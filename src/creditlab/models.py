"""Stage 3: champion (LendingClub sub-grade), challenger (gradient boosting) and overlay."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from . import config as C
from .features import assert_no_leakage


# --------------------------------------------------------------------------- champion
def champion_pd(df: pd.DataFrame) -> np.ndarray:
    """Grade-implied PD: the train-sample bad rate of each sub-grade (grade, then overall, as fallback).
    The sub-grade itself is a rank, not a probability; this mapping lets us test its calibration."""
    tr = df[df["sample"] == "train"]
    by_sub = tr.groupby("sub_grade")["bad"].mean()
    by_grade = tr.groupby("grade")["bad"].mean()
    overall = tr["bad"].mean()
    p = df["sub_grade"].map(by_sub)
    p = p.fillna(df["grade"].map(by_grade)).fillna(overall)
    return p.to_numpy(dtype=float)


# --------------------------------------------------------------------------- challenger
class Challenger:
    """HistGradientBoosting on raw application features (not WoE), with monotone constraints."""

    def __init__(self, features):
        assert_no_leakage(features)
        self.features = list(features)
        self.categorical = [f for f in self.features if f in C.CATEGORICAL_FEATURES]
        self.codes: dict[str, dict[str, int]] = {}
        self.model: HistGradientBoostingClassifier | None = None

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for f in self.features:
            if f in self.categorical:
                cols.append(X[f].astype(str).map(self.codes[f]).astype("float64").to_numpy())  # unseen -> NaN
            else:
                cols.append(X[f].to_numpy(dtype=float))
        return np.column_stack(cols)

    def fit(self, X: pd.DataFrame, y) -> "Challenger":
        for f in self.categorical:
            cats = sorted(X[f].astype(str).unique())
            self.codes[f] = {c: i for i, c in enumerate(cats)}
        mask = [f in self.categorical for f in self.features]
        mono = [C.CHALLENGER_MONOTONE.get(f, 0) for f in self.features]
        self.model = HistGradientBoostingClassifier(categorical_features=mask, monotonic_cst=mono,
                                                    **C.CHALLENGER_PARAMS)
        self.model.fit(self._matrix(X), np.asarray(y).astype(int))
        return self

    def pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._matrix(X))[:, 1]

    def importance(self, X: pd.DataFrame, y, n: int = C.PERM_SAMPLE) -> pd.DataFrame:
        """Permutation importance: drop in AUC when one feature's values are shuffled."""
        rng = np.random.default_rng(C.SEED)
        idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
        Xs, ys = self._matrix(X.iloc[idx]), np.asarray(y)[idx]
        r = permutation_importance(self.model, Xs, ys, scoring="roc_auc", n_repeats=C.PERM_REPEATS,
                                   random_state=C.SEED, n_jobs=1)
        t = pd.DataFrame({"feature": self.features, "auc_drop": r.importances_mean, "std": r.importances_std})
        t["gini_drop"] = 2 * t["auc_drop"]
        return t.sort_values("auc_drop", ascending=False).reset_index(drop=True)


def partial_dependence_check(ch: Challenger, X: pd.DataFrame, feature: str, grid_n: int = 15,
                             sample: int = 5000) -> pd.DataFrame:
    """Average predicted PD as one feature is swept across its range (others held at observed values).
    Used to check that the top drivers move in a direction that makes credit sense."""
    rng = np.random.default_rng(C.SEED)
    Xs = X.iloc[rng.choice(len(X), size=min(sample, len(X)), replace=False)].copy()
    if feature in ch.categorical:
        grid = list(ch.codes[feature].keys())
    else:
        grid = np.unique(np.nanquantile(X[feature].to_numpy(dtype=float), np.linspace(0.02, 0.98, grid_n)))
    rows = []
    for v in grid:
        Xs[feature] = v
        rows.append({"feature": feature, "value": v, "avg_pd": float(ch.pd(Xs).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- overlay
def fit_overlay(champ_pd: np.ndarray, sc_log_odds: np.ndarray, y, train_mask) -> tuple[np.ndarray, dict]:
    """Does my scorecard add information on top of LendingClub's grade?
    Logistic regression on [logit(grade-implied PD), scorecard log-odds], fitted on train only."""
    p = np.clip(champ_pd, 1e-4, 1 - 1e-4)
    Z = np.column_stack([np.log(p / (1 - p)), sc_log_odds])
    m = LogisticRegression(C=1e6, max_iter=5000)
    m.fit(Z[train_mask], np.asarray(y)[train_mask])
    return m.predict_proba(Z)[:, 1], {"coef_grade": float(m.coef_[0][0]), "coef_scorecard": float(m.coef_[0][1]),
                                      "intercept": float(m.intercept_[0])}
