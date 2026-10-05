"""Stage 2: application scorecard = logistic regression on WoE features, scaled to points."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config as C
from . import woe as W
from .features import assert_no_leakage
from .metrics import vif


def scaling(base_score=C.BASE_SCORE, base_odds=C.BASE_ODDS, pdo=C.PDO):
    """score = offset + factor * ln(good:bad odds). factor = PDO / ln 2."""
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)
    return factor, offset


@dataclass
class Scorecard:
    features: list[str]
    binnings: dict
    coef: np.ndarray
    intercept: float
    selection_log: list[str] = field(default_factory=list)

    def woe(self, X: pd.DataFrame) -> pd.DataFrame:
        return W.transform_all(X, self.binnings, self.features)

    def log_odds_bad(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + self.woe(X).to_numpy() @ self.coef

    def pd(self, X: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds_bad(X)))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        factor, offset = scaling()
        return offset - factor * self.log_odds_bad(X)     # higher score = safer

    def points_table(self) -> pd.DataFrame:
        factor, offset = scaling()
        k = len(self.features)
        rows = []
        for f, b in zip(self.features, self.coef):
            fb = self.binnings[f]
            for i, (lab, w) in enumerate(zip(fb.labels, fb.woe)):
                pts = -(b * w + self.intercept / k) * factor + offset / k
                rows.append({"feature": f, "bin": i, "label": lab, "woe": w, "coef": b, "points": pts,
                             "loans_train": int(fb.n[i]), "bad_rate_train": fb.bads[i] / fb.n[i]})
            pts0 = -(self.intercept / k) * factor + offset / k
            rows.append({"feature": f, "bin": -1, "label": "UNSEEN (neutral)", "woe": 0.0, "coef": b,
                         "points": pts0, "loans_train": 0, "bad_rate_train": np.nan})
        t = pd.DataFrame(rows)
        t["points_rounded"] = t["points"].round().astype(int)
        return t


def fit_lr(W_train: pd.DataFrame, y) -> LogisticRegression:
    # C very large = effectively unpenalised: the WoE transform already limits overfitting,
    # and an unpenalised fit keeps coefficients interpretable.
    m = LogisticRegression(C=1e6, max_iter=5000)
    m.fit(W_train.to_numpy(), np.asarray(y).astype(int))
    return m


def build(X_train: pd.DataFrame, y_train) -> tuple[Scorecard, dict]:
    candidates = list(X_train.columns)
    assert_no_leakage(candidates)
    log = []
    binnings = W.fit_all(X_train, y_train, candidates)
    ivt = W.iv_table(binnings)

    # 1. information value filter
    weak = ivt.loc[ivt["iv"] < C.IV_MIN, "feature"].tolist()
    for f in weak:
        log.append(f"dropped {f}: IV {binnings[f].iv:.4f} < {C.IV_MIN}")
    for f in ivt.loc[ivt["iv"] > C.IV_SUSPECT, "feature"]:
        log.append(f"FLAG {f}: IV {binnings[f].iv:.3f} > {C.IV_SUSPECT}; checked for leakage "
                   "(field is known at application time) and kept")
    keep = [f for f in ivt["feature"] if f not in weak]           # ordered by IV, highest first

    # 2. correlation filter, greedy by IV
    Wt = W.transform_all(X_train, binnings, keep)
    corr = Wt.corr()
    selected = []
    for f in keep:
        clash = [g for g in selected if abs(corr.loc[f, g]) > C.CORR_MAX]
        if clash:
            log.append(f"dropped {f}: |corr| {abs(corr.loc[f, clash[0]]):.2f} with {clash[0]} "
                       f"(higher IV) > {C.CORR_MAX}")
        else:
            selected.append(f)

    # 3. fit and enforce coefficient signs. With WoE = ln(good/bad) every coefficient must be
    #    NEGATIVE: more evidence of "good" must lower the predicted bad rate. A positive sign
    #    means the feature is fighting a correlated one and its points would read backwards.
    while True:
        model = fit_lr(Wt[selected], y_train)
        coef = model.coef_[0]
        wrong = [(f, c) for f, c in zip(selected, coef) if c >= 0]
        if not wrong:
            break
        worst = min(wrong, key=lambda fc: binnings[fc[0]].iv)[0]
        log.append(f"dropped {worst}: coefficient sign reversed in the multivariate fit (IV {binnings[worst].iv:.3f})")
        selected.remove(worst)

    v = vif(Wt[selected])
    sc = Scorecard(selected, {f: binnings[f] for f in selected}, coef, float(model.intercept_[0]), log)
    diagnostics = {
        "iv_table": ivt,
        "all_binnings": binnings,
        "vif": v,
        "correlation": corr,
        "selection_log": log,
    }
    return sc, diagnostics
