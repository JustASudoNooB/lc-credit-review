"""Model performance and stability metrics, written out by hand so each can be explained.

Convention: `risk` is any score where a HIGHER value means MORE likely to go bad
(a PD, a sub-grade rank, or minus a scorecard score).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gini(y, risk) -> float:
    """Gini = 2 * AUC - 1. 0 = random ranking, 1 = every bad ranked above every good."""
    return 2.0 * roc_auc_score(np.asarray(y), np.asarray(risk)) - 1.0


def ks(y, risk) -> float:
    """Kolmogorov-Smirnov: the largest gap between the cumulative share of bads and of goods
    captured as we move from the riskiest score downwards. Tied scores move together."""
    y = np.asarray(y).astype(int)
    risk = np.asarray(risk, dtype=float)
    vals, inv = np.unique(risk, return_inverse=True)
    bads = np.bincount(inv, weights=y, minlength=len(vals))
    tots = np.bincount(inv, minlength=len(vals)).astype(float)
    goods = tots - bads
    cb = np.cumsum(bads[::-1]) / bads.sum()
    cg = np.cumsum(goods[::-1]) / goods.sum()
    return float(np.max(np.abs(cb - cg)))


def psi_from_shares(expected, actual, eps: float = 1e-6) -> float:
    """Population Stability Index = sum((a - e) * ln(a / e)) over bins.
    Rule of thumb: < 0.10 stable, 0.10-0.25 watch, > 0.25 investigate."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi(expected_values, actual_values, bins: int = 10) -> float:
    """PSI of a continuous score; bins are deciles of the expected (development) distribution."""
    ev = np.asarray(expected_values, dtype=float)
    av = np.asarray(actual_values, dtype=float)
    edges = np.unique(np.quantile(ev, np.linspace(0, 1, bins + 1)[1:-1]))
    e_idx = np.searchsorted(edges, ev, side="right")
    a_idx = np.searchsorted(edges, av, side="right")
    k = len(edges) + 1
    e_share = np.bincount(e_idx, minlength=k) / len(ev)
    a_share = np.bincount(a_idx, minlength=k) / len(av)
    return psi_from_shares(e_share, a_share)


def psi_categorical(expected_codes, actual_codes) -> float:
    """PSI over discrete codes (used for CSI on WoE bins and for the sub-grade champion)."""
    e = pd.Series(expected_codes).value_counts(normalize=True)
    a = pd.Series(actual_codes).value_counts(normalize=True)
    idx = e.index.union(a.index)
    return psi_from_shares(e.reindex(idx, fill_value=0).to_numpy(), a.reindex(idx, fill_value=0).to_numpy())


def calibration_table(y, pd_hat, bins: int = 10) -> pd.DataFrame:
    """Predicted vs actual bad rate by decile of predicted PD."""
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "pd": np.asarray(pd_hat, dtype=float)})
    df["bucket"] = pd.qcut(df["pd"].rank(method="first"), bins, labels=False) + 1
    t = df.groupby("bucket").agg(loans=("y", "size"), predicted=("pd", "mean"), actual=("y", "mean"))
    t["gap_pp"] = (t["actual"] - t["predicted"]) * 100
    return t.reset_index()


def calibration_in_the_large(y, pd_hat) -> dict:
    p = float(np.mean(pd_hat))
    a = float(np.mean(y))
    return {"predicted_bad_rate": p, "actual_bad_rate": a, "gap_pp": (a - p) * 100,
            "relative_gap": (a - p) / a if a else float("nan")}


def bootstrap_gini(y, risks: dict, n_boot: int, seed: int, baseline: str | None = None) -> dict:
    """Bootstrap 95% CI for each model's Gini and, if `baseline` is given, for each model's
    Gini minus the baseline's on the SAME resample (paired, so the CI of the difference is tight)."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = {k: [] for k in risks}
    diffs = {k: [] for k in risks if baseline and k != baseline}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        g = {k: gini(yb, np.asarray(r)[idx]) for k, r in risks.items()}
        for k in risks:
            draws[k].append(g[k])
        for k in diffs:
            diffs[k].append(g[k] - g[baseline])
    out = {}
    for k, v in draws.items():
        v = np.array(v)
        out[k] = {"gini": gini(y, risks[k]), "ci_low": float(np.percentile(v, 2.5)),
                  "ci_high": float(np.percentile(v, 97.5))}
    for k, v in diffs.items():
        v = np.array(v)
        out[k]["diff_vs_" + baseline] = float(gini(y, risks[k]) - gini(y, risks[baseline]))
        out[k]["diff_ci_low"] = float(np.percentile(v, 2.5))
        out[k]["diff_ci_high"] = float(np.percentile(v, 97.5))
    return out


def vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factor: 1 / (1 - R^2) from regressing each column on all the others."""
    cols = list(X.columns)
    A = X.to_numpy(dtype=float)
    out = {}
    for j, c in enumerate(cols):
        yj = A[:, j]
        others = np.delete(A, j, axis=1)
        Z = np.column_stack([np.ones(len(A)), others])
        beta, *_ = np.linalg.lstsq(Z, yj, rcond=None)
        resid = yj - Z @ beta
        r2 = 1 - resid.var() / yj.var() if yj.var() > 0 else 0.0
        out[c] = float(1 / (1 - r2)) if r2 < 1 else float("inf")
    return pd.Series(out, name="vif")


def two_proportion_z(bad1: float, n1: float, bad2: float, n2: float) -> tuple[float, float]:
    """z-test that two bad rates differ; returns (z, two-sided p-value)."""
    if min(n1, n2) == 0:
        return float("nan"), float("nan")
    p1, p2 = bad1 / n1, bad2 / n2
    p = (bad1 + bad2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    return float(z), float(math.erfc(abs(z) / math.sqrt(2)))
