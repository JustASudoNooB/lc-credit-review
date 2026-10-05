"""Stage 4: underwriting strategy back-tests on the out-of-time (2015) book.

All loans in the data were approved by LendingClub. The question each test asks is:
"if we had to approve only X% of these loans, which X% should it have been, and what would
the bad rate and return have been?" Realised outcome fields (total_pymnt) are allowed here
because this evaluates a policy after the fact; they are never model inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M


def approve_best(risk, rate: float, seed: int = C.SEED) -> np.ndarray:
    """Approve the `rate` share with the lowest risk. Ties (e.g. a whole sub-grade sitting on the
    cut-off) are broken at random with a fixed seed so every policy hits exactly the same rate."""
    risk = np.asarray(risk, dtype=float)
    n = len(risk)
    tiebreak = np.random.default_rng(seed).random(n)
    order = np.lexsort((tiebreak, risk))
    k = int(round(rate * n))
    approved = np.zeros(n, dtype=bool)
    approved[order[:k]] = True
    return approved


def _book(d: pd.DataFrame) -> dict:
    funded = d["funded_amnt"].sum()
    return {
        "loans": int(len(d)),
        "bad_rate": float(d["bad"].mean()) if len(d) else float("nan"),
        "net_return": float((d["total_pymnt"].sum() - funded) / funded) if funded else float("nan"),
        "funded_amnt": float(funded),
    }


def swap_set(oot: pd.DataFrame, risk_a, risk_b, rate: float, name_a: str, name_b: str) -> dict:
    a = approve_best(risk_a, rate)
    b = approve_best(risk_b, rate)
    cells = {
        "Approved by both": a & b,
        f"Swap-in ({name_b} only)": b & ~a,
        f"Swap-out ({name_a} only)": a & ~b,
        "Declined by both": ~a & ~b,
    }
    rows = []
    for k, m in cells.items():
        rows.append({"cell": k, **_book(oot[m])})
    t = pd.DataFrame(rows)
    t["share_of_book"] = t["loans"] / len(oot)
    z, p = M.two_proportion_z(oot.loc[b & ~a, "bad"].sum(), (b & ~a).sum(),
                              oot.loc[a & ~b, "bad"].sum(), (a & ~b).sum())
    return {
        "table": t,
        "approval_rate": rate,
        "policy_a": {"name": name_a, **_book(oot[a])},
        "policy_b": {"name": name_b, **_book(oot[b])},
        "swap_in_vs_out_z": z,
        "swap_in_vs_out_p": p,
    }


def cutoff_table(oot: pd.DataFrame, risks: dict, grid=C.CUTOFF_GRID) -> pd.DataFrame:
    rows = []
    for name, r in risks.items():
        for rate in grid:
            m = approve_best(r, rate)
            rows.append({"model": name, "approval_rate": rate, **_book(oot[m])})
    return pd.DataFrame(rows)


def dti_cap_tests(oot: pd.DataFrame, col: str, caps, score: np.ndarray, bands: int = C.SCORE_BANDS) -> dict:
    """Decline every application above a DTI cap. Overall effect, then inside score bands to see
    whether the cap finds risk that the score has not already priced."""
    d = oot.assign(_score=score).dropna(subset=[col])
    base = _book(d)
    overall, within = [], []
    labels = [f"Q{i + 1}" + (" riskiest" if i == 0 else " safest" if i == bands - 1 else "") for i in range(bands)]
    d["_band"] = pd.qcut(d["_score"].rank(method="first"), bands, labels=labels)   # score: higher = safer
    for cap in caps:
        dec = d[col] > cap
        kept, declined = d[~dec], d[dec]
        z, p = M.two_proportion_z(declined["bad"].sum(), len(declined), kept["bad"].sum(), len(kept))
        overall.append({
            "dti_measure": col, "cap": cap, "declined_share": float(dec.mean()),
            "bad_rate_declined": _book(declined)["bad_rate"], "bad_rate_kept": _book(kept)["bad_rate"],
            "bad_rate_change_pp": (_book(kept)["bad_rate"] - base["bad_rate"]) * 100,
            "net_return_kept": _book(kept)["net_return"], "net_return_change_pp": (_book(kept)["net_return"] - base["net_return"]) * 100,
            "net_return_declined": _book(declined)["net_return"], "z": z, "p_value": p,
        })
        for band, g in d.groupby("_band", observed=True):
            gd = g[col] > cap
            z, p = M.two_proportion_z(g.loc[gd, "bad"].sum(), gd.sum(), g.loc[~gd, "bad"].sum(), (~gd).sum())
            within.append({
                "dti_measure": col, "cap": cap, "score_band": str(band),
                "band_loans": int(len(g)), "declined_share": float(gd.mean()),
                "bad_rate_above_cap": float(g.loc[gd, "bad"].mean()) if gd.any() else np.nan,
                "bad_rate_below_cap": float(g.loc[~gd, "bad"].mean()) if (~gd).any() else np.nan,
                "z": z, "p_value": p,
            })
    within = pd.DataFrame(within)
    within["lift_pp"] = (within["bad_rate_above_cap"] - within["bad_rate_below_cap"]) * 100
    within["significant_5pct"] = within["p_value"] < 0.05
    return {"base": base, "overall": pd.DataFrame(overall), "within_bands": within}
