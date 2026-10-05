"""Stage 5: root cause analysis of a change in bad rate, and an early-warning view.

Mix / rate decomposition between period 1 and period 2 over segments s:
    total change = sum_s w2*r2 - sum_s w1*r1
                 = sum_s (w2 - w1) * r1        <- mix: we lent more to riskier segments
                 + sum_s  w2 * (r2 - r1)       <- rate: the same segments performed worse
w = segment share of loans, r = segment bad rate. The two parts add up to the total exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _cells(df: pd.DataFrame, period_col: str, p: int, keys) -> pd.DataFrame:
    d = df[df[period_col] == p]
    g = d.groupby(keys, observed=True)["bad"].agg(loans="size", bads="sum")
    g["w"] = g["loans"] / g["loans"].sum()
    g["r"] = g["bads"] / g["loans"]
    return g


def mix_rate(df: pd.DataFrame, p1: int, p2: int, seg: str = "grade", period_col: str = "issue_year") -> dict:
    a = _cells(df, period_col, p1, [seg])
    b = _cells(df, period_col, p2, [seg])
    t = a[["w", "r", "loans"]].join(b[["w", "r", "loans"]], how="outer", lsuffix="1", rsuffix="2")
    t[["w1", "w2", "loans1", "loans2"]] = t[["w1", "w2", "loans1", "loans2"]].fillna(0)
    # a segment absent in one period: assume no rate change, so its effect shows up as mix
    t["r1"] = t["r1"].fillna(t["r2"])
    t["r2"] = t["r2"].fillna(t["r1"])
    t["mix"] = (t["w2"] - t["w1"]) * t["r1"]
    t["rate"] = t["w2"] * (t["r2"] - t["r1"])
    rate1 = float((t["w1"] * t["r1"]).sum())
    rate2 = float((t["w2"] * t["r2"]).sum())
    return {
        "segment": seg, "period_1": p1, "period_2": p2,
        "bad_rate_1": rate1, "bad_rate_2": rate2, "total_change": rate2 - rate1,
        "mix_effect": float(t["mix"].sum()), "rate_effect": float(t["rate"].sum()),
        "table": t.reset_index(),
    }


def drilldown(df: pd.DataFrame, p1: int, p2: int, dim: str, base: str = "grade",
              period_col: str = "issue_year") -> pd.DataFrame:
    """Which values of `dim` drive the deterioration INSIDE grades?
    Cells are grade x dim; each cell's rate effect w2*(r2 - r1) is summed by dim value."""
    a = _cells(df, period_col, p1, [base, dim])
    b = _cells(df, period_col, p2, [base, dim])
    t = a[["w", "r"]].join(b[["w", "r", "loans"]], how="outer", lsuffix="1", rsuffix="2").reset_index()
    t[["w1", "w2", "loans"]] = t[["w1", "w2", "loans"]].fillna(0)
    grade_r1 = _cells(df, period_col, p1, [base])["r"]
    t["r1"] = t["r1"].fillna(t[base].map(grade_r1))
    t["r2"] = t["r2"].fillna(t["r1"])
    t["rate_contribution"] = t["w2"] * (t["r2"] - t["r1"])
    out = t.groupby(dim, observed=True).agg(loans_period_2=("loans", "sum"), share_period_2=("w2", "sum"),
                                            rate_contribution=("rate_contribution", "sum")).reset_index()
    total = out["rate_contribution"].sum()
    out["share_of_within_grade_change"] = out["rate_contribution"] / total if total else np.nan
    # intensity: average within-grade bad-rate change for this segment, so a big segment
    # is not mistaken for a deteriorating one just because it is big
    out["avg_rate_change_pp"] = np.where(out["share_period_2"] > 0,
                                         out["rate_contribution"] / out["share_period_2"] * 100, np.nan)
    out.insert(0, "dimension", dim)
    out = out.rename(columns={dim: "segment"})
    out["segment"] = out["segment"].astype(str)
    return out.sort_values("rate_contribution", ascending=False).reset_index(drop=True)


def early_warning(df: pd.DataFrame, dims) -> pd.DataFrame:
    """For each segment: baseline bad rate (2012-2013), then the first issue quarter from which the
    segment ran at least EW_THRESHOLD_PP above its own baseline for EW_SUSTAIN quarters in a row."""
    rows = []
    for dim in dims:
        base = df[df["issue_year"].isin(C.EW_BASELINE_YEARS)].groupby(dim, observed=True)["bad"].mean()
        q = df.groupby([dim, "issue_q"], observed=True)["bad"].agg(loans="size", bad_rate="mean").reset_index()
        for seg, g in q.groupby(dim, observed=True):
            if seg not in base.index:
                continue
            g = g.sort_values("issue_q")
            b = base[seg]
            breach = (g["bad_rate"] - b >= C.EW_THRESHOLD_PP / 100) & (g["loans"] >= C.EW_MIN_LOANS)
            run, first = 0, None
            for qq, flag in zip(g["issue_q"], breach):
                run = run + 1 if flag else 0
                if run == C.EW_SUSTAIN:
                    idx = list(g["issue_q"]).index(qq) - (C.EW_SUSTAIN - 1)
                    first = g["issue_q"].iloc[idx]
                    break
            last = g.iloc[-1]
            rows.append({"dimension": dim, "segment": str(seg), "baseline_bad_rate": float(b),
                         "first_breach_quarter": first, "latest_quarter": last["issue_q"],
                         "latest_bad_rate": float(last["bad_rate"]),
                         "latest_vs_baseline_pp": float((last["bad_rate"] - b) * 100),
                         "loans_total": int(g["loans"].sum())})
    t = pd.DataFrame(rows)
    t["_order"] = t["first_breach_quarter"].fillna("9999Q9")
    return t.sort_values(["_order", "latest_vs_baseline_pp"], ascending=[True, False]).drop(columns="_order").reset_index(drop=True)
