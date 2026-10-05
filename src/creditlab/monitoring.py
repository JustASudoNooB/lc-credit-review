"""Stage 1: run the SQL monitoring queries and export tidy CSVs for Tableau."""
from __future__ import annotations

import pandas as pd

from . import config as C

QUERIES = {
    "static_pool_by_quarter_grade": "02_static_pool.sql",
    "mob_curves_by_quarter": "03_mob_curves.sql",
    "portfolio_by_quarter": "04_portfolio_by_quarter.sql",
}


def run(con) -> dict[str, pd.DataFrame]:
    C.TABLES.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, file in QUERIES.items():
        sql = (C.SQL_DIR / file).read_text(encoding="utf-8")
        df = con.execute(sql).df()
        df.to_csv(C.TABLES / f"{name}.csv", index=False)
        out[name] = df
    return out


def summary(tables: dict[str, pd.DataFrame]) -> dict:
    pq = tables["portfolio_by_quarter"]
    mob = tables["mob_curves_by_quarter"]
    by_year = (pq.assign(bads=pq["bad_rate"] * pq["loans"])
                 .groupby("issue_year")[["loans", "bads"]].sum())
    by_year["bad_rate"] = by_year["bads"] / by_year["loans"]
    # MOB at which half of each year's lifetime defaults had occurred
    m = mob.groupby(["issue_year", "mob"])[["new_defaults", "cohort_loans"]].sum().reset_index()
    half = {}
    for y, g in m.groupby("issue_year"):
        cum = g["new_defaults"].cumsum()
        total = cum.iloc[-1]
        half[int(y)] = int(g.loc[cum >= total / 2, "mob"].iloc[0]) if total > 0 else None
    worst_q = pq.loc[pq["bad_rate"].idxmax()]
    best_q = pq.loc[pq["bad_rate"].idxmin()]
    return {
        "quarters": int(len(pq)),
        "bad_rate_by_year": {int(k): float(v) for k, v in by_year["bad_rate"].items()},
        "loans_by_year": {int(k): int(v) for k, v in by_year["loans"].items()},
        "median_default_mob_by_year": half,
        "worst_quarter": {"issue_q": str(worst_q["issue_q"]), "bad_rate": float(worst_q["bad_rate"])},
        "best_quarter": {"issue_q": str(best_q["issue_q"]), "bad_rate": float(best_q["bad_rate"])},
    }
