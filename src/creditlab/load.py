"""Stage 0: raw file -> typed DuckDB tables -> modelling population (Parquet).

Also writes the data profile and the data-quality funnel (row count after
every filter), which are the first things a reviewer asks for.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def _sql(name: str, **params: str) -> str:
    text = (C.SQL_DIR / name).read_text(encoding="utf-8")
    for key, value in params.items():
        text = text.replace(f":{key}", value)
    return text


def _quoted_list(values) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def connect(db_path: Path = C.DB_FILE) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def build_base(raw_path: Path, con: duckdb.DuckDBPyConnection) -> dict:
    """Parse the raw file and build the `loans` population table. Returns DQ facts."""
    raw_sql_path = raw_path.resolve().as_posix().replace("'", "''")
    con.execute(_sql("00_parse_raw.sql", raw_path=raw_sql_path))
    con.execute(
        _sql(
            "01_build_loans.sql",
            start=C.ISSUE_START,
            end=C.ISSUE_END,
            term=str(C.TERM_MONTHS),
            bad_list=_quoted_list(C.BAD_STATUSES),
            good_list=_quoted_list(C.GOOD_STATUSES),
        )
    )
    funnel = data_quality_funnel(con)
    return {"funnel": funnel}


def data_quality_funnel(con) -> pd.DataFrame:
    """Row count after each cumulative filter, plus the statuses that were dropped."""
    steps = [
        ("rows read from raw file", "TRUE"),
        ("valid issue date", "issue_d IS NOT NULL"),
        (f"issued {C.ISSUE_START[:7]} to {C.ISSUE_END[:7]}",
         f"issue_d BETWEEN DATE '{C.ISSUE_START}' AND DATE '{C.ISSUE_END}'"),
        (f"term = {C.TERM_MONTHS} months", f"term_months = {C.TERM_MONTHS}"),
        ("individual applications", "application_type = 'INDIVIDUAL'"),
        ("final outcome (paid off or charged off)",
         f"(loan_status IN ({_quoted_list(C.BAD_STATUSES)}) OR loan_status IN ({_quoted_list(C.GOOD_STATUSES)}))"),
    ]
    rows, clause = [], []
    for label, cond in steps:
        clause.append(cond)
        n = con.execute(f"SELECT COUNT(*) FROM parsed WHERE {' AND '.join(clause)}").fetchone()[0]
        rows.append({"step": label, "rows": int(n)})
    funnel = pd.DataFrame(rows)
    funnel["dropped"] = funnel["rows"].shift(1).fillna(funnel["rows"]).astype(int) - funnel["rows"]
    return funnel


def excluded_statuses(con) -> pd.DataFrame:
    """Statuses removed by the final-outcome filter, inside the otherwise-eligible population."""
    return con.execute(f"""
        SELECT loan_status, COUNT(*) AS loans
        FROM parsed
        WHERE issue_d BETWEEN DATE '{C.ISSUE_START}' AND DATE '{C.ISSUE_END}'
          AND term_months = {C.TERM_MONTHS}
          AND application_type = 'INDIVIDUAL'
          AND NOT (loan_status IN ({_quoted_list(C.BAD_STATUSES)})
                   OR loan_status IN ({_quoted_list(C.GOOD_STATUSES)}))
        GROUP BY 1 ORDER BY 2 DESC
    """).df()


def profile(con) -> dict[str, pd.DataFrame]:
    """Formats and value counts a reviewer checks before trusting the parse."""
    out = {}
    out["status_counts"] = con.execute(
        "SELECT loan_status, COUNT(*) AS loans FROM parsed GROUP BY 1 ORDER BY 2 DESC").df()
    out["term_counts"] = con.execute(
        "SELECT term_months, COUNT(*) AS loans FROM parsed GROUP BY 1 ORDER BY 2 DESC").df()
    out["application_type_counts"] = con.execute(
        "SELECT application_type, COUNT(*) AS loans FROM parsed GROUP BY 1 ORDER BY 2 DESC").df()
    # parse failures: text present in the raw file but NULL after casting would show up here
    cols = [c for c in C.NUMERIC_FEATURES if c not in ("loan_to_income", "credit_hist_months")] + [
        "int_rate", "installment", "funded_amnt", "total_pymnt", "recoveries", "earliest_cr_line",
        "last_pymnt_d", "sub_grade"]
    missing = []
    for c in cols:
        n_null = con.execute(f"SELECT AVG(CASE WHEN {c} IS NULL THEN 1.0 ELSE 0 END) FROM loans").fetchone()[0]
        missing.append({"column": c, "missing_rate_in_population": float(n_null or 0.0)})
    out["missing_rates"] = pd.DataFrame(missing)
    out["bad_rate_by_year"] = con.execute("""
        SELECT issue_year, COUNT(*) AS loans, SUM(bad) AS bads, AVG(bad) AS bad_rate
        FROM loans GROUP BY 1 ORDER BY 1""").df()
    # does total_pymnt already include recoveries? (decides how realised return is computed)
    out["payment_identity"] = con.execute("""
        SELECT bad,
               AVG(CASE WHEN abs(total_pymnt - (total_rec_prncp + total_rec_int
                                 + total_rec_late_fee + recoveries)) < 1 THEN 1.0 ELSE 0 END)
                   AS share_total_pymnt_includes_recoveries,
               COUNT(*) AS loans
        FROM loans
        WHERE total_pymnt IS NOT NULL AND total_rec_prncp IS NOT NULL
        GROUP BY 1 ORDER BY 1""").df()
    return out


def export_population(con, path: Path = C.LOANS_PARQUET) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = con.execute("SELECT * FROM loans ORDER BY issue_d, id").df()
    df = assign_samples(df)
    df.to_parquet(path, index=False)
    return df


def assign_samples(df: pd.DataFrame) -> pd.DataFrame:
    """train / holdout (random, stratified, inside 2012-2014) and oot (2015)."""
    df = df.copy()
    df["sample"] = np.where(df["issue_year"] == C.OOT_YEAR, "oot", "dev")
    dev_idx = df.index[df["sample"] == "dev"]
    train_idx, hold_idx = train_test_split(
        dev_idx, test_size=C.HOLDOUT_SHARE, random_state=C.SEED, stratify=df.loc[dev_idx, "bad"])
    df.loc[train_idx, "sample"] = "train"
    df.loc[hold_idx, "sample"] = "holdout"
    df.loc[~df["issue_year"].isin(list(C.DEV_YEARS) + [C.OOT_YEAR]), "sample"] = "excluded"
    return df
