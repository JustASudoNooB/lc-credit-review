"""Generate a FAKE LendingClub-format file for testing the pipeline end to end.

THE NUMBERS THIS PRODUCES ARE NOT REAL AND MUST NEVER BE QUOTED ANYWHERE.
The file copies LendingClub's column names and string formats (" 36 months",
"Dec-2015", "10+ years", footer junk rows, out-of-scope statuses) so that the
loader, SQL, models and reports can be exercised before the real file arrives.

Usage:  python tools/make_synthetic_lc.py  [--rows 300000] [--out data/raw/SYNTHETIC_lc_test.csv.gz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GRADES = "ABCDEFG"
GRADE_SHARE_BASE = np.array([0.20, 0.31, 0.27, 0.14, 0.06, 0.018, 0.002])


def month_add(dates: pd.Series, months: np.ndarray) -> pd.Series:
    periods = dates.dt.to_period("M") + months.astype(int)
    return periods.dt.to_timestamp()


def fmt_month(d: pd.Series) -> pd.Series:
    out = d.dt.strftime("%b-%Y")
    return out.where(d.notna(), None)


def amort_payment(principal, rate_pct, n):
    r = rate_pct / 100.0 / 12.0
    return principal * r / (1 - (1 + r) ** (-n))


def balance_after(principal, rate_pct, n, k):
    r = rate_pct / 100.0 / 12.0
    pay = amort_payment(principal, rate_pct, n)
    return principal * (1 + r) ** k - pay * ((1 + r) ** k - 1) / r


def generate(rows: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = rows

    # issue dates 2011-2016, volume growing over time (2011 and 2016 test the date filter)
    months = pd.period_range("2011-01", "2016-12", freq="M")
    w = np.linspace(1, 6, len(months)) ** 1.6
    issue = months[rng.choice(len(months), size=n, p=w / w.sum())].to_timestamp()
    issue = pd.Series(issue)
    year = issue.dt.year.to_numpy()

    term = np.where(rng.random(n) < 0.74, 36, 60)
    app_type = np.where((year >= 2015) & (rng.random(n) < 0.03), "Joint App", "Individual")

    fico_low = np.clip(np.round((660 + rng.gamma(2.2, 16, n)) / 5) * 5, 660, 845)
    annual_inc = np.round(np.exp(rng.normal(np.log(68000), 0.55, n)), -2)
    annual_inc[rng.random(n) < 0.0008] = 0
    dti = np.round(np.clip(rng.normal(18, 8, n), 0, 39.9), 2)
    dti_missing = rng.random(n) < 0.0005
    emp = rng.choice(11, size=n, p=[0.08, 0.07, 0.09, 0.08, 0.06, 0.06, 0.05, 0.05, 0.05, 0.04, 0.37])
    emp_missing = rng.random(n) < 0.05
    home = rng.choice(["MORTGAGE", "RENT", "OWN", "ANY"], size=n, p=[0.495, 0.40, 0.1045, 0.0005])
    purposes = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
                "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
                "renewable_energy", "educational"]
    pp = np.array([0.585, 0.22, 0.058, 0.05, 0.02, 0.012, 0.011, 0.01, 0.007, 0.006, 0.005, 0.01,
                   0.0007, 0.0003])
    purpose = rng.choice(purposes, size=n, p=pp / pp.sum())
    inq = np.minimum(rng.poisson(0.7, n), 8)
    delinq = np.minimum(rng.poisson(0.3, n), 10)
    pub_rec = np.minimum(rng.poisson(0.18, n), 6)
    pub_bk = np.minimum(pub_rec, rng.binomial(pub_rec, 0.8))
    pub_bk_missing = (year <= 2012) & (rng.random(n) < 0.02)
    open_acc = rng.poisson(10.5, n) + 1
    total_acc = open_acc + rng.poisson(13, n)
    revol_util = np.round(np.clip(rng.beta(2.2, 2.0, n) * 105, 0, 140), 1)
    revol_bal = np.round(annual_inc * np.clip(rng.lognormal(-1.9, 0.7, n), 0, 3))
    mort_acc = np.where(home == "MORTGAGE", rng.poisson(2.2, n), rng.poisson(0.4, n)).astype(float)
    mort_missing = (year <= 2012) & (rng.random(n) < 0.03)
    hist_years = np.clip(rng.gamma(4.0, 4.0, n), 3, 50)
    earliest = month_add(issue, -(hist_years * 12).astype(int))

    loan_amnt = np.round(np.clip(annual_inc * rng.beta(2, 9, n) + 1000, 1000, 35000) / 25) * 25
    loan_amnt = np.where(annual_inc <= 0, np.round(rng.uniform(1000, 10000, n) / 25) * 25, loan_amnt)
    verified = np.where(loan_amnt > 15000, rng.choice(["Verified", "Source Verified", "Not Verified"], n,
                                                      p=[0.5, 0.35, 0.15]),
                        rng.choice(["Verified", "Source Verified", "Not Verified"], n, p=[0.25, 0.35, 0.40]))

    lti = np.where(annual_inc > 0, loan_amnt / np.maximum(annual_inc, 1), 0.3)
    pur_eff = pd.Series(purposes).map({"small_business": 0.55, "credit_card": -0.15, "car": -0.2,
                                       "wedding": -0.2, "medical": 0.2, "moving": 0.25,
                                       "house": 0.1, "home_improvement": -0.05}).fillna(0).to_numpy()
    pur_idx = pd.Series(purpose).map({p: i for i, p in enumerate(purposes)}).to_numpy()
    home_eff = pd.Series(home).map({"RENT": 0.15, "OWN": 0.0, "MORTGAGE": -0.08, "ANY": 0.0}).to_numpy()
    z_obs = (-0.032 * (fico_low - 695) + 0.035 * (dti - 18) + 0.30 * inq + 0.25 * delinq
             + 0.30 * pub_rec + 0.012 * (revol_util - 55) - 0.45 * np.log(np.maximum(annual_inc, 5000) / 68000)
             + 2.0 * (lti - 0.2) + 1.5 * pur_eff[pur_idx] + 1.5 * home_eff - 0.04 * emp
             - 0.020 * (hist_years - 15) - 0.10 * mort_acc)
    u_hidden = rng.normal(0, 0.45, n)              # information LendingClub had and we do not
    z_total = z_obs + u_hidden

    # LendingClub sub-grade: their view of risk, using the hidden information plus noise
    lc_view = z_obs + 0.85 * u_hidden + rng.normal(0, 0.35, n)
    shift = np.select([year <= 2012, year == 2013, year == 2014], [-0.15, -0.05, 0.05], 0.18)
    lc_view_shifted = lc_view + shift               # grade mix drifts riskier in later vintages
    cuts = np.quantile(lc_view, np.cumsum(GRADE_SHARE_BASE)[:-1])
    g_idx = np.searchsorted(cuts, lc_view_shifted)
    sub = np.zeros(n, dtype=int)
    for g in range(7):
        m = g_idx == g
        if m.sum() == 0:
            continue
        lo = -np.inf if g == 0 else cuts[g - 1]
        hi = np.inf if g == 6 else cuts[g]
        vals = lc_view_shifted[m]
        inner = np.clip(vals, np.nanmin(vals) if np.isinf(lo) else lo, np.nanmax(vals) if np.isinf(hi) else hi)
        q = pd.Series(inner).rank(pct=True).to_numpy()
        sub[m] = np.minimum((q * 5).astype(int), 4)
    grade = np.array(list(GRADES))[g_idx]
    sub_grade = np.char.add(grade.astype(str), (sub + 1).astype(str))
    rank = g_idx * 5 + sub + 1
    int_rate = np.round(5.3 + (rank - 1) * 0.62 + rng.normal(0, 0.15, n)
                        + np.select([year <= 2012, year == 2013], [0.4, 0.2], 0.0), 2)

    vintage = np.select([year <= 2012, year == 2013, year == 2014], [-0.12, -0.05, 0.05], 0.22)
    seg_drift = np.where((year == 2015) & (verified == "Not Verified") & (purpose == "debt_consolidation"),
                         0.25, 0.0)
    logit = -2.25 + 0.95 * z_total + vintage + seg_drift + np.where(term == 60, 0.5, 0)
    p_bad = 1 / (1 + np.exp(-logit))
    is_bad = rng.random(n) < p_bad

    funded = loan_amnt.copy()
    installment = np.round(amort_payment(funded, int_rate, term), 2)

    # outcome timing
    default_m = np.clip(np.round(rng.gamma(2.2, 6.0, n)), 0, term - 1).astype(int)
    never_paid = is_bad & (rng.random(n) < 0.04)
    prepay = (~is_bad) & (rng.random(n) < 0.38)
    prepay_m = rng.integers(4, term, n)
    paid_months = np.where(is_bad, default_m, np.where(prepay, prepay_m, term))
    last_pymnt = month_add(issue, paid_months)
    last_pymnt = last_pymnt.where(~never_paid, pd.NaT)

    bal = balance_after(funded, int_rate, term, paid_months)
    bal = np.clip(bal, 0, None)
    paid_sched = installment * paid_months
    prncp_paid = funded - bal
    int_paid = paid_sched - prncp_paid
    recov = np.where(is_bad, np.round(bal * rng.beta(1.2, 9, n), 2), 0.0)
    late_fee = np.where(is_bad & (rng.random(n) < 0.3), 15.0, 0.0)
    total_rec_prncp = np.where(is_bad, prncp_paid, funded)
    total_rec_int = np.where(is_bad, int_paid, int_paid)
    total_rec_prncp = np.where(never_paid, 0.0, total_rec_prncp)
    total_rec_int = np.where(never_paid, 0.0, total_rec_int)
    total_pymnt = np.round(total_rec_prncp + total_rec_int + late_fee + recov, 2)

    status = np.where(is_bad, np.where(rng.random(n) < 0.02, "Default", "Charged Off"), "Fully Paid")
    # loans still open at the snapshot (modified / late) and early policy-code loans
    open_late = (rng.random(n) < 0.006)
    status = np.where(open_late, rng.choice(["Current", "Late (31-120 days)", "In Grace Period"], n), status)
    still_running = (term == 60) & (year >= 2014)
    status = np.where(still_running & (rng.random(n) < 0.6), "Current", status)
    policy = (year == 2011) & (rng.random(n) < 0.05)
    status = np.where(policy & (status == "Fully Paid"), "Does not meet the credit policy. Status:Fully Paid", status)
    status = np.where(policy & (status == "Charged Off"), "Does not meet the credit policy. Status:Charged Off", status)

    def emp_str(e, miss):
        if miss:
            return None
        if e == 0:
            return "< 1 year"
        if e == 10:
            return "10+ years"
        return "1 year" if e == 1 else f"{e} years"

    df = pd.DataFrame({
        "id": np.arange(10_000_000, 10_000_000 + n).astype(str),
        "member_id": None,
        "loan_amnt": loan_amnt,
        "funded_amnt": funded,
        "funded_amnt_inv": funded,
        "term": np.where(term == 36, " 36 months", " 60 months"),
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_title": "Analyst",
        "emp_length": [emp_str(e, m) for e, m in zip(emp, emp_missing)],
        "home_ownership": home,
        "annual_inc": annual_inc,
        "verification_status": verified,
        "issue_d": fmt_month(issue),
        "loan_status": status,
        "pymnt_plan": "n",
        "purpose": purpose,
        "title": "Debt consolidation",
        "zip_code": "123xx",
        "addr_state": rng.choice(["CA", "NY", "TX", "FL", "IL", "NJ"], n),
        "dti": np.where(dti_missing, np.nan, dti),
        "delinq_2yrs": delinq,
        "earliest_cr_line": fmt_month(earliest),
        "fico_range_low": fico_low,
        "fico_range_high": fico_low + 4,
        "inq_last_6mths": inq,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "out_prncp": 0.0,
        "total_pymnt": total_pymnt,
        "total_rec_prncp": np.round(total_rec_prncp, 2),
        "total_rec_int": np.round(total_rec_int, 2),
        "total_rec_late_fee": late_fee,
        "recoveries": recov,
        "collection_recovery_fee": np.round(recov * 0.18, 2),
        "last_pymnt_d": fmt_month(last_pymnt),
        "last_fico_range_high": np.where(is_bad, 540, 720),   # leaky on purpose: must never be used
        "application_type": app_type,
        "mort_acc": np.where(mort_missing, np.nan, mort_acc),
        "pub_rec_bankruptcies": np.where(pub_bk_missing, np.nan, pub_bk),
        "hardship_flag": "N",
        "debt_settlement_flag": "N",
    })
    junk = pd.DataFrame({"id": ["Total amount funded in policy code 1: 1234567",
                                "Total amount funded in policy code 2: 7654321"]})
    return pd.concat([df, junk], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=300_000)
    ap.add_argument("--out", default="data/raw/SYNTHETIC_lc_test.csv.gz")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows)
    df.to_csv(out, index=False, compression="gzip")
    print(f"wrote {len(df):,} rows to {out} (SYNTHETIC - for pipeline testing only)")


if __name__ == "__main__":
    main()
