"""Project-wide paths and constants.

Every tunable number in the project lives here, so a reviewer can see each
modelling choice in one place and change it without touching the logic.
"""
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RAW_FILE = DATA_RAW / "accepted_2007_to_2018Q4.csv.gz"
DB_FILE = DATA_PROCESSED / "lc.duckdb"
LOANS_PARQUET = DATA_PROCESSED / "loans.parquet"
SCORED_PARQUET = DATA_PROCESSED / "scored.parquet"
SQL_DIR = ROOT / "sql"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
RESULTS_JSON = OUTPUTS / "results.json"
REPORTS = ROOT / "reports"

SEED = 42

# --------------------------------------------------------------------------- population
TERM_MONTHS = 36
ISSUE_START = "2012-01-01"
ISSUE_END = "2015-12-31"          # 36-month loans issued by Dec 2015 reach maturity by the Dec 2018 snapshot
DEV_YEARS = (2012, 2013, 2014)    # development sample (train + in-time holdout)
OOT_YEAR = 2015                   # out-of-time test sample
HOLDOUT_SHARE = 0.20

BAD_STATUSES = (
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
)
GOOD_STATUSES = (
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
)

# --------------------------------------------------------------------------- leakage control
# Fields only known after the loan is booked. They may be used to measure outcomes
# (monitoring, policy back-tests) but never as model inputs.
BANNED_FEATURES = {
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "out_prncp", "out_prncp_inv", "debt_settlement_flag",
    "loan_status", "pymnt_plan", "bad",
}
BANNED_PREFIXES = ("hardship_", "settlement_")

# LendingClub's own model outputs: used only as the champion benchmark.
# installment is excluded because it is computed from int_rate.
CHAMPION_ONLY = {"grade", "sub_grade", "sub_grade_rank", "int_rate", "installment"}

NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc", "loan_to_income", "dti", "emp_length_yrs", "fico",
    "inq_last_6mths", "delinq_2yrs", "open_acc", "pub_rec", "pub_rec_bankruptcies",
    "revol_bal", "revol_util", "total_acc", "mort_acc", "credit_hist_months",
]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose"]
CANDIDATE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXCLUDED_WITH_REASON = {
    "addr_state": "geographic proxy for protected characteristics (fair-lending risk); also unstable across vintages",
    "emp_title": "free text with thousands of values; would need its own NLP treatment",
    "grade / sub_grade / int_rate / installment": "LendingClub's own model outputs; kept only as the champion benchmark",
}

# --------------------------------------------------------------------------- binning and scorecard
FINE_BINS = 20                 # quantile bins before coarse classing
MIN_BIN_SHARE = 0.05           # every coarse bin holds at least 5% of non-missing train rows
MAX_BINS = 8
RARE_CATEGORY_SHARE = 0.01     # categories below 1% of train are pooled into RARE
WOE_SMOOTHING = 0.5            # additive smoothing so an empty cell never gives infinite WoE
IV_MIN = 0.02                  # below this a feature has no useful signal
IV_SUSPECT = 0.50              # above this a feature is checked for leakage
CORR_MAX = 0.70                # drop the lower-IV feature of any pair above this

BASE_SCORE = 600               # score at the base odds
BASE_ODDS = 50                 # good:bad odds of 50:1 at the base score
PDO = 20                       # points to double the odds

# --------------------------------------------------------------------------- validation
N_BOOTSTRAP = 200
PSI_BINS = 10
CALIBRATION_BINS = 10

# --------------------------------------------------------------------------- challenger
CHALLENGER_PARAMS = dict(
    learning_rate=0.05,
    max_iter=600,
    max_leaf_nodes=31,
    min_samples_leaf=200,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.15,   # early stopping on a held-back slice of the training data
    n_iter_no_change=30,
    scoring="loss",
    random_state=SEED,
)
# +1: predicted risk may only rise with the feature; -1: only fall. Keeps the GBM explainable.
CHALLENGER_MONOTONE = {
    "fico": -1,
    "dti": 1,
    "inq_last_6mths": 1,
    "revol_util": 1,
    "loan_to_income": 1,
}
PERM_SAMPLE = 50_000
PERM_REPEATS = 3

# --------------------------------------------------------------------------- strategy
TARGET_APPROVAL = 0.80
CUTOFF_GRID = [round(0.50 + 0.05 * i, 2) for i in range(11)]   # 50% .. 100%
DTI_CAPS_PRE = [15, 20, 25, 30, 35]
DTI_CAPS_POST = [20, 25, 30, 35, 40, 45]
REFERENCE_CAP_POST = 35        # cap used in the memo's headline policy finding
SCORE_BANDS = 5

# --------------------------------------------------------------------------- RCA and early warning
RCA_PERIOD_1 = 2013
RCA_PERIOD_2 = 2015
DTI_BAND_EDGES = [-1e9, 10, 20, 30, 1e9]
DTI_BAND_LABELS = ["<10", "10-20", "20-30", "30+"]
FICO_BAND_EDGES = [0, 680, 700, 720, 750, 1e9]
FICO_BAND_LABELS = ["<680", "680-699", "700-719", "720-749", "750+"]
EW_BASELINE_YEARS = (2012, 2013)
EW_THRESHOLD_PP = 2.0          # breach = bad rate at least 2 pp above the segment's own baseline
EW_SUSTAIN = 2                 # ... for 2 consecutive issue quarters
EW_MIN_LOANS = 300             # quarters with fewer loans are not judged

# --------------------------------------------------------------------------- review expectations
# Direction a credit analyst expects for each driver (+1: higher value -> riskier).
# Used to check that the challenger's top drivers make credit sense.
EXPECTED_DIRECTION = {
    "fico": -1, "dti": 1, "inq_last_6mths": 1, "revol_util": 1, "loan_to_income": 1,
    "annual_inc": -1, "mort_acc": -1, "emp_length_yrs": -1, "credit_hist_months": -1,
    "delinq_2yrs": 1, "pub_rec": 1, "pub_rec_bankruptcies": 1, "loan_amnt": 1,
}
