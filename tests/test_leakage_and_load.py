import gzip

import pytest

from creditlab import config as C
from creditlab import features as F
from creditlab import load
from creditlab.models import Challenger


# ------------------------------------------------------------------ leakage guard
def test_candidate_features_are_clean():
    F.assert_no_leakage(C.CANDIDATE_FEATURES)
    assert not set(C.CANDIDATE_FEATURES) & C.BANNED_FEATURES
    assert not set(C.CANDIDATE_FEATURES) & C.CHAMPION_ONLY
    assert not any(f.startswith(C.BANNED_PREFIXES) for f in C.CANDIDATE_FEATURES)


@pytest.mark.parametrize("bad", ["total_pymnt", "recoveries", "last_fico_range_high", "last_pymnt_d",
                                 "hardship_status", "settlement_amount", "loan_status"])
def test_post_origination_fields_are_rejected(bad):
    with pytest.raises(F.LeakageError):
        F.assert_no_leakage(["fico", bad])


@pytest.mark.parametrize("champ", ["grade", "sub_grade", "int_rate", "installment"])
def test_lendingclub_model_outputs_are_rejected(champ):
    with pytest.raises(F.LeakageError):
        F.assert_no_leakage(["fico", champ])
    with pytest.raises(F.LeakageError):
        Challenger(["fico", champ])


# ------------------------------------------------------------------ loader: formats, filters, target
HEADER = ("id,loan_amnt,funded_amnt,term,int_rate,installment,grade,sub_grade,emp_length,home_ownership,"
          "annual_inc,verification_status,issue_d,loan_status,purpose,addr_state,dti,delinq_2yrs,earliest_cr_line,"
          "fico_range_low,fico_range_high,inq_last_6mths,open_acc,pub_rec,revol_bal,revol_util,total_acc,"
          "total_pymnt,total_rec_prncp,total_rec_int,total_rec_late_fee,recoveries,last_pymnt_d,application_type,"
          "mort_acc,pub_rec_bankruptcies\n")


def _row(id_, term=" 36 months", issue="Dec-2015", status="Fully Paid", emp="10+ years", app="Individual",
         rate="13.56", util="45.2"):
    return (f"{id_},10000,10000,{term},{rate},339.6,C,C1,{emp},RENT,60000,Verified,{issue},{status},"
            f"debt_consolidation,CA,18.5,0,Aug-2003,690,694,1,10,0,5000,{util},20,"
            f"11000,10000,1000,0,0,Jan-2019,{app},1,0\n")


def test_loader_parses_lendingclub_formats(tmp_path):
    raw = tmp_path / "mini lc file.csv.gz"                      # space in the path on purpose
    rows = [
        _row(1),                                                 # kept, good
        _row(2, status="Charged Off", emp="< 1 year", rate=" 13.56%", util="45.2%"),   # kept, bad, % formats
        _row(3, status="Does not meet the credit policy. Status:Charged Off", emp="n/a", issue="Jan-2012"),
        _row(4, term=" 60 months"),                              # dropped: term
        _row(5, issue="Dec-2011"),                               # dropped: date
        _row(6, status="Current"),                               # dropped: no final outcome
        _row(7, app="Joint App"),                                # dropped: joint
        _row(8, status="Default", emp="3 years"),                # kept, bad
    ]
    junk = "Total amount funded in policy code 1: 123456\n"
    with gzip.open(raw, "wt", encoding="utf-8") as fh:
        fh.write(HEADER + "".join(rows) + junk)
    con = load.connect(tmp_path / "t.duckdb")
    out = load.build_base(raw, con)
    df = con.execute("SELECT * FROM loans ORDER BY id").df()
    assert df["id"].astype(int).tolist() == [1, 2, 3, 8]
    assert df["bad"].tolist() == [0, 1, 1, 1]
    assert df["emp_length_yrs"].tolist()[:2] == [10, 0]
    assert df["emp_length_yrs"].isna().tolist()[2]
    assert df["int_rate"].tolist()[:2] == [13.56, 13.56]
    assert df["revol_util"].tolist()[:2] == [45.2, 45.2]
    assert df["fico"].iloc[0] == 692
    assert df["sub_grade_rank"].iloc[0] == 11                   # C1 = 2*5 + 1
    assert df["credit_hist_months"].iloc[0] == 148              # Aug-2003 -> Dec-2015
    funnel = out["funnel"]
    assert funnel["rows"].iloc[0] == 9 and funnel["rows"].iloc[-1] == 4
