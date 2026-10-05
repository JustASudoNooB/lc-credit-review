import math

import numpy as np
import pandas as pd
import pytest

from creditlab import rca, scorecard, strategy


# ------------------------------------------------------------------ scorecard scaling
def test_scaling_base_point_and_pdo():
    factor, offset = scorecard.scaling(600, 50, 20)
    assert offset + factor * math.log(50) == pytest.approx(600)          # 50:1 odds -> 600
    assert offset + factor * math.log(100) == pytest.approx(620)         # doubling the odds -> +20


def test_points_sum_to_score():
    rng = np.random.default_rng(0)
    n = 20_000
    x1 = rng.normal(700, 40, n)
    x2 = rng.normal(18, 8, n)
    p = 1 / (1 + np.exp(0.03 * (x1 - 700) - 0.04 * (x2 - 18) + 2))
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({"fico": x1, "dti": x2})
    sc, _ = scorecard.build(X, y)
    pts = sc.points_table()
    row = X.iloc[[0]]
    total = 0.0
    for f in sc.features:
        b = sc.binnings[f].bin_index(row[f].to_numpy())[0]
        total += pts.loc[(pts.feature == f) & (pts.bin == b), "points"].iloc[0]
    assert total == pytest.approx(sc.score(row)[0])
    assert all(c < 0 for c in sc.coef)                                   # sign rule on WoE


# ------------------------------------------------------------------ RCA
def test_mix_plus_rate_equals_total():
    rng = np.random.default_rng(2)
    n = 30_000
    df = pd.DataFrame({
        "issue_year": rng.choice([2013, 2015], n),
        "grade": rng.choice(list("ABCDE"), n),
    })
    base = df["grade"].map({"A": 0.05, "B": 0.1, "C": 0.15, "D": 0.2, "E": 0.3})
    df["bad"] = (rng.random(n) < base + 0.03 * (df["issue_year"] == 2015)).astype(int)
    out = rca.mix_rate(df, 2013, 2015, "grade")
    assert out["mix_effect"] + out["rate_effect"] == pytest.approx(out["total_change"], abs=1e-12)
    assert out["bad_rate_1"] == pytest.approx(df.loc[df.issue_year == 2013, "bad"].mean())


def test_mix_rate_hand_example():
    # period 1: 50 A (10% bad), 50 B (30% bad) -> 20%
    # period 2: 20 A (10% bad), 80 B (40% bad) -> 34%
    # mix  = (0.2-0.5)*0.1 + (0.8-0.5)*0.3 = 0.06 ; rate = 0.2*0 + 0.8*0.1 = 0.08 ; total = 0.14
    rows = ([(2013, "A", 1)] * 5 + [(2013, "A", 0)] * 45 + [(2013, "B", 1)] * 15 + [(2013, "B", 0)] * 35
            + [(2015, "A", 1)] * 2 + [(2015, "A", 0)] * 18 + [(2015, "B", 1)] * 32 + [(2015, "B", 0)] * 48)
    df = pd.DataFrame(rows, columns=["issue_year", "grade", "bad"])
    out = rca.mix_rate(df, 2013, 2015, "grade")
    assert out["mix_effect"] == pytest.approx(0.06)
    assert out["rate_effect"] == pytest.approx(0.08)
    assert out["total_change"] == pytest.approx(0.14)


# ------------------------------------------------------------------ strategy
def test_approve_best_hits_exact_rate_with_ties():
    risk = np.repeat(np.arange(10), 100).astype(float)          # 10 tied groups
    a = strategy.approve_best(risk, 0.55)
    assert a.sum() == 550
    assert risk[a].max() <= risk[~a].min()                       # never approves riskier over safer


def test_swap_set_cells_partition_the_book():
    rng = np.random.default_rng(4)
    n = 5000
    oot = pd.DataFrame({"bad": rng.integers(0, 2, n), "funded_amnt": 10000.0,
                        "total_pymnt": rng.uniform(5000, 13000, n)})
    out = strategy.swap_set(oot, rng.random(n), rng.random(n), 0.8, "A", "B")
    t = out["table"]
    assert t["loans"].sum() == n
    assert t.loc[t.cell.str.startswith("Swap-in"), "loans"].iloc[0] == t.loc[t.cell.str.startswith("Swap-out"), "loans"].iloc[0]
