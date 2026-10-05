import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from creditlab import metrics as M


def _data(n=5000, seed=3):
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(risk * 1.2 - 1.5)))).astype(int)
    return y, risk


def test_gini_matches_auc():
    y, r = _data()
    assert M.gini(y, r) == pytest.approx(2 * roc_auc_score(y, r) - 1)


def test_ks_matches_scipy_two_sample():
    y, r = _data()
    assert M.ks(y, r) == pytest.approx(ks_2samp(r[y == 1], r[y == 0]).statistic)


def test_ks_handles_ties_like_scipy():
    y, r = _data()
    r = np.round(r, 1)                  # heavy ties, like a 35-level sub-grade
    assert M.ks(y, r) == pytest.approx(ks_2samp(r[y == 1], r[y == 0]).statistic)


def test_psi_identical_is_zero_and_known_value():
    assert M.psi_from_shares([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    # (0.6-0.5)*ln(0.6/0.5) + (0.4-0.5)*ln(0.4/0.5) = 0.1*ln1.2 + 0.1*ln1.25
    assert M.psi_from_shares([0.5, 0.5], [0.6, 0.4]) == pytest.approx(0.1 * np.log(1.2) + 0.1 * np.log(1.25))


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    assert M.psi(a, rng.normal(0, 1, 20000)) < 0.01
    assert M.psi(a, rng.normal(0.5, 1, 20000)) > 0.1


def test_bootstrap_ci_brackets_point_and_paired_diff():
    y, r = _data()
    noisy = r + np.random.default_rng(9).normal(0, 1.0, len(r))
    out = M.bootstrap_gini(y, {"a": r, "b": noisy}, n_boot=100, seed=1, baseline="a")
    assert out["a"]["ci_low"] <= out["a"]["gini"] <= out["a"]["ci_high"]
    assert out["b"]["diff_vs_a"] < 0 and out["b"]["diff_ci_high"] < 0      # noisier score ranks worse


def test_two_proportion_z_known_value():
    # 30/100 vs 20/100: pooled p = 0.25, se = sqrt(0.25*0.75*0.02) = 0.061237, z = 0.1/0.061237 = 1.633
    z, p = M.two_proportion_z(30, 100, 20, 100)
    assert z == pytest.approx(1.63299, rel=1e-4)
    assert p == pytest.approx(0.10247, rel=1e-3)


def test_calibration_in_the_large():
    c = M.calibration_in_the_large([0, 1, 0, 1], [0.25, 0.25, 0.25, 0.25])
    assert c["gap_pp"] == pytest.approx(25.0)
