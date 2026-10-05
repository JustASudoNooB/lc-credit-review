import numpy as np
import pandas as pd
import pytest

from creditlab import config as C
from creditlab import woe as W


def test_woe_iv_hand_computed():
    # bin 1: 80 good, 20 bad; bin 2: 20 good, 80 bad  (no smoothing so the arithmetic is exact)
    # share of goods = 0.8, 0.2 ; share of bads = 0.2, 0.8
    # WoE = ln(0.8/0.2) = ln 4 = 1.3863, ln(0.2/0.8) = -1.3863
    # IV  = (0.8-0.2)*1.3863 + (0.2-0.8)*(-1.3863) = 1.6636
    woe, ivp = W.woe_iv([100, 100], [20, 80], smoothing=0)
    assert woe == pytest.approx([np.log(4), -np.log(4)])
    assert ivp.sum() == pytest.approx(1.2 * np.log(4))


def test_smoothing_keeps_empty_cells_finite():
    woe, ivp = W.woe_iv([50, 50], [0, 10])
    assert np.isfinite(woe).all() and np.isfinite(ivp).all()


def _monotone_data(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(700, 40, n)
    p = 1 / (1 + np.exp(0.03 * (x - 700) + 2))
    y = (rng.random(n) < p).astype(int)
    x[rng.random(n) < 0.03] = np.nan
    return x, y


def test_numeric_binning_rules():
    x, y = _monotone_data()
    b = W.fit_numeric("fico", x, y)
    t = b.table()
    regular = t[t["label"] != "MISSING"]
    rates = regular["bad_rate"].to_numpy()
    assert b.direction == -1                                   # risk falls as the value rises
    assert np.all(np.diff(rates) <= 1e-12)                     # monotonic
    assert len(regular) <= C.MAX_BINS
    assert (regular["loans"] >= C.MIN_BIN_SHARE * regular["loans"].sum() - 1e-9).all()
    assert b.has_missing and t["label"].iloc[-1] == "MISSING"
    # every training row lands in a valid bin, and counts reconcile
    idx = b.bin_index(x)
    assert (idx >= 0).all()
    assert np.bincount(idx, minlength=len(b.labels)).tolist() == t["loans"].tolist()


def test_numeric_without_missing_maps_new_nan_to_neutral():
    x, y = _monotone_data()
    keep = ~np.isnan(x)
    b = W.fit_numeric("fico", x[keep], y[keep])
    assert not b.has_missing
    assert b.transform(np.array([np.nan]))[0] == 0.0


def test_categorical_rare_pooling_and_unseen():
    rng = np.random.default_rng(1)
    cats = rng.choice(["a", "b", "c", "tiny1", "tiny2"], size=20_000, p=[0.5, 0.3, 0.19, 0.005, 0.005])
    y = (rng.random(20_000) < pd.Series(cats).map({"a": 0.1, "b": 0.2, "c": 0.3, "tiny1": 0.5, "tiny2": 0.5})).astype(int)
    b = W.fit_categorical("purpose", cats, y)
    assert b.cat_to_bin["tiny1"] == b.cat_to_bin["tiny2"] == b.unseen_bin
    # an unseen category is scored with the rare group's evidence
    assert b.transform(["never_seen"])[0] == pytest.approx(b.woe[b.unseen_bin])
    rates = b.bads / b.n
    assert np.all(np.diff(rates) >= -1e-12)                    # groups ordered by bad rate
