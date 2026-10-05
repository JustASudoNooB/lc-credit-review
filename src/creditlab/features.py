"""Feature lists and the leakage guard.

Model inputs must be known at the moment of the credit decision. This module is
the single place that decides what a model may see, and `assert_no_leakage`
is called by every model before fitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


class LeakageError(ValueError):
    pass


def assert_no_leakage(features) -> None:
    features = list(features)
    banned = [f for f in features if f in C.BANNED_FEATURES or f.startswith(C.BANNED_PREFIXES)]
    champion = [f for f in features if f in C.CHAMPION_ONLY]
    if banned:
        raise LeakageError(f"post-origination fields used as features: {banned}")
    if champion:
        raise LeakageError(f"LendingClub model outputs used as features: {champion}")


def model_frame(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """Numeric features as float (NaN for missing), categoricals as plain str with 'MISSING'."""
    features = list(features or C.CANDIDATE_FEATURES)
    assert_no_leakage(features)
    out = pd.DataFrame(index=df.index)
    for f in features:
        if f in C.CATEGORICAL_FEATURES:
            s = df[f].astype("object")
            out[f] = s.where(s.notna(), "MISSING").astype(str)
        else:
            out[f] = pd.to_numeric(df[f], errors="coerce").astype("float64")
    return out


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dti_band"] = pd.cut(df["dti"], C.DTI_BAND_EDGES, labels=C.DTI_BAND_LABELS, right=False).astype(str)
    df["fico_band"] = pd.cut(df["fico"], C.FICO_BAND_EDGES, labels=C.FICO_BAND_LABELS, right=False).astype(str)
    return df


def y_of(df: pd.DataFrame) -> np.ndarray:
    return df["bad"].astype(int).to_numpy()
