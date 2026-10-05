"""Validation pack used for every model: discrimination, calibration, stability."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M

SAMPLES = ("train", "holdout", "oot")


def evaluate(scored: pd.DataFrame, risk_col: str, pd_col: str | None) -> dict:
    """Gini and KS on each sample; calibration and PSI on out-of-time."""
    out = {"by_sample": {}}
    for s in SAMPLES:
        d = scored[scored["sample"] == s]
        out["by_sample"][s] = {
            "loans": int(len(d)),
            "bad_rate": float(d["bad"].mean()),
            "gini": M.gini(d["bad"], d[risk_col]),
            "ks": M.ks(d["bad"], d[risk_col]),
        }
    tr = scored[scored["sample"] == "train"]
    oot = scored[scored["sample"] == "oot"]
    if scored[risk_col].nunique() <= 50:                      # discrete score (sub-grade)
        out["psi_train_vs_oot"] = M.psi_categorical(tr[risk_col], oot[risk_col])
    else:
        out["psi_train_vs_oot"] = M.psi(tr[risk_col], oot[risk_col], C.PSI_BINS)
    out["gini_drop_train_to_oot"] = out["by_sample"]["train"]["gini"] - out["by_sample"]["oot"]["gini"]
    if pd_col:
        out["calibration_oot"] = M.calibration_table(oot["bad"], oot[pd_col], C.CALIBRATION_BINS)
        out["citl_oot"] = M.calibration_in_the_large(oot["bad"], oot[pd_col])
        out["citl_holdout"] = M.calibration_in_the_large(
            scored.loc[scored["sample"] == "holdout", "bad"], scored.loc[scored["sample"] == "holdout", pd_col])
    return out


def csi(binnings: dict, features, X_train: pd.DataFrame, X_oot: pd.DataFrame) -> pd.DataFrame:
    """Characteristic Stability Index: PSI of each feature's bin distribution, train vs OOT."""
    rows = []
    for f in features:
        b = binnings[f]
        rows.append({"feature": f, "csi": M.psi_categorical(b.bin_index(X_train[f].to_numpy()),
                                                            b.bin_index(X_oot[f].to_numpy()))})
    t = pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)
    t["status"] = np.select([t["csi"] < 0.10, t["csi"] < 0.25], ["stable", "watch"], "investigate")
    return t


def strip_tables(ev: dict) -> dict:
    """JSON-safe copy (calibration table as records)."""
    out = {k: v for k, v in ev.items() if k != "calibration_oot"}
    if "calibration_oot" in ev:
        out["calibration_oot"] = ev["calibration_oot"].round(6).to_dict(orient="records")
    return out
