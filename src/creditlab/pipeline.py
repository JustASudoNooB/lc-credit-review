"""Runs stages 0-5 and writes outputs/results.json plus every table in outputs/tables/.

Nothing in reports/ or in the CV bullets is typed by hand: report.py reads results.json.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import load, models, monitoring, rca, scorecard, strategy, validation
from . import metrics as M


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return jsonable(o.to_dict(orient="records"))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return o


def _save(df: pd.DataFrame, name: str) -> None:
    C.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLES / f"{name}.csv", index=False)


def run(raw_path: Path, log=print) -> dict:
    t0 = time.time()
    synthetic = "SYNTHETIC" in raw_path.name.upper()
    R: dict = {"meta": {
        "data_source": raw_path.name,
        "synthetic_test_data": synthetic,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
    }}

    # ------------------------------------------------------------------ stage 0
    log("[0] parsing raw file and building the population ...")
    con = load.connect()
    built = load.build_base(raw_path, con)
    prof = load.profile(con)
    excl = load.excluded_statuses(con)
    df = load.export_population(con)
    _save(built["funnel"], "data_quality_funnel")
    _save(excl, "excluded_statuses")
    for k, v in prof.items():
        _save(v, f"profile_{k}")
    R["data"] = {
        "funnel": built["funnel"],
        "excluded_statuses": excl,
        "population_loans": int(len(df)),
        "sample_sizes": df["sample"].value_counts().to_dict(),
        "bad_rate_by_sample": df.groupby("sample")["bad"].mean().to_dict(),
        "bad_rate_by_year": prof["bad_rate_by_year"],
        "payment_identity": prof["payment_identity"],
        "missing_rates": prof["missing_rates"],
    }

    # ------------------------------------------------------------------ stage 1
    log("[1] SQL monitoring tables ...")
    mon = monitoring.run(con)
    R["monitoring"] = monitoring.summary(mon)
    con.close()

    # ------------------------------------------------------------------ stage 2
    log("[2] WoE scorecard ...")
    X = F.model_frame(df)
    y = F.y_of(df)
    tr = (df["sample"] == "train").to_numpy()
    oot = (df["sample"] == "oot").to_numpy()
    sc, diag = scorecard.build(X[tr], y[tr])
    df["sc_score"] = sc.score(X)
    df["sc_log_odds"] = sc.log_odds_bad(X)
    df["sc_pd"] = sc.pd(X)
    _save(diag["iv_table"], "iv_table")
    _save(pd.concat([b.table() for b in diag["all_binnings"].values()]), "woe_bins_all_candidates")
    _save(sc.points_table(), "scorecard_points")
    _save(diag["vif"].rename_axis("feature").reset_index(), "vif")
    csi = validation.csi(sc.binnings, sc.features, X[tr], X[oot])
    _save(csi, "csi_train_vs_oot")
    ev_sc = validation.evaluate(df, "sc_pd", "sc_pd")
    R["scorecard"] = {
        "features": sc.features,
        "n_candidates": len(C.CANDIDATE_FEATURES),
        "selection_log": sc.selection_log,
        "iv": diag["iv_table"][["feature", "iv", "iv_band"]],
        "vif_max": float(diag["vif"].max()),
        "coefficients": dict(zip(sc.features, map(float, sc.coef))),
        "intercept": sc.intercept,
        "score_range_oot": [float(df.loc[oot, "sc_score"].min()), float(df.loc[oot, "sc_score"].max())],
        "evaluation": validation.strip_tables(ev_sc),
        "csi": csi,
        "excluded_with_reason": C.EXCLUDED_WITH_REASON,
    }

    # ------------------------------------------------------------------ stage 3
    log("[3] champion vs challenger ...")
    df["champ_pd"] = models.champion_pd(df)
    df["champ_rank"] = df["sub_grade_rank"].astype(float)
    ch = models.Challenger(C.CANDIDATE_FEATURES).fit(X[tr], y[tr])
    df["ch_pd"] = ch.pd(X)
    df["ov_pd"], ov_coef = models.fit_overlay(df["champ_pd"].to_numpy(), df["sc_log_odds"].to_numpy(), y, tr)

    ev = {
        "champion": validation.evaluate(df, "champ_rank", "champ_pd"),
        "scorecard": ev_sc,
        "challenger": validation.evaluate(df, "ch_pd", "ch_pd"),
        "overlay": validation.evaluate(df, "ov_pd", "ov_pd"),
    }
    d_oot = df[oot]
    risks = {"champion": d_oot["champ_rank"], "scorecard": d_oot["sc_pd"],
             "challenger": d_oot["ch_pd"], "overlay": d_oot["ov_pd"]}
    boot = M.bootstrap_gini(
        d_oot["bad"], risks, C.N_BOOTSTRAP, C.SEED, baseline="champion")
    boot_vs_sc = M.bootstrap_gini(
        d_oot["bad"], {"scorecard": d_oot["sc_pd"], "challenger": d_oot["ch_pd"]},
        C.N_BOOTSTRAP, C.SEED, baseline="scorecard")
    imp = ch.importance(X[oot], y[oot])
    _save(imp, "challenger_permutation_importance")
    pdp = pd.concat([models.partial_dependence_check(ch, X[oot], f) for f in imp["feature"].head(4)])
    _save(pdp, "challenger_partial_dependence_top4")
    for name, e in ev.items():
        if "calibration_oot" in e:
            _save(e["calibration_oot"].assign(model=name), f"calibration_oot_{name}")
    R["model_review"] = {
        "evaluation": {k: validation.strip_tables(v) for k, v in ev.items()},
        "bootstrap_oot_vs_champion": boot,
        "bootstrap_challenger_vs_scorecard": boot_vs_sc["challenger"],
        "challenger_iterations": int(ch.model.n_iter_),
        "challenger_monotone": C.CHALLENGER_MONOTONE,
        "challenger_importance": imp,
        "overlay_coefficients": ov_coef,
        "pdp_direction": _pdp_direction(pdp),
    }

    # ------------------------------------------------------------------ stage 4
    log("[4] underwriting strategy ...")
    d_oot = df[oot].reset_index(drop=True)
    sw_sc = strategy.swap_set(d_oot, d_oot["champ_rank"], d_oot["sc_pd"], C.TARGET_APPROVAL, "LC sub-grade", "scorecard")
    sw_ch = strategy.swap_set(d_oot, d_oot["champ_rank"], d_oot["ch_pd"], C.TARGET_APPROVAL, "LC sub-grade", "challenger")
    _save(sw_sc["table"], "swap_set_grade_vs_scorecard")
    _save(sw_ch["table"], "swap_set_grade_vs_challenger")
    cut = strategy.cutoff_table(d_oot, {"LC sub-grade": d_oot["champ_rank"], "scorecard": d_oot["sc_pd"],
                                        "challenger": d_oot["ch_pd"]})
    _save(cut, "cutoff_strategy")
    dti_pre = strategy.dti_cap_tests(d_oot, "dti", C.DTI_CAPS_PRE, d_oot["sc_score"].to_numpy())
    dti_post = strategy.dti_cap_tests(d_oot, "dti_post", C.DTI_CAPS_POST, d_oot["sc_score"].to_numpy())
    _save(pd.concat([dti_pre["overall"], dti_post["overall"]]), "dti_cap_overall")
    _save(pd.concat([dti_pre["within_bands"], dti_post["within_bands"]]), "dti_cap_within_score_bands")
    best_ret = cut.loc[cut.groupby("model")["net_return"].idxmax()]
    R["strategy"] = {
        "swap_vs_scorecard": {k: v for k, v in sw_sc.items() if k != "table"} | {"table": sw_sc["table"]},
        "swap_vs_challenger": {k: v for k, v in sw_ch.items() if k != "table"} | {"table": sw_ch["table"]},
        "return_maximising_cutoff": best_ret,
        "cutoff_at_target": cut[cut["approval_rate"] == C.TARGET_APPROVAL],
        "dti_pre": {"base": dti_pre["base"], "overall": dti_pre["overall"],
                    "bands_significant": _sig_summary(dti_pre["within_bands"])},
        "dti_post": {"base": dti_post["base"], "overall": dti_post["overall"],
                     "bands_significant": _sig_summary(dti_post["within_bands"])},
        "reference_cap_post": C.REFERENCE_CAP_POST,
    }

    # ------------------------------------------------------------------ stage 5
    log("[5] RCA and early warning ...")
    df = F.add_bands(df)
    mr = rca.mix_rate(df, C.RCA_PERIOD_1, C.RCA_PERIOD_2, "grade")
    _save(mr["table"], "rca_mix_rate_by_grade")
    dims = ["purpose", "verification_status", "dti_band", "fico_band", "home_ownership"]
    drills = pd.concat([rca.drilldown(df, C.RCA_PERIOD_1, C.RCA_PERIOD_2, d) for d in dims])
    _save(drills, "rca_within_grade_drilldown")
    ew = rca.early_warning(df, ["grade", "verification_status", "purpose", "dti_band", "fico_band"])
    _save(ew, "early_warning")
    # contributions are additive within a dimension, not across dimensions: report the top segment of each
    top = (drills.sort_values("rate_contribution", ascending=False)
                 .groupby("dimension", sort=False).head(1).reset_index(drop=True))
    R["rca"] = {k: v for k, v in mr.items() if k != "table"} | {
        "mix_share_of_change": mr["mix_effect"] / mr["total_change"] if mr["total_change"] else None,
        "top_within_grade_drivers": top,
        "early_warning_first": ew.dropna(subset=["first_breach_quarter"]).head(5),
        "segments_breached": int(ew["first_breach_quarter"].notna().sum()),
        "segments_checked": int(len(ew)),
    }

    keep = ["id", "issue_d", "issue_year", "issue_q", "sample", "bad", "grade", "sub_grade", "sub_grade_rank",
            "funded_amnt", "total_pymnt", "dti", "dti_post", "fico", "purpose", "verification_status",
            "sc_score", "sc_pd", "champ_pd", "ch_pd", "ov_pd", "dti_band", "fico_band"]
    df[keep].to_parquet(C.SCORED_PARQUET, index=False)
    R["meta"]["runtime_seconds"] = round(time.time() - t0, 1)
    C.RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    C.RESULTS_JSON.write_text(json.dumps(jsonable(R), indent=2), encoding="utf-8")
    log(f"done in {R['meta']['runtime_seconds']}s -> {C.RESULTS_JSON}")
    return R


def _pdp_direction(pdp: pd.DataFrame) -> dict:
    out = {}
    for f, g in pdp.groupby("feature"):
        try:
            v = g["avg_pd"].to_numpy()
            x = pd.to_numeric(g["value"], errors="coerce").to_numpy()
            if np.isnan(x).any():
                out[f] = "categorical"
            else:
                out[f] = "rises" if v[-1] > v[0] else "falls"
        except Exception:
            out[f] = "n/a"
    return out


def _sig_summary(within: pd.DataFrame) -> list:
    g = within.groupby(["dti_measure", "cap"]).agg(bands=("score_band", "size"),
                                                   significant_bands=("significant_5pct", "sum"),
                                                   avg_lift_pp=("lift_pp", "mean")).reset_index()
    return g.to_dict(orient="records")
