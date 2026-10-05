"""Writes every narrative document from outputs/results.json. No number below is typed by hand."""
from __future__ import annotations

import json

import pandas as pd

from . import config as C

BANNER = ("> **SYNTHETIC TEST DATA.** These numbers come from a fake file used to test the code. "
          "They are not results and must not be quoted anywhere. Re-run `python run_all.py` on the real "
          "LendingClub file.\n\n")


def pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def ppf(x, d=2):
    """x is a fraction (0.0123) -> '+1.23 pp'."""
    return "n/a" if x is None else f"{x * 100:+.{d}f} pp"


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


class Facts:
    """Every number the documents use, pulled from results.json in one place."""

    def __init__(self, R: dict):
        self.R = R
        self.syn = R["meta"]["synthetic_test_data"]
        mr = R["model_review"]
        self.ev = mr["evaluation"]
        self.boot = mr["bootstrap_oot_vs_champion"]
        self.boot_cs = mr["bootstrap_challenger_vs_scorecard"]
        self.g = {k: self.boot[k]["gini"] for k in ("champion", "scorecard", "challenger", "overlay")}
        self.ks = {k: self.ev[k]["by_sample"]["oot"]["ks"] for k in self.g}
        self.psi = {k: self.ev[k]["psi_train_vs_oot"] for k in self.g}
        self.citl = {k: self.ev[k]["citl_oot"] for k in self.g}
        self.n_pop = R["data"]["population_loans"]
        self.n_oot = R["data"]["sample_sizes"].get("oot", 0)
        self.rca = R["rca"]
        st = R["strategy"]
        self.best_model = "challenger" if self.g["challenger"] >= self.g["scorecard"] else "scorecard"
        self.swap = st["swap_vs_challenger"] if self.best_model == "challenger" else st["swap_vs_scorecard"]
        self.swap_rows = {r["cell"].split(" (")[0]: r for r in self.swap["table"]}
        post = pd.DataFrame(st["dti_post"]["overall"])
        cap = st["reference_cap_post"]
        self.cap = cap
        self.cap_row = post[post["cap"] == cap].iloc[0].to_dict() if (post["cap"] == cap).any() else None
        sig = [s for s in st["dti_post"]["bands_significant"] if s["cap"] == cap]
        self.cap_sig = sig[0] if sig else None
        self.sc_feats = R["scorecard"]["features"]

    # ---- verdict for the model review
    def verdict(self) -> tuple[str, list[str]]:
        ch, ov = self.boot["challenger"], self.boot["overlay"]
        conditions = []
        if ch["diff_ci_low"] > 0:
            v = ("APPROVE WITH CONDITIONS: the challenger ranks 2015 loans better than the LendingClub "
                 f"sub-grade (Gini {ch['diff_vs_champion']:+.3f}, 95% CI {ch['diff_ci_low']:+.3f} "
                 f"to {ch['diff_ci_high']:+.3f}).")
        elif ov["diff_ci_low"] > 0:
            v = ("APPROVE AS AN OVERLAY, NOT A REPLACEMENT: neither new model beats the sub-grade on its own, "
                 "but adding the scorecard to the sub-grade improves ranking "
                 f"(Gini {ov['diff_vs_champion']:+.3f}, 95% CI {ov['diff_ci_low']:+.3f} to {ov['diff_ci_high']:+.3f}), "
                 "so the scorecard carries information the grade does not.")
        else:
            v = ("REJECT AS A REPLACEMENT; RETAIN AS A BENCHMARK: neither model, alone or as an overlay, "
                 "ranks 2015 loans significantly better than the LendingClub sub-grade.")
        for k, label in (("scorecard", "scorecard"), ("challenger", "challenger")):
            gap = self.citl[k]["gap_pp"]
            if abs(gap) >= 1.0:
                word = "under-predicts" if gap > 0 else "over-predicts"
                conditions.append(f"The {label} {word} the 2015 bad rate by {abs(gap):.1f} pp: recalibrate the "
                                  "intercept on the most recent complete vintage before any PD is used for pricing "
                                  "or provisioning.")
        for k in ("scorecard", "challenger"):
            p = self.psi[k]
            if p >= 0.10:
                conditions.append(f"Score PSI for the {k} is {p:.3f} (>= 0.10): investigate the population shift.")
        drop = self.ev["challenger"]["gini_drop_train_to_oot"]
        if drop > 0.05:
            conditions.append(f"The challenger loses {drop:.3f} Gini from train to out-of-time; it fits the "
                              "training sample more closely than it generalises, so monitor it monthly.")
        bad_dirs = [f for f, d in self.R["model_review"]["pdp_direction"].items()
                    if d in ("rises", "falls") and f in C.EXPECTED_DIRECTION
                    and (d == "rises") != (C.EXPECTED_DIRECTION[f] > 0)]
        if bad_dirs:
            conditions.append(f"Top drivers moving against credit intuition: {', '.join(bad_dirs)}. "
                              "Investigate before use.")
        conditions.append("Monitor monthly: score PSI and feature CSI (watch at 0.10, act at 0.25), Gini on each "
                          "new matured vintage, and calibration by decile.")
        return v, conditions


# =============================================================================== documents
def model_review(F: Facts) -> str:
    R = F.R
    d = R["data"]
    s = (BANNER if F.syn else "") + "# Model review: application scorecard and GBM challenger vs LendingClub sub-grade\n\n"
    s += f"Data: `{R['meta']['data_source']}`. Run: {R['meta']['run_utc']}. Every number below is read from `outputs/results.json`.\n\n"
    s += ("## 1. Purpose\n\nLendingClub's sub-grade (A1 to G5) is treated as the incumbent underwriting model. "
          "Two candidate models are built only from information available at application time and reviewed "
          "the way a model risk team reviews a decision model: discrimination, calibration, stability, "
          "driver sense-check, limitations, verdict.\n\n")
    s += "## 2. Data\n\nPopulation filters (row count after each step):\n\n"
    s += _md_table(pd.DataFrame(d["funnel"])) + "\n\n"
    ss = d["sample_sizes"]
    br = d["bad_rate_by_sample"]
    s += (f"Target: bad = charged off or default; good = fully paid. Loans still open at the snapshot are "
          f"excluded (see `outputs/tables/excluded_statuses.csv`).\n\n"
          f"| sample | definition | loans | bad rate |\n|---|---|---|---|\n"
          f"| train | 2012-2014, random 80% | {ss.get('train', 0):,} | {pct(br.get('train'))} |\n"
          f"| holdout | 2012-2014, random 20% | {ss.get('holdout', 0):,} | {pct(br.get('holdout'))} |\n"
          f"| out-of-time | 2015 | {ss.get('oot', 0):,} | {pct(br.get('oot'))} |\n\n")
    sc = R["scorecard"]
    s += "## 3. Design choices\n\n"
    s += (f"- {sc['n_candidates']} candidate application-time features. Excluded on purpose:\n")
    for k, v in sc["excluded_with_reason"].items():
        s += f"  - `{k}`: {v}\n"
    s += (f"- Binning: 20 quantile fine classes, merged to monotonic bad rates, minimum {C.MIN_BIN_SHARE:.0%} "
          f"of rows per bin, at most {C.MAX_BINS} bins, missing values in their own bin.\n"
          f"- Selection: IV >= {C.IV_MIN}, pairwise |correlation| <= {C.CORR_MAX} (keep higher IV), every "
          "coefficient negative on WoE.\n")
    s += "- Selection log:\n" + "".join(f"  - {line}\n" for line in sc["selection_log"])
    s += f"- Final scorecard features ({len(sc['features'])}): " + ", ".join(f"`{f}`" for f in sc["features"]) + "\n"
    s += f"- Largest VIF among final features: {sc['vif_max']:.2f}\n"
    s += (f"- Scaling: {C.BASE_SCORE} points at {C.BASE_ODDS}:1 good:bad odds, {C.PDO} points to double the odds "
          "(`outputs/tables/scorecard_points.csv`).\n")
    mono = ", ".join(f"`{k}` {'up' if v > 0 else 'down'}" for k, v in C.CHALLENGER_MONOTONE.items())
    s += (f"- Challenger: histogram gradient boosting on raw features, early stopping on 15% of train "
          f"({R['model_review']['challenger_iterations']} trees kept), monotone constraints: {mono}.\n"
          "- Overlay: logistic regression on [logit of grade-implied PD, scorecard log-odds], fitted on train.\n\n")
    s += "## 4. Discrimination\n\n"
    rows = []
    names = {"champion": "LC sub-grade (champion)", "scorecard": "WoE scorecard", "challenger": "GBM challenger",
             "overlay": "Grade + scorecard overlay"}
    for k, lab in names.items():
        e = F.ev[k]["by_sample"]
        b = F.boot[k]
        diff = "-" if k == "champion" else f"{b['diff_vs_champion']:+.3f} ({b['diff_ci_low']:+.3f} to {b['diff_ci_high']:+.3f})"
        rows.append({"model": lab, "train Gini": f3(e["train"]["gini"]), "holdout Gini": f3(e["holdout"]["gini"]),
                     "OOT Gini (95% CI)": f"{b['gini']:.3f} ({b['ci_low']:.3f}-{b['ci_high']:.3f})",
                     "OOT KS": f3(e["oot"]["ks"]), "OOT Gini vs champion (95% CI)": diff})
    s += _md_table(pd.DataFrame(rows)) + "\n\n"
    cs = F.boot_cs
    s += (f"Challenger minus scorecard on OOT: {cs['diff_vs_scorecard']:+.3f} Gini "
          f"(95% CI {cs['diff_ci_low']:+.3f} to {cs['diff_ci_high']:+.3f}). Confidence intervals come from "
          f"{C.N_BOOTSTRAP} paired bootstrap resamples of the 2015 loans.\n\n")
    s += "![Gini with confidence intervals](../outputs/figures/gini_with_ci.png)\n\n"
    s += "## 5. Stability\n\n| model | score PSI, train vs 2015 |\n|---|---|\n"
    for k, lab in names.items():
        s += f"| {lab} | {F.psi[k]:.3f} |\n"
    s += "\nFeature-level CSI (scorecard features):\n\n" + _md_table(pd.DataFrame(sc["csi"]).round(4)) + "\n\n"
    s += "## 6. Calibration\n\n| model | predicted 2015 bad rate | actual | gap |\n|---|---|---|---|\n"
    for k, lab in names.items():
        c = F.citl[k]
        s += f"| {lab} | {pct(c['predicted_bad_rate'], 2)} | {pct(c['actual_bad_rate'], 2)} | {c['gap_pp']:+.2f} pp |\n"
    s += ("\nA PD fitted on 2012-2014 is applied to 2015 loans. If 2015 performed worse than earlier vintages at the "
          "same risk profile, the model ranks correctly but under-predicts the level: that is a calibration "
          "problem, fixed by re-estimating the intercept, not by rebuilding the model.\n\n")
    s += "![Calibration](../outputs/figures/calibration_oot.png)\n\n"
    s += "## 7. Challenger drivers\n\n"
    imp = pd.DataFrame(R["model_review"]["challenger_importance"]).head(8)[["feature", "gini_drop"]].round(4)
    s += "Permutation importance on 2015 loans (Gini lost when the feature is shuffled):\n\n" + _md_table(imp) + "\n\n"
    s += "Direction check on the top drivers (average predicted PD as the feature is swept low to high):\n\n"
    s += "| feature | model | expected |\n|---|---|---|\n"
    for f, dct in R["model_review"]["pdp_direction"].items():
        exp = C.EXPECTED_DIRECTION.get(f)
        s += f"| {f} | {dct} | {'rises' if exp == 1 else 'falls' if exp == -1 else 'n/a'} |\n"
    s += ("\n## 8. Limitations\n\n"
          "- Reject inference: only loans LendingClub approved are observed. Every model here is trained and "
          "tested on loans the champion had already accepted, so none of these results says how the models "
          "would rank the applicants LendingClub declined.\n"
          "- The sub-grade uses bureau and platform data this dataset does not contain, so the comparison is "
          "partly a test of data, not only of method.\n"
          "- 2012-2015 is one benign US credit cycle. No recession is in the sample.\n"
          "- US unsecured personal loans. An Indian personal-loan book differs in bureau depth, income "
          "verification and collections; the method transfers, the coefficients do not.\n"
          "- Realised return ignores servicing fees, funding cost and time value of money.\n\n")
    v, conds = F.verdict()
    s += f"## 9. Conclusion\n\n**{v}**\n\nConditions:\n\n" + "".join(f"1. {c}\n" for c in conds)
    return s


def risk_memo(F: Facts) -> str:
    R = F.R
    r = F.rca
    up = r["total_change"] >= 0
    s = BANNER if F.syn else ""
    s += ("# Risk committee memo\n\n**Subject:** 2015 vintage performance, decision-model review and two "
          "underwriting options\n\n**Prepared by:** Akhand Raj  \n**Basis:** "
          f"{F.n_pop:,} LendingClub 36-month personal loans issued 2012-2015, all at final outcome\n\n")
    s += "## Findings\n\n"
    top = pd.DataFrame(r["top_within_grade_drivers"])
    big = top.sort_values("avg_rate_change_pp", ascending=False).iloc[0] if len(top) else None
    ew = pd.DataFrame(r["early_warning_first"])
    ew_txt = (f" The earliest sustained warning came from **{ew.iloc[0]['dimension']} = {ew.iloc[0]['segment']}** "
              f"in {ew.iloc[0]['first_breach_quarter']}." if len(ew) else "")
    s += (f"1. **The bad rate {'rose' if up else 'fell'} from {pct(r['bad_rate_1'])} ({r['period_1']} loans) to "
          f"{pct(r['bad_rate_2'])} ({r['period_2']} loans), {ppf(r['total_change'])}.** "
          f"Lending more to riskier grades explains {ppf(r['mix_effect'])}; the same grades performing worse "
          f"explains {ppf(r['rate_effect'])}.")
    if big is not None:
        s += (f" Inside grades, the sharpest deterioration was in **{big['dimension']} = {big['segment']}** "
              f"({big['avg_rate_change_pp']:+.1f} pp, {pct(big['share_period_2'], 0)} of {r['period_2']} loans).")
    s += ew_txt + "\n"
    g = F.g
    c_sc, c_ch = F.citl["scorecard"]["gap_pp"], F.citl["challenger"]["gap_pp"]
    def word(gap):
        return "under-predicts" if gap > 0 else "over-predicts"
    s += (f"2. **On 2015 loans, the LendingClub sub-grade ranks risk with Gini {g['champion']:.3f}; my application "
          f"scorecard {g['scorecard']:.3f}; the gradient-boosting challenger {g['challenger']:.3f}; grade plus "
          f"scorecard together {g['overlay']:.3f}.** Both new models were built on 2012-2014 data: the scorecard "
          f"{word(c_sc)} the 2015 bad rate by {abs(c_sc):.1f} pp and the challenger {word(c_ch)} it by "
          f"{abs(c_ch):.1f} pp.\n")
    si, so = F.swap_rows.get("Swap-in"), F.swap_rows.get("Swap-out")
    s += (f"3. **At an {C.TARGET_APPROVAL:.0%} approval rate, replacing the sub-grade with the {F.best_model} "
          f"swaps {si['loans']:,} loans: those brought in go bad at {pct(si['bad_rate'])}, those dropped at "
          f"{pct(so['bad_rate'])}** (portfolio bad rate {pct(F.swap['policy_a']['bad_rate'])} on the sub-grade "
          f"vs {pct(F.swap['policy_b']['bad_rate'])} on the {F.best_model}).")
    if F.cap_row:
        cr = F.cap_row
        s += (f" A post-loan DTI cap of {F.cap} (existing debts plus the new instalment) would decline "
              f"{pct(cr['declined_share'], 0)} of loans with a {pct(cr['bad_rate_declined'])} bad rate against "
              f"{pct(cr['bad_rate_kept'])} for those kept")
        if F.cap_sig:
            s += (f"; inside score quintiles the cap separates risk significantly in "
                  f"{F.cap_sig['significant_bands']} of {F.cap_sig['bands']}")
        s += ".\n"
    else:
        s += "\n"
    s += "\n## Recommendations\n\n"
    v, _ = F.verdict()
    if v.startswith("APPROVE WITH"):
        rec1 = (f"**Pilot the challenger** as the primary rank-ordering model on a champion/challenger split, "
                f"after recalibrating its intercept to the latest complete vintage.")
    elif v.startswith("APPROVE AS AN OVERLAY"):
        rec1 = ("**Keep the sub-grade as the primary model and add the scorecard as an overlay** (a second look "
                "at the cut-off margin), after recalibrating its intercept to the latest complete vintage.")
    else:
        rec1 = ("**Keep the sub-grade.** Retain the scorecard as an independent benchmark that is re-run on each "
                "new vintage to detect drift in the incumbent.")
    s += f"1. {rec1}\n"
    if F.cap_sig and F.cap_row:
        k, n = F.cap_sig["significant_bands"], F.cap_sig["bands"]
        if k >= (n + 1) // 2 + 1:
            rec2 = f"**Add a post-loan DTI cap of {F.cap}** as a hard policy rule: it finds risk the score has not priced in most score bands."
        elif k >= 1:
            rec2 = (f"**Apply a post-loan DTI cap of {F.cap} only in the score bands where it adds signal** "
                    f"({k} of {n}); elsewhere the score already prices the debt burden.")
        else:
            rec2 = (f"**Do not add a post-loan DTI cap of {F.cap} on top of the score**: inside score bands it "
                    "does not separate risk, so it would decline good loans for no gain.")
        s += f"2. {rec2}\n"
    watch = f"{ew.iloc[0]['dimension']} = {ew.iloc[0]['segment']}" if len(ew) else "the segments in early_warning.csv"
    s += (f"3. **Monitor monthly:** score PSI and feature CSI (watch at 0.10, act at 0.25), the vintage curves at "
          f"months-on-book 6 and 12, and the early-warning list, starting with {watch}.\n\n")
    s += ("Supporting detail: `reports/MODEL_REVIEW.md`, `outputs/tables/`, `outputs/figures/`.\n")
    return s


def cv_bullets(F: Facts) -> str:
    r = F.rca
    up = r["total_change"] >= 0
    g = F.g
    gap = F.citl[F.best_model]["gap_pp"]
    psi = F.psi[F.best_model]
    si, so = F.swap_rows.get("Swap-in"), F.swap_rows.get("Swap-out")
    b1 = (f"Built SQL vintage monitoring (DuckDB, CTEs, window functions) on {F.n_pop:,} LendingClub personal loans; "
          f"split the {r['period_1']}-{r['period_2']} bad-rate {'rise' if up else 'fall'} of "
          f"{abs(r['total_change']) * 100:.1f} pp into {r['mix_effect'] * 100:.1f} pp grade mix and "
          f"{r['rate_effect'] * 100:.1f} pp within-grade change.")
    cal = (f"; flagged a {abs(gap):.1f} pp {'under' if gap > 0 else 'over'}-prediction for recalibration"
           if abs(gap) >= 0.5 else f"; calibration within {abs(gap):.1f} pp")
    psi_txt = f"{psi:.2f}" if psi >= 0.01 else "below 0.01"
    b2 = (f"Reviewed LendingClub's grade vs a WoE logistic scorecard and a monotone GBM challenger on "
          f"{F.n_oot:,} out-of-time loans (Gini {g['champion']:.2f} / {g['scorecard']:.2f} / "
          f"{g['challenger']:.2f}); score PSI {psi_txt}{cal}.")
    b3a = (f"Back-tested underwriting at {C.TARGET_APPROVAL:.0%} approval with a swap-set analysis: moving from grade to "
           f"the {F.best_model} swaps {si['loans']:,} loans in at a {si['bad_rate'] * 100:.1f}% bad rate and out at "
           f"{so['bad_rate'] * 100:.1f}%.")
    b3b = ""
    if F.cap_row:
        cr = F.cap_row
        b3b = (f"Tested a post-loan DTI cap of {F.cap}: declines {cr['declined_share'] * 100:.0f}% of loans at a "
               f"{cr['bad_rate_declined'] * 100:.1f}% bad rate vs {cr['bad_rate_kept'] * 100:.1f}% kept"
               + (f", significant in {F.cap_sig['significant_bands']} of {F.cap_sig['bands']} score bands." if F.cap_sig else "."))

    def tex(x):
        return x.replace("%", "\\%").replace("&", "\\&")

    s = BANNER if F.syn else ""
    s += ("# CV bullets (generated from results.json)\n\nPick bullet 1, bullet 2, and one of 3a / 3b. "
          "Edit wording if you like, but keep every number exactly as printed.\n\n")
    for name, b in (("1", b1), ("2", b2), ("3a", b3a), ("3b", b3b)):
        if b:
            s += f"**{name}.** {b}  \n_{len(b)} characters_\n\n```latex\n\\item {tex(b)}\n```\n\n"
    return s


def calib_sentence(F: Facts) -> str:
    a, b = F.citl["scorecard"]["gap_pp"], F.citl["challenger"]["gap_pp"]
    hold = F.ev["scorecard"]["by_sample"]["holdout"]["gini"] - F.ev["scorecard"]["by_sample"]["oot"]["gini"]
    if max(abs(a), abs(b)) < 0.5:
        return f"Calibration on 2015 held within {max(abs(a), abs(b)):.1f} pp for both models."
    if a > 0 and b > 0:
        lead = f"Both models under-predicted the 2015 bad rate, by {a:.1f} and {b:.1f} pp."
    elif a < 0 and b < 0:
        lead = f"Both models over-predicted the 2015 bad rate, by {abs(a):.1f} and {abs(b):.1f} pp."
    else:
        lead = f"Calibration on 2015 was off by {a:+.1f} pp (scorecard) and {b:+.1f} pp (challenger)."
    if hold < 0.03:
        lead += (" The scorecard's Gini barely moved from holdout to 2015, so the ranking held and only the level "
                 "shifted: that is fixed by recalibrating the intercept, not by rebuilding the model.")
    return lead


def pitch(F: Facts) -> str:
    r = F.rca
    g = F.g
    v, _ = F.verdict()
    s = BANNER if F.syn else ""
    s += "# Two-minute project pitch (numbers filled from results.json)\n\n"
    s += (f"I took {F.n_pop:,} LendingClub 36-month personal loans issued 2012 to 2015, all old enough to have a final "
          f"outcome, and ran them the way a lender's credit risk team would.\n\n"
          f"First, monitoring. In SQL I built static-pool bad rates and vintage curves by issue quarter. The bad rate "
          f"went from {pct(r['bad_rate_1'])} for {r['period_1']} loans to {pct(r['bad_rate_2'])} for {r['period_2']}. "
          f"I split that change: {ppf(r['mix_effect'])} came from lending more to riskier grades, "
          f"{ppf(r['rate_effect'])} from the same grades performing worse.\n\n"
          f"Second, a model review. I treated LendingClub's own sub-grade as the incumbent model and built two "
          f"challengers using only application data: a weight-of-evidence logistic scorecard and a gradient-boosting "
          f"model with monotone constraints. Trained on 2012-2014, tested on 2015. Gini: sub-grade "
          f"{g['champion']:.2f}, scorecard {g['scorecard']:.2f}, boosting {g['challenger']:.2f}. My verdict: "
          f"{v.split(':')[0].lower()}. {calib_sentence(F)}\n\n"
          f"Third, strategy. At an {C.TARGET_APPROVAL:.0%} approval rate I compared which loans each model would "
          f"approve, and tested a debt-to-income cap that includes the new instalment, inside score bands, to see "
          f"whether it finds risk the score misses.\n\n"
          f"The biggest limitation is reject inference: I only see loans LendingClub approved.\n")
    return s


def readme(F: Facts) -> str:
    R = F.R
    r = F.rca
    g = F.g
    v, _ = F.verdict()
    s = BANNER if F.syn else ""
    s += ("# Retail credit model review and underwriting strategy (LendingClub 2012-2015)\n\n"
          "An end-to-end consumer-lending credit risk workflow on public loan-level data: SQL portfolio monitoring "
          "and root cause analysis, an application scorecard, a machine-learning challenger reviewed against the "
          "lender's own grade, underwriting cut-off and affordability tests, and a risk committee memo. Every "
          "number in the reports is generated by code from `outputs/results.json`.\n\n")
    s += "## Results\n\n| | |\n|---|---|\n"
    s += f"| Loans (36-month, issued 2012-2015, final outcome) | {F.n_pop:,} |\n"
    s += f"| Bad rate {r['period_1']} -> {r['period_2']} | {pct(r['bad_rate_1'])} -> {pct(r['bad_rate_2'])} ({ppf(r['total_change'])}) |\n"
    s += f"| ... from grade mix / within-grade | {ppf(r['mix_effect'])} / {ppf(r['rate_effect'])} |\n"
    s += (f"| Out-of-time Gini: LC sub-grade / scorecard / GBM / overlay | {g['champion']:.3f} / {g['scorecard']:.3f} / "
          f"{g['challenger']:.3f} / {g['overlay']:.3f} |\n")
    s += f"| Out-of-time KS: LC sub-grade / scorecard / GBM | {F.ks['champion']:.3f} / {F.ks['scorecard']:.3f} / {F.ks['challenger']:.3f} |\n"
    s += f"| Score PSI train vs 2015 (scorecard) | {F.psi['scorecard']:.3f} |\n"
    s += f"| Model review verdict | {v.split(':')[0].capitalize()} |\n\n"
    for fig, cap in (("vintage_curves", "Vintage curves"), ("rca_waterfall", "Root cause: mix vs rate"),
                     ("gini_with_ci", "Model discrimination"), ("calibration_oot", "Calibration"),
                     ("cutoff_strategy", "Cut-off strategy"), ("swap_set", "Swap-set")):
        s += f"![{cap}](outputs/figures/{fig}.png)\n\n"
    s += """## What is in here

| Stage | What it does | Where |
|---|---|---|
| 0 | Parse the raw file as text, cast explicitly, log every filter | `sql/00_parse_raw.sql`, `sql/01_build_loans.sql`, `outputs/tables/data_quality_funnel.csv` |
| 1 | Static-pool bad rates, months-on-book curves, quarterly health (CTEs, window functions) | `sql/02-04_*.sql` |
| 2 | Hand-written WoE binning, IV, correlation/VIF, logistic scorecard, points, Gini/KS/PSI/CSI, bootstrap CIs | `src/creditlab/woe.py`, `scorecard.py`, `metrics.py` |
| 3 | Champion (LC sub-grade) vs monotone GBM challenger vs overlay; permutation importance; model review | `models.py`, `reports/MODEL_REVIEW.md` |
| 4 | Swap-set analysis, cut-off strategy with realised return, pre/post-loan DTI caps inside score bands | `strategy.py` |
| 5 | Mix/rate decomposition, within-grade drill-down, early-warning by issue quarter | `rca.py` |
| 6 | Figures, risk committee memo, Tableau exports | `figures.py`, `report.py`, `reports/RISK_MEMO.md`, `docs/TABLEAU_GUIDE.md` |

## How to run

1. Download `accepted_2007_to_2018Q4.csv.gz` from Kaggle ("Lending Club Loan Data", wordsforthewise) into `data/raw/`.
2. Windows (PowerShell), from the repo folder:

```powershell
python -m pip install -r requirements.txt
python run_all.py
python -m pytest
```

`run_all.py` rebuilds everything from the raw file: tables, figures, reports and `outputs/results.json`.
The raw and processed data are not committed (see `.gitignore`).

## Design rules

- No leakage: model inputs are application-time fields only; a unit test fails if any post-origination
  field (payments, recoveries, last FICO, settlement fields) enters a feature list.
- LendingClub's grade, sub-grade, interest rate and instalment are its own model outputs and are used only as
  the benchmark.
- Out-of-time test: trained on 2012-2014, tested on 2015.
- Deterministic: seed 42 throughout.

## Limitations

Reject inference (only approved loans are observed); the sub-grade uses data not in this file; one benign
credit cycle; US market; realised return ignores fees, funding cost and time value. See `reports/MODEL_REVIEW.md`.
"""
    return s


def write_all() -> dict:
    R = json.loads(C.RESULTS_JSON.read_text(encoding="utf-8"))
    F = Facts(R)
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    docs = {
        C.REPORTS / "MODEL_REVIEW.md": model_review(F),
        C.REPORTS / "RISK_MEMO.md": risk_memo(F),
        C.REPORTS / "CV_BULLETS.md": cv_bullets(F),
        C.REPORTS / "PROJECT_PITCH.md": pitch(F),
        C.ROOT / "README.md": readme(F),
    }
    for path, text in docs.items():
        path.write_text(text, encoding="utf-8")
    return {"verdict": F.verdict()[0], "synthetic": F.syn}
