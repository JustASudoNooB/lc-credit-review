"""Static figures for the README, memo and model review (matplotlib, PNG).

Palette: first three categorical slots of a colour-blind-validated palette (blue, orange, aqua),
an ordinal blue ramp for the four vintages, hairline grey grid, text in ink colours only.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9893"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]     # oldest -> newest vintage

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2.0,
})


def _pct(ax, axis="y", decimals=0):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
    a.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=decimals))


def _finish(fig, name, note=None, synthetic=False):
    if synthetic:
        fig.text(0.99, -0.03, "SYNTHETIC TEST DATA - NOT REAL RESULTS", ha="right", va="top",
                 color="#d03b3b", fontsize=8, fontweight="bold")
    if note:
        fig.text(0.01, -0.03, note, ha="left", va="top", color=INK_2, fontsize=8)
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def vintage_curves(synthetic):
    mob = pd.read_csv(C.TABLES / "mob_curves_by_quarter.csv")
    y = mob.groupby(["issue_year", "mob"])[["new_defaults", "cohort_loans"]].sum().reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    years = sorted(y["issue_year"].unique())
    for col, yr in zip(ORDINAL[-len(years):], years):
        g = y[y["issue_year"] == yr].sort_values("mob")
        cum = g["new_defaults"].cumsum() / g["cohort_loans"]
        ax.plot(g["mob"], cum, color=col, label=str(yr))
        ax.text(g["mob"].iloc[-1] + 0.4, cum.iloc[-1], f"{yr}  {cum.iloc[-1]:.1%}", color=INK, va="center", fontsize=9)
    ax.set_xlim(0, 42)
    ax.set_xlabel("Months on book")
    ax.set_ylabel("Cumulative default rate")
    _pct(ax)
    ax.set_title("Vintage curves: 36-month loans by issue year")
    ax.legend(loc="upper left", title="Issue year", title_fontsize=9)
    _finish(fig, "vintage_curves", "Default month approximated as the month after the last payment.", synthetic)


def score_distribution(scored, R, synthetic):
    tr = scored.loc[scored["sample"] == "train", "sc_score"]
    oot = scored.loc[scored["sample"] == "oot", "sc_score"]
    bins = np.linspace(np.percentile(scored["sc_score"], 0.5), np.percentile(scored["sc_score"], 99.5), 40)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.hist(tr, bins=bins, density=True, histtype="step", color=S1, linewidth=2, label="Train (2012-2014)")
    ax.hist(oot, bins=bins, density=True, histtype="step", color=S2, linewidth=2, label="Out-of-time (2015)")
    psi = R["model_review"]["evaluation"]["scorecard"]["psi_train_vs_oot"]
    ax.set_title(f"Scorecard score distribution (PSI {psi:.3f})")
    ax.set_xlabel("Score (higher = safer)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    _finish(fig, "score_distribution", synthetic=synthetic)


def calibration(R, synthetic):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    names = [("champion", "LC sub-grade (grade-implied PD)", S1), ("scorecard", "WoE scorecard", S2),
             ("challenger", "GBM challenger", S3)]
    hi = 0
    for key, lab, col in names:
        t = pd.DataFrame(R["model_review"]["evaluation"][key]["calibration_oot"])
        ax.plot(t["predicted"], t["actual"], marker="o", markersize=5, color=col, label=lab)
        hi = max(hi, t["predicted"].max(), t["actual"].max())
    ax.plot([0, hi * 1.05], [0, hi * 1.05], color=NEUTRAL, linewidth=1, label="Perfect calibration")
    _pct(ax, "y")
    _pct(ax, "x")
    ax.set_xlabel("Predicted bad rate (decile mean)")
    ax.set_ylabel("Actual bad rate")
    ax.set_title("Calibration on 2015 out-of-time loans")
    ax.legend(loc="upper left", fontsize=8.5)
    _finish(fig, "calibration_oot", synthetic=synthetic)


def gini_ci(R, synthetic):
    b = R["model_review"]["bootstrap_oot_vs_champion"]
    order = [("champion", "LC sub-grade (champion)"), ("scorecard", "WoE scorecard"),
             ("challenger", "GBM challenger"), ("overlay", "Grade + scorecard overlay")]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for i, (k, lab) in enumerate(order):
        g, lo, hi = b[k]["gini"], b[k]["ci_low"], b[k]["ci_high"]
        ax.plot([lo, hi], [i, i], color=S1, linewidth=2)
        ax.plot(g, i, "o", color=S1, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(hi + 0.004, i, f"{g:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(order)), [o[1] for o in order])
    ax.invert_yaxis()
    ax.set_xlabel("Gini on 2015 loans (dot) with bootstrap 95% CI (line)")
    ax.set_title("Discrimination: who ranks risk best?")
    ax.grid(axis="y", visible=False)
    _finish(fig, "gini_with_ci", synthetic=synthetic)


def cutoff_curve(synthetic):
    t = pd.read_csv(C.TABLES / "cutoff_strategy.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (name, col) in [("LC sub-grade", S1), ("scorecard", S2), ("challenger", S3)]:
        g = t[t["model"] == name]
        ax.plot(g["approval_rate"], g["bad_rate"], marker="o", markersize=4, color=col, label=name)
    ax.axvline(C.TARGET_APPROVAL, color=NEUTRAL, linewidth=1)
    ax.text(C.TARGET_APPROVAL, ax.get_ylim()[1], f" {C.TARGET_APPROVAL:.0%} approval", color=INK_2, va="top", fontsize=8.5)
    _pct(ax, "x")
    _pct(ax, "y", 1)
    ax.set_xlabel("Approval rate (share of the 2015 book kept)")
    ax.set_ylabel("Bad rate of approved loans")
    ax.set_title("Cut-off strategy: bad rate at each approval rate")
    ax.legend(loc="upper left")
    _finish(fig, "cutoff_strategy", synthetic=synthetic)


def swap_set(R, synthetic):
    t = pd.DataFrame(R["strategy"]["swap_vs_scorecard"]["table"])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(t))
    ax.bar(x, t["bad_rate"], color=S1, width=0.6)
    for i, r in t.iterrows():
        ax.text(i, r["bad_rate"], f"{r['bad_rate']:.1%}\n{r['loans']:,} loans", ha="center", va="bottom",
                color=INK, fontsize=8.5)
    ax.set_xticks(x, [c.replace(" (", "\n(") for c in t["cell"]], fontsize=8.5)
    _pct(ax, "y")
    ax.set_ylim(0, t["bad_rate"].max() * 1.3)
    ax.set_ylabel("Bad rate")
    ax.set_title(f"Swap-set at {C.TARGET_APPROVAL:.0%} approval: LC sub-grade vs scorecard")
    ax.grid(axis="x", visible=False)
    _finish(fig, "swap_set", synthetic=synthetic)


def rca_waterfall(R, synthetic):
    r = R["rca"]
    steps = [(f"{r['period_1']} bad rate", r["bad_rate_1"], "total"),
             ("Grade mix", r["mix_effect"], "delta"),
             ("Within-grade", r["rate_effect"], "delta"),
             (f"{r['period_2']} bad rate", r["bad_rate_2"], "total")]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    level = 0.0
    for i, (lab, v, kind) in enumerate(steps):
        if kind == "total":
            ax.bar(i, v, color=NEUTRAL, width=0.55)
            level = v
            ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", color=INK, fontsize=9)
        else:
            col = S2 if lab == "Grade mix" else S1
            ax.bar(i, v, bottom=level, color=col, width=0.55)
            ax.text(i, level + max(v, 0), f"{v * 100:+.2f} pp", ha="center", va="bottom", color=INK, fontsize=9)
            level += v
    ax.set_xticks(range(len(steps)), [s[0] for s in steps])
    _pct(ax, "y")
    ax.set_ylim(0, max(r["bad_rate_1"], r["bad_rate_2"]) * 1.2)
    ax.set_title("Why the bad rate moved: grade mix vs within-grade change")
    ax.grid(axis="x", visible=False)
    _finish(fig, "rca_waterfall", synthetic=synthetic)


def all_figures():
    R = json.loads(C.RESULTS_JSON.read_text(encoding="utf-8"))
    syn = R["meta"]["synthetic_test_data"]
    scored = pd.read_parquet(C.SCORED_PARQUET)
    vintage_curves(syn)
    score_distribution(scored, R, syn)
    calibration(R, syn)
    gini_ci(R, syn)
    cutoff_curve(syn)
    swap_set(R, syn)
    rca_waterfall(R, syn)
