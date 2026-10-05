"""Weight-of-Evidence binning, written by hand.

Steps for a numeric feature (all on the TRAIN sample only):
  1. Fine classing: cut at the 5%, 10%, ..., 95% quantiles (20 bins; fewer if values repeat).
  2. Monotonic merge: while two neighbouring bins have bad rates moving against the feature's
     overall direction, merge the pair with the smallest gap. Every numeric feature ends up
     with points that move one way only, which a credit committee can follow.
  3. Size merge: any bin under 5% of rows is merged into the neighbour with the closer bad rate.
  4. Count cap: merge the closest neighbouring pair until there are at most 8 bins.
  5. Missing values get their own bin, so "not reported" is scored on its own evidence.
Categorical features: categories under 1% are pooled into RARE, groups are sorted by bad rate,
then steps 3 and 4 run on that order.

WoE of a bin = ln( share of all goods in the bin / share of all bads in the bin ).
Positive WoE = safer than average. IV = sum over bins of (share_good - share_bad) * WoE.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


def woe_iv(n, bads, smoothing: float = C.WOE_SMOOTHING):
    """WoE and IV contribution per bin from bin counts. Smoothing avoids ln(0) in empty cells."""
    n = np.asarray(n, dtype=float)
    b = np.asarray(bads, dtype=float)
    g = n - b
    k = len(n)
    dg = (g + smoothing) / (g.sum() + smoothing * k)
    db = (b + smoothing) / (b.sum() + smoothing * k)
    woe = np.log(dg / db)
    return woe, (dg - db) * woe


def _fmt(v: float) -> str:
    if np.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:,.4g}"


@dataclass
class FeatureBinning:
    feature: str
    kind: str                                   # "numeric" | "categorical"
    labels: list[str]
    n: np.ndarray
    bads: np.ndarray
    woe: np.ndarray
    iv_parts: np.ndarray
    has_missing: bool = False
    edges: list[float] = field(default_factory=list)        # numeric: upper bounds of all but the last bin
    cat_to_bin: dict[str, int] = field(default_factory=dict)
    unseen_bin: int | None = None              # categorical: where new categories go
    direction: int = 0                          # numeric: +1 risk rises with value, -1 falls

    @property
    def iv(self) -> float:
        return float(self.iv_parts.sum())

    def bin_index(self, values) -> np.ndarray:
        """Bin number for each value; -1 means 'not seen in training' and gets WoE 0 (neutral)."""
        if self.kind == "numeric":
            x = np.asarray(values, dtype=float)
            idx = np.searchsorted(np.asarray(self.edges, dtype=float), x, side="left")
            miss = np.isnan(x)
            idx[miss] = len(self.labels) - 1 if self.has_missing else -1
            return idx.astype(int)
        s = pd.Series(values).astype(str)
        default = -1 if self.unseen_bin is None else self.unseen_bin
        return s.map(self.cat_to_bin).fillna(default).astype(int).to_numpy()

    def transform(self, values) -> np.ndarray:
        idx = self.bin_index(values)
        out = np.zeros(len(idx), dtype=float)
        ok = idx >= 0
        out[ok] = self.woe[idx[ok]]
        return out

    def table(self) -> pd.DataFrame:
        t = pd.DataFrame({
            "feature": self.feature,
            "bin": np.arange(len(self.labels)),
            "label": self.labels,
            "loans": self.n.astype(int),
            "bads": self.bads.astype(int),
        })
        t["share"] = t["loans"] / t["loans"].sum()
        t["bad_rate"] = t["bads"] / t["loans"]
        t["woe"] = self.woe
        t["iv_part"] = self.iv_parts
        return t


# --------------------------------------------------------------------------- merging helpers
def _rates(blocks):
    return np.array([bl["b"] / bl["n"] for bl in blocks])


def _merge(blocks, i):
    """Merge block i with block i+1."""
    a, b = blocks[i], blocks[i + 1]
    merged = {"n": a["n"] + b["n"], "b": a["b"] + b["b"], "hi": b.get("hi"),
              "members": a.get("members", []) + b.get("members", []),
              "rare": a.get("rare", False) or b.get("rare", False)}
    return blocks[:i] + [merged] + blocks[i + 2:]


def _merge_small(blocks, min_n):
    while len(blocks) > 1:
        sizes = np.array([bl["n"] for bl in blocks])
        i = int(np.argmin(sizes))
        if sizes[i] >= min_n:
            break
        r = _rates(blocks)
        if i == 0:
            j = 0
        elif i == len(blocks) - 1:
            j = i - 1
        else:
            j = i - 1 if abs(r[i] - r[i - 1]) <= abs(r[i] - r[i + 1]) else i
        blocks = _merge(blocks, j)
    return blocks


def _merge_to_max(blocks, max_bins):
    while len(blocks) > max_bins:
        r = _rates(blocks)
        j = int(np.argmin(np.abs(np.diff(r))))
        blocks = _merge(blocks, j)
    return blocks


# --------------------------------------------------------------------------- fitting
def fit_numeric(feature: str, x, y) -> FeatureBinning:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(int)
    miss = np.isnan(x)
    xv, yv = x[~miss], y[~miss]
    edges = np.unique(np.quantile(xv, np.linspace(0, 1, C.FINE_BINS + 1)[1:-1]))
    idx = np.searchsorted(edges, xv, side="left")
    k = len(edges) + 1
    n = np.bincount(idx, minlength=k)
    b = np.bincount(idx, weights=yv, minlength=k)
    his = list(edges) + [np.inf]
    blocks = [{"n": float(n[i]), "b": float(b[i]), "hi": his[i]} for i in range(k) if n[i] > 0]

    # direction from the rank correlation of the raw feature with the outcome
    ranks = pd.Series(xv).rank().to_numpy()
    corr = np.corrcoef(ranks, yv)[0, 1] if len(xv) > 1 and yv.std() > 0 else 0.0
    direction = 1 if corr >= 0 else -1

    while len(blocks) > 1:
        r = _rates(blocks)
        viol = [i for i in range(len(blocks) - 1) if direction * (r[i + 1] - r[i]) < 0]
        if not viol:
            break
        i = min(viol, key=lambda j: abs(r[j + 1] - r[j]))
        blocks = _merge(blocks, i)

    blocks = _merge_small(blocks, C.MIN_BIN_SHARE * len(xv))
    blocks = _merge_to_max(blocks, C.MAX_BINS)

    final_edges = [bl["hi"] for bl in blocks[:-1]]
    lows = [-np.inf] + final_edges
    highs = final_edges + [np.inf]
    labels = [f"({_fmt(lo)}, {_fmt(hi)}]" if not np.isinf(hi) else f"> {_fmt(lo)}" for lo, hi in zip(lows, highs)]
    if np.isinf(lows[0]) and len(labels) > 0:
        labels[0] = f"<= {_fmt(highs[0])}" if not np.isinf(highs[0]) else "all"
    counts = [bl["n"] for bl in blocks]
    badc = [bl["b"] for bl in blocks]
    has_missing = bool(miss.any())
    if has_missing:
        labels.append("MISSING")
        counts.append(float(miss.sum()))
        badc.append(float(y[miss].sum()))
    woe, ivp = woe_iv(counts, badc)
    return FeatureBinning(feature, "numeric", labels, np.array(counts), np.array(badc), woe, ivp,
                          has_missing=has_missing, edges=[float(e) for e in final_edges], direction=direction)


def fit_categorical(feature: str, s, y) -> FeatureBinning:
    s = pd.Series(s).astype(str).reset_index(drop=True)
    y = pd.Series(np.asarray(y).astype(int))
    stats = pd.DataFrame({"cat": s, "y": y}).groupby("cat")["y"].agg(["size", "sum"])
    total = stats["size"].sum()
    rare_mask = stats["size"] < C.RARE_CATEGORY_SHARE * total
    blocks = [{"n": float(r["size"]), "b": float(r["sum"]), "members": [c], "rare": False}
              for c, r in stats[~rare_mask].iterrows()]
    if rare_mask.any():
        rs = stats[rare_mask]
        blocks.append({"n": float(rs["size"].sum()), "b": float(rs["sum"].sum()),
                       "members": list(rs.index), "rare": True})
    blocks.sort(key=lambda bl: bl["b"] / bl["n"])
    blocks = _merge_small(blocks, C.MIN_BIN_SHARE * total)
    blocks = _merge_to_max(blocks, C.MAX_BINS)

    cat_to_bin, labels, unseen = {}, [], None
    for i, bl in enumerate(blocks):
        for m in bl["members"]:
            cat_to_bin[m] = i
        names = sorted(bl["members"])
        label = ", ".join(names[:4]) + (f" +{len(names) - 4} more" if len(names) > 4 else "")
        labels.append(label)
        if bl["rare"]:
            unseen = i
    woe, ivp = woe_iv([bl["n"] for bl in blocks], [bl["b"] for bl in blocks])
    return FeatureBinning(feature, "categorical", labels, np.array([bl["n"] for bl in blocks]),
                          np.array([bl["b"] for bl in blocks]), woe, ivp,
                          cat_to_bin=cat_to_bin, unseen_bin=unseen)


def fit_all(X: pd.DataFrame, y, features) -> dict[str, FeatureBinning]:
    out = {}
    for f in features:
        if f in C.CATEGORICAL_FEATURES:
            out[f] = fit_categorical(f, X[f], y)
        else:
            out[f] = fit_numeric(f, X[f].to_numpy(), y)
    return out


def transform_all(X: pd.DataFrame, binnings: dict[str, FeatureBinning], features) -> pd.DataFrame:
    return pd.DataFrame({f: binnings[f].transform(X[f].to_numpy()) for f in features}, index=X.index)


def iv_table(binnings: dict[str, FeatureBinning]) -> pd.DataFrame:
    rows = [{"feature": f, "iv": b.iv, "bins": len(b.labels), "kind": b.kind,
             "direction": {1: "risk rises with value", -1: "risk falls with value", 0: "categorical"}[b.direction]}
            for f, b in binnings.items()]
    t = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    t["iv_band"] = pd.cut(t["iv"], [-1, 0.02, 0.1, 0.3, 0.5, 99],
                          labels=["none (<0.02)", "weak", "medium", "strong", "suspicious (>0.5)"]).astype(str)
    return t
