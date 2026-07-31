"""H1-I: does geometry+appearance reach the precision break-even that geometry
alone missed?

Break-even for a +0.005 composite gain (measured elsewhere): 4.07% precision on
family 44b6 and 6.38% on 6bba, among metric-visible candidates.  Geometry alone
reached 0.690% / 0.498%.

Two things are reported, and they are NOT the same claim:

  ORACLE   features and directions chosen using both families' labels.  This is
           an optimistic upper bound; with 92 positives the selection itself is
           fitted, so it cannot be believed as a deployment number.

  LOFO     leave-one-family-out.  Feature set and signs are chosen on the OTHER
           family only, then applied unchanged.  This is the honest transfer
           number and the one that decides whether appearance is deployable.

Precision is extrapolated to the full census whenever only part of the reliable
negatives have been extracted: an FP rate measured on the sampled negatives is
applied to the full reliable-negative count, and captured-positive fraction is
applied to the full positive count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BREAKEVEN = {"44b6": 0.0407, "6bba": 0.0638}
FULL_POS = {"44b6": 16, "6bba": 76}
FULL_RN = {"44b6": 54031, "6bba": 199319}


def rank_norm(x: np.ndarray) -> np.ndarray:
    """Rank-transform to [0,1]; NaNs -> 0.5 (uninformative)."""
    r = np.array(pd.Series(x).rank(pct=True).to_numpy(), dtype=float, copy=True)
    r[~np.isfinite(x)] = 0.5
    return r


def sweep(score: np.ndarray, is_pos: np.ndarray, n_pos_full: int, n_rn_full: int,
          min_tp: int = 3) -> dict:
    """Best extrapolated precision over all thresholds, requiring >= min_tp."""
    ok = np.isfinite(score)
    score, is_pos = score[ok], is_pos[ok]
    order = np.argsort(-score, kind="stable")
    p = is_pos[order]
    tp = np.cumsum(p)
    fp = np.cumsum(~p)
    n_pos, n_rn = int(p.sum()), int((~p).sum())
    if n_pos == 0 or n_rn == 0:
        return {}
    pos_frac = tp / n_pos
    fp_rate = fp / n_rn
    tp_full = pos_frac * n_pos_full
    fp_full = fp_rate * n_rn_full
    prec = tp_full / np.maximum(tp_full + fp_full, 1e-9)
    valid = tp >= min_tp
    if not valid.any():
        return {}
    i = int(np.argmax(np.where(valid, prec, -1)))
    return {
        "precision": float(prec[i]),
        "recall": float(pos_frac[i]),
        "tp_sampled": int(tp[i]),
        "fp_sampled": int(fp[i]),
        "fp_rate": float(fp_rate[i]),
        "k_sampled": int(i + 1),
        "n_pos_sampled": n_pos,
        "n_rn_sampled": n_rn,
    }


def build_score(df: pd.DataFrame, feats: list[str], signs: dict) -> np.ndarray:
    acc = np.zeros(len(df))
    for f in feats:
        acc += signs[f] * rank_norm(df[f].to_numpy(dtype=float))
    return acc / max(len(feats), 1)


def select(auc_tbl: pd.DataFrame, families: list[str], pool: list[str],
           n_max: int, min_sep: float) -> tuple[list[str], dict]:
    """Pick sign-consistent features using ONLY the listed families."""
    t = auc_tbl[auc_tbl.feature.isin(pool)].copy()
    cols = [f"auc_{f}" for f in families]
    t = t.dropna(subset=cols)
    dev = t[cols].to_numpy() - 0.5
    if len(families) > 1:
        consistent = np.all(np.sign(dev) == np.sign(dev[:, :1]), axis=1)
        t = t[consistent]
        dev = dev[consistent]
    t["sep"] = np.abs(dev).min(axis=1)
    t["sgn"] = np.sign(dev[:, 0])
    t = t[t.sep >= min_sep].sort_values("sep", ascending=False).head(n_max)
    return list(t.feature), dict(zip(t.feature, t.sgn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--auc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-feats", type=int, default=4)
    ap.add_argument("--min-sep", type=float, default=0.10)
    args = ap.parse_args()

    df = pd.read_parquet(args.cands)
    auc_tbl = pd.read_csv(args.auc)
    df["is_pos"] = (df.label == "positive").to_numpy()

    skip = {"cand_id", "crop", "family", "fold", "t", "mother", "d1", "d2",
            "rank", "label", "is_pos"}
    allf = [c for c in df.columns if c not in skip
            and pd.api.types.is_numeric_dtype(df[c])]
    geo = [c for c in allf if c.startswith("G_")]
    img = [c for c in allf if not c.startswith("G_")]

    report = {"n_rows": len(df), "pools": {"geometry": len(geo), "image": len(img)}}
    rows = []
    for mode, pool in (("geometry", geo), ("image", img), ("geometry+image", allf)):
        # ---- ORACLE: selection uses both families -------------------------
        f_or, s_or = select(auc_tbl, ["44b6", "6bba"], pool, args.n_feats, args.min_sep)
        for fam in ("44b6", "6bba"):
            sub = df[df.family == fam]
            if f_or:
                sc = build_score(sub, f_or, s_or)
                r = sweep(sc, sub.is_pos.to_numpy(), FULL_POS[fam], FULL_RN[fam])
            else:
                r = {}
            rows.append({"mode": mode, "eval": "ORACLE", "family": fam,
                         "n_feats": len(f_or), "feats": ";".join(f_or),
                         "breakeven": BREAKEVEN[fam], **r})
        # ---- LOFO: selection uses the other family only -------------------
        for fam in ("44b6", "6bba"):
            other = "6bba" if fam == "44b6" else "44b6"
            f_lo, s_lo = select(auc_tbl, [other], pool, args.n_feats, args.min_sep)
            sub = df[df.family == fam]
            if f_lo:
                sc = build_score(sub, f_lo, s_lo)
                r = sweep(sc, sub.is_pos.to_numpy(), FULL_POS[fam], FULL_RN[fam])
            else:
                r = {}
            rows.append({"mode": mode, "eval": "LOFO", "family": fam,
                         "n_feats": len(f_lo), "feats": ";".join(f_lo),
                         "breakeven": BREAKEVEN[fam], **r})

    res = pd.DataFrame(rows)
    res["passes"] = res.precision >= res.breakeven
    res["x_to_breakeven"] = res.breakeven / res.precision.replace(0, np.nan)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(res[["mode", "eval", "family", "n_feats", "precision", "breakeven",
               "x_to_breakeven", "recall", "tp_sampled", "fp_sampled",
               "passes"]].to_string(index=False))
    print()
    for m in res["mode"].unique():
        for e in ("ORACLE", "LOFO"):
            f = res[(res["mode"] == m) & (res["eval"] == e)].feats.iloc[0]
            print(f"{m:15s} {e:7s} -> {f}")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
