"""H1-I: operating-point curve with honest uncertainty.

The argmax-over-thresholds precision used in h1i_precision_gate.py sits on 3-6
events and is therefore unstable.  Here the operating point is instead pinned by
the FALSE-POSITIVE RATE f (the fraction of reliable negatives admitted), which
is the quantity a deployment threshold actually controls.  Recall r is then
measured, and its Clopper-Pearson interval (n = 16 or 76 positives) is
propagated into precision:

    precision(f) = r * P_full / (r * P_full + f * RN_full)

All scores are leave-one-family-out: feature set and signs come from the OTHER
family only, so nothing about the evaluated family is fitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from h1i_precision_gate import build_score, rank_norm, select

BREAKEVEN = {"44b6": 0.0407, "6bba": 0.0638}
FULL_POS = {"44b6": 16, "6bba": 76}
FULL_RN = {"44b6": 54031, "6bba": 199319}
FRACS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def cp_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def curve(df: pd.DataFrame, fam: str, feats: list[str], signs: dict) -> pd.DataFrame:
    sub = df[df.family == fam]
    sc = build_score(sub, feats, signs)
    is_pos = sub.label.to_numpy() == "positive"
    pos, neg = sc[is_pos], sc[~is_pos]
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    P, RN = FULL_POS[fam], FULL_RN[fam]
    rows = []
    for f in FRACS:
        thr = np.quantile(neg, 1.0 - f)
        k = int((pos > thr).sum())
        r = k / pos.size
        rlo, rhi = cp_interval(k, pos.size)
        def prec(rr):
            return rr * P / max(rr * P + f * RN, 1e-12)
        rows.append({
            "family": fam, "fp_frac": f, "tp": k, "n_pos_eval": pos.size,
            "recall": r, "precision": prec(r),
            "prec_lo": prec(rlo), "prec_hi": prec(rhi),
            "breakeven": BREAKEVEN[fam],
            "passes_point": prec(r) >= BREAKEVEN[fam],
            "passes_upper": prec(rhi) >= BREAKEVEN[fam],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--auc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-feats", type=int, default=6)
    ap.add_argument("--min-sep", type=float, default=0.10)
    args = ap.parse_args()

    df = pd.read_parquet(args.cands)
    auc_tbl = pd.read_csv(args.auc)
    skip = {"cand_id", "crop", "family", "fold", "t", "mother", "d1", "d2",
            "rank", "label"}
    allf = [c for c in df.columns if c not in skip
            and pd.api.types.is_numeric_dtype(df[c])]
    pools = {
        "geometry": [c for c in allf if c.startswith("G_")],
        "image": [c for c in allf if not c.startswith("G_")],
        "geometry+image": allf,
    }
    out = []
    for mode, pool in pools.items():
        for fam in ("44b6", "6bba"):
            other = "6bba" if fam == "44b6" else "44b6"
            fs, sg = select(auc_tbl, [other], pool, args.n_feats, args.min_sep)
            if not fs:
                continue
            c = curve(df, fam, fs, sg)
            c["mode"] = mode
            c["feats"] = ";".join(fs)
            out.append(c)
    res = pd.concat(out, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    show = ["mode", "family", "fp_frac", "tp", "recall", "precision",
            "prec_lo", "prec_hi", "breakeven", "passes_point", "passes_upper"]
    for m in pools:
        print(f"\n=== {m} (LOFO) ===")
        print(res[res["mode"] == m][show].to_string(index=False,
              float_format=lambda v: f"{v:.5f}"))


if __name__ == "__main__":
    main()
