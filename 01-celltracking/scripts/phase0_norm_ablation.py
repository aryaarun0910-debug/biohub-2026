"""Phase 0: per-frame (V3) vs precomputed per-volume quantile normalization, with statistics.

Runs the V3 DoG pipeline under BOTH normalizations on a stratified sample of crops from each
embryo, scores with the exact numpy edge metric, writes a per-crop CSV, and runs a PAIRED
Wilcoxon signed-rank test on per-crop adjusted edge Jaccard. Reports the p-value so the
"no meaningful difference" claim is evidence-backed, not asserted.

Writes reports/inventory/norm_ablation.csv.

Usage: python scripts/phase0_norm_ablation.py --per-embryo 15 --workers 6
"""

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from biotrack.metric import estimated_nodes  # noqa: E402
from biotrack.metric_numpy import score_sample  # noqa: E402
from run_v3_taxonomy import geff_to_sample, rows_to_sample  # noqa: E402

_spec = importlib.util.spec_from_file_location("dog_na", ROOT / "notebooks" / "kaggle_dog_infer.py")
dog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dog)

TRAIN = ROOT / "data" / "train"
OUT = ROOT / "reports" / "inventory" / "norm_ablation.csv"


def _run(name, mode):
    dog.NORM_MODE = mode
    rows = dog.infer_dataset(TRAIN / f"{name}.zarr")
    pred = rows_to_sample(rows)
    gt = geff_to_sample(str(TRAIN / f"{name}.geff"))
    n_est = estimated_nodes(str(TRAIN / f"{name}.geff"))
    s = score_sample(pred, gt, n_est)
    return s["adj_edge_jaccard"], s["node_recall"], (s["num_pred_nodes"] / n_est if n_est else float("nan"))


def worker(name):
    pf = _run(name, "per_frame")
    pc = _run(name, "precomputed")
    return {"crop": name, "fam": name.split("_")[0],
            "pf_adjJ": round(pf[0], 4), "pf_recall": round(pf[1], 4), "pf_ratio": round(pf[2], 4),
            "pc_adjJ": round(pc[0], 4), "pc_recall": round(pc[1], 4), "pc_ratio": round(pc[2], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-embryo", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    names = sorted(p.stem for p in TRAIN.glob("*.geff") if (TRAIN / f"{p.stem}.zarr").exists())
    by_fam = defaultdict(list)
    for n in names:
        by_fam[n.split("_")[0]].append(n)
    picks = [n for fam in by_fam.values() for n in fam[: args.per_embryo]]

    print(f"Norm ablation on {len(picks)} crops, {args.workers} workers...")
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            results = list(pool.imap_unordered(worker, picks))
    else:
        results = [worker(n) for n in picks]
    results.sort(key=lambda r: r["crop"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)

    pf = np.array([r["pf_adjJ"] for r in results]); pc = np.array([r["pc_adjJ"] for r in results])
    diffs = pf - pc
    print(f"\nWrote {OUT} ({len(results)} crops)")
    print(f"adjJ mean: per_frame={pf.mean():.4f}  precomputed={pc.mean():.4f}  "
          f"delta={pf.mean()-pc.mean():+.4f}  max|delta|={np.abs(diffs).max():.4f}")
    if np.allclose(diffs, 0.0):
        print(f"IDENTICAL on all {len(pf)} crops (delta=0). DoG uses a RELATIVE threshold "
              "(fraction of max response), so linear normalization does not move peaks -> "
              "normalization is not a lever for the DoG detector.")
    else:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(pf, pc)
        verdict = "no significant difference" if p > 0.05 else "SIGNIFICANT difference"
        print(f"paired Wilcoxon signed-rank: W={stat:.1f}  p={p:.3f} -> {verdict} (a=0.05, n={len(pf)})")


if __name__ == "__main__":
    main()
