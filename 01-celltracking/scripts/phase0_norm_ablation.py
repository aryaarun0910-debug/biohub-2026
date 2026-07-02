"""Phase 0: A/B per-frame (V3) vs precomputed per-volume quantile normalization.

Runs the V3 DoG pipeline under both normalizations on a few crops from each embryo,
scores with the exact numpy edge metric, and reports adj edge-J / recall / count ratio.
Decides whether the normalization is (part of) the V3 reproduction / recall gap.

Usage: python scripts/phase0_norm_ablation.py --per-embryo 3
"""

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from biotrack.metric import estimated_nodes  # noqa: E402
from biotrack.metric_numpy import score_sample  # noqa: E402
from run_v3_taxonomy import geff_to_sample, rows_to_sample  # noqa: E402

_spec = importlib.util.spec_from_file_location("dogv3", ROOT / "notebooks" / "kaggle_dog_infer.py")
dog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dog)

TRAIN = ROOT / "data" / "train"


def run(name: str, mode: str) -> dict:
    dog.NORM_MODE = mode
    rows = dog.infer_dataset(TRAIN / f"{name}.zarr")
    pred = rows_to_sample(rows)
    gt = geff_to_sample(str(TRAIN / f"{name}.geff"))
    n_est = estimated_nodes(str(TRAIN / f"{name}.geff"))
    s = score_sample(pred, gt, n_est)
    return {"adjJ": s["adj_edge_jaccard"], "recall": s["node_recall"],
            "ratio": s["num_pred_nodes"] / n_est if n_est else float("nan"),
            "w": s["edge_tp"] + s["edge_fp"] + s["edge_fn"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-embryo", type=int, default=3)
    args = ap.parse_args()

    names = sorted(p.stem for p in TRAIN.glob("*.geff"))
    by_fam = defaultdict(list)
    for n in names:
        by_fam[n.split("_")[0]].append(n)
    picks = [n for fam in by_fam.values() for n in fam[: args.per_embryo]]

    modes = ["per_frame", "precomputed"]
    agg = {m: defaultdict(float) for m in modes}
    print(f"{'crop':18s} {'fam':5s} " + "  ".join(f"{m:>22s}" for m in modes))
    print(f"{'':18s} {'':5s} " + "  ".join(f"{'adjJ  recall  ratio':>22s}" for _ in modes))
    for name in picks:
        line = f"{name[:18]:18s} {name.split('_')[0]:5s} "
        for m in modes:
            r = run(name, m)
            line += f"  {r['adjJ']:6.3f} {r['recall']:6.3f} {r['ratio']:5.2f} "
            agg[m]["adjw"] += r["adjJ"] * r["w"]; agg[m]["w"] += r["w"]
            agg[m]["recall"] += r["recall"]; agg[m]["ratio"] += r["ratio"]; agg[m]["n"] += 1
        print(line)

    print("\n=== aggregate ===")
    for m in modes:
        a = agg[m]
        print(f"  {m:12s}: adjJ(w)={a['adjw']/a['w']:.4f}  recall={a['recall']/a['n']:.3f}  "
              f"ratio={a['ratio']/a['n']:.3f}")


if __name__ == "__main__":
    main()
