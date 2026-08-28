r"""Node-recall retention: what the pipeline does to annotated cells the DETECTOR already found.

WHY
---
``LEVER-0028`` assumes the node deficit (``N_pred / estimated_number_of_nodes`` below 1) is a
DETECTION deficit and re-thresholds the detector to close it. Measured directly on the P28
substrate, the detector already emits at or above the estimate on both folds, and the deficit
first appears after the ILP solver. That relocates the question: the useful quantity is not how
many peaks exist, but how many annotated cells the detector found and the pipeline then LOST.

This instrument measures node recall at the two stages whose coordinates survive to disk:

  stage 1  DETECTOR   every local-max peak above ``BIOHUB_DET_THRESHOLD``, from the sidecar
  stage 2  FINAL      the champion export's node rows, after ILP + all post-processing

Both use the official one-to-one bipartite matching at ``MAX_DISTANCE = 7.0 um`` and scale
``(1.625, 0.40625, 0.40625)``, so the two recalls are directly comparable and the difference is
the pipeline's net effect on detected cells.

NET, NOT GROSS
--------------
Post-processing both removes nodes (ILP pruning, isolated-node pruning, the short-track filter)
and inserts them (synthetic gap nodes, safe divisions). This instrument reports the NET change
in matched cells. It does not attribute the net to individual stages, because the intermediate
graphs are not written to disk -- only their counts are, in ``run_stats.csv``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detpeak_curve import (  # noqa: E402
    DOWNSAMPLE,
    MAX_DISTANCE_UM,
    SCALE_UM,
    match_one_to_one,
    peaks_to_um,
    sigmoid,
)
from detpeak_fold_curve import load_gt  # noqa: E402


def crop_retention(sidecar: Path, gt_geff: Path, final_nodes: np.ndarray) -> dict:
    with np.load(sidecar, allow_pickle=False) as z:
        times = z["t"].astype(np.int64)
        zyx = z["zyx"].astype(np.int64)
        logits = z["logit"].astype(np.float64)
        threshold = float(z["pipeline_threshold"])
    keep = sigmoid(logits) > threshold
    gt, n_est = load_gt(gt_geff)
    n_gt = sum(len(v) for v in gt.values())
    scale = np.asarray(SCALE_UM, dtype=np.float64)

    det_matched = 0
    fin_matched = 0
    for frame, g in gt.items():
        if not len(g):
            continue
        m = keep & (times == frame)
        if m.any():
            det_matched += match_one_to_one(peaks_to_um(zyx[m]), g, MAX_DISTANCE_UM)
        fm = final_nodes[:, 0] == frame
        if fm.any():
            fin_matched += match_one_to_one(final_nodes[fm, 1:] * scale, g, MAX_DISTANCE_UM)
    return {
        "crop": sidecar.stem,
        "n_gt": int(n_gt),
        "n_est": n_est,
        "detector_nodes": int(keep.sum()),
        "final_nodes": int(len(final_nodes)),
        "detector_matched": int(det_matched),
        "final_matched": int(fin_matched),
        "detector_recall": det_matched / n_gt if n_gt else 0.0,
        "final_recall": fin_matched / n_gt if n_gt else 0.0,
        "retention_delta": (fin_matched - det_matched) / n_gt if n_gt else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--peaks-dir", type=Path, required=True)
    ap.add_argument("--final-csv", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--max-crops", type=int)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    final = (
        pl.read_csv(args.final_csv)
        .filter(pl.col("row_type") == "node")
        .select(["dataset", "t", "z", "y", "x"])
    )
    by_crop = {
        (k[0] if isinstance(k, tuple) else k): g.select(["t", "z", "y", "x"]).to_numpy().astype(np.float64)
        for k, g in final.group_by("dataset")
    }

    sidecars = sorted(args.peaks_dir.glob("*.npz"))
    if args.max_crops:
        sidecars = sidecars[: args.max_crops]

    rows = []
    for i, sidecar in enumerate(sidecars, 1):
        gt_geff = args.gt_dir / f"{sidecar.stem}.geff"
        if not gt_geff.exists():
            raise SystemExit(f"missing GT for {sidecar.stem}")
        if sidecar.stem not in by_crop:
            raise SystemExit(f"crop {sidecar.stem} absent from {args.final_csv}")
        rows.append(crop_retention(sidecar, gt_geff, by_crop[sidecar.stem]))
        print(f"  [{i}/{len(sidecars)}] {sidecar.stem}", flush=True)

    n_gt = sum(r["n_gt"] for r in rows)
    det = sum(r["detector_matched"] for r in rows)
    fin = sum(r["final_matched"] for r in rows)
    deltas = np.array([r["retention_delta"] for r in rows], dtype=np.float64)
    result = {
        "schema_version": 1,
        "peaks_dir": str(args.peaks_dir),
        "final_csv": str(args.final_csv),
        "n_crops": len(rows),
        "n_gt": n_gt,
        "detector_nodes_total": sum(r["detector_nodes"] for r in rows),
        "final_nodes_total": sum(r["final_nodes"] for r in rows),
        "detector_recall_pooled": det / n_gt if n_gt else 0.0,
        "final_recall_pooled": fin / n_gt if n_gt else 0.0,
        "detected_cells_lost_net": int(det - fin),
        "recall_lost_net": (det - fin) / n_gt if n_gt else 0.0,
        "per_crop_retention_delta": {
            "median": float(np.median(deltas)),
            "mean": float(deltas.mean()),
            "n_crops_worse": int((deltas < 0).sum()),
            "n_crops_better": int((deltas > 0).sum()),
        },
        "per_crop": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"DETPEAK_RETENTION crops={len(rows)} "
        f"detector_recall={result['detector_recall_pooled']:.4f} "
        f"final_recall={result['final_recall_pooled']:.4f} "
        f"net_detected_cells_lost={result['detected_cells_lost_net']:,} "
        f"({result['recall_lost_net']:.4f} recall)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
