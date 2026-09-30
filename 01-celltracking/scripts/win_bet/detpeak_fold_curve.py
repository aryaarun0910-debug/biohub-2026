r"""Fold-level detection-threshold curve replayed from a full DetPeak export.

WHAT THIS IS FOR
----------------
``PKT-0024`` STEP 2 has to freeze ONE crop-budget function before any GPU is spent. That
choice needs the detection surface of the P28 champion substrate measured directly
(``FACT-0350`` forbids inheriting the old p20/p19 control deltas), across the whole
threshold range the export makes available, on every crop of a fold at once.
``detpeak_curve.py`` answers this for ONE crop against a hand-supplied GT array; this driver
is the fold-level version that reads GT straight from ``data/train/<crop>.geff`` and reports
per crop as well as pooled.

WHAT IT MEASURES, AND WHAT IT CANNOT
------------------------------------
It measures the DETECTION surface only: one-to-one node matching under the official
``MAX_DISTANCE = 7.0 um`` at scale ``(1.625, 0.40625, 0.40625)``, so node recall, precision,
F1, the predicted count, and ``N_pred / estimated_number_of_nodes``.

It does NOT and cannot produce a tracking score. Measured on this export: the predictor's
saved graph holds fewer nodes than the detector emitted above threshold, because the ILP
solver drops nodes it cannot place in a track. Every peak newly admitted by a lower threshold
must therefore pass through edge prediction and the ILP before it becomes a node, and neither
runs on CPU from a sidecar. Any adjusted-edge number quoted from this instrument would be a
projection, not a measurement. Freeze the budget function here; measure the score on GPU.

THE EXPORT FLOOR IS REAL
------------------------
The sidecars carry peaks down to sigmoid 0.5, not to 0. Thresholds below the floor are
unmeasurable from this artifact and the driver refuses them rather than silently reporting
the floor's numbers.

COORDINATES
-----------
Peaks are written in the downsampled grid before ``coords[:, 1:] *= ds_arr``, so microns are
``zyx * [1,4,4] * (1.625, 0.40625, 0.40625)`` = ``zyx * 1.625`` isotropically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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

EXPORT_FLOOR = 0.5  # detpeak_export.py BIOHUB_DETPEAK_EXPORT_T
DEFAULT_THRESHOLDS = "0.50,0.60,0.70,0.80,0.85,0.90,0.93,0.95,0.96875,0.98,0.99"


def load_gt(geff_path: Path) -> tuple[dict[int, np.ndarray], float]:
    """GT node coordinates per frame in microns, plus estimated_number_of_nodes."""
    from biotrack.metric import estimated_nodes, load_graph

    graph = load_graph(geff_path)
    rows = graph.node_attrs().to_pandas()
    scale = np.asarray(SCALE_UM, dtype=np.float64)
    out: dict[int, np.ndarray] = {}
    times = rows["t"].to_numpy().astype(np.int64)
    coords = rows[["z", "y", "x"]].to_numpy().astype(np.float64) * scale
    for frame in np.unique(times):
        out[int(frame)] = coords[times == frame]
    return out, float(estimated_nodes(geff_path))


def crop_curve(
    sidecar: Path,
    gt_geff: Path,
    thresholds: list[float],
) -> dict:
    with np.load(sidecar, allow_pickle=False) as z:
        times = z["t"].astype(np.int64)
        zyx = z["zyx"].astype(np.int64)
        logits = z["logit"].astype(np.float64)
        pipeline_threshold = float(z["pipeline_threshold"])
    probs = sigmoid(logits)
    gt, n_est = load_gt(gt_geff)
    n_gt = sum(len(v) for v in gt.values())
    frames = np.unique(times)

    rows = []
    for thr in thresholds:
        keep = probs > thr
        matched = 0
        n_pred = int(keep.sum())
        for frame in frames:
            m = keep & (times == frame)
            if not m.any():
                continue
            g = gt.get(int(frame))
            if g is None or not len(g):
                continue
            matched += match_one_to_one(peaks_to_um(zyx[m]), g, MAX_DISTANCE_UM)
        recall = matched / n_gt if n_gt else 0.0
        precision = matched / n_pred if n_pred else 0.0
        f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
        rows.append({
            "threshold": float(thr),
            "n_pred": n_pred,
            "matched": int(matched),
            "node_recall": recall,
            "node_precision": precision,
            "node_f1": f1,
            "n_pred_over_n_est": (n_pred / n_est) if n_est and np.isfinite(n_est) else float("nan"),
        })
    return {
        "crop": sidecar.stem,
        "n_gt": int(n_gt),
        "n_est": n_est,
        "exported_peaks": int(len(logits)),
        "pipeline_threshold": pipeline_threshold,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--peaks-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--max-crops", type=int)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    thresholds = [float(v) for v in args.thresholds.split(",")]
    below = [t for t in thresholds if t < EXPORT_FLOOR]
    if below:
        raise SystemExit(
            f"thresholds {below} are below the export floor {EXPORT_FLOOR}; "
            "the sidecars contain no peaks there and the result would be a silent lie"
        )

    sidecars = sorted(args.peaks_dir.glob("*.npz"))
    if args.max_crops:
        sidecars = sidecars[: args.max_crops]
    if not sidecars:
        raise SystemExit(f"no sidecars under {args.peaks_dir}")

    per_crop = []
    for i, sidecar in enumerate(sidecars, 1):
        gt_geff = args.gt_dir / f"{sidecar.stem}.geff"
        if not gt_geff.exists():
            raise SystemExit(f"missing GT for {sidecar.stem}: {gt_geff}")
        per_crop.append(crop_curve(sidecar, gt_geff, thresholds))
        print(f"  [{i}/{len(sidecars)}] {sidecar.stem}", flush=True)

    pooled = []
    for idx, thr in enumerate(thresholds):
        n_pred = sum(c["rows"][idx]["n_pred"] for c in per_crop)
        matched = sum(c["rows"][idx]["matched"] for c in per_crop)
        n_gt = sum(c["n_gt"] for c in per_crop)
        n_est = sum(c["n_est"] for c in per_crop if np.isfinite(c["n_est"]))
        recall = matched / n_gt if n_gt else 0.0
        precision = matched / n_pred if n_pred else 0.0
        ratios = [
            c["rows"][idx]["n_pred_over_n_est"]
            for c in per_crop
            if np.isfinite(c["rows"][idx]["n_pred_over_n_est"])
        ]
        pooled.append({
            "threshold": float(thr),
            "n_pred": n_pred,
            "n_gt": n_gt,
            "matched": matched,
            "node_recall": recall,
            "node_precision": precision,
            "node_f1": (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0,
            "n_pred_over_n_est_pooled": (n_pred / n_est) if n_est else float("nan"),
            "n_pred_over_n_est_crop_median": float(np.median(ratios)) if ratios else float("nan"),
            "crops_under_estimate": int(sum(1 for r in ratios if r < 1.0)),
        })

    result = {
        "schema_version": 1,
        "peaks_dir": str(args.peaks_dir),
        "export_floor": EXPORT_FLOOR,
        "max_distance_um": MAX_DISTANCE_UM,
        "scale_um": list(SCALE_UM),
        "downsample": list(DOWNSAMPLE),
        "n_crops": len(per_crop),
        "thresholds": thresholds,
        "pooled": pooled,
        "per_crop": per_crop,
        "caveat": (
            "DETECTION surface only. The ILP solver drops nodes downstream, so a lower "
            "threshold's score cannot be derived from these numbers."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    best = max(pooled, key=lambda r: r["node_f1"])
    print(
        f"DETPEAK_FOLD_CURVE crops={len(per_crop)} "
        f"best_f1_threshold={best['threshold']} f1={best['node_f1']:.4f} "
        f"recall={best['node_recall']:.4f} precision={best['node_precision']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
