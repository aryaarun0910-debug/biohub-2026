"""Small conditional-recall audit for track-conditioned raw-image redetection.

This intentionally defaults to ONE crop. It audits only GT-known misses adjacent
to matched predicted tracks; it does not estimate the FP cost of firing searches
from every dangling track in a blind movie.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from biotrack.metric_numpy import MAX_DISTANCE, SCALE  # noqa: E402
from core import load_sample, local_peaks, query_rescue, track_conditioned_queries  # noqa: E402
from measure_oracles import discover  # noqa: E402


def open_volume(path: Path):
    import zarr

    root = zarr.open(path, mode="r")
    return root["0"] if hasattr(root, "keys") else root


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pred-dir", action="append", required=True, type=Path)
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--image-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--fold", choices=["44b6", "6bba"], required=True)
    ap.add_argument("--max-crops", type=int, default=1, help="safety scope; raise deliberately for broader runs")
    ap.add_argument("--search-radius-um", type=float, default=MAX_DISTANCE)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    preds = discover(args.pred_dir, {args.fold})
    selected = list(sorted(preds.items()))[: args.max_crops]
    totals = defaultdict(int)
    for stem, pred_path in selected:
        pred = load_sample(pred_path)
        gt = load_sample(args.gt_dir / f"{stem}.geff")
        queries = track_conditioned_queries(pred, gt)
        arr = open_volume(args.image_dir / f"{stem}.zarr")
        crop = defaultdict(int)
        for query in queries:
            crop["queries"] += 1
            center_distance = float(np.linalg.norm((query.center_zyx - query.truth_zyx) * np.asarray(SCALE)))
            crop["center_in_gate"] += center_distance <= MAX_DISTANCE
            peaks = local_peaks(
                np.asarray(arr[query.t]), query.center_zyx, search_radius_um=args.search_radius_um, top_k=args.top_k
            )
            for k in (1, 3, 5):
                crop[f"rescue_at_{k}"] += query_rescue(query, peaks[: min(k, len(peaks))])
            crop[f"model_{query.motion_model}"] += 1
        for key, value in crop.items():
            totals[key] += value
        n = crop["queries"] or 1
        print(
            f"{stem}: queries={crop['queries']} center-gate={crop['center_in_gate']/n:.3f} "
            f"rescue@1/3/5={crop['rescue_at_1']/n:.3f}/{crop['rescue_at_3']/n:.3f}/{crop['rescue_at_5']/n:.3f} "
            f"CV={crop['model_constant_velocity']} stationary={crop['model_stationary']}"
        )
    n = totals["queries"] or 1
    print(
        f"POOLED CONDITIONAL RECALL ({len(selected)} crops, {totals['queries']} queries): "
        f"center-gate={totals['center_in_gate']/n:.3f}, "
        f"rescue@1/3/5={totals['rescue_at_1']/n:.3f}/{totals['rescue_at_3']/n:.3f}/{totals['rescue_at_5']/n:.3f}"
    )
    print("Caveat: GT selects the audit cases. Measure blind-query FP/count cost before promoting this to inference.")


if __name__ == "__main__":
    main()
