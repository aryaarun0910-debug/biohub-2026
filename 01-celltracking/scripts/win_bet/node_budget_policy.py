"""Freeze GT-free per-crop detector thresholds from DetPeak sidecars.

This instrument only chooses thresholds. It does not claim a tracking score: newly admitted
peaks must pass through the full association and scorer round-trip before promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def threshold_for_budget(
    logits: np.ndarray,
    target_count: int,
    *,
    export_floor: float = 0.5,
) -> dict:
    """Choose a scalar probability threshold under the pipeline's strict ``>`` rule.

    A threshold cannot split equal logits. We therefore choose the attainable count closest
    to the requested budget, preferring the smaller count on an exact tie. No coordinates,
    crop identity, ground truth, or embryo identity enter the decision.
    """
    scores = sigmoid(np.asarray(logits, dtype=np.float64))
    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    if scores.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if not np.isfinite(scores).all():
        raise ValueError("logits must be finite")

    unique = np.unique(scores)
    # Candidate thresholds include the floor (all exported peaks), every score (strictly
    # excludes that tied group), and just below the maximum (admits the maximum group).
    candidates = np.unique(np.concatenate([
        np.asarray([float(export_floor), 1.0], dtype=np.float64),
        unique,
        np.nextafter(unique, -np.inf),
    ]))
    candidates = candidates[(candidates >= export_floor) & (candidates <= 1.0)]
    counts = np.asarray([(scores > threshold).sum() for threshold in candidates], dtype=np.int64)
    order = np.lexsort((-candidates, counts, np.abs(counts - int(target_count))))
    chosen = int(order[0])
    threshold = float(candidates[chosen])
    selected = int(counts[chosen])
    return {
        "threshold": threshold,
        "target_count": int(target_count),
        "selected_count": selected,
        "count_error": selected - int(target_count),
        "tie_limited": selected != int(target_count),
    }


def freeze_directory(
    peaks_dir: Path,
    estimates: dict[str, int],
    *,
    target_multiplier: float,
    export_floor: float = 0.5,
) -> dict:
    if not target_multiplier > 0:
        raise ValueError("target_multiplier must be positive")
    files = sorted(peaks_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no DetPeak sidecars under {peaks_dir}")
    missing = sorted(path.stem for path in files if path.stem not in estimates)
    extra = sorted(set(estimates) - {path.stem for path in files})
    if missing or extra:
        raise ValueError(f"estimate/sidecar mismatch: missing={missing}, extra={extra}")

    crops = {}
    for path in files:
        with np.load(path, allow_pickle=False) as sidecar:
            logits = sidecar["logit"]
            pipeline_threshold = float(sidecar["pipeline_threshold"])
            recorded_count = int(sidecar["pipeline_peak_count"])
        reconstructed = int(np.count_nonzero(sigmoid(logits) > pipeline_threshold))
        if reconstructed != recorded_count:
            raise RuntimeError(
                f"{path.stem}: pipeline reconstruction mismatch "
                f"({reconstructed} != {recorded_count})"
            )
        target = int(round(int(estimates[path.stem]) * target_multiplier))
        crops[path.stem] = {
            "estimated_number_of_nodes": int(estimates[path.stem]),
            "pipeline_threshold": pipeline_threshold,
            "pipeline_peak_count": recorded_count,
            **threshold_for_budget(logits, target, export_floor=export_floor),
        }
    return {
        "schema_version": 1,
        "policy": "scalar threshold nearest to round(estimated_number_of_nodes * target_multiplier)",
        "target_multiplier": float(target_multiplier),
        "export_floor": float(export_floor),
        "routes_on": ["candidate_score_distribution", "estimated_number_of_nodes"],
        "forbidden_routes": ["dataset", "embryo", "crop_name", "ground_truth"],
        "crops": crops,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--peaks-dir", type=Path, required=True)
    parser.add_argument("--estimates-json", type=Path, required=True)
    parser.add_argument("--target-multiplier", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    estimates = json.loads(args.estimates_json.read_text(encoding="utf-8"))
    result = freeze_directory(
        args.peaks_dir,
        {str(key): int(value) for key, value in estimates.items()},
        target_multiplier=args.target_multiplier,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"NODE_BUDGET_POLICY crops={len(result['crops'])} out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
