"""CLI for the offline Trackastra ``ctc`` zero-shot association experiment."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from .adapter import (
    DEFAULT_VOXEL_SIZE_UM,
    estimated_mask_bytes,
    export_trackastra_result,
    load_frozen_detections,
    rasterize_ellipsoid_masks,
    write_mask_mapping,
)


def image_array(path: Path):
    import zarr

    group = zarr.open_group(str(path), mode="r")
    return group["0"]


def validate_model_folder(path: Path, *, strict: bool) -> list[str]:
    required = ["config.yaml", "model.pt", "train_config.yaml"]
    missing = [name for name in required if not (path / name).is_file()]
    if strict and missing:
        raise FileNotFoundError(f"Trackastra model folder is missing: {missing}")
    return missing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Trackastra ctc zero-shot linking on frozen Biohub point detections."
    )
    parser.add_argument("--detections", required=True, type=Path, help="predicted .geff or point CSV")
    parser.add_argument("--images", required=True, type=Path, help="matching Biohub .zarr image")
    parser.add_argument("--model-dir", required=True, type=Path, help="offline extracted Trackastra ctc folder")
    parser.add_argument("--output-geff", type=Path, default=Path("trackastra_prediction.geff"))
    parser.add_argument("--edge-scores", type=Path, default=Path("trackastra_edge_scores.csv"))
    parser.add_argument("--mask-npy", type=Path, default=Path("trackastra_instances.npy"))
    parser.add_argument("--mapping-csv", type=Path, default=Path("trackastra_mask_mapping.csv"))
    parser.add_argument("--radius-um", type=float, default=2.5)
    parser.add_argument("--voxel-size-um", type=float, nargs=3, default=DEFAULT_VOXEL_SIZE_UM)
    parser.add_argument("--device", choices=["automatic", "cuda", "cpu", "mps"], default="automatic")
    parser.add_argument("--mode", choices=["greedy_nodiv", "greedy", "ilp"], default="greedy")
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan; allocate nothing")
    parser.add_argument("--prepare-only", action="store_true", help="write masks/mapping but do not import Trackastra")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    detections = load_frozen_detections(args.detections)
    image = image_array(args.images)
    shape = tuple(int(v) for v in image.shape)
    detections.validate_shape(shape)
    missing = validate_model_folder(args.model_dir, strict=not (args.dry_run or args.prepare_only))
    plan = {
        "detections": len(detections.node_ids),
        "frames": shape[0],
        "image_shape_tzyx": shape,
        "mask_gib": round(estimated_mask_bytes(shape, detections) / 2**30, 3),
        "radius_um": args.radius_um,
        "voxel_size_um": tuple(args.voxel_size_um),
        "model_missing": missing,
        "mode": args.mode,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    masks, mapping = rasterize_ellipsoid_masks(
        detections,
        shape,
        radius_um=args.radius_um,
        voxel_size_um=args.voxel_size_um,
        output_npy=args.mask_npy,
    )
    write_mask_mapping(mapping, args.mapping_csv)
    print(f"instances -> {args.mask_npy}; identity mapping -> {args.mapping_csv}")
    if args.prepare_only:
        return 0

    try:
        from trackastra.model import Trackastra
    except ImportError as exc:
        raise RuntimeError(
            "Trackastra is not installed. Stage an offline wheel/repository plus the ctc model first."
        ) from exc

    model = Trackastra.from_folder(args.model_dir, device=args.device, batch_size=args.batch_size)
    # We intentionally use these 0.5.2 private APIs so candidate scores survive.
    for name in ("_predict", "_track_from_predictions"):
        if not hasattr(model, name):
            raise RuntimeError(f"incompatible Trackastra API: {name} is absent (pin 0.5.2)")
    if "edge_threshold" not in inspect.signature(model._predict).parameters:
        raise RuntimeError("incompatible Trackastra _predict signature (pin 0.5.2)")

    # Trackastra normalises the image internally.  np.asarray materialises this
    # one crop; process crops sequentially to bound memory.
    predictions = model._predict(
        np.asarray(image),
        masks,
        edge_threshold=args.edge_threshold,
        batch_size=args.batch_size,
    )
    selected = model._track_from_predictions(predictions, mode=args.mode)
    export_trackastra_result(
        detections,
        mapping,
        predictions,
        selected,
        output_geff=args.output_geff,
        output_edge_scores=args.edge_scores,
    )
    print(f"prediction -> {args.output_geff}; candidate scores -> {args.edge_scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
