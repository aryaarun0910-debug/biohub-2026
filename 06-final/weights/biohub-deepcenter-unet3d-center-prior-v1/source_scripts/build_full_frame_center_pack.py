#!/usr/bin/env python
"""Build a standalone full-frame center detector artifact.

This artifact is intentionally separate from the main UNet+transformer support
pack. It carries a DeepCenterUNet3D checkpoint plus calibration diagnostics so
submission notebooks can use it as a conservative gated/blended node-rescue
prior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_FILES = [
    "SNAPSHOT_MANIFEST.json",
    "history.csv",
    "split_manifest.json",
    "gate_summary.json",
    "gate_threshold_metrics.csv",
    "gate_frame_metrics.csv",
    "gate_peak_samples.csv",
]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def file_count_and_bytes(path: Path) -> tuple[int, int]:
    count = 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            total += item.stat().st_size
    return count, total


def file_info(path: Path, relative_path: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_torch_checkpoint_summary(path: Path) -> dict[str, Any]:
    try:
        import torch  # type: ignore

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"read_error": repr(exc)}

    history = checkpoint.get("history", [])
    summary: dict[str, Any] = {
        "epoch": checkpoint.get("epoch"),
        "best_score": checkpoint.get("best_score"),
        "history_len": len(history) if isinstance(history, list) else None,
    }
    if isinstance(history, list) and history:
        summary["history_last"] = history[-1]
        best_row = max(history, key=lambda row: float(row.get("score", float("-inf"))))
        summary["history_best"] = best_row
    return summary


def choose_gate_thresholds(gate_summary: dict[str, Any]) -> dict[str, Any]:
    rows = gate_summary.get("threshold_metrics", []) if isinstance(gate_summary, dict) else []
    clean_rows: list[dict[str, float]] = []
    for row in rows:
        try:
            clean_rows.append(
                {
                    "threshold": float(row["threshold"]),
                    "precision_sparse": float(row["precision_sparse"]),
                    "recall_sparse": float(row["recall_sparse"]),
                    "f1_sparse": float(row["f1_sparse"]),
                    "pred_per_frame": float(row["pred_per_frame"]),
                }
            )
        except Exception:
            continue

    suggestions: dict[str, Any] = {}
    finite_rows = [
        row
        for row in clean_rows
        if row["precision_sparse"] == row["precision_sparse"]
        and row["recall_sparse"] == row["recall_sparse"]
        and row["f1_sparse"] == row["f1_sparse"]
    ]
    if finite_rows:
        suggestions["max_sparse_f1"] = max(finite_rows, key=lambda row: row["f1_sparse"])
        for target in [0.25, 0.35, 0.50]:
            eligible = [row for row in finite_rows if row["precision_sparse"] >= target]
            if eligible:
                suggestions[f"lowest_threshold_precision_ge_{target:g}"] = min(
                    eligible,
                    key=lambda row: row["threshold"],
                )
    return suggestions


def copy_source_scripts(output: Path) -> dict[str, dict[str, Any]]:
    dst_dir = output / "source_scripts"
    dst_dir.mkdir(parents=True, exist_ok=True)
    source_files = [
        (ROOT / "scripts" / "train_full_frame_center_detector.py", "train_full_frame_center_detector.py"),
        (ROOT / "scripts" / "build_full_frame_center_pack.py", "build_full_frame_center_pack.py"),
        (ROOT / "cloud_training" / "scripts" / "run_full_frame_center_training.sh", "run_full_frame_center_training.sh"),
        (ROOT / "cloud_training" / "scripts" / "package_and_upload_full_frame_center_pack.sh", "package_and_upload_full_frame_center_pack.sh"),
    ]
    replacements = {
        "cloud_training": "cloud_training",
        "gpu": "gpu",
        "GPU": "GPU",
        "gpu": "gpu",
    }
    copied: dict[str, dict[str, Any]] = {}
    for src, public_name in source_files:
        if not src.exists():
            continue
        dst = dst_dir / public_name
        shutil.copy2(src, dst)
        if dst.suffix in {".py", ".sh", ".md", ".txt"}:
            text = dst.read_text(errors="ignore")
            updated = text
            for old, new in replacements.items():
                updated = updated.replace(old, new)
            if updated != text:
                dst.write_text(updated)
        copied[public_name] = file_info(dst, f"source_scripts/{public_name}")
    return copied


def validate_source(source: Path) -> None:
    required = [
        source / "best.pt",
        source / "checkpoint_last.pt",
        source / "config.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Full-frame center source is missing required files:\n"
            + "\n".join(map(str, missing))
        )


def write_readme(output: Path, dataset_id: str) -> None:
    text = f"""# Biohub Full-Frame Center Detector Pack

This dataset contains a standalone DeepCenterUNet3D center-heatmap detector.
It is intended as an auxiliary gated/blended prior for Biohub cell tracking
submissions, not as a replacement for the primary graph model.

Important paths:

```text
weights/full_frame_center/best.pt
weights/full_frame_center/checkpoint_last.pt
weights/full_frame_center/config.json
weights/full_frame_center/history.csv
weights/full_frame_center/split_manifest.json
weights/full_frame_center/gate_summary.json
weights/full_frame_center/gate_threshold_metrics.csv
weights/full_frame_center/gate_frame_metrics.csv
weights/full_frame_center/gate_peak_samples.csv
ARTIFACT_MANIFEST.json
```

Coordinate contract:

```text
coordinate order: z, y, x
coordinate unit:  original voxel
voxel scale:      z=1.625, y=x=0.40625 microns/voxel
```

The detector runs on an XY-pooled image volume. A heatmap peak at pooled
coordinate `(z, y, x)` maps back to original voxel coordinates as:

```text
z_orig = z
y_orig = y * pool_factor + (pool_factor - 1) / 2
x_orig = x * pool_factor + (pool_factor - 1) / 2
```

The gate diagnostic files use sparse labels, so their precision/recall values
are calibration signals rather than complete-cell metrics. Use the model as a
high-confidence rescue source near graph gaps, short components, or unmatched
motion endpoints.

Kaggle input path:

```text
/kaggle/input/datasets/{dataset_id}/ARTIFACT_MANIFEST.json
```
"""
    (output / "README.md").write_text(text)


def write_dataset_metadata(output: Path, dataset_id: str, title: str, license_name: str) -> None:
    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": license_name}],
    }
    (output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def build_artifact(source: Path, output: Path, dataset_id: str, title: str, license_name: str) -> None:
    validate_source(source)
    weights_out = output / "weights" / "full_frame_center"
    copy_tree(source, weights_out)

    config = load_json(weights_out / "config.json")
    gate_summary = load_json(weights_out / "gate_summary.json")
    threshold_suggestions = choose_gate_thresholds(gate_summary)

    diagnostics: dict[str, dict[str, Any]] = {}
    for name in DIAGNOSTIC_FILES:
        path = weights_out / name
        if path.exists():
            diagnostics[name] = file_info(path, f"weights/full_frame_center/{name}")

    source_scripts = copy_source_scripts(output)
    output_count, output_bytes = file_count_and_bytes(output)

    manifest = {
        "artifact_name": output.name,
        "artifact_type": "biohub_full_frame_center_detector",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "model": {
            "method": "full_frame_center",
            "architecture": "DeepCenterUNet3D",
            "role": "auxiliary_center_prior",
            "best_checkpoint": file_info(weights_out / "best.pt", "weights/full_frame_center/best.pt"),
            "last_checkpoint": file_info(weights_out / "checkpoint_last.pt", "weights/full_frame_center/checkpoint_last.pt"),
            "best_checkpoint_summary": read_torch_checkpoint_summary(weights_out / "best.pt"),
            "last_checkpoint_summary": read_torch_checkpoint_summary(weights_out / "checkpoint_last.pt"),
            "config": config,
        },
        "training_contract": {
            "target": "sparse center heatmap",
            "loss": "positive-unlabeled weighted BCE on center heatmap logits",
            "positive_labels": "sparse GEFF node centroids rendered as Gaussian blobs",
            "negative_labels": "dark background sampled by image quantile",
            "ignored_or_low_weight_region": "bright unlabeled voxels receive small weight to reduce sparse-label false negatives",
            "normalization": {
                "type": "per-frame percentile normalization",
                "lo_pct": config.get("norm_lo_pct"),
                "hi_pct": config.get("norm_hi_pct"),
                "clip_lo": config.get("norm_clip_lo"),
                "clip_hi": config.get("norm_clip_hi"),
            },
        },
        "coordinate_contract": {
            "coordinate_order": ["z", "y", "x"],
            "coordinate_unit": "original_voxel",
            "voxel_scale_um": {"z": 1.625, "y": 0.40625, "x": 0.40625},
            "pool_factor": config.get("pool_factor"),
            "pooled_peak_to_original_voxel": {
                "z_orig": "z",
                "y_orig": "y * pool_factor + (pool_factor - 1) / 2",
                "x_orig": "x * pool_factor + (pool_factor - 1) / 2",
            },
        },
        "gate_and_blend_contract": {
            "recommended_use": "high-confidence node rescue near graph gaps, short components, or unmatched motion endpoints",
            "avoid": "adding all peaks directly to the node set without context gates",
            "candidate_features": [
                "full_frame_score",
                "nearest_existing_node_um",
                "frame_candidate_rank",
                "local_intensity_refined_centroid",
                "component_length",
                "gap_length",
                "motion_residual_um",
                "linefit_residual_um",
            ],
            "diagnostic_files": diagnostics,
            "threshold_suggestions_from_sparse_labels": threshold_suggestions,
            "warning": "Sparse-label precision/recall are calibration features, not complete-cell leaderboard estimates.",
        },
        "contents": {
            "files": output_count,
            "bytes": output_bytes,
            "source_scripts": source_scripts,
        },
    }
    (output / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_readme(output, dataset_id)
    write_dataset_metadata(output, dataset_id, title, license_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("weights/full_frame_center_v1"))
    parser.add_argument("--output", type=Path, default=Path("assets/biohub-full-frame-center-pack-v1"))
    parser.add_argument("--kaggle-id", default="pilkwang/biohub-full-frame-center-pack-v1")
    parser.add_argument("--title", default="Biohub Full-Frame Center Detector Pack V1")
    parser.add_argument("--license", default="CC0-1.0")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and args.clean:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    build_artifact(
        source=source,
        output=output,
        dataset_id=args.kaggle_id,
        title=args.title,
        license_name=args.license,
    )

    if args.zip:
        archive = shutil.make_archive(str(output), "zip", output)
        print(f"Wrote {archive}")

    count, total = file_count_and_bytes(output)
    print(f"Wrote {output}")
    print(f"Files: {count}")
    print(f"Bytes: {total:,}")
    print(f"Manifest: {output / 'ARTIFACT_MANIFEST.json'}")


if __name__ == "__main__":
    main()
