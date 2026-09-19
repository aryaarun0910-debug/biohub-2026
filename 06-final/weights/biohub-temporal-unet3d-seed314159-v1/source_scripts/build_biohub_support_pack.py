#!/usr/bin/env python
"""Build a lightweight Biohub tracking inference artifact.

The public full artifact contains:

    repo/
    weights/
    wheels/

For our notebook variants, only ``repo/`` and ``weights/`` are required when
dependencies are installed separately. This script creates a reproducible
support artifact with a manifest and optional Kaggle dataset metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "PublicNotebook" / "cellmot-baseline-artifacts"
DEFAULT_OUTPUT = ROOT / "PublicNotebook" / "biohub-tracking-support-pack-v1"
DEFAULT_REQUIREMENTS = ROOT / "requirements-unet-ilp.txt"
DEFAULT_PREDOWNLOAD_REQUIREMENTS = ROOT / "requirements-unet-ilp-kaggle-predownload.txt"
DEFAULT_KAGGLE_INSTALL_COMMAND = ROOT / "kaggle_dependency_install_command.txt"


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


def validate_source(source: Path) -> None:
    required = [
        source / "repo" / "scripts" / "predict_unet_transformer.py",
        source / "repo" / "scripts" / "train_unet_transformer.py",
        source / "repo" / "src" / "tracking_cellmot" / "models" / "temporal_unet.py",
        source / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth",
        source / "weights" / "unet_transformer" / "split_0" / "config.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source files:\n" + "\n".join(map(str, missing)))


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def sanitise_repo_names(repo_dir: Path) -> None:
    """Use neutral package/docs names inside the uploaded support artifact."""
    for generated_dir in ["predictions", "results", "__pycache__"]:
        path = repo_dir / generated_dir
        if path.exists():
            shutil.rmtree(path)

    old_pkg = repo_dir / "src" / "tracking_cellmot"
    new_pkg = repo_dir / "src" / "biohub_tracking"
    if old_pkg.exists():
        if new_pkg.exists():
            shutil.rmtree(new_pkg)
        old_pkg.rename(new_pkg)

    replacements = {
        "tracking_cellmot": "biohub_tracking",
        "tracking-cellmot": "biohub-tracking",
        "CellMot": "Biohub tracking",
        "cellmot": "biohub_tracking",
        "CELLMOT_DATA_DIR": "BIOHUB_DATA_DIR",
    }
    for path in repo_dir.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md", ".txt"}:
            continue
        text = path.read_text(errors="ignore")
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated)


def optional_file_info(path: Path, relative_path: str) -> dict | None:
    if not path.exists():
        return None
    return {
        "weight_path": relative_path,
        "weight_bytes": path.stat().st_size,
        "weight_sha256": sha256_file(path),
    }


def copy_source_scripts(output: Path) -> dict[str, dict]:
    source_output = output / "source_scripts"
    source_output.mkdir(parents=True, exist_ok=True)
    source_files = [
        (ROOT / "scripts" / "train_full_frame_center_detector.py", "train_full_frame_center_detector.py"),
        (ROOT / "scripts" / "build_biohub_support_pack.py", "build_biohub_support_pack.py"),
        (ROOT / "cloud_training" / "scripts" / "run_full_frame_center_training.sh", "run_full_frame_center_training.sh"),
        (ROOT / "cloud_training" / "scripts" / "package_and_upload_support_pack_version.sh", "package_and_upload_support_pack_version.sh"),
    ]
    copied: dict[str, dict] = {}
    neutral_replacements = {
        "cloud_training": "cloud_training",
        "run_main_training_oneshot.sh": "run_main_training_oneshot.sh",
        "unet_transformer_main_v1": "unet_transformer_main_v1",
        "gpu": "gpu",
        "GPU": "GPU",
        "gpu": "gpu",
    }
    for src, public_name in source_files:
        if not src.exists():
            continue
        dst = source_output / public_name
        shutil.copy2(src, dst)
        if dst.suffix in {".py", ".sh", ".md", ".txt"}:
            text = dst.read_text(errors="ignore")
            updated = text
            for old, new in neutral_replacements.items():
                updated = updated.replace(old, new)
            if updated != text:
                dst.write_text(updated)
        copied[public_name] = {
            "path": f"source_scripts/{dst.name}",
            "bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
        }
    return copied


def write_manifest(output: Path, source: Path, include_wheels: bool) -> None:
    weight = output / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
    config = output / "weights" / "unet_transformer" / "split_0" / "config.json"
    training_config = output / "weights" / "unet_transformer" / "split_0" / "training_config.json"
    full_frame_weight = output / "weights" / "full_frame_center" / "best.pt"
    full_frame_config = output / "weights" / "full_frame_center" / "config.json"
    repo_count, repo_bytes = file_count_and_bytes(output / "repo")
    weight_count, weight_bytes = file_count_and_bytes(output / "weights")
    notes = [
        "Use kaggle_dependency_install_command.txt in Kaggle dependency input before inference, or attach offline wheels.",
        "Do not quote zarr>=3.0.10,<4 in Kaggle dependency input; helper parsers may treat quotes as literal text.",
        "The model architecture is reconstructed from repo/ plus config.json.",
    ]
    if include_wheels:
        notes.insert(0, "This support artifact includes offline dependency wheels for code-submission reruns.")
    else:
        notes.insert(0, "This support artifact intentionally excludes dependency wheels.")

    unet_model = {
        "method": "unet_transformer",
        "weight_path": "weights/unet_transformer/split_0/edge_predictor_best.pth",
        "weight_bytes": weight.stat().st_size,
        "weight_sha256": sha256_file(weight),
        "config": json.loads(config.read_text()),
    }
    if training_config.exists():
        unet_model["training"] = json.loads(training_config.read_text())
    models = {"unet_transformer": unet_model}
    full_frame_info = optional_file_info(
        full_frame_weight,
        "weights/full_frame_center/best.pt",
    )
    if full_frame_info is not None:
        full_frame_info["method"] = "full_frame_center"
        full_frame_info["role"] = "auxiliary_center_prior"
        if full_frame_config.exists():
            full_frame_info["config"] = json.loads(full_frame_config.read_text())
        diagnostic_files = {}
        for name in [
            "history.csv",
            "split_manifest.json",
            "gate_summary.json",
            "gate_threshold_metrics.csv",
            "gate_frame_metrics.csv",
            "gate_peak_samples.csv",
        ]:
            path = full_frame_weight.parent / name
            if path.exists():
                diagnostic_files[name] = {
                    "path": f"weights/full_frame_center/{name}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        if diagnostic_files:
            full_frame_info["diagnostics"] = diagnostic_files
        models["full_frame_center"] = full_frame_info

    source_scripts = copy_source_scripts(output)
    compatibility = {
        "schema_version": 1,
        "coordinate_order": ["z", "y", "x"],
        "coordinate_unit": "original_voxel",
        "voxel_scale_um": {"z": 1.625, "y": 0.40625, "x": 0.40625},
        "models": {
            "unet_transformer": {
                "role": "primary_graph_generator",
                "outputs": [
                    "node detections in original z/y/x voxel coordinates",
                    "learned edge probabilities between adjacent frames",
                    "ILP candidate graph before notebook-level graph repair",
                ],
            },
            "full_frame_center": {
                "role": "auxiliary_center_prior",
                "outputs": [
                    "center heatmap on XY-pooled volume",
                    "peak coordinates mapped back to original z/y/x voxel coordinates",
                    "peak score usable as a conservative node-rescue gate",
                ],
                "recommended_use": (
                    "Use only as a high-confidence rescue prior near graph gaps, "
                    "short components, or unmatched motion endpoints. Avoid adding "
                    "all peaks directly to the node set."
                ),
            },
        },
        "gate_diagnostics": {
            "purpose": "threshold calibration for full_frame_center node rescue",
            "warning": "Sparse-label precision/recall are calibration features, not complete-cell metrics.",
            "files": [
                "weights/full_frame_center/gate_summary.json",
                "weights/full_frame_center/gate_threshold_metrics.csv",
                "weights/full_frame_center/gate_frame_metrics.csv",
                "weights/full_frame_center/gate_peak_samples.csv",
            ],
        },
    }

    manifest = {
        "artifact_name": output.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "public learned baseline artifact, repackaged locally",
        "compatibility": compatibility,
        "contents": {
            "repo": {"files": repo_count, "bytes": repo_bytes},
            "weights": {"files": weight_count, "bytes": weight_bytes},
            "wheels_included": include_wheels,
            "full_frame_center_included": full_frame_info is not None,
            "source_scripts": source_scripts,
        },
        "model": unet_model,
        "models": models,
        "notes": notes,
    }
    (output / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_readme(output: Path, include_wheels: bool) -> None:
    wheel_line = "- `wheels/`: offline dependency wheels for code-submission reruns." if include_wheels else ""
    not_included = "" if include_wheels else """
Not included:

- `wheels/`: dependency wheels are intentionally excluded to keep the artifact small.
"""
    text = """# Biohub Tracking Support Pack V1

This artifact contains the learned inference code and weights required by the
UNet + node-transformer + ILP submission notebooks. When present, it can also
contain a separately trained full-frame center detector used only as an
auxiliary node-rescue prior.

Included:

- `repo/`: inference/training source used by the notebook.
- `weights/`: `unet_transformer/split_0/edge_predictor_best.pth` and `config.json`.
- `source_scripts/`: local training/packaging scripts used to create the support pack.
- `requirements-unet-ilp.txt`: Python dependencies to install separately.
- `requirements-unet-ilp-kaggle-predownload.txt`: quote-free package list for Kaggle dependency helpers.
- `kaggle_dependency_install_command.txt`: one-line Kaggle dependency input command.
- `ARTIFACT_MANIFEST.json`: generated provenance and weight checksum.
{wheel_line}
{not_included}

The manifest includes a `compatibility` section that defines the shared
coordinate contract:

```text
coordinate order: z, y, x
coordinate unit:  original voxel
voxel scale:      z=1.625, y=x=0.40625 microns/voxel
```

If `weights/full_frame_center/` exists, the detector outputs are calibrated by:

```text
gate_summary.json
gate_threshold_metrics.csv
gate_frame_metrics.csv
gate_peak_samples.csv
```

These files are sparse-label calibration features. They are intended for
conservative node-rescue gates, not as complete-cell leaderboard estimates.

Kaggle dependency input command:

```bash
pip install tracksdata zarr>=3.0.10,<4 pyscipopt geff ilpy polars blosc2 dask imagecodecs pyarrow rustworkx sqlalchemy
```

Do not add quotes around `zarr>=3.0.10,<4` in Kaggle's dependency input.
""".format(wheel_line=wheel_line, not_included=not_included)
    (output / "README.md").write_text(text)


def write_kaggle_metadata(output: Path, kaggle_id: str, title: str) -> None:
    metadata = {
        "title": title,
        "id": kaggle_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--predownload-requirements", type=Path, default=DEFAULT_PREDOWNLOAD_REQUIREMENTS)
    parser.add_argument("--kaggle-install-command", type=Path, default=DEFAULT_KAGGLE_INSTALL_COMMAND)
    parser.add_argument(
        "--weights-source",
        type=Path,
        default=None,
        help=(
            "Optional trained weights source. Pass either a full weights/ directory "
            "or a method directory containing split_0/edge_predictor_best.pth; "
            "method directories are packaged as weights/unet_transformer/."
        ),
    )
    parser.add_argument(
        "--full-frame-center-source",
        type=Path,
        default=None,
        help=(
            "Optional directory containing a full-frame center detector "
            "best.pt/checkpoint_last.pt/config.json. It is packaged under "
            "weights/full_frame_center/."
        ),
    )
    parser.add_argument("--include-wheels", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--zip", action="store_true", help="Also create <output>.zip")
    parser.add_argument("--kaggle-id", default="pilkwang/biohub-tracking-support-pack-v1")
    parser.add_argument("--title", default="Biohub Tracking Support Pack V1")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validate_source(source)

    if output.exists() and args.clean:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    copy_tree(source / "repo", output / "repo")
    sanitise_repo_names(output / "repo")

    weights_source = args.weights_source.expanduser().resolve() if args.weights_source else source / "weights"
    if (weights_source / "split_0" / "edge_predictor_best.pth").exists():
        if (output / "weights").exists():
            shutil.rmtree(output / "weights")
        (output / "weights").mkdir(parents=True, exist_ok=True)
        copy_tree(weights_source, output / "weights" / "unet_transformer")
    else:
        copy_tree(weights_source, output / "weights")
    if args.include_wheels:
        copy_tree(source / "wheels", output / "wheels")
    elif (output / "wheels").exists():
        shutil.rmtree(output / "wheels")

    if args.full_frame_center_source is not None:
        full_frame_source = args.full_frame_center_source.expanduser().resolve()
        if not (full_frame_source / "best.pt").exists():
            raise FileNotFoundError(f"Missing full-frame detector best.pt: {full_frame_source}")
        full_frame_output = output / "weights" / "full_frame_center"
        copy_tree(full_frame_source, full_frame_output)

    if args.requirements.exists():
        shutil.copy2(args.requirements, output / "requirements-unet-ilp.txt")
    if args.predownload_requirements.exists():
        shutil.copy2(
            args.predownload_requirements,
            output / "requirements-unet-ilp-kaggle-predownload.txt",
        )
    if args.kaggle_install_command.exists():
        shutil.copy2(args.kaggle_install_command, output / "kaggle_dependency_install_command.txt")

    write_manifest(output, source, args.include_wheels)
    write_readme(output, args.include_wheels)
    write_kaggle_metadata(output, args.kaggle_id, args.title)

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
