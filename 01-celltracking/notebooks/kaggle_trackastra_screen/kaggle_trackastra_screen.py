"""One-crop Trackastra CTC zero-shot association screen on frozen fold-0 detections.

This is deliberately a falsification kernel, not a submission kernel. It changes
only edges, preserves every organizer detection coordinate, exports candidate
scores, and removes the ~0.8 GiB temporary instance mask before kernel output.
"""

from __future__ import annotations

import glob
import hashlib
import inspect
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path


WORK = Path("/kaggle/working")
MAX_CROPS_PER_FOLD = 10
FOLDS = (1,)
MODE = "greedy"
EDGE_THRESHOLD = 0.05
BATCH_SIZE = 16


def sh(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def one(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(f"No match: {pattern}")
    return Path(hits[0])


def find_image(name: str) -> Path:
    patterns = [
        f"/kaggle/input/competitions/biohub-cell-tracking-during-development/train/{name}.zarr",
        f"/kaggle/input/**/train/{name}.zarr",
        f"/kaggle/input/**/{name}.zarr",
    ]
    for pattern in patterns:
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"image zarr not found for {name}")


def prepare_bundle() -> tuple[Path, Path]:
    # Kaggle datasets unpack uploaded ZIPs. Prefer those unpacked directories,
    # while retaining ZIP support so the kernel is robust to either mount form.
    unpacked_model = sorted(glob.glob("/kaggle/input/**/ctc/ctc/model.pt", recursive=True))
    unpacked_code = sorted(glob.glob("/kaggle/input/**/adapter_code/adapter.py", recursive=True))
    if unpacked_model and unpacked_code:
        model_dir = Path(unpacked_model[0]).parent
        code_root = Path(unpacked_code[0]).parent
    else:
        ctc_zip = one("/kaggle/input/**/ctc.zip")
        code_zip = one("/kaggle/input/**/adapter_code.zip")
        digest = hashlib.sha256(ctc_zip.read_bytes()).hexdigest().upper()
        expected = "2EDE08481B2E1F4EA23A41DD3001E2151ECBCE7D978BE9829AD230F0A60AEA0E"
        if digest != expected:
            raise RuntimeError(f"Trackastra checkpoint hash mismatch: {digest}")
        model_root = WORK / "trackastra_model"
        code_root = WORK / "trackastra_adapter"
        for path in (model_root, code_root):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True)
        with zipfile.ZipFile(ctc_zip) as zf:
            zf.extractall(model_root)
        with zipfile.ZipFile(code_zip) as zf:
            zf.extractall(code_root)
        model_dir = model_root / "ctc"
    required = ["config.yaml", "model.pt", "train_config.yaml"]
    missing = [f for f in required if not (model_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"bad model bundle; missing {missing}")
    sys.path.insert(0, str(code_root))
    return model_dir, code_root


def main() -> None:
    t0 = time.time()
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    # OOF screen may use internet. The final submission will ship an offline
    # wheelhouse after this experiment proves signal.
    sh([
        sys.executable, "-m", "pip", "install", "-q",
        "trackastra==0.5.2", "tracksdata", "geff>=1.1.3.1.1",
        "zarr>=3.0.10,<4", "polars>=1.36",
    ])

    model_dir, _ = prepare_bundle()
    from adapter import (
        export_trackastra_result,
        load_frozen_detections,
        rasterize_ellipsoid_masks,
        write_mask_mapping,
    )
    import numpy as np
    import zarr
    from trackastra.model import Trackastra

    model = Trackastra.from_folder(model_dir, device="cuda", batch_size=BATCH_SIZE)
    for method in ("_predict", "_track_from_predictions"):
        if not hasattr(model, method):
            raise RuntimeError(f"Trackastra 0.5.2 API missing {method}")
    if "edge_threshold" not in inspect.signature(model._predict).parameters:
        raise RuntimeError("unexpected Trackastra _predict signature")

    temp_root = WORK / "trackastra_temp"
    temp_root.mkdir(exist_ok=True)
    for fold in FOLDS:
        pred_dirs = sorted(glob.glob(f"/kaggle/input/**/pred_geffs_split_{fold}", recursive=True))
        if not pred_dirs:
            seen = sorted(glob.glob("/kaggle/input/**/*.geff", recursive=True))[:20]
            raise FileNotFoundError(f"fold-{fold} prediction source absent; sample GEFFs={seen}")
        pred_dir = Path(pred_dirs[0])
        detections_paths = sorted(pred_dir.glob("*.geff"))[:MAX_CROPS_PER_FOLD]
        if not detections_paths:
            raise FileNotFoundError(f"no GEFFs under {pred_dir}")
        print(f"frozen predictions: {pred_dir}; crops={len(detections_paths)}", flush=True)
        geff_out = WORK / f"trackastra_geffs_split_{fold}"
        edge_out = WORK / f"trackastra_edges_split_{fold}"
        geff_out.mkdir(exist_ok=True)
        edge_out.mkdir(exist_ok=True)

        for crop_i, detections_path in enumerate(detections_paths, start=1):
            crop_t0 = time.time()
            name = detections_path.stem
            image_path = find_image(name)
            image = zarr.open_group(str(image_path), mode="r")["0"]
            detections = load_frozen_detections(detections_path)
            mask_path = temp_root / f"{name}_instances.npy"
            mapping_path = temp_root / f"{name}_mapping.csv"
            masks, mapping = rasterize_ellipsoid_masks(
                detections, tuple(int(v) for v in image.shape), output_npy=mask_path,
            )
            write_mask_mapping(mapping, mapping_path)
            print(
                f"fold={fold} [{crop_i}/{len(detections_paths)}] {name}: "
                f"nodes={len(detections.node_ids)} shape={tuple(image.shape)} "
                f"maskGiB={mask_path.stat().st_size/2**30:.3f}",
                flush=True,
            )

            predictions = model._predict(
                np.asarray(image), masks, edge_threshold=EDGE_THRESHOLD,
                batch_size=BATCH_SIZE,
            )
            selected = model._track_from_predictions(predictions, mode=MODE)
            export_trackastra_result(
                detections, mapping, predictions, selected,
                output_geff=geff_out / f"{name}.geff",
                output_edge_scores=edge_out / f"{name}.csv",
            )
            del masks, image, predictions, selected
            for path in (mask_path, mapping_path):
                if path.exists():
                    path.unlink()
            print(f"{name} complete in {(time.time()-crop_t0)/60:.1f} min", flush=True)

    if temp_root.exists():
        shutil.rmtree(temp_root)
    print(f"TRACKASTRA SCREEN COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
