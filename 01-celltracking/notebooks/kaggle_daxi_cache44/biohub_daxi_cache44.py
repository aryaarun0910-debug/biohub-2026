"""GPU cache of low-threshold DAXI peaks for every 44b6 OOF crop."""
from __future__ import annotations

import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter

START = time.time()
OUT = Path("/kaggle/working/daxi_cache44")
OUT.mkdir(parents=True, exist_ok=True)
LOW_THRESH = 0.15
TOPK = 6000
NMS = 3
PATCH_YX = 256
OVERLAP = 32
RUNTIME_GUARD_SECONDS = 8.25 * 60 * 60

# Kaggle's GPU image omits zarr. Use the submission kernel's pinned offline wheels.
wheels = []
for pattern in ("donfig-*.whl", "google_crc32c-*.whl", "numcodecs-*.whl", "zarr-*.whl"):
    matches = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not matches:
        raise FileNotFoundError(f"offline wheel missing: {pattern}")
    wheels.append(matches[0])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *wheels])
import zarr  # noqa: E402


def find_one(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[0])


@torch.inference_mode()
def foreground(model, frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    lo, hi = np.percentile(frame, (1.0, 99.9))
    frame = np.clip(
        (frame - np.float32(lo)) / np.float32(hi - lo + 1e-6), 0.0, 1.0
    ).astype(np.float32, copy=False)
    zdim, ydim, xdim = frame.shape
    result = np.zeros((zdim, ydim, xdim), dtype=np.float32)
    step = PATCH_YX - OVERLAP
    ys = list(range(0, max(1, ydim - OVERLAP), step)) or [0]
    xs = list(range(0, max(1, xdim - OVERLAP), step)) or [0]
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + PATCH_YX, ydim), min(x0 + PATCH_YX, xdim)
            y0, x0 = max(0, y1 - PATCH_YX), max(0, x1 - PATCH_YX)
            patch = np.ascontiguousarray(frame[:, y0:y1, x0:x1], dtype=np.float32)
            tensor = torch.from_numpy(patch)[None, None].cuda()
            if tensor.dtype != torch.float32:
                raise TypeError(f"DAXI input must be float32, got {tensor.dtype}")
            output = model(tensor)[0, 0].float().cpu().numpy()
            result[:, y0:y1, x0:x1] = np.maximum(result[:, y0:y1, x0:x1], output)
    return result


weight = find_one("/kaggle/input/**/unet-daxi.pt")
model = torch.jit.load(str(weight), map_location="cuda").eval()
train_root = find_one("/kaggle/input/**/train")
crops = sorted(path.name[:-5] for path in train_root.glob("44b6_*.zarr"))
if len(crops) != 71:
    raise RuntimeError(f"expected 71 44b6 crops, found {len(crops)}")
manifest = {
    "weight": str(weight), "train_root": str(train_root), "crops": crops,
    "threshold": LOW_THRESH, "topk": TOPK, "device": torch.cuda.get_device_name(0),
}
print(json.dumps(manifest, indent=2), flush=True)

for crop_index, crop in enumerate(crops):
    if time.time() - START > RUNTIME_GUARD_SECONDS:
        raise TimeoutError(f"runtime guard reached before crop {crop_index}: {crop}")
    arr = zarr.open(train_root / f"{crop}.zarr" / "0", mode="r")
    columns = {key: [] for key in ("t", "z", "y", "x", "resp")}
    for t in range(arr.shape[0]):
        response = foreground(model, np.asarray(arr[t]))
        peaks = np.argwhere((response == maximum_filter(response, size=NMS)) & (response > LOW_THRESH))
        if len(peaks):
            values = response[tuple(peaks.T)]
            if len(peaks) > TOPK:
                keep = np.argpartition(-values, TOPK - 1)[:TOPK]
                peaks, values = peaks[keep], values[keep]
            columns["t"].append(np.full(len(peaks), t, dtype=np.int16))
            columns["z"].append(peaks[:, 0].astype(np.int16))
            columns["y"].append(peaks[:, 1].astype(np.int16))
            columns["x"].append(peaks[:, 2].astype(np.int16))
            columns["resp"].append(values.astype(np.float16))
        if t % 25 == 0:
            print(f"{crop} t={t}/{arr.shape[0]}", flush=True)
    packed = {
        key: np.concatenate(values) if values else np.array([], dtype=np.float16)
        for key, values in columns.items()
    }
    np.savez_compressed(OUT / f"{crop}.npz", **packed)
    print(f"DONE {crop_index+1}/{len(crops)} {crop}: {len(packed['t'])} peaks "
          f"elapsed={(time.time()-START)/60:.1f}m", flush=True)

manifest["elapsed_seconds"] = time.time() - START
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
