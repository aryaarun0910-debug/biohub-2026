"""GPU cache of low-threshold DAXI candidates for a fixed 20-crop stratified gate."""
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

# Kaggle's July 2026 GPU image omits zarr. Install the pinned offline support
# wheels already used by the submission kernel; internet remains disabled.
wheel_patterns = ("donfig-*.whl", "google_crc32c-*.whl", "numcodecs-*.whl", "zarr-*.whl")
wheels = []
for pattern in wheel_patterns:
    matches = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not matches:
        raise FileNotFoundError(f"offline wheel missing: {pattern}")
    wheels.append(matches[0])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *wheels])
import zarr  # noqa: E402

START = time.time()
OUT = Path("/kaggle/working/daxi_cache20")
OUT.mkdir(parents=True, exist_ok=True)
LOW_THRESH = 0.15
TOPK = 6000
NMS = 3
PATCH_YX = 256
OVERLAP = 32
CROPS = [
    "44b6_d754aa59", "44b6_0db75fae", "44b6_d78e09d9", "44b6_90724892",
    "44b6_87bba6c4", "44b6_5f15d135", "44b6_d5e7d891", "44b6_415c0a3a",
    "44b6_71a4179f", "44b6_18ced818",
    "6bba_55b7eebe", "6bba_b204cac7", "6bba_4f99ce20", "6bba_705ec2c9",
    "6bba_6feeb0b1", "6bba_0c7fa718", "6bba_9a41d029", "6bba_cdcfe533",
    "6bba_07e24132", "6bba_57b7cc1e",
]


def find_one(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[0])


@torch.inference_mode()
def foreground(model, frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    lo, hi = np.percentile(frame, (1.0, 99.9))
    frame = np.clip((frame - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    zdim, ydim, xdim = frame.shape
    result = np.zeros_like(frame, dtype=np.float32)
    step = PATCH_YX - OVERLAP
    ys = list(range(0, max(1, ydim - OVERLAP), step)) or [0]
    xs = list(range(0, max(1, xdim - OVERLAP), step)) or [0]
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + PATCH_YX, ydim), min(x0 + PATCH_YX, xdim)
            y0, x0 = max(0, y1 - PATCH_YX), max(0, x1 - PATCH_YX)
            patch = np.ascontiguousarray(frame[:, y0:y1, x0:x1])
            output = model(torch.from_numpy(patch)[None, None].cuda())[0, 0].float().cpu().numpy()
            result[:, y0:y1, x0:x1] = np.maximum(result[:, y0:y1, x0:x1], output)
    return result


weight = find_one("/kaggle/input/**/unet-daxi.pt")
model = torch.jit.load(str(weight), map_location="cuda").eval()
train_root = find_one("/kaggle/input/**/train")
manifest = {"weight": str(weight), "train_root": str(train_root), "crops": CROPS,
            "threshold": LOW_THRESH, "topk": TOPK, "device": torch.cuda.get_device_name(0)}
print(json.dumps(manifest, indent=2), flush=True)

for crop in CROPS:
    target = OUT / f"{crop}.npz"
    zarr_path = train_root / f"{crop}.zarr" / "0"
    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)
    arr = zarr.open(zarr_path, mode="r")
    columns = {key: [] for key in ("t", "z", "y", "x", "resp")}
    for t in range(arr.shape[0]):
        response = foreground(model, np.asarray(arr[t]))
        peaks = np.argwhere((response == maximum_filter(response, size=NMS)) & (response > LOW_THRESH))
        if len(peaks):
            values = response[tuple(peaks.T)]
            if len(peaks) > TOPK:
                keep = np.argpartition(-values, TOPK - 1)[:TOPK]
                peaks, values = peaks[keep], values[keep]
            columns["t"].append(np.full(len(peaks), t, np.int16))
            columns["z"].append(peaks[:, 0].astype(np.int16))
            columns["y"].append(peaks[:, 1].astype(np.int16))
            columns["x"].append(peaks[:, 2].astype(np.int16))
            columns["resp"].append(values.astype(np.float16))
        if t % 10 == 0:
            print(f"{crop} t={t}/{arr.shape[0]}", flush=True)
    packed = {key: np.concatenate(values) if values else np.array([], dtype=np.float16)
              for key, values in columns.items()}
    np.savez_compressed(target, **packed)
    print(f"DONE {crop}: {len(packed['t'])} peaks elapsed={(time.time()-START)/60:.1f}m", flush=True)

manifest["elapsed_seconds"] = time.time() - START
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
