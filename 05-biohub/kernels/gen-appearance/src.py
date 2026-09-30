# Export DeepCenter appearance scores for every predicted node, on the TRAIN split.
#
# EXP-31's swap discriminator reaches AUC 0.80 on geometry alone and clears the 48.1% edge
# break-even at small K, but precision decays by top-1000. Appearance is the most likely way to
# hold it: EXP-26 found the DeepCenter score was the single strongest feature (0.657) where every
# geometric term sat at 0.52-0.58.
#
# This runs on Kaggle because the 82GB of image volumes are already mounted here; shipping them
# to Colab would cost more than the compute saves. Output is one small .npz per dataset holding
# (node_id, score) so the local swap model can join on it.
import json, os, subprocess, sys, shutil, time
from pathlib import Path
import numpy as np

OUT = Path("/kaggle/working")

def find(pred, depth=6):
    root = Path("/kaggle/input")
    for d in range(1, depth):
        for c in root.glob("/".join(["*"] * d)):
            if pred(c):
                return c
    return None

PACK = find(lambda c: c.is_dir() and (c / "weights/full_frame_center").is_dir())
TRAIN = find(lambda c: c.is_dir() and c.name == "train" and any(c.glob("*.zarr")))
print("PACK:", PACK, "\nTRAIN:", TRAIN, flush=True)
assert PACK and TRAIN

sys.path.insert(0, str(PACK / "source_scripts"))
import torch, zarr
from train_full_frame_center_detector import DeepCenterUNet3D

ck = torch.load(PACK / "weights/full_frame_center/best.pt", map_location="cpu", weights_only=False)
cfg = ck["config"] if isinstance(ck["config"], dict) else vars(ck["config"])
model = DeepCenterUNet3D(base_channels=int(cfg["base_channels"]))
model.load_state_dict(ck["model_state"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(dev)
pf = int(cfg.get("pool_factor", 4))
print(f"DeepCenter on {dev}, base_channels {cfg['base_channels']}, pool {pf}", flush=True)

def prep(v):
    z, y, x = v.shape
    p = v.reshape(z, y // pf, pf, x // pf, pf).max(axis=(2, 4))
    lo = np.percentile(p, cfg.get("norm_lo_pct", 50.0)); hi = np.percentile(p, cfg.get("norm_hi_pct", 99.5))
    p = (p - lo) / max(hi - lo, 1e-6)
    return np.clip(p, cfg.get("norm_clip_lo", -0.5), cfg.get("norm_clip_hi", 6.0)).astype(np.float32)

dst = OUT / "appearance"; dst.mkdir(exist_ok=True)
stems = sorted(p.stem for p in TRAIN.glob("*.zarr"))
t0 = time.time()
for i, stem in enumerate(stems):
    z = zarr.open(str(TRAIN / f"{stem}.zarr"), mode="r"); arr = z["0"] if "0" in z else z
    T = arr.shape[0]
    maps = np.zeros((T, arr.shape[1] // 1, arr.shape[2] // pf, arr.shape[3] // pf), np.float16)
    with torch.no_grad():
        for t in range(T):
            v = torch.from_numpy(prep(np.asarray(arr[t], np.float32))[None, None]).to(dev)
            maps[t] = torch.sigmoid(model(v)[0, 0]).cpu().numpy().astype(np.float16)
    np.savez_compressed(dst / f"{stem}.npz", heat=maps, pool_factor=pf)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(stems)}  {time.time()-t0:.0f}s", flush=True)
print(f"done {len(stems)} datasets in {time.time()-t0:.0f}s", flush=True)
shutil.make_archive(str(OUT / "appearance_maps"), "zip", root_dir=dst)
shutil.rmtree(dst, ignore_errors=True)
print("archived:", (OUT / "appearance_maps.zip").stat().st_size / 1e6, "MB", flush=True)
