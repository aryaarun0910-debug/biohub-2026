"""Cache DAXI low-threshold candidate peaks per crop (input to the isolated Stages 2-4 gate).

For each crop, runs the DAXI U-Net on every frame, keeps top-K local maxima above a low
threshold (voxel coords + response), and caches them. NO GT coordinates used. Bounded by
top-K/frame so tracklet construction stays tractable. Resumable (skips cached crops).

Output: artifacts/kaggle/daxi_cand/{crop}.npz  (t, z, y, x, resp)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from scipy.ndimage import maximum_filter  # noqa: E402

OUT = ROOT / "artifacts/kaggle/daxi_cand"
LOW_THRESH = 0.15
TOPK = 6000
NMS = 3
# representative crops (both folds); extend freely
CROPS = [
    "44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc",
    "6bba_05db0fb1", "6bba_05b6850b", "6bba_05db0fb1",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    net = torch.jit.load(str(ROOT / "weights" / "unet-daxi.pt"), map_location="cpu"); net.eval()
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--crops", nargs="+", default=None); a = ap.parse_args()
    crops = a.crops or sorted(set(CROPS))
    for crop in crops:
        outf = OUT / f"{crop}.npz"
        if outf.exists():
            print(f"skip {crop} (cached)", flush=True); continue
        zp = ROOT / "data" / "train" / f"{crop}.zarr"
        if not zp.exists():
            print(f"skip {crop} (no zarr)", flush=True); continue
        arr = zarr.open(zp / "0", mode="r")
        T = arr.shape[0]
        ts, zs, ys, xs, rs = [], [], [], [], []
        for t in range(T):
            frame = np.asarray(arr[t]).astype(np.float32)
            lo, hi = np.percentile(frame, 1), np.percentile(frame, 99.9)
            fn = np.clip((frame - lo) / (hi - lo + 1e-6), 0, 1)
            with torch.no_grad():
                fg = net(torch.tensor(fn[None, None]))[0, 0].numpy()
            mx = maximum_filter(fg, size=NMS)
            pk = np.argwhere((fg == mx) & (fg > LOW_THRESH))
            if len(pk) == 0:
                continue
            resp = fg[pk[:, 0], pk[:, 1], pk[:, 2]]
            if len(pk) > TOPK:
                keep = np.argpartition(-resp, TOPK)[:TOPK]; pk = pk[keep]; resp = resp[keep]
            ts.append(np.full(len(pk), t, np.int16)); zs.append(pk[:, 0].astype(np.int16))
            ys.append(pk[:, 1].astype(np.int16)); xs.append(pk[:, 2].astype(np.int16)); rs.append(resp.astype(np.float16))
        np.savez_compressed(outf, t=np.concatenate(ts), z=np.concatenate(zs), y=np.concatenate(ys),
                            x=np.concatenate(xs), resp=np.concatenate(rs))
        print(f"{crop}: cached {sum(len(a) for a in ts)} peaks over {T} frames", flush=True)


if __name__ == "__main__":
    main()
