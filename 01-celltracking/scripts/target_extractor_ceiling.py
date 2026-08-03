"""M2 pre-gate — recall ceiling of a (target, peak-extractor) pair, on PERFECT heatmaps.

Rasterise the GROUND-TRUTH centres into a heatmap, then run the deployed peak rule on it:

    pooled = max_pool3d(logits, pool_kernel, stride=1, padding=k//2)
    is_peak = (logits == pooled) & (sigmoid(logits) > threshold)

If recall < 1.0 on a perfect heatmap, the target/extractor pair is broken BEFORE any training
starts, and no amount of detector training can fix it. This costs no GPU and bounds M2's design
space: it says which sigma the target may use before Gaussian merging destroys recall.

Grid: the detector downsamples [1,4,4] against voxel scale (1.625, 0.40625, 0.40625), so the grid
is ISOTROPIC at 1.625 um/step. Sigma and pool windows are therefore quoted in grid units, and one
grid unit is 1.625 um on every axis.

Overlapping kernels are combined by MAXIMUM, not sum: summing invents a super-peak exactly
between two touching centres, which is the failure mode being tested for.
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys
from collections import defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
SCALE = (1.625, 0.40625, 0.40625)
DS = (1, 4, 4)
GRID_UM = 1.625          # isotropic after downsample


def rasterise(centres, shape, sigma):
    """Max-combined Gaussian heatmap. sigma in grid units; 0 => single-voxel targets."""
    vol = np.zeros(shape, dtype=np.float32)
    if sigma <= 0:
        for z, y, x in centres:
            vol[z, y, x] = 1.0
        return vol
    r = int(np.ceil(3 * sigma))
    dz, dy, dx = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1),
                             np.arange(-r, r + 1), indexing="ij")
    kern = np.exp(-(dz ** 2 + dy ** 2 + dx ** 2) / (2 * sigma ** 2)).astype(np.float32)
    Z, Y, X = shape
    for z, y, x in centres:
        z0, z1 = max(0, z - r), min(Z, z + r + 1)
        y0, y1 = max(0, y - r), min(Y, y + r + 1)
        x0, x1 = max(0, x - r), min(X, x + r + 1)
        sub = kern[z0 - (z - r):r + 1 + (z1 - z - 1),
                   y0 - (y - r):r + 1 + (y1 - y - 1),
                   x0 - (x - r):r + 1 + (x1 - x - 1)]
        np.maximum(vol[z0:z1, y0:y1, x0:x1], sub, out=vol[z0:z1, y0:y1, x0:x1])
    return vol


def peaks(vol, pool, thresh_on_value):
    """Deployed rule. `vol` is already a probability-like value in [0,1]."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(vol)[None, None]
    pad = tuple(k // 2 for k in pool)
    pooled = F.max_pool3d(t, pool, stride=1, padding=pad)
    is_peak = (t == pooled) & (t > thresh_on_value)
    return set(map(tuple, torch.nonzero(is_peak[0, 0]).numpy().tolist()))


def main():
    sys.path.insert(0, str(REPO / "src"))
    import tracksdata as td
    from biotrack.metric import load_graph

    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=int, default=12)
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()

    geffs = sorted(glob.glob(str(REPO / "data/train/*.geff")))
    sel = geffs[:: max(1, len(geffs) // a.crops)][: a.crops]

    # ---- 1. GT nearest-neighbour distance, physical um ------------------------------
    nn_all = defaultdict(list)
    frames_data = []
    for gp in sel:
        name = pathlib.Path(gp).stem
        fam = name.split("_")[0]
        g = load_graph(gp)
        na = g.node_attrs(attr_keys=["t", "z", "y", "x"])
        by_t = defaultdict(list)
        for t, z, y, x in zip(na["t"].to_list(), na["z"].to_list(),
                              na["y"].to_list(), na["x"].to_list()):
            by_t[int(t)].append((float(z), float(y), float(x)))
        picked = sorted(by_t, key=lambda k: -len(by_t[k]))[: a.frames]
        for t in picked:
            pts = np.array(by_t[t])
            if len(pts) < 2:
                continue
            phys = pts * np.array(SCALE)
            d = np.linalg.norm(phys[:, None, :] - phys[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            nn_all[fam].extend(d.min(axis=1).tolist())
            grid = np.stack([pts[:, 0] / DS[0], pts[:, 1] / DS[1], pts[:, 2] / DS[2]],
                            axis=1).round().astype(int)
            frames_data.append((fam, grid))

    print("=" * 78)
    print(f"GT NEAREST-NEIGHBOUR DISTANCE (physical um) — {len(frames_data)} dense frames")
    print("=" * 78)
    for fam in sorted(nn_all):
        v = np.array(nn_all[fam])
        print(f"  {fam}: n={len(v)}  p1={np.percentile(v,1):.2f}  p5={np.percentile(v,5):.2f}  "
              f"p50={np.percentile(v,50):.2f}  p95={np.percentile(v,95):.2f}  min={v.min():.2f}")
    allv = np.concatenate([np.array(x) for x in nn_all.values()])
    print(f"  ALL: p50 {np.median(allv):.2f} um = {np.median(allv)/GRID_UM:.2f} grid units")

    # ---- 2. recall ceiling on PERFECT heatmaps --------------------------------------
    print("\n" + "=" * 78)
    print("RECALL CEILING ON A PERFECT RASTERISED-GT HEATMAP (deployed peak rule)")
    print("=" * 78)
    print("  sigma=0 is the CURRENT single-voxel target. Grid step = 1.625 um.")
    print(f"\n  {'sigma':>7s} {'pool':>10s} {'recall':>9s} {'peaks/GT':>10s}  verdict")
    for sigma in (0.0, 0.5, 0.75, 1.0, 1.5, 2.0):
        for pool in ((3, 3, 3), (3, 9, 9), (5, 5, 5)):
            rec_n = rec_d = 0
            pk_n = 0
            for fam, grid in frames_data:
                Z = int(grid[:, 0].max()) + 8
                Y = int(grid[:, 1].max()) + 8
                X = int(grid[:, 2].max()) + 8
                if Z * Y * X > 40_000_000:
                    continue
                cen = [tuple(c) for c in grid]
                vol = rasterise(cen, (Z, Y, X), sigma)
                pk = peaks(vol, pool, 0.5)
                pk_n += len(pk)
                hit = sum(1 for c in cen if c in pk)
                rec_n += hit
                rec_d += len(cen)
            if rec_d:
                r = rec_n / rec_d
                verdict = "OK" if r > 0.999 else ("LOSES RECALL" if r < 0.99 else "marginal")
                print(f"  {sigma:7.2f} {str(pool):>10s} {r:9.4f} {pk_n/max(rec_d,1):10.3f}  {verdict}")

    print("\n  READING")
    print("    Any row below recall 1.0 is a ceiling the detector can never exceed,")
    print("    no matter how well it is trained. Pick sigma/pool from the OK rows only.")


if __name__ == "__main__":
    main()
