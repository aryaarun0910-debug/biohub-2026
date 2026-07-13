"""Track-before-detect FALSIFICATION diagnostic (Codex lateral-sweep edge #1).

Question: do the GT cells that DoG MISSES (no candidate within 7um) carry coherent LATENT signal
across time that single-frame thresholding discards? If integrating the raw DoG response over a few
frames separates missed-GT locations from background far better than a single frame does, then
track-before-detect can recover them -> a large, orthogonal recall lever.

Method per crop (worst-recall crops):
  - precompute the ISO DoG response volume for every frame;
  - find DoG-missed GT nodes (loose-NMS candidates vs GT, 7um);
  - at each missed node: SINGLE-frame response vs TEMPORAL sum over t-W..t+W (static window);
  - NULL: same at random background locations (>7um from any GT node).
  - report how much better temporal separates missed-GT from null vs single-frame (AUC-like:
    fraction of missed-GT above the null 95th percentile).

GO if temporal recovers a large fraction that single-frame does not. CPU, ~minutes/crop.
Usage: python scripts/tbd_diagnostic.py --n-crops 8 --window 2
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from biotrack import propose  # noqa: E402
from biotrack.propose import (ProposeConfig, dog_response, downsample_xy, normalize_frame,  # noqa: E402
                              open_volume, read_frame, propose_frame, to_raw)
from run_v3_taxonomy import geff_to_sample  # noqa: E402

TRAIN = ROOT / "data" / "train"
SCALE = np.array([1.625, 0.40625, 0.40625]); F = 4


def worst_recall_crops(n):
    rows = list(csv.DictReader((ROOT / "reports/inventory/v3_taxonomy.csv").open(encoding="utf-8")))
    rows.sort(key=lambda r: float(r["recall"]))
    return [r["crop"] for r in rows[:n]]


def _iso_idx(zyx):
    """RAW (z,y,x) -> ISO-frame integer index (Z unchanged, XY /4)."""
    return np.array([round(zyx[0]), round(zyx[1] / F), round(zyx[2] / F)], int)


def _sample(vol, idx, r=1):
    Z, Y, X = vol.shape
    z, y, x = idx
    if not (0 <= z < Z and 0 <= y < Y and 0 <= x < X):
        return 0.0
    z0, z1 = max(0, z - r), min(Z, z + r + 1)
    y0, y1 = max(0, y - r), min(Y, y + r + 1)
    x0, x1 = max(0, x - r), min(X, x + r + 1)
    return float(vol[z0:z1, y0:y1, x0:x1].max())


def diag_crop(crop, W, rng):
    arr = open_volume(TRAIN / f"{crop}.zarr")
    gt = geff_to_sample(str(TRAIN / f"{crop}.geff"))
    T = arr.shape[0]
    # response volumes + candidate misses per frame
    resp, missed = {}, {}
    for t in range(T):
        raw = read_frame(arr, t)
        iso = normalize_frame(downsample_xy(raw, F), (0.01, 0.997))
        resp[t] = dog_response(iso)
        gt_f = gt.zyx[gt.t == t]
        if len(gt_f) == 0:
            missed[t] = np.zeros((0, 3)); continue
        cand = to_raw(propose_frame(downsample_xy(raw, F), ProposeConfig(nms_dist_um=1.0)), F)[:, :3]
        if len(cand):
            from scipy.spatial import cKDTree
            d, _ = cKDTree(cand * SCALE).query(gt_f * SCALE)
            missed[t] = gt_f[d > 7.0]
        else:
            missed[t] = gt_f

    def integ(idx, t):
        single = _sample(resp[t], idx)
        temporal = sum(_sample(resp[tt], idx) for tt in range(max(0, t - W), min(T, t + W + 1)))
        return single, temporal

    pos_s, pos_te, null_s, null_te = [], [], [], []
    for t in range(T):
        for m in missed[t]:
            s, te = integ(_iso_idx(m), t)
            pos_s.append(s); pos_te.append(te)
        # null: random ISO locations far from any GT this frame
        gt_f = gt.zyx[gt.t == t]
        Z, Y, X = resp[t].shape
        for _ in range(max(5, len(missed[t]) * 3)):
            idx = np.array([rng.integers(0, Z), rng.integers(0, Y), rng.integers(0, X)])
            raw_pt = np.array([idx[0], idx[1] * F, idx[2] * F])
            if len(gt_f) and np.min(np.linalg.norm((gt_f - raw_pt) * SCALE, axis=1)) < 7.0:
                continue
            s, te = integ(idx, t)
            null_s.append(s); null_te.append(te)
    return dict(crop=crop, fam=crop.split("_")[0], n_missed=len(pos_s),
                pos_s=np.array(pos_s), pos_te=np.array(pos_te),
                null_s=np.array(null_s), null_te=np.array(null_te))


def frac_above(pos, null, q=95):
    if len(pos) == 0 or len(null) == 0:
        return float("nan")
    thr = np.percentile(null, q)
    return float(np.mean(pos > thr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-crops", type=int, default=8)
    ap.add_argument("--window", type=int, default=2)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    crops = worst_recall_crops(args.n_crops)
    print(f"worst-recall crops: {crops}\nW={args.window}\n")
    ps, pte, ns, nte = [], [], [], []
    for crop in crops:
        d = diag_crop(crop, args.window, rng)
        fs = frac_above(d["pos_s"], d["null_s"]); ft = frac_above(d["pos_te"], d["null_te"])
        print(f"{crop:18s} missed={d['n_missed']:4d}  single-frame recover={fs:.2f}  temporal recover={ft:.2f}")
        ps += list(d["pos_s"]); pte += list(d["pos_te"]); ns += list(d["null_s"]); nte += list(d["null_te"])
    print("\n=== POOLED (missed-GT above null 95th pct) ===")
    print(f"  single-frame: {frac_above(np.array(ps), np.array(ns)):.3f}")
    print(f"  temporal    : {frac_above(np.array(pte), np.array(nte)):.3f}")
    print("\nGO if temporal >> single-frame: missed cells carry coherent latent signal -> build track-before-detect.")


if __name__ == "__main__":
    main()
