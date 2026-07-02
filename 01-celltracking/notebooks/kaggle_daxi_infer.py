"""Kaggle inference: DAXI U-Net detector + motion-aware Hungarian linking.

First end-to-end submission. Self-contained: only torch, zarr, numpy, scipy,
skimage, pandas (all present in the Kaggle image) -- NO internet / offline install.

Pipeline per test dataset (<id>.zarr, shape T,Z,Y,X uint16):
  1. per-volume percentile normalize to [0,1]
  2. DAXI U-Net (patch, full-Z) -> foreground prob channel  (verified: in=(N,1,Z,Y,X),
     out=(N,2,Z,Y,X), ch0=foreground in [0,1], ch1=contour)
  3. peak_local_max on foreground with physical-distance NMS -> centroids per t
  4. motion-aware Hungarian linking t->t+1 gated by max physical distance
  5. write nodes+edges to submission.csv (schema verified locally against exact metric)

Divisions OFF for v1 (0.1 weight; false forks hurt edges -- add later, precision-first).

Kaggle setup:
  - Add DAXI weights as a dataset -> /kaggle/input/daxi-weights/unet-daxi.pt
  - Accelerator: GPU (T4). Run All. Output: /kaggle/working/submission.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

# ----------------------------------------------------------------------------- config
SCALE = (1.625, 0.40625, 0.40625)     # (z,y,x) micron/voxel — fixed competition scale
FG_THRESHOLD = 0.5                     # foreground prob to accept a peak
NMS_DIST_UM = 4.0                      # min physical distance between detections (micron)
LINK_MAX_UM = 10.0                     # max physical displacement for a t->t+1 link
PATCH_YX = 256                         # XY patch size for sliding-window inference
PATCH_OVERLAP = 32                     # XY overlap (blended by max)
NORM_Q = (0.001, 0.9999)              # per-volume percentile clip (DAXI recipe)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IN_DIR = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/test")
WEIGHTS = Path("/kaggle/input/daxi-weights/unet-daxi.pt")
OUT_CSV = Path("/kaggle/working/submission.csv")

SUBMISSION_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
]


# ----------------------------------------------------------------------------- detection
def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Percentile-clip a (Z,Y,X) volume to [0,1] (DAXI normalization)."""
    v = vol.astype(np.float32)
    lo, hi = (float(q) for q in np.quantile(v, NORM_Q))  # floats, avoid float64 promotion
    return np.clip((v - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def daxi_foreground(model, vol: np.ndarray) -> np.ndarray:
    """Run DAXI over one normalized (Z,Y,X) volume, return foreground prob (Z,Y,X).

    Tiles XY with overlap (full Z), blends overlaps by max.
    """
    Z, Y, X = vol.shape
    fg = np.zeros((Z, Y, X), np.float32)
    step = PATCH_YX - PATCH_OVERLAP
    ys = list(range(0, max(1, Y - PATCH_OVERLAP), step)) or [0]
    xs = list(range(0, max(1, X - PATCH_OVERLAP), step)) or [0]
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + PATCH_YX, Y), min(x0 + PATCH_YX, X)
            y0, x0 = max(0, y1 - PATCH_YX), max(0, x1 - PATCH_YX)
            patch = np.ascontiguousarray(vol[:, y0:y1, x0:x1], dtype=np.float32)
            t = torch.from_numpy(patch)[None, None].to(DEV)
            out = model(t)[0, 0].float().cpu().numpy()  # channel 0 = foreground
            fg[:, y0:y1, x0:x1] = np.maximum(fg[:, y0:y1, x0:x1], out)
    return fg


def detect_centroids(fg: np.ndarray, scale=SCALE) -> np.ndarray:
    """Peak detection with anisotropy-correct physical-distance NMS. Returns (N,3) (z,y,x) voxels.

    peak_local_max applies min_distance ISOTROPICALLY in voxel space. On anisotropic data
    (Z 4x coarser) we must tie it to the COARSEST axis (max scale) so the voxel filter never
    exceeds NMS_DIST_UM in any physical axis -- otherwise it collapses axially-separated nuclei
    (e.g. two peaks 4.875 um apart in Z). The true anisotropic separation is then enforced by
    the physical-distance cKDTree NMS below.
    """
    min_vox = max(1, int(NMS_DIST_UM / max(scale)))  # coarsest axis; safe pre-filter
    peaks = peak_local_max(fg, min_distance=min_vox, threshold_abs=FG_THRESHOLD, exclude_border=False)
    if len(peaks) == 0:
        return peaks.astype(np.float32)
    # physical-distance NMS: greedily keep highest-intensity peaks >= NMS_DIST_UM apart
    order = np.argsort(fg[tuple(peaks.T)])[::-1]
    peaks = peaks[order]
    phys = peaks * np.array(scale)
    keep = np.ones(len(peaks), bool)
    tree = cKDTree(phys)
    for i in range(len(peaks)):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(phys[i], NMS_DIST_UM):
            if j > i:
                keep[j] = False
    return peaks[keep].astype(np.float32)


# ----------------------------------------------------------------------------- linking
def link_frames(cents_by_t: list[np.ndarray], scale=SCALE) -> list[tuple[int, int, int, int]]:
    """Motion-aware Hungarian linking. Returns edges as (t, src_idx, t+1, tgt_idx)."""
    edges = []
    for t in range(len(cents_by_t) - 1):
        a, b = cents_by_t[t], cents_by_t[t + 1]
        if len(a) == 0 or len(b) == 0:
            continue
        pa, pb = a * np.array(scale), b * np.array(scale)
        cost = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
        cost[cost > LINK_MAX_UM] = 1e6
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if cost[r, c] < LINK_MAX_UM:
                edges.append((t, int(r), t + 1, int(c)))
    return edges


def prune_isolated(cents_by_t, edges):
    """Drop nodes with no incident edge (isolated nodes add count, no edge benefit)."""
    linked = {(t, i) for t, i, _, _ in edges} | {(t2, j) for _, _, t2, j in edges}
    return linked


# ----------------------------------------------------------------------------- submission
def build_rows(dataset: str, cents_by_t, edges, keep_isolated=False) -> list[dict]:
    """Assemble submission rows for one dataset. node_id is 1-based per dataset."""
    linked = prune_isolated(cents_by_t, edges)
    nid = {}
    rows = []
    counter = 1
    for t, cents in enumerate(cents_by_t):
        for i, (z, y, x) in enumerate(cents):
            if not keep_isolated and (t, i) not in linked:
                continue
            nid[(t, i)] = counter
            rows.append({
                "dataset": dataset, "row_type": "node", "node_id": counter,
                "t": int(t), "z": float(z), "y": float(y), "x": float(x),
                "source_id": -1, "target_id": -1,
            })
            counter += 1
    for t, i, t2, j in edges:
        if (t, i) in nid and (t2, j) in nid:
            rows.append({
                "dataset": dataset, "row_type": "edge", "node_id": -1,
                "t": -1, "z": -1, "y": -1, "x": -1,
                "source_id": nid[(t, i)], "target_id": nid[(t2, j)],
            })
    return rows


def infer_dataset(model, zarr_path: Path) -> list[dict]:
    grp = zarr.open_group(str(zarr_path), mode="r")
    arr = grp["0"]                       # (T,Z,Y,X)
    T = arr.shape[0]
    cents_by_t = []
    for t in range(T):
        vol = normalize_volume(np.asarray(arr[t]))
        fg = daxi_foreground(model, vol)
        cents_by_t.append(detect_centroids(fg))
    edges = link_frames(cents_by_t)
    return build_rows(zarr_path.stem, cents_by_t, edges)


def main():
    model = torch.jit.load(str(WEIGHTS), map_location=DEV).eval()
    all_rows = []
    datasets = sorted(IN_DIR.glob("*.zarr"))
    print(f"{len(datasets)} test datasets on {DEV}")
    for zp in datasets:
        rows = infer_dataset(model, zp)
        n = sum(r["row_type"] == "node" for r in rows)
        e = sum(r["row_type"] == "edge" for r in rows)
        print(f"  {zp.stem}: {n} nodes, {e} edges")
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    df = df[SUBMISSION_COLUMNS]
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
