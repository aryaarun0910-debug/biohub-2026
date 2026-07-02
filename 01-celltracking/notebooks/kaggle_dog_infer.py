"""Kaggle inference: multiscale DoG detector + V3 velocity-aware linking (the 0.842 anchor).

Reproduces the public "Rule-Based Baseline V3" (LB ~0.842), whose edge over the 0.826
ISAKA baseline is TEMPORAL LOGIC + refinement, not different DoG params (verified from
source, intel audit 2026-07-02 Section 2.2). Self-contained: numpy/scipy/skimage/zarr/pandas.

0.826 base config (ISAKA, verified): xy_downsample=4, dog_scales=[[1.5,4.0],[2.2,5.5]] um,
rel_threshold=0.045, reject below frame median, min_distance=3.2um, max_peaks=40000,
link 8um, gap-close 1 frame @6um, divisions off, per-frame 1-99.7 pct normalize.

V3 changes on top (verified diff):
  - NMS radius 3.2 -> 4.0 um
  - XY coord correction: +1.5 raw voxels (=(downsample-1)/2) when mapping to raw
  - refinement window (3,9,9) = radius (1,4,4) in (z,y,x)
  - two-pass velocity-aware Hungarian: pass1 6um gate, cost uses 0.5x inherited velocity;
    pass2 remaining nodes 8um gate
  - track filter: remove connected components shorter than 4 nodes (not just isolated)
  - divisions OFF

Kaggle: CPU is fine; Run All -> /kaggle/working/submission.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

# ----------------------------------------------------------------------------- config
SCALE = (1.625, 0.40625, 0.40625)
XY_DOWNSAMPLE = 4
ISO_UM = 1.625
XY_OFFSET = (XY_DOWNSAMPLE - 1) / 2.0        # 1.5 raw voxels: pooling-block center correction
NORM_Q = (0.01, 0.997)                        # per-frame 1-99.7 percentile
DOG_SCALE_PAIRS_UM = [(1.5, 4.0), (2.2, 5.5)]
DOG_REL_THRESHOLD = 0.045
NMS_DIST_UM = 4.0                             # V3: 3.2 -> 4.0
REFINE_RADIUS = (1, 4, 4)                     # V3: window (3,9,9)
REJECT_BELOW_MEDIAN = True
MAX_PEAKS = 40000
LINK_PASS1_UM = 6.0                           # V3 first pass gate (raw distance)
LINK_PASS2_UM = 8.0                           # V3 second pass gate
VELOCITY_FACTOR = 0.5                         # inherited-velocity prediction weight
GAP_CLOSE = True
GAP_MAX_UM = 6.0                              # V3 gap gate
MIN_TRACK_LEN = 4                             # V3: drop components < 4 nodes

IN_DIR = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/test")
OUT_CSV = Path("/kaggle/working/submission.csv")
SUBMISSION_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
]


# ----------------------------------------------------------------------------- detection
def downsample_xy(vol: np.ndarray, f: int = XY_DOWNSAMPLE) -> np.ndarray:
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))


def normalize_frame(vol: np.ndarray) -> np.ndarray:
    lo, hi = (float(q) for q in np.quantile(vol, NORM_Q))
    return np.clip((vol - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def multiscale_dog(vol: np.ndarray) -> np.ndarray:
    resp = np.zeros_like(vol, dtype=np.float32)
    for s_small, s_large in DOG_SCALE_PAIRS_UM:
        s1, s2 = s_small / ISO_UM, s_large / ISO_UM
        resp = np.maximum(resp, gaussian_filter(vol, s1) - gaussian_filter(vol, s2))
    return resp


def refine_com(vol: np.ndarray, coords: np.ndarray, radius=REFINE_RADIUS) -> np.ndarray:
    Z, Y, X = vol.shape
    rz, ry, rx = radius
    out = coords.astype(np.float32).copy()
    for i, (z, y, x) in enumerate(coords):
        z0, z1 = max(0, z - rz), min(Z, z + rz + 1)
        y0, y1 = max(0, y - ry), min(Y, y + ry + 1)
        x0, x1 = max(0, x - rx), min(X, x + rx + 1)
        w = vol[z0:z1, y0:y1, x0:x1]
        s = w.sum()
        if s <= 0:
            continue
        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
        out[i] = [(zz * w).sum() / s, (yy * w).sum() / s, (xx * w).sum() / s]
    return out


def detect_frame(raw_iso: np.ndarray) -> np.ndarray:
    """Detect centroids in one isotropic (downsampled) frame. Returns (N,3) (z,y,x) iso coords."""
    norm = normalize_frame(raw_iso)
    resp = multiscale_dog(norm)
    thr = DOG_REL_THRESHOLD * float(resp.max()) if resp.max() > 0 else DOG_REL_THRESHOLD
    min_vox = max(1, int(round(NMS_DIST_UM / ISO_UM)))
    peaks = peak_local_max(resp, min_distance=min_vox, threshold_abs=thr,
                           exclude_border=False, num_peaks=MAX_PEAKS)
    if len(peaks) == 0:
        return peaks.astype(np.float32)
    if REJECT_BELOW_MEDIAN:
        peaks = peaks[raw_iso[tuple(peaks.T)] >= float(np.median(raw_iso))]
        if len(peaks) == 0:
            return peaks.astype(np.float32)
    order = np.argsort(resp[tuple(peaks.T)])[::-1]
    peaks = peaks[order]
    keep = np.ones(len(peaks), bool)
    tree = cKDTree(peaks.astype(np.float32))
    min_sep = NMS_DIST_UM / ISO_UM
    for i in range(len(peaks)):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(peaks[i], min_sep):
            if j > i:
                keep[j] = False
    return refine_com(raw_iso, peaks[keep])


def to_raw_coords(cents_iso: np.ndarray) -> np.ndarray:
    """Iso-frame (z,y,x) -> RAW voxel coords for submission (undo XY downsample + center offset)."""
    if len(cents_iso) == 0:
        return cents_iso
    out = cents_iso.copy()
    out[:, 1] = out[:, 1] * XY_DOWNSAMPLE + XY_OFFSET
    out[:, 2] = out[:, 2] * XY_DOWNSAMPLE + XY_OFFSET
    return out


# ----------------------------------------------------------------------------- V3 linking
def link_frames(cents_by_t: list[np.ndarray]) -> list[tuple[int, int, int, int]]:
    """Two-pass velocity-aware Hungarian. Returns edges (t, src_idx, t+1, tgt_idx)."""
    edges = []
    # velocity[(t,i)] = physical (um) displacement into node i from its parent link
    velocity: dict[tuple[int, int], np.ndarray] = {}
    sc = np.array(SCALE)

    for t in range(len(cents_by_t) - 1):
        a, b = cents_by_t[t], cents_by_t[t + 1]
        if len(a) == 0 or len(b) == 0:
            continue
        pa, pb = a * sc, b * sc
        assigned_a, assigned_b = set(), set()

        # --- pass 1: velocity-aware, 6 um gate on RAW parent-candidate distance ---
        pred = pa.copy()
        for i in range(len(a)):
            v = velocity.get((t, i))
            if v is not None:
                pred[i] = pa[i] + VELOCITY_FACTOR * v
        raw_d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)   # gate distance
        cost = np.linalg.norm(pred[:, None, :] - pb[None, :, :], axis=2)  # assignment cost
        cost = cost.copy()
        cost[raw_d > LINK_PASS1_UM] = 1e6
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if raw_d[r, c] <= LINK_PASS1_UM:
                edges.append((t, int(r), t + 1, int(c)))
                velocity[(t + 1, int(c))] = pb[c] - pa[r]
                assigned_a.add(int(r)); assigned_b.add(int(c))

        # --- pass 2: remaining nodes, plain Hungarian, 8 um gate ---
        ra = [i for i in range(len(a)) if i not in assigned_a]
        rb = [j for j in range(len(b)) if j not in assigned_b]
        if ra and rb:
            d2 = np.linalg.norm(pa[ra][:, None, :] - pb[rb][None, :, :], axis=2)
            d2 = d2.copy(); d2[d2 > LINK_PASS2_UM] = 1e6
            ri2, ci2 = linear_sum_assignment(d2)
            for r, c in zip(ri2, ci2):
                if d2[r, c] <= LINK_PASS2_UM:
                    edges.append((t, ra[r], t + 1, rb[c]))
                    velocity[(t + 1, rb[c])] = pb[rb[c]] - pa[ra[r]]

    if GAP_CLOSE:
        linked_src = {(t, i) for t, i, _, _ in edges}
        linked_tgt = {(t2, j) for _, _, t2, j in edges}
        for t in range(len(cents_by_t) - 2):
            a, b = cents_by_t[t], cents_by_t[t + 2]
            ai = [i for i in range(len(a)) if (t, i) not in linked_src]
            bj = [j for j in range(len(b)) if (t + 2, j) not in linked_tgt]
            if not ai or not bj:
                continue
            d = np.linalg.norm((a[ai] * sc)[:, None, :] - (b[bj] * sc)[None, :, :], axis=2)
            d = d.copy(); d[d > GAP_MAX_UM] = 1e6
            ri, ci = linear_sum_assignment(d)
            for r, c in zip(ri, ci):
                if d[r, c] <= GAP_MAX_UM:
                    edges.append((t, ai[r], t + 2, bj[c]))
    return edges


# ----------------------------------------------------------------------------- submission
def _components(nodes: set, edges: list) -> list[set]:
    """Undirected connected components over linked nodes."""
    adj: dict = {n: set() for n in nodes}
    for t, i, t2, j in edges:
        adj[(t, i)].add((t2, j)); adj[(t2, j)].add((t, i))
    seen, comps = set(), []
    for n in nodes:
        if n in seen:
            continue
        stack, comp = [n], set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.add(u)
            stack.extend(adj[u] - seen)
        comps.append(comp)
    return comps


def build_rows(dataset: str, cents_by_t, edges) -> list[dict]:
    linked = {(t, i) for t, i, _, _ in edges} | {(t2, j) for _, _, t2, j in edges}
    # V3: keep only nodes in connected components with >= MIN_TRACK_LEN nodes
    keep = set()
    for comp in _components(linked, edges):
        if len(comp) >= MIN_TRACK_LEN:
            keep |= comp
    nid, rows, counter = {}, [], 1
    for t, cents in enumerate(cents_by_t):
        for i, (z, y, x) in enumerate(cents):
            if (t, i) not in keep:
                continue
            nid[(t, i)] = counter
            rows.append({"dataset": dataset, "row_type": "node", "node_id": counter,
                         "t": int(t), "z": float(z), "y": float(y), "x": float(x),
                         "source_id": -1, "target_id": -1})
            counter += 1
    for t, i, t2, j in edges:
        if (t, i) in nid and (t2, j) in nid:
            rows.append({"dataset": dataset, "row_type": "edge", "node_id": -1,
                         "t": -1, "z": -1, "y": -1, "x": -1,
                         "source_id": nid[(t, i)], "target_id": nid[(t2, j)]})
    return rows


def infer_dataset(zarr_path: Path) -> list[dict]:
    arr = zarr.open_group(str(zarr_path), mode="r")["0"]
    cents_by_t = [to_raw_coords(detect_frame(downsample_xy(np.asarray(arr[t]))))
                  for t in range(arr.shape[0])]
    return build_rows(zarr_path.stem, cents_by_t, link_frames(cents_by_t))


def main():
    all_rows = []
    datasets = sorted(IN_DIR.glob("*.zarr"))
    print(f"{len(datasets)} test datasets")
    for zp in datasets:
        rows = infer_dataset(zp)
        n = sum(r["row_type"] == "node" for r in rows)
        e = sum(r["row_type"] == "edge" for r in rows)
        print(f"  {zp.stem}: {n} nodes, {e} edges")
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    df[SUBMISSION_COLUMNS].to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
