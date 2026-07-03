"""Kaggle inference: over-propose -> same-cell dedup -> V3 velocity linking -> temporal smoothing.

Faithfully deploys the locally-validated op_bright_smooth pipeline (SMOOTH=True; full-199 min-fold
adjJ 0.6996, both folds above op_bright). Over-propose with LOOSE NMS (recovers subthreshold cells),
collapse same-cell duplicates via complete-linkage with a hard diameter cap (no chain-bridging), keep
the brightest representative per cell, link with the V3 two-pass velocity-aware Hungarian, then apply
V11 temporal coordinate smoothing (blend each node toward its edge-neighbours' mean, w=0.7).
Set SMOOTH=False for the pure op_bright calibration point. Self-contained: numpy/scipy/skimage +
tensorstore (Kaggle has no zarr). No internet needed.

NOTE: candidate clustering happens on the ISOTROPIC downsampled grid, so same_cell_sets scales coords
by ISO (uniform), NOT the anisotropic SCALE — the raw mapping only happens after dedup, in detect().
COM refinement is intentionally OMITTED here so node counts reproduce the local ablation exactly.

Kaggle: Accelerator none/GPU (CPU fine); Run All -> /kaggle/working/submission.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

SCALE = np.array([1.625, 0.40625, 0.40625]); ISO = 1.625; F = 4; XY_OFF = (F - 1) / 2.0
NORM_Q = (0.01, 0.997); SCALE_PAIRS = [(1.5, 4.0), (2.2, 5.5)]
OP_REL = 0.02; OP_NMS_UM = 1.0; MAX_PEAKS = 60000       # over-proposal
R_SAME_UM = 3.0                                          # same-cell dedup diameter cap
LINK1, LINK2, VEL, GAP, MINLEN = 6.0, 8.0, 0.5, 6.0, 4  # V3 linker
SMOOTH = True; SMOOTH_W = 0.7                            # V11 temporal coordinate smoothing (op_bright_smooth)
IN_DIR = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/test")
OUT_CSV = Path("/kaggle/working/submission.csv")
COLS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def open_vol(zp):
    try:
        import zarr
        return zarr.open_group(str(zp), mode="r")["0"]
    except ImportError:
        import tensorstore as ts
        return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(Path(zp) / "0")}}).result()


def read_frame(a, t):
    fr = a[t]
    return np.asarray(fr.read().result()) if hasattr(fr, "read") else np.asarray(fr)


def dsxy(v):
    Z, Y, X = v.shape
    return v[:, :(Y // F) * F, :(X // F) * F].astype(np.float32).reshape(Z, Y // F, F, X // F, F).mean((2, 4))


def norm(v):
    lo, hi = (float(q) for q in np.quantile(v, NORM_Q)); return np.clip((v - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def over_propose(iso):
    """Loose-NMS over-proposal; returns (M,4): z,y,x (ISO), response."""
    resp = None
    for s1, s2 in SCALE_PAIRS:
        d = gaussian_filter(iso, s1 / ISO) - gaussian_filter(iso, s2 / ISO)
        resp = d if resp is None else np.maximum(resp, d)
    thr = OP_REL * float(resp.max()) if resp.max() > 0 else OP_REL
    mind = max(1, int(round(OP_NMS_UM / ISO)))
    pk = peak_local_max(resp, min_distance=mind, threshold_abs=thr, exclude_border=False, num_peaks=MAX_PEAKS)
    if len(pk) == 0:
        return np.zeros((0, 4), np.float32)
    r = resp[pk[:, 0], pk[:, 1], pk[:, 2]]
    return np.column_stack([pk.astype(np.float32), r]).astype(np.float32)


def same_cell_sets(coords):
    n = len(coords)
    if n <= 1:
        return np.zeros(n, int)
    # coords are on the ISOTROPIC (downsampled) grid at ISO µm/voxel in all 3 axes, so scale
    # uniformly by ISO. Using the anisotropic SCALE here compresses XY 4x and merges distinct
    # nuclei (the raw->physical mapping only happens AFTER dedup, in detect()).
    Z = linkage(coords * ISO, method="complete")
    return fcluster(Z, t=R_SAME_UM, criterion="distance").astype(int) - 1


def detect(iso):
    """Over-propose -> same-cell dedup (brightest rep) -> RAW coords (matches local op_bright)."""
    cand = over_propose(iso)
    if len(cand) == 0:
        return np.zeros((0, 3), np.float32)
    coords = cand[:, :3]; labels = same_cell_sets(coords)
    reps = [idx[np.argmax(cand[idx, 3])] for lab in np.unique(labels) for idx in [np.where(labels == lab)[0]]]
    ref = coords[reps].astype(np.float32).copy()
    ref[:, 1] = ref[:, 1] * F + XY_OFF; ref[:, 2] = ref[:, 2] * F + XY_OFF
    return ref


def _hung(a, b, gate, pred=None):
    if len(a) == 0 or len(b) == 0:
        return []
    pa, pb = a * SCALE, b * SCALE
    q = pa if pred is None else pred
    cost = np.linalg.norm(q[:, None] - pb[None], axis=2); raw = np.linalg.norm(pa[:, None] - pb[None], axis=2)
    c = cost.copy(); c[raw > gate] = 1e6
    ri, ci = linear_sum_assignment(c)
    return [(int(r), int(cc)) for r, cc in zip(ri, ci) if raw[r, cc] <= gate]


def link(cbt):
    edges = []; vel = {}
    for t in range(len(cbt) - 1):
        a, b = cbt[t], cbt[t + 1]
        if len(a) == 0 or len(b) == 0:
            continue
        pa = a * SCALE; pred = pa.copy()
        for i in range(len(a)):
            if (t, i) in vel:
                pred[i] = pa[i] + VEL * vel[(t, i)]
        aa, bb = set(), set()
        for r, c in _hung(a, b, LINK1, pred):
            edges.append((t, r, t + 1, c)); vel[(t + 1, c)] = (b[c] - a[r]) * SCALE; aa.add(r); bb.add(c)
        ra = [i for i in range(len(a)) if i not in aa]; rb = [j for j in range(len(b)) if j not in bb]
        if ra and rb:
            for r, c in _hung(a[ra], b[rb], LINK2):
                edges.append((t, ra[r], t + 1, rb[c])); vel[(t + 1, rb[c])] = (b[rb[c]] - a[ra[r]]) * SCALE
    ls = {(t, i) for t, i, _, _ in edges}; lt = {(t2, j) for _, _, t2, j in edges}
    for t in range(len(cbt) - 2):
        a, b = cbt[t], cbt[t + 2]; ai = [i for i in range(len(a)) if (t, i) not in ls]; bj = [j for j in range(len(b)) if (t + 2, j) not in lt]
        if ai and bj:
            for r, c in _hung(a[ai], b[bj], GAP):
                edges.append((t, ai[r], t + 2, bj[c]))
    return edges


def comps(nodes, edges):
    adj = {n: set() for n in nodes}
    for t, i, t2, j in edges:
        adj[(t, i)].add((t2, j)); adj[(t2, j)].add((t, i))
    seen, out = set(), []
    for n in nodes:
        if n in seen:
            continue
        st, cc = [n], set()
        while st:
            u = st.pop()
            if u in seen:
                continue
            seen.add(u); cc.add(u); st.extend(adj[u] - seen)
        out.append(cc)
    return out


def smooth_coords(coord_by_id, edge_ids):
    """V11 temporal smoothing on RAW coords: out = w*self + (1-w)*mean(edge-neighbours' ORIGINAL pos).
    Single pass over original positions (matches biotrack run_phase1_ablation.smooth_sample)."""
    nbr = {k: [] for k in coord_by_id}
    for s, t in edge_ids:
        if s in coord_by_id and t in coord_by_id:
            nbr[s].append(t); nbr[t].append(s)
    orig = {k: np.asarray(v, np.float64) for k, v in coord_by_id.items()}
    out = {}
    for k, c in orig.items():
        ns = nbr[k]
        out[k] = SMOOTH_W * c + (1 - SMOOTH_W) * np.mean([orig[n] for n in ns], axis=0) if ns else c
    return out


def build(ds, cbt, edges):
    linked = {(t, i) for t, i, _, _ in edges} | {(t2, j) for _, _, t2, j in edges}
    keep = set().union(*[c for c in comps(linked, edges) if len(c) >= MINLEN]) if linked else set()
    nid, meta, coord_by_id, k = {}, {}, {}, 1
    for t, cs in enumerate(cbt):
        for i, (z, y, x) in enumerate(cs):
            if (t, i) not in keep:
                continue
            nid[(t, i)] = k; meta[k] = int(t); coord_by_id[k] = (float(z), float(y), float(x)); k += 1
    edge_ids = [(nid[(t, i)], nid[(t2, j)]) for t, i, t2, j in edges if (t, i) in nid and (t2, j) in nid]
    coords = smooth_coords(coord_by_id, edge_ids) if SMOOTH else coord_by_id
    rows = []
    for node_id in sorted(coord_by_id):
        z, y, x = coords[node_id]
        rows.append(dict(dataset=ds, row_type="node", node_id=node_id, t=meta[node_id], z=float(z), y=float(y), x=float(x), source_id=-1, target_id=-1))
    for s, t in edge_ids:
        rows.append(dict(dataset=ds, row_type="edge", node_id=-1, t=-1, z=-1, y=-1, x=-1, source_id=s, target_id=t))
    return rows


def infer(zp):
    a = open_vol(zp); cbt = [detect(norm(dsxy(read_frame(a, t)))) for t in range(a.shape[0])]
    return build(zp.stem, cbt, link(cbt))


def main():
    rows = []
    for zp in sorted(IN_DIR.glob("*.zarr")):
        r = infer(zp); print(zp.stem, sum(x["row_type"] == "node" for x in r), "nodes"); rows += r
    df = pd.DataFrame(rows).reset_index(drop=True); df.insert(0, "id", range(len(df)))
    df[COLS].to_csv(OUT_CSV, index=False); print("wrote", OUT_CSV, len(df))


if __name__ == "__main__":
    main()
