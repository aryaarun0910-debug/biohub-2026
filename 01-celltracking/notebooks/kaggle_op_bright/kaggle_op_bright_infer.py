"""Kaggle inference: over-propose -> same-cell dedup -> V3 velocity linking (op_bright).

Beats the V3 anchor locally (min-fold adjJ 0.677 -> 0.720, +0.043; hard-fold recall 0.827->0.889).
Over-propose with LOOSE NMS (recovers subthreshold cells), collapse same-cell duplicates via
complete-linkage with a hard diameter cap (no chain-bridging), keep the brightest representative
per cell, then link with the V3 two-pass velocity-aware Hungarian. Self-contained: numpy/scipy/
skimage + tensorstore (Kaggle has no zarr). No internet needed.

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
REFINE = (1, 4, 4)
LINK1, LINK2, VEL, GAP, MINLEN = 6.0, 8.0, 0.5, 6.0, 4  # V3 linker
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
    Z = linkage(coords * SCALE, method="complete")
    return fcluster(Z, t=R_SAME_UM, criterion="distance").astype(int) - 1


def refine_com(vol, coords):
    Z, Y, X = vol.shape; rz, ry, rx = REFINE; out = coords.astype(np.float32).copy()
    for i, (z, y, x) in enumerate(coords.astype(int)):
        z0, z1 = max(0, z - rz), min(Z, z + rz + 1); y0, y1 = max(0, y - ry), min(Y, y + ry + 1); x0, x1 = max(0, x - rx), min(X, x + rx + 1)
        w = vol[z0:z1, y0:y1, x0:x1]; s = w.sum()
        if s > 0:
            zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]; out[i] = [(zz * w).sum() / s, (yy * w).sum() / s, (xx * w).sum() / s]
    return out


def detect(iso):
    """Over-propose -> same-cell dedup (brightest rep) -> COM refine -> RAW coords."""
    cand = over_propose(iso)
    if len(cand) == 0:
        return np.zeros((0, 3), np.float32)
    coords = cand[:, :3]; labels = same_cell_sets(coords)
    reps = [idx[np.argmax(cand[idx, 3])] for lab in np.unique(labels) for idx in [np.where(labels == lab)[0]]]
    ref = refine_com(iso, coords[reps])
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


def build(ds, cbt, edges):
    linked = {(t, i) for t, i, _, _ in edges} | {(t2, j) for _, _, t2, j in edges}
    keep = set().union(*[c for c in comps(linked, edges) if len(c) >= MINLEN]) if linked else set()
    nid, rows, k = {}, [], 1
    for t, cs in enumerate(cbt):
        for i, (z, y, x) in enumerate(cs):
            if (t, i) not in keep:
                continue
            nid[(t, i)] = k; rows.append(dict(dataset=ds, row_type="node", node_id=k, t=int(t), z=float(z), y=float(y), x=float(x), source_id=-1, target_id=-1)); k += 1
    for t, i, t2, j in edges:
        if (t, i) in nid and (t2, j) in nid:
            rows.append(dict(dataset=ds, row_type="edge", node_id=-1, t=-1, z=-1, y=-1, x=-1, source_id=nid[(t, i)], target_id=nid[(t2, j)]))
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
