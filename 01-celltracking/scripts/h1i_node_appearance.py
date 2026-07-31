"""H1-I: per-node appearance feature extraction from raw image volumes.

Computes appearance descriptors ONCE PER UNIQUE NODE (never per candidate pair),
at the node's own frame and at temporal offsets dt in {-2,-1,0,+1,+2}.
Candidate-level features are formed later by cheap joins (h1i_candidate_eval.py).

All spatial moments are computed in MICRONS using the OME voxel spacing
(z, y, x) = (1.625, 0.40625, 0.40625) um.

Node coordinates in artifacts/.../graphs/{fold}/{crop}.parquet are VOXEL indices
into the (T=100, Z=64, Y=256, X=256) uint16 array at data/train/{crop}.zarr/0.

Implementation notes (this is the compute-bound step):
  * the sampling kernel holds only voxels inside the r<=6um sphere, sorted by
    radius, so every radial statistic is a contiguous slice;
  * frames are edge-padded once so box gathers are a single integer add with no
    per-voxel clipping;
  * every moment up to order 4 is obtained from ONE gemm (w @ MOMENT_MAT), and
    the axial (principal-axis) moments are recovered analytically from them.

Output: one parquet per crop, one row per (node_id, dt).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

# ---------------------------------------------------------------- constants

VOX = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)  # (z, y, x) um

R_INNER = 1.5   # um, peak search radius
R_CORE = 3.0    # um, "nucleus" core
R_MID = 4.5     # um
R_OUTER = 6.0   # um, local neighbourhood outer edge

HZ = int(np.ceil(R_OUTER / VOX[0]))   # 4
HY = int(np.ceil(R_OUTER / VOX[1]))   # 15
HX = int(np.ceil(R_OUTER / VOX[2]))   # 15

DTS = (-2, -1, 0, 1, 2)

# multi-indices (i,j,k) with i+j+k == n, in a fixed order
MULTI = {n: [t for t in itertools.product(range(n + 1), repeat=3) if sum(t) == n]
         for n in (1, 2, 3, 4)}
MULTI_OFF = {}
_c = 0
for _n in (1, 2, 3, 4):
    MULTI_OFF[_n] = _c
    _c += len(MULTI[_n])
N_MOM = _c  # 3 + 6 + 10 + 15 = 34


def _build_kernel():
    dz = np.arange(-HZ, HZ + 1) * VOX[0]
    dy = np.arange(-HY, HY + 1) * VOX[1]
    dx = np.arange(-HX, HX + 1) * VOX[2]
    DZ, DY, DX = np.meshgrid(dz, dy, dx, indexing="ij")
    r = np.sqrt(DZ ** 2 + DY ** 2 + DX ** 2)
    keep = r <= R_OUTER
    order = np.argsort(r[keep], kind="stable")

    dzk = DZ[keep][order].astype(np.float32)
    dyk = DY[keep][order].astype(np.float32)
    dxk = DX[keep][order].astype(np.float32)
    rk = r[keep][order].astype(np.float32)

    iz, iy, ix = np.meshgrid(np.arange(-HZ, HZ + 1), np.arange(-HY, HY + 1),
                             np.arange(-HX, HX + 1), indexing="ij")
    vox_off = np.stack([iz[keep][order], iy[keep][order], ix[keep][order]], axis=1)

    mat = np.empty((rk.size, N_MOM), dtype=np.float32)
    for n in (1, 2, 3, 4):
        for c, (i, j, k) in enumerate(MULTI[n]):
            mat[:, MULTI_OFF[n] + c] = (dzk ** i) * (dyk ** j) * (dxk ** k)

    # radius-sorted -> contiguous shell boundaries
    bnd = {name: int(np.searchsorted(rk, R, side="right"))
           for name, R in (("inner", R_INNER), ("s1", 1.5), ("s2", 3.0),
                           ("s3", 4.5), ("core", R_CORE), ("mid", R_MID))}
    return dzk, dyk, dxk, rk, vox_off, mat, bnd


DZK, DYK, DXK, RK, VOX_OFF, MOMENT_MAT, BND = _build_kernel()
K = RK.size
MOMENT_MAT = np.ascontiguousarray(MOMENT_MAT)

FEATURE_COLS = [
    "mass_core", "mass_outer", "peak_inner", "peak_mid", "shell_mean",
    "centroid_off_um", "lam1", "lam2", "lam3", "v1z", "v1y", "v1x",
    "shell0", "shell1", "shell2", "shell3", "axial_kurt", "axial_skew",
    "bg", "hi",
]


def _axial_moments(G: np.ndarray, v: np.ndarray) -> tuple:
    """Project raw 3D moments G (n, N_MOM) onto unit axis v (n,3) -> A1..A4."""
    out = []
    for n in (1, 2, 3, 4):
        acc = np.zeros(G.shape[0], dtype=np.float64)
        off = MULTI_OFF[n]
        for c, (i, j, k) in enumerate(MULTI[n]):
            coef = math.factorial(n) / (math.factorial(i) * math.factorial(j)
                                        * math.factorial(k))
            term = np.ones(G.shape[0])
            if i:
                term = term * v[:, 0] ** i
            if j:
                term = term * v[:, 1] ** j
            if k:
                term = term * v[:, 2] ** k
            acc += coef * term * G[:, off + c]
        out.append(acc)
    return out


def _node_features(box: np.ndarray, bg: float) -> dict:
    """box: (n, K) raw intensities inside the r<=6um sphere, radius sorted."""
    w = box.astype(np.float32)
    w -= np.float32(bg)
    np.maximum(w, 0.0, out=w)

    s0 = w[:, :BND["s1"]].sum(axis=1)
    s1 = w[:, BND["s1"]:BND["s2"]].sum(axis=1)
    s2 = w[:, BND["s2"]:BND["s3"]].sum(axis=1)
    s3 = w[:, BND["s3"]:].sum(axis=1)
    mass_core = s0 + s1
    mass_outer = mass_core + s2 + s3
    peak_inner = w[:, :BND["inner"]].max(axis=1)
    peak_mid = w[:, :BND["mid"]].max(axis=1)

    n_s = np.array([BND["s1"], BND["s2"] - BND["s1"], BND["s3"] - BND["s2"],
                    K - BND["s3"]], dtype=np.float32)

    G = (w @ MOMENT_MAT).astype(np.float64)          # (n, N_MOM) single gemm
    sw = np.maximum(mass_outer.astype(np.float64), 1e-6)
    G /= sw[:, None]

    o1, o2 = MULTI_OFF[1], MULTI_OFF[2]
    cz, cy, cx = G[:, o1], G[:, o1 + 1], G[:, o1 + 2]
    c = np.stack([cz, cy, cx], axis=1)
    # MULTI[2] order: (0,0,2),(0,1,1),(0,2,0),(1,0,1),(1,1,0),(2,0,0)
    mxx = G[:, o2 + 0] - cx * cx
    myx = G[:, o2 + 1] - cy * cx
    myy = G[:, o2 + 2] - cy * cy
    mzx = G[:, o2 + 3] - cz * cx
    mzy = G[:, o2 + 4] - cz * cy
    mzz = G[:, o2 + 5] - cz * cz

    n = w.shape[0]
    T = np.empty((n, 3, 3))
    T[:, 0, 0] = mzz
    T[:, 1, 1] = myy
    T[:, 2, 2] = mxx
    T[:, 0, 1] = T[:, 1, 0] = mzy
    T[:, 0, 2] = T[:, 2, 0] = mzx
    T[:, 1, 2] = T[:, 2, 1] = myx
    evals, evecs = np.linalg.eigh(T)
    lam3, lam2, lam1 = evals[:, 0], evals[:, 1], evals[:, 2]
    v1 = evecs[:, :, 2]
    v1 = v1 * np.sign(np.where(v1[:, 0:1] == 0, 1.0, v1[:, 0:1]))  # sign gauge

    A1, A2, A3, A4 = _axial_moments(G, v1)
    ab = np.einsum("ij,ij->i", c, v1)          # centroid projected on axis
    m2 = A2 - ab ** 2
    m3 = A3 - 3 * ab * A2 + 2 * ab ** 3
    m4 = A4 - 4 * ab * A3 + 6 * ab ** 2 * A2 - 3 * ab ** 4
    m2s = np.maximum(m2, 1e-9)
    axial_kurt = m4 / (m2s * m2s)
    axial_skew = m3 / m2s ** 1.5

    return {
        "mass_core": mass_core, "mass_outer": mass_outer,
        "peak_inner": peak_inner, "peak_mid": peak_mid,
        "shell_mean": s3 / n_s[3], "centroid_off_um": np.sqrt(cz**2 + cy**2 + cx**2),
        "lam1": lam1, "lam2": lam2, "lam3": lam3,
        "v1z": v1[:, 0], "v1y": v1[:, 1], "v1x": v1[:, 2],
        "shell0": s0 / n_s[0], "shell1": s1 / n_s[1],
        "shell2": s2 / n_s[2], "shell3": s3 / n_s[3],
        "axial_kurt": axial_kurt, "axial_skew": axial_skew,
    }


def extract_crop(crop: str, fold: int, root: Path, out_path: Path,
                 dts=DTS, block: int = 2048,
                 node_ids: np.ndarray | None = None) -> dict:
    t0 = time.time()
    g = pq.read_table(root / f"artifacts/kaggle/e0c_cache/graphs/{fold}/{crop}.parquet",
                      columns=["row_type", "node_id", "t", "z", "y", "x"]).to_pandas()
    nodes = g[g.row_type == "node"]
    if node_ids is not None:
        nodes = nodes[nodes.node_id.isin(node_ids)]
    nodes = nodes.reset_index(drop=True)

    arr = zarr.open_group(str(root / f"data/train/{crop}.zarr"), mode="r")["0"]
    T, Z, Y, X = arr.shape
    PY, PX = Y + 2 * HY, X + 2 * HX

    nt = nodes.t.to_numpy()
    nid = nodes.node_id.to_numpy()
    iz = np.clip(np.rint(nodes.z.to_numpy()), 0, Z - 1).astype(np.int64)
    iy = np.clip(np.rint(nodes.y.to_numpy()), 0, Y - 1).astype(np.int64)
    ix = np.clip(np.rint(nodes.x.to_numpy()), 0, X - 1).astype(np.int64)
    inside = ((iz >= HZ) & (iz < Z - HZ) & (iy >= HY) & (iy < Y - HY)
              & (ix >= HX) & (ix < X - HX))
    base = ((iz + HZ) * (PY * PX) + (iy + HY) * PX + (ix + HX)).astype(np.int64)
    off_flat = (VOX_OFF[:, 0].astype(np.int64) * (PY * PX)
                + VOX_OFF[:, 1].astype(np.int64) * PX
                + VOX_OFF[:, 2].astype(np.int64))

    by_time = {}
    for i, t in enumerate(nt):
        by_time.setdefault(int(t), []).append(i)
    by_time = {t: np.asarray(v, dtype=np.int64) for t, v in by_time.items()}

    need = {}
    for t, idxs in by_time.items():
        for dt in dts:
            f = t + dt
            if 0 <= f < T:
                need.setdefault(f, []).append((dt, idxs))

    out_chunks = []
    n_gathered = 0
    t_read = 0.0
    for f in sorted(need):
        tr = time.time()
        frame = np.asarray(arr[f])
        t_read += time.time() - tr
        sub = frame[::2, ::3, ::3]
        bg = float(np.percentile(sub, 20.0))
        hi = float(np.percentile(sub, 99.5))
        padded = np.pad(frame, ((HZ, HZ), (HY, HY), (HX, HX)), mode="edge").ravel()
        for dt, sel in need[f]:
            feats = {k: np.empty(sel.size, dtype=np.float32)
                     for k in FEATURE_COLS if k not in ("bg", "hi")}
            for s0 in range(0, sel.size, block):
                bidx = sel[s0:s0 + block]
                box = padded[base[bidx][:, None] + off_flat[None, :]]
                n_gathered += box.size
                for k, v in _node_features(box, bg).items():
                    feats[k][s0:s0 + bidx.size] = v
            df = pd.DataFrame(feats)
            df.insert(0, "node_id", nid[sel])
            df.insert(1, "dt", np.int8(dt))
            df.insert(2, "frame", np.int16(f))
            df["inside"] = inside[sel]
            df["bg"] = np.float32(bg)
            df["hi"] = np.float32(hi)
            out_chunks.append(df)

    res = pd.concat(out_chunks, ignore_index=True)
    res["crop"] = crop
    res["fold"] = np.int8(fold)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(res, preserve_index=False), out_path,
                   compression="zstd")
    return {"crop": crop, "n_nodes": int(len(nodes)), "n_rows": int(len(res)),
            "n_frames": len(need), "voxels_gathered": int(n_gathered),
            "read_s": round(t_read, 1), "seconds": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", nargs="+", help="fold:crop specs")
    ap.add_argument("--crops-file", help="file with one fold:crop per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--root", default=r"c:\Users\aryaa\Documents\Biohub-CellTracking-2026")
    ap.add_argument("--census", default="artifacts/kaggle/e0c_cache/fork_candidates/h0c_top3")
    ap.add_argument("--metric-visible-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    specs = list(args.crops or [])
    if args.crops_file:
        specs += [ln.strip() for ln in Path(args.crops_file).read_text().splitlines()
                  if ln.strip()]

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    stats = []
    for spec in specs:
        fold_s, crop = spec.split(":", 1)
        fold = int(fold_s)
        dest = out_dir / str(fold) / f"{crop}.parquet"
        if args.skip_existing and dest.exists():
            continue
        node_ids = None
        if args.metric_visible_only:
            c = pq.read_table(root / args.census / str(fold) / f"{crop}.parquet",
                              columns=["mother", "d1", "d2", "metric_visible"]).to_pandas()
            c = c[c.metric_visible]
            node_ids = np.unique(np.concatenate(
                [c.mother.values, c.d1.values, c.d2.values]))
        stats.append(extract_crop(crop, fold, root, dest, node_ids=node_ids))
        print(json.dumps(stats[-1]), flush=True)
    if stats:
        tot = sum(s["seconds"] for s in stats)
        print(json.dumps({"total_seconds": round(tot, 2), "n_crops": len(stats),
                          "mean_seconds_per_crop": round(tot / len(stats), 2)}), flush=True)


if __name__ == "__main__":
    main()
