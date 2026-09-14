#!/usr/bin/env python3
"""EXP-45 -- why every geometric signal sided with the impostor.

EXP-44: flow_knn 39.9%, flow_affine 38.4%, neigh_topo 39.1%, back_cycle 39.9%, accel 15.4%.
All below 50%, all clustered near 39%. That uniformity is the tell -- it is what you get when
every feature is dominated by the same term, distance.

A flow correction can only reorder two candidates if the correction is LARGER than the gap it
has to overturn. So measure three things directly:

    |mu|        magnitude of the local displacement field
    gap         d(i, true) - d(i, nearest)   on the hard set
    coherence   |mu| / spread of neighbour displacements

If |mu| is small relative to gap, no flow-based feature of ANY sophistication can work -- affine,
deformation field, transformer, it does not matter, because the correction it applies is smaller
than the ordering error it must fix. And if coherence is low, the tissue is not moving
collectively at all, which kills the premise that neighbours predict each other.
"""
import sys, warnings
from pathlib import Path
import numpy as np, tracksdata as td
from scipy.spatial import cKDTree

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
warnings.filterwarnings("ignore")
import div_sweep as D

PRED = Path("work/repro_out/tracking_repo/predictions/unknown/unet_transformer_val/split_0")
GT = Path("data/train_geff"); TOL = 7.0; K, SIG = 24, 25.0

def load(p):
    g = td.graph.IndexedRXGraph.from_geff(p); return g[0] if isinstance(g, tuple) else g

MU, GAP, COH, DTRUE, DNEAR, SPACING = [], [], [], [], [], []
for p in sorted(PRED.glob("*.geff")):
    gp = GT / f"{p.stem}.geff"
    if not gp.exists(): continue
    g = load(p)
    pos, tt = {}, {}
    for r in g.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        tt[i] = int(r["t"])
    succ = {}
    for r in g.edge_attrs().iter_rows(named=True):
        succ.setdefault(int(r["source_id"]), []).append(int(r["target_id"]))
    by_t = {}
    for i, t in tt.items(): by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}
    link_by_t = {}
    for s, ds in succ.items():
        if len(ds) == 1 and tt.get(ds[0]) == tt.get(s, -9) + 1:
            link_by_t.setdefault(tt[s], []).append((s, ds[0]))
    ftree = {t: (cKDTree(np.stack([pos[s] for s, _ in L])), L) for t, L in link_by_t.items() if len(L) >= 8}

    gt = load(gp)
    gpos, gtt = {}, {}
    for r in gt.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gtt[i] = int(r["t"])
    def match(q, t):
        ent = trees.get(t)
        if ent is None: return None
        tr, ids = ent; d, j = tr.query(q)
        return ids[int(j)] if d <= TOL else None

    for r in gt.edge_attrs().iter_rows(named=True):
        s, d = int(r["source_id"]), int(r["target_id"])
        if s not in gpos or d not in gpos: continue
        mi, mj = match(gpos[s], gtt[s]), match(gpos[d], gtt[d])
        if mi is None or mj is None: continue
        t = tt[mi]; ent = trees.get(t + 1)
        if ent is None: continue
        tr, ids = ent
        dist_all, _ = tr.query(pos[mi], k=min(6, len(ids)))
        nn = ids[int(tr.query(pos[mi])[1])]
        if nn == mj: continue
        ent2 = ftree.get(t)
        if ent2 is None: continue
        ft, L = ent2
        _, idx = ft.query(pos[mi], k=min(K, len(L)))
        sel = [L[q] for q in np.atleast_1d(idx) if L[q][0] != mi]
        if len(sel) < 6: continue
        P0 = np.stack([pos[a] for a, _ in sel]); dsp = np.stack([pos[b] - pos[a] for a, b in sel])
        w = np.exp(-(np.linalg.norm(P0 - pos[mi], axis=1) ** 2) / (2 * SIG ** 2)); w /= max(w.sum(), 1e-9)
        mu = (dsp * w[:, None]).sum(0)
        spread = float(np.sqrt(((np.linalg.norm(dsp - mu, axis=1) ** 2) * w).sum()))
        dt_, dn_ = float(np.linalg.norm(pos[mj] - pos[mi])), float(np.linalg.norm(pos[nn] - pos[mi]))
        MU.append(float(np.linalg.norm(mu))); GAP.append(dt_ - dn_); COH.append(float(np.linalg.norm(mu)) / (spread + 1e-9))
        DTRUE.append(dt_); DNEAR.append(dn_)
        SPACING.append(float(np.atleast_1d(dist_all)[1]) if len(np.atleast_1d(dist_all)) > 1 else np.nan)

MU, GAP, COH = np.array(MU), np.array(GAP), np.array(COH)
print(f"\n  hard-set cases: {len(MU)}\n")
print(f"  {'quantity':<42}{'median':>9}{'p90':>9}")
print("  " + "-" * 60)
print(f"  {'|mu|  local displacement field (um)':<42}{np.median(MU):>9.2f}{np.percentile(MU,90):>9.2f}")
print(f"  {'gap   d(true) - d(nearest)  (um)':<42}{np.median(GAP):>9.2f}{np.percentile(GAP,90):>9.2f}")
print(f"  {'d(i, true successor)        (um)':<42}{np.median(DTRUE):>9.2f}{np.percentile(DTRUE,90):>9.2f}")
print(f"  {'d(i, nearest impostor)      (um)':<42}{np.median(DNEAR):>9.2f}{np.percentile(DNEAR,90):>9.2f}")
print(f"  {'nearest-neighbour spacing at t+1 (um)':<42}{np.nanmedian(SPACING):>9.2f}{np.nanpercentile(SPACING,90):>9.2f}")
print(f"  {'coherence |mu| / spread':<42}{np.median(COH):>9.2f}{np.percentile(COH,90):>9.2f}")
print(f"\n  |mu| exceeds the gap it must overturn in {(MU>GAP).mean():.1%} of hard cases")
print(f"  local motion is coherent (|mu| > spread) in {(COH>1).mean():.1%} of hard cases")
