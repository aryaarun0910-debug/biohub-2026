#!/usr/bin/env python3
"""EXP-28 -- can a division STEAL a claimed daughter, and is the region separable?

EXP-27: 65.6% of real divisions are missed because the second daughter is already another
parent's assigned target, so add_safe_divisions -- which only considers UNCLAIMED targets --
never offers the candidate. That is two thirds of the division problem, untouched by every
experiment so far.

This asks the two questions that decide whether it is worth pursuing:
  1. How many candidates appear if claimed targets may be stolen? (the new pool's size)
  2. Is the TRUE division separable within that pool? (base rate and per-feature AUC)

The exchange rate demands better than 24% precision. EXP-26 found the unclaimed region sits at a
0.16% base rate and is hopeless. If the claimed region is materially richer, divisions reopen.
"""
import sys
from pathlib import Path
import numpy as np, tracksdata as td
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
import div_sweep as D

GT = Path("data/train_geff"); GRAPHS = Path("work/train_graphs"); TOL = 7.0
FEATS = ["parent_dist", "sister_dist", "child_dist", "cos",
         "arc_max", "arc_min", "arc_asym", "arc_sum", "rival_dist", "rival_margin"]
MAXP, MAXS = 12.0, 20.0

X, y = [], []
for p in sorted(GRAPHS.rglob("*.geff")):
    gp = GT / f"{p.stem}.geff"
    if not gp.exists():
        continue
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs(); e = g.edge_attrs()
    pos, tt = {}, {}
    for r in n.iter_rows(named=True):
        i = int(r["node_id"]); pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        tt[i] = int(r["t"])
    owner, first_child = {}, {}
    for r in e.iter_rows(named=True):
        s_, d_ = int(r["source_id"]), int(r["target_id"])
        owner[d_] = s_                       # who currently claims this target
        first_child.setdefault(s_, d_)

    gg = td.graph.IndexedRXGraph.from_geff(gp); gg = gg[0] if isinstance(gg, tuple) else gg
    gpos, gtt = {}, {}
    for r in gg.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gtt[i] = int(r["t"])
    gout = {}
    for r in gg.edge_attrs().iter_rows(named=True):
        gout.setdefault(int(r["source_id"]), []).append(int(r["target_id"]))
    gdiv = [(gpos[s], gpos[k[0]], gpos[k[1]]) for s, k in gout.items()
            if len(k) == 2 and s in gpos and k[0] in gpos and k[1] in gpos]
    if not gdiv:
        continue

    by_t = {}
    for i, t in tt.items():
        by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}

    # every (parent with one child) x (CLAIMED node at t+1) pair inside distance bounds
    for sid, cid in first_child.items():
        t = tt[sid]
        if tt.get(cid) != t + 1:
            continue
        ent = trees.get(t + 1)
        if ent is None:
            continue
        tree, ids = ent
        near = tree.query_ball_point(pos[sid], MAXP)
        for j in near:
            qid = ids[j]
            if qid == cid or qid not in owner:      # only CLAIMED targets: the new pool
                continue
            pd = float(np.linalg.norm(pos[qid] - pos[sid]))
            sd = float(np.linalg.norm(pos[qid] - pos[cid]))
            if sd > MAXS:
                continue
            rival = owner[qid]
            rd = float(np.linalg.norm(pos[qid] - pos[rival]))
            va, vb = pos[cid] - pos[sid], pos[qid] - pos[sid]
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na < 1e-9 or nb < 1e-9:
                continue
            f = dict(parent_dist=pd, sister_dist=sd, child_dist=float(na),
                     cos=float(va @ vb / (na * nb)), arc_max=max(na, nb), arc_min=min(na, nb),
                     arc_asym=abs(na - nb), arc_sum=na + nb,
                     rival_dist=rd, rival_margin=rd - pd)   # >0 means WE are the closer parent
            lab = 0
            for P, A, B in gdiv:
                if np.linalg.norm(pos[sid] - P) > TOL:
                    continue
                if ((np.linalg.norm(pos[cid] - A) <= TOL and np.linalg.norm(pos[qid] - B) <= TOL) or
                        (np.linalg.norm(pos[cid] - B) <= TOL and np.linalg.norm(pos[qid] - A) <= TOL)):
                    lab = 1; break
            X.append([f[k] for k in FEATS]); y.append(lab)

X, y = np.array(X), np.array(y)
print(f"\n  CLAIMED-TARGET candidate pool: {len(y):,}   true divisions {y.sum()}")
print(f"  base rate {y.mean():.4%}   (unclaimed region was 0.1606%)\n")
if y.sum() >= 3:
    print("  per-feature AUC:")
    for i, f in enumerate(FEATS):
        a = roc_auc_score(y, X[:, i]); print(f"    {f:<14} {max(a, 1-a):.3f}")
    best = max((max(roc_auc_score(y, X[:, i]), 1-roc_auc_score(y, X[:, i])), FEATS[i])
               for i in range(len(FEATS)))
    print(f"    best: {best[1]} at {best[0]:.3f}")
    print(f"\n  precision needed to pay: 24%  ->  lift required {0.24/max(y.mean(),1e-9):.0f}x")
