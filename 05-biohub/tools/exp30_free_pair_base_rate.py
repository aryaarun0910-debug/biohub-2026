#!/usr/bin/env python3
"""EXP-30 -- the base rate that decides whether BOTH_FREE_NO_EDGE is recoverable.

EXP-29 found 2,715 ground-truth edges where both endpoints were detected, NEITHER was claimed by
the linker, and no edge was made -- worth +0.0207 of score. Divisions died on base rate, not on
model quality, so the same question must be asked here BEFORE any modelling:

    among all (free source, free target) pairs at consecutive frames within a plausible radius,
    what fraction are real ground-truth edges?

If it is percent-scale, this is tractable in a way divisions never were. If it is 0.1%, it dies
the same death.
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
RADIUS = 10.0
npair = npos = 0
dists, labels = [], []

for p in sorted(GRAPHS.rglob("*.geff"))[:60]:
    gp = GT / f"{p.stem}.geff"
    if not gp.exists():
        continue
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs(); e = g.edge_attrs()
    pos, tt = {}, {}
    for r in n.iter_rows(named=True):
        i = int(r["node_id"]); pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        tt[i] = int(r["t"])
    claimed_t, linked_s = set(), set()
    for r in e.iter_rows(named=True):
        claimed_t.add(int(r["target_id"])); linked_s.add(int(r["source_id"]))

    gg = td.graph.IndexedRXGraph.from_geff(gp); gg = gg[0] if isinstance(gg, tuple) else gg
    gpos, gtt = {}, {}
    for r in gg.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gtt[i] = int(r["t"])
    gset = set()
    for r in gg.edge_attrs().iter_rows(named=True):
        gset.add((int(r["source_id"]), int(r["target_id"])))
    if not gset:
        continue

    by_t = {}
    for i, t in tt.items():
        by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}
    def match(pt, t):
        ent = trees.get(t)
        if ent is None: return None
        tree, ids = ent
        d, j = tree.query(pt)
        return ids[int(j)] if d <= TOL else None
    # the true free-pair edges, expressed in predicted-node ids
    true_pairs = set()
    for s, d in gset:
        if s in gpos and d in gpos:
            ms, md = match(gpos[s], gtt[s]), match(gpos[d], gtt[d])
            if ms is not None and md is not None:
                true_pairs.add((ms, md))

    for t, ids in by_t.items():
        src = [i for i in ids if i not in linked_s]
        nxt = [i for i in by_t.get(t + 1, []) if i not in claimed_t]
        if not src or not nxt:
            continue
        tree = cKDTree(np.stack([pos[i] for i in nxt]))
        for s_ in src:
            for j in tree.query_ball_point(pos[s_], RADIUS):
                q = nxt[j]
                npair += 1
                lab = 1 if (s_, q) in true_pairs else 0
                npos += lab
                dists.append(float(np.linalg.norm(pos[q] - pos[s_]))); labels.append(lab)

print(f"\n  free (source, target) pairs within {RADIUS}um: {npair:,}")
print(f"  of which real GT edges:                  {npos:,}")
print(f"  BASE RATE:                               {npos/max(npair,1):.3%}")
print(f"  (division candidate regions were 0.161% and 0.082%)")
if npos > 3:
    a = roc_auc_score(labels, -np.array(dists))
    print(f"\n  distance alone separates at AUC {max(a,1-a):.3f}")
    o = np.argsort(dists)
    yy = np.array(labels)[o]
    for K in (1000, 5000, 10000):
        if K <= len(yy):
            print(f"    nearest {K:,}: precision {yy[:K].mean():.1%}  recovers {int(yy[:K].sum()):,}")
