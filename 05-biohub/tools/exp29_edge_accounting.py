#!/usr/bin/env python3
"""EXP-29 -- the complete edge accounting, at REAL density, on their own graphs.

The edge term carries ten times the weight of divisions and sits at 0.926. EXP-17 concluded most
lost edges are unrecoverable ground-truth artefacts -- but it measured ORACLE detection at 6.7
cells per frame, and their pipeline runs at 203. That is exactly the error EXP-27 caught in the
division work: generalising from the wrong regime.

Their diagnostics also report wrong_association_edges = 0, which cannot be reconciled with EXP-27
showing 99 divisions whose daughter is assigned to a different parent. For a division the ground
truth holds two edges from one parent; assigning one daughter elsewhere IS a wrong association.
One of the two measurements is counting something narrower than it appears.

Every ground-truth edge is classified by what actually happened to it:

  REPRODUCED            the edge is in the prediction
  ENDPOINT_MISSING      source or target has no predicted node within 7um -- detection
  TARGET_TAKEN          the target is assigned to a DIFFERENT parent -- wrong association
  SOURCE_LINKED_AWAY    the source is linked to a different target, the true one left free
  BOTH_FREE_NO_EDGE     both endpoints free and unlinked -- pure assignment miss
"""
import sys
from collections import Counter
from pathlib import Path
import numpy as np, tracksdata as td
from scipy.spatial import cKDTree

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
import div_sweep as D

GT = Path("data/train_geff"); GRAPHS = Path("work/train_graphs"); TOL = 7.0
verdict, per_emb, dist_lost = Counter(), {}, []

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
    pred = set(); owner = {}; out_of = {}
    for r in e.iter_rows(named=True):
        s_, d_ = int(r["source_id"]), int(r["target_id"])
        pred.add((s_, d_)); owner[d_] = s_
        out_of.setdefault(s_, []).append(d_)

    gg = td.graph.IndexedRXGraph.from_geff(gp); gg = gg[0] if isinstance(gg, tuple) else gg
    gpos, gtt = {}, {}
    for r in gg.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gtt[i] = int(r["t"])
    gedges = [(int(r["source_id"]), int(r["target_id"]))
              for r in gg.edge_attrs().iter_rows(named=True)]
    if not gedges:
        continue

    by_t = {}
    for i, t in tt.items():
        by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}

    def match(point, t):
        ent = trees.get(t)
        if ent is None:
            return None
        tree, ids = ent
        d, j = tree.query(point)
        return ids[int(j)] if d <= TOL else None

    emb = p.stem[:4]; per_emb.setdefault(emb, Counter())
    for s, d in gedges:
        if s not in gpos or d not in gpos:
            continue
        ms, md = match(gpos[s], gtt[s]), match(gpos[d], gtt[d])
        if ms is None or md is None:
            v = "ENDPOINT_MISSING"
        elif (ms, md) in pred:
            v = "REPRODUCED"
        elif md in owner:
            v = "TARGET_TAKEN"
        elif out_of.get(ms):
            v = "SOURCE_LINKED_AWAY"
        else:
            v = "BOTH_FREE_NO_EDGE"
        verdict[v] += 1; per_emb[emb][v] += 1
        if v != "REPRODUCED":
            dist_lost.append(float(np.linalg.norm(gpos[d] - gpos[s])))

tot = sum(verdict.values())
print(f"\n  GROUND-TRUTH EDGES: {tot:,}\n")
for v, c in verdict.most_common():
    print(f"    {v:<20} {c:>7,}  {c/tot:>6.2%}")
lost = tot - verdict["REPRODUCED"]
print(f"\n  lost: {lost:,} ({lost/tot:.2%})")
if dist_lost:
    q = np.percentile(dist_lost, (50, 90, 99))
    print(f"  displacement of LOST edges: p50 {q[0]:.2f}um  p90 {q[1]:.2f}  p99 {q[2]:.2f}")
    print(f"  (EXP-17 at oracle density measured p50 12.19um for lost edges)")
print("\n  per embryo:")
for emb, c in sorted(per_emb.items()):
    s = sum(c.values())
    print(f"    {emb} (n={s:,}): " + "  ".join(f"{k}={v}" for k, v in c.most_common()))
