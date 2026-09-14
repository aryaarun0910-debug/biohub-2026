#!/usr/bin/env python3
"""EXP-41 -- how much of the fragmented pool is even ADDRESSABLE by endpoint stitching?

EXP-40 got 83% precision on flow-compensated stitches, far above the 48.1% break-even, but
recovered only ~10 of ~137 fragmented GT edges. Precision is not the constraint; REACH is.

A missing GT edge i->j can fail for structurally different reasons, and only one of them is
fixable by joining free endpoints:

  FREE       i has no outgoing edge AND j has no incoming edge   -> EXP-40 can propose it
  I_BUSY     i already has an outgoing edge (to something else)  -> needs a REWIRE
  J_BUSY     j already has an incoming edge (from something else)-> needs a REWIRE
  BOTH_BUSY  both                                                -> needs a double rewire
  NO_NODE    an endpoint was never detected                      -> unreachable (EXP-37: 0.34%)

The split decides the strategy. If most of the pool is FREE, tune the stitcher. If most is BUSY,
the gain needs edge REPLACEMENT, which is riskier: removing a wrong edge that the scorer never
charged (because its other endpoint is unannotated) costs nothing, but removing a correct one
costs a true positive.
"""
import sys, warnings
from pathlib import Path
import numpy as np, tracksdata as td
from scipy.spatial import cKDTree

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
warnings.filterwarnings("ignore")
import div_sweep as D

PRED = Path("work/repro_out/tracking_repo/predictions/unknown/unet_transformer/split_0")
GT = Path("data/train_geff"); TOL = 7.0

def load(p):
    g = td.graph.IndexedRXGraph.from_geff(p); return g[0] if isinstance(g, tuple) else g

tot = {k: 0 for k in ("FREE", "I_BUSY", "J_BUSY", "BOTH_BUSY", "NO_NODE", "PRESENT")}
for p in sorted(PRED.glob("*.geff")):
    gp = GT / f"{p.stem}.geff"
    if not gp.exists(): continue
    g = load(p)
    pos, tt = {}, {}
    for r in g.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        tt[i] = int(r["t"])
    edges = [(int(r["source_id"]), int(r["target_id"])) for r in g.edge_attrs().iter_rows(named=True)]
    eset = set(edges)
    outd, ind = {}, {}
    for s, d in edges: outd[s] = outd.get(s, 0) + 1; ind[d] = ind.get(d, 0) + 1

    gt = load(gp)
    gpos, gtt = {}, {}
    for r in gt.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gtt[i] = int(r["t"])
    ge = [(int(r["source_id"]), int(r["target_id"])) for r in gt.edge_attrs().iter_rows(named=True)]
    by_t = {}
    for i, t in tt.items(): by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}
    def match(q, t):
        ent = trees.get(t)
        if ent is None: return None
        tr, ids = ent; d, j = tr.query(q)
        return ids[int(j)] if d <= TOL else None

    per = {k: 0 for k in tot}
    for s, d in ge:
        if s not in gpos or d not in gpos: continue
        ms, md = match(gpos[s], gtt[s]), match(gpos[d], gtt[d])
        if ms is None or md is None: per["NO_NODE"] += 1; continue
        if (ms, md) in eset: per["PRESENT"] += 1; continue
        ib, jb = outd.get(ms, 0) > 0, ind.get(md, 0) > 0
        per["BOTH_BUSY" if (ib and jb) else "I_BUSY" if ib else "J_BUSY" if jb else "FREE"] += 1
    for k in per: tot[k] += per[k]
    miss = sum(per[k] for k in per if k != "PRESENT")
    print(f"  {p.stem}: GT edges {per['PRESENT']+miss:>5}  present {per['PRESENT']:>5}  "
          f"missing {miss:>4}  (FREE {per['FREE']}, I_BUSY {per['I_BUSY']}, "
          f"J_BUSY {per['J_BUSY']}, BOTH {per['BOTH_BUSY']}, NO_NODE {per['NO_NODE']})")

miss = sum(v for k, v in tot.items() if k != "PRESENT")
print(f"\n  GT edges {tot['PRESENT']+miss:,}   present {tot['PRESENT']:,}   MISSING {miss:,}\n")
for k in ("FREE", "I_BUSY", "J_BUSY", "BOTH_BUSY", "NO_NODE"):
    print(f"    {k:<10} {tot[k]:>5}   {tot[k]/miss:>6.1%} of missing")
print(f"\n  addressable by free-endpoint stitching (EXP-40): {tot['FREE']/miss:.1%}")
print(f"  requires rewiring an existing edge:              "
      f"{(tot['I_BUSY']+tot['J_BUSY']+tot['BOTH_BUSY'])/miss:.1%}")
