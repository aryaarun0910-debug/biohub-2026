#!/usr/bin/env python3
"""EXP-50 -- are missed divisions EXCLUDED from the candidate set before any gate sees them?

EXP-38b concluded divisions were closed: eight gate relaxations, divTP never moved, only FPs
added. That conclusion may rest on an upstream filter rather than on the gates. In
add_safe_divisions:

    cands = [i for i in kids_frame if i not in incoming and i not in used_t]

Only daughters with NO INCOMING EDGE are ever proposed. The error atlas puts 65.6% of missed
divisions in "contention" -- the daughter is already linked to some other parent. If that is
right, those divisions are excluded before any gate is evaluated, and no relaxation of max_um,
symmetry_tau or diverge_um could possibly recover them. The gates were never the binding
constraint.

This measures it directly on their validator output. For every GT division whose parent is
detected, classify why it is missing:

    PRESENT      already predicted as a division
    FREE         both daughters unlinked -> the gates COULD have proposed it
    CONTENDED    at least one daughter already has a parent -> structurally excluded
    NO_DAUGHTER  a daughter was never detected -> unreachable

If CONTENDED dominates, the division route is open and needs a different mechanism: allow
contended daughters as candidates and decide whether to steal them. The public notebook
"A dividing nucleus gets smaller, not dimmer" supplies a plausible deciding signal -- nucleus
VOLUME drops ~0.27 around a division while PEAK brightness holds, and the drop begins 1-2 frames
BEFORE the split, so it is predictive rather than merely descriptive.
"""
import sys, warnings
from pathlib import Path
import numpy as np, tracksdata as td
from scipy.spatial import cKDTree

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
warnings.filterwarnings("ignore")
import div_sweep as D

PRED = Path("work/repro_out/tracking_repo/predictions/unknown/unet_transformer_val/split_0")
GT = Path("data/train_geff"); TOL = 7.0

def load(p):
    g = td.graph.IndexedRXGraph.from_geff(p); return g[0] if isinstance(g, tuple) else g

tot = {k: 0 for k in ("PRESENT", "FREE", "CONTENDED", "NO_DAUGHTER", "NO_PARENT")}
for p in sorted(PRED.glob("*.geff")):
    gp = GT / f"{p.stem}.geff"
    if not gp.exists(): continue
    g = load(p)
    pos, tt = {}, {}
    for r in g.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        tt[i] = int(r["t"])
    outd, ind = {}, {}
    for r in g.edge_attrs().iter_rows(named=True):
        s, d = int(r["source_id"]), int(r["target_id"])
        outd[s] = outd.get(s, 0) + 1; ind[d] = ind.get(d, 0) + 1
    by_t = {}
    for i, t in tt.items(): by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}
    def match(q, t):
        ent = trees.get(t)
        if ent is None: return None
        tr, ids = ent; dd, j = tr.query(q)
        return ids[int(j)] if dd <= TOL else None

    gt = load(gp)
    gpos, gtt = {}, {}
    for r in gt.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gtt[i] = int(r["t"])
    gout = {}
    for r in gt.edge_attrs().iter_rows(named=True):
        gout.setdefault(int(r["source_id"]), []).append(int(r["target_id"]))

    per = {k: 0 for k in tot}
    for s, kids in gout.items():
        if len(kids) < 2 or s not in gpos: continue          # a GT division
        ms = match(gpos[s], gtt[s])
        if ms is None: per["NO_PARENT"] += 1; continue
        md = [match(gpos[k], gtt[k]) for k in kids if k in gpos]
        if any(x is None for x in md) or len(md) < 2:
            per["NO_DAUGHTER"] += 1; continue
        if outd.get(ms, 0) >= 2:
            per["PRESENT"] += 1; continue
        # the generator only proposes daughters with NO incoming edge
        if any(ind.get(x, 0) > 0 for x in md):
            per["CONTENDED"] += 1
        else:
            per["FREE"] += 1
    for k in per: tot[k] += per[k]
    if sum(per.values()):
        print(f"  {p.stem}: " + "  ".join(f"{k} {v}" for k, v in per.items() if v))

n = sum(tot.values())
print(f"\n  GT divisions with a detected parent: {n}\n")
for k in ("PRESENT", "FREE", "CONTENDED", "NO_DAUGHTER", "NO_PARENT"):
    print(f"    {k:<12} {tot[k]:>4}   {tot[k]/max(n,1):>6.1%}")
miss = n - tot["PRESENT"]
if miss:
    print(f"\n  of the {miss} MISSED divisions:")
    print(f"    reachable by the gates (FREE)      {tot['FREE']/miss:>6.1%}")
    print(f"    STRUCTURALLY EXCLUDED (CONTENDED)  {tot['CONTENDED']/miss:>6.1%}")
    print(f"    undetectable (NO_DAUGHTER)         {tot['NO_DAUGHTER']/miss:>6.1%}")
