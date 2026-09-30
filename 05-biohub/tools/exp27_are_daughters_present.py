#!/usr/bin/env python3
"""EXP-27 -- do the missing divisions even EXIST as candidates?

EXP-26 closed selection: the gate-rejected region cannot be separated. But that assumed the true
division is IN the candidate pool and merely buried. A candidate only exists if BOTH daughters
were detected AND the second one was left unclaimed by the linker.

So for every ground-truth division, ask which of three things happened:

  DAUGHTER_MISSING   a daughter has no predicted node within the scorer's 7um tolerance
                     -> detection failure; no selector can ever recover it
  DAUGHTER_CLAIMED   both daughters exist but the second is already some other parent's target
                     -> linker contention; the candidate is never offered
  CANDIDATE_EXISTS   both present, second unclaimed -> the candidate exists and selection owns it

This decides where the remaining effort belongs -- and whether replacing DeepCenter could help
at all, since DeepCenter is only the veto and the temporal UNet does the detecting.
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
verdict = Counter(); per_emb = {}

for p in sorted(GRAPHS.rglob("*.geff")):
    gp = GT / f"{p.stem}.geff"
    if not gp.exists():
        continue
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs(); e = g.edge_attrs()
    pos, tt = {}, {}
    for r in n.iter_rows(named=True):
        i = int(r["node_id"])
        pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        tt[i] = int(r["t"])
    claimed = {int(r["target_id"]) for r in e.iter_rows(named=True)}
    out_deg = Counter(int(r["source_id"]) for r in e.iter_rows(named=True))

    gg = td.graph.IndexedRXGraph.from_geff(gp); gg = gg[0] if isinstance(gg, tuple) else gg
    gpos, gt_t = {}, {}
    for r in gg.node_attrs().iter_rows(named=True):
        i = int(r["node_id"])
        gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gt_t[i] = int(r["t"])
    gout = {}
    for r in gg.edge_attrs().iter_rows(named=True):
        gout.setdefault(int(r["source_id"]), []).append(int(r["target_id"]))
    divs = [(s, k) for s, k in gout.items() if len(k) == 2]
    if not divs:
        continue

    # index predicted nodes per frame for a 7um lookup
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
    for s, kk in divs:
        if s not in gpos or kk[0] not in gpos or kk[1] not in gpos:
            continue
        ms = match(gpos[s], gt_t[s])
        m0 = match(gpos[kk[0]], gt_t[kk[0]])
        m1 = match(gpos[kk[1]], gt_t[kk[1]])
        if ms is None:
            v = "PARENT_MISSING"
        elif m0 is None or m1 is None:
            v = "DAUGHTER_MISSING"
        else:
            # which daughter is the linker's existing child, which is the extra one?
            extra = [m for m in (m0, m1) if m in claimed]
            free = [m for m in (m0, m1) if m not in claimed]
            if len(free) >= 1:
                v = "CANDIDATE_EXISTS"
            else:
                v = "DAUGHTER_CLAIMED"
        verdict[v] += 1; per_emb[emb][v] += 1

tot = sum(verdict.values())
print(f"\n  GROUND-TRUTH DIVISIONS EXAMINED: {tot}\n")
for v, c in verdict.most_common():
    print(f"    {v:<20} {c:>4}  {c/tot:>6.1%}")
print("\n  per embryo:")
for emb, c in sorted(per_emb.items()):
    s = sum(c.values())
    print(f"    {emb} (n={s}): " + "  ".join(f"{k}={v}" for k, v in c.most_common()))
print("\n  DAUGHTER_MISSING is a DETECTION failure -- unreachable by any selector.")
print("  DAUGHTER_CLAIMED is LINKER contention -- the candidate is never offered.")
print("  CANDIDATE_EXISTS is what EXP-26 showed is not separable.")
