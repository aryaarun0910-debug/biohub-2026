#!/usr/bin/env python3
"""EXP-37 -- is the remaining gap DETECTION or LINKING?

All four post-processing levers are closed (EXP-36). The only remaining route is
retraining the detector, which costs real GPU hours. Before spending them, measure
whether detection is actually the binding constraint.

The decisive quantity is the EDGE CEILING: the fraction of GT edges for which BOTH
endpoints have a predicted node within TOL. An edge missing an endpoint is
structurally unrecoverable -- no linker, however good, can produce it. So

    edge_ceiling  =  max achievable edge recall
    edgeJ_actual / edge_ceiling  =  how much of the reachable set we already get

If the ceiling is ~0.99 the detector is fine and linking is the problem (and linking
is closed, so we are done). If the ceiling is ~0.93 then detection is worth the GPU.

NOTE on regime: this does NOT ask "what if we predicted only annotated cells" -- that
is trivially the 1.1972 max and answers nothing, because the detector cannot know
which cells a human chose to label. It asks only what the CURRENT detector makes
reachable.
"""
import sys
from pathlib import Path
import numpy as np, tracksdata as td
from scipy.spatial import cKDTree

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
import div_sweep as D

GT = Path("data/train_geff"); GRAPHS = Path("work/train_graphs"); TOL = 7.0

def load(p):
    g = td.graph.IndexedRXGraph.from_geff(p); return g[0] if isinstance(g, tuple) else g

tot_n = tot_nm = tot_e = tot_er = 0
per = []
files = sorted(GRAPHS.rglob("*.geff"))
A = [p for p in files if p.stem.startswith("44b6")][:15]
B = [p for p in files if p.stem.startswith("6bba")][:15]
for p in A + B:
    gp = GT / f"{p.stem}.geff"
    if not gp.exists(): continue
    g = load(p)
    pos, tt = {}, {}
    for r in g.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM; tt[i] = int(r["t"])
    by_t = {}
    for i, t in tt.items(): by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}

    gg = load(gp)
    gpos, gtt = {}, {}
    for r in gg.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM; gtt[i] = int(r["t"])
    det = {}
    for i, q in gpos.items():
        ent = trees.get(gtt[i])
        if ent is None: det[i] = False; continue
        tr, ids = ent; d, _ = tr.query(q); det[i] = bool(d <= TOL)
    ge = [(int(r["source_id"]), int(r["target_id"])) for r in gg.edge_attrs().iter_rows(named=True)]
    reach = sum(1 for s, d in ge if det.get(s) and det.get(d))
    n, nm, e = len(gpos), sum(det.values()), len(ge)
    if e == 0: continue
    tot_n += n; tot_nm += nm; tot_e += e; tot_er += reach
    per.append((p.stem[:4], n, nm / n, e, reach / e))

print(f"\n  GT nodes {tot_n:,}   detected {tot_nm:,}   NODE RECALL {tot_nm/tot_n:.4f}")
print(f"  GT edges {tot_e:,}   reachable {tot_er:,}   EDGE CEILING {tot_er/tot_e:.4f}")
print(f"  structurally unrecoverable edges: {1-tot_er/tot_e:.2%}\n")
for emb in ("44b6", "6bba"):
    rows = [r for r in per if r[0] == emb]
    if not rows: continue
    n = sum(r[1] for r in rows); e = sum(r[3] for r in rows)
    nr = sum(r[1]*r[2] for r in rows)/n; er = sum(r[3]*r[4] for r in rows)/e
    print(f"    {emb}: node recall {nr:.4f}   edge ceiling {er:.4f}   ({len(rows)} datasets, {e:,} edges)")
print("\n  reading: edge ceiling is the MAX edge Jaccard recall any linker can reach")
print("  on this detector's output. Compare to our measured edgeJ to split the gap.")
