#!/usr/bin/env python3
"""EXP-43 -- where the linker disagrees with nearest-neighbour, who is right?

EXP-42: when a GT edge is missing, the true successor is the nearest candidate 37% of the time
and inside the top 2 80% of the time. Flow compensation does not reorder anything. So this is a
two-way discrimination the linker is losing, not a search problem.

The linker is a global optimiser, so it deliberately departs from nearest-neighbour to satisfy
constraints. The rewire question is therefore exactly:

    on judgeable nodes where the linker's target != the nearest candidate,
    how often is the GT successor the LINKER's pick, and how often the NEAREST?

This is the whole rewire route in one number, and it needs no classifier to measure. If nearest
wins the head-to-head, a blind "snap to nearest" rule is +EV and can be tested immediately. If
the linker wins, the pool is closed and no feature set over these candidates will open it,
because the linker already has more information than we would give a reranker.

Judgeable = the node matched a GT node, that GT node has a successor, and the successor was
itself detected. Everything else is unjudgeable under 2.82% annotation.
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

agree_ok = agree_bad = 0          # linker target == nearest
lk_win = nn_win = both_bad = 0    # they differ
for p in sorted(PRED.glob("*.geff")):
    gp = GT / f"{p.stem}.geff"
    if not gp.exists(): continue
    g = load(p)
    pos, tt = {}, {}
    for r in g.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        tt[i] = int(r["t"])
    out = {}
    for r in g.edge_attrs().iter_rows(named=True):
        out.setdefault(int(r["source_id"]), []).append(int(r["target_id"]))
    by_t = {}
    for i, t in tt.items(): by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}
    def match(q, t):
        ent = trees.get(t)
        if ent is None: return None
        tr, ids = ent; d, j = tr.query(q)
        return ids[int(j)] if d <= TOL else None

    gt = load(gp)
    gpos, gtt = {}, {}
    for r in gt.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gtt[i] = int(r["t"])
    for r in gt.edge_attrs().iter_rows(named=True):
        s, d = int(r["source_id"]), int(r["target_id"])
        if s not in gpos or d not in gpos: continue
        ms, md = match(gpos[s], gtt[s]), match(gpos[d], gtt[d])
        if ms is None or md is None: continue
        tgts = out.get(ms, [])
        if len(tgts) != 1: continue               # skip forks; a division is a different decision
        cur = tgts[0]
        ent = trees.get(tt[ms] + 1)
        if ent is None: continue
        tr, ids = ent
        nn = ids[int(tr.query(pos[ms])[1])]
        if cur == nn:
            agree_ok += (cur == md); agree_bad += (cur != md)
        else:
            if cur == md: lk_win += 1
            elif nn == md: nn_win += 1
            else: both_bad += 1

tot = agree_ok + agree_bad + lk_win + nn_win + both_bad
print(f"\n  judgeable single-successor nodes: {tot:,}\n")
print(f"  linker AGREES with nearest-neighbour : {agree_ok+agree_bad:,}"
      f"   correct {agree_ok:,}  wrong {agree_bad:,}")
print(f"  linker DIFFERS from nearest-neighbour: {lk_win+nn_win+both_bad:,}")
print(f"      linker right, nearest wrong : {lk_win:>5}")
print(f"      nearest right, linker wrong : {nn_win:>5}   <- the rewire pool")
print(f"      both wrong                  : {both_bad:>5}")
d = lk_win + nn_win + both_bad
if d:
    print(f"\n  snapping ALL disagreements to nearest: +{nn_win} TP / -{lk_win} TP"
          f"  = net {nn_win-lk_win:+d} edges")
    print(f"  head-to-head precision of 'nearest': {nn_win/max(lk_win+nn_win,1):.1%}"
          f"   (break-even 48.1%)")
