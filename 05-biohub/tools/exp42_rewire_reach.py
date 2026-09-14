#!/usr/bin/env python3
"""EXP-42 -- is the correct successor even REACHABLE when the linker picked the wrong one?

EXP-41: only 5.5% of missing GT edges have both endpoints free. 62.3% need a REWIRE -- the
source already points somewhere else, or the target already has a parent.

The economics of a rewire are much better than a fresh stitch. If node i matched a GT node and
its current outgoing edge does NOT go to the GT successor, that edge is ALREADY a false positive
(i is matched, so the edge is chargeable). Replacing it therefore costs nothing on FP and a
correct replacement wins three ways at once: TP+1, FP-1, FN-1.

But the decision has to be made WITHOUT ground truth. So the question this asks is narrow and
prior to any classifier:

    when the linker's choice is wrong, is the TRUE successor ranked highly by flow-compensated
    distance among the available candidates?

If the true successor is usually rank 1 or 2, a learned rewire rule has something to find. If it
is buried at rank 20, no classifier over these features will help and the pool is closed.

Rank is computed over ALL nodes at t+1 within radius, not just free ones, since a rewire may
legitimately steal a target. Reported only on judgeable sinks -- GT-matched, GT successor known
and itself detected -- because 2.82% annotation makes anything else unjudgeable.
"""
import sys, warnings
from pathlib import Path
import numpy as np, tracksdata as td
from scipy.spatial import cKDTree

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
warnings.filterwarnings("ignore")
import div_sweep as D

PRED = Path("work/repro_out/tracking_repo/predictions/unknown/unet_transformer_val/split_0")
GT = Path("data/train_geff"); TOL = 7.0; RADIUS = 15.0; FLOW_K, FLOW_SIG = 12, 25.0

def load(p):
    g = td.graph.IndexedRXGraph.from_geff(p); return g[0] if isinstance(g, tuple) else g

ranks_raw, ranks_flow, cases = [], [], []
for p in sorted(PRED.glob("*.geff")):
    gp = GT / f"{p.stem}.geff"
    if not gp.exists(): continue
    g = load(p)
    pos, tt = {}, {}
    for r in g.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        tt[i] = int(r["t"])
    edges = [(int(r["source_id"]), int(r["target_id"])) for r in g.edge_attrs().iter_rows(named=True)]
    eset = set(edges); cur_tgt, pred_of = {}, {}
    for s, d in edges: cur_tgt.setdefault(s, []).append(d); pred_of[d] = s
    disp_by_t = {}
    for s, d in edges:
        if tt.get(d) == tt.get(s, -9) + 1: disp_by_t.setdefault(tt[s], []).append((pos[s], pos[d]-pos[s]))
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
    ge = [(int(r["source_id"]), int(r["target_id"])) for r in gt.edge_attrs().iter_rows(named=True)]

    for s, d in ge:
        if s not in gpos or d not in gpos: continue
        ms, md = match(gpos[s], gtt[s]), match(gpos[d], gtt[d])
        if ms is None or md is None: continue
        if (ms, md) in eset: continue                 # already correct
        t = tt[ms]
        ent = trees.get(t + 1)
        if ent is None: continue
        tr, ids = ent
        cand = [ids[k] for k in tr.query_ball_point(pos[ms], RADIUS)]
        if md not in cand: cases.append("OUT_OF_RADIUS"); continue
        dd = disp_by_t.get(t, [])
        v = np.zeros(3)
        if len(dd) >= 3:
            P = np.stack([a for a, _ in dd]); V = np.stack([b for _, b in dd])
            ft = cKDTree(P); k = min(FLOW_K, len(dd))
            dist, idx = ft.query(pos[ms], k=k); dist = np.atleast_1d(dist); idx = np.atleast_1d(idx)
            w = np.exp(-(dist**2)/(2*FLOW_SIG**2))
            if w.sum() > 1e-9: v = (V[idx]*w[:, None]).sum(0)/w.sum()
        xhat = pos[ms] + v
        raw = sorted(cand, key=lambda c: np.linalg.norm(pos[c]-pos[ms]))
        flw = sorted(cand, key=lambda c: np.linalg.norm(pos[c]-xhat))
        ranks_raw.append(raw.index(md)+1); ranks_flow.append(flw.index(md)+1)
        cases.append("IN")

n = len(ranks_flow)
print(f"\n  judgeable missing edges with the true successor inside {RADIUS:g} um: {n}")
print(f"  out of radius entirely: {cases.count('OUT_OF_RADIUS')}\n")
if n:
    rr, rf = np.array(ranks_raw), np.array(ranks_flow)
    print(f"  {'rank of the TRUE successor':<34}{'raw dist':>10}{'flow-comp':>12}")
    for k in (1, 2, 3, 5, 10):
        print(f"    within top-{k:<2}                        {(rr<=k).mean():>9.1%}{(rf<=k).mean():>12.1%}")
    print(f"    median rank                       {np.median(rr):>9.0f}{np.median(rf):>12.0f}")
    print(f"\n  flow beats raw on {(rf<rr).sum()} cases, loses on {(rf>rr).sum()}, ties {(rf==rr).sum()}")
