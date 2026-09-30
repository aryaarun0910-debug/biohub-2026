#!/usr/bin/env python3
"""EXP-40 -- recover fragmented edges using a motion field built from the links we already trust.

Their pipeline's edge failures split as 17.1 fragmented per dataset against 9.9 lost to
detection (work/repro_out/validator_results.csv). Fragmentation is the largest identified
failure mode and EXP-37 showed detection is not the constraint (edge ceiling 0.9966).

A fragmented edge has BOTH endpoints detected on CONSECUTIVE frames and the linker simply
declined. So the candidate set is small and exact:

    sinks   = nodes at t   with no outgoing edge
    sources = nodes at t+1 with no incoming edge

The idea worth testing is that the accepted edges are themselves a dense sample of the local
tissue motion. For every accepted edge i->j take v = x_j - x_i; interpolate those spatially to
get v(x); then an orphan sink's successor should sit near x_i + v(x_i). If a region moves 6 um
together, raw nearest-neighbour confuses neighbours but flow-compensated distance does not.

Break-even edge precision is 48.1%. We do NOT need recall -- recovering 20-40% of the
fragmented pool at high precision is worth more than solving every ambiguous case.

SUBSTRATE: their validator output graphs, NOT work/train_graphs. Three experiments today were
misread because our cached graphs over-predict (median ratio +0.109) and carry ~187 native
linker forks per dataset where their pipeline carries almost none.

PRECISION IS MEASURED ONLY ON JUDGEABLE SINKS -- those matching a GT node whose GT successor
also matched a predicted node. Ground truth annotates 2.82% of cells; scoring a stitch against
an unannotated neighbour is the sparse-annotation trap, which has bitten five times.
"""
import sys, warnings
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
warnings.filterwarnings("ignore")
from geff import GeffMetadata
from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall
from biohub.contracts import SCALE
import div_sweep as D

PRED = Path("work/repro_out/tracking_repo/predictions/unknown/unet_transformer_val/split_0")
GT = Path("data/train_geff")
TOL = 7.0


def load(p):
    g = td.graph.IndexedRXGraph.from_geff(p); return g[0] if isinstance(g, tuple) else g


cache = []
for p in sorted(PRED.glob("*.geff")):
    gp = GT / f"{p.stem}.geff"
    if not gp.exists(): continue
    g = load(p)
    pos, tt = {}, {}
    for r in g.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        tt[i] = int(r["t"])
    edges = [(int(r["source_id"]), int(r["target_id"])) for r in g.edge_attrs().iter_rows(named=True)]

    gt = load(gp)
    gpos, gtt = {}, {}
    for r in gt.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gtt[i] = int(r["t"])
    ge = [(int(r["source_id"]), int(r["target_id"])) for r in gt.edge_attrs().iter_rows(named=True)]
    est = (GeffMetadata.read(gp).extra or {}).get("estimated_number_of_nodes")

    by_t = {}
    for i, t in tt.items(): by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}
    def match(q, t):
        ent = trees.get(t)
        if ent is None: return None
        tr, ids = ent; d, j = tr.query(q)
        return ids[int(j)] if d <= TOL else None
    # GT edge in PREDICTED node ids -- the only pairs a stitch can be judged against
    gt_pair = {}
    for s, d in ge:
        if s not in gpos or d not in gpos: continue
        ms, md = match(gpos[s], gtt[s]), match(gpos[d], gtt[d])
        if ms is not None and md is not None: gt_pair.setdefault(ms, set()).add(md)
    cache.append(dict(stem=p.stem, emb=p.stem[:4], pos=pos, t=tt, edges=edges,
                      gt=gt, est=float(est) if est else None, gt_pair=gt_pair, by_t=by_t))
    print(f"  {p.stem}: {len(pos):,} nodes  {len(edges):,} edges  judgeable GT pairs {len(gt_pair)}", flush=True)


def stitch(c, max_um, w_track, mutual, flow_k=12, flow_sigma=25.0):
    """Return proposed (sink, source) edges using flow-compensated prediction + Hungarian."""
    pos, tt, edges, by_t = c["pos"], c["t"], c["edges"], c["by_t"]
    out_deg, in_deg, pred_of = {}, {}, {}
    for s, d in edges:
        out_deg[s] = out_deg.get(s, 0) + 1; in_deg[d] = in_deg.get(d, 0) + 1
        pred_of[d] = s
    # trusted displacements per source frame
    disp_by_t = {}
    for s, d in edges:
        if tt.get(d) == tt.get(s, -9) + 1:
            disp_by_t.setdefault(tt[s], []).append((pos[s], pos[d] - pos[s]))
    proposed = []
    for t in sorted(by_t):
        nxt = by_t.get(t + 1)
        if not nxt: continue
        sinks = [i for i in by_t[t] if out_deg.get(i, 0) == 0]
        srcs = [j for j in nxt if in_deg.get(j, 0) == 0]
        if not sinks or not srcs: continue
        dd = disp_by_t.get(t, [])
        if len(dd) >= 3:
            P = np.stack([a for a, _ in dd]); V = np.stack([b for _, b in dd])
            ftree = cKDTree(P)
        else:
            ftree = None
        stree = cKDTree(np.stack([pos[j] for j in srcs]))
        cost = np.full((len(sinks), len(srcs)), 1e6)
        for a, i in enumerate(sinks):
            v_flow = np.zeros(3)
            if ftree is not None:
                k = min(flow_k, len(dd))
                dist, idx = ftree.query(pos[i], k=k)
                dist = np.atleast_1d(dist); idx = np.atleast_1d(idx)
                w = np.exp(-(dist ** 2) / (2 * flow_sigma ** 2))
                if w.sum() > 1e-9: v_flow = (V[idx] * w[:, None]).sum(0) / w.sum()
            v_trk = np.zeros(3); has_trk = False
            pr = pred_of.get(i)
            if pr is not None and tt.get(pr) == t - 1:
                v_trk = pos[i] - pos[pr]; has_trk = True
            v = w_track * v_trk + (1 - w_track) * v_flow if has_trk else v_flow
            xhat = pos[i] + v
            for b in stree.query_ball_point(xhat, max_um):
                cost[a, b] = float(np.linalg.norm(pos[srcs[b]] - xhat))
        if not np.isfinite(cost).any() or (cost < 1e6).sum() == 0: continue
        if mutual:
            keep = np.zeros_like(cost, bool)
            rb = cost.argmin(1); cb = cost.argmin(0)
            for a in range(len(sinks)):
                if cost[a, rb[a]] < 1e6 and cb[rb[a]] == a: keep[a, rb[a]] = True
            cost = np.where(keep, cost, 1e6)
        ri, ci = linear_sum_assignment(cost)
        for a, b in zip(ri, ci):
            if cost[a, b] < 1e6: proposed.append((sinks[a], srcs[b], float(cost[a, b])))
    return proposed


print("\n  === PRECISION on judgeable sinks (break-even 48.1%) ===", flush=True)
print(f"  {'config':<28} {'proposed':>9} {'judgeable':>10} {'correct':>8} {'PRECISION':>10}", flush=True)
GRID = [(r, wt, mu) for r in (4.0, 6.0, 9.0, 12.0) for wt in (0.0, 0.5) for mu in (True, False)]
best = None
for r, wt, mu in GRID:
    P = J = C = 0
    for c in cache:
        gp_ = c["gt_pair"]
        for i, j, _ in stitch(c, r, wt, mu):
            P += 1
            if i in gp_:                      # only judgeable if the sink's GT successor is known
                J += 1; C += (j in gp_[i])
    prec = C / J if J else float("nan")
    tag = f"r{r:g} wt{wt:g} {'mutual' if mu else 'hung'}"
    print(f"  {tag:<28} {P:>9,} {J:>10} {C:>8} {prec:>9.1%}", flush=True)
    if J >= 20 and (best is None or prec > best[0]): best = (prec, r, wt, mu)
print(f"\n  best judgeable precision: {best[0]:.1%} at r={best[1]:g} wt={best[2]:g} "
      f"{'mutual' if best[3] else 'hungarian'}", flush=True)
