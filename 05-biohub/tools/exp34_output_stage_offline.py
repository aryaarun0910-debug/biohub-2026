#!/usr/bin/env python3
"""EXP-34 -- evaluate the OUTPUT stage offline on 199 datasets, not their 8 stems.

kernels/v-outgrid is testing OUTPUT_MIN_TRACK_LEN and OUTPUT_EDGE_MAX_UM against their in-kernel
proxy, which uses 8 held-out stems carrying 12 divisions and over-reads the leaderboard by
+0.0041. We hold 199 graphs with 151 divisions, so the same configs can be judged on 25x the data
while the grid runs -- and unlike the rest of their post-processing, these two operations are pure
graph surgery we can reproduce exactly:

    OUTPUT_MIN_TRACK_LEN   drop connected components with fewer than N nodes
    OUTPUT_EDGE_MAX_UM     drop edges longer than X micrometres

This is also the only untested route into the LARGEST pool. The score is
edgeJ * (1.1 - 0.1*N_pred/n_total) + 0.1*divJ, their multiplier sits at 1.0015 of a possible
1.1000, and EXP-22 closed only SPATIAL targeting of N_pred -- never length-based pruning, where
removing a spurious short track drops its false edges AND lowers N_pred at the same time.
"""
import sys
from pathlib import Path
import numpy as np, tracksdata as td
from scipy.spatial import cKDTree

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
from geff import GeffMetadata
import div_sweep as D
from div_ci import paired_delta

GT = Path("data/train_geff"); GRAPHS = Path("work/train_graphs"); TOL = 7.0


def components(nodes, edges):
    parent = {n: n for n in nodes}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for s, d in edges:
        ra, rb = find(s), find(d)
        if ra != rb: parent[ra] = rb
    size = {}
    for n in nodes:
        r = find(n); size[r] = size.get(r, 0) + 1
    return {n: size[find(n)] for n in nodes}


def apply_output(nodes, pos, edges, min_len, edge_max):
    e = [(s, d) for s, d in edges
         if edge_max is None or np.linalg.norm(pos[d] - pos[s]) <= edge_max]
    if min_len and min_len > 1:
        sz = components(set(nodes), e)
        keep = {n for n in nodes if sz.get(n, 1) >= min_len}
        e = [(s, d) for s, d in e if s in keep and d in keep]
    else:
        keep = set(nodes)
    return keep, e


CFGS = [("theirs (6, 14.0)", 6, 14.0), ("minlen 3", 3, 14.0), ("minlen 4", 4, 14.0),
        ("minlen 8", 8, 14.0), ("minlen 10", 10, 14.0), ("edgemax 12", 6, 12.0),
        ("edgemax 10", 6, 10.0), ("edgemax 8", 6, 8.0), ("trim (8, 10.0)", 8, 10.0)]

cache = []
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
    edges = [(int(r["source_id"]), int(r["target_id"])) for r in e.iter_rows(named=True)]
    gg = td.graph.IndexedRXGraph.from_geff(gp); gg = gg[0] if isinstance(gg, tuple) else gg
    gpos, gtt = {}, {}
    for r in gg.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gtt[i] = int(r["t"])
    ge = [(int(r["source_id"]), int(r["target_id"])) for r in gg.edge_attrs().iter_rows(named=True)]
    est = (GeffMetadata.read(gp).extra or {}).get("estimated_number_of_nodes")
    if not ge or not est:
        continue
    by_t = {}
    for i, t in tt.items(): by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}
    def match(pt, t):
        ent = trees.get(t)
        if ent is None: return None
        tr_, ids = ent; d, j = tr_.query(pt)
        return ids[int(j)] if d <= TOL else None
    gt_pairs, gt_nodes = set(), set()
    for s, d in ge:
        if s in gpos and d in gpos:
            ms, md = match(gpos[s], gtt[s]), match(gpos[d], gtt[d])
            gt_nodes.update(x for x in (ms, md) if x is not None)
            if ms is not None and md is not None: gt_pairs.add((ms, md))
    cache.append((list(pos), pos, edges, gt_pairs, gt_nodes, float(est), p.stem[:4]))

print(f"  {len(cache)} datasets\n")
print(f"  {'config':<20} {'N_pred':>10} {'ratio':>7} {'mult':>7} {'edgeJ':>8} {'adj':>8} {'delta':>8}")
res, base_rows, base_adj = {}, None, None
for name, ml, em in CFGS:
    rows = []
    tot_np = tot_nt = 0
    for nodes, pos, edges, gtp, gtn, est, emb in cache:
        keep, e = apply_output(nodes, pos, edges, ml, em)
        s = set(e)
        tp = len(gtp & s)
        # THE SCORER ONLY CHARGES AN EDGE AS A FALSE POSITIVE IF AN ENDPOINT MATCHED A GT NODE.
        # Ground truth annotates 2.82% of cells, so counting all 4.1M predicted edges as FP gives
        # edgeJ 0.03 instead of 0.94. This is the third time today the sparse-annotation trap has
        # bitten (the 98.86% swap base rate, and their own precision_sparse metric).
        chargeable = sum(1 for a, b in s if a in gtn or b in gtn)
        fp = chargeable - tp; fn = len(gtp) - tp
        rows.append((tp, fp, fn)); tot_np += len(keep); tot_nt += est
    a = np.array(rows); TP, FP, FN = a.sum(0)
    eJ = TP / max(TP + FP + FN, 1)
    ratio = tot_np / tot_nt
    mult = max(0.0, 1.1 - 0.1 * ratio)
    adj = eJ * mult
    if base_rows is None: base_rows, base_adj = rows, adj
    res[name] = rows
    print(f"  {name:<20} {tot_np:>10,} {ratio:>7.3f} {mult:>7.4f} {eJ:>8.4f} {adj:>8.4f} {adj-base_adj:>+8.4f}")

print("\n  === PAIRED bootstrap on edge Jaccard, same 199 datasets ===")
for name, rows in res.items():
    if name.startswith("theirs"): continue
    d, lo, hi, pw = paired_delta(base_rows, rows)
    print(f"  {name:<20} edgeJ delta {d:+.4f} [{lo:+.4f},{hi:+.4f}]  P(not better) {pw:.3f}")
print("\n  NB the leaderboard resolves 0.001; anything smaller is unobservable there.")
