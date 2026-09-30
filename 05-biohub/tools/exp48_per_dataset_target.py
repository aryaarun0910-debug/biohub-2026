#!/usr/bin/env python3
"""EXP-48 -- prune to a per-dataset node-ratio TARGET, not a global minimum track length.

EXP-47 showed OUTPUT_MIN_TRACK_LEN is worth +0.0078..+0.0095 and that the whole gain is
multiplier: N_pred falls ~8% while edge Jaccard rises slightly. Short components are tracking
fragments and carry no annotated edges.

But a single global minlen is a blunt instrument, because the multiplier is applied PER DATASET:

    adj_i = J_i * (1 - 0.1 * (N_pred_i - n_total_i)/n_total_i)

and n_total_i is known exactly for all four test movies. Their ratios under minlen 6 differ
enormously -- 44b6_0b24845f sits at -0.368 while 6bba_05db0fb1 sits at +0.007 -- so one minlen
prunes them to wildly different places. Targeting a ratio directly should dominate.

Two questions:
  1. what target ratio maximises the score, and where does edge Jaccard start to break?
  2. does ordering by component TIME-SPAN or by size prune more nodes before the cliff?

The second matters because the binding constraint is not how many nodes we remove, it is how
many ANNOTATED edges we destroy on the way. A better ordering buys more multiplier per edge lost.
"""
import sys, warnings
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td

sys.path.insert(0, "src"); sys.path.insert(0, "tools"); sys.path.insert(0, "reference/royerlab-baseline/src")
warnings.filterwarnings("ignore")
from geff import GeffMetadata
from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall, summarise
from biohub.contracts import SCALE

GT = Path("data/train_geff")
PRED = Path("work/repro_out/tracking_repo/predictions/unknown/unet_transformer/split_0")
STEMS = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]


def load(p):
    g = td.graph.IndexedRXGraph.from_geff(p); return g[0] if isinstance(g, tuple) else g


cache = []
for s in STEMS:
    g = load(PRED / f"{s}.geff")
    nbi = {}
    for r in g.node_attrs().iter_rows(named=True):
        nbi[int(r["node_id"])] = {"t": int(r["t"]), "z": r["z"], "y": r["y"], "x": r["x"]}
    edges = [(int(r["source_id"]), int(r["target_id"])) for r in g.edge_attrs().iter_rows(named=True)]
    par = {n: n for n in nbi}
    def find(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb: par[ra] = rb
    members = {}
    for n in nbi: members.setdefault(find(n), []).append(n)
    comps = list(members.values())
    size = {id(c): len(c) for c in comps}
    span = {id(c): max(nbi[m]["t"] for m in c) - min(nbi[m]["t"] for m in c) + 1 for c in comps}
    gt = load(GT / f"{s}.geff")
    est = float((GeffMetadata.read(GT / f"{s}.geff").extra or {})["estimated_number_of_nodes"])
    cache.append((s, nbi, edges, comps, size, span, gt, est))


def build(nbi, keep, edges):
    G = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"): G.add_node_attr_key(k, pl.Float64, -999999.0)
    ks = [k for k in nbi if k in keep]
    ids = G.bulk_add_nodes([{"t": int(nbi[k]["t"]), "z": float(nbi[k]["z"]),
                             "y": float(nbi[k]["y"]), "x": float(nbi[k]["x"])} for k in ks])
    m = dict(zip(ks, ids))
    ee = [{"source_id": m[a], "target_id": m[b]} for a, b in edges if a in m and b in m]
    if ee: G.bulk_add_edges(ee)
    return G


def run(target, key):
    rows, info = [], []
    for s, nbi, edges, comps, size, span, gt, est in cache:
        rank = size if key == "size" else span
        # drop WHOLE components, smallest/shortest first, until the target is met.
        # An earlier version grouped by rank VALUE rather than by component, which removed every
        # component of a given size at once and made size and span produce identical output.
        order = sorted(comps, key=lambda c: (rank[id(c)], min(c)))
        want = int(est * (1.0 + target))
        drop = max(0, len(nbi) - want)
        keep = set(nbi); removed = 0
        for c in order:
            if removed >= drop: break
            for m in c: keep.discard(m)
            removed += len(c)
        e = [(a, b) for a, b in edges if a in keep and b in keep]
        pred = build(nbi, keep, e)
        er = evaluate(pred, gt, scale=tuple(SCALE), max_distance=7.0)
        rows.append(per_sample_metrics(er, est, node_recall(pred, gt) if e else 0.0))
        info.append((er.num_pred_nodes, est))
    return rows, info


print(f"\n  {'order':<7}{'target':>8}{'adj':>9}{'SCORE':>9}{'delta':>9}   per-stem edgeJ")
print("  " + "-" * 78)
base = None
for key in ("size", "span"):
    for target in (None, 0.00, -0.05, -0.10, -0.15, -0.20, -0.30):
        if target is None:
            if key == "span": continue
            rows = []
            for s, nbi, edges, comps, size, span, gt, est in cache:
                pred = build(nbi, set(nbi), edges)
                er = evaluate(pred, gt, scale=tuple(SCALE), max_distance=7.0)
                rows.append(per_sample_metrics(er, est, node_recall(pred, gt)))
            S = summarise(rows); base = S["score"]
            ej = " ".join(f"{r['edge_jaccard']:.3f}" for r in rows)
            print(f"  {'none':<7}{'-':>8}{S['adj_edge_jaccard']:>9.5f}{S['score']:>9.5f}"
                  f"{0.0:>+9.5f}   {ej}")
            continue
        rows, info = run(target, key)
        S = summarise(rows)
        ej = " ".join(f"{r['edge_jaccard']:.3f}" for r in rows)
        print(f"  {key:<7}{target:>+8.2f}{S['adj_edge_jaccard']:>9.5f}{S['score']:>9.5f}"
              f"{S['score']-base:>+9.5f}   {ej}")
print("\n  per-stem order: 44b6_0113de3b  44b6_0b24845f  6bba_05b6850b  6bba_05db0fb1")
print("  weights:        2.3%           2.3%           38.6%          56.9%")
