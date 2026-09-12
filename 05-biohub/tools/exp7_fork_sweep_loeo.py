#!/usr/bin/env python3
"""EXP-7 — sweep any two Config thresholds, LEAVE-ONE-EMBRYO-OUT, against the REAL score.

    tools/exp7_fork_sweep_loeo.py fork_parent_um=8,12 fork_sister_um=14,18

The pooled corpus is 125/151 divisions from 6bba, so a pooled sweep just fits 6bba. Train and
test are embryo-disjoint, so the honest protocol is: pick on one embryo, report on the other,
both directions, take the MIN. A setting that only works on the embryo it was chosen on is
worth nothing on a hidden one.

The objective is the COMPETITION SCORE (adj_edge_jaccard + 0.1*division_jaccard), not divJ
alone. Sweeping divJ alone is only valid for knobs alone that cannot move the edge term -- for
anything upstream of resolve it would report a divJ win that is a net loss, since the edge term
carries 10x the weight. All four stages run, because repair rewrites edges too.
"""
import sys
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td
sys.path.insert(0, "src")
sys.path.insert(0, "reference/royerlab-baseline/src")
sys.path.insert(0, "reference/royerlab-baseline")
from geff import GeffMetadata
from tracking_cellmot.metrics import evaluate, per_sample_metrics, summarise, node_recall
from biohub.contracts import Graph, Config, SCALE
from biohub.edges import score_edges
from biohub.refine import refine
from biohub.resolve import resolve
from biohub.repair import repair

GT = Path("data/train_geff")


def _load(p):
    g = td.graph.IndexedRXGraph.from_geff(p)
    return g[0] if isinstance(g, tuple) else g


def to_td(g):
    G = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        G.add_node_attr_key(k, pl.Float64, -999999.0)
    ids = G.bulk_add_nodes([{"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
                            for t, (z, y, x) in zip(g.t, g.zyx)])
    if len(g.edges):
        G.bulk_add_edges([{"source_id": ids[a], "target_id": ids[b]} for a, b in g.edges])
    return G


files = sorted(GT.glob("*.geff"))
CACHE = {}
for p in files:
    n = _load(p).node_attrs()
    CACHE[p] = (np.array([r["t"] for r in n.iter_rows(named=True)]),
                np.array([[r["z"], r["y"], r["x"]] for r in n.iter_rows(named=True)], float),
                _load(p),
                (GeffMetadata.read(p).extra or {}).get("estimated_number_of_nodes"))


def run(subset, cfg) -> dict:
    """Full stage chain per dataset, scored exactly as tools/run_pipeline.py scores it."""
    rows = []
    for p in subset:
        t, zyx, gt, est = CACHE[p]
        g = Graph(t=t, zyx=zyx, dataset=p.stem)
        for stage in (refine, score_edges, resolve, repair):
            g = stage(g, cfg)
        pred = to_td(g)
        er = evaluate(pred, gt, scale=tuple(SCALE), max_distance=7.0)
        rows.append(per_sample_metrics(er, float(est) if est else float("nan"),
                                       node_recall(pred, gt)))
    return summarise(rows)


A = [p for p in files if p.stem.startswith("44b6")]
B = [p for p in files if p.stem.startswith("6bba")]

AXES = [a.split("=", 1) for a in (sys.argv[1:] or
        ["fork_cos_max=1.0,0.0", "fork_divergence_min_um=-99.0,0.0"])]
(KA, VA), (KB, VB) = [(k, [float(x) for x in v.split(",")]) for k, v in AXES]
DEFAULT = (getattr(Config(), KA), getattr(Config(), KB))

res, det = {}, {}
print(f"  {KA:>20} {KB:>20} |   {'44b6 score':>10} {'adj_edge':>8} {'divJ':>7} "
      f"|   {'6bba score':>10} {'adj_edge':>8} {'divJ':>7}", flush=True)
for a in VA:
    for b in VB:
        cfg = Config(**{KA: a, KB: b})
        sa, sb = run(A, cfg), run(B, cfg)
        res[(a, b)] = (sa["score"], sb["score"]); det[(a, b)] = (sa, sb)
        print(f"  {a:>20.3f} {b:>20.3f} |   {sa['score']:>10.4f} {sa['adj_edge_jaccard']:>8.4f} "
              f"{sa['division_jaccard']:>7.4f} |   {sb['score']:>10.4f} "
              f"{sb['adj_edge_jaccard']:>8.4f} {sb['division_jaccard']:>7.4f}", flush=True)

print("\n=== LEAVE-ONE-EMBRYO-OUT: pick on one, report on the other (objective = SCORE) ===")
bA = max(res, key=lambda k: res[k][0]); bB = max(res, key=lambda k: res[k][1])
print(f"  picked on 44b6 -> {KA}={bA[0]} {KB}={bA[1]}   HELD-OUT 6bba score {res[bA][1]:.4f}")
print(f"  picked on 6bba -> {KA}={bB[0]} {KB}={bB[1]}   HELD-OUT 44b6 score {res[bB][0]:.4f}")
print(f"  MIN of the two directions: {min(res[bA][1], res[bB][0]):.4f}")
if DEFAULT in res:
    d = res[DEFAULT]
    print(f"\n  current default ({KA}={DEFAULT[0]}, {KB}={DEFAULT[1]}): "
          f"44b6 {d[0]:.4f}  6bba {d[1]:.4f}  min {min(d):.4f}")
bm = max(res, key=lambda k: min(res[k]))
sa, sb = det[bm]
print(f"  best by MIN across embryos: {KA}={bm[0]} {KB}={bm[1]} -> "
      f"44b6 {res[bm][0]:.4f}  6bba {res[bm][1]:.4f}  min {min(res[bm]):.4f}")
print(f"     44b6 adj_edge {sa['adj_edge_jaccard']:.4f} edgeJ {sa['edge_jaccard']:.4f} divJ "
      f"{sa['division_jaccard']:.4f} tp/fp/fn {sa['division_tp']}/{sa['division_fp']}/{sa['division_fn']}")
print(f"     6bba adj_edge {sb['adj_edge_jaccard']:.4f} edgeJ {sb['edge_jaccard']:.4f} divJ "
      f"{sb['division_jaccard']:.4f} tp/fp/fn {sb['division_tp']}/{sb['division_fp']}/{sb['division_fn']}")
