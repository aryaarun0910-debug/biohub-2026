#!/usr/bin/env python3
"""EXP-20 -- score division selectors on the 0.947 model's own linked graphs.

Loads their prediction .geff, strips any existing division (so every selector starts from the
same single-child graph), re-adds divisions with a given ranker, and scores against our GT.
"""
import sys
from pathlib import Path
import numpy as np, tracksdata as td

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
from tracking_cellmot.metrics import evaluate, per_sample_metrics, summarise, node_recall
from geff import GeffMetadata
from biohub.contracts import SCALE
import div_sweep as D

GT = Path("data/train_geff")


def load_geff(p):
    g = td.graph.IndexedRXGraph.from_geff(p)
    return g[0] if isinstance(g, tuple) else g


def to_dicts(g):
    n = g.node_attrs(); e = g.edge_attrs()
    nodes = {int(r["node_id"]): {"t": int(r["t"]), "z": float(r["z"]),
                                 "y": float(r["y"]), "x": float(r["x"])}
             for r in n.iter_rows(named=True)}
    edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"]),
              "edge_prob": r.get("edge_prob"), "distance_um": 0.0}
             for r in e.iter_rows(named=True)]
    return nodes, edges


def strip_divisions(nodes, edges):
    """Keep ONE outgoing edge per source so every ranker starts from the same graph.

    Which one matters: keeping whichever edge happened to come first makes the baseline depend
    on file order, and the re-added divisions then compete against an arbitrary sibling. Keep the
    highest edge_prob (ties broken by shorter distance), which is the child their linker would
    have chosen -- so stripping is undone exactly when a ranker re-proposes the same pair.
    """
    best = {}
    for e in edges:
        s = int(e["source_id"])
        p = float(e["edge_prob"]) if e.get("edge_prob") is not None else 0.0
        d = float(e.get("distance_um") or 0.0)
        cur = best.get(s)
        if cur is None or (p, -d) > (cur[0], -cur[1]):
            best[s] = (p, d, e)
    return [v[2] for v in best.values()]


def build(nodes, edges):
    G = td.graph.InMemoryGraph()
    import polars as pl
    for k in ("z", "y", "x"):
        G.add_node_attr_key(k, pl.Float64, -999999.0)
    order = sorted(nodes)
    idx = {nid: i for i, nid in enumerate(order)}
    ids = G.bulk_add_nodes([{"t": nodes[n]["t"], "z": nodes[n]["z"],
                             "y": nodes[n]["y"], "x": nodes[n]["x"]} for n in order])
    if edges:
        G.bulk_add_edges([{"source_id": ids[idx[e["source_id"]]],
                           "target_id": ids[idx[e["target_id"]]]} for e in edges])
    return G


PRED = Path("data/pubweights/biohub-tracking-support-pack-50ep-v1/repo/predictions")
preds = sorted(PRED.rglob("*.geff"))
print(f"  prediction graphs: {len(preds)}\n")

RANKERS = {"theirs (arc + 0.15*sister)": D.theirs, "parent_dist alone": D.parent_only}
for name, rank in RANKERS.items():
    rows, n_div = [], 0
    for p in preds:
        stem = p.stem
        gt_path = GT / f"{stem}.geff"
        if not gt_path.exists():
            continue
        g = load_geff(p)
        nodes, edges = to_dicts(g)
        base = strip_divisions(nodes, edges)
        out = D.add_safe_divisions(nodes, base, D.DivCfg(), rank=rank)
        n_div += len(out) - len(base)
        pred = build(nodes, out); gt = load_geff(gt_path)
        er = evaluate(pred, gt, scale=tuple(SCALE), max_distance=7.0)
        est = (GeffMetadata.read(gt_path).extra or {}).get("estimated_number_of_nodes")
        nr = node_recall(pred, gt) if len(out) else 0.0
        rows.append(per_sample_metrics(er, float(est) if est else float("nan"), nr))
    if not rows:
        print(f"  {name}: no scorable datasets"); continue
    s = summarise(rows)
    print(f"  {name:<30} divisions_added={n_div:<4} edgeJ {s['edge_jaccard']:.4f}  "
          f"divJ {s['division_jaccard']:.4f}  tp/fp/fn "
          f"{s['division_tp']}/{s['division_fp']}/{s['division_fn']}  SCORE {s['score']:.4f}")
