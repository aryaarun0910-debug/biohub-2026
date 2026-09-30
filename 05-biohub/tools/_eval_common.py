"""Shared scoring plumbing for run_pipeline.py and exp7_fork_sweep_loeo.py.

These two were duplicating `to_td` and the per-sample scoring call, which is exactly why the
sweep's claimed parity with the pipeline had to be argued rather than being structural. The
zero-edge guard in `score_one` also has to exist in only ONE place: evaluate() short-circuits
before writing match_node_id onto the prediction, so node_recall raises KeyError on any dataset
that resolves to zero edges -- which aborts a whole sweep grid rather than scoring that point 0.
"""
from __future__ import annotations
import polars as pl, tracksdata as td
from geff import GeffMetadata
from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall
from biohub.contracts import SCALE


def load_gt(p):
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


def score_one(g, gt, est=None, path=None):
    """One prediction -> one per-sample metric row. Survives a zero-edge prediction."""
    pred = to_td(g)
    er = evaluate(pred, gt, scale=tuple(SCALE), max_distance=7.0)
    if est is None and path is not None:
        est = (GeffMetadata.read(path).extra or {}).get("estimated_number_of_nodes")
    # evaluate() returns early on an empty prediction WITHOUT matching, so match_node_id is
    # absent and node_recall would raise. An empty prediction recalls nothing: that is 0.0.
    nr = node_recall(pred, gt) if len(g.edges) else 0.0
    return per_sample_metrics(er, float(est) if est else float("nan"), nr)
