#!/usr/bin/env python3
"""Score a real submission.csv on the four test movies against the released annotations.

tools/testproxy.py reads prediction .geff files; this reads the actual submission.csv a kernel
produced, so a variant can be judged BEFORE spending one of the day's submissions and before
waiting ~10 hours for the leaderboard.

This is the same substrate argument as testproxy: the test set is four movies, the organisers
released sparse annotations for all four, and the leaderboard scores held-out annotations on
those SAME movies. Their in-kernel ppsweep instead uses 8 TRAIN stems with EVEN embryo weighting,
which we now know is wrong -- the real score weights 6bba at 95.4% and 44b6 at 4.6%.

    python tools/score_submission_local.py work/minlen9_out/submission.csv
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
STEMS = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]


def main(path):
    df = pl.read_csv(path)
    rows = []
    print(f"  {'stem':<16}{'N_pred':>9}{'mult':>8}{'edgeJ':>8}{'adj':>8}{'divTP':>7}{'divFN':>7}{'w':>7}")
    for s in STEMS:
        d = df.filter(pl.col("dataset") == s)
        nodes = d.filter(pl.col("row_type") == "node")
        edges = d.filter(pl.col("row_type") == "edge")
        G = td.graph.InMemoryGraph()
        for k in ("z", "y", "x"): G.add_node_attr_key(k, pl.Float64, -999999.0)
        nid = nodes["node_id"].to_list()
        ids = G.bulk_add_nodes([{"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
                                for t, z, y, x in zip(nodes["t"], nodes["z"], nodes["y"], nodes["x"])])
        m = dict(zip(nid, ids))
        ee = [{"source_id": m[a], "target_id": m[b]}
              for a, b in zip(edges["source_id"], edges["target_id"]) if a in m and b in m]
        if ee: G.bulk_add_edges(ee)
        gt = td.graph.IndexedRXGraph.from_geff(GT / f"{s}.geff")
        gt = gt[0] if isinstance(gt, tuple) else gt
        est = float((GeffMetadata.read(GT / f"{s}.geff").extra or {})["estimated_number_of_nodes"])
        er = evaluate(G, gt, scale=tuple(SCALE), max_distance=7.0)
        r = per_sample_metrics(er, est, node_recall(G, gt) if ee else 0.0)
        rows.append(r)
        w = r["edge_tp"] + r["edge_fp"] + r["edge_fn"]
        print(f"  {s:<16}{len(nid):>9,}{1-0.1*((len(nid)-est)/est):>8.4f}{r['edge_jaccard']:>8.4f}"
              f"{r['adj_edge_jaccard']:>8.4f}{r['division_tp']:>7}{r['division_fn']:>7}{w:>7}")
    S = summarise(rows)
    dj = S["division_jaccard"]; dj = 0.0 if dj != dj else dj
    print(f"\n  adj {S['adj_edge_jaccard']:.5f}   divJ {dj:.4f}   ->  LOCAL TEST SCORE "
          f"{S['adj_edge_jaccard'] + 0.1*dj:.5f}")


main(sys.argv[1] if len(sys.argv) > 1 else "work/repro_out/submission.csv")
