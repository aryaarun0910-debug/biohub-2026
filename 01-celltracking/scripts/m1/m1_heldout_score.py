"""Score the one-shot M1 held-out evaluation against E0c on the full 128-crop 6bba family."""
from __future__ import annotations
import copy, json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts/win_bet"))
sys.path.insert(0, str(ROOT/"scripts/m1"))
import numpy as np, polars as pl
import coupled_arms as CA
from biotrack import wrapper as W
from biotrack.metric import estimated_nodes, per_sample_metrics, score_pred_graph
from biotrack.submission import submission_to_graphs
from coupled_replay import graph_from_cached_detections
from e0c_run import build_nodes_edges, graph_rows
from tracking_cellmot.metrics import EvaluationResult, summarise

CACHE = ROOT/"artifacts/kaggle/m1_heldout"
E0C = ROOT/"artifacts/kaggle/e0c_cache/graphs/1"
SCORES = ROOT/"artifacts/kaggle/m1_heldout_scores"; SCORES.mkdir(exist_ok=True, parents=True)

def _score(graph, crop):
    gt = str(ROOT/"data/train"/f"{crop}.geff")
    r = score_pred_graph(graph, gt)
    er = EvaluationResult(r["edge_tp"],r["edge_fp"],r["edge_fn"],
                          r["division_tp"],r["division_fp"],r["division_fn"],r["num_pred_nodes"])
    return per_sample_metrics(er, estimated_nodes(gt), r["node_recall"]), r

for p in sorted(CACHE.glob("*.npz")):
    crop = p.name.split("__")[0]
    out = SCORES/f"{crop}.json"
    if out.exists():
        continue
    CA.set_e0c_wrapper()
    nbi, raw = build_nodes_edges(graph_from_cached_detections(p))
    fn, fe, _ = W.filter_output_graph(copy.deepcopy(nbi), list(raw), dataset=crop)
    gdf = (pl.DataFrame(graph_rows(fn, fe)).with_columns(pl.lit(crop).alias("dataset"))
           .with_row_index("id"))
    m, rm = _score(submission_to_graphs(gdf)[crop], crop)
    e = (pl.read_parquet(E0C/f"{crop}.parquet").with_columns(pl.lit(crop).alias("dataset"))
         .with_row_index("id"))
    me, re_ = _score(submission_to_graphs(e)[crop], crop)
    out.write_text(json.dumps({"crop":crop,"m1":m,"e0c":me,
                               "m1_recall":rm["node_recall"],"e0c_recall":re_["node_recall"],
                               "m1_nodes":rm["num_pred_nodes"],"e0c_nodes":re_["num_pred_nodes"]}))
    print(f"{crop}: m1={m['adj_edge_jaccard']:.4f} e0c={me['adj_edge_jaccard']:.4f}", flush=True)
print("SCORING COMPLETE", flush=True)
