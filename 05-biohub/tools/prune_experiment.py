#!/usr/bin/env python3
"""Settle the node-pruning contradiction: does cutting nodes help, hurt, or cancel?

Two strategies on the same submission, because the reconciliation hypothesis is that
WHICH nodes you remove is the whole story:
  A) whole-track deletion, shortest tracks first  (what the competitor did)
  B) scattered random node deletion               (what threshold pruning looks like)
"""
import sys, json, random
from pathlib import Path
import polars as pl, tracksdata as td
from geff import GeffMetadata
sys.path.insert(0, "reference/royerlab-baseline/src"); sys.path.insert(0, "reference/royerlab-baseline")
from tracking_cellmot.metrics import evaluate, per_sample_metrics, summarise, node_recall
from scripts.csv_to_geffs import build_graph_from_rows
SCALE=(1.625,0.40625,0.40625); GT=Path("data/train_geff")

def load_gt(ds):
    g=td.graph.IndexedRXGraph.from_geff(GT/f"{ds}.geff"); g=g[0] if isinstance(g,tuple) else g
    v=(GeffMetadata.read(GT/f"{ds}.geff").extra or {}).get("estimated_number_of_nodes")
    return g, float(v) if v is not None else float("nan")

def tracks_of(nodes, edges):
    """union-find over edges -> component id per node"""
    parent={n:n for n in nodes}
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    for s,t in edges:
        if s in parent and t in parent:
            ra,rb=find(s),find(t)
            if ra!=rb: parent[ra]=rb
    return {n:find(n) for n in nodes}

def score(df, gts):
    rows=[]
    for ds,(gt,n_total) in gts.items():
        d=df.filter(pl.col("dataset")==ds)
        pred=build_graph_from_rows(d.filter(pl.col("row_type")=="node"), d.filter(pl.col("row_type")=="edge"))
        er=evaluate(pred,gt,scale=SCALE,max_distance=7.0)
        rows.append(per_sample_metrics(er,n_total,node_recall(pred,gt)))
    return summarise(rows)

def prune(df, frac, mode, seed=0):
    """Return a new submission frame with `frac` of nodes removed."""
    out=[]
    rng=random.Random(seed)
    for ds in df["dataset"].unique().to_list():
        d=df.filter(pl.col("dataset")==ds)
        nd=d.filter(pl.col("row_type")=="node"); ed=d.filter(pl.col("row_type")=="edge")
        ids=nd["node_id"].to_list()
        edges=list(zip(ed["source_id"].to_list(), ed["target_id"].to_list()))
        target=int(len(ids)*frac)
        if mode=="track":
            comp=tracks_of(set(ids), edges)
            sizes={}
            for n,c in comp.items(): sizes.setdefault(c,[]).append(n)
            order=sorted(sizes.values(), key=len)          # shortest tracks first
            drop=set()
            for grp in order:
                if len(drop)>=target: break
                drop.update(grp)
        else:
            drop=set(rng.sample(ids, min(target,len(ids))))
        keep_nodes=nd.filter(~pl.col("node_id").is_in(list(drop)))
        keep_edges=ed.filter(~pl.col("source_id").is_in(list(drop)) & ~pl.col("target_id").is_in(list(drop)))
        out += [keep_nodes, keep_edges]
    return pl.concat(out)

df=pl.read_csv(sys.argv[1] if len(sys.argv)>1 else "data/ablation/p15/submission.csv")
gts={ds:load_gt(ds) for ds in sorted(df["dataset"].unique().to_list()) if (GT/f"{ds}.geff").exists()}
df=df.filter(pl.col("dataset").is_in(list(gts)))
base=score(df,gts)
n0=df.filter(pl.col("row_type")=="node").height
print(f"{'strategy':<10} {'cut':>5} {'nodes':>8} {'edgeJ':>7} {'adj_edge':>9} {'delta':>8}")
print(f"{'baseline':<10} {0:>4}% {n0:>8} {base['edge_jaccard']:>7.4f} {base['adj_edge_jaccard']:>9.4f} {0:>8.4f}")
for mode in ("track","random"):
    for frac in (0.10,0.25,0.40):
        p=prune(df,frac,mode); s=score(p,gts)
        n=p.filter(pl.col("row_type")=="node").height
        print(f"{mode:<10} {int(frac*100):>4}% {n:>8} {s['edge_jaccard']:>7.4f} {s['adj_edge_jaccard']:>9.4f} "
              f"{s['adj_edge_jaccard']-base['adj_edge_jaccard']:>+8.4f}")
