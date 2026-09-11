#!/usr/bin/env python3
"""Validate src/biohub score_edges+resolve against EXP-5's oracle baseline.

EXP-5 arm B (no acceptance discriminators) gave divJ 0.3727, tp/fp/fn 142/230/9.
The new resolve adds the EXP-1 discriminators - cos(angle) and divergence persistence -
which should trade a little recall for a lot of precision.
"""
import sys; from pathlib import Path
import numpy as np, polars as pl, tracksdata as td
from geff import GeffMetadata
sys.path.insert(0,"src"); sys.path.insert(0,"reference/royerlab-baseline/src"); sys.path.insert(0,"reference/royerlab-baseline")
from tracking_cellmot.metrics import evaluate, per_sample_metrics, summarise, node_recall
from biohub.contracts import Graph, Config, SCALE
from biohub.edges import score_edges
from biohub.resolve import resolve
GT=Path("data/train_geff"); SC=tuple(SCALE); N=30

def load(p):
    g=td.graph.IndexedRXGraph.from_geff(p); return g[0] if isinstance(g,tuple) else g
def to_td(g):
    G=td.graph.InMemoryGraph()
    for k in ("z","y","x"): G.add_node_attr_key(k,pl.Float64,-999999.0)
    ids=G.bulk_add_nodes([{"t":int(t),"z":float(z),"y":float(y),"x":float(x)}
                          for t,(z,y,x) in zip(g.t,g.zyx)])
    if len(g.edges): G.bulk_add_edges([{"source_id":ids[a],"target_id":ids[b]} for a,b in g.edges])
    return G

files=sorted(GT.glob("*.geff"))
files=[f for f in files if f.stem.startswith("44b6")][:N//2]+[f for f in files if f.stem.startswith("6bba")][:N//2]
cfg=Config()
for label, fork, cfg_ in [("1:1 (no fork)",False,cfg),
                          ("fork, no discriminators",True,Config(fork_cos_max=1.0,fork_divergence_min_um=-99.0)),
                          ("fork + EXP-1 discriminators",True,cfg)]:
    rows=[]
    for p in files:
        gt=load(p); n=gt.node_attrs()
        t=np.array([r["t"] for r in n.iter_rows(named=True)])
        zyx=np.array([[r["z"],r["y"],r["x"]] for r in n.iter_rows(named=True)],float)
        g=resolve(score_edges(Graph(t=t,zyx=zyx,dataset=p.stem), cfg_), cfg_, allow_fork=fork)
        pred=to_td(g); gt2=load(p)
        er=evaluate(pred, gt2, scale=SC, max_distance=7.0)   # mutates pred with match info
        v=(GeffMetadata.read(p).extra or {}).get("estimated_number_of_nodes")
        rows.append(per_sample_metrics(er,float(v) if v else float("nan"),node_recall(pred,gt2)))
    s=summarise(rows)
    prec = s['division_tp']/max(s['division_tp']+s['division_fp'],1)
    print(f"  {label:<30} adj {s['adj_edge_jaccard']:.4f}  divJ {s['division_jaccard']:.4f}  "
          f"tp/fp/fn {s['division_tp']:>3}/{s['division_fp']:>4}/{s['division_fn']:>3}  prec {prec:.3f}")
