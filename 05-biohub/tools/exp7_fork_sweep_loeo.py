#!/usr/bin/env python3
"""EXP-7 — sweep fork acceptance thresholds, LEAVE-ONE-EMBRYO-OUT.

The pooled corpus is 125/151 divisions from 6bba, so a pooled sweep just fits 6bba. Train and
test are embryo-disjoint, so the honest protocol is: pick on one embryo, report on the other,
both directions, and take the MIN. A setting that only works on the embryo it was chosen on is
worth nothing on a hidden one.
"""
import sys, itertools
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td
sys.path.insert(0,"src"); sys.path.insert(0,"reference/royerlab-baseline/src"); sys.path.insert(0,"reference/royerlab-baseline")
from geff import GeffMetadata
from tracking_cellmot.metrics import evaluate
from biohub.contracts import Graph, Config, SCALE
from biohub.detect import detect_oracle
from biohub.edges import score_edges
from biohub.resolve import resolve
GT=Path("data/train_geff")

def load(p):
    g=td.graph.IndexedRXGraph.from_geff(p); return g[0] if isinstance(g,tuple) else g
def to_td(g):
    G=td.graph.InMemoryGraph()
    for k in ("z","y","x"): G.add_node_attr_key(k,pl.Float64,-999999.0)
    ids=G.bulk_add_nodes([{"t":int(t),"z":float(z),"y":float(y),"x":float(x)} for t,(z,y,x) in zip(g.t,g.zyx)])
    if len(g.edges): G.bulk_add_edges([{"source_id":ids[a],"target_id":ids[b]} for a,b in g.edges])
    return G

files=sorted(GT.glob("*.geff"))
cache={}
for p in files:
    n=load(p).node_attrs()
    cache[p]=(np.array([r["t"] for r in n.iter_rows(named=True)]),
              np.array([[r["z"],r["y"],r["x"]] for r in n.iter_rows(named=True)],float))

def divJ(subset, cfg):
    tp=fp=fn=0
    for p in subset:
        t,zyx=cache[p]
        g=resolve(score_edges(Graph(t=t,zyx=zyx,dataset=p.stem),cfg),cfg,allow_fork=True)
        er=evaluate(to_td(g), load(p), scale=tuple(SCALE), max_distance=7.0)
        tp+=er.division_tp; fp+=er.division_fp; fn+=er.division_fn
    return (tp/max(tp+fp+fn,1)), tp, fp, fn

A=[p for p in files if p.stem.startswith("44b6")]
B=[p for p in files if p.stem.startswith("6bba")]
GRID=[(c,d) for c in (1.0,0.0,-0.30) for d in (-99.0,0.0,1.0,2.0)]
res={}
print(f"  {'cos<=':>6} {'div>=':>6} | {'44b6 divJ':>10} {'tp/fp/fn':>12} | {'6bba divJ':>10} {'tp/fp/fn':>12}")
for c,d in GRID:
    cfg=Config(fork_cos_max=c, fork_divergence_min_um=d)
    ja,ta,fa,na = divJ(A,cfg); jb,tb,fb,nb = divJ(B,cfg)
    res[(c,d)]=(ja,jb)
    print(f"  {c:>6.2f} {d:>6.1f} | {ja:>10.4f} {f'{ta}/{fa}/{na}':>12} | {jb:>10.4f} {f'{tb}/{fb}/{nb}':>12}")

print("\n=== LEAVE-ONE-EMBRYO-OUT: pick on one, report on the other ===")
best_on_A=max(res, key=lambda k: res[k][0]); best_on_B=max(res, key=lambda k: res[k][1])
print(f"  picked on 44b6 -> cos<={best_on_A[0]} div>={best_on_A[1]}   HELD-OUT 6bba divJ {res[best_on_A][1]:.4f}")
print(f"  picked on 6bba -> cos<={best_on_B[0]} div>={best_on_B[1]}   HELD-OUT 44b6 divJ {res[best_on_B][0]:.4f}")
print(f"  MIN of the two directions: {min(res[best_on_A][1], res[best_on_B][0]):.4f}")
cur=res[(-0.30,1.0)]
print(f"\n  current default (cos<=-0.30, div>=1.0): 44b6 {cur[0]:.4f}  6bba {cur[1]:.4f}  min {min(cur):.4f}")
gates=res[(1.0,-99.0)]
print(f"  gates only (no discriminators):          44b6 {gates[0]:.4f}  6bba {gates[1]:.4f}  min {min(gates):.4f}")
best_min=max(res, key=lambda k: min(res[k]))
print(f"  best by MIN across embryos: cos<={best_min[0]} div>={best_min[1]} -> "
      f"44b6 {res[best_min][0]:.4f}  6bba {res[best_min][1]:.4f}  min {min(res[best_min]):.4f}")
