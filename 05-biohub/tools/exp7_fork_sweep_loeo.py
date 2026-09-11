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

# which two Config fields to sweep, and over what: `exp7 <fieldA>=v,v,v <fieldB>=v,v,v`
AXES = [a.split("=", 1) for a in (sys.argv[1:] or
        ["fork_cos_max=1.0,0.0,-0.30", "fork_divergence_min_um=-99.0,0.0,1.0,2.0"])]
(KA, VA), (KB, VB) = [(k, [float(x) for x in v.split(",")]) for k, v in AXES]
DEFAULT = (getattr(Config(), KA), getattr(Config(), KB))

res = {}
print(f"  {KA:>22} {KB:>22} | {'44b6 divJ':>10} {'tp/fp/fn':>12} | {'6bba divJ':>10} {'tp/fp/fn':>12}")
for a in VA:
    for b in VB:
        cfg = Config(**{KA: a, KB: b})
        ja, ta, fa, na = divJ(A, cfg); jb, tb, fb, nb = divJ(B, cfg)
        res[(a, b)] = (ja, jb)
        print(f"  {a:>22.2f} {b:>22.2f} | {ja:>10.4f} {f'{ta}/{fa}/{na}':>12} | "
              f"{jb:>10.4f} {f'{tb}/{fb}/{nb}':>12}", flush=True)

print("\n=== LEAVE-ONE-EMBRYO-OUT: pick on one, report on the other ===")
bA = max(res, key=lambda k: res[k][0]); bB = max(res, key=lambda k: res[k][1])
print(f"  picked on 44b6 -> {KA}={bA[0]} {KB}={bA[1]}   HELD-OUT 6bba divJ {res[bA][1]:.4f}")
print(f"  picked on 6bba -> {KA}={bB[0]} {KB}={bB[1]}   HELD-OUT 44b6 divJ {res[bB][0]:.4f}")
print(f"  MIN of the two directions: {min(res[bA][1], res[bB][0]):.4f}")
if DEFAULT in res:
    d = res[DEFAULT]
    print(f"\n  current default ({KA}={DEFAULT[0]}, {KB}={DEFAULT[1]}): "
          f"44b6 {d[0]:.4f}  6bba {d[1]:.4f}  min {min(d):.4f}")
bm = max(res, key=lambda k: min(res[k]))
print(f"  best by MIN across embryos: {KA}={bm[0]} {KB}={bm[1]} -> "
      f"44b6 {res[bm][0]:.4f}  6bba {res[bm][1]:.4f}  min {min(res[bm]):.4f}")
