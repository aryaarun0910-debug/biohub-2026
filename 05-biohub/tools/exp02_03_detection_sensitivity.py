#!/usr/bin/env python3
"""EXP-2 duplicate sensitivity + EXP-3 centroid-error cliff.

Oracle detection (GT node positions) perturbed, then linked and scored. The only
variable is the perturbation, so every drop is attributable to detection quality.
Tests two claims taken from public sources but never checked on this corpus:
  - a single duplicate 0.4um away HALVES the edge Jaccard
  - centroid error has a cliff at sigma ~2um (2.5um -16%, 3um -41%, 4um -74%)
"""
import sys
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td
from scipy.optimize import linear_sum_assignment
from geff import GeffMetadata
sys.path.insert(0,"reference/royerlab-baseline/src"); sys.path.insert(0,"reference/royerlab-baseline")
from tracking_cellmot.metrics import evaluate, per_sample_metrics, summarise, node_recall
GT=Path("data/train_geff"); SCALE=np.array([1.625,.40625,.40625]); SC=tuple(SCALE); GATE=10.0
N_DS=30

def load(p):
    g=td.graph.IndexedRXGraph.from_geff(p); return g[0] if isinstance(g,tuple) else g
def build(nodes,edges):
    g=td.graph.InMemoryGraph()
    for k in ("z","y","x"): g.add_node_attr_key(k,pl.Float64,-999999.0)
    ids=g.bulk_add_nodes([{"t":int(t),"z":float(z),"y":float(y),"x":float(x)} for t,z,y,x in nodes])
    if edges: g.bulk_add_edges([{"source_id":ids[a],"target_id":ids[b]} for a,b in edges])
    return g
def link(nodes):
    by_t={}
    for i,(t,_,_,_) in enumerate(nodes): by_t.setdefault(int(t),[]).append(i)
    P=np.array([[z,y,x] for _,z,y,x in nodes])*SCALE; E=[]
    for t in sorted(by_t):
        a,b=by_t.get(t,[]),by_t.get(t+1,[])
        if not a or not b: continue
        C=np.linalg.norm(P[a][:,None,:]-P[b][None,:,:],axis=2)
        M=np.where(C<=GATE,C,1e6); ri,ci=linear_sum_assignment(M)
        E+= [(a[r],b[c]) for r,c in zip(ri,ci) if M[r,c]<1e6]
    return E

def run(files, perturb, label):
    rows=[]
    for p in files:
        gt=load(p); n=gt.node_attrs()
        nodes=[(r["t"],r["z"],r["y"],r["x"]) for r in n.iter_rows(named=True)]
        nodes=perturb(nodes)
        pred=build(nodes,link(nodes))
        er=evaluate(pred,load(p),scale=SC,max_distance=7.0)
        v=(GeffMetadata.read(p).extra or {}).get("estimated_number_of_nodes")
        rows.append(per_sample_metrics(er,float(v) if v else float("nan"),node_recall(pred,load(p))))
    s=summarise(rows)
    print(f"  {label:<34} edgeJ {s['edge_jaccard']:.4f}  adj {s['adj_edge_jaccard']:.4f}")
    return s

files=sorted(GT.glob("*.geff"))
files=[f for f in files if f.stem.startswith("44b6")][:N_DS//2] + \
      [f for f in files if f.stem.startswith("6bba")][:N_DS//2]
rng=np.random.default_rng(0)
base=run(files, lambda n: n, "baseline (oracle)")
b=base["edge_jaccard"]

print("\n=== EXP-2  DUPLICATE DETECTIONS ===")
for frac in (0.02,0.05,0.20,1.00):
    for d_um in (0.4,2.0):
        def dup(nodes, frac=frac, d=d_um):
            out=list(nodes); k=max(1,int(len(nodes)*frac))
            idx=rng.choice(len(nodes),size=k,replace=False)
            off=d/SCALE                      # um -> voxels, along x
            for i in idx:
                t,z,y,x=nodes[i]; out.append((t,z,y,x+off[2]))
            return out
        s=run(files,dup,f"dup {int(frac*100):>3}% of nodes @ {d_um}um")
        print(f"       -> edgeJ {s['edge_jaccard']-b:+.4f} ({(s['edge_jaccard']/b-1)*100:+.1f}%)")

print("\n=== EXP-3  CENTROID ERROR ===")
for sig in (0.5,1.0,2.0,2.5,3.0,4.0):
    def jit(nodes, sig=sig):
        out=[]
        for t,z,y,x in nodes:
            n=rng.normal(0,sig,3)/SCALE      # um -> voxels per axis
            out.append((t,z+n[0],y+n[1],x+n[2]))
        return out
    s=run(files,jit,f"jitter sigma={sig}um")
    print(f"       -> edgeJ {s['edge_jaccard']-b:+.4f} ({(s['edge_jaccard']/b-1)*100:+.1f}%)")
