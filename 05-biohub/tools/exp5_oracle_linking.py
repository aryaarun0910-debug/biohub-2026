#!/usr/bin/env python3
"""EXP-5 — full-corpus scoring harness, driven by an ORACLE-DETECTION run.

Feeds the scorer GT node positions as if they were perfect detections, then links them
with candidate linkers. Any score below 1.0 is therefore attributable ENTIRELY to the
linker. Isolates the central claim: that divisions are destroyed by 1:1 linking rather
than never detected.

  arm A  nearest-neighbour, strictly 1:1 (Hungarian)  -> the public stack's shape
  arm B  same, but a node may take 2 children when both are within the division gates
"""
import sys
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td
from scipy.optimize import linear_sum_assignment
from geff import GeffMetadata
sys.path.insert(0, "reference/royerlab-baseline/src"); sys.path.insert(0, "reference/royerlab-baseline")
from tracking_cellmot.metrics import evaluate, per_sample_metrics, summarise, node_recall
GT = Path("data/train_geff"); SCALE = np.array([1.625, .40625, .40625]); SC = tuple(SCALE)
GATE_R = 10.0          # motion gate, um
PAR_MAX, SIS_MAX = 12.0, 18.0   # EXP-1: admits 99.3% of GT divisions

def load(p):
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    return g

def build(nodes, edges):
    g = td.graph.InMemoryGraph()
    for k in ("z","y","x"): g.add_node_attr_key(k, pl.Float64, -999999.0)
    ids = g.bulk_add_nodes([{"t":int(t),"z":float(z),"y":float(y),"x":float(x)} for t,z,y,x in nodes])
    if edges: g.bulk_add_edges([{"source_id":ids[a],"target_id":ids[b]} for a,b in edges])
    return g

def link(nodes, allow_fork):
    by_t = {}
    for i,(t,z,y,x) in enumerate(nodes): by_t.setdefault(int(t), []).append(i)
    P = np.array([[z,y,x] for _,z,y,x in nodes]) * SCALE
    edges = []
    for t in sorted(by_t):
        a, b = by_t.get(t, []), by_t.get(t+1, [])
        if not a or not b: continue
        C = np.linalg.norm(P[a][:,None,:] - P[b][None,:,:], axis=2)
        big = 1e6; M = np.where(C <= GATE_R, C, big)
        ri, ci = linear_sum_assignment(M)
        taken = set()
        for r,c in zip(ri,ci):
            if M[r,c] < big: edges.append((a[r], b[c])); taken.add(b[c])
        if allow_fork:
            for j,bj in enumerate(b):                     # second child for an unclaimed node
                if bj in taken: continue
                d = C[:, j]
                for r in np.argsort(d):
                    if d[r] > PAR_MAX: break
                    par = a[r]
                    kids = [e[1] for e in edges if e[0]==par]
                    if len(kids)!=1: continue
                    if np.linalg.norm(P[kids[0]]-P[bj]) <= SIS_MAX:
                        edges.append((par,bj)); taken.add(bj); break
    return edges

def main():
    files = sorted(GT.glob("*.geff"))
    arms = {"A 1:1 Hungarian": False, "B forking-allowed": True}
    out = {}
    for name, fork in arms.items():
        rows=[]
        for i,p in enumerate(files,1):
            gt = load(p)
            n = gt.node_attrs()
            idc = "node_id" if "node_id" in n.columns else n.columns[0]
            nodes = [(r["t"],r["z"],r["y"],r["x"]) for r in n.iter_rows(named=True)]
            e = link(nodes, fork)
            pred = build(nodes, e)
            er = evaluate(pred, load(p), scale=SC, max_distance=7.0)
            v = (GeffMetadata.read(p).extra or {}).get("estimated_number_of_nodes")
            rows.append(per_sample_metrics(er, float(v) if v else float("nan"),
                                           node_recall(pred, load(p))))
            if i%50==0: print(f"    {name}: {i}/{len(files)}", flush=True)
        s = summarise(rows); out[name]=s
        print(f"\n  {name}")
        print(f"    adj_edge_jaccard {s['adj_edge_jaccard']:.4f}   edge_jaccard {s['edge_jaccard']:.4f}")
        print(f"    division_jaccard {s['division_jaccard']:.4f}   tp/fp/fn "
              f"{s['division_tp']}/{s['division_fp']}/{s['division_fn']}")
        print(f"    score            {s['score']:.4f}")
    a,b = out["A 1:1 Hungarian"], out["B forking-allowed"]
    print(f"\n=== DELTA (B - A), oracle detection, all {len(files)} datasets ===")
    print(f"  adj_edge {b['adj_edge_jaccard']-a['adj_edge_jaccard']:+.4f}   "
          f"divJ {b['division_jaccard']-a['division_jaccard']:+.4f}   "
          f"score {b['score']-a['score']:+.4f}")

if __name__ == "__main__": main()
