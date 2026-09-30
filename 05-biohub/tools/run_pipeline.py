#!/usr/bin/env python3
"""Run the full six-stage pipeline over the whole corpus and score it.

Oracle detection until the dual-head model exists, so this measures the LINKER half of the
system against ground truth. Prints the stage ledger, the per-embryo split, and adj_edge /
division_jaccard separately - never their sum alone.
"""
import sys, time
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td
sys.path.insert(0,"src"); sys.path.insert(0,"reference/royerlab-baseline/src"); sys.path.insert(0,"reference/royerlab-baseline")
from geff import GeffMetadata
from tracking_cellmot.metrics import evaluate, per_sample_metrics, summarise, node_recall
sys.path.insert(0, str(Path(__file__).resolve().parent))
from biohub.contracts import check_env, Graph, Config, StageDelta, SCALE
from biohub.detect import detect_oracle
from biohub.refine import refine
from biohub.edges import score_edges
from biohub.resolve import resolve
from biohub.repair import repair
from biohub.submit import write

from _eval_common import load_gt as load, to_td, score_one

check_env()
GT=Path("data/train_geff"); cfg=Config()

files=sorted(GT.glob("*.geff"))
STAGES=("refine","score_edges","resolve","repair")
tot={s:StageDelta(s) for s in STAGES}; graphs=[]; rows=[]; per_emb={"44b6":[], "6bba":[]}
t0=time.time()
for i,p in enumerate(files,1):
    n=load(p).node_attrs()
    prev=detect_oracle(np.array([r["t"] for r in n.iter_rows(named=True)]),
                       np.array([[r["z"],r["y"],r["x"]] for r in n.iter_rows(named=True)],float), p.stem)
    for name,fn in [("refine",lambda x: refine(x,cfg)),("score_edges",lambda x: score_edges(x,cfg)),
                    ("resolve",lambda x: resolve(x,cfg)),("repair",lambda x: repair(x,cfg))]:
        t1=time.time(); cur=fn(prev); d=StageDelta.between(name,prev,cur,time.time()-t1); T=tot[name]
        for f in ("nodes_added","nodes_removed","edges_added","edges_removed",
                  "forks_created","forks_destroyed","wall_s"):
            setattr(T,f,getattr(T,f)+getattr(d,f))
        prev=cur
    graphs.append(prev)
    r=score_one(prev, load(p), path=p)
    rows.append(r); per_emb[p.stem[:4]].append(r)
    if i%50==0: print(f"    {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)

print("\n  STAGE LEDGER (all 199 datasets)")
for s in STAGES: print(tot[s].line())
info=write(graphs,"data/ablation/pipeline_full.csv")
print(f"\n  wrote {info['rows']:,} rows | {info['nodes']:,} nodes {info['edges']:,} edges {info['forks']} forks")
for name,rs in [("ALL", rows), ("44b6", per_emb["44b6"]), ("6bba", per_emb["6bba"])]:
    s=summarise(rs); prec=s['division_tp']/max(s['division_tp']+s['division_fp'],1)
    print(f"  {name:<5} n={len(rs):>3}  adj_edge {s['adj_edge_jaccard']:.4f}  edgeJ {s['edge_jaccard']:.4f}  "
          f"divJ {s['division_jaccard']:.4f}  tp/fp/fn {s['division_tp']:>3}/{s['division_fp']:>3}/{s['division_fn']:>3}  "
          f"prec {prec:.3f}  score {s['score']:.4f}")
print(f"\n  total {time.time()-t0:.0f}s")
