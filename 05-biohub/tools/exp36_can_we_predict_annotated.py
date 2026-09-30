#!/usr/bin/env python3
"""EXP-36 -- the one N-pred hypothesis never tested: can we predict WHICH cells were ANNOTATED?

Every previous attempt asked "is this node spurious?" and failed. The metric asks a different
question: did a human annotator label this cell? Only 2.82% were, and the score is

    edgeJ * (1.1 - 0.1 * N_pred/n_total) + 0.1 * divJ

so removing nodes that match no ground-truth node is free multiplier. Finding just 25% of them
takes rank 1; finding all of them is worth +0.068.

The bar is unusual and brutal: 97.3% of predicted nodes are unmatched, so CHANCE is already 97.3%
"safe". A useful classifier must beat that, not merely be good. EXP-22 (space, time), EXP-34
(track length) and v-outgrid (their scorer) all failed. This asks the question directly, as a
supervised problem, with the features we have.
"""
import sys, time
from pathlib import Path
import numpy as np, tracksdata as td
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
import div_sweep as D

GT = Path("data/train_geff"); GRAPHS = Path("work/train_graphs"); TOL = 7.0
FEATS = ["t", "z", "y", "x", "rho2", "rho5", "rho10", "nn1", "nn3",
         "indeg", "outdeg", "comp_len", "step_in", "step_out"]

X, y, emb = [], [], []
files = sorted(GRAPHS.rglob("*.geff"))
A = [p for p in files if p.stem.startswith("44b6")][:15]
B = [p for p in files if p.stem.startswith("6bba")][:15]
for p in A + B:
    gp = GT / f"{p.stem}.geff"
    if not gp.exists(): continue
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g,tuple) else g
    n = g.node_attrs(); e = g.edge_attrs()
    pos, tt = {}, {}
    for r in n.iter_rows(named=True):
        i=int(r["node_id"]); pos[i]=np.array([r["z"],r["y"],r["x"]])*D.VOXEL_SCALE_UM; tt[i]=int(r["t"])
    indeg, outdeg, nbr = {}, {}, {}
    for r in e.iter_rows(named=True):
        s_,d_=int(r["source_id"]),int(r["target_id"])
        outdeg[s_]=outdeg.get(s_,0)+1; indeg[d_]=indeg.get(d_,0)+1
        nbr.setdefault(s_,[]).append(d_); nbr.setdefault(d_,[]).append(s_)
    # connected-component length
    parent={i:i for i in pos}
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    for r in e.iter_rows(named=True):
        ra,rb=find(int(r["source_id"])),find(int(r["target_id"]))
        if ra!=rb: parent[ra]=rb
    size={}
    for i in pos: size[find(i)]=size.get(find(i),0)+1

    gg = td.graph.IndexedRXGraph.from_geff(gp); gg = gg[0] if isinstance(gg,tuple) else gg
    gpos,gtt={},{}
    for r in gg.node_attrs().iter_rows(named=True):
        i=int(r["node_id"]); gpos[i]=np.array([r["z"],r["y"],r["x"]])*D.VOXEL_SCALE_UM; gtt[i]=int(r["t"])
    by_t={}
    for i,t in tt.items(): by_t.setdefault(t,[]).append(i)
    trees={t:(cKDTree(np.stack([pos[i] for i in ids])),ids) for t,ids in by_t.items()}
    matched=set()
    for i,gp_ in gpos.items():
        ent=trees.get(gtt[i])
        if ent is None: continue
        tr_,ids=ent; d,j=tr_.query(gp_)
        if d<=TOL: matched.add(ids[int(j)])
    if not matched: continue

    for t,ids in by_t.items():
        tree,_=trees[t]
        P=np.stack([pos[i] for i in ids])
        for k,i in enumerate(ids):
            rho2=len(tree.query_ball_point(pos[i],2.0))-1
            rho5=len(tree.query_ball_point(pos[i],5.0))-1
            rho10=len(tree.query_ball_point(pos[i],10.0))-1
            dd,_=tree.query(pos[i],k=min(4,len(ids)))
            nn1=float(dd[1]) if len(ids)>1 else 99.0
            nn3=float(dd[3]) if len(ids)>3 else 99.0
            nb=nbr.get(i,[])
            si=float(np.linalg.norm(pos[nb[0]]-pos[i])) if nb else 0.0
            so=float(np.linalg.norm(pos[nb[-1]]-pos[i])) if nb else 0.0
            X.append([t,pos[i][0],pos[i][1],pos[i][2],rho2,rho5,rho10,nn1,nn3,
                      indeg.get(i,0),outdeg.get(i,0),size[find(i)],si,so])
            y.append(1 if i in matched else 0); emb.append(p.stem[:4])

X,y,emb=np.array(X,float),np.array(y),np.array(emb)
print(f"\n  nodes {len(y):,}   ANNOTATED (matched) {y.sum():,}   base rate {y.mean():.3%}")
print(f"  chance precision on 'unmatched' = {1-y.mean():.3%}  <-- the bar to beat\n")
print("  per-feature AUC for 'this node IS annotated':")
for i,f in enumerate(FEATS):
    a=roc_auc_score(y,X[:,i]); print(f"    {f:<9} {max(a,1-a):.3f}")
print("\n  === leave-one-embryo-out: precision on REMOVAL (predicting unmatched) ===")
for held in sorted(set(emb)):
    tr,te=emb!=held,emb==held
    mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9
    m=GradientBoostingClassifier(n_estimators=150,max_depth=3,random_state=0).fit((X[tr]-mu)/sd,y[tr])
    s=m.predict_proba((X[te]-mu)/sd)[:,1]        # P(annotated)
    o=np.argsort(s)                              # most confidently UNANNOTATED first
    yy=y[te][o]
    base=1-y[te].mean()
    print(f"    held-out {held}: AUC {roc_auc_score(y[te],s):.3f}  chance {base:.2%}")
    for frac in (0.1,0.25,0.5):
        K=int(frac*len(yy)); prec=1-yy[:K].mean()
        lift = "BEATS CHANCE" if prec>base else "below chance"
        print(f"       remove {frac:.0%} ({K:,}): {prec:.3%} truly unannotated   {lift}")
