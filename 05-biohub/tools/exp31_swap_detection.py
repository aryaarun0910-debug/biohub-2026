#!/usr/bin/env python3
"""EXP-31 -- the SWAP problem: a 2.3% base rate, not 0.15%.

Every closure so far died the same death: a candidate pool with a ~0.1% base rate against a
metric demanding 24-48% precision, so 150-320x lift was needed and ~2x was available.

But EXP-29's TARGET_TAKEN category is structurally different. 2,887 ground-truth edges have their
target assigned to a DIFFERENT parent. That is not "find a needle among 282,010 free pairs" -- it
is a binary question asked of each predicted edge: is this assignment WRONG, and should the target
belong to a rival parent instead? The pool is the set of predicted edges (~124k) and 2,887 are
wrong, so the base rate is 2.3% -- fifteen times richer than anything examined so far.

A swap is also worth more than an addition: it converts a false positive AND a false negative into
a true positive at once.

Measured here: the base rate, what features separate a wrong assignment from a right one, and what
a swap is worth.
"""
import sys
from pathlib import Path
import numpy as np, tracksdata as td
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
import div_sweep as D

GT = Path("data/train_geff"); GRAPHS = Path("work/train_graphs"); TOL = 7.0
FEATS = ["dist", "rival_dist", "rival_margin", "n_rivals", "src_outdeg",
         "tgt_indeg_free", "disp_ratio", "src_speed"]
X, y, EMB = [], [], []

for p in sorted(GRAPHS.rglob("*.geff")):          # ALL 199, not a slice
    gp = GT / f"{p.stem}.geff"
    if not gp.exists():
        continue
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs(); e = g.edge_attrs()
    pos, tt = {}, {}
    for r in n.iter_rows(named=True):
        i = int(r["node_id"]); pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        tt[i] = int(r["t"])
    pred = []
    inbound, outbound = {}, {}
    for r in e.iter_rows(named=True):
        s_, d_ = int(r["source_id"]), int(r["target_id"])
        pred.append((s_, d_)); inbound[d_] = s_
        outbound.setdefault(s_, []).append(d_)

    gg = td.graph.IndexedRXGraph.from_geff(gp); gg = gg[0] if isinstance(gg, tuple) else gg
    gpos, gtt = {}, {}
    for r in gg.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gtt[i] = int(r["t"])
    gedges = [(int(r["source_id"]), int(r["target_id"]))
              for r in gg.edge_attrs().iter_rows(named=True)]
    if not gedges:
        continue
    by_t = {}
    for i, t in tt.items():
        by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}
    def match(pt, t):
        ent = trees.get(t)
        if ent is None: return None
        tree, ids = ent
        d, j = tree.query(pt)
        return ids[int(j)] if d <= TOL else None
    # JUDGEABLE ONLY. Ground truth annotates 2.82% of cells, so most predicted edges have no GT
    # counterpart merely because their endpoints are unlabelled. Calling those "wrong" is the
    # same sparse-annotation artefact that corrupts their own precision_sparse metric -- and it
    # produced a 98.86% base rate on the first run of this experiment. An edge is judgeable only
    # when its SOURCE matches a GT node that HAS an outgoing GT edge; then we know the answer.
    truth, gt_target_of = set(), {}
    for s, d in gedges:
        if s in gpos and d in gpos:
            ms, md = match(gpos[s], gtt[s]), match(gpos[d], gtt[d])
            if ms is not None and md is not None:
                truth.add((ms, md)); gt_target_of.setdefault(ms, set()).add(md)
    if not truth:
        continue
    # which predicted targets are contested by a nearby rival parent?
    for s_, d_ in pred:
        t = tt[s_]
        ent = trees.get(t)
        if ent is None:
            continue
        tree, ids = ent
        rivals = [ids[j] for j in tree.query_ball_point(pos[d_], 10.0)
                  if ids[j] != s_ and tt[ids[j]] == t]
        if not rivals:
            continue
        dd = float(np.linalg.norm(pos[d_] - pos[s_]))
        rd = min(float(np.linalg.norm(pos[d_] - pos[r])) for r in rivals)
        if s_ not in gt_target_of:
            continue                                  # unjudgeable: source not annotated
        X.append([dd, rd, dd - rd, len(rivals), len(outbound.get(s_, [])),
                  1.0 if d_ not in inbound else 0.0, dd / max(rd, 1e-6), dd])
        y.append(0 if d_ in gt_target_of[s_] else 1)  # 1 = assignment is WRONG, and we KNOW it
        EMB.append(p.stem[:4])

X, y = np.array(X), np.array(y)
print(f"\n  contested predicted edges: {len(y):,}")
print(f"  of which WRONG:            {y.sum():,}")
print(f"  BASE RATE (wrong):         {y.mean():.2%}   <-- free-pair region was 0.152%\n")
if y.sum() > 5:
    print("  per-feature AUC for 'this assignment is wrong':")
    for i, f in enumerate(FEATS):
        a = roc_auc_score(y, X[:, i]); print(f"    {f:<14} {max(a, 1-a):.3f}")
    print(f"\n  edge break-even precision 48.1%  ->  lift required {0.481/max(y.mean(),1e-9):.1f}x")
    o = np.argsort(-X[:, 2])            # rank by rival_margin: rival is closer than we are
    yy = y[o]
    print("  ranking by rival_margin alone (rival closer than the assigned parent):")
    for K in (100, 500, 2000, 5000):
        if K <= len(yy):
            pr = yy[:K].mean()
            print(f"    top-{K:<5} precision {pr:>6.1%}  wrong found {int(yy[:K].sum()):>5}"
                  f"{'  <-- PAYS' if pr > 0.481 else ''}")

# ---- can a MODEL clear the 48.1% bar? LOEO, since train/test are embryo-disjoint ----
import pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
emb = np.array(EMB)
print("\n  === trained discriminator, leave-one-embryo-out ===")
for name, mk in (("logistic", lambda: LogisticRegression(max_iter=3000, class_weight="balanced")),
                 ("grad-boost", lambda: GradientBoostingClassifier(n_estimators=120, max_depth=3,
                                                                   random_state=0))):
    for held in sorted(set(emb)):
        tr, te = emb != held, emb == held
        if te.sum() < 50 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = mk().fit((X[tr] - mu) / sd, y[tr])
        s = m.predict_proba((X[te] - mu) / sd)[:, 1]
        auc = roc_auc_score(y[te], s)
        o = np.argsort(-s); yy = y[te][o]
        line = f"    {name:<11} held-out {held}: AUC {auc:.3f}  base {y[te].mean():.2%}  "
        for K in (50, 200, 1000):
            if K <= len(yy):
                pr = yy[:K].mean()
                line += f"top{K} {pr:.0%}{'*' if pr > 0.481 else ''}  "
        print(line)
print("    (* clears the 48.1% edge break-even)")
