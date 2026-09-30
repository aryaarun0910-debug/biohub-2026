#!/usr/bin/env python3
"""EXP-15 -- is the TP/FP fork signal reachable by a CLASSIFIER but not by a threshold?

The atlas (EXP-14) shows parent-arc separates true from false forks in distribution -- TRUE p50
6.91um vs FALSE p50 8.92um -- yet every threshold sweep we ran came out net-negative, because a
cut that removes 73% of false forks also removes 25% of true ones and divJ weights those equally.

That is the signature of a signal that exists but is not axis-aligned. This probes it honestly:
fit a small classifier on the fork geometry, LEAVE-ONE-EMBRYO-OUT, and see whether it beats the
best single threshold. If it does not, fork acceptance really is finished and the remaining work
is all upstream.
"""
import sys
from collections import Counter
from pathlib import Path
import numpy as np, tracksdata as td
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
from biohub.contracts import Config
from biohub.detect import detect_oracle
from biohub.edges import score_edges
from biohub.refine import refine
from biohub.resolve import resolve
from biohub.repair import repair

GT = Path("data/train_geff")


def feats(P, par, a, b, succ):
    va, vb = P[a] - P[par], P[b] - P[par]
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return None
    cos = float(va @ vb / (na * nb))
    sis = float(np.linalg.norm(P[a] - P[b]))
    g1, g2 = succ.get(a), succ.get(b)
    div = (float(np.linalg.norm(P[g1] - P[g2]) - sis)
           if g1 is not None and g2 is not None else 0.0)
    return [cos, sis, float(max(na, nb)), float(min(na, nb)),
            float(abs(na - nb)), div, float(na + nb)]


NAMES = ["cos", "sister", "arc_max", "arc_min", "arc_asym", "divergence", "arc_sum"]
X, y, emb = [], [], []
cfg = Config()
for p in sorted(GT.glob("*.geff")):
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs(); e = g.edge_attrs()
    ids = [r["node_id"] for r in n.iter_rows(named=True)]
    idx = {v: i for i, v in enumerate(ids)}
    t = np.array([r["t"] for r in n.iter_rows(named=True)])
    zyx = np.array([[r["z"], r["y"], r["x"]] for r in n.iter_rows(named=True)], float)
    ge = [(idx[r["source_id"]], idx[r["target_id"]]) for r in e.iter_rows(named=True)
          if r["source_id"] in idx and r["target_id"] in idx]
    kids = {}
    for s_, d_ in ge:
        kids.setdefault(s_, []).append(d_)
    gdiv = {s_: k for s_, k in kids.items() if len(k) == 2}

    gg = repair(resolve(score_edges(refine(detect_oracle(t.copy(), zyx.copy(), p.stem), cfg), cfg),
                        cfg), cfg)
    P = gg.um()
    succ = {}
    for s_, d_ in gg.edges:
        succ.setdefault(int(s_), int(d_))
    pk = {}
    for s_, d_ in gg.edges:
        pk.setdefault(int(s_), []).append(int(d_))
    for par, kk in pk.items():
        if len(kk) != 2:
            continue
        f = feats(P, par, kk[0], kk[1], succ)
        if f is None:
            continue
        X.append(f); emb.append(p.stem[:4])
        y.append(1 if (par in gdiv and set(kk) == set(gdiv[par])) else 0)

X, y, emb = np.array(X), np.array(y), np.array(emb)
print(f"  emitted forks: {len(y)}  true {y.sum()}  false {(1-y).sum()}\n")

print("  best SINGLE-THRESHOLD AUC per feature (the ceiling a gate can reach):")
for i, nm in enumerate(NAMES):
    a = roc_auc_score(y, -X[:, i])
    print(f"    {nm:<12} AUC {max(a, 1-a):.3f}")

print("\n  LEAVE-ONE-EMBRYO-OUT, classifier on all features:")
for name, mk in (("logistic", lambda: LogisticRegression(max_iter=2000, class_weight="balanced")),
                 ("grad-boost", lambda: GradientBoostingClassifier(n_estimators=60, max_depth=2,
                                                                   random_state=0))):
    aucs = []
    for held in ("44b6", "6bba"):
        tr, te = emb != held, emb == held
        if te.sum() < 5 or len(set(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = mk().fit((X[tr] - mu) / sd, y[tr])
        s = m.predict_proba((X[te] - mu) / sd)[:, 1]
        a = roc_auc_score(y[te], s) if len(set(y[te])) > 1 else float("nan")
        aucs.append(a)
        print(f"    {name:<11} held-out {held}: AUC {a:.3f}  (n={te.sum()}, true {y[te].sum()})")
    if aucs:
        print(f"    {name:<11} MIN across embryos: {np.nanmin(aucs):.3f}")

# ---- does it actually move divJ? AUC is not the metric we are scored on. ----
# With total GT divisions fixed per embryo, FN = GT - TP, so
#     divJ = TP / (TP + FP + FN) = TP / (GT + FP)
# Rejecting a fork moves TP or FP down by one; only a rejection that is FALSE helps.
GT_DIV = {"44b6": 26, "6bba": 125}
print("\n  === divJ at a real operating point, LEAVE-ONE-EMBRYO-OUT ===")
print("  (threshold chosen on the TRAINING embryo, applied blind to the held-out one)")
for held in ("44b6", "6bba"):
    tr, te = emb != held, emb == held
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit((X[tr]-mu)/sd, y[tr])
    # pick the threshold that maximises divJ ON THE TRAINING EMBRYO only
    otr = emb[tr][0] if len(set(emb[tr])) == 1 else None
    s_tr = m.predict_proba((X[tr]-mu)/sd)[:, 1]
    gtr = GT_DIV[otr] if otr else sum(GT_DIV.values()) - GT_DIV[held]
    best_t, best_j = 0.0, -1
    for t_ in np.linspace(0, 1, 101):
        keep = s_tr >= t_
        tp, fp = int(y[tr][keep].sum()), int((1-y[tr][keep]).sum())
        j = tp / max(gtr + fp, 1)
        if j > best_j: best_j, best_t = j, t_
    s_te = m.predict_proba((X[te]-mu)/sd)[:, 1]
    g_ = GT_DIV[held]
    tp0, fp0 = int(y[te].sum()), int((1-y[te]).sum())
    keep = s_te >= best_t
    tp1, fp1 = int(y[te][keep].sum()), int((1-y[te][keep]).sum())
    print(f"\n  held-out {held}: threshold {best_t:.2f} chosen on the other embryo")
    print(f"    before  tp {tp0:>3} fp {fp0:>3}  divJ {tp0/(g_+fp0):.4f}")
    print(f"    after   tp {tp1:>3} fp {fp1:>3}  divJ {tp1/(g_+fp1):.4f}   "
          f"delta {tp1/(g_+fp1) - tp0/(g_+fp0):+.4f}")
