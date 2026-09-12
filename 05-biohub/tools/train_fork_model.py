#!/usr/bin/env python3
"""Fit the fork-acceptance model and ship it as coefficients.

EXP-15: the true/false fork signal is real but NOT axis-aligned. Best single threshold reaches
AUC 0.806 in-sample; a logistic model reaches 0.832 HELD-OUT, and at a leave-one-embryo-out
operating point it lifts divJ +0.0359 on 44b6 and +0.0941 on 6bba. Gradient boosting overfits
at n=207, so linear it is -- 7 features, 8 numbers, no runtime dependency on sklearn.
"""
import sys, json
from pathlib import Path
import numpy as np, tracksdata as td
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
from biohub.contracts import Config
from biohub.detect import detect_oracle
from biohub.edges import score_edges
from biohub.refine import refine
from biohub.resolve import resolve, fork_features
from biohub.repair import repair

GT = Path("data/train_geff")
OUT = Path("src/biohub/fork_model.json")

X, y, emb = [], [], []
cfg = Config(fork_accept_p=-1.0)          # -1 disables the model so we can harvest ALL candidates
for p in sorted(GT.glob("*.geff")):
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs(); e = g.edge_attrs()
    idx = {r["node_id"]: i for i, r in enumerate(n.iter_rows(named=True))}
    t = np.array([r["t"] for r in n.iter_rows(named=True)])
    zyx = np.array([[r["z"], r["y"], r["x"]] for r in n.iter_rows(named=True)], float)
    kids = {}
    for r in e.iter_rows(named=True):
        if r["source_id"] in idx and r["target_id"] in idx:
            kids.setdefault(idx[r["source_id"]], []).append(idx[r["target_id"]])
    gdiv = {s: k for s, k in kids.items() if len(k) == 2}
    gg = repair(resolve(score_edges(refine(detect_oracle(t.copy(), zyx.copy(), p.stem), cfg), cfg),
                        cfg), cfg)
    P = gg.um(); succ = {}
    for s_, d_ in gg.edges: succ.setdefault(int(s_), int(d_))
    pk = {}
    for s_, d_ in gg.edges: pk.setdefault(int(s_), []).append(int(d_))
    for par, kk in pk.items():
        if len(kk) != 2: continue
        f = fork_features(P, par, kk[0], kk[1], succ)
        if f is None: continue
        X.append(f); y.append(1 if (par in gdiv and set(kk) == set(gdiv[par])) else 0)
        emb.append(p.stem[:4])

X, y, emb = np.array(X), np.array(y), np.array(emb)
mu, sd = X.mean(0), X.std(0) + 1e-9
m = LogisticRegression(max_iter=2000, class_weight="balanced").fit((X - mu) / sd, y)
OUT.write_text(json.dumps({
    "features": ["cos", "sister", "arc_max", "arc_min", "arc_asym", "divergence", "arc_sum"],
    "mu": mu.tolist(), "sd": sd.tolist(),
    "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
    "trained_on": {"forks": int(len(y)), "true": int(y.sum()), "false": int((1 - y).sum()),
                   "embryos": sorted(set(emb))},
    "note": "EXP-15. Trained on BOTH embryos for shipping; the honest generalisation estimate is "
            "the leave-one-embryo-out result, divJ +0.0359 (44b6) / +0.0941 (6bba).",
}, indent=2))
print(f"  wrote {OUT}  ({len(y)} forks, {y.sum()} true)")
for n_, c_ in zip(["cos","sister","arc_max","arc_min","arc_asym","divergence","arc_sum"], m.coef_[0]):
    print(f"    {n_:<12} {c_:+.3f}")
