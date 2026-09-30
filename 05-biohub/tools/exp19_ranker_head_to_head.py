#!/usr/bin/env python3
"""EXP-19 -- our learned ranker vs the 0.947 kernel's, on the same labelled forks.

The 0.947 family selects which divisions survive its cap with ONE line:

    score = parent_dist + 0.15 * sister_dist        (sorted ascending, lowest wins)

Because the cap binds at roughly the true biological division rate, the RANKING is what decides
division Jaccard. This scores their exact rule against ours on the same 207 labelled forks used
in EXP-15, leave-one-embryo-out, so the comparison is like-for-like.

CAVEAT, stated up front: our forks come from oracle detection at 6.7 cells/frame, theirs run at
~237. This measures ranking quality on identical data, not their score at their density.
"""
import sys
from pathlib import Path
import numpy as np, tracksdata as td
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
from biohub.contracts import Config
from biohub.detect import detect_oracle
from biohub.edges import score_edges
from biohub.refine import refine
from biohub.repair import repair
from biohub.resolve import resolve, fork_features

GT = Path("data/train_geff")
cfg = Config(fork_accept_p=-1.0)          # harvest every candidate fork, gate nothing
X, y, emb = [], [], []
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
        X.append(f); emb.append(p.stem[:4])
        y.append(1 if (par in gdiv and set(kk) == set(gdiv[par])) else 0)

X, y, emb = np.array(X), np.array(y), np.array(emb)
# their rule needs parent_dist (arc_max, index 2) and sister_dist (index 1); LOWER is better,
# so negate to make it a "higher = more likely a division" score.
theirs = -(X[:, 2] + 0.15 * X[:, 1])
print(f"  labelled forks: {len(y)}   true {y.sum()}   false {(1-y).sum()}\n")
print(f"  {'ranker':<34} {'pooled AUC':>11} {'44b6':>8} {'6bba':>8}")


def report(name, s):
    a = roc_auc_score(y, s)
    pe = [roc_auc_score(y[emb == e], s[emb == e]) if len(set(y[emb == e])) > 1 else float("nan")
          for e in ("44b6", "6bba")]
    print(f"  {name:<34} {a:>11.3f} {pe[0]:>8.3f} {pe[1]:>8.3f}")
    return a


report("0.947 kernel: -(arc + 0.15*sister)", theirs)
report("arc_max alone", -X[:, 2])
print()
print("  LEAVE-ONE-EMBRYO-OUT (train one embryo, rank the other):")
for held in ("44b6", "6bba"):
    tr, te = emb != held, emb == held
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit((X[tr] - mu) / sd, y[tr])
    ours = m.predict_proba((X[te] - mu) / sd)[:, 1]
    a_ours = roc_auc_score(y[te], ours)
    a_theirs = roc_auc_score(y[te], theirs[te])
    print(f"    held-out {held}: OURS {a_ours:.3f}   THEIRS {a_theirs:.3f}   "
          f"delta {a_ours - a_theirs:+.3f}   (n={te.sum()}, true {y[te].sum()})")

print("\n  precision at their operating point (top-K by each ranker, K = number of TRUE forks):")
for held in ("44b6", "6bba"):
    tr, te = emb != held, emb == held
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit((X[tr] - mu) / sd, y[tr])
    ours = m.predict_proba((X[te] - mu) / sd)[:, 1]
    K = int(y[te].sum())
    po = y[te][np.argsort(-ours)][:K].mean()
    pt = y[te][np.argsort(-theirs[te])][:K].mean()
    print(f"    held-out {held}: top-{K}  OURS {po:.1%} correct   THEIRS {pt:.1%}   "
          f"delta {po - pt:+.1%}")
