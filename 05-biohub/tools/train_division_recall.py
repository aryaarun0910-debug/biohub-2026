#!/usr/bin/env python3
"""Train the division-RECALL model on the 0.947 model's own linked graphs.

WHY RECALL, AND WHY A MODEL. Their selector rejects a fork on six hand-set conditions, and those
gates discard 74.8% of real Kaggle divisions before any ranking happens (EXP-21). Their own
diagnostics confirm it: div_tp 3, div_fp 1, div_fn 9 -- 25% recall at 75% precision, and the
global cap never binds. So the fix is not a better ORDER over the survivors, it is to stop
discarding them and then separate the enlarged pool with something better than four thresholds.

    gates WIDE  ->  learned score  ->  their cap enforces precision

The model is deliberately small: ~9 geometric features, logistic by default. Gradient boosting
already overfit at n=207 in EXP-15 (held-out AUC 0.728 against logistic's 0.832), and the whole
corpus carries only 151 divisions. This needs no GPU; it needs LABELLED CANDIDATES, which is what
the widened gates produce.

Everything is leave-one-embryo-out, both directions, minimum reported -- train and test are
embryo-disjoint by the host's own statement.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
import div_sweep as D

FEATS = ["parent_dist", "sister_dist", "child_dist", "cos", "diverge",
         "arc_max", "arc_min", "arc_asym", "arc_sum"]
GT = Path("data/train_geff")


def wide_cfg() -> D.DivCfg:
    """Gates opened so real divisions survive to be labelled. EXP-21 prices each closed gate:
    symmetry 37.7%, divergence 48.3%, parent arc 19.9%, sister 12.6% of real divisions lost."""
    return D.DivCfg(max_um=20.0, sister_max_um=30.0, existing_child_max_um=20.0,
                    symmetry_tau=0.0,          # 0 disables the symmetry test
                    diverge_um=-99.0, require_divergence=False,
                    require_mutual_nn=False,
                    frame_frac_cap=1.0, global_frac_cap=1.0)


def harvest(graph_dir: Path, cfg: D.DivCfg):
    """Every fork the WIDE gates admit, labelled against ground truth.

    A proposal is POSITIVE when it reproduces a real division: its parent matches a GT division
    parent within the scorer's 7um tolerance, and its two children match that division's two
    children (in either order) within the same tolerance. Anything else the gates admit is a
    negative -- which is exactly the population the model must learn to reject.
    """
    import tracksdata as td
    TOL = 7.0
    X, y, emb, ds = [], [], [], []
    for p in sorted(graph_dir.rglob("*.geff")):
        gt_path = GT / f"{p.stem}.geff"
        if not gt_path.exists():
            continue
        g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
        n = g.node_attrs(); e = g.edge_attrs()
        nodes = {int(r["node_id"]): {"t": int(r["t"]), "z": float(r["z"]),
                                     "y": float(r["y"]), "x": float(r["x"])}
                 for r in n.iter_rows(named=True)}
        edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"]),
                  "edge_prob": r.get("edge_prob"), "distance_um": 0.0}
                 for r in e.iter_rows(named=True)]
        if not edges:
            continue

        gg = td.graph.IndexedRXGraph.from_geff(gt_path); gg = gg[0] if isinstance(gg, tuple) else gg
        gn = gg.node_attrs(); ge = gg.edge_attrs()
        gpos = {int(r["node_id"]): np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
                for r in gn.iter_rows(named=True)}
        gout = {}
        for r in ge.iter_rows(named=True):
            gout.setdefault(int(r["source_id"]), []).append(int(r["target_id"]))
        gt_div = [(gpos[s], gpos[k[0]], gpos[k[1]]) for s, k in gout.items()
                  if len(k) == 2 and s in gpos and k[0] in gpos and k[1] in gpos]

        for f, sid, qid, _t in D.collect_proposals(nodes, edges, cfg):
            src = D._pos(nodes[sid]); cand = D._pos(nodes[qid])
            # the already-linked child completes the triple
            kid = next((D._pos(nodes[int(e_["target_id"])]) for e_ in edges
                        if int(e_["source_id"]) == sid), None)
            if kid is None:
                continue
            pos = 0
            for P, A, B in gt_div:
                if np.linalg.norm(src - P) > TOL:
                    continue
                if ((np.linalg.norm(kid - A) <= TOL and np.linalg.norm(cand - B) <= TOL) or
                        (np.linalg.norm(kid - B) <= TOL and np.linalg.norm(cand - A) <= TOL)):
                    pos = 1
                    break
            X.append([f[k] for k in FEATS]); y.append(pos)
            emb.append(p.stem[:4]); ds.append(p.stem)
    return X, y, emb, ds


def loeo(X, y, emb, model="logistic"):
    X, y, emb = np.asarray(X, float), np.asarray(y, int), np.asarray(emb)
    out = {}
    for held in sorted(set(emb)):
        tr, te = emb != held, emb == held
        if te.sum() < 5 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = (LogisticRegression(max_iter=3000, class_weight="balanced") if model == "logistic"
             else GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=0))
        m.fit((X[tr] - mu) / sd, y[tr])
        s = m.predict_proba((X[te] - mu) / sd)[:, 1]
        out[held] = dict(auc=float(roc_auc_score(y[te], s)), n=int(te.sum()),
                         pos=int(y[te].sum()), mu=mu.tolist(), sd=sd.tolist(),
                         coef=getattr(m, "coef_", [[None]])[0].tolist()
                         if model == "logistic" else None,
                         intercept=float(m.intercept_[0]) if model == "logistic" else None)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="work/train_graphs",
                    help="directory of prediction .geff from kernels/gen-train-graphs")
    ap.add_argument("--out", default="src/biohub/division_recall_model.json")
    a = ap.parse_args()
    gd = Path(a.graphs)
    if not gd.exists() or not list(gd.rglob("*.geff")):
        raise SystemExit(f"no prediction graphs under {gd} -- run kernels/gen-train-graphs first")
    print(f"  harvesting from {gd} with WIDE gates...", flush=True)
    X, y, emb, ds = harvest(gd, wide_cfg())
    print(f"  candidates {len(y)}  positive {int(np.sum(y))}")
    for mdl in ("logistic", "grad-boost"):
        r = loeo(X, y, emb, mdl)
        if r:
            print(f"  {mdl:<11} " + "  ".join(f"{k} AUC {v['auc']:.3f} (n={v['n']})"
                                              for k, v in r.items()))
