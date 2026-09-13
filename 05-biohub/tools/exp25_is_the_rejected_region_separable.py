#!/usr/bin/env python3
"""EXP-25 -- is the gate-rejected region SEPARABLE at all, or is it noise?

Everything now rests on one question. Their gates discard 74.8% of real divisions, and the metric
will pay for those at better than 24% precision. But v-sym and v-recall showed the discarded
region yields candidates that are right only 7-12.5% of the time. A discriminator helps ONLY if
the true divisions inside that region are separable from the false ones.

If they are not -- if a real division that fails the symmetry test looks exactly like the
thousands of false candidates that also fail it -- then no model recovers them, the 74.8% is
permanently lost, and division work should stop.

This asks the question directly on ground truth, with no model: inside the REJECTED region only,
how different are true and false candidates? Reported as per-feature AUC, which is what a
classifier could exploit.
"""
import sys
from pathlib import Path
import numpy as np, tracksdata as td
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
import div_sweep as D

GT = Path("data/train_geff"); GRAPHS = Path("work/train_graphs"); TOL = 7.0
FEATS = ["parent_dist", "sister_dist", "child_dist", "cos", "diverge",
         "arc_max", "arc_min", "arc_asym", "arc_sum"]

THEIRS = D.DivCfg()                       # what they accept
WIDE = D.DivCfg(max_um=12.0, sister_max_um=20.0, existing_child_max_um=12.0,
                symmetry_tau=0.0, diverge_um=-99.0, require_divergence=False,
                require_mutual_nn=True, frame_frac_cap=1.0, global_frac_cap=1.0)


def passes_theirs(f):
    """Would their published gates have accepted this candidate?"""
    if f["parent_dist"] > THEIRS.max_um or f["sister_dist"] > THEIRS.sister_max_um:
        return False
    if f["child_dist"] > THEIRS.existing_child_max_um:
        return False
    den = max((f["child_dist"] + f["parent_dist"]) / 2.0, 1e-6)
    if abs(f["child_dist"] - f["parent_dist"]) / den > THEIRS.symmetry_tau:
        return False
    return f["diverge"] >= THEIRS.diverge_um


rows_acc, rows_rej = [], []
files = sorted(GRAPHS.rglob("*.geff"))
for i, p in enumerate(files):
    gp = GT / f"{p.stem}.geff"
    if not gp.exists():
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
    gg = td.graph.IndexedRXGraph.from_geff(gp); gg = gg[0] if isinstance(gg, tuple) else gg
    gn = gg.node_attrs(); ge = gg.edge_attrs()
    gpos = {int(r["node_id"]): np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
            for r in gn.iter_rows(named=True)}
    gout = {}
    for r in ge.iter_rows(named=True):
        gout.setdefault(int(r["source_id"]), []).append(int(r["target_id"]))
    gtd = [(gpos[s], gpos[k[0]], gpos[k[1]]) for s, k in gout.items()
           if len(k) == 2 and s in gpos and k[0] in gpos and k[1] in gpos]
    if not gtd:
        continue
    first_child = {}
    for e_ in edges:
        first_child.setdefault(int(e_["source_id"]), int(e_["target_id"]))
    for f, sid, qid, _t in D.collect_proposals(nodes, edges, WIDE):
        src = D._pos(nodes[sid]); cand = D._pos(nodes[qid])
        kc = first_child.get(sid)
        if kc is None:
            continue
        kid = D._pos(nodes[kc])
        lab = 0
        for P, A, B in gtd:
            if np.linalg.norm(src - P) > TOL:
                continue
            if ((np.linalg.norm(kid - A) <= TOL and np.linalg.norm(cand - B) <= TOL) or
                    (np.linalg.norm(kid - B) <= TOL and np.linalg.norm(cand - A) <= TOL)):
                lab = 1; break
        (rows_acc if passes_theirs(f) else rows_rej).append(
            ([f[k] for k in FEATS], lab))
    if (i + 1) % 40 == 0:
        print(f"    {i+1}/{len(files)}  accepted {len(rows_acc):,}  rejected {len(rows_rej):,}",
              flush=True)


def report(name, rows):
    if not rows:
        print(f"  {name}: empty"); return
    X = np.array([r[0] for r in rows], float); y = np.array([r[1] for r in rows], int)
    print(f"\n  === {name} ===")
    print(f"    candidates {len(y):,}   true divisions {y.sum()}   base rate {y.mean():.4%}")
    if y.sum() < 2 or y.sum() == len(y):
        print("    not enough positives to separate"); return
    aucs = []
    for i, f in enumerate(FEATS):
        a = roc_auc_score(y, X[:, i]); aucs.append((max(a, 1 - a), f))
    for a, f in sorted(aucs, reverse=True):
        print(f"    {f:<14} AUC {a:.3f}")
    print(f"    best single feature: {max(aucs)[1]} at {max(aucs)[0]:.3f}")


report("ACCEPTED by their gates", rows_acc)
report("REJECTED by their gates  <-- the 74.8%, and the whole question", rows_rej)
