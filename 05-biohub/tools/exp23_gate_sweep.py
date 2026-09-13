#!/usr/bin/env python3
"""EXP-23 -- sweep the division gates on the 0.947 model's own graphs, at real density.

This is the experiment the whole campaign has been building toward. EXP-21 priced their gates
against ground-truth geometry and found they discard 74.8% of real divisions; their own in-kernel
diagnostics confirmed 25% recall. But both were indirect. Here their actual linked graphs, at 203
cells per frame, are re-divided under different gate settings and scored against our ground truth.

Reported with PAIRED bootstrap over datasets: unpaired, 151 divisions resolve only +-0.0068 of
score, which is larger than most of what we are chasing.
"""
import sys, time
from pathlib import Path
import numpy as np, tracksdata as td, polars as pl

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
import div_sweep as D
from div_ci import paired_delta, bootstrap_divJ

GT = Path("data/train_geff")
GRAPHS = Path("work/train_graphs")
TOL = 7.0


def load_pred(p):
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs(); e = g.edge_attrs()
    nodes = {int(r["node_id"]): {"t": int(r["t"]), "z": float(r["z"]),
                                 "y": float(r["y"]), "x": float(r["x"])}
             for r in n.iter_rows(named=True)}
    edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"]),
              "edge_prob": r.get("edge_prob"), "distance_um": 0.0}
             for r in e.iter_rows(named=True)]
    return nodes, edges


def gt_divisions(stem):
    p = GT / f"{stem}.geff"
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs(); e = g.edge_attrs()
    pos = {int(r["node_id"]): np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
           for r in n.iter_rows(named=True)}
    out = {}
    for r in e.iter_rows(named=True):
        out.setdefault(int(r["source_id"]), []).append(int(r["target_id"]))
    return [(pos[s], pos[k[0]], pos[k[1]]) for s, k in out.items()
            if len(k) == 2 and s in pos and k[0] in pos and k[1] in pos]


def strip(nodes, edges):
    """One child per source: highest edge_prob, ties by shorter distance."""
    best = {}
    for e in edges:
        s = int(e["source_id"])
        pr = float(e["edge_prob"]) if e.get("edge_prob") is not None else 0.0
        d = float(e.get("distance_um") or 0.0)
        cur = best.get(s)
        if cur is None or (pr, -d) > (cur[0], -cur[1]):
            best[s] = (pr, d, e)
    return [v[2] for v in best.values()]


def confusion(nodes, added, gtdiv):
    """Match emitted divisions to GT divisions within the scorer's tolerance."""
    used = set(); tp = 0
    for e in added:
        s, q = int(e["source_id"]), int(e["target_id"])
        P = D._pos(nodes[s]); C = D._pos(nodes[q])
        hit = None
        for i, (gp, ga, gb) in enumerate(gtdiv):
            if i in used or np.linalg.norm(P - gp) > TOL:
                continue
            if min(np.linalg.norm(C - ga), np.linalg.norm(C - gb)) <= TOL:
                hit = i; break
        if hit is not None:
            used.add(hit); tp += 1
    return tp, len(added) - tp, len(gtdiv) - tp


CFGS = {
    "theirs (published)":  D.DivCfg(),
    "diverge OFF":         D.DivCfg(require_divergence=False, diverge_um=-99.0),
    "symmetry OFF":        D.DivCfg(symmetry_tau=0.0),
    "diverge+symmetry OFF": D.DivCfg(require_divergence=False, diverge_um=-99.0, symmetry_tau=0.0),
    "mutual-NN OFF":       D.DivCfg(require_mutual_nn=False),
}

files = sorted(GRAPHS.rglob("*.geff"))
print(f"  {len(files)} graphs, real density\n", flush=True)
CACHE = {}
for p in files:
    CACHE[p] = (load_pred(p), gt_divisions(p.stem), p.stem[:4])

results = {}
for name, cfg in CFGS.items():
    t0 = time.time(); per = []
    for p in files:
        (nodes, edges), gtdiv, _ = CACHE[p]
        base = strip(nodes, edges)
        out = D.add_safe_divisions(nodes, base, cfg)
        per.append(confusion(nodes, out[len(base):], gtdiv))
    results[name] = per
    a = np.array(per)
    tp, fp, fn = a.sum(0)
    pt, lo, hi = bootstrap_divJ(per)
    print(f"  {name:<22} tp {tp:>4} fp {fp:>5} fn {fn:>4}  divJ {pt:.4f} "
          f"[{lo:.4f},{hi:.4f}]  recall {tp/max(tp+fn,1):.1%}  {time.time()-t0:.0f}s", flush=True)

print("\n  === PAIRED against their published gates (same 199 datasets) ===")
base = results["theirs (published)"]
for name, per in results.items():
    if name == "theirs (published)":
        continue
    d, lo, hi, pw = paired_delta(base, per)
    print(f"  {name:<22} delta divJ {d:+.4f} [{lo:+.4f},{hi:+.4f}]  "
          f"score {0.1*d:+.4f}  P(not better) {pw:.3f}")
