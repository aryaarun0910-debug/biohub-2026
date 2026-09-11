#!/usr/bin/env python3
"""EXPERIMENT 1 — division census over all 199 ground-truth graphs.

Pure GT. No model, no predictions, no images. Establishes the target geometry every
later division experiment aims at, and independently checks the dense-zebrahub numbers
against the actual competition data.
"""
import json, math, sys
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td
from geff import GeffMetadata

GT = Path("data/train_geff"); SCALE = np.array([1.625, 0.40625, 0.40625])

def load(p):
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs(); e = g.edge_attrs()
    return n, e

def main():
    div_rows, cont_rows = [], []
    per_ds, skipped = [], 0
    files = sorted(GT.glob("*.geff"))
    for i, p in enumerate(files, 1):
        try:
            n, e = load(p)
        except Exception as ex:
            skipped += 1; continue
        ncols = n.columns
        idc = "node_id" if "node_id" in ncols else ncols[0]
        pos = {r[idc]: np.array([r["t"], r["z"], r["y"], r["x"]], dtype=float)
               for r in n.iter_rows(named=True)}
        src = "source_id" if "source_id" in e.columns else e.columns[0]
        tgt = "target_id" if "target_id" in e.columns else e.columns[1]
        children = {}
        parents = {}
        for r in e.iter_rows(named=True):
            children.setdefault(r[src], []).append(r[tgt]); parents[r[tgt]] = r[src]
        nd = sum(1 for v in children.values() if len(v) >= 2)
        per_ds.append({"ds": p.stem, "nodes": len(pos), "edges": len(parents), "divisions": nd})
        for par, kids in children.items():
            if par not in pos: continue
            P = pos[par][1:] * SCALE
            if len(kids) == 1:
                k = kids[0]
                if k in pos: cont_rows.append(float(np.linalg.norm(pos[k][1:]*SCALE - P)))
            elif len(kids) >= 2:
                ks = [k for k in kids if k in pos][:2]
                if len(ks) < 2: continue
                D = [pos[k][1:]*SCALE for k in ks]
                v = [d - P for d in D]
                d0, d1 = float(np.linalg.norm(v[0])), float(np.linalg.norm(v[1]))
                sis = float(np.linalg.norm(D[0] - D[1]))
                nz = np.linalg.norm(v[0]) * np.linalg.norm(v[1])
                cos = float(np.dot(v[0], v[1]) / nz) if nz > 1e-9 else float("nan")
                # divergence persistence: grandchildren separation at t+2
                gsep = float("nan")
                gk = [children.get(k, []) for k in ks]
                if all(len(x) >= 1 for x in gk) and all(x[0] in pos for x in gk):
                    gsep = float(np.linalg.norm(pos[gk[0][0]][1:]*SCALE - pos[gk[1][0]][1:]*SCALE))
                # parent's own incoming step
                pstep = float("nan")
                gp = parents.get(par)
                if gp in pos: pstep = float(np.linalg.norm(P - pos[gp][1:]*SCALE))
                div_rows.append(dict(ds=p.stem, near=min(d0,d1), far=max(d0,d1), sister=sis,
                                     cos=cos, gsep=gsep, pstep=pstep, n_kids=len(kids)))
        if i % 50 == 0: print(f"  ...{i}/{len(files)}", flush=True)

    D = pl.DataFrame(div_rows); C = np.array(cont_rows); S = pl.DataFrame(per_ds)
    print(f"\n=== CORPUS ({len(files)} datasets, {skipped} unreadable) ===")
    print(f"  nodes {S['nodes'].sum():,}  edges {S['edges'].sum():,}  divisions {S['divisions'].sum()}")
    print(f"  datasets with zero divisions: {(S['divisions']==0).sum()}/{S.height}")
    print(f"  max children at a fork: {D['n_kids'].max()}")
    e44 = S.filter(pl.col('ds').str.starts_with('44b6')); e6b = S.filter(pl.col('ds').str.starts_with('6bba'))
    print(f"  44b6: {e44.height} datasets, {e44['divisions'].sum()} divisions | "
          f"6bba: {e6b.height} datasets, {e6b['divisions'].sum()} divisions")

    def q(a, name, unit="um"):
        a = np.asarray([x for x in a if x == x])
        if not len(a): print(f"  {name}: none"); return
        print(f"  {name:<26} n={len(a):>6}  median {np.median(a):6.3f}  "
              f"IQR {np.percentile(a,25):6.3f}-{np.percentile(a,75):6.3f}  "
              f"p90 {np.percentile(a,90):6.3f}  max {a.max():7.3f} {unit}")
    print("\n=== GEOMETRY ===")
    q(C, "continuation step")
    q(D["near"], "parent->nearer daughter")
    q(D["far"],  "parent->farther daughter")
    q(D["sister"], "sister separation")
    q(D["gsep"], "grandchild separation t+2")
    q(D["pstep"], "parent's own incoming step")
    q(D["cos"], "cos(angle) between arcs", "")

    print("\n=== BASE RATE: continuations vs divisions above each threshold ===")
    far = D["far"].to_numpy()
    for thr in (5,7,9,10,12,14):
        nc = int((C > thr).sum()); nd = int((far > thr).sum())
        print(f"  >{thr:>2} um   continuations {nc:>6}   divisions {nd:>4}   ratio {nc/max(nd,1):>7.1f}:1")

    print("\n=== GATE ADMISSIBILITY (fraction of GT divisions passing) ===")
    sis = D["sister"].to_numpy(); near = D["near"].to_numpy(); n = len(D)
    for pg, sg in [(4.7,7.2),(9.0,14.0),(10,15),(12,18)]:
        ok = int(((near<=pg) & (sis<=sg)).sum())
        print(f"  parent<={pg:<5} sister<={sg:<5}  {ok:>4}/{n}  {ok/n*100:5.1f}%")
    D.write_csv("data/ablation/exp1_divisions.csv"); S.write_csv("data/ablation/exp1_per_dataset.csv")
    print("\n  wrote data/ablation/exp1_divisions.csv and exp1_per_dataset.csv")

if __name__ == "__main__":
    main()
