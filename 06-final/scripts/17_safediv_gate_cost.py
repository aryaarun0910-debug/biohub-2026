"""Price the DEPLOYED pipeline's own safe_div gates against all 151 labelled divisions.

Deployed stage order (from the documented notebook) is:
  motion relink -> single-parent repair -> gap closing -> gap2 -> SAFE DIVISION
  -> prune isolated -> short-track filter
so safe_div sees the UNPRUNED post-relink graph. My earlier autopsy ran it
after pruning, which is the wrong configuration and inflated "daughter not
detected" from ~7 to 27.

Deployed values being priced:
  SAFE_DIV_MAX_UM 9.0 | SISTER_MAX 14.0 | SYMMETRY_TAU 0.6 | DIVERGE_UM 2.25
  EXISTING_CHILD_MAX 10.0 | REQUIRE_MUTUAL_NN on

Reports MARGINAL cost (true divisions this gate alone rejects) and CUMULATIVE
recall as gates are relaxed one at a time, ranked by recall recovered.

RECALL cost transfers to the deployed pipeline -- it is a property of real
division geometry and a shared detector. PRECISION cost does NOT: it depends on
the orphan pool, which is cleaner there. The board measures precision, one
change per submission.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd
from biohub import metric as MT

G = Path("artifacts/graphs")
split = json.loads(Path("artifacts/loeo_split.json").read_text())
fold_of = {f: e for e, d in split["folds"].items() for f in d["held_out_films"]}
GRID = 1.625
DEPLOY = dict(parent_max=9.0, sister_max=14.0, symmetry=0.6, diverge=2.25,
              existing_child_max=10.0, mutual_nn=True)

rows = []
for film in sorted(p.stem for p in G.glob("*.npz")):
    z = np.load(G / f"{film}.npz")
    ids = z["gt_ids"]; idx = {int(i): k for k, i in enumerate(ids)}
    gt_e = np.array([[idx[int(a)], idx[int(b)]] for a, b in z["gt_edges"]], np.int64).reshape(-1, 2)
    gt_out = {}
    for a, b in gt_e:
        gt_out.setdefault(int(a), []).append(int(b))
    divs = {g: c for g, c in gt_out.items() if len(c) >= 2}
    if not divs:
        continue
    # UNPRUNED graph -- what safe_div actually sees
    e = [(int(a), int(b)) for a, b, *_ in z["linked_edges"]]
    dt, dz = z["det_t"], z["det_zyx"].astype(np.float32)
    _, g2p = MT.match_nodes(z["gt_grid"], z["gt_t"], dz, dt)
    pos = dz.astype(np.float64) * GRID
    succ, indeg = {}, {}
    for s, t in e:
        succ.setdefault(s, []).append(t); indeg[t] = indeg.get(t, 0) + 1
    by_t = {}
    for i, t in enumerate(dt):
        by_t.setdefault(int(t), []).append(i)
    d = lambda a, b: float(np.linalg.norm(pos[a] - pos[b]))

    for g, ch in divs.items():
        r = dict(film=film, fold=fold_of[film])
        if g not in g2p or ch[0] not in g2p or ch[1] not in g2p:
            r["blocked_by"] = "detection"; rows.append(r); continue
        P, D1, D2 = g2p[g], g2p[ch[0]], g2p[ch[1]]
        kids = succ.get(P, [])
        if len(kids) != 1:
            r["blocked_by"] = "parent out-degree != 1"; rows.append(r); continue
        C = kids[0]
        Q = D2 if C == D1 else (D1 if C == D2 else None)
        if Q is None:
            r["blocked_by"] = "linker child is neither daughter"; rows.append(r); continue
        if indeg.get(Q, 0) != 0:
            r["blocked_by"] = "daughter held by another parent"; rows.append(r); continue
        dpc, dpq, dcq = d(P, C), d(P, Q), d(C, Q)
        orph = [j for j in by_t.get(int(dt[C]), []) if indeg.get(j, 0) == 0]
        nn = orph[int(np.argmin(np.linalg.norm(pos[orph] - pos[C], axis=1)))] if orph else None
        sc, sq = succ.get(C, []), succ.get(Q, [])
        div_ok = (len(sc) == 1 and len(sq) == 1 and d(sc[0], sq[0]) - dcq >= DEPLOY["diverge"])
        r.update(reachable=True, d_pc=dpc, d_pq=dpq, d_cq=dcq,
                 g_existing=dpc <= DEPLOY["existing_child_max"],
                 g_parent=dpq <= DEPLOY["parent_max"],
                 g_sister=dcq <= DEPLOY["sister_max"],
                 g_symmetry=abs(dpc - dpq) / max((dpc + dpq) / 2, 1e-9) <= DEPLOY["symmetry"],
                 g_mutual_nn=(nn == Q),
                 g_diverge=div_ok,
                 blocked_by=None)
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv("artifacts/safediv_gate_cost.csv", index=False)
N = len(df)
print(f"labelled divisions: {N}\n")
pre = df[df.blocked_by.notna()].blocked_by.value_counts()
print("blocked BEFORE any safe_div gate (deployed order, unpruned graph):")
for k, v in pre.items():
    print(f"  {k:<40} {v:>4}  {100*v/N:>5.1f}%")
reach = df[df.blocked_by.isna()].copy()
for _g in ["g_existing","g_parent","g_sister","g_symmetry","g_mutual_nn","g_diverge"]:
    reach[_g] = reach[_g].astype(bool)
print(f"\nreach the gates: {len(reach)}/{N} = {len(reach)/N:.1%}\n")

GATES = ["g_existing", "g_parent", "g_sister", "g_symmetry", "g_mutual_nn", "g_diverge"]
allpass = reach[GATES].all(axis=1)
print(f"pass ALL deployed gates: {allpass.sum()} = {allpass.sum()/N:.1%} of all divisions\n")
print(f"{'gate':<16}{'deployed':>10}{'rejects':>9}{'% of all':>10}{'recall if relaxed':>19}")
for gcol in GATES:
    rej = (~reach[gcol]).sum()
    relaxed = reach[[c for c in GATES if c != gcol]].all(axis=1).sum()
    val = {"g_existing": "10.0um", "g_parent": "9.0um", "g_sister": "14.0um",
           "g_symmetry": "tau 0.6", "g_mutual_nn": "on", "g_diverge": "2.25um"}[gcol]
    print(f"{gcol[2:]:<16}{val:>10}{rej:>9}{100*rej/N:>9.1f}%"
          f"{f'{allpass.sum()} -> {relaxed}':>19}")
print(f"\nrelax ALL gates: {len(reach)} proposable = {len(reach)/N:.1%}")
print("\npairwise: relax BOTH of the two that matter:")
_both = reach[[c for c in GATES if c not in ("g_symmetry","g_diverge")]].all(axis=1).sum()
print(f"  symmetry + diverge together: {allpass.sum()} -> {_both}  ({100*_both/N:.1f}% of all divisions)")
print("\nper fold (pass all deployed gates):")
print(reach.assign(pass_all=allpass).groupby("fold").pass_all.agg(["sum", "size"]).to_string())
print("\ngeometry of divisions BLOCKED by each gate (median um):")
for gcol, col in (("g_parent", "d_pq"), ("g_sister", "d_cq")):
    b = reach[~reach[gcol]]
    if len(b):
        print(f"  {gcol[2:]:<12} n={len(b):<3} {col}: median {b[col].median():.2f}, "
              f"p90 {b[col].quantile(.9):.2f}, max {b[col].max():.2f}")
s = reach[~reach.g_symmetry]
if len(s):
    asym = (abs(s.d_pc - s.d_pq) / ((s.d_pc + s.d_pq) / 2))
    print(f"  symmetry     n={len(s):<3} asymmetry: median {asym.median():.2f}, "
          f"p90 {asym.quantile(.9):.2f}, max {asym.max():.2f}")
