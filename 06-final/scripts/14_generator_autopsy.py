"""Why does the fork generator find only ~15% of labelled divisions?

For every labelled division, walk the gates IN ORDER and record which one
killed it. This is the per-event cause of death the ledger schema was built
for: "the generator is the bottleneck" is only actionable once you know which
gate does the killing.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd
from biohub import metric as MT, postprocess as PP, division as DV

G = Path("artifacts/graphs")
split = json.loads(Path("artifacts/loeo_split.json").read_text())
fold_of = {f: e for e, d in split["folds"].items() for f in d["held_out_films"]}
films = sorted(p.stem for p in G.glob("*.npz"))
cfg = DV.DivConfig(min_track_len=6)
GRID = 1.625

rows = []
for film in films:
    z = np.load(G / f"{film}.npz")
    ids = z["gt_ids"]; idx = {int(i): k for k, i in enumerate(ids)}
    gt_e = np.array([[idx[int(a)], idx[int(b)]] for a, b in z["gt_edges"]], np.int64).reshape(-1, 2)
    gt_out = {}
    for a, b in gt_e:
        gt_out.setdefault(int(a), []).append(int(b))
    divs = {g: c for g, c in gt_out.items() if len(c) >= 2}
    if not divs:
        continue

    e, dt, dz, _, _ = PP.prune_and_filter(
        [(int(a), int(b)) for a, b, *_ in z["linked_edges"]],
        z["det_t"], z["det_zyx"].astype(np.float32), min_track_len=6)
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
        c1, c2 = ch[0], ch[1]
        r = {"film": film, "fold": fold_of[film], "gt_parent": g, "cause": None}
        if g not in g2p:
            r["cause"] = "parent not detected"; rows.append(r); continue
        if c1 not in g2p or c2 not in g2p:
            r["cause"] = "daughter not detected"; rows.append(r); continue
        P, D1, D2 = g2p[g], g2p[c1], g2p[c2]
        kids = succ.get(P, [])
        # which daughter did the linker already take?
        if len(kids) == 0:
            r["cause"] = "parent has NO child in graph"; rows.append(r); continue
        if len(kids) >= 2:
            r["cause"] = "already a fork"; rows.append(r); continue
        C = kids[0]
        Q = D2 if C == D1 else (D1 if C == D2 else None)
        if Q is None:
            r["cause"] = "linker child is neither daughter"; rows.append(r); continue
        r.update(d_pc=d(P, C), d_pq=d(P, Q), d_cq=d(C, Q))
        if indeg.get(Q, 0) != 0:
            r["cause"] = "true daughter already has a parent"; rows.append(r); continue
        if d(P, C) > cfg.existing_child_max_um:
            r["cause"] = "existing-child distance"; rows.append(r); continue
        # the mutual-NN step: generator only ever considers the nearest orphan to C
        orphans = [j for j in by_t.get(int(dt[C]), []) if indeg.get(j, 0) == 0]
        if not orphans:
            r["cause"] = "no orphan in frame"; rows.append(r); continue
        nn = orphans[int(np.argmin(np.linalg.norm(pos[orphans] - pos[C], axis=1)))]
        r["n_orphans"] = len(orphans)
        if nn != Q:
            r["cause"] = "MUTUAL-NN: nearest orphan is not the daughter"
            r["nn_dist"] = d(C, nn); rows.append(r); continue
        if d(P, Q) > cfg.parent_max_um:
            r["cause"] = "parent-daughter gate"; rows.append(r); continue
        if not (cfg.sister_min_um <= d(C, Q) <= cfg.sister_max_um):
            r["cause"] = "sister gate"; rows.append(r); continue
        if abs(d(P, C) - d(P, Q)) / max((d(P, C) + d(P, Q)) / 2, 1e-9) > cfg.symmetry_tau:
            r["cause"] = "symmetry gate"; rows.append(r); continue
        if min(DV._chain_len(C, succ), DV._chain_len(Q, succ)) < cfg.min_track_len:
            r["cause"] = "CONSEQUENCE: track too short"; rows.append(r); continue
        sc, sq = succ.get(C, []), succ.get(Q, [])
        if len(sc) != 1 or len(sq) != 1:
            r["cause"] = "CONSEQUENCE: no unique grandchild"; rows.append(r); continue
        if d(sc[0], sq[0]) - d(C, Q) < cfg.diverge_um:
            r["cause"] = "CONSEQUENCE: divergence"; rows.append(r); continue
        r["cause"] = "REACHED (proposable)"
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv("artifacts/generator_autopsy.csv", index=False)
n = len(df)
print(f"labelled divisions examined: {n}\n")
c = df.cause.value_counts()
print(f"{'cause of death':<46} {'n':>4} {'%':>6}  {'cum%':>6}")
cum = 0
for k, v in c.items():
    cum += v
    print(f"{k:<46} {v:>4} {100*v/n:>5.1f}% {100*cum/n:>6.1f}%")
print(f"\nper fold:")
print(pd.crosstab(df.cause, df.fold).to_string())
if "n_orphans" in df:
    o = df.n_orphans.dropna()
    print(f"\norphans available in the daughter's frame: median {o.median():.0f}, "
          f"p90 {o.quantile(.9):.0f}, max {o.max():.0f}")
mn = df[df.cause.str.startswith("MUTUAL-NN", na=False)]
if len(mn):
    print(f"mutual-NN failures: {len(mn)}; the true daughter sits {mn.d_cq.median():.2f} um "
          f"from C while the nearest orphan sits {mn.nn_dist.median():.2f} um")
