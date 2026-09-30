"""Is the 44% pre-gate blockage reachable by a motion-relink parameter?

safe_div may only claim UNPARENTED nodes. 44% of true divisions fail because
the 1:1 Hungarian already gave the daughter to someone else, and 53.5% of those
incumbents are edges with no corresponding GT edge.

Motion relink runs two passes: tight 6.0 um, then relaxed 10.0 um on whatever
is still unmatched. If the thieving edges are predominantly RELAXED-pass, then
MOTION_RELINK_RELAXED_UM frees the daughter and safe_div can then claim it --
one environment variable, reaching a bigger population than every safe_div gate
combined.

The test is not just "is the thief far away". It is whether tightening would
ALSO have to be paid for elsewhere, so this counts both the true divisions
freed and the correct edges that the same change would destroy.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd
from biohub import metric as MT

G = Path("artifacts/graphs")
split = json.loads(Path("artifacts/loeo_split.json").read_text())
fold_of = {f: e for e, d in split["folds"].items() for f in d["held_out_films"]}
GRID, TIGHT = 1.625, 6.0

rows, edge_cost = [], []
for film in sorted(p.stem for p in G.glob("*.npz")):
    z = np.load(G / f"{film}.npz")
    ids = z["gt_ids"]; idx = {int(i): k for k, i in enumerate(ids)}
    gt_e = np.array([[idx[int(a)], idx[int(b)]] for a, b in z["gt_edges"]], np.int64).reshape(-1, 2)
    gt_set = {(int(a), int(b)) for a, b in gt_e}
    gt_out = {}
    for a, b in gt_e:
        gt_out.setdefault(int(a), []).append(int(b))
    le = z["linked_edges"]
    dt, dz = z["det_t"], z["det_zyx"].astype(np.float32)
    _, g2p = MT.match_nodes(z["gt_grid"], z["gt_t"], dz, dt)
    p2g = {v: k for k, v in g2p.items()}
    pos = dz.astype(np.float64) * GRID
    par = {int(b): (int(a), float(p), float(d)) for a, b, p, d in le}
    d = lambda a, b: float(np.linalg.norm(pos[a] - pos[b]))

    # cost side: how many CORRECT edges live in the relaxed band?
    for a, b, p, dist in le:
        ms, mt = p2g.get(int(a)), p2g.get(int(b))
        correct = ms is not None and mt is not None and (ms, mt) in gt_set
        edge_cost.append((float(dist), correct))

    divs = {g: c for g, c in gt_out.items() if len(c) >= 2}
    for g, ch in divs.items():
        if g not in g2p:
            continue
        P = g2p[g]
        for c in ch[:2]:
            if c not in g2p:
                continue
            D = g2p[c]
            inc = par.get(D)
            if inc is None or inc[0] == P:
                continue                      # orphan, or already correct
            thief, prob, dist = inc
            tg = p2g.get(thief)
            wrong = tg is None or (tg, c) not in gt_set
            rows.append(dict(film=film, fold=fold_of[film], thief_dist=dist,
                             thief_prob=prob, true_dist=d(P, D), wrong=wrong,
                             pass_="tight" if dist <= TIGHT else "relaxed"))

df = pd.DataFrame(rows)
df.to_csv("artifacts/daughter_theft.csv", index=False)
print(f"true daughters held by another parent: {len(df)}")
w = df[df.wrong]
print(f"  of which the incumbent edge is WRONG: {len(w)} ({len(w)/len(df):.1%})\n")

print("which motion-relink pass admitted the thieving edge?")
print(df.groupby(["wrong", "pass_"]).size().unstack(fill_value=0).to_string(), "\n")

print("geometry of the WRONG thieving edges (um):")
print(f"  thief distance : median {w.thief_dist.median():.2f}  p75 {w.thief_dist.quantile(.75):.2f}"
      f"  p90 {w.thief_dist.quantile(.9):.2f}  max {w.thief_dist.max():.2f}")
print(f"  true  distance : median {w.true_dist.median():.2f}  p75 {w.true_dist.quantile(.75):.2f}")
print(f"  thief is FARTHER than the true parent in {(w.thief_dist > w.true_dist).mean():.1%} of cases")
print(f"  thief learned prob: median {w.thief_prob.median():.3f}  p25 {w.thief_prob.quantile(.25):.3f}\n")

print("if MOTION_RELINK_RELAXED_UM were reduced, what is freed vs destroyed?")
ec = pd.DataFrame(edge_cost, columns=["dist", "correct"])
tot_correct = int(ec.correct.sum())
print(f"{'new relaxed cap':>16}{'true daughters freed':>22}{'correct edges lost':>20}{'ratio':>10}")
for cap in (10.0, 9.0, 8.0, 7.0, 6.5, 6.0):
    freed = int(((w.thief_dist > cap)).sum())
    lost = int(ec[(ec.dist > cap) & ec.correct].shape[0])
    print(f"{cap:>16.1f}{freed:>22}{lost:>20}{(freed/max(lost,1)):>10.4f}")
print(f"\n(total correct edges in the graph: {tot_correct:,})")
print("\nper fold (wrong thefts):")
print(w.groupby("fold").size().to_string())
