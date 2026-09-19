"""Close the baseline gap: gap closing, pruning, short-track filtering.

Pure CPU on the cached graphs. The target is the unparented-node pool, which is
what poisons division candidates. Node count is reported every time: removing
isolated nodes and fragments is precision; pushing n_pred below n_est is
prohibition #1.
"""
import sys, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd
from biohub import metric as MT, postprocess as PP

G = Path("artifacts/graphs")
split = json.loads(Path("artifacts/loeo_split.json").read_text())
fold_of = {f: e for e, d in split["folds"].items() for f in d["held_out_films"]}
films = sorted(p.stem for p in G.glob("*.npz"))


def load(f):
    z = np.load(G / f"{f}.npz")
    ids = z["gt_ids"]; idx = {int(i): k for k, i in enumerate(ids)}
    gt_e = np.array([[idx[int(a)], idx[int(b)]] for a, b in z["gt_edges"]], np.int64).reshape(-1, 2)
    return dict(det_t=z["det_t"], det_zyx=z["det_zyx"].astype(np.float32),
                gt_t=z["gt_t"], gt_grid=z["gt_grid"], n_est=float(z["est_nodes"]),
                edges=[(int(a), int(b)) for a, b, *_ in z["linked_edges"]], gt_e=gt_e)

D = {f: load(f) for f in films}


def run(gap_um, min_len, prune, label):
    rows, orph, tot = [], 0, 0
    for f in films:
        d = D[f]
        e, dt, dz = d["edges"], d["det_t"], d["det_zyx"]
        if gap_um > 0:
            e, dt, dz, _ = PP.close_gaps(e, dt, dz, max_um=gap_um)
        if min_len > 0 or prune:
            e, dt, dz, _, _ = PP.prune_and_filter(e, dt, dz, min_track_len=min_len,
                                                  prune_isolated=prune)
        ind = {}
        for s, t in e:
            ind[t] = ind.get(t, 0) + 1
        orph += sum(1 for i, t in enumerate(dt) if t > 0 and ind.get(i, 0) == 0)
        tot += len(dt)
        rows.append(MT.score_film(list(range(len(dt))), e, dz, dt,
                                  d["gt_grid"], d["gt_t"], d["gt_e"], d["n_est"]))
    a = MT.aggregate(rows)
    ratio = np.mean([r["n_pred"] / r["n_est"] for r in rows])
    below = sum(1 for r in rows if r["n_pred"] < r["n_est"])
    per = {fd: MT.aggregate([rows[i] for i, f in enumerate(films) if fold_of[f] == fd])
           for fd in sorted(set(fold_of.values()))}
    return dict(label=label, score=a["score"], adjJ=a["adj_J_edge"], J=a["J_edge"],
                mult=a["multiplier"], ratio=ratio, below_nest=below,
                orphan_pct=100 * orph / tot,
                **{f"{k}_score": v["score"] for k, v in per.items()})


cfgs = [
    (0.0,  0, False, "baseline"),
    (0.0,  0, True,  "prune isolated"),
    (0.0,  6, True,  "prune + minlen6"),
    (8.0,  0, False, "gap8"),
    (8.0,  6, True,  "gap8 + prune + minlen6"),
    (10.0, 6, True,  "gap10 + prune + minlen6"),
    (12.0, 6, True,  "gap12 + prune + minlen6"),
    (10.0, 4, True,  "gap10 + prune + minlen4"),
    (10.0, 8, True,  "gap10 + prune + minlen8"),
]
res = [run(*c) for c in cfgs]
df = pd.DataFrame(res); df.to_csv("artifacts/baseline_repair.csv", index=False)
b = res[0]
pd.set_option("display.width", 220)
print(f"{'config':<26} {'score':>7} {'d':>8} {'adjJ':>7} {'J':>7} {'mult':>6} "
      f"{'ratio':>6} {'<n_est':>7} {'orphan%':>8} {'44b6':>7} {'6bba':>7}")
for r in res:
    print(f"{r['label']:<26} {r['score']:>7.4f} {r['score']-b['score']:>+8.4f} "
          f"{r['adjJ']:>7.4f} {r['J']:>7.4f} {r['mult']:>6.4f} {r['ratio']:>6.3f} "
          f"{r['below_nest']:>7} {r['orphan_pct']:>8.2f} "
          f"{r['44b6_score']:>7.4f} {r['6bba_score']:>7.4f}")
