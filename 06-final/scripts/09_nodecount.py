"""Is the node-count term a precision opportunity or the forbidden lever?

Two DIFFERENT things share one formula:
  (a) deleting true tracks to push n_pred BELOW n_est, farming a multiplier > 1.
      Published post-mortem: +0.013 offline, -0.004 on the board. Forbidden.
  (b) removing genuine false positives to move n_pred/n_est from 1.287 toward
      1.0. That is detection precision, and it is not the same thing.

The ceiling adj_J at threshold t is reachability(t) x multiplier(t): raising t
cuts nodes (multiplier up) and loses GT edges (J down). The product has a max.
Cheap, because it needs matching only -- no relinking.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd
from biohub import metric as MT

G = Path("artifacts/graphs")
CACHE = Path("artifacts/cache")
TAUS = [0.965, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999]
split = json.loads(Path("artifacts/loeo_split.json").read_text())
fold_of = {f: e for e, d in split["folds"].items() for f in d["held_out_films"]}

rows = []
films = sorted(p.stem for p in G.glob("*.npz"))
t0 = time.perf_counter()
for n, film in enumerate(films):
    zc = np.load(CACHE / f"{film}.npz")
    zg = np.load(G / f"{film}.npz")
    det_p = zc["det_p"].astype(np.float32)
    order = np.argsort(zc["det_t"], kind="stable")      # graphs store sorted order
    det_p = det_p[order]
    det_t, det_zyx = zg["det_t"], zg["det_zyx"].astype(np.float32)
    gt_t, gt_grid, n_est = zg["gt_t"], zg["gt_grid"], float(zg["est_nodes"])
    ids = zc["gt_ids"]; idx = {int(i): k for k, i in enumerate(ids)}
    gt_e = np.array([[idx[int(a)], idx[int(b)]] for a, b in zc["gt_edges"]], np.int64).reshape(-1, 2)

    for tau in TAUS:
        keep = det_p >= tau
        if keep.sum() == 0:
            continue
        dt, dz = det_t[keep], det_zyx[keep]
        _, g2p = MT.match_nodes(gt_grid, gt_t, dz, dt)
        both = sum(1 for a, b in gt_e if int(a) in g2p and int(b) in g2p)
        n_pred = int(keep.sum())
        J_ceil = both / max(len(gt_e), 1)
        mult = 1.0 - 0.1 * (n_pred - n_est) / n_est
        rows.append({"film": film, "fold": fold_of[film], "tau": tau,
                     "n_pred": n_pred, "n_est": n_est, "ratio": n_pred / n_est,
                     "node_recall": len(g2p) / max(len(gt_t), 1),
                     "J_ceiling": J_ceil, "mult": mult,
                     "adjJ_ceiling": max(0.0, J_ceil * mult),
                     "weight": len(gt_e)})
    if (n + 1) % 50 == 0:
        print(f"  {n+1}/{len(films)}  {time.perf_counter()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv("artifacts/nodecount_sweep.csv", index=False)

def agg(g):
    w = g.weight
    return pd.Series({
        "ratio": np.average(g.ratio, weights=w),
        "node_recall": np.average(g.node_recall, weights=w),
        "J_ceiling": np.average(g.J_ceiling, weights=w),
        "mult": np.average(g["mult"], weights=w),
        "adjJ_ceiling": np.average(g.adjJ_ceiling, weights=w),
        "films_below_nest": int((g.ratio < 1.0).sum()),
    })

pd.set_option("display.width", 200)
for fold in ["ALL"] + sorted(set(fold_of.values())):
    sub = df if fold == "ALL" else df[df.fold == fold]
    t = sub.groupby("tau").apply(agg, include_groups=False)
    print(f"\n=== {fold} ({sub.film.nunique()} films) ===")
    print(t.to_string(float_format=lambda v: f"{v:.4f}"))
    best = t.adjJ_ceiling.idxmax()
    print(f"  max ceiling at tau={best}: adjJ={t.loc[best,'adjJ_ceiling']:.4f} "
          f"(ratio {t.loc[best,'ratio']:.3f}, recall {t.loc[best,'node_recall']:.4f})")

print("\n=== per-film ratio at the deployed threshold 0.965 ===")
d0 = df[df.tau == 0.965]
print(d0.groupby("fold").ratio.describe(percentiles=[.1, .5, .9]).to_string(
    float_format=lambda v: f"{v:.3f}"))
