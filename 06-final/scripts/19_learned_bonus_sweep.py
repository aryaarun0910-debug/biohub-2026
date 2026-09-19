"""Does MOTION_RELINK_LEARNED_BONUS stop true daughters being stolen?

Thieving edges have learned prob median 0.538 (p25 0.243) while sitting at a
median 3.81 um -- distance does NOT separate them, the model's own probability
does. The relink cost is

    C_ij = ||x_j - xhat_i|| + 0.05*||x_j - x_i|| - beta * p_ij

so beta (MOTION_RELINK_LEARNED_BONUS, deployed 1.0) trades micrometres against
probability. Raising it should let a confident long edge beat a doubtful short
one -- which is exactly the division case.

Reports, per beta: edge Jaccard, thefts of true daughters, and how many
divisions become PROPOSABLE (daughter unparented and parent out-degree 1).
This is an env var the deployed pipeline already exposes and that sits in
PP_SWEEP_KEYS, so a win here is one line.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd, torch
from biohub import io, model as M, edges as E, metric as MT

DEV = "mps"
CACHE, G = Path("artifacts/cache"), Path("artifacts/graphs")
split = json.loads(Path("artifacts/loeo_split.json").read_text())
fold_of = {f: e for e, d in split["folds"].items() for f in d["held_out_films"]}
BETAS = [float(x) for x in (sys.argv[1:] or ["1.0", "2.0", "4.0", "8.0"])]
films = sorted(p.stem for p in G.glob("*.npz"))
prim, _ = M.load("primary", DEV)

# cache the per-frame-pair probability matrices once; the sweep is then free
print("caching probability matrices ...", flush=True)
t0 = time.perf_counter()
PROB = {}
for n, film in enumerate(films):
    zc = np.load(CACHE / f"{film}.npz")
    order = np.argsort(zc["det_t"], kind="stable")
    dt, dz, df = zc["det_t"][order], zc["det_zyx"][order].astype(np.float32), zc["det_feat"][order]
    by_t = {int(t): np.where(dt == t)[0] for t in np.unique(dt)}
    mats = {}
    for t in sorted(by_t):
        si, ti = by_t.get(t), by_t.get(t + 1)
        if si is None or ti is None or not len(si) or not len(ti):
            continue
        lg = E.edge_logits(prim, torch.from_numpy(df[si]).to(DEV),
                           torch.from_numpy(df[ti]).to(DEV), dz[si], dz[ti], DEV)
        mats[t] = (si, ti, torch.softmax(torch.from_numpy(lg), 0).numpy().astype(np.float16))
    PROB[film] = (dt, dz, mats)
    if (n + 1) % 50 == 0:
        print(f"  {n+1}/{len(films)}  {time.perf_counter()-t0:.0f}s", flush=True)
print(f"cached in {(time.perf_counter()-t0)/60:.1f} min\n", flush=True)

GT = {}
for film in films:
    z = np.load(G / f"{film}.npz")
    ids = z["gt_ids"]; idx = {int(i): k for k, i in enumerate(ids)}
    gt_e = np.array([[idx[int(a)], idx[int(b)]] for a, b in z["gt_edges"]], np.int64).reshape(-1, 2)
    gt_out = {}
    for a, b in gt_e:
        gt_out.setdefault(int(a), []).append(int(b))
    GT[film] = (z["gt_t"], z["gt_grid"], float(z["est_nodes"]), gt_e, gt_out)

print(f"{'beta':>6}{'score':>9}{'adjJ':>9}{'J':>9}{'thefts':>8}{'proposable':>12}{'44b6':>9}{'6bba':>9}")
res = []
for beta in BETAS:
    rows, thefts, proposable = [], 0, 0
    for film in films:
        dt, dz, mats = PROB[film]
        gt_t, gt_grid, n_est, gt_e, gt_out = GT[film]
        e, prev = [], {}
        for t in sorted(mats):
            si, ti, prob = mats[t]
            s_um, t_um = dz[si] * E.GRID_UM, dz[ti] * E.GRID_UM
            pred = s_um.copy()
            for k, g in enumerate(si):
                p = prev.get(int(g))
                if p is not None:
                    pred[k] = s_um[k] + E.VELOCITY_WEIGHT * (s_um[k] - dz[p] * E.GRID_UM)
            for i, j, _, _, _ in E.link_frame_pair(s_um, t_um, prob.astype(np.float32),
                                                   pred, beta=beta):
                e.append((int(si[i]), int(ti[j]))); prev[int(ti[j])] = int(si[i])
        rows.append(MT.score_film(list(range(len(dt))), e, dz, dt, gt_grid, gt_t, gt_e, n_est))
        _, g2p = MT.match_nodes(gt_grid, gt_t, dz, dt)
        par, succ = {}, {}
        for s, tt in e:
            par[tt] = s; succ.setdefault(s, []).append(tt)
        for g, ch in gt_out.items():
            if len(ch) < 2 or g not in g2p:
                continue
            P = g2p[g]
            for c in ch[:2]:
                if c in g2p and par.get(g2p[c]) not in (None, P):
                    thefts += 1
            if len(succ.get(P, ())) == 1:
                kid = succ[P][0]
                other = [g2p[c] for c in ch[:2] if c in g2p and g2p[c] != kid]
                if other and par.get(other[0]) is None:
                    proposable += 1
    a = MT.aggregate(rows)
    per = {fd: MT.aggregate([rows[i] for i, f in enumerate(films) if fold_of[f] == fd])
           for fd in sorted(set(fold_of.values()))}
    print(f"{beta:>6.1f}{a['score']:>9.4f}{a['adj_J_edge']:>9.4f}{a['J_edge']:>9.4f}"
          f"{thefts:>8}{proposable:>12}{per['44b6']['score']:>9.4f}{per['6bba']['score']:>9.4f}",
          flush=True)
    res.append(dict(beta=beta, score=a["score"], adjJ=a["adj_J_edge"], J=a["J_edge"],
                    thefts=thefts, proposable=proposable))
pd.DataFrame(res).to_csv("artifacts/learned_bonus_sweep.csv", index=False)
