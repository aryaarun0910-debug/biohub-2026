"""Does the learned-bonus gain survive on deployed-quality probabilities?

beta was swept on single-view primary probabilities. The deployed pipeline uses
8-view TTA + dual-seed, which my ablation showed is worth +0.0314 on its own.
If part of that gain is ALSO "better probabilities -> better assignment", then
beta and TTA overlap and the +0.0096 shrinks on a real graph.

Tests beta = 1 vs 16 under both probability regimes on the same films, so the
interaction is measured rather than assumed.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd, torch
from biohub import io, model as M, edges as E, metric as MT, postprocess as PP, ensemble as EN

DEV = "mps"
N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 15
BETAS = [1.0, 4.0, 16.0]
tbl = pd.read_csv("artifacts/films.csv").set_index("film")
sel = []
for emb, g in tbl.groupby("embryo"):
    sel += list(g.sort_values(["divisions", "labelled_nodes"], ascending=False).index[:N_PER])
print(f"{len(sel)} films", flush=True)

prim, cfg = M.load("primary", DEV)
seco, _ = M.load("secondary", DEV)
rows = {(r, b): [] for r in ("single", "tta_dual") for b in BETAS}
t0 = time.perf_counter()

for n, film in enumerate(sel):
    zp = io.dataset_root() / "train" / f"{film}.zarr"
    q = io.quantiles(zp); ql, qh = float(q["0.001"]), float(q["0.999"])
    T = io.image_meta(zp)["shape"][0]
    zc = np.load(f"artifacts/cache/{film}.npz")
    gt_t, gt_grid, n_est = zc["gt_t"], zc["gt_grid"], float(zc["est_nodes"])
    ids = zc["gt_ids"]; idx = {int(i): k for k, i in enumerate(ids)}
    gt_e = np.array([[idx[int(a)], idx[int(b)]] for a, b in zc["gt_edges"]], np.int64).reshape(-1, 2)

    V = {k: {"t": [], "zyx": []} for k in ("single", "tta_dual")}
    fmap = {}
    for ws in range(0, T - M.WINDOW + 1):
        ts = list(range(ws, ws + M.WINDOW))
        imgs = torch.stack([M.normalise(io.read_frame(zp, t), ql, qh) for t in ts]).unsqueeze(0).to(DEV)
        with torch.no_grad():
            u1, d1 = prim.encode(imgs)
            uT, dT = EN.encode_tta(prim, imgs)
            _, dS = EN.encode_tta(seco, imgs)
        for i, t in enumerate(ts):
            if ws > 0 and i == 0:
                continue
            pk1 = M.peaks(d1[i][0])
            dB = dT[0, i]; pkB = M.peaks(dB)
            dC = EN.blend_detection(dB, dS[0, i]); pkC = M.peaks(dC)
            if not EN.retention_ok(len(pkC), len(pkB)):
                pkC = pkB
            for key, pk in (("single", pk1), ("tta_dual", pkC)):
                if len(pk):
                    V[key]["t"].append(np.full(len(pk), t, np.int16))
                    V[key]["zyx"].append(pk.to(torch.int16).cpu().numpy())
            fmap.setdefault(t, {})["single"] = u1[0, i]
            fmap[t]["tta_dual"] = uT[0, i]
        del imgs

    for regime in ("single", "tta_dual"):
        dt = np.concatenate(V[regime]["t"]); dz = np.concatenate(V[regime]["zyx"]).astype(np.float32)
        by_t = {int(t): np.where(dt == t)[0] for t in np.unique(dt)}
        mats = {}
        for t in sorted(by_t):
            si, ti = by_t.get(t), by_t.get(t + 1)
            if si is None or ti is None or not len(si) or not len(ti):
                continue
            fs = E.index_features(fmap[t][regime], dz[si])
            ft = E.index_features(fmap[t + 1][regime], dz[ti])
            lg = E.edge_logits(prim, fs, ft, dz[si], dz[ti], DEV)
            mats[t] = (si, ti, torch.softmax(torch.from_numpy(lg), 0).numpy())
        for beta in BETAS:
            e, prev = [], {}
            for t in sorted(mats):
                si, ti, prob = mats[t]
                s_um, t_um = dz[si] * E.GRID_UM, dz[ti] * E.GRID_UM
                pred = s_um.copy()
                for k, g in enumerate(si):
                    p = prev.get(int(g))
                    if p is not None:
                        pred[k] = s_um[k] + E.VELOCITY_WEIGHT * (s_um[k] - dz[p] * E.GRID_UM)
                for i, j, _, _, _ in E.link_frame_pair(s_um, t_um, prob, pred, beta=beta):
                    e.append((int(si[i]), int(ti[j]))); prev[int(ti[j])] = int(si[i])
            e2, dt2, dz2, _, _ = PP.prune_and_filter(e, dt, dz, min_track_len=6)
            rows[(regime, beta)].append(MT.score_film(list(range(len(dt2))), e2, dz2, dt2,
                                                      gt_grid, gt_t, gt_e, n_est))
    el = time.perf_counter() - t0
    print(f"[{n+1}/{len(sel)}] {film}  eta {el/(n+1)*(len(sel)-n-1)/60:.0f}min", flush=True)

print(f"\n{'regime':<12}{'beta':>6}{'score':>9}{'adjJ':>9}{'J':>9}{'vs beta=1':>11}")
out = []
for regime in ("single", "tta_dual"):
    base = MT.aggregate(rows[(regime, 1.0)])
    for b in BETAS:
        a = MT.aggregate(rows[(regime, b)])
        print(f"{regime:<12}{b:>6.1f}{a['score']:>9.4f}{a['adj_J_edge']:>9.4f}"
              f"{a['J_edge']:>9.4f}{a['score']-base['score']:>+11.4f}")
        out.append(dict(regime=regime, beta=b, score=a["score"], adjJ=a["adj_J_edge"],
                        J=a["J_edge"], vs_beta1=a["score"] - base["score"]))
pd.DataFrame(out).to_csv("artifacts/bonus_on_tta.csv", index=False)
g_single = out[2]["vs_beta1"]; g_tta = out[5]["vs_beta1"]
print(f"\nbeta gain on single-view probabilities : {g_single:+.4f}")
print(f"beta gain on TTA+dual probabilities    : {g_tta:+.4f}")
print(f"-> {'SURVIVES' if g_tta > 0.5*g_single else 'LARGELY ABSORBED by TTA+dual'}")
