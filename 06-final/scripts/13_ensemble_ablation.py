"""Which of the four missing components closes 0.845 -> 0.926?

One forward pass per window computes every rung, so the ladder is measured on
identical inputs. Stratified subset across both embryos, division-bearing films
preferred, because the GPU is shared.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd, torch
from biohub import io, model as M, edges as E, metric as MT, postprocess as PP, ensemble as EN

DEV = "mps"
N_PER_EMBRYO = int(sys.argv[1]) if len(sys.argv) > 1 else 20
films_tbl = pd.read_csv("artifacts/films.csv").set_index("film")
sel = []
for emb, g in films_tbl.groupby("embryo"):
    g = g.sort_values(["divisions", "labelled_nodes"], ascending=False)
    sel += list(g.index[:N_PER_EMBRYO])
print(f"{len(sel)} films: " + ", ".join(f"{e}={sum(1 for f in sel if f.startswith(e))}"
                                       for e in films_tbl.embryo.unique()), flush=True)

prim, cfg = M.load("primary", DEV)
seco, _ = M.load("secondary", DEV)
ROWS = {k: [] for k in ("A_base", "B_tta", "C_dual", "D_bidir", "E_consensus")}
t0 = time.perf_counter()

for n, film in enumerate(sel):
    zp = io.dataset_root() / "train" / f"{film}.zarr"
    q = io.quantiles(zp); ql, qh = float(q["0.001"]), float(q["0.999"])
    T = io.image_meta(zp)["shape"][0]
    zc = np.load(f"artifacts/cache/{film}.npz")
    gt_t, gt_grid, n_est = zc["gt_t"], zc["gt_grid"], float(zc["est_nodes"])
    ids = zc["gt_ids"]; idx = {int(i): k for k, i in enumerate(ids)}
    gt_e = np.array([[idx[int(a)], idx[int(b)]] for a, b in zc["gt_edges"]], np.int64).reshape(-1, 2)

    V = {k: {"t": [], "zyx": [], "feat": []} for k in ("A", "B", "C")}
    featmap = {}                       # frame -> {variant: (C,Z,Y,X)} for edge scoring
    for ws in range(0, T - M.WINDOW + 1):
        ts = list(range(ws, ws + M.WINDOW))
        imgs = torch.stack([M.normalise(io.read_frame(zp, t), ql, qh) for t in ts]).unsqueeze(0).to(DEV)
        with torch.no_grad():
            u1, d1 = prim.encode(imgs)                    # single view
            uT, dT = EN.encode_tta(prim, imgs)            # 4-view TTA
            uS, dS = EN.encode_tta(seco, imgs)            # secondary, 4-view
        for i, t in enumerate(ts):
            if ws > 0 and i == 0:
                continue
            dA, dB, dS_i = d1[i][0], dT[0, i], dS[0, i]
            dC = EN.blend_detection(dB, dS_i)
            pk_b = M.peaks(dB)
            pk_c = M.peaks(dC)
            if not EN.retention_ok(len(pk_c), len(pk_b)):
                dC = dB                                   # frame retention guard
                pk_c = pk_b
            for key, dd, pk, fm in (("A", dA, M.peaks(dA), u1[0, i]),
                                    ("B", dB, pk_b, u1[0, i]),
                                    ("C", dC, pk_c, uT[0, i])):
                if not len(pk):
                    continue
                V[key]["t"].append(np.full(len(pk), t, np.int16))
                V[key]["zyx"].append(pk.to(torch.int16).cpu().numpy())
                V[key]["feat"].append(E.index_features(fm, pk.cpu().numpy()).float().cpu().numpy())
            featmap.setdefault(t, {})["P"] = u1[0, i].cpu()
            featmap[t]["T"] = uT[0, i].cpu()
            featmap[t]["S"] = uS[0, i].cpu()
        del imgs

    def assemble(k):
        return (np.concatenate(V[k]["t"]), np.concatenate(V[k]["zyx"]).astype(np.float32),
                np.concatenate(V[k]["feat"]))

    def build(det_t, det_zyx, det_feat, fkey, bidir, consensus):
        by_t = {int(t): np.where(det_t == t)[0] for t in np.unique(det_t)}
        out, prev = [], {}
        for t in sorted(by_t):
            si, ti = by_t.get(t), by_t.get(t + 1)
            if si is None or ti is None or not len(si) or not len(ti):
                continue
            fs_key = featmap[t][fkey].to(DEV); ft_key = featmap[t + 1][fkey].to(DEV)
            fsrc = E.index_features(fs_key, det_zyx[si])
            ftgt = E.index_features(ft_key, det_zyx[ti])
            lg = E.edge_logits(prim, fsrc, ftgt, det_zyx[si], det_zyx[ti], DEV)
            if bidir:
                rv = E.edge_logits(prim, ftgt, fsrc, det_zyx[ti], det_zyx[si], DEV).T
                lg = EN.harmonic_bidirectional(lg, rv)
            if consensus:
                s_s = E.index_features(featmap[t]["S"].to(DEV), det_zyx[si])
                s_t = E.index_features(featmap[t + 1]["S"].to(DEV), det_zyx[ti])
                lg2 = E.edge_logits(seco, s_s, s_t, det_zyx[si], det_zyx[ti], DEV)
                lg = EN.low_margin_consensus(lg, lg2)
            prob = torch.softmax(torch.from_numpy(lg), 0).numpy()
            s_um, t_um = det_zyx[si] * E.GRID_UM, det_zyx[ti] * E.GRID_UM
            pred = s_um.copy()
            for k2, g in enumerate(si):
                p = prev.get(int(g))
                if p is not None:
                    pred[k2] = s_um[k2] + E.VELOCITY_WEIGHT * (s_um[k2] - det_zyx[p] * E.GRID_UM)
            for i, j, pp, dd, _ in E.link_frame_pair(s_um, t_um, prob, pred):
                out.append((int(si[i]), int(ti[j])))
                prev[int(ti[j])] = int(si[i])
        return out

    for name, key, fkey, bd, cs in (("A_base", "A", "P", False, False),
                                    ("B_tta", "B", "P", False, False),
                                    ("C_dual", "C", "T", False, False),
                                    ("D_bidir", "C", "T", True, False),
                                    ("E_consensus", "C", "T", True, True)):
        dt, dz, df = assemble(key)
        e = build(dt, dz, df, fkey, bd, cs)
        e2, dt2, dz2, _, _ = PP.prune_and_filter(e, dt, dz, min_track_len=6)
        ROWS[name].append((film, MT.score_film(list(range(len(dt2))), e2, dz2, dt2,
                                               gt_grid, gt_t, gt_e, n_est)))
    el = time.perf_counter() - t0
    print(f"[{n+1:>3}/{len(sel)}] {film}  {el/(n+1):.0f}s/film  "
          f"eta {el/(n+1)*(len(sel)-n-1)/60:.0f}min", flush=True)

out = []
for name, rows in ROWS.items():
    for fold in ["ALL"] + sorted(films_tbl.loc[sel].embryo.unique()):
        rr = [r for f, r in rows if fold == "ALL" or f.startswith(fold)]
        a = MT.aggregate(rr)
        out.append({"variant": name, "fold": fold, "score": a["score"],
                    "adjJ": a["adj_J_edge"], "J": a["J_edge"], "mult": a["multiplier"],
                    "divJ": a["div_jaccard"]})
df = pd.DataFrame(out); df.to_csv("artifacts/ensemble_ablation.csv", index=False)
piv = df[df.fold == "ALL"].set_index("variant")
base = piv.loc["A_base", "score"]
print("\n" + "=" * 78)
print(f"{'variant':<14} {'score':>8} {'vs A':>9} {'adjJ':>8} {'J':>8} {'mult':>7} {'divJ':>7}")
for v in ["A_base", "B_tta", "C_dual", "D_bidir", "E_consensus"]:
    r = piv.loc[v]
    print(f"{v:<14} {r.score:>8.4f} {r.score-base:>+9.4f} {r.adjJ:>8.4f} {r.J:>8.4f} "
          f"{r['mult']:>7.4f} {r.divJ:>7.4f}")
print("\nper fold:")
print(df.pivot_table(index="variant", columns="fold", values="score").to_string(
    float_format=lambda v: f"{v:.4f}"))
