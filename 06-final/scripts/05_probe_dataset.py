"""G3b prep: features + matching covariates for division parents and controls.

The matching IS the experiment. Unmatched, a probe relearns "this region is
bright" and hands back the AUC 0.73 raw intensity already gives. Controls are
drawn from the SAME film and SAME frame and matched on (log intensity, local
detection density), so the probe measures the INCREMENTAL information in the
representation.

Extracts a temporal window t-2..t+2 at the parent's own position, because two
independent observers believed the anaphase signature precedes the labelled
split frame and neither tested it.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, torch, pandas as pd
from scipy.spatial import cKDTree
from biohub import io, model as M

DEV, OFFSETS, N_CTRL = "mps", (-2, -1, 0, 1, 2), 12
CACHE = Path("artifacts/cache")
rng = np.random.default_rng(0)
m, cfg = M.load("primary", DEV)
C = cfg["unet_out_channels"]

films = pd.read_csv("artifacts/films.csv").set_index("film")
targets = films[films.divisions > 0].index.tolist()
print(f"{len(targets)} films carry a labelled division", flush=True)

rows = []
t0 = time.perf_counter()
for n, film in enumerate(targets):
    z = np.load(CACHE / f"{film}.npz")
    ids, gt_t, grid, edges = z["gt_ids"], z["gt_t"], z["gt_grid"], z["gt_edges"]
    idx_of = {int(i): k for k, i in enumerate(ids)}
    src, cnt = np.unique(edges[:, 0], return_counts=True)
    parents = [idx_of[int(s)] for s, c in zip(src, cnt) if c >= 2]
    if not parents:
        continue

    zp = io.dataset_root() / "train" / f"{film}.zarr"
    q = io.quantiles(zp); q_low, q_high = float(q["0.001"]), float(q["0.999"])
    T = io.image_meta(zp)["shape"][0]

    # local detection density per frame (detections are ~complete; GT is 3.6%)
    det_t, det_zyx = z["det_t"], z["det_zyx"].astype(np.float32)
    trees = {}
    for t in np.unique(gt_t):
        sel = det_t == t
        if sel.sum() >= 4:
            trees[int(t)] = cKDTree(det_zyx[sel] * 1.625)

    def density(t, c):
        tr = trees.get(int(t))
        if tr is None: return np.nan
        d, _ = tr.query(c[None] * 1.625, k=min(4, tr.n))
        return float(np.mean(d[0][1:])) if d.shape[1] > 1 else np.nan

    # intensity at every GT node of this film, for matching
    frames_needed = sorted({int(t) for t in gt_t})
    inten = np.full(len(ids), np.nan, np.float32)
    fcache = {}
    for t in frames_needed:
        f = M.normalise(io.read_frame(zp, t), q_low, q_high).numpy()
        fcache[t] = f
        for k in np.where(gt_t == t)[0]:
            cz, cy, cx = np.round(grid[k]).astype(int)
            cz, cy, cx = np.clip(cz, 0, 63), np.clip(cy, 0, 63), np.clip(cx, 0, 63)
            inten[k] = f[cz, max(cy-1,0):cy+2, max(cx-1,0):cx+2].max()

    dens = np.array([density(gt_t[k], grid[k]) for k in range(len(ids))], np.float32)

    # match controls within the same frame on (log intensity, density)
    chosen = []
    for p in parents:
        pool = np.where((gt_t == gt_t[p]) & (np.arange(len(ids)) != p))[0]
        pool = pool[~np.isnan(inten[pool]) & ~np.isnan(dens[pool])]
        if len(pool) < 2:
            continue
        key = lambda a: np.stack([np.log1p(inten[a]), dens[a]], 1)
        ref, cand = key(np.array([p]))[0], key(pool)
        s = cand.std(0); s[s == 0] = 1
        d = np.linalg.norm((cand - ref) / s, axis=1)
        chosen.append((p, pool[np.argsort(d)[:N_CTRL]]))

    # features across the temporal window at each selected position
    want = sorted({p for p, c in chosen} | {int(x) for _, c in chosen for x in c})
    need_frames = sorted({int(np.clip(gt_t[k] + o, 0, T - 1)) for k in want for o in OFFSETS})
    feats = {}
    for t in need_frames:
        imgs = torch.stack([
            M.normalise(io.read_frame(zp, tt), q_low, q_high)
            for tt in (t, min(t + 1, T - 1))
        ])
        with torch.no_grad():
            u, _ = m.encode(imgs.unsqueeze(0).to(DEV))
        feats[t] = u[0, 0]

    for p, ctrls in chosen:
        for k, lab in [(p, 1)] + [(int(c), 0) for c in ctrls]:
            r = {"film": film, "embryo": film.split("_")[0], "node": int(ids[k]),
                 "label": lab, "t": int(gt_t[k]),
                 "intensity": float(inten[k]), "density": float(dens[k])}
            for o in OFFSETS:
                tt = int(np.clip(gt_t[k] + o, 0, T - 1))
                fv = M.sample_features(feats[tt], torch.from_numpy(grid[k][None])).float().cpu().numpy()[0]
                for j in range(C):
                    r[f"f{o:+d}_{j}"] = float(fv[j])
            rows.append(r)
    el = time.perf_counter() - t0
    print(f"[{n+1:>3}/{len(targets)}] {film} pos={len(chosen)} rows={len(rows)} "
          f"eta {el/(n+1)*(len(targets)-n-1)/60:.0f}min", flush=True)

df = pd.DataFrame(rows)
df.to_csv("artifacts/probe_dataset.csv.gz", index=False)
print(f"\nwrote {len(df)} rows | positives {int(df.label.sum())} | films {df.film.nunique()}")
print(df.groupby('embryo').label.agg(['size','sum']).to_string())
