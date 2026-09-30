"""Build the baseline predicted graph for every film, off the cache.

Detection peaks are integer coords, where grid_sample (align_corners, bilinear)
and integer indexing agree exactly -- so the cached det_feat is valid for edge
scoring and the UNet never has to run again. Only the transformer runs here,
which is matmul-shaped and therefore the shape this machine is fast at.

Two graphs are produced per film:
  raw    -- greedy on softmax probs, max 1 parent / 2 children (forks allowed)
  linked -- one-to-one Hungarian, tight 6um then relaxed 10um (what the
            deployed pipeline actually ships; cannot express a division)
The pair is what the oracle needs: O1 asks what perfect forks would buy, and
the deployed graph is the one with forks already destroyed.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, torch
from biohub import io, model as M, edges as E

DEV = "mps"
CACHE, OUT = Path("artifacts/cache"), Path("artifacts/graphs")
OUT.mkdir(parents=True, exist_ok=True)
m, cfg = M.load("primary", DEV)
films = io.films("train")
t0 = time.perf_counter()

for n, film in enumerate(films):
    dst = OUT / f"{film}.npz"
    if dst.exists():
        continue
    z = np.load(CACHE / f"{film}.npz")
    det_t, det_zyx, det_feat = z["det_t"], z["det_zyx"].astype(np.float32), z["det_feat"]
    order = np.argsort(det_t, kind="stable")
    det_t, det_zyx, det_feat = det_t[order], det_zyx[order], det_feat[order]
    by_t = {int(t): np.where(det_t == t)[0] for t in np.unique(det_t)}

    raw_edges, lnk_edges = [], []
    prev_parent = {}                       # target index -> source index, for velocity
    for t in sorted(by_t):
        si, ti = by_t.get(t), by_t.get(t + 1)
        if si is None or ti is None or not len(si) or not len(ti):
            continue
        fs = torch.from_numpy(det_feat[si]).to(DEV)
        ft = torch.from_numpy(det_feat[ti]).to(DEV)
        lg = E.edge_logits(m, fs, ft, det_zyx[si], det_zyx[ti], DEV)
        prob = torch.softmax(torch.from_numpy(lg), dim=0).numpy()   # over sources per target

        # raw: greedy, 1 parent / 2 children, threshold 0.5 -- forks survive here
        cand = np.argwhere(prob > 0.5)
        if len(cand):
            cand = cand[np.argsort(-prob[cand[:, 0], cand[:, 1]])]
            nch, npa = {}, {}
            for i, j in cand:
                if nch.get(i, 0) >= 2 or npa.get(j, 0) >= 1:
                    continue
                raw_edges.append((int(si[i]), int(ti[j]), float(prob[i, j])))
                nch[i] = nch.get(i, 0) + 1; npa[j] = npa.get(j, 0) + 1

        # linked: one-to-one Hungarian with constant-velocity anchor
        s_um, t_um = det_zyx[si] * E.GRID_UM, det_zyx[ti] * E.GRID_UM
        pred = s_um.copy()
        for k, g in enumerate(si):
            p = prev_parent.get(int(g))
            if p is not None:
                pred[k] = s_um[k] + E.VELOCITY_WEIGHT * (s_um[k] - det_zyx[p] * E.GRID_UM)
        got = E.link_frame_pair(s_um, t_um, prob, pred)
        for i, j, p, d, ps in got:
            lnk_edges.append((int(si[i]), int(ti[j]), p, d))
            prev_parent[int(ti[j])] = int(si[i])

    np.savez_compressed(
        dst,
        det_t=det_t, det_zyx=det_zyx.astype(np.int16),
        raw_edges=np.array(raw_edges, np.float64).reshape(-1, 3),
        linked_edges=np.array(lnk_edges, np.float64).reshape(-1, 4),
        est_nodes=z["est_nodes"],
        gt_ids=z["gt_ids"], gt_t=z["gt_t"], gt_grid=z["gt_grid"], gt_edges=z["gt_edges"],
    )
    el = time.perf_counter() - t0
    nf = sum(1 for s, t, p in raw_edges if True)
    print(f"[{n+1:>3}/{len(films)}] {film}  nodes={len(det_t):>6}  raw={len(raw_edges):>6}"
          f"  linked={len(lnk_edges):>6}  {el/(n+1):.1f}s/film"
          f"  eta {el/(n+1)*(len(films)-n-1)/60:.0f}min", flush=True)
print(f"DONE {(time.perf_counter()-t0)/60:.1f} min")
