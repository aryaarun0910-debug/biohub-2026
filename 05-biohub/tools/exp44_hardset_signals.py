#!/usr/bin/env python3
"""EXP-44 -- on the HARD set only, which signal prefers the true successor over the near impostor?

EXP-43 localised every remaining edge error to one shape: the true successor exists, another cell
is closer, and the tracker takes the closer one. EXP-42 showed Gaussian-weighted kNN mean flow
does not reorder candidates -- but that measured RANK over all missing edges with the crudest
flow variant. This is the restricted, pairwise version, which is a strictly sharper instrument:

    H = { cases where d(i, nearest) < d(i, true) }

and for each feature f report  P( f(true) < f(nearest-wrong) | H ).

By construction Euclidean distance scores 0% on H. Anything materially above 50% is real signal
and says where to invest. Anything near 50% is noise dressed as a feature.

Features, all geometric, no appearance (EXP-32 closed appearance):
    f1 euclid        raw distance                         -- the control, must read 0%
    f2 flow_knn      residual vs Gaussian-weighted kNN mean displacement
    f3 flow_affine   residual vs a locally fitted affine map x -> A x + b
    f4 accel         |v_candidate - v_previous|, needs track history
    f5 neigh_topo    preservation of relative offsets to neighbours' successors
    f6 back_cycle    backward-flow residual from the candidate to i

Trusted links used to estimate local motion EXCLUDE every edge leaving i, so the tracker's own
(wrong) choice cannot bias the estimate toward itself.
"""
import sys, warnings
from pathlib import Path
import numpy as np, tracksdata as td
from scipy.spatial import cKDTree

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
warnings.filterwarnings("ignore")
import div_sweep as D

PRED = Path("work/repro_out/tracking_repo/predictions/unknown/unet_transformer_val/split_0")
GT = Path("data/train_geff"); TOL = 7.0
K_FLOW, K_TOPO, SIG = 24, 12, 25.0
FEATS = ["euclid", "flow_knn", "flow_affine", "accel", "neigh_topo", "back_cycle"]


def load(p):
    g = td.graph.IndexedRXGraph.from_geff(p); return g[0] if isinstance(g, tuple) else g


def affine_fit(P0, P1):
    """Least-squares x1 = A x0 + b with one reweighting pass to blunt outliers."""
    n = len(P0)
    if n < 6: return None
    X = np.hstack([P0, np.ones((n, 1))])
    w = np.ones(n)
    for _ in range(2):
        Xw = X * w[:, None]
        try: M, *_ = np.linalg.lstsq(Xw, P1 * w[:, None], rcond=None)
        except np.linalg.LinAlgError: return None
        r = np.linalg.norm(X @ M - P1, axis=1)
        s = np.median(r) + 1e-6
        w = 1.0 / (1.0 + (r / (3 * s)) ** 2)
    return M            # (4,3)


rows = []
for p in sorted(PRED.glob("*.geff")):
    gp = GT / f"{p.stem}.geff"
    if not gp.exists(): continue
    g = load(p)
    pos, tt = {}, {}
    for r in g.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        tt[i] = int(r["t"])
    edges = [(int(r["source_id"]), int(r["target_id"])) for r in g.edge_attrs().iter_rows(named=True)]
    succ, pred_of = {}, {}
    for s, d in edges:
        succ.setdefault(s, []).append(d); pred_of[d] = s
    by_t = {}
    for i, t in tt.items(): by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}
    # trusted single-successor links, indexed by source frame
    link_by_t = {}
    for s, d in edges:
        if tt.get(d) == tt.get(s, -9) + 1 and len(succ.get(s, [])) == 1:
            link_by_t.setdefault(tt[s], []).append((s, d))
    ftree = {t: (cKDTree(np.stack([pos[s] for s, _ in L])), L) for t, L in link_by_t.items() if len(L) >= 8}

    gt = load(gp)
    gpos, gtt = {}, {}
    for r in gt.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gtt[i] = int(r["t"])
    def match(q, t):
        ent = trees.get(t)
        if ent is None: return None
        tr, ids = ent; d, j = tr.query(q)
        return ids[int(j)] if d <= TOL else None

    for r in gt.edge_attrs().iter_rows(named=True):
        s, d = int(r["source_id"]), int(r["target_id"])
        if s not in gpos or d not in gpos: continue
        mi, mj = match(gpos[s], gtt[s]), match(gpos[d], gtt[d])
        if mi is None or mj is None: continue
        t = tt[mi]
        ent = trees.get(t + 1)
        if ent is None: continue
        tr, ids = ent
        nn = ids[int(tr.query(pos[mi])[1])]
        if nn == mj: continue                       # not hard: the true one IS nearest
        ent2 = ftree.get(t)
        if ent2 is None: continue
        ft, L = ent2
        k = min(K_FLOW, len(L))
        dist, idx = ft.query(pos[mi], k=k)
        dist = np.atleast_1d(dist); idx = np.atleast_1d(idx)
        sel = [L[q] for q in idx if L[q][0] != mi]          # never use i's own link
        if len(sel) < 6: continue
        P0 = np.stack([pos[a] for a, _ in sel]); P1 = np.stack([pos[b] for _, b in sel])
        dsp = P1 - P0
        dd = np.linalg.norm(P0 - pos[mi], axis=1)
        w = np.exp(-(dd ** 2) / (2 * SIG ** 2)); w = w / max(w.sum(), 1e-9)
        mu = (dsp * w[:, None]).sum(0)
        M = affine_fit(P0, P1)
        xh_knn = pos[mi] + mu
        xh_aff = (np.append(pos[mi], 1.0) @ M) if M is not None else xh_knn
        # previous velocity
        pr = pred_of.get(mi); vprev = None
        if pr is not None and tt.get(pr) == t - 1: vprev = pos[mi] - pos[pr]
        # neighbourhood topology anchors
        anch = sel[:K_TOPO]
        # backward flow field (t+1 -> t)
        bP0 = P1; bdsp = P0 - P1
        bw = np.exp(-(np.linalg.norm(bP0 - pos[mi], axis=1) ** 2) / (2 * SIG ** 2))

        def feats(c):
            f = {}
            f["euclid"] = float(np.linalg.norm(pos[c] - pos[mi]))
            f["flow_knn"] = float(np.linalg.norm(pos[c] - xh_knn))
            f["flow_affine"] = float(np.linalg.norm(pos[c] - xh_aff))
            f["accel"] = float(np.linalg.norm((pos[c] - pos[mi]) - vprev)) if vprev is not None else np.nan
            f["neigh_topo"] = float(np.mean([
                np.linalg.norm((pos[c] - pos[b]) - (pos[mi] - pos[a])) for a, b in anch]))
            bwc = np.exp(-(np.linalg.norm(bP0 - pos[c], axis=1) ** 2) / (2 * SIG ** 2))
            vb = (bdsp * bwc[:, None]).sum(0) / max(bwc.sum(), 1e-9)
            f["back_cycle"] = float(np.linalg.norm((pos[c] + vb) - pos[mi]))
            return f

        rows.append((feats(mj), feats(nn)))

print(f"\n  HARD set (true successor exists but is NOT the nearest cell): {len(rows)} cases\n")
print(f"  {'signal':<14}{'prefers TRUE':>14}{'n':>7}   verdict")
print("  " + "-" * 56)
for f in FEATS:
    pairs = [(a[f], b[f]) for a, b in rows if a[f] == a[f] and b[f] == b[f]]
    if not pairs: continue
    win = sum(1 for x, y in pairs if x < y)
    p = win / len(pairs)
    v = "signal" if p > 0.60 else "weak" if p > 0.55 else "noise"
    if f == "euclid": v = "control (0% by construction)"
    print(f"  {f:<14}{p:>13.1%}{len(pairs):>7}   {v}")
print("\n  break-even for acting on a signal is 48.1% edge precision, but these are")
print("  PAIRWISE preferences on the hard set, not precision -- a signal must first")
print("  beat 50% here before any threshold rule can be worth building.")
