r"""Division-proposal funnel -- where do the 146 missed divisions actually die?

WHY THIS EXISTS
---------------
Measured 2026-08-18 on the full deployment substrate: division TP=5, FP=613, FN=146, and the FP
and FN populations are DISJOINT (`fork_at_matched_mother = 0`). So no filter or re-ranking over
the emitted forks can recover a division -- the loss happens upstream, at PROPOSAL time.

`add_safe_divisions_postlink` (src/biotrack/wrapper.py:770-845) admits a second child only if a
chain of conditions all hold. This script walks a real GT division through that exact chain and
reports the survival count at each stage, so we learn WHICH condition kills them rather than
assuming it is the distance gates.

The conditions, in the order the wrapper applies them:
  (a) mother detected            -- a predicted node within MATCH_UM of the GT mother at t
  (b) both daughters detected    -- predicted nodes within MATCH_UM of each GT daughter at t+1
  (c) mother out-degree == 1     -- `source_ids` filter
  (d) linked child is a daughter -- the linker's existing child must BE one of the true daughters
  (e) other daughter is ORPHAN   -- `candidate_ids` excludes any node with an incoming edge
  (f) child_dist <= 7.8 um       -- SAFE_DIV_EXISTING_CHILD_MAX_UM
  (g) parent_dist <= 4.7 um      -- SAFE_DIV_MAX_UM
  (h) sister_dist <= 7.2 um      -- SAFE_DIV_SISTER_MAX_UM

(e) is the condition nobody has looked at: if the linker attached the second daughter to some
other parent, the true division is unproposable at ANY gate setting. Relaxing (g)/(h) would then
be worthless, and the whole division thesis dies cheaply -- which is the point of measuring first.

The script also sweeps (g)/(h) and reports, alongside admitted true divisions, the size of the
candidate pool each setting opens up -- the false-proposal exposure. Both numbers are needed:
division_jaccard = TP/(TP+FP+FN), so proposal without precision buys nothing.

This measures COUNTS, not scores, and is therefore immune to the LOEO->LB transfer problem
(see research/06-knowledge-system/internal-reports/loeo_lb_gap_2026-08-18.md).

Usage:
  .venv\Scripts\python.exe scripts\win_bet\div_proposal_funnel.py \
      --csv c:/temp/subvoxel_f0/loeo_split0_strict.csv.gz --gt-dir data/train
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])   # z, y, x um per level-0 voxel
MATCH_UM = 7.0                                 # scorer's matching radius
CHILD_MAX_UM = 7.8                             # SAFE_DIV_EXISTING_CHILD_MAX_UM
PARENT_MAX_UM = 4.7                            # SAFE_DIV_MAX_UM
SISTER_MAX_UM = 7.2                            # SAFE_DIV_SISTER_MAX_UM

STAGES = [
    "GT divisions",
    "(a) mother detected",
    "(b) both daughters detected",
    "(c) mother out-degree == 1",
    "(d) linked child IS a true daughter",
    "(e) other daughter is ORPHAN",
    "(f) child_dist <= 7.8",
    "(g) parent_dist <= 4.7",
    "(h) sister_dist <= 7.2  [ADMITTED]",
]


def load_pred(csv_path: Path):
    df = pd.read_csv(csv_path)
    nodes = df[df.row_type == "node"]
    edges = df[df.row_type == "edge"]
    return nodes, edges


def gt_divisions(geff_path: Path):
    g = zarr.open(str(geff_path), mode="r")
    nid = np.asarray(g["nodes/ids"][:])
    t = np.asarray(g["nodes/props/t/values"][:])
    z = np.asarray(g["nodes/props/z/values"][:])
    y = np.asarray(g["nodes/props/y/values"][:])
    x = np.asarray(g["nodes/props/x/values"][:])
    pos = {int(n): (int(t[i]), np.array([z[i], y[i], x[i]]) * SCALE) for i, n in enumerate(nid)}
    e = np.asarray(g["edges/ids"][:])
    out = collections.defaultdict(list)
    if e.ndim == 2 and len(e):
        for a, b in e:
            out[int(a)].append(int(b))
    divs = []
    for src, tg in out.items():
        if len(tg) >= 2 and src in pos:
            d = [tt for tt in tg[:2] if tt in pos]
            if len(d) == 2:
                divs.append((src, d[0], d[1]))
    return divs, pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--gt-dir", default="data/train")
    ap.add_argument("--sweep", action="store_true", default=True)
    args = ap.parse_args()

    nodes, edges = load_pred(Path(args.csv))
    counts = collections.Counter()
    # survivors reaching stage (f): (parent_dist, sister_dist) of the TRUE second daughter
    true_pairs: list[tuple[float, float]] = []
    # candidate-pool exposure: per (mother, frame) how many orphans lie within a generous radius
    pool_rows: list[tuple[float, float]] = []

    for ds, gnodes in nodes.groupby("dataset"):
        geff = Path(args.gt_dir) / f"{ds}.geff"
        if not geff.exists():
            continue
        divs, gpos = gt_divisions(geff)
        if not divs:
            continue
        gedges = edges[edges.dataset == ds]

        # predicted graph
        npos, nt = {}, {}
        zz = gnodes["z"].to_numpy() * SCALE[0]
        yy = gnodes["y"].to_numpy() * SCALE[1]
        xx = gnodes["x"].to_numpy() * SCALE[2]
        for i, nidv in enumerate(gnodes["node_id"].to_numpy()):
            npos[int(nidv)] = np.array([zz[i], yy[i], xx[i]])
            nt[int(nidv)] = int(gnodes["t"].to_numpy()[i])
        outdeg = collections.Counter()
        child_of = collections.defaultdict(list)
        incoming = set()
        for s, tg in zip(gedges["source_id"].to_numpy(), gedges["target_id"].to_numpy()):
            outdeg[int(s)] += 1
            child_of[int(s)].append(int(tg))
            incoming.add(int(tg))

        by_t = collections.defaultdict(list)
        for nidv, tt in nt.items():
            by_t[tt].append(nidv)
        trees = {tt: (cKDTree(np.stack([npos[i] for i in ids])), ids)
                 for tt, ids in by_t.items() if ids}

        def match(gt_t, gt_xyz):
            entry = trees.get(gt_t)
            if entry is None:
                return None
            tree, ids = entry
            d, j = tree.query(gt_xyz)
            return ids[j] if d <= MATCH_UM else None

        for src, d1, d2 in divs:
            counts["GT divisions"] += 1
            mt, mxyz = gpos[src]
            m = match(mt, mxyz)
            if m is None:
                continue
            counts["(a) mother detected"] += 1
            t1, p1 = gpos[d1]
            t2, p2 = gpos[d2]
            a = match(t1, p1)
            b = match(t2, p2)
            if a is None or b is None:
                continue
            counts["(b) both daughters detected"] += 1
            if outdeg.get(m, 0) != 1:
                continue
            counts["(c) mother out-degree == 1"] += 1
            c = child_of[m][0]
            if c not in (a, b):
                continue
            counts["(d) linked child IS a true daughter"] += 1
            other = b if c == a else a
            if other in incoming:
                continue
            counts["(e) other daughter is ORPHAN"] += 1
            child_dist = float(np.linalg.norm(npos[m] - npos[c]))
            if child_dist > CHILD_MAX_UM:
                continue
            counts["(f) child_dist <= 7.8"] += 1
            pd_ = float(np.linalg.norm(npos[m] - npos[other]))
            sd_ = float(np.linalg.norm(npos[c] - npos[other]))
            true_pairs.append((pd_, sd_))
            if pd_ > PARENT_MAX_UM:
                continue
            counts["(g) parent_dist <= 4.7"] += 1
            if sd_ > SISTER_MAX_UM:
                continue
            counts["(h) sister_dist <= 7.2  [ADMITTED]"] += 1

        # false-proposal exposure at a generous gate: orphans near a single-child mother
        for m, deg in outdeg.items():
            if deg != 1:
                continue
            c = child_of[m][0]
            tt = nt.get(m)
            if tt is None or nt.get(c) != tt + 1:
                continue
            entry = trees.get(tt + 1)
            if entry is None:
                continue
            tree, ids = entry
            for j in tree.query_ball_point(npos[m], 12.0):
                cid = ids[j]
                if cid in incoming or cid == c:
                    continue
                pool_rows.append((float(np.linalg.norm(npos[m] - npos[cid])),
                                  float(np.linalg.norm(npos[c] - npos[cid]))))

    print(f"\n=== DIVISION PROPOSAL FUNNEL  ({Path(args.csv).name}) ===")
    base = max(counts['GT divisions'], 1)
    prev = None
    for st in STAGES:
        n = counts[st]
        lost = "" if prev is None else f"   (lost {prev - n})"
        print(f"  {st:38s} {n:5d}   {n/base:6.1%}{lost}")
        prev = n

    if true_pairs:
        tp = np.array(true_pairs)
        print(f"\n  TRUE second daughters reaching the distance gates: {len(tp)}")
        print(f"    parent_dist um : median {np.median(tp[:,0]):.2f}  p90 {np.percentile(tp[:,0],90):.2f}  max {tp[:,0].max():.2f}   (gate {PARENT_MAX_UM})")
        print(f"    sister_dist um : median {np.median(tp[:,1]):.2f}  p90 {np.percentile(tp[:,1],90):.2f}  max {tp[:,1].max():.2f}   (gate {SISTER_MAX_UM})")
        pool = np.array(pool_rows) if pool_rows else np.empty((0, 2))
        print(f"\n  GATE SWEEP  (true admitted / false-candidate pool / ratio)")
        print(f"  {'parent':>7} {'sister':>7} {'TRUE':>6} {'POOL':>9} {'per-TP':>8}")
        for pmax in (4.7, 6.0, 8.0, 10.0, 12.0):
            for smax in (7.2, 10.0, 12.0, 15.0):
                t_ok = int(((tp[:, 0] <= pmax) & (tp[:, 1] <= smax)).sum())
                f_ok = int(((pool[:, 0] <= pmax) & (pool[:, 1] <= smax)).sum()) if len(pool) else 0
                print(f"  {pmax:7.1f} {smax:7.1f} {t_ok:6d} {f_ok:9d} {(f_ok/max(t_ok,1)):8.1f}")
    else:
        print("\n  NO true second daughter reached the distance gates -- the loss is upstream.")


if __name__ == "__main__":
    main()
