r"""Candidate-threshold sweep over the exported pre-ILP graph -- the ONLY lever class with
measured LB transfer.

WHY THIS EXISTS
---------------
The measured transfer law (2026-08-19) says only detection-surface / candidate-set changes
move the LB (control: local -0.0091 -> LB -0.0320, 3.5x amplified); division-term and
edge-permutation changes measured 0.000. `BIOHUB_DUAL_SEED_EDGE_THRESHOLD = 0.48` is the
constant that defines the candidate set, and the records show it is INHERITED -- "the
unchanged Pilkwang two-seed baseline", never swept by us.

This measures, over the exported fold-1 pre-ILP candidate graph, the GT REACH curve:
for each candidate threshold tau, how many true GT association edges are still present as
candidates, against how many candidates you carry. Reach is the ceiling on everything
downstream -- the solver cannot recover an edge that was never nominated
(`error_atlas_2026-08-19`: 17,067 detectable GT edges never nominated, while a PERFECT solver
over today's candidates is worth only +0.0012).

NOTE ON THE FLOOR: the export contains no edge below prob 0.500, because the candidate rule is
a column softmax > 0.5. The deployed 0.48 therefore sits BELOW the structural floor and cannot
bind -- any value <= 0.5 gives an identical candidate set. Only the upward direction is
measurable from this artifact.

Counts only -- no scoring, no model, no GPU.

Usage:
  .venv\Scripts\python.exe scripts\win_bet\candidate_threshold_sweep.py \
      --parquet c:/temp/preilp_f1_v2/preilp_split1.parquet --gt-dir data/train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 7.0  # scorer matching radius


def load_gt(path: Path):
    g = zarr.open(str(path), mode="r")
    nid = np.asarray(g["nodes/ids"][:]).astype(np.int64)
    t = np.asarray(g["nodes/props/t/values"][:]).astype(np.int64)
    pos = np.stack([
        np.asarray(g["nodes/props/z/values"][:]).astype(np.float64) * SCALE[0],
        np.asarray(g["nodes/props/y/values"][:]).astype(np.float64) * SCALE[1],
        np.asarray(g["nodes/props/x/values"][:]).astype(np.float64) * SCALE[2],
    ], axis=1)
    e = np.asarray(g["edges/ids"][:])
    if e.ndim != 2 or e.size == 0:
        e = np.zeros((0, 2), dtype=np.int64)
    return nid, t, pos, e.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--gt-dir", default="data/train")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print("loading pre-ILP export ...", flush=True)
    df = pd.read_parquet(args.parquet)
    nodes_all = df[df.row_type == "node"]
    edges_all = df[df.row_type == "edge"]

    TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
    reach = {tau: 0 for tau in TAUS}
    cand = {tau: 0 for tau in TAUS}
    gt_total = 0
    gt_detectable = 0

    for ds, gn in nodes_all.groupby("dataset"):
        gp = Path(args.gt_dir) / f"{ds}.geff"
        if not gp.exists():
            continue
        gnid, gt_t, gpos, ge = load_gt(gp)
        gt_total += len(ge)

        pos = np.stack([gn["z"].to_numpy() * SCALE[0],
                        gn["y"].to_numpy() * SCALE[1],
                        gn["x"].to_numpy() * SCALE[2]], axis=1)
        pt = gn["t"].to_numpy().astype(np.int64)
        pid = gn["node_id"].to_numpy().astype(np.int64)

        # match each GT node to the nearest predicted node in the same frame, within MATCH_UM
        gt2pred: dict[int, int] = {}
        for tt in np.unique(gt_t):
            pm = pt == tt
            if not pm.any():
                continue
            tree = cKDTree(pos[pm])
            ids_here = pid[pm]
            gm = gt_t == tt
            d, j = tree.query(gpos[gm])
            for gid, dd, jj in zip(gnid[gm], d, j):
                if dd <= MATCH_UM:
                    gt2pred[int(gid)] = int(ids_here[jj])

        # GT edges whose BOTH endpoints were detected -- the reachable ceiling
        pairs = []
        for a, b in ge:
            pa, pb = gt2pred.get(int(a)), gt2pred.get(int(b))
            if pa is not None and pb is not None and pa != pb:
                pairs.append((pa, pb))
        gt_detectable += len(pairs)
        if not pairs:
            continue
        want = set(pairs)

        gedg = edges_all[edges_all.dataset == ds]
        src = gedg["source_id"].to_numpy().astype(np.int64)
        tgt = gedg["target_id"].to_numpy().astype(np.int64)
        prob = gedg["edge_prob"].to_numpy()
        for tau in TAUS:
            m = prob >= tau
            cand[tau] += int(m.sum())
            have = set(zip(src[m].tolist(), tgt[m].tolist()))
            reach[tau] += len(want & have)

    print()
    print(f"GT edges total (fold-1 crops)        : {gt_total}")
    print(f"GT edges with BOTH endpoints detected: {gt_detectable} "
          f"({100.0 * gt_detectable / max(1, gt_total):.2f}% -- the detection ceiling)")
    print()
    base = reach[TAUS[0]]
    hdr = (f"{'tau':>6s} {'candidates':>12s} {'GT reached':>11s} "
           f"{'% of detectable':>15s} {'% of all GT':>12s} {'d vs 0.50':>10s}")
    print(hdr)
    print("-" * len(hdr))
    out = {"gt_total": gt_total, "gt_detectable": gt_detectable, "curve": {}}
    for tau in TAUS:
        r, c = reach[tau], cand[tau]
        print(f"{tau:6.2f} {c:12d} {r:11d} "
              f"{100.0 * r / max(1, gt_detectable):14.2f}% "
              f"{100.0 * r / max(1, gt_total):11.2f}% {r - base:10d}")
        out["curve"][f"{tau:.2f}"] = {
            "candidates": c, "gt_reached": r,
            "pct_of_detectable": 100.0 * r / max(1, gt_detectable),
            "pct_of_all_gt": 100.0 * r / max(1, gt_total),
            "delta_vs_0.50": r - base,
        }

    print()
    print("Deployed BIOHUB_DUAL_SEED_EDGE_THRESHOLD = 0.48 -> identical to tau=0.50 "
          "(export floor; a column softmax cannot fall below 0.5 for the argmax entry).")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
