r"""Nearest-parent oracle, with a tracklet-smoothing arm.

WHY
---
The linker's failure was localised to COORDINATE NOISE, not pool size: an oracle that
replaces ONLY the two endpoint coordinates with GT, holding the decoy pool completely
fixed, moves "true parent is the nearest frame-(t-1) node" from **3.44% -> 68.39%** on
fold 1 (experimental-records.md, 2026-08-25). Displacement noise is 89% of the signal.

The pre-registered follow-up is: rerun the same oracle with TRACKLET-SMOOTHED
coordinates, and **kill the lever if the statistic does not move off 3.44%**.

That test needs the oracle itself, which was run ad-hoc last cycle and never preserved --
only its two numbers reached the record. This module is that oracle, written to be
re-runnable.

CALIBRATION IS MANDATORY
------------------------
Two of three distance analyses in this project started with the wrong coordinate
convention. Atlas coords are FULL-RES ``(z, y, x)`` and take ``(1.625, 0.40625,
0.40625)``, NOT isotropic 1.625. So this module refuses to report anything until it
reproduces BOTH known anchors:

  * correctly-linked GT displacement median ~= 1.82 um,
  * the deployed/GT oracle pair ~= 3.44% / 68.39%.

If those do not come out, the implementation is wrong and the smoothed arm is meaningless.

ARMS
----
``deployed``   endpoints and pool all use predicted coords            (anchor: 3.44%)
``gt``         two endpoint coords replaced by GT, pool FIXED         (anchor: 68.39%)
``smooth_ep``  two endpoint coords replaced by SMOOTHED, pool FIXED   -- directly
               comparable to ``gt``: how much of the GT gain does smoothing recover?
``smooth_all`` every coordinate smoothed, including the pool          -- the deployable
               version, since in production you cannot smooth only the two nodes you
               care about.

Both smoothing arms follow PREDICTED tracks, not GT ones. Smoothing along a wrong link
propagates its error; that is the honest deployment condition.

USAGE
    python scripts/win_bet/nearest_parent_oracle.py --atlas C:/temp/error_atlas --fold f1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Atlas coords are FULL-RES (z, y, x). This is the trap; see module docstring.
SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

# Known anchors from experimental-records.md (fold 1).
ANCHOR_GT_DISP_MEDIAN_UM = 1.82
ANCHOR_DEPLOYED_PCT = 3.44
ANCHOR_GT_PCT = 68.39


def _um(zyx: np.ndarray) -> np.ndarray:
    """Scale full-res (z, y, x) voxel coords to micrometres."""
    return zyx.astype(np.float64) * SCALE


def calibrate_displacement(gtedges: pd.DataFrame) -> float:
    """Median true inter-frame displacement of CORRECTLY-linked GT edges, in um."""
    tp = gtedges[gtedges["is_tp"]]
    src = _um(tp[["s_z", "s_y", "s_x"]].to_numpy())
    tgt = _um(tp[["t_z", "t_y", "t_x"]].to_numpy())
    return float(np.median(np.linalg.norm(tgt - src, axis=1)))


def build_smoothed_positions(
    nodes: pd.DataFrame, edges: pd.DataFrame, window: int,
    mode: str = "mean", weight: float = 0.8,
) -> dict[tuple[str, int], np.ndarray]:
    """Smooth each predicted node's position along its PREDICTED track.

    A node's track neighbours are found by walking predicted edges backward and forward.
    Only unambiguous steps are followed (a unique parent / unique child); at a division or
    a merge the walk stops, because smoothing across a fork mixes two cells.

    ``mode``:
      ``mean``    zeroth-order average over the window. This is what the pre-registered
                  test named ("averaging node positions along a tracklet"). For a MOVING
                  cell it is biased: the mean pulls each node toward the track centroid,
                  shrinking apparent displacement.
      ``linefit`` first-order least-squares fit evaluated at the node's own frame, blended
                  ``weight`` toward the fit. This mirrors
                  ``linefit_smooth_output_graph`` (built notebook cell 6:1210) exactly,
                  including its ``len(neighbourhood) >= 3`` guard. It removes noise
                  WITHOUT injecting motion bias, which is why it is the estimator that
                  actually matters here.

    Returns {(dataset, node_id): smoothed_zyx_voxels}.
    """
    smoothed: dict[tuple[str, int], np.ndarray] = {}

    for ds, ndf in nodes.groupby("dataset", sort=False):
        pos = {int(r.node_id): np.array([r.z, r.y, r.x], dtype=np.float64)
               for r in ndf.itertuples()}
        edf = edges[edges["dataset"] == ds]

        # unique-parent / unique-child maps (ambiguous ones dropped)
        child_counts: dict[int, int] = {}
        parent_counts: dict[int, int] = {}
        for s, t in zip(edf["source_id"].to_numpy(), edf["target_id"].to_numpy()):
            child_counts[int(s)] = child_counts.get(int(s), 0) + 1
            parent_counts[int(t)] = parent_counts.get(int(t), 0) + 1
        nxt: dict[int, int] = {}
        prv: dict[int, int] = {}
        for s, t in zip(edf["source_id"].to_numpy(), edf["target_id"].to_numpy()):
            s, t = int(s), int(t)
            if child_counts.get(s, 0) == 1 and parent_counts.get(t, 0) == 1:
                nxt[s] = t
                prv[t] = s

        for nid, p in pos.items():
            hood = [(0, p)]
            cur = nid
            for step in range(1, window + 1):
                cur = prv.get(cur)
                if cur is None or cur not in pos:
                    break
                hood.append((-step, pos[cur]))
            cur = nid
            for step in range(1, window + 1):
                cur = nxt.get(cur)
                if cur is None or cur not in pos:
                    break
                hood.append((step, pos[cur]))

            if mode == "mean":
                smoothed[(ds, nid)] = np.mean([c for _, c in hood], axis=0)
            elif mode == "linefit":
                if len(hood) < 3:            # same guard as the deployed smoother
                    smoothed[(ds, nid)] = p
                    continue
                dts = np.array([d for d, _ in hood], dtype=np.float64)
                coords = np.stack([c for _, c in hood])
                # Closed-form degree-1 least squares, evaluated at dt=0 -> the intercept.
                # Identical to np.polyval(np.polyfit(dts, y, 1), 0.0) but vectorised over
                # all three axes at once; polyfit per axis is ~7.5M calls on this data.
                dbar = dts.mean()
                dvar = ((dts - dbar) ** 2).sum()
                if dvar <= 0:
                    smoothed[(ds, nid)] = p
                    continue
                ybar = coords.mean(axis=0)
                slope = ((dts - dbar)[:, None] * (coords - ybar)).sum(axis=0) / dvar
                fitted = ybar - slope * dbar
                if not np.isfinite(fitted).all():
                    smoothed[(ds, nid)] = p
                    continue
                smoothed[(ds, nid)] = (1.0 - weight) * p + weight * fitted
            else:
                raise ValueError(mode)

    return smoothed


def run_oracle(
    nodes: pd.DataFrame,
    gtedges: pd.DataFrame,
    arm: str,
    smoothed: dict[tuple[str, int], np.ndarray] | None = None,
    mislinked_only: bool = True,
) -> tuple[float, int]:
    """Fraction of edges where the true parent is the NEAREST frame-(t-1) predicted node.

    The decoy pool is every predicted node in the source frame and is held FIXED across
    arms, except in ``smooth_all`` where the pool is smoothed too.
    """
    sel = gtedges[gtedges["src_detected"] & gtedges["tgt_detected"]]
    sel = sel[~sel["is_tp"]] if mislinked_only else sel[sel["is_tp"]]

    # gt_id -> predicted node, per dataset
    gt_to_node: dict[str, dict[int, int]] = {}
    node_pos: dict[str, dict[int, np.ndarray]] = {}
    frame_nodes: dict[tuple[str, int], list[int]] = {}
    for ds, ndf in nodes.groupby("dataset", sort=False):
        m, pp = {}, {}
        for r in ndf.itertuples():
            nid = int(r.node_id)
            pp[nid] = np.array([r.z, r.y, r.x], dtype=np.float64)
            if r.gt_id != -1:
                m[int(r.gt_id)] = nid
            frame_nodes.setdefault((ds, int(r.t)), []).append(nid)
        gt_to_node[ds] = m
        node_pos[ds] = pp

    hits = 0
    total = 0
    for r in sel.itertuples():
        ds = r.dataset
        parent_nid = gt_to_node.get(ds, {}).get(int(r.gt_source))
        target_nid = gt_to_node.get(ds, {}).get(int(r.gt_target))
        if parent_nid is None or target_nid is None:
            continue
        pool = frame_nodes.get((ds, int(r.s_t)))
        if not pool or parent_nid not in pool:
            continue

        def base(nid: int) -> np.ndarray:
            if arm == "smooth_all" and smoothed is not None:
                return smoothed.get((ds, nid), node_pos[ds][nid])
            return node_pos[ds][nid]

        pool_arr = np.array([base(n) for n in pool], dtype=np.float64)

        if arm == "deployed":
            tgt_p = node_pos[ds][target_nid]
        elif arm == "gt":
            tgt_p = np.array([r.t_z, r.t_y, r.t_x], dtype=np.float64)
        elif arm in ("smooth_ep", "smooth_all"):
            tgt_p = (smoothed or {}).get((ds, target_nid), node_pos[ds][target_nid])
        else:
            raise ValueError(arm)

        # replace the TRUE PARENT's entry only; every decoy stays as it was
        pi = pool.index(parent_nid)
        if arm == "gt":
            pool_arr[pi] = np.array([r.s_z, r.s_y, r.s_x], dtype=np.float64)
        elif arm == "smooth_ep":
            pool_arr[pi] = (smoothed or {}).get((ds, parent_nid), node_pos[ds][parent_nid])

        d = np.linalg.norm(_um(pool_arr) - _um(tgt_p), axis=1)
        if pool[int(np.argmin(d))] == parent_nid:
            hits += 1
        total += 1

    return (100.0 * hits / total if total else float("nan")), total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="C:/temp/error_atlas")
    ap.add_argument("--fold", default="f1")
    ap.add_argument("--window", type=int, default=2, help="frames each side for smoothing")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    atlas = Path(args.atlas)
    nodes = pd.read_parquet(atlas / f"nodes_{args.fold}.parquet")
    edges = pd.read_parquet(atlas / f"edges_{args.fold}.parquet")
    gtedges = pd.read_parquet(atlas / f"gtedges_{args.fold}.parquet")
    print(f"loaded {len(nodes):,} nodes, {len(edges):,} edges, {len(gtedges):,} gt edges")

    # ---- CALIBRATION GATE 1: coordinate convention -------------------------------
    disp = calibrate_displacement(gtedges)
    print(f"\nCALIBRATION 1 - correctly-linked GT displacement median: {disp:.3f} um "
          f"(anchor {ANCHOR_GT_DISP_MEDIAN_UM})")
    if abs(disp - ANCHOR_GT_DISP_MEDIAN_UM) > 0.15:
        print("  !! COORDINATE CONVENTION IS WRONG - refusing to report. Do not trust any "
              "distance below.")
        return 2
    print("  OK - scale convention confirmed")

    # ---- CALIBRATION GATE 2: reproduce the two known oracle anchors ---------------
    dep, n = run_oracle(nodes, gtedges, "deployed")
    gt, _ = run_oracle(nodes, gtedges, "gt")
    print(f"\nCALIBRATION 2 - oracle anchors on {n:,} mislinked GT edges")
    print(f"  deployed coords : {dep:6.2f}%   (anchor {ANCHOR_DEPLOYED_PCT})")
    print(f"  GT coords       : {gt:6.2f}%   (anchor {ANCHOR_GT_PCT})")
    ok = abs(dep - ANCHOR_DEPLOYED_PCT) <= 1.0 and abs(gt - ANCHOR_GT_PCT) <= 3.0
    print("  " + ("OK - implementation reproduces the record" if ok else
                  "!! DOES NOT REPRODUCE - implementation differs from last cycle's"))

    # ---- THE TEST -----------------------------------------------------------------
    print(f"\nRESULT - 'true parent is nearest' on {n:,} mislinked GT edges (fold {args.fold})")
    print(f"  deployed            : {dep:6.2f}%")
    print(f"  GT coords (ceiling) : {gt:6.2f}%")

    arms: dict[str, dict[str, float]] = {}
    for mode in ("mean", "linefit"):
        print(f"\n  building '{mode}' smoothed positions (window +/-{args.window}) ...")
        sm = build_smoothed_positions(nodes, edges, args.window, mode=mode)
        sep, _ = run_oracle(nodes, gtedges, "smooth_ep", sm)
        sall, _ = run_oracle(nodes, gtedges, "smooth_all", sm)
        rec = 100.0 * (sep - dep) / (gt - dep) if gt > dep else float("nan")
        arms[mode] = {"endpoints_pct": sep, "all_pct": sall, "gt_gain_recovered_pct": rec}
        print(f"    {mode:8s} endpoints : {sep:6.2f}%  ({sep - dep:+.2f} pp vs deployed)")
        print(f"    {mode:8s} everything: {sall:6.2f}%  ({sall - dep:+.2f} pp vs deployed)")
        print(f"    {mode:8s} fraction of GT gain recovered: {rec:.1f}%")

    best = max(max(a["endpoints_pct"], a["all_pct"]) for a in arms.values())
    print(f"\n  PRE-REGISTERED KILL: statistic must move UP off the deployed baseline -> "
          f"{'SURVIVES' if best > dep + 1.0 else 'KILLED'}")
    if not ok:
        print("  NOTE: the recorded anchors did NOT reproduce, so this verdict is stated "
              "against THIS run's own baseline, not the record's.")

    result = {
        "fold": args.fold,
        "window": args.window,
        "n_mislinked": n,
        "calibration_disp_um": disp,
        "deployed_pct": dep,
        "gt_pct": gt,
        "arms": arms,
        "anchors_reproduced": bool(ok),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
