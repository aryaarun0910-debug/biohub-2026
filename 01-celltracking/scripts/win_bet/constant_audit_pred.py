r"""Constant audit, prediction side -- the gates that are calibrated against the PREDICTED
graph, not against ground truth.

WHY A SECOND SCRIPT
-------------------
`constant_audit.py` measures gates whose true distribution lives in GT (division geometry,
step length, track length). But three deployed gates are defined over the DETECTOR's output
density, and GT is a sparse subset annotation -- ~6.7 nodes/frame against the detector's
~153 -- so GT is the wrong ruler for them:

  BIOHUB_GAP_DENSITY_REFERENCE_UM  6.5   -- the neighbour-spacing set point the adaptive
                                            gap gate expands/contracts around
  BIOHUB_MOTION_RELINK_MAX_FRAME_NODES 2600 -- a hard bail-out: if ANY frame exceeds it,
                                            motion relink returns [] for the WHOLE crop
  BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP  0.00375 -- budget of safe-division edges per crop

This runs over the deployed LOEO submission CSVs (the real deployment substrate) and reports
where each set point sits in the distribution it is actually applied to.

Usage:
  .venv\Scripts\python.exe scripts\win_bet\constant_audit_pred.py \
      --csv c:/temp/subvoxel_f0/loeo_split0_strict.csv.gz \
            c:/temp/subvoxel_f1/loeo_split1_strict.csv.gz
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])


def summary(v: np.ndarray) -> dict:
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {"n": int(v.size), "mean": float(v.mean()), "p1": float(q[0]), "p5": float(q[1]),
            "p10": float(q[2]), "p25": float(q[3]), "median": float(q[4]), "p75": float(q[5]),
            "p90": float(q[6]), "p95": float(q[7]), "p99": float(q[8]), "max": float(v.max())}


def pct_of(v: np.ndarray, gate: float) -> float:
    return 100.0 * float(np.mean(v <= gate)) if v.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    nn_spacing: list[float] = []
    frame_nodes: list[int] = []
    edge_um: list[float] = []
    fork_parent_um: list[float] = []
    fork_sister_um: list[float] = []
    per_crop: list[dict] = []

    for csv in args.csv:
        print(f"reading {csv} ...", flush=True)
        df = pd.read_csv(csv)
        nodes = df[df.row_type == "node"]
        edges = df[df.row_type == "edge"]
        for ds, gn in nodes.groupby("dataset"):
            ge = edges[edges.dataset == ds]
            pos = np.stack([gn["z"].to_numpy() * SCALE[0],
                            gn["y"].to_numpy() * SCALE[1],
                            gn["x"].to_numpy() * SCALE[2]], axis=1)
            t = gn["t"].to_numpy().astype(np.int64)
            ids = gn["node_id"].to_numpy().astype(np.int64)
            idx = {int(n): i for i, n in enumerate(ids)}

            for tt in np.unique(t):
                m = t == tt
                frame_nodes.append(int(m.sum()))
                if m.sum() >= 5:
                    tree = cKDTree(pos[m])
                    d, _ = tree.query(pos[m], k=4)
                    nn_spacing.extend(np.median(d[:, 1:], axis=1).tolist())

            src = ge["source_id"].to_numpy().astype(np.int64)
            tgt = ge["target_id"].to_numpy().astype(np.int64)
            ok = np.array([(int(a) in idx and int(b) in idx) for a, b in zip(src, tgt)])
            if ok.any():
                a = np.array([idx[int(v)] for v in src[ok]])
                b = np.array([idx[int(v)] for v in tgt[ok]])
                edge_um.extend(np.linalg.norm(pos[b] - pos[a], axis=1).tolist())

            out = collections.defaultdict(list)
            for a, b in zip(src, tgt):
                if int(a) in idx and int(b) in idx:
                    out[int(a)].append(int(b))
            n_fork = 0
            for s, kids in out.items():
                if len(kids) >= 2:
                    n_fork += 1
                    i = idx[s]
                    d0, d1 = idx[kids[0]], idx[kids[1]]
                    fork_parent_um.append(float(np.linalg.norm(pos[d0] - pos[i])))
                    fork_parent_um.append(float(np.linalg.norm(pos[d1] - pos[i])))
                    fork_sister_um.append(float(np.linalg.norm(pos[d1] - pos[d0])))
            # residual gap-close opportunity: a track END at t and a track START at t+2 within
            # the effective gate (GAP_CLOSE_UM * 2 = 11.6 um). If gap-close had spare budget
            # and a wide enough gate these would already be consumed, so a large residue means
            # either the gate or the GAP_CLOSE_MAX_ADDED_* budget is binding.
            has_out = set(int(v) for v in src)
            has_in = set(int(v) for v in tgt)
            ends = np.array([i for i, n in enumerate(ids) if int(n) not in has_out])
            starts = np.array([i for i, n in enumerate(ids) if int(n) not in has_in])
            resid = 0
            if ends.size and starts.size:
                st_by_t: dict[int, list[int]] = collections.defaultdict(list)
                for i in starts:
                    st_by_t[int(t[i])].append(int(i))
                for i in ends:
                    cand_i = st_by_t.get(int(t[i]) + 2, [])
                    if cand_i:
                        dd = np.linalg.norm(pos[cand_i] - pos[i], axis=1)
                        resid += int((dd <= 11.6).any())
            per_crop.append({"dataset": ds, "nodes": int(len(gn)), "edges": int(len(ge)),
                             "forks": n_fork,
                             "gap_budget": min(2000, max(1, int(round(len(gn) * 0.05)))),
                             "residual_gap_ops": resid,
                             "global_cap": max(1, int(round(max(1, len(ge)) * 0.00375)))})

    out_json: dict = {}
    nn = np.array(nn_spacing)
    fn = np.array(frame_nodes, dtype=np.float64)
    eu = np.array(edge_um)
    fp = np.array(fork_parent_um)
    fs = np.array(fork_sister_um)

    print()
    print("=== PREDICTED-GRAPH DISTRIBUTIONS (deployment substrate) ===")
    print(f"nodes/frame                : {summary(fn)}")
    print(f"  MOTION_RELINK_MAX_FRAME_NODES=2600 -> max observed {fn.max():.0f}; "
          f"{pct_of(fn, 2600):.3f}% of frames under the cap")
    out_json["nodes_per_frame"] = summary(fn)

    print(f"local neighbour spacing um : {summary(nn)}")
    for g in (6.5,):
        print(f"  GAP_DENSITY_REFERENCE_UM={g} sits at percentile {pct_of(nn, g):.1f}% "
              f"(median spacing {np.median(nn):.2f} um)")
    out_json["nn_spacing_um"] = summary(nn)
    out_json["nn_spacing_um"]["pct_at_6.5"] = pct_of(nn, 6.5)

    # --- is the "adaptive" density term actually adaptive, or pinned at its clip?
    REF, GAIN, MAXD, BLEND = 6.5, 0.040, 0.125, 0.20
    blended = (1.0 - BLEND) * nn + BLEND * REF
    raw_delta = GAIN * (blended - REF)
    delta = np.clip(raw_delta, -MAXD, MAXD)
    print("  adaptive-density term (GAIN=0.040, clip +/-0.125, blend 0.20 toward 6.5):")
    print(f"    saturated HIGH at +0.125 : {100.0 * (raw_delta >= MAXD).mean():.1f}% of nodes")
    print(f"    saturated LOW  at -0.125 : {100.0 * (raw_delta <= -MAXD).mean():.1f}% of nodes")
    print(f"    contracting at all (<0)  : {100.0 * (raw_delta < 0).mean():.1f}% of nodes")
    print(f"    mean applied delta       : {delta.mean():+.4f} um/step "
          f"(on a {11.6:.1f} um gate -> {100.0 * 2 * delta.mean() / 11.6:+.2f}%)")
    print(f"    data-implied REFERENCE_UM (median spacing) = {np.median(nn):.2f} um")
    out_json["density_term"] = {
        "pct_saturated_high": 100.0 * float((raw_delta >= MAXD).mean()),
        "pct_saturated_low": 100.0 * float((raw_delta <= -MAXD).mean()),
        "pct_contracting": 100.0 * float((raw_delta < 0).mean()),
        "mean_delta_um": float(delta.mean()),
        "data_implied_reference_um": float(np.median(nn)),
    }

    print(f"emitted edge length um     : {summary(eu)}")
    for lab, g in (("MOTION_RELINK_TIGHT_UM", 6.0), ("MOTION_RELINK_RELAXED_UM", 10.0),
                   ("OUTPUT_EDGE_MAX_UM", 14.0)):
        print(f"  {lab}={g} -> {pct_of(eu, g):.2f}% of emitted edges at or below")
    out_json["edge_um"] = summary(eu)

    print(f"emitted fork parent dist um: {summary(fp)}")
    print(f"emitted fork sister dist um: {summary(fs)}")
    out_json["fork_parent_um"] = summary(fp)
    out_json["fork_sister_um"] = summary(fs)

    # ---- global division cap binding analysis
    fo = np.array([c["forks"] for c in per_crop], dtype=np.float64)
    cap = np.array([c["global_cap"] for c in per_crop], dtype=np.float64)
    ratio = fo / np.maximum(cap, 1)
    print()
    print("=== SAFE_DIV_GLOBAL_FRAC_CAP = 0.00375 BINDING ANALYSIS ===")
    print(f"crops={len(per_crop)}  forks total={int(fo.sum())}  cap total={int(cap.sum())}")
    print(f"  forks EXACTLY at cap : {int((fo == cap).sum())} crops")
    print(f"  forks at or above cap: {int((fo >= cap).sum())} crops "
          f"({100.0 * (fo >= cap).mean():.1f}%)")
    print(f"  forks/cap ratio: {summary(ratio)}")
    out_json["global_cap_binding"] = {
        "n_crops": len(per_crop), "forks_total": int(fo.sum()), "cap_total": int(cap.sum()),
        "n_exactly_at_cap": int((fo == cap).sum()),
        "n_at_or_above_cap": int((fo >= cap).sum()),
        "ratio": summary(ratio),
    }

    # ---- gap-close budget binding
    gb = np.array([c["gap_budget"] for c in per_crop], dtype=np.float64)
    rg = np.array([c["residual_gap_ops"] for c in per_crop], dtype=np.float64)
    print()
    print("=== GAP_CLOSE_MAX_ADDED_FRAC=0.05 / _ABS=2000 BINDING ANALYSIS ===")
    print(f"  per-crop synthetic-node budget: {summary(gb)}")
    print(f"  crops where the ABS cap (2000) is the binding one: "
          f"{int((gb >= 2000).sum())} of {len(per_crop)}")
    print(f"  residual UNCLOSED gap opportunities (end at t, start at t+2, <=11.6 um): "
          f"{summary(rg)}")
    print(f"  total residual {int(rg.sum())} vs total budget {int(gb.sum())} "
          f"-> budget is {'BINDING' if rg.sum() > gb.sum() else 'NOT binding'}")
    out_json["gap_budget"] = {"budget": summary(gb), "residual_ops": summary(rg),
                              "total_residual": int(rg.sum()), "total_budget": int(gb.sum())}

    if args.json:
        Path(args.json).write_text(json.dumps(out_json, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
