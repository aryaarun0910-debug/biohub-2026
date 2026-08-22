r"""Constant audit -- for every numeric gate in the deployed pipeline, where does the
deployed value sit inside the distribution of the TRUE quantity it is gating?

WHY THIS EXISTS
---------------
Four constants have already been caught sitting on the wrong side of a measurable property
of the data (BIOHUB_DET_THRESHOLD 0.96875 vs an analytic break-even ~0.50;
BIOHUB_SAFE_DIV_MAX_UM 4.66 vs a GT parent-daughter median of 7.42/8.87 um). Every one of
them was INHERITED from a public notebook and never swept. This script measures the true
distribution behind EVERY remaining distance / length / rate gate, and reports the
percentile the deployed value occupies.

The percentile IS the finding. A gate at the 10th percentile of the true values it was
meant to admit is discarding 90% of them.

Everything here is COUNTS and DISTANCES over ground truth -- no scoring, no model, no GPU.
It is therefore immune to the LOEO->LB transfer problem
(research/06-knowledge-system/internal-reports/loeo_lb_gap_2026-08-18.md).

Usage:
  .venv\Scripts\python.exe scripts\win_bet\constant_audit.py --gt-dir data/train
  .venv\Scripts\python.exe scripts\win_bet\constant_audit.py --gt-dir data/train --json out.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import zarr

SCALE = np.array([1.625, 0.40625, 0.40625])  # z, y, x um per level-0 voxel


def load_geff(path: Path):
    g = zarr.open(str(path), mode="r")
    nid = np.asarray(g["nodes/ids"][:]).astype(np.int64)
    t = np.asarray(g["nodes/props/t/values"][:]).astype(np.int64)
    z = np.asarray(g["nodes/props/z/values"][:]).astype(np.float64)
    y = np.asarray(g["nodes/props/y/values"][:]).astype(np.float64)
    x = np.asarray(g["nodes/props/x/values"][:]).astype(np.float64)
    pos = np.stack([z * SCALE[0], y * SCALE[1], x * SCALE[2]], axis=1)
    e = np.asarray(g["edges/ids"][:])
    if e.ndim != 2 or e.size == 0:
        e = np.zeros((0, 2), dtype=np.int64)
    return nid, t, pos, e.astype(np.int64)


def pct_of(values: np.ndarray, gate: float) -> float:
    """Fraction of the true distribution at or below the gate, in percent."""
    if values.size == 0:
        return float("nan")
    return 100.0 * float(np.mean(values <= gate))


def summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    q = np.percentile(values, [5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "p5": float(q[0]), "p10": float(q[1]), "p25": float(q[2]),
        "median": float(q[3]), "p75": float(q[4]), "p90": float(q[5]),
        "p95": float(q[6]), "p99": float(q[7]),
        "max": float(values.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="data/train")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    geffs = sorted(gt_dir.glob("*.geff"))

    # accumulators, split by embryo prefix and pooled
    acc: dict[str, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))

    n_nodes = n_edges = n_div = 0
    comp_sizes_by_prefix: dict[str, list[int]] = collections.defaultdict(list)
    track_lens_by_prefix: dict[str, list[int]] = collections.defaultdict(list)
    frame_div_rate: list[float] = []       # divisions / (cells with a forward link) per frame
    frame_node_counts: list[int] = []
    per_crop_div_frac_of_edges: list[float] = []

    for gp in geffs:
        ds = gp.stem
        prefix = ds.split("_", 1)[0]
        nid, t, pos, e = load_geff(gp)
        n_nodes += len(nid)
        n_edges += len(e)

        idx = {int(n): i for i, n in enumerate(nid)}
        out = collections.defaultdict(list)
        indeg = collections.Counter()
        for a, b in e:
            a, b = int(a), int(b)
            if a in idx and b in idx:
                out[a].append(b)
                indeg[b] += 1

        # ---- per-frame node counts
        for tt, c in collections.Counter(t.tolist()).items():
            frame_node_counts.append(c)

        # ---- edge displacement: continuation vs division
        cont_d, div_parent_d, sister_d = [], [], []
        n_div_here = 0
        for a, kids in out.items():
            ia = idx[a]
            if len(kids) == 1:
                b = kids[0]
                cont_d.append(float(np.linalg.norm(pos[idx[b]] - pos[ia])))
            elif len(kids) >= 2:
                n_div_here += 1
                d0, d1 = kids[0], kids[1]
                a0 = float(np.linalg.norm(pos[idx[d0]] - pos[ia]))
                a1 = float(np.linalg.norm(pos[idx[d1]] - pos[ia]))
                div_parent_d.append(a0)
                div_parent_d.append(a1)
                # The linker attaches ONE daughter first; SAFE_DIV_MAX_UM gates the
                # remaining (i.e. typically the FARTHER) one, while
                # SAFE_DIV_EXISTING_CHILD_MAX_UM gates the already-linked (NEARER) one.
                acc[prefix]["div_far_daughter_um"].append(max(a0, a1))
                acc[prefix]["div_near_daughter_um"].append(min(a0, a1))
                sister_d.append(float(np.linalg.norm(pos[idx[d1]] - pos[idx[d0]])))
        n_div += n_div_here
        if len(e):
            per_crop_div_frac_of_edges.append(n_div_here / len(e))

        acc[prefix]["cont_edge_um"] += cont_d
        acc[prefix]["div_parent_um"] += div_parent_d
        acc[prefix]["div_sister_um"] += sister_d

        # ---- two-frame displacement along a continuation chain a->b->c
        two_frame, mid_offset = [], []
        for a, kids in out.items():
            if len(kids) != 1:
                continue
            b = kids[0]
            kb = out.get(b, [])
            if len(kb) != 1:
                continue
            c = kb[0]
            pa, pb, pc = pos[idx[a]], pos[idx[b]], pos[idx[c]]
            two_frame.append(float(np.linalg.norm(pc - pa)))
            mid_offset.append(float(np.linalg.norm(pb - 0.5 * (pa + pc))))
        acc[prefix]["two_frame_um"] += two_frame
        acc[prefix]["mid_offset_um"] += mid_offset

        # ---- linefit residual: |node - line fit over +/-2 neighbours| on continuation chains
        # build continuation chains (successor unique AND predecessor unique)
        succ = {a: kids[0] for a, kids in out.items() if len(kids) == 1}
        starts = [int(n) for n in nid if indeg[int(n)] == 0 or True]
        seen = set()
        resid = []
        comp_track_lens = []
        for a in list(succ) + [int(n) for n in nid]:
            if a in seen:
                continue
            # only start a chain at a node whose predecessor does not uniquely lead to it
            chain = []
            cur = a
            while cur is not None and cur not in seen:
                seen.add(cur)
                chain.append(cur)
                cur = succ.get(cur)
            if len(chain) >= 2:
                comp_track_lens.append(len(chain))
            if len(chain) >= 5:
                P = np.stack([pos[idx[c]] for c in chain])
                for i in range(2, len(chain) - 2):
                    win = P[i - 2:i + 3]
                    tt_ = np.arange(-2, 3, dtype=np.float64)
                    fit = np.stack([
                        np.polyval(np.polyfit(tt_, win[:, k], 1), 0.0) for k in range(3)
                    ])
                    resid.append(float(np.linalg.norm(P[i] - fit)))
        acc[prefix]["linefit_resid_um"] += resid
        track_lens_by_prefix[prefix] += comp_track_lens

        # ---- connected component sizes (what OUTPUT_MIN_TRACK_LEN actually filters)
        parent = {int(n): int(n) for n in nid}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        for a, kids in out.items():
            for b in kids:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        comps = collections.Counter(find(int(n)) for n in nid)
        comp_sizes_by_prefix[prefix] += list(comps.values())

        # ---- per-frame division rate against the frame-cap denominator
        # frame cap denominator in the wrapper = #sources with out-degree exactly 1 at t
        div_at_t = collections.Counter()
        for a, kids in out.items():
            if len(kids) >= 2:
                div_at_t[int(t[idx[a]])] += 1
        src1_at_t = collections.Counter()
        for a, kids in out.items():
            if len(kids) == 1:
                src1_at_t[int(t[idx[a]])] += 1
        for tt in set(div_at_t) | set(src1_at_t):
            denom = src1_at_t.get(tt, 0)
            if denom > 0:
                frame_div_rate.append(div_at_t.get(tt, 0) / denom)

        # ---- nearest-neighbour spacing (k=3 median) -- GAP_DENSITY_REFERENCE_UM
        from scipy.spatial import cKDTree
        for tt in np.unique(t):
            m = t == tt
            if m.sum() < 5:
                continue
            tree = cKDTree(pos[m])
            d, _ = tree.query(pos[m], k=min(4, int(m.sum())))
            acc[prefix]["nn_spacing_um"] += list(np.median(d[:, 1:], axis=1))

    # ---------- report ----------
    print(f"crops={len(geffs)} nodes={n_nodes} edges={n_edges} divisions={n_div}")
    print()

    GATES = [
        # (label, accumulator key, deployed value, env var)
        ("OUTPUT_EDGE_MAX_UM",            "cont_edge_um",     14.0,   "BIOHUB_OUTPUT_EDGE_MAX_UM"),
        ("MOTION_RELINK_TIGHT_UM",        "cont_edge_um",      6.0,   "BIOHUB_MOTION_RELINK_TIGHT_UM"),
        ("MOTION_RELINK_RELAXED_UM",      "cont_edge_um",     10.0,   "BIOHUB_MOTION_RELINK_RELAXED_UM"),
        ("GAP_CLOSE_UM x2 (t->t+2 gate)", "two_frame_um",     11.6,   "BIOHUB_GAP_CLOSE_UM"),
        ("GAP_CLOSE_REUSE_UM",            "mid_offset_um",     3.2,   "BIOHUB_GAP_CLOSE_REUSE_UM"),
        ("GAP_REFINE_MAX_SHIFT_UM",       "linefit_resid_um",  3.2,   "BIOHUB_GAP_REFINE_MAX_SHIFT_UM"),
        ("SAFE_DIV_MAX_UM (far daughter)","div_far_daughter_um", 4.66, "BIOHUB_SAFE_DIV_MAX_UM"),
        ("SAFE_DIV_MAX_UM (both pooled)", "div_parent_um",     4.66,  "BIOHUB_SAFE_DIV_MAX_UM"),
        ("SAFE_DIV_EXIST_CHILD (near)",   "div_near_daughter_um", 7.65, "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"),
        ("SAFE_DIV_SISTER_MAX_UM",        "div_sister_um",     8.5,   "BIOHUB_SAFE_DIV_SISTER_MAX_UM"),
        ("GAP_DENSITY_REFERENCE_UM (GT)", "nn_spacing_um",     6.5,   "BIOHUB_GAP_DENSITY_REFERENCE_UM"),
    ]

    prefixes = sorted(acc)
    out_json: dict = {"n_crops": len(geffs), "n_nodes": n_nodes,
                      "n_edges": n_edges, "n_divisions": n_div, "gates": {}}

    hdr = f"{'gate':34s} {'deployed':>9s} " + " ".join(
        f"{p+' med':>10s} {p+' pct':>9s}" for p in prefixes
    ) + f" {'POOLED med':>11s} {'POOLED pct':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for label, key, val, env in GATES:
        pooled = np.array(sum((acc[p][key] for p in prefixes), []), dtype=np.float64)
        row = f"{label:34s} {val:9.3f} "
        rec = {"deployed": val, "env": env, "quantity": key, "by_prefix": {}}
        for p in prefixes:
            v = np.array(acc[p][key], dtype=np.float64)
            row += f" {np.median(v) if v.size else float('nan'):10.2f} {pct_of(v, val):8.1f}%"
            rec["by_prefix"][p] = {"median": float(np.median(v)) if v.size else None,
                                   "pct_at_gate": pct_of(v, val), **summary(v)}
        row += f" {np.median(pooled):11.2f} {pct_of(pooled, val):10.1f}%"
        rec["pooled"] = {"pct_at_gate": pct_of(pooled, val), **summary(pooled)}
        out_json["gates"][label] = rec
        print(row)

    # ---- count-style gates
    print()
    print("=== COUNT / RATE GATES ===")
    comps_pooled = np.array(sum(comp_sizes_by_prefix.values(), []), dtype=np.float64)
    print(f"GT connected-component sizes: {summary(comps_pooled)}")
    for m in (2, 4, 6, 8, 10):
        print(f"  MIN_TRACK_LEN={m:2d} would delete {pct_of(comps_pooled, m - 1):5.2f}% "
              f"of GT components ({int((comps_pooled < m).sum())} of {comps_pooled.size})")
    out_json["gt_component_sizes"] = summary(comps_pooled)
    out_json["min_track_len_curve"] = {
        str(m): {"pct_components_deleted": pct_of(comps_pooled, m - 1),
                 "n_components_deleted": int((comps_pooled < m).sum())}
        for m in range(2, 13)
    }

    tl = np.array(sum(track_lens_by_prefix.values(), []), dtype=np.float64)
    print(f"GT continuation-chain lengths: {summary(tl)}")
    out_json["gt_chain_lengths"] = summary(tl)

    fnc = np.array(frame_node_counts, dtype=np.float64)
    print(f"GT nodes per frame: {summary(fnc)}")
    print(f"  MOTION_RELINK_MAX_FRAME_NODES=2600 -> {pct_of(fnc, 2600):.2f}% of frames under cap")
    out_json["gt_nodes_per_frame"] = summary(fnc)

    fdr = np.array(frame_div_rate, dtype=np.float64)
    print(f"GT per-frame division rate (divisions / out-deg-1 sources): {summary(fdr)}")
    print(f"  SAFE_DIV_FRAME_FRAC_CAP=0.0076 sits at pct {pct_of(fdr, 0.0076):.1f}% "
          f"of true per-frame rates")
    out_json["gt_frame_div_rate"] = summary(fdr)
    out_json["gt_frame_div_rate"]["pct_at_0.0076"] = pct_of(fdr, 0.0076)

    gdf = np.array(per_crop_div_frac_of_edges, dtype=np.float64)
    print(f"GT divisions as a fraction of all edges (per crop): {summary(gdf)}")
    print(f"  SAFE_DIV_GLOBAL_FRAC_CAP=0.00375 sits at pct {pct_of(gdf, 0.00375):.1f}% "
          f"of true per-crop division fractions")
    out_json["gt_div_frac_of_edges"] = summary(gdf)
    out_json["gt_div_frac_of_edges"]["pct_at_0.00375"] = pct_of(gdf, 0.00375)

    # ---- joint division-gate admission sweep (all three gates must pass together)
    print()
    print("=== JOINT DIVISION-GATE SWEEP (fraction of the 151 GT divisions geometrically admissible) ===")
    far = np.array(sum((acc[p]["div_far_daughter_um"] for p in prefixes), []), dtype=np.float64)
    near = np.array(sum((acc[p]["div_near_daughter_um"] for p in prefixes), []), dtype=np.float64)
    sis = np.array(sum((acc[p]["div_sister_um"] for p in prefixes), []), dtype=np.float64)
    combos = [
        ("DEPLOYED", 4.66, 7.65, 8.5),
        ("public 0.923", 12.0, 12.0, 15.0),
        ("+1 step", 6.0, 8.0, 10.0),
        ("+2 step", 8.0, 10.0, 12.0),
        ("median-centred", 7.5, 6.0, 10.6),
        ("uncapped", 1e9, 1e9, 1e9),
    ]
    out_json["joint_div_sweep"] = {}
    print(f"{'setting':16s} {'parent':>8s} {'child':>7s} {'sister':>7s} {'admissible':>11s}")
    for name, pmax, cmax, smax in combos:
        ok = (far <= pmax) & (near <= cmax) & (sis <= smax)
        frac = 100.0 * float(ok.mean()) if ok.size else float("nan")
        print(f"{name:16s} {pmax:8.2f} {cmax:7.2f} {smax:7.2f} "
              f"{int(ok.sum()):5d}/{ok.size} {frac:6.1f}%")
        out_json["joint_div_sweep"][name] = {
            "parent_max": pmax, "child_max": cmax, "sister_max": smax,
            "n_admissible": int(ok.sum()), "n_total": int(ok.size), "pct": frac,
        }
    # one-at-a-time relaxation from the deployed point
    print("\n  one-at-a-time relaxation from DEPLOYED (4.66 / 7.65 / 8.5):")
    for lab, i in (("parent", 0), ("child", 1), ("sister", 2)):
        for v in (4.66, 6.0, 8.0, 10.0, 12.0, 14.0, 1e9):
            g = [4.66, 7.65, 8.5]
            g[i] = v
            ok = (far <= g[0]) & (near <= g[1]) & (sis <= g[2])
            print(f"    {lab:7s}={v:8.2f} -> {int(ok.sum()):3d}/{ok.size} admissible "
                  f"({100.0 * ok.mean():5.1f}%)")

    if args.json:
        Path(args.json).write_text(json.dumps(out_json, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
