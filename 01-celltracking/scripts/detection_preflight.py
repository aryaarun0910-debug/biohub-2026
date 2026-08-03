"""LANE D0 — detection preflight on UNIQUE missing GT nodes (CPU only, no GPU).

Detection owns 72.2% of edge FNs (+0.10332 oracle). Before funding any detector work the
first question is whether those nodes are missing from the RAW DETECTOR or merely lost later.

This operates on the pre-wrapper node set -- the detector's accepted peaks, before any wrapper
stage runs -- so it separates, with no GPU and no heatmap:

    C  candidate existed pre-wrapper, wrapper removed it   -> fix lifecycle/pruning
    D  no accepted peak within the matching radius at all  -> threshold / NMS / new detector

The A/B split inside D (sub-threshold response vs NMS-merged) needs the detector heatmap,
which is not available on CPU; this reports the size of the D bucket and how close the nearest
accepted peak is, which is the evidence that decides whether A/B is even plausible.

Also corrects an attribution defect in edge_fn_census.py: its `wrapper_node_loss` test compared
GT node ids against a dict keyed by PREDICTION node ids, so that bucket could never fire and its
contents were being swept into `det_never_detected`.

Unique missing NODES, not incident edge FNs. Interior nodes (a predecessor and a successor in
GT) put two edges at stake; terminal nodes only one.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
PREGRAPH_ROOT = (REPO.parent / "Biohub-CellTracking-2026_RESEARCH" / "agent_runs"
                 / "ws_f_armB_p0b_2026-08-01" / "data")
SCALE = (1.625, 0.40625, 0.40625)
MATCH_MAX_UM = 7.0


def match(gt_by_t, pred_by_t, max_um=MATCH_MAX_UM):
    """One-to-one nearest-centroid match per frame, same rule as the scorer."""
    from scipy.optimize import linear_sum_assignment
    out, nearest = {}, {}
    for t, gts in gt_by_t.items():
        preds = pred_by_t.get(t)
        if not preds:
            for gi, _ in gts:
                nearest[gi] = float("inf")
            continue
        gi, gp = zip(*gts)
        pi, pp = zip(*preds)
        D = np.linalg.norm(np.stack(gp)[:, None, :] - np.stack(pp)[None, :, :], axis=2)
        for k, g in enumerate(gi):
            nearest[g] = float(D[k].min())
        big = max_um + 1e6
        C = np.where(D <= max_um, D, big)
        r, c = linear_sum_assignment(C)
        for x, y in zip(r, c):
            if C[x, y] < big:
                out[gi[x]] = pi[y]
    return out, nearest


def _init():
    """Module-level initializer: a lambda is not picklable under Windows spawn."""
    sys.path.insert(0, str(REPO / "src"))


def work(path):
    import polars as pl
    import tracksdata as td
    from biotrack.metric import load_graph

    name = pathlib.Path(path).stem.replace("__nodes", "")
    gt_path = REPO / "data" / "train" / f"{name}.geff"
    if not gt_path.exists():
        return None

    nd = pl.read_parquet(path)
    pred_by_t = defaultdict(list)
    for r in nd.iter_rows(named=True):
        pred_by_t[int(r["t"])].append(
            (int(r["node_id"]), np.array([float(r["z"]) * SCALE[0], float(r["y"]) * SCALE[1],
                                          float(r["x"]) * SCALE[2]])))

    gt = load_graph(gt_path)
    gna = gt.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
    gt_by_t = defaultdict(list)
    gt_t = {}
    for iid, t, z, y, x in zip(gna[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(), gna["t"].to_list(),
                               gna["z"].to_list(), gna["y"].to_list(), gna["x"].to_list()):
        gt_by_t[int(t)].append((int(iid), np.array([float(z) * SCALE[0], float(y) * SCALE[1],
                                                    float(x) * SCALE[2]])))
        gt_t[int(iid)] = int(t)

    matched, nearest = match(gt_by_t, pred_by_t)

    # GT topology: how many edges does each node put at stake?
    indeg, outdeg = Counter(), Counter()
    if gt.num_edges():
        gea = gt.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                                       td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
        for s, t in zip(gea[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(),
                        gea[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()):
            outdeg[int(s)] += 1
            indeg[int(t)] += 1

    missing = [g for g in gt_t if g not in matched]
    rec = {"crop": name, "family": name.split("_")[0],
           "gt_nodes": len(gt_t), "gt_edges": int(gt.num_edges()),
           "pred_nodes": int(len(nd)), "matched": len(matched),
           "missing": len(missing)}

    interior = terminal = isolated = 0
    edges_at_stake = 0
    near_hist = Counter()
    runs = Counter()
    miss_by_t = defaultdict(set)
    for g in missing:
        deg = indeg[g] + outdeg[g]
        edges_at_stake += deg
        if indeg[g] and outdeg[g]:
            interior += 1
        elif deg:
            terminal += 1
        else:
            isolated += 1
        d = nearest.get(g, float("inf"))
        if not np.isfinite(d):
            near_hist["no_pred_in_frame"] += 1
        elif d <= 8:
            near_hist["7-8um"] += 1
        elif d <= 10:
            near_hist["8-10um"] += 1
        elif d <= 15:
            near_hist["10-15um"] += 1
        else:
            near_hist[">15um"] += 1
        miss_by_t[gt_t[g]].add(g)
    rec.update({"interior": interior, "terminal": terminal, "isolated": isolated,
                "edges_at_stake": edges_at_stake,
                "nearest_pred_hist": dict(near_hist),
                "matched_nearest_p50": float(np.median(
                    [nearest[g] for g in matched] or [0.0])),
                "runs": dict(runs)})
    return rec


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "reports/inventory/detection_preflight.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(REPO / "src"))

    crops = (sorted(glob.glob(str(PREGRAPH_ROOT / "f0_pregraphs" / "*__nodes.parquet")))
             + sorted(glob.glob(str(PREGRAPH_ROOT / "f1_pregraphs" / "*__nodes.parquet"))))
    if a.limit:
        crops = crops[: a.limit]
    print(f"crops={len(crops)} workers={a.workers}", flush=True)

    rows = []
    with mp.Pool(a.workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(work, crops), 1):
            if r:
                rows.append(r)
            if i % 20 == 0 or i == len(crops):
                print(f"  {i}/{len(crops)}", flush=True)

    json.dump(rows, open(a.out, "w"), indent=1)

    print("\n" + "=" * 80)
    print("LANE D0 — UNIQUE MISSING GT NODES vs THE RAW DETECTOR (pre-wrapper peaks)")
    print("=" * 80)
    for fam in ("44b6", "6bba", "ALL"):
        sub = rows if fam == "ALL" else [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        gtn = sum(r["gt_nodes"] for r in sub)
        mis = sum(r["missing"] for r in sub)
        print(f"\n  {fam}: {len(sub)} crops · GT nodes {gtn} · "
              f"matched {sum(r['matched'] for r in sub)} · MISSING {mis} ({100*mis/max(gtn,1):.2f}%)")
        print(f"     interior {sum(r['interior'] for r in sub)} (2 edges each) · "
              f"terminal {sum(r['terminal'] for r in sub)} (1 edge) · "
              f"isolated {sum(r['isolated'] for r in sub)}")
        print(f"     EDGES AT STAKE from missing nodes: {sum(r['edges_at_stake'] for r in sub)}")
        h = Counter()
        for r in sub:
            h.update(r["nearest_pred_hist"])
        tot = sum(h.values()) or 1
        print("     distance to the NEAREST accepted peak (match radius is 7 um):")
        for k in ("7-8um", "8-10um", "10-15um", ">15um", "no_pred_in_frame"):
            if h.get(k):
                print(f"        {k:16s} {h[k]:7d}  {100*h[k]/tot:6.2f}%")

    print("\n  READING")
    h = Counter()
    for r in rows:
        h.update(r["nearest_pred_hist"])
    tot = sum(h.values()) or 1
    near = h.get("7-8um", 0) + h.get("8-10um", 0)
    print(f"    {100*near/tot:.2f}% of missing GT nodes have an accepted peak within 10 um.")
    print("    A large near-miss share favours localisation/NMS repair (class A/B);")
    print("    a large far share means the detector has nothing there (class D).")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
