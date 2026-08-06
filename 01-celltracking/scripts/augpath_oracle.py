"""LANE C0 — augmenting-path association: ORACLE REALISABILITY GATE.

The deployed association runs a tight 6 um assignment, CONSUMES endpoints, then offers the
relaxed 10 um surface only the leftovers. A valid relaxed edge is therefore never even
enumerated when one of its endpoints was taken by a locally attractive tight assignment.

This measures the CEILING of repairing that, using GT only to choose. It is not a deployable
method; it answers "is there anything here worth building a selector for?".

Construction (fixed nodes, approximately fixed cardinality):
  * candidate surface = the UNION of tight and relaxed pairs (<= 10 um, consecutive frames),
    i.e. exactly the starvation fix -- the relaxed surface is no longer restricted to leftovers;
  * one joint assignment per frame pair over that union, cost = deployed relink cost minus a
    large bonus for GT-correct pairs, so the oracle maximises GT true positives while otherwise
    reproducing the deployed ordering on unannotated pairs;
  * cardinality capped at the deployed relink's own edge count for that frame pair, so this is a
    RE-AIM, not a widening;
  * in-degree <= 1 and out-degree <= 1 inside the relink, as deployed; no nodes added.

Then the complete wrapper runs on the re-aimed edge set and the exact scorer is applied.

GATE: >= 12% recovery of 6bba edge FNs AND >= +0.020 exact pooled oracle delta.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import pathlib
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[1]
PREGRAPH_ROOT = (REPO.parent / "Biohub-CellTracking-2026_RESEARCH" / "agent_runs"
                 / "ws_f_armB_p0b_2026-08-01" / "data")
SCALE = (1.625, 0.40625, 0.40625)
MATCH_MAX_UM = 7.0
TIGHT_UM, RELAXED_UM = 6.0, 10.0
GT_BONUS = 1.0e6

DEPLOY_ENV = {
    "BIOHUB_OUTPUT_FILTER_SHORT_TRACKS": "1", "BIOHUB_MOTION_RELINK_LEARNED_BONUS": "1.0",
    "BIOHUB_GAP_CLOSE_MAX_GAP": "2", "BIOHUB_GAP_CLOSE_UM": "5.8",
    "BIOHUB_GAP_DENSITY_ADAPTIVE": "1", "BIOHUB_GAP_DENSITY_REFERENCE_UM": "6.5",
    "BIOHUB_GAP_DENSITY_GAIN": "0.040", "BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM": "0.125",
    "BIOHUB_GAP_DENSITY_NEIGHBORS": "3", "BIOHUB_OUTPUT_MIN_TRACK_LEN": "6",
    "BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS": "1", "BIOHUB_OUTPUT_GAP2_RECOVERY": "0",
    "BIOHUB_SAFE_DIV_MAX_UM": "4.66", "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "8.5",
    "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "7.65",
    "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.0076", "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.00375",
    "BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE": "0",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
}
_S: dict = {}


def _build():
    os.environ.update(DEPLOY_ENV)
    spec = json.loads((REPO / "scripts/kaggle_specs/p2_armb_flowgate.json").read_text("utf-8"))
    edits = [e for e in spec["edits"]
             if e.get("code_file", "").endswith("armb_flow_gate.py")
             or (e["kind"] == "replace" and "GATE_COMPENSATION" in e.get("new", ""))]
    text = (REPO / "src/biotrack/wrapper.py").read_text("utf-8")
    for e in edits:
        if e["kind"] == "insert_before":
            text = text.replace(
                e["anchor"], "\n" + (REPO / e["code_file"]).read_text("utf-8") + "\n" + e["anchor"], 1)
        else:
            text = text.replace(e["old"], e["new"], 1)
    text = text.replace('ARMB_FLOW_GATE = os.environ.get("BIOHUB_ARMB_FLOW_GATE", "1") != "0"',
                        "ARMB_FLOW_GATE = True")
    p = REPO / "artifacts" / "_augpath" / f"_wrapper_{os.getpid()}.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    s = importlib.util.spec_from_file_location(f"augpath_{os.getpid()}", p)
    m = importlib.util.module_from_spec(s)
    sys.modules[s.name] = m
    s.loader.exec_module(m)
    m.TEST_DIR = None
    return m


def init_worker():
    global _S
    sys.path.insert(0, str(REPO / "src"))
    _S = {"m": _build()}


def _match_gt_to_pred(gt_pos_by_t, pred_pos_by_t):
    """Nearest-centroid matching per frame, <= MATCH_MAX_UM, one-to-one (same rule as scorer)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    out = {}
    for t, gts in gt_pos_by_t.items():
        preds = pred_pos_by_t.get(t)
        if not preds:
            continue
        gi, gp = zip(*gts)
        pi, pp = zip(*preds)
        G = np.stack(gp)
        P = np.stack(pp)
        D = np.linalg.norm(G[:, None, :] - P[None, :, :], axis=2)
        big = MATCH_MAX_UM + 1e6
        C = np.where(D <= MATCH_MAX_UM, D, big)
        r, c = linear_sum_assignment(C)
        for a, b in zip(r, c):
            if C[a, b] < big:
                out[gi[a]] = pi[b]
    return out


def work(path):
    import numpy as np
    import polars as pl
    import tracksdata as td
    from scipy.optimize import linear_sum_assignment
    from biotrack.metric import load_graph, score_pred_graph

    m = _S["m"]
    name = pathlib.Path(path).stem
    if name.endswith("__nodes"):
        name = name[: -len("__nodes")]
    gt_path = REPO / "data" / "train" / f"{name}.geff"
    if not gt_path.exists():
        return None

    _p = pathlib.Path(path)
    if _p.name.endswith("__nodes.parquet"):
        nd = pl.read_parquet(_p)
        ed = pl.read_parquet(str(_p).replace("__nodes.parquet", "__edges.parquet"))
        nodes = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]),
                                     "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])}
                 for r in nd.iter_rows(named=True)}
        edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"]),
                  "edge_prob": float(r["edge_prob"])} for r in ed.iter_rows(named=True)]
    else:
        d = pl.read_parquet(_p)
        nd = d.filter(pl.col("row_type") == "node")
        ed = d.filter(pl.col("row_type") == "edge")
        nodes = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]),
                                     "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])}
                 for r in nd.iter_rows(named=True)}
        edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"])}
                 for r in ed.iter_rows(named=True)]

    def pos(n):
        return np.array([n["z"] * SCALE[0], n["y"] * SCALE[1], n["x"] * SCALE[2]])

    # ---- GT -> pre-wrapper pred node identity -----------------------------------------
    gt = load_graph(gt_path)
    gna = gt.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
    gt_pos_by_t = defaultdict(list)
    for iid, t, z, y, x in zip(gna[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(), gna["t"].to_list(),
                               gna["z"].to_list(), gna["y"].to_list(), gna["x"].to_list()):
        gt_pos_by_t[int(t)].append((int(iid), np.array([float(z) * SCALE[0],
                                                        float(y) * SCALE[1],
                                                        float(x) * SCALE[2]])))
    pred_pos_by_t = defaultdict(list)
    for nid, n in nodes.items():
        pred_pos_by_t[int(n["t"])].append((nid, pos(n)))
    g2p = _match_gt_to_pred(gt_pos_by_t, pred_pos_by_t)

    gea = gt.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                                   td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
    gt_pairs = set()
    gt_edges_total = gt.num_edges()
    for s, t in zip(gea[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(),
                    gea[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()):
        a, b = g2p.get(int(s)), g2p.get(int(t))
        if a is not None and b is not None:
            gt_pairs.add((a, b))

    # ---- DEPLOYED arm B ---------------------------------------------------------------
    nD, eD, _ = m.filter_output_graph({k: dict(v) for k, v in nodes.items()},
                                      [dict(e) for e in edges], dataset=name)
    from_graph = _to_graph(nD, eD)
    rowD = score_pred_graph(from_graph, gt_path)

    # ---- ORACLE re-aim of the relink ---------------------------------------------------
    orig_relink = m.motion_relink_edges
    stats_moves = defaultdict(int)

    def oracle_relink(nodes_by_id, stats, learned_edge_probs=None):
        """MINIMAL-INTERVENTION oracle: start from the deployed matching and apply only
        alternating moves that actually convert a GT FN into a TP. Everything else is left
        exactly as deployed, so the composite delta measures recovered edges rather than the
        side-effects of arbitrary churn among unannotated pairs."""
        deployed = orig_relink(nodes_by_id, stats, learned_edge_probs)
        P = {nid: pos(n) for nid, n in nodes_by_id.items()}
        T_OF = {nid: int(n["t"]) for nid, n in nodes_by_id.items()}
        sel_src, sel_tgt = {}, {}
        for e in deployed:
            s, tt = int(e["source_id"]), int(e["target_id"])
            sel_src[s] = tt
            sel_tgt[tt] = s
        n_dep = len(deployed)

        def near(x, y):
            return float(np.linalg.norm(P[y] - P[x])) <= RELAXED_UM

        # only GT pairs that are actually reachable: both nodes present, consecutive, <=10um
        targets = [(u, v) for (u, v) in gt_pairs
                   if u in T_OF and v in T_OF and T_OF[v] == T_OF[u] + 1
                   and sel_src.get(u) != v and near(u, v)]
        for u, v in targets:
            vu, uv = sel_src.get(u), sel_tgt.get(v)
            if vu is None and uv is None:                       # both free: pure augment
                sel_src[u], sel_tgt[v] = v, u
                stats_moves["added"] += 1
                stats_moves["gt_gained"] += 1
                stats_moves["changed"] += 1
            elif vu is not None and uv is None:                 # source taken -> 1-for-1 swap
                if (u, vu) in gt_pairs:
                    continue                                    # never break a true edge
                del sel_tgt[vu]
                sel_src[u], sel_tgt[v] = v, u
                stats_moves["swap1"] += 1
                stats_moves["gt_gained"] += 1
                stats_moves["changed"] += 2
            elif vu is None and uv is not None:                 # target taken -> 1-for-1 swap
                if (uv, v) in gt_pairs:
                    continue
                del sel_src[uv]
                sel_src[u], sel_tgt[v] = v, u
                stats_moves["swap1"] += 1
                stats_moves["gt_gained"] += 1
                stats_moves["changed"] += 2
            else:                                               # both taken -> 2-for-2
                if (u, vu) in gt_pairs or (uv, v) in gt_pairs:
                    continue
                del sel_tgt[vu]
                del sel_src[uv]
                sel_src[u], sel_tgt[v] = v, u
                stats_moves["gt_gained"] += 1
                if near(uv, vu) and uv not in sel_src and vu not in sel_tgt:
                    sel_src[uv], sel_tgt[vu] = vu, uv    # reconnect the displaced pair
                    stats_moves["swap2_reconnected"] += 1
                    stats_moves["changed"] += 2
                else:
                    stats_moves["swap2_dropped"] += 1
                    stats_moves["changed"] += 3
        stats_moves["pairs"] += 1
        stats_moves["deployed"] += n_dep
        stats_moves["oracle"] += len(sel_src)
        stats_moves["gt_reachable"] += len(targets)
        out = [{"source_id": s, "target_id": tt, "edge_prob": 0.0,
                "distance_um": float(np.linalg.norm(P[tt] - P[s])),
                "motion_distance_um": float(np.linalg.norm(P[tt] - P[s])),
                "motion_relinked": 1, "motion_pass": "oracle"}
               for s, tt in sorted(sel_src.items())]
        stats["motion_relink_edges"] = len(out)
        return out

    m.motion_relink_edges = oracle_relink
    try:
        nO, eO, _ = m.filter_output_graph({k: dict(v) for k, v in nodes.items()},
                                          [dict(e) for e in edges], dataset=name)
    finally:
        m.motion_relink_edges = orig_relink
    rowO = score_pred_graph(_to_graph(nO, eO), gt_path)

    keep = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
            "num_pred_nodes", "total_node_ratio", "node_recall")
    return {"crop": name, "family": name.split("_")[0],
            "deployed": {k: rowD[k] for k in keep},
            "oracle": {k: rowO[k] for k in keep},
            "moves": dict(stats_moves), "gt_edges": gt_edges_total,
            "gt_pairs_mapped": len(gt_pairs)}


def _to_graph(out_nodes, out_edges):
    import polars as pl
    import tracksdata as td
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    ids = sorted(out_nodes)
    internal = g.bulk_add_nodes([
        {"t": int(out_nodes[i]["t"]),
         "z": float(max(0, int(round(float(out_nodes[i]["z"]))))),
         "y": float(max(0, int(round(float(out_nodes[i]["y"]))))),
         "x": float(max(0, int(round(float(out_nodes[i]["x"])))))}
        for i in ids])
    remap = {sid: internal[k] for k, sid in enumerate(ids)}
    if out_edges:
        g.bulk_add_edges([{"source_id": remap[int(e["source_id"])],
                           "target_id": remap[int(e["target_id"])]} for e in out_edges])
    return g


ALPHA, DIVW = 0.1, 0.1


def composite(rows, arm):
    num = den = 0.0
    dtp = dfp = dfn = 0
    for r in rows:
        d = r[arm]
        tp, fp, fn = d["edge_tp"], d["edge_fp"], d["edge_fn"]
        w = tp + fp + fn
        if w <= 0:
            continue
        tnr = d["total_node_ratio"]
        if tnr != tnr:
            tnr = 0.0
        num += w * max(0.0, (tp / w) * (1 - ALPHA * tnr))
        den += w
        dtp += d["division_tp"]
        dfp += d["division_fp"]
        dfn += d["division_fn"]
    adj = num / den if den else float("nan")
    dd = dtp + dfp + dfn
    return adj + DIVW * (dtp / dd if dd else 0.0)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--substrate", choices=("prewrapper", "postwrapper"),
                    default="prewrapper")
    # No default. The previous default wrote `augpath_oracle.json`, and the file left at
    # that path was the RETRACTED post-wrapper run (40 crops, not 199) -- quarantined as
    # `augpath_oracle_RETRACTED_postwrapper_40crops.json`. A default output path let a
    # retracted artifact sit where a valid one was expected, and would have let a rerun
    # silently overwrite it. `--out` is now required and must name its substrate.
    ap.add_argument("--out", required=True,
                    help="output path; include the substrate in the name, e.g. "
                         "reports/inventory/augpath_oracle_prewrapper_199.json")
    a = ap.parse_args()

    if a.substrate == "prewrapper":
        crops = (sorted(glob.glob(str(PREGRAPH_ROOT / "f0_pregraphs" / "*__nodes.parquet")))
                 + sorted(glob.glob(str(PREGRAPH_ROOT / "f1_pregraphs" / "*__nodes.parquet"))))
    else:
        crops = sorted(glob.glob(str(REPO / "artifacts/kaggle/p0strict_cache/graphs/*/*.parquet")))
    crops = crops[::a.stride]
    if a.limit:
        crops = crops[: a.limit]
    print(f"crops={len(crops)} workers={a.workers}", flush=True)
    rows = []
    with mp.Pool(a.workers, initializer=init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(work, crops), 1):
            if r:
                rows.append(r)
            if i % 5 == 0 or i == len(crops):
                print(f"  {i}/{len(crops)}", flush=True)

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)

    cd, co = composite(rows, "deployed"), composite(rows, "oracle")
    print("\n" + "=" * 78)
    print(f"GATE C0 — AUGMENTING-PATH ORACLE   ({len(rows)} crops)")
    print("=" * 78)
    print(f"  deployed composite : {cd:.6f}")
    print(f"  oracle   composite : {co:.6f}")
    print(f"  ORACLE DELTA       : {co-cd:+.6f}      gate >= +0.020  "
          f"{'PASS' if co-cd >= 0.020 else 'FAIL'}")

    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam]
        fnd = sum(r["deployed"]["edge_fn"] for r in sub)
        fno = sum(r["oracle"]["edge_fn"] for r in sub)
        rec = (fnd - fno) / max(fnd, 1)
        line = (f"  {fam}: FN {fnd} -> {fno}   recovery {100*rec:6.2f}%"
                f"   composite {composite(sub,'deployed'):.6f} -> {composite(sub,'oracle'):.6f}")
        if fam == "6bba":
            line += f"   gate >= 12% {'PASS' if rec >= 0.12 else 'FAIL'}"
        print(line)

    mv = defaultdict(int)
    for r in rows:
        for k, v in r["moves"].items():
            mv[k] += v
    print(f"\n  moves: frame-pairs {mv['pairs']}  deployed edges {mv['deployed']}  "
          f"oracle edges {mv['oracle']}  changed {mv['changed']}  GT gained {mv['gt_gained']}")
    if mv["changed"]:
        print(f"  TRUE-MOVE BASE RATE (GT gained / edges changed) : "
              f"{mv['gt_gained']/mv['changed']:.5f}   "
              f"({mv['changed']/max(mv['gt_gained'],1):.1f} changes per useful one)")
    print(f"  cardinality preserved: deployed {mv['deployed']} vs oracle {mv['oracle']} "
          f"(delta {mv['oracle']-mv['deployed']:+d})")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
