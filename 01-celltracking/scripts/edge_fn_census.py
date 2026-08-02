"""LANE B — edge-FN first-failure-stage census on the P0-strict substrate.

Runs the COMPLETE wrapper for both arms on every cached P0-strict crop:

    arm A  = deployed P0-B path (BIOHUB_ARMB_FLOW_GATE=0)   -> per-crop score (Lane A input,
                                                               and the baseline parity check)
    arm B  = flow-compensated relink gate                    -> per-crop score + FN attribution

For arm B every GT edge that we miss is attributed to the FIRST stage that lost it, using the
relink's own `_CANDIDATE_SINK` surface plus per-stage edge snapshots. GT<->pred node identity
comes from `MATCHED_NODE_ID`, which the exact scorer writes onto the predicted graph, so the
attribution uses the same matcher as the score.

Stages (first one that applies wins):
  det_never_detected            GT endpoint has no matched predicted node at all
  wrapper_node_loss             endpoint existed pre-wrapper but the wrapper dropped it
  assoc_enum_beyond_cap         both endpoints present, pair never within the relaxed radius
  assoc_bipartite_source_taken  pair enumerated; source went to a different target
  assoc_bipartite_target_taken  pair enumerated; target took a different source
  assoc_bipartite_both_taken    pair enumerated; both endpoints consumed elsewhere
  assoc_enum_relaxed_starved    pair only reachable in the relaxed pass, endpoint consumed in tight
  assoc_post_relink_deleted     pair WAS selected by the relink but a later stage removed it
  final_endpoint_only           endpoints survive, pair never enumerated for another reason

Writes one JSON per crop shard plus a combined roll-up. No GPU, no submission.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import pathlib
import sys
from collections import Counter, defaultdict

REPO = pathlib.Path(__file__).resolve().parents[1]
PREGRAPH_ROOT = (REPO.parent / "Biohub-CellTracking-2026_RESEARCH" / "agent_runs"
                 / "ws_f_armB_p0b_2026-08-01" / "data")

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
TIGHT_UM, RELAXED_UM = 6.0, 10.0
_S: dict = {}


def _build(gate_on: bool):
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
                        f"ARMB_FLOW_GATE = {gate_on!r}")
    tag = "B" if gate_on else "A"
    # per-process filename: workers would otherwise race on the same generated module
    path = REPO / "artifacts" / "_fncensus" / f"_wrapper_{tag}_{os.getpid()}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    s = importlib.util.spec_from_file_location(f"fncensus_{tag}_{os.getpid()}", path)
    m = importlib.util.module_from_spec(s)
    sys.modules[s.name] = m
    s.loader.exec_module(m)
    m.TEST_DIR = None
    return m


def init_worker():
    global _S
    sys.path.insert(0, str(REPO / "src"))
    _S = {"A": _build(False), "B": _build(True)}


def _instrument(mod):
    """Snapshot the edge set after each mutating stage. Returns (snaps, restore)."""
    snaps: dict[str, set] = {}
    orig = {
        "gap": mod.close_single_frame_gaps,
        "gap2": mod.recover_strict_gap2,
        "safediv": mod.add_safe_divisions_postlink,
        "short": mod.filter_short_track_components,
    }

    def keys(edges):
        return {(int(e["source_id"]), int(e["target_id"])) for e in edges}

    def w_gap(n, e, s, **kw):
        snaps["post_relink_repair"] = keys(e)
        r = orig["gap"](n, e, s, **kw)
        snaps["post_gap"] = keys(r[1])
        return r

    def w_gap2(n, e, s, **kw):
        r = orig["gap2"](n, e, s, **kw)
        snaps["post_gap2"] = keys(r[1])
        return r

    def w_sd(n, e, s, **kw):
        r = orig["safediv"](n, e, s, **kw)
        snaps["post_safediv"] = keys(r)
        return r

    def w_short(n, e, s, **kw):
        snaps["pre_shorttrack"] = keys(e)
        r = orig["short"](n, e, s, **kw)
        snaps["post_shorttrack"] = keys(r[1])
        snaps["_nodes_post_shorttrack"] = set(r[0])
        return r

    mod.close_single_frame_gaps = w_gap
    mod.recover_strict_gap2 = w_gap2
    mod.add_safe_divisions_postlink = w_sd
    mod.filter_short_track_components = w_short

    def restore():
        mod.close_single_frame_gaps = orig["gap"]
        mod.recover_strict_gap2 = orig["gap2"]
        mod.add_safe_divisions_postlink = orig["safediv"]
        mod.filter_short_track_components = orig["short"]

    return snaps, restore


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
    return g, remap, ids


def work(path):
    import numpy as np
    import polars as pl
    import tracksdata as td
    from biotrack.metric import load_graph, score_pred_graph

    name = pathlib.Path(path).stem
    if name.endswith("__nodes"):
        name = name[: -len("__nodes")]
    gt_path = REPO / "data" / "train" / f"{name}.geff"
    if not gt_path.exists():
        return None
    fam = name.split("_")[0]

    p = pathlib.Path(path)
    if p.name.endswith("__nodes.parquet"):
        # PRE-WRAPPER substrate (WS-F f0/f1 pregraphs): carries edge_prob and the full
        # relink node population. This is the ONLY substrate that can express a pre-wrapper
        # gate change -- see reports/inventory/wsf_ROUTE1_VERDICT.json.
        nd = pl.read_parquet(p)
        ed = pl.read_parquet(str(p).replace("__nodes.parquet", "__edges.parquet"))
        base_nodes = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]),
                                          "z": float(r["z"]), "y": float(r["y"]),
                                          "x": float(r["x"])}
                      for r in nd.iter_rows(named=True)}
        base_edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"]),
                       "edge_prob": float(r["edge_prob"])}
                      for r in ed.iter_rows(named=True)]
    else:
        d = pl.read_parquet(p)
        nd = d.filter(pl.col("row_type") == "node")
        ed = d.filter(pl.col("row_type") == "edge")
        base_nodes = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]),
                                          "z": float(r["z"]), "y": float(r["y"]),
                                          "x": float(r["x"])}
                      for r in nd.iter_rows(named=True)}
        base_edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"])}
                      for r in ed.iter_rows(named=True)]

    out = {"crop": name, "family": fam, "prewrapper_nodes": len(base_nodes)}

    # ---------------- arm A: score only (Lane A input + parity) ----------------
    mA = _S["A"]
    gA, _, _ = _to_graph(*mA.filter_output_graph(
        {k: dict(v) for k, v in base_nodes.items()}, [dict(e) for e in base_edges],
        dataset=name)[:2])
    rowA = score_pred_graph(gA, gt_path)
    out["armA"] = {k: rowA[k] for k in
                   ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp",
                    "division_fn", "num_pred_nodes", "node_recall", "total_node_ratio",
                    "edge_jaccard", "adj_edge_jaccard")}

    # ---------------- arm B: score + FN attribution ----------------
    mB = _S["B"]
    snaps, restore = _instrument(mB)
    sink: list = []
    mB._CANDIDATE_SINK = sink
    try:
        nB, eB, statsB = mB.filter_output_graph(
            {k: dict(v) for k, v in base_nodes.items()}, [dict(e) for e in base_edges],
            dataset=name)
    finally:
        mB._CANDIDATE_SINK = None
        restore()

    gB, remapB, idsB = _to_graph(nB, eB)
    rowB = score_pred_graph(gB, gt_path)
    out["armB"] = {k: rowB[k] for k in
                   ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp",
                    "division_fn", "num_pred_nodes", "node_recall", "total_node_ratio",
                    "edge_jaccard", "adj_edge_jaccard")}
    out["armB_stats"] = {k: statsB.get(k, 0) for k in
                         ("motion_relink_edges", "safe_divisions_added", "gap_added_edges",
                          "short_track_edges_removed", "pruned_isolated_nodes")}

    # ---- compact the candidate surface -------------------------------------------------
    enum_pass: dict[tuple, str] = {}
    sel_src: dict[int, int] = {}
    sel_tgt: dict[int, int] = {}
    tight_src: set = set()
    tight_tgt: set = set()
    for c in sink:
        s, t = int(c["source_id"]), int(c["target_id"])
        p = c["pass"]
        if (s, t) not in enum_pass or p == "tight":
            enum_pass[(s, t)] = p
        if c["selected"]:
            sel_src[s] = t
            sel_tgt[t] = s
            if p == "tight":
                tight_src.add(s)
                tight_tgt.add(t)
    del sink

    # ---- GT <-> pred identity from the scorer's own matcher ----------------------------
    na = gB.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID,
                                  td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    internal_to_sub = {internal: sub for sub, internal in remapB.items()}
    gt_to_pred: dict[int, int] = {}
    for iid, mid in zip(na[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
                        na[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list()):
        if mid is not None and int(mid) >= 0:
            gt_to_pred[int(mid)] = internal_to_sub[iid]

    gt = load_graph(gt_path)
    gea = gt.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                                   td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
    final_edges = {(int(e["source_id"]), int(e["target_id"])) for e in eB}
    surviving_nodes = set(nB)

    stage = Counter()
    detail: list = []
    pos = {i: np.array([base_nodes[i]["z"] * 1.625, base_nodes[i]["y"] * 0.40625,
                        base_nodes[i]["x"] * 0.40625]) for i in base_nodes}

    for gs, gtt in zip(gea[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(),
                       gea[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()):
        pu, pv = gt_to_pred.get(int(gs)), gt_to_pred.get(int(gtt))
        if pu is not None and pv is not None and (pu, pv) in final_edges:
            continue                                            # a TP, not an FN
        # ---- endpoint failures ----
        if pu is None or pv is None:
            missing = [x for x, p in ((int(gs), pu), (int(gtt), pv)) if p is None]
            lost = any(m in base_nodes and m not in surviving_nodes for m in missing)
            stage["wrapper_node_loss" if lost else "det_never_detected"] += 1
            detail.append({"stage": "wrapper_node_loss" if lost else "det_never_detected"})
            continue
        # ---- both endpoints present: why is the pair absent? ----
        if (pu, pv) in snaps.get("post_relink_repair", set()) or \
           (pu, pv) in snaps.get("post_safediv", set()):
            stage["assoc_post_relink_deleted"] += 1
            detail.append({"stage": "assoc_post_relink_deleted", "src": pu, "tgt": pv})
            continue
        dist = (float(np.linalg.norm(pos[pv] - pos[pu]))
                if pu in pos and pv in pos else float("inf"))
        src_taken = pu in sel_src and sel_src[pu] != pv
        tgt_taken = pv in sel_tgt and sel_tgt[pv] != pu
        p = enum_pass.get((pu, pv))

        if p is None:
            # NEVER ENUMERATED. `assign_pass` only receives endpoints still unmatched, so a
            # pair whose source (or target) was consumed by the tight pass is never even
            # offered to the relaxed pass -- that is starvation, not "beyond cap".
            if pu not in base_nodes or pv not in base_nodes:
                st = "assoc_synthetic_endpoint"
            elif (pu in nB and pv in nB
                  and int(nB[pv]["t"]) != int(nB[pu]["t"]) + 1):
                st = "assoc_nonconsecutive"
            elif dist > RELAXED_UM:
                st = "assoc_enum_beyond_cap"
            elif pu in tight_src or pv in tight_tgt:
                st = "assoc_enum_relaxed_starved"
            else:
                st = "final_endpoint_only"
        elif p == "relaxed" and (pu in tight_src or pv in tight_tgt):
            st = "assoc_enum_relaxed_starved"
        elif src_taken and tgt_taken:
            st = "assoc_bipartite_both_taken"
        elif src_taken:
            st = "assoc_bipartite_source_taken"
        elif tgt_taken:
            st = "assoc_bipartite_target_taken"
        else:
            st = "final_endpoint_only"
        stage[st] += 1
        detail.append({"stage": st, "src": pu, "tgt": pv, "pass": p,
                       "dist_um": round(dist, 4) if np.isfinite(dist) else None,
                       "src_taken": bool(src_taken), "tgt_taken": bool(tgt_taken)})

    out["fn_stages"] = dict(stage)
    out["fn_total_attributed"] = sum(stage.values())
    out["fn_detail"] = detail
    out["gt_edges"] = gt.num_edges()
    out["gt_matched_nodes"] = len(gt_to_pred)
    out["gt_nodes"] = gt.num_nodes()
    # base-rate denominators: how big is the surface a selector would have to rank?
    out["candidate_surface"] = len(enum_pass)
    out["candidate_surface_tight"] = sum(1 for v in enum_pass.values() if v == "tight")
    out["candidate_surface_relaxed"] = sum(1 for v in enum_pass.values() if v == "relaxed")
    out["relink_selected"] = len(sel_src)
    out["final_edges"] = len(final_edges)
    return out


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(REPO / "reports/inventory/edge_fn_census.json"))
    ap.add_argument("--substrate", choices=("prewrapper", "postwrapper"), default="prewrapper",
                    help="prewrapper = WS-F f0/f1 pregraphs (has edge_prob, full relink node "
                         "population). postwrapper = p0strict_cache, which ROUTE1_VERDICT "
                         "rejects for any pre-wrapper change.")
    a = ap.parse_args()

    if a.substrate == "prewrapper":
        base = pathlib.Path(PREGRAPH_ROOT)
        crops = sorted(glob.glob(str(base / "f0_pregraphs" / "*__nodes.parquet"))) + \
            sorted(glob.glob(str(base / "f1_pregraphs" / "*__nodes.parquet")))
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
    print(f"\nwrote {a.out}  ({len(rows)} crops)")

    agg = defaultdict(Counter)
    for r in rows:
        for k, v in r["fn_stages"].items():
            agg[r["family"]][k] += v
            agg["ALL"][k] += v
    print("\nEDGE-FN FIRST-FAILURE STAGE")
    fams = [f for f in ("44b6", "6bba") if f in agg]
    print(f"  {'stage':32s} " + "".join(f"{f:>10s}" for f in fams) + f"{'ALL':>10s}{'%':>8s}")
    tot = sum(agg["ALL"].values()) or 1
    for st in sorted(agg["ALL"], key=lambda s: -agg["ALL"][s]):
        print(f"  {st:32s} " + "".join(f"{agg[f][st]:10d}" for f in fams)
              + f"{agg['ALL'][st]:10d}{100*agg['ALL'][st]/tot:7.2f}%")
    print(f"  {'TOTAL':32s} " + "".join(f"{sum(agg[f].values()):10d}" for f in fams)
          + f"{tot:10d}")


if __name__ == "__main__":
    main()
