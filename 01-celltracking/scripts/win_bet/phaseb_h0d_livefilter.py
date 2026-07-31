"""H0d — honest live-path replay of the frozen H0c cascade, with the REAL wrapper
component filter applied after the fork edits, plus a node-retention guard.

WHY THIS EXISTS
---------------
`scripts/win_bet/phaseb_h0c_replay.py` re-used one immutable `node_rows` frame for both
the baseline and the treatment score, so `N_pred` COULD NOT differ and the
`node_count_invariant` assertion tested nothing. The live path is not invariant:
`src/biotrack/wrapper.py::filter_short_track_components` keeps a component when

    len(members) >= min_track_len or (OUTPUT_KEEP_DIVISION_COMPONENTS and has_division)

so a short component that survives ONLY because it contains a fork is DELETED once the
fork is suppressed. This script measures that instead of assuming it away.

WHAT IS REPLAYED
----------------
The cached graphs under `artifacts/kaggle/**/graphs` are already POST-wrapper, so the
honest replay is: take the cached graph as the state a division module would edit, apply
the edits, then re-run the two wrapper stages that consume edge topology --

    1. the `OUTPUT_PRUNE_ISOLATED` block of `filter_output_graph` (replicated verbatim,
       4 lines, marked below), and
    2. `wrapper.filter_short_track_components` (the REAL function, imported, not copied).

Both stages are idempotent on their own output, so re-running them on the UNEDITED cached
graph must be the identity. That identity is asserted per crop (`refilter_identity`); it is
what makes any node deletion in a treatment arm attributable to the edits rather than to
the replay design.

`OUTPUT_LINEFIT_SMOOTH` is deliberately NOT re-run: it is a coordinate blend, not
idempotent, and re-applying it to already-smoothed cached coordinates would double-smooth
both arms. Its topology-sensitive footprint is reported instead as `linefit_exposed_nodes`.

ARMS (each scored with the exact patched scorer)
-----------------------------------------------
  base              cached graph, unedited                      (the E0c/surface baseline)
  h0c_post          suppress-all + add-replace, NO re-filter     (== published H0c)
  h0c_refilt        suppress-all + add-replace, re-filter, no guard      (the live path)
  h0c_refilt_guard  suppress-all + add-replace, re-filter, retention guard
  supp_post         suppress-all only, NO re-filter
  supp_refilt       suppress-all only, re-filter, no guard

NODE-RETENTION GUARD (GT-free, deployment-legal)
------------------------------------------------
Before any edit, find every component that the wrapper keeps ONLY through its division
exemption: `len(members) < min_track_len and has_division`. Those node ids are whitelisted
and re-added to the post-filter keep set. The whitelist is a pure function of the
pre-edit predicted graph, so it uses nothing that is unavailable at test time.

The degenerate "keep everything that survived the pre-edit filter" guard is also computed
and asserted equal to the full node set (it must be, because the cache is post-filter);
that identity is exactly why `h0c_post` is the no-deletion upper bound and needs no
separate scoring.

MOTHER-COLLISION ASSERTION
--------------------------
`gt_to_sub[int(m)] = s` in the D0P/H0c lineage overwrites silently when two predicted
nodes match the same GT node. Harmless under E0c (bipartite matching is one-to-one there)
but a latent bug on any new node population, so `build_gt_maps` here raises unless
`--allow-gt-collisions` is passed, in which case the collision count is recorded per crop.

The proposer is NOT re-implemented: `shortlist` and `CFG` are imported verbatim from
`phaseb_h0c_replay` (config hash 04eeac97500d) so every surface is measured with one
literally identical frozen definition.

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\phaseb_h0d_livefilter.py --surfaces e0c --workers 6
  .venv\\Scripts\\python.exe scripts\\win_bet\\phaseb_h0d_livefilter.py --surfaces e0c,v122 --workers 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from phaseb_d0p_proposer import SCALE  # noqa: E402
from phaseb_h0c_replay import CFG, CFG_HASH, shortlist  # noqa: E402  (frozen, verbatim)

DEFAULT_OUT = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH"
                   r"\agent_runs\agent2\phaseb_h0d_livefilter.json")

# Graph surfaces. Every one of these stores the identical parquet schema
# (row_type, node_id, t, z, y, x, source_id, target_id).
#
# `min_len` is the SHORT-TRACK THRESHOLD THE SURFACE WAS ACTUALLY BUILT WITH, and it is NOT
# the wrapper default. E0c set `OUTPUT_MIN_TRACK_LEN = 7` (scripts/win_bet/e0c_run.py), v122
# sets `BIOHUB_OUTPUT_MIN_TRACK_LEN = 6`. Using the wrapper's own default (6) against the E0c
# cache silently makes the re-filter a no-op superset and hides every deletion, so the value
# is taken from each surface's own status manifest per crop and only cross-checked against
# the constant here.
SURFACES: dict[str, dict] = {
    "e0c": {"graphs": "artifacts/kaggle/e0c_cache/graphs",
            "status": "artifacts/kaggle/e0c_cache/status", "arm": None, "min_len": 7},
    "v122": {"graphs": "artifacts/kaggle/coupled_cache/arms/D",
             "status": "artifacts/kaggle/coupled_cache/shared_status", "arm": "D", "min_len": 6},
    "ilp_c": {"graphs": "artifacts/kaggle/coupled_cache/arms/C",
              "status": "artifacts/kaggle/coupled_cache/shared_status", "arm": "C", "min_len": 7},
    "clean903": {"graphs": "artifacts/kaggle/clean903_wrapper_oof_cache/graphs",
                 "status": "artifacts/kaggle/clean903_wrapper_oof_cache/status",
                 "arm": None, "min_len": 6},
}
E0C_ANCHOR = {0: 0.7595, 1: 0.6490}
ARMS = ("base", "h0c_post", "h0c_refilt", "h0c_refilt_guard", "supp_post", "supp_refilt")


# ---------------------------------------------------------------- surface plumbing
def surface_dir(surface: str) -> Path:
    return ROOT / SURFACES[surface]["graphs"]


def surface_min_track_len(surface: str, split: int, crop: str) -> int:
    """Read the short-track threshold this surface was built with, from its own manifest."""
    spec = SURFACES[surface]
    p = ROOT / spec["status"] / f"{split}__{crop}.json"
    declared = int(spec["min_len"])
    if not p.exists():
        return declared
    d = json.loads(p.read_text())
    if spec["arm"] is not None:
        stats = (d.get("arms", {}).get(spec["arm"], {}) or {}).get("wrapper_stats", {}) or {}
    else:
        stats = d.get("diagnostics", {}) or {}
    got = stats.get("short_track_min_len_effective")
    if got is None:
        return declared
    got = int(got)
    assert got == declared, (f"{surface}/{split}/{crop}: manifest short_track_min_len_effective="
                            f"{got} but SURFACES declares {declared}")
    return got


def surface_crops(surface: str, split: int) -> list[str]:
    d = surface_dir(surface) / str(split)
    return sorted(p.stem for p in d.glob("*.parquet")) if d.is_dir() else []


def common_crops(surfaces: list[str], split: int) -> list[str]:
    """Intersection across surfaces (ENVIRONMENT_TRAPS: never compare unequal crop sets)."""
    sets = [set(surface_crops(s, split)) for s in surfaces]
    keep = set.intersection(*sets) if sets else set()
    return sorted(c for c in keep if (ROOT / "data" / "train" / f"{c}.geff").exists())


def load_surface_tables(surface: str, split: int, crop: str):
    """Same contract as phaseb_d0p_proposer.load_e0c_tables, for an arbitrary surface."""
    df = pl.read_parquet(surface_dir(surface) / str(split) / f"{crop}.parquet")
    nd = df.filter(pl.col("row_type") == "node").sort("node_id")
    sub = np.asarray(nd["node_id"].to_list(), dtype=np.int64)
    t = np.asarray(nd["t"].to_list(), dtype=np.int64)
    pos = np.stack([np.asarray(nd["z"].to_list(), float) * SCALE[0],
                    np.asarray(nd["y"].to_list(), float) * SCALE[1],
                    np.asarray(nd["x"].to_list(), float) * SCALE[2]], axis=1)
    edges = [(int(r["source_id"]), int(r["target_id"]))
             for r in df.filter(pl.col("row_type") == "edge").iter_rows(named=True)]
    return sub, t, pos, edges, nd


# ---------------------------------------------------------------- graph helpers
def components_and_outdeg(node_ids, edge_set):
    """Union-find components + out-degree, matching the wrapper's own bookkeeping."""
    parent = {int(n): int(n) for n in node_ids}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    outc: dict[int, int] = {}
    for a, b in edge_set:
        a, b = int(a), int(b)
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        outc[a] = outc.get(a, 0) + 1

    comps: dict[int, list[int]] = {}
    for n in node_ids:
        comps.setdefault(find(int(n)), []).append(int(n))
    return comps, outc


def division_exempt_nodes(node_ids, edge_set, min_len: int):
    """Nodes in components the wrapper keeps ONLY via `has_division` (len < min_len)."""
    comps, outc = components_and_outdeg(node_ids, edge_set)
    ex_nodes: set[int] = set()
    ex_comps = 0
    for members in comps.values():
        if len(members) >= min_len:
            continue
        if any(outc.get(n, 0) >= 2 for n in members):
            ex_comps += 1
            ex_nodes.update(members)
    return ex_nodes, ex_comps


def degree_signature(node_ids, edge_set):
    ind, outd = defaultdict(int), defaultdict(int)
    for a, b in edge_set:
        outd[int(a)] += 1
        ind[int(b)] += 1
    return {int(n): (ind[int(n)], outd[int(n)]) for n in node_ids}


def wrapper_component_filter(nodes_by_id: dict, edge_set):
    """The REAL live-path node filter: prune-isolated block + filter_short_track_components.

    Returns (kept_node_ids, kept_edge_set, stats).
    """
    import biotrack.wrapper as W

    stats: dict[str, int] = defaultdict(int)
    nodes = dict(nodes_by_id)
    edges = [{"source_id": int(a), "target_id": int(b)} for a, b in sorted(edge_set)]

    # --- verbatim slice of wrapper.filter_output_graph (OUTPUT_PRUNE_ISOLATED block) ---
    if W.OUTPUT_PRUNE_ISOLATED:
        incident = {int(e["source_id"]) for e in edges} | {int(e["target_id"]) for e in edges}
        if incident:
            kept_nodes = {nid: n for nid, n in nodes.items() if nid in incident}
            stats["pruned_isolated_nodes"] = len(nodes) - len(kept_nodes)
            nodes = kept_nodes
            edges = [e for e in edges
                     if int(e["source_id"]) in nodes and int(e["target_id"]) in nodes]
    # --- end verbatim slice -----------------------------------------------------------

    nodes, edges = W.filter_short_track_components(nodes, edges, stats, dataset=None)
    return (set(nodes),
            {(int(e["source_id"]), int(e["target_id"])) for e in edges},
            dict(stats))


# ---------------------------------------------------------------- GT plumbing
def build_gt_maps(nd, edges, gt, allow_collisions: bool):
    """Predicted<->GT id maps with the MOTHER-COLLISION ASSERTION.

    The D0P/H0c lineage writes `gt_to_sub[int(m)] = s` without checking whether `m` was
    already claimed. Under one-to-one bipartite matching that cannot happen, but nothing
    in the code enforces it, and the assumption is silently node-population dependent.
    """
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE

    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    internal = g.bulk_add_nodes([{"t": int(a), "z": float(z), "y": float(y), "x": float(x)}
                                 for a, z, y, x in zip(nd["t"], nd["z"], nd["y"], nd["x"])])
    s2i = {int(s): internal[i] for i, s in enumerate(nd["node_id"].to_list())}
    if edges:
        g.bulk_add_edges([{"source_id": s2i[a], "target_id": s2i[b]} for a, b in edges])
    g.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    na = g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID,
                                 td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    i2g = {int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
           for r in na.iter_rows(named=True)}

    gt_to_sub: dict[int, int] = {}
    sub_to_gt: dict[int, int] = {}
    collisions: list[tuple[int, int, int]] = []
    for s, iid in s2i.items():
        m = i2g.get(iid)
        if m in (None, -1):
            continue
        m = int(m)
        if m in gt_to_sub:                      # <-- the previously unguarded overwrite
            collisions.append((m, gt_to_sub[m], s))
            continue                            # first writer wins; never silently clobber
        gt_to_sub[m] = s
        sub_to_gt[s] = m
    if collisions and not allow_collisions:
        raise AssertionError(
            f"mother-collision: {len(collisions)} GT node(s) claimed by >1 predicted node; "
            f"first three {collisions[:3]}. The gt_to_sub map is not injective on this node "
            f"population -- re-run with --allow-gt-collisions to quantify instead of abort."
        )
    return gt_to_sub, sub_to_gt, len(collisions)


# ---------------------------------------------------------------- edit operators
def edit_edges(edges, retained, suppress: bool, reconstruct: bool):
    """Frozen suppress-all (lowest-id retention) then add-replace. Copied semantics from
    phaseb_h0c_replay.replay_one; iteration order over `retained` is preserved so the
    published H0c numbers reproduce exactly."""
    es = {(int(a), int(b)) for a, b in edges}
    par: dict[int, set[int]] = {}
    ch_: dict[int, set[int]] = {}
    for a, b in es:
        ch_.setdefault(a, set()).add(b)
        par.setdefault(b, set()).add(a)

    forks_pre = sum(1 for k in ch_.values() if len(k) >= 2)
    suppressed = 0
    if suppress:
        for m in [n for n, k in ch_.items() if len(k) >= 2]:
            keep = min(ch_[m])
            for k in list(ch_[m]):
                if k != keep:
                    es.discard((m, k)); ch_[m].discard(k); par.get(k, set()).discard(m)
                    suppressed += 1

    steals = 0
    used: set[int] = set()
    if reconstruct:
        for pM, (p1, p2) in retained.items():
            if p1 in used or p2 in used:
                continue
            for k in list(ch_.get(pM, set())):
                if k not in (p1, p2):
                    es.discard((pM, k)); ch_[pM].discard(k); par.get(k, set()).discard(pM)
            for pc in (p1, p2):
                for s in list(par.get(pc, set())):
                    if s != pM:
                        es.discard((s, pc)); par[pc].discard(s); ch_.get(s, set()).discard(pc)
                        steals += 1
                es.add((pM, pc)); ch_.setdefault(pM, set()).add(pc); par.setdefault(pc, set()).add(pM)
            used |= {p1, p2}
    return es, {"forks_pre": forks_pre, "suppressed_edges": suppressed, "steals": steals}


# ---------------------------------------------------------------- per-crop worker
def replay_one(args) -> dict:
    surface, split, crop, allow_coll = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import biotrack.wrapper as W
    from biotrack.metric import load_graph, score_pred_graph
    from biotrack.submission import submission_to_graphs

    # ENVIRONMENT_TRAPS: module-global config leaks. Pin the constants this replay depends
    # on, so a stray BIOHUB_* env var cannot silently change the filter, and set the
    # short-track threshold from the SURFACE'S OWN manifest (E0c=7, v122=6, ...).
    assert W.OUTPUT_FILTER_SHORT_TRACKS is True, "OUTPUT_FILTER_SHORT_TRACKS must be on"
    assert W.OUTPUT_KEEP_DIVISION_COMPONENTS is True, "OUTPUT_KEEP_DIVISION_COMPONENTS must be on"
    assert W.OUTPUT_PRUNE_ISOLATED is True, "OUTPUT_PRUNE_ISOLATED must be on"
    assert not W.SHORT_TRACK_MIN_LEN_BY_DATASET, "per-dataset override must be empty"
    min_len = surface_min_track_len(surface, split, crop)
    W.OUTPUT_MIN_TRACK_LEN = min_len
    assert int(W.short_track_min_len_for_dataset(None)) == min_len

    t0 = time.time()
    sub, t, pos, edges, nd = load_surface_tables(surface, split, crop)
    gt_geff = str(ROOT / "data" / "train" / f"{crop}.geff")

    node_rows = nd.select("row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")
    all_nodes = {int(n) for n in sub}
    nodes_by_id = {int(n): {"t": int(a), "z": float(z), "y": float(y), "x": float(x)}
                   for n, a, z, y, x in zip(nd["node_id"], nd["t"], nd["z"], nd["y"], nd["x"])}
    base_edge_set = {(int(a), int(b)) for a, b in edges}

    def score(keep_nodes: set[int], edge_set) -> dict:
        nr = (node_rows if len(keep_nodes) == len(all_nodes)
              else node_rows.filter(pl.col("node_id").is_in(sorted(keep_nodes))))
        er = pl.DataFrame([{"row_type": "edge", "node_id": -1, "t": -1, "z": -1.0, "y": -1.0,
                            "x": -1.0, "source_id": a, "target_id": b}
                           for a, b in sorted(edge_set)])
        o = pl.concat([nr, er]) if er.height else nr
        g = submission_to_graphs(
            o.with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))[crop]
        return score_pred_graph(g, gt_geff)

    # ---- GT-FREE proposer, verbatim frozen definition -----------------------
    prop, n_pre = shortlist(sub, t, pos, edges)

    # ---- idempotence of the live filter on the UNEDITED cached graph --------
    k0, e0, st0 = wrapper_component_filter(nodes_by_id, base_edge_set)
    refilter_identity = (k0 == all_nodes and e0 == base_edge_set)

    # ---- pre-edit division-exempt components (the guard whitelist) ----------
    exempt_nodes, exempt_comps = division_exempt_nodes(all_nodes, base_edge_set, min_len)

    # ---- GT only after generation -------------------------------------------
    gt = load_graph(gt_geff)
    gt_to_sub, _sub_to_gt, n_coll = build_gt_maps(nd, edges, gt, allow_coll)
    ids = [int(n) for n in gt.node_ids()]
    outdeg = dict(zip(ids, gt.out_degree(ids)))
    n_gt_div = reachable = 0
    retained: dict[int, tuple[int, int]] = {}
    for n in ids:
        if outdeg[n] < 2:
            continue
        ch = [int(c) for c in gt.successors(int(n))][:2]
        if len(ch) != 2:
            continue
        n_gt_div += 1
        pM, p1, p2 = gt_to_sub.get(n), gt_to_sub.get(ch[0]), gt_to_sub.get(ch[1])
        if pM is None or p1 is None or p2 is None or p1 == p2:
            continue
        reachable += 1
        if (min(p1, p2), max(p1, p2)) in {k for _, k in prop.get(pM, ())}:
            retained[pM] = (min(p1, p2), max(p1, p2))

    # ---- arms ---------------------------------------------------------------
    es_h0c, ed_h0c = edit_edges(edges, retained, suppress=True, reconstruct=True)
    es_sup, ed_sup = edit_edges(edges, {}, suppress=True, reconstruct=False)

    k_h0c, e_h0c, st_h0c = wrapper_component_filter(nodes_by_id, es_h0c)
    k_sup, e_sup, st_sup = wrapper_component_filter(nodes_by_id, es_sup)

    guard_keep = k_h0c | (exempt_nodes & all_nodes)
    guard_edges = {(a, b) for a, b in es_h0c if a in guard_keep and b in guard_keep}

    # degenerate "everything that survived the pre-edit filter" guard: must be all nodes
    guard_full_is_identity = (k0 == all_nodes)

    rows = {
        "base": score(all_nodes, base_edge_set),
        "h0c_post": score(all_nodes, es_h0c),
        "h0c_refilt": score(k_h0c, e_h0c),
        "h0c_refilt_guard": score(guard_keep, guard_edges),
        "supp_post": score(all_nodes, es_sup),
        "supp_refilt": score(k_sup, e_sup),
    }

    # ---- how many of the exempt components actually lost their exemption ----
    _, outc_after = components_and_outdeg(all_nodes, es_h0c)
    exempt_lost_nodes = len(exempt_nodes - k_h0c)
    comps_pre, outc_pre = components_and_outdeg(all_nodes, base_edge_set)
    exempt_comps_lost = 0
    for members in comps_pre.values():
        if len(members) >= min_len:
            continue
        if not any(outc_pre.get(n, 0) >= 2 for n in members):
            continue
        if not (set(members) & k_h0c):
            exempt_comps_lost += 1

    sig_b = degree_signature(all_nodes, base_edge_set)
    sig_t = degree_signature(all_nodes, es_h0c)
    linefit_exposed = sum(1 for n in k_h0c if sig_b[n] != sig_t[n])

    out = {"surface": surface, "split": split, "crop": crop, "min_track_len": min_len,
           "n_nodes_cached": len(all_nodes), "n_edges_cached": len(base_edge_set),
           "refilter_identity": bool(refilter_identity),
           "refilter_identity_nodes": int(len(all_nodes) - len(k0)),
           "refilter_identity_edges": int(len(base_edge_set) - len(e0)),
           "guard_full_is_identity": bool(guard_full_is_identity),
           "gt_collisions": int(n_coll),
           "n_gt_div": n_gt_div, "reachable": reachable, "retained": len(retained),
           "cand_pre_topk": n_pre, "cand_shortlist": sum(len(v) for v in prop.values()),
           "mothers": len(prop),
           "forks_pre": ed_h0c["forks_pre"], "suppressed_edges": ed_h0c["suppressed_edges"],
           "steals": ed_h0c["steals"],
           "exempt_comps": exempt_comps, "exempt_nodes": len(exempt_nodes),
           "exempt_comps_lost": exempt_comps_lost, "exempt_lost_nodes": exempt_lost_nodes,
           "del_nodes_h0c_refilt": len(all_nodes) - len(k_h0c),
           "del_nodes_supp_refilt": len(all_nodes) - len(k_sup),
           "del_nodes_guard": len(all_nodes) - len(guard_keep),
           "pruned_isolated_h0c": int(st_h0c.get("pruned_isolated_nodes", 0)),
           "shorttrack_nodes_h0c": int(st_h0c.get("short_track_nodes_removed", 0)),
           "shorttrack_comps_h0c": int(st_h0c.get("short_track_components_removed", 0)),
           "pruned_isolated_supp": int(st_sup.get("pruned_isolated_nodes", 0)),
           "shorttrack_nodes_supp": int(st_sup.get("short_track_nodes_removed", 0)),
           "linefit_exposed_nodes": linefit_exposed,
           "runtime_s": time.time() - t0}
    keys = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
            "node_recall", "num_pred_nodes")
    for arm, r in rows.items():
        out.update({f"{arm}__{k}": r[k] for k in keys})
    return out


# ---------------------------------------------------------------- aggregation
def aggregate(res: list[dict], surface: str, fold: int, fam: str) -> dict:
    from biotrack.metric import estimated_nodes, per_sample_metrics
    from tracking_cellmot.metrics import EvaluationResult, summarise

    keys = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
            "num_pred_nodes")
    per_arm_rows: dict[str, list[dict]] = {a: [] for a in ARMS}
    n_est_by_crop = {}
    for r in res:
        ne = estimated_nodes(str(ROOT / "data" / "train" / f"{r['crop']}.geff"))
        n_est_by_crop[r["crop"]] = ne
        for a in ARMS:
            per_arm_rows[a].append(per_sample_metrics(
                EvaluationResult(*[r[f"{a}__{k}"] for k in keys]), ne, r[f"{a}__node_recall"]))

    summ = {a: summarise(per_arm_rows[a]) for a in ARMS}

    def mult(row):  # count multiplier = 1 - 0.1 * (N_pred - N_est)/N_est
        return 1.0 - 0.1 * row["total_node_ratio"]

    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in per_arm_rows["base"]]
    tw = float(sum(w)) or 1.0

    def wmean(vals):
        return float(sum(wi * v for wi, v in zip(w, vals)) / tw)

    mult_stats = {}
    base_rows = per_arm_rows["base"]
    for a in ARMS:
        rows_a = per_arm_rows[a]
        m_b = [mult(r) for r in base_rows]
        m_a = [mult(r) for r in rows_a]
        j_b = [r["edge_jaccard"] for r in base_rows]
        j_a = [r["edge_jaccard"] for r in rows_a]
        mult_stats[a] = {
            "mult_w": wmean(m_a),
            "d_mult_w": wmean([x - y for x, y in zip(m_a, m_b)]),
            "n_pred_total": int(sum(r["num_pred_nodes"] for r in rows_a)),
            # first-order decomposition of d(adjJ) = d(J)*mult_b + J_b*d(mult) + cross
            "term_rawJ": wmean([(x - y) * mb for x, y, mb in zip(j_a, j_b, m_b)]),
            "term_mult": wmean([jb * (x - y) for jb, x, y in zip(j_b, m_a, m_b)]),
            "term_cross": wmean([(ja - jb) * (ma - mb)
                                 for ja, jb, ma, mb in zip(j_a, j_b, m_a, m_b)]),
        }

    agg = {k: int(sum(r[k] for r in res)) for k in (
        "n_nodes_cached", "n_edges_cached", "n_gt_div", "reachable", "retained",
        "cand_pre_topk", "cand_shortlist", "mothers", "forks_pre", "suppressed_edges",
        "steals", "exempt_comps", "exempt_nodes", "exempt_comps_lost", "exempt_lost_nodes",
        "del_nodes_h0c_refilt", "del_nodes_supp_refilt", "del_nodes_guard",
        "pruned_isolated_h0c", "shorttrack_nodes_h0c", "shorttrack_comps_h0c",
        "pruned_isolated_supp", "shorttrack_nodes_supp", "linefit_exposed_nodes",
        "gt_collisions")}
    agg["n_crops"] = len(res)
    agg["min_track_len"] = sorted({int(r["min_track_len"]) for r in res})
    agg["refilter_identity_all"] = all(r["refilter_identity"] for r in res)
    agg["guard_full_is_identity_all"] = all(r["guard_full_is_identity"] for r in res)
    agg["est_nodes_total"] = float(sum(n_est_by_crop.values()))

    return {"surface": surface, "fold": fold, "family": fam,
            "summary": {a: summ[a] for a in ARMS},
            "mult": mult_stats, "agg": agg,
            "n_est_total": agg["est_nodes_total"]}


def report(block: dict) -> None:
    fam, surface = block["family"], block["surface"]
    s, m, a = block["summary"], block["mult"], block["agg"]
    print(f"\n########## H0d  surface={surface}  {fam}  ({a['n_crops']} crops) ##########")
    print(f"  cached graph: nodes={a['n_nodes_cached']:,} edges={a['n_edges_cached']:,} "
          f"N_est={a['est_nodes_total']:,.0f} existing_forks={a['forks_pre']:,} "
          f"min_track_len={a['min_track_len']}")
    print(f"  live-filter idempotent on the unedited graph: {a['refilter_identity_all']}   "
          f"'keep pre-edit survivors' guard == identity: {a['guard_full_is_identity_all']}")
    print(f"  GT: divisions={a['n_gt_div']} reachable={a['reachable']} "
          f"retained_by_frozen_proposer={a['retained']} "
          f"({a['retained']/max(a['reachable'],1):.3f} of reachable)  "
          f"gt_collisions={a['gt_collisions']}")
    print(f"  shortlist: pre-topK={a['cand_pre_topk']:,} top3={a['cand_shortlist']:,} "
          f"mothers={a['mothers']:,} steals={a['steals']:,} "
          f"suppressed_edges={a['suppressed_edges']:,}")
    print(f"  division-exempt short components (kept ONLY by has_division): "
          f"{a['exempt_comps']:,} comps / {a['exempt_nodes']:,} nodes; "
          f"lost after suppression: {a['exempt_comps_lost']:,} comps / "
          f"{a['exempt_lost_nodes']:,} nodes")
    print(f"  node deletions under the real filter: h0c={a['del_nodes_h0c_refilt']:,} "
          f"(isolated {a['pruned_isolated_h0c']:,} + short-track {a['shorttrack_nodes_h0c']:,}), "
          f"suppress-only={a['del_nodes_supp_refilt']:,}, with guard={a['del_nodes_guard']:,}")
    print(f"  linefit-exposed nodes (topology changed, coordinates NOT re-smoothed): "
          f"{a['linefit_exposed_nodes']:,}")
    print(f"  {'arm':<18}{'composite':>10}{'delta':>9}{'adjEdgeJ':>10}{'rawJ':>8}"
          f"{'divJ':>8}{'mult':>8}{'dmult':>9}{'N_pred':>11}{'dNodes':>9}")
    b = s["base"]["score"]
    for arm in ARMS:
        r, mm = s[arm], m[arm]
        print(f"  {arm:<18}{r['score']:>10.4f}{r['score']-b:>+9.4f}"
              f"{r['adj_edge_jaccard']:>10.4f}{r['edge_jaccard']:>8.4f}"
              f"{r['division_jaccard']:>8.4f}{mm['mult_w']:>8.4f}{mm['d_mult_w']:>+9.5f}"
              f"{mm['n_pred_total']:>11,}"
              f"{mm['n_pred_total']-m['base']['n_pred_total']:>+9,}")
    for arm in ("h0c_post", "h0c_refilt", "h0c_refilt_guard"):
        mm = m[arm]
        print(f"    {arm}: d(adjJ) decomposition  rawJ={mm['term_rawJ']:+.5f} "
              f"mult={mm['term_mult']:+.5f} cross={mm['term_cross']:+.5f}")
    for arm in ARMS:
        r = s[arm]
        print(f"    {arm}: div TP{r['division_tp']}/FP{r['division_fp']}/FN{r['division_fn']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--surfaces", default="e0c")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--limit", type=int, default=0, help="first N crops per fold (smoke)")
    ap.add_argument("--allow-gt-collisions", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--per-crop", default="", help="optional parquet path for per-crop rows")
    args = ap.parse_args()

    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    folds = [int(f) for f in args.folds.split(",") if f.strip()]
    for s in surfaces:
        if s not in SURFACES:
            raise SystemExit(f"unknown surface {s}; known: {sorted(SURFACES)}")

    print(f"H0d live-filter replay | frozen proposer cfg {CFG_HASH}: {CFG}")
    print(f"surfaces={surfaces} folds={folds}")

    blocks, all_rows = [], []
    for surface in surfaces:
        for fold in folds:
            fam = "44b6" if fold == 0 else "6bba"
            crops = common_crops(surfaces, fold)
            if args.limit:
                crops = crops[:args.limit]
            if not crops:
                print(f"  [skip] {surface} fold {fold}: no crops")
                continue
            tasks = [(surface, fold, c, args.allow_gt_collisions) for c in crops]
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                res = list(ex.map(replay_one, tasks))
            blk = aggregate(res, surface, fold, fam)
            blk["wall_s"] = time.time() - t0
            report(blk)
            print(f"  wall={blk['wall_s']:.0f}s")
            blocks.append(blk)
            all_rows.extend(res)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": CFG, "config_hash": CFG_HASH,
                               "surfaces": {s: SURFACES[s] for s in surfaces},
                               "e0c_anchor": E0C_ANCHOR, "blocks": blocks},
                              indent=2, default=float))
    print(f"\nwrote {out}")
    if args.per_crop:
        p = Path(args.per_crop)
        p.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(all_rows).write_parquet(p)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
