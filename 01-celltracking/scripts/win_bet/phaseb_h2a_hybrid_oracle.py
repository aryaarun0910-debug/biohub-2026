"""H2a — HYBRID-SUBSTRATE division oracle: v122 ordinary tracking + clean903 division surface.

WHY THIS EXISTS
---------------
`phaseb_h0d_livefilter.py` measured that the division ceiling is SUBSTRATE-DEPENDENT:

    surface    reach 44b6   reach 6bba
    e0c        20/26        93/125
    v122       15/26        68/125     <- best ordinary tracking (pooled OOF 0.69909)
    clean903   20/26        96/125     <- best division reachability
    ilp_c      15/26        67/125

`reachable` there means "the GT mother AND both GT daughters each have a predicted node
matched to them", i.e. the division is representable at all on that node set. v122 loses
25-27% of reachable divisions purely because its DETECTOR/ILP stage dropped a mother or a
daughter -- the wrapper and the proposer never get a chance.

H2a refuses the choice between the two. It keeps the v122 graph as the ordinary-tracking
substrate everywhere and introduces AUXILIARY NODES from the clean903 detection surface ONLY
at GT-reachable divisions v122 cannot represent, then re-optimises only the local affected
component and re-runs the REAL wrapper component filter.

This file delivers the GT-ASSISTED ORACLE CEILING. Every triple admitted here is a true GT
division, so nothing in it is submittable. It exists to answer one question before any
deployable proposer work is funded:

    do the recovered divisions outrun the exact count-multiplier charge of the added nodes?

WHAT IS AND IS NOT CHARGED
--------------------------
Added nodes are NOT free. The scorer's per-sample adjusted edge Jaccard is

    J_adj = max(0, J * (1 - 0.1 * (N_pred - N_est) / N_est))

and N_est is NOT available at test time, so an added node is a permanent, unhedgeable cost.
Every arm here is scored with the exact patched scorer on its OWN node set, and the count
multiplier is additionally reported analytically (it is a pure function of N_pred and N_est)
and decomposed first-order into rawJ / mult / cross contributions.

Local reassignment damage (parent steals) and the wrapper's own deletions are likewise
measured, never assumed: the two node populations are different, so nothing about node
invariance carries over from the H0c/H0d lineage.

DEDUPLICATION AND MATCHING SOUNDNESS
------------------------------------
v122 node ids and clean903 node ids are BOTH 1..N and mean nothing to each other. The two
surfaces are reconciled through the only common anchor that exists: each is matched to the
SAME GT graph with the SAME `DistanceMatching(max_distance=7.0, scale=DEFAULT_SCALE)` the
official scorer uses. Two predicted nodes matched to the same GT node are the same biological
object. So:

  * a GT role (mother / daughter) already carried by a v122 node NEVER gets an auxiliary node
    -- that is the definition of "absent from v122" used to select the recovery set;
  * at most one auxiliary node is ever created per GT node id (`aux_by_gt` registry), so two
    recovery triples sharing a cell share the node;
  * auxiliary ids are offset into a disjoint range and the disjointness is asserted.

That is sound for GT-anchored identity but says nothing about GEOMETRIC near-duplicates, so a
second, independent audit is run: for every auxiliary node, the distance in microns to the
nearest v122 node in the SAME FRAME is measured, together with whether that neighbour is
itself GT-matched. `dup_collisions` counts auxiliary nodes landing inside the 7 um matching
radius of an existing v122 node -- those are the ones that can steal a match. The final
arbiter is the scorer itself, which re-runs matching on the hybrid node set; any stolen match
shows up directly in `node_recall`, `edge_tp` and `division_tp`.

`build_gt_maps` (imported from H0d) raises on a mother-collision unless
`--allow-gt-collisions`; clean903 is a high-recall greedy surface, so collisions are EXPECTED
there and are counted per surface rather than assumed away.

ARMS (all scored with the exact patched scorer, all re-filtered by the REAL wrapper)
-----------------------------------------------------------------------------------
  base              v122 cached graph, unedited
  main_oracle       v122 + suppress-all + add-replace over v122-REACHABLE GT divisions
                    (the substrate-limited ceiling: this is what v122 can ever reach)
  hybrid_div_only   v122 UNSUPPRESSED + auxiliary nodes and add-replace for the RECOVERY set
                    only (isolates the marginal value and marginal cost of the aux surface)
  hybrid_full       v122 + suppress-all + add-replace over (reachable UNION recovery), with
                    auxiliary nodes (the full hybrid ceiling)

`main_oracle` uses ALL reachable triples, not the frozen H0c top-3 shortlist, because this is
a reachability ceiling. The frozen proposer is still imported VERBATIM (cfg hash 04eeac97500d)
and run on the HYBRID node set as a deployability diagnostic: `recovery_in_shortlist` reports
how many recovered triples the frozen GT-free generator would even propose once the auxiliary
nodes exist. Nothing is retuned.

PRIMARY METRIC
--------------
EXACT POOLED OOF: every crop from both folds goes into ONE combined
`tracking_cellmot.metrics.summarise()` call. Per-family composites are DIAGNOSTICS ONLY --
44b6 carries ~15% of edge mass, so averaging or min-ing the families optimises a different
objective (see scripts/metric/verify_pooled_objective.py).

Usage:
  smoke (<=3 crops, no full-corpus run):
    .venv\\Scripts\\python.exe scripts\\win_bet\\phaseb_h2a_hybrid_oracle.py \
        --crops 6bba_48816121,6bba_09961292,6bba_afb141ff --workers 3 --allow-gt-collisions
  full corpus (199 crops) -- REQUIRES HUMAN GO-AHEAD:
    .venv\\Scripts\\python.exe scripts\\win_bet\\phaseb_h2a_hybrid_oracle.py \
        --folds 0,1 --workers 6 --allow-gt-collisions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
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
from phaseb_h0d_livefilter import (  # noqa: E402  (surface handling + collision guard)
    SURFACES, build_gt_maps, components_and_outdeg, division_exempt_nodes, edit_edges,
    load_surface_tables, surface_crops, surface_min_track_len, wrapper_component_filter,
)

DEFAULT_OUT = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026\_evidence"
                   r"\agent_runs\hybrid\phaseb_h2a_hybrid_oracle.json")

# Hybrid configuration. FROZEN BEFORE EXECUTION; hashed into the output so a retune is visible.
HCFG = {
    "main_surface": "v122",          # ordinary tracking substrate (best pooled OOF arm)
    "aux_surface": "clean903",       # auxiliary DIVISION surface (best measured reachability)
    "anchor": "gt_distance_matching",  # the only common id space between the two populations
    "aux_id_offset": 10_000_000,     # auxiliary node ids live here; disjointness is asserted
    "dedup": "one_aux_node_per_gt_node",
    "inject_when": "gt_triple_reachable_on_aux_and_not_on_main",
    "local_reopt": "add_replace",    # remove other mother out-edges, steal daughters' parents
    "refilter": "real_wrapper_component_filter",
    "min_len_source": "surface_status_manifest",
}
HCFG_HASH = hashlib.sha256(json.dumps(HCFG, sort_keys=True).encode()).hexdigest()[:12]

ARMS = ("base", "main_oracle", "hybrid_div_only", "hybrid_full")
SCORE_KEYS = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
              "node_recall", "num_pred_nodes")
COUNT_KEYS = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
              "num_pred_nodes")
# v122 pooled OOF anchor and per-family diagnostics (journal 2026-07-19 / coupled decomposition)
V122_ANCHOR = {"pooled": 0.69909, "44b6": 0.6962, "6bba": 0.6997}


# ---------------------------------------------------------------- census helpers
def gt_division_triples(gt) -> list[tuple[int, int, int]]:
    """Every GT node with out-degree >= 2, as (mother, child0, child1) GT ids."""
    ids = [int(n) for n in gt.node_ids()]
    outdeg = dict(zip(ids, gt.out_degree(ids)))
    out = []
    for n in ids:
        if outdeg.get(n, 0) < 2:
            continue
        ch = [int(c) for c in gt.successors(int(n))][:2]
        if len(ch) != 2:
            continue
        out.append((n, ch[0], ch[1]))
    return out


def map_triple(triple, gt_to_sub):
    """(mother, d1, d2) in GT ids -> predicted ids, or None if any role is unrepresented."""
    m, c1, c2 = triple
    pM, p1, p2 = gt_to_sub.get(m), gt_to_sub.get(c1), gt_to_sub.get(c2)
    if pM is None or p1 is None or p2 is None or p1 == p2:
        return None
    return pM, min(p1, p2), max(p1, p2)


def nearest_same_frame(pos_um: np.ndarray, t_arr: np.ndarray, q_pos: np.ndarray, q_t: int):
    """(distance_um, index) to the nearest node in frame ``q_t``; (inf, -1) if the frame
    is empty. Brute force per query: the query count is O(#aux nodes) = tiny."""
    idx = np.flatnonzero(t_arr == q_t)
    if idx.size == 0:
        return float("inf"), -1
    d = np.linalg.norm(pos_um[idx] - q_pos[None, :], axis=1)
    j = int(np.argmin(d))
    return float(d[j]), int(idx[j])


# ---------------------------------------------------------------- per-crop worker
def hybrid_one(args) -> dict:
    split, crop, allow_coll, arms = args
    arms = tuple(arms)
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import biotrack.wrapper as W
    from biotrack.metric import MAX_DISTANCE, load_graph, score_pred_graph
    from biotrack.submission import submission_to_graphs
    from tracksdata.options import set_options

    set_options(show_progress=False)   # cosmetic only: the GT-map matches print a tqdm bar

    main_s, aux_s = HCFG["main_surface"], HCFG["aux_surface"]
    off = int(HCFG["aux_id_offset"])

    # ENVIRONMENT_TRAPS: pin every wrapper constant the live filter depends on, and take the
    # short-track threshold from the MAIN surface's own manifest (v122 = 6, NOT the E0c 7).
    assert W.OUTPUT_FILTER_SHORT_TRACKS is True, "OUTPUT_FILTER_SHORT_TRACKS must be on"
    assert W.OUTPUT_KEEP_DIVISION_COMPONENTS is True, "OUTPUT_KEEP_DIVISION_COMPONENTS must be on"
    assert W.OUTPUT_PRUNE_ISOLATED is True, "OUTPUT_PRUNE_ISOLATED must be on"
    assert not W.SHORT_TRACK_MIN_LEN_BY_DATASET, "per-dataset override must be empty"
    min_len = surface_min_track_len(main_s, split, crop)
    W.OUTPUT_MIN_TRACK_LEN = min_len
    assert int(W.short_track_min_len_for_dataset(None)) == min_len

    t0 = time.time()
    sub_m, t_m, pos_m, edges_m, nd_m = load_surface_tables(main_s, split, crop)
    sub_a, t_a, pos_a, edges_a, nd_a = load_surface_tables(aux_s, split, crop)
    gt_geff = str(ROOT / "data" / "train" / f"{crop}.geff")

    main_nodes = {int(n) for n in sub_m}
    assert max(main_nodes) < off, (f"{crop}: main node id {max(main_nodes)} >= aux offset "
                                  f"{off}; the auxiliary id range is not disjoint")
    base_edges = {(int(a), int(b)) for a, b in edges_m}
    nodes_by_id_main = {int(n): {"t": int(a), "z": float(z), "y": float(y), "x": float(x)}
                        for n, a, z, y, x in zip(nd_m["node_id"], nd_m["t"], nd_m["z"],
                                                 nd_m["y"], nd_m["x"])}
    node_rows_main = nd_m.select("row_type", "node_id", "t", "z", "y", "x",
                                 "source_id", "target_id")
    aux_raw = {int(n): (int(a), float(z), float(y), float(x))
               for n, a, z, y, x in zip(nd_a["node_id"], nd_a["t"], nd_a["z"],
                                        nd_a["y"], nd_a["x"])}

    # ---- idempotence of the live filter on the UNEDITED main graph ----------
    k0, e0, _ = wrapper_component_filter(nodes_by_id_main, base_edges)
    refilter_identity = (k0 == main_nodes and e0 == base_edges)
    exempt_nodes, exempt_comps = division_exempt_nodes(main_nodes, base_edges, min_len)

    # ---- GT anchor: match BOTH surfaces to the SAME GT graph -----------------
    gt = load_graph(gt_geff)
    gt2m, m2gt, coll_m = build_gt_maps(nd_m, edges_m, gt, allow_coll)
    gt2a, _a2gt, coll_a = build_gt_maps(nd_a, edges_a, gt, allow_coll)

    triples = gt_division_triples(gt)
    main_tri: dict[int, tuple[int, int]] = {}
    reach_main = reach_aux = 0
    recovery: list[tuple[int, int, int]] = []      # GT triples reachable on aux, not on main
    for tri in triples:
        pm = map_triple(tri, gt2m)
        pa = map_triple(tri, gt2a)
        if pm is not None:
            reach_main += 1
            main_tri[pm[0]] = (pm[1], pm[2])
        if pa is not None:
            reach_aux += 1
        if pa is not None and pm is None:
            recovery.append(tri)

    # ---- build the auxiliary node population --------------------------------
    aux_by_gt: dict[int, int] = {}       # GT node id -> new hybrid node id (dedup registry)
    aux_rows: list[dict] = []
    aux_pos_um: list[np.ndarray] = []
    aux_t: list[int] = []
    aux_role = {"mother": 0, "daughter": 0}
    rec_tri: dict[int, tuple[int, int]] = {}
    rec_needs_aux = {1: 0, 2: 0, 3: 0}
    n_next = off + 1

    def hybrid_id(gt_node: int, role: str) -> int:
        nonlocal n_next
        if gt_node in gt2m:
            return int(gt2m[gt_node])
        if gt_node in aux_by_gt:
            return aux_by_gt[gt_node]
        a_sub = int(gt2a[gt_node])
        tt, z, y, x = aux_raw[a_sub]
        nid = n_next
        n_next += 1
        aux_by_gt[gt_node] = nid
        aux_rows.append({"row_type": "node", "node_id": nid, "t": tt, "z": z, "y": y, "x": x,
                         "source_id": -1, "target_id": -1})
        aux_pos_um.append(np.array([z * SCALE[0], y * SCALE[1], x * SCALE[2]], float))
        aux_t.append(tt)
        aux_role[role] += 1
        return nid

    for tri in recovery:
        m, c1, c2 = tri
        rec_needs_aux[sum(1 for g in tri if g not in gt2m)] += 1
        hM = hybrid_id(m, "mother")
        h1 = hybrid_id(c1, "daughter")
        h2 = hybrid_id(c2, "daughter")
        if h1 == h2:                      # cannot happen: distinct GT ids, injective registry
            continue
        rec_tri[hM] = (min(h1, h2), max(h1, h2))

    aux_ids = sorted(aux_by_gt.values())
    assert not (set(aux_ids) & main_nodes), f"{crop}: auxiliary/main node id collision"

    # ---- duplicate-collision audit (geometric, independent of the GT anchor) --
    dup_lt_max = dup_lt_half = 0
    dup_nn_matched = 0
    nn_d = []
    for p, tt in zip(aux_pos_um, aux_t):
        d, j = nearest_same_frame(pos_m, t_m, p, tt)
        nn_d.append(d)
        if d < MAX_DISTANCE:
            dup_lt_max += 1
            if int(sub_m[j]) in m2gt:
                dup_nn_matched += 1
        if d < 0.5 * MAX_DISTANCE:
            dup_lt_half += 1

    # ---- hybrid node population ---------------------------------------------
    hybrid_nodes = main_nodes | set(aux_ids)
    nodes_by_id_hyb = dict(nodes_by_id_main)
    for r in aux_rows:
        nodes_by_id_hyb[int(r["node_id"])] = {"t": int(r["t"]), "z": float(r["z"]),
                                              "y": float(r["y"]), "x": float(r["x"])}
    node_rows_all = (pl.concat([node_rows_main, pl.DataFrame(aux_rows).select(
        node_rows_main.columns).cast(node_rows_main.schema)])
        if aux_rows else node_rows_main)

    # ---- edge edits ----------------------------------------------------------
    es_main, ed_main = edit_edges(edges_m, main_tri, suppress=True, reconstruct=True)
    es_hdiv, ed_hdiv = edit_edges(edges_m, rec_tri, suppress=False, reconstruct=True)
    union_tri = {**main_tri, **rec_tri}
    es_hfull, ed_hfull = edit_edges(edges_m, union_tri, suppress=True, reconstruct=True)

    # ---- REAL wrapper filter, per arm on its OWN node population -------------
    k_main, e_main, st_main = wrapper_component_filter(nodes_by_id_main, es_main)
    k_hdiv, e_hdiv, st_hdiv = wrapper_component_filter(nodes_by_id_hyb, es_hdiv)
    k_hfull, e_hfull, st_hfull = wrapper_component_filter(nodes_by_id_hyb, es_hfull)

    aux_survived_hfull = len(set(aux_ids) & k_hfull)
    aux_survived_hdiv = len(set(aux_ids) & k_hdiv)

    # ---- artificial-hub audit: components made ONLY of auxiliary nodes -------
    comps_h, outc_h = components_and_outdeg(k_hfull, e_hfull)
    aux_only_comps = aux_only_short_div = 0
    for members in comps_h.values():
        if not members or not set(members) <= set(aux_ids):
            continue
        aux_only_comps += 1
        if len(members) < min_len and any(outc_h.get(n, 0) >= 2 for n in members):
            aux_only_short_div += 1

    arm_graphs = {
        "base": (main_nodes, base_edges, node_rows_main),
        "main_oracle": (k_main, e_main, node_rows_main),
        "hybrid_div_only": (k_hdiv, e_hdiv, node_rows_all),
        "hybrid_full": (k_hfull, e_hfull, node_rows_all),
    }

    def score(keep_nodes: set[int], edge_set, nrows) -> dict:
        nr = (nrows if len(keep_nodes) == nrows.height
              else nrows.filter(pl.col("node_id").is_in(sorted(keep_nodes))))
        er = pl.DataFrame([{"row_type": "edge", "node_id": -1, "t": -1, "z": -1.0, "y": -1.0,
                            "x": -1.0, "source_id": a, "target_id": b}
                           for a, b in sorted(edge_set)])
        o = pl.concat([nr, er.cast(nr.schema)]) if er.height else nr
        g = submission_to_graphs(
            o.with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))[crop]
        return score_pred_graph(g, gt_geff)

    # EXACT-IDENTITY SHORTCUT. On a crop with no recovery triple no auxiliary node exists, so
    # `hybrid_div_only` and `hybrid_full` are the SAME GRAPH as `base` and `main_oracle`
    # respectively -- not approximately, identically. Most crops are in that state (151 GT
    # divisions over 199 crops), so the scorer is skipped only after the node set AND the edge
    # set are compared element-wise. `node_rows_all` filtered to a keep set contained in the
    # main population is row-identical to `node_rows_main` filtered to it, so the row frame
    # does not need to enter the comparison. Every reuse is recorded in `arm_reused`.
    rows: dict[str, dict] = {}
    reused: dict[str, str] = {}
    for a in arms:
        keep_a, edges_a_, _ = arm_graphs[a]
        src = None
        for b in arms:
            if b == a or b not in rows:
                continue
            keep_b, edges_b_, _ = arm_graphs[b]
            if keep_a == keep_b and edges_a_ == edges_b_:
                src = b
                break
        if src is not None:
            rows[a] = dict(rows[src])
            reused[a] = src
        else:
            rows[a] = score(*arm_graphs[a])
    n_pred_by_arm = {a: len(v[0]) for a, v in arm_graphs.items()}

    # ---- deployability diagnostic: does the FROZEN GT-free proposer see them? -
    # Run the verbatim H0c shortlist on the HYBRID node set (base edges, so the flow estimate
    # is the deployment-time one) and ask whether each recovered pair is in its mother's top-3.
    rec_in_shortlist = 0
    if rec_tri:
        h_sub = np.concatenate([sub_m.astype(np.int64),
                                np.asarray(aux_ids, dtype=np.int64)]) if aux_ids else sub_m
        h_t = np.concatenate([t_m, np.asarray(aux_t, dtype=np.int64)]) if aux_ids else t_m
        h_pos = (np.concatenate([pos_m, np.stack(aux_pos_um)], axis=0) if aux_ids else pos_m)
        prop_h, _ = shortlist(h_sub, h_t, h_pos, edges_m)
        for pM, pair in rec_tri.items():
            if pair in {k for _, k in prop_h.get(pM, ())}:
                rec_in_shortlist += 1

    out = {
        "split": split, "crop": crop, "min_track_len": min_len,
        "n_nodes_main": len(main_nodes), "n_edges_main": len(base_edges),
        "n_nodes_aux_surface": int(len(sub_a)),
        "refilter_identity": bool(refilter_identity),
        "gt_collisions_main": int(coll_m), "gt_collisions_aux": int(coll_a),
        "n_gt_div": len(triples), "reach_main": reach_main, "reach_aux": reach_aux,
        "recovery": len(recovery), "recovery_triples_built": len(rec_tri),
        "recovery_in_shortlist": rec_in_shortlist,
        "aux_nodes_added": len(aux_ids),
        "aux_role_mother": aux_role["mother"], "aux_role_daughter": aux_role["daughter"],
        "rec_needs_1": rec_needs_aux[1], "rec_needs_2": rec_needs_aux[2],
        "rec_needs_3": rec_needs_aux[3],
        "dup_collisions_lt_maxdist": dup_lt_max, "dup_collisions_lt_half": dup_lt_half,
        "dup_nn_gt_matched": dup_nn_matched,
        "aux_nn_dist_min": float(min(nn_d)) if nn_d else float("nan"),
        "aux_nn_dist_med": float(np.median(nn_d)) if nn_d else float("nan"),
        "aux_survived_hybrid_full": aux_survived_hfull,
        "aux_survived_hybrid_div": aux_survived_hdiv,
        "aux_only_comps": aux_only_comps, "aux_only_short_div_comps": aux_only_short_div,
        "exempt_comps": exempt_comps, "exempt_nodes": len(exempt_nodes),
        "steals_main": ed_main["steals"], "steals_hdiv": ed_hdiv["steals"],
        "steals_hfull": ed_hfull["steals"],
        "suppressed_main": ed_main["suppressed_edges"],
        "suppressed_hfull": ed_hfull["suppressed_edges"],
        "forks_pre": ed_main["forks_pre"],
        "del_nodes_main_oracle": len(main_nodes) - len(k_main),
        "del_nodes_hybrid_div": len(hybrid_nodes) - len(k_hdiv),
        "del_nodes_hybrid_full": len(hybrid_nodes) - len(k_hfull),
        "shorttrack_nodes_main": int(st_main.get("short_track_nodes_removed", 0)),
        "shorttrack_nodes_hfull": int(st_hfull.get("short_track_nodes_removed", 0)),
        "pruned_isolated_hfull": int(st_hfull.get("pruned_isolated_nodes", 0)),
        "pruned_isolated_hdiv": int(st_hdiv.get("pruned_isolated_nodes", 0)),
        "scored_arms": list(arms), "runtime_s": time.time() - t0,
        "arm_reused": json.dumps(reused), "n_scorer_calls": len(arms) - len(reused),
    }
    out.update({f"npred__{a}": v for a, v in n_pred_by_arm.items()})
    for arm, r in rows.items():
        out.update({f"{arm}__{k}": r[k] for k in SCORE_KEYS})
    return out


# ---------------------------------------------------------------- aggregation
def build_rows(res: list[dict], arms: tuple[str, ...]):
    """Per-sample metric rows per arm + N_est per crop. No pooling yet."""
    from biotrack.metric import estimated_nodes, per_sample_metrics
    from tracking_cellmot.metrics import EvaluationResult

    per_arm: dict[str, list[dict]] = {a: [] for a in arms}
    n_est = []
    for r in res:
        ne = estimated_nodes(str(ROOT / "data" / "train" / f"{r['crop']}.geff"))
        n_est.append(ne)
        for a in arms:
            per_arm[a].append(per_sample_metrics(
                EvaluationResult(*[r[f"{a}__{k}"] for k in COUNT_KEYS]),
                ne, r[f"{a}__node_recall"]))
    return per_arm, n_est


def pooled_block(res: list[dict], per_arm: dict[str, list[dict]], n_est: list[float],
                 arms: tuple[str, ...], label: str) -> dict:
    """ONE combined summarise() over every crop in `res`. This is the primary objective."""
    from tracking_cellmot.metrics import summarise

    summ = {a: summarise(per_arm[a]) for a in arms}
    # With no scored arm (census-only) there are no measured edge volumes; fall back to the
    # cached edge count, which is the same quantity the scorer would weight by up to TP/FP/FN
    # bookkeeping. It is only ever used to weight the ANALYTIC count multiplier.
    w = ([r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in per_arm[arms[0]]] if arms
         else [r["n_edges_main"] for r in res])
    tw = float(sum(w)) or 1.0

    def wmean(v):
        return float(sum(wi * x for wi, x in zip(w, v)) / tw)

    mult = {a: [1.0 - 0.1 * (r[f"npred__{a}"] - ne) / ne for r, ne in zip(res, n_est)]
            for a in ARMS}
    stats = {}
    for a in ARMS:
        stats[a] = {"mult_w": wmean(mult[a]),
                    "d_mult_w": wmean([x - y for x, y in zip(mult[a], mult["base"])]),
                    "n_pred_total": int(sum(r[f"npred__{a}"] for r in res)),
                    "scored": a in arms}
        if a in arms and "base" in arms:
            mb = [1.0 - 0.1 * r["total_node_ratio"] for r in per_arm["base"]]
            ma = [1.0 - 0.1 * r["total_node_ratio"] for r in per_arm[a]]
            jb = [r["edge_jaccard"] for r in per_arm["base"]]
            ja = [r["edge_jaccard"] for r in per_arm[a]]
            stats[a].update({
                "term_rawJ": wmean([(x - y) * m for x, y, m in zip(ja, jb, mb)]),
                "term_mult": wmean([b * (x - y) for b, x, y in zip(jb, ma, mb)]),
                "term_cross": wmean([(p - q) * (x - y)
                                     for p, q, x, y in zip(ja, jb, ma, mb)]),
            })

    agg = {k: int(sum(r[k] for r in res)) for k in (
        "n_nodes_main", "n_edges_main", "n_nodes_aux_surface", "n_gt_div", "reach_main",
        "reach_aux", "recovery", "recovery_triples_built", "recovery_in_shortlist",
        "aux_nodes_added", "aux_role_mother", "aux_role_daughter", "rec_needs_1",
        "rec_needs_2", "rec_needs_3", "dup_collisions_lt_maxdist", "dup_collisions_lt_half",
        "dup_nn_gt_matched", "aux_survived_hybrid_full", "aux_survived_hybrid_div",
        "aux_only_comps", "aux_only_short_div_comps", "exempt_comps", "exempt_nodes",
        "steals_main", "steals_hdiv", "steals_hfull", "suppressed_main", "suppressed_hfull",
        "forks_pre", "del_nodes_main_oracle", "del_nodes_hybrid_div", "del_nodes_hybrid_full",
        "shorttrack_nodes_main", "shorttrack_nodes_hfull", "pruned_isolated_hfull",
        "pruned_isolated_hdiv", "gt_collisions_main", "gt_collisions_aux", "n_scorer_calls")}
    agg["n_crops"] = len(res)
    agg["crops_with_recovery"] = int(sum(1 for r in res if r["recovery_triples_built"]))
    agg["refilter_identity_all"] = all(r["refilter_identity"] for r in res)
    agg["min_track_len"] = sorted({int(r["min_track_len"]) for r in res})
    agg["n_est_total"] = float(sum(n_est))
    agg["edge_mass"] = tw
    dists = [r["aux_nn_dist_min"] for r in res if r["aux_nodes_added"]]
    agg["aux_nn_dist_min_overall"] = float(min(dists)) if dists else float("nan")

    # MATCH-STEALING DETECTOR. Injecting a node re-runs the scorer's own bipartite matching
    # on a changed node population, so an auxiliary node CAN take a GT match away from an
    # incumbent v122 node. If that happened anywhere, node_recall falls on that crop. The
    # worst per-crop delta vs `base` is reported, never averaged away.
    for a in arms:
        if "base" not in arms:
            continue
        d = [r[f"{a}__node_recall"] - r["base__node_recall"] for r in res]
        stats[a]["node_recall_delta_min"] = float(min(d))
        stats[a]["node_recall_delta_w"] = wmean(d)
        stats[a]["crops_recall_lost"] = int(sum(1 for x in d if x < -1e-12))
    return {"label": label, "summary": summ, "mult": stats, "agg": agg}


def report(block: dict, arms: tuple[str, ...]) -> None:
    s, m, a = block["summary"], block["mult"], block["agg"]
    print(f"\n########## H2a hybrid  [{block['label']}]  ({a['n_crops']} crops) ##########")
    print(f"  main={HCFG['main_surface']} aux={HCFG['aux_surface']} "
          f"min_track_len={a['min_track_len']} refilter_idempotent={a['refilter_identity_all']}")
    print(f"  substrate: main nodes={a['n_nodes_main']:,} edges={a['n_edges_main']:,} "
          f"| aux surface nodes={a['n_nodes_aux_surface']:,} | N_est={a['n_est_total']:,.0f} "
          f"| edge mass={a['edge_mass']:,.0f}")
    print(f"  GT divisions={a['n_gt_div']}  reach_main={a['reach_main']}  "
          f"reach_aux={a['reach_aux']}  RECOVERY(aux-only)={a['recovery']} "
          f"-> triples built={a['recovery_triples_built']}  "
          f"union={a['reach_main']+a['recovery']}")
    print(f"  frozen-proposer diagnostic: {a['recovery_in_shortlist']}/"
          f"{a['recovery_triples_built']} recovered pairs are in the mother's top-3 "
          f"(cfg {CFG_HASH})")
    print(f"  auxiliary nodes ADDED={a['aux_nodes_added']} "
          f"(mother {a['aux_role_mother']} / daughter {a['aux_role_daughter']}); "
          f"triples needing 1/2/3 aux = {a['rec_needs_1']}/{a['rec_needs_2']}/{a['rec_needs_3']}")
    print(f"  duplicate collisions: {a['dup_collisions_lt_maxdist']} aux nodes within 7.0 um "
          f"of a main node in the same frame ({a['dup_nn_gt_matched']} of those neighbours are "
          f"GT-matched); {a['dup_collisions_lt_half']} within 3.5 um; "
          f"min nn distance {a['aux_nn_dist_min_overall']:.3f} um")
    print(f"  aux survival after the REAL filter: hybrid_full {a['aux_survived_hybrid_full']}/"
          f"{a['aux_nodes_added']}, hybrid_div_only {a['aux_survived_hybrid_div']}/"
          f"{a['aux_nodes_added']}; aux-only components={a['aux_only_comps']} "
          f"(short+division-exempt {a['aux_only_short_div_comps']})")
    print(f"  local reassignment: steals main={a['steals_main']} hdiv={a['steals_hdiv']} "
          f"hfull={a['steals_hfull']}; suppressed edges main={a['suppressed_main']} "
          f"hfull={a['suppressed_hfull']} (pre-existing forks {a['forks_pre']})")
    print(f"  wrapper deletions: main_oracle={a['del_nodes_main_oracle']} "
          f"hybrid_div={a['del_nodes_hybrid_div']} hybrid_full={a['del_nodes_hybrid_full']}")
    print(f"  gt collisions: main={a['gt_collisions_main']} aux={a['gt_collisions_aux']}")
    print(f"  crops with >=1 recovery triple: {a['crops_with_recovery']}/{a['n_crops']}; "
          f"scorer calls {a['n_scorer_calls']} of {a['n_crops']*len(arms)} "
          f"(exact-identity reuse saved {a['n_crops']*len(arms)-a['n_scorer_calls']})")
    print(f"  {'arm':<18}{'POOLED':>10}{'delta':>9}{'adjEdgeJ':>10}{'rawJ':>8}{'divJ':>8}"
          f"{'mult':>9}{'dmult':>10}{'N_pred':>12}{'dNodes':>8}{'divTP/FP/FN':>16}")
    b = s["base"]["score"] if "base" in s else float("nan")
    for arm in ARMS:
        mm = m[arm]
        if arm in s:
            r = s[arm]
            print(f"  {arm:<18}{r['score']:>10.5f}{r['score']-b:>+9.5f}"
                  f"{r['adj_edge_jaccard']:>10.5f}{r['edge_jaccard']:>8.5f}"
                  f"{r['division_jaccard']:>8.4f}{mm['mult_w']:>9.5f}{mm['d_mult_w']:>+10.6f}"
                  f"{mm['n_pred_total']:>12,}"
                  f"{mm['n_pred_total']-m['base']['n_pred_total']:>+8,}"
                  f"{r['division_tp']:>7}/{r['division_fp']}/{r['division_fn']}")
        else:
            print(f"  {arm:<18}{'(not scored)':>10}{'':>9}{'':>10}{'':>8}{'':>8}"
                  f"{mm['mult_w']:>9.5f}{mm['d_mult_w']:>+10.6f}{mm['n_pred_total']:>12,}"
                  f"{mm['n_pred_total']-m['base']['n_pred_total']:>+8,}")
    for arm in arms:
        mm = m[arm]
        if "term_rawJ" in mm:
            print(f"    {arm}: d(adjEdgeJ) = rawJ {mm['term_rawJ']:+.6f} "
                  f"+ mult {mm['term_mult']:+.6f} + cross {mm['term_cross']:+.6f}")
    if "base" in arms:
        print("  match-stealing audit (node_recall vs base; negative == an auxiliary node "
              "took a GT match from a v122 node)")
        for arm in arms:
            mm = m[arm]
            print(f"    {arm:<18} d_node_recall worst-crop "
                  f"{mm['node_recall_delta_min']:+.6f} weighted "
                  f"{mm['node_recall_delta_w']:+.6f}  crops that LOST recall: "
                  f"{mm['crops_recall_lost']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--crops", default="", help="explicit comma-separated crop stems (smoke)")
    ap.add_argument("--limit", type=int, default=0, help="first N crops per fold (smoke)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--allow-gt-collisions", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--per-crop", default="")
    args = ap.parse_args()

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}; known: {ARMS}")
    folds = [int(f) for f in args.folds.split(",") if f.strip()]
    main_s, aux_s = HCFG["main_surface"], HCFG["aux_surface"]

    print(f"H2a hybrid oracle | hybrid cfg {HCFG_HASH}: {HCFG}", flush=True)
    print(f"frozen proposer cfg {CFG_HASH}: {CFG}", flush=True)
    print(f"surfaces: main={main_s} {SURFACES[main_s]['graphs']} | "
          f"aux={aux_s} {SURFACES[aux_s]['graphs']}", flush=True)

    want = {c.strip() for c in args.crops.split(",") if c.strip()}
    tasks = []
    for fold in folds:
        cm, ca = set(surface_crops(main_s, fold)), set(surface_crops(aux_s, fold))
        crops = sorted(c for c in (cm & ca)
                       if (ROOT / "data" / "train" / f"{c}.geff").exists())
        missing = sorted((cm | ca) - (cm & ca))
        if missing:
            print(f"  [warn] fold {fold}: {len(missing)} crops not on BOTH surfaces, dropped")
        if want:
            crops = [c for c in crops if c in want]
        elif args.limit:
            crops = crops[:args.limit]
        tasks += [(fold, c, args.allow_gt_collisions, arms) for c in crops]
    if not tasks:
        raise SystemExit("no crops selected")
    print(f"crops={len(tasks)} folds={folds} scored_arms={arms}", flush=True)

    t0 = time.time()
    if args.workers <= 1:
        res = [hybrid_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(hybrid_one, tasks))
    wall = time.time() - t0

    per_arm, n_est = build_rows(res, arms)
    blocks = []
    # PRIMARY: one combined summarise() over every crop, both folds pooled.
    blocks.append(pooled_block(res, per_arm, n_est, arms, "POOLED (primary objective)"))
    report(blocks[0], arms)
    # DIAGNOSTIC ONLY: per-family composites.
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        idx = [i for i, r in enumerate(res) if r["split"] == fold]
        if not idx:
            continue
        sub_res = [res[i] for i in idx]
        sub_arm = {a: [per_arm[a][i] for i in idx] for a in arms}
        blk = pooled_block(sub_res, sub_arm, [n_est[i] for i in idx], arms,
                           f"DIAGNOSTIC family {fam}")
        report(blk, arms)
        blocks.append(blk)

    print(f"\nwall={wall:.0f}s  per-crop mean={wall/len(tasks):.1f}s  "
          f"(sum of worker runtimes {sum(r['runtime_s'] for r in res):.0f}s)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"hybrid_config": HCFG, "hybrid_config_hash": HCFG_HASH,
                               "proposer_config": CFG, "proposer_config_hash": CFG_HASH,
                               "surfaces": {main_s: SURFACES[main_s], aux_s: SURFACES[aux_s]},
                               "v122_anchor": V122_ANCHOR, "n_crops": len(tasks),
                               "wall_s": wall, "blocks": blocks},
                              indent=2, default=float))
    print(f"wrote {out}")
    if args.per_crop:
        p = Path(args.per_crop)
        p.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(res).write_parquet(p)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
