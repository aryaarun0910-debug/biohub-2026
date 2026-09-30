r"""THE CPU REACH ORACLE - LEVER-0045 FALSIFIER 0, owned by PKT-0044.

THE QUESTION, AND THE ONE ANSWER THAT WOULD BE DISHONEST
--------------------------------------------------------
Our own ledger (FACT-0422) measured the entire parent-RESCORING prize at 240 fold-0 GT edges while
1,110 fold-0 and 14,007 fold-1 true edges are NEVER OFFERED to any scorer at all, and FACT-0369
shows the deployed rule admits at most one parent per target BY ARITHMETIC, so no rescoring can
reach them. This module asks whether a TEMPORAL proposal mechanism can.

IT DOES NOT ASK WHETHER A TRUE EDGE "EXISTS INSIDE AN EIGHT-FRAME WINDOW". Both endpoints of every
category-3 edge already exist in ADJACENT frames - that is what category 3 MEANS - so such a number
is near-vacuous and would pass regardless of merit. What is measured here is whether the FROZEN,
GT-FREE rule in ``reach_rule.py`` actually NOMINATES the true parent.

THE FREEZE IS DEMONSTRABLE, NOT ASSERTED
----------------------------------------
``reach_rule.py`` contains the rule and nothing else; it refuses to write its own freeze if its
body mentions the ground truth at all. Its pickle sha256 was recorded in PKT-0044 before this file
existed. This module imports it and re-checks the hash on every run - a run whose rule hash does
not match the packet's recorded value REFUSES.

THE CATEGORY DEFINITIONS ARE NOT FORKED
---------------------------------------
A second, incompatible definition of "category 3" would make this result incomparable with
FACT-0422, which is the whole reason the ledger exists. So this module imports
``assoc_lost_edge_ledger._match_per_frame`` (the official one-to-one 7 um matcher) and applies the
committed first-match-wins ordering verbatim, then ASSERTS its per-crop counts against the
committed payloads ``_evidence/assoc/pathology/lost_edge_ledger_f{0,1}.json``. Any disagreement on
any crop is a hard refusal.

CALIBRATION BEFORE ANYTHING DERIVED IS TRUSTED
----------------------------------------------
Atlas coords are full-res (z,y,x) at (1.625, 0.40625, 0.40625) and are NOT isotropic; two of three
distance analyses in one day started wrong on exactly this. The module reproduces FACT-0040's GT
displacement median before a single proposal distance is read, and refuses otherwise.

THE ORACLE
----------
Every category-3 GT edge the frozen rule nominates is first split by its state in the DEPLOYED
final graph. An edge that motion relinking already recovered is an availability diagnostic, not
headroom. PERFECT SELECTION is simulated only for the genuinely-lost subset: insert the true edge,
drop whatever parent the target currently has. Nothing else changes. Both arms go through the
OFFICIAL scorer. Reported per fold SEPARATELY, never pooled alone (AGENTS.md section 4).

USAGE
    python scripts/win_bet/reach_oracle.py --fold 0 --out-dir _evidence/assoc/reach
    python scripts/win_bet/reach_oracle.py --fold 1 --out-dir _evidence/assoc/reach
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reach_rule  # noqa: E402
from assoc_lost_edge_ledger import (  # noqa: E402  - the committed definitions, NOT forked
    CATEGORIES,
    SUBSTRATE,
    LedgerRefusal,
    _match_per_frame,
)
from detpeak_curve import SCALE_UM  # noqa: E402

SCALE = np.asarray(SCALE_UM, dtype=np.float64)

# Recorded in PKT-0044 BEFORE this module existed. A run whose rule differs is not this experiment.
PACKET_RULE_SHA256 = "58252510152dd0d8d15792b54ee92bb8bbc965e6d426a06c3051f8594cb807eb"

# FACT-0040: median true inter-frame displacement of GT edges, FOLD 1, in um. The anchor is
# fold-1-scoped, so BOTH folds calibrate against fold-1 GT - the distance code is the same code and
# a scale error would show up identically. The committed instrument (nearest_parent_oracle.py)
# refuses outside +/-0.15 of its anchor; the same tolerance is used here.
ANCHOR_GT_DISP_MEDIAN_UM = 1.817
ANCHOR_TOL_UM = 0.15
CALIBRATION_CROPS = 12  # first N fold-1 crops in committed-list order

CONTEXTS = tuple(reach_rule.FROZEN_RULE["context_lengths"])


class ReachRefusal(RuntimeError):
    """Fail closed."""


def _cat3_identity(edge: dict) -> tuple[int, int, int, int]:
    """Stable identity used to prove that reach subsets are disjoint and exhaustive."""
    required = ("peak_source", "peak_target", "final_source", "final_target")
    missing = [key for key in required if key not in edge]
    if missing:
        raise ReachRefusal(f"category-3 edge is missing identity fields: {missing}")
    return tuple(int(edge[key]) for key in required)


def partition_reached_cat3(reached: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split temporally reached category-3 edges by their deployed final-graph state.

    ``category 3`` says that an edge was never offered to the learned selector.  It does not say
    that the final pipeline failed to recover it: motion relinking may independently recreate the
    same scored edge.  The two populations therefore have different meanings and must never be
    combined in an oracle headroom number.

    Returns ``(genuinely_lost, already_recovered)`` and refuses on missing state, duplicate edge
    identities, overlap, or a non-exhaustive partition.
    """
    identities: set[tuple[int, int, int, int]] = set()
    genuinely_lost: list[dict] = []
    already_recovered: list[dict] = []
    for edge in reached:
        identity = _cat3_identity(edge)
        if identity in identities:
            raise ReachRefusal(f"duplicate reached category-3 edge identity: {identity}")
        identities.add(identity)
        if "already_in_final_graph" not in edge:
            raise ReachRefusal(
                f"reached category-3 edge {identity} has no deployed final-graph state")
        if edge["already_in_final_graph"] is True:
            already_recovered.append(edge)
        elif edge["already_in_final_graph"] is False:
            genuinely_lost.append(edge)
        else:
            raise ReachRefusal(
                f"reached category-3 edge {identity} has non-boolean final-graph state")

    lost_ids = {_cat3_identity(edge) for edge in genuinely_lost}
    recovered_ids = {_cat3_identity(edge) for edge in already_recovered}
    if lost_ids & recovered_ids:
        raise ReachRefusal("genuinely-lost and already-recovered reach subsets overlap")
    if lost_ids | recovered_ids != identities:
        raise ReachRefusal("reach partition is not exhaustive")
    return genuinely_lost, already_recovered


def reconcile_reach_partition(
        total: int, genuinely_lost: int, already_recovered: int, *, label: str) -> None:
    """Fail closed unless a lost/recovered aggregation is non-negative and exhaustive."""
    values = (total, genuinely_lost, already_recovered)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ReachRefusal(f"{label}: reach partition counts must be non-negative integers")
    if genuinely_lost + already_recovered != total:
        raise ReachRefusal(
            f"{label}: reach partition does not reconcile: {genuinely_lost} genuinely lost + "
            f"{already_recovered} already recovered != {total} total")


def calibrate(gt_dir: Path) -> dict:
    """Reproduce FACT-0040 before ANY derived distance is trusted, on either fold.

    Atlas coords are FULL-RES (z,y,x) at (1.625, 0.40625, 0.40625) and are NOT isotropic. Two of
    three distance analyses in one day started wrong on exactly this, which is why this runs first
    and refuses rather than warns.
    """
    from biotrack.metric import load_graph

    ledger = json.loads(
        Path("_evidence/assoc/pathology/lost_edge_ledger_f1.json").read_text())
    crops = [r["crop"] for r in ledger["per_crop"]][:CALIBRATION_CROPS]
    d: list[float] = []
    for c in crops:
        g = load_graph(gt_dir / f"{c}.geff")
        n = g.node_attrs().to_pandas()
        e = g.edge_attrs().to_pandas()
        idx = {int(v): i for i, v in enumerate(n["node_id"].to_numpy())}
        P = n[["t", "z", "y", "x"]].to_numpy().astype(np.float64)
        for s, t in zip(e["source_id"], e["target_id"]):
            i, j = idx.get(int(s)), idx.get(int(t))
            if i is None or j is None:
                continue
            d.append(float(np.linalg.norm((P[j, 1:] - P[i, 1:]) * SCALE)))
    med = float(np.median(d))
    ok = abs(med - ANCHOR_GT_DISP_MEDIAN_UM) <= ANCHOR_TOL_UM
    if not ok:
        raise ReachRefusal(
            f"COORDINATE CONVENTION IS WRONG: fold-1 GT inter-frame displacement median "
            f"{med:.4f} um against the FACT-0040 anchor {ANCHOR_GT_DISP_MEDIAN_UM} "
            f"(tolerance {ANCHOR_TOL_UM}). Refusing to report any distance-derived quantity.")
    return {"anchor_fact": "FACT-0040", "anchor_um": ANCHOR_GT_DISP_MEDIAN_UM,
            "tolerance_um": ANCHOR_TOL_UM, "measured_um": med, "n_edges": len(d),
            "n_crops": len(crops), "fold_measured_on": 1, "scale_um": list(SCALE_UM),
            "passed": True}


# ---------------------------------------------------------------------------------------------
# category 3 identities, by the COMMITTED ordering
# ---------------------------------------------------------------------------------------------
def classify_crop(crop: str, gt_geff: Path, pre: pl.DataFrame, fin: pl.DataFrame) -> dict:
    """The committed first-match-wins ordering, but keeping EDGE IDENTITIES, not only counts.

    Every branch below is the corresponding branch of ``assoc_lost_edge_ledger.crop_ledger``, in
    the same order, on the same matcher. The counts it produces are asserted against the committed
    payload by the caller, which is what makes "not forked" a check rather than a promise.
    """
    from biotrack.metric import load_graph

    pre_nodes = pre.filter(pl.col("row_type") == "node")
    pre_edges = pre.filter(pl.col("row_type") == "edge")
    fin_nodes = fin.filter(pl.col("row_type") == "node")
    fin_edges = fin.filter(pl.col("row_type") == "edge")

    offered = {(int(s), int(t)) for s, t in zip(pre_edges["source_id"], pre_edges["target_id"])}
    final_edges = {(int(s), int(t)) for s, t in zip(fin_edges["source_id"], fin_edges["target_id"])}
    parent_of: dict[int, list[int]] = defaultdict(list)
    for s, t in final_edges:
        parent_of[t].append(s)
    # the DEPLOYED candidate parent of each target, in PEAK space. Used only to ask whether the
    # frozen rule's rank-1 nominee is the parent the deployed softmax already chose.
    deployed_parent: dict[int, int] = {int(t): int(s) for s, t in offered}

    gt_graph = load_graph(gt_geff)
    gtn = gt_graph.node_attrs().to_pandas()
    gt_ids = gtn["node_id"].to_numpy().astype(np.int64)
    gt_row = {int(v): i for i, v in enumerate(gt_ids)}
    gt_tzyx = gtn[["t", "z", "y", "x"]].to_numpy().astype(np.float64)
    gte = gt_graph.edge_attrs().to_pandas()

    to_peak = _match_per_frame(
        gt_tzyx,
        pre_nodes["t"].to_numpy().astype(np.int64),
        pre_nodes.select(["z", "y", "x"]).to_numpy().astype(np.float64) * SCALE,
        pre_nodes["node_id"].to_numpy().astype(np.int64))
    to_final = _match_per_frame(
        gt_tzyx,
        fin_nodes["t"].to_numpy().astype(np.int64),
        fin_nodes.select(["z", "y", "x"]).to_numpy().astype(np.float64) * SCALE,
        fin_nodes["node_id"].to_numpy().astype(np.int64))

    counts = {k: 0 for k in CATEGORIES}
    counts["recovered"] = 0
    cat3: list[dict] = []
    gt_disp_um: list[float] = []
    gt_edges = 0

    for s, t in zip(gte["source_id"], gte["target_id"]):
        su, tv = gt_row.get(int(s)), gt_row.get(int(t))
        if su is None or tv is None:
            continue
        gt_edges += 1
        disp = float(np.linalg.norm((gt_tzyx[tv, 1:] - gt_tzyx[su, 1:]) * SCALE))

        if su not in to_peak or tv not in to_peak:
            counts["cat1_endpoint_absent_from_peaks"] += 1
            continue
        if su not in to_final or tv not in to_final:
            counts["cat2_endpoint_removed_by_node_selection"] += 1
            continue
        a, b = to_peak[su], to_peak[tv]
        if (a, b) not in offered and (b, a) not in offered:
            counts["cat3_true_edge_never_offered"] += 1
            fa, fb = to_final[su], to_final[tv]
            # A category-3 edge is one that was NEVER OFFERED at pre-ILP. That is a statement
            # about the candidate surface, NOT about the emitted graph: motion_relink_edges
            # rebuilds the whole edge list from geometry (stage_map order 4), so it can and does
            # emit pairs the candidate rule never proposed. Splitting the two is the difference
            # between a prize and a number.
            present = (fa, fb) in final_edges or (fb, fa) in final_edges
            cat3.append({
                "peak_source": int(a), "peak_target": int(b),
                "final_source": int(fa), "final_target": int(fb),
                "gt_frame_target": int(gt_tzyx[tv, 0]),
                "gt_displacement_um": disp,
                "already_in_final_graph": bool(present),
                "target_has_final_parent": bool(parent_of.get(fb)),
                "deployed_candidate_parent": deployed_parent.get(int(b), -1),
            })
            continue
        fa, fb = to_final[su], to_final[tv]
        if (fa, fb) in final_edges or (fb, fa) in final_edges:
            counts["recovered"] += 1
            gt_disp_um.append(disp)
            continue
        if parent_of.get(fb):
            counts["cat4_wrong_parent_selected"] += 1
            continue
        counts["cat5_selected_then_overwritten_provisional"] += 1

    return {"crop": crop, "gt_edges": gt_edges, "counts": counts, "cat3": cat3,
            "recovered_displacement_um": gt_disp_um}


# ---------------------------------------------------------------------------------------------
# the frozen rule, applied to a crop
# ---------------------------------------------------------------------------------------------
def crop_proposals(pre: pl.DataFrame, context: int) -> tuple[dict, int, int]:
    """{target_node_id: [(source_node_id, um), ...]} plus (deployed edges, proposed pairs)."""
    nodes = pre.filter(pl.col("row_type") == "node").sort("node_id")
    edges = pre.filter(pl.col("row_type") == "edge")

    node_id = nodes["node_id"].to_numpy().astype(np.int64)
    t = nodes["t"].to_numpy().astype(np.int64)
    zyx = nodes.select(["z", "y", "x"]).to_numpy().astype(np.float64)
    index_of = {int(v): i for i, v in enumerate(node_id)}

    parent = np.full(len(node_id), -1, dtype=np.int64)
    n_dep = 0
    for s, tt in zip(edges["source_id"], edges["target_id"]):
        si, ti = index_of.get(int(s)), index_of.get(int(tt))
        n_dep += 1
        if si is None or ti is None:
            continue
        if parent[ti] != -1:
            raise ReachRefusal("a target has two candidate parents - FACT-0369 says this is "
                               "impossible on the deployed surface, so the substrate is wrong")
        parent[ti] = si

    props = reach_rule.propose(t, zyx, node_id, parent, context)
    n_prop = sum(len(v) for v in props.values())
    return props, n_dep, n_prop


# ---------------------------------------------------------------------------------------------
# the oracle graph edit and the official score
# ---------------------------------------------------------------------------------------------
def apply_oracle(fin: pl.DataFrame, inserts: list[tuple[int, int]]) -> tuple[pl.DataFrame, dict]:
    """Perfect selection over the newly reachable edges only. Everything else untouched."""
    nodes = fin.filter(pl.col("row_type") == "node")
    edges = fin.filter(pl.col("row_type") == "edge")
    pairs = [(int(s), int(t)) for s, t in zip(edges["source_id"], edges["target_id"])]

    children = defaultdict(set)
    parent = {}
    for s, t in pairs:
        children[s].add(t)
        parent[t] = s

    # HARNESS SELF-PROOF, unconditional and on every crop: the parent-map representation this
    # oracle edits must round-trip the DEPLOYED edge set exactly before anything is inserted. The
    # machinery that measures a ceiling must be the machinery that reproduces the control - the
    # FACT-0384 adapter-proof pattern.
    original = set(pairs)
    if {(s, t) for t, s in parent.items()} != original:
        raise ReachRefusal(
            f"the oracle's parent-map rebuild does not round-trip the deployed edge set "
            f"({len(parent)} against {len(original)} edges), so the emitted graph has a target "
            "with more than one parent and this representation is wrong for it")
    stats = {"inserted": 0, "displaced_a_wrong_parent": 0, "target_had_no_parent": 0,
             "skipped_outdegree": 0, "already_present": 0}
    for fa, fb in inserts:
        if parent.get(fb) == fa:
            stats["already_present"] += 1
            continue
        # the source may not take a third child: GT lineage degree is at most 2 and a 3-child
        # node is not a graph any real selector could emit. Skipping is the conservative choice.
        if len(children[fa] - {fb}) >= 2:
            stats["skipped_outdegree"] += 1
            continue
        if fb in parent:
            children[parent[fb]].discard(fb)
            stats["displaced_a_wrong_parent"] += 1
        else:
            stats["target_had_no_parent"] += 1
        parent[fb] = fa
        children[fa].add(fb)
        stats["inserted"] += 1

    new_pairs = sorted((s, t) for t, s in parent.items())
    if not stats["inserted"] and set(new_pairs) != original:
        raise ReachRefusal("zero insertions changed the edge set - the harness is wrong")
    crop = nodes["dataset"][0]
    m = len(new_pairs)
    edge_rows = pl.DataFrame({
        "dataset": [crop] * m, "row_type": ["edge"] * m,
        "node_id": [-1] * m, "t": [-1] * m,
        "z": [-1.0] * m, "y": [-1.0] * m, "x": [-1.0] * m,
        "source_id": [int(a) for a, _ in new_pairs],
        "target_id": [int(b) for _, b in new_pairs],
    })
    keep = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    nodes = nodes.select(keep).with_columns(
        pl.col("z").cast(pl.Float64), pl.col("y").cast(pl.Float64), pl.col("x").cast(pl.Float64))
    return pl.concat([nodes, edge_rows.select(keep)]), stats


def score_rows(rows: pl.DataFrame, gt_geff: Path) -> dict:
    """The OFFICIAL scorer, byte-for-byte the path ceiling_ladder.py uses."""
    import tempfile

    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, estimated_nodes, load_graph
    from biotrack.submission import read_submission, submission_to_graphs
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
        tmp = Path(fh.name)
    rows.drop("id", strict=False).with_row_index("id").write_csv(tmp)
    try:
        graphs = submission_to_graphs(read_submission(tmp))
        pred = graphs[next(iter(graphs))]
        gt = load_graph(gt_geff)
        er = evaluate(pred, gt, scale=DEFAULT_SCALE, max_distance=MAX_DISTANCE)
        recall = node_recall(pred, gt) if pred.num_edges() and pred.num_nodes() else 0.0
        return per_sample_metrics(er, estimated_nodes(gt_geff), recall)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fold", type=int, required=True, choices=(0, 1))
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--out-dir", type=Path, default=Path("_evidence/assoc/reach"))
    ap.add_argument("--max-crops", type=int)
    ap.add_argument("--no-score", action="store_true",
                    help="reach only; skip the official scorer (smoke)")
    args = ap.parse_args(argv)

    t0 = time.time()

    # ---- the freeze, re-checked ---------------------------------------------------------
    reach_rule.assert_gt_free()
    h = reach_rule.rule_hash()
    if h["rule_pickle_sha256"] != PACKET_RULE_SHA256:
        raise ReachRefusal(
            f"the proposal rule has CHANGED since PKT-0044 recorded it: "
            f"{h['rule_pickle_sha256']} against {PACKET_RULE_SHA256}. Tuning the rule after a "
            "result is the one thing this packet forbids, so this run refuses.")

    # ---- CALIBRATION GATE: nothing derived from a distance is read before this passes ----
    calib = calibrate(args.gt_dir)
    print(f"  CALIBRATION OK  fold-1 GT displacement median {calib['measured_um']:.4f} um "
          f"over {calib['n_edges']:,} edges (FACT-0040 anchor {calib['anchor_um']})", flush=True)

    cfg = SUBSTRATE[args.fold]
    pre_all = pl.read_parquet(cfg["preilp"])
    fin_all = pl.read_csv(cfg["final_csv"]) if cfg["final_csv"] else None
    ledger = json.loads(
        Path(f"_evidence/assoc/pathology/lost_edge_ledger_f{args.fold}.json").read_text())
    committed = {r["crop"]: r for r in ledger["per_crop"]}

    crops = sorted(pre_all["dataset"].unique().to_list())
    if args.max_crops:
        crops = crops[: args.max_crops]
    complete = args.max_crops is None

    per_crop: list[dict] = []
    disp_pool: list[float] = []

    for i, crop in enumerate(crops, 1):
        gt_geff = args.gt_dir / f"{crop}.geff"
        if not gt_geff.exists():
            raise ReachRefusal(f"missing GT for {crop}")
        pre = pre_all.filter(pl.col("dataset") == crop)
        if fin_all is not None:
            fin = fin_all.filter(pl.col("dataset") == crop)
        else:
            p = Path(cfg["final_dir"]) / f"{crop}.parquet"
            if not p.is_file():
                raise ReachRefusal(f"{crop}: no persisted final graph at {p}")
            fin = pl.read_parquet(p)

        cl = classify_crop(crop, gt_geff, pre, fin)
        want = committed[crop]
        for k in list(CATEGORIES) + ["recovered", "gt_edges"]:
            got = cl["gt_edges"] if k == "gt_edges" else cl["counts"][k]
            if int(got) != int(want[k]):
                raise ReachRefusal(
                    f"{crop}: category identities disagree with the COMMITTED ledger on {k} "
                    f"({got} against {want[k]}). A second definition of category 3 would make "
                    "this result incomparable with FACT-0422, so nothing is written.")
        disp_pool.extend(cl["recovered_displacement_um"])

        row = {"crop": crop, "gt_edges": cl["gt_edges"],
               "cat3": cl["counts"]["cat3_true_edge_never_offered"],
               "cat2": cl["counts"]["cat2_endpoint_removed_by_node_selection"],
               "cat4": cl["counts"]["cat4_wrong_parent_selected"],
               "cat5": cl["counts"]["cat5_selected_then_overwritten_provisional"],
               "cat3_already_in_final_graph": int(sum(
                   c["already_in_final_graph"] for c in cl["cat3"])),
               "cat3_absent_from_final_graph": int(sum(
                   not c["already_in_final_graph"] for c in cl["cat3"])),
               "cat3_target_has_final_parent": int(sum(
                   c["target_has_final_parent"] for c in cl["cat3"])),
               "contexts": {}}
        reconcile_reach_partition(
            row["cat3"], row["cat3_absent_from_final_graph"],
            row["cat3_already_in_final_graph"], label=f"{crop}: category-3 final graph")
        if not args.no_score:
            row["score_control"] = score_rows(fin, gt_geff)

        for L in CONTEXTS:
            props, n_dep, n_prop = crop_proposals(pre, L)
            new_pairs = 0
            reached, ranks, resid = [], [], []
            ranks_absent, ranks_recovered = [], []
            resid_absent, resid_recovered = [], []
            rank1_is_deployed_parent = 0
            rank1_is_deployed_parent_absent = 0
            rank1_is_deployed_parent_recovered = 0
            for c in cl["cat3"]:
                picks = props.get(c["peak_target"], [])
                hit = next((r for r, (s, _d) in enumerate(picks, 1)
                            if s == c["peak_source"]), None)
                if hit is None:
                    continue
                reached.append(c)
                ranks.append(hit)
                resid.append(picks[hit - 1][1])
                if not c["already_in_final_graph"]:
                    ranks_absent.append(hit)
                    resid_absent.append(picks[hit - 1][1])
                else:
                    ranks_recovered.append(hit)
                    resid_recovered.append(picks[hit - 1][1])
                if picks and picks[0][0] == c["deployed_candidate_parent"]:
                    rank1_is_deployed_parent += 1
                    if c["already_in_final_graph"]:
                        rank1_is_deployed_parent_recovered += 1
                    else:
                        rank1_is_deployed_parent_absent += 1
            # multiplier: proposals that are NOT already deployed candidate edges
            offered = {(int(s), int(t)) for s, t in zip(
                pre.filter(pl.col("row_type") == "edge")["source_id"],
                pre.filter(pl.col("row_type") == "edge")["target_id"])}
            for tgt, picks in props.items():
                for s, _d in picks:
                    if (s, tgt) not in offered:
                        new_pairs += 1

            genuinely_lost, already_recovered = partition_reached_cat3(reached)
            entry = {"context": L, "deployed_edges": n_dep, "proposed_pairs": n_prop,
                     "proposed_new_pairs": new_pairs,
                     # `temporally_reached` is deliberately not called `newly_reachable`: some
                     # of these edges have already been recovered by the deployed final chain.
                     "cat3_temporally_reached": len(reached),
                     "cat3_genuinely_lost_reached": len(genuinely_lost),
                     "cat3_already_recovered_reached": len(already_recovered),
                     "rank_hist": {str(r): int(sum(1 for x in ranks if x == r))
                                   for r in range(1, reach_rule.FROZEN_RULE["cap_k"] + 1)},
                     "rank_hist_genuinely_lost": {
                         str(r): int(sum(1 for x in ranks_absent if x == r))
                         for r in range(1, reach_rule.FROZEN_RULE["cap_k"] + 1)},
                     "rank_hist_already_recovered": {
                         str(r): int(sum(1 for x in ranks_recovered if x == r))
                         for r in range(1, reach_rule.FROZEN_RULE["cap_k"] + 1)},
                     "rank1_is_the_deployed_candidate_parent": int(rank1_is_deployed_parent),
                     "rank1_is_the_deployed_candidate_parent_genuinely_lost": int(
                         rank1_is_deployed_parent_absent),
                     "rank1_is_the_deployed_candidate_parent_already_recovered": int(
                         rank1_is_deployed_parent_recovered),
                     "residual_um_median": float(np.median(resid)) if resid else None,
                     "residual_um_median_genuinely_lost": (
                         float(np.median(resid_absent)) if resid_absent else None),
                     "residual_um_median_already_recovered": (
                         float(np.median(resid_recovered)) if resid_recovered else None),
                     "reached_target_has_final_parent": int(sum(
                         c["target_has_final_parent"] for c in reached)),
                     "genuinely_lost_target_has_final_parent": int(sum(
                         c["target_has_final_parent"] for c in genuinely_lost)),
                     "already_recovered_target_has_final_parent": int(sum(
                         c["target_has_final_parent"] for c in already_recovered))}
            reconcile_reach_partition(
                entry["cat3_temporally_reached"], entry["cat3_genuinely_lost_reached"],
                entry["cat3_already_recovered_reached"], label=f"{crop}: context {L}")
            if not args.no_score:
                # Only the genuinely lost population can create score headroom. Passing the
                # already-recovered subset into surgery would make an availability diagnostic
                # masquerade as an oracle intervention, even if apply_oracle later skipped it.
                trt, st = apply_oracle(fin, [(c["final_source"], c["final_target"])
                                             for c in genuinely_lost])
                if st["already_present"]:
                    raise ReachRefusal(
                        f"{crop}: oracle surgery received {st['already_present']} already-"
                        "recovered edges; the reach partition is wrong")
                entry["oracle_edit"] = st
                entry["score_oracle"] = score_rows(trt, gt_geff)
            row["contexts"][str(L)] = entry
        per_crop.append(row)
        msg = "  ".join(
            f"L{L}={row['contexts'][str(L)]['cat3_genuinely_lost_reached']} lost+"
            f"{row['contexts'][str(L)]['cat3_already_recovered_reached']} recovered"
            for L in CONTEXTS)
        print(f"  [{i}/{len(crops)}] {crop}  cat3={row['cat3']}  {msg}", flush=True)

    payload = build_payload(args, cfg, ledger, per_crop, complete, disp_pool, t0, calib)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"reach_oracle_f{args.fold}.json"
    p.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print_report(payload)
    print(f"\nREACH_ORACLE_COMPLETE -> {p}")
    return 0


def build_payload(args, cfg, ledger, per_crop, complete, disp_pool, t0, calib) -> dict:
    from tracking_cellmot.metrics import summarise

    tot_cat3 = int(sum(r["cat3"] for r in per_crop))
    tot_present = int(sum(r["cat3_already_in_final_graph"] for r in per_crop))
    tot_absent = int(sum(r["cat3_absent_from_final_graph"] for r in per_crop))
    reconcile_reach_partition(
        tot_cat3, tot_absent, tot_present, label="aggregate category-3 final graph")
    ctx: dict[str, dict] = {}
    for L in CONTEXTS:
        k = str(L)
        rows = [r["contexts"][k] for r in per_crop]
        reached = int(sum(x["cat3_temporally_reached"] for x in rows))
        reached_absent = int(sum(x["cat3_genuinely_lost_reached"] for x in rows))
        reached_recovered = int(sum(x["cat3_already_recovered_reached"] for x in rows))
        reconcile_reach_partition(
            reached, reached_absent, reached_recovered, label=f"aggregate context {L}")
        dep = int(sum(x["deployed_edges"] for x in rows))
        newp = int(sum(x["proposed_new_pairs"] for x in rows))
        ranks, ranks_absent, ranks_recovered = defaultdict(int), defaultdict(int), defaultdict(int)
        for x in rows:
            for r, n in x["rank_hist"].items():
                ranks[r] += int(n)
            for r, n in x["rank_hist_genuinely_lost"].items():
                ranks_absent[r] += int(n)
            for r, n in x["rank_hist_already_recovered"].items():
                ranks_recovered[r] += int(n)
        e = {
            "context": L,
            "cat3_total": tot_cat3,
            "cat3_already_in_final_graph": tot_present,
            "cat3_absent_from_final_graph": tot_absent,
            "cat3_temporally_reached": reached,
            "cat3_temporal_reach_share": reached / max(tot_cat3, 1),
            "cat3_genuinely_lost_reached": reached_absent,
            "cat3_genuinely_lost_reach_share": reached_absent / max(tot_absent, 1),
            "cat3_already_recovered_reached": reached_recovered,
            "true_parent_rank": dict(sorted(ranks.items())),
            "true_parent_rank_genuinely_lost": dict(sorted(ranks_absent.items())),
            "true_parent_rank_already_recovered": dict(sorted(ranks_recovered.items())),
            "true_parent_rank1_share_of_reached": ranks["1"] / max(reached, 1),
            "rank1_is_the_deployed_candidate_parent": int(sum(
                x["rank1_is_the_deployed_candidate_parent"] for x in rows)),
            "rank1_is_the_deployed_candidate_parent_genuinely_lost": int(sum(
                x["rank1_is_the_deployed_candidate_parent_genuinely_lost"] for x in rows)),
            "rank1_is_the_deployed_candidate_parent_already_recovered": int(sum(
                x["rank1_is_the_deployed_candidate_parent_already_recovered"] for x in rows)),
            "deployed_candidate_edges": dep,
            "proposed_new_pairs": newp,
            "candidate_multiplier": (dep + newp) / max(dep, 1),
            "reached_target_has_final_parent": int(sum(
                x["reached_target_has_final_parent"] for x in rows)),
            "genuinely_lost_target_has_final_parent": int(sum(
                x["genuinely_lost_target_has_final_parent"] for x in rows)),
            "already_recovered_target_has_final_parent": int(sum(
                x["already_recovered_target_has_final_parent"] for x in rows)),
            "per_crop_genuinely_lost_reach_share": [
                (r["contexts"][k]["cat3_genuinely_lost_reached"]
                 / r["cat3_absent_from_final_graph"]
                 if r["cat3_absent_from_final_graph"] else None)
                for r in per_crop],
        }
        if "score_oracle" in rows[0]:
            ctl = summarise([{"dataset": r["crop"], **r["score_control"]} for r in per_crop])
            trt = summarise([{"dataset": r["crop"], **r["contexts"][k]["score_oracle"]}
                             for r in per_crop])
            e["score_control"] = {kk: (None if isinstance(v, float) and np.isnan(v) else v)
                                  for kk, v in ctl.items()}
            e["score_oracle"] = {kk: (None if isinstance(v, float) and np.isnan(v) else v)
                                 for kk, v in trt.items()}
            e["oracle_score_delta"] = float(trt["score"] - ctl["score"])
            e["oracle_raw_edge_jaccard_delta"] = float(trt["edge_jaccard"] - ctl["edge_jaccard"])
            e["oracle_division_jaccard_delta"] = float(
                trt["division_jaccard"] - ctl["division_jaccard"])
            edit = defaultdict(int)
            for r in per_crop:
                for kk, v in r["contexts"][k]["oracle_edit"].items():
                    edit[kk] += int(v)
            e["oracle_edit"] = dict(edit)
        ctx[k] = e

    return {
        "schema_version": 2,
        "heartbeat": "REACH_ORACLE_COMPLETE",
        "packet": "PKT-0044", "lever": "LEVER-0045", "falsifier": "FALSIFIER 0",
        "fold": args.fold, "embryo": cfg["embryo"], "n_crops": len(per_crop),
        "complete_fold": complete,
        "frozen_rule": {**reach_rule.rule_hash(), "rule": reach_rule.FROZEN_RULE},
        "calibration": {
            **calib,
            "this_fold_recovered_edge_displacement_median_um": (
                float(np.median(disp_pool)) if disp_pool else None),
            "this_fold_recovered_edges": len(disp_pool),
        },
        "ledger_agreement": {
            "source": f"_evidence/assoc/pathology/lost_edge_ledger_f{args.fold}.json",
            "checked": "all six categories plus recovered plus gt_edges, on every crop",
            "committed_cat3": ledger["totals"]["cat3_true_edge_never_offered"],
            "measured_cat3": tot_cat3,
            "agrees": tot_cat3 == ledger["totals"]["cat3_true_edge_never_offered"] or not complete,
        },
        "category_2_overlap": {
            "structural": "ZERO BY THE COMMITTED ORDERING. Category 2 is tested BEFORE category 3, "
                          "so every category-3 edge already has BOTH endpoints present in the "
                          "emitted graph. The category-2 drain FACT-0430 warns about is already "
                          "removed from this population - it is not a further haircut on it.",
            "drained_before_cat3": ledger["fact_0370_reconciliation"],
        },
        "category_5_overlap": {
            "structural": "ZERO BY CONSTRUCTION. Category 5 requires the edge to have been "
                          "OFFERED; a category-3 edge never was. The measurable relation is the "
                          "state of the target in the emitted graph, reported as "
                          "`reached_target_has_final_parent`.",
        },
        "contexts": ctx,
        "per_crop": per_crop,
        "promotable": False,
        "elapsed_s": round(time.time() - t0, 1),
    }


def print_report(p: dict) -> None:
    print(f"\nREACH ORACLE  fold {p['fold']} ({p['embryo']})  crops {p['n_crops']}  "
          f"complete={p['complete_fold']}")
    print(f"  calibration {p['calibration']['measured_um']:.3f} um vs FACT-0040 anchor "
          f"{p['calibration']['anchor_um']}")
    e0 = next(iter(p["contexts"].values()))
    print(f"  cat3 {e0['cat3_total']:,}  of which ALREADY IN THE EMITTED GRAPH "
          f"{e0['cat3_already_in_final_graph']:,} "
          f"({e0['cat3_already_in_final_graph'] / max(e0['cat3_total'], 1):.4f})  -  "
          f"genuinely absent {e0['cat3_absent_from_final_graph']:,}")
    for k, e in p["contexts"].items():
        print(f"\n  context {k} frames")
        print(f"    cat3 temporal reach    {e['cat3_temporally_reached']:,} / {e['cat3_total']:,}"
              f"  ({e['cat3_temporal_reach_share']:.4f})")
        print(f"    ... GENUINELY LOST     {e['cat3_genuinely_lost_reached']:,} / "
              f"{e['cat3_absent_from_final_graph']:,}  "
              f"({e['cat3_genuinely_lost_reach_share']:.4f})"
              f"   <- the only part that can pay")
        print(f"    ... ALREADY RECOVERED  {e['cat3_already_recovered_reached']:,}"
              "   <- availability diagnostic only")
        print(f"    true-parent rank       {e['true_parent_rank']}   "
              f"genuinely-lost {e['true_parent_rank_genuinely_lost']}   "
              f"already-recovered {e['true_parent_rank_already_recovered']}")
        print(f"    candidate multiplier   {e['candidate_multiplier']:.3f}x  "
              f"(+{e['proposed_new_pairs']:,} pairs on {e['deployed_candidate_edges']:,})")
        if "oracle_score_delta" in e:
            print(f"    ORACLE score delta     {e['oracle_score_delta']:+.5f}   "
                  f"rawJ {e['oracle_raw_edge_jaccard_delta']:+.5f}   "
                  f"divJ {e['oracle_division_jaccard_delta']:+.5f}")
            print(f"    oracle edit            {e['oracle_edit']}")


if __name__ == "__main__":
    raise SystemExit(main())
