r"""THE UNIFIED LOST-EDGE LEDGER - every recoverable miss attributed to the stage responsible.

WHY ONE LEDGER
--------------
This campaign has repeatedly measured a gain at one stage and then discovered the stage did not
own the loss. ``FACT-0364`` is the canonical case: ``motion_relink_edges`` replaces the solver's
whole edge list at a median 99.9% coverage, so an ILP-stage edge gain is overwritten before it
reaches the score. ``FACT-0376`` is the other: widening candidates raised GT reach by 924 edges
and the chain converted 28. A per-stage proxy cannot tell those apart. One ledger, one denominator,
mutually exclusive categories, can.

THE SIX CATEGORIES, IN ORDER. FIRST MATCH WINS - the ordering is what makes them exclusive, and
it is applied literally:

  1  endpoint_absent_from_peaks        an endpoint was never detected at all
  2  endpoint_removed_by_node_selection  detected, then dropped before the final graph
  3  true_edge_never_offered           both endpoints survive, no candidate edge between them
  4  wrong_parent_selected             offered, and the final graph gives the child a DIFFERENT parent
  5  selected_then_overwritten         offered, and the final graph gives the child NO parent
  6  division_topology_failure         structurally unreachable under this ordering - see below

Plus ``recovered`` - the GT edge is present in the final graph. The instrument ASSERTS that the
six categories plus ``recovered`` sum to the GT edge total on EVERY crop and REFUSES if they do
not, so exhaustiveness is a contract rather than a claim.

CATEGORY 5 IS PROVISIONAL AND SAYS SO
-------------------------------------
"Correct edge selected initially, then overwritten" needs the ILP's own edge list BEFORE
``motion_relink_edges`` replaces it, and no committed artifact carries it - the replay persists
final graphs (``C:/temp/p30_f0/graphs_ctl``) and per-crop metrics, not the intermediate solve.
PKT-0040 (Agent 3) owns that stage boundary. So this module measures the OBSERVABLE shape of
category 5 - the true edge was offered and the child ends with no parent at all - labels it
``provisional: true``, and does NOT fold its mass into category 4. A category silently absorbing
another's mass is exactly the failure this ledger exists to prevent, so the two are separated by
an observable (does the child have a parent?) rather than by an assumption.

CATEGORY 6 IS STRUCTURALLY EMPTY UNDER THIS ORDERING, AND THAT IS A FINDING, NOT A BUG
--------------------------------------------------------------------------------------
Every missing GT edge whose endpoints survive and which was offered ends in 4 or 5 by the time
the ordering reaches 6, because "the child has a different parent" and "the child has no parent"
between them exhaust the ways an edge can be absent. Reporting 6 as an exact 0 on the edge
denominator is therefore correct AND vacuous, so a SECOND ledger is reported on the DIVISION
denominator - GT division events, not edges - where the question is meaningful. The two
denominators are never added. ``FACT-0371`` governs the reading: divisions are downstream of
association, and nothing here is read as division recovery.

MATCHING IS THE OFFICIAL SCORER'S
---------------------------------
One-to-one bipartite assignment within 7 um per frame at scale (1.625, 0.40625, 0.40625) -
``detpeak_curve.match_one_to_one_pairs``, the same implementation behind FACT-0354/0355/0357 and
FACT-0370. Every GT node is matched TWICE: once into the detector peak set (the pre-ILP export)
and once into the FINAL graph. Categories 1 and 2 are the difference between those two matchings.

RECONCILIATION, NOT RE-DERIVATION
---------------------------------
``FACT-0370`` measured fold 1's three-way split - reachable / both-endpoints-detected-but-no-
candidate / undetected endpoint - on the SAME pre-ILP export this module reads (verified byte
identical: C:/temp/p34_f1/preilp_split1.parquet and C:/temp/preilp_f1_v2/preilp_split1.parquet
share a sha256). That split IGNORES node selection, so it cannot be compared to the six categories
directly. This module therefore emits FACT-0370's three-way split ALONGSIDE the ledger, computed
from the same matching, and ASSERTS it against the published fold-1 counts. A ledger whose
reconciliation fails is a defect in the ledger.

    python scripts/win_bet/assoc_lost_edge_ledger.py --fold 1 --out-dir _evidence/assoc/pathology
    python scripts/win_bet/assoc_lost_edge_ledger.py --fold 0 --out-dir _evidence/assoc/pathology
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

from detpeak_curve import MAX_DISTANCE_UM, SCALE_UM, match_one_to_one_pairs  # noqa: E402

SCALE = np.asarray(SCALE_UM, dtype=np.float64)

CATEGORIES = (
    "cat1_endpoint_absent_from_peaks",
    "cat2_endpoint_removed_by_node_selection",
    "cat3_true_edge_never_offered",
    "cat4_wrong_parent_selected",
    "cat5_selected_then_overwritten_provisional",
    "cat6_division_topology_failure",
)

# The substrates, bound by path. Both pre-ILP exports are at the DEPLOYED candidate floor - fold 1
# min edge_prob 0.5000002 over 2,162,040 edges (FACT-0369), fold 0 min 0.5000010 over 1,973,714
# (FACT-0376's `candidates_deployed`) - so "offered" in this ledger means offered to the DEPLOYED
# pipeline, not offered on the widened 0.1 acquisition surface FACT-0382 describes.
SUBSTRATE: dict[int, dict] = {
    0: {
        "embryo": "44b6",
        "preilp": "C:/temp/p30_f0/preilp_split0.parquet",
        "final_dir": "C:/temp/p30_f0/graphs_ctl",     # per-crop parquet, float coordinates
        "final_csv": None,
        "n_crops": 71,
    },
    1: {
        "embryo": "6bba",
        "preilp": "C:/temp/p34_f1/preilp_split1.parquet",
        "final_dir": None,
        # fold 1 has no per-crop persisted graph; the champion export is the final graph, and its
        # coordinates are INTEGER (submission format). Rounding is at most half a voxel - 0.20 um
        # in y/x and 0.81 um in z against a 7 um gate - so it can only matter for a match already
        # at the boundary. Recorded here rather than discovered later.
        "final_csv": "C:/temp/p34_f1/loeo_split1_champion.csv.gz",
        "n_crops": 128,
    },
}

# FACT-0370's published fold-1 three-way split. Quoted as an ASSERTION TARGET: the run fails if
# the reconciliation does not land on it, which is the only use a literal has in an instrument.
FACT_0370_F1 = {"gt_edges": 109057, "reachable": 81053,
                "unreachable_no_candidate": 17003, "undetected_endpoint": 11001}

# FACT-0376's fold-0 OPPORTUNITY counts, which are the solid half of that fact - its SUSPECT flag is
# on the mechanism claim (a NET edge-TP count), not on these. ``reached_deployed`` must equal this
# ledger's fold-0 ``reachable``, and ``gt_detectable`` must equal its GT total minus the undetected
# endpoints, because both are the same quantity computed by different code on the same export.
FACT_0376_F0 = {"gt_detectable": 19665, "reached_deployed": 18466}


class LedgerRefusal(RuntimeError):
    """Fail closed. A ledger that cannot account for every edge is not a ledger."""


def assert_exhaustive(counts: dict, gt_edges: int, where: str) -> None:
    """The six categories plus ``recovered`` MUST account for every GT edge.

    Extracted so the contract is callable - and therefore testable by planting the violation -
    rather than being an inline expression no test can reach. A category silently absorbing
    another's mass, or quietly counting nothing, is invisible in a payload; this is the only thing
    that makes it a crash.
    """
    total = sum(int(counts.get(k, 0)) for k in CATEGORIES) + int(counts.get("recovered", 0))
    if total != int(gt_edges):
        raise LedgerRefusal(
            f"{where}: the ledger does not account for every GT edge - six categories plus "
            f"recovered sum to {total} against {gt_edges} GT edges. Exhaustiveness is a contract."
        )


def _match_per_frame(gt_tzyx: np.ndarray, pred_t: np.ndarray, pred_xyz_um: np.ndarray,
                     pred_ids: np.ndarray) -> dict[int, int]:
    """GT row index -> predicted node id, official one-to-one within 7 um, frame by frame."""
    out: dict[int, int] = {}
    for frame in np.unique(gt_tzyx[:, 0]).astype(np.int64):
        gm = np.nonzero(gt_tzyx[:, 0] == frame)[0]
        pm = np.nonzero(pred_t == frame)[0]
        if not len(pm) or not len(gm):
            continue
        for g, p, _d in match_one_to_one_pairs(pred_xyz_um[pm], gt_tzyx[gm, 1:] * SCALE,
                                               MAX_DISTANCE_UM):
            out[int(gm[g])] = int(pred_ids[pm][p])
    return out


def crop_ledger(crop: str, gt_geff: Path, pre: pl.DataFrame, fin: pl.DataFrame) -> dict:
    from biotrack.metric import load_graph

    pre_nodes = pre.filter(pl.col("row_type") == "node")
    pre_edges = pre.filter(pl.col("row_type") == "edge")
    fin_nodes = fin.filter(pl.col("row_type") == "node")
    fin_edges = fin.filter(pl.col("row_type") == "edge")
    if pre_nodes.height == 0:
        raise LedgerRefusal(f"{crop}: pre-ILP export has no nodes")
    if fin_nodes.height == 0:
        raise LedgerRefusal(f"{crop}: final graph has no nodes")

    offered = {(int(s), int(t)) for s, t in zip(pre_edges["source_id"], pre_edges["target_id"])}
    final_edges = {(int(s), int(t)) for s, t in zip(fin_edges["source_id"], fin_edges["target_id"])}
    parent_of: dict[int, list[int]] = defaultdict(list)
    children_of: dict[int, list[int]] = defaultdict(list)
    for s, t in final_edges:
        parent_of[t].append(s)
        children_of[s].append(t)

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
    # FACT-0370's three-way split, computed from the SAME matching so the reconciliation is real
    reach = {"reachable": 0, "unreachable_no_candidate": 0, "undetected_endpoint": 0}
    gt_edges = 0
    gt_children: dict[int, list[int]] = defaultdict(list)

    for s, t in zip(gte["source_id"], gte["target_id"]):
        su, tv = gt_row.get(int(s)), gt_row.get(int(t))
        if su is None or tv is None:
            continue
        gt_edges += 1
        gt_children[int(s)].append(int(t))

        # --- FACT-0370's split, node selection ignored ------------------------------------
        if su not in to_peak or tv not in to_peak:
            reach["undetected_endpoint"] += 1
        elif (to_peak[su], to_peak[tv]) in offered or (to_peak[tv], to_peak[su]) in offered:
            reach["reachable"] += 1
        else:
            reach["unreachable_no_candidate"] += 1

        # --- the six categories, FIRST MATCH WINS ----------------------------------------
        if su not in to_peak or tv not in to_peak:
            counts["cat1_endpoint_absent_from_peaks"] += 1
            continue
        if su not in to_final or tv not in to_final:
            counts["cat2_endpoint_removed_by_node_selection"] += 1
            continue
        a, b = to_peak[su], to_peak[tv]
        if (a, b) not in offered and (b, a) not in offered:
            counts["cat3_true_edge_never_offered"] += 1
            continue
        fa, fb = to_final[su], to_final[tv]
        if (fa, fb) in final_edges or (fb, fa) in final_edges:
            counts["recovered"] += 1
            continue
        if parent_of.get(fb):
            counts["cat4_wrong_parent_selected"] += 1
            continue
        counts["cat5_selected_then_overwritten_provisional"] += 1

    assert_exhaustive(counts, gt_edges, crop)
    if sum(reach.values()) != gt_edges:
        raise LedgerRefusal(f"{crop}: the FACT-0370 reconciliation split does not sum to gt_edges")

    # --- the DIVISION denominator, reported separately and never added to the edge one -----
    div_total = div_recovered = div_partial = div_lost = 0
    for parent, kids in gt_children.items():
        if len(kids) < 2:
            continue
        div_total += 1
        pu = gt_row.get(parent)
        got = 0
        for kid in kids:
            kv = gt_row.get(kid)
            if pu is None or kv is None or pu not in to_final or kv not in to_final:
                continue
            if (to_final[pu], to_final[kv]) in final_edges:
                got += 1
        if got == len(kids):
            div_recovered += 1
        elif got > 0:
            div_partial += 1
        else:
            div_lost += 1
    if div_recovered + div_partial + div_lost != div_total:
        raise LedgerRefusal(f"{crop}: division ledger does not sum")

    return {
        "crop": crop, "gt_edges": gt_edges, **counts,
        "gt_nodes": int(len(gt_ids)),
        "gt_nodes_matched_to_peaks": len(to_peak),
        "gt_nodes_matched_to_final": len(to_final),
        "pred_nodes_preilp": int(pre_nodes.height),
        "pred_nodes_final": int(fin_nodes.height),
        "fact_0370_split": reach,
        "divisions": {"gt_division_events": div_total, "fully_recovered": div_recovered,
                      "partially_recovered": div_partial, "wholly_lost": div_lost},
    }


def load_final(fold: int) -> pl.DataFrame | None:
    cfg = SUBSTRATE[fold]
    if cfg["final_csv"]:
        return pl.read_csv(cfg["final_csv"])
    return None


def final_for_crop(fold: int, crop: str, cached: pl.DataFrame | None) -> pl.DataFrame:
    cfg = SUBSTRATE[fold]
    if cached is not None:
        return cached.filter(pl.col("dataset") == crop)
    p = Path(cfg["final_dir"]) / f"{crop}.parquet"
    if not p.is_file():
        raise LedgerRefusal(f"{crop}: no persisted final graph at {p}")
    return pl.read_parquet(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fold", type=int, required=True, choices=(0, 1))
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--out-dir", type=Path, default=Path("_evidence/assoc/pathology"))
    ap.add_argument("--max-crops", type=int)
    ap.add_argument("--crop-stride", type=int, default=1)
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = SUBSTRATE[args.fold]
    pre = pl.read_parquet(cfg["preilp"])
    cached_final = load_final(args.fold)
    crops = sorted(pre["dataset"].unique().to_list())[:: args.crop_stride]
    complete = args.max_crops is None and args.crop_stride == 1
    if args.max_crops:
        crops = crops[: args.max_crops]

    rows = []
    for i, crop in enumerate(crops, 1):
        gt = args.gt_dir / f"{crop}.geff"
        if not gt.exists():
            raise LedgerRefusal(f"missing GT for {crop}")
        rows.append(crop_ledger(crop, gt,
                                pre.filter(pl.col("dataset") == crop),
                                final_for_crop(args.fold, crop, cached_final)))
        print(f"  [{i}/{len(crops)}] {crop}", flush=True)

    tot = {k: int(sum(r[k] for r in rows)) for k in list(CATEGORIES) + ["recovered", "gt_edges"]}
    assert_exhaustive(tot, tot["gt_edges"], f"pooled fold {args.fold}")
    split = {k: int(sum(r["fact_0370_split"][k] for r in rows))
             for k in ("reachable", "unreachable_no_candidate", "undetected_endpoint")}
    div = {k: int(sum(r["divisions"][k] for r in rows))
           for k in ("gt_division_events", "fully_recovered", "partially_recovered",
                     "wholly_lost")}

    recon = {"published": FACT_0370_F1 if args.fold == 1 else None,
             "measured_here": {"gt_edges": tot["gt_edges"], **split},
             "complete_fold": complete}
    if args.fold == 0 and complete:
        recon["published"] = FACT_0376_F0
        detectable = tot["gt_edges"] - split["undetected_endpoint"]
        agree0 = (split["reachable"] == FACT_0376_F0["reached_deployed"]
                  and detectable == FACT_0376_F0["gt_detectable"])
        recon["gt_detectable_derived"] = detectable
        recon["agrees_with_fact_0376"] = bool(agree0)
        if not agree0:
            raise LedgerRefusal(
                "RECONCILIATION FAILED against FACT-0376 on the fold-0 deployed candidate export: "
                f"reachable {split['reachable']} against reached_deployed "
                f"{FACT_0376_F0['reached_deployed']}, and gt_edges minus undetected endpoints "
                f"{detectable} against gt_detectable {FACT_0376_F0['gt_detectable']}. A ledger "
                "that contradicts a published opportunity count is a defect in the ledger."
            )
    if args.fold == 1 and complete:
        agree = (split["reachable"] == FACT_0370_F1["reachable"]
                 and split["unreachable_no_candidate"] == FACT_0370_F1["unreachable_no_candidate"]
                 and split["undetected_endpoint"] == FACT_0370_F1["undetected_endpoint"]
                 and tot["gt_edges"] == FACT_0370_F1["gt_edges"])
        recon["agrees_with_fact_0370"] = bool(agree)
        if not agree:
            raise LedgerRefusal(
                "RECONCILIATION FAILED against FACT-0370 on the byte-identical pre-ILP export: "
                f"measured {split} over {tot['gt_edges']} GT edges against published "
                f"{FACT_0370_F1}. A ledger whose totals contradict FACT-0370 is a defect in the "
                "ledger until proven otherwise, so nothing is written."
            )

    payload = {
        "schema_version": 1,
        "heartbeat": "ASSOC_LOST_EDGE_LEDGER_COMPLETE",
        "packet": "PKT-0038", "lever": "LEVER-0041",
        "fold": args.fold, "embryo": cfg["embryo"], "n_crops": len(rows),
        "complete_fold": complete,
        "substrate": {k: cfg[k] for k in ("preilp", "final_dir", "final_csv")},
        "candidate_floor": "DEPLOYED - the pre-ILP export is the >0.5 softmax candidate set "
                           "(FACT-0369); 'offered' therefore means offered to the deployed "
                           "pipeline, not on the widened 0.1 surface of FACT-0382",
        "matching": {"rule": "official one-to-one bipartite per frame",
                     "max_distance_um": MAX_DISTANCE_UM, "scale_um": list(SCALE_UM),
                     "instrument": "detpeak_curve.match_one_to_one_pairs"},
        "totals": tot,
        "shares": {k: tot[k] / max(tot["gt_edges"], 1)
                   for k in list(CATEGORIES) + ["recovered"]},
        "recoverable_misses": {
            "definition": "categories 2-5: the GT edge is missing and at least one endpoint "
                          "survived detection, so a pipeline change could in principle recover it",
            "n": sum(tot[k] for k in CATEGORIES[1:]),
            "shares_of_recoverable": {
                k: tot[k] / max(sum(tot[c] for c in CATEGORIES[1:]), 1) for k in CATEGORIES[1:]},
        },
        "cat5_provisional": {
            "provisional": True,
            "observable_used": "the true edge was offered and the child ends with NO parent in "
                               "the final graph",
            "why_not_definitive": "separating ILP rejection from motion-relink overwrite needs the "
                                  "solver's edge list BEFORE motion_relink_edges replaces it "
                                  "(FACT-0364) and no committed artifact carries it; PKT-0040 owns "
                                  "that stage boundary. Its mass is NOT folded into category 4.",
        },
        "cat6_note": "structurally 0 on the EDGE denominator under the declared first-match-wins "
                     "ordering: an absent edge whose endpoints survive and which was offered is "
                     "already category 4 or 5. The meaningful division measurement is reported on "
                     "its own denominator below and the two are never added (FACT-0371).",
        "divisions": div,
        "fact_0370_reconciliation": recon,
        "per_crop": rows,
        "promotable": False,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"lost_edge_ledger_f{args.fold}.json"
    p.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    print(f"\nLOST-EDGE LEDGER  fold {args.fold} ({cfg['embryo']})  crops {len(rows)}  "
          f"GT edges {tot['gt_edges']:,}")
    for k in CATEGORIES:
        print(f"  {k:46s} {tot[k]:8,d}  {tot[k] / max(tot['gt_edges'], 1):7.4f}")
    print(f"  {'recovered':46s} {tot['recovered']:8,d}  "
          f"{tot['recovered'] / max(tot['gt_edges'], 1):7.4f}")
    rec = payload["recoverable_misses"]
    print(f"\n  recoverable misses (categories 2-5): {rec['n']:,}")
    for k, v in rec["shares_of_recoverable"].items():
        print(f"    {k:46s} {v:7.4f}")
    print(f"\n  divisions {json.dumps(div)}")
    print(f"  FACT-0370 reconciliation: {json.dumps(recon.get('measured_here'))}"
          f"  agrees={recon.get('agrees_with_fact_0370')}")
    print(f"\nASSOC_LOST_EDGE_LEDGER_COMPLETE -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
