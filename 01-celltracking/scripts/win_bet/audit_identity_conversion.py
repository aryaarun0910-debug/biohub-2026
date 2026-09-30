r"""What is FACT-0376's NET +28 true edges actually made of? (PKT-0033 target 5, READ-ONLY)

THE QUESTION
------------
``FACT-0376`` reports ``delta_edge_tp: 28`` for the LEVER-0037 arm over 71 fold-0 crops. It is a
NET count. Its own ``validity_reason`` says the alternatives cannot be separated: "28 net could be
28 gained and 0 lost, or hundreds gained and nearly as many displaced". This module separates them
from persisted FINAL GRAPHS, in GT-EDGE SPACE, which is the only space where identity is comparable
across the two arms - the arms have different node counts, so predicted ``node_id`` is not.

WHAT IT COMPUTES, AND WHY EACH PART IS RECOVERABLE
--------------------------------------------------
For each arm the official scorer is run (``div_reach_steal.score_crop``'s exact path), which writes
``match_node_id`` onto every predicted node and ``matched_edge_mask`` onto every predicted edge.
A true positive is therefore identifiable as the GT PAIR ``(match[src], match[tgt])``:

  gained  GT edges recovered by the candidate arm and not by the control
  lost    GT edges the control recovered and the candidate lost - the "displaced" quantity
  churn   gained + lost, the number a NET figure hides
  net     gained - lost, which MUST equal the retained per-crop ``delta_edge_tp``. That equality
          is the calibration this module refuses to report without (AGENTS.md: calibrate any
          derived quantity against an independently known value).

Each LOST edge is then attributed, because +156,408 nodes (FACT-0376) can break a GT node's
one-to-one match without any parent choice changing:

  node_match_churn   an endpoint is no longer matched to its GT node at all
  wrong_parent       both endpoints still matched, and the GT target's matched node has a
                     DIFFERENT predicted parent - a genuine association failure
  no_parent          both endpoints still matched and the target has no predicted parent

WHAT IS NOT RECOVERABLE FROM THESE ARTIFACTS - STATED SO NOBODY INFERS IT
-------------------------------------------------------------------------
``p28_full_chain_replay.py --save-graph`` persists ONLY the FINAL frame. The ILP's raw solved edge
list and the ``motion_relink_edges`` output are discarded, and only their COUNTS survive in the
replay's ``comparison`` block. So an edge lost at the ILP stage cannot be told from one overwritten
by the relink (FACT-0364: the relink replaces the whole edge list). Stage attribution needs both
arms re-run with those intermediates persisted; it cannot be added to a finished arm.

READ-ONLY. This module opens the replay's outputs and writes nothing into them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _tp_pairs_and_matching(frame: pl.DataFrame, gt_geff: Path, ea, sc):
    """Run the official scorer on one arm and return its identity-level state.

    Returns (tp_gt_pairs, gt_to_pred, pred_parent, counts) where
      tp_gt_pairs  set of (gt_source, gt_target) recovered as true positives
      gt_to_pred   gt node id -> predicted node id it matched (this arm's matching)
      pred_parent  predicted node id -> its predicted parent id (in-degree is <= 1 by contract)
    """
    import tracksdata as td
    K = td.DEFAULT_ATTR_KEYS
    _td, MAX_DISTANCE, estimated_nodes, load_graph, _dm, evaluate, node_recall, per_sample, _s = sc
    from div_reach_steal import SCALE

    g, _ = ea.build_graph(frame)
    gt = load_graph(gt_geff)
    er = evaluate(g, gt, scale=SCALE, max_distance=MAX_DISTANCE)   # writes matching onto g
    rec = node_recall(g, gt) if g.num_edges() and g.num_nodes() else 0.0
    row = per_sample(er, estimated_nodes(gt_geff), rec)

    na = g.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID]).to_pandas()
    match = {int(n): int(m) for n, m in zip(na[K.NODE_ID], na[K.MATCHED_NODE_ID])}
    gt_to_pred = {m: n for n, m in match.items() if m != -1}

    tp: set[tuple[int, int]] = set()
    pred_parent: dict[int, int] = {}
    if g.num_edges():
        eattrs = g.edge_attrs(
            attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET, K.MATCHED_EDGE_MASK]).to_pandas()
        for s, t, m in zip(eattrs[K.EDGE_SOURCE], eattrs[K.EDGE_TARGET],
                           eattrs[K.MATCHED_EDGE_MASK]):
            pred_parent[int(t)] = int(s)
            if bool(m):
                ms, mt = match.get(int(s), -1), match.get(int(t), -1)
                if ms != -1 and mt != -1:
                    tp.add((ms, mt))
    return tp, gt_to_pred, pred_parent, {
        "edge_tp": int(er.edge_tp), "edge_fp": int(er.edge_fp), "edge_fn": int(er.edge_fn),
        "num_pred_nodes": int(er.num_pred_nodes),
        "edge_jaccard": float(row["edge_jaccard"]),
    }


def analyse(crop: str, ctl_frame: pl.DataFrame, cand_path: Path, gt_geff: Path, ea, sc) -> dict:
    ctl_tp, ctl_map, _ctl_par, ctl_counts = _tp_pairs_and_matching(
        ctl_frame, gt_geff, ea, sc)
    cand_tp, cand_map, cand_par, cand_counts = _tp_pairs_and_matching(
        pl.read_parquet(cand_path), gt_geff, ea, sc)

    gained = sorted(cand_tp - ctl_tp)
    lost = sorted(ctl_tp - cand_tp)

    # attribute every LOST GT edge
    node_churn = wrong_parent = no_parent = 0
    for gs, gt_t in lost:
        ps, pt = cand_map.get(gs), cand_map.get(gt_t)
        if ps is None or pt is None:
            node_churn += 1
        elif pt in cand_par:
            wrong_parent += 1 if cand_par[pt] != ps else 0
            # parent equal but not a TP would mean the mask disagreed; count it as association
            no_parent += 0 if cand_par[pt] != ps else 1
        else:
            no_parent += 1

    # the calibration: identity net MUST reproduce the count-level delta
    net_identity = len(gained) - len(lost)
    net_counts = cand_counts["edge_tp"] - ctl_counts["edge_tp"]
    return {
        "crop": crop,
        "control": ctl_counts, "candidate": cand_counts,
        "gained": len(gained), "lost": len(lost),
        "churn": len(gained) + len(lost),
        "net_identity": net_identity, "net_from_counts": net_counts,
        "calibrated": net_identity == net_counts,
        "lost_attribution": {
            "node_match_churn": node_churn,
            "wrong_parent": wrong_parent,
            "no_predicted_parent_or_mask": no_parent,
        },
        "gt_nodes_matched_control": len(ctl_map),
        "gt_nodes_matched_candidate": len(cand_map),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # WHICH CONTROL. lever37_conversion.py scores its control from the CHAMPION SUBMISSION CSV
    # (its --control-csv), not from the replayed control arm. Measured on 2026-08-29: on 3 of the
    # first 10 crops the replayed control scores +9 edge TP against the champion despite
    # `control_exact: true`, because that flag compares 15 STAGE COUNTERS and not edge identity.
    # So a decomposition of FACT-0376's own 28 MUST use the champion CSV, or it decomposes a
    # different number. `--control-csv` is therefore the default and `--control-dir` the variant.
    ap.add_argument("--control-csv", type=Path,
                    default=Path("C:/temp/p30_f0/loeo_split0_champion.csv.gz"))
    ap.add_argument("--control-dir", type=Path, default=Path("C:/temp/p30_f0/graphs_ctl"))
    ap.add_argument("--control-from-replay", action="store_true",
                    help="use the replayed control graphs instead of the champion submission")
    ap.add_argument("--candidate-dir", type=Path, default=Path("C:/temp/p30_f0/graphs_l37"))
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--conversion-json", type=Path,
                    default=ROOT / "_evidence" / "ecb" / "conversion_l37.json")
    ap.add_argument("--out", type=Path,
                    default=Path("C:/temp/audit_assoc/identity_conversion.json"))
    args = ap.parse_args()

    from div_reach_steal import _ea_atlas, _scorer
    ea, sc = _ea_atlas(), _scorer()

    cand = {p.stem: p for p in args.candidate_dir.glob("*.parquet")}
    if args.control_from_replay:
        ctl_paths = {p.stem: p for p in args.control_dir.glob("*.parquet")}
        ctl_frames = {c: pl.read_parquet(p) for c, p in ctl_paths.items()}
        control_source = f"replayed control graphs {args.control_dir}"
    else:
        from biotrack.submission import read_submission
        sub = read_submission(args.control_csv)
        frame = sub if isinstance(sub, pl.DataFrame) else pl.DataFrame(sub)
        ctl_frames = {c: frame.filter(pl.col("dataset") == c)
                      for c in frame["dataset"].unique().to_list()}
        control_source = f"champion submission {args.control_csv} (the control FACT-0376 used)"
    shared = sorted(set(ctl_frames) & set(cand))
    if not shared:
        raise SystemExit(
            f"no crop has BOTH arms ({len(ctl_frames)} control, {len(cand)} candidate) - "
            "refusing to report a decomposition from one arm")
    ctl = ctl_frames

    retained = {}
    if args.conversion_json.exists():
        retained = {r["crop"]: r for r in
                    json.loads(args.conversion_json.read_text(encoding="utf-8"))["per_crop"]}

    rows, uncalibrated = [], []
    for i, crop in enumerate(shared, 1):
        gt_geff = args.gt_dir / f"{crop}.geff"
        if not gt_geff.exists():
            raise SystemExit(f"{crop}: ground truth missing - refusing to skip it silently")
        r = analyse(crop, ctl[crop], cand[crop], gt_geff, ea, sc)
        r["retained_delta_edge_tp"] = (
            retained[crop]["delta_edge_tp"] if crop in retained else None)
        r["matches_retained_replay"] = (
            None if r["retained_delta_edge_tp"] is None
            else r["net_identity"] == int(r["retained_delta_edge_tp"]))
        if not r["calibrated"]:
            uncalibrated.append(crop)
        rows.append(r)
        print(f"  [{i}/{len(shared)}] {crop} gained={r['gained']} lost={r['lost']} "
              f"net={r['net_identity']} (retained {r['retained_delta_edge_tp']}) "
              f"lost_attr={r['lost_attribution']}", flush=True)

    tot = {k: sum(r[k] for r in rows) for k in ("gained", "lost", "churn", "net_identity")}
    attr = {k: sum(r["lost_attribution"][k] for r in rows)
            for k in ("node_match_churn", "wrong_parent", "no_predicted_parent_or_mask")}
    disagree = [r["crop"] for r in rows if r["matches_retained_replay"] is False]
    report = {
        "schema_version": 1,
        "PARTIAL": len(shared) < len(ctl),
        "crops_analysed": len(shared),
        "control_source": control_source,
        "control_arm_crops_available": len(ctl),
        "candidate_arm_crops_available": len(cand),
        "sampling": ("NAME ORDER, NOT A RANDOM SAMPLE - these crops are whatever the in-flight "
                     "replay has finished; no fold-0 figure may be read off this"),
        "totals": tot, "lost_attribution": attr,
        "crops_where_identity_net_disagrees_with_counts": uncalibrated,
        "crops_where_identity_net_disagrees_with_retained_replay": disagree,
        "per_crop": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    print(f"\n  crops {len(shared)} of {len(ctl)} control / {len(cand)} candidate persisted"
          f"\n  GAINED {tot['gained']}   LOST {tot['lost']}   CHURN {tot['churn']}"
          f"   NET {tot['net_identity']}"
          f"\n  lost attributed: {attr}")
    if uncalibrated:
        print("\nCALIBRATION FAILED on:", uncalibrated,
              "\n  identity net does not reproduce the scorer's own edge_tp delta - the "
              "heartbeat is withheld.", flush=True)
        return 1
    if disagree:
        print("\nRETAINED-REPLAY DISAGREEMENT on:", disagree,
              "\n  these graphs do not reproduce the recorded per-crop delta; the heartbeat is "
              "withheld because the two runs are then not the same measurement.", flush=True)
        return 1
    print(f"\nIDENTITY_CONVERSION_AUDIT_COMPLETE crops={len(shared)} out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
