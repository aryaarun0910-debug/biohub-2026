r"""The PKT-0046 panel: read the reversed-ladder crop payloads and report through assoc_report.

THREE THINGS FAIL CLOSED BEFORE ANY ARM IS READ, in this order:

1. COMPLETENESS. Every crop of the fold must have a payload with the right heartbeat. A silently
   missing crop is invisible in a panel - the ``all_passed`` trap ``FACT-0417`` records.
2. THE INSTRUMENT'S OWN CALIBRATION. The ``calibration_perfect`` arm - GT nodes with GT edges -
   must score raw edge Jaccard 1.0000, which is what ``FACT-0368``'s harness scores it at. If it
   does not, this harness is wrong and no other arm means anything.
3. THE CONTROL. It must reproduce the recorded deployed numbers of the anchor artifact
   (``_evidence/ceiling/ladder_f{0,1}.json``, the instrument output behind ``FACT-0368`` and
   ``FACT-0371``) on the SAME crops. A control that does not reproduce HALTS the packet.

Every per-arm summary is built by ``assoc_report.build_report`` and nothing else, so no channel
can be quietly selected after the fact.

THE COUNT-FAIR READING, AND WHY IT IS NOT A RIVAL SUMMARY
--------------------------------------------------------
``perfect_membership_full`` deletes ~99% of predicted nodes, so its ADJUSTED score collects an
under-production bonus (``FACT-0191``) that no method can earn. The honest headline is therefore
the count-fair one, and it is assembled from the report's OWN channels using the identity
``assoc_report`` documents at module level:

    delta score = delta adj-edge-J + DIV_WEIGHT * delta division-J

with ``delta adj-edge-J`` taken as the report's ``count_adjustment.raw_channel`` - association
quality re-scored at the CONTROL's node-count multiplier. That is exactly the construction
``FACT-0371`` used for its count-fair headroom. It selects no channel the report did not produce.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import assoc_report  # noqa: E402
from revladder import ARMS, FOLDS, HEARTBEAT  # noqa: E402

CALIBRATION_ARM = "calibration_perfect"
CONTROL_ARM = "control"
TREATED = tuple(a for a in ARMS if a != CONTROL_ARM)
ANCHOR_KEYS = ("edge_jaccard", "adj_edge_jaccard", "score", "node_recall",
               "division_tp", "division_fp", "division_fn", "num_pred_nodes")


def load_fold(fold: int, payload_dir: Path, gt_dir: Path, only: set[str] | None):
    from revladder_batch import crops_for

    names = crops_for(fold, gt_dir)
    if only is not None:
        names = [c for c in names if c in only]
        missing_requested = only - set(names)
        if missing_requested:
            raise SystemExit(f"anchor names not in fold {fold}: {sorted(missing_requested)[:5]}")
    rows: dict[str, list[dict]] = {a: [] for a in ARMS}
    diag: list[dict] = []
    missing, bad = [], []
    for crop in names:
        p = payload_dir / f"f{fold}_{crop}.json"
        if not p.is_file():
            missing.append(crop)
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:                       # noqa: BLE001
            bad.append(f"{crop}: unreadable ({exc})")
            continue
        if d.get("heartbeat") != HEARTBEAT:
            bad.append(f"{crop}: heartbeat {d.get('heartbeat')!r}")
            continue
        for a in ARMS:
            if a not in d["arms"]:
                bad.append(f"{crop}: arm {a} absent")
                break
            rows[a].append(d["arms"][a])
        diag.append(d["arms"]["_diag"])
    return names, rows, diag, missing, bad


def check_control(fold: int, control_rows: list[dict], anchor: dict, summarise) -> dict:
    got = summarise(control_rows)
    got = {**got, "num_pred_nodes": int(sum(r["num_pred_nodes"] for r in control_rows))}
    want = anchor["ladder"]["deployed"]
    deltas, failures = {}, []
    for k in ANCHOR_KEYS:
        if k not in want:
            continue
        g, w = float(got[k]), float(want[k])
        deltas[k] = {"recorded": w, "reproduced": g, "delta": g - w}
        tol = 0.0 if k in ("division_tp", "division_fp", "division_fn", "num_pred_nodes") else 1e-9
        if abs(g - w) > tol:
            failures.append(f"{k}: recorded {w!r}, reproduced {g!r}")
    return {
        "anchor_artifact": anchor.get("_path"),
        "anchor_crops": anchor["n_crops"],
        "compared_crops": len(control_rows),
        "reproduces": not failures,
        "failures": failures,
        "fields": deltas,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fold", type=int, required=True, choices=(0, 1))
    ap.add_argument("--payload-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--anchor", type=Path, required=True,
                    help="_evidence/ceiling/ladder_f{0,1}.json - the FACT-0368/0371 instrument output")
    ap.add_argument("--anchor-subset", action="store_true",
                    help="compare the control only on the crops the anchor actually scored "
                         "(fold 1's anchor is a stride-4 sample of 32 of 128)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from tracking_cellmot.metrics import SCORE_DIVISION_WEIGHT, summarise

    anchor = json.loads(args.anchor.read_text(encoding="utf-8"))
    anchor["_path"] = str(args.anchor)
    anchor_names = {c["crop"] for c in anchor["per_crop"]}

    names, rows, diag, missing, bad = load_fold(args.fold, args.payload_dir, args.gt_dir, None)
    blockers = []
    if missing:
        blockers.append(f"{len(missing)} crop payloads missing: {missing[:5]}")
    if bad:
        blockers.append(f"{len(bad)} malformed payloads: {bad[:5]}")
    if blockers:
        print("PANEL_REFUSED\n  " + "\n  ".join(blockers))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"heartbeat": "REVLADDER_PANEL_REFUSED", "blockers": blockers}, indent=2),
            encoding="utf-8")
        return 2

    # ---- GATE 2: the instrument's own calibration -------------------------------------------
    cal = summarise(rows[CALIBRATION_ARM])
    cal_raw = float(cal["edge_jaccard"])
    cal_ok = abs(cal_raw - 1.0) < 1e-9
    per_crop_cal_bad = [r["dataset"] for r in rows[CALIBRATION_ARM]
                        if abs(float(r["edge_jaccard"]) - 1.0) > 1e-9]

    # ---- GATE 3: the control reproduces the recorded numbers ---------------------------------
    if args.anchor_subset:
        idx = [i for i, c in enumerate(names) if c in anchor_names]
        ctrl_for_check = [rows[CONTROL_ARM][i] for i in idx]
    else:
        ctrl_for_check = rows[CONTROL_ARM]
    control_check = check_control(args.fold, ctrl_for_check, anchor, summarise)

    reports = {}
    for arm in TREATED:
        reports[arm] = assoc_report.build_report(
            model=f"revladder:{arm}",
            fold=args.fold,
            control=rows[CONTROL_ARM],
            candidate=rows[arm],
            summarise=summarise,
            conversions=None,
            notes=(
                "PKT-0046 reversed ceiling ladder. This arm is an ORACLE and is NOT promotable; "
                "the verdict blockers below are expected and correct. The control is the P28 "
                "champion export, bound by EXACT REPRODUCTION of the FACT-0368/FACT-0371 anchor "
                "artifact rather than by the deployed_controls.json manifest, which carries no "
                "entry for this substrate."
            ),
        )
        ch = reports[arm]["channels"]
        # Derived from the report's OWN channels via the identity assoc_report documents.
        reports[arm]["count_fair"] = {
            "basis": "assoc_report channels count_adjustment.raw_channel and "
                     "division_counts.division_jaccard_delta; identity is assoc_report's own "
                     "delta_score = delta_adj + DIV_WEIGHT * delta_division_J",
            "delta_adj_count_fair": ch["count_adjustment"]["raw_channel"],
            "delta_division_term": SCORE_DIVISION_WEIGHT * ch["division_counts"]["division_jaccard_delta"],
            "delta_score_count_fair": (
                ch["count_adjustment"]["raw_channel"]
                + SCORE_DIVISION_WEIGHT * ch["division_counts"]["division_jaccard_delta"]
            ),
        }

    summaries = {a: summarise(rows[a]) for a in ARMS}
    for a in ARMS:
        summaries[a] = {**summaries[a],
                        "num_pred_nodes": int(sum(r["num_pred_nodes"] for r in rows[a]))}

    payload = {
        "schema_version": 1,
        "heartbeat": "REVLADDER_PANEL_COMPLETE",
        "fold": args.fold,
        "n_crops": len(names),
        "calibration_gate": {
            "arm": CALIBRATION_ARM,
            "raw_edge_jaccard": cal_raw,
            "expected": 1.0,
            "passes": bool(cal_ok),
            "crops_off_one": per_crop_cal_bad,
        },
        "control_check": control_check,
        "arm_summaries": summaries,
        "reports": reports,
        "diagnostics": {
            # WHY N_est IS RECORDED HERE. The `perfect_membership_full` arm's whole gain comes
            # from DELETING unmatched predicted nodes. That is only "node correction" if those
            # nodes are spurious detections. If the pipeline already emits FEWER nodes than the
            # crop is estimated to contain, they are overwhelmingly real cells the annotation
            # does not label, and no detector can remove them - it would emit MORE. So this ratio
            # decides how that arm may be read, and it is measured here rather than cited.
            "control_n_pred": int(sum(r["num_pred_nodes"] for r in rows[CONTROL_ARM])),
            "control_n_est": float(sum(
                r["num_pred_nodes"] / (1.0 + r["total_node_ratio"])
                for r in rows[CONTROL_ARM] if r["total_node_ratio"] == r["total_node_ratio"]
            )),
            "n_gt_nodes": int(sum(d["n_gt_nodes"] for d in diag)),
            "n_gt_edges": int(sum(d["n_gt_edges"] for d in diag)),
            "n_deployed_nodes": int(sum(d["n_deployed_nodes"] for d in diag)),
            "n_deployed_edges": int(sum(d["n_deployed_edges"] for d in diag)),
            "n_matched": int(sum(d["n_matched"] for d in diag)),
            "n_missing_gt": int(sum(d["n_missing_gt"] for d in diag)),
            "n_unmatched_deployed": int(sum(d["n_unmatched_deployed"] for d in diag)),
            "n_edges_surviving_full": int(sum(d["n_edges_surviving_full"] for d in diag)),
        },
        # The composition of the deployed false-positive edge set, which falls straight out of the
        # arms: `perfect_membership_full` keeps EVERY edge whose endpoints both match an annotated
        # cell, so whatever FP survives there is a wrong link BETWEEN TWO ANNOTATED CELLS, and the
        # rest are links from an annotated cell into an unannotated one.
        "false_positive_composition": {
            "control_edge_fp": reports["perfect_membership_full"]["channels"]["final_graph_edges"]["control"]["edge_fp"],
            "fp_between_two_annotated_cells": reports["perfect_membership_full"]["channels"]["final_graph_edges"]["candidate"]["edge_fp"],
            "fp_into_an_unannotated_node": (
                reports["perfect_membership_full"]["channels"]["final_graph_edges"]["control"]["edge_fp"]
                - reports["perfect_membership_full"]["channels"]["final_graph_edges"]["candidate"]["edge_fp"]
            ),
        },
        "scope_limit": (
            "Bounds NODE CORRECTION UNDER THE FROZEN CONSUMER only. A real detector also changes "
            "localisation, node features, candidate availability and association decisions, so a "
            "miss here does NOT kill detector-derived representations or candidate-generation "
            "effects."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    print(f"\nREVLADDER_PANEL fold={args.fold} crops={len(names)}")
    print(f"  CALIBRATION  {CALIBRATION_ARM} rawJ={cal_raw:.6f} "
          f"-> {'PASS' if cal_ok else 'FAIL - harness suspect, do not read other arms'}")
    print(f"  CONTROL      reproduces anchor on {control_check['compared_crops']} crops -> "
          f"{'PASS' if control_check['reproduces'] else 'FAIL - HALT'}")
    for f in control_check["failures"]:
        print(f"                 {f}")
    print(f"\n  {'arm':<30} {'rawJ':>9} {'adjJ':>9} {'score':>9} {'divJ':>8} "
          f"{'divTP/FP/FN':>14} {'recall':>8} {'nodes':>12}")
    for a in ARMS:
        s = summaries[a]
        print(f"  {a:<30} {s['edge_jaccard']:9.6f} {s['adj_edge_jaccard']:9.6f} "
              f"{s['score']:9.6f} {s['division_jaccard']:8.4f} "
              f"{int(s['division_tp']):4d}/{int(s['division_fp']):4d}/{int(s['division_fn']):4d} "
              f"{s['node_recall']:8.5f} {s['num_pred_nodes']:12,}")
    print(f"\n  {'arm vs control':<30} {'d_rawJ':>10} {'d_adj':>10} {'count_ch':>10} "
          f"{'d_score':>10} {'COUNT-FAIR d_score':>20}")
    for a in TREATED:
        ch = reports[a]["channels"]
        print(f"  {a:<30} {ch['edge_jaccard_raw']['delta']:+10.6f} "
              f"{ch['count_adjustment']['delta_adj']:+10.6f} "
              f"{ch['count_adjustment']['count_channel']:+10.6f} "
              f"{ch['score']['delta']:+10.6f} "
              f"{reports[a]['count_fair']['delta_score_count_fair']:+20.6f}")
    return 0 if (cal_ok and control_check["reproduces"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
