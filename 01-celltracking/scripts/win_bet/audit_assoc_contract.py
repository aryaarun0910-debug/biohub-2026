r"""ADVERSARIAL AUDIT of the learned-association cycle's own instruments (PKT-0033).

This is not a unit test and it asserts nothing about science. It EXERCISES named failure modes
of `assoc_parent_dataset.py` and `assoc_report.py` and reports what each one actually does, so
that a divergence between a docstring and the code cannot survive by being read rather than run.

FAIL-CLOSED CONTRACT
--------------------
Every probe records ``executed: true`` only after its body has run to completion. The heartbeat
``ASSOC_CONTRACT_AUDIT_COMPLETE`` is printed ONLY when every declared probe reports executed. A
probe that raises, or a probe name that is declared and never populated, suppresses the heartbeat
and sets exit code 1. The ABSENCE of the heartbeat is the alarm - a silent no-op cannot pass here,
which is the rule earned by the tmp-write test that matched a PATTERN in generated source instead
of executing it (FACT-0373 note).

PROBES
------
  P1 tie_break_rule       - the frozen contract claims ties break by LOWER SOURCE INDEX. Run it.
  P2 two_predecessor      - the contract claims two-predecessor GT cells are DROPPED. Run it, and
                            census the real GT for whether the case is live or latent.
  P3 fold_degeneracy      - recompute the contested/single split for both folds without calling
                            evaluate(), and recompute it at the DEPLOYED floor.
  P4 feature_label_wall   - no feature may be derived from GT. Checked structurally: every feature
                            must be produced above the line where the GT graph is loaded.
  P5 conversions_guard    - can parent_conversions' target-set guard be satisfied by a SUBSET?
  P6 net_only_report      - can build_report's verdict be satisfied by a bare net figure?
  P7 bootstrap_direction  - is `excludes_zero` directional, or does a significant LOSS satisfy it?
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DECLARED = [
    "tie_break_rule", "two_predecessor", "fold_degeneracy",
    "feature_label_wall", "conversions_guard", "net_only_report", "bootstrap_direction",
]


# ---------------------------------------------------------------------------------------------
def p_tie_break_rule() -> dict:
    """The docstring says ties break by lower source index. Does the code do that?"""
    from assoc_parent_dataset import evaluate

    base = [
        {"crop": "c", "target": 10, "source": 99, "score": 0.5, "is_true_parent": 1,
         "true_parent_is_candidate": 1, "target_has_true_parent": 1},
        {"crop": "c", "target": 10, "source": 7, "score": 0.5, "is_true_parent": 0,
         "true_parent_is_candidate": 1, "target_has_true_parent": 1},
    ]
    forward = evaluate(pl.DataFrame(base), "score")["parent_top1"]
    reverse = evaluate(pl.DataFrame(list(reversed(base))), "score")["parent_top1"]
    # If the rule were "lower source index wins", the true parent (source 99) always loses => 0.0.
    no_source = evaluate(pl.DataFrame(base).drop("source"), "score")["parent_top1"]
    return {
        "executed": True,
        "top1_true_parent_row_first": forward,
        "top1_true_parent_row_second": reverse,
        "documented_rule_lower_source_index_would_give": 0.0,
        "runs_without_a_source_column": True,
        "top1_without_source_column": no_source,
        "order_dependent": forward != reverse,
        "verdict": ("DIVERGES: tie-break is by ROW POSITION (np.argsort stable), not source index; "
                    "evaluate() never reads the source column"
                    if forward != reverse else "matches the documented rule"),
    }


def p_two_predecessor(gt_dir: Path, limit: int | None) -> dict:
    """The docstring says a GT cell with two predecessors would be DROPPED. What does the code do?

    Two parts. (a) BEHAVIOUR: reproduce the exact predecessor construction on a synthetic
    two-predecessor edge list. (b) CENSUS: is the case live in the real GT, or latent?
    """
    import pandas as pd

    # (a) the construction at assoc_parent_dataset.py:123, reproduced verbatim
    gte = pd.DataFrame({"source_id": [1, 2], "target_id": [9, 9]})
    predecessor = {int(t): int(s) for s, t in zip(gte["source_id"], gte["target_id"])}
    behaviour = {
        "edges_in": [[1, 9], [2, 9]],
        "predecessor_map": predecessor,
        "target_9_dropped": 9 not in predecessor,
        "target_9_resolved_to": predecessor.get(9),
        "rule_actually_applied": "LAST EDGE WINS (dict comprehension overwrite)",
    }

    # (b) census over the real ground truth
    from biotrack.metric import load_graph
    files = sorted(gt_dir.glob("*.geff"))
    if limit:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"no .geff under {gt_dir} - refusing to report an empty census")
    multi_pred = multi_child = 0
    crops_with_multi_pred = []
    total_nodes = total_edges = 0
    for f in files:
        g = load_graph(f)
        e = g.edge_attrs().to_pandas()
        total_nodes += len(g.node_attrs().to_pandas())
        total_edges += len(e)
        cin = collections.Counter(e["target_id"].tolist())
        cout = collections.Counter(e["source_id"].tolist())
        n_mp = sum(1 for v in cin.values() if v > 1)
        multi_pred += n_mp
        multi_child += sum(1 for v in cout.values() if v > 2)
        if n_mp:
            crops_with_multi_pred.append(f.stem)
    return {
        "executed": True,
        "behaviour": behaviour,
        "census": {
            "geff_files_scanned": len(files),
            "gt_nodes": total_nodes,
            "gt_edges": total_edges,
            "gt_nodes_with_two_or_more_predecessors": multi_pred,
            "crops_affected": crops_with_multi_pred,
            "gt_nodes_with_three_or_more_children": multi_child,
        },
        "verdict": ("DIVERGES but LATENT: the code silently keeps the LAST predecessor rather than "
                    "dropping the cell; the GT contains no such cell, so nothing is mis-labelled today"
                    if multi_pred == 0 else
                    "DIVERGES AND LIVE: the GT contains multi-predecessor cells and they are guessed, "
                    "not dropped"),
    }


def p_fold_degeneracy(f0: Path, f1: Path, deployed_floor: float) -> dict:
    """Recompute the contested/single split WITHOUT calling evaluate(), on both folds, and repeat
    it at the DEPLOYED candidate floor - the surface the pipeline actually consumes."""
    out = {"executed": True, "folds": {}, "deployed_floor": deployed_floor}
    for name, path in (("fold0", f0), ("fold1", f1)):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - refusing to report a fold as clean unseen")
        t = pl.read_parquet(path)
        d = t.filter(pl.col("true_parent_is_candidate") == 1)
        g = (d.group_by(["crop", "target"])
              .agg(n=pl.len(),
                   top=pl.col("is_true_parent").sort_by(pl.col("prob"), descending=True).first(),
                   ties_at_top=(pl.col("prob") == pl.col("prob").max()).sum()))
        c = g.filter(pl.col("n") > 1)
        # the same population restricted to the DEPLOYED rule
        dep = (d.filter(pl.col("prob") > deployed_floor)
                .group_by(["crop", "target"]).agg(n=pl.len()))
        out["folds"][name] = {
            "rows": t.height,
            "decidable_targets": g.height,
            "max_candidates_per_target": int(g["n"].max()),
            "contested": int(c.height),
            "contested_top1": float(c["top"].sum() / c.height) if c.height else None,
            "contested_errors": int(c.height - c["top"].sum()) if c.height else 0,
            "single": int(g.height - c.height),
            "single_top1_must_be_one": float(
                (g.filter(pl.col("n") == 1)["top"].sum()) / max(g.height - c.height, 1)),
            "overall_top1": float(g["top"].sum() / g.height),
            "targets_with_a_tie_at_top_prob": int(g.filter(pl.col("ties_at_top") > 1).height),
            "at_deployed_floor": {
                "targets": int(dep.height),
                "max_candidates_per_target": int(dep["n"].max()) if dep.height else 0,
                "contested": int(dep.filter(pl.col("n") > 1).height),
            },
            "degenerate_for_ranking": bool(c.height == 0),
            "degenerate_at_deployed_floor": bool(dep.filter(pl.col("n") > 1).height == 0),
        }
    return out


def p_feature_label_wall(src: Path) -> dict:
    """No feature may see the label. Proven STRUCTURALLY: every feature must be computed above the
    line where the GT graph enters build_crop, and no FEATURES name may be a label column."""
    from assoc_parent_dataset import FEATURES
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_crop")
    gt_line = min(
        (n.lineno for n in ast.walk(fn)
         if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "load_graph"),
        default=None)
    if gt_line is None:
        raise RuntimeError("load_graph call not found in build_crop - the wall cannot be located")
    # every producer of a feature quantity, and the line it is assigned on
    producers: dict[str, int] = {}
    for n in ast.walk(fn):
        # AnnAssign (`n_cand: dict[int, int] = {}`) is NOT an ast.Assign - missing it once made
        # this probe report a hardcoded CLEAN verdict beside a False check. Both forms are read.
        targets = (n.targets if isinstance(n, ast.Assign)
                   else [n.target] if isinstance(n, ast.AnnAssign) else [])
        for tgt in targets:
            for name in ([tgt.id] if isinstance(tgt, ast.Name) else
                         [e.id for e in getattr(tgt, "elts", []) if isinstance(e, ast.Name)]):
                producers.setdefault(name, n.lineno)
    watched = ["best", "ranks", "n_cand", "out_deg", "prob", "node_zyx"]
    label_names = {"is_true_parent", "target_has_true_parent",
                   "true_parent_is_candidate", "target_matched_gt"}
    lines = {k: producers.get(k) for k in watched}
    unlocated = [k for k, v in lines.items() if v is None]
    if unlocated:
        raise RuntimeError(
            f"could not locate the producer of {unlocated} in build_crop - the wall is unproven")
    above = all(v < gt_line for v in lines.values())
    overlap = sorted(set(FEATURES) & label_names)
    clean = above and not overlap
    return {
        "executed": True,
        "gt_enters_build_crop_at_line": gt_line,
        "feature_producer_lines": lines,
        "all_feature_producers_above_gt_line": above,
        "features_declared": FEATURES,
        "features_that_are_label_columns": overlap,
        # derived from the checks, never asserted alongside them
        "verdict": ("CLEAN: every feature quantity is produced strictly above the GT load and no "
                    "feature is a label column" if clean else
                    f"LEAK RISK: above_gt_line={above} label_columns_in_features={overlap}"),
    }


def p_conversions_guard() -> dict:
    """Can the target-set guard be satisfied by SUBSETTING both maps to the targets a model won?"""
    from assoc_report import parent_conversions
    before = {("c", i): int(i % 3 != 0) for i in range(300)}
    after = dict(before)
    for i in range(0, 300, 3):           # 100 gains
        after[("c", i)] = 1
    for i in range(1, 300, 3):           # 100 losses
        after[("c", i)] = 0
    honest = parent_conversions(before, after)

    # the bypass: restrict BOTH maps to the targets that improved, then call the same function
    won = [k for k in before if not before[k] and after[k]]
    sub_b = {k: before[k] for k in won}
    sub_a = {k: after[k] for k in won}
    raised = None
    try:
        cherry = parent_conversions(sub_b, sub_a)
    except ValueError as exc:                       # pragma: no cover - would be the good outcome
        cherry, raised = None, str(exc)

    # and the degenerate case: two empty maps
    empty_raised = None
    try:
        empty = parent_conversions({}, {})
    except ValueError as exc:
        empty, empty_raised = None, str(exc)
    return {
        "executed": True,
        "honest_full_surface": honest,
        "cherry_picked_subset": cherry,
        "subset_raised": raised,
        "guard_blocks_subsetting": raised is not None,
        "empty_maps": empty,
        "empty_raised": empty_raised,
        "verdict": ("BYPASSABLE: the guard only compares the two maps to EACH OTHER, so any common "
                    "subset passes; it holds no reference to the frozen surface's target count"
                    if raised is None else "guard blocks subsetting"),
    }


def _fake_arm(n: int, tp: int, fp: int, fn: int, ratio: float) -> list[dict]:
    j = tp / (tp + fp + fn)
    return [{"edge_tp": tp, "edge_fp": fp, "edge_fn": fn, "edge_jaccard": j,
             "total_node_ratio": ratio} for _ in range(n)]


def _summarise(rows: list[dict]) -> dict:
    from assoc_report import ADJUSTMENT_ALPHA, SCORE_DIVISION_WEIGHT
    tp = sum(r["edge_tp"] for r in rows)
    fp = sum(r["edge_fp"] for r in rows)
    fn = sum(r["edge_fn"] for r in rows)
    j = tp / max(tp + fp + fn, 1)
    mult = 1.0 - ADJUSTMENT_ALPHA * rows[0]["total_node_ratio"]
    adj = max(0.0, j * mult)
    return {"edge_jaccard": j, "adj_edge_jaccard": adj, "division_jaccard": 0.5,
            "division_tp": 10, "division_fp": 1, "division_fn": 1, "node_recall": 0.99,
            "score": adj + SCORE_DIVISION_WEIGHT * 0.5}


def p_net_only_report() -> dict:
    """Can a bare NET figure - the exact thing FACT-0376 could not decompose - satisfy the verdict?"""
    from assoc_report import build_report
    control = _fake_arm(20, 900, 100, 100, 0.0)
    candidate = _fake_arm(20, 950, 100, 100, 0.0)      # unambiguously better association
    net_only = {"net": 28}                              # NO gained, NO lost, NO churn, NO targets
    rep = build_report(model="probe-net-only", fold=0, control=control, candidate=candidate,
                       summarise=_summarise, conversions=net_only, draws=200)
    missing_none = build_report(model="probe-no-conv", fold=0, control=control, candidate=candidate,
                                summarise=_summarise, conversions=None, draws=200)
    return {
        "executed": True,
        "conversions_passed_in": net_only,
        "stored_verbatim": rep["channels"]["parent_conversions"],
        "verdict_with_net_only": rep["verdict"],
        "verdict_with_conversions_none": missing_none["verdict"],
        "net_only_is_promotable": rep["verdict"]["promotable"],
        "identity_check_value": rep["channels"]["score"]["identity_check"],
        "verdict_reads_identity_check": "identity" in json.dumps(rep["verdict"]),
        "conclusion": ("NOT CLOSED: build_report stores `conversions` verbatim and verdict() reads "
                       "only ['net'], so a bare net figure - the precise quantity FACT-0376 could "
                       "not decompose - is accepted as a satisfied gained/lost/churn contract"
                       if rep["verdict"]["promotable"] else "net-only figure is rejected"),
    }


def p_bootstrap_direction() -> dict:
    """`excludes_zero` is two-sided. Does a candidate that is significantly WORSE satisfy it?"""
    from assoc_report import build_report, paired_bootstrap_score
    control = _fake_arm(20, 900, 100, 100, 0.0)
    worse = _fake_arm(20, 700, 100, 100, 0.0)
    boot = paired_bootstrap_score(control, worse, _summarise, draws=500)
    rep = build_report(model="probe-worse", fold=0, control=control, candidate=worse,
                       summarise=_summarise,
                       conversions={"targets": 10, "gained": 5, "lost": 1, "net": 4,
                                    "held_correct": 4, "churn": 6},
                       draws=500)
    return {
        "executed": True,
        "delta_score": rep["channels"]["score"]["delta"],
        "bootstrap": boot,
        "excludes_zero_on_a_significant_LOSS": boot["excludes_zero"],
        "blockers": rep["verdict"]["blockers"],
        "bootstrap_blocker_fired": any("bootstrap" in b for b in rep["verdict"]["blockers"]),
        "conclusion": ("NON-DIRECTIONAL: a significantly NEGATIVE delta satisfies the "
                       "'interval excludes zero' condition; the check is carried only by the "
                       "separate raw-edge-Jaccard blocker"
                       if boot["excludes_zero"] else "interval covered zero on this probe"),
    }


# ---------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--f0", type=Path, default=Path("C:/temp/assoc/f0.parquet"))
    ap.add_argument("--f1", type=Path, default=Path("C:/temp/assoc/f1.parquet"))
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--gt-limit", type=int)
    ap.add_argument("--deployed-floor", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("C:/temp/audit_assoc/contract_audit.json"))
    args = ap.parse_args()

    probes: dict[str, dict] = {}
    failures: list[str] = []
    runners = {
        "tie_break_rule": p_tie_break_rule,
        "two_predecessor": lambda: p_two_predecessor(args.gt_dir, args.gt_limit),
        "fold_degeneracy": lambda: p_fold_degeneracy(args.f0, args.f1, args.deployed_floor),
        "feature_label_wall": lambda: p_feature_label_wall(
            Path(__file__).resolve().parent / "assoc_parent_dataset.py"),
        "conversions_guard": p_conversions_guard,
        "net_only_report": p_net_only_report,
        "bootstrap_direction": p_bootstrap_direction,
    }
    for name in DECLARED:
        try:
            probes[name] = runners[name]()
            if not probes[name].get("executed"):
                failures.append(f"{name}: body completed without setting executed")
        except Exception as exc:                       # noqa: BLE001 - a probe crash IS the finding
            import traceback
            probes[name] = {"executed": False, "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-1500:]}
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    for name in DECLARED:
        if name not in probes:
            failures.append(f"{name}: DECLARED BUT NEVER RUN")

    report = {"schema_version": 1, "declared_probes": DECLARED,
              "executed_probes": sorted(k for k, v in probes.items() if v.get("executed")),
              "failures": failures, "probes": probes}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    for name in DECLARED:
        p = probes.get(name, {})
        mark = "ok " if p.get("executed") else "XX "
        print(f"  {mark}{name:<20} {p.get('verdict') or p.get('conclusion') or p.get('error', '')}")
    if failures:
        print("\nAUDIT INCOMPLETE - the heartbeat is withheld:", flush=True)
        for f in failures:
            print("   ", f)
        return 1
    print(f"\nASSOC_CONTRACT_AUDIT_COMPLETE probes={len(DECLARED)} out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
