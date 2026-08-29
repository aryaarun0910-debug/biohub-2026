r"""Report-completeness gate: a summary that assoc_report.py did not produce is not a result.

WHAT IS REJECTED, AND WHY EACH REJECTION IS ALREADY PAID FOR
-------------------------------------------------------------
  * NO RAW ASSOCIATION CHANNEL. LEVER-0036 died on an adjusted figure that rose while the raw
    channel did not (FACT-0375), and LEVER-0037 died on the mirror image (FACT-0376). A result
    that reports only the adjusted number cannot be read at all.
  * NO COUNT-ADJUSTMENT CONTRIBUTION. The metric multiplier pays for predicting fewer nodes
    (FACT-0191/FACT-0192), so an adjusted delta with no decomposition into association and node
    count is unattributable.
  * NO NODE RECALL. Without it a gain bought by dropping cells looks like a gain.
  * NO DIVISION TP/FP/FN. An association gain is never read as division recovery (FACT-0371),
    and that can only be checked if divisions are reported separately at every gate.
  * A POOLED HEADLINE. Pooling has hidden a full inversion in this project - the lever ranking
    flips by fold (FACT-0031), and FACT-0385's embryo inversion would have been invisible pooled.
    Both embryo directions are reported separately or the result is rejected.

THE VERDICT IS RECOMPUTED, NOT READ. `verdict()` is called here on the submitted report rather
than trusting whatever the producer recorded next to it - the producer and the promotion decision
must not be the same statement. If the recomputed verdict disagrees with the recorded one, that
disagreement is itself the finding.

FAIL CLOSED, and the failure mode this guards is specific: a hand-written summary that looks
plausible. `--selftest` builds a complete synthetic report and then removes each required piece in
turn, asserting rejection every time, including the bare `{"net": N}` shape that once satisfied
the conversion contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

HEARTBEAT = "RESULT_COMPLETENESS_COMPLETE"
EXPECTED_EMBRYO = {0: "44b6", 1: "6bba"}


def check_one(report: dict, mod) -> dict:
    missing: list[str] = []
    if report.get("heartbeat") != "ASSOC_REPORT_COMPLETE":
        missing.append("heartbeat != ASSOC_REPORT_COMPLETE (not produced by assoc_report.py)")
    ch = report.get("channels")
    if not isinstance(ch, dict):
        missing.append("channels block absent")
        ch = {}
    for name in mod.CHANNELS:
        if name not in ch:
            missing.append(f"channel {name} absent")
    ca = ch.get("count_adjustment") or {}
    for k in ("delta_adj", "raw_channel", "count_channel", "count_adjustment_artifact"):
        if k not in ca:
            missing.append(f"count_adjustment.{k} absent")
    if "delta" not in (ch.get("edge_jaccard_raw") or {}):
        missing.append("edge_jaccard_raw.delta absent (raw association not reported)")
    nr = ch.get("node_recall") or {}
    for k in ("control", "candidate", "delta"):
        if k not in nr:
            missing.append(f"node_recall.{k} absent")
    dc = ch.get("division_counts") or {}
    for arm in ("control", "candidate"):
        for k in ("division_tp", "division_fp", "division_fn"):
            if k not in (dc.get(arm) or {}):
                missing.append(f"division_counts.{arm}.{k} absent")
    if "final_graph_edges" not in ch:
        missing.append("final_graph_edges absent (the FACT-0376 quantity that holds the gate)")
    pc = ch.get("parent_conversions")
    if isinstance(pc, dict) and set(pc) <= {"net"}:
        missing.append("parent_conversions is a bare net figure - gained/lost/churn required")
    if report.get("fold") not in (0, 1):
        missing.append("fold is not 0 or 1 - a pooled headline is not a result")

    recomputed = None
    try:
        recomputed = mod.verdict(report)
    except Exception as exc:
        missing.append(f"verdict() could not be recomputed: {type(exc).__name__}: {exc}")
    recorded = report.get("verdict")
    disagreement = (
        isinstance(recorded, dict) and recomputed is not None
        and bool(recorded.get("promotable")) != bool(recomputed["promotable"])
    )
    if disagreement:
        missing.append("recorded verdict disagrees with the recomputed one")
    return {
        "model": report.get("model"),
        "fold": report.get("fold"),
        "embryo_expected": EXPECTED_EMBRYO.get(report.get("fold")),
        "missing": missing,
        "recomputed_verdict": recomputed,
        "recorded_verdict": recorded,
        "accepted": not missing,
    }


def _synthetic() -> dict:
    return {
        "schema_version": 1, "heartbeat": "ASSOC_REPORT_COMPLETE",
        "model": "synthetic", "fold": 0, "n_crops": 71,
        "channels": {
            "edge_jaccard_raw": {"control": 0.88, "candidate": 0.89, "delta": 0.01},
            "count_adjustment": {"delta_adj": 0.01, "raw_channel": 0.01, "count_channel": 0.0,
                                 "count_adjustment_artifact": False},
            "parent_conversions": {"gained": 90, "lost": 50, "churn": 140, "net": 40,
                                   "stage": "pre_ILP_candidate_ranking"},
            "final_graph_edges": {"delta_tp": 12},
            "node_recall": {"control": 0.99, "candidate": 0.99, "delta": 0.0},
            "division_counts": {
                "control": {"division_tp": 5, "division_fp": 67, "division_fn": 17},
                "candidate": {"division_tp": 5, "division_fp": 53, "division_fn": 17},
                "delta_tp": 0, "delta_fp": -14, "delta_fn": 0, "division_jaccard_delta": 0.02},
            "score": {"delta": 0.012, "identity_check": 0.0},
        },
        "paired_bootstrap": {"lo": 0.002, "hi": 0.02, "favourable": True},
    }


def selftest(mod) -> int:
    base = _synthetic()
    ok = check_one(json.loads(json.dumps(base)), mod)
    failures = []
    if not ok["accepted"]:
        failures.append(("BASELINE", ok["missing"]))
    mutations = {
        "no_heartbeat": lambda r: r.pop("heartbeat"),
        "no_raw_channel": lambda r: r["channels"].pop("edge_jaccard_raw"),
        "no_count_adjustment": lambda r: r["channels"].pop("count_adjustment"),
        "no_node_recall": lambda r: r["channels"].pop("node_recall"),
        "no_division_counts": lambda r: r["channels"].pop("division_counts"),
        "no_final_graph_edges": lambda r: r["channels"].pop("final_graph_edges"),
        "bare_net_conversions": lambda r: r["channels"].__setitem__("parent_conversions",
                                                                    {"net": 28}),
        "pooled_headline": lambda r: r.__setitem__("fold", "pooled"),
        "recorded_verdict_disagrees": lambda r: r.__setitem__(
            "verdict", {"promotable": True, "blockers": []}) or r["channels"]
        ["edge_jaccard_raw"].__setitem__("delta", -0.01),
    }
    for name, mutate in mutations.items():
        r = json.loads(json.dumps(base))
        mutate(r)
        res = check_one(r, mod)
        print(f"  selftest {name:28s} -> rejected={not res['accepted']} "
              f"({res['missing'][:1]})")
        if res["accepted"]:
            failures.append((name, "ACCEPTED a defective report"))
    if failures:
        for n, m in failures:
            print(f"SELFTEST FAILED {n}: {m}")
        return 1
    print(f"{HEARTBEAT} selftest=PASS mutations={len(mutations)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", type=Path, nargs="*", default=[])
    ap.add_argument("--require-both-folds", action="store_true",
                    help="reject unless BOTH embryo directions are present among the reports")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    import assoc_report as mod  # noqa: PLC0415

    if args.selftest:
        return selftest(mod)
    if not args.out:
        raise SystemExit("--out is required unless --selftest")

    rows = []
    for p in args.report:
        path = p if p.is_absolute() else ROOT / p
        if not path.is_file():
            rows.append({"report": str(path), "accepted": False,
                         "missing": ["report file not found"]})
            continue
        try:
            rows.append({"report": str(path),
                         **check_one(json.loads(path.read_text(encoding="utf-8")), mod)})
        except json.JSONDecodeError as exc:
            rows.append({"report": str(path), "accepted": False,
                         "missing": [f"unreadable JSON: {exc}"]})
    folds = {r.get("fold") for r in rows}
    both = {0, 1} <= folds
    result = {
        "schema_version": 1,
        "reports": rows,
        "both_embryo_directions_present": both,
        "passed": bool(rows) and all(r["accepted"] for r in rows)
        and (both or not args.require_both_folds),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for r in rows:
        print(f"  {'ACCEPT' if r['accepted'] else 'REJECT'}  {r.get('model')} fold={r.get('fold')}"
              + ("" if r["accepted"] else f"  {r['missing'][:3]}"))
    print(f"{HEARTBEAT} reports={len(rows)} passed={result['passed']} -> {args.out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
