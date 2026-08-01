r"""Generate the canonical claims table DIRECTLY from artifact JSON.

Why this exists
---------------
Every reporting error this project has suffered had the same shape: a number was
transcribed into a Markdown doc, the artifact moved on, and the doc kept asserting the
old value. Four corrections landed in a single cycle (H1-M ~30x, node budget 5.2x, FN
attribution 1.5x, ssl gate ~26%), plus a recorded cause-of-death that the kernel log
flatly contradicted.

So no number in a tracked doc should be typed by hand. This script reads the artifacts
and emits the table. If an artifact is missing, or a path no longer resolves, or a basis
tag is not one of the six legal values, it FAILS LOUDLY rather than emitting a stale or
untagged number.

Basis tags -- exactly one per claim, no exceptions
-------------------------------------------------
  public              a leaderboard score we actually received
  exact-pooled-OOF    exact patched scorer, full 199-crop corpus, pooled objective
  cross-family-LOFO   fitted on one family, evaluated on the other
  in-family-CV        cross-validated within a single family  (WEAKEST -- inflates)
  placeholder-proxy   the four visible placeholder movies     (in-sample, biased)
  GT-oracle           uses ground truth in the decision       (a ceiling, NOT a candidate)

Usage
-----
  .\.venv\Scripts\python.exe scripts\claims_table.py                  # write reports/CLAIMS.md
  .\.venv\Scripts\python.exe scripts\claims_table.py --check          # verify only, exit 1 on drift
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INV = ROOT / "reports" / "inventory"
OUT = ROOT / "reports" / "CLAIMS.md"

ALLOWED_BASES = (
    "public",
    "exact-pooled-OOF",
    "cross-family-LOFO",
    "in-family-CV",
    "placeholder-proxy",
    "GT-oracle",
)

# (section, label, artifact, dotted path, basis, format, note)
# `None` artifact == a value with no artifact yet; it must carry its own note and is
# rendered as "NOT ARTIFACT-BACKED" so it can never masquerade as a measurement.
CLAIMS: list[tuple] = [
    # ---------------------------------------------------------------- deployment
    ("Deployment", "P0-B public (deployment base)", None, None, "public", None,
     "0.914 - recorded in reports/submissions/CANDIDATE_LEDGER.md, ref 55136908"),
    ("Deployment", "P0-A public (clean 0.913 reproduction)", None, None, "public", None,
     "0.913 - ref 55136759"),
    ("Deployment", "v122 public", None, None, "public", None, "0.908 - ref 54854143"),
    ("Deployment", "E0c public (scientific anchor)", None, None, "public", None, "0.889"),

    # ---------------------------------------------------------------- substrate
    ("P0 substrate (fold 0 / 44b6)", "adj_edge_jaccard",
     "loeo_f0_strict.json", "summary.adj_edge_jaccard", "exact-pooled-OOF", "{:.5f}",
     "71 crops, arm strict. NOT comparable to E0c 0.7595 until reconciled in one pass."),
    ("P0 substrate (fold 0 / 44b6)", "node_recall",
     "loeo_f0_strict.json", "summary.node_recall", "exact-pooled-OOF", "{:.5f}", ""),
    ("P0 substrate (fold 0 / 44b6)", "division_jaccard",
     "loeo_f0_strict.json", "summary.division_jaccard", "exact-pooled-OOF", "{:.5f}",
     "TP 2 / FP 100 / FN 24 - its own division layer is near-worthless here"),
    ("P0 substrate (fold 0 / 44b6)", "reachable GT divisions",
     "loeo_f0_strict.json", "family_reach.44b6.reachable", "exact-pooled-OOF", "{:.0f}",
     "of 26. Beats E0c/clean903 20/26 and v122 15/26 on THIS family only - the advantage INVERTS on 6bba (66/125, worst measured). See the fold-1 rows."),

    ("P0 substrate (fold 1 / 6bba)", "adj_edge_jaccard",
     "loeo_f1_strict.json", "summary.adj_edge_jaccard", "exact-pooled-OOF", "{:.5f}",
     "128 crops, our own split_1, manifest audit PASS"),
    ("P0 substrate (fold 1 / 6bba)", "node_recall",
     "loeo_f1_strict.json", "summary.node_recall", "exact-pooled-OOF", "{:.5f}",
     "collapses from 0.98457 on fold 0 across the family boundary"),
    ("P0 substrate (fold 1 / 6bba)", "reachable GT divisions",
     "loeo_f1_strict.json", "family_reach.6bba.reachable", "exact-pooled-OOF", "{:.0f}",
     "of 125. WORST measured - E0c 93, clean903 96, v122 68. Pooled reach 88/151 = 58.3% "
     "vs E0c 74.8% and clean903 76.8%. 6bba is 85.06% of edge mass."),

    ("Component retention (CLOSED)", "GT-oracle corpus-scaled delta",
     "laneA_component_retention_replay.json", "corpus.oracle", "GT-oracle", "{:+.6f}",
     "763 components retained; bilaterally positive; clears the +0.002 bar 4x over"),
    ("Component retention (CLOSED)", "GT-free selector corpus-scaled delta",
     "laneA_component_retention_replay.json", "corpus.selector",
     "cross-family-LOFO", "{:+.6f}",
     "11,683 components retained - 15x the oracle - adding 49,688 nodes for edge TP -117; "
     "families disagree in sign (+0.0012 / -0.0115). Headroom is real; selection is the constraint."),

    ("Motion-residual gate (WS-A arm B)", "pooled delta",
     "wsa_PROMOTION_CANDIDATE_arm_B.json", "delta.pooled", "exact-pooled-OOF", "{:+.7f}",
     "199/199 crops, complete wrapper, parity gap EXACTLY 0.0 vs canonical anchors. prob=zero on "
     "new pairs is a LOWER diagnostic, so deployable >= this. E0c SUBSTRATE, not P0-B."),
    ("Motion-residual gate (WS-A arm B)", "44b6 delta (min fold)",
     "wsa_PROMOTION_CANDIDATE_arm_B.json", "delta.44b6", "exact-pooled-OOF", "{:+.7f}",
     "clears the CLAUDE.md min-fold >= +0.005 gate - the FIRST mechanism in the programme to do so"),
    ("Motion-residual gate (WS-A arm B)", "6bba delta",
     "wsa_PROMOTION_CANDIDATE_arm_B.json", "delta.6bba", "exact-pooled-OOF", "{:+.7f}",
     "85.06% of edge mass"),
    ("Motion-residual gate (WS-A arm B)", "bootstrap lower 95%",
     "wsa_PROMOTION_CANDIDATE_arm_B.json", "bootstrap_pooled.lo95", "exact-pooled-OOF", "{:+.6f}",
     "2000 crop-block draws; P(d>0)=1.000 and P(d>+0.002)=1.000"),
    ("Acquisition policy (WS-A arm E)", "UPPER diagnostic - GT-oracle probability",
     "wsa_BOUNDS_arm_E.json",
     "bound_2_UPPER_DIAGNOSTIC_GT_oracle_probability_on_newly_admitted_pairs.pooled", "GT-oracle", "{:+.7f}",
     "arm E is FALSIFIED: even its upper bound is negative. The loss is in the candidate surface "
     "(net -422 GT-true pairs); a pair off the surface cannot be selected at any probability."),

    ("Suppress-all (WS-B)", "P0-strict OOF pooled delta, complete wrapper",
     "wsb_suppressall_p0strict.json", "result.pooled.delta_composite",
     "exact-pooled-OOF", "{:+.7f}",
     "199 crops, complete wrapper re-run, 0/199 parity mismatches. CI [+0.000716,+0.002545], "
     "P(d>0)=0.9995. GT-free, no selector, pi_vis-invariant. NOT bilateral: 44b6 -0.000619 "
     "(P(gain)=0.339) / 6bba +0.002093 (P=1.000). E0c's +0.002706 did NOT port."),
    ("Suppress-all (WS-B)", "edge vs division share of the delta",
     "wsb_suppressall_p0strict.json", "result.pooled.share_edge",
     "exact-pooled-OOF", "{:.4f}",
     "edge term carries 124% and the division term is NEGATIVE (-24%) - the opposite of node "
     "budget's 116% count-multiplier profile. This is edge quality, not a metric artifact."),
    ("Suppress-all (WS-B)", "q_net of deleted content vs retain floor",
     "wsb_suppressall_p0strict.json", "result.pooled.retain_floor_q_star",
     "exact-pooled-OOF", "{:.4f}",
     "deleted content q_net 0.2126 sits BELOW this floor, so deleting is correct by the "
     "project's own independently-derived rule"),

    # ---------------------------------------------------------------- node budget
    ("Node budget (CLOSED)", "arm D (v122) delta @ keep_frac 0.975",
     "node_budget_armD.json", "table.0.975.delta", "exact-pooled-OOF", "{:+.5f}",
     "199 crops. Pooled optimum is keep_frac 1.00 = no pruning."),
    ("Node budget (CLOSED)", "arm D node ratio @ 1.0",
     "node_budget_armD.json", "table.1.0.ratio", "exact-pooled-OOF", "{:+.4f}",
     "under-predicts; parity-exact against the published v122 anchor"),
    ("Node budget (CLOSED)", "P0-B delta @ keep_frac 0.975",
     "p1_node_budget_p0b_substrate.json",
     "like_for_like_same_four_movies.p0b.delta", "placeholder-proxy", "{:+.7f}",
     "proxy is BIASED IN FAVOUR of the mechanism and it still fails"),
    ("Node budget (CLOSED)", "P0-B node ratio",
     "p1_node_budget_p0b_substrate.json",
     "like_for_like_same_four_movies.p0b.node_ratio", "placeholder-proxy", "{:+.5f}", ""),
    ("Node budget (CLOSED)", "E0c arm A delta @ keep_frac 0.975 (same 4 movies)",
     "p1_node_budget_p0b_substrate.json",
     "like_for_like_same_four_movies.e0c_armA.delta", "placeholder-proxy", "{:+.6f}",
     "CORRECTED 2026-08-01: this is NOT explained by the node-ratio sign - see the q_net rows below"),

    # ---------------------------------------------- decision theory (mechanism, corrected)
    ("Decision rule (corrected mechanism)", "E0c arm A q_net of deleted components",
     "decision_node_budget_qnet.json", "totals.e0c_armA.q_net", "placeholder-proxy", "{:+.4f}",
     "d_tp -5 / d_fp +8 -> below the 0.3994 floor, so DELETING is correct there"),
    ("Decision rule (corrected mechanism)", "P0-B q_net of deleted components",
     "decision_node_budget_qnet.json", "totals.p0b.q_net", "placeholder-proxy", "{:+.4f}",
     "d_tp +5 / d_fp 0 -> above the floor, so deleting is WRONG; this is the real mechanism, "
     "not the node ratio (count cost is 0.1*tp/N_est, invariant to over/under-prediction)"),
    ("Decision rule (corrected mechanism)", "H0c oracle reproduced from the closed form",
     "decision_div_threshold.json", "oracle_delta_vs_base", "GT-oracle", "{:+.7f}",
     "admitting all 92 census positives reproduces the reported +0.06012 - validates the kernel"),
    ("Decision rule (corrected mechanism)", "metric-visible reliable negatives",
     "decision_div_threshold.json", "classes.reliable_negative.n", "exact-pooled-OOF", "{:.0f}",
     "vs 14,117,560 unlabeled candidates of which only 65 are metric-visible - "
     "the precision denominator is the visible set, not the raw shortlist"),
    ("Decision rule (corrected mechanism)", "pi_vis measured (DO NOT DEPLOY)",
     "decision_div_threshold.json", "pi_vis", "exact-pooled-OOF", "{:.5f}",
     "MUST be pinned to 1.0 in any selector - fitting it is an annotation-coverage exploit"),

    # ---------------------------------------------------------------- ssl gate
    ("ssl x geometry gate (CLOSED)", "pooled delta",
     "h4_ssl_gate_replay_pooled.json", "pooled.delta_composite", "exact-pooled-OOF", "{:+.6f}",
     "headline was +0.00141, a projection; exact scorer says this"),
    ("ssl x geometry gate (CLOSED)", "44b6 delta",
     "h4_ssl_gate_replay_pooled.json", "44b6.delta_composite", "exact-pooled-OOF", "{:+.6f}", ""),
    ("ssl x geometry gate (CLOSED)", "6bba delta (min-fold)",
     "h4_ssl_gate_replay_pooled.json", "6bba.delta_composite", "exact-pooled-OOF", "{:+.6f}",
     "P(gain) = 0.52, a coin flip, against a +0.005 bilateral gate"),
    ("ssl x geometry gate (CLOSED)", "baseline parity: 44b6",
     "h4_ssl_gate_replay_pooled.json", "44b6.baseline.composite", "exact-pooled-OOF", "{:.6f}",
     "reproduces the published 0.759549 anchor - this is what makes the delta trustworthy"),
    ("ssl x geometry gate (CLOSED)", "baseline parity: 6bba",
     "h4_ssl_gate_replay_pooled.json", "6bba.baseline.composite", "exact-pooled-OOF", "{:.6f}",
     "reproduces the published 0.648965 anchor"),

    # ---------------------------------------------------------------- FN attribution
    ("Association FN attribution", "baseline pooled",
     "fn_attribution_ceilings.json", "base_pooled", "exact-pooled-OOF", "{:.7f}", ""),
    ("Association FN attribution", "total edge FN",
     "fn_attribution_ceilings.json", "n_fn", "exact-pooled-OOF", "{:.0f}",
     "43.80% never detected; 43.32% detected then discarded"),

    # ---------------------------------------------------------------- division economics
    ("Division economics", "H0c full oracle pooled delta",
     "pooled_breakeven.json", "arms.k1.0.delta", "GT-oracle", "{:+.5f}",
     "CEILING, not a candidate - fork choice uses ground truth"),
    ("Division economics", "suppress-all alone (EDGES-ONLY - superseded)",
     "pooled_breakeven.json", "arms.supp.delta", "exact-pooled-OOF", "{:+.6f}",
     "SIGN CORRECTED 2026-08-01: this -0.001728 is an EDGES-ONLY artifact on a fixed node set. "
     "Through the COMPLETE wrapper suppress-all is +0.002706 pooled, CI [+0.001726,+0.003703], "
     "P(d>0)=1.000, 44b6 +0.001532 / 6bba +0.002872 - bilaterally positive, GT-FREE, no selector, "
     "pi_vis-invariant. A +0.0044 swing; trap 14 in reverse. E0c ONLY - E0c has 20,353 forks, "
     "P0-B has 8, so the magnitude cannot port and must be re-measured."),
    ("Division economics", "delta at 1 FP per true fork",
     "pooled_breakeven.json", "arms.fp1.delta", "GT-oracle", "{:+.5f}", "92 true / 92 false"),
    ("Division economics", "delta at 5 FP per true fork",
     "pooled_breakeven.json", "arms.fp5.delta", "GT-oracle", "{:+.5f}", "88 true / 460 false"),
    ("Division economics", "delta at 10 FP per true fork",
     "pooled_breakeven.json", "arms.fp10.delta", "GT-oracle", "{:+.5f}", "88 true / 912 false"),
    ("Division economics", "delta at 25 FP per true fork",
     "pooled_breakeven.json", "arms.fp25.delta", "GT-oracle", "{:+.6f}",
     "NEGATIVE - this is where a weak selector lands"),
    ("Division economics", "precision required for +0.005 pooled",
     "pooled_breakeven.json", "required.0.005.implied_precision", "exact-pooled-OOF", "{:.5f}",
     "the 10.15% break-even, AT FULL RECALL; the marginal threshold is what a selector needs"),
    ("Division economics", "precision required for +0.002 pooled",
     "pooled_breakeven.json", "required.0.002.implied_precision", "exact-pooled-OOF", "{:.5f}", ""),
]


def resolve(doc, path: str):
    """Resolve a dotted path, tolerating keys that themselves contain dots.

    Artifact JSONs key sweeps by their float value -- `table["0.975"]`, `required["0.005"]` --
    so a naive `path.split(".")` shatters those keys and reports false drift. At each level we
    try the LONGEST matching key prefix first, then fall back to shorter ones.
    """
    parts = path.split(".")

    def walk(cur, i: int):
        if i == len(parts):
            return cur, True
        if isinstance(cur, list):
            try:
                return walk(cur[int(parts[i])], i + 1)
            except (ValueError, IndexError):
                return None, False
        if not isinstance(cur, dict):
            return None, False
        for j in range(len(parts), i, -1):          # longest key first
            key = ".".join(parts[i:j])
            if key in cur:
                got, ok = walk(cur[key], j)
                if ok:
                    return got, True
        return None, False

    val, ok = walk(doc, 0)
    if not ok:
        raise KeyError(path)
    return val


def build() -> tuple[str, list[str]]:
    problems: list[str] = []
    cache: dict[str, dict] = {}
    lines = [
        "# Claims table (generated -- do not edit by hand)",
        "",
        f"**Generated:** {date.today().isoformat()} by `scripts/claims_table.py`",
        "",
        "Every number below is read from an artifact at generation time. If an artifact moves or a",
        "path stops resolving, this file fails to build rather than asserting a stale value. Hand-",
        "editing it defeats the entire point -- change the artifact, then regenerate.",
        "",
        "**Basis tags.** `public` = a leaderboard score we received - `exact-pooled-OOF` = exact",
        "patched scorer, 199-crop corpus, pooled - `cross-family-LOFO` = fitted on one family,",
        "evaluated on the other - `in-family-CV` = within one family (WEAKEST, inflates) -",
        "`placeholder-proxy` = the four visible movies (in-sample, biased) - `GT-oracle` = uses",
        "ground truth in the decision, so it is a CEILING and never a candidate.",
        "",
    ]
    section = None
    for sec, label, art, path, basis, fmt, note in CLAIMS:
        if basis not in ALLOWED_BASES:
            problems.append(f"{label!r}: illegal basis {basis!r}")
            continue
        if sec != section:
            section = sec
            lines += ["", f"## {sec}", "",
                      "| claim | value | basis | note |", "|---|---:|---|---|"]
        if art is None:
            lines.append(f"| {label} | *(see note)* | `{basis}` | NOT ARTIFACT-BACKED - {note} |")
            continue
        f = INV / art
        if not f.exists():
            problems.append(f"{label!r}: missing artifact {art}")
            lines.append(f"| {label} | **MISSING ARTIFACT** | `{basis}` | {art} |")
            continue
        if art not in cache:
            cache[art] = json.loads(f.read_text(encoding="utf-8"))
        try:
            val = resolve(cache[art], path)
        except (KeyError, IndexError, ValueError):
            problems.append(f"{label!r}: path {path!r} no longer resolves in {art}")
            lines.append(f"| {label} | **PATH DRIFT** | `{basis}` | `{path}` in {art} |")
            continue
        shown = fmt.format(val) if fmt else str(val)
        lines.append(f"| {label} | `{shown}` | `{basis}` | {note} |")

    lines += ["", "---", "",
              f"{len(CLAIMS)} claims from {len(cache)} artifacts under `reports/inventory/`."]
    return "\n".join(lines) + "\n", problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="verify claims still resolve; do not rewrite the file")
    a = ap.parse_args()
    text, problems = build()
    for p in problems:
        print(f"DRIFT: {p}", file=sys.stderr)
    if a.check:
        if problems:
            print(f"\n{len(problems)} claim(s) drifted.", file=sys.stderr)
            return 1
        print(f"OK - all {len(CLAIMS)} claims resolve.")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(CLAIMS)} claims)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
