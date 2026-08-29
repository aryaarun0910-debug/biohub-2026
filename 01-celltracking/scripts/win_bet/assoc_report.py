r"""THE SHARED MODEL-REPORT CONTRACT. Every learned surface reports through this module.

WHY THIS EXISTS
---------------
Three learned capabilities are being advanced at once - parent ranking, false-fork rejection and
the HOCT head. If each reports its own headline number the cycle produces three unfalsifiable
claims instead of a capability ladder, and worse, each is free to pick whichever channel happens
to have moved. So the decomposition lives here, once, and no ranker may be promoted on a summary
this module did not produce.

THE OBJECTIVE, AND WHY IT MUST BE SPLIT
---------------------------------------
The campaign objective is the official combined score. Read from the scorer itself
(``tracking_cellmot.metrics``), per crop:

    total_node_ratio = (N_pred - N_est) / N_est
    adj_edge_J_i     = max(0, edge_J_i * (1 - ALPHA * total_node_ratio_i))
    adj_edge_J       = weighted mean of adj_edge_J_i, weights w_i = TP_i + FP_i + FN_i
    score            = adj_edge_J + DIV_WEIGHT * division_J

so

    delta score = delta adj-edge-J + DIV_WEIGHT * delta division-J

THE TRAP THIS MODULE EXISTS TO CATCH. ``adj_edge_J`` is a PRODUCT of association quality and a
node-count multiplier, and that multiplier has no upper cap - under-producing nodes is PAID
(FACT-0191/FACT-0192). Two levers have already died exactly here, in opposite directions:
LEVER-0036 posted +0.00264 adjusted while RAW edge Jaccard went NEGATIVE at -0.00084, the entire
gain being a 12.5% smaller node set collecting a larger bonus (FACT-0375); LEVER-0037 was the
mirror, its adjusted loss exceeding its raw loss because extra candidates retained nodes and spent
the bonus (FACT-0376). A ranker that quietly drops nodes can therefore post a positive score while
associating WORSE. So the adjusted delta is decomposed against counterfactual arms:

    A = adj(J_ctrl,  m_ctrl)      the control
    B = adj(J_cand,  m_ctrl)      candidate association, control node-count multiplier
    C = adj(J_ctrl,  m_cand)      control association, candidate node-count multiplier
    D = adj(J_cand,  m_cand)      the candidate

    raw_channel   = B - A         what better ASSOCIATION bought
    count_channel = C - A         what the NODE COUNT bought
    residual      = D - B - C + A the interaction the product leaves over

``count_adjustment_artifact`` is set when the adjusted delta is positive while the raw channel is
not. Under the standing rule that is NON-PROMOTIONAL whatever the pooled score did.

WHAT A COMPLETE REPORT MUST CARRY (host, 2026-08-29). All six, always, never a subset:
raw edge Jaccard; the count-adjustment contribution; parent-choice conversions; node recall;
division TP/FP/FN; the combined score. Division is reported SEPARATELY at every gate and an
association gain is never read as division recovery (FACT-0371).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Read the weights from the scorer rather than restating them - a restated constant is a number
# that can go stale, which is the whole reason the registry exists.
try:  # pragma: no cover - taken whenever the real scorer is installed
    from tracking_cellmot.metrics import ADJUSTMENT_ALPHA, SCORE_DIVISION_WEIGHT
except Exception:  # pragma: no cover - keeps the module importable for unit tests
    ADJUSTMENT_ALPHA, SCORE_DIVISION_WEIGHT = 0.1, 0.1

CHANNELS = (
    "edge_jaccard_raw",
    "count_adjustment",
    "parent_conversions",
    "node_recall",
    "division_counts",
    "score",
)


def _multiplier(row: dict) -> float:
    return float(1.0 - ADJUSTMENT_ALPHA * row["total_node_ratio"])


def _weighted_adj(rows: list[dict], jaccards: list[float], multipliers: list[float]) -> float:
    """The scorer's own weighted mean, with edge Jaccard and multiplier supplied separately.

    Weights are w_i = TP_i + FP_i + FN_i taken from ``rows``, and the max(0, .) clip is applied
    per crop exactly as ``per_sample_metrics`` applies it.
    """
    weights, values = [], []
    for row, jac, mult in zip(rows, jaccards, multipliers):
        if jac != jac or mult != mult:
            continue
        weights.append(row["edge_tp"] + row["edge_fp"] + row["edge_fn"])
        values.append(max(0.0, jac * mult))
    total = sum(weights)
    if total <= 0:
        return float("nan")
    return float(sum(w * v for w, v in zip(weights, values)) / total)


def decompose_adjusted(control: list[dict], candidate: list[dict]) -> dict:
    """Split the adjusted-edge-Jaccard delta into an association channel and a node-count channel.

    Both arms must be the SAME crops in the SAME order - a paired comparison is the only one the
    bootstrap and the sign counts are valid for.
    """
    if len(control) != len(candidate):
        raise ValueError(f"unpaired arms: {len(control)} control crops vs {len(candidate)}")
    j_ctrl = [r["edge_jaccard"] for r in control]
    j_cand = [r["edge_jaccard"] for r in candidate]
    m_ctrl = [_multiplier(r) for r in control]
    m_cand = [_multiplier(r) for r in candidate]

    a = _weighted_adj(control, j_ctrl, m_ctrl)
    b = _weighted_adj(control, j_cand, m_ctrl)
    c = _weighted_adj(control, j_ctrl, m_cand)
    d = _weighted_adj(candidate, j_cand, m_cand)
    raw_channel = b - a
    count_channel = c - a
    return {
        "control_adj": a,
        "candidate_adj": d,
        "delta_adj": d - a,
        "raw_channel": raw_channel,
        "count_channel": count_channel,
        "residual": d - b - c + a,
        "count_adjustment_artifact": bool((d - a) > 0 and raw_channel <= 0),
    }


CONVERSION_KEYS = ("targets", "gained", "lost", "net", "held_correct", "churn")


def _edge_counts(rows: list[dict]) -> dict:
    """FINAL-GRAPH edge counts, summed from the per-crop metric rows.

    This is the genuine FACT-0376 quantity - true edges recovered through the COMPLETE chain -
    and it is deliberately computed here from the rows rather than read out of a caller's
    ``summarise``, because not every summariser in this tree returns the count columns.
    """
    return {k: int(sum(r[k] for r in rows)) for k in ("edge_tp", "edge_fp", "edge_fn")}


def parent_conversions(before: dict, after: dict) -> dict:
    """Parent-choice movement on the FROZEN PRE-ILP surface, so a wash cannot look like a win.

    ``before``/``after`` are per-target correctness maps keyed by (crop, target): 1 correct, 0 not.
    A net top-1 delta alone hides the trade - FACT-0376's 28 net true edges could have been 28
    gained and none lost, or hundreds gained and nearly as many displaced, and PKT-0027 rule (5)
    exists precisely because that distinction was not recoverable after the fact.

    STAGE WARNING, AND IT IS NOT PEDANTRY. What this measures is CANDIDATE RANKING BEFORE THE
    SOLVER. It is NOT the FACT-0376 quantity, which is a final-graph edge-TP delta, and the two
    can disagree completely: FACT-0364 established that ``motion_relink_edges`` replaces the whole
    edge list downstream, covering a median 99.9% of the solver's raw edges with zero fallbacks,
    so a better pre-ILP ranking survives only as a scoring prior. The final-graph quantity is
    reported separately in the ``final_graph_edges`` channel and it is the one promotion turns on.
    """
    keys = set(before) & set(after)
    if len(keys) != len(before) or len(keys) != len(after):
        raise ValueError(
            f"target sets differ: {len(before)} before, {len(after)} after, {len(keys)} shared - "
            "a ranker must be scored on the SAME targets as the baseline"
        )
    gained = sum(1 for k in keys if not before[k] and after[k])
    lost = sum(1 for k in keys if before[k] and not after[k])
    held = sum(1 for k in keys if before[k] and after[k])
    return {
        "stage": "pre_ILP_candidate_ranking",   # NOT the final-graph FACT-0376 quantity
        "targets": len(keys),
        "gained": gained,
        "lost": lost,
        "net": gained - lost,
        "held_correct": held,
        "top1_before": sum(before[k] for k in keys) / max(len(keys), 1),
        "top1_after": sum(after[k] for k in keys) / max(len(keys), 1),
        "churn": gained + lost,
    }


def paired_bootstrap_score(control: list[dict], candidate: list[dict], summarise,
                           draws: int = 2000, seed: int = 20260829) -> dict:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        idx = rng.integers(0, len(control), len(control))
        deltas.append(
            float(summarise([candidate[i] for i in idx])["score"]
                  - summarise([control[i] for i in idx])["score"])
        )
    values = np.asarray(deltas)
    lo, hi = float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
    return {
        "mean": float(values.mean()), "ci95": [lo, hi],
        "draws": draws, "seed": seed,
        # `excludes_zero` is the two-sided FACT: significant in EITHER direction. A significantly
        # WORSE candidate satisfies it, so promotion reads `favourable`, which is directional.
        "excludes_zero": bool(lo > 0 or hi < 0),
        "favourable": bool(lo > 0),
    }


def build_report(*, model: str, fold: int, control: list[dict], candidate: list[dict], summarise,
                 conversions: dict | None = None, notes: str = "",
                 draws: int = 2000, seed: int = 20260829) -> dict:
    """The only summary a learned surface may be promoted on."""
    if conversions is not None:
        # A bare {"net": 28} - the exact figure FACT-0376 could NOT decompose - must not satisfy
        # the gained/lost/churn contract just because `verdict` happens to read only one key.
        missing = [k for k in CONVERSION_KEYS if k not in conversions]
        if missing:
            raise ValueError(
                f"conversions is missing {missing}: a net figure alone does not satisfy the "
                "gained/lost/churn contract. Build it with parent_conversions()."
            )
    ctrl, cand = summarise(control), summarise(candidate)
    decomposition = decompose_adjusted(control, candidate)
    boot = paired_bootstrap_score(control, candidate, summarise, draws, seed)
    delta_score = float(cand["score"] - ctrl["score"])
    delta_div_j = float(cand["division_jaccard"] - ctrl["division_jaccard"])
    edges_ctrl, edges_cand = _edge_counts(control), _edge_counts(candidate)

    report = {
        "schema_version": 1,
        "heartbeat": "ASSOC_REPORT_COMPLETE",
        "model": model,
        "fold": fold,
        "n_crops": len(control),
        "channels": {
            "edge_jaccard_raw": {
                "control": float(ctrl["edge_jaccard"]),
                "candidate": float(cand["edge_jaccard"]),
                "delta": float(cand["edge_jaccard"] - ctrl["edge_jaccard"]),
            },
            "count_adjustment": decomposition,
            # THE genuine FACT-0376 quantity: net true edges through the COMPLETE chain. Kept
            # distinct from `parent_conversions`, which is a pre-ILP candidate-ranking figure and
            # survives the solver only as a scoring prior (FACT-0364).
            "final_graph_edges": {
                "control": edges_ctrl,
                "candidate": edges_cand,
                "delta_tp": edges_cand["edge_tp"] - edges_ctrl["edge_tp"],
                "delta_fp": edges_cand["edge_fp"] - edges_ctrl["edge_fp"],
                "delta_fn": edges_cand["edge_fn"] - edges_ctrl["edge_fn"],
                "added_fp_per_gained_tp": (
                    (edges_cand["edge_fp"] - edges_ctrl["edge_fp"])
                    / (edges_cand["edge_tp"] - edges_ctrl["edge_tp"])
                    if edges_cand["edge_tp"] != edges_ctrl["edge_tp"] else None
                ),
            },
            "parent_conversions": conversions,
            "node_recall": {
                "control": float(ctrl["node_recall"]),
                "candidate": float(cand["node_recall"]),
                "delta": float(cand["node_recall"] - ctrl["node_recall"]),
            },
            "division_counts": {
                "control": {k: int(ctrl[k]) for k in ("division_tp", "division_fp", "division_fn")},
                "candidate": {k: int(cand[k]) for k in ("division_tp", "division_fp", "division_fn")},
                "delta_tp": int(cand["division_tp"] - ctrl["division_tp"]),
                "delta_fp": int(cand["division_fp"] - ctrl["division_fp"]),
                "delta_fn": int(cand["division_fn"] - ctrl["division_fn"]),
                "division_jaccard_delta": delta_div_j,
            },
            "score": {
                "control": float(ctrl["score"]),
                "candidate": float(cand["score"]),
                "delta": delta_score,
                # delta score should equal delta adjusted + weight * delta division Jaccard.
                # A non-zero residual means the arms were not scored by the same scorer.
                "identity_check": float(
                    delta_score - decomposition["delta_adj"] - SCORE_DIVISION_WEIGHT * delta_div_j
                ),
            },
        },
        "paired_bootstrap": boot,
        "notes": notes,
    }
    report["verdict"] = verdict(report)
    return report


def verdict(report: dict) -> dict:
    """The promotion reading, applied identically to every model. Advisory, never automatic.

    These conditions are already paid for in evidence; none of them is new policy:
      - a count-adjustment artifact is non-promotional whatever the pooled score (FACT-0375);
      - RAW association must actually improve - the seam-calibration precondition;
      - division TP must not decline, so an association gain is never banked as division
        recovery (FACT-0371);
      - the paired interval must be FAVOURABLE, not merely significant - a significantly WORSE
        candidate also excludes zero;
      - opportunity must CONVERT into net true edges through the COMPLETE chain, which is the
        genuine FACT-0376 quantity. The pre-ILP parent-conversion figure is required as evidence
        of MECHANISM but is not sufficient on its own, because motion relink replaces the
        solver's edge list downstream (FACT-0364);
      - the score identity must hold, or the two arms were not scored by the same scorer and no
        channel in the report can be trusted.
    """
    ch = report["channels"]
    blockers = []
    if ch["count_adjustment"]["count_adjustment_artifact"]:
        blockers.append("count_adjustment_artifact: adjusted rose while the raw channel did not")
    if ch["edge_jaccard_raw"]["delta"] <= 0:
        blockers.append("raw edge Jaccard did not improve")
    if ch["division_counts"]["delta_tp"] < 0:
        blockers.append("division true positives declined")
    if not report["paired_bootstrap"]["favourable"]:
        blockers.append("paired bootstrap interval is not favourable (lower bound not above zero)")
    if ch["final_graph_edges"]["delta_tp"] <= 0:
        blockers.append("no net true-edge conversion through the complete chain (the FACT-0376 quantity)")
    if ch["parent_conversions"] is None:
        blockers.append("pre-ILP parent conversions not reported")
    elif ch["parent_conversions"]["net"] <= 0:
        blockers.append("no net pre-ILP parent-choice conversion")
    if abs(ch["score"]["identity_check"]) > 1e-6:
        blockers.append(
            f"score identity violated by {ch['score']['identity_check']:.2e}: the arms were not "
            "scored by the same scorer"
        )
    return {"promotable": not blockers, "blockers": blockers}


def write(report: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    ch = report["channels"]
    print(
        f"\n{report['heartbeat']} model={report['model']} fold={report['fold']} "
        f"crops={report['n_crops']}\n"
        f"  raw edge Jaccard      {ch['edge_jaccard_raw']['delta']:+.5f}\n"
        f"  adjusted (total)      {ch['count_adjustment']['delta_adj']:+.5f}"
        f"   = raw {ch['count_adjustment']['raw_channel']:+.5f}"
        f" + count {ch['count_adjustment']['count_channel']:+.5f}"
        f" + resid {ch['count_adjustment']['residual']:+.5f}\n"
        f"  node recall           {ch['node_recall']['delta']:+.5f}\n"
        f"  final-graph edges     TP {ch['final_graph_edges']['delta_tp']:+d} / "
        f"FP {ch['final_graph_edges']['delta_fp']:+d} / "
        f"FN {ch['final_graph_edges']['delta_fn']:+d}   <- the FACT-0376 quantity\n"
        f"  division TP/FP/FN     {ch['division_counts']['delta_tp']:+d} / "
        f"{ch['division_counts']['delta_fp']:+d} / {ch['division_counts']['delta_fn']:+d}\n"
        f"  parent conv (pre-ILP) {ch['parent_conversions']}\n"
        f"  score identity        {ch['score']['identity_check']:+.2e}\n"
        f"  SCORE                 {ch['score']['delta']:+.5f}  "
        f"ci95={report['paired_bootstrap']['ci95']}\n"
        f"  verdict               {report['verdict']}",
        flush=True,
    )
