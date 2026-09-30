"""The shared model-report contract must actually catch what it claims to catch.

These are SOFTWARE contract tests. They assert that the decomposition is arithmetically faithful
to the scorer and that the guards fire on the two pathologies already measured (FACT-0375's
count-adjustment artifact and a parent-choice wash). They decide nothing scientific.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import assoc_report as ar  # noqa: E402


def crop(edge_tp, edge_fp, edge_fn, n_pred, n_est, div=(0, 0, 0), node_recall=0.99):
    """One per-crop metric row in exactly the layout ``per_sample_metrics`` returns."""
    denom = edge_tp + edge_fp + edge_fn
    ratio = (n_pred - n_est) / n_est
    jac = edge_tp / denom if denom else float("nan")
    return {
        "edge_tp": edge_tp, "edge_fp": edge_fp, "edge_fn": edge_fn,
        "division_tp": div[0], "division_fp": div[1], "division_fn": div[2],
        "num_pred_nodes": n_pred, "node_recall": node_recall,
        "total_node_ratio": ratio, "edge_jaccard": jac,
        "adj_edge_jaccard": max(0.0, jac * (1 - ar.ADJUSTMENT_ALPHA * ratio)),
    }


def fake_summarise(rows):
    """Micro-averaged, matching the scorer's aggregation closely enough for the identity check."""
    tp = sum(r["edge_tp"] for r in rows)
    fp = sum(r["edge_fp"] for r in rows)
    fn = sum(r["edge_fn"] for r in rows)
    dtp = sum(r["division_tp"] for r in rows)
    dfp = sum(r["division_fp"] for r in rows)
    dfn = sum(r["division_fn"] for r in rows)
    weights = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rows]
    adj = sum(w * r["adj_edge_jaccard"] for w, r in zip(weights, rows)) / sum(weights)
    div_denom = dtp + dfp + dfn
    div_j = dtp / div_denom if div_denom else 0.0
    return {
        "edge_jaccard": tp / (tp + fp + fn),
        "adj_edge_jaccard": adj,
        "division_jaccard": div_j,
        "division_tp": dtp, "division_fp": dfp, "division_fn": dfn,
        "node_recall": sum(r["node_recall"] for r in rows) / len(rows),
        "score": adj + ar.SCORE_DIVISION_WEIGHT * div_j,
    }


def test_decomposition_reproduces_the_adjusted_delta_exactly():
    """raw + count + residual must sum to the adjusted delta, or the split is decorative."""
    control = [crop(900, 50, 50, 1000, 1000), crop(800, 100, 100, 990, 1000)]
    candidate = [crop(920, 40, 40, 1005, 1000), crop(830, 90, 80, 985, 1000)]
    d = ar.decompose_adjusted(control, candidate)
    total = d["raw_channel"] + d["count_channel"] + d["residual"]
    assert total == pytest.approx(d["delta_adj"], abs=1e-12)


def test_count_adjustment_artifact_fires_on_the_lever_0036_pathology():
    """RAW edge Jaccard down, adjusted up on a smaller node set - FACT-0375's exact shape."""
    control = [crop(900, 50, 50, 1000, 1000), crop(900, 50, 50, 1000, 1000)]
    # Fewer true edges, on a node set 15% smaller: raw edge Jaccard falls 0.900 -> 0.890 while
    # the under-production multiplier 1.015 lifts the adjusted figure to 0.9034.
    candidate = [crop(890, 45, 65, 850, 1000), crop(890, 45, 65, 850, 1000)]
    d = ar.decompose_adjusted(control, candidate)
    assert d["raw_channel"] < 0, "the arms must actually associate worse for this test to mean anything"
    assert d["delta_adj"] > 0, "and the adjusted figure must rise, which is the trap"
    assert d["count_adjustment_artifact"] is True


def test_no_artifact_flag_when_the_raw_channel_carries_the_gain():
    control = [crop(900, 50, 50, 1000, 1000)] * 2
    candidate = [crop(930, 35, 35, 1000, 1000)] * 2
    d = ar.decompose_adjusted(control, candidate)
    assert d["raw_channel"] > 0
    assert d["count_adjustment_artifact"] is False


def test_unpaired_arms_are_refused():
    with pytest.raises(ValueError, match="unpaired arms"):
        ar.decompose_adjusted([crop(900, 50, 50, 1000, 1000)], [])


def test_parent_conversions_separates_gained_from_displaced():
    """A wash must not read as a win: 10 gained and 10 lost is net 0 with churn 20."""
    before = {("c", i): (1 if i < 50 else 0) for i in range(100)}
    after = dict(before)
    for i in range(50, 60):      # ten newly correct
        after[("c", i)] = 1
    for i in range(0, 10):       # ten newly wrong
        after[("c", i)] = 0
    conv = ar.parent_conversions(before, after)
    assert conv["gained"] == 10 and conv["lost"] == 10
    assert conv["net"] == 0 and conv["churn"] == 20
    assert conv["top1_before"] == conv["top1_after"]


def test_parent_conversions_refuses_a_different_target_set():
    """Scoring a ranker on targets the baseline never saw is the easiest way to fake a gain."""
    with pytest.raises(ValueError, match="SAME targets"):
        ar.parent_conversions({("c", 1): 1}, {("c", 1): 1, ("c", 2): 1})


def test_verdict_blocks_a_count_artifact_even_when_the_score_rises():
    control = [crop(900, 50, 50, 1000, 1000)] * 4
    candidate = [crop(890, 45, 65, 850, 1000)] * 4
    report = ar.build_report(
        model="artifact-arm", fold=0, control=control, candidate=candidate,
        summarise=fake_summarise,
        conversions=ar.parent_conversions({("c", 1): 0}, {("c", 1): 1}),
        draws=50,
    )
    assert report["channels"]["score"]["delta"] > 0
    assert report["verdict"]["promotable"] is False
    assert any("count_adjustment_artifact" in b for b in report["verdict"]["blockers"])


def test_verdict_blocks_when_division_true_positives_decline():
    """An association gain is never banked as division recovery (FACT-0371)."""
    control = [crop(900, 50, 50, 1000, 1000, div=(5, 60, 20))] * 4
    candidate = [crop(940, 30, 30, 1000, 1000, div=(4, 60, 21))] * 4
    report = ar.build_report(
        model="div-losing-arm", fold=0, control=control, candidate=candidate,
        summarise=fake_summarise,
        conversions=ar.parent_conversions({("c", 1): 0}, {("c", 1): 1}),
        draws=50,
    )
    assert report["channels"]["division_counts"]["delta_tp"] < 0
    assert "division true positives declined" in report["verdict"]["blockers"]


def test_missing_conversions_blocks_promotion():
    control = [crop(900, 50, 50, 1000, 1000)] * 4
    candidate = [crop(940, 30, 30, 1000, 1000)] * 4
    report = ar.build_report(model="no-conv", fold=0, control=control, candidate=candidate,
                             summarise=fake_summarise, conversions=None, draws=50)
    assert "pre-ILP parent conversions not reported" in report["verdict"]["blockers"]


def test_a_bare_net_figure_is_refused_as_a_conversion_contract():
    """{"net": 28} is the exact figure FACT-0376 could not decompose. It must not satisfy the gate."""
    control = [crop(900, 50, 50, 1000, 1000)] * 4
    candidate = [crop(940, 30, 30, 1000, 1000)] * 4
    with pytest.raises(ValueError, match="a net figure alone does not satisfy"):
        ar.build_report(model="bare-net", fold=0, control=control, candidate=candidate,
                        summarise=fake_summarise, conversions={"net": 28}, draws=50)


def test_a_significantly_worse_candidate_is_not_promotable():
    """`excludes_zero` is two-sided; promotion must read the directional `favourable` instead."""
    control = [crop(940, 30, 30, 1000, 1000)] * 6
    candidate = [crop(860, 70, 70, 1000, 1000)] * 6
    report = ar.build_report(
        model="worse-arm", fold=0, control=control, candidate=candidate,
        summarise=fake_summarise,
        conversions=ar.parent_conversions({("c", 1): 0}, {("c", 1): 1}), draws=200,
    )
    assert report["paired_bootstrap"]["excludes_zero"] is True, "it IS significant - the wrong way"
    assert report["paired_bootstrap"]["favourable"] is False
    assert report["verdict"]["promotable"] is False


def test_final_graph_edge_conversion_is_the_gated_quantity():
    """A pre-ILP ranking win with no final-graph edge gain must not promote (FACT-0364)."""
    control = [crop(900, 50, 50, 1000, 1000)] * 4
    candidate = [crop(900, 45, 50, 1000, 1000)] * 4      # FP down, TP unchanged
    report = ar.build_report(
        model="no-edge-conversion", fold=0, control=control, candidate=candidate,
        summarise=fake_summarise,
        conversions=ar.parent_conversions({("c", i): 0 for i in range(10)},
                                          {("c", i): 1 for i in range(10)}),
        draws=50,
    )
    assert report["channels"]["final_graph_edges"]["delta_tp"] == 0
    assert report["channels"]["parent_conversions"]["net"] == 10, "pre-ILP ranking clearly improved"
    assert any("FACT-0376 quantity" in b for b in report["verdict"]["blockers"])


def test_parent_conversions_is_labelled_pre_ilp():
    conv = ar.parent_conversions({("c", 1): 0}, {("c", 1): 1})
    assert conv["stage"] == "pre_ILP_candidate_ranking"


def test_a_broken_score_identity_blocks_promotion():
    """A mismatched scorer must not pass silently into the JSON - finding 9."""
    control = [crop(900, 50, 50, 1000, 1000)] * 4
    candidate = [crop(940, 30, 30, 1000, 1000)] * 4

    def wrong_summarise(rows):
        out = fake_summarise(rows)
        out["score"] = out["score"] + 0.01      # a scorer that does not match its own channels
        return out

    report = ar.build_report(
        model="mismatched-scorer", fold=0, control=control, candidate=candidate,
        summarise=wrong_summarise,
        conversions=ar.parent_conversions({("c", 1): 0}, {("c", 1): 1}), draws=50,
    )
    assert report["channels"]["score"]["identity_check"] == pytest.approx(0.0, abs=1e-9), (
        "a constant offset cancels in the delta"
    )

    def scaling_summarise(rows):
        out = fake_summarise(rows)
        out["score"] = out["score"] * 1.5
        return out

    report = ar.build_report(
        model="scaling-scorer", fold=0, control=control, candidate=candidate,
        summarise=scaling_summarise,
        conversions=ar.parent_conversions({("c", 1): 0}, {("c", 1): 1}), draws=50,
    )
    assert abs(report["channels"]["score"]["identity_check"]) > 1e-6
    assert any("score identity violated" in b for b in report["verdict"]["blockers"])


def test_score_identity_holds():
    """delta score == delta adjusted + weight * delta division Jaccard, or the arms disagree."""
    control = [crop(900, 50, 50, 1000, 1000, div=(5, 60, 20)),
               crop(800, 90, 90, 995, 1000, div=(3, 40, 15))]
    candidate = [crop(930, 40, 35, 1002, 1000, div=(7, 55, 18)),
                 crop(820, 85, 80, 998, 1000, div=(4, 38, 14))]
    report = ar.build_report(model="identity", fold=0, control=control, candidate=candidate,
                             summarise=fake_summarise, draws=50)
    assert report["channels"]["score"]["identity_check"] == pytest.approx(0.0, abs=1e-9)
