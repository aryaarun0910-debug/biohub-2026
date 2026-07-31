"""Regression lock on the scoring objective.

The leaderboard POOLS all samples with edge-volume weighting. It does not average, min, or
otherwise treat the two embryo families as equally important. 44b6 carries ~15% of total edge
mass, so a 50/50 family criterion is a different objective from the one being scored.

These tests fail if anyone restores 50/50 family weighting, or a bilateral-delta gate, as the
PRIMARY selection criterion. Min-fold remains legitimate as a robustness CONSTRAINT reported
alongside the pooled score -- it is just not the thing being optimised.

See scripts/verify_pooled_objective.py for the full proof against cached per-crop rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracking_cellmot.metrics import ADJUSTMENT_ALPHA, SCORE_DIVISION_WEIGHT, summarise  # noqa: E402


def _row(edge_tp, edge_fp, edge_fn, n_pred, n_est, div=(0, 0, 0)):
    """One per-sample metric row, mirroring per_sample_metrics()."""
    denom = edge_tp + edge_fp + edge_fn
    j = edge_tp / denom if denom else float("nan")
    ratio = (n_pred - n_est) / n_est
    return {
        "edge_tp": edge_tp, "edge_fp": edge_fp, "edge_fn": edge_fn,
        "division_tp": div[0], "division_fp": div[1], "division_fn": div[2],
        "num_pred_nodes": n_pred, "node_recall": 1.0,
        "total_node_ratio": ratio, "edge_jaccard": j,
        "adj_edge_jaccard": max(0.0, j * (1 - ADJUSTMENT_ALPHA * ratio)),
    }


def _tiny_family_imbalance():
    """A small family carrying little edge mass, and a large one carrying most of it.

    Mirrors the real corpus: 44b6 is ~15% of edge mass but 50% of a min-fold criterion.
    """
    small = [_row(80, 10, 10, 1000, 1000, div=(1, 0, 0))]          # mass 100, J 0.80
    large = [_row(600, 200, 200, 10000, 10000, div=(0, 0, 5))]      # mass 1000, J 0.60
    return small, large


def test_summarise_is_edge_mass_weighted_not_family_averaged():
    """The pooled score must track the LARGE family, not sit midway between the two."""
    small, large = _tiny_family_imbalance()
    pooled = summarise(small + large)["score"]
    s_small = summarise(small)["score"]
    s_large = summarise(large)["score"]
    mean_family = 0.5 * (s_small + s_large)

    assert s_small > s_large, "fixture should have the small family scoring higher"
    # pooled must sit far closer to the mass-dominant family than to the family mean
    assert abs(pooled - s_large) < abs(pooled - s_small)
    assert abs(pooled - mean_family) > 1e-3, (
        "pooled score coincides with the family average; the objective distinction is untestable"
    )


def test_pooled_score_reconstructs_from_edge_volume_weights():
    """Rebuild summarise() by hand so the weighting is explicit and auditable."""
    small, large = _tiny_family_imbalance()
    rows = small + large
    num = sum((r["edge_tp"] + r["edge_fp"] + r["edge_fn"]) * r["adj_edge_jaccard"] for r in rows)
    den = sum(r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rows)
    tp = sum(r["division_tp"] for r in rows)
    fp = sum(r["division_fp"] for r in rows)
    fn = sum(r["division_fn"] for r in rows)
    expected = num / den + SCORE_DIVISION_WEIGHT * (tp / (tp + fp + fn))
    assert summarise(rows)["score"] == pytest.approx(expected, abs=1e-12)


def test_bilateral_delta_gate_can_reject_a_pooled_better_candidate():
    """The historical failure mode, in miniature.

    A candidate that loses on the low-mass family but wins big on the high-mass family is
    BETTER on the objective actually being scored, yet a both-families-must-improve gate
    rejects it. This is what closed v122, C_survival, Bp_ilp and B_detpop.
    """
    base_small = [_row(80, 10, 10, 1000, 1000)]
    base_large = [_row(600, 200, 200, 10000, 10000)]
    cand_small = [_row(70, 15, 15, 1000, 1000)]      # worse on the small family
    cand_large = [_row(700, 150, 150, 10000, 10000)]  # much better on the large family

    base_pooled = summarise(base_small + base_large)["score"]
    cand_pooled = summarise(cand_small + cand_large)["score"]
    d_small = summarise(cand_small)["score"] - summarise(base_small)["score"]
    d_large = summarise(cand_large)["score"] - summarise(base_large)["score"]

    assert cand_pooled > base_pooled, "candidate must be better on the pooled objective"
    assert d_small < 0 < d_large, "candidate must lose on the small family and win on the large"
    bilateral_pass = (d_small > 0) and (d_large > 0)
    assert not bilateral_pass, (
        "the bilateral gate accepted this candidate; the fixture no longer reproduces the "
        "historical contradiction"
    )


def test_real_corpus_mass_imbalance_if_cache_present():
    """On the real cache, 44b6 must be a minority of edge mass -- the premise of all of this."""
    import glob
    import json

    files = glob.glob(str(ROOT / "artifacts/kaggle/coupled_cache/scores/A__*.json"))
    if len(files) < 199:
        pytest.skip("coupled per-crop cache not present")
    rows = [json.loads(Path(f).read_text()) for f in files]
    mass = lambda rs: sum(r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rs)  # noqa: E731
    m0 = mass([r for r in rows if r["crop"].startswith("44b6")])
    m1 = mass([r for r in rows if r["crop"].startswith("6bba")])
    share = m0 / (m0 + m1)
    assert 0.10 < share < 0.20, f"44b6 edge-mass share {share:.4f} outside the measured ~0.149"
    assert share < 0.5, "44b6 is a minority of edge mass; 50/50 family weighting is not the metric"
