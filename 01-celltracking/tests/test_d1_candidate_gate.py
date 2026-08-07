import numpy as np

import d1_candidate_gate as G


def test_weighted_auc_respects_horvitz_mass_and_ties():
    # Positive 2 beats the negative at 1 (weight 3), ties the negative at 2 (weight 1),
    # and loses to the negative at 3 (weight 2): (3 + .5) / 6.
    got = G._weighted_auc(
        np.array([2.0]), np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]))
    assert got == (3.0 + 0.5) / 6.0


def test_frontier_reports_recall_at_deployable_precision():
    row = G._frontier(
        np.array([5.0, 4.0]), np.array([3.0, 2.0]), np.array([1.0, 1.0]))
    assert row["recall_at_precision"]["0.90"] == 1.0
    assert row["operating_points"][0]["precision_ht"] == 1.0
