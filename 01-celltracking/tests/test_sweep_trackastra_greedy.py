from scripts.sweep_trackastra_greedy import agreement_bonus, greedy_edges


def test_greedy_respects_parent_child_capacities_and_threshold():
    candidates = [
        (1, 10, 0.9),
        (2, 10, 0.8),
        (1, 11, 0.7),
        (1, 12, 0.6),
        (3, 13, 0.4),
    ]
    assert greedy_edges(candidates, threshold=0.5, allow_divisions=True) == [(1, 10), (1, 11)]
    assert greedy_edges(candidates, threshold=0.5, allow_divisions=False) == [(1, 10)]


def test_agreement_bonus_changes_only_shared_edge_logit():
    candidates = [(1, 10, 0.5), (1, 11, 0.5)]
    fused = agreement_bonus(candidates, {(1, 11)}, bonus=1.0)
    assert fused[0] == candidates[0]
    assert fused[1][2] > 0.7
