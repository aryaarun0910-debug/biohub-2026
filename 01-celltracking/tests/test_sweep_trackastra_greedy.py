from scripts.sweep_trackastra_greedy import greedy_edges


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
