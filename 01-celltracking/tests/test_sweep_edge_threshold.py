from scripts.sweep_edge_threshold import selected_edges


def test_selected_edges_thresholds_and_caps_each_parent():
    rows = [(1, 10, 0.7), (1, 11, 0.9), (2, 20, 0.4), (2, 21, 0.8)]
    assert set(selected_edges(rows, threshold=0.5, max_children=1)) == {(1, 11), (2, 21)}
    assert set(selected_edges(rows, threshold=0.5, max_children=2)) == {
        (1, 10),
        (1, 11),
        (2, 21),
    }
