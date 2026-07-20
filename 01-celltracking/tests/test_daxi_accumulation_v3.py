from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.win_bet.daxi_accumulation_v3 import (  # noqa: E402
    E0cFlowField, crop_bootstrap, choose_threshold, evaluate_threshold,
)


def test_spatial_flow_uses_local_motion_and_frame_mode_uses_frame_median():
    pos, times, edges = {}, {}, []
    node_id = 1
    # Two spatial groups move in opposite Y directions; each has >=3 anchors.
    for y0, displacement in ((0.0, 2.0), (100.0, -2.0)):
        for offset in (0.0, 5.0, 10.0, 15.0):
            source, target = node_id, node_id + 1
            pos[source] = np.array([0.0, y0 + offset, 0.0])
            pos[target] = np.array([0.0, y0 + offset + displacement, 0.0])
            times[source], times[target] = 0, 1
            edges.append((source, target))
            node_id += 2
    field = E0cFlowField(pos, times, edges, neighbours=4, radius_um=20.0)
    local = field.displacement(0, np.array([0.0, 5.0, 0.0]), direction="forward", spatial=True)
    frame = field.displacement(0, np.array([0.0, 5.0, 0.0]), direction="forward", spatial=False)
    assert local[1] > 1.5
    assert abs(frame[1]) < 0.1


def test_crop_bootstrap_weights_crops_not_events():
    rows = ([{"crop": "a", "value": 1.0}] * 100
            + [{"crop": "b", "value": -1.0}])
    result = crop_bootstrap(rows, "value", draws=2000, seed=1)
    assert result["n_crops"] == 2
    assert abs(result["mean"]) < 1e-9


def test_threshold_is_fit_on_train_and_applied_unchanged():
    train = [
        {"pos_spatial_gain5": value, "ctrl_spatial_gain5": control}
        for value, control in zip((0.9, 0.8, 0.7, 0.1), (0.0, 0.0, 0.1, 0.2))
    ]
    selected = choose_threshold(train, max_control_fpr=0.05)
    assert selected is not None
    test = [{"pos_spatial_gain5": 0.75, "ctrl_spatial_gain5": 0.05}]
    result = evaluate_threshold(test, selected[0])
    assert result["threshold"] == selected[0]
    assert result["tpr"] == 1.0 and result["control_fpr"] == 0.0
