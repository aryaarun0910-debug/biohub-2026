from __future__ import annotations

import numpy as np
import pytest

from scripts.kaggle_edits.h1r_det_train import (
    DEPLOYED_SIGMOID_THR,
    Zh001rWindows,
    best_operating_point,
    eval_thresholds,
    global_max_nodes,
)


def test_eval_thresholds_includes_deployed_and_rejects_invalid() -> None:
    thresholds = eval_thresholds("0.5,0.9")
    assert "deployed" in thresholds
    assert len(thresholds) == 3
    assert np.isclose(
        thresholds["deployed"],
        np.log(DEPLOYED_SIGMOID_THR / (1.0 - DEPLOYED_SIGMOID_THR)),
    )
    with pytest.raises(ValueError):
        eval_thresholds("0.5,1.0")


def test_best_operating_point_breaks_f1_ties_on_precision() -> None:
    name, stats = best_operating_point({
        "loose": {"f1": 0.8, "precision": 0.7},
        "strict": {"f1": 0.8, "precision": 0.9},
    })
    assert name == "strict"
    assert stats["precision"] == 0.9


def test_global_max_prevents_validation_truncation(tmp_path) -> None:
    np.save(tmp_path / "zh001r_iso.npy", np.zeros((2, 2, 4, 4, 4), dtype=np.uint8))
    np.savez(
        tmp_path / "zh001r_nodes.npz",
        f0=np.array([[0, 0, 0, 0]], dtype=np.float32),
        f1=np.array([[1, 1, 1, 1]], dtype=np.float32),
        f2=np.array([[0, 0, 0, 0]], dtype=np.float32),
        f3=np.array([
            [1, 0, 0, 0], [1, 1, 1, 1], [1, 2, 2, 2],
        ], dtype=np.float32),
    )
    max_nodes = global_max_nodes(tmp_path)
    assert max_nodes == 3
    validation = Zh001rWindows(tmp_path, [1], max_nodes=max_nodes)
    item = validation[0]
    assert item["coords"].shape == (2, 3, 3)
    assert item["masks"].sum().item() == 4
