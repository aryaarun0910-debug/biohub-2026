from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from scripts.win_bet.validate_detpeak_export import validate


def _csv(path, x=1):
    pl.DataFrame({"id": [0], "dataset": ["crop"], "row_type": ["node"],
                  "node_id": [1], "t": [0], "z": [0], "y": [0], "x": [x],
                  "source_id": [-1], "target_id": [-1]}).write_csv(path)


def test_export_validation_checks_heartbeat_and_graph_parity(tmp_path):
    sidecar = tmp_path / "crop.npz"
    logits = np.asarray([1.0, 3.0], dtype=np.float32)
    np.savez(sidecar, t=[0, 0], zyx=np.zeros((2, 3)), logit=logits,
             pipeline_threshold=0.8, pipeline_peak_count=1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"graph_unchanged": True}))
    smoke, control = tmp_path / "smoke.csv", tmp_path / "control.csv"
    _csv(smoke); _csv(control)
    got = validate([sidecar], smoke, control, manifest)
    assert got["pass"] and got["observational_graph_parity"]

    _csv(control, x=2)
    with pytest.raises(ValueError, match="parity"):
        validate([sidecar], smoke, control, manifest)
