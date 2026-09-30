"""Software contract for the full-export audit gate.

The load-bearing claim of ``audit_detpeak_full_export.py`` is that it distinguishes a
divergence AT detection (which invalidates the sidecars) from one strictly downstream of it
(which does not). These tests pin that distinction and the sidecar reconstruction contract.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scripts.win_bet.audit_detpeak_full_export import (
    audit_graph_parity,
    audit_sidecars,
    audit_stages,
)

STATS_COLUMNS = ("dataset", "raw_nodes", "raw_edges", "short_track_nodes_removed", "nodes", "edges")


def _stats(path, raw_nodes=10, short_track_nodes_removed=2, nodes=8):
    pl.DataFrame({
        "dataset": ["crop"],
        "raw_nodes": [raw_nodes],
        "raw_edges": [9],
        "short_track_nodes_removed": [short_track_nodes_removed],
        "nodes": [nodes],
        "edges": [7],
    }).write_csv(path)


def _sidecar(path, logits, threshold, count):
    np.savez(
        path,
        t=np.zeros(len(logits), dtype=np.int16),
        zyx=np.zeros((len(logits), 3), dtype=np.int16),
        logit=np.asarray(logits, dtype=np.float32),
        pipeline_threshold=np.asarray(threshold, dtype=np.float64),
        pipeline_peak_count=np.asarray(count, dtype=np.int64),
    )


def _csv(path, x=1, rows=1):
    pl.DataFrame({
        "id": list(range(rows)),
        "dataset": ["crop"] * rows,
        "row_type": ["node"] * rows,
        "node_id": list(range(rows)),
        "t": [0] * rows,
        "z": [0] * rows,
        "y": [0] * rows,
        "x": [x] * rows,
        "source_id": [-1] * rows,
        "target_id": [-1] * rows,
    }).write_csv(path)


def test_sidecar_reconstruction_is_enforced(tmp_path):
    peaks = tmp_path / "peaks"
    peaks.mkdir()
    # sigmoid(3.0) > 0.8 but sigmoid(1.0) is not, so exactly one peak survives
    _sidecar(peaks / "crop.npz", [1.0, 3.0], 0.8, 1)
    got = audit_sidecars(peaks)
    assert got["n_sidecars"] == 1
    assert got["pipeline_peaks_total"] == 1
    assert got["exported_peaks_total"] == 2

    _sidecar(peaks / "crop.npz", [1.0, 3.0], 0.8, 2)
    with pytest.raises(ValueError, match="peak count mismatch"):
        audit_sidecars(peaks)


def test_sidecars_must_agree_on_one_pipeline_threshold(tmp_path):
    peaks = tmp_path / "peaks"
    peaks.mkdir()
    _sidecar(peaks / "a.npz", [3.0], 0.8, 1)
    _sidecar(peaks / "b.npz", [3.0], 0.9, 1)
    with pytest.raises(ValueError, match="disagree on the pipeline threshold"):
        audit_sidecars(peaks)


def test_downstream_divergence_keeps_the_detection_contract(tmp_path):
    """A short-track-filter difference must NOT invalidate the candidate export."""
    control, export = tmp_path / "c.csv", tmp_path / "e.csv"
    _stats(control, raw_nodes=10, short_track_nodes_removed=2, nodes=8)
    _stats(export, raw_nodes=10, short_track_nodes_removed=1, nodes=9)
    got = audit_stages(control, export)
    assert got["detection_identical"] is True
    assert "detection" not in got["stages_differing"]
    assert got["earliest_differing_stage_per_crop"]["crop"] == "short_track_filter"


def test_detection_divergence_breaks_the_contract(tmp_path):
    """A raw_nodes difference means the sidecars describe a different substrate."""
    control, export = tmp_path / "c.csv", tmp_path / "e.csv"
    _stats(control, raw_nodes=10)
    _stats(export, raw_nodes=11)
    got = audit_stages(control, export)
    assert got["detection_identical"] is False
    assert got["earliest_differing_stage_per_crop"]["crop"] == "detection"


def test_graph_parity_reports_per_crop_deltas(tmp_path):
    control, export = tmp_path / "c.csv", tmp_path / "e.csv"
    _csv(control, x=1, rows=1)
    _csv(export, x=1, rows=1)
    assert audit_graph_parity(control, export)["exact_parity"] is True

    _csv(export, x=1, rows=2)
    got = audit_graph_parity(control, export)
    assert got["exact_parity"] is False
    assert got["divergent_crops"]["crop"]["node_delta"] == 1
