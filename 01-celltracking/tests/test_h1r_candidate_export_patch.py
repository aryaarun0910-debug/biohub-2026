from pathlib import Path

import numpy as np
import pytest

from scripts.kaggle_edits import h1r_candidate_export_patch as P


VENDOR = (
    Path(__file__).resolve().parent.parent
    / "vendor"
    / "kaggle-cell-tracking"
    / "scripts"
    / "predict_unet_transformer.py"
)


def test_three_way_logits_retain_argmax_below_threshold():
    probs = np.array([[0.40], [0.35], [0.25]], dtype=np.float32)
    logits = np.array([[1.0], [0.5], [-0.25]], dtype=np.float32)

    rows = P._h1r_select_candidate_rows(
        probs,
        logits,
        threshold=0.48,
        topk=1,
        source_mask=np.ones(3, dtype=bool),
        target_mask=np.ones(1, dtype=bool),
        source_indices=np.array([10, 11, 12]),
        target_indices=np.array([20]),
        source_frame=4,
        target_frame=5,
    )

    assert len(rows) == 1
    assert rows[0][:4] == (10, 20, 4, 5)
    assert rows[0][4] == pytest.approx(1.0)
    assert rows[0][5] == pytest.approx(0.40)
    assert rows[0][6] is False
    assert rows[0][7] == 1


def test_export_is_threshold_union_topk_and_respects_valid_masks():
    probs = np.array(
        [
            [0.60, 0.99],
            [0.95, 0.99],  # masked source: must never export
            [0.30, 0.99],
        ],
        dtype=np.float32,
    )
    logits = np.array([[0.2, 9.0], [5.0, 8.0], [2.0, 7.0]], dtype=np.float32)

    rows = P._h1r_select_candidate_rows(
        probs,
        logits,
        threshold=0.48,
        topk=1,
        source_mask=np.array([True, False, True]),
        target_mask=np.array([True, False]),
        source_indices=np.array([100, 101, 102]),
        target_indices=np.array([200, 201]),
        source_frame=7,
        target_frame=8,
    )

    # source 100 is threshold-qualified; source 102 is the valid top logit.
    assert {(row[0], row[1]) for row in rows} == {(100, 200), (102, 200)}
    by_source = {row[0]: row for row in rows}
    assert by_source[100][6] is True and by_source[100][7] == 0
    assert by_source[102][6] is False and by_source[102][7] == 1
    assert all(row[0] != 101 and row[1] != 201 for row in rows)


def test_non_adjacent_pair_fails_closed():
    with pytest.raises(ValueError, match="adjacent frames"):
        P._h1r_select_candidate_rows(
            np.ones((1, 1)),
            np.ones((1, 1)),
            threshold=0.48,
            topk=1,
            source_mask=np.ones(1, dtype=bool),
            target_mask=np.ones(1, dtype=bool),
            source_indices=np.array([1]),
            target_indices=np.array([2]),
            source_frame=3,
            target_frame=5,
        )


def test_topk_zero_disables_export():
    rows = P._h1r_select_candidate_rows(
        np.array([[0.9]]),
        np.array([[2.0]]),
        threshold=0.48,
        topk=0,
        source_mask=np.ones(1, dtype=bool),
        target_mask=np.ones(1, dtype=bool),
        source_indices=np.array([1]),
        target_indices=np.array([2]),
        source_frame=0,
        target_frame=1,
    )
    assert rows == []


def test_sidecar_is_self_contained_and_columnar(tmp_path):
    path = tmp_path / "movie.edge_candidates.npz"
    rows = [(0, 1, 0, 1, 1.25, 0.4, False, 1)]
    coords = np.array([[0, 2, 3, 4], [1, 5, 6, 7]], dtype=np.int16)
    P._h1r_save_candidate_sidecar(
        path,
        rows,
        coords=coords,
        topk=5,
        threshold=0.48,
        activation="softmax",
    )

    with np.load(path) as sidecar:
        assert sidecar["schema_version"].item() == 1
        assert sidecar["edge_topk"].item() == 5
        np.testing.assert_array_equal(sidecar["node_coords_tzyx"], coords)
        np.testing.assert_array_equal(sidecar["source_index"], [0])
        np.testing.assert_array_equal(sidecar["target_index"], [1])
        assert sidecar["edge_logit"].item() == pytest.approx(1.25)
        assert sidecar["edge_prob"].item() == pytest.approx(0.4)


def test_patch_compiles_is_idempotent_and_preserves_legacy_selection(tmp_path):
    original = VENDOR.read_text(encoding="utf-8")
    assert original.count(P._LEGACY_CANDIDATE_BLOCK) == 1
    target = tmp_path / "predict_unet_transformer.py"
    target.write_text(original, encoding="utf-8")

    P.apply_h1r_candidate_export_patch(target)
    patched = target.read_text(encoding="utf-8")
    compile(patched, str(target), "exec")

    # The final graph still uses the exact vendored threshold/greedy candidate block.
    assert patched.count(P._LEGACY_CANDIDATE_BLOCK) == 1
    assert "candidate_rows = [] if _H1R_EDGE_TOPK > 0 else None" in patched
    assert "if candidate_rows is not None:" in patched

    P.apply_h1r_candidate_export_patch(target)
    assert target.read_text(encoding="utf-8") == patched


def test_patch_fails_closed_on_vendor_drift(tmp_path):
    target = tmp_path / "predict_unet_transformer.py"
    drifted = VENDOR.read_text(encoding="utf-8").replace(
        "if probs[i, j] > cfg.threshold",
        "if probs[i, j] >= cfg.threshold",
        1,
    )
    target.write_text(drifted, encoding="utf-8")

    with pytest.raises(AssertionError, match="matched 0 times"):
        P.apply_h1r_candidate_export_patch(target)
