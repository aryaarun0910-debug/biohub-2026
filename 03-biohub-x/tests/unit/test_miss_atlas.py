"""The miss classification is an ordered rule over measured features; hold the order and the thresholds.

A miss earns every flag its features justify, and its primary class is the
first flag in CLASSES order. The order is part of the instrument: a cell that
is both near a face and dim is a boundary miss, because a face effect explains
the dimness and not the reverse.
"""

from __future__ import annotations

import math

from biohubx.evaluation.miss_atlas import CLASSES, MissRecord, Thresholds, classify, summarise


def record(**overrides: object) -> MissRecord:
    base: dict[str, object] = {
        "dataset": "6bba_test",
        "embryo": "6bba",
        "frame": 3,
        "node_id": 7,
        "voxel_zyx": (10.0, 100.0, 100.0),
        "physical_um_zyx": (16.25, 40.6, 40.6),
        "intensity_percentile": 0.95,
        "response_by_scale": {"2um": 0.1, "3um": 0.2},
        "response_max": 0.2,
        "response_percentile": 0.97,
        "frame_cutoff": 0.15,
        "nearest_proposal_um": 25.0,
        "nearest_vector_um_zyx": (0.0, 0.0, 25.0),
        "nearest_proposal_matched_to_other": False,
        "boundary_um": 30.0,
        "neighbour_distances_um": [20.0, 25.0],
        "neighbours_within_10um": 0,
        "has_previous": True,
        "has_next": True,
        "previous_matched": False,
        "next_matched": False,
        "local_max_within_radius": False,
        "local_max_um": math.inf,
        "local_max_response": float("nan"),
        "local_max_below_cutoff": False,
    }
    base.update(overrides)
    return MissRecord(**base)  # type: ignore[arg-type]


def test_a_miss_with_no_explanation_is_unexplained() -> None:
    flags, primary = classify(record(), Thresholds())
    assert flags == [] and primary == "unexplained"


def test_boundary_outranks_dim_and_both_flags_are_kept() -> None:
    flags, primary = classify(record(boundary_um=3.0, intensity_percentile=0.5), Thresholds())
    assert primary == "boundary"
    assert flags == ["boundary", "dim"]


def test_merged_needs_a_nearby_proposal_that_serves_another_cell_or_a_close_neighbour() -> None:
    thresholds = Thresholds()
    shared = record(nearest_proposal_um=9.0, nearest_proposal_matched_to_other=True)
    assert classify(shared, thresholds)[1] == "merged"
    crowded = record(nearest_proposal_um=9.0, neighbour_distances_um=[5.0, 30.0])
    assert classify(crowded, thresholds)[1] == "merged"
    alone = record(nearest_proposal_um=9.0)
    flags, primary = classify(alone, thresholds)
    assert primary == "localization" and "merged" not in flags


def test_localisation_is_beyond_the_official_radius_and_within_fourteen() -> None:
    assert classify(record(nearest_proposal_um=14.0), Thresholds())[1] == "localization"
    assert classify(record(nearest_proposal_um=14.1), Thresholds())[1] == "unexplained"


def test_temporal_dropout_requires_both_existing_neighbours_matched() -> None:
    both = record(previous_matched=True, next_matched=True)
    assert classify(both, Thresholds())[1] == "temporal_dropout"
    one_side = record(previous_matched=True, next_matched=None, has_next=False)
    assert classify(one_side, Thresholds())[1] == "temporal_dropout"
    half = record(previous_matched=True, next_matched=False)
    assert "temporal_dropout" not in classify(half, Thresholds())[0]


def test_low_response_means_a_discarded_local_maximum_within_the_radius() -> None:
    kept = record(
        local_max_within_radius=True, local_max_um=3.0, local_max_response=0.3, local_max_below_cutoff=False
    )
    assert classify(kept, Thresholds())[1] == "unexplained"
    dropped = record(
        local_max_within_radius=True, local_max_um=3.0, local_max_response=0.1, local_max_below_cutoff=True
    )
    assert classify(dropped, Thresholds())[1] == "low_response"


def test_annotation_ambiguity_is_an_isolated_annotation_with_no_peak_at_all() -> None:
    isolated = record(has_previous=False, has_next=False, previous_matched=None, next_matched=None)
    assert classify(isolated, Thresholds())[1] == "annotation_ambiguity"


def test_summary_counts_every_class_in_order() -> None:
    rows = [record(boundary_um=1.0), record(intensity_percentile=0.1), record()]
    for r in rows:
        r.flags, r.primary_class = classify(r, Thresholds())
    summary = summarise(rows, annotated_nodes=30)
    assert summary["misses"] == 3 and summary["miss_fraction"] == 0.1
    assert list(summary["primary_class_counts"]) == list(CLASSES)
    assert summary["primary_class_counts"]["boundary"] == 1
    assert summary["primary_class_counts"]["dim"] == 1
    assert summary["primary_class_counts"]["unexplained"] == 1
