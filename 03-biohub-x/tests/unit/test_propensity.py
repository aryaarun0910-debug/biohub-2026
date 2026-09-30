"""The propensity probe: does it see a planted propensity, and does it stay quiet without one?

Two synthetic worlds. In one, "annotated" depends only on the DoG response, so
every other feature is noise and the probe must report no signal. In the other,
"annotated" depends on position and persistence beyond the response, and the
probe must report a signal and name the temporal group. The decision table is
tested on its own, because it is applied mechanically and a wrong branch there
spends or withholds GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from biohubx.evaluation.propensity import (
    FEATURE_NAMES,
    CandidateTable,
    ProbeParameters,
    PropensityError,
    auc,
    columns_for,
    decide,
    fit_logistic,
    held_out_auc,
    probe,
)

PARAMETERS = ProbeParameters(
    density_radius_um=10.0,
    persistence_radius_um=5.0,
    motion_cap_um=15.0,
    l2=0.01,
    permutations=12,
    seed=0,
    null_quantile=0.95,
)


def _world(rng: np.random.Generator, *, movies: int, rows: int, propensity: bool) -> list[CandidateTable]:
    tables = []
    for m in range(movies):
        features = rng.normal(size=(rows, len(FEATURE_NAMES)))
        features[:, FEATURE_NAMES.index("persist_prev")] = rng.integers(0, 2, size=rows)
        features[:, FEATURE_NAMES.index("persist_next")] = rng.integers(0, 2, size=rows)
        logit = 2.0 * features[:, FEATURE_NAMES.index("dog")] - 4.0
        if propensity:
            logit = (
                logit
                + 2.5 * features[:, FEATURE_NAMES.index("z")]
                + 3.0 * features[:, FEATURE_NAMES.index("persist_next")]
            )
        labels = rng.random(rows) < 1.0 / (1.0 + np.exp(-logit))
        if labels.sum() == 0:
            labels[0] = True
        tables.append(
            CandidateTable(
                dataset=f"m{m}",
                first_frame=0,
                frames=10,
                features=features,
                labels=labels,
                estimate=float(rows) / 4,
                annotated_nodes=int(labels.sum()),
            )
        )
    return tables


def test_auc_is_the_mann_whitney_statistic() -> None:
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([False, False, True, True])
    # Positives 0.35 and 0.8 against unlabelled 0.1 and 0.4: three of four pairs ordered.
    assert auc(scores, labels) == pytest.approx(0.75)
    assert auc(np.array([1.0, 1.0, 1.0]), np.array([True, False, True])) == pytest.approx(0.5)


def test_auc_refuses_a_single_class() -> None:
    with pytest.raises(PropensityError, match="at least one positive"):
        auc(np.array([0.1, 0.2]), np.array([False, False]))


def test_the_logistic_fit_recovers_a_planted_direction() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4000, 3))
    y = rng.random(4000) < 1.0 / (1.0 + np.exp(-(3.0 * x[:, 0] - 2.0)))
    weights = fit_logistic(x, y, l2=0.01)
    assert weights[0] > 1.0
    assert abs(weights[1]) < 0.4 and abs(weights[2]) < 0.4


def test_held_out_evaluation_is_grouped_by_movie() -> None:
    """A movie's own labels must not inform the model that scores it."""
    tables = _world(np.random.default_rng(2), movies=3, rows=600, propensity=False)
    result = held_out_auc(tables, columns_for(["control"]), l2=0.01)
    assert set(result["per_movie"]) == {"m0", "m1", "m2"}
    assert 0.5 < result["pooled"] <= 1.0
    with pytest.raises(PropensityError, match="at least two movies"):
        held_out_auc(tables[:1], columns_for(["control"]), l2=0.01)


def test_a_world_without_propensity_reports_no_signal() -> None:
    tables = _world(np.random.default_rng(3), movies=4, rows=800, propensity=False)

    report = probe(tables, PARAMETERS)

    assert report["decision"]["verdict"] == "propensity_signal_absent"
    assert report["null"]["all"]["signal"] is False
    assert report["positives_total"] > 0


def test_a_planted_propensity_is_seen_and_the_temporal_group_is_named() -> None:
    tables = _world(np.random.default_rng(4), movies=4, rows=800, propensity=True)

    report = probe(tables, PARAMETERS)

    assert report["decision"]["verdict"] == "propensity_signal_present"
    assert report["null"]["all"]["signal"] is True
    assert report["null"]["temporal"]["signal"] is True
    assert report["delta_auc"]["all"] > report["null"]["all"]["threshold"]
    assert report["decision"]["A4"].startswith("hold; neighbouring")


def test_the_probe_refuses_windows_with_no_positive_anywhere() -> None:
    tables = _world(np.random.default_rng(5), movies=2, rows=100, propensity=False)
    for table in tables:
        table.labels[:] = False
    with pytest.raises(PropensityError, match="no positives"):
        probe(tables, PARAMETERS)


def test_one_embryo_never_kills_the_objective_on_its_own() -> None:
    absent = decide(signal=False, temporal=False, non_temporal=False)
    assert absent["verdict"] == "propensity_signal_absent"
    assert "continue" in absent["next_action"]

    temporal = decide(signal=True, temporal=True, non_temporal=True)
    assert temporal["verdict"] == "propensity_signal_present"
    assert temporal["A4"].startswith("hold; neighbouring")
    assert "weak, embryo-specific" in temporal["next_action"]
    assert "residual-propensity test" in temporal["A3"]

    spatial_only = decide(signal=True, temporal=False, non_temporal=True)
    assert spatial_only["A4"].startswith("hold pending")


def test_column_selection_keeps_the_control_first_and_never_duplicates() -> None:
    cols = columns_for(["control", "temporal", "control"])
    assert cols[0] == FEATURE_NAMES.index("dog")
    assert len(cols) == len(set(cols)) == 4
