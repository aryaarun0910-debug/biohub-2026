"""The division-verifier substrate must hold its own leakage contract mechanically.

These are SOFTWARE contract tests (CLAUDE.md rule 4). They assert that the leakage guard fires,
that no identifier can enter the feature space, that the feature function cannot encode daughter
order or how a row was constructed, and that the metric adapter's accounting arithmetic is the
one written in its docstring. They decide nothing scientific - the adapter's scientific proof is
reproducing the deployed fold-0 division counts, which needs the real export and lives in
``scripts/win_bet/divverify_adapter.py prove``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import divverify_dataset as dv  # noqa: E402


def _rows(fold, n, label=0, source="pipeline_fork", embryo=None):
    embryo = embryo or dv.FOLD_EMBRYO.get(fold, "ZSNS003")
    return pl.DataFrame({
        "fold": [fold] * n,
        "embryo": [embryo] * n,
        "crop": [f"{embryo}_c{i}" for i in range(n)],
        "source": [source] * n,
        "label": [label] * n,
        **{f: [0.0] * n for f in dv.FEATURES},
    })


# ------------------------------------------------------------------ the leakage contract
def test_features_contain_no_identifier():
    """Guard (2) of the leakage contract: the closed feature list may not carry fold identity."""
    assert not (set(dv.FEATURES) & set(dv.IDENTIFIER_COLUMNS))


def test_training_rows_exclude_the_judged_fold():
    table = pl.concat([_rows(0, 3), _rows(1, 4)], how="vertical")
    for judged in (0, 1):
        train = dv.training_rows_for(table, judged)
        assert judged not in set(train["fold"].to_list())
        dv.assert_no_leak(train, judged)


def test_assert_no_leak_raises_on_a_judged_fold_row():
    """A written rule is not a lock. The lock must actually fire."""
    table = pl.concat([_rows(1, 4), _rows(0, 1)], how="vertical")
    with pytest.raises(RuntimeError, match="LEAK"):
        dv.assert_no_leak(table, 0)


def test_assert_no_leak_raises_on_the_held_out_embryo_even_if_the_fold_column_lies():
    """Fold is a label; the embryo is the physical fact. Both are checked."""
    mislabelled = _rows(1, 3, embryo=dv.FOLD_EMBRYO[0])
    with pytest.raises(RuntimeError, match="held-out embryo"):
        dv.assert_no_leak(mislabelled, 0)


def test_counterfactuals_carry_the_fold_of_the_crop_they_were_built_from():
    """A counterfactual built from a fold-0 crop IS a fold-0 row - falsifier (b) turns on this."""
    ctx = _toy_context()
    seed = [{"mother": 1, "d1": 10, "d2": 11, "t": 0, "source": "gt_tuple"}]
    rows = dv.counterfactual_rows(ctx, seed, "44b6_toy", 0, "44b6", np.random.default_rng(0))
    assert rows, "the counterfactual constructor produced nothing - a silent no-op"
    assert {r["fold"] for r in rows} == {0}
    table = pl.DataFrame(rows).select(["fold", "embryo", "crop", "label"])
    with pytest.raises(RuntimeError, match="LEAK"):
        dv.assert_no_leak(table, 0)


# ------------------------------------------------------------------ the feature function
def _toy_context():
    """A tiny crop: one mother at t=0 with two daughters at t=1, plus filler."""
    ids = np.array([1, 2, 3, 10, 11, 12, 13, 20, 21], dtype=np.int64)
    t = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2], dtype=np.int64)
    pos = np.array([
        [0.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0],
        [0.0, -4.0, 0.0], [0.0, 4.0, 0.0], [0.0, 24.0, 0.0], [0.0, 0.0, 24.0],
        [0.0, -6.0, 0.0], [0.0, 6.0, 0.0],
    ], dtype=np.float64)
    edges = [(1, 10), (1, 11), (2, 12), (3, 13), (10, 20), (11, 21)]
    return dv.CropContext(ids, t, pos, edges)


def test_triple_features_is_symmetric_in_daughter_order():
    """d1/d2 order must be inert, or the label could be read off the ordering convention."""
    ctx = _toy_context()
    a = dv.triple_features(ctx, 1, 10, 11)
    b = dv.triple_features(ctx, 1, 11, 10)
    for f in dv.FEATURES:
        assert a[f] == pytest.approx(b[f]), f"{f} depends on daughter order"


def test_triple_features_rejects_a_degenerate_triple():
    ctx = _toy_context()
    assert dv.triple_features(ctx, 1, 10, 10) is None
    assert dv.triple_features(ctx, 1, 1, 11) is None
    assert dv.triple_features(ctx, 99, 10, 11) is None


def test_triple_features_does_not_depend_on_how_the_row_was_produced():
    """The same triple must produce the same vector whether it arrived as a pipeline fork or as a
    constructed counterfactual - otherwise 'is this a real physics or a generator artifact' is
    unanswerable by construction."""
    ctx = _toy_context()
    seed = [{"mother": 1, "d1": 10, "d2": 11, "t": 0, "source": "gt_tuple"}]
    rows = dv.counterfactual_rows(ctx, seed, "44b6_toy", 0, "44b6", np.random.default_rng(0),
                                  per_seed=4)
    assert rows
    for r in rows:
        direct = dv.triple_features(ctx, r["mother"], r["d1"], r["d2"])
        for f in dv.FEATURES:
            assert r[f] == pytest.approx(direct[f]), f"{f} encodes the construction route"


def test_temporal_counterfactuals_are_actually_temporally_incoherent():
    """The construction must produce what it claims: a skip fork spans two frames, a same-frame
    fork spans none. A mislabelled negative teaches the wrong physics."""
    ctx = _toy_context()
    seed = [{"mother": 1, "d1": 10, "d2": 11, "t": 0, "source": "gt_tuple"}]
    rows = dv.counterfactual_rows(ctx, seed, "44b6_toy", 0, "44b6", np.random.default_rng(0),
                                  per_seed=3)
    skip = [r for r in rows if r["neg_kind"] == "cf_temporal_skip"]
    same = [r for r in rows if r["neg_kind"] == "cf_temporal_same"]
    assert skip and same, "temporal counterfactuals were not constructed"
    assert all(max(r["dt_d1"], r["dt_d2"]) == 2 for r in skip)
    assert all(min(r["dt_d1"], r["dt_d2"]) == 0 for r in same)


def test_wrong_parent_counterfactual_keeps_both_daughters():
    ctx = _toy_context()
    seed = [{"mother": 1, "d1": 10, "d2": 11, "t": 0, "source": "gt_tuple"}]
    rows = dv.counterfactual_rows(ctx, seed, "44b6_toy", 0, "44b6", np.random.default_rng(0),
                                  per_seed=2)
    wp = [r for r in rows if r["neg_kind"] == "cf_wrong_parent"]
    assert wp
    for r in wp:
        assert {r["d1"], r["d2"]} == {10, 11}
        assert r["mother"] != 1


def test_wrong_sister_counterfactual_keeps_the_mother_and_one_daughter():
    ctx = _toy_context()
    seed = [{"mother": 1, "d1": 10, "d2": 11, "t": 0, "source": "gt_tuple"}]
    rows = dv.counterfactual_rows(ctx, seed, "44b6_toy", 0, "44b6", np.random.default_rng(0),
                                  per_seed=2)
    ws = [r for r in rows if r["neg_kind"] == "cf_wrong_sister"]
    assert ws
    for r in ws:
        assert r["mother"] == 1
        assert len({r["d1"], r["d2"]} & {10, 11}) == 1


# ------------------------------------------------------------------ the external contract
def test_external_shared_space_drops_the_constant_columns():
    """dt_* and out_degree are constants in the external table; keeping them would hand a mixed
    model a perfect in-domain/external discriminator."""
    for f in ("dt_d1", "dt_d2", "out_degree"):
        assert f not in dv.EXTERNAL_SHARED
    assert set(dv.EXTERNAL_SHARED) < set(dv.FEATURES)


# ------------------------------------------------------------------ the metric adapter
def test_adapter_accounting_arithmetic():
    import divverify_adapter as da

    ad = object.__new__(da.DivisionMetricAdapter)
    ad.base = {"division_tp": 5, "division_fp": 67, "division_fn": 21}
    ad.tp_forks = {1, 2}
    ad.fp_forks = {10, 11, 12}
    ad.ignored_forks = {90, 91}

    assert ad.predict_counts(set()) == {
        "division_tp": 5, "division_fp": 67, "division_fn": 21,
        "rejections_that_cut_a_charged_fp": 0,
        "rejections_that_cost_a_true_positive": 0,
        "rejections_the_metric_ignores": 0,
    }
    # rejecting a fork the metric ignores must move nothing - the point the adapter exists to make
    inert = ad.predict_counts({90, 91})
    assert (inert["division_tp"], inert["division_fp"], inert["division_fn"]) == (5, 67, 21)
    assert inert["rejections_the_metric_ignores"] == 2
    # rejecting an FP cuts FP; rejecting a TP moves it to FN
    cut = ad.predict_counts({10, 11})
    assert (cut["division_tp"], cut["division_fp"], cut["division_fn"]) == (5, 65, 21)
    cost = ad.predict_counts({1})
    assert (cost["division_tp"], cost["division_fp"], cost["division_fn"]) == (4, 67, 22)


def test_adapter_expectation_matches_the_registry_scope():
    """The one number written outside the registry is a TEST EXPECTATION, so a registry change
    fails loudly here instead of passing on a stale constant (FACT-0375 scope, control arm)."""
    import divverify_adapter as da

    assert da.DEPLOYED_F0_DIVISION == {"division_tp": 5, "division_fp": 67, "division_fn": 21}


def test_adapter_weakest_child_is_deterministic_and_prefers_a_dead_branch():
    import divverify_adapter as da

    ad = object.__new__(da.DivisionMetricAdapter)
    ad.name = "toy"
    ad.pos = {1: np.zeros(3), 10: np.array([0.0, 1.0, 0.0]), 11: np.array([0.0, 9.0, 0.0])}
    ad.children_of = {1: [10, 11], 10: [99]}
    # 11 does not persist, so it goes first even though 10 is nearer
    assert ad.weakest_child(1) == 11
    ad.children_of = {1: [10, 11], 10: [99], 11: [98]}
    # both persist: the LARGER displacement is dropped
    assert ad.weakest_child(1) == 11
    assert ad.weakest_child(1) == 11  # deterministic


def test_adapter_refuses_a_node_that_is_not_a_fork():
    import divverify_adapter as da

    ad = object.__new__(da.DivisionMetricAdapter)
    ad.name = "toy"
    ad.pos = {1: np.zeros(3), 10: np.ones(3)}
    ad.children_of = {1: [10]}
    with pytest.raises(RuntimeError, match="not a fork"):
        ad.weakest_child(1)


# ==================================================================================================
# PKT-0035 - the route-neutral feature space, the association-mistake class, and the operating
# points. Software contracts only: these assert that the guards fire and that the arithmetic is the
# one the docstrings claim. Nothing here decides whether the verifier works.
# ==================================================================================================
def test_route_neutral_excludes_every_route_reading_column():
    """FACT-0385's trap was out_degree and competing_parents reading the CONSTRUCTION ROUTE. The
    fix is mechanical removal, so the removal has to actually hold."""
    assert set(dv.ROUTE_READING) <= set(dv.FEATURES)
    assert not (set(dv.ROUTE_NEUTRAL) & set(dv.ROUTE_READING))
    assert set(dv.ROUTE_NEUTRAL) < set(dv.FEATURES)
    for f in ("out_degree", "competing_parents"):
        assert f not in dv.ROUTE_NEUTRAL


def test_feature_list_is_v1_then_v2_and_carries_no_identifier():
    assert dv.FEATURES == dv.FEATURES_V1 + dv.FEATURES_V2
    assert len(set(dv.FEATURES)) == len(dv.FEATURES), "a feature is listed twice"
    assert not (set(dv.FEATURES) & set(dv.IDENTIFIER_COLUMNS))


def test_external_shared_space_is_pinned_to_v1():
    """The v2 features have no counterpart in the external event table, so a mixed fit can only
    ever happen in the v1 space. Pinning it stops a future v2 feature silently entering."""
    assert set(dv.EXTERNAL_SHARED) < set(dv.FEATURES_V1)


def test_mislinked_by_frame_groups_by_the_target_frame():
    ctx = _toy_context()
    # (1 -> 10) and (2 -> 12) both land at t=1; (10 -> 20) lands at t=2
    mis = dv.mislinked_by_frame(ctx, {(1, 10), (2, 12), (10, 20)})
    assert sorted(mis[1]) == [10, 12]
    assert mis[2] == [20]
    assert dv.mislinked_by_frame(ctx, set()) == {}


def test_assoc_mistake_builds_a_fork_from_a_charged_mislink():
    """The class is seeded from the association layer's own charged errors, so the mother and the
    charged target must both survive into the triple."""
    ctx = _toy_context()
    rows = dv.assoc_mistake_rows(ctx, {(2, 12)}, "44b6_toy", 0, "44b6",
                                 lambda m, n: False, np.random.default_rng(0), per_crop=3)
    assert rows, "the association-mistake constructor produced nothing - a silent no-op"
    for r in rows:
        assert r["neg_kind"] == "cf_assoc_mistake"
        assert r["label"] == 0
        assert r["mother"] == 2
        assert 12 in (r["d1"], r["d2"]), "the charged target did not survive into the triple"
        assert r["derived_from"] == "charged_fp_edge:2:12"


def test_assoc_mistake_skips_a_mother_that_is_already_an_emitted_fork():
    """Node 1 has two children in the toy graph, so its triples ARE the natural FP population and
    building them here would count the same rows twice."""
    ctx = _toy_context()
    assert dv.assoc_mistake_rows(ctx, {(1, 10)}, "44b6_toy", 0, "44b6",
                                 lambda m, n: False, np.random.default_rng(0)) == []


def test_assoc_mistake_refuses_a_link_that_is_not_one_frame_apart():
    """A fork spans exactly one frame gap. A charged edge that does not is not a fork shape."""
    ctx = _toy_context()
    assert dv.assoc_mistake_rows(ctx, {(2, 20)}, "44b6_toy", 0, "44b6",
                                 lambda m, n: False, np.random.default_rng(0)) == []


def test_assoc_mistake_refuses_a_triple_that_is_really_a_division():
    """A constructed negative that is secretly a true division would teach the exact opposite of
    what the class is for, so the guard must refuse it rather than mislabel it."""
    ctx = _toy_context()
    rng = np.random.default_rng(0)
    rows = dv.assoc_mistake_rows(ctx, {(2, 12)}, "44b6_toy", 0, "44b6",
                                 lambda m, n: True, rng)
    assert rows == []


def test_assoc_mistake_fabricates_nothing_without_a_charged_edge():
    ctx = _toy_context()
    assert dv.assoc_mistake_rows(ctx, set(), "44b6_toy", 0, "44b6", lambda m, n: False,
                                 np.random.default_rng(0)) == []


def test_assoc_mistake_respects_the_per_crop_cap():
    ctx = _toy_context()
    rows = dv.assoc_mistake_rows(ctx, {(2, 12), (3, 13)}, "44b6_toy", 0, "44b6",
                                 lambda m, n: False, np.random.default_rng(0), per_crop=1)
    assert len(rows) == 1


def test_assoc_mistake_rows_carry_the_fold_of_the_crop_they_came_from():
    ctx = _toy_context()
    rows = dv.assoc_mistake_rows(ctx, {(2, 12)}, "44b6_toy", 0, "44b6", lambda m, n: False,
                                 np.random.default_rng(0))
    table = pl.DataFrame(rows).select(["fold", "embryo", "crop", "label"])
    with pytest.raises(RuntimeError, match="LEAK"):
        dv.assert_no_leak(table, 0)


# ------------------------------------------------------------------ the verifier's arithmetic
def _vrows(**cols):
    n = len(next(iter(cols.values())))
    base = {"crop": [f"c{i}" for i in range(n)], "mother": list(range(n)),
            "d1": list(range(n)), "d2": list(range(100, 100 + n)), "fold": [0] * n,
            "embryo": ["44b6"] * n, "source": ["pipeline_fork"] * n, "neg_kind": [""] * n}
    base.update(cols)
    return pl.DataFrame(base)


def test_verifier_feature_sets_are_subsets_of_the_closed_list():
    import divverify_verifier as vv

    for name, cols in vv.FEATURE_SETS.items():
        assert set(cols) <= set(dv.FEATURES), f"{name} names a feature that does not exist"
        assert not (set(cols) & set(dv.IDENTIFIER_COLUMNS)), f"{name} carries an identifier"
    assert set(vv.FEATURE_SETS["route_neutral"]) == set(dv.ROUTE_NEUTRAL)


def test_verifier_populations_select_exactly_the_declared_classes():
    import divverify_verifier as vv

    t = _vrows(
        official_label=["tp_fork", "fp_fork", "gt_positive", "ignored", "constructed_negative",
                        "constructed_negative"],
        label=[1, 0, 1, 0, 0, 0],
    ).with_columns(
        pl.Series("source", ["pipeline_fork"] * 4 + ["counterfactual"] * 2),
        pl.Series("neg_kind", ["", "natural_fp", "", "ignored_fork", "cf_wrong_parent",
                               "cf_assoc_mistake"]),
    )
    assert sorted(vv.population_rows(t, "emitted_charged")["official_label"].to_list()) == \
        ["fp_fork", "tp_fork"]
    assert sorted(vv.population_rows(t, "genuine_tuples")["official_label"].to_list()) == \
        ["fp_fork", "gt_positive", "tp_fork"]
    assert vv.population_rows(t, "genuine_plus_cf").height == 5
    kinds = set(vv.population_rows(t, "genuine_plus_assoc")["neg_kind"].to_list())
    assert "cf_assoc_mistake" in kinds and "cf_wrong_parent" not in kinds
    # the ignored forks never enter any training population
    for pop in vv.POPULATIONS:
        assert "ignored" not in vv.population_rows(t, pop)["official_label"].to_list()


def test_verifier_deployment_population_is_the_charged_forks_only():
    import divverify_verifier as vv

    t = _vrows(official_label=["tp_fork", "fp_fork", "ignored", "gt_positive"],
               label=[1, 0, 0, 1]).with_columns(
        pl.Series("source", ["pipeline_fork"] * 3 + ["gt_tuple"]))
    assert sorted(vv.deployment_rows(t, 0)["official_label"].to_list()) == ["fp_fork", "tp_fork"]
    # ... while the DEPLOYABLE rule sees every emitted fork, ignored ones included (FACT-0383)
    assert vv.all_emitted_forks(t, 0).height == 3


def test_verifier_operating_point_reports_a_lost_true_fork_as_a_failure():
    """PRESERVATION IS THE PRIMARY CONSTRAINT. An operating point that costs a true fork is a
    failure of that operating point and must not be netted off against the FPs it cut."""
    import divverify_verifier as vv

    y = np.array([1, 0, 0, 0])
    p = np.array([0.10, 0.01, 0.02, 0.90])
    ok = vv.operating_point(y, p, 0.05)
    assert ok["true_forks_lost"] == 0 and ok["false_forks_cut"] == 2
    assert ok["preserves_every_true_fork"] and ok["verdict"] == "OK"
    bad = vv.operating_point(y, p, 0.5)
    assert bad["true_forks_lost"] == 1
    assert not bad["preserves_every_true_fork"]
    assert "FAILS the primary constraint" in bad["verdict"]


def test_verifier_transferred_threshold_comes_from_the_training_positives_only():
    import divverify_verifier as vv

    y = np.array([1, 1, 0, 0])
    oof = np.array([0.4, 0.7, 0.9, 0.1])
    assert vv.threshold_from_training(y, oof) == pytest.approx(0.4)
    assert vv.threshold_from_training(np.array([0, 0]), np.array([0.5, 0.5])) == 0.0


def test_verifier_oracle_cut_counts_false_forks_before_the_first_true_one():
    import divverify_verifier as vv

    y = np.array([0, 0, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    r = vv.oracle_cut(y, s)
    assert r["fp_rejected_before_losing_any_tp"] == 2
    assert r["fp_rejected_before_losing_a_second_tp"] == 3
    assert r["is_an_oracle"] is True
    # a fold where every false fork sorts below every true one is the perfect case
    perfect = vv.oracle_cut(np.array([0, 0, 1]), np.array([0.1, 0.2, 0.9]))
    assert perfect["fp_rejected_before_losing_any_tp"] == 2


def test_adapter_paired_bootstrap_refuses_unpaired_arms():
    import divverify_adapter as da

    with pytest.raises(RuntimeError, match="paired bootstrap"):
        da._paired_bootstrap([{"score": 1.0}], [], lambda rs: {"score": 0.0})
    with pytest.raises(RuntimeError, match="paired bootstrap"):
        da._paired_bootstrap([], [], lambda rs: {"score": 0.0})
