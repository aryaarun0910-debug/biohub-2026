"""Focused CPU-small tests for the PKT-0049 BIOHUB-X graph-owning matcher."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.win_bet.biohubx import contract as C
from scripts.win_bet.biohubx import model as M


def small_model(*, depth: int = 4) -> M.BiohubXMatcher:
    torch.manual_seed(7)
    return M.BiohubXMatcher(
        M.MatcherConfig(
            node_feature_dim=5,
            model_dim=16,
            depth=depth,
            temporal_window=2,
            num_heads=4,
            feedforward_dim=24,
            dropout=0.0,
        )
    )


def small_inputs():
    source_features = torch.tensor(
        [[1.0, 0.0, 0.2, 0.3, 0.4], [0.1, 1.0, 0.7, 0.2, 0.5]], dtype=torch.float32
    )
    target_features = torch.tensor(
        [[0.9, 0.1, 0.3, 0.2, 0.6], [0.2, 0.8, 0.6, 0.4, 0.1], [0.4, 0.2, 0.1, 0.9, 0.7]],
        dtype=torch.float32,
    )
    source_pos = torch.tensor([[0.0, 1.0, 2.0], [1.625, 3.0, 4.0]], dtype=torch.float32)
    target_pos = torch.tensor(
        [[0.0, 1.4, 2.4], [1.625, 3.4, 4.4], [3.25, 8.0, 9.0]], dtype=torch.float32
    )
    # Sparse offered surface: target 0 sees both sources; targets 1 and 2 see source 1 only.
    candidate_source = torch.tensor([0, 1, 1, 1], dtype=torch.long)
    candidate_target = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    return source_features, target_features, source_pos, target_pos, candidate_source, candidate_target


def manual_output() -> M.MatcherOutput:
    # Four offered edges across three targets and two sources. Target 0 continues source 0;
    # targets 1 and 2 are the two division-labelled daughters of source 1.
    logits = torch.tensor(
        [
            [8.0, -4.0, -4.0],
            [-4.0, -4.0, 8.0],
            [-4.0, 8.0, -4.0],
            [-4.0, 7.0, -4.0],
        ],
        dtype=torch.float32,
    )
    return M.MatcherOutput(
        pair_logits=logits,
        no_parent_logits=torch.tensor([-5.0, -4.0, -3.0]),
        candidate_source_index=torch.tensor([0, 1, 1, 1], dtype=torch.long),
        candidate_target_index=torch.tensor([0, 0, 1, 2], dtype=torch.long),
        n_source=2,
        n_target=3,
    )


def test_temporal_window_and_transformer_depth_are_distinct_configuration_axes():
    for depth in M.SUPPORTED_DEPTHS:
        cfg = M.MatcherConfig(node_feature_dim=5, depth=depth, temporal_window=2)
        assert cfg.depth == depth
        assert cfg.temporal_window == 2

    with pytest.raises(ValueError, match="T is not transformer depth D"):
        M.MatcherConfig(node_feature_dim=5, depth=4, temporal_window=5)
    with pytest.raises(ValueError, match="depth D"):
        M.MatcherConfig(node_feature_dim=5, depth=5, temporal_window=2)


def test_cpu_small_forward_has_requested_depth_and_explicit_no_parent_logit():
    model = small_model(depth=4)
    sf, tf, sp, tp, cs, ct = small_inputs()
    output = model(sf, tf, sp, tp, candidate_source_index=cs, candidate_target_index=ct)
    assert len(model.encoder.layers) == 4
    assert output.pair_logits.shape == (4, 3)
    assert output.no_parent_logits.shape == (3,)
    assert M.TEMPORAL_WINDOW == 2
    assert C.CLASSES == ("continuation", "division", "neither")


def test_both_pair_head_and_no_parent_head_receive_gradient():
    model = small_model()
    sf, tf, sp, tp, cs, ct = small_inputs()
    output = model(sf, tf, sp, tp, candidate_source_index=cs, candidate_target_index=ct)
    loss = output.pair_logits.square().mean() + output.no_parent_logits.square().mean()
    loss.backward()
    assert model.pair_head[-1].weight.grad is not None
    assert model.no_parent_head[-1].weight.grad is not None
    assert float(model.pair_head[-1].weight.grad.abs().sum()) > 0.0
    assert float(model.no_parent_head[-1].weight.grad.abs().sum()) > 0.0


def test_candidate_rows_include_one_learned_no_parent_row_and_pass_v1_contract():
    model = small_model()
    sf, tf, sp, tp, cs, ct = small_inputs()
    output = model(sf, tf, sp, tp, candidate_source_index=cs, candidate_target_index=ct)
    table = model.candidate_table(
        output,
        crop="44b6_cpu_smoke",
        t_target=1,
        source_uids=np.array([10, 11], dtype=np.int64),
        target_uids=np.array([20, 21, 22], dtype=np.int64),
    )
    assert table["source_uid"].tolist().count(C.NO_PARENT) == 3
    assert len(table["source_uid"]) == 4 + 3  # offered sparse edges + one abstain row/target
    np_rows = table["source_uid"] == C.NO_PARENT
    assert np.unique(table["p_neither"][np_rows]).size > 1
    assert np.allclose(
        table["p_continuation"] + table["p_division"] + table["p_neither"], 1.0
    )
    offered = list(
        zip(table["crop"], table["t_target"], table["target_uid"], table["source_uid"])
    )
    report = C.verify_candidates(table, offered_pairs=offered)
    assert report["contract"] == "biohubx_io_v1"
    assert report["n_targets"] == 3
    assert report["coverage"]["n_uncovered"] == 0


def test_explicit_no_parent_logit_can_make_a_target_a_root():
    pair_logits = torch.tensor([[6.0, -2.0, -3.0]], dtype=torch.float32)
    output = M.MatcherOutput(
        pair_logits=pair_logits,
        no_parent_logits=torch.tensor([12.0]),
        candidate_source_index=torch.tensor([0], dtype=torch.long),
        candidate_target_index=torch.tensor([0], dtype=torch.long),
        n_source=1,
        n_target=1,
    )
    decisions = M.BiohubXMatcher.select_parents(output)
    assert decisions.parent_index.tolist() == [C.NO_PARENT]
    assert decisions.relation_index.tolist() == [C.CLASSES.index("neither")]


def test_forward_refuses_duplicate_or_out_of_bounds_sparse_candidates():
    model = small_model()
    sf, tf, sp, tp, cs, ct = small_inputs()
    with pytest.raises(ValueError, match="unique"):
        model(sf, tf, sp, tp, candidate_source_index=torch.tensor([0, 0]),
              candidate_target_index=torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="source index is out of bounds"):
        model(sf, tf, sp, tp, candidate_source_index=torch.tensor([2]),
              candidate_target_index=torch.tensor([0]))


def test_matcher_owns_a_legal_division_graph_without_incumbent_relink():
    model = small_model()
    owned = model.emit_graph(
        manual_output(),
        crop="44b6_cpu_smoke",
        source_uids=np.array([10, 11], dtype=np.int64),
        target_uids=np.array([20, 21, 22], dtype=np.int64),
        source_zyx_vox=np.array([[0, 2, 3], [1, 6, 7]], dtype=np.int64),
        target_zyx_vox=np.array([[0, 3, 4], [1, 7, 8], [1, 8, 9]], dtype=np.int64),
        t_source=0,
        t_target=1,
    )
    assert owned.table["parent_uid"].tolist() == [-1, -1, 10, 11, 11]
    assert owned.decisions.relation_index.tolist() == [0, 1, 1]
    assert all("relink" not in stage.lower() for stage in owned.consumer_chain)
    report = C.verify_graph(owned.table, consumer_chain=owned.consumer_chain)
    assert report["divisions"] == 1
    assert report["max_out_degree"] == 2
    assert report["relink_applied"] is False


def test_graph_owner_is_bound_to_candidate_graph_and_clean_source_bytes():
    """A consumer-chain string alone is not provenance; bind the producer and both artifacts."""
    model = small_model()
    output = manual_output()
    candidates = model.candidate_table(
        output,
        crop="44b6_cpu_smoke",
        t_target=1,
        source_uids=np.array([10, 11], dtype=np.int64),
        target_uids=np.array([20, 21, 22], dtype=np.int64),
    )
    owned = model.emit_graph(
        output,
        crop="44b6_cpu_smoke",
        source_uids=np.array([10, 11], dtype=np.int64),
        target_uids=np.array([20, 21, 22], dtype=np.int64),
        source_zyx_vox=np.array([[0, 2, 3], [1, 6, 7]], dtype=np.int64),
        target_zyx_vox=np.array([[0, 3, 4], [1, 7, 8], [1, 8, 9]], dtype=np.int64),
        t_source=0,
        t_target=1,
    )
    receipt = C.make_consumer_receipt(
        producer_path=M.__file__,
        producer_symbol="emit_graph",
        candidates=candidates,
        graph=owned.table,
        consumer_chain=owned.consumer_chain,
    )
    report = C.verify_consumer_provenance(receipt, candidates=candidates, graph=owned.table)
    assert report["source_scanned_for_incumbent"] is True
    assert report["artifacts_content_bound"] is True


def test_emit_graph_refuses_fractional_export_and_non_adjacent_frames():
    model = small_model()
    kwargs = dict(
        output=manual_output(),
        crop="44b6_cpu_smoke",
        source_uids=np.array([10, 11], dtype=np.int64),
        target_uids=np.array([20, 21, 22], dtype=np.int64),
        source_zyx_vox=np.array([[0, 2, 3], [1, 6, 7]], dtype=np.int64),
        target_zyx_vox=np.array([[0, 3, 4], [1, 7, 8], [1, 8, 9]], dtype=np.float64),
        t_source=0,
        t_target=1,
    )
    kwargs["target_zyx_vox"][0, 2] = 4.5
    with pytest.raises(ValueError, match="integer voxel"):
        model.emit_graph(**kwargs)

    kwargs["target_zyx_vox"][0, 2] = 4.0
    kwargs["t_target"] = 2
    with pytest.raises(ValueError, match="exactly one frame"):
        model.emit_graph(**kwargs)
