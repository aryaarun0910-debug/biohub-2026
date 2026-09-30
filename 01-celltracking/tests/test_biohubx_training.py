"""End-to-end CPU software smoke for the radical Biohub-X T=2 path."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.win_bet.biohubx import contract as C
from scripts.win_bet.biohubx import model as M
from scripts.win_bet.biohubx import synthetic_t2 as S
from scripts.win_bet.biohubx import training as T


def fixture_batch(*, dropout: float = 0.25) -> dict:
    i = np.arange(24, dtype=np.float64)
    points = np.column_stack((2 + i % 8, 10 + (5 * i) % 70, 20 + (7 * i) % 100))
    crops = np.asarray(["44b6_software"] * len(points))
    cfg = S.WarpConfig(
        seed=71,
        k_hard_negatives=4,
        source_dropout=dropout,
        affine_zyx=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation_um_zyx=(0.0, 0.8125, 0.40625),
        local_control_points=0,
        local_amplitude_um_zyx=(0.0, 0.0, 0.0),
        jitter_std_um_zyx=(0.0, 0.0, 0.0),
    )
    displacement = np.tile(np.asarray(cfg.translation_um_zyx), (len(points), 1))
    stats = S.displacement_statistics(displacement)
    calibration = S.CalibrationTarget(
        name="software_fixture_not_science",
        median_um=stats["median_um"],
        p99_um=stats["p99_um"],
        per_axis_median_um_zyx=tuple(stats["per_axis_median_um_zyx"]),
        median_tolerance_um=1e-12,
        p99_relative_tolerance=1e-12,
        per_axis_tolerance_um_zyx=(1e-12, 1e-12, 1e-12),
    )
    return S.generate_t2(
        points,
        np.arange(500, 500 + len(points), dtype=np.int64),
        crops,
        split_name="train",
        split_by_crop={"44b6_software": "train", "6bba_software": "validation"},
        calibration_target=calibration,
        config=cfg,
    )


def make_model(depth: int) -> M.BiohubXMatcher:
    torch.manual_seed(101)
    return M.BiohubXMatcher(M.MatcherConfig(
        node_feature_dim=8,
        model_dim=16,
        depth=depth,
        num_heads=4,
        feedforward_dim=24,
        dropout=0.0,
    ))


@pytest.mark.parametrize("depth", M.SUPPORTED_DEPTHS)
def test_depth_ladder_forward_backward_uses_one_sparse_surface(depth: int):
    prepared = T.prepare_synthetic_t2(fixture_batch())
    model = make_model(depth)
    output = model(
        prepared.source_features,
        prepared.target_features,
        prepared.source_positions_um,
        prepared.target_positions_um,
        candidate_source_index=prepared.candidate_source_index,
        candidate_target_index=prepared.candidate_target_index,
    )
    loss = T.supervised_t2_loss(output, prepared)
    loss.total.backward()
    assert output.pair_logits.shape[0] == prepared.candidate_source_index.numel()
    assert loss.n_positive_pairs + loss.n_negative_pairs == output.pair_logits.shape[0]
    assert loss.n_abstentions > 0
    assert torch.isfinite(loss.total)
    assert all(layer.self_attn.in_proj_weight.grad is not None for layer in model.encoder.layers)


def test_source_dropout_is_compacted_not_leaked_into_context_tokens():
    raw = fixture_batch(dropout=0.35)
    prepared = T.prepare_synthetic_t2(raw)
    assert prepared.source_features.shape[0] == int(raw["source_visible"].sum())
    assert np.array_equal(prepared.visible_source_row, np.flatnonzero(raw["source_visible"]))
    assert int(prepared.candidate_source_index.max()) < prepared.source_features.shape[0]

    broken = dict(raw)
    broken["candidate_source_index"] = raw["candidate_source_index"].copy()
    dropped = int(np.flatnonzero(~raw["source_visible"])[0])
    broken["candidate_source_index"][0] = dropped
    with pytest.raises(ValueError, match="source dropout removed"):
        T.prepare_synthetic_t2(broken)


def _node_table(raw: dict, prepared: T.PreparedT2Batch,
                source_uids: np.ndarray, target_uids: np.ndarray) -> tuple[dict, np.ndarray]:
    source_vox = np.rint(raw["source_vox_zyx"][prepared.visible_source_row]).astype(np.int64)
    target_vox = np.rint(raw["target_vox_zyx"]).astype(np.int64)
    vox = np.concatenate((source_vox, target_vox), axis=0)
    uid = np.concatenate((source_uids, target_uids))
    n_source, n_target = len(source_uids), len(target_uids)
    row = np.arange(len(uid), dtype=np.float64)
    scale = np.asarray(C.SCALE_UM)
    nodes = {
        "crop": np.full(len(uid), "44b6_software", dtype=object),
        "node_uid": uid,
        "t": np.concatenate((np.zeros(n_source, dtype=np.int64),
                             np.ones(n_target, dtype=np.int64))),
        "instance_label": np.concatenate((np.arange(1, n_source + 1, dtype=np.int64),
                                          np.arange(1, n_target + 1, dtype=np.int64))),
        "z_um": vox[:, 0] * scale[0],
        "y_um": vox[:, 1] * scale[1],
        "x_um": vox[:, 2] * scale[2],
        "z_vox": vox[:, 0], "y_vox": vox[:, 1], "x_vox": vox[:, 2],
        # Software-fixture values exercise the contract. They are not FOCUS predictions.
        "center_confidence": 0.55 + 0.4 * (row + 1) / (len(row) + 1),
        "volume_vox": 20 + (np.arange(len(uid)) % 11),
        "division_prob": 0.01 + 0.08 * ((row * 7) % len(row)) / max(len(row) - 1, 1),
    }
    embeddings = np.concatenate(
        (prepared.source_features.numpy(), prepared.target_features.numpy()), axis=0)
    return nodes, embeddings


def test_generator_matcher_graph_owner_and_complete_contract_connect_end_to_end():
    raw = fixture_batch()
    prepared = T.prepare_synthetic_t2(raw)
    model = make_model(6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            prepared.source_features,
            prepared.target_features,
            prepared.source_positions_um,
            prepared.target_positions_um,
            candidate_source_index=prepared.candidate_source_index,
            candidate_target_index=prepared.candidate_target_index,
        )
        loss = T.supervised_t2_loss(output, prepared)
        loss.total.backward()
        optimizer.step()

    source_uids = np.arange(1000, 1000 + len(prepared.visible_source_row), dtype=np.int64)
    target_uids = np.arange(2000, 2000 + len(raw["target_um_zyx"]), dtype=np.int64)
    candidates = model.candidate_table(
        output,
        crop="44b6_software", t_target=1,
        source_uids=source_uids, target_uids=target_uids,
    )
    source_vox = np.rint(raw["source_vox_zyx"][prepared.visible_source_row]).astype(np.int64)
    target_vox = np.rint(raw["target_vox_zyx"]).astype(np.int64)
    owned = model.emit_graph(
        output,
        crop="44b6_software",
        source_uids=source_uids,
        target_uids=target_uids,
        source_zyx_vox=source_vox,
        target_zyx_vox=target_vox,
        t_source=0,
        t_target=1,
    )
    nodes, embeddings = _node_table(raw, prepared, source_uids, target_uids)
    receipt = C.make_consumer_receipt(
        producer_path=M.__file__, producer_symbol="emit_graph",
        candidates=candidates, graph=owned.table, consumer_chain=owned.consumer_chain,
    )
    offered = list(zip(
        candidates["crop"], candidates["t_target"],
        candidates["target_uid"], candidates["source_uid"],
    ))
    report = C.verify_complete_artifact(
        nodes=nodes,
        embeddings=embeddings,
        candidates=candidates,
        graph=owned.table,
        consumer_receipt=receipt,
        offered_pairs=offered,
        where="software_fixture_not_science",
    )
    assert set(report.stages) == C.ContractReport.REQUIRED_STAGES
    assert report.stages["X_cross_stage"]["emitted_edges_offered"] is True


def test_loss_refuses_a_different_candidate_surface():
    prepared = T.prepare_synthetic_t2(fixture_batch())
    model = make_model(4)
    output = model(
        prepared.source_features,
        prepared.target_features,
        prepared.source_positions_um,
        prepared.target_positions_um,
        candidate_source_index=prepared.candidate_source_index,
        candidate_target_index=prepared.candidate_target_index,
    )
    tampered = T.PreparedT2Batch(
        **{**prepared.__dict__,
           "candidate_target_index": prepared.candidate_target_index.roll(1)}
    )
    with pytest.raises(ValueError, match="surface does not match"):
        T.supervised_t2_loss(output, tampered)
