"""Contracts for the HOCT association head and its DETERMINED feature contract (PKT-0029).

Software contracts only. Whether HOCT chooses better parents is an experiment result, and it is
not decidable here at all - the 32-channel node features are GPU-bound.

WHY THIS FILE CHANGED ON 2026-08-29
-----------------------------------
It previously asserted `not hasattr(HoctAssociationHead, "forward")`, encoding PKT-0029's premise
that the publisher shipped no model code and that a forward pass would require INVENTING a
feature layout. The premise is refuted: the publisher's own module is on Kaggle under MIT and
strict-loads both checkpoints, so the layout is READ. The assertion that replaces it is the one
that actually protects us - that a strict load is NOT sufficient.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.win_bet.hoct_head import (
    CONTRACT_SLOTS,
    D_MODEL,
    EDGE_NEIGHBORS,
    GATE_UM,
    N_HEADS,
    NODE_EXTRAS,
    PAIR_EXTRAS,
    RELATION_IN,
    SCALE_ZYX_UM,
    HoctAssociationHead,
    build_edge_neighborhood,
    candidate_graph,
    edge_relation_features,
    load_checkpoint,
    parent_probability,
    run_equivalence,
)

CKPT_DIR = Path("C:/temp/arch_inventory/hoct/weights")
PUBLISHER = Path("C:/temp/hoct/code/hoct_edge_transformer.py")
FOLDS = [
    CKPT_DIR / "fold0_hoct_hard_negative_9294.pt",
    CKPT_DIR / "fold1_hoct_hard_negative_fold1_9294.pt",
]


def _toy(n_source: int = 24, n_target: int = 22, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    source = rng.uniform(0.0, 30.0, size=(n_source, 3)).astype(np.float32)
    target = (source[:n_target] + rng.normal(0.0, 2.0, size=(n_target, 3))).astype(np.float32)
    edge_source, edge_target = candidate_graph(source, target)
    neighbors = build_edge_neighborhood(edge_source, edge_target, source, target)
    return {
        "source_features": torch.from_numpy(rng.normal(size=(n_source, 32)).astype(np.float32)),
        "target_features": torch.from_numpy(rng.normal(size=(n_target, 32)).astype(np.float32)),
        "source_um": torch.from_numpy(source),
        "target_um": torch.from_numpy(target),
        "edge_source": torch.from_numpy(edge_source),
        "edge_target": torch.from_numpy(edge_target),
        "neighbors": torch.from_numpy(neighbors),
    }


def test_reconstruction_matches_the_published_shapes():
    """The dims the checkpoint DOES determine. Getting any of these wrong fails the strict load,
    which is why they are constants rather than magic numbers."""
    model = HoctAssociationHead()
    assert model.node_projection.in_features == 35        # 32 UNet channels + 3 extras
    assert model.node_projection.out_features == D_MODEL
    assert model.frame_embedding.num_embeddings == 2      # window_size 2
    assert len(model.edge_blocks) == 3
    assert model.edge_projection[0].in_features == 196    # 96 src + 96 tgt + 4 pair extras
    assert model.edge_blocks[0].relation_bias[-1].out_features == N_HEADS
    assert model.edge_blocks[0].relation_bias[0].in_features == RELATION_IN
    assert (NODE_EXTRAS, PAIR_EXTRAS, RELATION_IN) == (3, 4, 13)


def test_a_strict_load_cannot_see_the_trunk_contract():
    """THE LOAD-BEARING TEST, and the one the previous version of this file did not have.

    Activation, normalisation order and the positional encoding change the computed function
    WITHOUT changing a single parameter shape. The earlier reconstruction got all three wrong,
    strict-loaded both checkpoints cleanly, and would have scored a different model than the one
    published - the exact silent failure PKT-0029 falsifier (b) warned about. This asserts the
    divergence is real, so nobody can conclude a clean strict load is sufficient.
    """
    inputs = _toy()
    correct = HoctAssociationHead()
    torch.manual_seed(0)
    state = correct.state_dict()

    wrong = HoctAssociationHead()
    wrong.load_state_dict(state, strict=True)         # same weights, defective trunk
    wrong.node_encoder = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, D_MODEL * 2, dropout=0.0,
            activation="relu", batch_first=True, norm_first=False,   # torch defaults
        ),
        2,
    )
    wrong.node_encoder.load_state_dict(correct.node_encoder.state_dict(), strict=True)
    correct.eval()
    wrong.eval()
    with torch.no_grad():
        a = correct(**inputs)["edge_logits"]
        b = wrong(**inputs)["edge_logits"]
    assert not torch.allclose(a, b, atol=1e-4), (
        "a defective node encoder produced identical logits - this test cannot protect anything"
    )


def test_relation_features_are_in_the_determined_order():
    """The 13 relation slots, asserted against quantities computed independently here. Slot 12
    and 13 are the endpoint-sharing flags, which is what makes competing parents visible."""
    inputs = _toy()
    relation, valid, midpoints = edge_relation_features(
        inputs["source_um"], inputs["target_um"],
        inputs["edge_source"], inputs["edge_target"], inputs["neighbors"], GATE_UM,
    )
    assert relation.shape[-1] == RELATION_IN
    assert midpoints.shape == (len(inputs["edge_source"]), 3)
    # An edge is always its own first neighbour, so against itself: all deltas zero, cosine 1,
    # line distance 0, and BOTH shared flags 1.
    self_relation = relation[:, 0, :]
    assert torch.allclose(self_relation[:, :9], torch.zeros_like(self_relation[:, :9]), atol=1e-5)
    assert torch.allclose(self_relation[:, 9], torch.ones_like(self_relation[:, 9]), atol=1e-5)
    assert torch.allclose(self_relation[:, 10], torch.zeros_like(self_relation[:, 10]), atol=1e-5)
    assert torch.all(self_relation[:, 11] == 1.0)
    assert torch.all(self_relation[:, 12] == 1.0)
    assert valid[:, 0].all()


def test_candidate_graph_is_a_geometric_ball_not_our_deployed_rule():
    """FACT-0369 says our deployed rule admits AT MOST ONE parent per target by arithmetic.
    HOCT's admits every source inside gate_um. This asserts the difference rather than leaving
    it as prose, because it is the largest upstream incompatibility on this lever."""
    rng = np.random.default_rng(3)
    source = rng.uniform(0.0, 25.0, size=(40, 3)).astype(np.float32)
    target = rng.uniform(0.0, 25.0, size=(35, 3)).astype(np.float32)
    edge_source, edge_target = candidate_graph(source, target, gate_um=GATE_UM)
    distances = np.linalg.norm(source[edge_source] - target[edge_target], axis=-1)
    assert distances.max() <= GATE_UM + 1e-5
    counts = np.bincount(edge_target, minlength=len(target))
    assert counts.max() > 1, "the gate produced no contested target - the test proves nothing"


def test_edge_neighbourhood_puts_endpoint_sharing_edges_first():
    """Competing parents share a target and competing daughters share a source. If those lose the
    truncation to 64 slots to merely nearby segments, the head cannot see the competition."""
    inputs = _toy(n_source=40, n_target=38, seed=11)
    neighbors = inputs["neighbors"].numpy()
    edge_source = inputs["edge_source"].numpy()
    edge_target = inputs["edge_target"].numpy()
    assert neighbors.shape[1] == EDGE_NEIGHBORS
    for edge in range(min(len(edge_source), 50)):
        row = neighbors[edge]
        row = row[row >= 0]
        assert row[0] == edge
        sharing = {
            index for index in range(len(edge_source))
            if edge_source[index] == edge_source[edge] or edge_target[index] == edge_target[edge]
        }
        if len(sharing) <= EDGE_NEIGHBORS:
            assert sharing.issubset(set(row.tolist()))


def test_parent_probability_includes_an_abstain_mass():
    """HOCT normalises over each target's incoming edges PLUS a quiet logit, so the incoming
    probabilities sum to strictly LESS than one. Our deployed rule has no such alternative."""
    edge_target = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    edge_logits = np.array([1.0, 0.5, -0.2, 0.3, 2.0])
    quiet_logits = np.array([0.1, -1.0])
    probability = parent_probability(edge_logits, quiet_logits, edge_target)
    for target in (0, 1):
        total = probability[edge_target == target].sum()
        assert 0.0 < total < 1.0


def test_scale_matches_the_campaign_calibration_anchor():
    """FACT-0040's note: atlas coords are FULL-RES (z,y,x) at (1.625, 0.40625, 0.40625), NOT
    isotropic. The publisher's SCALE_ZYX_UM is the same, which is why their coordinates are
    directly comparable to ours."""
    assert tuple(SCALE_ZYX_UM) == (1.625, 0.40625, 0.40625)


def test_contract_slots_cover_each_block_exactly():
    assert CONTRACT_SLOTS["node_extras"] == (32, 35)
    assert CONTRACT_SLOTS["pair_extras"] == (192, 196)
    assert CONTRACT_SLOTS["relation"] == (0, 13)


def test_equivalence_fails_closed_when_the_publisher_source_is_absent(tmp_path):
    """A gate that quietly does nothing is indistinguishable from a gate that passed. AGENTS.md
    names this failure class explicitly, so absence must RAISE, not skip."""
    with pytest.raises(SystemExit):
        run_equivalence(FOLDS[0], tmp_path / "not_here.py", tolerance=1e-5)


@pytest.mark.parametrize("path", FOLDS, ids=lambda p: p.name)
def test_published_checkpoints_load_strict(path):
    """Both folds must load with strict=True. Necessary, and NOT sufficient - see
    test_a_strict_load_cannot_see_the_trunk_contract."""
    if not path.exists():
        pytest.skip(f"checkpoint not present: {path}")
    model, meta = load_checkpoint(path)
    assert sum(p.numel() for p in model.parameters()) == 447376
    assert meta.get("method") == "hoct_hard_negative_v1"
    assert meta.get("gate_um") == GATE_UM
    assert meta.get("feature_dim") == 32


@pytest.mark.parametrize("path", FOLDS, ids=lambda p: p.name)
def test_our_module_computes_the_publishers_function(path):
    """The gate that makes the reimplementation trustworthy. Skips LOUDLY when the publisher
    source is not on this machine, naming exactly what went unverified."""
    if not path.exists() or not PUBLISHER.exists():
        pytest.skip(
            f"NOT VERIFIED HERE: numerical equivalence against {PUBLISHER} requires the MIT "
            "publisher source and the CC BY checkpoints; fetch with kaggle datasets download "
            "-d rudispresence/biohub-stabledet-hoct-code",
        )
    result = run_equivalence(path, PUBLISHER, tolerance=1e-5)
    assert result["passed"], result
    assert result["candidate_edges"] > 0


def test_fork_head_exists_but_is_not_wired_into_any_scoring_path():
    """FACT-0347: its published division recall is zero, on 110 positive pairs (FACT-0362).
    Reconstructed so the strict load covers the whole checkpoint, and deliberately unused."""
    model = HoctAssociationHead()
    assert model.fork_head[0].in_features == 197
    source = Path("scripts/win_bet/hoct_head.py").read_text(encoding="utf-8")
    assert "fork_head(" not in source, "the fork head must not be invoked anywhere"
    assert "division_head(" not in source, "the division head must not be invoked anywhere"


def test_every_slot_group_is_perturbed_by_the_same_mechanism():
    """All three contract groups must be permuted by reordering the WEIGHT COLUMNS that read them,
    so their sensitivities are comparable.

    An earlier version permuted the coordinate AXES for node_extras. That is an isometry: it
    preserves every distance and simultaneously relabels the displacement and relation channels, so
    it measured global axis-convention robustness rather than the 3 node slots, and its number was
    not comparable with the other two. This asserts each group reads a distinct, correctly sized
    weight block and that permuting it actually changes the logits.
    """
    from scripts.win_bet.hoct_head import _forward_with_permuted_block

    inputs = _toy(n_source=36, n_target=34, seed=5)
    model = HoctAssociationHead()
    model.eval()
    with torch.no_grad():
        base = model(**inputs)["edge_logits"].numpy()

    for group, (start, stop) in CONTRACT_SLOTS.items():
        width = stop - start
        reversed_order = np.arange(width)[::-1].copy()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        other = _forward_with_permuted_block(model, inputs, group, reversed_order)
        assert other.shape == base.shape
        assert not np.allclose(base, other), f"permuting {group} changed nothing"
        # The helper must restore every weight it touched, or the next measurement is contaminated.
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, before[name]), f"{group} left {name} modified"

    with pytest.raises(ValueError):
        _forward_with_permuted_block(model, inputs, "relation", np.arange(3))
