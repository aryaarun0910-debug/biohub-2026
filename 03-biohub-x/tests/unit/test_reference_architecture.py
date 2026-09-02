"""The compatibility definition loads the reference exactly, or refuses.

Torch lives in the optional ``model-cpu`` group and the checkpoints are external
machine-local artifacts, so these skip rather than fail where either is absent.
What they must never do is pass while the architecture is subtly wrong, which is
why every assertion here is about exactness: exact keys, exact shapes, exact
dimensions.
"""

from __future__ import annotations

import pathlib

import pytest

torch = pytest.importorskip("torch", reason="the model-cpu dependency group is not installed")

from biohubx.reference.architecture import (  # noqa: E402
    REFERENCE_UNET_OUT_CHANNELS,
    ReferenceArchitectureError,
    ReferenceEdgeModel,
    ReferenceSpec,
    load_reference_model,
    spec_from_state_dict,
)

ARTIFACT_IDS = {
    "support_pack": "reference.pilkwang.support_pack_50ep.edge_predictor_best",
    "temporal_seed": "reference.pilkwang.temporal_seed314159.edge_predictor_best",
}


def checkpoints() -> dict[str, pathlib.Path]:
    """Resolve the quarantined checkpoints through the registry, never a literal path.

    The registry is the one file permitted to carry a machine-local absolute
    path, and it is also where the quarantine that governs these weights is
    recorded, so reading them from anywhere else would bypass both.
    """
    from biohubx.artifacts import ARTIFACT_REGISTRY_PATH, load_artifact_registry

    root = pathlib.Path(__file__).resolve().parents[2]
    registry = load_artifact_registry(root / ARTIFACT_REGISTRY_PATH)
    by_id = {record.id: record for record in registry.artifacts}
    found: dict[str, pathlib.Path] = {}
    for name, artifact_id in ARTIFACT_IDS.items():
        record = by_id.get(artifact_id)
        if record is not None and record.external_path is not None:
            found[name] = pathlib.Path(record.external_path)
    return found


PUBLISHED_SPEC = ReferenceSpec(
    unet_out_channels=32,
    unet_layers=(32, 64, 128),
    pos_feat_dim=32,
    window_size=2,
    downsample=(1, 4, 4),
)


def available() -> dict[str, pathlib.Path]:
    return {name: path for name, path in checkpoints().items() if path.is_file()}


# --- the architecture, checkable without any checkpoint ---------------------


def test_a_synthetic_forward_pass_has_the_expected_dimensions() -> None:
    """Detection is one logit per voxel, per frame of the window."""
    model = ReferenceEdgeModel(PUBLISHED_SPEC)
    model.eval()
    window = torch.zeros((1, 2, 8, 16, 16), dtype=torch.float32)

    with torch.no_grad():
        features, logits = model.detect(window)

    assert tuple(features.shape) == (1, 2, REFERENCE_UNET_OUT_CHANNELS, 8, 16, 16)
    assert tuple(logits.shape) == (1, 2, 1, 8, 16, 16)
    assert bool(torch.isfinite(features).all())
    assert bool(torch.isfinite(logits).all())


def test_a_wrongly_ranked_window_is_refused() -> None:
    model = ReferenceEdgeModel(PUBLISHED_SPEC)
    with pytest.raises(ReferenceArchitectureError, match=r"\(B, W, Z, Y, X\)"):
        model.detect(torch.zeros((2, 8, 16, 16), dtype=torch.float32))


def test_the_spec_is_derived_from_the_checkpoint_not_assumed() -> None:
    state = {
        "detect_head.weight": torch.zeros((1, 32, 1, 1, 1)),
        "transformer.proj.weight": torch.zeros((128, 64)),
    }
    assert spec_from_state_dict(state).pos_feat_dim == 32

    wider = {
        "detect_head.weight": torch.zeros((1, 16, 1, 1, 1)),
        "transformer.proj.weight": torch.zeros((128, 80)),
    }
    derived = spec_from_state_dict(wider)
    assert (derived.unet_out_channels, derived.pos_feat_dim) == (16, 64)


def test_a_checkpoint_from_another_architecture_is_refused() -> None:
    with pytest.raises(ReferenceArchitectureError, match="not this architecture"):
        spec_from_state_dict({"something.else": torch.zeros((1, 1))})


def test_a_projection_too_narrow_for_a_positional_embedding_is_refused() -> None:
    with pytest.raises(ReferenceArchitectureError, match="no room for a positional embedding"):
        spec_from_state_dict(
            {
                "detect_head.weight": torch.zeros((1, 64, 1, 1, 1)),
                "transformer.proj.weight": torch.zeros((128, 64)),
            }
        )


def test_a_missing_checkpoint_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ReferenceArchitectureError, match="no checkpoint at"):
        load_reference_model(pathlib.Path("does-not-exist.pth"))


# --- the published checkpoints, where this machine holds them ---------------


@pytest.mark.parametrize("name", sorted(ARTIFACT_IDS))
def test_each_published_checkpoint_loads_with_exact_key_coverage(name: str) -> None:
    paths = available()
    if name not in paths:
        pytest.skip(f"{name} checkpoint is not on this machine")

    state = torch.load(paths[name], map_location="cpu", weights_only=True)
    model = ReferenceEdgeModel(spec_from_state_dict(state))

    expected = set(model.state_dict())
    published = set(state)
    assert published - expected == set(), "the checkpoint carries keys the definition does not"
    assert expected - published == set(), "the definition carries keys the checkpoint does not"

    for key, tensor in state.items():
        assert tuple(model.state_dict()[key].shape) == tuple(tensor.shape), f"shape differs at {key}"


@pytest.mark.parametrize("name", sorted(ARTIFACT_IDS))
def test_each_published_checkpoint_runs_finite_and_frozen(name: str) -> None:
    paths = available()
    if name not in paths:
        pytest.skip(f"{name} checkpoint is not on this machine")

    model, spec = load_reference_model(paths[name])
    assert spec == PUBLISHED_SPEC
    assert not model.training, "a loaded reference must be in eval mode"
    assert all(not p.requires_grad for p in model.parameters()), "reference weights must be frozen"

    with torch.no_grad():
        _, logits = model.detect(torch.zeros((1, spec.window_size, 8, 16, 16), dtype=torch.float32))
    assert tuple(logits.shape) == (1, spec.window_size, 1, 8, 16, 16)
    assert bool(torch.isfinite(logits).all())


def test_the_two_published_checkpoints_are_different_models() -> None:
    paths = available()
    if len(paths) < 2:
        pytest.skip("both checkpoints are needed to compare them")

    first = torch.load(paths["support_pack"], map_location="cpu", weights_only=True)
    second = torch.load(paths["temporal_seed"], map_location="cpu", weights_only=True)
    assert set(first) == set(second)
    assert any(not torch.equal(first[key], second[key]) for key in first), (
        "the two published weights are identical, so they are one model published twice"
    )
