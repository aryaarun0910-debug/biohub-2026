"""The analyzer measures; these hold the measurements to what a toy model makes certain.

A three-tap convolution stack has a receptive field the arithmetic gives; a
stride-two layer halves an axis; a checkpoint missing a key must be named;
reloading must reproduce output bit for bit. If any of these drift, every
number the analyzer reports about a real detector is suspect.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch
from torch import nn

from biohubx.training.inspect import (
    activation_census,
    downsampling,
    empirical_receptive_field,
    gradient_norm_by_block,
    input_channel_probes,
    output_statistics,
    parameter_census,
    shape_trace,
    state_dict_coverage,
    strict_reload_determinism,
    temporal_perturbation,
    to_jsonable,
)


class Stack(nn.Module):
    """Two 3-tap 3D convolutions, no padding shrinkage concerns: padding keeps the shape."""

    def __init__(self, stride: int = 1) -> None:
        super().__init__()
        self.first = nn.Conv3d(2, 4, kernel_size=3, padding=1, stride=stride)
        self.second = nn.Conv3d(4, 1, kernel_size=3, padding=1)
        self.register_buffer("scale", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.second(torch.relu(self.first(x))) * self.scale
        return out


def run(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = model(x)
    return out


def example() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 2, 9, 11, 13)


def test_coverage_names_missing_unexpected_and_mismatched_keys() -> None:
    model = Stack()
    state = model.state_dict()
    ok = state_dict_coverage(model, state)
    assert ok["strict_ok"] and ok["model_keys"] == 5

    broken = dict(state)
    del broken["second.bias"]
    broken["extra.weight"] = torch.zeros(1)
    broken["first.weight"] = torch.zeros(4, 2, 5, 5, 5)
    report = state_dict_coverage(model, broken)
    assert report["missing_in_checkpoint"] == ["second.bias"]
    assert report["unexpected_in_checkpoint"] == ["extra.weight"]
    assert report["shape_mismatch"] == ["first.weight"]
    assert report["strict_ok"] is False


def test_census_counts_parameters_buffers_and_frozen() -> None:
    model = Stack()
    model.second.weight.requires_grad_(False)
    census = parameter_census(model)
    first = 4 * 2 * 27 + 4
    second = 1 * 4 * 27 + 1
    assert census["total_parameters"] == first + second
    assert census["frozen_parameters"] == 1 * 4 * 27
    assert census["total_buffers"] == 1
    assert census["by_module"]["first"]["trainable"] == first


def test_shape_trace_and_downsampling_see_a_stride() -> None:
    model = Stack(stride=2)
    x = example()
    rows = shape_trace(model, run, x)
    assert [r["module"] for r in rows] == ["first", "second"]
    assert rows[0]["output"]["shape"] == [1, 4, 5, 6, 7]
    out = run(model, x)
    down = downsampling(tuple(x.shape), tuple(out.shape), (1.625, 1.625, 1.625))
    assert down["factors"] == [1.8, 1.8333, 1.8571]
    assert down["output_spacing_um"][0] == 2.925


def test_receptive_field_of_two_three_tap_convolutions_is_five() -> None:
    model = Stack()
    field = empirical_receptive_field(model, run, example(), spatial_axes=3)
    assert field["extent_voxels"] == [5, 5, 5]
    assert field["clipped_by_input"] == [False, False, False]


def test_strict_reload_reproduces_output_exactly() -> None:
    torch.manual_seed(1)
    model = Stack()
    report = strict_reload_determinism(Stack, model, run, example())
    assert report["output_identical_after_reload"] and report["output_identical_between_runs"]
    assert report["max_abs_difference_after_reload"] == 0.0


def test_probes_notice_an_inert_channel_and_missing_temporal_context() -> None:
    model = Stack()
    with torch.no_grad():
        model.first.weight[:, 1] = 0.0  # channel 1 is wired to nothing
    probes = input_channel_probes(model, run, example(), channel_axis=1)
    assert probes[1]["inert"] is True and probes[0]["inert"] is False

    # This stack treats the leading spatial axis as "time" for the probe: it is a
    # 3D convolution, so reversing that axis reorders the output exactly and the
    # frozen-neighbour probe does move the output. Both numbers are reported.
    temporal = temporal_perturbation(model, run, example(), time_axis=2)
    assert temporal["frames"] == 9
    assert temporal["neighbours_frozen_relative_change"] > 0.0


def test_gradient_norms_and_activation_census_and_statistics_are_finite() -> None:
    model = Stack()
    x = example()
    model(x).sum().backward()
    norms = gradient_norm_by_block(model)
    assert norms["total"] > 0 and set(norms["by_block"]) == {"first", "second"}
    census = activation_census(model, run, x)
    assert census["total_elements"] == 4 * 9 * 11 * 13 + 9 * 11 * 13
    stats = output_statistics(model(x))
    assert stats["non_finite"] == 0 and 0.0 <= stats["fraction_above_0_5"] <= 1.0
    assert to_jsonable({"a": torch.tensor([1.0]), "b": float("inf")}) == {"a": [1.0], "b": "inf"}
