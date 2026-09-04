"""How a session is spent, tested on CPU where the mechanisms must degrade honestly.

AMP and the batch ladder need a CUDA device; on CPU they must say so and fall
back rather than pretend. The counters and the matched sampler are pure and are
tested for their arithmetic. The ceiling check is tested for refusing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from biohubx.evaluation.propensity import FEATURE_NAMES
from biohubx.proposals import rescore
from biohubx.training import runtime
from biohubx.training.targets import positive_unlabelled_loss


def _patches(n: int, channels: int = 3) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(n, channels, 9, 9, 9)


def test_amp_is_refused_without_a_cuda_device_and_says_why() -> None:
    model = rescore.build_scorer("A3")
    labels = torch.zeros(64, dtype=torch.bool)
    labels[:3] = True
    outcome = runtime.verify_amp(
        model, _patches(64), labels, prior=0.05, device=torch.device("cpu"), loss_fn=positive_unlabelled_loss
    )
    assert outcome["requested"] is True and outcome["enabled"] is False
    assert "CUDA" in outcome["reason"]


def test_the_batch_ladder_is_not_climbed_on_cpu() -> None:
    model = rescore.build_scorer("A3")
    labels = torch.zeros(64, dtype=torch.bool)
    labels[0] = True
    outcome = runtime.autotune_batch(
        model,
        _patches(64),
        labels,
        prior=0.05,
        device=torch.device("cpu"),
        loss_fn=positive_unlabelled_loss,
        amp=False,
        ceiling_bytes=runtime.MEMORY_CEILING_BYTES,
    )
    assert outcome["chosen"] == runtime.BATCH_LADDER[0]
    assert outcome["rungs"] == []


def test_the_memory_ceiling_is_recorded_as_unenforced_on_cpu() -> None:
    assert runtime.memory_ceiling(torch.device("cpu"))["enforced"] is False
    runtime.check_ceiling(torch.device("cpu"), 1, "anywhere")  # zero peak on CPU never exceeds


def test_the_throughput_meter_reports_the_ratios_a_session_is_judged_by() -> None:
    meter = runtime.ThroughputMeter()
    meter.examples, meter.positives, meter.batches = 4096, 20, 4
    meter.skipped_no_positive, meter.clamped = 3, 1
    meter.compute_seconds, meter.wait_seconds = 2.0, 0.5
    out = meter.to_dict()
    assert out["examples_per_second"] == 2048.0
    assert out["positives_per_batch"] == 5.0
    assert out["clamp_frequency"] == 0.25
    assert out["data_wait_fraction"] == 0.2
    assert out["gpu_utilisation_median"] is None


def test_matched_sampling_reweights_the_unlabelled_side_toward_the_positives_circumstances() -> None:
    """Positives sit at high z; after matching, unlabelled weight mass moves toward high z."""
    rng = np.random.default_rng(0)
    tables, labels = [], []
    for _ in range(2):
        rows = 2000
        x = rng.normal(size=(rows, len(FEATURE_NAMES)))
        z = FEATURE_NAMES.index("z")
        y = rng.random(rows) < 1.0 / (1.0 + np.exp(-(3.0 * x[:, z] - 4.0)))
        y[:5] = True
        tables.append(x)
        labels.append(y)
    weights, record = runtime.matched_sampling_weights(tables, labels)
    z = FEATURE_NAMES.index("z")
    for x, y, w in zip(tables, labels, weights, strict=True):
        unl = ~y
        assert w[y].min() == w[y].max() == 1.0
        assert np.isclose(w[unl].sum(), unl.sum())
        weighted_z = float((w[unl] * x[unl, z]).sum() / w[unl].sum())
        plain_z = float(x[unl, z].mean())
        assert weighted_z > plain_z + 0.3
    assert "dog" not in record["matched_on"] and "intensity" not in record["matched_on"]
    assert record["coefficients"]["z"] > 0.5


def test_matched_sampling_refuses_without_a_positive() -> None:
    x = [np.zeros((10, len(FEATURE_NAMES)))]
    with pytest.raises(runtime.RuntimeRefusal, match="at least one positive"):
        runtime.matched_sampling_weights(x, [np.zeros(10, dtype=bool)])
