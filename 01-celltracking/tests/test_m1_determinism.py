"""Determinism tests for M1 — the twin-run proof.

The vendored trainer seeded nothing: `np.random.default_rng()` with no seed inside
`__getitem__`, no `torch.manual_seed` before model construction, no `--seed` on the CLI.
These tests prove the replacement is reproducible: identical seeds must give identical
sampled transforms, identical initial weights and identical early losses on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "m1"))

import m1_augment as M  # noqa: E402
import m1_determinism as DT  # noqa: E402

torch = pytest.importorskip("torch")


def test_sample_rng_reproducible_and_decorrelated():
    a = DT.sample_rng(7, 0, 0).random(8)
    b = DT.sample_rng(7, 0, 0).random(8)
    np.testing.assert_array_equal(a, b)                     # same (seed, epoch, index)
    assert not np.array_equal(a, DT.sample_rng(7, 1, 0).random(8))   # epoch varies
    assert not np.array_equal(a, DT.sample_rng(7, 0, 1).random(8))   # index varies
    assert not np.array_equal(a, DT.sample_rng(8, 0, 0).random(8))   # seed varies


def test_augmentation_is_reproducible_from_seed_epoch_index():
    vol = np.random.default_rng(0).random((2, 6, 16, 16), dtype=np.float32)
    one = M.apply_augmentations(vol, DT.sample_rng(42, 3, 11))
    two = M.apply_augmentations(vol, DT.sample_rng(42, 3, 11))
    np.testing.assert_array_equal(one, two)
    other = M.apply_augmentations(vol, DT.sample_rng(42, 4, 11))
    assert not np.array_equal(one, other)


def test_transform_fingerprint_matches_across_twin_runs():
    """Compares the transforms two runs SAMPLE, independent of any GPU nondeterminism."""
    assert DT.transform_fingerprint(2026, 0) == DT.transform_fingerprint(2026, 0)
    assert DT.transform_fingerprint(2026, 0) != DT.transform_fingerprint(2026, 1)
    assert DT.transform_fingerprint(2026, 0) != DT.transform_fingerprint(2027, 0)


def _tiny_model():
    return torch.nn.Sequential(
        torch.nn.Conv3d(1, 4, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv3d(4, 1, 3, padding=1))


def test_seeding_gives_identical_initial_weights():
    DT.seed_everything(1234)
    h1 = DT.state_hash(_tiny_model())
    DT.seed_everything(1234)
    h2 = DT.state_hash(_tiny_model())
    assert h1 == h2, "identical seed must give identical initial weights"
    DT.seed_everything(4321)
    assert DT.state_hash(_tiny_model()) != h1, "different seed must change init"


def test_twin_short_runs_match_on_cpu():
    """End-to-end: two identical short CPU runs must produce identical losses.

    CPU only. GPU parity is not asserted because some cuDNN 3D-convolution backward
    kernels have no deterministic implementation -- documented in m1_determinism.
    """
    def run(seed):
        DT.seed_everything(seed)
        model = _tiny_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        losses = []
        for step in range(5):
            rng = DT.sample_rng(seed, 0, step)
            vol = rng.random((1, 1, 4, 8, 8)).astype(np.float32)
            aug = M.apply_augmentations(vol[0, 0], rng)
            x = torch.from_numpy(aug)[None, None]
            y = torch.from_numpy(vol[0])[None]
            loss = torch.nn.functional.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        return losses, DT.state_hash(model)

    l1, h1 = run(2026)
    l2, h2 = run(2026)
    assert l1 == pytest.approx(l2, abs=0.0), f"losses diverged: {l1} vs {l2}"
    assert h1 == h2, "final weights diverged under identical seed"

    l3, h3 = run(2027)
    assert h3 != h1, "different seed should give a different trajectory"


def test_seed_everything_reports_determinism_flags():
    info = DT.seed_everything(11)
    assert info["seed"] == 11
    assert info["cudnn_deterministic"] is True
    assert info["cudnn_benchmark"] is False
    assert info["cublas_workspace_config"] == ":4096:8"
    assert "use_deterministic_algorithms" in info


def test_worker_init_is_deterministic():
    init = DT.worker_init_fn(99)
    init(0); a = np.random.random(4)
    init(0); b = np.random.random(4)
    np.testing.assert_array_equal(a, b)
    init(1); c = np.random.random(4)
    assert not np.array_equal(a, c)
