"""M1 driver tests: baseline-fidelity locks and deterministic resumable sampling."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "m1"))

import m1_augment as M1A  # noqa: E402
import m1_driver as MD  # noqa: E402

TRAINER = ROOT / "vendor/kaggle-cell-tracking/scripts/train_unet_transformer.py"


def _default_of(func_name: str, arg: str) -> float:
    """Read a default straight out of the vendored source, so the lock tracks reality."""
    tree = ast.parse(TRAINER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            args = node.args.args + node.args.kwonlyargs
            defaults = list(node.args.defaults) + list(node.args.kw_defaults)
            pad = [None] * (len(args) - len(node.args.defaults)) if not node.args.kwonlyargs else []
            mapping = dict(zip([a.arg for a in node.args.args][-len(node.args.defaults):],
                               node.args.defaults))
            if arg in mapping:
                return ast.literal_eval(mapping[arg])
    raise AssertionError(f"{func_name}({arg}) default not found")


def test_baseline_loss_weights_match_train_not_train_epoch():
    """The bug this locks: train_epoch's OWN defaults are (0.1, 0.1) while train() -- the
    baseline path -- passes (1e1, 1e-2). Omitting them trains a 100x smaller detection
    weight and a 10x larger negative weight, making M1 incomparable to the baseline."""
    assert MD.BASELINE_LOSS_WEIGHTS["det_loss_weight"] == _default_of("train", "det_loss_weight")
    assert MD.BASELINE_LOSS_WEIGHTS["det_neg_weight"] == _default_of("train", "det_neg_weight")
    assert MD.BASELINE_LOSS_WEIGHTS["pool_kernel_um"] == _default_of("train", "pool_kernel_um")
    # and they must NOT equal train_epoch's misleading defaults
    assert MD.BASELINE_LOSS_WEIGHTS["det_loss_weight"] != _default_of("train_epoch", "det_loss_weight")
    assert MD.BASELINE_LOSS_WEIGHTS["det_neg_weight"] != _default_of("train_epoch", "det_neg_weight")


def test_all_source_patch_anchors_exist():
    src = TRAINER.read_text()
    assert M1A.RNG_PATCH_OLD in src, "determinism anchor missing"
    assert MD.TELEMETRY_OLD in src, "telemetry anchor missing"
    assert MD.SAMPLE_PATCH_OLD in src, "sample-identity anchor missing"


def test_patched_trainer_parses_and_contains_all_patches():
    src = TRAINER.read_text()
    patched = MD.patch_trainer_source(src, 20260729, M1A)
    patched = patched.replace(MD.SAMPLE_PATCH_OLD, MD.SAMPLE_PATCH_NEW, 1)
    ast.parse(patched)
    assert "default_rng([M1_SEED" in patched
    assert "STEP_LOG.append" in patched
    assert "SAMPLE_LOG.append" in patched
    assert patched.index("STEP_LOG.append") > patched.index("optimizer.step()")


def test_sampler_is_deterministic_for_same_seed_and_epoch():
    a = MD.ResumableSampler(100, seed=7, epoch=3)
    b = MD.ResumableSampler(100, seed=7, epoch=3)
    assert list(a) == list(b)
    assert a.permutation_hash() == b.permutation_hash()


def test_sampler_order_varies_by_epoch_and_seed():
    base = MD.ResumableSampler(100, seed=7, epoch=0)
    assert list(base) != list(MD.ResumableSampler(100, seed=7, epoch=1))
    assert list(base) != list(MD.ResumableSampler(100, seed=8, epoch=0))


def test_resume_continues_instead_of_replaying():
    """The defect this fixes: a fresh shuffle replays consumed samples."""
    full = MD.ResumableSampler(50, seed=11, epoch=2, length=40)
    order = list(full)
    first = list(MD.ResumableSampler(50, seed=11, epoch=2, start=0, length=20))
    resumed = MD.ResumableSampler.resume(
        MD.ResumableSampler(50, seed=11, epoch=2, length=40).state(), step_in_epoch=20)
    second = list(resumed)
    assert first == order[:20]
    assert second == order[20:40], "resume must continue, not restart"
    assert not set(first) & set(second), "resumed samples must not repeat consumed ones"
    assert first + second == order


def test_resume_rejects_permutation_mismatch():
    state = MD.ResumableSampler(50, seed=11, epoch=2, length=40).state()
    state["permutation_hash"] = "deadbeefdeadbeef"
    with pytest.raises(SystemExit):
        MD.ResumableSampler.resume(state, step_in_epoch=20)


def test_sampler_length_accounts_for_start():
    s = MD.ResumableSampler(100, seed=1, epoch=0, start=30, length=80)
    assert len(s) == 50 and len(list(s)) == 50


def test_sampler_state_roundtrip_fields():
    s = MD.ResumableSampler(64, seed=5, epoch=9, start=8, length=40)
    st = s.state()
    for k in ("n", "seed", "epoch", "start", "length", "permutation_hash"):
        assert k in st
