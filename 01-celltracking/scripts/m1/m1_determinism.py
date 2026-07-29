"""Deterministic seeding for M1.

The vendored trainer is not reproducible: `FrameWindowDataset.__getitem__` calls
`np.random.default_rng()` with NO seed, model construction happens without
`torch.manual_seed`, and the CLI never exposed `--seed`. Two runs with the same nominal
configuration therefore differ in both sampled augmentations and initial weights.

This module fixes all of that in one place:
  * `seed_everything()` seeds Python, NumPy, torch (CPU + all CUDA devices) and sets the
    cuDNN/cuBLAS determinism flags BEFORE any model is constructed;
  * `sample_rng()` derives a per-sample generator deterministically from
    (base_seed, epoch, sample_index), so augmentation is reproducible AND still varies
    across epochs and samples;
  * `state_hash()` / `checkpoint_hash()` give hashes for the initial state and every saved
    checkpoint so a run can be proven identical after the fact.

DOCUMENTED RESIDUAL NONDETERMINISM: some cuDNN kernels (notably certain 3D convolution
backward paths) have no deterministic implementation. `torch.use_deterministic_algorithms`
is therefore requested in warn-only mode by default; where a nondeterministic kernel is
unavoidable, bitwise-identical GPU losses are NOT guaranteed, though seeds, data order and
augmentation remain reproducible. CPU runs are fully deterministic and are what the twin-run
test asserts on.
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True, warn_only: bool = True) -> dict:
    """Seed every RNG and set determinism flags. Call BEFORE building the model."""
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS needs this set before first CUDA context for deterministic GEMMs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    info = {"seed": seed, "deterministic_requested": deterministic}
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            info["use_deterministic_algorithms"] = f"True(warn_only={warn_only})"
        except Exception as exc:  # noqa: BLE001
            info["use_deterministic_algorithms"] = f"unavailable: {exc}"
        info["cudnn_deterministic"] = True
        info["cudnn_benchmark"] = False
    info["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    return info


def sample_rng(base_seed: int, epoch: int, index: int) -> np.random.Generator:
    """Deterministic per-sample generator.

    Reproducible for a given (seed, epoch, index) yet decorrelated across all three, so
    augmentation differs between epochs and samples without ever depending on wall clock,
    worker id or iteration order.
    """
    return np.random.default_rng([base_seed, epoch, index])


def worker_init_fn(base_seed: int):
    """DataLoader worker seeding; keeps workers decorrelated but reproducible."""
    def _init(worker_id: int) -> None:
        import torch
        s = (base_seed + 100003 * (worker_id + 1)) % (2 ** 32)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
    return _init


def state_hash(model) -> str:
    """Order-independent hash of all model parameters (proves identical initial state)."""
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def checkpoint_hash(path) -> str:
    from pathlib import Path
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def transform_fingerprint(base_seed: int, epoch: int, n: int = 32) -> str:
    """Hash of the first n sampled augmentation parameter draws.

    Lets two runs be compared on the *transforms they sampled*, independent of any GPU
    kernel nondeterminism in the loss.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import m1_augment as M

    h = hashlib.sha256()
    for i in range(n):
        rng = sample_rng(base_seed, epoch, i)
        draws = [rng.random() for _ in range(len(M.ORDER))]
        h.update(np.asarray(draws, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]
