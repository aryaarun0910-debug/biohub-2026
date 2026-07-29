"""Frozen M1 round-one configuration and its hash.

One seed. One architecture (the organizer UNet+transformer, unchanged) so the TRAINING
change is the only variable. E0c greedy + faithful E0c wrapper downstream -- the failed
v122 ILP is NOT included; if M1 itself passes bilaterally, ILP interaction may be retested
once in a later round.

Direction 1 runs FIRST (train 44b6, hold out 6bba) because 6bba is the current minimum fold
at 0.6490. If it fails, no GPU is spent on direction 0.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "m1"))

import m1_augment as M1A  # noqa: E402

# Checkpoints are saved on a fixed schedule and selected AFTER training by exact scoring.
# Running the exact graph pipeline every epoch would be prohibitively expensive.
CHECKPOINT_EPOCHS = [10, 15, 20, 25, 30]

M1_CONFIG = {
    "round": "M1-v1",
    "architecture": "organizer UNet+transformer (unchanged)",
    "seeds": 1,
    "seed": 20260729,
    "epochs": 30,
    "early_stopping": {"enabled": True, "monitor": "inner_val_loss", "patience": 6},
    "checkpoint_epochs": CHECKPOINT_EPOCHS,
    "checkpoint_selection": {
        "method": "exact patched composite on the frozen inner-validation manifest",
        "downstream": "E0c greedy graph construction + faithful E0c wrapper",
        "weighting": "edge-volume weighted",
        "soup": False,
        "note": "held-out family scored EXACTLY ONCE, after selection",
    },
    "downstream": {
        "graph": "E0c greedy (max_parents=1, max_children=2)",
        "wrapper": "faithful E0c wrapper (min-track 7, gap-refine on)",
        "ilp": None,
        "rationale": "isolate the model change; v122 ILP failed bilaterally (min-fold -0.0633)",
    },
    "augmentation": {
        "module": "scripts/m1/m1_augment.py",
        "fingerprint": M1A.config_fingerprint(),
        "included": ["coordinate-correct flip (trainer)", "brightness", "gamma", "contrast",
                     "poisson+read noise", "separable axial/lateral PSF blur",
                     "frame intensity drift"],
        "excluded_round_one": ["physical rescaling/anisotropy", "localization jitter",
                               "missing-slice corruption"],
        "ranges": M1A.DEFAULT.to_dict(),
        "range_provenance": "fixed physical priors + training-family statistics only",
    },
    "sampling": "density-balanced so sparse and dense regimes contribute comparably",
    "validation_manifest": "scripts/m1/val_manifests.json (frozen, committed pre-training)",
    "determinism": {
        "module": "scripts/m1/m1_determinism.py",
        "seeds": ["python", "numpy", "torch", "torch.cuda"],
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
        "documented_residual": ("some cuDNN 3D-conv backward kernels have no deterministic "
                                "implementation; use_deterministic_algorithms is warn-only "
                                "and bitwise GPU loss parity is not claimed"),
    },
    "directions": {
        "1": {"train_family": "44b6", "held_out": "6bba", "order": "FIRST",
              "baseline_E0c": 0.6490, "gate": "+0.005 over 0.6490, no major regime collapse"},
        "0": {"train_family": "6bba", "held_out": "44b6", "order": "only if direction 1 passes",
              "baseline_E0c": 0.7595, "gate": "+0.005 over 0.7595"},
    },
    "promotion": "both folds improve AND min-fold >= +0.005",
}


def config_hash() -> str:
    return hashlib.sha256(json.dumps(M1_CONFIG, sort_keys=True).encode()).hexdigest()[:16]


if __name__ == "__main__":
    print(json.dumps(M1_CONFIG, indent=2))
    print("\nM1_CONFIG_HASH:", config_hash())
