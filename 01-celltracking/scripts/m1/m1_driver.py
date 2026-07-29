"""M1 training driver — owns the loop; reuses the vendored `train_epoch` for step fidelity.

`train()` is the wrong reuse boundary for M1: it runs `evaluate()` every epoch (the proxy
evaluation we must not do) and saves "best" by acc*recall (a forbidden selection metric).
`train_epoch` is the right boundary — it is the actual optimisation step (losses, backward,
clipping) — so this driver replicates train()'s dataset/model construction exactly, then
runs its own epoch loop with:

  * the baseline budget: batch 1, 800 iters/epoch, 30 epochs = 24,000 steps, LR 1e-4;
  * NO per-epoch validation and NO proxy-metric checkpoint selection;
  * checkpoints at a fixed schedule containing model + optimizer + RNG + counters + hashes;
  * resume that restores all of the above;
  * per-step telemetry (timing, losses) via a small source patch.

The inner-validation crops are EXCLUDED from the training loader: the manifest's
`train_crops` is the family minus the 12 validation crops (verified disjoint before use).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

STEP_LOG: list[dict] = []          # filled by the patched train_epoch

# Source patch: append per-step telemetry. Anchored on the accumulator lines that follow
# optimizer.step(), so it runs exactly once per optimizer step.
TELEMETRY_OLD = """        total_edge_loss += edge_loss.item() * B
        total_det_loss += det_loss.item() * B
        n_samples += B"""
TELEMETRY_NEW = """        total_edge_loss += edge_loss.item() * B
        total_det_loss += det_loss.item() * B
        n_samples += B
        try:
            import m1_driver as _m1d
            _m1d.STEP_LOG.append({
                "edge_loss": float(edge_loss.item()), "det_loss": float(det_loss.item()),
                "t_data": float(t1 - t0), "t_fwd": float(t2 - t1), "t_bwd": float(t3 - t2),
                "crop": str(batch.get("name", [""])[0]) if isinstance(batch, dict) else "",
            })
        except Exception:
            pass"""


def patch_trainer_source(src: str, seed: int, augment_module) -> str:
    """Apply the determinism and telemetry patches; abort loudly if either anchor is gone."""
    if augment_module.RNG_PATCH_OLD not in src:
        raise SystemExit("ABORT -- unseeded RNG anchor missing; determinism patch would no-op")
    if TELEMETRY_OLD not in src:
        raise SystemExit("ABORT -- telemetry anchor missing; per-step metrics unavailable")
    src = src.replace(augment_module.RNG_PATCH_OLD, augment_module.RNG_PATCH_NEW, 1)
    src = src.replace(TELEMETRY_OLD, TELEMETRY_NEW, 1)
    nl = chr(10)
    src = src.replace("import numpy as np",
                      nl.join(["import numpy as np", f"M1_SEED = {seed}", "M1_EPOCH = [0]"]), 1)
    return src


def save_checkpoint(path: Path, *, model, optimizer, epoch: int, global_step: int,
                    seed: int, config_hash: str, aug_fingerprint: str,
                    manifest_sha: str, torch_mod) -> str:
    """Resumable checkpoint. Weights-only is insufficient for multi-session training."""
    import hashlib
    blob = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),          # AdamW moments -- mandatory
        "epoch": epoch, "global_step": global_step,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch_mod.get_rng_state(),
        "rng_cuda": torch_mod.cuda.get_rng_state_all() if torch_mod.cuda.is_available() else [],
        "aug_epoch": epoch,                            # drives sample_rng(seed, epoch, idx)
        "seed": seed, "config_hash": config_hash,
        "aug_fingerprint": aug_fingerprint, "manifest_sha256": manifest_sha,
    }
    tmp = path.with_suffix(".pt.tmp")
    torch_mod.save(blob, tmp)
    import os
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path: Path, *, model, optimizer, torch_mod,
                    expect_config_hash: str | None = None) -> dict:
    """Restore model, optimizer, counters and every RNG stream."""
    blob = torch_mod.load(path, weights_only=False)
    if expect_config_hash and blob.get("config_hash") != expect_config_hash:
        raise SystemExit(f"ABORT -- checkpoint config hash {blob.get('config_hash')} "
                         f"!= expected {expect_config_hash}")
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    random.setstate(blob["rng_python"])
    np.random.set_state(blob["rng_numpy"])
    torch_mod.set_rng_state(blob["rng_torch"].cpu() if hasattr(blob["rng_torch"], "cpu")
                            else blob["rng_torch"])
    if torch_mod.cuda.is_available() and blob.get("rng_cuda"):
        torch_mod.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s
                                          for s in blob["rng_cuda"]])
    return {"epoch": blob["epoch"], "global_step": blob["global_step"],
            "aug_epoch": blob.get("aug_epoch", blob["epoch"]),
            "config_hash": blob.get("config_hash"),
            "manifest_sha256": blob.get("manifest_sha256")}


def step_stats(log: list[dict]) -> dict:
    if not log:
        return {}
    tot = np.array([r["t_data"] + r["t_fwd"] + r["t_bwd"] for r in log])
    d = np.array([r["t_data"] for r in log])
    f = np.array([r["t_fwd"] for r in log])
    b = np.array([r["t_bwd"] for r in log])
    e = np.array([r["edge_loss"] for r in log])
    dl = np.array([r["det_loss"] for r in log])
    return {
        "n_steps": len(log),
        "step_seconds": {"median": float(np.median(tot)), "p90": float(np.percentile(tot, 90)),
                         "max": float(tot.max()), "mean": float(tot.mean())},
        "data_wait_s": float(d.sum()), "forward_s": float(f.sum()), "backward_s": float(b.sum()),
        "data_wait_frac": float(d.sum() / tot.sum()),
        "gpu_compute_frac": float((f.sum() + b.sum()) / tot.sum()),
        "edge_loss": {"first10": float(e[:10].mean()), "last10": float(e[-10:].mean()),
                      "min": float(e.min()), "max": float(e.max())},
        "det_loss": {"first10": float(dl[:10].mean()), "last10": float(dl[-10:].mean())},
        "nonfinite": int((~np.isfinite(e)).sum() + (~np.isfinite(dl)).sum()),
    }


# ---------------------------------------------------------------- baseline loss weights
# CRITICAL: train_epoch()'s OWN defaults are (0.1, 0.1), but train() -- the baseline path --
# passes (1e1, 1e-2). Calling train_epoch without these silently trains with a 100x smaller
# detection weight and a 10x larger negative weight, which would make M1 incomparable to the
# baseline. Locked here and asserted in tests/test_m1_driver.py.
BASELINE_LOSS_WEIGHTS = {
    "det_loss_weight": 1e1,
    "det_neg_weight": 1e-2,
    "pool_kernel_um": 5.0,
}


class ResumableSampler:
    """Deterministic epoch ordering with exact mid-epoch resume.

    A fresh shuffle on resume is NOT acceptable here: it replays already-consumed samples
    instead of continuing. The epoch permutation is derived from (seed, epoch) so it can be
    reconstructed exactly, and iteration starts at the next unconsumed index.
    """

    def __init__(self, n: int, seed: int, epoch: int, start: int = 0,
                 length: int | None = None):
        self.n, self.seed, self.epoch = n, seed, epoch
        self.perm = np.random.default_rng([seed, epoch]).permutation(n)
        self.start = start
        self.length = min(length if length is not None else n, n)

    def __iter__(self):
        return iter(self.perm[self.start:self.length].tolist())

    def __len__(self) -> int:
        return max(0, self.length - self.start)

    def permutation_hash(self) -> str:
        import hashlib
        return hashlib.sha256(self.perm.tobytes()).hexdigest()[:16]

    def state(self) -> dict:
        return {"n": self.n, "seed": self.seed, "epoch": self.epoch,
                "start": self.start, "length": self.length,
                "permutation_hash": self.permutation_hash()}

    @classmethod
    def resume(cls, state: dict, step_in_epoch: int):
        s = cls(state["n"], state["seed"], state["epoch"],
                start=step_in_epoch, length=state["length"])
        if s.permutation_hash() != state["permutation_hash"]:
            raise SystemExit("ABORT -- epoch permutation hash mismatch on resume; the data "
                             "ordering could not be reconstructed")
        return s


# Per-sample identity log, used by the resume test to prove that resumed steps consume the
# SAME samples with the SAME augmentations. Disabled during full training (hashing every
# augmented volume is expensive).
SAMPLE_LOG: list[dict] = []
SAMPLE_LOG_ENABLED = [False]

SAMPLE_PATCH_OLD = """            meta = {**meta, "coords": c, "masks": m}"""
SAMPLE_PATCH_NEW = """            meta = {**meta, "coords": c, "masks": m}
        try:
            import m1_driver as _m1d
            if _m1d.SAMPLE_LOG_ENABLED[0]:
                import hashlib as _hl
                _m1d.SAMPLE_LOG.append({
                    "idx": int(idx),
                    "img_hash": _hl.sha256(
                        imgs.detach().cpu().float().numpy().tobytes()).hexdigest()[:12],
                })
        except Exception:
            pass"""


# Capture the gradient norm: clip_grad_norm_ RETURNS the pre-clip total norm but the
# vendored code discards it, so grad-norm statistics are otherwise unavailable.
GRADNORM_OLD = "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)"
GRADNORM_NEW = "        _m1_gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)"

TELEMETRY_RICH_OLD = '"t_bwd": float(t3 - t2),'
_NL = chr(10)
TELEMETRY_RICH_NEW = _NL.join([
    '"t_bwd": float(t3 - t2),',
    '                "grad_norm": float(_m1_gn),',
    '                "n_pos_targets": int(targets.sum().item()),',
])


def patch_all(src: str, seed: int, augment_module) -> str:
    """Apply every source patch: determinism, telemetry, grad-norm, sample identity."""
    src = patch_trainer_source(src, seed, augment_module)
    for old, new, label in ((GRADNORM_OLD, GRADNORM_NEW, "grad-norm"),
                            (TELEMETRY_RICH_OLD, TELEMETRY_RICH_NEW, "rich telemetry"),
                            (SAMPLE_PATCH_OLD, SAMPLE_PATCH_NEW, "sample identity")):
        if old not in src:
            raise SystemExit(f"ABORT -- {label} anchor missing; patch would no-op")
        src = src.replace(old, new, 1)
    return src
