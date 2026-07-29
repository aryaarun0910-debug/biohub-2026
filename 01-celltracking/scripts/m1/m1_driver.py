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
