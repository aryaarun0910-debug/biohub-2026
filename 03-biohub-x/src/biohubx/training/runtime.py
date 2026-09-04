"""How a GPU session is spent: verified precision, a measured batch size, a hard ceiling, and honest counters.

[[D-0045]]: 13 GiB peak allocation per active T4 is a ceiling and not a target;
the operational target is examples per second and positives per step. Every
mechanism here is measured in the run that uses it and recorded in that run's
manifest, so a session that was spent badly says so.

Mixed precision is enabled only after an in-run verification: one fixed batch
is scored in full precision and under autocast and the logits and loss are
compared to a declared tolerance. A failure falls back to full precision and
records that it did, rather than training on numbers nobody checked.

The batch size is chosen by a ladder on the first training movie's own
patches, forward and backward at each rung, until the ceiling or the top rung;
the knee is the smallest rung reaching a declared fraction of the best
throughput. Positives per batch follow from the rung and are recorded beside
it, because with a 0.2 percent positive rate a small batch has none most of the
time and the loop skips it.

Propensity matching is E08's objective change and nothing else's: the
unlabelled risk is importance-weighted by a circumstance propensity fit on the
training movies, so the unlabelled distribution the risk is taken over has
the positives' circumstances and they carry no separating signal. Intensity and the response are excluded from
that fit by declaration; they are cellness cues as much as circumstances.

Consumer: ``biohubx.training.rescore_loop.run_loop``.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler

from biohubx.evaluation import propensity

MEMORY_CEILING_BYTES = 13 * 1024**3
"""D-0045's hard ceiling on peak allocation per active device."""

BATCH_LADDER: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)
"""Rungs tried in order; the ladder stops at the first rung that breaks the ceiling or fails."""

KNEE_FRACTION = 0.90
"""The smallest rung reaching this fraction of the best measured throughput is the knee."""

AMP_LOSS_TOLERANCE = 1e-2
"""Relative loss difference full precision versus autocast above which AMP is refused."""

AMP_LOGIT_TOLERANCE = 5e-2
"""Max absolute logit difference above which AMP is refused."""

MATCHED_WEIGHT_CAP = 20.0
"""Importance weights are clipped to [1/cap, cap] before normalisation.

Eighteen positives against twelve thousand unlabelled rows fit a propensity
whose odds span many orders of magnitude, and an unclipped weight lets a
handful of rows carry the whole unlabelled risk. The cap bounds how far the
matched distribution can move from the observed one; the effective sample
size after clipping is recorded so the bound's cost is visible.
"""

MATCHED_ON: tuple[str, ...] = (
    "z",
    "y",
    "x",
    "face_um",
    "t",
    "density",
    "persist_prev",
    "persist_next",
    "motion_um",
)
"""Circumstances the E08 sampler matches on. ``dog`` and ``intensity`` are excluded by declaration."""


class RuntimeRefusal(RuntimeError):
    """The session would exceed a ceiling or train on unverified numbers."""


def memory_ceiling(device: torch.device, ceiling_bytes: int = MEMORY_CEILING_BYTES) -> dict[str, Any]:
    """Cap the allocator so the ceiling is enforced by CUDA, not only observed afterwards."""
    if device.type != "cuda":
        return {"enforced": False, "reason": "no accelerator"}
    total = int(torch.cuda.get_device_properties(device).total_memory)
    fraction = min(1.0, ceiling_bytes / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return {"enforced": True, "ceiling_bytes": ceiling_bytes, "total_bytes": total, "fraction": fraction}


def peak_allocated(device: torch.device) -> int:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


def check_ceiling(device: torch.device, ceiling_bytes: int, where: str) -> None:
    peak = peak_allocated(device)
    if peak > ceiling_bytes:
        raise RuntimeRefusal(
            f"peak allocation {peak} bytes exceeded the {ceiling_bytes} byte ceiling at {where}"
        )


def autocast_context(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager[Any]:
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def verify_amp(
    model: torch.nn.Module,
    patches: torch.Tensor,
    labels: torch.Tensor,
    *,
    prior: float,
    device: torch.device,
    loss_fn: Callable[..., tuple[torch.Tensor, dict[str, float]]],
) -> dict[str, Any]:
    """Full precision against autocast on one fixed batch; AMP is enabled only if they agree."""
    if device.type != "cuda":
        return {"requested": True, "enabled": False, "reason": "autocast float16 needs a CUDA device"}
    model.eval()
    with torch.no_grad():
        full = model(patches.to(device)).float()
        loss_full, _ = loss_fn(full, labels.to(device), prior=prior)
        with torch.autocast("cuda", dtype=torch.float16):
            mixed = model(patches.to(device))
        mixed = mixed.float()
        loss_mixed, _ = loss_fn(mixed, labels.to(device), prior=prior)
    model.train()
    logit_gap = float((full - mixed).abs().max().item()) if full.numel() else 0.0
    loss_gap = float(abs(loss_full.item() - loss_mixed.item()) / max(abs(loss_full.item()), 1e-8))
    ok = logit_gap <= AMP_LOGIT_TOLERANCE and loss_gap <= AMP_LOSS_TOLERANCE
    return {
        "requested": True,
        "enabled": ok,
        "max_abs_logit_difference": logit_gap,
        "relative_loss_difference": loss_gap,
        "logit_tolerance": AMP_LOGIT_TOLERANCE,
        "loss_tolerance": AMP_LOSS_TOLERANCE,
        "reason": "verified"
        if ok
        else "autocast disagreed with full precision beyond tolerance; full precision kept",
    }


def autotune_batch(
    model: torch.nn.Module,
    patches: torch.Tensor,
    labels: torch.Tensor,
    *,
    prior: float,
    device: torch.device,
    loss_fn: Callable[..., tuple[torch.Tensor, dict[str, float]]],
    amp: bool,
    ceiling_bytes: int,
    ladder: Sequence[int] = BATCH_LADDER,
    positive_rate: float = 0.0,
) -> dict[str, Any]:
    """Forward and backward at each rung on training patches; the throughput knee below the ceiling wins.

    Runs on a throwaway optimizer so nothing here trains the model; the model's
    parameters are restored afterwards. On CPU the ladder is not climbed: the
    first rung is returned, because a CPU exercise measures nothing about the
    card.
    """
    if device.type != "cuda":
        return {"chosen": int(ladder[0]), "rungs": [], "reason": "no accelerator; first rung"}
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rungs: list[dict[str, Any]] = []
    best = 0.0
    scaler = GradScaler("cuda", enabled=amp)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    for size in ladder:
        if size > patches.shape[0]:
            break
        torch.cuda.reset_peak_memory_stats(device)
        try:
            chosen = torch.arange(size)
            x = patches[chosen].to(device)
            y = labels[chosen].to(device)
            if int(y.sum()) == 0:
                y = y.clone()
                y[0] = True
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with autocast_context(device, amp):
                logits = model(x)
                loss, _ = loss_fn(logits.float(), y, prior=prior)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            peak = peak_allocated(device)
            rate = size / elapsed if elapsed > 0 else 0.0
            rung = {
                "batch": int(size),
                "seconds": round(elapsed, 4),
                "examples_per_second": round(rate, 1),
                "peak_allocated_bytes": peak,
                "expected_positives_per_batch": round(size * positive_rate, 2),
                "under_ceiling": peak <= ceiling_bytes,
            }
            rungs.append(rung)
            if peak > ceiling_bytes:
                break
            best = max(best, rate)
        except torch.OutOfMemoryError:
            rungs.append({"batch": int(size), "out_of_memory": True, "under_ceiling": False})
            torch.cuda.empty_cache()
            break
        finally:
            del_names = [name for name in ("x", "y", "logits", "loss") if name in locals()]
            for name in del_names:
                locals().pop(name, None)
    model.load_state_dict(state, strict=True)
    torch.cuda.reset_peak_memory_stats(device)
    eligible = [r for r in rungs if r.get("under_ceiling") and "examples_per_second" in r]
    if not eligible:
        raise RuntimeRefusal("no batch rung fits under the memory ceiling")
    knee = next(r for r in eligible if r["examples_per_second"] >= KNEE_FRACTION * best)
    return {
        "chosen": int(knee["batch"]),
        "rungs": rungs,
        "best_examples_per_second": round(best, 1),
        "knee_fraction": KNEE_FRACTION,
    }


@dataclass
class ThroughputMeter:
    """Counters a session is judged by, recorded rather than remembered."""

    examples: int = 0
    positives: int = 0
    batches: int = 0
    skipped_no_positive: int = 0
    clamped: int = 0
    compute_seconds: float = 0.0
    wait_seconds: float = 0.0
    utilisation_samples: list[float] = field(default_factory=list)

    def sample_utilisation(self, device: torch.device) -> None:
        if device.type != "cuda":
            return
        try:
            self.utilisation_samples.append(float(torch.cuda.utilization(device)))
        except Exception:  # pynvml may be absent; the counter is optional, the run is not
            return

    def to_dict(self) -> dict[str, Any]:
        total = self.compute_seconds + self.wait_seconds
        return {
            "examples": self.examples,
            "positives": self.positives,
            "batches": self.batches,
            "skipped_no_positive": self.skipped_no_positive,
            "clamped_batches": self.clamped,
            "clamp_frequency": round(self.clamped / self.batches, 4) if self.batches else None,
            "positives_per_batch": round(self.positives / self.batches, 3) if self.batches else None,
            "examples_per_second": round(self.examples / self.compute_seconds, 1)
            if self.compute_seconds
            else None,
            "compute_seconds": round(self.compute_seconds, 3),
            "data_wait_seconds": round(self.wait_seconds, 3),
            "data_wait_fraction": round(self.wait_seconds / total, 4) if total else None,
            "gpu_utilisation_median": (
                float(np.median(self.utilisation_samples)) if self.utilisation_samples else None
            ),
        }


def matched_sampling_weights(
    circumstances: Sequence[np.ndarray], labels: Sequence[np.ndarray], *, l2: float = 0.01
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Per-movie importance weights for unlabelled rows, matching their circumstances to the positives'.

    One logistic propensity model over ``MATCHED_ON`` is fit on every training
    movie's rows together (positives against unlabelled), standardised on the
    same rows. An unlabelled row's weight is its odds of being a positive under
    that model, which is the importance ratio that reweights the unlabelled
    distribution onto the positive one; positives keep weight one. Weights are
    normalised per movie so the loss's class prior is unchanged.
    """
    columns = [propensity.FEATURE_NAMES.index(name) for name in MATCHED_ON]
    x_all = np.vstack([c[:, columns] for c in circumstances])
    y_all = np.concatenate([np.asarray(lab, dtype=bool) for lab in labels])
    if y_all.sum() == 0:
        raise RuntimeRefusal("propensity matching needs at least one positive across the training movies")
    mean = x_all.mean(axis=0)
    std = x_all.std(axis=0) + 1e-9
    weights_vector = propensity.fit_logistic((x_all - mean) / std, y_all, l2=l2)
    per_movie: list[np.ndarray] = []
    effective: list[float] = []
    for c, lab in zip(circumstances, labels, strict=True):
        z = ((c[:, columns] - mean) / std) @ weights_vector[:-1] + weights_vector[-1]
        odds = np.clip(np.exp(np.clip(z, -20.0, 20.0)), 1.0 / MATCHED_WEIGHT_CAP, MATCHED_WEIGHT_CAP)
        w = np.where(np.asarray(lab, dtype=bool), 1.0, odds)
        unl = ~np.asarray(lab, dtype=bool)
        if unl.any():
            w[unl] = w[unl] / w[unl].sum() * unl.sum()
        per_movie.append(w.astype(np.float64))
        effective.append(float(w[unl].sum() ** 2 / (w[unl] ** 2).sum()) if unl.any() else 0.0)
    return per_movie, {
        "matched_on": list(MATCHED_ON),
        "weight_cap": MATCHED_WEIGHT_CAP,
        "coefficients": {
            name: round(float(v), 4) for name, v in zip(MATCHED_ON, weights_vector[:-1], strict=True)
        },
        "effective_unlabelled_per_movie": [round(e, 1) for e in effective],
    }
