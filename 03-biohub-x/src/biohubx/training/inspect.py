"""Measure a detector architecture before anyone trains it: what it is, what it costs, what it can see.

Every function here takes a model and returns numbers a preregistration can
quote: state-dict coverage under strict loading, parameter and buffer census,
the shapes each module sees, downsampling and the physical spacing of the
output, the empirical receptive field, graph capture, activation volume,
timings, memory, the profiler's expensive operators, output calibration,
gradient norms by block, inert-channel and temporal-perturbation probes, and a
deterministic strict-reload comparison. With an extractor attached it also
turns the output into proposals and scores the oracle ceiling over them.

None of this is a finding. A parameter count, a FLOP estimate or a pleasing
feature map says nothing about cell tracking; the analyzer exists so that
Stage 0 of the training funnel can kill an impossible or duplicative design
on measurement rather than on taste, and so the survivors' costs are known
before a GPU is asked for. PyTorch built-ins only.

Consumer: ``biohubx model inspect``, for the learned-proposal experiment family.
"""

from __future__ import annotations

import contextlib
import math
import os
import pathlib
import platform
import statistics
import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

Forward = Callable[[nn.Module, torch.Tensor], torch.Tensor]
"""How to run the model on one input batch and get the logits the loss sees."""


class InspectError(ValueError):
    """The analyzer cannot measure what it was asked to measure."""


# ---------------------------------------------------------------------------
# Identity and census
# ---------------------------------------------------------------------------


def state_dict_coverage(model: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Strict coverage: every model key present in the checkpoint with the same shape, and nothing extra.

    A checkpoint that loads with ``strict=False`` and silently leaves a head at
    its random initialisation is the failure this is written against.
    """
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        key for key in set(expected) & set(state) if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    return {
        "model_keys": len(expected),
        "checkpoint_keys": len(state),
        "missing_in_checkpoint": missing,
        "unexpected_in_checkpoint": unexpected,
        "shape_mismatch": mismatched,
        "strict_ok": not missing and not unexpected and not mismatched,
    }


def parameter_census(model: nn.Module) -> dict[str, Any]:
    """Parameters and buffers by top-level child, with trainable and frozen counts."""
    by_module: dict[str, dict[str, int]] = OrderedDict()
    for name, child in model.named_children():
        params = list(child.parameters())
        by_module[name] = {
            "parameters": sum(p.numel() for p in params),
            "trainable": sum(p.numel() for p in params if p.requires_grad),
            "frozen": sum(p.numel() for p in params if not p.requires_grad),
            "buffers": sum(b.numel() for b in child.buffers()),
        }
    total = list(model.parameters())
    return {
        "total_parameters": sum(p.numel() for p in total),
        "trainable_parameters": sum(p.numel() for p in total if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in total if not p.requires_grad),
        "total_buffers": sum(b.numel() for b in model.buffers()),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in total),
        "by_module": by_module,
    }


# ---------------------------------------------------------------------------
# Shapes, downsampling, receptive field
# ---------------------------------------------------------------------------


def _shape_of(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
    if isinstance(value, list | tuple):
        return [_shape_of(item) for item in value]
    return None


@contextlib.contextmanager
def _hooks(model: nn.Module, hook: Callable[..., None], *, leaves_only: bool) -> Iterator[None]:
    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if leaves_only and any(True for _ in module.children()):
            continue
        handles.append(module.register_forward_hook(lambda m, i, o, _name=name: hook(_name, m, i, o)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def shape_trace(model: nn.Module, forward: Forward, example: torch.Tensor) -> list[dict[str, Any]]:
    """Input and output shapes and dtypes of every leaf module, in execution order."""
    rows: list[dict[str, Any]] = []

    def record(name: str, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        rows.append(
            {
                "module": name,
                "type": type(module).__name__,
                "inputs": [_shape_of(i) for i in inputs],
                "output": _shape_of(output),
            }
        )

    model.eval()
    with torch.no_grad(), _hooks(model, record, leaves_only=True):
        forward(model, example)
    return rows


def downsampling(
    input_shape: tuple[int, ...], output_shape: tuple[int, ...], input_spacing_um: tuple[float, ...]
) -> dict[str, Any]:
    """Per-axis stride between input and output, and the physical spacing that implies.

    Axes are compared from the right, which is where the spatial axes live;
    ``input_spacing_um`` is given for those same trailing axes.
    """
    n = len(input_spacing_um)
    if len(input_shape) < n or len(output_shape) < n:
        raise InspectError("fewer spatial axes than spacings were given")
    factors = []
    for i_dim, o_dim in zip(input_shape[-n:], output_shape[-n:], strict=True):
        factors.append(round(i_dim / o_dim, 4) if o_dim else math.inf)
    return {
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "factors": factors,
        "input_spacing_um": list(input_spacing_um),
        "output_spacing_um": [round(s * f, 4) for s, f in zip(input_spacing_um, factors, strict=True)],
    }


def empirical_receptive_field(
    model: nn.Module, forward: Forward, example: torch.Tensor, *, spatial_axes: int
) -> dict[str, Any]:
    """The input extent one central output voxel depends on, measured by gradient.

    Theoretical receptive fields are only determinable for plain convolutional
    stacks; a UNet with skip connections has several. What is always measurable
    is the region of the input whose perturbation reaches one output voxel,
    which is the number a crop size has to respect. Measured in eval mode on a
    constant-plus-noise input so that no activation is exactly at a kink.
    """
    model.eval()
    probe = example.detach().clone().float()
    probe = probe + 1e-3 * torch.randn_like(probe)
    probe.requires_grad_(True)
    output = forward(model, probe)
    centre = tuple(s // 2 for s in output.shape)
    output[centre].backward()  # type: ignore[no-untyped-call]
    if probe.grad is None:
        raise InspectError("no gradient reached the input; the output does not depend on it")
    grad = probe.grad.abs()
    extent: list[int] = []
    for axis in range(probe.ndim - spatial_axes, probe.ndim):
        reduce_axes = tuple(a for a in range(probe.ndim) if a != axis)
        nonzero = torch.nonzero(grad.sum(dim=reduce_axes) > 0).flatten()
        extent.append(int(nonzero.max() - nonzero.min() + 1) if nonzero.numel() else 0)
    return {
        "method": "gradient of one central output voxel with respect to the input",
        "output_voxel": list(centre),
        "extent_voxels": extent,
        "input_extent_voxels": list(probe.shape[-spatial_axes:]),
        "clipped_by_input": [e >= s for e, s in zip(extent, probe.shape[-spatial_axes:], strict=True)],
    }


# ---------------------------------------------------------------------------
# Graph capture and activation census
# ---------------------------------------------------------------------------


def capture_graph(model: nn.Module, forward: Forward, example: torch.Tensor) -> dict[str, Any]:
    """torch.export first, torch.fx second, and a named failure when neither captures."""

    class Wrapped(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return forward(self.inner, x)

    wrapped = Wrapped(model).eval()
    try:
        exported = torch.export.export(wrapped, (example,))
        ops: dict[str, int] = {}
        for node in exported.graph.nodes:
            if node.op == "call_function":
                key = (
                    str(node.target).split(".")[-1]
                    if hasattr(node.target, "__module__")
                    else str(node.target)
                )
                ops[key] = ops.get(key, 0) + 1
        return {
            "method": "torch.export",
            "captured": True,
            "nodes": len(list(exported.graph.nodes)),
            "call_function_ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])[:25]),
        }
    except Exception as export_error:
        try:
            traced = torch.fx.symbolic_trace(wrapped)
            ops = {}
            for node in traced.graph.nodes:
                if node.op in ("call_function", "call_module", "call_method"):
                    key = f"{node.op}:{node.target}" if node.op != "call_module" else f"module:{node.target}"
                    ops[key] = ops.get(key, 0) + 1
            return {
                "method": "torch.fx.symbolic_trace",
                "captured": True,
                "export_failure": f"{type(export_error).__name__}: {str(export_error)[:200]}",
                "nodes": len(list(traced.graph.nodes)),
                "ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])[:25]),
            }
        except Exception as fx_error:
            return {
                "method": None,
                "captured": False,
                "export_failure": f"{type(export_error).__name__}: {str(export_error)[:200]}",
                "fx_failure": f"{type(fx_error).__name__}: {str(fx_error)[:200]}",
            }


def activation_census(model: nn.Module, forward: Forward, example: torch.Tensor) -> dict[str, Any]:
    """Elements and bytes every leaf module emits, and the largest single activation."""
    totals = {"elements": 0, "bytes": 0}
    largest: dict[str, Any] = {"module": None, "elements": 0, "bytes": 0}

    def record(name: str, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        tensors = (
            [output]
            if isinstance(output, torch.Tensor)
            else [o for o in output if isinstance(o, torch.Tensor)]
        )
        for t in tensors:
            totals["elements"] += t.numel()
            totals["bytes"] += t.numel() * t.element_size()
            if t.numel() > largest["elements"]:
                largest.update({"module": name, "elements": t.numel(), "bytes": t.numel() * t.element_size()})

    model.eval()
    with torch.no_grad(), _hooks(model, record, leaves_only=True):
        forward(model, example)
    return {"total_elements": totals["elements"], "total_bytes": totals["bytes"], "largest": largest}


# ---------------------------------------------------------------------------
# Cost: timings, memory, profiler
# ---------------------------------------------------------------------------


def _rss_bytes() -> int | None:
    """Resident set size without a third-party dependency."""
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        query = kernel32.K32GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        if query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    try:
        with pathlib.Path("/proc/self/status").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


@dataclass
class StepTimings:
    forward_seconds: list[float] = field(default_factory=list)
    forward_backward_seconds: list[float] = field(default_factory=list)
    optimizer_step_seconds: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        def stats(values: list[float]) -> dict[str, float]:
            return (
                {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "n": len(values),
                }
                if values
                else {}
            )

        return {
            "forward_seconds": stats(self.forward_seconds),
            "forward_backward_seconds": stats(self.forward_backward_seconds),
            "optimizer_step_seconds": stats(self.optimizer_step_seconds),
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timings_and_memory(
    model: nn.Module,
    forward: Forward,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    example: torch.Tensor,
    *,
    device: torch.device,
    warmup: int = 1,
    repeats: int = 3,
) -> dict[str, Any]:
    """Forward, forward+backward and optimizer step timings after warm-up, with peak memory."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    timings = StepTimings()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(warmup + repeats):
        measured = iteration >= warmup
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            forward(model, example)
        _sync(device)
        t1 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(forward(model, example))
        loss.backward()  # type: ignore[no-untyped-call]
        _sync(device)
        t2 = time.perf_counter()
        optimizer.step()
        _sync(device)
        t3 = time.perf_counter()
        if measured:
            timings.forward_seconds.append(t1 - t0)
            timings.forward_backward_seconds.append(t2 - t1)
            timings.optimizer_step_seconds.append(t3 - t2)
    memory: dict[str, Any] = {"cpu_peak_rss_bytes": _rss_bytes()}
    if device.type == "cuda":
        memory["cuda_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        memory["cuda_max_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return {"timings": timings.summary(), "memory": memory, "warmup": warmup, "repeats": repeats}


def profile_operators(
    model: nn.Module,
    forward: Forward,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    example: torch.Tensor,
    *,
    device: torch.device,
    top: int = 12,
) -> list[dict[str, Any]]:
    """The operators that cost the most self time in one forward and backward pass."""
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    model.train()
    with profile(activities=activities) as prof:
        loss = loss_fn(forward(model, example))
        loss.backward()  # type: ignore[no-untyped-call]
    rows = []
    key = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
    for event in sorted(prof.key_averages(), key=lambda e: -getattr(e, key, 0.0))[:top]:
        rows.append(
            {
                "operator": event.key,
                "calls": int(event.count),
                "self_cpu_ms": round(event.self_cpu_time_total / 1000.0, 3),
                "self_device_ms": round(getattr(event, "self_device_time_total", 0.0) / 1000.0, 3),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Behaviour: calibration, gradients, probes, determinism
# ---------------------------------------------------------------------------


def output_statistics(logits: torch.Tensor) -> dict[str, Any]:
    flat = logits.detach().float().flatten()
    probabilities = torch.sigmoid(flat)
    quantiles = torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=flat.dtype)
    return {
        "elements": int(flat.numel()),
        "logit_min": float(flat.min()),
        "logit_max": float(flat.max()),
        "logit_mean": float(flat.mean()),
        "logit_std": float(flat.std()) if flat.numel() > 1 else 0.0,
        "probability_quantiles": {
            f"q{q:g}": float(v)
            for q, v in zip(
                quantiles.tolist(), torch.quantile(probabilities, quantiles).tolist(), strict=True
            )
        },
        "fraction_above_0_5": float((probabilities > 0.5).float().mean()),
        "fraction_above_0_9": float((probabilities > 0.9).float().mean()),
        "non_finite": int((~torch.isfinite(flat)).sum()),
    }


def gradient_norm_by_block(model: nn.Module) -> dict[str, Any]:
    """L2 gradient norm per top-level child, after a backward pass has run."""
    blocks: dict[str, float] = {}
    total = 0.0
    zero_blocks = []
    for name, child in model.named_children():
        squares = [float((p.grad**2).sum()) for p in child.parameters() if p.grad is not None]
        norm = math.sqrt(sum(squares)) if squares else 0.0
        blocks[name] = norm
        total += sum(squares)
        if not squares or norm == 0.0:
            zero_blocks.append(name)
    return {"total": math.sqrt(total), "by_block": blocks, "zero_gradient_blocks": zero_blocks}


def input_channel_probes(
    model: nn.Module, forward: Forward, example: torch.Tensor, *, channel_axis: int
) -> list[dict[str, Any]]:
    """Zero each input channel in turn and measure how much the output moves.

    A channel whose removal moves nothing is inert: either the model ignores it
    or the design never wired it. Relative L2 change against the unperturbed
    output, so the number is comparable across architectures.
    """
    model.eval()
    with torch.no_grad():
        baseline = forward(model, example)
        scale = float(baseline.norm()) or 1.0
        rows = []
        for channel in range(example.shape[channel_axis]):
            probe = example.clone()
            index: list[Any] = [slice(None)] * example.ndim
            index[channel_axis] = channel
            probe[tuple(index)] = 0.0
            changed = forward(model, probe)
            relative = float((changed - baseline).norm()) / scale
            rows.append({"channel": channel, "relative_output_change": relative, "inert": relative < 1e-6})
    return rows


def temporal_perturbation(
    model: nn.Module, forward: Forward, example: torch.Tensor, *, time_axis: int
) -> dict[str, Any]:
    """How much the output depends on temporal context: reversed order, and frozen neighbours.

    Reversal keeps every frame and changes only their order; a model that uses
    no temporal context returns the same per-frame output reordered, so the
    reordered difference is zero. Replacing the neighbours of the centre frame
    with the centre frame itself removes motion cues while keeping intensities.
    """
    model.eval()
    frames = example.shape[time_axis]
    with torch.no_grad():
        baseline = forward(model, example)
        scale = float(baseline.norm()) or 1.0
        reversed_input = torch.flip(example, dims=[time_axis])
        reversed_output = forward(model, reversed_input)
        reordered_difference = (
            float((torch.flip(reversed_output, dims=[time_axis]) - baseline).norm()) / scale
        )
        centre = frames // 2
        frozen = example.clone()
        index: list[Any] = [slice(None)] * example.ndim
        for f in range(frames):
            if f == centre:
                continue
            index[time_axis] = f
            src: list[Any] = [slice(None)] * example.ndim
            src[time_axis] = centre
            frozen[tuple(index)] = example[tuple(src)]
        frozen_output = forward(model, frozen)
        frozen_difference = float((frozen_output - baseline).norm()) / scale
    return {
        "frames": int(frames),
        "reversed_order_relative_change": reordered_difference,
        "neighbours_frozen_relative_change": frozen_difference,
        "uses_temporal_context": reordered_difference > 1e-6 or frozen_difference > 1e-6,
    }


def strict_reload_determinism(
    build: Callable[[], nn.Module], model: nn.Module, forward: Forward, example: torch.Tensor
) -> dict[str, Any]:
    """Save, rebuild, load strictly, compare outputs exactly; run twice and compare again."""
    model.eval()
    with torch.no_grad():
        first = forward(model, example)
        second = forward(model, example)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rebuilt = build().to(example.device)
    rebuilt.load_state_dict(state, strict=True)
    rebuilt.eval()
    with torch.no_grad():
        reloaded = forward(rebuilt, example)
    return {
        "strict_reload_ok": True,
        "output_identical_after_reload": bool(torch.equal(first, reloaded)),
        "output_identical_between_runs": bool(torch.equal(first, second)),
        "max_abs_difference_after_reload": float((first - reloaded).abs().max()),
    }


def environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "threads": torch.get_num_threads(),
        "pid": os.getpid(),
    }


def precision_dtype(name: str) -> torch.dtype:
    table = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    if name not in table:
        raise InspectError(f"precision must be one of {sorted(table)}, got {name!r}")
    return table[name]


def to_jsonable(value: Any) -> Any:
    """Numbers and containers only; tensors and arrays become lists or scalars."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor | np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value
