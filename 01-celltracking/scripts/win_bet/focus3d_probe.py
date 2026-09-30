"""FOCUS-3D bounded CPU probe - PKT-0041 / LEVER-0043 steps 3 and 4-preparation.

WHY THIS FILE EXISTS AS A COMMITTED INSTRUMENT
----------------------------------------------
An auditor vetoed the first pass of these numbers, correctly. AGENTS.md requires a MEASURED fact to
be computed from a NAMED ARTIFACT by code that is COMMITTED, and a VERIFIED one to be re-derived at
a named file:line or by rerunning committed code. The first pass produced a parameter count from a
scratchpad script, and - this is the part that matters - **a parameter count taken from
instantiating the architecture with random initialisation is numerically indistinguishable from one
taken off the released tensors.** Nothing in that artifact let a reader tell them apart. The
precedent is FACT-0043, an anchor that proved unreproducible because its script was never committed.

So this file exists, and it makes the distinction MECHANICAL rather than a matter of the author's
care. Every payload carries `evidence_source`, which is one of:

    ARCHITECTURE_INSTANTIATION   built from the public config, RANDOM weights, no checkpoint read.
                                 A count from here can never be MEASURED or VERIFIED. The payload
                                 hard-codes provenance_ceiling: UNVERIFIED and says why.
    RELEASED_TENSORS             read out of a named checkpoint file whose sha256 is in the payload.
                                 Only this mode can support VERIFIED.

Refusing to emit the stronger label is the point. A mode that cannot see tensors says so in its own
output rather than leaving the reader to notice.

THREE SUBCOMMANDS
-----------------
  arch    Build FOCUS-3D from the publisher's config and report its shape. No checkpoint.
  ckpt    Read a checkpoint's ACTUAL tensors: sha256, wrapper structure, key prefixes, non-tensor
          keys, dtypes, parameter count, and the missing/unexpected sets against the built model.
  radius  Derive FOCUS-3D's `cell_radius` from OUR dense deployed node surface, with the FACT-0040
          calibration guard run first.

A SILENT NO-OP IN THE PUBLISHER'S LOADER, WHICH `ckpt` EXISTS TO CATCH
---------------------------------------------------------------------
`inference_win.build_predictor` calls `load_state_dict(..., strict=False)` and the two prints that
would report the missing and unexpected key counts are COMMENTED OUT in the released source. A
checkpoint whose keys do not match therefore loads SILENTLY and the model runs on random weights.
On a 300-query segmenter that looks exactly like "the foreign model does not join to our nodes" -
i.e. it would counterfeit falsifier (d). `ckpt` reports both counts and fails closed on request.

USAGE
-----
    python scripts/win_bet/focus3d_probe.py arch   --runtime C:/temp/focus3d/src/focus3d_runtime \
                                                   --config  C:/temp/focus3d/src/configs/3d_test.yaml
    python scripts/win_bet/focus3d_probe.py ckpt   --runtime ... --config ... \
                                                   --checkpoint C:/temp/focus3d/model_final_nuclei.pth
    python scripts/win_bet/focus3d_probe.py radius --preilp  C:/temp/p34_f1/preilp_split1.parquet \
                                                   --geff-dir data/train

Every subcommand prints a positive heartbeat as its LAST line. Its absence is the alarm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# Atlas scale, full-res (z, y, x) um per voxel. NOT isotropic - AGENTS.md s4, and two of three
# distance analyses in one day started with the wrong convention.
SCALE = np.array([1.625, 0.40625, 0.40625])

HEARTBEAT = "FOCUS3D_PROBE_COMPLETE"


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_runtime(runtime: Path, config: Path):
    """Import the publisher's pure-PyTorch runtime and build the model from its config."""
    runtime = runtime.resolve()
    if not runtime.is_dir():
        raise SystemExit(f"runtime root not found: {runtime}")
    if not config.is_file():
        raise SystemExit(f"config not found: {config}")
    sys.path.insert(0, str(runtime))
    from focus3d.segmentation.FOCUS3D.inference_win import (  # noqa: E402
        build_maskformer_model_from_cfg,
        setup_cfg,
    )

    cfg = setup_cfg(str(config), "", device="cpu")
    return build_maskformer_model_from_cfg(cfg).eval()


def _state_compatibility(model_state: dict, checkpoint_state: dict) -> dict:
    """Compare names, shapes and dtypes without letting ``strict=False`` hide a mismatch."""
    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_state)
    shared = sorted(model_keys & checkpoint_keys)
    shape_mismatch = [
        key for key in shared
        if tuple(model_state[key].shape) != tuple(checkpoint_state[key].shape)
    ]
    dtype_mismatch = [
        key for key in shared
        if getattr(model_state[key], "dtype", None) != getattr(checkpoint_state[key], "dtype", None)
    ]
    return {
        "missing": sorted(model_keys - checkpoint_keys),
        "unexpected": sorted(checkpoint_keys - model_keys),
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "shared": len(shared),
    }


def _attest_training_only_criterion_buffer(runtime: Path, config: Path,
                                            checkpoint_state: dict) -> dict:
    """Bind the sole inference-excluded tensor to publisher source and config values.

    The downloaded Windows runtime cannot instantiate its training criterion because it omits
    ``utils.misc_win``.  We therefore do not pretend a full-model strict load ran.  Instead this
    attestation proves the narrower fact needed by inference: the exact extra key is registered by
    the publisher's criterion source, the inference builder explicitly omits that criterion, and
    the released tensor equals ``ones(num_classes + 1)`` with the configured no-object weight in
    its last slot.  Any second extra key, source drift, shape drift, or value drift refuses.
    """
    import torch
    import yaml

    runtime = runtime.resolve()
    criterion_hits = list(runtime.rglob("criterion_win.py"))
    model_hits = list(runtime.rglob("maskformer_model_win.py"))
    if len(criterion_hits) != 1 or len(model_hits) != 1:
        raise RuntimeError(
            "training-only buffer attestation requires exactly one criterion_win.py and "
            "maskformer_model_win.py")
    criterion_path, model_path = criterion_hits[0], model_hits[0]
    criterion_text = criterion_path.read_text(encoding="utf-8")
    model_text = model_path.read_text(encoding="utf-8")
    required_criterion = "self.register_buffer('empty_weight', empty_weight)"
    if required_criterion not in criterion_text:
        raise RuntimeError("publisher source no longer registers criterion.empty_weight")
    if "if build_criterion:" not in model_text or "criterion = None" not in model_text:
        raise RuntimeError("publisher inference source no longer makes criterion omission explicit")

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    num_classes = int(cfg["MODEL"]["SEM_SEG_HEAD"]["NUM_CLASSES"])
    no_object_weight = float(cfg["MODEL"]["MASK_FORMER"]["NO_OBJECT_WEIGHT"])
    key = "criterion.empty_weight"
    if key not in checkpoint_state:
        raise RuntimeError(f"released checkpoint is missing the attested {key}")
    value = checkpoint_state[key]
    expected = torch.ones(num_classes + 1, dtype=value.dtype, device=value.device)
    expected[-1] = no_object_weight
    if tuple(value.shape) != (num_classes + 1,) or not torch.equal(value, expected):
        raise RuntimeError(
            f"{key} does not equal the publisher source/config construction; "
            f"shape={tuple(value.shape)}, expected_shape={(num_classes + 1,)}")
    return {
        "key": key,
        "shape": list(value.shape),
        "values": value.detach().cpu().tolist(),
        "num_classes": num_classes,
        "no_object_weight": no_object_weight,
        "criterion_source": str(criterion_path),
        "criterion_source_sha256": _sha256(criterion_path),
        "inference_builder_source": str(model_path),
        "inference_builder_source_sha256": _sha256(model_path),
        "source_registers_buffer": True,
        "inference_explicitly_omits_criterion": True,
        "value_matches_config": True,
    }


def _write_payload(path: str | None, payload: dict) -> None:
    """Persist evidence atomically; stdout alone is not a registry-grade artifact."""
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps({**payload, "heartbeat": HEARTBEAT}, indent=2, default=str),
                   encoding="utf-8")
    tmp.replace(output)


def _emit(payload: dict, output: str | None = None) -> None:
    _write_payload(output, payload)
    print(json.dumps(payload, indent=2, default=str))
    print(HEARTBEAT)


# ---------------------------------------------------------------------------------------------
# arch
# ---------------------------------------------------------------------------------------------
def cmd_arch(args: argparse.Namespace) -> None:
    model = _load_runtime(Path(args.runtime), Path(args.config))
    state = model.state_dict()

    per_module: dict[str, int] = {}
    for name, param in model.named_parameters():
        parts = name.split(".")
        key = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
        per_module[key] = per_module.get(key, 0) + param.numel()

    total = sum(p.numel() for p in model.parameters())
    _emit({
        "mode": "arch",
        "evidence_source": "ARCHITECTURE_INSTANTIATION",
        "provenance_ceiling": "UNVERIFIED",
        "provenance_ceiling_reason": (
            "Built from the public config with RANDOM initialisation. No checkpoint was read, so "
            "this count describes the architecture the config specifies and NOT the released "
            "weights. A count from this mode is numerically identical to one taken off the real "
            "tensors and must never be labelled MEASURED or VERIFIED. Run `ckpt` for that."
        ),
        "config": str(Path(args.config).resolve()),
        "runtime_root": str(Path(args.runtime).resolve()),
        "model_class": type(model).__name__,
        "total_parameters": total,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "buffer_elements": sum(b.numel() for b in model.buffers()),
        "state_dict_entries": len(state),
        "state_dict_numel": int(sum(v.numel() for v in state.values())),
        "fp32_weight_bytes": total * 4,
        "per_second_level_module": dict(sorted(per_module.items(), key=lambda kv: -kv[1])[:12]),
    }, getattr(args, "output", None))


# ---------------------------------------------------------------------------------------------
# ckpt
# ---------------------------------------------------------------------------------------------
def cmd_ckpt(args: argparse.Namespace) -> None:
    import torch

    path = Path(args.checkpoint)
    if not path.is_file():
        raise SystemExit(
            f"checkpoint not found: {path}\n"
            "This mode is the ONLY one that can support VERIFIED provenance. Without the file, "
            "report the architecture count from `arch` as UNVERIFIED and say so."
        )

    payload: dict = {
        "mode": "ckpt",
        "evidence_source": "RELEASED_TENSORS",
        "provenance_ceiling": "VERIFIED",
        "checkpoint": str(path.resolve()),
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": _sha256(path),
    }

    raw = torch.load(str(path), map_location="cpu", weights_only=False)

    # Mirror inference_win._extract_model_state_dict / _clean_state_dict_keys exactly, and RECORD
    # which branch fired rather than inferring it later from a log header.
    wrapper = None
    state = raw
    if isinstance(raw, dict):
        payload["top_level_keys"] = sorted(k for k in raw if isinstance(k, str))[:40]
        for candidate in ("model", "state_dict", "model_state_dict"):
            if candidate in raw and isinstance(raw[candidate], dict):
                wrapper, state = candidate, raw[candidate]
                break
    payload["wrapper_key"] = wrapper
    payload["is_bare_state_dict"] = wrapper is None

    tensors = {k: v for k, v in state.items() if hasattr(v, "numel")}
    non_tensor = {k: str(type(v)) for k, v in state.items() if not hasattr(v, "numel")}
    payload["state_dict_entries"] = len(state)
    payload["tensor_entries"] = len(tensors)
    payload["non_tensor_key_count"] = len(non_tensor)
    payload["non_tensor_keys"] = dict(list(non_tensor.items())[:20])
    payload["total_parameters_in_file"] = int(sum(int(v.numel()) for v in tensors.values()))
    payload["dtypes"] = sorted({str(v.dtype) for v in tensors.values()})

    prefixes: dict[str, int] = {}
    for key in tensors:
        head = key.split(".")[0]
        prefixes[head] = prefixes.get(head, 0) + 1
    payload["key_prefix_histogram"] = dict(sorted(prefixes.items(), key=lambda kv: -kv[1])[:15])
    payload["module_prefix_present"] = any(k.startswith("module.") for k in tensors)
    payload["model_prefix_present"] = any(k.startswith("model.") for k in tensors)

    if args.runtime and args.config:
        model = _load_runtime(Path(args.runtime), Path(args.config))
        cleaned = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("model."):
                key = key[len("model."):]
            cleaned[key] = value
        comparison = _state_compatibility(model.state_dict(), cleaned)
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        payload["built_model_state_dict_entries"] = len(model.state_dict())
        payload["missing_keys_count"] = len(missing)
        payload["unexpected_keys_count"] = len(unexpected)
        payload["missing_keys_sample"] = list(missing)[:15]
        payload["unexpected_keys_sample"] = list(unexpected)[:15]
        payload["strict_load_clean"] = (not missing) and (not unexpected)
        payload["shape_mismatch_count"] = len(comparison["shape_mismatch"])
        payload["dtype_mismatch_count"] = len(comparison["dtype_mismatch"])
        payload["shape_mismatch_sample"] = comparison["shape_mismatch"][:15]
        payload["dtype_mismatch_sample"] = comparison["dtype_mismatch"][:15]

        # Inference deliberately omits the loss criterion.  This is an allowlist, not a wildcard:
        # all inference keys must exist with identical shape and dtype, and the only permitted
        # checkpoint-only key is the criterion's registered class-weight buffer.  The attestation
        # below binds that exception to the publisher source and config; it is not a wildcard.
        allowed_training_only = {"criterion.empty_weight"}
        unexpected_set = set(comparison["unexpected"])
        payload["training_only_unexpected_keys"] = sorted(
            unexpected_set & allowed_training_only)
        payload["inference_state_clean"] = bool(
            not comparison["missing"]
            and not comparison["shape_mismatch"]
            and not comparison["dtype_mismatch"]
            and unexpected_set <= allowed_training_only
            and getattr(model, "criterion", None) is None
        )

        payload["training_only_buffer_attestation"] = \
            _attest_training_only_criterion_buffer(
                Path(args.runtime), Path(args.config), cleaned)
        payload["training_only_buffer_attested"] = True
        payload["why_this_check_exists"] = (
            "inference_win.build_predictor loads with strict=False and its missing/unexpected "
            "prints are commented out, so a key mismatch runs on random weights SILENTLY and "
            "counterfeits LEVER-0043 falsifier (d)."
        )
        payload["checkpoint_runtime_contract_clean"] = bool(
            payload["inference_state_clean"] and payload["training_only_buffer_attested"])
        if args.fail_on_mismatch and not payload["checkpoint_runtime_contract_clean"]:
            _emit(payload, getattr(args, "output", None))
            raise SystemExit(
                "REFUSED: checkpoint does not match the inference projection and full training "
                "buffer attestation "
                f"(inference_missing={len(missing)}, inference_unexpected={len(unexpected)}, "
                f"shape_mismatch={len(comparison['shape_mismatch'])}, "
                f"dtype_mismatch={len(comparison['dtype_mismatch'])})."
            )

    _emit(payload, getattr(args, "output", None))


# ---------------------------------------------------------------------------------------------
# radius
# ---------------------------------------------------------------------------------------------
def _calibration_guard(geff_dir: Path, limit_edges: int = 20000) -> dict:
    """Reproduce the GT displacement median before trusting any derived distance (FACT-0040).

    AGENTS.md: calibrate any derived quantity against an independently known value BEFORE using it.
    """
    import zarr

    crops = sorted(p for p in geff_dir.iterdir() if p.suffix == ".geff" and p.name.startswith("6bba"))
    out = {}
    for crop in crops:
        graph = zarr.open(str(crop), mode="r")
        props = graph["nodes/props"]
        ids = np.asarray(graph["nodes/ids"][:]).ravel()
        pts = np.stack([
            np.asarray(props["z/values"][:]).ravel(),
            np.asarray(props["y/values"][:]).ravel(),
            np.asarray(props["x/values"][:]).ravel(),
        ], axis=1) * SCALE
        edges = np.asarray(graph["edges/ids"][:])
        if len(edges) < 50:
            continue
        index = {int(v): i for i, v in enumerate(ids)}
        pairs = edges[:limit_edges]
        src = np.array([index[int(a)] for a, _ in pairs])
        dst = np.array([index[int(b)] for _, b in pairs])
        out[crop.stem] = round(float(np.median(np.linalg.norm(pts[dst] - pts[src], axis=1))), 4)
        if len(out) >= 4:
            break
    return out


def cmd_radius(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq
    from scipy.spatial import cKDTree

    payload: dict = {
        "mode": "radius",
        "evidence_source": "OUR_OWN_DEPLOYED_SURFACE",
        "preilp": str(Path(args.preilp).resolve()),
    }

    if args.geff_dir:
        guard = _calibration_guard(Path(args.geff_dir))
        payload["calibration_gt_displacement_median_um"] = guard
        payload["calibration_anchor"] = "FACT-0040 (fold-1 GT inter-frame displacement median)"
        payload["calibration_note"] = (
            "Must reproduce FACT-0040 before the radius below is trusted. If it does not, the "
            "coordinate convention or the anisotropic scale is wrong and nothing else here counts."
        )

    handle = pq.ParquetFile(args.preilp)
    columns = handle.schema_arrow.names
    needed = [c for c in ("dataset", "t", "z", "y", "x") if c in columns]
    if len(needed) < 5:
        raise SystemExit(f"expected dataset/t/z/y/x in {args.preilp}, found {columns}")
    table = handle.read(columns=needed).to_pandas()

    medians, counts = [], []
    for _, group in table.groupby(["dataset", "t"]):
        pts = np.unique(group[["z", "y", "x"]].to_numpy(dtype=np.float64), axis=0)
        if len(pts) < args.min_nodes:
            continue
        scaled = pts * SCALE
        dist, _ = cKDTree(scaled).query(scaled, k=2)
        medians.append(float(np.median(dist[:, 1])))
        counts.append(len(pts))
        if len(medians) >= args.max_groups:
            break

    if not medians:
        raise SystemExit("no group had enough nodes; a silent empty result is not a measurement")

    arr = np.array(medians)
    nn_um = float(np.median(arr))
    payload.update({
        "groups_measured": len(medians),
        "median_nodes_per_group": int(np.median(counts)),
        "median_nn_distance_um": round(nn_um, 4),
        "p25_nn_um": round(float(np.percentile(arr, 25)), 4),
        "p75_nn_um": round(float(np.percentile(arr, 75)), 4),
        "median_nn_xy_px": round(nn_um / SCALE[2], 3),
        "cell_radius_xy_px": round(nn_um / SCALE[2] / 2.0, 3),
        "focus3d_reference_cell_radius_px": 15.0,
        "implied_xy_resample_factor": round(nn_um / SCALE[2] / 2.0 / 15.0, 4),
        "z_ratio_for_focus3d": round(float(SCALE[0] / SCALE[2]), 6),
        "do_not_use_gt_for_this": (
            "The GT is SPARSE (FACT-0354 validity note), so GT nearest-neighbour distance measures "
            "ANNOTATION spacing, not nucleus spacing - it returns 22-37 um and an implied radius of "
            "27-46 px, which is wrong. Only the dense predicted surface answers this."
        ),
    })
    _emit(payload, getattr(args, "output", None))


# ---------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_arch = sub.add_parser("arch", help="architecture shape from the config; RANDOM weights")
    p_arch.add_argument("--runtime", required=True)
    p_arch.add_argument("--config", required=True)
    p_arch.add_argument("--output")
    p_arch.set_defaults(func=cmd_arch)

    p_ckpt = sub.add_parser("ckpt", help="read the RELEASED tensors; the only VERIFIED-capable mode")
    p_ckpt.add_argument("--checkpoint", required=True)
    p_ckpt.add_argument("--runtime")
    p_ckpt.add_argument("--config")
    p_ckpt.add_argument("--fail-on-mismatch", action="store_true")
    p_ckpt.add_argument("--output")
    p_ckpt.set_defaults(func=cmd_ckpt)

    p_rad = sub.add_parser("radius", help="cell_radius from OUR dense deployed node surface")
    p_rad.add_argument("--preilp", required=True)
    p_rad.add_argument("--geff-dir", help="run the FACT-0040 calibration guard first")
    p_rad.add_argument("--min-nodes", type=int, default=20)
    p_rad.add_argument("--max-groups", type=int, default=40)
    p_rad.add_argument("--output")
    p_rad.set_defaults(func=cmd_radius)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
