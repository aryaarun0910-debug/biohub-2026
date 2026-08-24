"""Fail-closed AdaBN detector path for the unchanged P3 association substrate.

The calibrated checkpoint is never installed as the primary model.  A second model supplies
only detector logits; the original P3 model still supplies ``unet_out`` and every edge logit.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path


PRIMARY_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
CALIBRATED_SHA256 = "1371c35a63a55366e6daba77f025b7c39b57a7ba7a63d516cd70b2c3d5e358de"
CONFIG_SHA256 = "fdeff5809574543376eeee4fe83d9c344f45680535b505c9d712ecd77b966a57"
SUMMARY_SHA256 = "eb965ba98e403f50f4b6f74dee38ab624f9e34ef365ebb2163e972e0e6300421"
AUDIT_SHA256 = "a67bfab726835a509ac25647560945889160623efcd3f133e276416db67cc0f8"
CALIBRATION_MODE = "detection_only_bn_buffers"
_BN_BUFFER_SUFFIXES = (".running_mean", ".running_var", ".num_batches_tracked")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_adabn_artifact_contract(audit: dict, summary: dict) -> dict:
    """Accept only the completed LR=0 calibration control, never a trained checkpoint."""
    contract = {
        "audit_contract": audit.get("contract"),
        "summary_contract": summary.get("trunk_contract"),
        "lr": float(audit.get("lr", float("nan"))),
        "selection_threshold": summary.get("selection_threshold"),
        "epochs_run": int(summary.get("epochs_run", -1)),
        "finite_optimizer_steps": int(audit.get("finite_optimizer_steps", 0)),
    }
    if contract["audit_contract"] != "adabn_control" \
            or contract["summary_contract"] != "adabn_control":
        raise RuntimeError(f"AdaBN consumer rejects non-control artifact: {contract}")
    if contract["lr"] != 0.0:
        raise RuntimeError(f"AdaBN consumer requires an exact LR=0 run: {contract}")
    if contract["selection_threshold"] != "deployed" or contract["epochs_run"] != 1:
        raise RuntimeError(f"AdaBN consumer rejects mismatched control protocol: {contract}")
    if contract["finite_optimizer_steps"] <= 0:
        raise RuntimeError(f"AdaBN consumer requires finite forward/backward evidence: {contract}")
    return contract


def validate_adabn_state_contract(primary: dict, calibrated: dict) -> list[str]:
    """Prove that the only checkpoint changes are U-Net BatchNorm running buffers."""
    if set(primary) != set(calibrated):
        raise RuntimeError({
            "missing": sorted(set(primary) - set(calibrated)),
            "unexpected": sorted(set(calibrated) - set(primary)),
        })

    expected_buffers = {
        key for key in primary
        if key.startswith("unet.") and key.endswith(_BN_BUFFER_SUFFIXES)
    }
    changed: set[str] = set()
    incompatible: list[str] = []
    for key in sorted(primary):
        before, after = primary[key], calibrated[key]
        if before.shape != after.shape or before.dtype != after.dtype:
            incompatible.append(key)
            continue
        if not before.equal(after):
            changed.add(key)
    if incompatible:
        raise RuntimeError(f"AdaBN tensor shape/dtype drift: {incompatible}")

    forbidden = sorted(changed - expected_buffers)
    if forbidden:
        raise RuntimeError(
            "AdaBN checkpoint changes learned/association tensors: " + ", ".join(forbidden)
        )
    missing = sorted(expected_buffers - changed)
    if missing:
        raise RuntimeError(
            "AdaBN checkpoint did not update the complete BN-buffer set: " + ", ".join(missing)
        )
    if len(changed) != 30:
        raise RuntimeError(f"AdaBN checkpoint expected 30 changed BN buffers, got {len(changed)}")
    return sorted(changed)


def patch_detection_only_predictor(path: Path) -> str:
    """Wire calibrated logits into detection while leaving association on primary ``unet_out``."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")

    signature_old = """    secondary_low_margin_max: float = 0.2,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:"""
    signature_new = """    secondary_low_margin_max: float = 0.2,
    adabn_detection_model: UNetNodeTransformer | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:"""
    if source.count(signature_old) != 1:
        raise RuntimeError("AdaBN patch expected one final P3 predict_video signature")
    source = source.replace(signature_old, signature_new, 1)

    encode_old = """        unet_out, det_logits = model.encode(imgs)
        # unet_out: (1, W, C, *spatial_down), det_logits: list of W × (1, 1, *spatial_down)
"""
    encode_new = """        # PRIMARY P3 PATH: identity-view features are association-only and read-only.
        unet_out, _primary_det_logits_unused = model.encode(imgs)
        if adabn_detection_model is None:
            raise RuntimeError("AdaBN detection model is required by this isolated candidate")
        # CALIBRATED PATH: only detector logits escape; adapted feature maps are discarded.
        _adabn_features_unused, det_logits = adabn_detection_model.encode(imgs)
        del _adabn_features_unused, _primary_det_logits_unused
        # unet_out remains the original P3 association representation.
"""
    if source.count(encode_old) != 1:
        raise RuntimeError("AdaBN patch expected one primary encode block")
    source = source.replace(encode_old, encode_new, 1)

    tta_start = source.index("        if cfg.det_tta:", source.index(encode_new))
    tta_end = source.index("        secondary_unet_out = None", tta_start)
    tta_block = source[tta_start:tta_end]
    if tta_block.count("model.encode(") != 4:
        raise RuntimeError(
            f"AdaBN patch expected four deployed P3 TTA call sites (seven views), "
            f"found {tta_block.count('model.encode(')}"
        )
    calibrated_tta = tta_block.replace(
        "model.encode(", "adabn_detection_model.encode("
    )
    source = source[:tta_start] + calibrated_tta + source[tta_end:]

    load_old = """    model, window_size, downsample = load_model(weights_path, device)

    secondary_model = None
"""
    load_new = """    model, window_size, downsample = load_model(weights_path, device)

    adabn_weights_text = os.environ.get("BIOHUB_ADABN_DETECTION_WEIGHTS", "").strip()
    if not adabn_weights_text:
        raise RuntimeError("BIOHUB_ADABN_DETECTION_WEIGHTS is required")
    adabn_detection_model, adabn_window_size, adabn_downsample = load_model(
        Path(adabn_weights_text), device,
    )
    if adabn_window_size != window_size or adabn_downsample != downsample:
        raise RuntimeError(
            "AdaBN detector grid differs from the P3 association grid: "
            f"primary={(window_size, downsample)}, "
            f"calibrated={(adabn_window_size, adabn_downsample)}"
        )

    secondary_model = None
"""
    if source.count(load_old) != 1:
        raise RuntimeError("AdaBN patch expected one final P3 model-load block")
    source = source.replace(load_old, load_new, 1)

    call_old = """                secondary_mix_temperature=secondary_mix_temperature,
                secondary_low_margin_max=secondary_low_margin_max,
            )"""
    call_new = """                secondary_mix_temperature=secondary_mix_temperature,
                secondary_low_margin_max=secondary_low_margin_max,
                adabn_detection_model=adabn_detection_model,
            )"""
    if source.count(call_old) != 1:
        raise RuntimeError("AdaBN patch expected one final P3 predict_video call")
    source = source.replace(call_old, call_new, 1)

    # Structural fail-closed assertions on the assembled consumer, not on a partial mirror.
    if source.count("adabn_detection_model.predict_edges"):
        raise RuntimeError("AdaBN model reached association prediction")
    if source.count("adabn_detection_model._index_features"):
        raise RuntimeError("AdaBN feature maps reached association indexing")
    if source.count("model._index_features(\n                unet_out") != 2:
        raise RuntimeError("P3 association no longer indexes exactly two original unet_out frames")
    compile(source, str(path), "exec")
    path.write_text(source, encoding="utf-8")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def install_adabn_detection_only(
    repo_dir: Path,
    weights_relative: str,
    input_root: Path = Path("/kaggle/input"),
    working_root: Path = Path("/kaggle/working"),
) -> dict:
    """Resolve the exact completed control, audit it, and patch the deployed predictor."""
    repo_dir, input_root, working_root = map(Path, (repo_dir, input_root, working_root))
    primary_path = repo_dir / weights_relative
    if sha256_file(primary_path) != PRIMARY_SHA256:
        raise RuntimeError("P3 primary association checkpoint hash mismatch")

    candidates = [p for p in input_root.rglob("edge_predictor_best.pth")
                  if sha256_file(p) == CALIBRATED_SHA256]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one exact AdaBN checkpoint, got {candidates}")
    source_dir = candidates[0].parent
    files = {
        "config.json": CONFIG_SHA256,
        "summary.json": SUMMARY_SHA256,
        "training_audit.json": AUDIT_SHA256,
    }
    for name, expected in files.items():
        path = source_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"AdaBN control sidecar hash mismatch: {path}")

    import torch
    primary = torch.load(primary_path, map_location="cpu", weights_only=True)
    calibrated = torch.load(candidates[0], map_location="cpu", weights_only=True)
    changed = validate_adabn_state_contract(primary, calibrated)
    audit = json.loads((source_dir / "training_audit.json").read_text(encoding="utf-8"))
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    protocol = validate_adabn_artifact_contract(audit, summary)

    target = working_root / "h1r_adabn_detection_only"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("edge_predictor_best.pth", "config.json"):
        shutil.copy2(source_dir / name, target / name)
    os.environ["BIOHUB_ADABN_DETECTION_WEIGHTS"] = str(target / "edge_predictor_best.pth")
    predictor = repo_dir / "scripts" / "predict_unet_transformer.py"
    predictor_sha256 = patch_detection_only_predictor(predictor)

    record = {
        "mode": CALIBRATION_MODE,
        "primary_sha256": PRIMARY_SHA256,
        "calibrated_sha256": CALIBRATED_SHA256,
        "changed_tensors": changed,
        "predictor_sha256": predictor_sha256,
        **protocol,
    }
    (working_root / "adabn_detection_only_contract.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("H1R_ADABN_DETECTION_ONLY", json.dumps(record, sort_keys=True))
    return record


# Importing for unit tests is inert; the deployment notebook defines these globals.
if "REPO_DIR" in globals() and "WEIGHTS_RELATIVE" in globals():
    install_adabn_detection_only(REPO_DIR, WEIGHTS_RELATIVE)
