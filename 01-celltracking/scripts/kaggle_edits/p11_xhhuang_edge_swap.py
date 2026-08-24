# P11: fail-closed primary-checkpoint swap to xhhuang's v6 edge predictor.
#
# This block runs after the P9 prediction command has been assembled and before
# any inference worker starts.  The attached artifact is addressed by one exact
# Kaggle mount path: discovery by filename/size is intentionally forbidden.
import hashlib as _p11_hashlib
import json as _p11_json
from pathlib import Path as _P11_Path


_P11_DATASET_ROOT = _P11_Path("/kaggle/input/biohub-edge-predictor-v6-weights")
_P11_WEIGHT_PATH = _P11_DATASET_ROOT / "split_0_edge_predictor_best.pth"
_P11_CONFIG_PATH = _P11_DATASET_ROOT / "split_0_config.json"
_P11_EXPECTED_WEIGHT_BYTES = 8_357_783
_P11_EXPECTED_WEIGHT_SHA256 = (
    "19cfbbeb082f54845564b77d48528998d43cfe1f8b1023101425427c834aa68f"
)
_P11_EXPECTED_STATE_SCHEMA_SHA256 = (
    "5011bba0806057be37c5090fb7b7a1c081a145d64e626f02a25cf6c715af0868"
)
_P11_EXPECTED_CONFIG = {
    "unet_out_channels": 32,
    "unet_layers": [32, 64, 128],
    "downsample": [1, 4, 4],
    "window_size": 2,
    "pool_kernel_um": 5.0,
}


def _p11_sha256_file(path: _P11_Path) -> str:
    digest = _p11_hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _p11_state_schema(state: dict[str, _torch.Tensor]) -> list[dict[str, object]]:
    if not isinstance(state, dict):
        raise RuntimeError(
            "P11 checkpoint contract failed: expected a bare state_dict, "
            f"got {type(state).__name__}"
        )
    bad = [key for key, value in state.items() if not isinstance(key, str) or not _torch.is_tensor(value)]
    if bad:
        raise RuntimeError(
            "P11 checkpoint contract failed: non-tensor or non-string state entries: "
            f"{bad[:5]}"
        )
    return [
        {"key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(state.items())
    ]


for _p11_required in (_P11_WEIGHT_PATH, _P11_CONFIG_PATH):
    if not _p11_required.is_file():
        raise FileNotFoundError(
            "P11 checkpoint contract failed: exact attached path is missing: "
            f"{_p11_required}"
        )

_p11_weight_bytes = _P11_WEIGHT_PATH.stat().st_size
if _p11_weight_bytes != _P11_EXPECTED_WEIGHT_BYTES:
    raise RuntimeError(
        "P11 checkpoint byte-size mismatch: "
        f"expected {_P11_EXPECTED_WEIGHT_BYTES}, got {_p11_weight_bytes}"
    )

_p11_weight_sha256 = _p11_sha256_file(_P11_WEIGHT_PATH)
if _p11_weight_sha256 != _P11_EXPECTED_WEIGHT_SHA256:
    raise RuntimeError(
        "P11 checkpoint SHA256 mismatch: "
        f"expected {_P11_EXPECTED_WEIGHT_SHA256}, got {_p11_weight_sha256}"
    )

try:
    _p11_external_config = _p11_json.loads(_P11_CONFIG_PATH.read_text(encoding="utf-8"))
except Exception as exc:
    raise RuntimeError(f"P11 external config is unreadable: {exc}") from exc
if _p11_external_config != _P11_EXPECTED_CONFIG:
    raise RuntimeError(
        "P11 external config mismatch: "
        f"expected {_P11_EXPECTED_CONFIG}, got {_p11_external_config}"
    )

_p11_reference_weight = REPO_DIR / WEIGHTS_RELATIVE
_p11_reference_config_path = _p11_reference_weight.parent / "config.json"
for _p11_required in (_p11_reference_weight, _p11_reference_config_path):
    if not _p11_required.is_file():
        raise FileNotFoundError(
            "P11 reference-model contract failed: exact P9 artifact path is missing: "
            f"{_p11_required}"
        )

try:
    _p11_reference_config = _p11_json.loads(
        _p11_reference_config_path.read_text(encoding="utf-8")
    )
except Exception as exc:
    raise RuntimeError(f"P11 reference config is unreadable: {exc}") from exc
_p11_model_config_keys = (
    "unet_out_channels", "unet_layers", "downsample", "window_size", "pool_kernel_um"
)
_p11_reference_model_config = {
    key: _p11_reference_config.get(key) for key in _p11_model_config_keys
}
if _p11_reference_model_config != _P11_EXPECTED_CONFIG:
    raise RuntimeError(
        "P11 P9/reference config mismatch: "
        f"expected {_P11_EXPECTED_CONFIG}, got {_p11_reference_model_config}"
    )

_p11_external_state = _torch.load(
    _P11_WEIGHT_PATH, map_location="cpu", weights_only=True
)
_p11_reference_state = _torch.load(
    _p11_reference_weight, map_location="cpu", weights_only=True
)
_p11_external_schema = _p11_state_schema(_p11_external_state)
_p11_reference_schema = _p11_state_schema(_p11_reference_state)
if len(_p11_external_schema) != 136 or len(_p11_reference_schema) != 136:
    raise RuntimeError(
        "P11 state-count mismatch: expected external/reference 136/136, got "
        f"{len(_p11_external_schema)}/{len(_p11_reference_schema)}"
    )
if _p11_external_schema != _p11_reference_schema:
    _p11_external_by_key = {row["key"]: row for row in _p11_external_schema}
    _p11_reference_by_key = {row["key"]: row for row in _p11_reference_schema}
    _p11_bad_keys = sorted(
        key
        for key in set(_p11_external_by_key) | set(_p11_reference_by_key)
        if _p11_external_by_key.get(key) != _p11_reference_by_key.get(key)
    )
    raise RuntimeError(
        "P11 state schema is not 136/136 shape-compatible with P9; "
        f"first mismatches: {_p11_bad_keys[:8]}"
    )
_p11_schema_bytes = _p11_json.dumps(
    _p11_external_schema, sort_keys=True, separators=(",", ":")
).encode("utf-8")
_p11_schema_sha256 = _p11_hashlib.sha256(_p11_schema_bytes).hexdigest()
if _p11_schema_sha256 != _P11_EXPECTED_STATE_SCHEMA_SHA256:
    raise RuntimeError(
        "P11 state-schema SHA256 mismatch: "
        f"expected {_P11_EXPECTED_STATE_SCHEMA_SHA256}, got {_p11_schema_sha256}"
    )

# Prove that the command mutation changes exactly one field: the primary
# --weights value.  The independent secondary seed and every P9 setting remain untouched.
if predict_cmd.count("--weights") != 1:
    raise RuntimeError(
        f"P11 expected exactly one --weights selector, got {predict_cmd}"
    )
_p11_weights_index = predict_cmd.index("--weights") + 1
if _p11_weights_index >= len(predict_cmd):
    raise RuntimeError(f"P11 --weights selector has no value: {predict_cmd}")
if predict_cmd[_p11_weights_index] != WEIGHTS_RELATIVE:
    raise RuntimeError(
        "P11 primary weight precondition failed: expected "
        f"{WEIGHTS_RELATIVE!r}, got {predict_cmd[_p11_weights_index]!r}"
    )
_p11_predict_cmd_before = list(predict_cmd)
predict_cmd[_p11_weights_index] = str(_P11_WEIGHT_PATH)
_p11_changed_indices = [
    index
    for index, (before, after) in enumerate(zip(_p11_predict_cmd_before, predict_cmd, strict=True))
    if before != after
]
if _p11_changed_indices != [_p11_weights_index]:
    raise RuntimeError(
        "P11 violated the isolated-swap contract; changed command indices "
        f"{_p11_changed_indices}"
    )

del _p11_external_state, _p11_reference_state
print(
    "P11 PRIMARY EDGE CHECKPOINT VERIFIED AND REBOUND: "
    f"{_P11_WEIGHT_PATH} | sha256={_p11_weight_sha256} | "
    "state=136/136 exact key+shape+dtype compatibility | "
    f"config={_p11_external_config}",
    flush=True,
)
