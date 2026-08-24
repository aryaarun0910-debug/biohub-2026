import copy
import ast
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "kaggle_specs" / "p11_xhhuang_edge_swap.json"
EDIT = ROOT / "scripts" / "kaggle_edits" / "p11_xhhuang_edge_swap.py"
BASE = ROOT / "notebooks" / "kaggle_p9_coupled_division" / "biohub-p9-coupled-division.ipynb"
BUILT = (
    ROOT
    / "notebooks"
    / "kaggle_p11_xhhuang_edge_swap"
    / "biohub-p11-xhhuang-edge-swap.ipynb"
)

DATASET = "xhhuang/biohub-edge-predictor-v6-weights"
WEIGHT_SHA256 = "19cfbbeb082f54845564b77d48528998d43cfe1f8b1023101425427c834aa68f"
CONFIG_SHA256 = "e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a"
SCHEMA_SHA256 = "5011bba0806057be37c5090fb7b7a1c081a145d64e626f02a25cf6c715af0868"
EXPECTED_CONFIG = {
    "unet_out_channels": 32,
    "unet_layers": [32, 64, 128],
    "downsample": [1, 4, 4],
    "window_size": 2,
    "pool_kernel_um": 5.0,
}


def _source(cell):
    return "".join(cell.get("source", []))


def _discovery_helpers():
    tree = ast.parse(EDIT.read_text(encoding="utf-8"))
    selected = []
    wanted = {"_p11_sha256_file", "_p11_find_unique_hashed_file"}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            selected.append(node)
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(EDIT), "exec"), namespace)
    return namespace["_p11_find_unique_hashed_file"]


def test_spec_is_one_primary_checkpoint_selector_with_exact_artifact_contract():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["base_sha256"] == "6b93ef47b7f1b0fc623a9bd058eb11ffaec5c047ee6ae8cb48d6b84a3fc2ccc9"
    assert spec["datasets"].count(DATASET) == 1
    assert len(spec["edits"]) == 1
    assert spec["edits"][0] == {
        "kind": "insert_before",
        "cell_match": "available_gpu_count = _torch.cuda.device_count()",
        "anchor": "available_gpu_count = _torch.cuda.device_count()",
        "code_file": "scripts/kaggle_edits/p11_xhhuang_edge_swap.py",
        "expect": 1,
    }

    artifact = spec["provenance"]["artifact"]
    assert artifact["dataset"] == DATASET
    assert artifact["expected_filename"] == "split_0_edge_predictor_best.pth"
    assert "/kaggle/input" in artifact["mount_resolution"]
    assert artifact["bytes"] == 8_357_783
    assert artifact["sha256"] == WEIGHT_SHA256
    assert artifact["state_tensors"] == 136
    assert artifact["state_schema_sha256"] == SCHEMA_SHA256
    assert artifact["config_filenames"] == ["split_0_config.json", "config.json"]
    assert artifact["config_bytes"] == 165
    assert artifact["config_sha256"] == CONFIG_SHA256
    assert artifact["config"] == EXPECTED_CONFIG


def test_provenance_does_not_promote_owner_score_or_invent_normalization():
    provenance = json.loads(SPEC.read_text(encoding="utf-8"))["provenance"]
    normalization = provenance["training_normalization"].lower()
    owner_evidence = provenance["owner_leaderboard_evidence"].lower()
    assert "unknown" in normalization
    assert "no training code" in normalization
    assert "0.920" in owner_evidence
    assert "non-isolated" in owner_evidence
    assert "do not attribute" in owner_evidence


def test_runtime_edit_fails_closed_on_discovered_artifact_identity():
    source = EDIT.read_text(encoding="utf-8")
    compile(source, str(EDIT), "exec")
    assert '"/kaggle/input"' in source
    assert '"split_0_edge_predictor_best.pth"' in source
    assert '"split_0_config.json"' in source
    assert WEIGHT_SHA256 in source
    assert CONFIG_SHA256 in source
    assert SCHEMA_SHA256 in source
    assert "_P11_EXPECTED_WEIGHT_BYTES = 8_357_783" in source
    assert "len(_p11_external_schema) != 136" in source
    assert "_p11_external_schema != _p11_reference_schema" in source
    assert "_p11_external_config != _P11_EXPECTED_CONFIG" in source
    assert "_p11_reference_model_config != _P11_EXPECTED_CONFIG" in source
    assert 'predict_cmd.count("--weights") != 1' in source
    assert "_p11_changed_indices != [_p11_weights_index]" in source
    assert "root.rglob(filename)" in source
    assert "len(exact_matches) != 1" in source
    assert "BIOHUB_P11" not in source


@pytest.mark.parametrize(
    "relative",
    [
        Path("biohub-edge-predictor-v6-weights") / "split_0_edge_predictor_best.pth",
        Path("datasets") / "xhhuang" / "biohub-edge-predictor-v6-weights" / "split_0_edge_predictor_best.pth",
    ],
    ids=["new_mount", "legacy_mount"],
)
def test_discovery_accepts_legacy_and_new_mount_shapes(tmp_path, relative):
    payload = b"verified-xh-checkpoint"
    expected = tmp_path / relative
    expected.parent.mkdir(parents=True)
    expected.write_bytes(payload)
    resolved = _discovery_helpers()(
        tmp_path,
        ("split_0_edge_predictor_best.pth",),
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )
    assert resolved == expected


def test_discovery_fails_closed_when_artifact_is_absent(tmp_path):
    with pytest.raises(RuntimeError, match="got 0"):
        _discovery_helpers()(
            tmp_path,
            ("split_0_edge_predictor_best.pth",),
            hashlib.sha256(b"missing").hexdigest(),
            len(b"missing"),
        )


def test_discovery_fails_closed_on_duplicate_exact_hash_matches(tmp_path):
    payload = b"duplicate-xh-checkpoint"
    for mount in ("legacy", "new"):
        path = tmp_path / mount / "split_0_edge_predictor_best.pth"
        path.parent.mkdir()
        path.write_bytes(payload)
    with pytest.raises(RuntimeError, match="got 2"):
        _discovery_helpers()(
            tmp_path,
            ("split_0_edge_predictor_best.pth",),
            hashlib.sha256(payload).hexdigest(),
            len(payload),
        )


def test_config_prefers_colocated_exact_hash_then_supports_hashed_fallback(tmp_path):
    payload = b"verified-config"
    weight_parent = tmp_path / "opaque-mount" / "nested"
    weight_parent.mkdir(parents=True)
    colocated = weight_parent / "split_0_config.json"
    colocated.write_bytes(payload)
    find = _discovery_helpers()
    assert find(
        tmp_path,
        ("split_0_config.json", "config.json"),
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        preferred_parent=weight_parent,
    ) == colocated

    colocated.unlink()
    fallback = tmp_path / "another-mount" / "config.json"
    fallback.parent.mkdir()
    fallback.write_bytes(payload)
    assert find(
        tmp_path,
        ("split_0_config.json", "config.json"),
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        preferred_parent=weight_parent,
    ) == fallback


def test_edit_payload_hash_is_pinned_by_built_manifest():
    manifest_path = BUILT.parent / "build_manifest.json"
    if not manifest_path.exists():
        pytest.skip("run kaggle_factory build first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["edits"]) == 1
    assert manifest["edits"][0]["payload_sha256"] == hashlib.sha256(
        json.dumps(
            json.loads(SPEC.read_text(encoding="utf-8"))["edits"][0], sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def test_built_notebook_diff_is_only_the_pre_worker_checkpoint_swap():
    if not BUILT.exists():
        pytest.skip("run kaggle_factory build first")

    base = json.loads(BASE.read_text(encoding="utf-8"))
    built = json.loads(BUILT.read_text(encoding="utf-8"))
    assert len(base["cells"]) == len(built["cells"])
    changed = [
        index
        for index, (base_cell, built_cell) in enumerate(
            zip(base["cells"], built["cells"], strict=True)
        )
        if base_cell != built_cell
    ]
    assert changed == [5]

    payload = EDIT.read_text(encoding="utf-8")
    anchor = "available_gpu_count = _torch.cuda.device_count()"
    expected = _source(base["cells"][5]).replace(anchor, "\n" + payload + "\n" + anchor, 1)
    assert _source(built["cells"][5]) == expected
    assert _source(built["cells"][5]).index(payload) < _source(built["cells"][5]).index(anchor)

    normalized = copy.deepcopy(built)
    normalized["cells"][5] = copy.deepcopy(base["cells"][5])
    assert normalized == base


def test_built_surface_keeps_p9_secondary_harmonic_and_coupled_divisions():
    if not BUILT.exists():
        pytest.skip("run kaggle_factory build first")

    source = "\n".join(_source(cell) for cell in json.loads(BUILT.read_text(encoding="utf-8"))["cells"])
    assert source.count("P11 PRIMARY EDGE CHECKPOINT VERIFIED AND REBOUND") == 1
    assert source.count("def add_safe_divisions_postlink(") == 2
    assert source.count("COUPLED_DIV_DIVERGE_UM =") == 1
    assert 'os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.20"' in source
    assert "BIOHUB_SECONDARY_WEIGHTS" in source
    assert 'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = \'8.0\'' in source
    assert 'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = \'11.0\'' in source
    assert 'os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = \'10.0\'' in source
