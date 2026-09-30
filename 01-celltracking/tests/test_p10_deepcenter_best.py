import copy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "kaggle_specs" / "p10_deepcenter_best.json"
BASE = ROOT / "notebooks" / "kaggle_p9_coupled_division" / "biohub-p9-coupled-division.ipynb"
BUILT = ROOT / "notebooks" / "kaggle_p10_deepcenter_best" / "biohub-p10-deepcenter-best.ipynb"

BEST_PATH = "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt"
LAST_PATH = "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/checkpoint_last.pt"


def _source(cell):
    return "".join(cell.get("source", []))


def test_spec_is_one_scientific_selector_and_pins_verified_artifact():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["base_sha256"] == "6b93ef47b7f1b0fc623a9bd058eb11ffaec5c047ee6ae8cb48d6b84a3fc2ccc9"
    assert len(spec["edits"]) == 1
    assert spec["edits"][0] == {
        "kind": "env",
        "cell_match": "BIOHUB_PRESET",
        "vars": {
            "BIOHUB_DEEPCENTER_CHECKPOINT": BEST_PATH,
            "BIOHUB_DEEPCENTER_EXPECTED_EPOCH": "2",
        },
        "expect": 1,
    }

    artifact = spec["provenance"]["artifact"]
    assert artifact["dataset"] == "pilkwang/biohub-deepcenter-unet3d-center-prior-v1"
    assert artifact["artifact_manifest_sha256"] == "3bfe97304e9bbc3b3481a095392a1e83937315325bac8b987eb527d2951b96f3"
    assert artifact["snapshot_manifest_sha256"] == "e7d791b1500610602f33a1e0576f04804a734762df306ff008226b9f2c608d6e"
    assert artifact["best_checkpoint"] == {
        "path": "weights/full_frame_center/best.pt",
        "bytes": 37876911,
        "sha256": "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0",
        "epoch": 2,
        "best_score": -0.04500306242456039,
    }
    assert artifact["last_checkpoint"] == {
        "path": "weights/full_frame_center/checkpoint_last.pt",
        "bytes": 37918921,
        "sha256": "8164d1ffa07f87e0506027a0392edeab7939a32bd5e3f756377c0d72885cf127",
        "epoch": 500,
        "best_score": -0.04500306242456039,
    }


def test_p9_base_has_the_last_epoch500_contract_once():
    base = json.loads(BASE.read_text(encoding="utf-8"))
    source = "\n".join(_source(cell) for cell in base["cells"])
    assert source.count(f'os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = "{LAST_PATH}"') == 1
    assert source.count('os.environ["BIOHUB_DEEPCENTER_EXPECTED_EPOCH"] = "500"') == 1
    assert BEST_PATH not in source


def test_built_notebook_diff_is_only_the_checkpoint_selector_override():
    if not BUILT.exists():
        pytest.skip("run kaggle_factory build first")

    base = json.loads(BASE.read_text(encoding="utf-8"))
    built = json.loads(BUILT.read_text(encoding="utf-8"))
    changed = [
        i
        for i, (base_cell, built_cell) in enumerate(zip(base["cells"], built["cells"], strict=True))
        if base_cell != built_cell
    ]
    assert changed == [2]
    assert len(base["cells"]) == len(built["cells"])

    expected_suffix = (
        "\n\n# --- factory env overrides ---\n"
        f"os.environ[\"BIOHUB_DEEPCENTER_CHECKPOINT\"] = '{BEST_PATH}'\n"
        "os.environ[\"BIOHUB_DEEPCENTER_EXPECTED_EPOCH\"] = '2'\n"
    )
    assert _source(built["cells"][2]) == _source(base["cells"][2]).rstrip("\n") + expected_suffix

    normalized = copy.deepcopy(built)
    normalized["cells"][2] = copy.deepcopy(base["cells"][2])
    assert normalized == base


def test_built_selector_is_last_write_and_keeps_p9_science_surface():
    if not BUILT.exists():
        pytest.skip("run kaggle_factory build first")

    nb = json.loads(BUILT.read_text(encoding="utf-8"))
    source = "\n".join(_source(cell) for cell in nb["cells"])
    config_source = _source(nb["cells"][2])
    assert config_source.rfind(BEST_PATH) > config_source.rfind(LAST_PATH)
    assert config_source.rfind('BIOHUB_DEEPCENTER_EXPECTED_EPOCH\"] = \'2\'') > config_source.rfind(
        'BIOHUB_DEEPCENTER_EXPECTED_EPOCH\"] = "500"'
    )
    assert 'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = \'8.0\'' in source
    assert 'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = \'11.0\'' in source
    assert 'os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = \'10.0\'' in source
    assert 'os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.20"' in source
    assert source.count("def add_safe_divisions_postlink(") == 2
    assert source.count("COUPLED_DIV_DIVERGE_UM =") == 1
