"""Software contract for scripts/kaggle_specs/p24_deepcenter_best_veto.json (LEVER-0024).

P24 is the P10 checkpoint selector PLUS the DeepCenter safe-division veto switched on - the
0.927 public lineage's DeepCenter bundle, env-only on the P9 champion. These tests pin the
spec to exactly that and nothing else; whether it helps the leaderboard is EXP-0025's job.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "kaggle_specs" / "p24_deepcenter_best_veto.json"
P10 = ROOT / "scripts" / "kaggle_specs" / "p10_deepcenter_best.json"
BASE = ROOT / "notebooks" / "kaggle_p9_coupled_division" / "biohub-p9-coupled-division.ipynb"
BUILT = ROOT / "notebooks" / "kaggle_p24_deepcenter_best_veto" / "biohub-p24-deepcenter-best-veto.ipynb"

BEST_PATH = "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt"
VETO_OFF_LINE = "os.environ[\"BIOHUB_DEEPCENTER_SAFE_DIV_VETO\"] = '0'"
VETO_READ_LINE = 'DEEPCENTER_SAFE_DIV_VETO = os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_VETO", "1") != "0"'


def _source(cell):
    return "".join(cell.get("source", []))


def test_spec_is_p10_plus_the_safe_division_veto_and_nothing_else():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    p10 = json.loads(P10.read_text(encoding="utf-8"))
    assert spec["base_notebook"] == p10["base_notebook"]
    assert spec["base_sha256"] == p10["base_sha256"]
    assert spec["datasets"] == p10["datasets"]
    assert spec["expects_submission"] is True
    assert len(spec["edits"]) == 1
    expected_vars = dict(p10["edits"][0]["vars"])
    expected_vars["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"
    assert spec["edits"][0] == {
        "kind": "env",
        "cell_match": "BIOHUB_PRESET",
        "vars": expected_vars,
        "expect": 1,
    }
    # The checkpoint artifact pin is the verified one from P10, byte for byte.
    assert spec["provenance"]["artifact"]["best_checkpoint"] == p10["provenance"]["artifact"]["best_checkpoint"]
    assert spec["provenance"]["artifact"]["last_checkpoint"] == p10["provenance"]["artifact"]["last_checkpoint"]
    # A different out_dir and slug than P10 - the sibling-overwrite trap (AGENTS.md section 5).
    assert spec["out_dir"] != p10["out_dir"]
    assert spec["slug"] != p10["slug"]


def test_p9_base_switches_the_veto_off_exactly_once_so_the_override_is_meaningful():
    base = json.loads(BASE.read_text(encoding="utf-8"))
    source = "\n".join(_source(cell) for cell in base["cells"])
    assert source.count(VETO_OFF_LINE) == 1
    # The notebook reads the flag in a LATER cell, so an override appended to the preset cell wins.
    assert VETO_READ_LINE in source
    preset_cell = next(i for i, c in enumerate(base["cells"]) if VETO_OFF_LINE in _source(c))
    read_cell = next(i for i, c in enumerate(base["cells"]) if VETO_READ_LINE in _source(c))
    assert preset_cell < read_cell


def test_built_notebook_diff_is_only_the_three_env_overrides():
    if not BUILT.exists():
        pytest.skip("run kaggle_factory build first")
    base = json.loads(BASE.read_text(encoding="utf-8"))
    built = json.loads(BUILT.read_text(encoding="utf-8"))
    assert len(base["cells"]) == len(built["cells"])
    changed = [
        i for i, (b, c) in enumerate(zip(base["cells"], built["cells"], strict=True)) if b != c
    ]
    assert changed == [2]
    expected_suffix = (
        "\n\n# --- factory env overrides ---\n"
        f"os.environ[\"BIOHUB_DEEPCENTER_CHECKPOINT\"] = '{BEST_PATH}'\n"
        "os.environ[\"BIOHUB_DEEPCENTER_EXPECTED_EPOCH\"] = '2'\n"
        "os.environ[\"BIOHUB_DEEPCENTER_SAFE_DIV_VETO\"] = '1'\n"
    )
    base_text = _source(base["cells"][2]).rstrip("\n")
    assert _source(built["cells"][2]) == base_text + expected_suffix
