import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EDIT = ROOT / "scripts" / "kaggle_edits" / "p12_forward_acceleration_lookahead.py"
SPEC = ROOT / "scripts" / "kaggle_specs" / "p12_forward_acceleration_lookahead.json"
BASE = ROOT / "notebooks" / "kaggle_p9_coupled_division" / "biohub-p9-coupled-division.ipynb"
BUILT = (
    ROOT
    / "notebooks"
    / "kaggle_p12_forward_acceleration"
    / "biohub-p12-forward-acceleration.ipynb"
)


@pytest.fixture
def p12(monkeypatch):
    spec = importlib.util.spec_from_file_location("p12_forward_acceleration", EDIT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUTPUT_MOTION_RELINK", True, raising=False)
    monkeypatch.setattr(module, "MOTION_RELINK_MAX_FRAME_NODES", 2600, raising=False)
    monkeypatch.setattr(module, "MOTION_RELINK_TIGHT_UM", 6.0, raising=False)
    monkeypatch.setattr(module, "MOTION_RELINK_RELAXED_UM", 10.0, raising=False)
    monkeypatch.setattr(module, "MOTION_RELINK_VELOCITY_WEIGHT", 0.5, raising=False)
    monkeypatch.setattr(module, "MOTION_RELINK_LEARNED_BONUS", 1.0, raising=False)
    monkeypatch.setattr(
        module,
        "_position_um",
        lambda node: np.asarray([node["z"], node["y"], node["x"]], dtype=float),
        raising=False,
    )
    return module


def node(t, z=0.0, y=0.0, x=0.0):
    return {"t": t, "z": z, "y": y, "x": x}


def blank_stats():
    return {
        "motion_relink_skipped_large_frame": 0,
        "motion_relink_tight_edges": 0,
        "motion_relink_relaxed_edges": 0,
        "motion_relink_frames": 0,
    }


def test_semantic_straight_continuation_receives_full_bonus(p12):
    positions = {
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([0.0, 1.0, 0.0]),
        3: np.array([0.0, 2.0, 0.0]),
        4: np.array([0.0, 1.0, 3.0]),
    }
    residual, count = p12.p12_forward_acceleration_lookahead(
        1, 2, {1: 0, 2: 1, 3: 2, 4: 2}, {0: [1], 1: [2], 2: [3, 4]}, positions, 10.0
    )
    assert count == 2
    assert residual == pytest.approx(0.0)
    assert p12.p12_forward_acceleration_bonus(residual, 4.0, 0.20) == pytest.approx(0.20)


def test_next_step_gate_is_inclusive_and_excludes_outside(p12):
    positions = {
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([0.0, 1.0, 0.0]),
        3: np.array([0.0, 11.0, 0.0]),
        4: np.array([0.0, 11.001, 0.0]),
    }
    residual, count = p12.p12_forward_acceleration_lookahead(
        1, 2, {1: 0, 2: 1, 3: 2, 4: 2}, {2: [3, 4]}, positions, 10.0
    )
    assert count == 1
    assert residual == pytest.approx(9.0)


@pytest.mark.parametrize(
    "residual, expected",
    [(None, 0.0), (0.0, 0.20), (2.0, 0.10), (4.0, 0.0), (8.0, 0.0)],
)
def test_bonus_is_bounded_linear_rule(p12, residual, expected):
    assert p12.p12_forward_acceleration_bonus(residual, 4.0, 0.20) == pytest.approx(expected)


def test_lookahead_can_break_equal_cost_assignment_toward_straight_motion(p12, monkeypatch):
    # Candidate 2 continues sharply; candidate 3 continues exactly straight to node 4.
    # Their P9 costs are equal, so the disabled control deterministically takes lower id 2.
    nodes = {
        1: node(0),
        2: node(1, x=1.0),
        3: node(1, y=1.0),
        4: node(2, y=2.0),
    }
    monkeypatch.setattr(p12, "P12_USE_FORWARD_ACCELERATION_LOOKAHEAD", False)
    control = p12.motion_relink_edges(nodes, blank_stats())
    assert (control[0]["source_id"], control[0]["target_id"]) == (1, 2)

    monkeypatch.setattr(p12, "P12_USE_FORWARD_ACCELERATION_LOOKAHEAD", True)
    stats = blank_stats()
    treatment = p12.motion_relink_edges(nodes, stats)
    assert (treatment[0]["source_id"], treatment[0]["target_id"]) == (1, 3)
    assert stats["forward_lookahead_evaluated_edges"] > 0
    assert stats["forward_lookahead_supported_edges"] > 0
    assert stats["forward_lookahead_bonus_edges"] > 0
    assert stats["forward_lookahead_bonus_milli_sum"] > 0


def test_spec_is_atomic_p9_derivative_with_exact_rule():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["base_sha256"] == "6b93ef47b7f1b0fc623a9bd058eb11ffaec5c047ee6ae8cb48d6b84a3fc2ccc9"
    assert spec["datasets"] == json.loads(
        (ROOT / "scripts" / "kaggle_specs" / "p9_coupled_division.json").read_text(encoding="utf-8")
    )["datasets"]
    assert len(spec["edits"]) == 2
    assert spec["edits"][0]["vars"] == {
        "BIOHUB_USE_FORWARD_ACCELERATION_LOOKAHEAD": "1",
        "BIOHUB_FORWARD_LOOKAHEAD_NEXT_STEP_UM": "10.0",
        "BIOHUB_FORWARD_LOOKAHEAD_MAX_ACCEL_UM": "4.0",
        "BIOHUB_FORWARD_LOOKAHEAD_MAX_BONUS": "0.20",
    }
    assert spec["edits"][1] == {
        "kind": "insert_before",
        "cell_match": "def close_single_frame_gaps(",
        "anchor": "def close_single_frame_gaps(",
        "code_file": "scripts/kaggle_edits/p12_forward_acceleration_lookahead.py",
        "expect": 1,
    }
    assert spec["provenance"]["public_source"]["isolation_warning"]


def test_built_notebook_changes_only_config_and_motion_relink_cell():
    if not BUILT.exists():
        pytest.skip("run kaggle_factory build first")
    base = json.loads(BASE.read_text(encoding="utf-8"))
    built = json.loads(BUILT.read_text(encoding="utf-8"))
    assert len(base["cells"]) == len(built["cells"])
    changed = [
        index
        for index, (base_cell, built_cell) in enumerate(zip(base["cells"], built["cells"], strict=True))
        if base_cell != built_cell
    ]
    assert changed == [2, 6]

    normalized = copy.deepcopy(built)
    normalized["cells"][2] = copy.deepcopy(base["cells"][2])
    normalized["cells"][6] = copy.deepcopy(base["cells"][6])
    assert normalized == base


def test_built_surface_keeps_p9_science_and_has_one_p12_override():
    if not BUILT.exists():
        pytest.skip("run kaggle_factory build first")
    nb = json.loads(BUILT.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    assert source.count("def motion_relink_edges(") == 2
    assert source.count("def p12_forward_acceleration_lookahead(") == 1
    assert source.count("P12_FORWARD_LOOKAHEAD_MAX_BONUS =") == 1
    assert source.count("def add_safe_divisions_postlink(") == 2
    assert source.count("COUPLED_DIV_DIVERGE_UM =") == 1
    assert 'os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.20"' in source
    assert 'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = \'8.0\'' in source
    assert 'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = \'11.0\'' in source
    assert 'os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = \'10.0\'' in source
    assert 'os.environ["BIOHUB_DEEPCENTER_EXPECTED_EPOCH"] = "500"' in source
