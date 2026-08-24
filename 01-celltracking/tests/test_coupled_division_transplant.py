import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "scripts" / "kaggle_edits" / "coupled_division_transplant.py"
SPEC = ROOT / "scripts" / "kaggle_specs" / "p9_coupled_division.json"
BUILT = ROOT / "notebooks" / "kaggle_p9_coupled_division" / "biohub-p9-coupled-division.ipynb"


@pytest.fixture
def coupled(monkeypatch):
    spec = importlib.util.spec_from_file_location("coupled_division_transplant", PATCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "VOXEL_SCALE_UM", (1.0, 1.0, 1.0), raising=False)
    monkeypatch.setattr(module, "OUTPUT_SAFE_DIVISIONS", True, raising=False)
    monkeypatch.setattr(module, "SAFE_DIV_MAX_UM", 8.0, raising=False)
    monkeypatch.setattr(module, "SAFE_DIV_SISTER_MAX_UM", 11.0, raising=False)
    monkeypatch.setattr(module, "SAFE_DIV_EXISTING_CHILD_MAX_UM", 10.0, raising=False)
    monkeypatch.setattr(module, "SAFE_DIV_FRAME_FRAC_CAP", 1.0, raising=False)
    monkeypatch.setattr(module, "SAFE_DIV_GLOBAL_FRAC_CAP", 1.0, raising=False)
    monkeypatch.setattr(module, "DEEPCENTER_SAFE_DIV_VETO", False, raising=False)
    monkeypatch.setattr(module, "DEEPCENTER_SAFE_DIV_THRESHOLD", 0.0, raising=False)
    monkeypatch.setattr(
        module,
        "edge_distance_um",
        lambda a, b: float(((module._coupled_point_um(a) - module._coupled_point_um(b)) ** 2).sum() ** 0.5),
        raising=False,
    )
    monkeypatch.setattr(module, "node_point", lambda node: module._coupled_point_um(node), raising=False)
    return module


def node(t, y, x=0.0, z=0.0):
    return {"t": t, "z": z, "y": y, "x": x}


def edge(source, target):
    return {"source_id": source, "target_id": target, "edge_prob": 1.0}


def valid_case():
    # 1 -> 2 establishes C1.  2 -> 3 is the existing daughter; 4 is its
    # orphan sister.  Both daughters continue to frame t+2 and separate by 3 um.
    nodes = {
        1: node(0, 0),
        2: node(1, 0),
        3: node(2, 1),
        4: node(2, -4),
        5: node(3, 2),
        6: node(3, -6),
    }
    edges = [edge(1, 2), edge(2, 3), edge(3, 5), edge(4, 6)]
    return nodes, edges


def added_pairs(result):
    return {
        (int(e["source_id"]), int(e["target_id"]))
        for e in result
        if e.get("safe_division") == 1
    }


def test_valid_coupled_division_is_added(coupled):
    nodes, edges = valid_case()
    stats = {}
    result = coupled.add_safe_divisions_postlink(nodes, edges, stats)
    assert added_pairs(result) == {(2, 4)}
    assert stats["safe_divisions_added"] == 1


def test_c1_rejects_track_start_parent(coupled):
    nodes, edges = valid_case()
    edges = [e for e in edges if (e["source_id"], e["target_id"]) != (1, 2)]
    result = coupled.add_safe_divisions_postlink(nodes, edges, {})
    assert added_pairs(result) == set()


def test_c2_keeps_only_existing_childs_nearest_orphan(coupled):
    nodes, edges = valid_case()
    nodes.update({7: node(2, 2), 8: node(3, 7)})
    edges.append(edge(7, 8))
    # Node 7 is 1 um from linked child 3, so public C2 must exclude node 4.
    # Its t+2 separation grows 5 um, allowing it through C3.
    result = coupled.add_safe_divisions_postlink(nodes, edges, {})
    assert added_pairs(result) == {(2, 7)}


@pytest.mark.parametrize("next_y,accepted", [(-5.25, True), (-5.249, False)])
def test_c3_divergence_boundary_is_inclusive(coupled, next_y, accepted):
    nodes, edges = valid_case()
    # Birth separation is 5.0.  Successors at y=2 and y=next_y have the
    # requested separation growth; exact 2.25 must pass.
    nodes[6] = node(3, next_y)
    result = coupled.add_safe_divisions_postlink(nodes, edges, {})
    assert bool(added_pairs(result)) is accepted


def test_c3_requires_both_t2_successors(coupled):
    nodes, edges = valid_case()
    edges = [e for e in edges if (e["source_id"], e["target_id"]) != (4, 6)]
    result = coupled.add_safe_divisions_postlink(nodes, edges, {})
    assert added_pairs(result) == set()


@pytest.mark.parametrize("gate", ["existing_child", "parent", "sister"])
def test_8_11_10_physical_gates_reject_outside_points(coupled, gate):
    nodes, edges = valid_case()
    if gate == "existing_child":
        nodes[3] = node(2, 10.001)  # p-c1 > 10
    elif gate == "parent":
        nodes[4] = node(2, -8.001)  # p-q > 8 while c1-q < 11
    else:
        nodes[3] = node(2, 10.0)
        nodes[4] = node(2, -1.001)  # c1-q > 11 while p-q < 8
    result = coupled.add_safe_divisions_postlink(nodes, edges, {})
    assert added_pairs(result) == set()


@pytest.mark.parametrize("gate", ["existing_child", "parent", "sister"])
def test_8_11_10_physical_gate_boundaries_are_inclusive(coupled, gate):
    nodes, edges = valid_case()
    if gate == "existing_child":
        nodes[3] = node(2, 10.0)
        nodes[4] = node(2, 4.0)
        nodes[5] = node(3, 12.25)
        nodes[6] = node(3, 3.0)  # separation grows from 6 to 9.25
    elif gate == "parent":
        nodes[3] = node(2, 1.0)
        nodes[4] = node(2, -8.0)
        nodes[5] = node(3, 2.0)
        nodes[6] = node(3, -9.25)  # separation grows from 9 to 11.25
    else:
        nodes[3] = node(2, 10.0)
        nodes[4] = node(2, -1.0)
        nodes[5] = node(3, 11.0)
        nodes[6] = node(3, -2.25)  # separation grows from 11 to 13.25
    result = coupled.add_safe_divisions_postlink(nodes, edges, {})
    assert added_pairs(result) == {(2, 4)}


def test_target_dedupe_and_source_degree_contract(coupled):
    nodes, edges = valid_case()
    # A second mid-track parent and existing daughter can nominate the same orphan.
    nodes.update({11: node(0, 0.5), 12: node(1, 0.5), 13: node(2, 1.5), 15: node(3, 2.5)})
    edges.extend([edge(11, 12), edge(12, 13), edge(13, 15)])
    result = coupled.add_safe_divisions_postlink(nodes, edges, {})
    coupled._coupled_assert_degrees(result, "test")
    targets = [int(e["target_id"]) for e in result]
    assert len(targets) == len(set(targets))
    out = {}
    for e in result:
        out[int(e["source_id"])] = out.get(int(e["source_id"]), 0) + 1
    assert max(out.values()) <= 2


def test_spec_pins_public_provenance_and_atomic_constants():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["base_sha256"] == "70b636b64455f5fd24caa7ffbed661c43003f64764ba77f8768f8f53f9380ca8"
    assert spec["provenance"]["public_function_sha256"] == "b0d50a39e07624bc5a29207215fb4ec11908b49a8b60662e3527a513b1b35a5d"
    assert {x["kernel_id"] for x in spec["provenance"]["public_sources"]} == {130667075, 131694028}
    env = spec["edits"][0]["vars"]
    assert env == {
        "BIOHUB_SAFE_DIV_MAX_UM": "8.0",
        "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "11.0",
        "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "10.0",
        "BIOHUB_SAFE_DIV_DIVERGE_UM": "2.25",
    }


def test_built_notebook_contains_one_override_and_p3_harmonic_surface():
    if not BUILT.exists():
        pytest.skip("run kaggle_factory build first")
    nb = json.loads(BUILT.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    assert source.count("def add_safe_divisions_postlink(") == 2
    assert source.count("COUPLED_DIV_DIVERGE_UM =") == 1
    assert 'os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.20"' in source
    assert source.count("harmonic_prob = 1.0 / (") == 1
    assert source.count("assert_degree_invariants(edges, \"export\")") == 1
