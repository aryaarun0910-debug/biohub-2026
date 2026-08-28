from pathlib import Path

from scripts.win_bet.p28_full_chain_replay import (
    DEFAULT_NOTEBOOK,
    REQUIRED,
    load_p28_module,
    submission_frame,
)


def test_extracts_exact_p28_chain_and_champion_constants(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    module = load_p28_module(DEFAULT_NOTEBOOK, tmp_path, checkpoint)
    assert REQUIRED <= module.__dict__.keys()
    assert module.TEST_DIR == tmp_path
    assert module.MOTION_RELINK_LEARNED_BONUS == 1.0
    assert module.SAFE_DIV_MAX_UM == 8.0
    assert module.SAFE_DIV_SISTER_MAX_UM == 11.0
    assert module.SAFE_DIV_EXISTING_CHILD_MAX_UM == 10.0
    assert module.DEEPCENTER_EXPECTED_EPOCH == 2
    assert module.DEEPCENTER_SAFE_DIV_VETO is True
    assert len(module._source_sha256) == 64


def test_submission_frame_preserves_node_identity_and_edges():
    nodes = {
        4: {"t": 0, "z": 1.0, "y": 2.0, "x": 3.0},
        9: {"t": 1, "z": 2.0, "y": 3.0, "x": 4.0},
    }
    frame = submission_frame("crop", nodes, [{"source_id": 4, "target_id": 9}])
    assert frame.height == 3
    assert frame.filter(frame["row_type"] == "node")["node_id"].to_list() == [4, 9]
    edge = frame.filter(frame["row_type"] == "edge").row(0, named=True)
    assert (edge["source_id"], edge["target_id"]) == (4, 9)
