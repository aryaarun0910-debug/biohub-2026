"""MANDATORY pre-push gate: every external API the audit calls, executed against real data.

This exists because v3 burned a GPU run on
    AttributeError: module 'tracksdata.io' has no attribute 'load_geff'
a call that takes seconds to verify locally against data/train and was never checked. The rule
is now explicit: no external loader/API may be pushed to a kernel until it has been resolved,
signature-inspected and executed here on a real crop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN = ROOT / "data" / "train"
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)

pytestmark = pytest.mark.skipif(not TRAIN.exists(), reason="data/train not present")


def _one_geff() -> Path:
    hits = sorted(TRAIN.glob("*.geff"))
    if not hits:
        pytest.skip("no GT geff available")
    return hits[0]


def test_geff_loader_call_used_by_the_audit_resolves_and_runs():
    """The audit must use the SAME call as our exact scorer, and it must actually work."""
    import tracksdata as td

    assert not hasattr(td.io, "load_geff"), (
        "tracksdata.io.load_geff exists now -- re-check which call the audit should use"
    )
    assert hasattr(td.graph, "IndexedRXGraph"), "IndexedRXGraph missing from this build"
    assert hasattr(td.graph.IndexedRXGraph, "from_geff"), "from_geff missing"

    res = td.graph.IndexedRXGraph.from_geff(_one_geff())
    g = res[0] if isinstance(res, tuple) else res
    assert g.num_nodes() > 0


def test_audit_block_uses_the_verified_loader_and_not_the_broken_one():
    src = (ROOT / "scripts" / "kaggle_edits" / "d1_response_audit.py").read_text(encoding="utf-8")
    assert "IndexedRXGraph.from_geff(" in src
    # Match an actual CALL, not a mention: the block deliberately names the broken API in a
    # comment recording why v3 failed, and that comment must not trip this gate.
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert ".io.load_geff(" not in code, "audit still CALLS the non-existent loader"


def test_gt_fields_the_audit_reads_exist_with_expected_shapes():
    """t / z / y / x, read as ATTRIBUTES -- never decoded from node ids."""
    from biotrack.metric import load_graph

    g = load_graph(_one_geff())
    na = g.node_attrs(attr_keys=["t", "z", "y", "x"])
    for col in ("t", "y", "x", "z"):
        assert col in na.columns, f"GT attribute {col} missing"
    ts = na["t"].to_list()
    assert len(ts) == g.num_nodes() and all(int(t) >= 0 for t in ts)
    for col in ("z", "y", "x"):
        vals = na[col].to_list()
        assert all(float(v) >= 0 for v in vals), f"negative GT coordinate in {col}"


def test_output_grid_geometry_matches_what_the_audit_assumes():
    """scale * downsample must be isotropic 1.625 um, and the pool kernel (3,3,3)."""
    sys.path.insert(0, str(ROOT / "vendor" / "kaggle-cell-tracking" / "scripts"))
    from predict_unet_transformer import pool_kernel_from_um

    step = tuple(s * d for s, d in zip(VOXEL_SCALE_UM, (1, 4, 4)))
    assert step == pytest.approx((1.625, 1.625, 1.625), abs=1e-9)
    assert tuple(pool_kernel_from_um(5.0, step)) == (3, 3, 3)


def test_scorer_matching_constants_are_the_ones_the_partition_uses():
    from biotrack.d1_partition import MATCH_UM
    from biotrack.metric import MAX_DISTANCE

    assert MATCH_UM == MAX_DISTANCE == 7.0, (
        "the partition must use the scorer's own match radius, not a private constant"
    )


def test_cube_corner_bound_is_what_explains_the_v4_distances():
    """15 um axis-aligned box -> 25.98 um corner. v4 reported max 24.4 um, i.e. a corner."""
    import math

    from biotrack.d1_partition import SEARCH_UM

    assert SEARCH_UM * math.sqrt(3) == pytest.approx(25.98, abs=0.01)
    assert 24.4 < SEARCH_UM * math.sqrt(3)
