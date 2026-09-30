"""``OUTPUT_LINEFIT_SMOOTH`` must not be able to push a node out of the volume.

P0-C emitted node 15274 of ``44b6_0b24845f`` at ``t=43`` with ``z=64`` (``z_max=63``):
the node is a track ENDPOINT, so its line fit is one-sided and extrapolates. The repair
restores the original detector coordinate per axis, and never clamps to the face.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from biotrack import wrapper as W  # noqa: E402


def _stats():
    s = {"linefit_smoothed_nodes": 0, "linefit_skipped_nodes": 0}
    for a in W.VOLUME_AXES:
        s[f"linefit_volume_fallback_{a}"] = 0
        s[f"linefit_volume_original_invalid_{a}"] = 0
    return s


def _chain(coords):
    """A single unbranched track, one node per frame, at the given (z, y, x) triples."""
    nodes = {
        i + 1: {"node_id": i + 1, "t": i, "z": float(z), "y": float(y), "x": float(x)}
        for i, (z, y, x) in enumerate(coords)
    }
    edges = [{"source_id": i, "target_id": i + 1} for i in range(1, len(coords))]
    return nodes, edges


def _run(nodes, edges, guard: bool):
    stats = _stats()
    prev = W.OUTPUT_VOLUME_GUARD
    W.OUTPUT_VOLUME_GUARD = guard
    try:
        out = W.linefit_smooth_output_graph(nodes, edges, stats)
    finally:
        W.OUTPUT_VOLUME_GUARD = prev
    return out, stats


# z track that reproduces the P0-C failure shape: the endpoint sits ON the top face and
# the next two frames dive, so the one-sided fit extrapolates the endpoint above it.
ON_FACE = [(63, 183, 248), (63, 188, 248), (59, 192, 248), (59, 192, 247)]
# same failure, but the endpoint's ORIGINAL is 62 -- a clamp and a restore differ here.
OFF_FACE = [(62, 183, 248), (63, 188, 248), (50, 192, 248), (50, 192, 247)]


def test_endpoint_extrapolation_leaves_the_volume_without_the_guard():
    out, stats = _run(*_chain(ON_FACE), guard=False)
    assert int(round(out[1]["z"])) > W.OUTPUT_VOLUME_ZYX[0] - 1
    assert stats["linefit_volume_fallback_z"] == 0


def test_guard_restores_the_original_and_does_not_clamp():
    out, stats = _run(*_chain(ON_FACE), guard=True)
    assert out[1]["z"] == 63.0                      # the ORIGINAL, byte for byte
    assert stats["linefit_volume_fallback_z"] == 1

    # On-face originals cannot distinguish a restore from a clamp, so repeat with an
    # original that is strictly inside: a clamp would write 63, a restore writes 62.
    unguarded, _ = _run(*_chain(OFF_FACE), guard=False)
    assert int(round(unguarded[1]["z"])) > W.OUTPUT_VOLUME_ZYX[0] - 1
    out2, stats2 = _run(*_chain(OFF_FACE), guard=True)
    assert stats2["linefit_volume_fallback_z"] == 1
    assert out2[1]["z"] == 62.0
    assert out2[1]["z"] != float(W.OUTPUT_VOLUME_ZYX[0] - 1)


def test_fallback_is_axis_wise_not_node_wise():
    # z extrapolates out of the top face; y and x stay well inside and must keep smoothing
    nodes, edges = _chain([(63, 100, 100), (63, 104, 108), (59, 108, 116), (59, 112, 124)])
    out_g, stats = _run({k: dict(v) for k, v in nodes.items()}, edges, guard=True)
    out_u, _ = _run({k: dict(v) for k, v in nodes.items()}, edges, guard=False)
    assert stats["linefit_volume_fallback_z"] == 1
    assert stats["linefit_volume_fallback_y"] == 0
    assert stats["linefit_volume_fallback_x"] == 0
    assert out_g[1]["z"] != out_u[1]["z"]
    assert out_g[1]["y"] == out_u[1]["y"]   # untouched axes are bit-identical
    assert out_g[1]["x"] == out_u[1]["x"]


def test_guard_is_a_no_op_when_nothing_leaves_the_volume():
    nodes, edges = _chain([(30, 100, 100), (31, 104, 108), (32, 108, 116), (33, 112, 124)])
    out_g, stats = _run({k: dict(v) for k, v in nodes.items()}, edges, guard=True)
    out_u, _ = _run({k: dict(v) for k, v in nodes.items()}, edges, guard=False)
    assert all(stats[f"linefit_volume_fallback_{a}"] == 0 for a in W.VOLUME_AXES)
    for nid in out_g:
        for a in W.VOLUME_AXES:
            assert out_g[nid][a] == out_u[nid][a]


def test_rounding_is_part_of_the_predicate():
    # the writer emits int(round(v)), so 63.5 already names voxel 64
    assert W.coordinate_in_volume(63.4, 0)
    assert not W.coordinate_in_volume(63.6, 0)
    assert not W.coordinate_in_volume(-0.6, 0)
    assert W.coordinate_in_volume(255.0, 1)
    assert not W.coordinate_in_volume(255.6, 2)


def test_an_already_invalid_original_is_counted_not_masked():
    # detector coordinate itself outside the volume: restoring it is correct (smoothing is
    # not to blame) but it must be reported so the structural audit still fails loudly
    nodes, edges = _chain([(70, 100, 100), (69, 104, 108), (67, 108, 116), (66, 112, 124)])
    out, stats = _run(nodes, edges, guard=True)
    assert stats["linefit_volume_fallback_z"] >= 1
    assert stats["linefit_volume_original_invalid_z"] >= 1
    assert out[1]["z"] == 70.0  # not clamped to 63
