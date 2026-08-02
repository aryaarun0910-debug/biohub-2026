"""The pipeline's lineage-degree invariants must be enforced, not merely assumed.

Every stage downstream of the relink assumes in-degree <= 1 and out-degree <= 2, and the
submission guard rejects any movie that breaks them. Nothing enforced out-degree <= 2.

`add_safe_divisions_postlink` builds a proposal list that may contain SEVERAL candidates for
the same source, then dedupes admissions by TARGET only (`used_targets` / `incoming`) and
never advances `out_by_source`. A source with one existing child can therefore be handed two
safe divisions in the same frame and reach out-degree 3. That is what blocked the first arm-B
deployment run: exactly one node in 121,003 at out-degree 3, zero in-degree violations.

The relink is NOT the site -- it replaces the whole edge list and assigns one-to-one per
frame pair, so it emits out-degree <= 1 by construction. `test_relink_cannot_exceed_out_degree_one`
locks that, because a guard placed there would be a silent no-op.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def wrapper():
    for key, value in {
        "BIOHUB_SAFE_DIV_MAX_UM": "4.66",
        "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "8.5",
        "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "7.65",
        "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.0076",
        "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.00375",
    }.items():
        os.environ[key] = value
    from biotrack import wrapper as mod

    for name, value in {
        "SAFE_DIV_MAX_UM": 4.66,
        "SAFE_DIV_SISTER_MAX_UM": 8.5,
        "SAFE_DIV_EXISTING_CHILD_MAX_UM": 7.65,
        "SAFE_DIV_FRAME_FRAC_CAP": 0.0076,
        "SAFE_DIV_GLOBAL_FRAC_CAP": 0.00375,
        "OUTPUT_SAFE_DIVISIONS": True,
    }.items():
        setattr(mod, name, value)
    return mod


def _node(node_id, t, z=0.0, y=0.0, x=0.0):
    return {"node_id": node_id, "t": t, "z": z, "y": y, "x": x}


def _degrees(edges):
    out, inn = defaultdict(int), defaultdict(int)
    for edge in edges:
        out[int(edge["source_id"])] += 1
        inn[int(edge["target_id"])] += 1
    return out, inn


def _two_candidate_graph(n_chains=210, n_candidates=2):
    """Well-separated 3-frame chains, plus `n_candidates` unlinked nodes flanking
    chain 0's source. Sized so frame_cap and global_cap both resolve to 2."""
    nodes, edges = {}, []
    node_id = 0
    for chain in range(n_chains):
        y = chain * 100.0
        a, b, c = node_id, node_id + 1, node_id + 2
        node_id += 3
        nodes[a], nodes[b], nodes[c] = _node(a, 0, y=y), _node(b, 1, y=y), _node(c, 2, y=y)
        edges.append({"source_id": a, "target_id": b, "edge_prob": 0.9, "distance_um": 0.0})
        edges.append({"source_id": b, "target_id": c, "edge_prob": 0.9, "distance_um": 0.0})
    for offset in range(n_candidates):
        nodes[node_id] = _node(node_id, 1, x=4.0 * (1 if offset % 2 == 0 else -1))
        node_id += 1
    return nodes, edges


def test_safe_divisions_cannot_create_out_degree_three(wrapper):
    """Two admissible candidates on one source must yield a fork, never a triple fork."""
    nodes, edges = _two_candidate_graph()
    stats = defaultdict(int)

    result = wrapper.add_safe_divisions_postlink(nodes, edges, stats)

    out, inn = _degrees(result)
    assert max(out.values()) == 2, "source must stop at a two-way division"
    assert max(inn.values()) == 1, "target-side dedupe must still hold"
    assert stats["safe_divisions_added"] == 1
    assert stats["safe_division_skipped_outdegree"] == 1, (
        "the third child must be rejected at admission and counted"
    )


def test_guard_is_inert_when_only_one_candidate_exists(wrapper):
    """A legitimate single division is untouched -- the P0-B path must not move."""
    nodes, edges = _two_candidate_graph(n_candidates=1)
    stats = defaultdict(int)

    result = wrapper.add_safe_divisions_postlink(nodes, edges, stats)

    out, _ = _degrees(result)
    assert max(out.values()) == 2
    assert stats["safe_divisions_added"] == 1
    assert stats["safe_division_skipped_outdegree"] == 0


def test_rejected_proposal_does_not_consume_the_cap_budget(wrapper):
    """A rejected third child frees its slot for the next valid proposal.

    `SAFE_DIV_GLOBAL_FRAC_CAP` is a budget on how many safe divisions may be added. The guard
    `continue`s rather than counting the rejection, so when the cap binds the fix is a one-edge
    SWAP, not a pure removal: the invalid third child is dropped and a valid division that the
    cap had crowded out is admitted in its place.

    Observed on real data -- P0-strict crop 44b6_a2bb48bb, global cap binding at exactly 166:
    removed (16693, 17368), added (38172, 38903), total safe divisions 166 before and after.
    Locked here because it is a deliberate choice: an invalid proposal must not spend budget.
    """
    nodes, edges = {}, []
    node_id = 0
    for chain in range(220):                       # 660 nodes / 440 edges
        y = chain * 100.0                          # global_cap = round(440*0.00375) = 2
        a, b, c = node_id, node_id + 1, node_id + 2
        node_id += 3
        nodes[a], nodes[b], nodes[c] = _node(a, 0, y=y), _node(b, 1, y=y), _node(c, 2, y=y)
        edges.append({"source_id": a, "target_id": b, "edge_prob": 0.9, "distance_um": 0.0})
        edges.append({"source_id": b, "target_id": c, "edge_prob": 0.9, "distance_um": 0.0})

    src0, mid1 = 0, 4                              # chain 0 t=0 source; chain 1 t=1 middle
    cand_a, cand_b, cand_c = node_id, node_id + 1, node_id + 2
    nodes[cand_a] = _node(cand_a, 1, x=3.0)        # nearest -> ranks first
    nodes[cand_b] = _node(cand_b, 1, x=-5.0)       # the would-be third child
    nodes[cand_c] = _node(cand_c, 2, y=100.0, x=4.0)   # crowded out by the cap pre-fix

    stats = defaultdict(int)
    result = wrapper.add_safe_divisions_postlink(nodes, edges, stats)

    added = {(int(e["source_id"]), int(e["target_id"]))
             for e in result if e.get("safe_division") == 1}
    out, inn = _degrees(result)

    assert stats["safe_division_skipped_outdegree"] == 1
    assert max(out.values()) == 2, "no triple fork"
    assert max(inn.values()) == 1
    assert (src0, cand_b) not in added, "the third child must be rejected"
    assert (src0, cand_a) in added, "the better-ranked division survives"
    assert (mid1, cand_c) in added, (
        "the freed cap slot must go to the next valid proposal, not be wasted"
    )
    assert stats["safe_divisions_added"] == 2, "the budget is still fully spent"


def test_relink_cannot_exceed_out_degree_one(wrapper):
    """The relink is structurally one-to-one, so an out-degree guard there is a no-op.

    Locked so nobody 'fixes' the invariant at the wrong site again.
    """
    rng = np.random.default_rng(20260802)
    worst_out = worst_in = 0
    for _ in range(60):
        nodes = {}
        node_id = 0
        for t in range(4):
            for _ in range(int(rng.integers(3, 25))):
                nodes[node_id] = _node(
                    node_id, t,
                    z=float(rng.uniform(0, 6)),
                    y=float(rng.uniform(0, 25)),
                    x=float(rng.uniform(0, 25)),
                )
                node_id += 1
        stats = defaultdict(int)
        relinked = wrapper.motion_relink_edges(nodes, stats, {})
        if relinked:
            out, inn = _degrees(relinked)
            worst_out = max(worst_out, max(out.values()))
            worst_in = max(worst_in, max(inn.values()))
    assert worst_out <= 1, "relink emitted a fork -- the one-to-one assumption broke"
    assert worst_in <= 1


def test_assert_degree_invariants_rejects_a_triple_fork(wrapper):
    bad = [
        {"source_id": 1, "target_id": 2},
        {"source_id": 1, "target_id": 3},
        {"source_id": 1, "target_id": 4},
    ]
    with pytest.raises(RuntimeError, match="out-degree>2"):
        wrapper.assert_degree_invariants(bad, "unit test")


def test_assert_degree_invariants_rejects_a_merge(wrapper):
    bad = [
        {"source_id": 1, "target_id": 3},
        {"source_id": 2, "target_id": 3},
    ]
    with pytest.raises(RuntimeError, match="in-degree>1"):
        wrapper.assert_degree_invariants(bad, "unit test")


def test_assert_degree_invariants_accepts_a_legal_division(wrapper):
    ok = [
        {"source_id": 1, "target_id": 2},
        {"source_id": 1, "target_id": 3},
        {"source_id": 2, "target_id": 4},
    ]
    wrapper.assert_degree_invariants(ok, "unit test")


def test_stats_key_is_preinitialised_in_filter_output_graph():
    """`stats` is a plain dict; a missing key would KeyError inside the pipeline."""
    source = (ROOT / "src" / "biotrack" / "wrapper.py").read_text(encoding="utf-8")
    assert '"safe_division_skipped_outdegree": 0,' in source
