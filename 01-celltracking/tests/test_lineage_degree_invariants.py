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


sys.path.insert(0, str(ROOT / "scripts" / "core"))
import baseline_contract as BC  # noqa: E402


@pytest.fixture(scope="module")
def wrapper():
    """Configured from the OPERATIONAL BASE, read out of the built notebook.

    This fixture used to hard-code 4.66 / 8.5 / 7.65 - `p3_harmonic`'s geometry, which CLAUDE.md
    explicitly says is NOT the champion. Two other committed places held the same stale triple
    (`constant_audit.py`, `scripts/d1/gt_division_gates.py`) and nothing anywhere encoded the
    operational base's 7.0 / 12.0 / 10.0. Three independent copies, three independent chances to
    go stale, and all three took it. So the numbers are no longer written here at all: they come
    from `baseline_contract`, which regenerates from the notebook and has a --check drift lock.

    The two frac caps were NOT stale, and that is worth noticing rather than glossing - a fixture
    can be half-current, which is harder to spot than one that is wholly wrong.
    """
    geometry = BC.safe_division()
    for key, value in geometry.items():
        os.environ[key] = repr(value) if not isinstance(value, float) else f"{value}"
    from biotrack import wrapper as mod

    for key, value in geometry.items():
        setattr(mod, key.removeprefix("BIOHUB_"), value)
    setattr(mod, "OUTPUT_SAFE_DIVISIONS", True)
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


def test_gap_close_cannot_give_a_node_two_parents(wrapper):
    """A node consumed as a gap-close TARGET must not be reusable as a gap-close MIDDLE.

    `close_single_frame_gaps` kept two disjoint consumption sets. `used_starts` recorded
    targets, `used_isolated` recorded reused middles, and NEITHER admission guard consulted
    the other. So a node taken as a target (recorded in `used_starts`/`incoming`) stayed
    eligible as a reused middle, where only `used_isolated` was checked -- and picked up a
    SECOND incoming edge.

    Found on the real pre-wrapper substrate: the full-corpus census died at crop 185/199 with
    `in-degree>1 on [3863]`. It never fired on the post-wrapper cache, which lacks 83,260 of
    the nodes the gap-closer actually sees, so the wrong substrate had been masking it.

    Reachable ordering, which this reconstructs: frames are walked in ascending t, so the
    pair whose TARGET is X is processed before the pair that wants X as its MIDDLE.
    """
    mod = wrapper
    for name, value in {
        "OUTPUT_GAP_CLOSE": True,
        "GAP_CLOSE_MAX_GAP": 1,
        "GAP_CLOSE_UM": 40.0,
        "GAP_CLOSE_REUSE_EXISTING": True,
        "GAP_CLOSE_REUSE_UM": 40.0,
        "GAP_DENSITY_ADAPTIVE": False,
        "DEEPCENTER_GAP_VETO": False,
        "OUTPUT_GAP2_RECOVERY": False,
    }.items():
        if hasattr(mod, name):
            setattr(mod, name, value)

    #   t=0  S2(isolated end)            t=1  S1(end, has a parent)
    #   t=2  X (isolated -> start AND reuse candidate)
    #   t=3  T1(start)
    # pair B (t=0): S2 -> [synthetic] -> X          consumes X as a TARGET
    # pair A (t=1): S1 -> [X reused]   -> T1        would give X a second parent
    nodes = {
        0: _node(0, 0, x=0.0),          # S2
        1: _node(1, 1, x=0.0),          # S1
        2: _node(2, 2, x=0.0),          # X  (isolated)
        3: _node(3, 3, x=0.0),          # T1
        4: _node(4, 0, x=200.0),        # parent of S1, keeps S1 out of starts
    }
    edges = [{"source_id": 4, "target_id": 1, "edge_prob": 0.9, "distance_um": 0.0}]
    stats = defaultdict(int)

    new_nodes, new_edges = mod.close_single_frame_gaps(nodes, edges, stats)

    _, inn = _degrees(new_edges)
    assert max(inn.values()) <= 1, (
        f"gap close gave a node two parents: "
        f"{sorted(n for n, d in inn.items() if d > 1)}"
    )
    mod.assert_degree_invariants(new_edges, "gap close regression")


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
