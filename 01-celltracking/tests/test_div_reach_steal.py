"""Software contracts for scripts/win_bet/div_reach_steal.py (LEVER-0025 instrument, PKT-0021).

These pin the pure planner and the edit applier on hand-built toy graphs. They say nothing about
whether the mechanism scores - that is the packet's job, through the scorer round-trip.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "div_reach_steal", ROOT / "scripts" / "win_bet" / "div_reach_steal.py")
drs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drs)

D, G, C1, C2, GC2 = 900, 901, 902, 903, 904   # GT ids (divider, grandparent, children, grandchild)


def toy(edges, t_of):
    parent_of, children_of = {}, {}
    for s, t in edges:
        children_of.setdefault(s, []).append(t)
        assert t not in parent_of
        parent_of[t] = s
    pos = {n: (float(t_of[n]), float(n), 0.0) for n in t_of}
    return parent_of, children_of, pos


def test_direct_steal_removes_the_wrong_parent_edge_and_adds_mother_to_daughter():
    # M(5) -> a(6) linked; b(6) matched to C2 but owned by P(5).
    M, P, a, b = 1, 2, 3, 4
    t_of = {M: 5, P: 5, a: 6, b: 6}
    parent_of, children_of, pos = toy([(M, a), (P, b)], t_of)
    plan = drs.plan_division(t_div=5, divider=D, parent_ids={M}, daughter_ids=[{a}, {b}],
                             node_to_gt={M: D, a: C1, b: C2}, parent_of=parent_of,
                             children_of=children_of, t_of=t_of, pos_um=pos)
    assert plan == {"route": "direct", "fork": M, "keep": [a], "add": [(M, b)],
                    "remove": [(P, b)], "daughter_kinds": ["direct"]}


def test_orphan_second_daughter_needs_no_removal():
    M, a, b = 1, 3, 4
    t_of = {M: 5, a: 6, b: 6}
    parent_of, children_of, pos = toy([(M, a)], t_of)
    plan = drs.plan_division(t_div=5, divider=D, parent_ids={M}, daughter_ids=[{a}, {b}],
                             node_to_gt={M: D, a: C1, b: C2}, parent_of=parent_of,
                             children_of=children_of, t_of=t_of, pos_um=pos)
    assert plan["add"] == [(M, b)] and plan["remove"] == []


def test_predecessor_route_forks_at_the_unmatched_detection_under_a_matched_grandparent():
    # G'(4) matched to the grandparent; its child x(5) is NOT matched to the divider; x -> a(6).
    Gp, x, P, a, b = 10, 11, 12, 13, 14
    t_of = {Gp: 4, x: 5, P: 5, a: 6, b: 6}
    parent_of, children_of, pos = toy([(Gp, x), (x, a), (P, b)], t_of)
    plan = drs.plan_division(t_div=5, divider=D, parent_ids={Gp}, daughter_ids=[{a}, {b}],
                             node_to_gt={Gp: G, a: C1, b: C2}, parent_of=parent_of,
                             children_of=children_of, t_of=t_of, pos_um=pos)
    assert plan["route"] == "pred" and plan["fork"] == x
    assert plan["add"] == [(x, b)] and plan["remove"] == [(P, b)]


def test_grandchild_evidence_admits_an_unmatched_daughter_whose_successor_matches():
    # b(6) is unmatched, but b -> gc(7) and gc is matched to a grandchild of C2.
    M, a, b, gc, P = 1, 3, 4, 5, 6
    t_of = {M: 5, P: 5, a: 6, b: 6, gc: 7}
    parent_of, children_of, pos = toy([(M, a), (P, b), (b, gc)], t_of)
    plan = drs.plan_division(t_div=5, divider=D, parent_ids={M}, daughter_ids=[{a}, {gc}],
                             node_to_gt={M: D, a: C1, gc: GC2}, parent_of=parent_of,
                             children_of=children_of, t_of=t_of, pos_um=pos)
    assert plan["add"] == [(M, b)] and plan["remove"] == [(P, b)]
    assert plan["daughter_kinds"] == ["grand"]


def test_direct_match_is_preferred_over_grandchild_evidence_then_nearest():
    M, a, b_direct, b_grand, gc, P1, P2 = 1, 3, 4, 5, 6, 7, 8
    t_of = {M: 5, P1: 5, P2: 5, a: 6, b_direct: 6, b_grand: 6, gc: 7}
    parent_of, children_of, pos = toy([(M, a), (P1, b_direct), (P2, b_grand), (b_grand, gc)], t_of)
    plan = drs.plan_division(t_div=5, divider=D, parent_ids={M}, daughter_ids=[{a}, {b_direct, gc}],
                             node_to_gt={M: D, a: C1, b_direct: C2, gc: GC2}, parent_of=parent_of,
                             children_of=children_of, t_of=t_of, pos_um=pos)
    assert plan["add"] == [(M, b_direct)]


def test_existing_fork_or_wrong_child_is_left_alone():
    M, a, b, w = 1, 3, 4, 5
    t_of = {M: 5, a: 6, b: 6, w: 6}
    # already out-degree 2
    parent_of, children_of, pos = toy([(M, a), (M, w)], t_of)
    assert drs.plan_division(t_div=5, divider=D, parent_ids={M}, daughter_ids=[{a}, {b}],
                             node_to_gt={M: D, a: C1, b: C2}, parent_of=parent_of,
                             children_of=children_of, t_of=t_of, pos_um=pos) is None
    # single child that is in no lineage
    parent_of, children_of, pos = toy([(M, w)], t_of)
    assert drs.plan_division(t_div=5, divider=D, parent_ids={M}, daughter_ids=[{a}, {b}],
                             node_to_gt={M: D, w: 777}, parent_of=parent_of,
                             children_of=children_of, t_of=t_of, pos_um=pos) is None


def test_apply_plans_preserves_degree_invariants_and_skips_colliding_plans():
    edges = {(1, 3), (2, 4), (5, 6)}
    p1 = {"route": "direct", "fork": 1, "keep": [3], "add": [(1, 4)], "remove": [(2, 4)],
          "daughter_kinds": ["direct"], "removed_edge_is_full_tp": False}
    # collides: steals the same daughter 4 for another fork
    p2 = {"route": "direct", "fork": 5, "keep": [6], "add": [(5, 4)], "remove": [(2, 4)],
          "daughter_kinds": ["direct"], "removed_edge_is_full_tp": False}
    new_edges, applied = drs.apply_plans(edges, [p1, p2])
    assert applied == [p1]
    assert new_edges == {(1, 3), (1, 4), (5, 6)}
    parents = {}
    for s, t in new_edges:
        assert t not in parents
        parents[t] = s
    out = {}
    for s, _t in new_edges:
        out[s] = out.get(s, 0) + 1
    assert max(out.values()) <= 2
