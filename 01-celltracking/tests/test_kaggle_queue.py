"""Software contracts for scripts/core/kaggle_queue.py - the scheduler logic with fakes.

Nothing here touches Kaggle. The runner's side effects are injected, so these tests pin the
ordering, the slot cap (FACT-0061), producer -> consumer dependencies, the terminal-state
handling, and the 'submit only what was named' rule.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "core"))
import kaggle_queue as Q  # noqa: E402
import kaggle_factory as KF  # noqa: E402


def item(name, slug=None, deps=(), expects=False, status="pending"):
    return Q.Item(spec_path=f"{name}.json", name=name, slug=slug or name, expects_submission=expects,
                  deps=list(deps), fetch=["x"], status=status)


def test_plan_respects_slot_cap_and_order():
    items = [item("a"), item("b"), item("c")]
    plan = Q.plan_pushes(items, max_running=2)
    assert [i.name for i in plan] == ["a", "b"]
    items[0].status = "running"
    assert [i.name for i in Q.plan_pushes(items, 2)] == ["b"]


def test_consumer_waits_for_its_producer_to_complete():
    prod = item("h1r_edge_s5")
    cons = item("deploy_h1r_edge_s5_loeo_f0", deps=["h1r_edge_s5"])
    other = item("p23")
    items = [prod, cons, other]
    assert [i.name for i in Q.plan_pushes(items, 2)] == ["h1r_edge_s5", "p23"]
    prod.status = "running"; other.status = "running"
    assert Q.plan_pushes(items, 2) == []
    prod.status = "complete"
    assert [i.name for i in Q.plan_pushes(items, 2)] == ["deploy_h1r_edge_s5_loeo_f0"]


def test_tick_pushes_polls_and_routes_terminal_states():
    items = [item("a", expects=True), item("b"), item("c")]
    statuses = {"a": "RUNNING", "b": "RUNNING"}
    pushed, completed = [], []

    def status_fn(slug):
        return statuses[slug]

    def push_fn(it):
        pushed.append(it.name)
        return 0

    def complete_fn(it):
        completed.append(it.name)
        it.status = "fetched"

    ev = Q.tick(items, status_fn=status_fn, push_fn=push_fn, complete_fn=complete_fn, max_running=2, now=0.0)
    assert pushed == ["a", "b"] and [i.status for i in items] == ["running", "running", "pending"]
    statuses["a"] = "COMPLETE"; statuses["b"] = "ERROR (OOM)"
    statuses["c"] = "RUNNING"
    ev = Q.tick(items, status_fn=status_fn, push_fn=push_fn, complete_fn=complete_fn, max_running=2, now=3600.0)
    assert completed == ["a"]
    assert items[0].status == "fetched" and items[1].status == "error"
    # the freed slots are refilled in the same tick
    assert items[2].status == "running" and pushed == ["a", "b", "c"]
    assert any("COMPLETE" in e for e in ev) and any("ERROR" in e for e in ev)
    statuses["c"] = "COMPLETE"
    Q.tick(items, status_fn=status_fn, push_fn=push_fn, complete_fn=complete_fn, max_running=2, now=7200.0)
    assert Q.all_done(items)


def test_failed_push_marks_error_and_does_not_block_the_queue():
    items = [item("bad"), item("good")]

    def push_fn(it):
        return 1 if it.name == "bad" else 0

    Q.tick(items, status_fn=lambda s: "RUNNING", push_fn=push_fn, complete_fn=lambda it: None, max_running=2, now=0.0)
    assert items[0].status == "error" and items[1].status == "running"


def test_post_complete_failure_is_recorded_not_raised():
    items = [item("a", status="running", expects=True)]

    def complete_fn(it):
        raise RuntimeError("audit failed")

    Q.tick(items, status_fn=lambda s: "COMPLETE", push_fn=lambda it: 0, complete_fn=complete_fn, max_running=2, now=1.0)
    assert items[0].status == "complete" and "audit failed" in items[0].note


def test_state_round_trip(tmp_path):
    items = [item("a", status="running"), item("b", deps=["a"])]
    p = tmp_path / "state.json"
    Q.save_state(p, items)
    back = Q.load_state(p)
    assert [i.name for i in back] == ["a", "b"] and back[1].deps == ["a"] and back[0].status == "running"


def test_default_fetch_by_spec_kind():
    sub = {"expects_submission": True, "edits": []}
    loeo = {"expects_submission": False, "edits": [{"kind": "env", "vars": {"BIOHUB_LOEO_FOLD": "0", "BIOHUB_LOEO_ARM": "strict"}}]}
    train = {"expects_submission": False, "edits": []}
    assert Q.default_fetch(sub) == ["submission.csv", "run_stats.csv"]
    assert Q.default_fetch(loeo) == ["loeo_split0_strict.csv.gz", "loeo_manifest.json", "run_stats.csv"]
    assert Q.default_fetch(train) == ["metrics.json", "summary.json", "config.json"]


def test_real_specs_form_a_valid_queue_with_the_consumer_after_its_producer():
    names = ["p24_deepcenter_best_veto", "p25_recipe_parity_loeo_f0", "p23_icom155",
             "h1r_edge_s5", "deploy_h1r_edge_s5_loeo_f0"]
    paths = [ROOT / "scripts" / "kaggle_specs" / f"{n}.json" for n in names]
    specs = [KF.load_spec(p) for p in paths]
    slugs = {s["slug"] for s in specs}
    items = [Q.item_from_spec(p, s, slugs) for p, s in zip(paths, specs)]
    by = {i.name: i for i in items}
    assert by["deploy_h1r_edge_s5_loeo_f0"].deps == ["biohub-h1r-edge-s5"]
    assert by["h1r_edge_s5"].deps == []
    assert by["p24_deepcenter_best_veto"].expects_submission and by["p23_icom155"].expects_submission
    assert not by["p25_recipe_parity_loeo_f0"].expects_submission
    assert by["p25_recipe_parity_loeo_f0"].fetch[0] == "loeo_split0_strict.csv.gz"
    assert len(slugs) == 5, "every queued spec must have a distinct slug"
    # a submission that was never named is never submitted, by construction of make_complete_fn
    assert "submit_names" in Q.make_complete_fn.__code__.co_varnames
