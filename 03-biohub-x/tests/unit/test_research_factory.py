"""The controller/worker boundary, and the four properties it exists to enforce.

Each test is one way parallel research could quietly become something other than
what was dispatched: a request edited after the fact, an attempt overwritten by
its own retry, a taint dropped on the way through, a worker awarding itself
standing, or two disagreeing results averaged into one that neither produced.

The lock test is the reason the module exists at all. Sequential ``RL-`` ids
allocated by a plain read, modify, write are correct for one writer and lossy for
two, and the fix has to be demonstrated rather than asserted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from biohubx.research.factory import (
    TAINTS,
    WORKER_STATUSES,
    FactoryError,
    collect,
    dispatch,
    record_result,
)
from biohubx.research.ledger import LedgerError, ledger_lock

CONTEXT = {
    "commit": "0" * 40,
    "dependency_lock": "raw_artifact_sha256:sha256:" + "a" * 64,
    "image": "local-python-3.12",
}

JOB = {
    "job": "mechanism-extraction",
    "question": "what mechanism does the public notebook actually use",
    "falsifier": "no mechanism is identifiable from its source alone",
    "inputs": ["a public notebook URL"],
    "split": "not applicable; this job reads no competition data",
    "seed": 0,
    "budget": {"requests": 50, "bytes": 100_000_000, "minutes": 30},
    "taints": ["reference_only"],
}


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "registry").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _dispatch_one(tmp_path: Path, job: dict[str, Any] | None = None) -> Path:
    root = _repo(tmp_path)
    outcome = dispatch(root, campaign="RX-02", jobs=[dict(job or JOB)], context=dict(CONTEXT))
    run_id = str(outcome["dispatched"][0]["run_id"])
    return root / "artifacts/factory/RX-02" / run_id / "request.json"


def _draft(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "status": "integration_only",
        "conclusion": "the notebook thresholds a response map and links greedily",
        "answers_falsifier": True,
        "taints": ["reference_only"],
        "claims": [{"claim": "greedy linking", "source": "RL-0001", "locator": "cell 4"}],
        "ledger_additions": [],
        "budget_used": {"requests": 3, "bytes": 12_000, "minutes": 4},
    }
    body.update(overrides)
    return body


# --- immutability -----------------------------------------------------------


def test_a_request_edited_after_dispatch_is_refused(tmp_path: Path) -> None:
    """A job is frozen when it is issued. If it can be edited afterwards, nothing
    downstream is pinned to anything."""
    request_path = _dispatch_one(tmp_path)
    body = json.loads(request_path.read_text(encoding="utf-8"))
    body["budget"]["minutes"] = 9999
    request_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(FactoryError, match="immutable"):
        record_result(tmp_path, request_path=request_path, draft=_draft())


def test_a_result_edited_after_it_was_written_is_caught_at_collection(tmp_path: Path) -> None:
    request_path = _dispatch_one(tmp_path)
    record_result(tmp_path, request_path=request_path, draft=_draft())
    result_path = request_path.parent / "result.json"
    body = json.loads(result_path.read_text(encoding="utf-8"))
    body["conclusion"] = "something the worker never concluded"
    result_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(FactoryError, match="changed after it was written"):
        collect(tmp_path, campaign="RX-02")


# --- unique retries ---------------------------------------------------------


def test_a_retry_takes_a_new_run_id_and_leaves_the_first_attempt_standing(tmp_path: Path) -> None:
    """A second attempt at a question is a second record, not a correction of the
    first. Both are collected and both keep their own digests."""
    root = _repo(tmp_path)
    first = dispatch(root, campaign="RX-02", jobs=[dict(JOB)], context=dict(CONTEXT))
    second = dispatch(root, campaign="RX-02", jobs=[dict(JOB)], context=dict(CONTEXT))

    assert first["dispatched"][0]["run_id"] == "mechanism-extraction-a01"
    assert second["dispatched"][0]["run_id"] == "mechanism-extraction-a02"
    assert (root / "artifacts/factory/RX-02/mechanism-extraction-a01/request.json").is_file()
    assert (root / "artifacts/factory/RX-02/mechanism-extraction-a02/request.json").is_file()
    parent = json.loads(
        (root / "artifacts/factory/RX-02/mechanism-extraction-a02/request.json").read_text(encoding="utf-8")
    )
    assert parent["parent_run_id"] == "mechanism-extraction-a01"


def test_a_second_result_for_one_run_is_refused(tmp_path: Path) -> None:
    request_path = _dispatch_one(tmp_path)
    record_result(tmp_path, request_path=request_path, draft=_draft())

    with pytest.raises(FactoryError, match="never overwritten"):
        record_result(tmp_path, request_path=request_path, draft=_draft(conclusion="a different answer"))


# --- monotonic taints -------------------------------------------------------


def test_a_result_may_add_a_taint(tmp_path: Path) -> None:
    request_path = _dispatch_one(tmp_path)

    recorded = record_result(
        tmp_path,
        request_path=request_path,
        draft=_draft(taints=["reference_only", "unknown_training_data"]),
    )

    assert recorded["taints"] == ["reference_only", "unknown_training_data"]


def test_a_result_may_not_drop_a_taint_it_inherited(tmp_path: Path) -> None:
    """The whole point of a taint is that a later step cannot forget it."""
    request_path = _dispatch_one(tmp_path)

    with pytest.raises(FactoryError, match="monotonic"):
        record_result(tmp_path, request_path=request_path, draft=_draft(taints=[]))


def test_an_invented_taint_is_refused(tmp_path: Path) -> None:
    request_path = _dispatch_one(tmp_path)

    with pytest.raises(FactoryError, match="unknown taints"):
        record_result(
            tmp_path, request_path=request_path, draft=_draft(taints=["reference_only", "looks_fine"])
        )


def test_the_campaign_carries_every_taint_any_run_carried(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    dispatch(root, campaign="RX-02", jobs=[dict(JOB), {**JOB, "job": "second"}], context=dict(CONTEXT))
    record_result(
        root,
        request_path=root / "artifacts/factory/RX-02/mechanism-extraction-a01/request.json",
        draft=_draft(taints=["reference_only", "networked_runtime"]),
    )
    record_result(
        root,
        request_path=root / "artifacts/factory/RX-02/second-a01/request.json",
        draft=_draft(taints=["reference_only"]),
    )

    collection = collect(root, campaign="RX-02")

    assert collection["campaign_taints"] == ["networked_runtime", "reference_only"]


# --- workers cannot promote -------------------------------------------------


@pytest.mark.parametrize("status", ["challenger", "promoted"])
def test_a_worker_cannot_declare_standing(tmp_path: Path, status: str) -> None:
    """Standing is decided in the registries under AGENTS.md section 4, and a
    worker that could award it to itself is a worker that decides promotions."""
    request_path = _dispatch_one(tmp_path)

    with pytest.raises(FactoryError, match="may not declare"):
        record_result(tmp_path, request_path=request_path, draft=_draft(status=status))


def test_a_worker_reports_one_of_three_statuses(tmp_path: Path) -> None:
    assert WORKER_STATUSES == ("integration_only", "killed", "invalid")
    request_path = _dispatch_one(tmp_path)

    with pytest.raises(FactoryError, match="status must be one of"):
        record_result(tmp_path, request_path=request_path, draft=_draft(status="interesting"))


# --- budget, sources and conflict -------------------------------------------


def test_a_run_that_outgrew_its_budget_is_refused(tmp_path: Path) -> None:
    request_path = _dispatch_one(tmp_path)

    with pytest.raises(FactoryError, match="exceeded its declared budget"):
        record_result(
            tmp_path,
            request_path=request_path,
            draft=_draft(budget_used={"requests": 5000, "bytes": 1, "minutes": 1}),
        )


def test_a_claim_with_no_source_is_refused(tmp_path: Path) -> None:
    request_path = _dispatch_one(tmp_path)

    with pytest.raises(FactoryError, match="no source"):
        record_result(
            tmp_path, request_path=request_path, draft=_draft(claims=[{"claim": "it works", "source": ""}])
        )


def test_two_attempts_that_disagree_are_compared_and_neither_is_rewritten(tmp_path: Path) -> None:
    """Averaging two disagreeing results produces a number neither run measured.
    The disagreement is the finding, so both stand and the collection names it."""
    root = _repo(tmp_path)
    dispatch(root, campaign="RX-02", jobs=[dict(JOB)], context=dict(CONTEXT))
    dispatch(root, campaign="RX-02", jobs=[dict(JOB)], context=dict(CONTEXT))
    record_result(
        root,
        request_path=root / "artifacts/factory/RX-02/mechanism-extraction-a01/request.json",
        draft=_draft(conclusion="greedy linking"),
    )
    record_result(
        root,
        request_path=root / "artifacts/factory/RX-02/mechanism-extraction-a02/request.json",
        draft=_draft(conclusion="an integer program, not greedy linking", status="killed"),
    )

    collection = collect(root, campaign="RX-02")

    comparison = next(c for c in collection["comparisons"] if c["job"] == "mechanism-extraction")
    assert comparison["agree"] is False
    assert "disagree" in comparison["resolution"]
    assert sorted(comparison["attempts"]) == ["mechanism-extraction-a01", "mechanism-extraction-a02"]
    assert len(comparison["conclusions"]) == 2
    assert len(collection["runs"]) == 2


# --- the ledger lock --------------------------------------------------------


def test_only_the_controller_writes_the_ledger_and_it_takes_a_lock(tmp_path: Path) -> None:
    """Two writers allocating RL ids under a plain read, modify, write lose one
    record. The merge happens once, in collect, and it is locked."""
    root = _repo(tmp_path)
    (root / "registry/research-ledger.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "policy": "D-0039", "entries": []}), encoding="utf-8"
    )
    dispatch(root, campaign="RX-02", jobs=[dict(JOB), {**JOB, "job": "second"}], context=dict(CONTEXT))
    for run_id, url in (
        ("mechanism-extraction-a01", "https://example.org/one"),
        ("second-a01", "https://example.org/two"),
    ):
        record_result(
            root,
            request_path=root / "artifacts/factory/RX-02" / run_id / "request.json",
            draft=_draft(
                ledger_additions=[{"url": url, "kind": "notebook", "campaign": "RX-02", "note": "n"}]
            ),
        )

    collection = collect(root, campaign="RX-02")

    assert collection["ledger"]["added"] == ["RL-0001", "RL-0002"]
    entries = yaml.safe_load((root / "registry/research-ledger.yaml").read_text(encoding="utf-8"))["entries"]
    assert [entry["id"] for entry in entries] == ["RL-0001", "RL-0002"]
    assert {entry["status"] for entry in entries} == {"reference_only"}


def test_a_held_lock_refuses_rather_than_writing_anyway(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with (
        ledger_lock(root),
        pytest.raises(LedgerError, match="was held for more than"),
        ledger_lock(root, timeout=0.1),
    ):
        pass


def test_the_lock_is_released_even_when_the_body_raises(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with pytest.raises(RuntimeError), ledger_lock(root):
        raise RuntimeError("the body failed")

    with ledger_lock(root, timeout=0.1):
        pass


# --- what is frozen, and what is not written --------------------------------


def test_every_request_pins_the_things_a_rerun_would_need(tmp_path: Path) -> None:
    request = json.loads(_dispatch_one(tmp_path).read_text(encoding="utf-8"))

    for field in (
        "commit",
        "dependency_lock",
        "image",
        "inputs",
        "split",
        "seed",
        "budget",
        "falsifier",
        "taints",
        "write_scope",
    ):
        assert field in request, field
    assert request["write_scope"] == "artifacts/factory/RX-02/mechanism-extraction-a01"


def test_a_worker_writes_only_inside_its_own_run_directory(tmp_path: Path) -> None:
    """The scope is declared in the request and honoured by the only writer there
    is: `record_result` writes the result beside the request and nowhere else."""
    root = _repo(tmp_path)
    request_path = _dispatch_one(tmp_path)
    before = {path for path in root.rglob("*") if path.is_file()}

    record_result(root, request_path=request_path, draft=_draft())

    written = {path for path in root.rglob("*") if path.is_file()} - before
    run_dir = request_path.parent
    assert written
    assert all(path.parent == run_dir for path in written), sorted(str(p) for p in written)


def test_the_dossier_is_generated_from_the_collection(tmp_path: Path) -> None:
    """A hand-maintained dossier is a second place for a fact to be true."""
    root = _repo(tmp_path)
    dispatch(root, campaign="RX-02", jobs=[dict(JOB)], context=dict(CONTEXT))
    record_result(
        root,
        request_path=root / "artifacts/factory/RX-02/mechanism-extraction-a01/request.json",
        draft=_draft(),
    )

    collect(root, campaign="RX-02")

    body = json.loads((root / "artifacts/factory/RX-02/DOSSIER.json").read_text(encoding="utf-8"))
    assert body["generated_from"] == "COLLECTION.json"
    entry = body["jobs"]["mechanism-extraction"]
    assert entry["question"] == JOB["question"]
    assert entry["falsifier"] == JOB["falsifier"]
    assert entry["attempts"][0]["status"] == "integration_only"
    assert set(entry["taints"]) <= set(TAINTS)
