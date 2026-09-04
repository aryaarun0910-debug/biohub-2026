"""Controller and workers: parallel research without a shared writer.

The research ledger allocates sequential ``RL-`` ids with a plain read, modify,
write. One writer is fine. Two are not: both read the same highest id, both
choose the next one, and the second save silently drops the first record. That
is the reason this module exists, and it is the reason the boundary is drawn
where it is.

::

    controller
      |- immutable job -> worker A -> isolated result.json
      |- immutable job -> worker B -> isolated result.json
      '- immutable job -> worker C -> isolated result.json
                              |
                   controller verifies and collects

Only the controller writes Git or a canonical registry. A worker receives inputs
it may read and one directory it may write, ``artifacts/factory/<campaign>/<run
id>/``, which is outside Git by ``.gitignore``. It records what it acquired; it
does not merge it. The merge happens once, in ``collect``, under a lock.

Four properties are enforced here rather than asked for.

*Immutability.* A request is written once and never rewritten; a run directory
that exists is never dispatched into again. A result is written once. Both carry
typed digests, and ``collect`` recomputes both and refuses on any difference, so
an edited request or result is detectable rather than merely discouraged.

*Unique retries.* A retry is a new run id, never a second write to an old one.
Attempt 2 of a job sits beside attempt 1 with its own request, its own result and
its own digests, and both are collected.

*Monotonic taints.* A taint travels forward. A result may add taints and may
never drop one its request carried, so "this rests on a source whose training
data is unknown" cannot be laundered by a later step that forgets to mention it.

*Workers cannot promote.* ``challenger`` and ``promoted`` are refused at the
worker boundary. A worker reports ``integration_only``, ``killed`` or
``invalid``; standing is decided by the controller, in the registries, under the
rules AGENTS.md section 4 already sets.

No experiment tracker, no task database, no per-agent Markdown and no registry
per arm. The run directories are the queue, the registries stay canonical, and
the dossier is a generated view over both.

Consumers: ``biohubx research dispatch``, ``biohubx research worker`` and
``biohubx research collect``.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from biohubx.artifacts import atomic_write_text
from biohubx.hashing import DigestKind, digest_file
from biohubx.research.controller import next_action, state_from_registries
from biohubx.research.ledger import ledger_lock

FACTORY_ROOT = Path("artifacts/factory")
"""Where worker output lives. Ignored by Git, because a worker writes no history."""

TAINTS: tuple[str, ...] = (
    "reference_only",
    "unknown_training_data",
    "public_test_duplicate",
    "heldout_seen",
    "networked_runtime",
)
"""Every taint a result may carry. Monotonic: added, never removed."""

WORKER_STATUSES: tuple[str, ...] = ("integration_only", "killed", "invalid")
"""What a worker may conclude about its own run."""

CONTROLLER_ONLY_STATUSES: tuple[str, ...] = ("challenger", "promoted")
"""Standing. Not a worker's to declare, and refused by name so the refusal reads."""

REQUEST_FILE = "request.json"
RESULT_FILE = "result.json"
RESULT_DIGEST_FILE = "result.digest"
DISPATCH_FILE = "DISPATCH.json"
COLLECTION_FILE = "COLLECTION.json"
DOSSIER_FILE = "DOSSIER.json"


class FactoryError(ValueError):
    """The factory cannot proceed as asked, and says which property would break."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    """Write once, atomically, and return the file's typed canonical digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return digest_file(path, DigestKind.CANONICAL_TEXT).token


def digest_of(path: Path) -> str:
    return digest_file(path, DigestKind.CANONICAL_TEXT).token


@dataclass(frozen=True, slots=True)
class JobRequest:
    """One immutable unit of work, frozen at the moment it is dispatched."""

    campaign: str
    job: str
    run_id: str
    attempt: int
    parent_run_id: str | None
    question: str
    falsifier: str
    inputs: tuple[str, ...]
    split: str
    seed: int
    budget: dict[str, int]
    commit: str
    dependency_lock: str
    image: str
    taints: tuple[str, ...]
    write_scope: str
    dispatched_utc: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["inputs"] = list(self.inputs)
        payload["taints"] = list(self.taints)
        return payload


def frozen_context(root: Path) -> dict[str, str]:
    """Commit, dependency lock and image, resolved once and pinned into every request.

    A dirty tree is refused rather than recorded. A research result that cannot
    name the exact code that produced it is a result nobody can re-run, and the
    whole point of an immutable request is that its context is immutable too.
    """
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    if revision.returncode != 0:
        raise FactoryError("cannot resolve the repository commit; a dispatched job must pin one")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False
    ).stdout.strip()
    if dirty:
        raise FactoryError(
            "the working tree is dirty; commit before dispatching, because a job pins the commit "
            "it was frozen at and an uncommitted tree pins nothing"
        )
    lock = root / "uv.lock"
    if not lock.is_file():
        raise FactoryError("no uv.lock to pin; the dependency set must be nameable")
    return {
        "commit": revision.stdout.strip(),
        "dependency_lock": digest_file(lock, DigestKind.RAW_ARTIFACT).token,
        # No container is approved for research work, so the image is the local
        # interpreter and it is named rather than left implicit.
        "image": f"local-python-{os.sys.version_info.major}.{os.sys.version_info.minor}",  # type: ignore[attr-defined]
    }


def campaign_root(root: Path, campaign: str) -> Path:
    return root / FACTORY_ROOT / campaign


def existing_runs(root: Path, campaign: str, job: str) -> list[str]:
    """Every run directory already allocated to this job, in order."""
    base = campaign_root(root, campaign)
    if not base.is_dir():
        return []
    return sorted(item.name for item in base.iterdir() if item.is_dir() and item.name.startswith(f"{job}-a"))


def dispatch(
    root: Path,
    *,
    campaign: str,
    jobs: list[dict[str, Any]],
    context: dict[str, str],
) -> dict[str, Any]:
    """Freeze one request per job and write it where exactly one worker will read it.

    A retry is a new run id. Nothing is ever dispatched into a directory that
    exists, so an attempt cannot be overwritten by a later one that thought it
    was the same job.
    """
    base = campaign_root(root, campaign)
    dispatched: list[dict[str, Any]] = []
    for spec in jobs:
        job = str(spec["job"])
        prior = existing_runs(root, campaign, job)
        attempt = len(prior) + 1
        run_id = f"{job}-a{attempt:02d}"
        run_dir = base / run_id
        if run_dir.exists():
            raise FactoryError(f"{run_id} already exists; a retry takes a new run id, never an old one")
        taints = tuple(str(taint) for taint in spec.get("taints", ()))
        unknown = sorted(set(taints) - set(TAINTS))
        if unknown:
            raise FactoryError(f"unknown taints {unknown}; the vocabulary is {list(TAINTS)}")
        request = JobRequest(
            campaign=campaign,
            job=job,
            run_id=run_id,
            attempt=attempt,
            parent_run_id=prior[-1] if prior else None,
            question=str(spec["question"]),
            falsifier=str(spec["falsifier"]),
            inputs=tuple(str(item) for item in spec.get("inputs", ())),
            split=str(spec.get("split", "not applicable; this job reads no competition data")),
            seed=int(spec.get("seed", 0)),
            budget={str(k): int(v) for k, v in dict(spec["budget"]).items()},
            commit=context["commit"],
            dependency_lock=context["dependency_lock"],
            image=context["image"],
            taints=taints,
            write_scope=(FACTORY_ROOT / campaign / run_id).as_posix(),
            dispatched_utc=_now(),
        )
        digest = _write_json(run_dir / REQUEST_FILE, request.to_dict())
        dispatched.append({"run_id": run_id, "job": job, "attempt": attempt, "request_digest": digest})

    record_path = base / DISPATCH_FILE
    existing = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    runs = dict(existing.get("runs", {}))
    for item in dispatched:
        runs[str(item["run_id"])] = item
    payload = {
        "schema_version": 1,
        "campaign": campaign,
        "updated_utc": _now(),
        "runs": runs,
    }
    _write_json(record_path, payload)
    return {"campaign": campaign, "dispatched": dispatched, "runs_total": len(runs)}


def _recorded_request_digest(root: Path, campaign: str, run_id: str) -> str:
    record_path = campaign_root(root, campaign) / DISPATCH_FILE
    if not record_path.is_file():
        raise FactoryError(f"no {DISPATCH_FILE} for campaign {campaign}; nothing was dispatched")
    runs = json.loads(record_path.read_text(encoding="utf-8")).get("runs", {})
    if run_id not in runs:
        raise FactoryError(f"{run_id} is not a dispatched run of {campaign}")
    return str(runs[run_id]["request_digest"])


def record_result(root: Path, *, request_path: Path, draft: dict[str, Any]) -> dict[str, Any]:
    """Validate a worker's draft against its own request and write it once.

    Every refusal here is a property from the module docstring, checked rather
    than trusted: the request is the one that was dispatched, the status is one a
    worker may reach, the taints only grew, the budget held, and nothing is being
    overwritten.
    """
    if not request_path.is_file():
        raise FactoryError(f"no request at {request_path}")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    run_dir = request_path.parent
    campaign, run_id = str(request["campaign"]), str(request["run_id"])

    observed = digest_of(request_path)
    recorded = _recorded_request_digest(root, campaign, run_id)
    if observed != recorded:
        raise FactoryError(
            f"the request at {request_path} is {observed} and dispatch recorded {recorded}; "
            "a request is immutable and this one changed after it was issued"
        )

    result_path = run_dir / RESULT_FILE
    if result_path.exists():
        raise FactoryError(
            f"{run_id} already has a result; an attempt is never overwritten, so a further "
            "attempt at this job must be dispatched as a new run"
        )

    status = str(draft.get("status", ""))
    if status in CONTROLLER_ONLY_STATUSES:
        raise FactoryError(
            f"a worker may not declare {status!r}; standing is the controller's to record in the "
            f"registries. A worker reports one of {list(WORKER_STATUSES)}"
        )
    if status not in WORKER_STATUSES:
        raise FactoryError(f"status must be one of {list(WORKER_STATUSES)}, got {status!r}")

    taints = tuple(str(taint) for taint in draft.get("taints", ()))
    unknown = sorted(set(taints) - set(TAINTS))
    if unknown:
        raise FactoryError(f"unknown taints {unknown}; the vocabulary is {list(TAINTS)}")
    inherited = set(request.get("taints", ()))
    dropped = sorted(inherited - set(taints))
    if dropped:
        raise FactoryError(
            f"the result drops taints its request carried: {dropped}; taints are monotonic, so a "
            "result may add one and may never lose one"
        )

    budget = dict(request.get("budget", {}))
    used = {str(k): int(v) for k, v in dict(draft.get("budget_used", {})).items()}
    overruns = sorted(key for key, value in used.items() if key in budget and value > int(budget[key]))
    if overruns:
        raise FactoryError(
            f"the run exceeded its declared budget on {overruns}: used {used} against {budget}; "
            "a run that outgrew its envelope is reported as invalid, not recorded as if it had not"
        )

    for claim in draft.get("claims", []):
        if not str(claim.get("source", "")).strip():
            raise FactoryError(f"a claim with no source is not a claim: {claim}")

    body = {
        "schema_version": 1,
        "campaign": campaign,
        "job": str(request["job"]),
        "run_id": run_id,
        "attempt": int(request["attempt"]),
        "request_digest": recorded,
        "commit": str(request["commit"]),
        "dependency_lock": str(request["dependency_lock"]),
        "image": str(request["image"]),
        "seed": int(request["seed"]),
        "split": str(request["split"]),
        "status": status,
        "taints": sorted(set(taints)),
        "conclusion": str(draft.get("conclusion", "")),
        "answers_falsifier": bool(draft.get("answers_falsifier", False)),
        "claims": list(draft.get("claims", [])),
        "ledger_additions": list(draft.get("ledger_additions", [])),
        "budget_used": used,
        "notes": str(draft.get("notes", "")),
        "recorded_utc": _now(),
    }
    digest = _write_json(result_path, body)
    atomic_write_text(run_dir / RESULT_DIGEST_FILE, digest + "\n")
    return {"run_id": run_id, "result_digest": digest, "status": status, "taints": body["taints"]}


def _merge_ledger_additions(root: Path, additions: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold every worker's acquisitions into the canonical ledger, once, under the lock.

    Ids are allocated inside the lock, which is the whole point: the sequential
    allocation that is safe for one writer and lossy for two happens exactly here
    and nowhere else.
    """
    from biohubx.research.ledger import LedgerEntry, find_by_url, load_ledger, next_id, save_ledger

    added: list[str] = []
    already: list[str] = []
    with ledger_lock(root):
        entries = load_ledger(root)
        for addition in additions:
            url = str(addition.get("url", "")).strip()
            if not url:
                continue
            if find_by_url(entries, url) is not None:
                already.append(url)
                continue
            entry_id = next_id(entries)
            entries.append(
                LedgerEntry(
                    id=entry_id,
                    url=url,
                    kind=str(addition.get("kind", "other")),
                    campaign=str(addition.get("campaign", "")),
                    branch=str(addition.get("branch", "")),
                    note=str(addition.get("note", "")),
                    recorded_utc=_now(),
                    fetched=bool(addition.get("fetched", False)),
                    fetched_utc=addition.get("fetched_utc"),
                    http_status=addition.get("http_status"),
                    bytes=addition.get("bytes"),
                    raw_digest=addition.get("raw_digest"),
                    cache_path=addition.get("cache_path"),
                    payload_present=bool(addition.get("payload_present", False)),
                    status="reference_only",
                )
            )
            added.append(entry_id)
        if added:
            save_ledger(root, entries)
    return {"added": added, "already_present": sorted(set(already))}


def collect(root: Path, *, campaign: str) -> dict[str, Any]:
    """Verify every run, merge what the workers acquired, and compare what they concluded.

    Conflicting results are kept apart. Two runs of one job that reach different
    conclusions produce a comparison naming both, not an average and not a
    rewrite of the earlier one: a disagreement between two attempts is the
    finding, and resolving it is work, not arithmetic.
    """
    base = campaign_root(root, campaign)
    if not base.is_dir():
        raise FactoryError(f"no campaign directory at {FACTORY_ROOT / campaign}")

    runs: list[dict[str, Any]] = []
    pending: list[str] = []
    additions: list[dict[str, Any]] = []
    for run_dir in sorted(item for item in base.iterdir() if item.is_dir()):
        request_path = run_dir / REQUEST_FILE
        if not request_path.is_file():
            raise FactoryError(f"{run_dir.name} has no {REQUEST_FILE}")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        observed_request = digest_of(request_path)
        recorded_request = _recorded_request_digest(root, campaign, run_dir.name)
        if observed_request != recorded_request:
            raise FactoryError(
                f"{run_dir.name}'s request is {observed_request} and dispatch recorded "
                f"{recorded_request}; it was changed after it was issued"
            )
        result_path = run_dir / RESULT_FILE
        if not result_path.is_file():
            pending.append(run_dir.name)
            continue
        observed_result = digest_of(result_path)
        claimed = (run_dir / RESULT_DIGEST_FILE).read_text(encoding="utf-8").strip()
        if observed_result != claimed:
            raise FactoryError(
                f"{run_dir.name}'s result is {observed_result} and the worker recorded {claimed}; "
                "it was changed after it was written"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        additions.extend(result.get("ledger_additions", []))
        runs.append(
            {
                "run_id": run_dir.name,
                "job": str(request["job"]),
                "attempt": int(request["attempt"]),
                "question": str(request["question"]),
                "falsifier": str(request["falsifier"]),
                "request_digest": recorded_request,
                "result_digest": observed_result,
                "commit": str(result["commit"]),
                "status": str(result["status"]),
                "taints": list(result["taints"]),
                "conclusion": str(result["conclusion"]),
                "answers_falsifier": bool(result["answers_falsifier"]),
                "claims": list(result["claims"]),
                "budget_used": dict(result["budget_used"]),
            }
        )

    merged = _merge_ledger_additions(root, additions)
    campaign_state = state_from_registries(root)

    by_job: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_job.setdefault(str(run["job"]), []).append(run)
    comparisons = []
    for job, attempts in sorted(by_job.items()):
        conclusions = {str(item["conclusion"]) for item in attempts}
        statuses = {str(item["status"]) for item in attempts}
        comparisons.append(
            {
                "job": job,
                "attempts": [item["run_id"] for item in attempts],
                "agree": len(conclusions) <= 1 and len(statuses) <= 1,
                "conclusions": sorted(conclusions),
                "statuses": sorted(statuses),
                "resolution": (
                    "one attempt"
                    if len(attempts) == 1
                    else (
                        "attempts agree"
                        if len(conclusions) <= 1 and len(statuses) <= 1
                        else "attempts disagree; both stand, neither is merged into the other"
                    )
                ),
            }
        )

    collection = {
        "schema_version": 1,
        "campaign": campaign,
        "collected_utc": _now(),
        "runs": runs,
        "pending": pending,
        "comparisons": comparisons,
        "ledger": merged,
        # Monotonic across the campaign: once any run carries a taint, the
        # campaign carries it, whatever a later run forgets to mention.
        "campaign_taints": sorted({taint for run in runs for taint in run["taints"]}),
        # The controller's decision, from the registries and the probe report
        # alone. Workers propose; this is what spends GPU or unlocks a stage.
        "controller_state": campaign_state.to_dict(),
        "next_action": next_action(campaign_state).to_dict(),
        "standing": (
            "no run in this campaign carries standing; a worker reports "
            f"{list(WORKER_STATUSES)} and promotion is decided in the registries"
        ),
    }
    _write_json(base / COLLECTION_FILE, collection)
    _write_json(base / DOSSIER_FILE, dossier(collection))
    return collection


def dossier(collection: dict[str, Any]) -> dict[str, Any]:
    """The dossier as a view, derived here and never maintained by hand.

    AGENTS.md section 3 forbids a Markdown file per research source, and a
    hand-written dossier is the same failure one level up: a second place for a
    fact to be true. This is generated from ``COLLECTION.json`` every time it is
    asked for, so it cannot disagree with the records it summarises.
    """
    jobs: dict[str, dict[str, Any]] = {}
    for run in collection["runs"]:
        entry = jobs.setdefault(
            str(run["job"]),
            {
                "question": run["question"],
                "falsifier": run["falsifier"],
                "attempts": [],
                "claims": [],
                "taints": [],
            },
        )
        entry["attempts"].append(
            {
                "run_id": run["run_id"],
                "status": run["status"],
                "conclusion": run["conclusion"],
                "answers_falsifier": run["answers_falsifier"],
                "result_digest": run["result_digest"],
                "commit": run["commit"],
            }
        )
        entry["claims"].extend(run["claims"])
        entry["taints"] = sorted(set(entry["taints"]) | set(run["taints"]))
    comparisons = {str(item["job"]): item for item in collection["comparisons"]}
    for job, entry in jobs.items():
        entry["comparison"] = comparisons.get(job, {})
    return {
        "schema_version": 1,
        "generated_from": COLLECTION_FILE,
        "generated_utc": collection["collected_utc"],
        "campaign": collection["campaign"],
        "campaign_taints": collection["campaign_taints"],
        "standing": collection["standing"],
        "pending": collection["pending"],
        "next_action": collection.get("next_action"),
        "jobs": jobs,
    }
