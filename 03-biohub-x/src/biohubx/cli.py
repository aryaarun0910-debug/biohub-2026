"""The single typed entry point into Biohub-X.

There is no ``scripts/`` directory. Every operation the system can perform is a
subcommand here, and every subcommand obeys the same contract:

* it refuses missing or ambiguous inputs rather than guessing;
* it never falls back to a default data path;
* it prints a positive heartbeat so a silent run is distinguishable from a
  hung one;
* it writes its outputs atomically;
* it writes a manifest recording what it read, what it produced, and the typed
  digests of both.

Commands are added when their first real consumer exists. The commands the
system will eventually expose are listed in SYSTEM.md; only the ones below are
implemented, and an unimplemented command is absent rather than stubbed.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, cast

import typer
import yaml

from biohubx import __version__
from biohubx.artifacts import (
    ARTIFACT_REGISTRY_PATH,
    MANIFEST_DIR,
    ArtifactRecord,
    ArtifactRegistry,
    ProvenanceStatus,
    atomic_write_text,
    load_artifact_registry,
    strip_provisional_review_note,
    summarise_checks,
    verify_registry,
)
from biohubx.hashing import CANONICALIZATION_VERSION, DigestKind, NotTextError, digest_file
from biohubx.packaging.audit import AUDIT_KERNEL

app = typer.Typer(
    name="biohubx",
    help="Biohub-X: object-centric 3D+t cell tracking.",
    no_args_is_help=True,
    add_completion=False,
)
artifacts_app = typer.Typer(name="artifacts", help="Artifact identity and provenance.", no_args_is_help=True)
official_app = typer.Typer(name="official", help="Pinned official competition source.", no_args_is_help=True)
evaluate_app = typer.Typer(name="evaluate", help="Authoritative metric calibration.", no_args_is_help=True)
data_app = typer.Typer(name="data", help="Read-only dataset validation.", no_args_is_help=True)
infer_app = typer.Typer(name="infer", help="Emit a lineage graph.", no_args_is_help=True)
train_app = typer.Typer(name="train", help="Train a Biohub-X model.", no_args_is_help=True)
package_app = typer.Typer(name="package", help="Build a runnable package.", no_args_is_help=True)
research_app = typer.Typer(name="research", help="Ledger public research intake.", no_args_is_help=True)
model_app = typer.Typer(
    name="model", help="Measure a model architecture before it is trained.", no_args_is_help=True
)
submission_app = typer.Typer(
    name="submission", help="Write and validate the competition CSV; never submit.", no_args_is_help=True
)
app.add_typer(artifacts_app)
app.add_typer(official_app)
app.add_typer(evaluate_app)
app.add_typer(data_app)
app.add_typer(infer_app)
app.add_typer(train_app)
app.add_typer(package_app)
app.add_typer(research_app)
app.add_typer(model_app)
app.add_typer(submission_app)


def repository_root() -> Path:
    """The repository root, derived from this file rather than the process CWD.

    Deriving it from ``__file__`` means a command run from anywhere resolves the
    same repository, and there is no environment variable that can silently
    redirect it.
    """
    return Path(__file__).resolve().parents[2]


def heartbeat(command: str, stage: str, detail: str = "") -> None:
    """Emit one heartbeat line. Heartbeats go to stderr so stdout stays parseable."""
    elapsed = time.monotonic() - _STARTED
    line = f"[biohubx {elapsed:7.2f}s] {command} | {stage}"
    if detail:
        line = f"{line} | {detail}"
    print(line, file=sys.stderr, flush=True)


_STARTED = time.monotonic()


def _write_manifest(name: str, payload: dict[str, object]) -> Path:
    """Write a run manifest atomically to a stable, per-command path."""
    manifest = {
        "manifest_version": 1,
        "command": name,
        "written_utc": datetime.now(UTC).isoformat(),
        "canonicalization_version": CANONICALIZATION_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "biohubx_version": __version__,
        **payload,
    }
    path = repository_root() / MANIFEST_DIR / f"{name.replace(' ', '-')}.json"
    atomic_write_text(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Print version and exit."),
    ] = False,
) -> None:
    """Biohub-X command line."""


@artifacts_app.command("digest")
def artifacts_digest(
    path: Annotated[Path, typer.Argument(help="File to digest.")],
    kind: Annotated[
        DigestKind,
        typer.Option("--kind", help="Which identity to compute. There is no default."),
    ],
) -> None:
    """Print the typed digest token for a file.

    The identity kind is a required option: a digest whose kind was chosen for
    you is not an identity you can safely record.
    """
    command = "artifacts digest"
    heartbeat(command, "start", f"path={path} kind={kind.value}")
    if not path.is_file():
        heartbeat(command, "refused", f"not a file: {path}")
        raise typer.Exit(code=2)
    try:
        digest = digest_file(path, kind)
    except NotTextError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    heartbeat(command, "done", f"bytes={path.stat().st_size}")
    typer.echo(digest.token)


@artifacts_app.command("verify")
def artifacts_verify(
    registry: Annotated[
        Path | None,
        typer.Option("--registry", help="Artifact registry. Defaults to the repository registry."),
    ] = None,
    deep: Annotated[
        bool,
        typer.Option("--deep", help="Re-derive tree digests by reading every byte. Slow."),
    ] = False,
) -> None:
    """Check every recorded identity, at the depth each artifact warrants.

    Files inside the repository are always re-derived from their bytes. Dataset
    trees are outside the repository and take minutes to read, so by default
    they are checked for presence and shape and are reported as not deeply
    verified. ``--deep`` re-derives them.

    Exits non-zero if any recorded identity no longer holds. This is the command
    that makes a recorded hash an assertion rather than a note.
    """
    command = "artifacts verify"
    root = repository_root()
    registry_path = registry if registry is not None else root / ARTIFACT_REGISTRY_PATH
    heartbeat(command, "start", f"registry={registry_path}")

    try:
        loaded = load_artifact_registry(registry_path)
    except (FileNotFoundError, ValueError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    heartbeat(command, "loaded", f"artifacts={len(loaded.artifacts)} deep={deep}")
    checks = verify_registry(loaded, root, deep=deep)
    for check in checks:
        status = "ok" if check.ok else "FAIL"
        detail = f"id={check.artifact_id} slot={check.slot} depth={check.depth.value}"
        if check.detail:
            detail = f"{detail} {check.detail}"
        heartbeat(command, status, detail)

    failed = [check for check in checks if not check.ok]
    totals = summarise_checks(checks)
    # An in-repository registry is recorded relatively so the manifest stays
    # portable; an external one is recorded absolutely because that IS the
    # provenance. Manifests are therefore local evidence and are not committed.
    recorded_registry = (
        str(registry_path.relative_to(root)) if registry_path.is_relative_to(root) else str(registry_path)
    )
    manifest_path = _write_manifest(
        command,
        {
            "registry_path": recorded_registry,
            **totals,
            "checks": [check.model_dump() for check in checks],
        },
    )
    summary = (
        f"checked={totals['checked']} failed={totals['failed']} "
        f"deep={totals['deep']} shape_only={totals['shape_only']} absent={totals['absent']}"
    )
    heartbeat(command, "done", f"{summary} manifest={manifest_path.relative_to(root)}")
    if totals["shape_only"] and not deep:
        # Stated rather than left to be inferred from a count: a shape check is
        # not an identity check, and a reader must not take the pass for one.
        heartbeat(
            command,
            "note",
            f"{totals['shape_only']} tree artifact(s) were checked by shape only; "
            "re-run with --deep to re-derive their identities",
        )
    if failed:
        raise typer.Exit(code=1)


@artifacts_app.command("register")
def artifacts_register(
    source: Annotated[
        Path,
        typer.Option("--from", help="A data-fingerprint report to register."),
    ],
    registry: Annotated[
        Path | None,
        typer.Option("--registry", help="Artifact registry. Defaults to the repository registry."),
    ] = None,
) -> None:
    """Merge fingerprinted dataset identities into the artifact registry.

    Registration is a command rather than a hand edit so that no digest is ever
    retyped. An id already present with a different identity is a conflict and
    is refused: that means the data changed under a name the registry already
    claims, and only a person can decide whether that is a re-download or a
    problem.
    """
    command = "artifacts register"
    root = repository_root()
    registry_path = registry if registry is not None else root / ARTIFACT_REGISTRY_PATH
    heartbeat(command, "start", f"from={source} registry={registry_path}")

    if not source.is_file():
        heartbeat(command, "refused", f"no fingerprint report at {source}")
        raise typer.Exit(code=2)
    report = json.loads(source.read_text(encoding="utf-8"))
    if report.get("status") != "fingerprinted":
        heartbeat(
            command,
            "refused",
            f"report status is {report.get('status')!r}; only a completed fingerprint carries "
            "identities, and a plan carries none",
        )
        raise typer.Exit(code=2)

    try:
        loaded = load_artifact_registry(registry_path)
    except (FileNotFoundError, ValueError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    existing = {record.id: record for record in loaded.artifacts}
    added: list[str] = []
    unchanged: list[str] = []
    conflicts: list[str] = []
    records = list(loaded.artifacts)

    for entry in report["entries"]:
        record = ArtifactRecord.model_validate(
            {
                "id": entry["artifact_id"],
                "kind": entry["kind"],
                "external_path": entry["external_path"],
                "schema": entry["schema"],
                "digests": {"tree": entry["tree_digest"]},
                "shape": {
                    "file_count": entry["file_count"],
                    "empty_directory_count": entry["empty_directory_count"],
                    "total_bytes": entry["total_bytes"],
                },
                "provenance": entry["provenance"],
            }
        )
        previous = existing.get(record.id)
        if previous is None:
            records.append(record)
            added.append(record.id)
        elif previous.digests == record.digests and previous.shape == record.shape:
            unchanged.append(record.id)
        else:
            conflicts.append(
                f"{record.id}: recorded {previous.digests.get('tree')} now {record.digests['tree']}"
            )

    if conflicts:
        for conflict in conflicts:
            heartbeat(command, "FAIL", conflict)
        heartbeat(
            command,
            "refused",
            f"{len(conflicts)} artifact id(s) already record a different identity; nothing written",
        )
        raise typer.Exit(code=1)

    merged = ArtifactRegistry(schema_version=1, artifacts=records)
    document = {
        "schema_version": merged.schema_version,
        "artifacts": [
            record.model_dump(mode="json", by_alias=True, exclude_none=True) for record in merged.artifacts
        ],
    }
    atomic_write_text(registry_path, yaml.safe_dump(document, sort_keys=False, width=100))
    _write_manifest(
        command,
        {
            "source": str(source.relative_to(root)) if source.is_relative_to(root) else str(source),
            "added": added,
            "unchanged": unchanged,
            "registry_size": len(records),
        },
    )
    heartbeat(
        command,
        "done",
        f"added={len(added)} unchanged={len(unchanged)} registry_artifacts={len(records)}",
    )
    typer.echo(json.dumps({"added": added, "unchanged": unchanged}, sort_keys=True))


class Eligibility(StrEnum):
    """An explicit competition-eligibility decision. There is no default."""

    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not-eligible"


@artifacts_app.command("clear")
def artifacts_clear(
    reviewed_by: Annotated[
        str,
        typer.Option("--reviewed-by", help="Name of the person who read the terms. Required."),
    ],
    source_url: Annotated[str, typer.Option("--source-url", help="Where the terms were read.")],
    access_restrictions: Annotated[
        str,
        typer.Option("--access-restrictions", help="What the holder may not do, licence aside."),
    ],
    eligibility: Annotated[
        Eligibility,
        typer.Option("--eligibility", help="Explicit competition-eligibility decision."),
    ],
    id_prefix: Annotated[
        str,
        typer.Option("--id-prefix", help="Clear artifacts whose id starts with this."),
    ],
    data_license: Annotated[
        str | None, typer.Option("--data-license", help="Licence on the data itself.")
    ] = None,
    code_license: Annotated[
        str | None, typer.Option("--code-license", help="Licence on accompanying code.")
    ] = None,
    weight_license: Annotated[
        str | None, typer.Option("--weight-license", help="Licence on model weights.")
    ] = None,
    note: Annotated[str | None, typer.Option("--note", help="Anything else the review found.")] = None,
    registry: Annotated[
        Path | None,
        typer.Option("--registry", help="Artifact registry. Defaults to the repository registry."),
    ] = None,
) -> None:
    """Record a completed terms review against artifacts already registered.

    Clearance is a human act. This command writes down a review that has
    happened; it does not perform one, and it cannot judge eligibility. Every
    piece of evidence is a required option precisely so that none of it can be
    left to a default.

    The selected artifacts are verified before anything is written. A clearance
    says that someone read the terms covering THESE bytes, so applying one to an
    artifact whose recorded identity no longer holds would attach a real review
    to data it was never about.
    """
    command_name = "artifacts clear"
    root = repository_root()
    registry_path = registry if registry is not None else root / ARTIFACT_REGISTRY_PATH
    heartbeat(command_name, "start", f"id_prefix={id_prefix!r} reviewed_by={reviewed_by!r}")

    if not (data_license or code_license or weight_license):
        heartbeat(command_name, "refused", "a clearance needs at least one licence recorded")
        raise typer.Exit(code=2)

    try:
        loaded = load_artifact_registry(registry_path)
    except (FileNotFoundError, ValueError) as exc:
        heartbeat(command_name, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    selected = [record for record in loaded.artifacts if record.id.startswith(id_prefix)]
    if not selected:
        heartbeat(command_name, "refused", f"no artifact id starts with {id_prefix!r}")
        raise typer.Exit(code=2)
    heartbeat(command_name, "selected", f"artifacts={len(selected)} of {len(loaded.artifacts)}")

    # A review is about specific bytes. Verify before attaching it to them.
    subset = ArtifactRegistry(schema_version=1, artifacts=selected)
    checks = verify_registry(subset, root)
    failed = [check for check in checks if not check.ok]
    if failed:
        for check in failed[:10]:
            heartbeat(command_name, "FAIL", f"id={check.artifact_id} {check.detail}")
        heartbeat(
            command_name,
            "refused",
            f"{len(failed)} selected artifact(s) no longer match their recorded identity; "
            "a review cannot be attached to bytes it was not about",
        )
        raise typer.Exit(code=1)
    totals = summarise_checks(checks)
    heartbeat(
        command_name,
        "verified",
        f"deep={totals['deep']} shape_only={totals['shape_only']} absent={totals['absent']}",
    )

    cleared: list[str] = []
    records: list[ArtifactRecord] = []
    for record in loaded.artifacts:
        if not record.id.startswith(id_prefix):
            records.append(record)
            continue
        provenance = record.provenance.model_dump(exclude_none=True)
        # Records written before the note was fixed carry a sentence saying the
        # review has not happened. Leaving it on a cleared record would let a
        # reader believe the prose over the status.
        stripped = strip_provisional_review_note(provenance.get("note"))
        if stripped is None:
            provenance.pop("note", None)
        else:
            provenance["note"] = stripped
        provenance.update(
            {
                "status": ProvenanceStatus.EXTERNAL_CLEARED.value,
                "source_url": source_url,
                "access_restrictions": access_restrictions,
                "competition_eligible": eligibility is Eligibility.ELIGIBLE,
                "reviewed_by": reviewed_by,
                "reviewed_utc": datetime.now(UTC),
            }
        )
        for field, value in (
            ("data_license", data_license),
            ("code_license", code_license),
            ("weight_license", weight_license),
            ("note", note),
        ):
            if value:
                provenance[field] = value
        updated = record.model_dump(by_alias=True, exclude_none=True)
        updated["provenance"] = provenance
        records.append(ArtifactRecord.model_validate(updated))
        cleared.append(record.id)

    merged = ArtifactRegistry(schema_version=1, artifacts=records)
    document = {
        "schema_version": merged.schema_version,
        "artifacts": [
            record.model_dump(mode="json", by_alias=True, exclude_none=True) for record in merged.artifacts
        ],
    }
    atomic_write_text(registry_path, yaml.safe_dump(document, sort_keys=False, width=100))
    _write_manifest(
        command_name,
        {
            "id_prefix": id_prefix,
            "cleared": len(cleared),
            "reviewed_by": reviewed_by,
            "source_url": source_url,
            "access_restrictions": access_restrictions,
            "competition_eligible": eligibility is Eligibility.ELIGIBLE,
        },
    )
    heartbeat(
        command_name,
        "done",
        f"cleared={len(cleared)} eligibility={eligibility.value} reviewed_by={reviewed_by!r}",
    )
    typer.echo(json.dumps({"cleared": len(cleared), "eligibility": eligibility.value}, sort_keys=True))


@artifacts_app.command("revoke")
def artifacts_revoke(
    reason: Annotated[
        str,
        typer.Option("--reason", help="Why the review is being withdrawn. Recorded on each record."),
    ],
    id_prefix: Annotated[
        str,
        typer.Option("--id-prefix", help="Revoke artifacts whose id starts with this."),
    ],
    registry: Annotated[
        Path | None,
        typer.Option("--registry", help="Artifact registry. Defaults to the repository registry."),
    ] = None,
) -> None:
    """Withdraw a recorded terms review, returning artifacts to uncleared.

    Used when a clearance turns out to rest on something that was not
    established: an attestation from someone not in a position to give it, terms
    that were paraphrased rather than read, a reviewer who did not review.

    Returning to `external_uncleared` is the honest end state, not
    `not-eligible`. Recording not-eligible would assert that a review happened
    and reached a negative conclusion. Uncleared asserts only that no review
    stands, which is what is actually true after a withdrawal.

    The reason is written onto every affected record, so the registry shows that
    a clearance was attempted and withdrawn rather than looking as though none
    was ever made.
    """
    command_name = "artifacts revoke"
    root = repository_root()
    registry_path = registry if registry is not None else root / ARTIFACT_REGISTRY_PATH
    heartbeat(command_name, "start", f"id_prefix={id_prefix!r}")

    try:
        loaded = load_artifact_registry(registry_path)
    except (FileNotFoundError, ValueError) as exc:
        heartbeat(command_name, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    selected = [record for record in loaded.artifacts if record.id.startswith(id_prefix)]
    if not selected:
        heartbeat(command_name, "refused", f"no artifact id starts with {id_prefix!r}")
        raise typer.Exit(code=2)

    withdrawn: list[str] = []
    records: list[ArtifactRecord] = []
    for record in loaded.artifacts:
        if not record.id.startswith(id_prefix):
            records.append(record)
            continue
        provenance = record.provenance.model_dump(exclude_none=True)
        # Every field that was part of the review claim goes. What the artifact
        # IS survives untouched; only what anyone said about its terms is removed.
        for field in (
            "source_url",
            "data_license",
            "code_license",
            "weight_license",
            "access_restrictions",
            "competition_eligible",
            "reviewed_by",
            "reviewed_utc",
        ):
            provenance.pop(field, None)
        provenance["status"] = ProvenanceStatus.EXTERNAL_UNCLEARED.value
        existing = strip_provisional_review_note(provenance.get("note"))
        stamp = datetime.now(UTC).isoformat()
        withdrawal = f"Terms review withdrawn {stamp}: {reason}"
        provenance["note"] = f"{existing} {withdrawal}".strip() if existing else withdrawal

        updated = record.model_dump(by_alias=True, exclude_none=True)
        updated["provenance"] = provenance
        records.append(ArtifactRecord.model_validate(updated))
        withdrawn.append(record.id)

    merged = ArtifactRegistry(schema_version=1, artifacts=records)
    document = {
        "schema_version": merged.schema_version,
        "artifacts": [
            record.model_dump(mode="json", by_alias=True, exclude_none=True) for record in merged.artifacts
        ],
    }
    atomic_write_text(registry_path, yaml.safe_dump(document, sort_keys=False, width=100))
    _write_manifest(command_name, {"id_prefix": id_prefix, "withdrawn": len(withdrawn), "reason": reason})
    heartbeat(command_name, "done", f"withdrawn={len(withdrawn)} status=external_uncleared")
    typer.echo(json.dumps({"withdrawn": len(withdrawn), "reason": reason}, sort_keys=True))


@official_app.command("verify-source")
def official_verify_source() -> None:
    """Verify every typed identity in the official-source lock."""
    command = "official verify-source"
    root = repository_root()
    lock_path = root / "registry/official_source.yaml"
    heartbeat(command, "start", "lock=registry/official_source.yaml")
    if not lock_path.is_file():
        heartbeat(command, "refused", "official source lock is missing")
        raise typer.Exit(code=2)
    try:
        document = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        sources = document["sources"]
        commit = document["repository"]["commit"]
        if commit != "075fc5f5a52d11077f9dc2b074644618f26939e2":
            raise ValueError("official source commit differs from the adapter pin")
        checks: list[dict[str, object]] = []
        for source in sources:
            path = root / source["vendored_path"]
            for slot, token in source["digests"].items():
                expected_kind = DigestKind.RAW_ARTIFACT if slot == "raw" else DigestKind.CANONICAL_TEXT
                observed = digest_file(path, expected_kind).token
                checks.append(
                    {
                        "path": source["vendored_path"],
                        "slot": slot,
                        "expected": token,
                        "observed": observed,
                        "ok": observed == token,
                    }
                )
    except (FileNotFoundError, KeyError, TypeError, ValueError, NotTextError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    failed = [check for check in checks if not check["ok"]]
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "verified" if not failed else "failed",
        "source_commit": commit,
        "checked": len(checks),
        "failed": len(failed),
        "checks": checks,
    }
    report = root / "artifacts/official-source-verify.json"
    atomic_write_text(report, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_manifest(command, payload)
    heartbeat(command, "done", f"checked={len(checks)} failed={len(failed)}")
    typer.echo(json.dumps(payload, sort_keys=True))
    if failed:
        raise typer.Exit(code=1)


@evaluate_app.command("fixture")
def evaluate_fixture() -> None:
    """Run all deterministic Phase-1 fixtures through the official adapter."""
    command = "evaluate fixture"
    heartbeat(command, "start", "fixtures=15")
    from biohubx.evaluation.official_metric import evaluate_calibration_fixtures

    try:
        fixtures = evaluate_calibration_fixtures()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "calibrated",
        "fixtures": fixtures,
    }
    root = repository_root()
    report = root / "artifacts/evaluate-fixture.json"
    atomic_write_text(report, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_manifest(command, {"fixture_count": len(fixtures), "report": "artifacts/evaluate-fixture.json"})
    heartbeat(command, "done", f"fixtures={len(fixtures)} report=artifacts/evaluate-fixture.json")
    typer.echo(json.dumps(payload, sort_keys=True))


@data_app.command("validate")
def data_validate(
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Explicit data root. BIOHUB_DATA_ROOT is the only environment fallback."),
    ] = None,
    synthetic: Annotated[
        bool,
        typer.Option("--synthetic", help="Validate the dry in-memory fixture; never accesses real data."),
    ] = False,
) -> None:
    """Validate an explicit data root without copying or modifying it."""
    command = "data validate"
    from biohubx.data.validation import (
        synthetic_validation_report,
        validate_data_root,
        write_validation_report,
    )

    env_root = os.environ.get("BIOHUB_DATA_ROOT")
    if synthetic and (root is not None or env_root is not None):
        heartbeat(command, "refused", "synthetic mode cannot be combined with a real root")
        raise typer.Exit(code=2)
    if synthetic:
        heartbeat(command, "start", "mode=synthetic")
        report = synthetic_validation_report()
    else:
        resolved = root if root is not None else (Path(env_root) if env_root else None)
        if resolved is None:
            heartbeat(command, "refused", "provide --root or BIOHUB_DATA_ROOT")
            raise typer.Exit(code=2)
        heartbeat(command, "start", "mode=explicit-read-only")
        try:
            manifests = validate_data_root(resolved)
        except (FileNotFoundError, OSError, ValueError) as exc:
            heartbeat(command, "refused", str(exc))
            raise typer.Exit(code=2) from exc
        report = {
            "schema_version": 1,
            "mode": "real-read-only",
            "status": "valid",
            "datasets": [asdict(manifest) for manifest in manifests],
        }

    datasets = report["datasets"]
    assert isinstance(datasets, list)
    mode = report["mode"]

    output = repository_root() / "artifacts/data-validation.json"
    write_validation_report(output, report)
    _write_manifest(command, {"mode": mode, "dataset_count": len(datasets)})
    heartbeat(command, "done", f"datasets={len(datasets)} report=artifacts/data-validation.json")
    typer.echo(json.dumps(report, sort_keys=True))


HEARTBEAT_INTERVAL_SECONDS = 15.0
"""How often a long-running read reports progress."""


def _periodic_progress(command: str, label: str) -> Callable[[int, int], None]:
    """A progress hook that reports on elapsed time rather than file count.

    A Zarr array here is chunked one timepoint per file, so an artifact holds on
    the order of a hundred files. A count-based trigger would never fire during
    a multi-minute read, and the command would look hung, which is the one thing
    every command here promises not to do.
    """
    last = [time.monotonic()]

    def progress(files: int, total_bytes: int) -> None:
        now = time.monotonic()
        if now - last[0] >= HEARTBEAT_INTERVAL_SECONDS:
            last[0] = now
            heartbeat(command, "hashing", f"{label} files={files} bytes={total_bytes}")

    return progress


def _permitted_gpu_counts(raw: str) -> tuple[int, ...]:
    """Parse ``--allow-gpu-counts`` into the set the packaged guard will enforce.

    A malformed value must not silently widen the guard, so an unparseable
    entry, a negative count or an empty list raises rather than defaulting.
    Zero is refused too: an accelerator package that would accept no accelerator
    is the check standing next to the thing it is supposed to be sitting on.
    """
    try:
        counts = tuple(sorted({int(part) for part in raw.split(",") if part.strip()}))
    except ValueError as exc:
        raise ValueError(f"--allow-gpu-counts must be comma-separated integers, got {raw!r}") from exc
    if not counts or any(count < 1 for count in counts):
        raise ValueError(
            f"--allow-gpu-counts must name at least one positive count, got {raw!r}; "
            "a GPU package that accepts zero GPUs checks nothing"
        )
    return counts


SLICE_GRAPH_PATH = Path("artifacts/slice-graph.json")


@infer_app.command("synthetic")
def infer_synthetic(
    noise: Annotated[float, typer.Option("--noise", help="Fixture noise standard deviation.")] = 0.01,
    seed: Annotated[int, typer.Option("--seed", help="Fixture seed.")] = 0,
    annotated_fraction: Annotated[
        float,
        typer.Option("--annotated-fraction", help="Fraction of cells a sparse annotator labelled."),
    ] = 1.0,
) -> None:
    """Run the whole slice on the deterministic fixture and emit a legal graph.

    Writes the emitted graph and the stage-by-stage report atomically, and
    records the graph's raw digest so that `evaluate slice` scores exactly the
    bytes this command produced. No competition data, no GPU, no network.
    """
    command = "infer synthetic"
    from biohubx.evaluation.official_metric import UnscorablePredictionError
    from biohubx.tracking.pipeline import SliceConfig, run_slice

    heartbeat(command, "start", f"seed={seed} noise={noise} annotated_fraction={annotated_fraction}")
    try:
        result = run_slice(SliceConfig(seed=seed, noise=noise, annotated_fraction=annotated_fraction))
    except (ValueError, UnscorablePredictionError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    heartbeat(command, "detected", f"instances={len(result.instances.instances)}")
    reach = result.candidates.reach
    if reach is not None:
        heartbeat(command, "candidates", f"edges={len(result.candidates.edges)} reach={reach.reach:.3f}")
    heartbeat(
        command,
        "decoded",
        f"nodes={result.decode_report.nodes} edges={result.decode_report.edges} "
        f"divisions={result.decode_report.divisions} abstentions={result.decode_report.abstentions}",
    )

    root = repository_root()
    graph_path = root / SLICE_GRAPH_PATH
    atomic_write_text(
        graph_path,
        json.dumps(result.emitted.integer_export(), indent=2, sort_keys=True) + "\n",
    )
    report = result.report()
    report_path = root / "artifacts/slice-report.json"
    atomic_write_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")

    digest = digest_file(graph_path, DigestKind.RAW_ARTIFACT)
    _write_manifest(
        command,
        {
            "config": result.config.to_dict(),
            "emitted_graph": str(SLICE_GRAPH_PATH),
            "emitted_graph_digest": digest.token,
            "report": "artifacts/slice-report.json",
        },
    )
    heartbeat(command, "done", f"graph={SLICE_GRAPH_PATH} digest={digest.token[:38]}...")
    typer.echo(json.dumps(report, sort_keys=True))


@evaluate_app.command("slice")
def evaluate_slice(
    graph: Annotated[
        Path | None,
        typer.Option("--graph", help="Emitted graph to score. Defaults to the repository artifact."),
    ] = None,
) -> None:
    """Score a previously emitted slice graph against the fixture it came from.

    Refuses when the graph on disk does not match the configuration that
    produced it: scoring a stale artifact against a freshly generated ground
    truth would silently compare two different runs.
    """
    command = "evaluate slice"
    from biohubx.tracking.pipeline import run_slice

    root = repository_root()
    graph_path = graph if graph is not None else root / SLICE_GRAPH_PATH
    heartbeat(command, "start", f"graph={graph_path}")
    if not graph_path.is_file():
        heartbeat(
            command, "refused", f"no emitted graph at {graph_path}; run `biohubx infer synthetic` first"
        )
        raise typer.Exit(code=2)

    stored = json.loads(graph_path.read_text(encoding="utf-8"))
    result = run_slice()
    regenerated = result.emitted.integer_export()
    if stored != regenerated:
        heartbeat(
            command,
            "refused",
            "the stored graph does not match a fresh run of the default configuration; "
            "it came from different settings or a different revision",
        )
        raise typer.Exit(code=2)

    payload = {"schema_version": 1, "graph": str(SLICE_GRAPH_PATH), "official": result.score.to_dict()}
    atomic_write_text(
        root / "artifacts/slice-score.json", json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _write_manifest(command, payload)
    heartbeat(command, "done", f"score={result.score.score:.6f}")
    typer.echo(json.dumps(payload, sort_keys=True))


@infer_app.command("real")
def infer_real(
    dataset: Annotated[str, typer.Option("--dataset", help="Registered competition dataset id.")],
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Competition data root. BIOHUB_DATA_ROOT is the only fallback."),
    ] = None,
    split: Annotated[str, typer.Option("--split", help="Official split holding the dataset.")] = "train",
    first_frame: Annotated[int, typer.Option("--first-frame", help="First frame of the window.")] = 0,
    frames: Annotated[int, typer.Option("--frames", help="Number of frames in the window.")] = 12,
    crop_z: Annotated[str, typer.Option("--crop-z", help="Half-open voxel range start:stop.")] = "0:32",
    crop_y: Annotated[str, typer.Option("--crop-y", help="Half-open voxel range start:stop.")] = "128:256",
    crop_x: Annotated[str, typer.Option("--crop-x", help="Half-open voxel range start:stop.")] = "128:256",
    detection_threshold: Annotated[
        float, typer.Option("--detection-threshold", help="Normalised intensity threshold.")
    ] = 0.70,
) -> None:
    """Run the whole slice on one preregistered window of real competition data.

    CPU only, no GPU, no network and no download. The window is bounded
    explicitly rather than defaulted to a whole movie, because a preflight whose
    cost is unknown before it starts is not a preflight. Scores against the
    dataset's own annotated ground truth restricted to the same window.
    """
    command = "infer real"
    from biohubx.data.competition import (
        CompetitionLayoutError,
        WindowSelection,
        competition_root,
        load_window,
    )
    from biohubx.evaluation.official_metric import EstimatedTotalNodes, UnscorablePredictionError
    from biohubx.tracking.pipeline import SliceConfig, run_chain

    def span(text: str, name: str) -> tuple[int, int]:
        parts = text.split(":")
        if len(parts) != 2:
            heartbeat(command, "refused", f"{name} must be start:stop, got {text!r}")
            raise typer.Exit(code=2)
        return int(parts[0]), int(parts[1])

    z0, z1 = span(crop_z, "--crop-z")
    y0, y1 = span(crop_y, "--crop-y")
    x0, x1 = span(crop_x, "--crop-x")
    selection = WindowSelection(
        dataset_id=dataset,
        first_frame=first_frame,
        frames=frames,
        z_start=z0,
        z_stop=z1,
        y_start=y0,
        y_stop=y1,
        x_start=x0,
        x_stop=x1,
    )
    last = first_frame + frames - 1
    heartbeat(command, "start", f"dataset={dataset} split={split} frames={first_frame}..{last}")

    started = time.monotonic()
    try:
        window = load_window(competition_root(root), selection, split=split)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    heartbeat(
        command,
        "loaded",
        f"volume={tuple(window.volume.shape)} annotated_nodes={len(window.annotated.nodes)} "
        f"annotated_edges={len(window.annotated.edges)}",
    )

    settings = SliceConfig(detection_threshold=detection_threshold)
    try:
        outcome = run_chain(
            window.volume,
            dataset=window.annotated.dataset,
            annotated=window.annotated,
            scale=window.scale,
            estimate=EstimatedTotalNodes.declared(window.window_estimated_total_nodes),
            settings=settings,
        )
    except (ValueError, UnscorablePredictionError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    heartbeat(command, "detected", f"instances={len(outcome.instances.instances)}")
    reach = outcome.candidates.reach
    if reach is None:
        heartbeat(command, "refused", "the run measured no candidate reach")
        raise typer.Exit(code=2)
    heartbeat(command, "candidates", f"edges={len(outcome.candidates.edges)} reach={reach.reach:.3f}")
    heartbeat(
        command,
        "decoded",
        f"nodes={outcome.decode_report.nodes} edges={outcome.decode_report.edges} "
        f"divisions={outcome.decode_report.divisions}",
    )

    root_path = repository_root()
    graph_path = root_path / "artifacts/real-preflight-graph.json"
    atomic_write_text(
        graph_path, json.dumps(outcome.emitted.integer_export(), indent=2, sort_keys=True) + "\n"
    )
    payload = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "window": window.to_dict(),
        "config": settings.to_dict(),
        "detection": {"instances": len(outcome.instances.instances)},
        "candidates": {
            "radius_um": outcome.candidates.radius_um,
            "edges": len(outcome.candidates.edges),
            "reach": {
                "true_edges": reach.true_edges,
                "reachable_true_edges": reach.reachable_true_edges,
                "unreachable_missing_endpoint": reach.unreachable_missing_endpoint,
                "unreachable_outside_radius": reach.unreachable_outside_radius,
                "fraction": reach.reach,
            },
        },
        "decode": outcome.decode_report.to_dict(),
        "official": outcome.score.to_dict(),
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    report_path = root_path / "artifacts/real-preflight.json"
    atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_manifest(
        command,
        {
            "emitted_graph": "artifacts/real-preflight-graph.json",
            "emitted_graph_digest": digest_file(graph_path, DigestKind.RAW_ARTIFACT).token,
            "report": "artifacts/real-preflight.json",
        },
    )
    heartbeat(command, "done", f"score={outcome.score.score:.6f}")
    typer.echo(json.dumps(payload, sort_keys=True))


def _peak_rss_bytes() -> int | None:
    """Best-effort peak resident set size, or None where the platform will not say.

    Reported as evidence about cost, never as a measurement that gates anything,
    which is why an unavailable value is None rather than a zero that would read
    as a real number.
    """
    if sys.platform != "win32":
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    try:
        import ctypes
        from ctypes import wintypes

        # Signatures must be declared. Left to ctypes' defaults, GetCurrentProcess
        # returns a truncated 32-bit int on 64-bit Windows and the call fails
        # silently, which is how a real measurement becomes a quiet None.
        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_Counters),
            wintypes.DWORD,
        ]
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        if not kernel32.K32GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.PeakWorkingSetSize)
    except Exception:
        return None


@infer_app.command("reference")
def infer_reference(
    weights: Annotated[Path, typer.Option("--weights", help="Quarantined reference checkpoint.")],
    dataset: Annotated[
        str, typer.Option("--dataset", help="Registered competition dataset id.")
    ] = "6bba_2540cd90",
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Competition data root. BIOHUB_DATA_ROOT is the only fallback."),
    ] = None,
    split: Annotated[str, typer.Option("--split", help="Official split holding the dataset.")] = "train",
    first_frame: Annotated[int, typer.Option("--first-frame", help="First frame of the window.")] = 0,
    frames: Annotated[int, typer.Option("--frames", help="Number of frames in the window.")] = 12,
    crop_z: Annotated[str, typer.Option("--crop-z", help="Half-open voxel range start:stop.")] = "0:32",
    crop_y: Annotated[str, typer.Option("--crop-y", help="Half-open voxel range start:stop.")] = "128:256",
    crop_x: Annotated[str, typer.Option("--crop-x", help="Half-open voxel range start:stop.")] = "128:256",
    detection_threshold: Annotated[
        float,
        typer.Option("--detection-threshold", help="Sigmoid threshold on the detection logits."),
    ] = 0.965,
) -> None:
    """Load a quarantined reference checkpoint on CPU and count what it proposes.

    Two runs. First a synthetic forward pass, which checks the architecture
    produces finite output of the expected shape before any real byte is read.
    Then a no-gradient detection pass over the preregistered E01 window.

    integration_only. Both published checkpoints are quarantined: one trained on
    all 199 annotated movies and the other has no recorded training split, so
    nothing measured here may support a held-out finding (D-0020). CPU only, no
    training, no GPU, no corpus-wide inference.
    """
    command = "infer reference"
    import torch

    from biohubx.data.competition import (
        CompetitionLayoutError,
        WindowSelection,
        competition_root,
        load_window,
    )
    from biohubx.reference.architecture import ReferenceArchitectureError, load_reference_model

    def span(text: str, name: str) -> tuple[int, int]:
        parts = text.split(":")
        if len(parts) != 2:
            heartbeat(command, "refused", f"{name} must be start:stop, got {text!r}")
            raise typer.Exit(code=2)
        return int(parts[0]), int(parts[1])

    heartbeat(command, "start", f"weights={weights.name} threshold={detection_threshold}")
    torch.set_grad_enabled(False)
    try:
        model, spec = load_reference_model(weights)
    except ReferenceArchitectureError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    parameters = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    heartbeat(command, "loaded", f"strict=True params={parameters} buffers={buffers} spec={spec.to_dict()}")

    # Run 1: synthetic. Cheap, deterministic, and it fails before real data is read.
    synthetic_started = time.perf_counter()
    synthetic_in = torch.zeros((1, spec.window_size, 8, 16, 16), dtype=torch.float32)
    synthetic_features, synthetic_logits = model.detect(synthetic_in)
    synthetic_seconds = time.perf_counter() - synthetic_started
    synthetic_finite = bool(torch.isfinite(synthetic_logits).all())
    expected_logits = (1, spec.window_size, 1, 8, 16, 16)
    if tuple(synthetic_logits.shape) != expected_logits or not synthetic_finite:
        heartbeat(
            command,
            "refused",
            f"synthetic forward produced {tuple(synthetic_logits.shape)} finite={synthetic_finite}, "
            f"expected {expected_logits} finite=True",
        )
        raise typer.Exit(code=2)
    heartbeat(
        command,
        "synthetic",
        f"in={tuple(synthetic_in.shape)} features={tuple(synthetic_features.shape)} "
        f"logits={tuple(synthetic_logits.shape)} finite=True {synthetic_seconds:.3f}s",
    )

    z0, z1 = span(crop_z, "--crop-z")
    y0, y1 = span(crop_y, "--crop-y")
    x0, x1 = span(crop_x, "--crop-x")
    selection = WindowSelection(
        dataset_id=dataset,
        first_frame=first_frame,
        frames=frames,
        z_start=z0,
        z_stop=z1,
        y_start=y0,
        y_stop=y1,
        x_start=x0,
        x_stop=x1,
    )
    try:
        window = load_window(competition_root(root), selection, split=split)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    # The reference downsamples by striding, not pooling, so the smoke strides too.
    dz, dy, dx = spec.downsample
    strided = window.volume[:, ::dz, ::dy, ::dx]
    heartbeat(
        command,
        "loaded",
        f"volume={tuple(window.volume.shape)} strided={tuple(strided.shape)} "
        f"annotated_nodes={len(window.annotated.nodes)}",
    )

    real_started = time.perf_counter()
    proposals_per_frame: list[int] = []
    feature_shape: tuple[int, ...] = ()
    logit_shape: tuple[int, ...] = ()
    all_finite = True
    for index in range(0, strided.shape[0] - spec.window_size + 1, spec.window_size):
        chunk = torch.from_numpy(strided[index : index + spec.window_size]).unsqueeze(0)
        features, logits = model.detect(chunk)
        feature_shape = tuple(features.shape)
        logit_shape = tuple(logits.shape)
        all_finite &= bool(torch.isfinite(logits).all())
        above = (torch.sigmoid(logits) >= detection_threshold).sum(dim=(2, 3, 4, 5))
        proposals_per_frame.extend(int(v) for v in above[0])
    real_seconds = time.perf_counter() - real_started

    payload = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "quarantine": (
            "The checkpoint is reference_only. One published weight trained on all 199 annotated "
            "movies and the other records no training split, so no number here may support a "
            "held-out finding."
        ),
        "checkpoint": {
            "path_name": weights.name,
            "raw_digest": digest_file(weights, DigestKind.RAW_ARTIFACT).token,
            "strict_load": True,
            "parameters": parameters,
            "buffers": buffers,
            "spec": spec.to_dict(),
        },
        "synthetic_forward": {
            "input_shape": list(synthetic_in.shape),
            "feature_shape": list(synthetic_features.shape),
            "logit_shape": list(synthetic_logits.shape),
            "finite": synthetic_finite,
            "seconds": round(synthetic_seconds, 4),
        },
        "real_smoke": {
            "window": window.to_dict(),
            "volume_shape": list(window.volume.shape),
            "strided_shape": list(strided.shape),
            "feature_shape": list(feature_shape),
            "logit_shape": list(logit_shape),
            "finite": all_finite,
            "detection_threshold": detection_threshold,
            "voxels_per_frame": int(strided.shape[1] * strided.shape[2] * strided.shape[3]),
            "proposal_voxels_per_frame": proposals_per_frame,
            "proposal_voxels_total": sum(proposals_per_frame),
            "seconds": round(real_seconds, 3),
        },
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    report_path = repository_root() / "artifacts/reference-smoke.json"
    atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_manifest(command, {"report": "artifacts/reference-smoke.json", "checkpoint": weights.name})
    heartbeat(
        command,
        "smoke",
        f"features={feature_shape} logits={logit_shape} finite={all_finite} "
        f"above_threshold={sum(proposals_per_frame)} in {real_seconds:.2f}s",
    )
    heartbeat(command, "done", "report=artifacts/reference-smoke.json")
    typer.echo(json.dumps(payload, sort_keys=True))


@train_app.command("preflight")
def train_preflight(
    dataset: Annotated[
        str, typer.Option("--dataset", help="Registered competition dataset id.")
    ] = "6bba_2540cd90",
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Competition data root. BIOHUB_DATA_ROOT is the only fallback."),
    ] = None,
    split: Annotated[str, typer.Option("--split", help="Official split holding the dataset.")] = "train",
    first_frame: Annotated[int, typer.Option("--first-frame", help="First frame of the window.")] = 0,
    frames: Annotated[int, typer.Option("--frames", help="Number of frames in the window.")] = 12,
    crop_z: Annotated[str, typer.Option("--crop-z", help="Half-open voxel range start:stop.")] = "0:32",
    crop_y: Annotated[str, typer.Option("--crop-y", help="Half-open voxel range start:stop.")] = "128:256",
    crop_x: Annotated[str, typer.Option("--crop-x", help="Half-open voxel range start:stop.")] = "128:256",
    seed: Annotated[int, typer.Option("--seed", help="Deterministic initialisation seed.")] = 0,
    peak_quantile: Annotated[
        float,
        typer.Option(
            "--peak-quantile",
            help="Heatmap quantile used as the peak threshold; an untrained detector is uncalibrated.",
        ),
    ] = 0.999,
    learning_rate: Annotated[float, typer.Option("--learning-rate", help="Optimizer step size.")] = 1e-4,
) -> None:
    """Prove the E03 training loop runs end to end on CPU before any GPU is asked for.

    Ten stages, each of which has failed silently in someone's pipeline before:
    forward, masked loss, backward, optimizer step, atomic checkpoint, strict
    reload, bit-identical output after reload, peak extraction, suppression, and
    a legal proposal graph.

    The detector starts from deterministic random initialisation. No quarantined
    checkpoint initialises, supervises, selects or tunes it (D-0021, D-0022). The
    architecture is the vendored CC0 definition; the weights are Biohub-X's own
    and begin as noise.

    integration_only. CPU only, one tiny window, no fold, no promotion.
    """
    command = "train preflight"
    import numpy as np
    import torch

    from biohubx.data.competition import (
        CompetitionLayoutError,
        WindowSelection,
        competition_root,
        load_window,
    )
    from biohubx.evaluation.official_metric import UnscorablePredictionError
    from biohubx.proposals.peaks import HeatmapProposalError, instances_from_heatmap
    from biohubx.reference.architecture import REFERENCE_DOWNSAMPLE, ReferenceEdgeModel, ReferenceSpec
    from biohubx.tracking.pipeline import SliceConfig, run_chain
    from biohubx.training.targets import (
        PriorError,
        TargetConstructionError,
        class_prior_from_estimate,
        positive_mask,
        positive_unlabelled_loss,
    )

    def span(text: str, name: str) -> tuple[int, int]:
        parts = text.split(":")
        if len(parts) != 2:
            heartbeat(command, "refused", f"{name} must be start:stop, got {text!r}")
            raise typer.Exit(code=2)
        return int(parts[0]), int(parts[1])

    z0, z1 = span(crop_z, "--crop-z")
    y0, y1 = span(crop_y, "--crop-y")
    x0, x1 = span(crop_x, "--crop-x")
    selection = WindowSelection(
        dataset_id=dataset,
        first_frame=first_frame,
        frames=frames,
        z_start=z0,
        z_stop=z1,
        y_start=y0,
        y_stop=y1,
        x_start=x0,
        x_stop=x1,
    )
    heartbeat(command, "start", f"dataset={dataset} seed={seed} objective=positive-unlabelled")
    started = time.perf_counter()

    try:
        window = load_window(competition_root(root), selection, split=split)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    dz, dy, dx = REFERENCE_DOWNSAMPLE
    grid_np = window.volume[:, ::dz, ::dy, ::dx]
    grid_voxels = int(np.prod(grid_np.shape))
    try:
        positives, placed = positive_mask(
            tuple(grid_np.shape), window.annotated, downsample=REFERENCE_DOWNSAMPLE
        )
        prior = class_prior_from_estimate(window.window_estimated_total_nodes, grid_voxels)
    except (TargetConstructionError, PriorError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    heartbeat(
        command,
        "target",
        f"positive={placed} unlabelled={grid_voxels - placed} prior={prior:.3e} "
        f"unplaceable={len(window.annotated.nodes) - placed} "
        "(no voxel called background; F-0020 falsified the band)",
    )

    # Deterministic random initialisation. Nothing from a published checkpoint.
    torch.manual_seed(seed)
    spec = ReferenceSpec(
        unet_out_channels=32,
        unet_layers=(32, 64, 128),
        pos_feat_dim=32,
        window_size=2,
        downsample=REFERENCE_DOWNSAMPLE,
    )
    model = ReferenceEdgeModel(spec)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    grid = torch.from_numpy(grid_np).unsqueeze(0)

    # Stage 1-2: forward and positive-unlabelled risk.
    _, logits = model.detect(grid)
    flat = logits[0, :, 0]
    try:
        loss, risk_terms = positive_unlabelled_loss(flat, positives, prior=prior)
    except (TargetConstructionError, PriorError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    loss_before = float(loss.item())

    # Stage 3-4: backward and one optimizer step.
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore[no-untyped-call]
    squared = [(p.grad**2).sum() for p in model.parameters() if p.grad is not None]
    if not squared:
        heartbeat(command, "refused", "the backward pass produced no gradient at all")
        raise typer.Exit(code=2)
    grad_norm = float(torch.sqrt(torch.stack(squared).sum()))
    optimizer.step()
    with torch.no_grad():
        _, stepped = model.detect(grid)
        after, after_terms = positive_unlabelled_loss(stepped[0, :, 0], positives, prior=prior)
        loss_after = float(after.item())
    heartbeat(
        command,
        "step",
        f"loss {loss_before:.8f} -> {loss_after:.8f} grad_norm={grad_norm:.4e} "
        f"negative_risk={after_terms['negative_risk']:.6f} "
        f"clamped={bool(after_terms['negative_risk_clamped'])}",
    )

    # Stage 5: atomic checkpoint. A partially written checkpoint that still loads
    # is worse than one that does not exist.
    root_path = repository_root()
    checkpoint_path = root_path / "artifacts/preflight-detector.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    staged = checkpoint_path.with_suffix(".pt.partial")
    torch.save(model.state_dict(), staged)
    staged.replace(checkpoint_path)

    # Stage 6-7: strict reload, and identical output from the reloaded weights.
    model.eval()
    with torch.no_grad():
        _, reference_logits = model.detect(grid)
    reloaded = ReferenceEdgeModel(spec)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    reloaded.eval()
    with torch.no_grad():
        _, reloaded_logits = reloaded.detect(grid)
    identical = bool(torch.equal(reference_logits, reloaded_logits))
    if not identical:
        heartbeat(command, "refused", "the reloaded checkpoint did not reproduce its own output")
        raise typer.Exit(code=2)
    heartbeat(command, "checkpoint", f"strict reload OK, output identical, {checkpoint_path.name}")

    # Stage 8-9: peak extraction and suppression.
    with torch.no_grad():
        heatmap = torch.sigmoid(reloaded_logits)[0, :, 0].numpy()
    # One optimizer step from noise leaves the detector uncalibrated: it has
    # correctly learned that almost everything is negative, and no voxel reaches
    # any fixed probability. A quantile threshold exercises extraction and
    # suppression on their own terms. E03 uses an absolute threshold, once there
    # is a trained detector whose scale means something.
    peak_threshold = float(np.quantile(heatmap, peak_quantile))
    heatmap_max = float(heatmap.max())
    try:
        instances = instances_from_heatmap(
            heatmap,
            dataset=window.annotated.dataset,
            downsample=REFERENCE_DOWNSAMPLE,
            threshold=peak_threshold,
            scale=window.scale,
        )
    except HeatmapProposalError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    heartbeat(
        command,
        "proposals",
        f"instances={len(instances.instances)} threshold={peak_threshold:.6f} "
        f"(quantile {peak_quantile}) heatmap_max={heatmap_max:.6f}",
    )

    # Stage 10: a legal proposal graph, scored through the pinned official path.
    from biohubx.evaluation.official_metric import EstimatedTotalNodes

    try:
        outcome = run_chain(
            window.volume,
            dataset=window.annotated.dataset,
            annotated=window.annotated,
            scale=window.scale,
            estimate=EstimatedTotalNodes.declared(window.window_estimated_total_nodes),
            settings=SliceConfig(),
            instances=instances,
        )
    except (ValueError, UnscorablePredictionError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    reach = outcome.candidates.reach
    if reach is None:
        heartbeat(command, "refused", "the run measured no candidate reach")
        raise typer.Exit(code=2)
    heartbeat(
        command,
        "graph",
        f"nodes={outcome.decode_report.nodes} edges={outcome.decode_report.edges} "
        f"reach={reach.reach:.3f} node_recall={outcome.score.node_recall:.3f}",
    )

    payload = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "seed": seed,
        "initialisation": "deterministic random; no published checkpoint touched it",
        "window": window.to_dict(),
        "target": {
            "objective": "non-negative positive-unlabelled",
            "positive_voxels": placed,
            "unlabelled_voxels": int(np.prod(grid_np.shape)) - placed,
            "class_prior": prior,
            "prior_source": "the dataset's own estimated_number_of_nodes, prorated to the window",
            "unplaceable_nodes": len(window.annotated.nodes) - placed,
            "no_voxel_called_background": True,
            "why": "F-0020 falsified every intensity band on this corpus",
            "risk_terms_before": risk_terms,
            "risk_terms_after": after_terms,
        },
        "training_step": {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "gradient_norm": grad_norm,
            "learning_rate": learning_rate,
        },
        "checkpoint": {
            "path": "artifacts/preflight-detector.pt",
            "written_atomically": True,
            "strict_reload": True,
            "output_identical_after_reload": identical,
        },
        "proposals": {
            "peak_quantile": peak_quantile,
            "peak_threshold": peak_threshold,
            "heatmap_max_probability": heatmap_max,
            "instances": len(instances.instances),
            "note": (
                "An untrained detector has no calibrated probability scale, so the preflight "
                "thresholds by quantile. E03 uses an absolute threshold."
            ),
        },
        "graph": {
            "nodes": outcome.decode_report.nodes,
            "edges": outcome.decode_report.edges,
            "reach": reach.reach,
        },
        "official": outcome.score.to_dict(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    report_path = root_path / "artifacts/train-preflight.json"
    atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_manifest(command, {"report": "artifacts/train-preflight.json", "seed": seed})
    heartbeat(command, "done", f"score={outcome.score.score:.6f} report=artifacts/train-preflight.json")
    typer.echo(json.dumps(payload, sort_keys=True))


@evaluate_app.command("retention")
def evaluate_retention(
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Competition data root. BIOHUB_DATA_ROOT is the only fallback."),
    ] = None,
    retentions: Annotated[
        str,
        typer.Option("--retentions", help="Comma-separated edge-retention values to sweep."),
    ] = "1.0,0.98,0.96,0.94,0.92,0.90,0.88,0.85",
    seed: Annotated[int, typer.Option("--seed", help="Seed selecting which edges survive.")] = 0,
) -> None:
    """Measure what the node-count adjustment is worth against the edges it costs.

    Runs on the annotated training corpus only. The four duplicated public-test
    fixtures are never read, because each is byte-identical to a train volume
    (D-0015) and scoring against them would be scoring against training data.

    Every number comes from the pinned official scorer aggregated the official
    way, and each embryo fold is reported on its own.
    """
    command = "evaluate retention"
    from biohubx.data.competition import CompetitionLayoutError, competition_root, load_ground_truth
    from biohubx.evaluation.retention import crossover, sweep, verify_decoys_are_free

    try:
        data_root = competition_root(root)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    train_dir = data_root / "train"
    train = sorted(path.stem for path in train_dir.glob("*.geff"))
    if not train:
        heartbeat(command, "refused", f"no annotated training datasets under {train_dir}")
        raise typer.Exit(code=2)
    heartbeat(command, "start", f"datasets={len(train)} seed={seed}")

    started = time.monotonic()
    corpus = []
    for index, dataset_id in enumerate(train):
        corpus.append(load_ground_truth(data_root, dataset_id))
        if (index + 1) % 50 == 0:
            heartbeat(command, "loaded", f"{index + 1}/{len(train)}")
    heartbeat(command, "loaded", f"{len(corpus)}/{len(train)} ground-truth graphs")

    guard = verify_decoys_are_free(corpus[0])
    if not guard["counts_unchanged"]:
        heartbeat(command, "refused", "decoy nodes changed edge or division counts on real data")
        raise typer.Exit(code=2)
    heartbeat(command, "verified", f"decoys free on {guard['dataset']}")

    values = tuple(float(item) for item in retentions.split(","))
    points = sweep(corpus, retentions=values, seed=seed)
    heartbeat(command, "swept", f"points={len(points)}")

    embryos = sorted({point.embryo for point in points})
    crossings = [crossover(points, embryo) for embryo in embryos]
    payload = {
        "schema_version": 1,
        "provenance_status": "oracle_upper_bound",
        "datasets": len(corpus),
        "seed": seed,
        "retentions": list(values),
        "decoy_guard": guard,
        "points": [point.to_dict() for point in points],
        "crossover": crossings,
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    report_path = repository_root() / "artifacts/retention-oracle.json"
    atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_manifest(command, {"report": "artifacts/retention-oracle.json", "datasets": len(corpus)})
    for entry in crossings:
        heartbeat(
            command,
            "crossover",
            f"embryo={entry['embryo']} dense_adj={entry['dense_adjusted_edge_jaccard']:.4f} "
            f"breakeven={entry['lowest_retention_that_matches_dense']}",
        )
    heartbeat(command, "done", "report=artifacts/retention-oracle.json")
    typer.echo(json.dumps({"crossover": crossings}, sort_keys=True))


@evaluate_app.command("oracle-ceiling")
def evaluate_oracle_ceiling(
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Competition data root. BIOHUB_DATA_ROOT is the only fallback."),
    ] = None,
    frames: Annotated[int, typer.Option("--frames", help="Frames per movie window.")] = 10,
    radii: Annotated[str, typer.Option("--radii", help="Frozen DoG bank, um.")] = "2.0,3.0",
    local_maxima: Annotated[bool, typer.Option("--local-maxima", help="Frozen: strict peaks.")] = True,
    response_quantile: Annotated[float, typer.Option("--response-quantile")] = 0.95,
    suppression_radius: Annotated[float, typer.Option("--suppression-radius")] = 4.0,
    max_movies: Annotated[
        int, typer.Option("--max-movies", help="Cap movies per embryo. 0 uses every annotated movie.")
    ] = 0,
    per_scale_union: Annotated[
        bool,
        typer.Option("--per-scale-union", help="E05 source: strict peaks per scale, unioned (F-0034)."),
    ] = False,
    budget_ratios: Annotated[
        str | None,
        typer.Option(
            "--budget-ratios",
            help=(
                "E05 selection: comma-separated multiples of the window estimate to keep, by "
                "response rank. Every ratio is scored from one detection pass; omit for no truncation."
            ),
        ),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Report path. Defaults to artifacts/e06-oracle.json.")
    ] = None,
) -> None:
    """Score the association ceiling over frozen proposals, per embryo, officially.

    E06-ORACLE. Every emitted edge is a ground-truth edge whose endpoints both
    matched a proposal one-to-one within the official radius; every proposal is a
    node whether it matched or not. The result is what a perfect linker would
    score on these proposals. It is a diagnostic ceiling and can never be
    promoted, because it chooses edges with ground truth.

    The frozen configuration is passed explicitly and recorded in the report;
    the defaults here restate D-0041 and the E06 declaration, and a run that
    changes them is a different measurement. Reads a bounded window of each
    annotated training movie and never the public-test directory. Each embryo
    is reported on its own.
    """
    command = "evaluate oracle-ceiling"
    from biohubx.data.competition import (
        CompetitionLayoutError,
        WindowSelection,
        competition_root,
        load_ground_truth,
        load_window,
    )
    from biohubx.evaluation.official_metric import EstimatedTotalNodes, metric_row, summarise_fold
    from biohubx.evaluation.oracle import oracle_graph
    from biohubx.evaluation.proposals import truncate_to_budget
    from biohubx.proposals import dog

    try:
        data_root = competition_root(root)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    # Ratio keys are the strings the operator typed, so a report reads back the
    # way it was asked for; "none" is the untruncated arm and is always scored.
    ratios: list[tuple[str, float | None]] = [("none", None)]
    if budget_ratios:
        try:
            ratios += [(item.strip(), float(item)) for item in budget_ratios.split(",") if item.strip()]
        except ValueError as exc:
            heartbeat(command, "refused", f"--budget-ratios must be numbers: {budget_ratios!r}")
            raise typer.Exit(code=2) from exc
        if any(value is not None and value <= 0 for _, value in ratios):
            heartbeat(command, "refused", "a budget ratio must be positive")
            raise typer.Exit(code=2)

    train_dir = data_root / "train"
    movies = sorted(path.stem for path in train_dir.glob("*.geff"))
    if not movies:
        heartbeat(command, "refused", f"no annotated training datasets under {train_dir}")
        raise typer.Exit(code=2)
    bank = tuple(float(item) for item in radii.split(","))
    frozen = {
        "source": "biohubx.proposals.dog",
        "local_maxima_only": local_maxima,
        "radii_um": list(bank),
        "response_quantile": response_quantile,
        "suppression_radius_um": suppression_radius,
        "refine_centroids": False,
        "per_scale_union": per_scale_union,
        "truncation": "none" if budget_ratios is None else "by response rank to ratio x window estimate",
        "budget_ratios": [key for key, _ in ratios],
        "frames_per_movie": frames,
    }
    by_embryo: dict[str, list[str]] = {}
    for dataset_id in movies:
        by_embryo.setdefault(dataset_id.split("_", 1)[0], []).append(dataset_id)
    if max_movies:
        by_embryo = {embryo: ids[:max_movies] for embryo, ids in by_embryo.items()}
    heartbeat(
        command,
        "start",
        " ".join(f"{embryo}={len(ids)}" for embryo, ids in sorted(by_embryo.items()))
        + f" frozen={json.dumps(frozen, sort_keys=True)}",
    )

    started = time.monotonic()
    per_embryo_rows: dict[str, list[dict[str, Any]]] = {}
    accounting: dict[str, dict[str, float]] = {}
    unscored: dict[str, list[dict[str, object]]] = {}

    def slot(ratio_key: str, embryo_id: str) -> str:
        # The report keeps the E06-ORACLE shape when only the untruncated arm
        # runs, so F-0033 and an E05 A0 read identically.
        return embryo_id if len(ratios) == 1 else f"{embryo_id}@{ratio_key}"

    for embryo, dataset_ids in sorted(by_embryo.items()):
        for index, dataset_id in enumerate(dataset_ids):
            truth = load_ground_truth(data_root, dataset_id)
            annotated_frames = sorted({node.frame for node in truth.lineage.nodes})
            if not annotated_frames:
                continue
            first = min(annotated_frames[0], max(0, truth.frames - frames))
            depth, height, width = _open_volume(data_root / "train" / f"{dataset_id}.zarr").shape[1:]
            try:
                window = load_window(
                    data_root,
                    WindowSelection(dataset_id, first, frames, 0, depth, 0, height, 0, width),
                    split="train",
                )
            except ValueError as exc:
                heartbeat(command, "skip", f"{dataset_id}: {exc}")
                continue
            if not window.annotated.nodes:
                continue
            detected = dog.detect_instances(
                window.volume,
                dataset=window.annotated.dataset,
                radii_um=bank,
                response_quantile=response_quantile,
                suppression_radius_um=suppression_radius,
                local_maxima_only=local_maxima,
                per_scale_union=per_scale_union,
            )
            for ratio_key, ratio in ratios:
                instances = (
                    detected
                    if ratio is None
                    else truncate_to_budget(
                        detected, round(ratio * float(window.window_estimated_total_nodes))
                    )
                )
                key = slot(ratio_key, embryo)
                ceiling = oracle_graph(instances, window.annotated)
                bucket = accounting.setdefault(
                    key,
                    {
                        "movies": 0,
                        "proposals": 0,
                        "estimated_nodes": 0.0,
                        "annotated_nodes": 0,
                        "matched_nodes": 0,
                        "annotated_edges": 0,
                        "retained_edges": 0,
                        "retained_divisions": 0,
                    },
                )
                bucket["movies"] += 1
                bucket["proposals"] += ceiling.proposals
                bucket["estimated_nodes"] += float(window.window_estimated_total_nodes)
                bucket["annotated_nodes"] += ceiling.annotated_nodes
                bucket["matched_nodes"] += ceiling.matched_nodes
                bucket["annotated_edges"] += ceiling.annotated_edges
                bucket["retained_edges"] += ceiling.retained_edges
                bucket["retained_divisions"] += ceiling.retained_divisions
                if ceiling.graph is None:
                    unscored.setdefault(key, []).append(
                        {
                            "dataset": dataset_id,
                            "annotated_edges": ceiling.annotated_edges,
                            "matched_nodes": ceiling.matched_nodes,
                        }
                    )
                    continue
                row = metric_row(
                    ceiling.graph,
                    window.annotated,
                    estimated_total_nodes=EstimatedTotalNodes.declared(
                        float(window.window_estimated_total_nodes)
                    ),
                )
                per_embryo_rows.setdefault(key, []).append(row)
            if (index + 1) % 25 == 0:
                heartbeat(command, "scored", f"{embryo} {index + 1}/{len(dataset_ids)}")
        heartbeat(command, "embryo", f"{embryo} done")

    summaries: dict[str, dict[str, object]] = {}
    for embryo, rows in sorted(per_embryo_rows.items()):
        fold = summarise_fold(rows)
        acc = accounting[embryo]
        summaries[embryo] = {
            **fold.to_dict(),
            "windows_scored": len(rows),
            "windows_unscored_no_retained_edge": len(unscored.get(embryo, [])),
            "annotated_edges_in_unscored_windows": sum(
                int(str(u["annotated_edges"])) for u in unscored.get(embryo, [])
            ),
            "proposals": int(acc["proposals"]),
            "estimated_nodes": round(acc["estimated_nodes"], 1),
            "node_ratio": round((acc["proposals"] - acc["estimated_nodes"]) / acc["estimated_nodes"], 4),
            "annotated_nodes": int(acc["annotated_nodes"]),
            "matched_nodes": int(acc["matched_nodes"]),
            "node_match_fraction": round(acc["matched_nodes"] / acc["annotated_nodes"], 4)
            if acc["annotated_nodes"]
            else 0.0,
            "annotated_edges": int(acc["annotated_edges"]),
            "retained_edges": int(acc["retained_edges"]),
            "edge_retention": round(acc["retained_edges"] / acc["annotated_edges"], 4)
            if acc["annotated_edges"]
            else 0.0,
            "retained_divisions": int(acc["retained_divisions"]),
            "supports_0_950": bool(fold.score >= 0.950),
        }
        heartbeat(
            command,
            "ceiling",
            f"{embryo}: score={fold.score:.4f} adj_edge={fold.adjusted_edge_jaccard:.4f} "
            f"raw_edge={fold.edge_jaccard:.4f} node_ratio={summaries[embryo]['node_ratio']} "
            f"retention={summaries[embryo]['edge_retention']} "
            f"unscored={summaries[embryo]['windows_unscored_no_retained_edge']}",
        )

    payload = {
        "schema_version": 1,
        "provenance_status": "diagnostic_ceiling",
        "experiment": "E06",
        "measurement": "E06-ORACLE",
        "frozen_proposals": frozen,
        "per_embryo": summaries,
        "unscored_windows": unscored,
        "both_support_0_950": bool(summaries)
        and all(
            bool(s["supports_0_950"]) for k, s in summaries.items() if "@" not in k or k.endswith("@none")
        ),
        "slots": "embryo, or embryo@ratio when --budget-ratios is given; @none is the untruncated arm",
        "runtime_seconds": round(time.monotonic() - started, 3),
        "limits": (
            "A ceiling, never a method: edges are chosen with ground truth. Scores are per window and "
            "aggregated the official way per embryo; they are not comparable with whole-movie scores. "
            "A window with no retained edge cannot be scored by the pinned scorer (F-0009) and is counted "
            "separately with its annotated edges, which a real linker would also have missed."
        ),
    }
    report_path = out if out is not None else repository_root() / "artifacts/e06-oracle.json"
    atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True) + chr(10))
    _write_manifest(command, {"report": str(report_path), "experiment": "E06-ORACLE"})
    heartbeat(command, "done", f"both embryos support 0.950: {payload['both_support_0_950']}")
    typer.echo(json.dumps({k: v for k, v in payload.items() if k != "unscored_windows"}, sort_keys=True))


@evaluate_app.command("miss-atlas")
def evaluate_miss_atlas(
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Competition data root. BIOHUB_DATA_ROOT is the only fallback."),
    ] = None,
    embryo: Annotated[str, typer.Option("--embryo", help="Embryo prefix whose misses are audited.")] = "6bba",
    frames: Annotated[int, typer.Option("--frames", help="Frames per movie window, as E06-ORACLE.")] = 10,
    radii: Annotated[str, typer.Option("--radii", help="DoG bank, um.")] = "2.0,3.0",
    response_quantile: Annotated[float, typer.Option("--response-quantile")] = 0.95,
    suppression_radius: Annotated[float, typer.Option("--suppression-radius")] = 4.0,
    per_scale_union: Annotated[
        bool,
        typer.Option("--per-scale-union", help="The A2 source of F-0035: strict peaks per scale, unioned."),
    ] = True,
    local_maxima: Annotated[
        bool, typer.Option("--local-maxima", help="Strict peaks on the fused response.")
    ] = True,
    max_movies: Annotated[
        int, typer.Option("--max-movies", help="Cap movies. 0 uses every annotated movie.")
    ] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Report path. Defaults to artifacts/miss-atlas-<embryo>.json."),
    ] = None,
) -> None:
    """Describe and classify every annotated cell the proposal source failed to reach.

    Same windows as E06-ORACLE, so the misses are the ones F-0033 and F-0035
    counted. For each: physical position, intensity percentile, per-scale DoG
    response, distance and vector to the nearest proposal, boundary distance,
    neighbour distances and density, whether the nearest proposal already
    serves another cell, presence and matching in the neighbouring frames, and
    whether a strict local maximum the quantile discarded lies within the
    official radius. Then one ordered classification. The atlas motivates arms;
    it is not a finding.
    """
    command = "evaluate miss-atlas"
    from biohubx.data.competition import (
        CompetitionLayoutError,
        WindowSelection,
        competition_root,
        load_ground_truth,
        load_window,
    )
    from biohubx.evaluation.miss_atlas import CLASSES, Thresholds, audit_window, representative, summarise
    from biohubx.proposals import dog

    try:
        data_root = competition_root(root)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    movies = sorted(path.stem for path in (data_root / "train").glob(f"{embryo}*.geff"))
    if not movies:
        heartbeat(command, "refused", f"no annotated {embryo} movie under {data_root / 'train'}")
        raise typer.Exit(code=2)
    if max_movies:
        movies = movies[:max_movies]
    bank = tuple(float(item) for item in radii.split(","))
    thresholds = Thresholds()
    source = {
        "source": "biohubx.proposals.dog",
        "local_maxima_only": local_maxima,
        "per_scale_union": per_scale_union,
        "radii_um": list(bank),
        "response_quantile": response_quantile,
        "suppression_radius_um": suppression_radius,
        "refine_centroids": False,
        "truncation": "none",
        "frames_per_movie": frames,
    }
    heartbeat(command, "start", f"{embryo}={len(movies)} source={json.dumps(source, sort_keys=True)}")

    started = time.monotonic()
    records = []
    annotated_total = 0
    per_movie: list[dict[str, object]] = []
    for index, dataset_id in enumerate(movies):
        truth = load_ground_truth(data_root, dataset_id)
        annotated_frames = sorted({node.frame for node in truth.lineage.nodes})
        if not annotated_frames:
            continue
        first = min(annotated_frames[0], max(0, truth.frames - frames))
        depth, height, width = _open_volume(data_root / "train" / f"{dataset_id}.zarr").shape[1:]
        try:
            window = load_window(
                data_root,
                WindowSelection(dataset_id, first, frames, 0, depth, 0, height, 0, width),
                split="train",
            )
        except ValueError as exc:
            heartbeat(command, "skip", f"{dataset_id}: {exc}")
            continue
        if not window.annotated.nodes:
            continue
        instances = dog.detect_instances(
            window.volume,
            dataset=window.annotated.dataset,
            radii_um=bank,
            response_quantile=response_quantile,
            suppression_radius_um=suppression_radius,
            local_maxima_only=local_maxima,
            per_scale_union=per_scale_union,
        )
        found = audit_window(
            dataset_id=dataset_id,
            embryo=embryo,
            volume=window.volume,
            annotated=window.annotated,
            instances=instances,
            first_frame=first,
            radii_um=bank,
            response_quantile=response_quantile,
            thresholds=thresholds,
        )
        nodes_in_window = sum(1 for n in window.annotated.nodes if first <= n.frame < first + frames)
        annotated_total += nodes_in_window
        records.extend(found)
        per_movie.append(
            {
                "dataset": dataset_id,
                "first_frame": first,
                "annotated_nodes": nodes_in_window,
                "proposals": len(instances.instances),
                "misses": len(found),
                "classes": {
                    c: sum(1 for r in found if r.primary_class == c)
                    for c in CLASSES
                    if any(r.primary_class == c for r in found)
                },
            }
        )
        if (index + 1) % 16 == 0:
            heartbeat(command, "audited", f"{index + 1}/{len(movies)} misses={len(records)}")

    summary = summarise(records, annotated_total)
    payload = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "role": "miss atlas; motivates arms, is not a finding until a preregistered probe reproduces it",
        "embryo": embryo,
        "source": source,
        "thresholds": thresholds.to_dict(),
        "class_order": list(CLASSES),
        "summary": summary,
        "per_movie": per_movie,
        "representatives": representative(records),
        "records": [r.to_dict() for r in records],
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    report_path = out if out is not None else repository_root() / f"artifacts/miss-atlas-{embryo}.json"
    atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True, default=str) + chr(10))
    _write_manifest(command, {"report": str(report_path), "embryo": embryo})
    heartbeat(command, "summary", json.dumps(summary["primary_class_counts"], sort_keys=True))
    heartbeat(
        command,
        "done",
        f"misses={summary['misses']} of {annotated_total} annotated ({summary['miss_fraction']}) "
        f"report={report_path}",
    )
    typer.echo(json.dumps({"report": str(report_path), "summary": summary}, sort_keys=True))


@evaluate_app.command("proposals")
def evaluate_proposals(
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Competition data root. BIOHUB_DATA_ROOT is the only fallback."),
    ] = None,
    frames: Annotated[int, typer.Option("--frames", help="Annotated frames per movie.")] = 2,
    ratios: Annotated[
        str, typer.Option("--ratios", help="Comma-separated node budgets, as multiples of the estimate.")
    ] = "0.5,1.0,2.0,3.0,4.0,6.0,8.0",
    max_movies: Annotated[
        int, typer.Option("--max-movies", help="Cap movies per embryo. 0 uses every annotated movie.")
    ] = 0,
    audit_misses: Annotated[
        bool,
        typer.Option("--audit-misses", help="Describe every annotated node DoG reached or missed."),
    ] = False,
    audit_ratio: Annotated[
        float, typer.Option("--audit-ratio", help="Budget ratio at which misses are audited.")
    ] = 1.0,
    refine: Annotated[
        bool, typer.Option("--refine", help="Sub-voxel centroid refinement of DoG peaks (H-11).")
    ] = False,
    suppression_radius: Annotated[
        float, typer.Option("--suppression-radius", help="Physical suppression radius in um.")
    ] = 4.0,
    radii: Annotated[
        str,
        typer.Option("--radii", help="Comma-separated nucleus radii in um for the DoG bank (H-11b)."),
    ] = "2.0,3.0,4.5",
    local_maxima: Annotated[
        bool,
        typer.Option("--local-maxima", help="Only true 3D local maxima of the response are candidates."),
    ] = False,
    per_scale_union: Annotated[
        bool,
        typer.Option(
            "--per-scale-union", help="H-11c: strict peaks at each scale, unioned, then suppressed."
        ),
    ] = False,
    out: Annotated[
        Path | None, typer.Option("--out", help="Report path. Defaults to artifacts/proposals.json.")
    ] = None,
) -> None:
    """Compare proposal sources at a matched node budget, per embryo.

    The E04 probe. Every source is truncated to the same budget, taken from each
    movie's own official estimated_number_of_nodes, so the comparison measures
    detector quality rather than how much each source was allowed to emit.

    Reads a bounded window of each annotated training movie and never the
    public-test directory. Reports each embryo separately, because a pooled number
    over two embryos differing twelvefold in annotation density would describe
    neither.
    """
    command = "evaluate proposals"
    import numpy as np

    from biohubx.data.competition import (
        CompetitionLayoutError,
        WindowSelection,
        competition_root,
        load_ground_truth,
        load_window,
    )
    from biohubx.evaluation.proposals import (
        measure_reachability,
        normalise_for_classical,
        truncate_to_budget,
    )
    from biohubx.evaluation.residuals import NodeAudit, audit_nodes, one_to_one_recall, summarise
    from biohubx.proposals import classical, dog

    try:
        data_root = competition_root(root)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    train_dir = data_root / "train"
    movies = sorted(path.stem for path in train_dir.glob("*.geff"))
    if not movies:
        heartbeat(command, "refused", f"no annotated training datasets under {train_dir}")
        raise typer.Exit(code=2)
    budgets = tuple(float(item) for item in ratios.split(","))
    if any(value <= 0 for value in budgets):
        heartbeat(command, "refused", f"every budget ratio must be positive, got {budgets}")
        raise typer.Exit(code=2)
    bank = tuple(float(item) for item in radii.split(","))
    if not bank or any(value <= 0 for value in bank):
        heartbeat(command, "refused", f"the scale bank must be non-empty positive radii, got {bank}")
        raise typer.Exit(code=2)

    by_embryo: dict[str, list[str]] = {}
    for dataset_id in movies:
        by_embryo.setdefault(dataset_id.split("_", 1)[0], []).append(dataset_id)
    if max_movies:
        by_embryo = {embryo: ids[:max_movies] for embryo, ids in by_embryo.items()}
    heartbeat(
        command,
        "start",
        " ".join(f"{embryo}={len(ids)}" for embryo, ids in sorted(by_embryo.items()))
        + f" frames={frames} ratios={list(budgets)}",
    )

    started = time.monotonic()
    # embryo -> source -> ratio -> [reached, annotated, proposals, budget, matched]
    totals: dict[str, dict[str, dict[float, list[float]]]] = {}
    audits: list[dict[str, object]] = []
    skipped = 0
    for embryo, dataset_ids in sorted(by_embryo.items()):
        for index, dataset_id in enumerate(dataset_ids):
            truth = load_ground_truth(data_root, dataset_id)
            annotated_frames = sorted({node.frame for node in truth.lineage.nodes})
            if not annotated_frames:
                skipped += 1
                continue
            first = min(annotated_frames[0], max(0, truth.frames - frames))
            depth, height, width = _open_volume(data_root / "train" / f"{dataset_id}.zarr").shape[1:]
            try:
                window = load_window(
                    data_root,
                    WindowSelection(dataset_id, first, frames, 0, depth, 0, height, 0, width),
                    split="train",
                )
            except ValueError as exc:
                heartbeat(command, "skip", f"{dataset_id}: {exc}")
                skipped += 1
                continue
            if not window.annotated.nodes:
                skipped += 1
                continue

            dataset = window.annotated.dataset
            estimate = float(window.window_estimated_total_nodes)
            # Detected once per source at a permissive setting, then truncated to
            # each budget. Re-detecting per budget would multiply the cost and
            # could not change which proposals rank highest.
            # Both sources are cut at a quantile of their own response, not at an
            # absolute number. An absolute intensity threshold admits millions of
            # candidates on a full-extent frame and means something different on
            # every movie; a quantile bounds the candidate pool symmetrically and
            # leaves the budget, not the threshold, as the thing being matched.
            normalised = normalise_for_classical(window.volume)
            intensity_cut = float(np.clip(np.quantile(normalised, 0.997), 1e-6, 1.0 - 1e-6))
            # A permissive quantile, so the largest budget in the sweep can still
            # be filled where the detector has that many peaks to offer.
            proposal_sets = {
                "classical": classical.detect_instances(normalised, dataset=dataset, threshold=intensity_cut),
                "dog": dog.detect_instances(
                    window.volume,
                    dataset=dataset,
                    radii_um=bank,
                    response_quantile=0.95,
                    suppression_radius_um=suppression_radius,
                    refine_centroids=refine,
                    local_maxima_only=local_maxima,
                    per_scale_union=per_scale_union,
                ),
            }
            for name, instances in proposal_sets.items():
                for ratio in budgets:
                    budget = max(1, round(ratio * estimate))
                    kept = truncate_to_budget(instances, budget)
                    reach = measure_reachability(kept, window.annotated, estimated_total_nodes=estimate)
                    matched, _ = one_to_one_recall(kept, window.annotated)
                    bucket = (
                        totals.setdefault(embryo, {})
                        .setdefault(name, {})
                        .setdefault(ratio, [0.0, 0.0, 0.0, 0.0, 0.0])
                    )
                    bucket[0] += reach.reached
                    bucket[1] += reach.annotated_nodes
                    bucket[2] += reach.proposals
                    bucket[3] += budget
                    bucket[4] += matched
                    if audit_misses and name == "dog" and ratio == audit_ratio:
                        stride = dog.isotropic_plane_stride()
                        responses = {
                            local: dog.dog_response(
                                dog.normalise_frame(window.volume[local][:, ::stride, ::stride]),
                                radii_um=bank,
                                voxel_um=1.625,
                            )
                            for local in range(window.volume.shape[0])
                        }
                        audits.extend(
                            audit.to_dict()
                            for audit in audit_nodes(
                                dataset_id=dataset_id,
                                embryo=embryo,
                                volume=window.volume,
                                instances=kept,
                                annotated=window.annotated,
                                response_per_frame=responses,
                                plane_stride=stride,
                                first_frame=first,
                            )
                        )
            if (index + 1) % 25 == 0:
                heartbeat(command, "measured", f"{embryo} {index + 1}/{len(dataset_ids)}")
        heartbeat(command, "embryo", f"{embryo} done")

    rows: list[dict[str, object]] = []
    for embryo, sources in sorted(totals.items()):
        for name, per_ratio in sorted(sources.items()):
            for ratio, (reached, annotated, proposals, allowed, matched_total) in sorted(per_ratio.items()):
                rows.append(
                    {
                        "embryo": embryo,
                        "source": name,
                        "budget_ratio": ratio,
                        "reachability": round(reached / annotated, 6) if annotated else 0.0,
                        "one_to_one_node_recall": round(matched_total / annotated, 6) if annotated else 0.0,
                        "annotated_nodes": int(annotated),
                        "reached": int(reached),
                        "proposals": int(proposals),
                        "budget_allowed": int(allowed),
                        # When a source cannot fill its budget the comparison is
                        # at its natural operating point, not at a matched one,
                        # and saying so is the difference between a measurement
                        # and a misreading.
                        "budget_filled": bool(allowed) and proposals >= allowed * 0.99,
                    }
                )

    # Selection happens on one embryo and is judged on the other, never both.
    folds = []
    embryos = sorted(totals)
    for select_on in embryos:
        evaluate_on = [other for other in embryos if other != select_on]
        if len(evaluate_on) != 1:
            continue
        held_out = evaluate_on[0]
        candidates = [row for row in rows if row["embryo"] == select_on and row["source"] == "dog"]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda row: (float(str(row["reachability"])), -float(str(row["budget_ratio"]))),
        )
        ratio = float(str(best["budget_ratio"]))
        held = next(
            row
            for row in rows
            if row["embryo"] == held_out and row["source"] == "dog" and row["budget_ratio"] == ratio
        )
        baseline = next(
            row
            for row in rows
            if row["embryo"] == held_out and row["source"] == "classical" and row["budget_ratio"] == ratio
        )
        folds.append(
            {
                "fold_id": f"fold_{select_on}",
                "selected_on": select_on,
                "evaluated_on": held_out,
                "selected_budget_ratio": ratio,
                "selection_reachability": best["reachability"],
                "held_out_dog_reachability": held["reachability"],
                "held_out_classical_reachability": baseline["reachability"],
                "improves_over_classical": float(str(held["reachability"]))
                > float(str(baseline["reachability"])),
            }
        )
        heartbeat(
            command,
            "fold",
            f"select={select_on} ratio={ratio} -> held-out {held_out}: "
            f"dog={float(str(held['reachability'])):.4f} "
            f"classical={float(str(baseline['reachability'])):.4f}",
        )

    payload = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "experiment": "E04",
        "frames_per_movie": frames,
        "budget_ratios": list(budgets),
        "dog_refine_centroids": refine,
        "dog_suppression_radius_um": suppression_radius,
        "dog_radii_um": list(bank),
        "dog_local_maxima_only": local_maxima,
        "dog_per_scale_union": per_scale_union,
        "movies_skipped": skipped,
        "measure": "reachability: an annotated node with some proposal within 7 um, per frame",
        "rows": rows,
        "folds": folds,
        "miss_audit": (
            {
                "budget_ratio": audit_ratio,
                "nodes_audited": len(audits),
                "summary": summarise([NodeAudit(**item) for item in audits]),  # type: ignore[arg-type]
                "nodes": audits,
            }
            if audit_misses
            else None
        ),
        "both_folds_improve": bool(folds) and all(bool(fold["improves_over_classical"]) for fold in folds),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "limits": (
            "Reachability upper-bounds the official one-to-one node recall and is not it. Budgets "
            "come from each movie's own estimated_number_of_nodes, which is official metadata, so "
            "nothing about the held-out embryo enters the selection. A bounded window of each movie "
            "is read, not the whole movie."
        ),
    }
    report_path = out if out is not None else repository_root() / "artifacts/proposals.json"
    atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True) + chr(10))
    _write_manifest(command, {"report": str(report_path), "experiment": "E04"})
    heartbeat(command, "done", f"both folds improve: {payload['both_folds_improve']}")
    typer.echo(json.dumps({k: v for k, v in payload.items() if k != "rows"}, sort_keys=True))


@evaluate_app.command("track-length")
def evaluate_track_length(
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Competition data root. BIOHUB_DATA_ROOT is the only fallback."),
    ] = None,
    minimums: Annotated[
        str,
        typer.Option("--minimums", help="Comma-separated minimum frame spans to sweep."),
    ] = "2,3,4,5,6,8,10",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Report path. Defaults to artifacts/track-length.json."),
    ] = None,
) -> None:
    """What a minimum-track-length filter costs in correct edges, per embryo fold.

    The H-04 probe. Three public notebooks prune short tracks and isolated nodes
    ([[R-0010]]), and [[F-0018]] set the bar any real filter must clear: about 0.94
    edge retention to buy the step to 0.950, and below 0.92 it stops paying at all.

    Measured on the annotated graphs themselves, which is the cheapest possible
    falsification. Against ground truth the filter meets a perfect graph, so what
    it destroys here it destroys at best on a real one; a policy that already fails
    the bar against perfect input cannot pass it against a detector's output.

    Reads ground-truth graphs only. No volume is opened and the four duplicated
    public-test fixtures are never read ([[D-0015]]).
    """
    command = "evaluate track-length"
    from biohubx.data.competition import CompetitionLayoutError, competition_root, load_ground_truth
    from biohubx.evaluation.track_length import filter_cost

    try:
        data_root = competition_root(root)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    train_dir = data_root / "train"
    train = sorted(path.stem for path in train_dir.glob("*.geff"))
    if not train:
        heartbeat(command, "refused", f"no annotated training datasets under {train_dir}")
        raise typer.Exit(code=2)
    spans = tuple(int(item) for item in minimums.split(","))
    heartbeat(command, "start", f"datasets={len(train)} minimums={list(spans)}")

    started = time.monotonic()
    corpus = []
    for index, dataset_id in enumerate(train):
        corpus.append(load_ground_truth(data_root, dataset_id))
        if (index + 1) % 50 == 0:
            heartbeat(command, "loaded", f"{index + 1}/{len(train)}")
    heartbeat(command, "loaded", f"{len(corpus)} ground-truth graphs")

    rows: list[dict[str, object]] = []
    for keep_divisions in (True, False):
        for minimum in spans:
            per_embryo: dict[str, list[int]] = {}
            for truth in corpus:
                cost = filter_cost(truth.lineage, minimum_frames=minimum, keep_divisions=keep_divisions)
                bucket = per_embryo.setdefault(truth.embryo, [0, 0, 0, 0])
                bucket[0] += cost.edges_after
                bucket[1] += cost.edges_before
                bucket[2] += cost.nodes_after
                bucket[3] += cost.nodes_before
            for embryo, (edges_after, edges_before, nodes_after, nodes_before) in sorted(per_embryo.items()):
                retention = 1.0 if edges_before == 0 else edges_after / edges_before
                rows.append(
                    {
                        "embryo": embryo,
                        "minimum_frames": minimum,
                        "keep_divisions": keep_divisions,
                        "edge_retention": round(retention, 6),
                        "node_retention": round(1.0 if nodes_before == 0 else nodes_after / nodes_before, 6),
                        "edges_kept": edges_after,
                        "edges_total": edges_before,
                        "nodes_kept": nodes_after,
                        "nodes_total": nodes_before,
                        # F-0018 put break-even at 0.92 and the 0.950 step at 0.94.
                        "clears_break_even_0_92": retention >= 0.92,
                        "clears_target_0_94": retention >= 0.94,
                    }
                )
            heartbeat(
                command,
                "swept",
                f"keep_divisions={keep_divisions} minimum={minimum} "
                + " ".join(
                    f"{row['embryo']}={row['edge_retention']:.4f}" for row in rows[-len(per_embryo) :]
                ),
            )

    # The division exemption is the policy the public notebooks actually run, so
    # the headline verdict is taken from that arm.
    verdict: dict[int, bool] = {}
    for minimum in spans:
        arm = [row for row in rows if row["minimum_frames"] == minimum and row["keep_divisions"] is True]
        verdict[minimum] = bool(arm) and all(bool(row["clears_target_0_94"]) for row in arm)
    clearing = [minimum for minimum, passed in verdict.items() if passed]
    largest_clearing = max(clearing) if clearing else None

    payload = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "hypothesis": "H-04",
        "datasets": len(corpus),
        "bar": {
            "break_even_edge_retention": 0.92,
            "target_edge_retention": 0.94,
            "source": "F-0018, both embryo folds",
        },
        "rows": rows,
        "largest_minimum_clearing_0_94_on_both_folds": largest_clearing,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "limits": (
            "Measured on ground-truth graphs, where every edge is correct by construction, so the "
            "retention reported is what the filter costs against a perfect graph. A real detector's "
            "output can only do worse. This says nothing about whether the filter reaches the "
            "annotated node count, because ground truth is already at it."
        ),
    }
    report_path = out if out is not None else repository_root() / "artifacts/track-length.json"
    atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True) + chr(10))
    _write_manifest(command, {"report": str(report_path), "datasets": len(corpus)})
    heartbeat(
        command,
        "done",
        f"largest minimum clearing 0.94 on both folds: {largest_clearing}",
    )
    typer.echo(json.dumps({k: v for k, v in payload.items() if k != "rows"}, sort_keys=True))


@evaluate_app.command("mask-audit")
def evaluate_mask_audit(
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Competition data root. BIOHUB_DATA_ROOT is the only fallback."),
    ] = None,
    max_frames: Annotated[
        int,
        typer.Option("--max-frames", help="Frames sampled per dataset. 0 reads every annotated frame."),
    ] = 0,
    compare_maxpool: Annotated[
        bool,
        typer.Option(
            "--compare-maxpool/--no-compare-maxpool",
            help="Also measure a max-pooled grid. Reads full resolution, so 16x the bytes.",
        ),
    ] = False,
) -> None:
    """Measure what a confident-background band would cost, at every annotated node.

    E03-MASK-AUDIT. For each annotated cell this reads the image where the cell
    is and records its intensity percentile within its own frame, whether it is a
    local maximum, and how it would fare under each candidate band. A node below
    a band is a node that band would have called background: the falsifier for
    `confident_background`, counted rather than argued about.

    Bands are selected per fold from the TRAINING embryo alone. The evaluation
    embryo's nodes are measured under the resulting rule afterwards, as a
    false-negative proxy, and never used to choose it.

    CPU only. Reads training volumes and never the public-test directory.
    """
    command = "evaluate mask-audit"
    import numpy as np

    from biohubx.data.competition import CompetitionLayoutError, competition_root, load_ground_truth
    from biohubx.training.targets import CANDIDATE_BANDS, report_bands, select_band

    try:
        data_root = competition_root(root)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    train_dir = data_root / "train"
    datasets = sorted(path.stem for path in train_dir.glob("*.geff"))
    if not datasets:
        heartbeat(command, "refused", f"no annotated training datasets under {train_dir}")
        raise typer.Exit(code=2)
    heartbeat(command, "start", f"datasets={len(datasets)} max_frames={max_frames or 'all'}")

    started = time.perf_counter()
    totals: dict[str, int] = {
        "annotated_nodes": 0,
        "frames_total": 0,
        "frames_read": 0,
        "nodes_in_read_frames": 0,
        "nodes_measured": 0,
        "collisions": 0,
    }
    per_dataset: list[dict[str, object]] = []
    by_embryo: dict[str, list[float]] = {}
    by_embryo_frame_norm: dict[str, list[float]] = {}
    local_max_hits: dict[str, list[int]] = {}

    for index, dataset_id in enumerate(datasets):
        truth = load_ground_truth(data_root, dataset_id)
        volume = _open_volume(data_root / "train" / f"{dataset_id}.zarr")
        frames: dict[int, list[tuple[int, int, int]]] = {}
        for node in truth.lineage.nodes:
            frames.setdefault(node.frame, []).append(
                (int(node.voxel.z), int(node.voxel.y) // 4, int(node.voxel.x) // 4)
            )
        chosen = sorted(frames)
        if max_frames:
            step = max(1, len(chosen) // max_frames)
            chosen = chosen[::step][:max_frames]

        ranks: list[float] = []
        frame_ranks: list[float] = []
        is_local_max: list[int] = []
        nodes_in_chosen = sum(len(frames[f]) for f in chosen)
        collisions = 0
        seen: set[tuple[int, int, int, int]] = set()
        for frame in chosen:
            if compare_maxpool:
                full = np.asarray(volume[frame], dtype=np.float32)
                grid = full[:, ::4, ::4]
            else:
                full = None
                grid = np.asarray(volume[frame, ::1, ::4, ::4], dtype=np.float32)
            # Max-pooling instead of striding is the obvious suspect for a dim
            # annotated node: striding samples every fourth voxel and can miss a
            # cell centre entirely. Measuring both settles whether dim nodes are
            # a property of the cells or an artefact of the grid.
            if full is not None:
                depth, height, width = full.shape
                pooled = (
                    full[:, : (height // 4) * 4, : (width // 4) * 4]
                    .reshape(depth, height // 4, 4, width // 4, 4)
                    .max(axis=(2, 4))
                )
                flat_pooled = np.sort(pooled.reshape(-1))
            flat = np.sort(grid.reshape(-1))
            for z, y, x in frames[frame]:
                if not (0 <= z < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= x < grid.shape[2]):
                    continue
                value = float(grid[z, y, x])
                # Percentile rank is invariant to any monotonic normalisation, so
                # this number does not move if the intensity scaling changes.
                rank = float(np.searchsorted(flat, value, side="left")) / flat.size
                ranks.append(rank)
                # Two annotated cells can land on the same grid voxel once y and x
                # are divided by four. Both become one positive, so the count the
                # class prior is compared against is not the annotated count.
                key = (frame, z, y, x)
                if key in seen:
                    collisions += 1
                else:
                    seen.add(key)
                if full is not None and y < pooled.shape[1] and x < pooled.shape[2]:
                    frame_ranks.append(
                        float(np.searchsorted(flat_pooled, float(pooled[z, y, x]), side="left"))
                        / flat_pooled.size
                    )
                z0, z1 = max(0, z - 1), min(grid.shape[0], z + 2)
                y0, y1 = max(0, y - 1), min(grid.shape[1], y + 2)
                x0, x1 = max(0, x - 1), min(grid.shape[2], x + 2)
                is_local_max.append(int(value >= float(grid[z0:z1, y0:y1, x0:x1].max())))

        if not ranks:
            continue
        array = np.array(ranks, dtype=np.float64)
        by_embryo.setdefault(truth.embryo, []).extend(ranks)
        by_embryo_frame_norm.setdefault(truth.embryo, []).extend(frame_ranks)
        local_max_hits.setdefault(truth.embryo, []).extend(is_local_max)
        grid_voxels_movie = truth.frames * volume.shape[1] * (volume.shape[2] // 4) * (volume.shape[3] // 4)
        movie_voxels = truth.frames * volume.shape[1] * volume.shape[2] * volume.shape[3]
        prior_grid = truth.estimated_total_nodes / grid_voxels_movie
        prior_full = truth.estimated_total_nodes / movie_voxels
        totals["annotated_nodes"] += len(truth.lineage.nodes)
        totals["frames_total"] += len(frames)
        totals["frames_read"] += len(chosen)
        totals["nodes_in_read_frames"] += nodes_in_chosen
        totals["nodes_measured"] += int(array.size)
        totals["collisions"] += collisions
        per_dataset.append(
            {
                "dataset": dataset_id,
                "embryo": truth.embryo,
                "frames_total": len(frames),
                "frames_read": len(chosen),
                "annotated_nodes": len(truth.lineage.nodes),
                "nodes_in_read_frames": nodes_in_chosen,
                "nodes_measured": int(array.size),
                "grid_collisions": collisions,
                "annotated_density": len(truth.lineage.nodes) / truth.estimated_total_nodes,
                "class_prior_grid": prior_grid,
                "class_prior_full_resolution": prior_full,
                "prior_within_bounds": bool(0.0 < prior_grid < 1.0 and 0.0 < prior_full < 1.0),
                "percentile_min": float(array.min()),
                "percentile_p01": float(np.quantile(array, 0.01)),
                "percentile_median": float(np.median(array)),
                "local_maximum_fraction": float(np.mean(is_local_max)),
                "maxpool_percentile_min": float(np.min(frame_ranks)) if frame_ranks else None,
            }
        )
        if (index + 1) % 25 == 0:
            heartbeat(command, "scanned", f"{index + 1}/{len(datasets)} datasets")

    heartbeat(command, "scanned", f"{len(per_dataset)}/{len(datasets)} datasets with measurable nodes")

    embryos = sorted(by_embryo)
    embryo_reports = {
        embryo: [report.to_dict() for report in report_bands(np.array(by_embryo[embryo]), CANDIDATE_BANDS)]
        for embryo in embryos
    }

    folds = []
    for train_embryo in embryos:
        evaluate_embryo = [e for e in embryos if e != train_embryo]
        band = select_band(np.array(by_embryo[train_embryo]), CANDIDATE_BANDS, required_coverage=1.0)
        entry: dict[str, object] = {
            "fold": f"train_{train_embryo}",
            "train_embryo": train_embryo,
            "evaluate_embryo": evaluate_embryo[0] if evaluate_embryo else None,
            "selected_band": band,
            "selected_using": f"{train_embryo} annotations only",
            "evaluation_data_used_for_selection": False,
        }
        if band is None:
            entry["verdict"] = (
                "no band covers every training annotation; a negative loss cannot be "
                "justified on this fold and a positive-unlabelled objective is required"
            )
        elif evaluate_embryo:
            held = np.array(by_embryo[evaluate_embryo[0]])
            below = int((held < band).sum())
            entry["false_negative_proxy_on_evaluation_embryo"] = {
                "embryo": evaluate_embryo[0],
                "nodes": int(held.size),
                "would_be_called_background": below,
                "rate": below / held.size if held.size else 0.0,
                "note": "measured after the band was fixed; it did not select the band",
            }
        folds.append(entry)

    def column(name: str) -> list[float]:
        return [float(str(row[name])) for row in per_dataset]

    densities = column("annotated_density")
    medians = column("percentile_median")
    correlation = float(np.corrcoef(densities, medians)[0, 1]) if len(densities) > 2 else float("nan")
    priors = column("class_prior_grid")
    payload = {
        "schema_version": 1,
        "provenance_status": "measured",
        "experiment": "E03-MASK-AUDIT",
        "datasets_scanned": len(per_dataset),
        "population_reconciliation": {
            "annotated_nodes_in_corpus": totals["annotated_nodes"],
            "annotated_frames_in_corpus": totals["frames_total"],
            "frames_read": totals["frames_read"],
            "nodes_in_read_frames": totals["nodes_in_read_frames"],
            "nodes_measured": totals["nodes_measured"],
            "excluded_by_frame_sampling": totals["annotated_nodes"] - totals["nodes_in_read_frames"],
            "excluded_by_falling_outside_grid": (totals["nodes_in_read_frames"] - totals["nodes_measured"]),
            "coverage_of_corpus": (
                totals["nodes_measured"] / totals["annotated_nodes"] if totals["annotated_nodes"] else 0.0
            ),
            "grid_collisions": totals["collisions"],
            "note": (
                "excluded_by_frame_sampling is the only exclusion under the operator's "
                "control: it is zero when --max-frames is 0. Nodes outside the grid are "
                "nodes whose coordinates fall beyond the strided array, which is a "
                "property of the data. Collisions are annotated cells that share a grid "
                "voxel after dividing y and x by four; they are measured, not excluded."
            ),
        },
        "class_prior_audit": {
            "units": (
                "cells per voxel of the grid the model predicts on. A movie is "
                "(T, Z, Y, X); the model sees (T, Z, Y/4, X/4), so the grid prior is "
                "sixteen times the full-resolution prior. A crop prorates the movie "
                "estimate by its frame and voxel fraction before dividing."
            ),
            "grid_prior_min": min(priors) if priors else None,
            "grid_prior_max": max(priors) if priors else None,
            "grid_prior_median": float(np.median(priors)) if priors else None,
            "all_within_open_unit_interval": all(0.0 < p < 1.0 for p in priors),
            "datasets_out_of_bounds": [
                row["dataset"] for row in per_dataset if not row["prior_within_bounds"]
            ],
        },
        "selection_bias_probe": {
            "question": (
                "nnPU assumes labelled positives are Selected Completely At Random "
                "from all positives (SCAR). If annotators reached for bright cells "
                "first, sparsely annotated datasets should show brighter annotations "
                "than densely annotated ones."
            ),
            "pearson_density_vs_median_percentile": correlation,
            "datasets": len(per_dataset),
            "reading": (
                "A correlation near zero is consistent with SCAR and does not prove it. "
                "A negative correlation means sparser annotation picks brighter cells, "
                "which violates SCAR and biases the prior."
            ),
        },
        "nodes_measured": sum(len(v) for v in by_embryo.values()),
        "bands_evaluated": list(CANDIDATE_BANDS),
        "by_embryo": {
            embryo: {
                "nodes": len(by_embryo[embryo]),
                "percentile_min": float(np.min(by_embryo[embryo])),
                "percentile_p01": float(np.quantile(by_embryo[embryo], 0.01)),
                "percentile_median": float(np.median(by_embryo[embryo])),
                "local_maximum_fraction": float(np.mean(local_max_hits[embryo])),
                "maxpool_percentile_min": (
                    float(np.min(by_embryo_frame_norm[embryo])) if by_embryo_frame_norm.get(embryo) else None
                ),
                "maxpool_bands": (
                    [
                        report.to_dict()
                        for report in report_bands(np.array(by_embryo_frame_norm[embryo]), CANDIDATE_BANDS)
                    ]
                    if by_embryo_frame_norm.get(embryo)
                    else None
                ),
                "bands": embryo_reports[embryo],
            }
            for embryo in embryos
        },
        "folds": folds,
        "grid_reduction_sensitivity": (
            "Every node is measured twice, once on the strided grid the reference "
            "uses and once on a max-pooled grid of the same shape. If striding were "
            "discarding cell centres the two would disagree; the report records "
            "whether they do."
        ),
        "normalisation_sensitivity": (
            "Percentile rank within a frame is invariant to any monotonic intensity "
            "normalisation, so every number here is unchanged by the choice of "
            "quantile scaling. What a normalisation can move is the absolute "
            "threshold a band maps to, never which nodes fall inside it."
        ),
        "per_dataset": per_dataset,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    report_path = repository_root() / "artifacts/mask-audit.json"
    atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_manifest(command, {"report": "artifacts/mask-audit.json", "datasets": len(per_dataset)})
    for embryo in embryos:
        worst = float(np.min(by_embryo[embryo]))
        heartbeat(command, "embryo", f"{embryo} nodes={len(by_embryo[embryo])} worst_percentile={worst:.4f}")
    for entry in folds:
        heartbeat(command, "fold", f"{entry['fold']} band={entry['selected_band']}")
    heartbeat(command, "done", "report=artifacts/mask-audit.json")
    typer.echo(json.dumps({"folds": folds, "by_embryo": payload["by_embryo"]}, sort_keys=True))


def _open_volume(path: Path) -> Any:
    """Open a competition volume array, narrowing zarr's loose union once."""
    import zarr

    group: Any = zarr.open(str(path), mode="r")
    return group["0"]


@train_app.command("probe")
def train_probe(
    arm: Annotated[str, typer.Option("--arm", help="E07 arm: A3 or A4.")],
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Family config. Defaults to configs/e07-candidate-rescoring.yaml."),
    ] = None,
    section: Annotated[str, typer.Option("--section")] = "E07",
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Competition data root. BIOHUB_DATA_ROOT is the only fallback."),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Training-embryo movie. Defaults to the first annotated 44b6 movie."),
    ] = None,
    first_frame: Annotated[int, typer.Option("--first-frame")] = 0,
    frames: Annotated[int, typer.Option("--frames", help="Frames in the probe window.")] = 3,
    steps: Annotated[int, typer.Option("--steps", help="Optimizer steps after the first backward.")] = 3,
    learning_rate: Annotated[float, typer.Option("--learning-rate")] = 1e-3,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    batch: Annotated[int, typer.Option("--batch", help="Candidates per forward batch.")] = 512,
    out: Annotated[
        Path | None, typer.Option("--out", help="Report path. Defaults to artifacts/e07-probe-<arm>.json.")
    ] = None,
) -> None:
    """Stage 1: one deterministic training crop, forward, nnPU loss, backward, steps, strict reload.

    Kills a design that produces no gradient, an unstable or non-finite loss,
    an inert input channel, or an output that its own checkpoint does not
    reproduce. Then decodes at the count-tied threshold and scores the kept
    candidates through the oracle ceiling beside A0 on the same window, so the
    arm's contribution is measured where it must appear. Training data only;
    the held-out embryo is never read here.
    """
    command = "train probe"
    import time

    import numpy as np
    import torch

    from biohubx.contracts.instances import InstanceSet
    from biohubx.data.competition import (
        CompetitionLayoutError,
        WindowSelection,
        competition_root,
        load_window,
    )
    from biohubx.evaluation.official_metric import EstimatedTotalNodes, metric_row, summarise_fold
    from biohubx.evaluation.oracle import oracle_graph
    from biohubx.proposals import dog, rescore
    from biohubx.training import inspect as measure
    from biohubx.training.targets import PriorError, TargetConstructionError, positive_unlabelled_loss

    root_path = repository_root()
    config_path = config if config is not None else root_path / "configs/e07-candidate-rescoring.yaml"
    try:
        family = yaml.safe_load(config_path.read_text(encoding="utf-8"))[section]
        candidates_block = family["candidates"]
        baseline_block = family["baseline"]["proposals"]
        bank = tuple(float(r) for r in candidates_block["radii_um"])
        suppression = float(candidates_block["suppression_radius_um"])
        pool_quantile = float(candidates_block["response_quantile"])
        baseline_quantile = float(baseline_block["response_quantile"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        heartbeat(command, "refused", f"cannot read the family from {config_path} [{section}]: {exc}")
        raise typer.Exit(code=2) from exc
    if arm not in rescore.ARMS:
        heartbeat(command, "refused", f"unknown arm {arm!r}; the family declares {sorted(rescore.ARMS)}")
        raise typer.Exit(code=2)
    config_digest = digest_file(config_path, DigestKind.CANONICAL_TEXT).token
    torch_device = torch.device(device)

    try:
        data_root = competition_root(root)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    movie = dataset
    if movie is None:
        found = sorted(p.stem for p in (data_root / "train").glob("44b6*.geff"))
        if not found:
            heartbeat(command, "refused", "no annotated 44b6 movie under the data root")
            raise typer.Exit(code=2)
        movie = found[0]
    depth, height, width = _open_volume(data_root / "train" / f"{movie}.zarr").shape[1:]
    try:
        window = load_window(
            data_root,
            WindowSelection(movie, first_frame, frames, 0, depth, 0, height, 0, width),
            split="train",
        )
    except (CompetitionLayoutError, ValueError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    started = time.monotonic()
    estimate = float(window.window_estimated_total_nodes)
    baseline = dog.detect_instances(
        window.volume,
        dataset=window.annotated.dataset,
        radii_um=bank,
        response_quantile=baseline_quantile,
        suppression_radius_um=suppression,
        local_maxima_only=True,
        per_scale_union=True,
    )
    pool = rescore.extract_candidates(
        window.volume,
        dataset=window.annotated.dataset,
        radii_um=bank,
        suppression_radius_um=suppression,
        response_quantile=pool_quantile,
    )
    labels = rescore.candidate_labels(pool, window.annotated).to(torch_device)
    try:
        prior = rescore.candidate_prior(estimate, len(pool.instances))
    except rescore.RescoreError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    patches = rescore.candidate_patches(window.volume, pool, arm=arm, radii_um=bank).to(torch_device)
    heartbeat(
        command,
        "start",
        f"{arm} {movie} frames={frames} annotated={len(window.annotated.nodes)} "
        f"baseline={len(baseline.instances)} pool={len(pool.instances)} positives={int(labels.sum())} "
        f"prior={prior:.4f} patches={tuple(patches.shape)}",
    )

    def build() -> torch.nn.Module:
        return rescore.build_scorer(arm, seed=seed)

    model = build().to(torch_device)

    def forward(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        pieces = [m(part) for part in rescore.batches(x, batch)]
        return torch.cat(pieces) if pieces else torch.zeros(0, device=x.device)

    def loss_of(logits: torch.Tensor) -> torch.Tensor:
        return positive_unlabelled_loss(logits, labels, prior=prior)[0]

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trajectory: list[dict[str, float | bool]] = []
    try:
        logits = forward(model, patches)
        loss, terms = positive_unlabelled_loss(logits, labels, prior=prior)
    except (TargetConstructionError, PriorError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    loss_first = float(loss.item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore[no-untyped-call]
    gradients = measure.gradient_norm_by_block(model)
    trajectory.append(
        {"step": 0, "loss": loss_first, "negative_risk_clamped": bool(terms["negative_risk_clamped"])}
    )
    for step in range(1, steps + 1):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logits = forward(model, patches)
        loss, terms = positive_unlabelled_loss(logits, labels, prior=prior)
        loss.backward()  # type: ignore[no-untyped-call]
        trajectory.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "negative_risk_clamped": bool(terms["negative_risk_clamped"]),
            }
        )
    optimizer.step()
    finite = all(np.isfinite(t["loss"]) for t in trajectory)
    heartbeat(
        command,
        "steps",
        f"loss {loss_first:.6f} -> {trajectory[-1]['loss']:.6f} grad_norm={gradients['total']:.4e} "
        f"zero_blocks={gradients['zero_gradient_blocks']}",
    )

    channel_probes = measure.input_channel_probes(model, forward, patches, channel_axis=1)
    determinism = measure.strict_reload_determinism(build, model, forward, patches)
    inert = [p["channel"] for p in channel_probes if p["inert"]]

    model.eval()
    with torch.no_grad():
        final_logits = forward(model, patches)
    threshold = rescore.threshold_for_count(final_logits, round(estimate))
    kept = rescore.keep_by_threshold(pool, final_logits, threshold)

    def ceiling_of(instances: InstanceSet) -> dict[str, object]:
        graph = oracle_graph(instances, window.annotated)
        row: dict[str, object] = {
            "proposals": graph.proposals,
            "node_ratio": (graph.proposals - estimate) / estimate if estimate else None,
            "matched_nodes": graph.matched_nodes,
            "annotated_nodes": graph.annotated_nodes,
            "match_fraction": graph.matched_nodes / graph.annotated_nodes if graph.annotated_nodes else None,
            "retained_edges": graph.retained_edges,
            "annotated_edges": graph.annotated_edges,
            "score": None,
        }
        if graph.graph is not None:
            metric = metric_row(
                graph.graph, window.annotated, estimated_total_nodes=EstimatedTotalNodes.declared(estimate)
            )
            row["score"] = summarise_fold([metric]).score
        return row

    ceilings = {"A0": ceiling_of(baseline), "pool": ceiling_of(pool), "arm_kept": ceiling_of(kept)}
    kill_reasons = []
    if gradients["total"] == 0.0:
        kill_reasons.append("zero gradient")
    if not finite:
        kill_reasons.append("non-finite loss")
    if inert:
        kill_reasons.append(f"inert input channels {inert}")
    if not determinism["output_identical_after_reload"]:
        kill_reasons.append("strict reload does not reproduce the output")
    report = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "stage": "1, gradient probe on one deterministic training crop",
        "experiment": section,
        "arm": rescore.describe(arm),
        "config_digest": config_digest,
        "window": window.to_dict(),
        "seed": seed,
        "device": str(torch_device),
        "estimate": estimate,
        "prior": prior,
        "counts": {
            "annotated": len(window.annotated.nodes),
            "baseline_A0": len(baseline.instances),
            "pool": len(pool.instances),
            "positives": int(labels.sum()),
            "kept_at_count_tied_threshold": len(kept.instances),
        },
        "threshold": threshold,
        "census": measure.parameter_census(model),
        "trajectory": trajectory,
        "gradients": gradients,
        "channel_probes": channel_probes,
        "determinism": determinism,
        "ceilings": ceilings,
        "kill_reasons": kill_reasons,
        "survives_stage_1": not kill_reasons,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "limits": (
            "A handful of steps from random initialisation on one window; the kept set and its ceiling say "
            "whether the decode path works, not whether the arm can learn. Stage 3 answers that."
        ),
    }
    report_path = out if out is not None else root_path / f"artifacts/e07-probe-{arm}.json"
    atomic_write_text(
        report_path, json.dumps(measure.to_jsonable(report), indent=2, sort_keys=True) + chr(10)
    )
    _write_manifest(command, {"report": str(report_path), "arm": arm, "config_digest": config_digest})
    heartbeat(
        command,
        "ceilings",
        json.dumps(
            {
                k: {"n": v["proposals"], "match": v["match_fraction"], "score": v["score"]}
                for k, v in ceilings.items()
            }
        ),
    )
    heartbeat(command, "done", f"survives={not kill_reasons} kill={kill_reasons} report={report_path}")
    typer.echo(
        json.dumps(
            {
                "arm": arm,
                "survives_stage_1": not kill_reasons,
                "kill_reasons": kill_reasons,
                "ceilings": ceilings,
            },
            sort_keys=True,
        )
    )


@train_app.command("rescore")
def train_rescore(
    arm: Annotated[str, typer.Option("--arm", help="E07 arm: A3 or A4.")],
    fold: Annotated[str, typer.Option("--fold", help="fold_44b6 or fold_6bba from the family config.")],
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Family config. Defaults to configs/e07-candidate-rescoring.yaml."),
    ] = None,
    section: Annotated[str, typer.Option("--section")] = "E07",
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Competition data root. BIOHUB_DATA_ROOT is the only fallback."),
    ] = None,
    train_movies: Annotated[
        int, typer.Option("--train-movies", help="First N movies of the training embryo. 0 = all.")
    ] = 1,
    evaluate_movies: Annotated[
        int, typer.Option("--evaluate-movies", help="First M movies of the held-out embryo. 0 = all.")
    ] = 1,
    frames: Annotated[int, typer.Option("--frames", help="Frames per movie window.")] = 10,
    epochs: Annotated[int, typer.Option("--epochs")] = 1,
    batch: Annotated[int, typer.Option("--batch")] = 512,
    learning_rate: Annotated[float, typer.Option("--learning-rate")] = 1e-3,
    count_ratio: Annotated[
        float,
        typer.Option("--count-ratio", help="Kept candidates per movie = round(ratio x its own estimate)."),
    ] = 1.0,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    cache: Annotated[
        Path | None, typer.Option("--cache", help="Patch cache directory. Defaults to artifacts/cache/e07.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Report path. Defaults to artifacts/e07-<fold>-<arm>.json.")
    ] = None,
) -> None:
    """Train one E07 arm on the fold's training embryo and score its kept candidates on the other.

    Stage 2 and 3 of the funnel run this same loop on Kaggle; on the workstation
    it is the CPU smoke that proves the loop end to end on a movie or two. The
    count-tied threshold is frozen on the training movies before the held-out
    embryo is read, and the held-out ceiling is reported beside A0 and the pool
    on the same windows. A diagnostic ceiling, never a promotable number.
    """
    command = "train rescore"
    import torch

    from biohubx.data.competition import CompetitionLayoutError, competition_root
    from biohubx.proposals import rescore
    from biohubx.training import inspect as measure
    from biohubx.training.rescore_loop import LoopSpec, run_loop

    root_path = repository_root()
    config_path = config if config is not None else root_path / "configs/e07-candidate-rescoring.yaml"
    try:
        family = yaml.safe_load(config_path.read_text(encoding="utf-8"))[section]
        candidates_block = family["candidates"]
        baseline_block = family["baseline"]["proposals"]
        folds = {f["id"]: f for f in family["folds"]}
        chosen = folds[fold]
        spec = LoopSpec(
            arm=arm,
            train_embryo=str(chosen["train_on"]),
            evaluate_embryo=str(chosen["evaluate_on"]),
            train_movies=train_movies,
            evaluate_movies=evaluate_movies,
            frames=frames,
            epochs=epochs,
            batch=batch,
            learning_rate=learning_rate,
            seed=seed,
            radii_um=tuple(float(r) for r in candidates_block["radii_um"]),
            suppression_radius_um=float(candidates_block["suppression_radius_um"]),
            pool_quantile=float(candidates_block["response_quantile"]),
            baseline_quantile=float(baseline_block["response_quantile"]),
            config_digest=digest_file(config_path, DigestKind.CANONICAL_TEXT).token,
            count_ratio=count_ratio,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        heartbeat(command, "refused", f"cannot read fold {fold!r} of {section} from {config_path}: {exc}")
        raise typer.Exit(code=2) from exc
    if arm not in rescore.ARMS:
        heartbeat(command, "refused", f"unknown arm {arm!r}; the family declares {sorted(rescore.ARMS)}")
        raise typer.Exit(code=2)
    try:
        data_root = competition_root(root)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        heartbeat(command, "refused", "cuda was requested and is not available")
        raise typer.Exit(code=2)
    cache_dir = cache if cache is not None else root_path / "artifacts/cache/e07"
    report_path = out if out is not None else root_path / f"artifacts/e07-{fold}-{arm}.json"
    checkpoint_path = report_path.with_suffix(".pt")
    heartbeat(command, "start", json.dumps(spec.to_dict(), sort_keys=True))

    def log(stage: str, detail: str) -> None:
        heartbeat(command, stage, detail)

    result = run_loop(
        data_root, spec, device=torch_device, cache_dir=cache_dir, checkpoint_path=checkpoint_path, log=log
    )
    payload = {
        "schema_version": 1,
        "provenance_status": "diagnostic_ceiling",
        "experiment": section,
        "fold": fold,
        "device": str(torch_device),
        "execution": measure.environment(),
        "checkpoint": str(checkpoint_path.relative_to(root_path))
        if checkpoint_path.is_relative_to(root_path)
        else str(checkpoint_path),
        **result.to_dict(),
        "limits": (
            "Oracle ceilings over kept candidates on bounded windows of a fixed movie "
            "subset; comparable with A0 "
            "and the pool on the same windows and with nothing else. A promotable number needs Stage 4."
        ),
    }
    atomic_write_text(
        report_path, json.dumps(measure.to_jsonable(payload), indent=2, sort_keys=True) + chr(10)
    )
    _write_manifest(
        command, {"report": str(report_path), "fold": fold, "arm": arm, "config_digest": spec.config_digest}
    )
    heartbeat(
        command,
        "done",
        f"held-out {spec.evaluate_embryo}: arm={result.held_out.get('score')} "
        f"A0={result.baseline_held_out.get('score')} "
        f"pool={result.pool_held_out.get('score')} "
        f"reload_identical={result.reload_identical} report={report_path}",
    )
    typer.echo(
        json.dumps(
            {"fold": fold, "arm": arm, "held_out": payload["held_out"], "threshold": result.threshold},
            sort_keys=True,
        )
    )


@package_app.command("kaggle")
def package_kaggle(
    owner: Annotated[
        str,
        typer.Option(
            "--owner",
            help="Kaggle account slug owning the kernel, as it appears in the account URL.",
        ),
    ],
    fold: Annotated[
        str, typer.Option("--fold", help="Fold id from configs/e03-clean-folds.yaml.")
    ] = "fold_44b6",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Where to stage the package. Defaults to artifacts/kaggle-package."),
    ] = None,
    epochs: Annotated[int, typer.Option("--epochs", help="Training epochs.")] = 2,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Movies per optimizer step.")] = 2,
    learning_rate: Annotated[float, typer.Option("--learning-rate", help="Optimizer step size.")] = 1e-4,
    max_movies: Annotated[
        int, typer.Option("--max-movies", help="Cap training movies. 0 uses the whole training embryo.")
    ] = 2,
    allow_gpu_counts: Annotated[
        str,
        typer.Option(
            "--allow-gpu-counts",
            help="Comma-separated visible-GPU counts the run accepts. Training pins to cuda:0 "
            "whatever it is handed.",
        ),
    ] = "1,2",
    runtime_ceiling: Annotated[
        int, typer.Option("--runtime-ceiling", help="Seconds after which the run refuses to continue.")
    ] = 2400,
    allow_dirty: Annotated[
        bool,
        typer.Option(
            "--allow-dirty",
            help="Build from an uncommitted tree. The package records it and is not pushable.",
        ),
    ] = False,
    root: Annotated[
        Path | None, typer.Option("--root", help="Competition data root, for the pre-push gate.")
    ] = None,
    expect_device: Annotated[
        str,
        typer.Option("--expect-device", help="Substring the allocated GPU name must contain."),
    ] = "Tesla T4",
    smoke_id: Annotated[
        str,
        typer.Option("--smoke-id", help="Attempt identity, carried in the package, log and manifest."),
    ] = "E03-SMOKE-03",
    wheelhouse: Annotated[
        Path | None,
        typer.Option("--wheelhouse", help="Local wheelhouse. Defaults to artifacts/wheelhouse."),
    ] = None,
    expect_published: Annotated[
        str | None,
        typer.Option(
            "--expect-published",
            help="Refuse unless the wheelhouse publishes this tree identity.",
        ),
    ] = None,
    expect_kernel: Annotated[
        str | None,
        typer.Option("--expect-kernel", help="Refuse unless the build would create exactly this kernel id."),
    ] = None,
) -> None:
    """Stage a Kaggle package for one fold, and optionally run its entry point on CPU.

    The package carries code and constants: no competition bytes, no external
    weights. Its entry point re-verifies mounted input identity, asserts fold
    membership, refuses public-test paths and quarantined checkpoints, and emits
    a heartbeat at every stage. Building is local and pushes nothing.
    """
    command = "package kaggle"
    try:
        permitted_counts = _permitted_gpu_counts(allow_gpu_counts)
    except ValueError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    import shutil
    import subprocess
    import tempfile

    from biohubx.hashing import tree_digest
    from biohubx.packaging import prepush
    from biohubx.packaging.kaggle import (
        FoldSpec,
        PackageSpec,
        PackagingError,
        archive_digest,
        archive_inventory,
        build_notebook,
        check_payload_contents,
        deterministic_archive,
        forbidden_content,
        kernel_id_for,
        kernel_metadata,
    )
    from biohubx.packaging.preflight import WHEELHOUSE_SLUG
    from biohubx.packaging.wheelhouse import published_relative_paths, stage_published_payload

    root_path = repository_root()
    config_path = root_path / "configs/e03-clean-folds.yaml"
    if not config_path.is_file():
        heartbeat(command, "refused", f"no fold configuration at {config_path}")
        raise typer.Exit(code=2)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["E03"]
    match = [item for item in config["folds"] if item["id"] == fold]
    if not match:
        heartbeat(command, "refused", f"unknown fold {fold!r}; known: {[i['id'] for i in config['folds']]}")
        raise typer.Exit(code=2)
    entry = match[0]

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root_path, capture_output=True, text=True, check=False
    )
    if revision.returncode != 0:
        heartbeat(command, "refused", "cannot resolve the repository commit; a package must pin one")
        raise typer.Exit(code=2)
    commit = revision.stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root_path, capture_output=True, text=True, check=False
    ).stdout.strip()
    if dirty and not allow_dirty:
        heartbeat(
            command,
            "refused",
            "the working tree is dirty, so the commit this package would pin does not describe "
            "the bytes it ships; commit first, or pass --allow-dirty for an unpushable build",
        )
        raise typer.Exit(code=2)

    # E03 needs zarr, which the image does not carry, so the wheelhouse is an
    # input to this package and its identity travels inside it (D-0032, D-0033).
    house = wheelhouse if wheelhouse is not None else root_path / "artifacts/wheelhouse"
    if not (house / "wheels").is_dir():
        heartbeat(command, "refused", f"no wheelhouse under {house}; E03 cannot import zarr without it")
        raise typer.Exit(code=2)
    bundle = tree_digest(house, on_file=_periodic_progress(command, "upload bundle"))
    with tempfile.TemporaryDirectory(prefix="biohubx-published-") as scratch:
        payload_root = Path(scratch) / "payload"
        payload_root.mkdir()
        stage_published_payload(house, payload_root, published_relative_paths(bundle))
        published = tree_digest(payload_root, on_file=_periodic_progress(command, "published payload"))
    if expect_published is not None and expect_published != published.digest.token:
        heartbeat(
            command,
            "refused",
            f"the wheelhouse publishes {published.digest.token}, not the authorised {expect_published}",
        )
        raise typer.Exit(code=2)
    heartbeat(
        command,
        "wheelhouse",
        f"published payload {published.digest.token}"
        + (" (pinned)" if expect_published is not None else " (NOT pinned to an authorisation)"),
    )

    spec = PackageSpec(
        commit=commit,
        config_path="configs/e03-clean-folds.yaml",
        config_digest=digest_file(config_path, DigestKind.CANONICAL_TEXT).token,
        fold=FoldSpec(
            fold_id=entry["id"],
            train_embryo=entry["train_embryo"],
            evaluate_embryo=entry["evaluate_embryo"],
            seed=entry["seed"],
        ),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        # Canonical casing. Attempt 2 requested nvidiaTeslaT4 and Kaggle
        # allocated a P100 (D-0028). Whether the casing caused it is not
        # established, so this is a correction, not a fix: the guarantee is the
        # runtime guard refusing any device that is not a Tesla T4.
        accelerator="NvidiaTeslaT4",
        allowed_gpu_counts=permitted_counts,
        expected_device_substring=expect_device,
        smoke=True,
        smoke_id=smoke_id,
        wheelhouse_slug=WHEELHOUSE_SLUG,
        wheelhouse_tree=published.digest.token,
        max_movies=max_movies or None,
        runtime_ceiling_seconds=runtime_ceiling,
    )
    heartbeat(command, "start", f"fold={spec.fold.fold_id} commit={commit[:12]} dirty={bool(dirty)}")

    kernel_title = f"Biohub-X E03 {spec.fold.fold_id.replace('_', ' ')}"
    try:
        kernel_id = kernel_id_for(owner, kernel_title)
    except PackagingError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    if expect_kernel and kernel_id != expect_kernel:
        heartbeat(
            command,
            "refused",
            f"the kernel this build would create is {kernel_id}, not the expected {expect_kernel}",
        )
        raise typer.Exit(code=2)
    heartbeat(command, "kernel", f"id={kernel_id} title={kernel_title!r}")

    staging = out if out is not None else root_path / "artifacts/kaggle-package"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "src").mkdir(parents=True)
    shutil.copytree(
        root_path / "src/biohubx",
        staging / "src/biohubx",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(config_path, staging / "e03-clean-folds.yaml")

    # The package ships the registry's tree digests for exactly the artifacts
    # this fold will read. Without them the entry point refuses, because an
    # identity check over an empty set verifies nothing.
    registry = load_artifact_registry(root_path / ARTIFACT_REGISTRY_PATH)
    prefix = f"competition.train.{spec.fold.train_embryo}_"
    input_digests = {
        record.id.removeprefix("competition.train."): record.digests["tree"]
        for record in registry.artifacts
        if record.id.startswith(prefix) and "tree" in record.digests
    }
    if not input_digests:
        heartbeat(command, "refused", f"no registered train artifacts for embryo {spec.fold.train_embryo}")
        raise typer.Exit(code=2)
    input_shapes = {
        record.id.removeprefix("competition.train."): [
            record.shape.file_count,
            record.shape.empty_directory_count,
            record.shape.total_bytes,
        ]
        for record in registry.artifacts
        if record.id.startswith(prefix) and record.shape is not None
    }
    shipped: dict[str, object] = dict(spec.to_dict())
    shipped["input_digests"] = input_digests
    shipped["input_shapes"] = input_shapes
    heartbeat(
        command,
        "digests",
        f"shipped identities for {len(input_digests)} {spec.fold.train_embryo} artifacts",
    )
    # The package travels inside the notebook. Nothing is imported from
    # /kaggle/input, because no dataset or model source supplies it.
    try:
        archive_bytes = deterministic_archive(root_path / "src/biohubx")
        check_payload_contents(archive_bytes)
    except PackagingError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    inventory = archive_inventory(archive_bytes)
    payload_token = archive_digest(archive_bytes)
    heartbeat(
        command,
        "payload",
        f"bytes={len(archive_bytes)} files={len(inventory)} digest={payload_token[:46]}",
    )
    notebook = build_notebook(
        spec,
        shipped=shipped,
        payload=archive_bytes,
        wheelhouse_payload={
            "tree": published.digest.token,
            "records": [
                [
                    record.kind,
                    "-" if record.content_sha256 is None else record.content_sha256,
                    "-" if record.size_bytes is None else str(record.size_bytes),
                    record.relative_path,
                ]
                for record in published.records
            ],
        },
    )
    atomic_write_text(staging / "run.ipynb", json.dumps(notebook, indent=1, sort_keys=True) + chr(10))
    atomic_write_text(
        staging / "kernel-metadata.json",
        json.dumps(
            kernel_metadata(spec, slug=kernel_id, title=kernel_title),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    atomic_write_text(staging / "spec.json", json.dumps(shipped, indent=2, sort_keys=True) + "\n")

    offenders = forbidden_content(staging)
    if offenders:
        heartbeat(command, "refused", f"package would ship forbidden content: {offenders[:5]}")
        raise typer.Exit(code=2)

    files = sorted(p for p in staging.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)
    listing = {
        str(p.relative_to(staging)).replace("\\", "/"): digest_file(p, DigestKind.RAW_ARTIFACT).token
        for p in files
    }
    manifest_text = json.dumps(
        {
            "schema_version": 1,
            "spec": spec.to_dict(),
            "files": listing,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "contains_competition_bytes": False,
            "contains_external_weights": False,
            "repository_clean_at_build": not dirty,
            "pushable": not dirty,
        },
        indent=2,
        sort_keys=True,
    )
    atomic_write_text(staging / "PACKAGE_MANIFEST.json", manifest_text + "\n")
    package_digest = digest_file(staging / "PACKAGE_MANIFEST.json", DigestKind.CANONICAL_TEXT)
    heartbeat(
        command,
        "staged",
        f"files={len(files)} bytes={total_bytes} digest={package_digest.token[:52]}",
    )

    push_command = (
        f"kaggle kernels push -p {staging.relative_to(root_path)}"
        if staging.is_relative_to(root_path)
        else f"kaggle kernels push -p {staging}"
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "package": {
            "path": str(staging.relative_to(root_path))
            if staging.is_relative_to(root_path)
            else str(staging),
            "manifest_digest": package_digest.token,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "push_command": push_command,
            "pushed": False,
        },
        "spec": spec.to_dict(),
    }

    # The pre-push gate. Not optional: a package that has not proved it converts
    # and speaks is a package that can spend a GPU session on a stack trace,
    # which is exactly what E03-SMOKE did (D-0026). Everything here runs before
    # any network call, and nothing below this point contacts Kaggle.
    from biohubx.data.competition import CompetitionLayoutError, competition_root
    from biohubx.packaging.entry import EntryRefusal
    from biohubx.packaging.prepush import PrePushError

    try:
        data_root = competition_root(root)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    gate_out = staging.parent / "kaggle-smoke"
    gate_out.mkdir(parents=True, exist_ok=True)
    os.environ["BIOHUBX_OUTPUT"] = str(gate_out)
    heartbeat(command, "pre-push", "validating, converting and exercising before any network action")
    try:
        # The published payload staged as a real tree, so the notebook's own
        # resolver and identity check run against a mount rather than a mock.
        with tempfile.TemporaryDirectory(prefix="biohubx-gate-mount-") as mount:
            gate_mount = Path(mount) / "input"
            gate_payload = gate_mount / "biohubx-wheelhouse-zarr-cp312-linux"
            gate_payload.mkdir(parents=True)
            stage_published_payload(house, gate_payload, published_relative_paths(bundle))
            gate = prepush.run_all(
                notebook,
                shipped,
                data_root=data_root,
                interpreter=sys.executable,
                wheelhouse_root=gate_mount,
            )
    except (PrePushError, EntryRefusal, PackagingError) as exc:
        heartbeat(command, "refused", f"pre-push gate failed, nothing was sent: {exc}")
        raise typer.Exit(code=2) from exc
    heartbeat(
        command,
        "pre-push",
        f"nbformat ok, nbconvert produced {gate.converted_bytes} bytes, "
        f"{gate.stage_lines_seen} heartbeat lines across {len(gate.stage_names)} stages",
    )
    payload["pre_push_gate"] = gate.to_dict()
    payload["bootstrap"] = {
        "embedded_payload_digest": payload_token,
        "embedded_payload_bytes": len(archive_bytes),
        "embedded_file_count": len(inventory),
        "inventory": inventory,
        "extracts_to": "/kaggle/working/biohubx-package",
        "imports_from_kaggle_input": False,
    }
    # A smoke from an earlier build is not evidence about this one. The manifest
    # records the commit it ran at, so a stale one is named as stale rather than
    # presented beside a fresh package as though it belonged to it. This is the
    # same refusal `evaluate slice` makes about a stored graph.
    smoke_manifest = gate_out / f"result-{spec.fold.fold_id}.json"
    if smoke_manifest.is_file():
        stored = json.loads(smoke_manifest.read_text(encoding="utf-8"))
        stored_commit = str(stored.get("spec", {}).get("commit", ""))
        if stored_commit == commit:
            payload["local_smoke"] = stored
        else:
            payload["local_smoke"] = {
                "state": "stale, not this build",
                "ran_at_commit": stored_commit,
                "this_package_pins": commit,
                "note": "re-run with --smoke-local to exercise the entry point at this commit",
            }
            heartbeat(
                command,
                "local-smoke",
                f"stale: the stored smoke ran at {stored_commit[:12]}, this package pins {commit[:12]}",
            )
    else:
        payload["local_smoke"] = {"state": "not run; pass --smoke-local to exercise the entry point"}

    report = root_path / "artifacts/kaggle-package.json"
    atomic_write_text(report, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_manifest(command, {"report": "artifacts/kaggle-package.json", "fold": spec.fold.fold_id})
    heartbeat(command, "done", f"NOT PUSHED. push with: {push_command}")
    typer.echo(json.dumps(payload, sort_keys=True))


@package_app.command("rescore")
def package_rescore(
    owner: Annotated[
        str,
        typer.Option(
            "--owner", help="Kaggle account slug owning the kernel, as it appears in the account URL."
        ),
    ],
    arm: Annotated[str, typer.Option("--arm", help="E07 arm: A3 or A4.")],
    fold: Annotated[
        str, typer.Option("--fold", help="fold_44b6 or fold_6bba from the family config.")
    ] = "fold_44b6",
    smoke_id: Annotated[
        str, typer.Option("--smoke-id", help="Attempt identity, carried in the package, log and manifest.")
    ] = "E07-SMOKE-01",
    train_movies: Annotated[
        int, typer.Option("--train-movies", help="First N movies of the training embryo.")
    ] = 2,
    evaluate_movies: Annotated[
        int, typer.Option("--evaluate-movies", help="First M movies of the held-out embryo.")
    ] = 2,
    frames: Annotated[int, typer.Option("--frames", help="Frames per movie window.")] = 10,
    epochs: Annotated[int, typer.Option("--epochs")] = 2,
    batch: Annotated[int, typer.Option("--batch")] = 512,
    learning_rate: Annotated[float, typer.Option("--learning-rate")] = 1e-3,
    count_ratio: Annotated[float, typer.Option("--count-ratio")] = 1.0,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    allow_gpu_counts: Annotated[
        str,
        typer.Option(
            "--allow-gpu-counts",
            help="Comma-separated visible-GPU counts the run accepts. Training pins to cuda:0 "
            "whatever it is handed.",
        ),
    ] = "1,2",
    expect_device: Annotated[
        str, typer.Option("--expect-device", help="Substring the allocated GPU name must contain.")
    ] = "Tesla T4",
    runtime_ceiling: Annotated[
        int, typer.Option("--runtime-ceiling", help="Seconds after which the run refuses to continue.")
    ] = 1800,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Family config. Defaults to configs/e07-candidate-rescoring.yaml."),
    ] = None,
    wheelhouse: Annotated[
        Path | None,
        typer.Option("--wheelhouse", help="The metric wheelhouse. Defaults to artifacts/wheelhouse-metric."),
    ] = None,
    expect_published: Annotated[
        str | None,
        typer.Option("--expect-published", help="Refuse unless the wheelhouse publishes this tree identity."),
    ] = None,
    root: Annotated[
        Path | None, typer.Option("--root", help="Competition data root, for the pre-push gate.")
    ] = None,
    expect_kernel: Annotated[
        str | None,
        typer.Option("--expect-kernel", help="Refuse unless the build would create exactly this kernel id."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Where to stage. Defaults to artifacts/kaggle-rescore.")
    ] = None,
    allow_dirty: Annotated[
        bool, typer.Option("--allow-dirty", help="Build from an uncommitted tree; recorded and unpushable.")
    ] = False,
) -> None:
    """Stage the E07 kernel: one arm, one fold, bounded movies and epochs, gated locally.

    The same notebook shape as the E03 package: the source travels inside the
    notebook, the wheelhouse is located and verified by identity before pip,
    the entry point runs behind the GPU, input and wheelhouse guards. What runs
    is `entry.run_rescore`: train the re-scorer on the fold's training movies,
    freeze the decode, score the held-out movies through the oracle ceiling
    beside A0. The pre-push gate exercises the notebook locally on one movie,
    two frames and one epoch. Nothing here contacts Kaggle.
    """
    command = "package rescore"
    try:
        permitted_counts = _permitted_gpu_counts(allow_gpu_counts)
    except ValueError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    import shutil
    import subprocess
    import tempfile

    from biohubx.data.competition import CompetitionLayoutError, competition_root
    from biohubx.hashing import tree_digest
    from biohubx.packaging import prepush
    from biohubx.packaging.entry import EntryRefusal
    from biohubx.packaging.kaggle import (
        FoldSpec,
        PackageSpec,
        PackagingError,
        archive_digest,
        archive_inventory,
        build_notebook,
        check_payload_contents,
        deterministic_archive,
        forbidden_content,
        kernel_id_for,
        kernel_metadata,
    )
    from biohubx.packaging.metric_preflight import METRIC_WHEELHOUSE_SLUG
    from biohubx.packaging.prepush import PrePushError
    from biohubx.packaging.wheelhouse import published_relative_paths, stage_published_payload
    from biohubx.proposals import rescore

    root_path = repository_root()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root_path, capture_output=True, text=True, check=False
    )
    if revision.returncode != 0:
        heartbeat(command, "refused", "cannot resolve the repository commit; a package must pin one")
        raise typer.Exit(code=2)
    commit = revision.stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root_path, capture_output=True, text=True, check=False
    ).stdout.strip()
    if dirty and not allow_dirty:
        heartbeat(
            command,
            "refused",
            "the working tree is dirty; commit first, or pass --allow-dirty for an unpushable build",
        )
        raise typer.Exit(code=2)

    config_path = config if config is not None else root_path / "configs/e07-candidate-rescoring.yaml"
    try:
        family = yaml.safe_load(config_path.read_text(encoding="utf-8"))["E07"]
        candidates_block = family["candidates"]
        baseline_block = family["baseline"]["proposals"]
        chosen = {f["id"]: f for f in family["folds"]}[fold]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        heartbeat(command, "refused", f"cannot read fold {fold!r} from {config_path}: {exc}")
        raise typer.Exit(code=2) from exc
    if arm not in rescore.ARMS:
        heartbeat(command, "refused", f"unknown arm {arm!r}; the family declares {sorted(rescore.ARMS)}")
        raise typer.Exit(code=2)

    house = wheelhouse if wheelhouse is not None else root_path / "artifacts/wheelhouse-metric"
    if not (house / "wheels").is_dir():
        heartbeat(command, "refused", f"no wheels under {house}; assemble the metric wheelhouse first")
        raise typer.Exit(code=2)
    bundle = tree_digest(house, on_file=_periodic_progress(command, "upload bundle"))
    with tempfile.TemporaryDirectory(prefix="biohubx-published-") as scratch:
        payload_root = Path(scratch) / "payload"
        payload_root.mkdir()
        stage_published_payload(house, payload_root, published_relative_paths(bundle))
        published = tree_digest(payload_root, on_file=_periodic_progress(command, "published payload"))
    if expect_published is not None and expect_published != published.digest.token:
        heartbeat(
            command,
            "refused",
            f"the wheelhouse publishes {published.digest.token}, not the authorised {expect_published}",
        )
        raise typer.Exit(code=2)
    heartbeat(
        command,
        "wheelhouse",
        f"published payload {published.digest.token}"
        + (" (pinned)" if expect_published else " (NOT pinned to an authorisation)"),
    )

    spec = PackageSpec(
        commit=commit,
        config_path="configs/e07-candidate-rescoring.yaml",
        config_digest=digest_file(config_path, DigestKind.CANONICAL_TEXT).token,
        fold=FoldSpec(
            fold_id=fold,
            train_embryo=str(chosen["train_on"]),
            evaluate_embryo=str(chosen["evaluate_on"]),
            seed=seed,
        ),
        epochs=epochs,
        batch_size=batch,
        learning_rate=learning_rate,
        accelerator="NvidiaTeslaT4",
        allowed_gpu_counts=permitted_counts,
        expected_device_substring=expect_device,
        smoke=True,
        smoke_id=smoke_id,
        wheelhouse_slug=METRIC_WHEELHOUSE_SLUG,
        wheelhouse_tree=published.digest.token,
        max_movies=train_movies or None,
        runtime_ceiling_seconds=runtime_ceiling,
    )
    loop = {
        "arm": arm,
        "train_movies": train_movies,
        "evaluate_movies": evaluate_movies,
        "frames": frames,
        "epochs": epochs,
        "batch": batch,
        "learning_rate": learning_rate,
        "count_ratio": count_ratio,
        "radii_um": [float(r) for r in candidates_block["radii_um"]],
        "suppression_radius_um": float(candidates_block["suppression_radius_um"]),
        "pool_quantile": float(candidates_block["response_quantile"]),
        "baseline_quantile": float(baseline_block["response_quantile"]),
    }
    heartbeat(
        command,
        "start",
        f"E07 {arm} {fold} commit={commit[:12]} dirty={bool(dirty)} loop={json.dumps(loop, sort_keys=True)}",
    )

    kernel_title = f"Biohub-X E07 {arm} {fold.replace('_', ' ')}"
    try:
        kernel_id = kernel_id_for(owner, kernel_title)
    except PackagingError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    if expect_kernel and kernel_id != expect_kernel:
        heartbeat(
            command,
            "refused",
            f"the kernel this build would create is {kernel_id}, not the expected {expect_kernel}",
        )
        raise typer.Exit(code=2)
    heartbeat(command, "kernel", f"id={kernel_id} title={kernel_title!r}")

    staging = out if out is not None else root_path / "artifacts/kaggle-rescore"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # The registry's digests for exactly the movies this run opens: the first N of
    # the training embryo and the first M of the held-out embryo, in sorted order.
    registry = load_artifact_registry(root_path / ARTIFACT_REGISTRY_PATH)
    input_digests: dict[str, str] = {}
    input_shapes: dict[str, list[int]] = {}
    for embryo, count in (
        (spec.fold.train_embryo, train_movies),
        (spec.fold.evaluate_embryo, evaluate_movies),
    ):
        prefix = f"competition.train.{embryo}_"
        ids = sorted(
            {
                r.id.removeprefix("competition.train.").rsplit(".", 1)[0]
                for r in registry.artifacts
                if r.id.startswith(prefix)
            }
        )
        wanted = set(ids[:count] if count else ids)
        for record in registry.artifacts:
            name = record.id.removeprefix("competition.train.")
            if record.id.startswith(prefix) and name.rsplit(".", 1)[0] in wanted:
                if "tree" in record.digests:
                    input_digests[name] = record.digests["tree"]
                if record.shape is not None:
                    input_shapes[name] = [
                        record.shape.file_count,
                        record.shape.empty_directory_count,
                        record.shape.total_bytes,
                    ]
    if not input_digests:
        heartbeat(command, "refused", "no registered train artifacts for the movies this run would open")
        raise typer.Exit(code=2)
    shipped: dict[str, object] = dict(spec.to_dict())
    shipped.update(
        {"experiment": "E07", "loop": loop, "input_digests": input_digests, "input_shapes": input_shapes}
    )
    heartbeat(
        command, "digests", f"shipped identities for {len(input_digests)} artifacts across both embryos"
    )

    try:
        archive_bytes = deterministic_archive(root_path / "src/biohubx")
        check_payload_contents(archive_bytes)
    except PackagingError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    inventory = archive_inventory(archive_bytes)
    payload_token = archive_digest(archive_bytes)
    heartbeat(
        command, "payload", f"bytes={len(archive_bytes)} files={len(inventory)} digest={payload_token[:46]}"
    )
    notebook = build_notebook(
        spec,
        shipped=shipped,
        payload=archive_bytes,
        wheelhouse_payload={
            "tree": published.digest.token,
            "records": [
                [
                    record.kind,
                    "-" if record.content_sha256 is None else record.content_sha256,
                    "-" if record.size_bytes is None else str(record.size_bytes),
                    record.relative_path,
                ]
                for record in published.records
            ],
        },
    )
    atomic_write_text(staging / "run.ipynb", json.dumps(notebook, indent=1, sort_keys=True) + chr(10))
    atomic_write_text(
        staging / "kernel-metadata.json",
        json.dumps(kernel_metadata(spec, slug=kernel_id, title=kernel_title), indent=2, sort_keys=True)
        + chr(10),
    )
    atomic_write_text(staging / "spec.json", json.dumps(shipped, indent=2, sort_keys=True) + chr(10))
    offenders = forbidden_content(staging)
    if offenders:
        heartbeat(command, "refused", f"the package would ship forbidden content: {offenders[:5]}")
        raise typer.Exit(code=2)

    files = sorted(path for path in staging.rglob("*") if path.is_file())
    listing = {
        path.relative_to(staging).as_posix(): digest_file(path, DigestKind.RAW_ARTIFACT).token
        for path in files
    }
    total_bytes = sum(path.stat().st_size for path in files)
    manifest_text = json.dumps(
        {
            "schema_version": 1,
            "experiment": "E07",
            "spec": spec.to_dict(),
            "loop": loop,
            "files": listing,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "contains_competition_bytes": False,
            "contains_external_weights": False,
            "repository_clean_at_build": not dirty,
            "pushable": not dirty,
        },
        indent=2,
        sort_keys=True,
    )
    atomic_write_text(staging / "PACKAGE_MANIFEST.json", manifest_text + chr(10))
    package_digest = digest_file(staging / "PACKAGE_MANIFEST.json", DigestKind.CANONICAL_TEXT)
    heartbeat(command, "staged", f"files={len(files)} bytes={total_bytes} digest={package_digest.token[:52]}")

    payload: dict[str, object] = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "experiment": "E07",
        "package": {
            "path": str(staging.relative_to(root_path))
            if staging.is_relative_to(root_path)
            else str(staging),
            "manifest_digest": package_digest.token,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "pushed": False,
        },
        "spec": spec.to_dict(),
        "loop": loop,
        "kernel": {"id": kernel_id, "title": kernel_title},
    }

    try:
        data_root = competition_root(root)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    gate_out = staging.parent / "kaggle-rescore-gate"
    gate_out.mkdir(parents=True, exist_ok=True)
    os.environ["BIOHUBX_OUTPUT"] = str(gate_out)
    heartbeat(
        command, "pre-push", "validating, converting and exercising on one movie, two frames, one epoch"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="biohubx-gate-mount-") as mount:
            gate_mount = Path(mount) / "input"
            gate_payload = gate_mount / METRIC_WHEELHOUSE_SLUG.rsplit("/", 1)[-1]
            gate_payload.mkdir(parents=True)
            stage_published_payload(house, gate_payload, published_relative_paths(bundle))
            gate = prepush.run_all(
                notebook, shipped, data_root=data_root, interpreter=sys.executable, wheelhouse_root=gate_mount
            )
    except (PrePushError, EntryRefusal, PackagingError) as exc:
        heartbeat(command, "refused", f"pre-push gate failed, nothing was sent: {exc}")
        raise typer.Exit(code=2) from exc
    heartbeat(
        command,
        "pre-push",
        f"nbformat ok, nbconvert produced {gate.converted_bytes} bytes, "
        f"{gate.stage_lines_seen} heartbeat lines across {len(gate.stage_names)} stages",
    )
    payload["pre_push_gate"] = gate.to_dict()
    payload["bootstrap"] = {
        "embedded_payload_digest": payload_token,
        "embedded_payload_bytes": len(archive_bytes),
        "embedded_file_count": len(inventory),
        "extracts_to": "/kaggle/working/biohubx-package",
        "imports_from_kaggle_input": False,
    }
    payload["wheelhouse_identity"] = {
        "published_payload": published.digest.token,
        "upload_bundle": bundle.digest.token,
        "slug": METRIC_WHEELHOUSE_SLUG,
    }
    atomic_write_text(
        root_path / "artifacts/kaggle-rescore.json", json.dumps(payload, indent=2, sort_keys=True) + chr(10)
    )
    _write_manifest(
        command,
        {"report": "artifacts/kaggle-rescore.json", "digest": package_digest.token, "arm": arm, "fold": fold},
    )
    heartbeat(command, "done", "NOT PUSHED; push goes through package transport under an envelope")
    typer.echo(
        json.dumps(
            {
                "package": payload["package"],
                "kernel": payload["kernel"],
                "gate_stages": gate.to_dict().get("stage_names"),
            },
            sort_keys=True,
        )
    )


@package_app.command("audit")
def package_audit(
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Where to stage the audit. Defaults to artifacts/kaggle-audit."),
    ] = None,
    expect_kernel: Annotated[
        str, typer.Option("--expect-kernel", help="Refuse unless the build targets exactly this kernel id.")
    ] = AUDIT_KERNEL,
    allow_dirty: Annotated[
        bool,
        typer.Option(
            "--allow-dirty", help="Build from an uncommitted tree. The package records it and is unpushable."
        ),
    ] = False,
) -> None:
    """Stage a CPU-only diagnostic that measures the Kaggle runtime, and push nothing.

    One traceback established that zarr is absent. It did not establish the
    Python version, platform tag or ABI a wheel would have to match, and a
    wheelhouse built on inference is a second wasted session waiting to happen.

    The diagnostic reads no competition data, trains nothing, requests no
    accelerator and imports no part of Biohub-X, so it runs on a bare image. It
    goes through the same conversion and isolated-execution gate as the training
    package before it is allowed to exist as a pushable artifact.
    """
    command = "package audit"
    import shutil
    import subprocess

    from biohubx.packaging import prepush
    from biohubx.packaging.audit import (
        AUDIT_ID,
        AUDIT_RUNTIME_CEILING_SECONDS,
        EXPECTED_AUDIT_STAGES,
        AuditError,
        AuditSpec,
        audit_kernel_metadata,
        build_audit_notebook,
    )
    from biohubx.packaging.kaggle import forbidden_content
    from biohubx.packaging.prepush import PrePushError

    root_path = repository_root()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root_path, capture_output=True, text=True, check=False
    )
    if revision.returncode != 0:
        heartbeat(command, "refused", "cannot resolve the repository commit; a package must pin one")
        raise typer.Exit(code=2)
    commit = revision.stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root_path, capture_output=True, text=True, check=False
    ).stdout.strip()
    if dirty and not allow_dirty:
        heartbeat(
            command,
            "refused",
            "the working tree is dirty, so the commit this audit would pin does not describe "
            "the bytes it ships; commit first, or pass --allow-dirty for an unpushable build",
        )
        raise typer.Exit(code=2)

    spec = AuditSpec(
        audit_id=AUDIT_ID,
        commit=commit,
        kernel=expect_kernel,
        runtime_ceiling_seconds=AUDIT_RUNTIME_CEILING_SECONDS,
    )
    heartbeat(command, "start", f"audit={spec.audit_id} kernel={spec.kernel} commit={commit[:12]}")

    try:
        notebook = build_audit_notebook(spec)
    except AuditError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    metadata = audit_kernel_metadata(kernel=spec.kernel)
    if metadata["enable_gpu"] or metadata["enable_internet"]:
        heartbeat(command, "refused", "the audit must be CPU only with internet disabled")
        raise typer.Exit(code=2)
    if any(metadata[key] for key in ("dataset_sources", "model_sources", "competition_sources")):
        heartbeat(command, "refused", "the audit must attach no sources of any kind")
        raise typer.Exit(code=2)

    staging = out if out is not None else root_path / "artifacts/kaggle-audit"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    atomic_write_text(staging / "run.ipynb", json.dumps(notebook, indent=1, sort_keys=True) + chr(10))
    atomic_write_text(
        staging / "kernel-metadata.json", json.dumps(metadata, indent=2, sort_keys=True) + chr(10)
    )
    atomic_write_text(
        staging / "audit-spec.json", json.dumps(spec.to_dict(), indent=2, sort_keys=True) + chr(10)
    )

    offenders = forbidden_content(staging)
    if offenders:
        heartbeat(command, "refused", f"the audit would ship forbidden content: {offenders[:5]}")
        raise typer.Exit(code=2)

    heartbeat(command, "pre-push", "validating, converting and running isolated before any network action")
    try:
        gate = prepush.run_all(
            notebook,
            spec.to_dict(),
            data_root=root_path,
            interpreter=sys.executable,
            expected=EXPECTED_AUDIT_STAGES,
            needs_data=False,
        )
    except PrePushError as exc:
        heartbeat(command, "refused", f"pre-push gate failed, nothing was sent: {exc}")
        raise typer.Exit(code=2) from exc
    heartbeat(
        command,
        "pre-push",
        f"nbformat ok, nbconvert produced {gate.converted_bytes} bytes, "
        f"{gate.stage_lines_seen} heartbeat lines across {len(gate.stage_names)} stages",
    )

    files = sorted(path for path in staging.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in files)
    listing = {
        str(path.relative_to(staging)).replace(chr(92), "/"): digest_file(path, DigestKind.RAW_ARTIFACT).token
        for path in files
    }
    manifest_text = json.dumps(
        {
            "schema_version": 1,
            "audit": spec.to_dict(),
            "kernel_metadata": metadata,
            "files": listing,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "contains_competition_bytes": False,
            "contains_external_weights": False,
            "imports_biohubx": False,
            "repository_clean_at_build": not dirty,
            "pushable": not dirty,
        },
        indent=2,
        sort_keys=True,
    )
    atomic_write_text(staging / "AUDIT_MANIFEST.json", manifest_text + chr(10))
    package_digest = digest_file(staging / "AUDIT_MANIFEST.json", DigestKind.CANONICAL_TEXT)

    push_command = (
        f"kaggle kernels push -p {staging.relative_to(root_path)}"
        if staging.is_relative_to(root_path)
        else f"kaggle kernels push -p {staging}"
    )
    payload = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "audit": spec.to_dict(),
        "package": {
            "path": str(staging.relative_to(root_path)),
            "manifest_digest": package_digest.token,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "push_command": push_command,
            "pushed": False,
        },
        "kernel_metadata": metadata,
        "pre_push_gate": gate.to_dict(),
        "probes": list(prepush_closure()),
    }
    report = root_path / "artifacts/kaggle-audit.json"
    atomic_write_text(report, json.dumps(payload, indent=2, sort_keys=True) + chr(10))
    _write_manifest(command, {"report": "artifacts/kaggle-audit.json", "audit": spec.audit_id})
    heartbeat(
        command,
        "staged",
        f"files={len(files)} bytes={total_bytes} digest={package_digest.token[:52]}",
    )
    heartbeat(command, "done", f"NOT PUSHED. push with: {push_command}")
    typer.echo(json.dumps(payload, sort_keys=True))


def prepush_closure() -> tuple[str, ...]:
    """The import names the audit probes, surfaced so the report lists them."""
    from biohubx.packaging.audit import RUNTIME_CLOSURE

    return RUNTIME_CLOSURE


@package_app.command("preflight")
def package_preflight(
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Where to stage it. Defaults to artifacts/kaggle-preflight."),
    ] = None,
    wheelhouse: Annotated[
        Path | None,
        typer.Option(
            "--wheelhouse", help="Local wheelhouse to validate against. Defaults to artifacts/wheelhouse."
        ),
    ] = None,
    expect_published: Annotated[
        str | None,
        typer.Option(
            "--expect-published",
            help="Refuse unless the published payload has this tree identity. State the authorised one.",
        ),
    ] = None,
    allow_dirty: Annotated[
        bool,
        typer.Option("--allow-dirty", help="Build from an uncommitted tree. Records it and is unpushable."),
    ] = False,
) -> None:
    """Stage the wheelhouse installation and readability preflight, and push nothing.

    Two questions in order: does the wheelhouse install offline on the measured
    image, and can the result decode a real competition chunk. A wheelhouse that
    installs but cannot decode blosc has answered neither, which is why the
    competition is attached for exactly one read.

    Before either question, the package checks that the tree it mounted is the one
    it was authorised against, using an identity and a walk it carries itself. The
    mounted dataset supplies bytes and nothing else, because
    ``requirements-offline.txt`` cannot authenticate itself: a substituted dataset
    shipping its own matching requirements file satisfies ``--require-hashes``
    exactly ([[D-0032]]).
    """
    command = "package preflight"
    import shutil
    import subprocess
    import tempfile

    from biohubx.hashing import tree_digest
    from biohubx.packaging.audit import AuditSpec
    from biohubx.packaging.kaggle import forbidden_content
    from biohubx.packaging.preflight import (
        PREFLIGHT_ID,
        PREFLIGHT_KERNEL,
        PREFLIGHT_RUNTIME_CEILING_SECONDS,
        WHEELHOUSE_SLUG,
        build_preflight_notebook,
        preflight_kernel_metadata,
    )
    from biohubx.packaging.wheelhouse import published_relative_paths, stage_published_payload

    root_path = repository_root()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root_path, capture_output=True, text=True, check=False
    )
    if revision.returncode != 0:
        heartbeat(command, "refused", "cannot resolve the repository commit; a package must pin one")
        raise typer.Exit(code=2)
    commit = revision.stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root_path, capture_output=True, text=True, check=False
    ).stdout.strip()
    if dirty and not allow_dirty:
        heartbeat(
            command,
            "refused",
            "the working tree is dirty, so the commit this package would pin does not describe "
            "the bytes it ships; commit first, or pass --allow-dirty for an unpushable build",
        )
        raise typer.Exit(code=2)

    house = wheelhouse if wheelhouse is not None else root_path / "artifacts/wheelhouse"
    wheels = sorted((house / "wheels").glob("*.whl")) if house.is_dir() else []
    if not wheels:
        heartbeat(command, "refused", f"no wheels under {house}; build the wheelhouse first")
        raise typer.Exit(code=2)
    sdists = [p.name for p in house.rglob("*") if p.suffix in {".gz", ".zip"} and p.is_file()]
    if sdists:
        heartbeat(command, "refused", f"the wheelhouse contains source distributions: {sdists[:3]}")
        raise typer.Exit(code=2)

    spec = AuditSpec(
        audit_id=PREFLIGHT_ID,
        commit=commit,
        kernel=PREFLIGHT_KERNEL,
        runtime_ceiling_seconds=PREFLIGHT_RUNTIME_CEILING_SECONDS,
    )
    heartbeat(command, "start", f"preflight={spec.audit_id} kernel={spec.kernel} wheels={len(wheels)}")

    # The identity the package will carry is the published payload's, not the
    # upload bundle's: the bundle includes dataset-metadata.json, which Kaggle
    # consumes rather than stores, so a kernel never mounts it (D-0031).
    bundle = tree_digest(house, on_file=_periodic_progress(command, "upload bundle"))
    with tempfile.TemporaryDirectory(prefix="biohubx-published-") as scratch:
        payload_root = Path(scratch) / "payload"
        payload_root.mkdir()
        stage_published_payload(house, payload_root, published_relative_paths(bundle))
        published = tree_digest(payload_root, on_file=_periodic_progress(command, "published payload"))
    if expect_published is not None and expect_published != published.digest.token:
        heartbeat(
            command,
            "refused",
            f"the wheelhouse publishes {published.digest.token}, not the authorised "
            f"{expect_published}; the package would carry an identity nobody approved",
        )
        raise typer.Exit(code=2)
    heartbeat(
        command,
        "identity",
        f"published payload files={published.file_count} bytes={published.total_bytes} "
        f"{published.digest.token}"
        + (" (pinned)" if expect_published is not None else " (NOT pinned to an authorisation)"),
    )

    notebook = build_preflight_notebook(
        spec,
        published_payload={
            "tree": published.digest.token,
            # Serialised exactly as TreeRecord.line does, including the "-" for an
            # absent field. `or "-"` would turn a zero-byte file into "-" and
            # silently disagree with the authoritative listing.
            "records": [
                [
                    record.kind,
                    "-" if record.content_sha256 is None else record.content_sha256,
                    "-" if record.size_bytes is None else str(record.size_bytes),
                    record.relative_path,
                ]
                for record in published.records
            ],
        },
    )
    metadata = preflight_kernel_metadata()
    if metadata["enable_gpu"] or metadata["enable_internet"]:
        heartbeat(command, "refused", "the preflight must be CPU only with internet disabled")
        raise typer.Exit(code=2)
    if metadata["dataset_sources"] != [WHEELHOUSE_SLUG]:
        heartbeat(command, "refused", "the preflight must attach the wheelhouse and nothing else")
        raise typer.Exit(code=2)

    staging = out if out is not None else root_path / "artifacts/kaggle-preflight"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    atomic_write_text(staging / "run.ipynb", json.dumps(notebook, indent=1, sort_keys=True) + chr(10))
    atomic_write_text(
        staging / "kernel-metadata.json", json.dumps(metadata, indent=2, sort_keys=True) + chr(10)
    )
    atomic_write_text(
        staging / "preflight-spec.json",
        json.dumps({**spec.to_dict(), "preflight_id": spec.audit_id}, indent=2, sort_keys=True) + chr(10),
    )

    offenders = forbidden_content(staging)
    if offenders:
        heartbeat(command, "refused", f"the preflight would ship forbidden content: {offenders[:5]}")
        raise typer.Exit(code=2)

    # The notebook is validated and converted, but not executed: it installs
    # packages and reads the corpus, neither of which belongs in a build step.
    from biohubx.packaging import prepush
    from biohubx.packaging.prepush import PrePushError

    try:
        prepush.check_required_metadata(notebook)
        prepush.validate_notebook(notebook)
        converted = prepush.convert_notebook(notebook)
    except PrePushError as exc:
        heartbeat(command, "refused", f"pre-push gate failed, nothing was sent: {exc}")
        raise typer.Exit(code=2) from exc
    heartbeat(command, "pre-push", f"nbformat ok, nbconvert produced {converted} bytes")

    files = sorted(path for path in staging.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in files)
    manifest_text = json.dumps(
        {
            "schema_version": 1,
            "preflight": {**spec.to_dict(), "preflight_id": spec.audit_id},
            "kernel_metadata": metadata,
            "wheelhouse_validated": {
                "path": str(house.relative_to(root_path)) if house.is_relative_to(root_path) else str(house),
                "wheels": [path.name for path in wheels],
                "source_distributions": [],
            },
            # Two identities, kept apart. The published payload is what the kernel
            # mounts and what the notebook carries and checks. The upload bundle is
            # what this machine sent; it is recorded for provenance and is not what
            # anything verifies at runtime, because no kernel ever sees it (D-0031).
            "wheelhouse_identity": {
                "published_payload": {
                    "tree": published.digest.token,
                    "file_count": published.file_count,
                    "total_bytes": published.total_bytes,
                    "embedded_in_the_notebook": True,
                    "verified_at_runtime_before_pip": True,
                    "pinned_to_authorisation": expect_published is not None,
                },
                "upload_bundle": {
                    "tree": bundle.digest.token,
                    "file_count": bundle.file_count,
                    "total_bytes": bundle.total_bytes,
                    "embedded_in_the_notebook": False,
                    "verified_at_runtime_before_pip": False,
                    "why_not": "Kaggle consumes dataset-metadata.json, so no kernel mounts this tree",
                },
            },
            "files": {
                path.relative_to(staging).as_posix(): digest_file(path, DigestKind.RAW_ARTIFACT).token
                for path in files
            },
            "file_count": len(files),
            "total_bytes": total_bytes,
            "contains_competition_bytes": False,
            "contains_external_weights": False,
            "repository_clean_at_build": not dirty,
            "pushable": not dirty,
        },
        indent=2,
        sort_keys=True,
    )
    atomic_write_text(staging / "PREFLIGHT_MANIFEST.json", manifest_text + chr(10))
    package_digest = digest_file(staging / "PREFLIGHT_MANIFEST.json", DigestKind.CANONICAL_TEXT)

    payload = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "preflight": {**spec.to_dict(), "preflight_id": spec.audit_id},
        "package": {
            "path": str(staging.relative_to(root_path)),
            "manifest_digest": package_digest.token,
            "file_count": len(files) + 1,
            "total_bytes": total_bytes,
            "push_command": f"kaggle kernels push -p {staging.relative_to(root_path)}",
            "pushed": False,
        },
        "kernel_metadata": metadata,
        "nbconvert_bytes": converted,
        "wheelhouse_identity": {
            "published_payload": published.digest.token,
            "published_payload_pinned_to_authorisation": expect_published is not None,
            "upload_bundle": bundle.digest.token,
        },
    }
    atomic_write_text(
        root_path / "artifacts/kaggle-preflight.json", json.dumps(payload, indent=2, sort_keys=True) + chr(10)
    )
    _write_manifest(command, {"report": "artifacts/kaggle-preflight.json", "preflight": spec.audit_id})
    heartbeat(command, "staged", f"files={len(files) + 1} digest={package_digest.token[:52]}")
    heartbeat(command, "done", "NOT PUSHED; the mounted tree is verified before pip runs")
    typer.echo(json.dumps(payload, sort_keys=True))


@package_app.command("transport")
def package_transport(
    package: Annotated[
        Path | None,
        typer.Option("--package", help="Staged package. Defaults to artifacts/kaggle-package."),
    ] = None,
    expect_kernel: Annotated[
        str,
        typer.Option("--expect-kernel", help="The kernel address the push must create."),
    ] = "aryaarun07/biohub-x-e03-fold-44b6",
    expect_digest: Annotated[
        str | None,
        typer.Option("--expect-digest", help="Refuse unless the staged manifest has this digest."),
    ] = None,
    push: Annotated[
        bool,
        typer.Option("--push", help="Actually send it. Refused without --campaign-envelope."),
    ] = False,
    campaign_envelope: Annotated[
        str | None,
        typer.Option("--campaign-envelope", help="The approval this push is drawn against."),
    ] = None,
    accelerator: Annotated[
        str,
        typer.Option("--accelerator", help="Accelerator requested on the command line as well."),
    ] = "NvidiaTeslaT4",
    expect_dataset: Annotated[
        str | None,
        typer.Option(
            "--expect-dataset",
            help="The one dataset the kernel may attach. Defaults to the zarr wheelhouse.",
        ),
    ] = None,
    timeout: Annotated[int, typer.Option("--timeout", help="Seconds Kaggle may run the kernel for.")] = 2400,
    interpreter: Annotated[
        str,
        typer.Option("--interpreter", help="Launcher for the Kaggle CLI, invoked as a module."),
    ] = "py -3.14",
) -> None:
    """Validate a staged package for transport, and stop before sending it.

    Everything here runs before any network call, because a package that is wrong
    costs a GPU session to discover remotely and nothing to discover locally. The
    checks are: the manifest digest re-derived from its bytes, every staged file
    re-derived against the manifest's inventory, no file the builder did not
    generate, no quarantined name or foreign path, no external checkpoint, exactly
    the registered wheelhouse and the official competition as sources, and no
    Docker image pin, because Biohub-X approves none and an unapproved one is an
    unreviewed runtime.

    Sending is a separate act and is refused unless ``--push`` is given together
    with the campaign envelope it is drawn against. When it does send, it invokes
    the CLI as a module, captures both streams, parses the kernel URL Kaggle
    returns, and treats any divergence between that address and the expected one
    as a failed attempt.

    The credentials file belongs to the Kaggle CLI alone. Nothing here reads,
    prints, copies or records it.
    """
    command = "package transport"
    import re
    import subprocess

    from biohubx.packaging.kaggle import forbidden_content
    from biohubx.packaging.preflight import WHEELHOUSE_SLUG

    expected_dataset = expect_dataset or WHEELHOUSE_SLUG
    root_path = repository_root()
    # Resolved, because the report names the package relative to the repository
    # and a relative --package would fail that at the very end, after validation.
    staging = (package if package is not None else root_path / "artifacts/kaggle-package").resolve()
    manifest_path = staging / "PACKAGE_MANIFEST.json"
    if not manifest_path.is_file():
        heartbeat(command, "refused", f"no PACKAGE_MANIFEST.json under {staging}")
        raise typer.Exit(code=2)

    failures: list[str] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = digest_file(manifest_path, DigestKind.CANONICAL_TEXT).token
    heartbeat(command, "digest", digest)
    if expect_digest is not None and digest != expect_digest:
        failures.append(f"staged manifest is {digest}, not the authorised {expect_digest}")

    # Every staged file must be one the builder recorded, and vice versa. A file
    # the manifest does not name is a file nobody reviewed.
    recorded = dict(manifest.get("files", {}))
    on_disk = {
        path.relative_to(staging).as_posix(): path for path in sorted(staging.rglob("*")) if path.is_file()
    }
    unexpected = sorted(set(on_disk) - set(recorded) - {"PACKAGE_MANIFEST.json"})
    absent = sorted(set(recorded) - set(on_disk))
    if unexpected:
        failures.append(f"files the builder did not generate: {unexpected[:5]}")
    if absent:
        failures.append(f"files the manifest names but the directory lacks: {absent[:5]}")
    for name, token in sorted(recorded.items()):
        if name in on_disk and digest_file(on_disk[name], DigestKind.RAW_ARTIFACT).token != token:
            failures.append(f"{name} does not match the digest the manifest recorded")
    heartbeat(command, "inventory", f"files={len(on_disk)} recorded={len(recorded)}")

    # Names and paths that must never travel. The authoritative quarantine list is
    # the constant in the isolation contract test, so it is read from there rather
    # than copied here: two lists would drift, and the copy would be the one that
    # silently stopped matching. src/ does not import tests/, so it is parsed.
    import ast

    isolation = root_path / "tests/contracts/test_repository_isolation.py"
    quarantined_names: tuple[str, ...] = ()
    for statement in ast.parse(isolation.read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "QUARANTINED_NAMES" for target in statement.targets
        ):
            quarantined_names = tuple(
                element.value
                for element in getattr(statement.value, "elts", [])
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
    if not quarantined_names:
        failures.append("could not read the authoritative quarantine list; refusing rather than guessing")

    text_blob = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in on_disk.values()
        if path.suffix in {".json", ".ipynb", ".py", ".yaml", ".txt"}
    )
    for quarantined in quarantined_names:
        if quarantined in text_blob:
            failures.append(f"a quarantined name appears in the staged package: {quarantined}")
    # An external checkpoint is refused as a FILE and as an attached SOURCE, never
    # as a mentioned string. The package legitimately ships the guard code that
    # names `reference.pilkwang` and refuses `.pth`, so scanning text for those
    # names flags the very thing protecting the run. That is the adjacent check
    # again: the name is not the artifact.
    staged_offenders = forbidden_content(staging)
    if staged_offenders:
        failures.append(f"the package stages forbidden content: {staged_offenders[:5]}")

    metadata = json.loads((staging / "kernel-metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_sources") != [expected_dataset]:
        failures.append(
            f"dataset sources are {metadata.get('dataset_sources')}, "
            f"not the expected {expected_dataset} alone"
        )
    if metadata.get("competition_sources") != ["biohub-cell-tracking-during-development"]:
        failures.append(f"competition sources are {metadata.get('competition_sources')}")
    if metadata.get("model_sources") or metadata.get("kernel_sources"):
        failures.append("model or kernel sources are attached and none is approved")
    attached = list(metadata.get("dataset_sources", [])) + list(metadata.get("model_sources", []))
    external_weights = sorted(name for name in attached if name != expected_dataset)
    if external_weights:
        failures.append(f"an unapproved external source is attached: {external_weights[:3]}")
    if metadata.get("enable_internet") is not False:
        failures.append("internet is not disabled")
    if "docker_image" in metadata:
        failures.append("the package pins a Docker image and Biohub-X has approved none")
    if metadata.get("id") != expect_kernel:
        failures.append(f"kernel id is {metadata.get('id')!r}, not {expect_kernel!r}")
    heartbeat(command, "metadata", f"id={metadata.get('id')} sources ok={not failures}")

    for line in failures:
        heartbeat(command, "refused", line)
    if failures:
        raise typer.Exit(code=2)
    heartbeat(command, "validated", f"{len(on_disk)} files, every check passed")

    pushed: dict[str, object] = {"attempted": False}
    if push:
        if not campaign_envelope:
            heartbeat(
                command,
                "refused",
                "--push needs --campaign-envelope naming the approval it is drawn against; "
                "sending is a separate authorisation from building",
            )
            raise typer.Exit(code=2)
        heartbeat(command, "push", f"envelope={campaign_envelope} kernel={expect_kernel}")
        environment = dict(os.environ)
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        # The CLI is invoked as a module through the launcher, not through the
        # console script: kaggle is not importable in this project's virtualenv,
        # and the .exe shim is not what the envelope authorises. The accelerator
        # and ceiling are passed on the command line as well as carried in the
        # metadata, so a request that one form drops is still made by the other.
        # A CPU package must not request an accelerator, and a GPU package must.
        # The Kaggle API sets machine_shape from this flag when it is given, so
        # for a CPU kernel the flag is omitted rather than sent as a name the
        # platform might read as a request.
        wants_gpu = bool(metadata.get("enable_gpu"))
        accelerator_flag = None if accelerator.strip().lower() in {"", "none"} else accelerator
        if wants_gpu and accelerator_flag is None:
            heartbeat(command, "refused", "the package enables a GPU but no accelerator was requested")
            raise typer.Exit(code=2)
        if not wants_gpu and accelerator_flag is not None:
            heartbeat(
                command, "refused", f"the package is CPU only but --accelerator {accelerator} was requested"
            )
            raise typer.Exit(code=2)
        argv = [*interpreter.split(), "-m", "kaggle", "kernels", "push", "-p", str(staging)]
        if accelerator_flag is not None:
            argv += ["--accelerator", accelerator_flag]
        argv += ["--timeout", str(timeout)]
        heartbeat(command, "push", " ".join(argv))
        completed = subprocess.run(argv, capture_output=True, text=True, env=environment, check=False)
        found = re.search(r"kaggle\.com/code/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", completed.stdout or "")
        observed = f"{found.group(1)}/{found.group(2)}" if found else ""
        pushed = {
            "attempted": True,
            "envelope": campaign_envelope,
            "argv": argv,
            "accelerator_requested": accelerator_flag or "none (CPU, metadata-driven)",
            "timeout_seconds": timeout,
            "returncode": completed.returncode,
            "stdout_tail": (completed.stdout or "")[-1500:],
            "stderr_tail": (completed.stderr or "")[-1500:],
            "observed_kernel": observed,
            "expected_kernel": expect_kernel,
            "slug_matches": observed == expect_kernel,
        }
        if completed.returncode != 0 or observed != expect_kernel:
            heartbeat(
                command,
                "refused",
                f"push failed or diverged: returncode={completed.returncode} observed={observed!r}",
            )
            atomic_write_text(
                root_path / "artifacts/kaggle-transport.json",
                json.dumps({"schema_version": 1, "validated": True, "push": pushed}, indent=2, sort_keys=True)
                + chr(10),
            )
            raise typer.Exit(code=2)
        heartbeat(command, "push", f"created {observed}")

    payload = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "package": str(staging.relative_to(root_path)),
        "manifest_digest": digest,
        "file_count": len(on_disk),
        "expected_kernel": expect_kernel,
        "kernel_metadata": metadata,
        "checks_passed": True,
        "push": pushed,
    }
    atomic_write_text(
        root_path / "artifacts/kaggle-transport.json", json.dumps(payload, indent=2, sort_keys=True) + chr(10)
    )
    _write_manifest(command, {"report": "artifacts/kaggle-transport.json", "digest": digest})
    heartbeat(command, "done", "NOT PUSHED" if not push else "pushed once")
    typer.echo(json.dumps(payload, sort_keys=True))


@package_app.command("metric-preflight")
def package_metric_preflight(
    wheelhouse: Annotated[
        Path | None,
        typer.Option("--wheelhouse", help="The metric wheelhouse. Defaults to artifacts/wheelhouse-metric."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Where to stage it. Defaults to artifacts/kaggle-metric-preflight."),
    ] = None,
    expect_published: Annotated[
        str | None,
        typer.Option(
            "--expect-published", help="Refuse unless the published payload has this tree identity."
        ),
    ] = None,
    root: Annotated[
        Path | None,
        typer.Option(
            "--root",
            help="Competition data root for the local baseline. BIOHUB_DATA_ROOT is the only fallback.",
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            help="Movie for the tiny real-data score. Defaults to the first annotated 44b6 movie.",
        ),
    ] = None,
    frames: Annotated[int, typer.Option("--frames", help="Frames in the real-data window.")] = 2,
    allow_dirty: Annotated[
        bool,
        typer.Option("--allow-dirty", help="Build from an uncommitted tree. Records it and is unpushable."),
    ] = False,
) -> None:
    """Stage the metric preflight: five checks on the target, baseline computed here, nothing sent.

    Strict imports of the whole closure, the vendored official source hashed
    against the registry, the scorer's characterisation fixtures, behavioural
    equivalence against this machine, and one tiny real-data score through the
    frozen proposals of D-0041. The baseline the target must reproduce is computed
    here first and travels inside the spec, so the kernel compares rather than
    merely reports.
    """
    command = "package metric-preflight"
    import shutil
    import subprocess
    import tempfile

    import yaml

    from biohubx.data.competition import CompetitionLayoutError, competition_root
    from biohubx.hashing import tree_digest
    from biohubx.packaging import prepush
    from biohubx.packaging.audit import AuditSpec
    from biohubx.packaging.kaggle import (
        DEFAULT_PACKAGE_ROOT,
        PackagingError,
        archive_digest,
        deterministic_archive,
        forbidden_content,
    )
    from biohubx.packaging.metric_preflight import (
        METRIC_PREFLIGHT_ID,
        METRIC_PREFLIGHT_KERNEL,
        METRIC_PREFLIGHT_RUNTIME_CEILING_SECONDS,
        METRIC_WHEELHOUSE_SLUG,
        build_metric_preflight_notebook,
        metric_preflight_kernel_metadata,
    )
    from biohubx.packaging.metric_preflight_runtime import (
        EXPECTED_ABSENT,
        FROZEN_PROPOSALS,
        OFFICIAL_SOURCE_FILES,
        local_baseline,
    )
    from biohubx.packaging.prepush import PrePushError
    from biohubx.packaging.wheelhouse import (
        WheelhouseError,
        published_relative_paths,
        requirement_hashes,
        stage_published_payload,
    )

    root_path = repository_root()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root_path, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root_path, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty and not allow_dirty:
        heartbeat(
            command,
            "refused",
            "the working tree is dirty, so the commit this package would pin does not describe "
            "the bytes it ships; commit first, or pass --allow-dirty for an unpushable build",
        )
        raise typer.Exit(code=2)

    house = wheelhouse if wheelhouse is not None else root_path / "artifacts/wheelhouse-metric"
    wheels = sorted((house / "wheels").glob("*.whl")) if house.is_dir() else []
    if not wheels:
        heartbeat(command, "refused", f"no wheels under {house}; assemble the wheelhouse first")
        raise typer.Exit(code=2)
    sdists = [p.name for p in house.rglob("*") if p.suffix in {".gz", ".zip"} and p.is_file()]
    if sdists:
        heartbeat(command, "refused", f"the wheelhouse contains source distributions: {sdists[:3]}")
        raise typer.Exit(code=2)
    requirements_path = house / "requirements-offline.txt"
    try:
        required = requirement_hashes(requirements_path.read_text(encoding="utf-8"))
    except (OSError, WheelhouseError) as exc:
        heartbeat(command, "refused", f"cannot read what pip would enforce: {exc}")
        raise typer.Exit(code=2) from exc
    shipped_versions = {
        name: {"version": version, "sha256": sha} for name, (version, sha) in sorted(required.items())
    }

    spec = AuditSpec(
        audit_id=METRIC_PREFLIGHT_ID,
        commit=commit,
        kernel=METRIC_PREFLIGHT_KERNEL,
        runtime_ceiling_seconds=METRIC_PREFLIGHT_RUNTIME_CEILING_SECONDS,
    )
    heartbeat(command, "start", f"preflight={spec.audit_id} kernel={spec.kernel} wheels={len(wheels)}")

    bundle = tree_digest(house, on_file=_periodic_progress(command, "upload bundle"))
    with tempfile.TemporaryDirectory(prefix="biohubx-published-") as scratch:
        payload_root = Path(scratch) / "payload"
        payload_root.mkdir()
        stage_published_payload(house, payload_root, published_relative_paths(bundle))
        published = tree_digest(payload_root, on_file=_periodic_progress(command, "published payload"))
    if expect_published is not None and expect_published != published.digest.token:
        heartbeat(
            command,
            "refused",
            f"the wheelhouse publishes {published.digest.token}, not the authorised "
            f"{expect_published}; the package would carry an identity nobody approved",
        )
        raise typer.Exit(code=2)
    heartbeat(
        command,
        "identity",
        f"published payload files={published.file_count} bytes={published.total_bytes} "
        f"{published.digest.token}"
        + (" (pinned)" if expect_published is not None else " (NOT pinned to an authorisation)"),
    )

    # The registry's digests for the vendored official source, keyed the way the
    # runtime hashes them: relative to the biohubx package directory.
    official = yaml.safe_load((root_path / "registry/official_source.yaml").read_text(encoding="utf-8"))
    official_expected: dict[str, str] = {}
    for item in official.get("sources", []) if isinstance(official, dict) else []:
        vendored = str(item.get("vendored_path", ""))
        relative = vendored.removeprefix("src/biohubx/")
        if relative in OFFICIAL_SOURCE_FILES:
            official_expected[relative] = str(item["digests"]["raw"]).removeprefix(
                "raw_artifact_sha256:sha256:"
            )
    missing_digests = sorted(set(OFFICIAL_SOURCE_FILES) - set(official_expected))
    if missing_digests:
        heartbeat(command, "refused", f"registry/official_source.yaml pins no digest for {missing_digests}")
        raise typer.Exit(code=2)

    try:
        data_root = competition_root(root)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    movie = dataset
    if movie is None:
        candidates = sorted(p.stem for p in (data_root / "train").glob("44b6*.geff"))
        if not candidates:
            heartbeat(command, "refused", "no annotated 44b6 movie under the data root to score")
            raise typer.Exit(code=2)
        movie = candidates[0]
    heartbeat(command, "baseline", f"fixtures, official source, and {movie} x {frames} frames, locally")
    try:
        baseline = local_baseline(
            data_root=data_root,
            dataset_id=movie,
            frames=frames,
            package_dir=root_path / "src/biohubx",
            official_expected=official_expected,
            shipped_versions=shipped_versions,
        )
    except (ValueError, OSError, CompetitionLayoutError) as exc:
        heartbeat(command, "refused", f"the local baseline could not be computed: {exc}")
        raise typer.Exit(code=2) from exc
    real = baseline["real_data"]
    heartbeat(
        command,
        "baseline",
        f"fixtures={baseline['fixtures_count']} digest={baseline['fixtures_digest'][:16]} "
        f"real proposals={real['proposals']} matched={real['matched_nodes']}/{real['annotated_nodes']} "
        f"score={real.get('score')}",
    )

    archive_bytes = deterministic_archive(root_path / "src/biohubx")
    shipped = {
        **spec.to_dict(),
        "preflight_id": spec.audit_id,
        "payload_digest": archive_digest(archive_bytes),
        "package_root": DEFAULT_PACKAGE_ROOT,
        "frozen_proposals": FROZEN_PROPOSALS,
        "expected_absent": list(EXPECTED_ABSENT),
        "wheelhouse_slug": METRIC_WHEELHOUSE_SLUG,
        "baseline": baseline,
    }
    try:
        notebook = build_metric_preflight_notebook(
            shipped,
            payload=archive_bytes,
            wheelhouse_payload={
                "tree": published.digest.token,
                "records": [
                    [
                        record.kind,
                        "-" if record.content_sha256 is None else record.content_sha256,
                        "-" if record.size_bytes is None else str(record.size_bytes),
                        record.relative_path,
                    ]
                    for record in published.records
                ],
            },
        )
        metadata = metric_preflight_kernel_metadata()
    except PackagingError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    if metadata["enable_gpu"] or metadata["enable_internet"]:
        heartbeat(command, "refused", "the metric preflight must be CPU only with internet disabled")
        raise typer.Exit(code=2)
    if metadata["dataset_sources"] != [METRIC_WHEELHOUSE_SLUG]:
        heartbeat(
            command, "refused", "the metric preflight must attach the metric wheelhouse and nothing else"
        )
        raise typer.Exit(code=2)

    staging = out if out is not None else root_path / "artifacts/kaggle-metric-preflight"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    atomic_write_text(staging / "run.ipynb", json.dumps(notebook, indent=1, sort_keys=True) + chr(10))
    atomic_write_text(
        staging / "kernel-metadata.json", json.dumps(metadata, indent=2, sort_keys=True) + chr(10)
    )
    atomic_write_text(
        staging / "preflight-spec.json", json.dumps(shipped, indent=2, sort_keys=True) + chr(10)
    )

    offenders = forbidden_content(staging)
    if offenders:
        heartbeat(command, "refused", f"the preflight would ship forbidden content: {offenders[:5]}")
        raise typer.Exit(code=2)
    try:
        prepush.check_required_metadata(notebook)
        prepush.validate_notebook(notebook)
        converted = prepush.convert_notebook(notebook)
    except PrePushError as exc:
        heartbeat(command, "refused", f"pre-push gate failed, nothing was sent: {exc}")
        raise typer.Exit(code=2) from exc
    heartbeat(command, "pre-push", f"nbformat ok, nbconvert produced {converted} bytes")

    files = sorted(path for path in staging.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in files)
    manifest_text = json.dumps(
        {
            "schema_version": 1,
            "preflight": {**spec.to_dict(), "preflight_id": spec.audit_id},
            "kernel_metadata": metadata,
            "wheelhouse_validated": {
                "path": str(house.relative_to(root_path)) if house.is_relative_to(root_path) else str(house),
                "wheels": [path.name for path in wheels],
                "source_distributions": [],
            },
            "wheelhouse_identity": {
                "published_payload": {
                    "tree": published.digest.token,
                    "file_count": published.file_count,
                    "total_bytes": published.total_bytes,
                    "embedded_in_the_notebook": True,
                    "verified_at_runtime_before_pip": True,
                    "pinned_to_authorisation": expect_published is not None,
                },
                "upload_bundle": {
                    "tree": bundle.digest.token,
                    "file_count": bundle.file_count,
                    "total_bytes": bundle.total_bytes,
                    "embedded_in_the_notebook": False,
                    "verified_at_runtime_before_pip": False,
                    "why_not": "Kaggle consumes dataset-metadata.json, so no kernel mounts this tree",
                },
            },
            "payload": {"digest": shipped["payload_digest"], "bytes": len(archive_bytes)},
            "baseline": {
                "fixtures_digest": baseline["fixtures_digest"],
                "fixtures_count": baseline["fixtures_count"],
                "real_data": real,
                "official_source": official_expected,
            },
            "files": {
                path.relative_to(staging).as_posix(): digest_file(path, DigestKind.RAW_ARTIFACT).token
                for path in files
            },
            "file_count": len(files),
            "total_bytes": total_bytes,
            "contains_competition_bytes": False,
            "contains_external_weights": False,
            "repository_clean_at_build": not dirty,
            "pushable": not dirty,
        },
        indent=2,
        sort_keys=True,
    )
    atomic_write_text(staging / "PACKAGE_MANIFEST.json", manifest_text + chr(10))
    package_digest = digest_file(staging / "PACKAGE_MANIFEST.json", DigestKind.CANONICAL_TEXT)

    payload = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "preflight": {**spec.to_dict(), "preflight_id": spec.audit_id},
        "package": {
            "path": str(staging.relative_to(root_path)),
            "manifest_digest": package_digest.token,
            "file_count": len(files) + 1,
            "total_bytes": total_bytes,
            "pushed": False,
        },
        "kernel_metadata": metadata,
        "nbconvert_bytes": converted,
        "wheelhouse_identity": {
            "published_payload": published.digest.token,
            "published_payload_pinned_to_authorisation": expect_published is not None,
            "upload_bundle": bundle.digest.token,
        },
        "baseline": {
            "fixtures_digest": baseline["fixtures_digest"],
            "real_data": real,
            "local_environment": baseline["local_environment"],
        },
    }
    atomic_write_text(
        root_path / "artifacts/kaggle-metric-preflight.json",
        json.dumps(payload, indent=2, sort_keys=True) + chr(10),
    )
    _write_manifest(command, {"report": "artifacts/kaggle-metric-preflight.json", "preflight": spec.audit_id})
    heartbeat(command, "staged", f"files={len(files) + 1} digest={package_digest.token[:52]}")
    heartbeat(command, "done", "NOT PUSHED; the mounted tree is verified before pip runs")
    typer.echo(json.dumps(payload, sort_keys=True))


@package_app.command("retrieve")
def package_retrieve(
    kernel: Annotated[str, typer.Option("--kernel", help="Kernel id, owner/slug.")],
    only: Annotated[
        list[str],
        typer.Option("--only", help="An output filename to keep. Repeatable. Nothing else is kept."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory to place the kept files in.")],
    allow_missing: Annotated[
        bool,
        typer.Option("--allow-missing", help="Do not refuse when a named file is absent."),
    ] = False,
) -> None:
    """Retrieve only the named outputs of a kernel, and account for everything else.

    An authorisation naming two artifacts should not be able to pull a third.
    ``kaggle kernels output`` has no per-file mode: it fetches the whole output
    directory. Retrieving E03-WHEELHOUSE-PREFLIGHT-01 attempt 2 therefore also
    brought back the run's own roundtrip.zarr, which nobody had asked for and
    which had to be deleted afterwards.

    **This is containment, not prevention, and the difference is recorded rather
    than glossed.** The platform still sends every output; what changes is that
    they land in a temporary directory, only the named files are moved to ``out``,
    the rest are deleted, and the report lists every file that arrived with what
    happened to it. A reviewer can see what was discarded instead of taking it on
    trust, which is the honest version of a guarantee this CLI cannot make.
    """
    command = "package retrieve"
    import shutil
    import subprocess
    import tempfile

    root_path = repository_root()
    wanted = sorted(set(only))
    if not wanted:
        heartbeat(command, "refused", "no --only given; a retrieval that names nothing keeps nothing")
        raise typer.Exit(code=2)
    if any("/" in name or "\\" in name or name in {".", ".."} for name in wanted):
        heartbeat(command, "refused", f"--only takes bare filenames, got {wanted}")
        raise typer.Exit(code=2)

    heartbeat(command, "start", f"kernel={kernel} keeping={wanted}")
    with tempfile.TemporaryDirectory(prefix="biohubx-retrieve-") as scratch:
        quarantine = Path(scratch)
        completed = subprocess.run(
            ["kaggle", "kernels", "output", kernel, "-p", str(quarantine)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            heartbeat(
                command, "refused", f"kaggle returned {completed.returncode}: {completed.stderr[-300:]}"
            )
            raise typer.Exit(code=2)

        arrived = sorted(p for p in quarantine.rglob("*") if p.is_file())
        kept: list[dict[str, object]] = []
        discarded: list[str] = []
        out.mkdir(parents=True, exist_ok=True)
        for item in arrived:
            relative = item.relative_to(quarantine).as_posix()
            if item.name in wanted and item.parent == quarantine:
                destination = out / item.name
                shutil.copyfile(item, destination)
                kept.append(
                    {
                        "name": item.name,
                        "bytes": destination.stat().st_size,
                        "digest": digest_file(destination, DigestKind.RAW_ARTIFACT).token,
                    }
                )
            else:
                discarded.append(relative)
        missing = sorted(set(wanted) - {str(entry["name"]) for entry in kept})

    for entry in kept:
        heartbeat(command, "kept", f"{entry['name']} bytes={entry['bytes']} {entry['digest']}")
    for name in discarded:
        heartbeat(command, "discarded", name)
    if missing and not allow_missing:
        heartbeat(command, "refused", f"named outputs the kernel did not produce: {missing}")
        raise typer.Exit(code=2)

    payload = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "kernel": kernel,
        "requested": wanted,
        "kept": kept,
        "discarded": discarded,
        "missing": missing,
        "destination": str(out),
        "containment_note": (
            "The Kaggle CLI fetches a kernel's whole output directory; there is no per-file mode. "
            "Everything arrived in a temporary directory, only the named files were copied out, and "
            "the rest were deleted with the directory. Discarded names are listed above so the "
            "difference between what was sent and what was kept is auditable."
        ),
    }
    report_path = root_path / "artifacts/kaggle-retrieve.json"
    atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True) + chr(10))
    _write_manifest(command, {"report": str(report_path), "kernel": kernel, "kept": len(kept)})
    heartbeat(command, "done", f"kept={len(kept)} discarded={len(discarded)} into {out}")
    typer.echo(json.dumps(payload, sort_keys=True))


@package_app.command("wheelhouse")
def package_wheelhouse(
    path: Annotated[
        Path | None,
        typer.Option("--path", help="The staged wheelhouse. Defaults to artifacts/wheelhouse."),
    ] = None,
    remote: Annotated[
        Path | None,
        typer.Option(
            "--remote",
            help="A downloaded copy of the published dataset, checked against the published payload.",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Report path. Defaults to artifacts/wheelhouse-identity.json."),
    ] = None,
    built_wheel: Annotated[
        list[str] | None,
        typer.Option(
            "--built-wheel",
            help=(
                "FILENAME=sha256:HEX for a wheel Biohub-X built itself; the hash comes from its "
                "registry entry, not from uv.lock. Repeatable."
            ),
        ),
    ] = None,
) -> None:
    """Record what the wheelhouse is, as the two different trees it actually is.

    The upload bundle is what this machine sends. The published payload is what a
    kernel mounts. Kaggle consumes ``dataset-metadata.json`` as configuration
    instead of storing it, so the two differ by exactly that file and one digest
    would name neither tree. Both are computed here from one walk of one
    directory, so an inventory difference reads as the platform behaviour it is
    rather than being investigated as a substitution.

    With ``--remote``, a downloaded copy of the published dataset is compared with
    the published payload by name, size and content, and any difference refuses.

    uv.lock is the authority for every wheel except one kind: a wheel Biohub-X
    built from a git source the lock pins without an artifact (R-0014). Such a
    wheel is accepted only when named with ``--built-wheel`` and the digest its
    registry entry records, so the authority is stated by the operator and
    recorded in the report as ``biohubx_build`` rather than inferred.
    """
    command = "package wheelhouse"
    import tempfile

    from biohubx.hashing import tree_digest
    from biohubx.packaging.wheelhouse import (
        KAGGLE_CONFIGURATION_FILENAME,
        WheelhouseError,
        canonical_package_name,
        compare_manifests,
        locked_wheels,
        published_relative_paths,
        requirement_hashes,
        stage_published_payload,
    )

    root_path = repository_root()
    house = path if path is not None else root_path / "artifacts/wheelhouse"
    if not house.is_dir():
        heartbeat(command, "refused", f"no wheelhouse directory at {house}")
        raise typer.Exit(code=2)

    wheels_dir = house / "wheels"
    wheel_paths = sorted(wheels_dir.glob("*.whl")) if wheels_dir.is_dir() else []
    if not wheel_paths:
        heartbeat(command, "refused", f"no wheels under {wheels_dir}; there is nothing to identify")
        raise typer.Exit(code=2)
    sdists = [item.name for item in house.rglob("*") if item.suffix in {".gz", ".zip"} and item.is_file()]
    if sdists:
        heartbeat(command, "refused", f"the wheelhouse contains source distributions: {sdists[:3]}")
        raise typer.Exit(code=2)

    requirements_path = house / "requirements-offline.txt"
    if not requirements_path.is_file():
        heartbeat(command, "refused", f"no {requirements_path.name}; nothing states what pip would enforce")
        raise typer.Exit(code=2)

    heartbeat(command, "start", f"path={house} wheels={len(wheel_paths)}")

    bundle = tree_digest(house, on_file=_periodic_progress(command, "upload bundle"))
    heartbeat(
        command,
        "upload-bundle",
        f"files={bundle.file_count} bytes={bundle.total_bytes} {bundle.digest.token}",
    )

    configuration = [
        record
        for record in bundle.records
        if record.kind == "f" and record.relative_path == KAGGLE_CONFIGURATION_FILENAME
    ]
    if not configuration:
        heartbeat(
            command,
            "refused",
            f"{KAGGLE_CONFIGURATION_FILENAME} is absent from the root, so this is not an upload "
            "bundle and the two identities would name one tree",
        )
        raise typer.Exit(code=2)

    relative_paths = published_relative_paths(bundle)
    with tempfile.TemporaryDirectory(prefix="biohubx-published-") as scratch:
        staged = Path(scratch) / "payload"
        staged.mkdir()
        stage_published_payload(house, staged, relative_paths)
        published = tree_digest(staged, on_file=_periodic_progress(command, "published payload"))
    heartbeat(
        command,
        "published-payload",
        f"files={published.file_count} bytes={published.total_bytes} {published.digest.token}",
    )

    built: dict[str, str] = {}
    for item in built_wheel or []:
        filename, separator, stated_hash = item.partition("=")
        if not separator or not stated_hash.startswith("sha256:") or len(stated_hash) != 71:
            heartbeat(command, "refused", f"--built-wheel takes FILENAME=sha256:HEX, got {item!r}")
            raise typer.Exit(code=2)
        if filename not in {p.name for p in wheel_paths}:
            heartbeat(command, "refused", f"--built-wheel names {filename}, which is not staged")
            raise typer.Exit(code=2)
        built[filename] = stated_hash.removeprefix("sha256:")
    try:
        locked = locked_wheels(
            (root_path / "uv.lock").read_text(encoding="utf-8"),
            [item.name for item in wheel_paths if item.name not in built],
        )
        required = requirement_hashes(requirements_path.read_text(encoding="utf-8"))
    except WheelhouseError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    staged_content = {record.relative_path: record for record in bundle.records if record.kind == "f"}
    wheel_rows: list[dict[str, object]] = []
    wheel_failures: list[str] = []
    for wheel_path in wheel_paths:
        record = staged_content[f"wheels/{wheel_path.name}"]
        if wheel_path.name in built:
            # A built wheel: the bytes on disk must be the registered build, and
            # requirements-offline.txt must enforce that same hash, or pip would
            # accept a wheel the registry never described.
            stem_package, _, stem_version = (
                wheel_path.name.split("-", 2)[0],
                "",
                wheel_path.name.split("-", 2)[1],
            )
            canonical = canonical_package_name(stem_package)
            stated = required.get(canonical)
            matches_build = record.content_sha256 == built[wheel_path.name]
            matches_requirements = stated is not None and stated[1] == built[wheel_path.name]
            if not matches_build:
                wheel_failures.append(f"{wheel_path.name} does not match the registered build digest")
            if stated is None:
                wheel_failures.append(f"{canonical} is staged but {requirements_path.name} omits it")
            elif not matches_requirements:
                wheel_failures.append(
                    f"{canonical} carries a requirements hash that is not the registered build"
                )
            wheel_rows.append(
                {
                    "package": canonical,
                    "version": stem_version,
                    "filename": wheel_path.name,
                    "size_bytes": record.size_bytes,
                    "sha256": record.content_sha256,
                    "hash_authority": "biohubx_build",
                    "matches_uv_lock": False,
                    "matches_registered_build": matches_build,
                    "matches_requirements_offline": matches_requirements,
                }
            )
            continue
        entry = locked[wheel_path.name]
        canonical = canonical_package_name(entry.package)
        stated = required.get(canonical)
        # The hash is the authority; the size is compared only where uv.lock
        # states one, so an entry without a size is not read as a size of zero.
        matches_lock = record.content_sha256 == entry.sha256 and (
            entry.size_bytes is None or record.size_bytes == entry.size_bytes
        )
        matches_requirements = stated is not None and stated[1] == entry.sha256
        if not matches_lock:
            wheel_failures.append(f"{wheel_path.name} does not match the bytes uv.lock locks")
        if stated is None:
            wheel_failures.append(f"{entry.package} is staged but {requirements_path.name} omits it")
        elif not matches_requirements:
            wheel_failures.append(f"{entry.package} carries a requirements hash uv.lock does not lock")
        wheel_rows.append(
            {
                "package": entry.package,
                "version": entry.version,
                "filename": entry.filename,
                "size_bytes": record.size_bytes,
                "sha256": record.content_sha256,
                "hash_authority": "uv.lock",
                "matches_uv_lock": matches_lock,
                "matches_requirements_offline": matches_requirements,
            }
        )
    staged_packages = {canonical_package_name(item.package) for item in locked.values()} | {
        canonical_package_name(name.split("-", 1)[0]) for name in built
    }
    for canonical in required.keys() - staged_packages:
        wheel_failures.append(f"{canonical} is required offline but no wheel is staged for it")
    if wheel_failures:
        for failure in wheel_failures:
            heartbeat(command, "refused", failure)
        raise typer.Exit(code=2)
    heartbeat(command, "wheels", f"{len(wheel_rows)} verified against uv.lock and {requirements_path.name}")

    remote_report: dict[str, object] | None = None
    if remote is not None:
        if not remote.is_dir():
            heartbeat(command, "refused", f"no downloaded dataset at {remote}")
            raise typer.Exit(code=2)
        observed = tree_digest(remote, on_file=_periodic_progress(command, "published dataset"))
        differences = compare_manifests(published, observed)
        remote_report = {
            "path": str(remote),
            "tree": observed.digest.token,
            "file_count": observed.file_count,
            "total_bytes": observed.total_bytes,
            "matches_published_payload": not differences,
            "differences": [asdict(difference) for difference in differences],
        }
        if differences:
            for difference in differences[:10]:
                heartbeat(
                    command,
                    "refused",
                    f"{difference.relative_path}: {difference.reason} "
                    f"expected={difference.expected} observed={difference.observed}",
                )
            raise typer.Exit(code=2)
        heartbeat(
            command,
            "published-dataset",
            f"files={observed.file_count} bytes={observed.total_bytes} identical to the published payload",
        )

    report_path = out if out is not None else root_path / "artifacts/wheelhouse-identity.json"
    payload = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "wheelhouse": {
            "path": str(house.relative_to(root_path)) if house.is_relative_to(root_path) else str(house)
        },
        "upload_bundle": {
            "tree": bundle.digest.token,
            "file_count": bundle.file_count,
            "total_bytes": bundle.total_bytes,
        },
        "published_payload": {
            "tree": published.digest.token,
            "file_count": published.file_count,
            "total_bytes": published.total_bytes,
            "excludes": {
                "relative_path": configuration[0].relative_path,
                "size_bytes": configuration[0].size_bytes,
                "sha256": configuration[0].content_sha256,
                "why": "Kaggle reads it to create the dataset and does not store it as data",
            },
        },
        "wheels": wheel_rows,
        "published_dataset": remote_report,
    }
    atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True) + chr(10))
    _write_manifest(
        command,
        {
            "report": str(report_path),
            "upload_bundle": bundle.digest.token,
            "published_payload": published.digest.token,
        },
    )
    heartbeat(command, "done", f"two identities recorded in {report_path.name}")
    typer.echo(json.dumps(payload, sort_keys=True))


@data_app.command("fingerprint")
def data_fingerprint(
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Explicit data root. BIOHUB_DATA_ROOT is the only environment fallback."),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Fingerprint one dataset id only. Repeat the command for more."),
    ] = None,
    plan: Annotated[
        bool,
        typer.Option("--plan", help="Measure the work without hashing anything, and exit."),
    ] = False,
) -> None:
    """Compute a typed tree identity for every dataset artifact under a root.

    Layout validation answers whether something is the right shape. This answers
    whether it is the same data, by reading every byte of every chunk file once
    and recording a tree digest per artifact.

    That is the only thing strong enough to bind an experiment to its inputs
    here. The archive a download produced is normally deleted after extraction,
    and a competition slug is a mutable name rather than an identity, so neither
    can stand in for the content of the tree that experiments actually read.

    ``--plan`` performs the same walk with the same refusals but reads nothing,
    so the cost of the real pass is known before committing to it and a layout
    or reparse problem surfaces in seconds rather than after minutes of reading.

    Writes a report whose entries are ready to register in
    registry/artifacts.yaml, naming the selection it covers so a partial run is
    never mistaken for a complete one. It never modifies the data.
    """
    command_name = "data fingerprint"
    from biohubx.data.validation import validate_data_root
    from biohubx.hashing import tree_digest, tree_shape

    env_root = os.environ.get("BIOHUB_DATA_ROOT")
    resolved = root if root is not None else (Path(env_root) if env_root else None)
    if resolved is None:
        heartbeat(command_name, "refused", "provide --root or BIOHUB_DATA_ROOT")
        raise typer.Exit(code=2)

    mode = "plan" if plan else "hash"
    heartbeat(command_name, "start", f"root={resolved} dataset={dataset or 'all'} mode={mode}")
    try:
        manifests = validate_data_root(resolved)
    except (FileNotFoundError, OSError, ValueError) as exc:
        heartbeat(command_name, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    selected = [item for item in manifests if dataset is None or item.dataset_id == dataset]
    if not selected:
        known = sorted(item.dataset_id for item in manifests)
        heartbeat(command_name, "refused", f"no dataset {dataset!r} under this root; known: {known}")
        raise typer.Exit(code=2)

    heartbeat(command_name, "validated", f"datasets={len(selected)} of {len(manifests)}")

    entries: list[dict[str, object]] = []
    hashed_bytes = 0
    for manifest in selected:
        split_dir = resolved if manifest.relative_path == "." else resolved / manifest.relative_path
        for artifact in manifest.artifacts:
            path = split_dir / artifact.name
            label = f"{manifest.split}/{artifact.name}"

            if plan:
                try:
                    shape = tree_shape(path)
                except (OSError, ValueError) as exc:
                    heartbeat(command_name, "refused", f"{artifact.name}: {exc}")
                    raise typer.Exit(code=2) from exc
                hashed_bytes += shape.total_bytes
                entries.append(
                    {
                        "dataset_id": manifest.dataset_id,
                        "split": manifest.split,
                        "artifact": artifact.name,
                        "file_count": shape.file_count,
                        "empty_directory_count": shape.empty_directory_count,
                        "total_bytes": shape.total_bytes,
                    }
                )
                heartbeat(
                    command_name,
                    "measured",
                    f"{label} files={shape.file_count} bytes={shape.total_bytes}",
                )
                continue

            heartbeat(command_name, "hashing", label)
            try:
                tree = tree_digest(path, on_file=_periodic_progress(command_name, label))
            except (OSError, ValueError) as exc:
                heartbeat(command_name, "refused", f"{artifact.name}: {exc}")
                raise typer.Exit(code=2) from exc
            hashed_bytes += tree.total_bytes
            suffix = artifact.name.rsplit(".", 1)[-1]
            entries.append(
                {
                    "artifact_id": f"competition.{manifest.split}.{manifest.dataset_id}.{suffix}",
                    "dataset_id": manifest.dataset_id,
                    "split": manifest.split,
                    "split_role": manifest.split_role,
                    "artifact": artifact.name,
                    "kind": f"competition_dataset_{suffix}",
                    "schema": manifest.schema,
                    # Machine-local evidence of where the bytes were read from.
                    # The digest is the part that travels between machines.
                    "external_path": str(path),
                    "tree_digest": tree.digest.token,
                    "file_count": tree.file_count,
                    "empty_directory_count": tree.empty_directory_count,
                    "total_bytes": tree.total_bytes,
                    "provenance": {
                        "status": "external_uncleared",
                        "original_path": str(path),
                        # Describes the dataset only. Whether the terms have
                        # been reviewed is what `status` is for, and saying it
                        # twice is how the two came to disagree.
                        "note": (
                            f"Competition dataset {manifest.dataset_id!r}, split "
                            f"{manifest.split!r}, role {manifest.split_role}."
                        ),
                    },
                }
            )
            heartbeat(
                command_name,
                "hashed",
                f"{artifact.name} files={tree.file_count} bytes={tree.total_bytes} "
                f"{tree.digest.token[:34]}...",
            )

    report: dict[str, object] = {
        "schema_version": 1,
        "mode": "real-read-only",
        "status": "planned" if plan else "fingerprinted",
        # Naming the selection is what stops a one-dataset run being read later
        # as a statement about the whole root.
        "selection": dataset or "all",
        "datasets_selected": len(selected),
        "datasets_available": len(manifests),
        "entries": entries,
    }
    name = "data-fingerprint-plan.json" if plan else "data-fingerprint.json"
    output = repository_root() / "artifacts" / name
    atomic_write_text(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_manifest(
        f"{command_name} {mode}",
        {
            "root": str(resolved),
            "selection": dataset or "all",
            "entries": len(entries),
            "total_bytes": hashed_bytes,
            "report": f"artifacts/{name}",
        },
    )
    heartbeat(
        command_name,
        "done",
        f"mode={mode} entries={len(entries)} bytes={hashed_bytes} report=artifacts/{name}",
    )
    typer.echo(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    app()


@research_app.command("intake")
def research_intake(
    url: Annotated[str, typer.Option("--url", help="Public http(s) source.")],
    kind: Annotated[
        str,
        typer.Option(
            "--kind", help="paper | repository | notebook | discussion | page | dataset_listing | other"
        ),
    ],
    campaign: Annotated[str, typer.Option("--campaign", help="Research campaign id, e.g. RX-01.")],
    branch: Annotated[str, typer.Option("--branch", help="Stream within the campaign.")],
    note: Annotated[str, typer.Option("--note", help="Why this source is being read.")],
    fetch: Annotated[bool, typer.Option("--fetch", help="Download the bytes into research/cache.")] = False,
    max_bytes: Annotated[
        int, typer.Option("--max-bytes", help="Per-item cap. Raise deliberately for a large acquisition.")
    ] = 256 * 1024 * 1024,
) -> None:
    """Record a public research source, and optionally fetch it, under D-0039.

    Every acquired artifact gets a ledger record with its source, time, size and
    raw digest, so a claim in registry/reference.yaml can point at an exact place a
    reader can re-fetch. A URL already in the ledger is returned rather than
    fetched again. The request and byte budgets are counted over the whole
    ledger, not the session, so splitting work across branches cannot evade them.

    Payloads live under research/cache, which Git ignores; only the
    repository-relative path is recorded. Acquiring bytes permits inspection, not
    use as evidence: nothing here can become a finding.
    """
    command = "research intake"
    from biohubx.research.ledger import LedgerError, budget_used, intake, load_ledger

    root_path = repository_root()
    heartbeat(command, "start", f"{kind} {campaign}/{branch} fetch={fetch}")
    try:
        entry, created = intake(
            root_path,
            url=url,
            kind=kind,
            campaign=campaign,
            branch=branch,
            note=note,
            do_fetch=fetch,
            max_bytes=max_bytes,
        )
    except LedgerError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    requests_used, bytes_used = budget_used(load_ledger(root_path))
    heartbeat(
        command,
        "recorded" if created else "already-known",
        f"{entry.id} bytes={entry.bytes} digest={(entry.raw_digest or 'none')[:44]}",
    )
    heartbeat(command, "budget", f"requests={requests_used}/2000 bytes={bytes_used}")
    heartbeat(command, "done", entry.id)
    typer.echo(json.dumps(entry.to_dict(), sort_keys=True))


@research_app.command("evict")
def research_evict(
    entry_id: Annotated[str, typer.Option("--id", help="Ledger id whose payload to delete.")],
) -> None:
    """Delete a cached payload and keep its record, digest and source intact."""
    command = "research evict"
    from biohubx.research.ledger import LedgerError, evict

    try:
        entry = evict(repository_root(), entry_id)
    except LedgerError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    heartbeat(command, "done", f"{entry.id} payload_present={entry.payload_present} digest kept")
    typer.echo(json.dumps(entry.to_dict(), sort_keys=True))


@research_app.command("sandbox")
def research_sandbox(
    workdir: Annotated[Path, typer.Option("--workdir", help="The one host directory mounted at /work.")],
    command: Annotated[list[str], typer.Option("--cmd", help="Command token. Repeat for each token.")],
    image: Annotated[
        str, typer.Option("--image", help="Base image, must be present with a repo digest.")
    ] = "python:3.12-slim",
    timeout: Annotated[int, typer.Option("--timeout", help="Wall-clock seconds.")] = 1800,
    cpus: Annotated[str, typer.Option("--cpus")] = "2",
    memory: Annotated[str, typer.Option("--memory")] = "4g",
    allow_network: Annotated[
        bool, typer.Option("--allow-network", help="Declare network. Off by default.")
    ] = False,
    env: Annotated[
        list[str] | None, typer.Option("--env", help="KEY=VALUE passed into the container. Repeatable.")
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option(
            "--report", help="Where to write the run record. Defaults to artifacts/sandbox-run.json."
        ),
    ] = None,
) -> None:
    """Run one command on third-party code inside the disposable sandbox.

    D-0039 permits third-party execution only with no secrets, no general host
    mount, no external writes, bounded processes and resources, and networking
    disabled unless separately declared. This command is that sandbox: a pinned
    image by digest, --network none, a read-only root with tmpfs scratch, exactly
    one bind mount, a non-root user, CPU, memory and PID limits, capabilities
    dropped, and a timeout. The run record carries all of it.
    """
    name = "research sandbox"
    from biohubx.research.sandbox import SandboxError, run_in_sandbox

    environment: dict[str, str] = {}
    for item in env or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            heartbeat(name, "refused", f"--env takes KEY=VALUE, got {item!r}")
            raise typer.Exit(code=2)
        environment[key] = value
    heartbeat(
        name, "start", f"image={image} network={'declared' if allow_network else 'none'} workdir={workdir}"
    )
    try:
        run = run_in_sandbox(
            list(command),
            workdir=workdir,
            image=image,
            cpus=cpus,
            memory=memory,
            timeout_seconds=timeout,
            allow_network=allow_network,
            env=environment,
        )
    except SandboxError as exc:
        heartbeat(name, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    report_path = report if report is not None else repository_root() / "artifacts/sandbox-run.json"
    atomic_write_text(report_path, json.dumps(run.to_dict(), indent=2, sort_keys=True) + chr(10))
    heartbeat(name, "image", run.image_digest)
    heartbeat(
        name, "exit", f"returncode={run.returncode} timed_out={run.timed_out} elapsed={run.elapsed_seconds}s"
    )
    if run.stdout_tail.strip():
        for line in run.stdout_tail.strip().splitlines()[-12:]:
            heartbeat(name, "stdout", line[:160])
    if run.returncode != 0:
        for line in run.stderr_tail.strip().splitlines()[-12:]:
            heartbeat(name, "stderr", line[:160])
        heartbeat(name, "done", f"FAILED, record at {report_path}")
        raise typer.Exit(code=1)
    heartbeat(name, "done", f"record at {report_path}")


def _inspect_rescorer(
    *,
    command: str,
    arm: str,
    config_path: Path,
    section: str,
    root: Path | None,
    dataset: str | None,
    first_frame: int,
    frames: int,
    device: str,
    precision: str,
    seed: int,
    out: Path | None,
) -> None:
    """The analyzer's measurements on a candidate re-scorer: patches in, one logit per candidate out."""
    import time

    import torch

    from biohubx.data.competition import (
        CompetitionLayoutError,
        WindowSelection,
        competition_root,
        load_window,
    )
    from biohubx.evaluation.official_metric import EstimatedTotalNodes, metric_row, summarise_fold
    from biohubx.evaluation.oracle import oracle_graph
    from biohubx.proposals import rescore
    from biohubx.training import inspect as measure
    from biohubx.training.targets import positive_unlabelled_loss

    root_path = repository_root()
    try:
        family = yaml.safe_load(config_path.read_text(encoding="utf-8"))[section]
        candidates_block = family["candidates"]
        bank = tuple(float(r) for r in candidates_block["radii_um"])
        suppression = float(candidates_block["suppression_radius_um"])
        pool_quantile = float(candidates_block["response_quantile"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        heartbeat(command, "refused", f"cannot read the family from {config_path} [{section}]: {exc}")
        raise typer.Exit(code=2) from exc
    if arm not in rescore.ARMS:
        heartbeat(command, "refused", f"unknown arm {arm!r}; the family declares {sorted(rescore.ARMS)}")
        raise typer.Exit(code=2)
    config_digest = digest_file(config_path, DigestKind.CANONICAL_TEXT).token
    torch_device = torch.device(device)
    try:
        dtype = measure.precision_dtype(precision)
    except measure.InspectError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    try:
        data_root = competition_root(root)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    movie = dataset
    if movie is None:
        found = sorted(p.stem for p in (data_root / "train").glob("44b6*.geff"))
        if not found:
            heartbeat(command, "refused", "no annotated 44b6 movie under the data root")
            raise typer.Exit(code=2)
        movie = found[0]
    depth, height, width = _open_volume(data_root / "train" / f"{movie}.zarr").shape[1:]
    try:
        window = load_window(
            data_root,
            WindowSelection(movie, first_frame, frames, 0, depth, 0, height, 0, width),
            split="train",
        )
    except (CompetitionLayoutError, ValueError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    started = time.monotonic()
    estimate = float(window.window_estimated_total_nodes)
    pool = rescore.extract_candidates(
        window.volume,
        dataset=window.annotated.dataset,
        radii_um=bank,
        suppression_radius_um=suppression,
        response_quantile=pool_quantile,
    )
    labels = rescore.candidate_labels(pool, window.annotated).to(torch_device)
    prior = rescore.candidate_prior(estimate, len(pool.instances))
    patches = rescore.candidate_patches(window.volume, pool, arm=arm, radii_um=bank).to(torch_device)
    heartbeat(
        command,
        "start",
        f"{arm} {movie} frames={frames} pool={len(pool.instances)} positives={int(labels.sum())} "
        f"patches={tuple(patches.shape)} device={torch_device} precision={precision}",
    )

    def build() -> torch.nn.Module:
        return rescore.build_scorer(arm, seed=seed)

    model = build().to(torch_device)

    def forward(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if dtype is torch.float32:
            return torch.cat([m(part) for part in rescore.batches(x, 512)])
        with torch.autocast(device_type=torch_device.type, dtype=dtype):
            return torch.cat([m(part) for part in rescore.batches(x, 512)]).float()

    def loss_fn(logits: torch.Tensor) -> torch.Tensor:
        return positive_unlabelled_loss(logits, labels, prior=prior)[0]

    one = patches[:1]
    census = measure.parameter_census(model)
    report: dict[str, object] = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "consumer": "E07, Stage 0 of the training funnel",
        "identity": {
            "config": str(config_path.relative_to(root_path))
            if config_path.is_relative_to(root_path)
            else str(config_path),
            "section": section,
            "config_digest": config_digest,
            "arm": rescore.describe(arm),
            "initialisation": f"deterministic random, seed {seed}",
        },
        "input": {
            "window": window.to_dict(),
            "axes": "(N candidates, C channels, 9, 9, 9) patches on the isotropic 1.625 um grid",
            "patches_shape": list(patches.shape),
            "dtype": str(patches.dtype).removeprefix("torch."),
            "grid_spacing_um": [1.625, 1.625, 1.625],
            "pool": len(pool.instances),
            "positives": int(labels.sum()),
            "class_prior": prior,
        },
        "execution": {"device": str(torch_device), "precision": precision, **measure.environment()},
        "coverage": None,
        "census": census,
        "shape_trace": measure.shape_trace(model, lambda m, x: m(x), one),
        "downsampling": {
            "note": "not applicable: one scalar per candidate; the patch is the receptive field's upper bound"
        },
        "receptive_field": measure.empirical_receptive_field(model, lambda m, x: m(x), one, spatial_axes=3),
        "graph": measure.capture_graph(model, lambda m, x: m(x), one),
        "activations": measure.activation_census(model, lambda m, x: m(x), patches[:512]),
    }
    field = report["receptive_field"]
    heartbeat(
        command,
        "census",
        f"parameters={census['total_parameters']:,} "
        f"receptive_field={field['extent_voxels'] if isinstance(field, dict) else field}",
    )
    cost = measure.timings_and_memory(model, forward, loss_fn, patches, device=torch_device)
    report["cost"] = cost
    heartbeat(command, "cost", json.dumps(cost["timings"]))
    report["profile"] = measure.profile_operators(model, forward, loss_fn, patches, device=torch_device)
    model.eval()
    with torch.no_grad():
        logits = forward(model, patches)
    report["output"] = measure.output_statistics(logits)
    model.train()
    model.zero_grad(set_to_none=True)
    loss_fn(forward(model, patches)).backward()  # type: ignore[no-untyped-call]
    gradients = measure.gradient_norm_by_block(model)
    report["gradients"] = gradients
    probes = measure.input_channel_probes(model, forward, patches, channel_axis=1)
    report["input_channel_probes"] = probes
    report["temporal"] = {
        "note": "no time axis on a patch; the temporal arm's frame channels are probed as channels above"
    }
    report["determinism"] = measure.strict_reload_determinism(build, model, forward, patches)
    model.eval()
    with torch.no_grad():
        final = forward(model, patches)
    threshold = rescore.threshold_for_count(final, round(estimate))
    kept = rescore.keep_by_threshold(pool, final, threshold)
    graph = oracle_graph(kept, window.annotated)
    extractor: dict[str, object] = {
        "rule": "count-tied threshold, round(window estimate) candidates kept",
        "threshold": threshold,
        "proposals": graph.proposals,
        "estimated_nodes": estimate,
        "node_ratio": (graph.proposals - estimate) / estimate if estimate else None,
        "annotated_nodes": graph.annotated_nodes,
        "matched_nodes": graph.matched_nodes,
        "match_fraction": graph.matched_nodes / graph.annotated_nodes if graph.annotated_nodes else None,
        "retained_edges": graph.retained_edges,
        "annotated_edges": graph.annotated_edges,
        "oracle": None,
    }
    if graph.graph is not None:
        row = metric_row(
            graph.graph, window.annotated, estimated_total_nodes=EstimatedTotalNodes.declared(estimate)
        )
        extractor["oracle"] = summarise_fold([row]).to_dict()
    report["extractor"] = extractor
    inert = [p["channel"] for p in probes if p["inert"]]
    heartbeat(
        command,
        "gradients",
        f"total={gradients['total']:.4e} zero_blocks={gradients['zero_gradient_blocks']} inert={inert}",
    )
    report["runtime_seconds"] = round(time.monotonic() - started, 3)
    report_path = out if out is not None else root_path / f"artifacts/model-inspect-{arm}.json"
    atomic_write_text(
        report_path, json.dumps(measure.to_jsonable(report), indent=2, sort_keys=True) + chr(10)
    )
    _write_manifest(command, {"report": str(report_path), "config_digest": config_digest, "arm": arm})
    heartbeat(command, "done", f"report={report_path}")
    typer.echo(
        json.dumps(
            {
                "report": str(report_path),
                "arm": arm,
                "parameters": census["total_parameters"],
                "extractor": extractor,
            },
            sort_keys=True,
        )
    )


@model_app.command("inspect")
def model_inspect(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Experiment config carrying the model block. Defaults to configs/e03-clean-folds.yaml.",
        ),
    ] = None,
    section: Annotated[str, typer.Option("--section", help="Config section holding `model`.")] = "E03",
    checkpoint: Annotated[
        Path | None, typer.Option("--checkpoint", help="Optional state_dict to load strictly and measure.")
    ] = None,
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Competition data root. BIOHUB_DATA_ROOT is the only fallback."),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            help="Registered movie for the representative input. Defaults to the first annotated 44b6 movie.",
        ),
    ] = None,
    first_frame: Annotated[int, typer.Option("--first-frame")] = 0,
    frames: Annotated[int, typer.Option("--frames", help="Frames in the representative window.")] = 4,
    crop_z: Annotated[str, typer.Option("--crop-z", help="Half-open voxel range start:stop.")] = "0:32",
    crop_y: Annotated[str, typer.Option("--crop-y", help="Half-open voxel range start:stop.")] = "128:256",
    crop_x: Annotated[str, typer.Option("--crop-x", help="Half-open voxel range start:stop.")] = "128:256",
    device: Annotated[str, typer.Option("--device", help="cpu or cuda.")] = "cpu",
    precision: Annotated[str, typer.Option("--precision", help="fp32, bf16 or fp16 (autocast).")] = "fp32",
    seed: Annotated[
        int, typer.Option("--seed", help="Deterministic initialisation when no checkpoint is given.")
    ] = 0,
    pos_feat_dim: Annotated[
        int, typer.Option("--pos-feat-dim", help="Reference spec field the config omits.")
    ] = 32,
    arm: Annotated[
        str | None,
        typer.Option("--arm", help="An E07 re-scorer arm (A3, A4) instead of the reference detector."),
    ] = None,
    peak_quantile: Annotated[
        float, typer.Option("--peak-quantile", help="Heatmap quantile used as the extractor threshold.")
    ] = 0.999,
    out: Annotated[
        Path | None, typer.Option("--out", help="Report path. Defaults to artifacts/model-inspect.json.")
    ] = None,
) -> None:
    """Measure a detector: identity, coverage, census, shapes, spacing, receptive field, cost, behaviour.

    Stage 0 of the training funnel. Every number here is an instrument reading
    about an architecture on one registered input; none is a finding. With the
    heatmap extractor attached, the output is also turned into proposals and the
    oracle ceiling over them is scored through the pinned official path, which
    is what tells a design whether it can even carry the metric.
    """
    command = "model inspect"
    if arm is not None:
        _inspect_rescorer(
            command=command,
            arm=arm,
            config_path=config
            if config is not None
            else repository_root() / "configs/e07-candidate-rescoring.yaml",
            section=section if section != "E03" else "E07",
            root=root,
            dataset=dataset,
            first_frame=first_frame,
            frames=frames,
            device=device,
            precision=precision,
            seed=seed,
            out=out,
        )
        return
    import time

    import numpy as np
    import torch

    from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE
    from biohubx.data.competition import (
        CompetitionLayoutError,
        WindowSelection,
        competition_root,
        load_window,
    )
    from biohubx.evaluation.official_metric import EstimatedTotalNodes, metric_row, summarise_fold
    from biohubx.evaluation.oracle import oracle_graph
    from biohubx.proposals.peaks import HeatmapProposalError, instances_from_heatmap
    from biohubx.reference.architecture import ReferenceEdgeModel, ReferenceSpec
    from biohubx.training import inspect as measure
    from biohubx.training.targets import (
        PriorError,
        TargetConstructionError,
        class_prior_from_estimate,
        positive_mask,
        positive_unlabelled_loss,
    )

    root_path = repository_root()
    config_path = config if config is not None else root_path / "configs/e03-clean-folds.yaml"
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))[section]
        model_block = document["model"]
        downsample = tuple(int(d) for d in model_block["downsample"])
        spec = ReferenceSpec(
            unet_out_channels=int(model_block["unet_out_channels"]),
            unet_layers=tuple(int(layer) for layer in model_block["unet_layers"]),
            pos_feat_dim=pos_feat_dim,
            window_size=int(model_block["window_size"]),
            downsample=downsample,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        heartbeat(command, "refused", f"cannot read a model block from {config_path} [{section}]: {exc}")
        raise typer.Exit(code=2) from exc
    config_digest = digest_file(config_path, DigestKind.CANONICAL_TEXT).token

    try:
        torch_device = torch.device(device)
        dtype = measure.precision_dtype(precision)
    except (RuntimeError, measure.InspectError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        heartbeat(command, "refused", "cuda was requested and is not available")
        raise typer.Exit(code=2)

    def build() -> torch.nn.Module:
        torch.manual_seed(seed)
        return ReferenceEdgeModel(spec)

    model = build().to(torch_device)
    coverage: dict[str, Any] | None = None
    checkpoint_digest: str | None = None
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        checkpoint_digest = digest_file(checkpoint, DigestKind.RAW_ARTIFACT).token
        coverage = measure.state_dict_coverage(model, state)
        heartbeat(
            command,
            "coverage",
            f"strict_ok={coverage['strict_ok']} missing={len(coverage['missing_in_checkpoint'])} "
            f"unexpected={len(coverage['unexpected_in_checkpoint'])}",
        )
        if not coverage["strict_ok"]:
            heartbeat(
                command,
                "refused",
                "the checkpoint does not cover the model strictly; nothing below would describe it",
            )
            raise typer.Exit(code=2)
        model.load_state_dict(state, strict=True)

    try:
        data_root = competition_root(root)
    except CompetitionLayoutError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    movie = dataset
    if movie is None:
        candidates = sorted(p.stem for p in (data_root / "train").glob("44b6*.geff"))
        if not candidates:
            heartbeat(command, "refused", "no annotated 44b6 movie under the data root")
            raise typer.Exit(code=2)
        movie = candidates[0]

    def span(text: str, name: str) -> tuple[int, int]:
        try:
            start, stop = (int(part) for part in text.split(":"))
        except ValueError as exc:
            heartbeat(command, "refused", f"{name} must be start:stop, got {text!r}")
            raise typer.Exit(code=2) from exc
        return start, stop

    z0, z1 = span(crop_z, "--crop-z")
    y0, y1 = span(crop_y, "--crop-y")
    x0, x1 = span(crop_x, "--crop-x")
    selection = WindowSelection(movie, first_frame, frames, z0, z1, y0, y1, x0, x1)
    try:
        window = load_window(data_root, selection, split="train")
    except (CompetitionLayoutError, ValueError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    dz, dy, dx = downsample
    grid_np = np.ascontiguousarray(window.volume[:, ::dz, ::dy, ::dx]).astype(np.float32)
    grid = torch.from_numpy(grid_np).unsqueeze(0).to(torch_device)
    grid_spacing = (
        OFFICIAL_VOXEL_SCALE.z_um * dz,
        OFFICIAL_VOXEL_SCALE.y_um * dy,
        OFFICIAL_VOXEL_SCALE.x_um * dx,
    )
    try:
        positives, placed = positive_mask(tuple(grid_np.shape), window.annotated, downsample=downsample)
        prior = class_prior_from_estimate(window.window_estimated_total_nodes, int(np.prod(grid_np.shape)))
    except (TargetConstructionError, PriorError) as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc
    positives = positives.to(torch_device)
    heartbeat(
        command,
        "start",
        f"{movie} frames={frames} grid={tuple(grid_np.shape)} spacing_um={grid_spacing} "
        f"positives={placed} prior={prior:.3e} device={torch_device} precision={precision}",
    )

    def forward(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        # The analyzer hands back a plain nn.Module; the detection path lives on the
        # reference class, and the cast says so instead of letting __getattr__ guess.
        detector = cast(ReferenceEdgeModel, m)
        if dtype is torch.float32:
            return detector.detect(x)[1]
        with torch.autocast(device_type=torch_device.type, dtype=dtype):
            return detector.detect(x)[1].float()

    def loss_fn(logits: torch.Tensor) -> torch.Tensor:
        return positive_unlabelled_loss(logits[0, :, 0], positives, prior=prior)[0]

    started = time.monotonic()
    report: dict[str, object] = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "consumer": "the learned-proposal experiment family; Stage 0 of the training funnel",
        "identity": {
            "config": str(config_path.relative_to(root_path))
            if config_path.is_relative_to(root_path)
            else str(config_path),
            "section": section,
            "config_digest": config_digest,
            "spec": spec.to_dict(),
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "checkpoint_digest": checkpoint_digest,
            "initialisation": "checkpoint"
            if checkpoint is not None
            else f"deterministic random, seed {seed}",
        },
        "input": {
            "window": window.to_dict(),
            "axes": "(B, W, Z, Y, X); one intensity channel, added inside the model",
            "grid_shape": list(grid_np.shape),
            "dtype": str(grid.dtype).removeprefix("torch."),
            "grid_spacing_um_zyx": list(grid_spacing),
            "downsample_from_voxels": list(downsample),
            "positives_on_grid": placed,
            "class_prior": prior,
        },
        "execution": {"device": str(torch_device), "precision": precision, **measure.environment()},
        "coverage": coverage,
    }
    census = measure.parameter_census(model)
    report["census"] = census
    heartbeat(
        command, "census", f"parameters={census['total_parameters']:,} buffers={census['total_buffers']:,}"
    )

    trace = measure.shape_trace(model, forward, grid)
    report["shape_trace"] = trace
    model.eval()
    with torch.no_grad():
        logits = forward(model, grid)
    spatial = measure.downsampling(tuple(grid.shape), tuple(logits.shape), grid_spacing)
    temporal_factor = grid.shape[1] / logits.shape[1] if logits.shape[1] else float("inf")
    report["downsampling"] = {
        **spatial,
        "temporal_factor": temporal_factor,
        "logits_shape": list(logits.shape),
    }
    heartbeat(
        command,
        "shapes",
        f"leaf_modules={len(trace)} logits={tuple(logits.shape)} spatial_factors={spatial['factors']} "
        f"output_spacing_um={spatial['output_spacing_um']}",
    )

    try:
        field = measure.empirical_receptive_field(model, forward, grid, spatial_axes=3)
    except measure.InspectError as exc:
        field = {"error": str(exc)}
    report["receptive_field"] = field
    heartbeat(command, "receptive-field", json.dumps(field.get("extent_voxels", field)))

    graph = measure.capture_graph(model, forward, grid)
    report["graph"] = graph
    heartbeat(command, "graph", f"method={graph['method']} captured={graph['captured']}")
    report["activations"] = measure.activation_census(model, forward, grid)
    cost = measure.timings_and_memory(model, forward, loss_fn, grid, device=torch_device)
    report["cost"] = cost
    heartbeat(command, "cost", json.dumps(cost["timings"]))
    report["profile"] = measure.profile_operators(model, forward, loss_fn, grid, device=torch_device)

    model.eval()
    with torch.no_grad():
        logits = forward(model, grid)
    report["output"] = measure.output_statistics(logits)
    model.train()
    model.zero_grad(set_to_none=True)
    loss_fn(forward(model, grid)).backward()  # type: ignore[no-untyped-call]
    gradients = measure.gradient_norm_by_block(model)
    report["gradients"] = gradients
    heartbeat(
        command,
        "gradients",
        f"total={gradients['total']:.4e} zero_blocks={gradients['zero_gradient_blocks']}",
    )
    report["input_frame_probes"] = measure.input_channel_probes(model, forward, grid, channel_axis=1)
    report["temporal"] = measure.temporal_perturbation(model, forward, grid, time_axis=1)
    report["determinism"] = measure.strict_reload_determinism(build, model, forward, grid)
    heartbeat(command, "determinism", json.dumps(report["determinism"]))

    model.eval()
    with torch.no_grad():
        heatmap = torch.sigmoid(forward(model, grid))[0, :, 0].cpu().numpy()
    threshold = float(np.quantile(heatmap, peak_quantile))
    extractor: dict[str, object] = {"threshold": threshold, "threshold_quantile": peak_quantile}
    try:
        instances = instances_from_heatmap(
            heatmap,
            dataset=window.annotated.dataset,
            downsample=downsample,
            threshold=threshold,
            scale=window.scale,
        )
        ceiling = oracle_graph(instances, window.annotated)
        estimate = float(window.window_estimated_total_nodes)
        extractor.update(
            {
                "proposals": ceiling.proposals,
                "estimated_nodes": estimate,
                "node_ratio": (ceiling.proposals - estimate) / estimate if estimate else None,
                "annotated_nodes": ceiling.annotated_nodes,
                "matched_nodes": ceiling.matched_nodes,
                "match_fraction": ceiling.matched_nodes / ceiling.annotated_nodes
                if ceiling.annotated_nodes
                else None,
                "retained_edges": ceiling.retained_edges,
                "annotated_edges": ceiling.annotated_edges,
            }
        )
        if ceiling.graph is not None:
            row = metric_row(
                ceiling.graph, window.annotated, estimated_total_nodes=EstimatedTotalNodes.declared(estimate)
            )
            extractor["oracle"] = summarise_fold([row]).to_dict()
        else:
            extractor["oracle"] = None
    except (HeatmapProposalError, ValueError) as exc:
        extractor["error"] = str(exc)
    report["extractor"] = extractor
    oracle_summary = extractor.get("oracle")
    oracle_score = oracle_summary.get("score") if isinstance(oracle_summary, dict) else None
    heartbeat(
        command,
        "extractor",
        f"proposals={extractor.get('proposals')} match={extractor.get('match_fraction')} "
        f"oracle={oracle_score}",
    )

    report["runtime_seconds"] = round(time.monotonic() - started, 3)
    report_path = out if out is not None else root_path / "artifacts/model-inspect.json"
    atomic_write_text(
        report_path, json.dumps(measure.to_jsonable(report), indent=2, sort_keys=True) + chr(10)
    )
    _write_manifest(command, {"report": str(report_path), "config_digest": config_digest})
    heartbeat(command, "done", f"report={report_path}")
    typer.echo(
        json.dumps(
            {
                "report": str(report_path),
                "parameters": census["total_parameters"],
                "oracle": oracle_summary,
            },
            sort_keys=True,
        )
    )


@submission_app.command("validate")
def submission_validate(
    csv_path: Annotated[Path, typer.Option("--csv", help="A submission CSV to validate strictly.")],
    expect_datasets: Annotated[
        str | None,
        typer.Option("--expect-datasets", help="Comma-separated dataset names that must all be present."),
    ] = None,
) -> None:
    """Every reason the platform could refuse the file, found here first."""
    command = "submission validate"
    from biohubx.submission import validate_csv

    expected = [item.strip() for item in expect_datasets.split(",")] if expect_datasets else None
    report = validate_csv(csv_path, expected_datasets=expected)
    for refusal in report.refusals[:20]:
        heartbeat(command, "refused", refusal)
    heartbeat(command, "done", f"rows={report.rows} datasets={len(report.datasets)} ok={report.ok}")
    typer.echo(json.dumps(report.to_dict(), sort_keys=True))
    if not report.ok:
        raise typer.Exit(code=1)


@submission_app.command("rehearse")
def submission_rehearse(
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Staging directory. Defaults to artifacts/submission-rehearsal."),
    ] = None,
    datasets: Annotated[
        int, typer.Option("--datasets", help="Synthetic datasets to emit, one seed each.")
    ] = 3,
    movies: Annotated[
        int,
        typer.Option(
            "--movies", help="Assumed test movies for the size forecast; an assumption, stated as one."
        ),
    ] = 40,
) -> None:
    """The whole submission path on synthetic data: emit graphs, write, validate, round-trip, package.

    Synthetic only, by design: the slice pipeline's deterministic fixture is
    tracked and its graph written exactly as a real fold's would be. The package
    it stages carries digests, row counts, the size forecast and the refusal
    catalogue, and it cannot be pushed or submitted from here; a competition
    submission is an act Arya Arun authorises individually.
    """
    command = "submission rehearse"
    import shutil
    import time

    from biohubx.submission import COLUMNS, forecast, round_trip, validate_csv, write_csv
    from biohubx.tracking.pipeline import SliceConfig, run_slice

    started = time.monotonic()
    staging = out if out is not None else repository_root() / "artifacts/submission-rehearsal"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    graphs = {}
    emitted: dict[str, dict[str, object]] = {}
    for seed in range(1, datasets + 1):
        result = run_slice(SliceConfig(seed=seed))
        name = f"synthetic-slice-v1-s{seed}"
        graphs[name] = result.emitted
        emitted[name] = {
            "seed": seed,
            "nodes": len(result.emitted.nodes),
            "edges": len(result.emitted.edges),
            "divisions": result.decode_report.divisions,
        }
    heartbeat(command, "emitted", json.dumps({k: (v["nodes"], v["edges"]) for k, v in emitted.items()}))

    csv_path = staging / "submission.csv"
    written = write_csv(graphs, csv_path)
    heartbeat(command, "written", f"rows={written.rows} bytes={written.bytes} digest={written.sha256[:16]}")
    validation = validate_csv(csv_path, expected_datasets=sorted(graphs))
    for refusal in validation.refusals[:10]:
        heartbeat(command, "refused", refusal)
    trip = round_trip(graphs, csv_path)
    heartbeat(command, "round-trip", f"identical={trip['identical']} differences={trip['differences'][:3]}")

    # E05 measured about 300 proposals per frame per movie at A2 over both embryos
    # (253,543 over 710 frames on 44b6, 330,820 over 1,280 on 6bba); a 100-frame
    # movie at that density is the assumption behind the forecast.
    projection = forecast(written, nodes_per_movie=30000.0, edges_per_node=0.97, movies=movies)
    catalogue = [
        f"header not exactly {list(COLUMNS)}",
        "a row without exactly ten fields",
        "id not contiguous from 0",
        "row_type other than node or edge",
        "a node row with edge fields set, or a negative id, frame or coordinate",
        "a node_id repeated within a dataset",
        "an edge row with node fields set",
        "an edge whose endpoint is not a node of the same dataset",
        "an edge that does not span exactly one frame forward",
        "a self loop",
        "out-degree above 2 or in-degree above 1",
        "a dataset expected and absent, or present and unexpected",
        "an empty graph set, which is refused before any file exists",
    ]
    manifest = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "kind": "submission rehearsal on synthetic data; no competition bytes, no push, no submission",
        "schema_source": (
            "RL-0051 and RL-0052, the official geffs_to_csv.py and csv_to_geffs.py at "
            "075fc5f5a52d11077f9dc2b074644618f26939e2"
        ),
        "columns": list(COLUMNS),
        "emitted": emitted,
        "written": written.to_dict(),
        "validation": validation.to_dict(),
        "round_trip": trip,
        "forecast": projection,
        "refusal_catalogue": catalogue,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "pushed": False,
        "submitted": False,
        "submission_requires": "Arya Arun's explicit instruction; no command in this repository submits",
    }
    atomic_write_text(
        staging / "REHEARSAL_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + chr(10)
    )
    digest = digest_file(staging / "REHEARSAL_MANIFEST.json", DigestKind.CANONICAL_TEXT).token
    report = {
        "schema_version": 1,
        "provenance_status": "integration_only",
        "staging": str(staging.relative_to(repository_root()))
        if staging.is_relative_to(repository_root())
        else str(staging),
        "manifest_digest": digest,
        "csv_digest": f"raw_artifact_sha256:sha256:{written.sha256}",
        "rows": written.rows,
        "bytes": written.bytes,
        "validation_ok": validation.ok,
        "round_trip_identical": trip["identical"],
        "forecast": projection,
        "pushed": False,
        "submitted": False,
    }
    atomic_write_text(
        repository_root() / "artifacts/submission-rehearsal.json",
        json.dumps(report, indent=2, sort_keys=True) + chr(10),
    )
    _write_manifest(command, {"report": "artifacts/submission-rehearsal.json", "digest": digest})
    heartbeat(
        command,
        "done",
        f"validated={validation.ok} round_trip={trip['identical']} "
        f"forecast={projection['projected_megabytes']}MB "
        f"for {movies} movies; NOT submitted",
    )
    typer.echo(json.dumps(report, sort_keys=True))
    if not validation.ok or not trip["identical"]:
        raise typer.Exit(code=1)
