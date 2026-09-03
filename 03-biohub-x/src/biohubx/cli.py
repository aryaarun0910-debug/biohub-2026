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
from typing import Annotated, Any

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
app.add_typer(artifacts_app)
app.add_typer(official_app)
app.add_typer(evaluate_app)
app.add_typer(data_app)
app.add_typer(infer_app)
app.add_typer(train_app)
app.add_typer(package_app)


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
    gpus: Annotated[int, typer.Option("--gpus", help="GPUs the run expects and asserts.")] = 1,
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
    ] = "T4",
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
    import shutil
    import subprocess

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
        accelerator="nvidiaTeslaT4",
        expected_gpu_count=gpus,
        expected_device_substring=expect_device,
        smoke=True,
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
    notebook = build_notebook(spec, shipped=shipped, payload=archive_bytes)
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
        gate = prepush.run_all(notebook, shipped, data_root=data_root, interpreter=sys.executable)
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
    smoke_manifest = gate_out / f"result-{spec.fold.fold_id}.json"
    if smoke_manifest.is_file():
        payload["local_smoke"] = json.loads(smoke_manifest.read_text(encoding="utf-8"))

    report = root_path / "artifacts/kaggle-package.json"
    atomic_write_text(report, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_manifest(command, {"report": "artifacts/kaggle-package.json", "fold": spec.fold.fold_id})
    heartbeat(command, "done", f"NOT PUSHED. push with: {push_command}")
    typer.echo(json.dumps(payload, sort_keys=True))


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
    """
    command = "package preflight"
    import shutil
    import subprocess

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

    notebook = build_preflight_notebook(spec)
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
    }
    atomic_write_text(
        root_path / "artifacts/kaggle-preflight.json", json.dumps(payload, indent=2, sort_keys=True) + chr(10)
    )
    _write_manifest(command, {"report": "artifacts/kaggle-preflight.json", "preflight": spec.audit_id})
    heartbeat(command, "staged", f"files={len(files) + 1} digest={package_digest.token[:52]}")
    heartbeat(command, "done", "NOT PUSHED and the dataset it needs does not exist yet")
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

    try:
        locked = locked_wheels(
            (root_path / "uv.lock").read_text(encoding="utf-8"),
            [item.name for item in wheel_paths],
        )
        required = requirement_hashes(requirements_path.read_text(encoding="utf-8"))
    except WheelhouseError as exc:
        heartbeat(command, "refused", str(exc))
        raise typer.Exit(code=2) from exc

    staged_content = {record.relative_path: record for record in bundle.records if record.kind == "f"}
    wheel_rows: list[dict[str, object]] = []
    wheel_failures: list[str] = []
    for wheel_path in wheel_paths:
        entry = locked[wheel_path.name]
        record = staged_content[f"wheels/{wheel_path.name}"]
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
                "matches_uv_lock": matches_lock,
                "matches_requirements_offline": matches_requirements,
            }
        )
    for canonical in required.keys() - {canonical_package_name(item.package) for item in locked.values()}:
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
