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
from typing import Annotated

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
app.add_typer(artifacts_app)
app.add_typer(official_app)
app.add_typer(evaluate_app)
app.add_typer(data_app)
app.add_typer(infer_app)


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
