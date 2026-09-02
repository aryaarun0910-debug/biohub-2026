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
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

from biohubx import __version__
from biohubx.artifacts import (
    ARTIFACT_REGISTRY_PATH,
    MANIFEST_DIR,
    atomic_write_text,
    load_artifact_registry,
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
app.add_typer(artifacts_app)
app.add_typer(official_app)
app.add_typer(evaluate_app)
app.add_typer(data_app)


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
) -> None:
    """Re-derive every digest in the artifact registry from the bytes on disk.

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

    heartbeat(command, "loaded", f"artifacts={len(loaded.artifacts)}")
    checks = verify_registry(loaded, root)
    for check in checks:
        status = "ok" if check.ok else "FAIL"
        detail = f"id={check.artifact_id} slot={check.slot}"
        if check.detail:
            detail = f"{detail} {check.detail}"
        heartbeat(command, status, detail)

    failed = [check for check in checks if not check.ok]
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
            "checked": len(checks),
            "failed": len(failed),
            "checks": [check.model_dump() for check in checks],
        },
    )
    heartbeat(
        command,
        "done",
        f"checked={len(checks)} failed={len(failed)} manifest={manifest_path.relative_to(root)}",
    )
    if failed:
        raise typer.Exit(code=1)


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


if __name__ == "__main__":
    app()
