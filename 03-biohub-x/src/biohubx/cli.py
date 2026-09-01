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
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

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
app.add_typer(artifacts_app)


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


if __name__ == "__main__":
    app()
