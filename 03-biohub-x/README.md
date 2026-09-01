# Biohub-X

An object-centric, graph-owning 3D+t cell-tracking system for the Biohub
"Cell Tracking During Development" task.

Biohub-X is an independent attempt. Its beliefs come from the research
primitives dossier snapshotted under [research/](research/), from official
competition code and documentation, from primary papers and official project
repositories, and from measurements produced inside this repository. Nothing
else has standing here.

**Current state: Phase 0.** The repository has an identity contract, a registry,
and a command line. It has no scientific capability yet: no data loading, no
model, no metric, no tracking. See [SYSTEM.md](SYSTEM.md) for what exists and
what is next.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11-3.13 (the repository
pins 3.12 in `.python-version`).

```
uv sync --all-groups
```

This creates `.venv/` and installs `biohubx` in editable mode.

## Usage

Every operation is a subcommand of one typed CLI. There is no `scripts/`
directory.

```
uv run biohubx --version
uv run biohubx artifacts digest research/primitives-dossier.md --kind raw_artifact_sha256
uv run biohubx artifacts verify
```

`artifacts verify` re-derives every digest recorded in
[registry/artifacts.yaml](registry/artifacts.yaml) from the bytes on disk and
exits non-zero if any recorded identity no longer holds. It writes a run
manifest to `artifacts/manifests/`.

Commands print a heartbeat to stderr and their result to stdout, so a run that
produces nothing is distinguishable from a run that hung.

## Tests

```
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

`tests/contracts/` holds the tests that must never be relaxed: repository
isolation, the absence of absolute paths in code and configuration, line-ending
policy, and artifact-registry integrity.

## Documents

The repository has four governing documents and four registries, and no other
research hierarchy.

| File | Holds |
| --- | --- |
| [SYSTEM.md](SYSTEM.md) | architecture, contracts, honest current state |
| [AGENTS.md](AGENTS.md) | operating and safety rules for anyone working here |
| [DECISIONS.md](DECISIONS.md) | short architectural decision records |
| [registry/experiments.yaml](registry/experiments.yaml) | every experiment and its single result status |
| [registry/models.yaml](registry/models.yaml) | every model and its provenance |
| [registry/artifacts.yaml](registry/artifacts.yaml) | every asserted artifact identity |
| [registry/findings.yaml](registry/findings.yaml) | every measured claim |

Scored values live in the registries once. They are never restated in prose.
