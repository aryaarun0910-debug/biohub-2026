# AGENTS

Operating and safety rules for anyone, human or agent, working in Biohub-X.
Read this before making a change.

## 1. Isolation

Biohub-X is a blind, independent attempt. A prior campaign on this task exists
elsewhere on some machines. Treat it as nonexistent.

You may not open, search, read the history of, import from, copy from, depend
on, submodule, symlink to, compare against during design, or modify any prior
campaign repository. This covers code, notebooks, tests, reports, facts,
experiment results, configurations, weights, artifacts and conclusions.

No prior belief enters Biohub-X unless Biohub-X reproduces it independently.

The authoritative quarantine list is the `QUARANTINED_NAMES` constant in
`tests/contracts/test_repository_isolation.py`. That test also enforces the
general properties that make isolation checkable: the repository is its own Git
toplevel, no tracked file carries an absolute filesystem path outside the
allow-listed provenance records, and no import path resolves outside the
repository, its virtual environment and the standard library.

If you believe the quarantine list needs to change, stop and ask. Do not edit it
as part of another change.

## 2. What may be believed

Only four sources have standing:

1. `research/primitives-dossier.md`, status `reference_only`.
2. Official competition code, metric, documentation and data.
3. Primary research papers and official project repositories.
4. Measurements produced inside Biohub-X and recorded in `registry/findings.yaml`.

A number that is not in a registry is not a result. A number restated in prose
is a duplicate and must be removed in favour of the registry ID.

## 3. Repository rules

- No general-purpose `scripts/` directory. Operations are subcommands of
  `biohubx.cli`.
- No substantive model logic in notebooks. Notebooks are generated from the
  tested package.
- No manually duplicated score values.
- No untyped artifact hashes. Every digest states its kind, and its
  canonicalization version when one applies.
- No hidden absolute paths in code or configuration.
- No silent optional imports. If a dependency is missing, fail loudly.
- No zero-filling a missing feature. A missing feature is an error or an
  explicit, tested, named sentinel.
- No `load_state_dict(..., strict=False)` without an audited exception recorded
  in `DECISIONS.md`.
- No experiment without a falsifier.
- No new module without a named consumer. The sparse tree is deliberate.
- "Implemented" is not a scientific result.
- `main` contains only contracts, reusable system components and promoted
  implementations. Failed experiments live in Git history and machine-readable
  reports, not in dead production files.

## 4. Experiments

Every experiment declares, before it runs: ID; system or component; hypothesis;
falsifier; inputs and their digests; data split; model configuration; budget;
expected outputs; forbidden changes; promotion metric; runtime ceiling;
provenance status.

Every completed experiment ends in exactly one status:

`promoted` | `challenger` | `killed` | `invalid` | `integration_only`

"Interesting" and "implemented" are not statuses.

Promotion is decided by the official competition metric, reported per held-out
domain and direction, never only pooled.

## 5. Data and splits

Split by source embryo, movie or direction before augmentation. Augmented
siblings never cross a split. Respect unlabeled and ignore regions rather than
treating them as background.

Pseudo-labels are permitted only after exact correspondence has established
learnability. Record the producing model and its confidence, never treat them as
ground truth, and always compare training with and without them.

## 6. Provenance

Before any external model, checkpoint or dataset enters the system, record:
source repository; pinned commit or release; code licence; weight licence;
training-data provenance; competition eligibility; SHA-256; runtime
preprocessing; exact role.

"Publicly downloadable" is not "cleared". Code licence is not weight licence,
and weight licence is not training-data provenance. No network download is
trusted by filename or size alone.

Large weights are never committed to Git.

## 7. Compute and external actions - stop and report

These require an explicit go-ahead. Prepare the artifacts, print the gate, and
stop:

- downloading model weights or large external datasets;
- any GPU run;
- any Kaggle kernel push;
- any competition submission;
- any change to `.gitattributes`, the canonicalization version, or the
  quarantine list.

Preflight on CPU with preregistered tiny inputs first, and report the exact
inputs before requesting the run.

## 8. Branches

`main` is always installable and testable. One branch per experiment family. A
challenger enters `main` only on promotion. Killed branches are tagged and
removed from active development; their reports stay reachable by commit or tag.

## 9. Before you commit

```
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run biohubx artifacts verify
```

Contract tests in `tests/contracts/` are not to be relaxed to make a change
pass. If a contract test blocks you, the change is probably wrong.
