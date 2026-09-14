# Biohub cell tracking

Campaign evidence, experiment records and tooling for the Kaggle competition
*Biohub — Cell Tracking During Development* (closes 2026-09-29).

**Start at [notes/STATE.md](notes/STATE.md)** — the single honest account of where things are:
the metric and its exchange rates, which scoring terms are closed and by which experiment, and
the corrections made along the way. [notes/RANK1-PLAN.md](notes/RANK1-PLAN.md) is the forward
plan only.

Current position: the public **0.947** reproduced exactly, with a measured **+0.0082** variant
(extended post-process grid plus `OUTPUT_MIN_TRACK_LEN=9`) submitted and awaiting scoring.

## What the tooling does

| tool | purpose |
|---|---|
| `tools/score_submission_local.py` | score a real submission.csv on the FOUR actual test movies against released annotations — the only trustworthy offline instrument |
| `tools/multiplier.py` | compute the metric's node-count multiplier exactly, offline, with no leaderboard feedback |
| `tools/mkkernel.py` | build a kernel variant with env overrides that survive the notebook's drift guard |
| `tools/kb.py` | the only interface to the knowledge base; `check` audits provenance |
| `tools/testproxy.py` | score prediction geffs (raw stage) on the four test movies |
| `tools/fastpp.py` | **shelved** — local post-processing loop; under-read a known delta by 3x |

The test set is four movies and `estimated_number_of_nodes` ships for all four, so the
multiplier term is exactly computable without submitting. The score weight-averages by annotated
edge count, which makes it **95.4% 6bba / 4.6% 44b6** — not an even embryo split.

## Start from a clean checkout

Python 3.11 or 3.12 is sufficient for the knowledge tools; they use the standard library.
No Kaggle credentials, private archive clones, or GPU are needed for this bootstrap:

```bash
python3 tools/sync_source_rows.py import
python3 tools/ingest_kaggle.py
python3 tools/sync_source_rows.py export
python3 tools/sync_source_rows.py check
python3 tools/ask.py status
python3 -m unittest discover -s tests -v
```

Import restores the tracked knowledge core with its original identities and relationships.
It is safe to repeat on identical knowledge; it refuses to overwrite differing knowledge.
To inspect a separate restore, pass `--db /path/to/new.db` to the sync command. Ingestion
refreshes competition indexes from digest-verified frozen evidence, using acquisition dates.
It does **not** fetch today's leaderboard. Export records any deliberate refresh changes.

Optional archive registry ingesters require PyYAML (`python3 -m pip install -r requirements-knowledge.txt`).
The reference model has separate dependencies in `reference/royerlab-baseline/pyproject.toml`;
its moving `tracksdata` dependency must be pinned before campaign evaluation.

## Query and maintain

```bash
python3 tools/ask.py counts
python3 tools/ask.py facts division
python3 tools/ask.py facts prior-campaign
python3 tools/ask.py audit
python3 tools/ask.py expts
python3 tools/ask.py todo
```

Facts show confidence, validity, source, quote and review dates. UNKNOWN validity is not
verification. `audit` exposes missing provenance, expired reviews and suspect/invalid records.
Historical rows remain queryable in SQLite even when excluded from active views.

Before a session: pull, then run snapshot `check`. After editing knowledge: `export`, tests,
`check`, commit and push. Never run simultaneous SQLite writers. Read
[the database contract](db/DATABASE.md) before changing rows.

## Layout

| Path | Purpose |
|---|---|
| `db/source_rows.jsonl` | Lossless, tracked knowledge snapshot; `db/*.db` is a local working copy |
| `db/schema.sql` | Schema, including independent fact validity |
| `tools/` | Knowledge queries, snapshot, ingestion and standalone-script smoke checks |
| `evidence/` | Frozen Kaggle responses and acquisition-time digest index |
| `knowledge/` | Navigation and conclusions; use DB keys for measurements |
| `notes/` | Reviewed approach and experiment plan |
| `reference/royerlab-baseline/` | Vendored upstream reference, not modified by the review |
| `repos/` | Ignored private campaign clones, optional for bootstrap |
| `tests/` | Offline regression tests for the knowledge tools |

`tools/ingest_repos.py` refreshes optional archive clones with GitHub authentication.
`tools/ingest_registries.py` imports their registries; `tools/ingest_bx_findings.py` applies
curated findings. Export afterward. Keep archives pinned for reproducible experiment use.

`tools/colab_guard.sh` is for standalone scripts with an explicit smoke-check contract
(documented in `tools/preflight.py`). Receipts cover the script bytes, not imports, datasets
or checkpoints. A successful CLI exit does not prove remote numerical parity or teardown;
inspect the remote report and session state before relying on it.
