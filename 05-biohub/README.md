# Biohub cell tracking development base

This repository preserves campaign evidence, experiment records and archived findings for
Biohub cell tracking. It includes the organizers' reference code; a current campaign
training/inference package is the next milestone, not an already verified deliverable.

Read [the approach review](notes/APPROACH-REVIEW.md) for the critique and reasoning behind
the fixes, and [the campaign plan](notes/RANK1-PLAN.md) for the next validation gates.

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
