# Knowledge storage contract

SQLite is the local query/edit store. `source_rows.jsonl` is its versioned knowledge core.
Table counts are available through `python3 tools/ask.py counts`; do not copy them into prose.

## Persistence

The v2 snapshot includes every column of `source`, `repo`, `fact`, `experiment`, `artifact`
`decision`, `lb_snapshot`, `forum_topic` and `public_kernel`, including primary keys, foreign keys, dates and supersession links. These
records include manual interpretation and corrections even when an original ingest exists.
A topic-name or source-ref heuristic cannot reliably decide whether they are disposable.

`repo_commit` and `repo_file` are disposable indexes that require the optional private
clones to rebuild. Competition snapshots and annotations are preserved because historical
responses and manual triage may not be recoverable from the latest raw evidence. Store
checkpoints and competition data separately by content digest.

```bash
python3 tools/sync_source_rows.py export
python3 tools/sync_source_rows.py check
python3 tools/sync_source_rows.py import --db /path/to/new.db
```

Export is deterministic and atomically replaces the tracked file. Restore validates the full
snapshot in isolation, then writes one transaction with foreign keys enforced. Repeating an
identical restore is a no-op. Restoring over differing knowledge is refused: save/export that
work and restore to another path. Integer IDs from independently edited databases must never
be silently merged. New checkouts import **before** running ingesters.

The legacy snapshot lost fields and relationships. The v2 snapshot was exported from the
surviving complete database, not reconstructed from the lossy file. A pre-change backup was
kept outside the repository during the migration. Do not downgrade the export format.

## Evidence and validity

Every new fact needs `source_id` and a verbatim supporting `quote`. `source` records the
artifact location, acquisition time, method and content digest when available. Legacy
`git:...` labels are identifiers, not verified content hashes. Legacy missing provenance
remains visible in `ask.py audit`; do not invent a source to clear that report.

- `claim_type` describes observation, claim, inference, decision or profile.
- `confidence` describes strength of support.
- `validity` separately records VALID, SUSPECT, INVALID or UNKNOWN, with `validity_reason`.
- `status` records active, superseded or retracted. Keep history and set `superseded_by`
  to the replacement when a contradiction is resolved.

UNKNOWN is the default for legacy facts; a high-confidence observation is not automatically
valid. Registry ingestion preserves explicit validity and supersession relationships.
Active query views exclude suspect/invalid facts and show provenance and review dates.

Declare an experiment's hypothesis, parameter candidates, data/checkpoint/code/scorer hashes,
training/evaluation membership, budget and falsifier in its config/notes **before** running.
Record invalid runs as invalid evidence, never as model wins or scientific dead ends.

## Refresh

Kaggle ingestion verifies every indexed artifact before writing and uses its original
acquisition timestamp. Replaying evidence does not make it fresh. Registry ingesters update
existing identities instead of deleting facts/experiments referenced elsewhere. Export all
knowledge changes after refresh. Keep one writer and run the offline tests before pushing.
