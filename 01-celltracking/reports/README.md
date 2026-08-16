# Reports

Evidence and topic notes, not competing instructions. The live instruction is always
`../HANDOFF.md`; stable boundaries are in `../SYSTEM_DESIGN.md`.

## Topic markdowns (append here; read when relevant)

- `CLAIMS.md` — generated, artifact-backed claims (regenerate via `scripts/claims_table.py`; never hand-edit).
- `EXPERIMENT_LEDGER.md` — compact experiment index (append one row per experiment).
- `METRIC_SEMANTICS_VERIFIED.md` — immutable scorer semantics.
- `ENVIRONMENT_TRAPS.md` — reproducibility and environment failures.
- `CORPUS_CTLD_CENSUS.md` — corpus C/T/L/D census (detection/arbitration accounting).
- `submissions/CANDIDATE_LEDGER.md` — submission-by-submission record + public scores.
- `journal/JOURNAL.md` — historical chronology; a record, never a current plan.
- `inventory/` — machine-readable artifacts (CSVs) the claims table reads.

## Conventions

- No per-cycle "CYCLE_OUTPUT" dumps. A cycle's findings go into CLAIMS (via artifacts), one
  EXPERIMENT_LEDGER row, and the journal; raw output stays in the sibling research store.
- Historical/removed material is recoverable from git tags `pre-lean-2026-08-07`,
  `biohub-closed-2026-08-07`. Archived notebooks are under `../notebooks/_archive/`.
