# Biohub Cell Tracking 2026 — operating instructions

## Start here

Read, in order:

1. `HANDOFF.md`
2. `reports/NEXT_DECISION.md`
3. `reports/EXPERIMENT_LEDGER.md`
4. `reports/METRIC_SEMANTICS_VERIFIED.md`

Older plans and transfer briefs were removed from the active tree on 2026-07-30. Recover
them only when necessary from Git tag `pre-lean-2026-07-30`; they are not live instructions.

## Standing workflow

After every experiment or meaningful result:

1. Append a dated, terse entry to `reports/journal/JOURNAL.md`: execution, exact numbers,
   decision, and next step.
2. Update `HANDOFF.md` and `reports/NEXT_DECISION.md` if the conclusion or active queue
   changes.
3. Commit code, small canonical results, and documentation together with a descriptive
   message and a `Co-Authored-By` trailer.

Negative evidence must remain in the journal/ledger. Do not resurrect a closed method
without identifying a genuinely unmeasured mechanism and a cheap falsification gate.

## Measurement rules

- Baseline: E0c exact wrapper, public `0.889`, LOEO OOF `0.7595 / 0.6490`.
- Promotion: both embryo families improve, min-fold delta at least `+0.005`, no major
  regime collapse, exact patched scorer.
- Use at most two confirmatory parameterizations per round.
- Never tune on the four visible placeholder movies.
- Never use family/crop identity as a deployment router.
- Public metric exploits, negative-time/out-of-volume nodes, and artificial hubs/forks are
  quarantined and may not enter a submission.
- External data/models require URL, license, checksum, and transformation provenance.
- Test-time weight updates require written host clearance.

## Active track

Only `reports/NEXT_DECISION.md` is active. Generic retraining, more seeds, full-data fitting,
and ensembling remain blocked. GPU may be spent only after the clean-public candidate-breadth
or family-boundary oracle gate passes.

## Environment

- Main environment: `.venv` (Python 3.12).
- Trackastra environment is historical and may be removed locally.
- Tests: `.\.venv\Scripts\python.exe -m pytest -q`
- Exact scoring:
  `.\.venv\Scripts\python.exe scripts\score_oof.py --pred-dir <dir> --gt-dir data\train`

Preserve unrelated dirty user files. Generated data, weights, caches, and Kaggle outputs
remain ignored and should not be committed.
