# Agent contract for Biohub

## Scope and current approach

This is the knowledge and development base for Biohub cell tracking. The target is rank 1;
that is an objective, not evidence of which technique will work. Read
[notes/APPROACH-REVIEW.md](notes/APPROACH-REVIEW.md) and
[notes/RANK1-PLAN.md](notes/RANK1-PLAN.md) before proposing experiments.

The historical division-gate census motivates an ablation. It does not establish that the
entire leaderboard gap is divisions or that wider gates improve total score. The archive
withdraws the earlier edge-parity argument (`prior-campaign/FACT-0161`, `FACT-0162`) and
records a prior ranking ablation (`FACT-0174`). Do not repeat superseded conclusions.

## Knowledge workflow

- Run `git pull --ff-only`, then `python3 tools/sync_source_rows.py check` before editing.
  On a fresh checkout, follow README bootstrap instructions first.
- Read `db/DATABASE.md` before writing rows. SQLite is single-writer; never use simultaneous
  writers. If the snapshot has drifted, preserve/export the existing work before adding yours.
- A measured value lives in the DB once. Prose cites keys and describes reasoning.
- Every new fact needs a retrievable source and a verbatim quote. Validity and confidence
  are separate. UNKNOWN is not verified. Never invent provenance for legacy rows.
- Preserve contradicted facts, set status to superseded and link the replacement. Do not
  delete historical failures or silently restore them to active during refresh.
- Export after any knowledge mutation; run the tests and snapshot check before commit/push.
- Never write to `~/.claude/`. Project knowledge belongs here. Scratch work belongs in
  `~/Work/TestBench/BiohubTestBench/`, outside this repository.

## Experimental rules

- Preregister hypothesis, artifacts, parameter candidates, budget and falsifier before a run.
- Every learned component must exclude the evaluation embryo. Record training membership
  and content hashes, including cached prediction ancestry. Never select on `data/test`.
- Score both embryo directions with the pinned official scorer and report each score term.
  A pooled result or an oracle reachability census is not sufficient for promotion.
- Treat edgeless evaluation as a failed run, never a fabricated zero score.
- Inspect borrowed pipelines for implausible timepoints, hub nodes and count manipulation.
- Historical dead ends remain scoped to their substrate and validity; reopening needs a
  changed premise. Query active records and their provenance before spending compute.

## Compute and packaging

Use PyTorch for the submission path and select the device at runtime. Benchmark numerical
parity on the actual installed stack. `torch.autocast` is thread-local: test any multi-device
path explicitly. Verify all Kaggle data sources are attached before saving a submission run.
Commit a runnable, dependency-pinned baseline before further model development. The ignored
`repos/` trees are archives, not the active submission package. Do not promise GPU parity,
automatic teardown, runtime, or rank improvements without measured evidence.
