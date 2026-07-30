# Biohub Cell Tracking 2026

Lean competition workspace for 3D+t zebrafish cell detection, association, and lineage
reconstruction.

## Start here

1. [`HANDOFF.md`](HANDOFF.md) — current scores, decisions, and execution state.
2. [`reports/NEXT_DECISION.md`](reports/NEXT_DECISION.md) — the only active research track.
3. [`reports/EXPERIMENT_LEDGER.md`](reports/EXPERIMENT_LEDGER.md) — compact record of closed
   methods and canonical evidence.
4. [`reports/METRIC_SEMANTICS_VERIFIED.md`](reports/METRIC_SEMANTICS_VERIFIED.md) — verified
   scorer behaviour.
5. [`reports/journal/JOURNAL.md`](reports/journal/JOURNAL.md) — chronological experiment log.

Everything removed during the 2026-07-30 lean reset remains recoverable from Git tag
`pre-lean-2026-07-30` (commit `7897511`).

## Current truth

- Private-safe baseline: **E0c**, public **0.889**, exact LOEO OOF **0.7595 / 0.6490**.
- Best clean public deployment probe: **v122**, public **0.908**, but it fails the private-safe
  both-family OOF gate.
- M1 domain-randomised training failed cross-family transfer: **+0.0010**, versus the
  preregistered **+0.005** gate.
- No model, wrapper, ILP, selector, division, redetection, or temporal method is promoted.
- Generic additional seeds, full-data training, and ensembling are blocked until a mechanism
  crosses the embryo-family boundary.

## Active repository

| Path | Purpose |
|---|---|
| `src/biotrack/` | scorer integration, graph conversion, and the E0c wrapper |
| `scripts/` | baseline reproduction, exact scoring, and environment preflight |
| `tests/` | active metric and wrapper regression tests |
| `notebooks/` | retained baseline/public deployment and OOF kernels |
| `reports/` | current decision, evidence ledger, metric semantics, and journal |
| `data/`, `weights/`, `artifacts/` | ignored local data and reproducible run artifacts |

## Core commands

```powershell
# Tests
.\.venv\Scripts\python.exe -m pytest -q

# Exact OOF scoring
.\.venv\Scripts\python.exe scripts\score_oof.py `
  --pred-dir <prediction-directory> --gt-dir data\train

# Rebuild/score the authoritative E0c wrapper cache
.\.venv\Scripts\python.exe scripts\win_bet\e0c_run.py --shard 0/4
.\.venv\Scripts\python.exe scripts\win_bet\e0c_score.py --workers 4
```

## Non-negotiables

- Select on embryo-held-out evidence, never the four visible placeholder movies.
- Gate against E0c with both folds positive, min-fold gain at least `+0.005`, and no major
  regime collapse.
- Keep public metric exploits, negative-time nodes, out-of-volume nodes, and artificial hubs
  out of every submission.
- External sources require URL, license, and checksum.
- Record and commit every result, including failures.
