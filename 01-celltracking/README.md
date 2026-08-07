# Biohub Cell Tracking 2026

Lean execution repository for the Kaggle Biohub cell-tracking competition.

## Current position

- Public score: **0.915** (P3 harmonic).
- Leader at the last verified check: approximately **0.949**.
- Active scientific question: can rejected local maxima in the 6bba detector be
  re-accepted with useful precision?
- No scoring candidate is currently ready to submit.

Read only these files to start:

1. `HANDOFF.md` - current state and next command.
2. `SYSTEM_DESIGN.md` - stable architecture and repository boundaries.
3. `reports/CYCLE_OUTPUT_2026-08-07.md` - evidence behind the current decision.

Historical scripts and reports removed by the August 7 lean reset remain available at
Git tag `pre-lean-2026-08-07`. Raw research and large artifacts live in the sibling
`Biohub-CellTracking-2026_RESEARCH` store.

## Commands

```powershell
# Fast correctness gate for active development
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_scoring.py tests/test_metric_parity.py `
  tests/test_d1_partition.py tests/test_d1_postprocess.py `
  tests/test_d1_v6_export.py tests/test_d1f_probe.py

# Full regression gate before Kaggle or submission work
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\claims_table.py --check
```

The project is Python-first. Native code is introduced only after profiling proves that a
stable pure function is a material bottleneck and a Python parity reference exists.
