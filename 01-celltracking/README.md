# Biohub Cell Tracking 2026 — archived

This project was permanently closed on **2026-08-07**. It is retained as a reproducible,
read-only research archive. Do not launch training, Kaggle kernels, submissions, monitors,
or unfinished experiments from this repository.

## Current position

- Final verified public score: **0.915** (P3 harmonic).
- Public rank at closure: **143**.
- Active compute: **none**.
- Closure rationale and preservation details: `HANDOFF.md`.

Read only these files for historical context:

1. `HANDOFF.md` - permanent closure record.
2. `SYSTEM_DESIGN.md` - stable architecture and repository boundaries.
3. `reports/CYCLE_OUTPUT_2026-08-07.md` - evidence behind the current decision.

Historical scripts and reports removed by the August 7 lean reset remain available at
Git tag `pre-lean-2026-08-07`. Raw research and large artifacts live in the sibling
`Biohub-CellTracking-2026_RESEARCH` store.

## Commands

```powershell
# Historical correctness gate (read-only verification)
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
