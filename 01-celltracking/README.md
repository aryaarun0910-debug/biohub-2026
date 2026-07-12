# Biohub Cell Tracking 2026

Competition repository for 3D+t zebrafish cell detection, association and
lineage reconstruction.

## Start here

1. [HANDOFF.md](HANDOFF.md) — current scores, decisions and immediate work.
2. [Three-layer victory architecture](reports/THREE_LAYER_WIN_ARCHITECTURE_2026-07-12.md) — data, algorithms and inference/compute strategy.
3. [WIN_PLAN.md](reports/WIN_PLAN.md) — evidence-gated execution plan.
4. [Reports index](reports/README.md) — current, reference and historical documents.
5. [Journal](reports/journal/JOURNAL.md) — chronological experiment record.

## Current truth

- Public best: **0.889** from the 400-epoch learned baseline.
- Trackastra as a weak hint: **0.889**, neutral.
- Direct Trackastra replacement: **0.865**, rejected for deployment.
- Full OOF fusion plus incident-node pruning: **0.6948 / 0.6044** on held-out
  `44b6 / 6bba`, but this did not transfer as a wholesale hidden association
  replacement.
- Public leader: 0.968 outlier; next frontier approximately 0.90–0.91.
- Private leaderboard is 71% and uses hidden embryos; the four local test movies
  are execution placeholders.

## Repository map

| Path | Purpose |
|---|---|
| `src/biotrack/` | metric, submission conversion and classical pipeline code |
| `scripts/` | experiments, exact scoring, Trackastra, solver and adaptation tools |
| `tests/` | metric, graph, adapter and experiment regression tests |
| `notebooks/` | Kaggle kernels; each subdirectory is one deployable job |
| `reports/` | plans, evidence, research, runbooks and journal |
| `data/` | ignored 81.6 GB extracted competition train/test data |
| `weights/` | ignored DAXI and Trackastra weights |
| `artifacts/kaggle/oof_clean/` | canonical learned OOF predictions |
| `artifacts/kaggle/trackastra_full_v6/` | canonical full Trackastra candidate tables and pruned fusion graphs |
| `artifacts/kaggle/lb897_*` | final anchor/fusion submission artifacts |
| `.venv/` | main Python 3.12 metric/development environment |
| `.venv-trackastra/` | isolated Trackastra 0.5.2 environment |

Generated data, environments, weights and artifacts are ignored by Git. Keep
only canonical outputs needed for current comparisons; Kaggle outputs are
re-downloadable.

## Core commands

```powershell
# Tests
.\.venv\Scripts\python.exe -m pytest -q

# Exact GEFF OOF scoring
.\.venv\Scripts\python.exe scripts\score_oof.py `
  --pred-dir <prediction-directory> --gt-dir data\train

# Kaggle kernel status
.\.venv\Scripts\kaggle.exe kernels status <owner/kernel-slug>

# Competition submissions
.\.venv\Scripts\kaggle.exe competitions submissions `
  biohub-cell-tracking-during-development
```

## Rules

- Optimize on embryo-held-out evidence, not the four visible placeholder movies.
- No private-label reconstruction, evaluator defects or submission probing.
- Public external data/models require license, URL, checksum and transformation
  records.
- No submission without exact OOF, runtime, offline dependency and graph-integrity
  gates.

## Environment

- Python: 3.12 via `.venv`.
- Local GPU: MX350 2 GB; use only for smoke tests.
- Heavy inference/training: Kaggle T4/T4x2 or gated cloud compute.
- Voxel scale `(z,y,x)`: `(1.625, 0.40625, 0.40625)` microns.
- Training data: 199 crops from two embryo families (`44b6`, `6bba`).
