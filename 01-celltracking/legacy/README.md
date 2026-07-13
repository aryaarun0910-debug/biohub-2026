# legacy/ — superseded code, preserved for provenance

Moved here 2026-07-13 during a utilization audit. These files belong to abandoned or
superseded strategy branches. They are **kept, not deleted** (git-tracked, reversible)
so the decision history stays intact — but they are NOT part of the active pipeline and
should not be run or imported by current work.

## What moved and why

| Path | Superseded branch |
|---|---|
| `scripts/phase0_norm_ablation.py`, `phase1_recall_frontier.py` | early normalization / recall-frontier ablations (pre-learned-stack) |
| `scripts/spotiflow_zeroshot_screen.py` | Spotiflow detector screen — abandoned (detection near-ceiling) |
| `scripts/tbd_diagnostic.py` | naive track-before-detect — killed (failed its 20% gate at 5.5%) |
| `scripts/build_inventory.py`, `make_animation.py`, `make_figures.py` | one-off inventory / figure generation |
| `notebooks/kaggle_op_bright/` | classical over-propose+arbitrate kernel (~0.81) |
| `notebooks/kaggle_spotiflow_screen/` | Spotiflow screen kernel |
| `notebooks/kaggle_ablate/` | early ablation kernel |

## What was deliberately KEPT (not dead)

- `scripts/run_phase1_ablation.py`, `run_v3_taxonomy.py`, `validate_metric_parity.py` —
  **test-imported** (tests/test_metric_parity.py); moving them breaks the suite.
- `scripts/metric_solver/`, `oracle_redetect/`, `target_adapt/`, `trackastra_zero_shot/`,
  `sweep_edge_threshold.py`, `sweep_trackastra_greedy.py` — **test-covered** lever
  implementations still referenced by the roadmap.
- `src/biotrack/` — all active or test-covered (`metric`, `wrapper`, `submission`,
  `arbitrate`, `metric_numpy`; `propose`/`cache` retained for the kept classical scripts).
- `notebooks/kaggle_lb897_trackastra/` (deploy wrapper source) + the OOF/predict/pack/
  submit pipeline kernels — active or reference.
- `vendor/`, `tests/`, `reports/` — active (reports/ is organized by its own README).

Verified: `pytest` green (50 passed) after the move.
