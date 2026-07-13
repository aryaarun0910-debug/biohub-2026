# Current handoff

**Updated:** 2026-07-12
**Branch:** `master`

## Objective

Win the private leaderboard while remaining prize-eligible. Public score is a
deployment signal; model selection is embryo-held-out.

## Scores and decisions

| System | Public | OOF `44b6` | OOF `6bba` | Decision |
|---|---:|---:|---:|---|
| **E0b wrapper (strong; NOT yet exact)** | 0.889 | 0.7601 | 0.6450 | strong baseline; **E0c parity pending** before authoritative |
| raw greedy OOF (pre-wrapper) | — | 0.6562 | 0.5593 | weak; do NOT gate against this |
| fork-suppressed organizer | — | 0.6595 | 0.5680 | old safe OOF baseline |
| Trackastra hint inside motion relinker | 0.889 | — | — | neutral; do not repeat |
| direct Trackastra + pruning | 0.865 | 0.6948 | 0.6044 | BELOW wrapper on both folds — rejected |

**E0b (2026-07-13), strong but deployment-inexact.** The extracted pure-0.889 wrapper
OOF is **0.7601 / 0.6450** (min-fold 0.6456) via `scripts/win_bet/e0_replay.py` over
`artifacts/kaggle/oof_clean`. It already **exceeds** the direct Trackastra-fusion OOF
(0.6948 / 0.6044) on both folds — proving fusion's +0.035/+0.036 "gain" was an artifact
of the weak greedy (0.656/0.559); against the real wrapper fusion regresses
(−0.065/−0.041), which is why it lost hidden (0.865 vs 0.889). Trackastra replacement
is conclusively dead. **But E0b is not deployment-exact — two mismatches vs the saved
0.889 `run_stats.csv`:** (1) min-track-len — the submission used **7** everywhere
(effective 6 only on the `6bba_05b6850b` public-test movie), E0b used 6 for all crops;
(2) image-based **synthetic-gap refinement** was ON in the submission (gap_refined
144/988/72/931), OFF in E0b. **E0c must fix both, prove exact/explained parity on the
four saved test movies vs `run_stats.csv`, then re-score OOF — only then authoritative.**
The `6bba_05b6850b`=6 exception is a specific public-test movie and must NOT generalize
to the OOF family (use 7 uniformly for OOF).

**TO RESUME E0c after reboot** (the background OOF run is killed on shutdown; wrapper is
already parity-verified, so just re-run and record):
```powershell
.\.venv\Scripts\python.exe scripts\win_bet\e0_replay.py --pred-dir artifacts\kaggle\oof_clean\pred_geffs_split_0
.\.venv\Scripts\python.exe scripts\win_bet\e0_replay.py --pred-dir artifacts\kaggle\oof_clean\pred_geffs_split_1
```
Then record both-fold numbers in journal + HANDOFF, mark E0c authoritative, and commit.
Then Phase B: export the full pre-assignment candidate surface (not final edges), label
via scorer pred→GT matching, compare vs the wrapper's composite decision, gate on exact
graph-level gain over E0c (not AUC).

## Current architecture direction

Maintain the 0.889 graph as the hedge while testing a new three-layer system:

1. **Data:** dense public trajectory supervision, corruption, PU labels and
   acquisition-regime metadata.
2. **Algorithms:** track-before-detect lineage field, baseline-preserving
   selective repairs, component posterior and legitimate division prediction.
3. **Inference/compute:** cached candidates, uncertainty cascades, target motion
   calibration and complementary ensembles.

Primary design: [THREE_LAYER_WIN_ARCHITECTURE_2026-07-12.md](reports/THREE_LAYER_WIN_ARCHITECTURE_2026-07-12.md).

## Immediate queue

1. Bracketed-miss analysis: determine what fraction of missing endpoints can be
   predicted from a track on both sides.
2. Motion-compensated response integration using raw pre-NMS detector evidence.
3. Baseline-preserving selective edge replacement; never replace the whole
   association graph.
4. Component-confidence and count-aware pruning curves across both embryos.
5. Public March-22 dense-track edge/existence/fork training table.
6. Forward/backward consistency gate before any learned target adaptation.

## Canonical local artifacts

- `data/train`, `data/test`: extracted competition data.
- `artifacts/kaggle/oof_clean`: canonical learned OOF GEFFs.
- `artifacts/kaggle/trackastra_full_v6/trackastra_edges_split_{0,1}`:
  complete Trackastra candidate tables.
- `artifacts/kaggle/trackastra_full_v6/materialized_fused_pruned_*`:
  authoritative pruned fusion OOF graphs.
- `artifacts/kaggle/lb897_calibration`: 0.889 anchor artifact.
- `artifacts/kaggle/lb897_trackastra_v2`: neutral hint artifact.
- `artifacts/kaggle/lb897_trackastra_v3`: rejected direct-replacement artifact.
- `artifacts/kaggle/weights_dataset`: canonical fold-weight bundle.

## Active Kaggle references

- Anchor submission: `54534923`, score 0.889.
- Neutral hint submission: `54588144`, score 0.889.
- Direct Trackastra submission: `54601594`, score 0.865.
- Deployment kernel: `aryaarun07/biohub-lb897-trackastra-fusion`.

## Guardrails

- The unmatched-fork division evaluator pathology is diagnostic only and must
  never enter a submission.
- Exact public-source trajectory transfer into an identified hidden crop needs
  written host clearance; generic public-data pretraining is permitted.
- Preserve dirty user files unless their ownership and purpose are established.
- Do not infer hidden quality from the four visible placeholder movies.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git status --short
```
