# Current handoff

**Updated:** 2026-07-12
**Branch:** `master`

## Objective

Win the private leaderboard while remaining prize-eligible. Public score is a
deployment signal; model selection is embryo-held-out.

## Scores and decisions

| System | Public | OOF `44b6` | OOF `6bba` | Decision |
|---|---:|---:|---:|---|
| **E0b pure-0.889 wrapper (authoritative)** | **0.889** | **0.7601** | **0.6450** | **production baseline — gate all deltas vs this** |
| raw greedy OOF (pre-wrapper) | — | 0.6562 | 0.5593 | weak; do NOT gate against this |
| fork-suppressed organizer | — | 0.6595 | 0.5680 | old safe OOF baseline |
| Trackastra hint inside motion relinker | 0.889 | — | — | neutral; do not repeat |
| direct Trackastra + pruning | 0.865 | 0.6948 | 0.6044 | BELOW wrapper on both folds — rejected |

**E0b (2026-07-13):** the pure-0.889 wrapper OOF is **0.7601 / 0.6450** (min-fold
0.6456), reproduced locally via `scripts/win_bet/e0_replay.py` over
`artifacts/kaggle/oof_clean`. This is now THE baseline. It **exceeds** the direct
Trackastra-fusion OOF (0.6948 / 0.6044) on both folds — proving that fusion's
+0.035/+0.036 "gain" was an artifact of comparing to the weak greedy (0.656/0.559);
against the real wrapper the fusion regresses (−0.065/−0.041), which is why it lost
hidden (0.865 vs 0.889). The direct system scored 0.9065 on the four labelled dummy
movies but only 0.865 hidden. Dummy-movie score is not a promotion gate.

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
