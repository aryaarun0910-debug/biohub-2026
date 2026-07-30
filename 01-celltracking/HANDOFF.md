# Current handoff

**Updated:** 2026-07-30  
**Branch:** `master`  
**Authoritative experiment state:** commit `7897511`

## Outcome

E0c remains the private-safe baseline. Nothing is promoted.

| System | Public | OOF 44b6 | OOF 6bba | Decision |
|---|---:|---:|---:|---|
| **E0c wrapper** | **0.889** | **0.7595** | **0.6490** | authoritative private-safe floor |
| clean v122 deployment probe | 0.908 | 0.6962 | 0.6997 | public-positive, bilateral OOF fail |
| M1 epoch 10 | — | not run cross-family | 0.6499 | fail: `+0.0010`, CI spans zero |

The official scorer is pinned at patch commit `075fc5f`. E0c reproduces exactly across all
199 crops. Gate every future delta against E0c: both families positive, min-fold at least
`+0.005`, and no major regime regression.

## What the campaign established

Seven independent approaches failed exact both-family gates:

1. breadth reranking saturated;
2. learned division posterior had effectively zero high-precision recall;
3. isolated de-novo detection recovered only 0–9% even under oracle gating;
4. temporal accumulation was null on 44b6 and harmful on 6bba;
5. the v122 ILP gained on 6bba but lost `0.0633` on 44b6;
6. an A/D selector had an oracle ceiling barely above the gate and did not transfer;
7. M1 improved same-family validation by `+0.0074` but only `+0.0010` cross-family.

The shared failure is embryo-family transfer. M1 also worsened raw linking
(`0.6499 -> 0.6421`); its small composite gain came from emitting 20% fewer nodes and
receiving count-multiplier credit. It is not “nearly passing.”

Full figures, canonical files, and recovery commits are indexed in
[`reports/EXPERIMENT_LEDGER.md`](reports/EXPERIMENT_LEDGER.md). The complete pre-clean tree
is recoverable from Git tag `pre-lean-2026-07-30`.

## Active decision

Do not run more generic training, extra seeds, full-data fitting, or ensembles.

The only active work is [`reports/NEXT_DECISION.md`](reports/NEXT_DECISION.md):

- quarantine newly popular public 0.95-style notebooks because their scored output contains
  negative-time hub/fork augmentation;
- isolate the clean pre-exploit pipeline delta;
- cheaply test the one apparently unmeasured mechanism: expanded top-two transformer edge
  candidates before ILP;
- in parallel, perform a CPU-only family-boundary decomposition using existing E0c/M1
  artifacts.

GPU is allowed only after the cheap candidate/oracle gate passes. If neither branch exposes
a bilateral oracle ceiling of at least `+0.01`, stop research compute and retain a final
submission hedge: E0c for private robustness and v122 for public strength.

## Canonical assets

- `artifacts/kaggle/oof_clean/` — fold-specific organizer OOF predictions.
- `artifacts/kaggle/e0c_cache/` — authoritative post-wrapper graphs and candidate surface.
- `artifacts/kaggle/weights_dataset/` — fold-specific checkpoints/configs.
- `reports/inventory/e0c_score_full.txt` — exact E0c score.
- `reports/inventory/coupled_score_2026-07-29.txt` — v122 decomposition.
- `reports/inventory/m1_selection.json` and `m1_heldout_result.json` — M1 verdict.

## Guardrails

- No public-score exploitation, negative-time nodes, out-of-volume nodes, or synthetic hubs.
- No routing on family/crop identity and no tuning on visible placeholder test movies.
- No test-time gradient updates without written host clearance.
- Preserve unrelated dirty user files.
- Journal and commit every experiment, including negative results.

Before writing a Kaggle kernel or calling the trainer directly, read
[`reports/ENVIRONMENT_TRAPS.md`](reports/ENVIRONMENT_TRAPS.md) — six environment defects
(trainer loss-weight defaults, broken image polars, package naming, pack/scorer vintage,
`np.savez_compressed` naming, resume data stream) that have each already cost time here.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git status --short
```
