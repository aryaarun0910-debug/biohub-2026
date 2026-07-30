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

Eight independent approaches failed exact both-family gates:

1. breadth reranking saturated;
2. learned division posterior had effectively zero high-precision recall;
3. isolated de-novo detection recovered only 0–9% even under oracle gating;
4. temporal accumulation was null on 44b6 and harmful on 6bba;
5. the v122 ILP gained on 6bba but lost `0.0633` on 44b6;
6. an A/D selector had an oracle ceiling barely above the gate and did not transfer;
7. M1 improved same-family validation by `+0.0074` but only `+0.0010` cross-family;
8. pre-ILP candidate breadth at the `10 µm` cap lost `0.1596 / 0.1496`, and still lost
   `0.1319 / 0.1281` when handed a perfect oracle edge probability.

The shared failure is embryo-family transfer. M1 also worsened raw linking
(`0.6499 -> 0.6421`); its small composite gain came from emitting 20% fewer nodes and
receiving count-multiplier credit. It is not “nearly passing.”

Full figures, canonical files, and recovery commits are indexed in
[`reports/EXPERIMENT_LEDGER.md`](reports/EXPERIMENT_LEDGER.md). The complete pre-clean tree
is recoverable from Git tag `pre-lean-2026-07-30`.

## Active decision

Do not run more generic training, extra seeds, full-data fitting, or ensembles.

The only active work is [`reports/NEXT_DECISION.md`](reports/NEXT_DECISION.md). Its branch A
(expanded pre-ILP candidate breadth, the one apparently unmeasured mechanism in the clean
public pipeline) was **closed on 2026-07-30 with zero GPU spent** — the organizer GEFFs are
pruned to one parent per target at `p >= 0.5`, so the mechanism was genuinely unmeasured, but
its oracle ceiling collapses in practice: widening enumeration to the `10 µm` cap loses
`0.1596 / 0.1496` and still loses `0.1319 / 0.1281` under a perfect edge probability, because
the one-to-one per-frame assignment lets each false edge displace a true one. Reproduce with
`scripts/branchA_gate.py --stage all`.

**The primary track is now divisions.** E0c scores `division_jaccard` `0.0000 / 0.0057` — it
harvests essentially none of a term worth up to `+0.1`. D0' (2026-07-30) measured the
composition Oracle C never tested — remove existing false forks, then reconstruct reachable
true ones — at **`+0.0783 / +0.0737`** bilateral (GT-free child-retention control; `+0.0818 /
+0.0783` with it). Suppression and reconstruction are worthless apart and strongly
super-additive together, because E0c's fork layer is essentially noise: 11,441 / 9,012 forks,
of which `0` and `2` sit on a true divider. Reproduce with
`scripts\win_bet\phaseb_oracle_d0prime.py`.

That is a GT-informed oracle ceiling, not a result. Fork selection is oracle in every arm and
nothing there is submittable. The live question is realizability, and the previous division
classifier was killed on the wrong instrument: it was gated at precision `0.9`, whereas after
suppression `J = k/(26+m)`, so a detector at 30–50% precision clears the `+0.005` gate.

Active queue is in [`reports/NEXT_DECISION.md`](reports/NEXT_DECISION.md): D0 operating-point
reanalysis (needs an inference-only re-run — no saved posterior predictions survive), then D1
covariance fork audit, then D2 joint re-optimiser. Branch B is demoted behind divisions; the
stop rule is suspended while the division track is live.

GPU remains blocked until a mechanism passes a deployable gate. Final hedges stay E0c for
private robustness and v122 for public strength, with a third slot reserved for the division
track if it graduates.

## Canonical assets

- `artifacts/kaggle/oof_clean/` — fold-specific organizer OOF predictions.
- `artifacts/kaggle/e0c_cache/` — authoritative post-wrapper graphs and candidate surface.
- `artifacts/kaggle/weights_dataset/` — fold-specific checkpoints/configs.
- `reports/inventory/e0c_score_full.txt` — exact E0c score.
- `reports/inventory/coupled_score_2026-07-29.txt` — v122 decomposition.
- `reports/inventory/m1_selection.json` and `m1_heldout_result.json` — M1 verdict.
- `reports/inventory/branchA_*.json` — branch A gate evidence (surface audit, oracle
  ceiling, widened run, oracle-probability ceiling, filter control).

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
