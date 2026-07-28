# Instrument audit — does any OOF statistic rank our public submissions?

**Date:** 2026-07-28
**Reproduce:** `.\.venv\Scripts\python.exe scripts\win_bet\instrument_audit.py`

## Why this was run

Model selection has been gated on embryo-held-out OOF (min-fold ≥ +0.005) for the whole
campaign, but the relationship between that gate and the deployment metric had never been
measured. The audit asks one question: **is min-fold OOF directionally discriminative for
public score?** It cannot and does not attempt a public-score calibration — six public
observations cannot support one.

## Primary result — the audit CANNOT be completed as specified

Qualifying evidence requires post-patch scoring (official `075fc5f`) **and** proven
deployment parity. **Exactly one submission qualifies (n = 1).** Rank correlation on
qualifying data is not computable.

| sub | public | family | 44b6 | 6bba | min-fold | edge-wt | tier | parity |
|---|---:|---|---:|---:|---:|---:|:--:|---|
| 54290725 | 0.815 | classical V3 (DoG) | 0.6320 | 0.7560 | 0.6320 | 0.7369 | B | unproven |
| 54301967 | 0.741 | classical op_bright | 0.6840 | 0.7510 | 0.6840 | 0.7407 | B | unproven |
| 54601594 | 0.865 | Trackastra direct + pruning | 0.6948 | 0.6044 | 0.6044 | 0.6183 | B | unproven |
| 54534923 | 0.889 | E0c learned wrapper | 0.7595 | 0.6490 | 0.6490 | 0.6660 | **A** | **PROVEN** |
| 54588144 | 0.889 | Trackastra hint in relinker | — | — | — | — | U | n/a |
| 54854143 | 0.908 | clean v122 coupled det/ILP | — | — | — | — | U | n/a |

Tier definitions: **A** = post-patch + deployment parity proven. **B** = exact all-199 OOF
with matching edge-volume weighting, but pre-patch and/or parity unproven. **U** = no exact
OOF; excluded entirely and never estimated.

Per-submission disqualifications:

- `54601594` — OOF computed 2026-07-12, **pre-patch**; `075fc5f` changed the division term.
- `54301967` — kernel `aryaarun07/biohub-op-bright` never parity-checked against the local
  199-crop config.
- `54290725` — same parity gap; additionally see the public-score drift below.
- `54588144` — no OOF artifact exists (HANDOFF records `—` for both folds).
- `54854143` — no OOF yet. This is exactly what the C0/C1 experiment must produce.

## Two incidental findings that stand on their own

1. **Aggregation is comparable.** `scripts/run_phase1_ablation.py::fold_summary` weights by
   `w = edge_tp + edge_fp + edge_fn`, identical to `tracking_cellmot.metrics.summarise`.
   Verified before computing anything; had it differed, the table would be meaningless.
2. **The public column itself moved.** `JOURNAL.md` records `54290725` at **0.807** on
   2026-07-03; the Kaggle API now returns **0.815**. The patch rescore therefore *did* shift
   a non-exploit submission, contrary to the host's "non-exploit submissions should not
   change." **Any historically recorded public score must be refreshed from the API before
   use.**

## Degraded diagnostic (tier A+B, n = 4) — NOT a calibration

| statistic | Spearman | Kendall |
|---|---:|---:|
| min_fold | −0.400 | −0.333 |
| edge_weighted | −0.800 | −0.667 |
| arith_mean | −0.400 | −0.333 |
| harm_mean | −0.400 | −0.333 |
| **44b6_only** | **+0.800** | **+0.667** |
| 6bba_only | −0.600 | −0.333 |

Leave-one-out sensitivity (min_fold): drop `54301967` → **+0.500**; drop `54534923` →
**−1.000**; drop either other → −0.500.

Child-vs-parent sign concordance:

| pair | Δpublic | Δmin-fold | verdict |
|---|---:|---:|---|
| 54301967 vs 54290725 | −0.074 | +0.052 | **INVERTED** |
| 54601594 vs 54534923 | −0.024 | −0.045 | AGREE |

## Interpretation — this does NOT convict min-fold

The negative coefficients are **not** evidence that min-fold is broken:

1. **Family confound.** The n=4 set mixes two pipeline families and the sign is driven by
   that: classical systems show *high* OOF / *low* public; learned systems the reverse.
   Within-family, only one child-parent pair exists per family.
2. **Total instability.** Leave-one-out swings from −1.000 to +0.500. A single observation
   controls the result — and the observation carrying the positive signal is E0c, the only
   tier-A point.
3. **The inversion is unattributable.** `op_bright` drives every discordant pair, and its
   deployment parity is unproven (predicted "~0.81", scored 0.741). An instrument failure
   and a config-drift failure are **indistinguishable** with the artifacts that exist.

`44b6_only = +0.800` is the only positive statistic. At n = 4 it should be treated as noise,
but it is a cheap hypothesis to re-test once a second qualifying observation exists.

## Limitations (explicit)

- n = 1 qualifying; n = 4 degraded. No rank statistic here is inferentially meaningful.
- No public-score mapping was fitted, and none should be.
- Density/regime slice comparison is **unavailable** for every tier-B submission.
- Three of four degraded points are pre-patch, so their public and OOF values come from
  different scoring epochs.
- The audit cannot distinguish instrument failure from deployment-parity failure.

## Decision

- **Keep LOEO min-fold as the private-safety gate.** It is not refuted, and it must not be
  retuned to six public observations.
- **Do not promote any deployment-ranking statistic** on this evidence, including
  `44b6_only`.
- **The instrument is uncalibrated and no existing artifact can fix it.** The only route to
  a second qualifying observation is deployment-parity post-patch OOF for the coupled
  operating point — i.e. the C0/C1 experiment. Phase 1 is therefore doubly load-bearing:
  the primary attack *and* the sole source of calibration evidence.
