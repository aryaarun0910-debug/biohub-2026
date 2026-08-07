# Cycle output — first lean cycle (post-reset)

**Date:** 2026-08-07 · **Baseline in:** `7143ef0` · **Head out:** see git log
**GPU spent: 0. Kaggle pushes: 0. Submissions: 0. Score movement: 0.000.**
Public stays **0.915** (P3 harmonic). 552 tests, 53 claims, `0 0` with origin.

## 1. What this cycle was asked to do

Stage 1 of the lean sequence: the two free CPU measurements, before any GPU.
Stage 2: verify the rebuilt 2x2 smoke. Stage 3+ were explicitly not authorised.

## 2. What came back

### Corpus C/L census — DELIVERED, and it closes a lane

199/199 crops, 133,318 annotated GT nodes, scorer-exact.

| | 44b6 (71) | 6bba (128) | pooled |
|---|---:|---:|---:|
| detection recall | 95.836% | 91.175% | **91.881%** |
| arbitration-loss nodes (`L_assign`) | 41 | 18 | **59** |
| no accepted peak <=7 um (`NA7`) | 800 | 9,965 | **10,765** |

**Split/merge arbitration is CLOSED**: 59 nodes -> **+0.000581 pooled**. The prior
+0.00045 and +0.0012 bracket it; **+0.019 is refuted by ~31x**. Cause identified, not
asserted: the old 3-crop `M = 1,521` was the POST-wrapper node match of a degraded smoke
submission (adj-J 0.33 on 6bba vs deployed 0.648) differenced against a PRE-wrapper peak
count of 1,794, so the "273-node class" was measuring wrapper/solver loss.

**Detection is NOT closed**: +0.08088 at p=1, +0.03764 at p=0.7/phi=2, bilaterally
positive to p ~ 0.6, break-even **p ~ 0.55**.

### The result that changes the plan

Acceptance **0.990 -> 0.969**, all 199 crops: **~530-590k nodes admitted to recover 1,656
GT**. Marginal precision **~0.003 where the ceiling needs ~0.55** — two orders of
magnitude short. Verified independently on the only two crops holding both thresholds:
`44b6_0113de3b` +6,420 nodes -> **+0** GT; `6bba_05b6850b` +381 nodes -> **+1** GT.

**Re-acceptance cannot be a threshold; it must be a ranker.** This prices the HANDOFF's
stage-5 fallback ("one scalar node-budgeted threshold before retraining") at **-0.0234
pooled**. That fallback is not cheaper — it is closed.

### Probability-at-the-deployed-gate oracle — NOT RETURNED

Still running at the session limit. It carries the KILL rule (< +0.015 closes every
association-scoring proposal by arithmetic). **Nothing in this document depends on it**,
and no decision below assumes either outcome.

## 3. Two defects fixed that would each have wasted a GPU run

1. **D1-F would have raised `ContractError` on the artifact, after the GPU was paid for.**
   `validate_manifest` hard-required a top-level `n_views` that the aggregator never wrote
   — and never should: this block makes **8 encode calls over 7 distinct views**, because
   `rot90(imgs,1,dims=(-2,-1)).transpose(-1,-2)` IS `imgs.flip(-1)`. The probe now REFUSES
   the merged field and checks what the two counts assert about each other.
2. **Nothing assembled the four kernel outputs into a readable shape.**
   `scripts/assemble_d1_factorial.py` merges them into `basis_N/`, re-deriving role from
   `SPLIT_SOURCE_FAMILY`, checking the sha256 the kernel observed for its loaded weights,
   and refusing a basis that is not exactly one source + one target — the routed-only shape
   that made the probe refuse in the first place.

## 4. Corrections issued this cycle

- **The scorer MAXIMISES `1/(1+d)`.** It does not minimise distance. Gate A's matcher was
  wrong; the census matcher agrees node-for-node with `metric_numpy.match_nodes`.
- **`clean903_wrapper_oof_cache` is "clean903 wrapper on frozen 0.990 detections"** — a
  strict subset of deployed 0.969, unusable for metric claims. Confirmed independently:
  its `raw_nodes` for `44b6_0113de3b` is exactly the 28,119 of the `det-0.99` npz.
- **"44b6 holds 2.42% of at-stake edge mass" is substrate-specific**, not a corpus
  property — **10.80%** on the deployed export. The stable quantity is the scorer weight
  share **14.86% / 85.14%**. This WEAKENS the private-embryo portability worry; it does not
  remove it.
- My own admitted-node count is 527,276 vs the lane's 588,766 (11.7%, most likely nodes
  carrying no candidate edge). Marginal precision 0.00314 vs 0.00281 — immaterial.

## 5. What is NOT done

- **The oracle lane has not reported.**
- **Corpus `T` is not CPU-reachable.** It needs the detector's rejected maxima and no
  logit volume exists on disk for any crop. The 3-crop extrapolation is 3,100-7,400 nodes,
  **2.4x wide, with zero 44b6 basis**. Producing corpus T is now the strongest reason to
  run the smoke — stronger than the plumbing check it was scoped as.
- **`phi` has never been measured for a TARGETED re-acceptance.** The sign of the ceiling
  below p ~ 0.6 is entirely a phi argument.
- Stages 3, 4 and 5 not started. No GPU authorised.

## 6. Decisions outstanding

1. **Re-scope the smoke.** It was plumbing; it is now the only route to corpus `T` and to a
   first `phi` measurement on a targeted re-acceptance. Recommend running it for that.
2. **Retire the scalar-threshold fallback** from the stage sequence. It is measured and
   negative; leaving it in the plan implies an option that does not exist.
3. **`.gitignore` negation** for `data/d1_factorial/` — still untouched, per standing rule.
