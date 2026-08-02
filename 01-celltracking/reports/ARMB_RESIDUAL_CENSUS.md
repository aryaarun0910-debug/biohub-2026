# Arm-B residual-error census + sister-gate design gate

**Date:** 2026-08-02 · **Substrate:** P0-strict OOF, 50 cached crops · **Scorer:** exact patched
**Basis:** CPU replay of the corrected arm-B wrapper, coordinates exported exactly as the deployed
kernel (`max(0, round(v))`), scored against `data/train` GT. `prob = 0` proxy (established
adequate, ≤2.26e-05). **No GPU, no submission slot.**

Written while Kaggle scores submission `55181562`. Nothing here was launched.

## 1. Where arm B's remaining error actually is

```
score 0.734918 = adj_edge_J 0.734418 + 0.1 · div_J 0.005000
edge     TP/FP/FN : 28904 / 4193 / 6704
division TP/FP/FN : 1 / 151 / 48      precision 0.658%   recall 2.04%
mean node_recall  : 0.902756
mean total_node_ratio : −0.147596
```

### Corrected oracle ceilings

A repaired FN becomes a **TP**; it does not vanish from the denominator. My first pass scaled FN
out of the denominator instead, which understated every recall ceiling and produced a nonsensical
division reading. Corrected:

| oracle | Δ score |
|---|---:|
| **edge recall perfect (FN → TP)** | **+0.170959** |
| division perfect (both) | +0.099500 |
| edge precision perfect (FP = 0) | +0.082648 |
| **division recall perfect** | **+0.024000** |
| edge: 10% of FN recovered | +0.017096 |
| division: 50% of FN recovered | +0.012000 |
| edge: 10% of FP removed | +0.007317 |
| division precision perfect | +0.001541 |
| node ratio → 0 | **−0.008205** |

**Edge recall is worth ~7× everything division-related.** Recovering a mere **10% of missed edges
(+0.0171)** exceeds the entire +0.011 needed for 0.925 — and beats *perfect* division precision by
11×.

### By family — 6bba is the whole problem

| family | n | score | edge FP | edge FN | div FP | div FN |
|---|---:|---:|---:|---:|---:|---:|
| 44b6 | 18 | 0.901429 | 453 | 324 | 31 | 7 |
| 6bba | 32 | **0.701218** | 3,740 | **6,380** | 120 | 41 |

6bba carries **95.2% of the missed edges** at a score 0.200 below 44b6. Consistent with 6bba being
85.06% of edge mass and "worst measured" (correction 3).

### A live warning on node count

`total_node_ratio = −0.1476`: we under-predict node count by ~15%, and the count adjustment
`J·(1 − 0.1·ratio)` **pays us +0.0082 for it**. Any primitive that adds nodes surrenders part of
that bonus before it earns anything. Gap-close, gap2 recovery and any candidate-broadening
proposal must be scored net of this. It is not a free axis.

## 2. Division sister-gate — designed, measured, and CLOSED as a relaxation

### First: target the LIVE gate, not the dead one

Correction 6 killed `DIV_SISTER_MAX_UM = 8.0` — it sits inside `OUTPUT_DIVISION_GEOMETRY_FILTER`,
which defaults `"0"` and is **never set in the deployment env**. Re-confirmed this session.

The **live** sister gate is `SAFE_DIV_SISTER_MAX_UM = 8.5`, set explicitly in cell02 and applied in
`add_safe_divisions_postlink`. Any sister-gate work must target that constant. Designing against
`DIV_SISTER_MAX_UM` would repeat correction 6 exactly.

### The gate is genuinely mis-centred

GT division geometry, all 199 training crops, 151 true divisions:

| quantity | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|
| **sister (daughter↔daughter)** | **10.570** | 14.357 | 15.343 | 20.300 |
| parent → nearer daughter | 4.083 | 6.858 | 7.410 | 11.207 |
| parent → farther daughter | 7.130 | 10.050 | 11.781 | 13.529 |

**The median true division has its daughters 10.57 µm apart — the gate sits at 8.5 µm, below the
median.** It rejects **70.9%** of true divisions, versus 42.4% for `SAFE_DIV_MAX_UM = 4.66` and
40.4% for `SAFE_DIV_EXISTING_CHILD_MAX_UM = 7.65`. It is the hardest-binding of the three, and it
is the only one mis-centred rather than merely tight. Only **26.5%** of true divisions clear all
three gates.

### Why relaxing it still loses — the falsification

Break-even precision for *adding* k division candidates, from
`J' = (TP+p)/(D+k−p) > TP/D` ⟹ `p/k > TP/(D+TP)`:

with TP=1, D=200 → **0.498%**. Trivially clearable — *if the division term were the only term*.

It is not. Every safe division also emits an **edge**, which lands in edge FP when the division is
wrong. Measured both sides together:

| added k | div gain @ q=5% | edge cost @ q=5% | **net** |
|---:|---:|---:|---:|
| 50 | +0.000914 | −0.000858 | +0.000056 |
| 100 | +0.001534 | −0.001714 | **−0.000180** |
| 200 | +0.002321 | −0.003419 | **−0.001098** |
| 400 | +0.003121 | −0.006804 | **−0.003683** |

**The added candidates must reach ≈20–25% precision to be net positive**, not 0.5%.

What would relaxing 8.5 → 12.0 µm actually deliver? True divisions unlocked: **+23 of 151**
(1.58×). Candidate volume grows as r³: **(12/8.5)³ = 2.81×**. Scaling to this sample, relaxation
admits ≈274 extra candidates carrying ≈7.5 extra true divisions — **precision ≈ 2.7%**.

At q≈2.7% and k≈274 the division gain is ≈+0.0011 against an edge cost of ≈−0.0047:
**net ≈ −0.0036. FALSIFIED.**

This is the project's signature failure repeated a seventh time: **the oracle clears (perfect
division recall = +0.024) and the selector fails (achievable 2.7% vs required 20–25%).**

### What is NOT falsified

A **re-aim at constant admission count** — arm B's own winning move, change the gate *quantity*
not the radius — remains open, because it incurs no extra edge cost. Any precision improvement is
then pure gain. It must still be worth having: at constant k, the whole division term tops out at
**+0.024** even with perfect recall.

**My first attempt at that measurement was invalid and is withdrawn.** I tested
`sister_dist / local_spacing` using GT node density, and it looked strongly falsified (CV 0.2997 →
0.4845, i.e. 62% *worse*; family transfer also worse, p50 ratio 0.783 → 0.707). But GT geffs
annotate only ~670 nodes per crop against ~25,000 predicted — GT is **~2.7% as dense**, so GT
spacing (p50 22.9 µm) is not the quantity a deployed gate would compute (which uses the prediction
graph via the already-deployed `frame_local_spacing`, GAP_DENSITY_NEIGHBORS = 3). The measurement
falsifies a variant nobody would ship. It neither supports nor refutes the deployable one.

## 3. Recommended next primitive — and it is not divisions

**Attack 6bba edge recall.** It is worth +0.171 at the oracle, needs only 10% capture to clear the
+0.011 to 0.925, and 95% of the mass sits in one family. Divisions cap out at +0.024 perfect and
+0.0015 for perfect precision.

Before funding anything, the two standing rules apply. **Substrate transfer:** any constant must
be re-measured on 6bba, never inherited from 44b6. **Oracle-clears/selector-fails:** do not fund
another selector over an existing candidate population without a base-rate argument — and the
edge-FN population is exactly that, so the base-rate argument must come first.

Re-run the sister-gate falsification with:
```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe scripts\armb_census.py --stride 4 --workers 6 --gate B
```

## 4. Caveats

50 crops, **49 true divisions and 1 TP** — division counts are small-sample and the ratios are
fragile. `prob = 0` proxy. The local wrapper has no DeepCenter gap veto (safe divisions are
unaffected — deployment sets `DEEPCENTER_SAFE_DIV_VETO = 0` — but gap-close differs slightly).
Two crops emitted `No matching nodes found` during scoring. The r³ candidate-growth model is an
approximation, not a measurement; the measured quantity is the true-division unlock (+23), and the
2.81× is inferred.
