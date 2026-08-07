# Gate C — exact TTA-group audit and the true-D4 arm

**Audited:** 2026-08-07 · Raw work: `..._RESEARCH\agent_runs\v7_diagnostic_20260807\gateC\`

# VERDICT: **KILL the true-D4 arm.** Keep the deployed seven-view path.

**Ceiling +0.00068** at the 95% upper bound — **22.1× short of +0.015, 7.4× short of even
+0.005**. **Cost 6.5–11.2 T4-h**, mutually exclusive with the conditional 7.88 T4-h split-1 job.

The defect is real, is now measured to several decimal places, and **is not measurably costly.**

---

## Task 1 — the seven-view finding, re-derived by EXECUTION

Not by reading. The lane `exec`s every top-level statement of every notebook cell that touches
`predict_unet_transformer.py` against a sandbox copy, then runs the resulting block against a
tracing `model.encode`, identifying each tensor against D4 by **bit-exact index-permutation
equality**. Faithfulness check: the replay reproduces the real Kaggle log's four patch prints
and the D1 block hash.

**8 encode calls · 7 distinct views · divisor 8.** Weight multiset:

```
{ identity 1/8 · rot90_k1 1/8 · rot90_k2 1/8 · rot90_k3 1/8
  flipY 1/8 · transpose 1/8 · flipX 2/8 · anti-transpose 0/8 }
```

Call 7's non-square shape is the tell (`12,7`, not `7,12`). Identical in f0, f1 and harmonic;
square and non-square; 5-D and 6-D.

**Two facts the earlier work did not establish:**

1. **Every forward/inverse pair in the deployed block is a bit-exact round-trip.** The defect is
   purely the *choice* of view — there is no misalignment. The output is mis-**weighted**, not
   corrupted.
2. **The secondary block carries the identical collision.** Dead in the strict smoke
   (`loeo_retarget.py:130`, `secondary_enabled: false` in the pulled manifest) but **live at
   weight 0.475 on the public P3 harmonic path** — so **the public detector carries the defect
   twice.**

v6 re-verified after it merged: the `A0` block's `det_logits` path and encode-input sequence are
**bit-identical to pre-v6**, still 7 distinct views. *(`kaggle_p3_harmonic` was not rebuilt.)*

## Task 2 — the arm

`TRUE_D4_BLOCK`, built and verified, **never applied**. 40/40 forward/inverse round-trips
bit-exact across 5-D, 6-D, square, non-square and W=1; D4 closed with all 8 distinct.
**Inference-cost-neutral** — 8 encode calls either way. **Squareness confirmed irrelevant**,
as §0b already claimed.

## Task 3 — measured on real checkpoints, 3 crops, 63 frames, both folds

CPU validated against the deployed GPU logits at the audited voxels (max 3.8e-6, zero threshold
flips); deployed-block reconstruction bit-identical.

| quantity | result |
|---|---|
| accepted peaks changing identity | **19.5%** (44b6 10.7%; 6bba 21.6% / 24.4%), flat across thresholds 0.50–0.99 |
| max \|Δlogit\| | **1.331** on 44b6 frame 0 — replicates the prior 1.330 exactly |
| features | rel-L2 0.008–0.017, cosine ≥ 0.9999 |
| `unet_out` | **untouched by both arms — the view set is not a representation lever** |
| edges (`predict_edges` RERUN, not bounded) | on genuinely unchanged node pairs, max \|Δlogit\| 0.30 and **max \|Δ softmax prob\| 0.723** |
| **GT recall — the decisive number** | **net +2 of 659** annotated GT (11 gained, 9 lost) |

Per C7, families separately: **44b6 is saturated** (12/12 under both arms); **6bba nets +2 of
647.**

### Correction to a number I quoted

I previously reported "201 of 1,780 accepted peaks (~11%) change identity." **The fraction
replicates; the counts do not.** At the deployed operating point that frame has **256** accepted
peaks with a symmetric difference of **45** — the 1780/201 pair was measured at a much looser
criterion than the deployed one. Quote the fraction, not the counts.

### C3, confirmed with a number

**`max |Δ softmax prob| = 0.723` on node pairs that did not themselves change.** That is the
cross-attention/source-softmax coupling correction C3 asserts, now measured rather than argued:
a changed node population moves existing edge probabilities by up to 0.72 — through a fixed
absolute 0.5 admission threshold. No association artifact survives a detector change.

---

## The primitive worth keeping

**Orientation dependence, measurable at zero extra encodes.** The TTA average over multiset `S`
under a re-orientation `g` depends only on the coset `Sg`:

| | distinct cosets | accepted-set symmetric difference | max \|Δlogit\| |
|---|---:|---:|---:|
| deployed 7-view | **8** | up to **35%** | 2.567 |
| true D4 | **1** | **0** | ≤ 2.9e-6 |

That is the honest operational content of *"the deployed average is not a group average"*: the
deployed detector's output **depends on how the volume happens to be oriented**, and a true group
average would not. It is a **variance / reproducibility** property — and **the metric does not
pay for it.**

Which is exactly why this closes: the defect is decisively measurable and decisively unprofitable.
