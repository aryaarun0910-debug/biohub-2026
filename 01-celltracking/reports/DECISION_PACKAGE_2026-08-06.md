# Decision package — v5 audit + 10-lane swarm

**Date:** 2026-08-06 · **Commits:** `2c91bc6` → `7f943b5` · **Platform:** P3 harmonic **0.915**
· leader ~0.948 · **submissions this cycle: 0** · **GPU spent this cycle: 0**

Basis tags: `EXACT-OOF` · `LOEO` · `IN-FAMILY` · `ORACLE` · `PUBLIC-LB` · `LITERATURE` ·
`HYPOTHESIS` · `SOURCE-EXACT` (deterministic read of pinned source/checkpoint) ·
`FIXTURE` (synthetic, calibrates scale only).

---

## 1. What actually happened

The cycle began intending to launch a 199-crop D1/D1-F export. **It was not launched, and that
is the correct outcome.** Four blockers were found, three of which would have silently
corrupted the result rather than failing loudly.

| # | blocker | would it have failed loudly? | status |
|---|---|---|---|
| B1 | kernel emits no `d1_class` / `matched`; the partition is a missing CPU stage | no — audit simply unrunnable | **RESOLVED** |
| B2 | `d1f_probe.py` unloadable; H0 fitted not read; **H1–H4 byte-identical** | **no** — four identical arms read as convergent evidence | open, spec'd |
| B3 | features identity-view, logits post-**8**-view TTA | **no** — would have returned a plausible verdict | open, spec'd |
| B4 | polars schema drop kills gt-only columns on **28/199 crops** | **no** — reports `COMPLETE` with correct `gt_rows` | **FIXED + locked** |
| B5 | family and checkpoint perfectly confounded | **no** — returns "representation deficit" w.p. ≈1 | open, needs your call |

Plus one trap caught *while writing the fix*: every lane proposed accumulating the TTA mean
into `unet_out`, which is read downstream by `predict_edges`. That would have destroyed the
0.915 association substrate under cover of a bug fix. Locked by test.

**Net:** 4 commits, tests 90 → 110, one instrument built (`scripts/d1_postprocess.py`), one
spec written (`reports/D1_V6_SPEC.md`), zero GPU, zero submissions.

---

## 2. Deduplicated mechanism map

Twelve distinct mechanisms were proposed across ten lanes. After deduplication and reconciling
against recorded failures, **four survive**, three are conditional, five are dead.

### Alive

| mechanism | lever | evidence | cost |
|---|---|---|---|
| **M1 · Re-acceptance head on the frozen peak set** | head + calibration | partition says T=54.5% of unmatched `IN-FAMILY, 3 crops` | 0 extra GPU |
| **M2 · Count-constrained GE on blob-corrected peak count** | head training | units proven `EXACT`; guard demonstrated `FIXTURE` | 0 extra GPU |
| **M3 · Relative/top-1 admission replacing absolute 0.5** | dense deployment | in-degree ≤1 is arithmetic `SOURCE-EXACT` | 1 CPU replay |
| **M4 · Sampling-phase augmentation** | encoder + head training | 38.5% of 6bba GT flip peak status across phase `IMG-SURROGATE` | training run |

### Conditional

| mechanism | gate |
|---|---|
| **M5 · Temporal pseudo-positives, ghost-only** | ghost real-cell precision ≥ 0.50 on 6bba (L3-K2, 20 CPU-min) |
| **M6 · Masked autoencoding** | only if a *valid* D1-F reads "representation" |
| **M7 · Association re-measurement** | only after a detector change lands, and only via `η = ΔTP/ΔA` |

### Dead this cycle

**JEPA-lite** (NO-GO: pretext signal 0.0092, 85% at noise floor, 10⁴× below smallest published
scale) · **mask-only H1 as a candidate** (best recall, *worst* score, over-detects 3.92× —
measured) · **nnPU** (non-SCAR; parked pending a densely-annotated region) · **illumination
re-aim** (sign-opposite across families, both CIs excluding zero, 199/199) · **anti-aliased
decimation / phase-union / anisotropy pool / axial deconvolution / drift normalisation** (all
falsified for ~15 CPU-minutes total).

---

## 3. The three executable candidates, ranked by expected value per GPU hour

### #1 · v6 export + D1-F done correctly — **EV/GPU-hour: effectively infinite**

Not a scoring candidate; the **gate every other candidate depends on**. Costs **~0 extra GPU
FLOPs** — the flipped encoder passes already run and discard three feature maps into `_`. The
change binds and accumulates them into a separate tensor.

Unlocks: M1, M2, M5, M6, and the entire representation-vs-head question. Blocks nothing.

**Do this first regardless of everything else in this document.**

### #2 · M1 — re-acceptance on the frozen peak set — **EV/GPU-hour: high, bounded**

Peak extraction stays on `w_ckpt`; only the *acceptance score* becomes `w′`. The peak set is
then exactly unchanged, so **the CPU measurement is exact and there is no CPU→GPU gap** — the
usual killer of probe results.

- **Ceiling:** `|T ∪ L|` × 6.7547e-06 pooled. On the 3-crop smoke T+L = 81.5% of unmatched
  `IN-FAMILY` — encouraging, but 3 crops, one deliberately extreme. **Not a corpus estimate.**
- **Pre-gate, 1 CPU-minute:** count corpus `|T ∪ L|`. If **< 3,400**, the route cannot reach
  +0.015 at any precision and dies before a model is fitted.
- **Why it is cheap:** the 33-parameter head is exactly the family that commutes with D4 TTA,
  so **N candidate heads score from one encoder pass**.

### #3 · M2 — count-constrained GE, blob-corrected — **EV/GPU-hour: moderate, economics hostile**

- **Must clear +1.184% relative J just to break even** (it surrenders +0.008556 by hitting the
  count target), **+3.259%** to promote. 44b6 pays twice what 6bba does, and promotion needs
  44b6 ≥ −0.002. `LOEO`
- **Two implementation traps, both measured:** the naive Topaz target is unit-wrong by ~34×
  (mass vs peaks), and it fails *silently because the count converges*; and the GE term must
  never see the exported row set (prevalence 7–489× too high; fixture collapses to 0.0000).
- **Kill rule that matters:** if raw `edge_jaccard` is flat while adjusted J rises, that is
  count-matching, not detection. **Report unadjusted J beside adjusted J for every arm.**

**Not ranked:** M4 (sampling-phase augmentation) is the most *interesting* new mechanism but
requires training, and **we have never trained this model** — no preflight, no wall-clock, no
resume test. It is a Phase-2 item, not a 72-hour item.

---

## 4. Dependencies on the v5 census

| consumer | dependency | status |
|---|---|---|
| M/C/T/L/D class counts | CPU postprocessor over the export | **3 crops only** — `IN-FAMILY`, diagnostic |
| D1-F / representation verdict | TTA-mean features | **BLOCKED** until v6 |
| `|T ∪ L|` pre-gate for M1 | corpus partition | **BLOCKED** until v6 full-199 |
| H2 `pi_crop` | `N_est` metadata only | **READY** — needs no export |
| H3 temporal table | ghost provenance flags | **BLOCKED** — needs a wrapper flag, CPU-only |
| association work | per-node `det_logit`, pre-threshold probs | not in v6 scope; deliberate |

**The 3-crop partition may not be quoted as a census.** D=0 and T=54.5% are real and
invariant-checked, but one crop was deliberately selected as extreme.

---

## 5. 72-hour execution schedule

**Submissions: zero.** The bar is +0.020 pooled and nothing on the shelf is within reach of it.
Slots consumed 11; do not spend a twelfth on a probe result.

| window | track | work | GPU |
|---|---|---|---|
| **0–4 h** | CPU | six cheap gates (§6). Any one can redirect everything after it | 0 |
| **4–10 h** | CPU | implement v6 patch per `D1_V6_SPEC.md`; local composition gate; injection idempotency; `unet_out` read-only test | 0 |
| **10–11 h** | **GPU** | push **3-crop v6 smoke**, both folds | ~0.5 T4-h |
| **11–14 h** | CPU | v6 acceptance: manifests 3/3 · partition parity 52/52 · **head/TTA logit parity** · **association bit-identity vs v5** · peak VRAM | 0 |
| **14–20 h** | CPU | rewrite `d1f_probe.py` against Lane 2's integration manifest (fitter already built + 76 tests) | 0 |
| **20–28 h** | **GPU** | full-199 v6 export, 2 shards | ~7 T4-h |
| **28–36 h** | CPU | global audit: 199/199 · 20,197 + 113,121 GT · finite features · partition invariants · no fold leakage · hashes | 0 |
| **36–44 h** | CPU | **D1-F both directions**, capacity ladder P0→P5, ranking *and* calibration reported separately | 0 |
| **44–48 h** | — | **verdict checkpoint.** Representation vs head vs calibration | 0 |
| **48–72 h** | contingent | if HEAD/CAL → build M1 candidate + M3 admission replay. If REPRESENTATION → training preflight (never run) before anything else | ≤ 4 T4-h |

**Total GPU ≈ 11.5 T4-hours.** Fold 1 already consumes 76.6% of a 9-hour session at current
settings, so **there is no room for a second encoder pass — heads must share one kernel.**

---

## 6. The six cheap gates (hours 0–4, all CPU, any can redirect the rest)

| # | gate | cost | kills / decides |
|---|---|---|---|
| 1 | **Recall-ceiling re-audit at true density** (9.68 µm spacing, anisotropic PSF) | 30 min | re-prices the *entire* detection route |
| 2 | `\|T ∪ L\|` corpus count *(needs v6; run at hour 28)* | 1 min | M1 lives or dies by arithmetic |
| 3 | **SAR separability test** (propensity vs classifier attributes) | 2 h | admits or forbids all PU machinery |
| 4 | **L3-K2 ghost precision** | 20 min | M5 lives or is BLOCKED |
| 5 | **Margin census** `m_j = 1 − 0.5/p_j` | 15 min | whether densification helps or *hurts* association |
| 6 | **Quarantine `augpath_oracle.json`** | 5 min | stops a retracted number being re-quoted |

Gate 1 is the highest-leverage item in this document. **If the ceiling at realistic density
comes back below 0.97, the Gaussian-target line reopens and the +0.10332 detection oracle must
be re-priced before any GPU is spent at all.**

---

## 7. Kill rules

**Instrument-level (abort before any science):**
- H0 parity residual `max|r| > 1e-4`, **or** residual systematically signed / correlated with
  the logit. **No "proceed with caveat" branch — the near-miss is the dangerous case.**
- Association not bit-identical to v5 on the v6 smoke ⇒ §0's trap was walked into.
- Any implemented arm byte-identical to another ⇒ pairwise objective fingerprints must raise.
- A null D1-F on identity-view features **may not buy GPU** (noise asymmetry biases it toward
  the expensive conclusion).

**Mechanism-level:**
- **M1:** `|T ∪ L| < 3,400`, or cannot reach 2,678 TP @ 80% precision / 4,080 @ 60%.
- **M2:** raw `edge_jaccard` flat while adjusted J rises ⇒ count-matching, not detection.
- **M3:** any re-aim that admits more pairs *in aggregate* is branch A and is refused.
- **M5:** ghost real-cell precision < 0.50 on 6bba ⇒ **BLOCKED, not emitted.**
- **M6/JEPA:** `C_copy > 0.99` on TTA-mean features ⇒ closed for zero GPU.

**Promotion:** pooled ≥ +0.015 · LCB ≥ +0.008 · 6bba ≥ +0.015 · 44b6 ≥ −0.002.
**Submission:** ≥ +0.020 pooled. **Statistical:** any single crop > 25% of the point estimate
⇒ not corpus-general, clears no gate.

---

## 8. Do-not-run list

| do not run | why |
|---|---|
| full-199 **v5** export | four blockers; and never run v5 then repeat 199 for v6 |
| `scripts/d1f_probe.py` | six independent defects; four produce a plausible number rather than an error |
| any GE term on the **exported row set** | prevalence 7–489× too high; fixture collapses to 0.0000 |
| nnPU as a classifier | non-SCAR; no dense region to estimate π |
| JEPA-lite | NO-GO; and MAE is a prerequisite via the teacher trilemma |
| mask-only H1 as a candidate | best recall, worst score — it is an ablation |
| velocity / raw-distance / gate-radius sweeps | closed; lever ≤ 2.26e-05 pooled |
| pre-ILP candidate breadth / gate widening | −0.1596/−0.1496; **worse with a perfect oracle probability** |
| anything on `p0strict_cache` | route 1 dead, measured four ways |
| `augpath_oracle.json`'s numbers | retracted post-wrapper run, 40 crops not 199 |
| top-K parent export as a ranking fix | true-parent-preferred 9.36% vs ~50% break-even — 5.3× short |
| appearance channel for association | AUC 0.657/0.496, 7–232× dependent on existing scorers |
| Topaz **repo** reuse | GPL-3.0 — clean-room from the paper only |
| public LB as an instrument below ~0.005 | ~64× less sensitive than OOF |

---

## 9. The proprietary combination

Every component below is individually public. **The combination is not, because it is specific
to arithmetic facts about this pipeline that no public notebook has cause to exploit.**

> **CTPU-RA — Count-constrained, TTA-exact, Re-Acceptance head.**
>
> 1. Exploit the **exact linearity** of the 33-parameter head under D4 TTA:
>    `detect_head(mean_v aligned f_v) ≡ mean_v aligned logit_v`. This is what makes a
>    frozen-feature probe *deployable rather than merely diagnostic*, and it lets **N candidate
>    heads score from one encoder pass** — decisive when fold 1 already eats 76.6% of a session.
> 2. Fit the head as a **re-acceptance rule on the frozen peak set**, not as a new detector.
>    The peak set is then exactly unchanged, the node population is controlled, and the CPU
>    measurement is *exact* — removing the CPU→GPU gap that has killed previous probe results.
> 3. Constrain with the count prior on the **blob-corrected soft peak count**, not on
>    probability mass, plus the **second-moment guard** against diffuse smearing.
> 4. Replace the **absolute 0.5 admission** on a source-softmax with a **relative/top-1** rule.
>    This is the one change that defuses the dilution law (`m_j = 1 − 0.5/p_j`), under which a
>    denser detector *degrades* already-correct edges.
>
> Public work does (1) never, (2) rarely, (3) as mass, and (4) not at all. Steps 2 and 4 exist
> only because we measured that in-degree is ≤ 1 by arithmetic and that our metric does not
> forgive surplus — neither of which applies to the ILP-filtered pipelines the literature
> assumes.

**Falsification:** step 1 is an assertion (parity gate). Step 2's ceiling is `|T ∪ L|`,
1 CPU-minute. Step 3 is fixture-validated. Step 4 is one CPU replay. **The whole combination
can be killed for under an hour of CPU before any GPU is spent.**

---

## 10. Measured vs hypothesised — the separation, explicitly

### Measured, and I would defend each

| finding | basis |
|---|---|
| deployed TTA is **8 views**, `/_nv`, two blocks, guard is a `print` | `SOURCE-EXACT` |
| `unet_out` read by `predict_edges` at L442/L445, after the TTA block | `SOURCE-EXACT` |
| trap 21 fires on **28/199 crops**; smoke crops all start at frame 0 | `EXACT` |
| 3-crop partition: M 1469 · C 289 · T 849 · L 420 · **D 0**; 52/52 parity | `IN-FAMILY` |
| wrapper destroys **16 GT matches (1.05%)** on every crop measured | `IN-FAMILY` |
| `N_est` is a per-movie total — reproduces `total_node_ratio` 199/199, err 0.000e+00 | `EXACT` |
| π pooled 9.0577e-4; label fraction 0.028215; ratio ~31× | `EXACT` |
| count target surrenders **+0.008556**; needs +3.259% rel-J to promote | `LOEO` |
| node-ratio is **0.268%** of FP cost; recovered-node value / added-node cost = **546×** | `EXACT-OOF` |
| uniform rows are **1/4096** ⇒ **8.3178-logit** intercept correction | `EXACT` |
| `unet.head` has no activation ⇒ 32-D feature is an invertible affine image, rank 32 | `SOURCE-EXACT` |
| `cos(w_split0, w_split1) = −0.155`; fold bases independent (rel-L2 1.41542 ≈ √2) | `SOURCE-EXACT` |
| true nuclear spacing **9.68 µm** vs axial FWHM ~10 µm; 60.7% within own extent | `IMG-CENSUS` |
| illumination re-aim sign-opposite, both CIs excluding zero | `IMG-SURROGATE` 199/199 |
| 38.5% of annotated 6bba nuclei flip peak status across sampling phase | `IMG-SURROGATE` |
| in-degree ≤ 1 by arithmetic (source-softmax + absolute 0.5) | `SOURCE-EXACT` |
| 96.69% of the pseudo-positive pool is at already-accepted voxels | `EXACT-OOF` |

### Hypothesised — must not be quoted as ours

- Every `FIXTURE` number (H1 vs H2 adjJ, the 34× blob correction's *magnitude*, guard ρ values).
  These calibrate **scale and sign**, not separability on real data.
- All literature-reported gains (CellSeg3D F1 0.74, Nishimura 0.853→0.881, Kervadec DSC).
- Every projected T4-hour figure for work not yet run.
- The interpretation of background relL2 = 0.1009 as a *noise floor* (the number is exact; the
  label is inference).
- **That D=0 and T=54.5% generalise.** Three crops. One chosen as extreme.

### The one thing I got wrong, recorded

I reported the TTA view list as "exactly 4" from the vendored file. That file is replaced at
build time; the deployed count is 8. **The base file is not the run.** Every constant quoted
from `vendor/` in the reports is subject to the same error and should be re-checked against the
generated notebook.

---

## 11. The single recommendation

**Run the six CPU gates, then build v6, then measure D1-F properly. Do not submit.**

The detection route is still the only asset large enough to close 0.033. But this cycle
established that **our instrument was wrong in four independent ways, three of them silent**,
and that two "settled" numbers (the 1.0000 recall ceiling and the node-ratio constraint) rest
on foundations narrower than their use. Fix the instrument, re-price the ceiling, and let the
first honest D1-F decide between a 33-parameter refit and an encoder programme.

**Do not let a probe result buy a submission slot.** Sampled GT/background F1 is not the
competition objective; only dense inference → P3 harmonic → complete wrapper → exact patched
pooled scorer can promote an arm.
