# Cycle output — 2026-08-07 · v7 diagnostic cycle

**Start:** `39705de` · **End:** `95d4b46` (pushed, `0 0`) · **Tests:** 224 → **487**
**GPU spent: 0** · **Kaggle pushes: 0** · **Submissions: 0** · **Score movement: 0.000**
Public **~0.915** · leader **~0.949**

Delivered **before** any GPU spend, as instructed. Phase 3 (smoke, then pilot) has not started.

---

## 1. The question this cycle was asked

> Decide, with the smallest reliable spend, whether the detection branch contains a realistic
> route to ≥ +0.020 pooled OOF and ultimately 0.925+ public.

**Answer: the ceiling permits it; no mechanism has yet been shown to reach it; and one
irreducible risk could close it regardless.** Detail in §6.

---

## 2. The four decision gates — all returned, all free

| gate | verdict | the number that decided it |
|---|---|---|
| **A** recall ceiling at true density | **PROCEED** | ≤ **3,410 of 15,296** misses (22.3%) optically unresolvable; ceiling **+0.062–0.069**, above +0.020 for any precision **p ≥ 0.55** |
| **B** synthetic corpus (disc. 732103) | **REJECT** | licence unverifiable **in fact**; single fit-free statistic separates corpora at **AUC 1.0000** |
| **C** true-D4 TTA arm | **KILL** | ceiling **+0.00068** — 22.1× short of +0.015; net **+2 of 659** GT recovered |
| **R1/R2/R3** research | delivered | see §5 |

### A — the density objection was half right, and it was the wrong half

23–34% of the true nuclear population **is** merged under the measured PSF. But
`MAX_DISTANCE = 7.0 µm` **exceeds the entire merge scale**, so a merged pair still sites its
maximum inside the scorer radius **97.7–98.3%** of the time.

**Merging is falsified as the cause of the misses.** 85.94% of GT centres already have a local
maximum within 7 µm — but only **58.27%** have an *accepted* one, and **66.30%** of unmatched
GT still carry an unaccepted maximum inside the radius. Crowding stratification runs
**backwards** (accepted rises 15.06% → 89.19% across crowding quartiles), and 44b6 is 3.4×
denser while missing **9.7× fewer**.

> **The detection deficit is a calibration/acceptance problem, not an optical one.**
> That is the most actionable finding of the cycle.

### C — the defect is real, measured, and not worth fixing

The deployed TTA makes **8 encode calls over 7 distinct views** (`rot90(1)∘transpose` **is**
`flip(-1)`), so `flip(-1)` carries 2/8 and the true anti-transpose 0/8. Re-derived by
*executing* the notebook's patch strings against a tracing `encode`, not by reading.

Two facts newly established: every forward/inverse pair **is** a bit-exact round-trip, so the
output is mis-*weighted*, not corrupted; and **the secondary block carries the identical
collision**, dead in the strict smoke but **live at weight 0.475 on the public path** — the
public detector carries the defect twice.

Correcting it moves 19.5% of accepted peaks and `max |Δlogit| = 1.331` — and recovers **net +2
of 659 annotated GT**. Large, and worth almost nothing. **KILL.**

**C3 is now confirmed with a number.** With `predict_edges` rerun rather than bounded, node
pairs that did not themselves change show **max |Δ softmax prob| = 0.723** through a fixed
absolute 0.5 admission threshold. No association artifact survives a detector change.

---

## 3. The finding that changes policy

> ### The +0.020 submission bar rests on a statistical error.

`CYCLE3_LANE_O_AND_D0.md` §1 concluded "four-movie sampling variance is FALSIFIED" and that
conclusion is what raised the bar. It tested the hypothesis with a **point probability where a
tail probability is required** — it observed the flat outcome and quoted `P(flat) = 0.0322` as
a p-value. From its **own published table**:

```
P(down) + P(flat) = 0.0683 + 0.0322 = 0.1005
at the actual public mix:  0.0737 + 0.0352 = 0.1089
```

**At p ≈ 0.10–0.18, sampling variance is not falsified at any conventional level.** We *fail to
reject* it. Section retracted; hypotheses B/C/D remain live.

**And the bar gates the wrong variable.** Scaling arm B's per-crop deltas leaves the panel
t-statistic **invariant at ≈1.15**: a 4× larger arm has `P(public up)` 0.914 against the 1×
arm's 0.890 — while a **homogeneous** +0.005 has `P(up)` **1.0000**. The decision variable is
**cross-crop homogeneity** (`t4 = 2/CV`). Arm B's CV is **1.71** against the 1.22 needed.

> **Arm B never had a 95% chance of showing on a four-movie panel at any magnitude whatsoever.**
> Raising a magnitude bar could not have fixed that, and cannot now.

Two corollaries. The leaderboard is an **asymmetric instrument** — a public win is worth
BF ≤ 2.4–3.9, a clear public loss **13–46:1**; weak confirmer, strong refuter. And **churn does
not predict transfer** (Spearman **−0.63**; "always predict 0.000" beats it 3/4) — its two
apparent successes were one fact stated twice.

---

## 4. Every instrument we relied on was broken

This is the cycle's real output.

| instrument | defect | would it have failed loudly? |
|---|---|---|
| `d1f_probe.py` | **six independent verdict-path defects** | **no** |
| `target_extractor_ceiling.py` | **degenerate by construction** | **no** |
| Lane O panel analysis | point probability used as a p-value | **no** |
| `d1_postprocess.py` | could not read a v6 export at all | yes — `SystemExit` |
| v6 spec | advertised a gate **no code read** | **no** |

**D1-F's six**, every one found by *running* it: a label-shuffled gate at `0.5 ± 0.05` where the
statistic has **sd 0.148** (a standard error of a 20-seed *mean* applied to one draw);
`rank_net` that algebraically collapses to `lin−null`, clearing its threshold even when the
refit ranked **worse** than H0 (−0.000146); a `mean + 3·sd` ceiling on a `[0,1]` statistic
making LINEAR_HEAD **unreachable**; a null regime that **crashed** instead of returning the
verdict built for it; CALIBRATION returned from a corpus with **no** feature/label association;
and **H3/H4 silently BLOCKED** — the arms whose distinctness the instrument exists to prove.

**The 1.0000 recall ceiling was degenerate.** The script max-combines unit-amplitude kernels and
tests `vol == pooled`, so every centre is a peak **by construction** — **27 nuclei in a single
pool window still returns 1.0000**. It could only ever fail on voxel collisions, measured
0/133,318. It may no longer be quoted as evidence about detectability.

**The v6 spec advertised a gate that did not exist**: the assembler emitted
`BIOHUB_D1_REQUIRE_TTA_VIEWS=8` that no code read. It "worked" only because both real audit
defaults happen to be correct — and the single number it recorded was the 8/7 conflation.

---

## 5. Research: what is now known

- **Acceptance is a LEVEL SET.** `(logits == maxpool) & (sigmoid > τ)` is invariant under any
  strictly increasing φ, and the local-max test runs on raw logits. So temperature, Platt, beta,
  binning and isotonic are **all worth exactly one scalar** here. A bare `CALIBRATION` verdict
  was uninterpretable; it is now split into `CALIBRATION_GLOBAL` / `CALIBRATION_PER_CROP`.
- **A prior-shift/SLD/BBSE correction is wrong-signed** — it prescribes **raising** 6bba's
  threshold by 0.808 logits, the family with 9.7× the miss rate and 85% of edge mass. Refused at
  intake in code.
- **The field is over-split, not merged**: **86.0%** of accepted peaks already have another
  accepted peak inside the 10 µm axial FWHM. The `(5,3,3)` pool is excluded by geometry at
  corpus scale (≈−0.026 pooled).
- **The division metric's tolerance is exactly ±1 frame**, making ±1 temporal NMS
  metric-mandatory rather than a tuning choice. The division route's floor is **99, not 151**
  divisions on the strict substrate.
- **`royerlab/hoct`** — from the host lab. MIT, weights released, native 3D, writes GEFF, built
  on tracksdata, default tile is our exact volume. Edge-centric; its stated motivation is
  verbatim the pathology behind three of our closed lanes. **Blocking risk: mask-based.**
- **The field is one shared pipeline** — ~149 teams at exactly 0.913; the organizer baseline has
  had no commit since the metric patch we already pin.
- **New licence blocks:** DECODE GPL-3.0 · MAE CC BY-NC · ModelsGenesis non-commercial ·
  MDPAFOF, Bayesian-Crowd-Counting, and two competitor repos **unlicensed**.
  **New clears:** tapnet/TAPIR/LocoTrack **Apache-2.0 including checkpoints**, netcal Apache-2.0.

---

## 6. Where 0.925+ actually stands

**For:** the ceiling permits it (+0.062–0.069 at p ≥ 0.55). The deficit is calibration, not
optics — and calibration is cheap. T = 54.5% of unmatched GT is the population that lever
addresses, and the threshold lever is **not** closed on 6bba (the earlier "closed" result was
measured where recall was already 100.0%/99.54% — no headroom existed there).

**Against, and I weight this more heavily:**

- **44b6 holds 2.42% of at-stake edge mass. Its entire detection oracle is +0.0025.** If the
  private embryo behaves like 44b6, the route **cannot** clear +0.020 at any recall or
  precision. Irreducible on two embryos — not a missing experiment.
- **The recovery requirement is 1.27–2.56× worse than I stated.** 65.8% of at-stake edge mass
  has **both** endpoints missing, so nodes do not convert independently. +0.020 needs
  **≈5,400–7,600 nodes at 70% precision**, not 2,961. Recovery **order** is a real lever:
  coherent is worth ~2× random.
- **The transfer instrument is weak in the direction we need.** A public win is worth
  BF ≤ 3.9; homogeneity, not magnitude, decides visibility.

**The biggest unmeasured parameter in the project:** two independent competitors state the
scored test contains **hidden movies beyond the four visible placeholders**, and is
embryo-disjoint. If ≥16 movies, arm B's flat is **200:1** against and transfer is a real
problem; if 4, it is **6:1** and mostly noise. **These are opposite worlds**, and the question
is believed resolvable for free from a consumed submission's scored-run log.

---

## 7. Corrections issued this cycle

**Mine:** the TTA view count (8 encode calls, **7 distinct**); the partition total (GT **3,079**,
M **1,521**, and the class split is **6bba-only, 2 crops, zero 44b6 representation**);
"+0.020 needs 2,961 recoveries" (**5,400–7,600**); "M1 is CPU-exact" (**it is not**);
"`|T∪L|` is the ceiling" (**it is not**); "201 of 1,780 peaks" (fraction replicates, **counts
do not** — they were taken at a looser criterion than deployed); `Y == X` as a safety property
(**refuted**); "the patch guard degrades silently" (**overstated**).

**Inherited:** Lane O's falsification (**retracted**); the **+0.136531 oracle** (inflated
1.658× — `28,732` is a **degree sum**, unique at-stake edges are **17,327**; **+0.10332
survives**, corroborated to 0.67%); CellSeg3D described as an MAE (**it is a SoftNCuts W-Net**);
"Calibrated Teacher +0.7–1.1 mAP" (that is the *combined* method; calibration-only is +0.2 on
the setting closest to ours).

---

## 8. What is NOT done

- **Phase 3 has not started.** No smoke, no pilot, no GPU.
- **The cross-encoded smoke build is still running.** It is a hard dependency: D1-F's
  `run_direction` **correctly refuses routed-only data**, and our smoke is routed-only. Without
  it, the 6 checkpoint×crop cells do not exist.
- **Everything in D1-F is FIXTURE.** No v6 export exists; v5 smoke is forbidden as training data.
- **Two ~1 CPU-hour measurements worth doing before any GPU:**
  1. **Corpus C census** — our split/merge ceiling disagrees with itself **43×** (66 nodes /
     +0.00045 vs 184 / +0.0012 vs the 3-crop C class implying ~2,830 / +0.019), straddling the
     bar. **Nobody should quote 18.5% as a corpus ceiling until this runs.**
  2. **Prob-oracle-at-gate** — the probability term can shift a pairwise cost by at most
     ~1.0 µm-equivalent against a 1.329 µm flow residual, so any true pair losing by >1 µm on
     geometry is unreachable by *any* edge-probability improvement. **Kill gate: oracle <+0.015
     closes every association-scoring proposal by arithmetic, zero GPU.** The oracle has only
     ever been run at the *widened* 10 µm gate, never the deployed one.

## 9. Decisions outstanding

1. **`.gitignore` negation** for `data/d1_factorial/` — manifests are tracked only by force-add,
   so regenerations are invisible to `git status`. Not done: `.gitignore` is under your
   do-not-touch instruction.
2. **Re-derive or retire the +0.020 bar** now that its basis is retracted. My recommendation:
   replace a magnitude bar with a **homogeneity** gate (`CV ≤ 1.22` for 95% visibility), since
   that is the variable that actually governs whether anything shows.
3. **Scope of the eventual split-1 run** — 7.88 T4-h, ~85% of the objective. Still recommended
   over the impossible 22.44 T4-h factorial, but **not yet earned**: the pilot has not run.

---

## 10. Standing rules unchanged

No submission from a probe or sampled-row metric. Promotion ≥ +0.015 OOF. **+0.020 is POLICY —
and now policy resting on a retracted argument.** The 199 crops come from **two embryos**;
crop-block bootstrap measures within-embryo variation and does not estimate private-embryo
generalisation — **report both transfer directions separately.** **The base file is not the
run.** Strict LOEO is **not** the public P3 detector.
