# Roadmap addendum — corrections, M2-CTPU, and the Phase-0 design

**Written:** 2026-08-05 · Supersedes the loss-provenance and D1-interpretation claims in
`ROADMAP_DETECTION.md`. Platform P3 harmonic **0.915**.

---

## C1 · Loss provenance — CORRECTED. I quoted a documented trap.

**I was wrong by 10×, and the project had already written the trap down before I hit it.**
`reports/ENVIRONMENT_TRAPS.md` line 1 and JOURNAL 2026-07-29 both record it; the literature review
at JOURNAL:53 states the host baseline uses `neg_weight=0.01`. I quoted the function-local default.

| symbol | value | source |
|---|---:|---|
| `compute_detection_loss(neg_weight=…)` | 0.1 | local default — **the trap** |
| `train_epoch(det_neg_weight=…)` | 0.1 | local default — **the trap** |
| **`train(det_neg_weight=…)`** | **1e-2 = 0.01** | the baseline path, passed positionally |
| **CLI `--det-neg-weight`** | **1e-2 = 0.01** | argparse default, help text agrees |

**`det_neg_weight = 0.01` is well-supported** — CLI default, `train()` default and the independent
literature review all agree. Locked by `tests/test_m1_driver.py` reading `train()`'s call site
via AST.

### A third discrepancy, not previously recorded

`det_loss_weight` is **internally inconsistent in the vendored source**:

| path | value |
|---|---:|
| `train()` signature | `1e1` = **10.0** |
| CLI `--det-loss-weight` | `1e0` = **1.0** |
| CLI help text | *"default: 1e1"* |

The CLI default contradicts its own help string. If the organizer invoked the CLI without the flag
they got **1.0**, not 10.0. **The shipped `config_split_{0,1}.json` records only architecture**
(`downsample`, `pool_kernel_um`, `unet_layers`, `unet_out_channels`, `window_size`, sha256
`e9b4e396c58081bc`, identical for both folds) — **no loss weights, no command, no log.**

### RESOLVED 2026-08-06 — a retained training_config settles it

The ordered search found a **retained `training_config.json` from an actual run of this trainer**
(`../Biohub-CellTracking-2026_RESEARCH/.../secondary_seed_weights/unet_transformer/split_0/`,
sha256 `4f29349439e133ad`):

```
det_loss_weight = 1.0        <-- the CLI default (1e0), NOT train()'s 1e1
det_neg_weight  = 0.01
pool_kernel_um  = 5.0        downsample = [1,4,4]    unet_out_channels = 32
method = unet_transformer_alltrain_seed314159_v1     train_datasets = 199
```

**`det_loss_weight = 1.0` and `det_neg_weight = 0.01`.** The CLI path was used, so `train()`'s
`1e1` signature default was never in play and the help-text contradiction is inert.

**Scope caveat, stated precisely:** this config belongs to the **secondary** seed model
(`alltrain_seed314159_v1`, all 199 datasets, fold 0), not to the primary support-pack 50-epoch
weights. It is direct evidence of how *this trainer is actually invoked* — same script, same CLI —
but it is not the primary pack's own log. Treat `1.0 / 0.01` as **operative and well-evidenced**,
not as byte-proven for the primary checkpoint.

**Status: `det_neg_weight = 0.01` and `det_loss_weight = 1.0`, evidenced.**
Neither may be quoted as runtime-proven until a training log or command is recovered. Any M2 arm
must fix both explicitly and regression-lock them.

### What this does to the thesis

Per-voxel weights are `w_pos = 1/n_pos` and `w_neg = 0.01/n_neg`, so **aggregate loss mass is
1.0 positive against 0.01 negative — 100:1 in favour of positives.**

- **Survives:** unannotated nuclei *do* receive negative gradient. There is no ignore mask. The
  sign of the effect is real.
- **Substantially weakened:** the magnitude. Describing the detector as "explicitly trained to
  suppress" the cells it misses overstates a term carrying **1% of the detection loss**.
- **Direct consequence:** the masked/ignore-radius fix removes something already small. **H1
  alone should now be expected to buy little**, and my earlier framing of it as *the* confirmed
  fix was overconfident.

This is why H5 and D1-F below are load-bearing rather than optional.

---

## C2 · D1 class D ≠ representation failure. `detect_head` is 33 parameters.

`self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)` with
`unet_out_channels = 32` → **32 weights + 1 bias = 33 parameters.** A low logit at a missing GT
nucleus is consistent with three different causes:

1. the frozen 32-D feature at that voxel is uninformative → **encoder**;
2. the feature separates but a 33-parameter linear map cannot exploit it → **head**;
3. the head separates but the operating point is wrong → **calibration/threshold**.

**Observing class D does not distinguish these, and must not authorise encoder retraining.**

### D1-F — frozen-feature linear probe (the decisive gate)

Export 32-D features once, fit on CPU, and ask: *does a better linear map on the **existing**
features recover the missing nuclei?*

- **Probe recovers them** → head/calibration problem. Fix the head. **Do not retrain the encoder.**
- **Probe cannot** → representation deficit is established, and only then is encoder work justified.

---

## C3 · Linajea masking is unsafe as a standalone deployment loss

Linajea deliberately leaves background **unconstrained** — *"we are not training our cell indicator
network on any background regions"* — and accepts surplus NMS candidates because a downstream ILP
filters them.

**Our metric does not forgive surplus.** It charges added nodes through `total_node_ratio` (we
currently *collect* **+0.008633** for under-predicting and would surrender it), and spurious
detections can steal an optimal-assignment match from an annotated cell.

**Therefore mask-only (H1) is an ablation, never the final design.** It must be paired with a
count or PU constraint that bounds how much mass the model may place on background.

---

## New primary mechanism — M2-CTPU

**Count- and Temporal-consistency Positive-Unlabelled residual detector.** Our own combination.
No leaderboard notebook is imported.

Three components, each answering a defect the corrections above expose:

1. **Masked local loss** — stop the (weak) wrong negative signal on unannotated nuclei.
2. **Training-only count prior** — replace the removed background constraint with a *global* one,
   so removing local negatives cannot licence unbounded surplus.
3. **Temporal pseudo-positives** — supply the positives that annotation sparsity withholds, from
   evidence that is GT-free at inference.

### Training-only count prior

Labelled training GEFFs carry `estimated_number_of_nodes`. **Unavailable on hidden test movies —
never a runtime input, never a routing feature.** Training supervision only.

```
pi_crop  =  estimated_number_of_nodes / (n_frames * Z_out * Y_out * X_out)
```

**Verify units before use** — `estimated_number_of_nodes` is a per-movie total, and `Z_out/Y_out/X_out`
must be the **downsampled output grid** `[1,4,4]`, not the raw volume. Sample **uniform voxels**,
not positive-centred batches, or the constraint measures the sampler rather than the field.

`N_est` is an estimate, not truth. **Pre-register sensitivity at ×0.5 / ×1.0 / ×2.0**, select only
on inner training-family validation, apply unchanged to the held-out family.

Implement as Topaz-style generalized expectation or a weak KL count loss.

**Do not confuse annotation frequency with `π`.** Our 0.655–8.529% is the labelling frequency.
Worse: **lineage annotations are biased positives, not SCAR** — annotators follow trackable
lineages, so labelled cells are systematically easier than unlabelled ones. Ordinary nnPU
assumptions (labelled-completely-at-random) are therefore **unsafe here** and nnPU stays parked.

### Temporal pseudo-positives

Generated **only from the training family**, using the fold-honest teacher. Held-out-family paths
must be structurally impossible to construct, not merely unused.

Reliable-positive rule (all must hold): node on a long unbranched track · forward/reverse
association agreement · low kNN-flow residual · stable under TTA/augmentation · no gap-close,
division or repair-generated edge · **no crop or family identity anywhere**.

**Treat these as positives only. Unmatched voxels are NOT negatives.**

Report coverage: tracks, nodes, crops, time regimes, intensity and density regimes. **Group and
weight by track and crop** so a handful of long tracks cannot dominate.

---

## Phase 0 — run in parallel where resources allow

| | task | resource |
|---|---|---|
| **A** | finish the P3 pregraph census rebase | 1 GPU (pregraph export) + overnight CPU |
| **B** | D1 A/B/D on **fold-honest** checkpoints | GPU |
| **C** | baseline training / resume preflight | GPU |
| **D** | sampled frozen-feature export | **same encoder pass as B** |

**B and D share one encoder pass.** Do not cache full U-Net volumes — export only 32-D vectors plus
provenance for: every labelled centre · points inside the local Linajea training mask · uniformly
sampled output-grid positions per crop · P3 high-confidence unannotated track nodes ·
under-threshold local maxima · temporal controls. Small enough to fit on CPU.

---

## Causal head-only arms (frozen organizer encoder)

| arm | change |
|---|---|
| **H0** | original detection head, exact parity |
| **H1** | Linajea-style local masked loss only *(ablation, not a candidate)* |
| **H2** | masked + training-only count/GE constraint |
| **H3** | masked + temporal pseudo-positives |
| **H4** | **full M2-CTPU** — masked + count + temporal |
| **H5** | **positive-exposure-balanced baseline, loss unchanged** |

**H5 is mandatory.** Annotation density and training-set size are confounded in the fold asymmetry
(44b6: 0.655% density *and* 5.6× fewer annotated nodes). H5 separates *"wrong negative
supervision"* from *"simply fewer positives"*. Without it the natural experiment proves nothing.

Heads are tiny — multiple 1×1 heads evaluate in one GPU encoder pass. Test both **replacement** and
**calibrated max-fusion** with the original head. **Never select on held-out-family results.**

---

## Evaluation — every arm, no exceptions

```
new detections → P3 harmonic association → complete current wrapper → exact patched pooled scorer
```

Report: held-out node recall and missing-node recovery · added-node match precision · edge TP/FP/FN
and **assignment stealing** · count multiplier · 44b6 / 6bba / pooled composite · **crop-block**
bootstrap interval · graph churn and structural invariants.

**Node recall improving is necessary, not sufficient. Do not translate recovered nodes into
composite gain** — recovered nodes cost node-ratio bonus and can steal assignments.

Promotion: pooled ≥ +0.015 · LCB ≥ +0.008 · 6bba ≥ +0.015 · 44b6 ≥ −0.002.
Submission: ≥ +0.020 pooled, or a separately justified high-information causal probe.

---

## If head-only fails — progressive unfreezing

Only then is this an encoder problem. Unfreeze in order: detection head → final decoder block →
larger encoder sections **only if required**. **Keep the transformer frozen initially.**

If shared U-Net layers change, preserve the working P3 association representation with a
**frozen-teacher distillation loss** on fixed candidate sets (features and raw edge logits).
Blind full-model fine-tuning risks gaining detections while destroying the 0.915 association
substrate — the single largest downside risk in this roadmap.

---

## JEPA — deferred to JEPA-lite

Not until CTPU improves and plateaus. Then: existing 3D U-Net · predict `t+1` target latents from
`t` · **align the target with the existing kNN flow first**, or the task learns bulk displacement
instead of cell identity · EMA or frozen target encoder · **1–3 epoch pilot** · linear-probe and
exact detector evaluation · feature-variance, covariance-rank and collapse diagnostics.

V-JEPA's published evidence comes from enormous video and model scale. It supports latent temporal
prediction **conceptually**; it does **not** establish that our ~19,900-frame 3D setting benefits.

---

## What changed in my confidence

| claim | before | now |
|---|---|---|
| background suppression strength | `neg_weight=0.1` | **0.01**, aggregate 1% of detection loss |
| `det_loss_weight` | 10.0 assumed | **1.0**, from a retained training_config |
| masked loss is *the* fix | high | **low-to-moderate** — it removes a 1% term |
| class D ⇒ encoder problem | assumed | **rejected** — 33-parameter head, D1-F decides |
| Linajea as final design | proposed | **ablation only** — leaves background unconstrained |
| detection ≫ association | +0.095 vs +0.033 | **unchanged** |
