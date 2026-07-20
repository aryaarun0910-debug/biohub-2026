# Biohub expansive attack regime

**Date:** 2026-07-20  
**Objective:** reach the clean public frontier immediately, then win the private board
with model/data diversity that survives embryo-held-out validation.

## Situation

- Live public leader = 0.982; twenty teams are at 0.964 or above. Our deployed E0c is
  0.889.
- The board is contaminated by public kernels that add negative-time, out-of-volume
  hub/fork structures to exploit division scoring. Those predictions are quarantined:
  they are incompatible with the patched organizer metric and create prize-audit risk.
- The clean public lineage has independently reached 0.909. Its current v122 candidate
  changes detector threshold/ILP calibration together: detector 0.9690, appearance 0.0,
  disappearance 1.5, min-track 6, density-adaptive gap closing. Our failed clean-0.903
  OOF test covered only the wrapper delta on frozen 0.990 detections; it did not test
  this coupled detector/ILP operating point.
- Every previously identified repair-only lever is exhausted. Winning now requires a
  stronger detection/association model or a legal source/data advantage.

## Attack fronts

### Front 0: recover the clean leaderboard frontier now

1. Reproduce public clean v122 exactly in a private internet-off T4 kernel.
2. Audit output for valid times/coordinates, consecutive edges, out-degree <=2, and no
   artificial hubs/forks.
3. Submit immediately if the audit passes. This is a deployment probe, not evidence of
   private generalization.
4. If v122 underperforms the confirmed 0.909 v120 anchor, reproduce v120's detector
   threshold 0.96875 with the same disappearance cost 1.5 as the fallback.

### Front 1: establish the new honest baseline

Regenerate embryo-held-out predictions at the coupled operating point instead of
post-processing frozen 0.990 detections. Run only two configurations in round one:

- C0: v120-equivalent, threshold 0.96875 / disappearance 1.5.
- C1: v122-equivalent, threshold 0.9690 / disappearance 1.5.

Run the patched exact scorer on both full families. Promote only if both folds improve
and min-fold gains at least 0.005 over E0c. Cache probability peaks/candidate edges once
so threshold and graph calibration do not rerun the U-Net.

### Front 2: train models that the public monoculture does not have

1. Train a full-data organizer UNet+transformer on both competition embryos, reserving
   stratified internal validation only for checkpoint selection. Save last and proxy-best
   checkpoints; never select on public LB.
2. Train a second seed or independently augmented checkpoint only after the first model
   improves both embryo-held-out folds.
3. Ensemble probability/edge logits before graph solving. Do not average final graphs.
4. Add clean D4 spatial TTA as a separate ablation; it must pass the exact both-fold gate.
5. Do not retry Trackastra graph replacement: its bilateral OOF/public failure already
   establishes that it is not useful diversity at the final-graph level.

### Front 3: use external breadth where it can still matter

Geometry-only Zebrahub reranking is dead; external data must train the image model.

1. Build a provenance-locked pretraining corpus from NIS3D (CC-BY-4.0) plus public
   Zebrahub/March-22 imagery and dense trajectories after recording a usable license.
2. Pretrain center evidence and next-frame affinity/motion heads with physical resampling,
   PSF/noise/gamma perturbations, missing-node corruption, and localization jitter.
3. Fine-tune on competition crops; evaluate strict embryo-held-out folds.
4. Kill if the min-fold node/edge ceiling does not improve by 0.005. Scale to multiple
   seeds only after this first transfer gate passes.

### Front 4: legal target-time robustness

- Estimate movie quantiles, PSF/blur, density, displacement scale, and developmental
  stage without labels; choose normalization and a predeclared model mixture from those
  descriptors.
- Build a low-resolution fingerprint index for public Biohub movies, initially using a
  match only to select normalization/motion priors.
- Direct registration and transfer of public source trajectories into identified hidden
  clips remains quarantined until the organizer explicitly approves that use in writing.
- No gradient weight updates on test movies without written clearance.

## Compute allocation

- Kaggle T4 now: clean v122 reproduction and submission artifact.
- Next Kaggle T4: full OOF cache for C0/C1, then all-data training.
- Subsequent GPU only after gates: external pretraining; one seed first.
- Local CPU continuously: exact patched scoring, output audits, crop bootstraps, model
  disagreement analysis, provenance manifests, and submission validation.
- Keep one authoritative artifact per run with kernel version, code hash, model/data
  hashes, configuration, runtime, and exact score.

## Promotion policy

- Public score moves deployment priority but never selects a private model by itself.
- A new core model must beat E0c on both held-out embryos with min-fold +0.005.
- A public submission must be in-volume, temporally valid, metric-patch-compatible,
  internet-off, reproducible, and provenance-clean.
- No more than two confirmatory parameterizations per round. Expansion comes from
  independent models/data, not dozens of constants fitted to two embryos.

## Active operations

- `aryaarun07/biohub-clean-v122-reproduction` v1: RUNNING on T4.
- On completion: download, audit, submit, journal exact public score.
- In parallel: prepare C0/C1 OOF cache kernel and the full-data training kernel.

