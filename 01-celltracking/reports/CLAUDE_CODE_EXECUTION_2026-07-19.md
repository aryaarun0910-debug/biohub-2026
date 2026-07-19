# Claude Code execution order — aggressive attrition campaign

**Date:** 2026-07-19  
**Mission:** maximize private-leaderboard performance without metric exploitation or public-test overfitting.

## Operating doctrine

Keep every compute lane doing useful work, but require a falsifiable gate before expanding a branch. The unit
of aggression is **information gained per GPU-hour**, not model size. Preserve the frozen E0c graph as fallback;
all learned changes are selective augmentations unless they beat E0c on both embryo-held-out directions.

Before acting, read `CLAUDE.md`, `HANDOFF.md`, and the 2026-07-14 final synthesis. Preserve the dirty files
listed in `HANDOFF.md`. Record and commit every result according to `CLAUDE.md`.

## Track 0 — scoring epoch reset (ACTIVE; finish first)

1. Confirm requirements pin official scorer commit `075fc5f`.
2. Run the focused project metric tests and the upstream patched division suites.
3. Finish the full 199-crop E0c rescore.
4. Re-run no-fork and Division Oracle C add-only/add-replace under the patch.
5. Update `HANDOFF.md`, final synthesis, roadmap, win bet, and journal with post-patch numbers; commit.

Nothing may be promoted across the scoring-epoch boundary without this rescore.

## Track 1 — clean-frontier capture (CPU OOF, then Kaggle inference)

Port the public `yusuketogashi/biohub-clean-approach-no-metric-hacking` configuration as isolated wrapper
flags. Its declared base is clean LB 0.903. Reproduce only the coherent published configuration—no sweep:

- detector threshold 0.9700;
- gap close 5.8 um, max gap 2;
- local/prefix density adaptation (reference 6.5 um, gain 0.040, cap 0.125 um, blend 0.20);
- min track length 6;
- conservative safe-division geometry/caps;
- no off-volume nodes, cross-clip edges, artificial forks, or test-ID conditionals.

Run deployment-exact two-fold OOF against patched E0c. Promote only if both folds improve and min-fold delta
is at least +0.005 with no regime-slice regression. If it fails, retain E0c and do not tune on test movies.

## Track 2 — corrected temporal-signal falsification (Kaggle T4 + local CPU)

The July 14 pilot is not adequately powered for a permanent kill: 44b6 n=3, and controls were selected using
cached full-volume scores but evaluated using patch scores. Repair it before any temporal model:

1. Sample at least 10 crops per embryo family, stratified by density, depth, time, and miss rate.
2. Target at least 200 isolated-miss events per family where possible.
3. Score positive and control tubes through the identical inference path.
4. Match controls within crop/frame/depth/density and center-response quantile.
5. Compare single-frame, static 3/5/7-frame, GT-motion, and estimated-flow accumulation.
6. Bootstrap by crop; report motion-minus-static effect and precision at the allowable node budget.
7. Insert selected nodes/edges into the exact patched graph and score the composite.

GO only if both families show positive crop-bootstrap lower bounds and >=15–20% isolated-miss recovery without
adjusted-edge-J loss. Otherwise permanently kill temporal training.

## Track 3 — compact affinity-field model (GPU only after Track 2 GO)

### Interdisciplinary formulation

Treat tracking as a mixture of:

- **radar/astronomy track-before-detect:** integrate sub-threshold evidence before declaring an object;
- **Lagrangian tissue mechanics:** predict motion relative to a smooth material flow, not global coordinates;
- **positive-unlabeled learning:** sparse annotations are positives, never exhaustive background labels;
- **error-correcting codes:** forward, backward, and skip-frame paths must agree;
- **survival/hazard modeling:** birth, continuation, disappearance, and division are competing events;
- **selective prediction:** abstain to E0c whenever the learned posterior is not calibrated.

### Minimal model

Input two or three adjacent 3D volumes. Use a compact 3D U-Net with time as channels or shared features plus
warping. Heads: peak evidence, forward displacement/affinity, backward displacement/affinity, calibrated
existence confidence. Division is deferred until the patched oracle proves material ceiling.

Pretrain on public dense Zebrahub/Ultrack trajectories plus synthetic motion/intensity corruptions. Fine-tune
with masked peak-margin and PU losses on competition labels. Add path-consistency loss across t->t+1->t+2
versus t->t+2. Generate dense pseudo-links only where E0c, Trackastra, and an independent flow/Ultrack view
agree. Train one seed. Expand to three seeds only after bilateral improvement.

The model never owns the complete graph. It proposes node/edge repairs; E0c remains fallback.

## Track 4 — red-team wildcards (cheap gates only)

Run these only as bounded falsifications, not open-ended projects:

1. **Photon-statistics matched filter:** whiten local background and integrate likelihood ratios along motion
   tubes instead of averaging sigmoid scores. Gate on Track 2's same event set.
2. **Topology-aware uncertainty:** use local graph persistence under perturbations (threshold, TTA, linker) as
   confidence. Repair only edges whose winning topology is stable across perturbations.
3. **Competing-risk lineage solver:** calibrate continuation/birth/death/division hazards by developmental
   stage and local density, then solve under one-parent/two-child constraints. Gate against E0c, never generic
   likelihood.
4. **Counterfactual corruptions:** delete/shift/attenuate known GT nodes and train the repair scorer to undo
   only those corruptions. This creates dense supervised repair examples without declaring unlabeled cells
   negative.
5. **Sequential probability ratio test:** accumulate evidence over time and stop as soon as a tube is clearly
   real or noise, reducing both inference cost and false nodes.

Each wildcard gets one implementation and one bilateral test. Kill immediately if min-fold gain is <+0.005.

## Compute ownership

- **Local CPU (max 4 data workers):** exact scorer, graph transforms, bootstraps, calibration, manifests,
  provenance, OOF reports. Cache everything; never rerun inference for a threshold sweep.
- **Local MX350:** tiny patch smoke tests only.
- **Kaggle T4:** response-map caching and first-seed compact model.
- **Kaggle T4x2:** independent folds/seeds or true data parallelism only after a gate. Do not waste two GPUs on
  batch sizes dominated by synchronization.

Run Track 0/1 CPU work concurrently with Track 2 GPU preparation. At most two confirmatory configurations per
round because the effective validation sample size is two embryos.

## Submission gates

A submission requires patched exact OOF, both folds up, min-fold >=+0.005, regime-slice audit, runtime <9 h,
internet-off dry run, immutable commit/config/hashes, and provenance/license manifest. Public LB is a deployment
check only. Never use the patched division exploit or any test-ID-specific logic.
