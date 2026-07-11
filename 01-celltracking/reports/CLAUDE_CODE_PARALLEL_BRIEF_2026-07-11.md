# Claude Code parallel mission — Biohub private-board attack

**Date:** 2026-07-11  
**Objective:** produce work that is additive to Codex's active tracks and moves
the prize-eligible private score, not another public-only fork.

## Read first

- `reports/INTEL_0968_2026-07-11.md`
- `reports/PROVENANCE_0968_AUDIT_2026-07-11.md`
- `reports/WIN_PLAN.md`
- `reports/COMMANDER_LOG_2026-07-10.md`
- `vendor/kaggle-cell-tracking/src/tracking_cellmot/metrics.py`
- `vendor/kaggle-cell-tracking/src/tracking_cellmot/division_metrics.py`

Assume Codex may be wrong. Red-team the conclusions, but support every contrary
claim with executable evidence.

## New verified situation

1. Our public calibration is 0.889, rank 358/1,054 in the July-11 snapshot.
2. Kevin moved from 0.896 on submission 17 to 0.968 on submission 18: exactly
   +0.072 in one attempt. His method is private.
3. Score is adjusted-edge-J plus `0.1 * division-J`. The node-undercount
   multiplier is unclipped, so the score supremum is approximately 1.2.
4. The division evaluator's unmatched-fork/weak-component defect has been
   reproduced locally: it can add almost +0.1 without changing edge TP/FP/FN.
   **This is quarantined and must never enter a submission.** Treat it as
   forbidden/DQ-risk unless the host explicitly rules otherwise in writing.
5. Published `div07`/`division_prior09` safe geometry failed our bilateral OOF
   slice: 0 TP from 7 proposed forks. Cheap geometry does not transfer.
6. Full-OOF isolated-node pruning is a real permitted lever:
   - held-out `44b6`: 0.65610 -> **0.67448**, +0.01839
   - held-out `6bba`: 0.55892 -> **0.56970**, +0.01078
7. Trackastra/organizer agreement fusion passed bilateral 10-crop transfer:
   +0.0196 combined on `44b6`; +0.0398 on `6bba`. Full 199-crop candidate export
   has completed and Codex is downloading/scoring it now.
8. Public Biohub March-22 tracks are genuinely aligned competition supervision.
   Two recovered crop mappings give 96–98% GT-node recall at 7 um; direct public
   trajectories score 0.8490 and 0.9016 adjusted edge-J. This proves same-source
   value but does not prove the hidden embryo is public.
9. The four visible test movies are only byte-identical execution placeholders.
   The host confirms leaderboard scoring substitutes a larger private set.

## Division of labour — do not duplicate Codex

Codex currently owns:

- downloading and exact-scoring full 199-crop Trackastra fusion;
- isolated/short-component pruning sweeps on frozen predictions;
- metric bookkeeping, deployment integration, Kaggle kernels and submissions;
- monitoring the 0.968 outlier and any host metric response.

Claude Code owns the following parallel fronts.

### Front C1 — dense public-trajectory pretraining/calibration (highest EV)

Turn the proven March-22 alignment into a prize-safe association asset—not exact
hidden-label transfer.

Deliverables:

1. Inspect the provenance report and reproduce its two mappings/checks.
2. Build a compact training table from public dense trajectories with physical
   motion, acceleration, neighbor-coherence, density, track age, split geometry,
   and corrupted detection hypotheses.
3. Train the smallest useful edge/fork posterior or calibrator. Prefer a model
   that trains locally or in one Kaggle T4 job and consumes organizer/Trackastra
   candidate features rather than another full detector.
4. Evaluate strictly leave-one-embryo-out on all available competition OOF:
   train/calibrate without evaluated-embryo labels, freeze, score exact metric,
   reverse direction.
5. Go gate: at least +0.010 edge-J on both folds or +0.015 min-fold with neither
   fold worse than -0.003. Record hashes, commands and per-fold results.

Rules boundary: public-data pretraining and generic priors are permitted. Exact
source fingerprinting followed by direct trajectory-label transfer remains
`NEEDS WRITTEN HOST CLEARANCE`; do not implement it in a submission.

### Front C2 — legitimate learned division posterior

The goal is to earn the division bucket biologically, not imitate the evaluator
defect.

Features to test:

- parent-track deceleration/contraction and age;
- two-daughter near-symmetry and separation vector;
- daughter intensity/shape change and combined mass conservation;
- organizer and Trackastra edge logits/agreement;
- local tissue velocity residual and neighbor coherence;
- density, depth, developmental time and frame-freeze/jump indicators.

Use candidate forks already proposed by the linkers plus conservative nearby
unassigned daughters. Calibrate probabilities on one embryo, freeze threshold,
evaluate the other, then reverse. Output exact division TP/FP/FN/J, edge delta,
count delta and combined delta.

Go gate: positive combined delta on both folds. Do not optimize aggregate score
if one embryo loses. Do not use weak-component membership, unmatched artificial
forks, boundary-edge tricks, or any other evaluator-defect feature.

### Front C3 — frame-freeze/global-jump robust association

The latest discussion reports exact duplicate frames and sudden global volume
jumps. Verify frequency in all train crops, then add a frame-pair regime detector:

- frozen frame: reuse identity/nearest compatible assignment;
- ordinary motion: learned/Trackastra fusion;
- global jump: estimate robust population transform before local association.

Use only image/detection evidence. Gate on exact bilateral OOF, with special
reporting for affected frame pairs. Go at +0.003 min-fold or a large error drop
on jump frames without global regression.

## Required output

Write `reports/CLAUDE_PARALLEL_RESULTS_2026-07-11.md` containing:

- executive verdict and ranked findings;
- exact commands, artifact paths and hashes;
- per-embryo metric tables;
- what was killed and why;
- changes made and tests run;
- the single result Codex should integrate first.

Commit only Claude-owned files. Preserve all unrelated dirty files. Do not push
a Kaggle submission; Codex owns submission selection so evidence remains clean.

## The forcing question

Which prize-eligible mechanism best explains a plausible private improvement of
at least +0.03 after removing public-board artifacts: public dense-trajectory
calibration, legitimate division prediction, frame-regime-aware linking, or
aggressive metric-aware pruning? Prove the answer on both embryos rather than
arguing from intuition.
