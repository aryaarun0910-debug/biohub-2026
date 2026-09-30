# 03 WHAT THE TWO PRIOR CAMPAIGNS LEARNED

## Read this warning first

The second campaign, Biohub-X, was founded on a rule that no belief from the first
campaign could enter it unless Biohub-X reproduced that belief itself. That rule was
load-bearing: it is why Biohub-X's own findings can be trusted.

Arya dissolved that rule on 2026-09-07 when he ended both campaigns. This document
exists because of that decision. The consequence is real and should be stated plainly:
**the new campaign is not blind.** Anything you carry from below is inherited, not
reproduced. If a number matters enough to bet a submission on, measure it again.

Two further cautions:

- The monolith labels every recorded fact on two axes, provenance and validity. Some facts are `SUSPECT` or `INVALID`. Never quote a number from it without checking its validity field.
- One internal leak was found and voided. `EXP-0019` scored the `6bba` embryo using weights trained on `6bba`, a leave-one-embryo-out cross-validation leak. Its facts are marked `INVALID` and seven downstream facts and eight work packets were traced. The repository caught this itself and added a guard. Any number in that lineage stays retracted.

## Where the detail lives

All paths are inside `aryaarun0910-debug/Biohub-CellTracking-2026`.

| File | What it holds |
|---|---|
| `research/00-system/handoff.md` | The richest single document: cycle by cycle, what was believed, what killed it, what was licensed next. |
| `research/06-knowledge-system/experimental-records.md` | One compact row per experiment across the whole campaign. |
| `research/06-knowledge-system/lab-notebooks.md` | Day by day reasoning, including the dead ends. |
| `research/06-knowledge-system/failed-experiments.md` | Small and dense: closed lines with measured deltas. |
| `research/06-knowledge-system/internal-reports/` | 35 deep-dive reports. Metric forensics, retrain recipes, public-code teardown, the leave-one-embryo-out versus leaderboard gap. |
| `research/03-experimentation/quality-control.md` | The platform traps, reproduced in document 02. |
| `research/00-system/registry/facts.yaml` | About 460 facts with provenance and validity. |
| `research/00-system/registry/levers.yaml` | 46 hypotheses with their kill conditions. |
| `research/07-outputs/submissions.md` | Every submission with its source hash and leaderboard result. |

For Biohub-X, `DECISIONS.md` holds 49 decisions with reasoning, `SYSTEM.md` describes
every command, and `registry/reference.yaml` holds 64 reference records.

## Scores

| | Best public score | When |
|---|---|---|
| Monolith | 0.925, rank 207 of 2,693 | 2026-08-24 |
| Biohub-X | 0.496 | 2026-09-06 |
| Competition leader | 0.962 | 2026-08-24 |

The monolith's own note on its champion says the edge over its neighbours lived in a
bundle of three things together, a trained checkpoint, a veto step and a lambda
setting, and that one knob it had suspected, a 0.60 threshold, had **zero** effect.
Bundles are how that campaign kept fooling itself. Change one thing at a time.

Biohub-X's 0.496 is not a fair comparison. It was six days old, its detector was the
classical difference-of-Gaussians baseline, and it never trained a learned detector on
a GPU. Its value is not its score.

## What is closed, with magnitudes

From the monolith's own failed-experiments ledger. These are directions that were
tried and measured, so re-trying them costs days you do not have.

- Scalar-threshold re-acceptance: **−0.023**. Worse.
- Ground-truth-free component selector: **−0.008**. Worse.
- Node-budget pruning: the optimum was **no pruning at all**.
- Split and merge repair: **+0.0006**. Indistinguishable from noise.
- Edge test-time augmentation: hurts.
- HOCT was not a drop-in linker. CoTracker lost morphology through divisions.

## What Biohub-X established independently, and is worth keeping

These were measured on the throttled laptop and are stated in its registries.

- Sparse annotation makes any "background" label a fiction. Both campaigns converged on non-negative positive-unlabelled losses with a prior built from the window's own estimate. Do not skip this and call unannotated voxels background.
- A diagnostic label is a hypothesis until its counterfactual is measured. Biohub-X killed one of its own detector arms on a saturation heuristic, then found the heuristic was measuring background magnitude rather than whether the ranking still worked. The replacement rule looks at usable peak count, ties at the selection cut, and coverage against the classical baseline at a matched budget. That correction is worth carrying.
- Association is two separate problems. Ranking, which parent is best, and acceptance, whether to take any parent at all. Biohub-X's learned matcher ranked well and accepted badly, and treating those as one number hid it for days.
- Multi-threaded CPU training on that machine differed run to run by about 4e-3. Exact-reproduction checks need a pinned thread count or a measured noise floor beside the difference.
- The classical difference-of-Gaussians detector at a matched candidate budget is a genuinely strong baseline. Any learned detector has to beat it at the same number of proposals, not at an unlimited number.

## The methodology, which is the part worth copying wholesale

Both campaigns independently arrived at the same shape, and it is the reason either
produced trustworthy numbers at all.

- One registry is the only home for numbers. Prose never restates a measurement, because a stale number gets read more often than the live one. The monolith measured this: a superseded score appeared 244 times across 36 files while the live score appeared 31 times across 7.
- Declare the experiment, and its falsifier, before running it.
- Bind every artifact by content digest, and verify the digest before loading.
- Gate every commit on lint, types, tests and an artifact-integrity check.

Keep all four. They cost almost nothing and they are why the failures above are
believable.
