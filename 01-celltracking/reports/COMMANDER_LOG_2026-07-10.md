# Commander log — 2026-07-10

## Control snapshot

- Kaggle account: `aryaarun07`; access-token authentication healthy.
- Use explicit CLI: `.venv/Scripts/kaggle.exe` (global PATH has no Kaggle CLI).
- Live public leaderboard downloaded at 11:29 BST: 1,014 rows, top 0.910, #10 0.901, #20 0.898, 385 ≥0.880, median 0.839.
- Our submissions: 0.807 valid anchor; 0.727 known broken-coordinate artifact.
- No Biohub kernel running.
- Complete kernels: meta-oof, train-oof-f0/f1, predict-score-f0/f1, op-bright, v3-anchor.
- Error kernels: ablate (fold-1 weights not found), Spotiflow (CUDA binary/architecture incompatibility).
- GPU quota: 16.8493h remaining before reset at 2026-07-11 00:00 UTC.

## Work started

1. Downloading all completed learned OOF predictions and fold weights locally.
2. Building Trackastra `ctc` zero-shot adapter in isolated new files.
3. Building endpoint/candidate oracle tooling and redetection diagnostic in isolated new files.
4. Rewriting `WIN_PLAN.md` around the private shuffle.

## Artifact locations

- Kaggle pulls: `artifacts/kaggle/` (local/generated; keep out of commits).
- Live leaderboard snapshot: `scratch_live_lb/` (local/generated).
- Fold weights dataset: `artifacts/kaggle/weights_dataset/`.

## Next gates

- Count and exact-score all recovered GEFFs.
- Verify output completeness; re-download with `PYTHONUTF8=1` where Windows extraction truncated.
- Run Trackastra and oracle unit tests.
- Decide first GPU launch from measured oracle/adapter readiness, not narrative confidence.

## Exact organizer learned-baseline OOF

Recovered the complete packed predictions from the two held-out-embryo kernels
(71 `44b6` crops and 128 `6bba` crops) and scored them with the authoritative
local metric:

| Held-out embryo | adj edge J | division J | node recall | score |
|---|---:|---:|---:|---:|
| `44b6` | 0.6554 | 0.0081 | 0.9536 | **0.6562** |
| `6bba` | 0.5589 | 0.0041 | 0.8966 | **0.5593** |

Division predictions are actively harmful: `44b6` has 5 TP / 594 FP / 21 FN;
`6bba` has 45 TP / 10,786 FP / 80 FN. The learned detector is valuable, but
the organizer association/division output does not generalize across embryos.
The next gate is therefore frozen-coordinate relinking and division suppression,
with redetection focused especially on the 10.34% missing `6bba` nodes.

## Measured edge headroom

The full 199-crop oracle sweep separates present edge selection from endpoint
availability:

| Held-out embryo | baseline edge J | best subset of selected edges | assignment-feasible endpoint ceiling |
|---|---:|---:|---:|
| `44b6` | 0.6561 | 0.7343 | **0.9179** |
| `6bba` | 0.5589 | 0.7112 | **0.8756** |

For `44b6`, 14,760 GT edges are supported by the current selected edge set,
18,464 are assignment-feasible with existing endpoints, and 19,826 GT edges
exist. For `6bba`, those counts are 78,339 / 96,496 / 109,057. Thus pure
edge pruning has material immediate value, while relinking existing detections
is the dominant measured lever. Redetection remains necessary to push beyond
the roughly 0.876 hard-fold endpoint ceiling.

## Division/fork suppression ablation

Keeping only the highest-`edge_prob` child for each parent removes 70,628 edges
on `44b6` and 150,568 on `6bba`. Exact OOF becomes:

| Held-out embryo | original score | fork-suppressed score | delta |
|---|---:|---:|---:|
| `44b6` | 0.6562 | **0.6595** | +0.0033 |
| `6bba` | 0.5593 | **0.5680** | +0.0087 |

This is a safe floor improvement on both folds, but it captures only a small
fraction of the oracle selection headroom. Keep suppression in the baseline;
do not confuse it with the required association solution.

## Blind dangling-track redetection gate

The first ten crops from each held-out embryo were tested with identical blind
query logic and exact post-hoc scoring. The default confidence 1.0 policy lost
score on 19/20 crops; losses reached -0.0581 on `44b6`. Confidence thresholds
1.25, 1.5, 1.75, 2.0, and 3.0 were then frozen and applied to both folds.
Threshold 1.25 remained negative overall; thresholds 1.5 and above selected
almost no proposals and produced no useful gain.

**Decision:** current raw-intensity dangling-track redetection is a no-go. The
GT-conditioned top-3 rescue result proves image signal exists, but confidence
alone cannot identify it safely. Revisit only after the linker supplies stronger
residual/cycle-consistency features; do not spend more cycles threshold-fishing.

## Organizer edge-probability sweep

With detections frozen and forks capped at one child, thresholds 0.0 through
0.5 are identical because every selected organizer edge has `edge_prob >= 0.5`.
Thresholds above 0.5 monotonically destroy performance on both folds (for
example, threshold 0.7 gives edge J 0.448 on `44b6` and 0.516 on `6bba`).

**Decision:** the subset-oracle gap cannot be captured by a global confidence
cutoff. It requires contextual association or a structured solver.

A matching displacement-distance sweep also failed as a hard rule. `44b6`
preferred no gate; `6bba` improved only +0.00056 at a 7-unit maximum, which was
slightly negative on `44b6`. Retain distance as a soft linker feature only.

## Trackastra zero-shot first result

On frozen organizer detections for `44b6_0113de3b`, Trackastra CTC zero-shot
greedy association changed only the edges:

| linker | nodes | edges | exact adj edge J |
|---|---:|---:|---:|
| organizer | 28,119 | 25,139 | 0.75769 |
| Trackastra | 28,119 | 26,945 | **0.83158** |

The gain is **+0.07390** on identical coordinates. Trackastra produced two false
divisions on a crop with no GT divisions. This is a high-EV association signal,
not yet a fold-level result. A fixed-settings 10-crop-per-fold screen was launched
immediately as kernel version 4.

Cached candidate scores permit tracker-threshold ablation without rerunning the
GPU. On this same crop, the default threshold 0.5 reproduces 0.83158 exactly;
threshold 0.8 materializes and authoritative-scores at **0.87425**. This is a
one-crop tuning result only. The threshold is not eligible for promotion until
chosen on one held-out embryo and transferred unchanged to the other.

Forcing Trackastra to one child per parent reduced edge J to 0.82869 on the
first crop, despite removing its two false divisions. Unlike the organizer
graph, Trackastra divisions must not be suppressed blindly.

## Live public-code correction

Authenticated Kaggle kernel search on 2026-07-10 shows the visible frontier has
moved beyond a uniform 50-epoch baseline. Public `lb897-baseline` descendants
expose a 400-epoch temporal model, spatial D4 detection TTA, ILP, motion relinking,
one-frame gap repair, safe divisions, and short-track filtering/recovery. A public
DoG+Trackastra notebook and offline Trackastra model/package bundle also exist.

**Implication:** Trackastra alone is not an asymmetric secret. The potentially
decisive stack is learned detections + embryo-held-out Trackastra calibration +
metric-aware fusion + legal target-embryo adaptation. Public code will be used
as deployment reference, never as evidence of private generalization.

## 16:59 UTC leaderboard refresh

Authenticated full download contains 1,030 ranked teams. #1 remains 0.910,
#10 is 0.901, #20 is 0.900, and the top-20 spread is 0.010. There are 401 teams
at or above 0.880; median is 0.842. Our account is rank 650 at 0.807 before the
running LB897 calibration kernel. Public saturation is increasing, while the
winning public score is unchanged.

## Deployment calibration and offline packaging

The current public `lb897-baseline` notebook was run unchanged as private kernel
`aryaarun07/biohub-lb897-calibration` version 1. Its output has 239,901 contiguous
unique rows (122,035 nodes, 117,866 edges), covers all four test datasets, and has
SHA256 `284DECB6C3618879E06647761187FF8D97A4DF6240AA2AFA503D69BEAAD262AA`.
Code-competition submission ref **54534923** is pending. This is deployment/public
calibration only; its dataset-specific public-movie tuning is not private evidence.

The support-pack manifest now identifies itself as the 400-epoch snapshot despite
the legacy `50ep` dataset slug. Model weight SHA256 is
`12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`.

A lean 54.8 MB Trackastra wheel archive was built and uploaded privately with
SHA256 `31317ff2c670837f096ff42773e5443d96549d48b506a3a4b2de1533b4d15911`.
An internet-off CPU smoke kernel installed the support-pack graph stack, installed
Trackastra/edt/lz4 without dependencies, loaded the CTC checkpoint, and printed
`OFFLINE TRACKASTRA SMOKE PASS`.

## Trackastra 10-crop `44b6` gate

On the first ten `44b6` OOF crops, organizer combined score is **0.6880**. Pure
Trackastra default is 0.6065; its best simple cached setting is no-divisions at
threshold 0.5 with edge J 0.6418. The one-crop 0.874 result did not generalize.

Edge-level agreement fusion is materially better: add +3.0 to the Trackastra
logit when the organizer independently proposes the same edge, then run no-div
greedy at threshold 0.5. Authoritative materialized edge J is **0.7076**, a
+0.0213 edge-J gain over organizer edge J on the slice. Fold-1 is running with
the same candidate generation; bonus/threshold/mode must transfer unchanged.

## 2026-07-11 cross-embryo promotion gate

The frozen fold-0 agreement rule transferred unchanged to the first ten held-out
`6bba` crops. Organizer combined score was **0.6035**; pure Trackastra was
**0.6116**; agreement fusion (bonus 3, threshold 0.5, no divisions) was
**0.6433**. The deployable gain is therefore **+0.0398 absolute** on the exact
combined metric, with identical detections.

A coarse grid chosen on `6bba` preferred bonus 3 / threshold 0.8 / no divisions
at 0.6451. Applied unchanged back to the first ten `44b6` crops it scored
**0.7042**, still materially above the organizer's 0.6880 baseline (although
below the 0.7076 fold-0-selected setting). The broad bilateral-positive surface
passes the generalization gate. Kaggle kernel version 6 now exports Trackastra
candidates for the complete 199-crop OOF set; no leaderboard submission will be
made until that exact full-OOF result is scored.

## 2026-07-11 public calibration and leaderboard refresh

Submission 54534923 completed at **0.889**, not the source notebook's claimed
0.897. Our team `Arya Arun` is rank **358 / 1,054** in the 07:35 UTC full
leaderboard snapshot. The board has a new **0.968** outlier at rank 1; rank 2 is
0.910 and rank 3 is 0.908. Treat the outlier as an unresolved intelligence item,
not as evidence that our private target or legal strategy should change.

## 2026-07-11 full Trackastra fusion OOF

Kaggle kernel version 6 exported Trackastra candidates for all 199 OOF crops.
The fold-0-selected configuration was frozen unchanged: organizer-agreement
logit bonus 3.0, greedy threshold 0.5, one child per parent/no divisions.

Authoritative materialized exact scores:

| Held-out embryo | organizer | fork-suppressed organizer | Trackastra agreement fusion | gain vs safe organizer |
|---|---:|---:|---:|---:|
| `44b6` (71 crops) | 0.6562 | 0.6595 | **0.6801** | **+0.0206** |
| `6bba` (128 crops) | 0.5593 | 0.5680 | **0.5809** | **+0.0129** |

Both full folds pass the predeclared association promotion gate. The earlier
10-crop improvements were directionally valid but overstated the hard-fold
effect; full-fold numbers now supersede them. Fusion is promoted to the legal
private-board stack. It currently emits no divisions, so division recovery must
be evaluated as an explicit posterior after association rather than by relaxing
Trackastra's child capacity.

The research-swarm claim that Trackastra had already captured a large portion of
the +0.18 oracle headroom was premature before this run. The corrected statement
is: Trackastra captures a reproducible +0.013 to +0.021 across embryos, leaving
substantial calibrated-association, pruning, target-adaptation and candidate
headroom.

## 2026-07-11 fusion plus incident-node pruning

The fusion materializer now has a tested `--prune-isolated` mode that exports
only detections incident to a selected edge. This stacks cleanly with Trackastra
agreement fusion on every OOF crop:

| Held-out embryo | fusion | fusion + pruning | pruning delta | gain vs safe organizer |
|---|---:|---:|---:|---:|
| `44b6` | 0.6801 | **0.6948** | +0.0147 | **+0.0353** |
| `6bba` | 0.5809 | **0.6044** | +0.0235 | **+0.0364** |

Node recall falls only from 0.9536 to 0.9459 on `44b6` and 0.8966 to 0.8933
on `6bba`; removing assignment-stealing isolated detections and improving the
count multiplier more than compensates. This establishes a balanced legal stack
before target adaptation, Dinkelbach calibration or legitimate divisions.

Deployment caveat: the public 0.889 wrapper already prunes isolated output nodes.
Therefore the entire OOF pruning delta cannot be added to 0.889 as a public-score
forecast. Trackastra association fusion is the new deployment increment; pruning
is a required invariant that preserves its measured benefit.
