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
