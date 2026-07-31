# Research synthesis

**Updated:** 2026-07-31
**Store:** `C:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH\research.sqlite` (66 findings,
outside the Git worktree). Raw downloads, snapshots and agent notes stay there, never here.
**Grades:** A = recovered and independently reproduced; B = source/method available with a
checkable result; C = self-report without reproducible artifact; D = hypothesis only.
Current: 13×A, 42×B, 11×C.

This file is updated only when evidence changes a decision, opens a testable branch, or closes
one. It is not a log.

## 1. The decision this cycle changed

The division track's headline number is intact as a ceiling but its *deployable* claim is much
weaker than the H0c report implied, and the reason is now precise.

**The binding unknown is mother-level abstention, not pair ranking.** Three independent sources
converge:

- **Our own red-team measurement.** H0b's "median rank 0" is a *within-mother* statistic. 81,966
  (44b6) and 83,703 (6bba) mothers carry candidates and every one has a rank-0 pair. Swept as a
  global threshold over all 14,371,002 rows the frozen ranker yields `+0.0011 / -0.0010` —
  under the bar and negative on one family, FP:TP 930:1 / 979:1 at 50% recall.
- **Every surveyed tracker** (btrack, Trackastra, OrganoidTracker, laptrack) frames division as
  competition against an explicit terminate / no-split / null alternative, not as ranking.
- **Our own census**: the fork layer carries 0 and 2 true divisions out of 11.4k / 9.0k.

Break-even is now quantified: minimum precision-among-metric-visible for `+0.005` is
**4.07% / 6.38%**; the frozen geometric ranker delivers **0.107% / 0.101%** — short by 38× / 63×.
Closing that gap is the whole problem.

## 2. Convergent evidence worth acting on

**Daughter anti-parallelism (3 independent sources).** Our G smoke test measured
`cos(daughter angle)` = **−0.583** for positives vs **+0.020** for reliable negatives. btrack
implements exactly this as its division prior (`hypothesis.cc::P_branch:658`, MIT). The mitosis
literature gives it as daughter-axis/principal-axis alignment. It is dimensionless.

**Dimensionality may explain the family boundary.** Our frozen ranking feature (flow-midpoint
residual) is in absolute µm and therefore scales with tissue velocity and crop scale;
`cos θ` does not. Eight levers have died crossing the family boundary. Prefer dimensionless
features — testable with one float column and zero scorer passes.

**Persistence-confirmed divisions (independent rediscovery).** A public notebook adds a second
daughter only if both branches survive to `t+2` *and* the daughters' midpoint sits where
constant velocity predicts the parent should be. That is nearly our frozen H0c proposer
(flow-midpoint residual + persistence), arrived at independently.

**The division term is unharvested by the entire public field.** The strongest *clean* public
submission emits 276 divisions on 116,663 nodes (0.24%). The 0.913–0.916 public plateau is an
edge-Jaccard plateau. This matches our measured `division_jaccard` of 0.0000 / 0.0057 and is the
best available argument that the gap to 0.942 lives in divisions.

## 3. Closed or downgraded

- **Reverse-time association → D-grade hypothesis.** No reference implementation of literal
  forward/backward cycle-consistency exists in any surveyed repo; Trackastra's causal norm is
  per-direction normalisation, not a reciprocity test. The only support is one public
  self-report. Exact V19 code is recovered and source-locked, so the *feature* remains cheap to
  test — but it is not an established mechanism and the R arm should not be priced as one.
- **PU-learning machinery → not needed.** We hold 253,350 reliably-labelled negatives; on the
  metric-visible subset this is a rare-event *supervised* problem. PU-aware *reporting* (firing
  rate on unlabeled mothers) is still required.
- **Ultrack wholesale → closed.** It changes segmentation hypotheses and destroys the
  node-population invariance every H0c gate depends on. Branch A already falsified the mechanism.
- **cell-tracker-gnn → licence-blocked** (CC BY-NC). Recorded as `falsified` so it is not
  rediscovered.

## 4. Hard constraints now quantified

| constraint | value | consequence |
|---|---|---|
| labelled positives | 16 (44b6) / 76 (6bba) | at EPV≥10, the 44b6 direction supports **1.6 parameters** |
| division quantum | `0.1/26 = 0.00385` on 44b6 | the `+0.005` bar is ~1.3 quanta |
| SD of composite delta | ±0.0086 on 44b6 | **1.72× the bar** |
| P(spurious bilateral pass) | 12.7% at 5 trials, 24.3% at 10 | bilateral claims are underpowered by default |
| annotation density | 0.00985 vs 0.08978 (**9.11×**) | the unlabeled-FP subsidy is a density artifact, not a metric property |
| parent stealing | 97.25% / 96.90% of shortlist rows | untested exposure to branch A's falsified failure mode |

## 5. Corrections to previously reported results

- **Node invariance was circular.** `phaseb_h0c_replay.py:118` reuses `node_rows` for both arms,
  so `N_pred` could not differ; the assertion tested nothing. The live pipeline is *not*
  invariant — `wrapper.py:858` keeps short components only when they contain a division, so
  suppression deletes them (14,378 / 12,780 nodes; count-multiplier shift +0.00055 / +0.00061,
  a lower bound). H0c remains valid as a ceiling for edge-only edits on a **fixed** node set.
- **`suppress_all` free-gain was the GT-informed arm.** GT-free control is `-0.00004 / -0.00204`.
  Suppression alone contributes no free gain.

## 6. Safety: quarantined public notebooks

Many of the highest-voted public notebooks inject a hub node at `t = -1000` with coordinates
`(-10000, -10000, -10000)` plus synthetic fork chains at negative time. At least ten kernels,
including several with 60–137 votes, and one with `BIOHUB_AUGMENT_HUB` defaulting to **ON**.
"Reproduce the top public notebook" walks directly into the exploit. Full list in the store.
These may never enter a submission.

## 7. Open, ranked by (expected benefit × transfer plausibility) ÷ cost

1. **Mother-level abstention gate** — the load-bearing unknown. Softmax over the mother's top-3
   plus an explicit abstain term (Trackastra `quiet_softmax`, BSD-3; laptrack self-calibrating
   percentile, BSD-3). Must be judged on exact composite, not ranking.
2. **D-12 steal-exclusion re-run** — 97% of shortlist rows need parent stealing; re-run H0c
   without them. If retention collapses from 80.0%/81.7%, the track inherits branch A's
   falsified failure mode. ~1 h CPU, decisive.
3. **Dimensionless-feature transfer test** — one float column, zero scorer passes.
4. **Portability audit onto v122 nodes** — nothing has been measured on that population, and
   v122's mechanism *is* node-population change. Reachability is a pure function of the node set.
5. **Image battery** (mass ratio, saddle depth, condensation jump, µm-scaled second moments) —
   requires a `.zarr` pass; graphs carry only `t,x,y,z`. Defer until 1–2 resolve.
