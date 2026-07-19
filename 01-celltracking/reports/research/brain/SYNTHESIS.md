# Brain synthesis — evidence-updated strategy

**SUPERSEDED 2026-07-14:** the plan below was executed and its winning-scale bets (division
posterior, isolated detection, breadth reranking) were falsified. See
[FINAL_SYNTHESIS_2026-07-14.md](FINAL_SYNTHESIS_2026-07-14.md) for the current conclusion.
Kept for historical record of the reasoning that led there.

**Date:** 2026-07-12

This synthesis cross-reads the six research lanes and the vendored scorer. Lane
reports are research inputs; this file is the corrected operating interpretation.

## Conclusions that survive review

1. **Measure the correct baseline first.** Trackastra improved two weak OOF
   organizer graphs but lost when it replaced the mature hidden deployment graph.
   Every new method must be evaluated against fold-specific OOF reproduction of
   the complete 0.889 wrapper.
2. **Association and component discipline are the highest-ROI near-term fronts.**
   Detection still has endpoint headroom, but the from-scratch 4D lineage field is
   too expensive before cheaper association/count hypotheses are exhausted.
3. **Baseline-preserving selective repair is safer than wholesale replacement.**
   It is not regression-proof: added nodes and edges can steal assignments, create
   counted FPs and change the count multiplier.
4. **Hidden `N_est` is unavailable.** Use known counts only for OOF analysis. A
   deployable policy must be scale-free or infer count from images/graph features.
5. **The 0.968 mechanism is unknown.** Public overfit, aggressive pruning/count
   effects, a genuine edge/division method and the quarantined evaluator defect
   remain hypotheses.
6. **Limit concurrent confirmatory bets.** Two embryos provide weak independence;
   pre-register at most two promotion experiments per round and preserve negative
   results.

## Corrected gate doctrine

A candidate clears all of:

1. Full 0.889-wrapper fold-specific OOF baseline, not organizer-greedy OOF.
2. Positive delta on both embryo families and min-fold at least +0.005.
3. No material regression across freeze/jump, density, depth, intensity and
   developmental-time slices.
4. Exact metric scoring including count and divisions.
5. Deployment features available on hidden test images—no hidden GT metadata.
6. Runtime, internet-off, provenance and graph-integrity gates.

Public score is a deployment check, not a rescue mechanism.

## Revised portfolio

| Priority | Bet | Verdict |
|---|---|---|
| 0 | Reproduce complete 0.889 wrapper on fold-specific OOF | **DO NOW** |
| 1 | Scale-free component/edge selection; global Dinkelbach lambda as an ablation | **DO AFTER E0** |
| 2 | Gradient-free forward/backward consistency features | **DO AFTER E0** |
| 3 | Selective baseline edge repair using Trackastra/flow/OT candidates | **DO NEXT** |
| 4 | Dense public-track edge/existence/fork posterior | **GATED, HIGH UPSIDE** |
| 5 | Legitimate division posterior after the edge graph is frozen | **GATED** |
| 6 | Multi-threshold detection into structured selection | **GATED** |
| 7 | From-scratch 4D lineage field | **DEFER/KILL FOR CURRENT ROUND** |
| 8 | Standalone embryo-wide OT/FGW | **KILL AS STANDALONE** |
| 9 | Learned test-time weight updates | **BLOCKED pending written clearance** |

Track-before-detect remains a small endpoint-recovery falsification experiment,
not the main architecture build.

## E0 — establish the actual production baseline

The deployed 0.889 wrapper includes learned detections/edges, motion relinking,
gap and safe-division logic, isolated/short-component pruning and smoothing.

E0 must:

1. Use fold-0 raw predictions for held-out `44b6` and fold-1 raw predictions for
   held-out `6bba`.
2. Apply the complete wrapper post-processing with one global dataset-agnostic
   configuration. The visible-test-specific min-track exception is excluded.
3. Export one GEFF per crop plus `run_stats.csv`.
4. Score exact per-crop edge TP/FP/FN, adjusted J, count ratio, node recall and
   division TP/FP/FN/J.
5. Join regime metadata and publish embryo plus slice summaries.

Applying the deployed split-0 checkpoint to every crop would be in-sample and is
not valid OOF. Merely rescoring the four deployed graphs is not E0.

## E1 — scale-free component and metric selection

On frozen E0 graphs:

- compute component persistence, image support, motion consistency, edge
  confidence, density/boundary context and model agreement;
- calibrate TP/counted-FP/ignored outcomes out of embryo;
- test global component-confidence and short-track policies;
- test one global Dinkelbach lambda as an edge/component selection ablation;
- use `N_est` only to understand OOF score effects, never as model input or a
  hidden deployment constraint.

Gate: +0.005 min-fold, both embryos positive, no major regime regression.

## E2 — forward/backward consistency

Compute cycle/path consistency on existing candidate edges without weight updates.
Use it as a calibrated feature for selective repair or rejection—not as an
automatic guarantee. Freeze on one embryo, transfer to the other and reverse.

Gate: identical to E1.

E1 and E2 are the only confirmatory bets in the first round.

## Deferred research value

The long-form architecture and lane reports remain useful for later rounds:

- dense public trajectories are the strongest source of training breadth;
- PU/masked losses are mandatory if the detector is retrained;
- target motion/path consistency is promising if rules permit adaptation;
- dedicated division prediction has up to +0.1 mathematical contribution but
  extreme calibration risk with 26 versus 125 GT divisions.

## Immediate action

Implement E0 as a reproducible post-processing module extracted from the deployed
kernel, apply it to both canonical OOF directories, and publish the authoritative
production-wrapper OOF table before tuning E1 or E2.
