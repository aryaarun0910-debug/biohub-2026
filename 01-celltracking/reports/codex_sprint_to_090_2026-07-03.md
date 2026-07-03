# Codex sprint brief — path to 0.90 on PRIVATE by end of week

**Target:** 0.90 private by ~2026-07-08 (5 days). Current public leader 0.875; public plateau
0.842. This is aggressive: our measured DoG recall ceiling is ~0.90, so classical tuning alone
CANNOT reach 0.90 — the sprint almost certainly requires TRAINING a learned detector. Plan for that.

## Measured state (evidence in repo, private: github.com/aryaarun0910-debug/Biohub-CellTracking-2026)
- **V3 anchor** reproduced + submitted (score PENDING). Local V3 adjJ 0.632 (44b6) / 0.756 (6bba),
  min-fold 0.632. Exact numpy metric validated == organizer (edge term; divisions via tracksdata).
- **Bottleneck is DETECTION RECALL**, measured over all 199 crops: 77-80% of missed edges are
  no-candidate (no detection within 7um), lost-assignment ~0-2%, association 18-23%. Recall 0.82/0.88.
- **Over-proposal frontier (step 1a):** NMS radius (not threshold) is the recall lever; loosening
  NMS 3.2->1.0um lifts hard-crop recall 0.65->0.78 (ratio 1.33), but **DoG recall plateaus ~0.78-0.90
  < 0.95**. Finer scales hurt. => classical over-proposal is a PARTIAL lever.
- Count ratios <1 (we under-detect). Same-cell conflict sets (complete-linkage + diameter cap,
  chain-bridging proven impossible) + arbitration scaffolding built. Ablation harness scores any
  config across BOTH embryo folds on the exact metric (min-fold gate).
- Infra: 199 crops local, embryo-held-out CV, tests + CI, candidate caching, Kaggle kernel pipeline
  (note: Kaggle image has tensorstore, NOT zarr). Local GPU = MX350 (2GB, cannot train). Training
  must run on Kaggle (T4x2, 12h, no internet) or cloud.

## The question for Codex
Given 5 days and the above, design the HIGHEST-EV path to 0.90 on PRIVATE. Be concrete and honest.

1. **Score decomposition to 0.90.** From min-fold ~0.63-0.76, what stack of gains realistically sums
   to 0.90, and which is the single biggest lever? (recall via learned detector? arbitration? count
   calibration? tissue-flow linking? divisions?) Give a numeric budget.
2. **Training NOW - which model, which recipe, what compute?** Pull learned detection forward:
   - Spotiflow-3D (residual detector, Linajea soft-mask) vs the host temporal-U-Net+transformer vs a
     DAXI-fine-tune vs an ensemble. Which trains fastest to a real recall gain on the held-out embryo?
   - Exact training recipe for 5 days: data (2 embryos + public ZSNS/DAXI pretraining?), loss
     (mask scheme), augmentation, epochs, and how to fit Kaggle 12h / whether we need cloud A100.
   - How to AVOID the known failure (a small 3D U-Net "failed to generalize from 2 embryos") -
     pretraining? heavy aug? residual-only over DoG?
3. **Parallelization plan.** What runs concurrently this week: arbitration/linking (CPU, local) vs
   detector training (GPU, cloud/Kaggle) vs count calibration? Sequence so nothing blocks.
4. **Realism check.** Is 0.90 in 5 days achievable, or is the honest ceiling ~0.86-0.88? If 0.90 is
   a stretch, what's the max-EV target and the exact experiments that would prove/disprove 0.90 fast.
5. **Private-set robustness.** With only 2 training embryos, how do we push recall hard WITHOUT
   overfitting them (the whole game is the disjoint hidden embryo)? Concrete safeguards.

## Deliverable
A 5-day hour-blocked sprint plan: each block = task, owner (CPU-local vs GPU-train), expected
min-fold delta, go/no-go gate on the exact metric, and the compute needed. Lead with the single
experiment most likely to break past the DoG recall ceiling. End with the honest probability of
hitting 0.90 and the fallback target.
