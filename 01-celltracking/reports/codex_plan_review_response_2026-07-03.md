# Codex response — adversarial review of the post-swarm pivot

**Date:** 2026-07-03  
**Reviewed:** `reports/codex_plan_review_2026-07-03.md`, the four `gap_*.md` reports, vendored organizer code/metric, public notebook sources, public 50-epoch artifact metadata, authenticated rules, and the live leaderboard.

## Verdict

The **pivot to the organizer learned stack is correct**. Shipping the public-weight inference path is the highest-EV immediate move because it validates the production stack and should establish a much stronger public calibration point without training.

The proposed plan nevertheless over-rotates in three ways:

1. **The 0.896 public frontier is contaminated as evidence of private quality.** All four current `test/` IDs also exist under `train/` with labels locally, and public training notebooks explicitly name them `TEST4` and exclude them. Participants can therefore tune the present public board almost exactly. That may be permitted competition-data use, but it makes the current top scores poor evidence of disjoint-embryo generalization.
2. **The plan double-counts post-processing.** The public 0.856-style notebook already invokes the learned detector/transformer with `use_ilp=1`, division cost, one-frame gap recovery, synthetic-node refinement, component pruning, and optional safe divisions/gap2. “Public weights baseline, then add motile + gaps + divisions to reach 0.89” is not an additive decomposition.
3. **Step 2 is already implemented, while step 3 proposes unnecessary integration.** `src/biotrack/metric.py` already wraps the authoritative organizer scorer including the count adjustment and division term. The organizer inference script already calls `tracksdata.solvers.ILPSolver`, which itself uses `ilpy` and falls back from Gurobi to SCIP. Adding motile separately recreates an existing layer.

## Strongest case against step 1

Ship the public weights, but treat the result as an **infrastructure/calibration submission**, not proof of the winning architecture.

Reasons for skepticism:

- The `biohub-tracking-support-pack-50ep-v1` manifest calls itself a repackaged public artifact. Its name implies 50 epochs, but the manifest does not record epoch, training set, validation split, or clean held-out score.
- It contains one `split_0` checkpoint. Unless provenance proves otherwise, scoring it on the 199 training crops is mostly in-sample and cannot establish embryo generalization.
- The artifact contains its own older `biohub_tracking` repository while HEAD vendors the renamed `tracking_cellmot` code. Load the checkpoint with the artifact's own pinned code first; port only after tensor/output parity.
- The public 0.856 claim appears to include ILP and gap post-processing, not weights alone.
- Public-score gains can be produced by tuning on the four labelled `TEST4` movies and may reverse on the final hidden embryos.

Even with those objections, no alternative beats the payoff-per-hour of an unchanged public-notebook reproduction. The correction is to narrow what that experiment is allowed to prove.

## Revised execution order

### 0. Resolve the rules gate — complete

The authenticated rules explicitly allow public external data and models when free/equally accessible or otherwise reasonable. They also state that an incompatible license on input data/pretrained models does not have to be relicensed under the winner's MIT code license.

Community weights mounted from a free public Kaggle Dataset are therefore **permitted**. Preserve the dataset URL, checksum, license/provenance, artifact code and exact environment for winner reproducibility.

This does **not** cure a third-party license violation. PAC-MAP's NonCommercial model/assets remain prize-use risk; do not use them in a prize-eligible submission without written permission. NIS3D CC-BY is clean. Zebrahub data requires a data-license check; broad pretraining is different from identifying and transferring labels into a suspected private test source.

### 1. Reproduce the public 50ep stack unchanged

- Use the support pack's own repo, config and checkpoint.
- Run an offline dependency/install smoke test, one-video inference, then all four visible test movies.
- Record runtime, peak memory, checkpoint SHA, node/edge/division counts and artifact-resolution path.
- Locally score the generated four graphs against the corresponding known labels using the authoritative evaluator. The local score and Kaggle public score should agree closely; if they do not, stop for train/serve diagnosis.
- Submit one unchanged reproduction. Do not tune from its public score.

**Go gate:** exact artifact loaded, output complete, <9 h projected hidden inference, local/Kaggle parity understood.

### 2. Use the real metric that already exists; harden it

Do not build a second evaluator. Use `biotrack.metric.score_many/score_submission`, which calls organizer `evaluate`, then `per_sample_metrics` and `summarise`.

Important trap: organizer `evaluate_datasets()` computes raw edge Jaccard plus division Jaccard and does **not** apply the per-sample node-count adjustment. It is not the exact final leaderboard gate. The existing wrapper is safer.

Hardening work:

- Add parity on several real dense crops between NumPy edge counts and the authoritative matcher, including tie/collision frames.
- Add exact division fixtures and one real division-bearing crop.
- Pin the evaluator commit independently from the model artifact. Reusing organizer code is not statistical leakage, but shared model/evaluator drift can hide implementation bugs.
- Make every run emit `edge TP/FP/FN`, adjusted edge J, count ratio, division TP/FP/FN/J, and final score.

### 3. Establish a clean learned OOF baseline

The public one-checkpoint artifact cannot answer private robustness. Obtain genuinely fold-specific public checkpoints if they exist; otherwise train short organizer fold models now. Training is needed for measurement even if the public weights remain stronger on the public board.

- Fold A: train on 6bba, evaluate all 44b6.
- Fold B: train on 44b6, evaluate all 6bba.
- Freeze inference parameters before evaluating the opposite embryo.
- Never use the first-10 alphabetical screen as a go/no-go set. Use all crops, or a fixed stratified screen spanning embryo, density, intensity, depth and time.

This clean OOF score becomes the optimization target. Public LB is only a serve check.

### 4. Tune the existing ILP and division path before integrating anything new

The vendored prediction script already exposes:

- `--use-ilp`
- `--ilp-edge-weight`
- appearance/disappearance weights
- `--ilp-division-weight`
- detection threshold and TTA

The solver is division-native and SCIP-backed. Start with factorial ablations on frozen learned detections:

1. Greedy vs existing tracksdata ILP.
2. Detection threshold/count trajectory.
3. ILP division weight `{off, 1.0, 0.8, 0.4, 0.2}`.
4. One-frame gap recovery off/on.
5. Division geometry filter off/on.
6. Only then micro-safe post-hoc divisions or gap2.

Do not integrate motile separately unless the existing ILP fails correctness, runtime or candidate-graph expressiveness.

### 5. Retrain for score only after the clean baseline decomposition

Retraining remains the ceiling lever, but choose it from measured residual errors:

- Detection no-candidate dominated: NIS3D pretraining / longer organizer training / detector ensemble.
- Association dominated: transformer/link-cost training or wider temporal windows.
- Count dominated: threshold/count regressor using image-derived features; `estimated_number_of_nodes` is unavailable at inference.
- Division dominated: division-aware sampling/loss and ILP cost calibration.

## Divisions — what to do and what not to claim

“Divisions are free money” is directionally useful but mathematically sloppy. A false fork can add a division FP and an edge FP; public learned pipelines may already emit divisions through ILP.

The two train embryos contain about 82 annotated divisions. Illustrative outcomes:

- `TP=10, FP=5, FN=72` gives division J `10/87 = 0.115`, worth **+0.0115** before edge effects.
- `TP=25, FP=10, FN=57` gives division J `25/92 = 0.272`, worth **+0.0272**.

Therefore divisions can exceed the entire classical gain, but first measure the learned baseline's existing division TP/FP/FN. The cheapest reliable route is an **existing ILP division-weight sweep plus geometry filtering**, not a new motile integration and not an ungated nearest-neighbour fork heuristic.

## What the 0.890–0.896 teams probably do

The swarm's claim that they “almost certainly” run a specific organizer+ILP+gap stack is not evidenced. Their methods are private.

The strongest explanation for the sudden public frontier is **public-test specialization**:

- the visible test consists of four movie IDs with corresponding labelled train copies;
- public notebooks explicitly isolate those IDs as `TEST4`;
- detection threshold, count, gap and division parameters can therefore be optimized directly against the public scoring movies.

Likely additional ingredients, in descending confidence:

1. 50ep learned detector/transformer or an equivalent learned detector.
2. Exact four-movie threshold/count and division tuning.
3. Existing ILP plus conservative gap and division post-processing.
4. Fold/checkpoint ensembling or detector union; flip TTA is already present in the organizer code.
5. Public-label-aware local optimization, which may win the public board but is dangerous for the private 71%.

The 0.04 gap from the shared 0.856 stack to 0.896 could come largely from the 0.1 division bucket and public-specific tuning; it does not require a new architecture. Conversely, none of this proves private superiority.

## Red-team finding disposition

| Finding | Disposition | Action |
|---|---|---|
| #1 clobbered 199-crop V3 file | **Inline now** | Regenerate atomically; separate filenames for screens vs full runs; never overwrite record-of-truth CSVs. |
| #2 stale HANDOFF table | **Inline now** | Correct after active runs finish; mark corrupted 0.727 as deployment failure, not model score. |
| #3 biased first-10/20 screen | **INTERRUPT model selection** | Replace with fixed stratified screen or full folds before any go decision. |
| #4 divisions scheduled last | **INTERRUPT after baseline reproduction** | Measure existing learned divisions; move ILP division sweep ahead of new detector research. |
| #5 op_bright marginal vs v3_smooth unknown | **Inline / classical fallback** | Finish full run; do not let it delay learned stack. |
| #6 one-sided count bonus | **Inline experiment** | Sweep undercount empirically. The bonus is only 1% at 0.9× count, so do not assume aggressive undercount wins. |
| #7 unreproduced 0.854 ceiling/V11 identity | **Drop the claim** | Preserve only measured results. |
| #8 local→LB offset n=1 | **INTERRUPT score forecasting** | Delete offset-based projections until multiple clean submissions exist. |
| #9 no nested split / same-embryo tuning | **INTERRUPT retraining/tuning** | Use embryo-held-out configs and freeze before opposite-fold evaluation; acknowledge two-embryo uncertainty. |
| #10 dense matcher tie risk | **Inline metric hardening** | Add real dense-crop parity; final gates already use authoritative scorer. |

### New finding — public TEST4 overlap

**INTERRUPT leaderboard-driven strategy.** Treat the current public leaderboard as a deployment benchmark, not a generalization benchmark. Report both `public_TEST4` and clean embryo-held-out scores on every candidate; optimize only the latter.

## DQ / prize risks

- **Public community weights:** permitted by the external-data/model clause if free and equally accessible. Record provenance and reproduce with the public artifact.
- **NIS3D:** permitted; CC-BY attribution required.
- **PAC-MAP NC assets:** avoid without written commercial/prize-use permission.
- **Zebrahub:** pretraining requires explicit data-license verification. Do not identify a private crop and transfer its public trajectory labels without written host approval.
- **Metric defects / score probing:** do not use the quarantined division evaluator defect or submission-based private-label reconstruction.

## The one change I would fight for

**Insert a clean, full-metric, embryo-held-out learned baseline gate immediately after the unchanged public-weight smoke submission—and make it the sole optimization target.**

That one change prevents three failure modes at once: chasing a four-movie leakable public board, double-counting components already present in the 0.856 stack, and postponing the only measurement that predicts the private disjoint embryos. Ship the free weights now, but do not believe them until a clean fold experiment says what transfers.
