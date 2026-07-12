# Three-layer victory architecture

**Date:** 2026-07-12  
**Objective:** maximize private-set accuracy, precision and prize-eligible score by
treating data, algorithms and inference/compute as three compounding systems.

## Executive thesis

The field largely shares one pipeline:

```text
framewise detector -> discrete points -> pairwise linker -> graph repairs
```

That pipeline destroys weak temporal evidence before association, learns from
two sparsely labelled embryos, and optimizes generic tracking likelihood rather
than the competition metric. Incremental tuning can remain as a hedge, but the
main architectural bet should change the unit of prediction from a cell in one
frame to a trajectory supported across time.

The proposed system is a **Lagrangian Lineage Field**:

```text
4D image window
    -> continuous nuclear-evidence and tissue-motion fields
    -> motion-compensated trajectory tubes
    -> jointly decoded nodes, edges and divisions
    -> metric-aware component selection
```

Winning requires all three layers to reinforce one another:

1. **Data layer:** manufacture transferable supervision and calibrated domain
   variation from public dense tracks, sparse competition labels and unlabeled
   target movies.
2. **Algorithm layer:** integrate evidence through time before detection, model
   tissue motion, predict branching lineages and optimize expected metric value.
3. **Compute and inference layer:** allocate GPU/CPU work by information value,
   stream large volumes efficiently, ensemble only complementary models and adapt
   safely to each hidden embryo.

## Current evidence and constraints

- Public anchor: **0.889**.
- Trackastra used only as a hint: **0.889**, neutral.
- Direct Trackastra association: **0.865** public, despite local dummy score
  0.9065. It is rejected as a wholesale deployment replacement.
- Full embryo-held-out OOF fusion plus incident-node pruning improves the safe
  organizer by **+0.0353 / +0.0364**, proving complementary association and
  pruning signal exists but also proving two-embryo OOF is not sufficient for
  hidden transfer.
- Public Biohub March-22 dense trajectories align with competition-source crops
  at 96-98% sparse-GT node recall. They are valuable same-source training data.
- Simple geometric division proposals produced 0 true positives from 7 proposals
  in the initial bilateral screen.
- The score rewards correct edges, legitimate undercount and divisions. Recall
  alone is not the objective.
- Exact hidden-source trajectory transfer and evaluator defects remain
  quarantined; the system must remain prize-eligible.

# Layer 1: data and supervision

## 1.1 Build a unified trajectory lake

Create one physical-coordinate schema for:

- competition crops and sparse GT;
- public March-22 images and dense Ultrack trajectories;
- ZSNS001-005/Zebrahub images, tracks and divisions;
- DAXI weights and intermediate detector responses;
- NIS3D and Cell Tracking Challenge nuclear datasets;
- synthetic corrupted trajectories;
- model candidate edges and uncertainty;
- per-frame acquisition-regime metadata.

Canonical records:

```text
movie_id, source, license, embryo, t, z_um, y_um, x_um,
track_id, parent_track_id, label_confidence, image_support,
motion, acceleration, density, division, corruption_type
```

Every artifact requires URL, license, checksum, transformation history and
coordinate convention. Data provenance is part of the model, not paperwork.

## 1.2 Turn dense tracks into domain-randomized supervision

Dense public trajectories should not be copied into test predictions. They should
teach transferable physics and lineage structure.

Generate training variants by:

- deleting 5-40% of nodes in structured and random patterns;
- adding intensity-correlated and random false detections;
- perturbing coordinates with separate axial/lateral noise;
- varying PSF, blur, shot noise and photobleaching;
- changing temporal cadence and inserting frozen frames;
- adding sudden global translations and local non-rigid deformation;
- varying density, crop boundaries and developmental speed;
- hiding most labels to reproduce positive-unlabelled supervision;
- corrupting divisions and candidate graphs.

The model must learn to recover the clean trajectory and calibrated uncertainty,
not memorize embryo appearance.

## 1.3 Treat sparse labels correctly

Competition annotations are positives, not exhaustive negatives.

Loss stack:

1. positive heatmap/point localization loss around annotations;
2. masked background loss only in trusted negative regions;
3. non-negative positive-unlabelled risk as an ablation;
4. teacher-student pseudo-trajectories gated by temporal persistence;
5. forward/backward and skipped-frame consistency on all unlabeled images;
6. confidence weighting from annotation/public-track provenance.

## 1.4 Build a regime and difficulty index

For every frame pair/crop compute:

- frozen-frame indicator;
- global translation/rotation estimate;
- local strain and divergence;
- density and depth;
- intensity/photobleaching statistics;
- detector count and confidence distribution;
- association entropy;
- missing-endpoint and division likelihood;
- predicted domain distance from training embryos.

This enables regime-specific inference and prevents average metrics from hiding
where an architecture fails.

## Layer-1 gates

- Public trajectory alignment median error below 3 um.
- Synthetic corruption reproduces measured OOF FN/FP/count distributions.
- A model trained on public tracks improves both competition embryos when all
  labels from the evaluated embryo are hidden.
- No artifact without a documented prize-compatible license enters production.

# Layer 2: algorithms and decision systems

## 2.1 Main bet: Lagrangian Lineage Field

Predict continuous fields from 7-11-frame anisotropic 3D windows:

| Head | Output | Purpose |
|---|---|---|
| nuclear evidence | 1 channel | subthreshold centre likelihood |
| forward motion | 3 channels | physical displacement to `t+1` |
| backward motion | 3 channels | cycle and occlusion consistency |
| existence | 1 channel | probability a trajectory persists |
| division | 1 channel | local branching hazard |
| daughter vectors | 6 channels | two daughter displacements |
| uncertainty | 1-4 channels | localization/existence calibration |

Use an anisotropic 3D encoder per frame, multi-scale temporal attention and
deformable warping. Avoid full-volume dense 4D convolution.

Decoder:

1. estimate global motion;
2. estimate local tissue flow;
3. warp evidence into material coordinates;
4. integrate weak evidence along candidate trajectory tubes;
5. declare nodes only after temporal integration;
6. decode ordinary paths and supported bifurcations;
7. send only ambiguous regions to a structured solver;
8. retain components by expected metric contribution.

## 2.2 Hedge A: baseline-preserving selective repair

The 0.889 graph remains the production hedge. Never replace it wholesale again.

For every uncertain baseline edge, generate alternatives from Trackastra, flow,
optimal transport and three-frame motion. Replace only when:

- baseline confidence is low;
- challenger margin is calibrated out of embryo;
- forward/backward paths agree;
- local tissue-flow residual improves;
- neighbor motion becomes more coherent;
- no collision, parent or lineage constraint is violated.

This hedge is cheaper and should reach production before the full lineage field.

## 2.3 Hedge B: track-before-detect without a new backbone

Use existing raw detector responses:

- predict paths from reliable track endpoints;
- sample low-threshold evidence along 3-7-frame tubes;
- compensate for local tissue motion;
- compare against matched background/null paths;
- promote missing nodes only with both temporal and image support.

This tests the central architectural thesis before expensive training.

## 2.4 Hedge C: population geometry and optimal transport

For ambiguous dense regions, match local neighborhoods rather than individual
appearance:

- unbalanced/partial optimal transport for births and deaths;
- fused Gromov-Wasserstein for kNN geometry;
- local coherent point drift or affine tissue deformation;
- dustbin assignments and multiple short hypotheses.

Apply only where baseline association entropy is high; do not run embryo-wide.

## 2.5 Division as a dedicated posterior

Predict divisions after a strong ordinary association graph.

Candidate triplet features:

- parent contraction, intensity and track age;
- daughter displacement symmetry and sister separation;
- combined daughter versus parent intensity/mass;
- Trackastra/organizer/field agreement;
- local divergence and neighbor motion;
- density, depth and developmental time;
- image patches before and after the event.

Accept a second daughter only from a calibrated posterior. Never lower the global
division cost as a substitute for prediction.

## 2.6 Metric-aware component and edge selection

Calibrate three outcomes separately for every edge/component:

```text
pTP, p(metric-counted FP), p(ignored/unannotated)
```

Then optimize expected adjusted edge-J with lineage constraints and a count
budget. Use Dinkelbach-style fractional optimization as a solver layer, with
lambda selected out of embryo.

Component confidence includes:

- length/persistence;
- average image support;
- calibrated edge probability;
- model agreement;
- motion smoothness;
- birth/death plausibility;
- density and boundary context;
- target-domain uncertainty.

## Layer-2 gates

- No promotion from one crop or one embryo.
- Freeze configuration on one embryo; transfer unchanged; reverse direction.
- Minimum go gate: +0.005 exact score on both embryos.
- Major architecture gate: +0.010 on both embryos plus stress-test robustness.
- A public submission cannot rescue a model that fails cross-embryo gates.

# Layer 3: compute, inference and deployment

## 3.1 Separate compute by purpose

| Resource | Work |
|---|---|
| local CPU | metric, pruning, calibration, candidate analysis, small models |
| local MX350 | smoke tests and tiny inference only |
| Kaggle T4 | full learned inference, Trackastra and falsification screens |
| Kaggle T4x2/cloud A100 | gated training only |
| storage datasets | immutable weights, wheels, cached candidates, provenance |

GPU jobs must not perform tasks that can be cached or computed locally.

## 3.2 Cache the expensive boundaries

Version and cache:

- raw detector response volumes;
- detections at several recall thresholds;
- Trackastra/organizer/flow candidate edges;
- image and motion features;
- public trajectory training shards;
- calibrated probabilities;
- final graph components.

This turns most architecture experiments into CPU-only reselection instead of
re-running 3D models.

## 3.3 Cascade inference by uncertainty

```text
fast baseline
    -> confidence/regime classification
        -> easy regions: retain baseline
        -> uncertain links: selective ensemble
        -> weak/missing endpoints: track-before-detect
        -> division candidates: fork posterior
        -> severe global jumps: registration + re-association
```

Heavy models should process only ambiguous windows. This improves both runtime
and precision by reducing unnecessary intervention.

## 3.4 Test-time target calibration

For each hidden embryo estimate without labels:

- intensity and PSF statistics;
- count/density trajectory;
- global and local motion distributions;
- frame freezes/time jumps;
- uncertainty and model disagreement;
- division-rate prior only as a soft statistic.

Adapt normalization, motion calibration and small association layers through
path/cycle consistency. Keep the source model frozen and preserve a fallback.

## 3.5 Ensemble for complementary error, not model count

Measure pairwise error overlap. A model enters the ensemble only if it recovers
errors the baseline misses at acceptable marginal precision. The final ensemble
should contain different mechanisms:

- learned temporal graph baseline;
- lineage-field or track-before-detect evidence;
- Trackastra structural association;
- population-flow/OT in ambiguous regions;
- dedicated division posterior.

## 3.6 Submission and reproducibility gate

Before Kaggle:

- exact two-embryo OOF and stress tests;
- runtime projection below 9 hours;
- internet-off dependency smoke;
- output schema and graph integrity;
- immutable commit/config/checkpoint hashes;
- artifact license/provenance manifest;
- predicted reason the submission should differ from the current anchor.

# Interdisciplinary bet portfolio

Do not bet everything on one architecture. Allocate compute like a portfolio.

| Bet | Origin | EV | Cost | Initial allocation | Kill gate |
|---|---|---:|---:|---:|---|
| motion-compensated track-before-detect | radar/astronomy | very high | medium | 20% | recover >=20% missed endpoints at >=70% marginal precision |
| Lagrangian Lineage Field | continuum mechanics/video | very high | high | 25% | response-integration prototype +0.005 both folds |
| selective baseline repair | MOT/SLAM | high | low | 15% | +0.005 both folds, no stress regression |
| public dense-track edge/fork posterior | developmental biology/GNN | high | medium | 15% | +0.010 both folds or +0.015 min-fold |
| component/count optimization | decision theory/OR | high | low | 10% | transferable component-confidence curve |
| forward/backward target adaptation | self-supervised MOT | medium-high | medium | 7% | zero-gradient consistency gate improves both folds |
| local unbalanced OT/FGW | geometry/transport | medium | medium | 3% | +0.003 min-fold on ambiguous subset without global loss |
| dedicated division posterior | lineage biology | high ceiling | medium | 5% | positive combined delta both folds; target division-J >=0.25 |

Reallocate weekly toward demonstrated cross-embryo marginal gain. Kill weak bets
quickly; retain negative results in the journal.

# Efficiency and accuracy scorecard

Each experiment reports:

| Axis | Metric |
|---|---|
| accuracy | raw and adjusted edge-J |
| precision | edge TP / metric-FP and component marginal precision |
| recall | node, endpoint and candidate-edge recall |
| lineage | division TP/FP/FN/J |
| generalization | per-embryo score and min-fold |
| robustness | freeze/jump/density/depth/intensity slices |
| compute | GPU-hours, CPU-hours, peak RAM/VRAM |
| efficiency | score delta per GPU-hour and per engineering day |
| complementarity | errors recovered beyond the anchor |
| eligibility | license/rules/provenance status |

# Execution roadmap

## Phase 0: 24-48 hour falsification

1. Run the bracketed-miss analysis.
2. Implement response-volume track-before-detect on the hardest crops.
3. Run baseline-preserving edge-replacement calibration.
4. Build component-confidence/pruning features and cross-fold curves.
5. Audit and shard March-22 dense trajectories for association training.

Go/no-go decides whether to train the lineage field.

## Phase 1: one-week prototypes

1. Train dense-track edge/existence/fork posterior.
2. Build a small motion/evidence-field model on public trajectories.
3. Add acquisition-regime detection and global-jump correction.
4. Run forward/backward consistency gating.
5. Establish cached multi-model candidate tables.

## Phase 2: two-to-four-week architecture

1. Train the complete Lagrangian Lineage Field.
2. Add motion-compensated temporal evidence decoding.
3. Train the division and component posteriors.
4. Add target-embryo calibration.
5. Optimize expected metric value through the lineage solver.
6. Ensemble only proven complementary mechanisms.

# The decisive first experiment

The cheapest test of whether the field thinks about the problem incorrectly is:

> Can motion-compensated temporal integration recover genuinely missed endpoints
> that no single-frame detector sees?

Use raw pre-NMS response volumes and reliable tracks. Warp 3-7 neighboring
frames into track coordinates, integrate evidence, compare against matched null
paths and freeze every parameter across embryos.

Proceed to the full lineage field only if this yields:

- >=20% recovery of missing endpoints at >=70% marginal precision; or
- >=+0.005 exact score on both embryos.

# Final commander bet

The strongest new architecture is a **motion-compensated 4D track-before-detect
lineage field**, pretrained on dense same-source trajectories and decoded through
a metric-aware component solver.

The hedge is the 0.889 baseline with selective, uncertainty-gated repairs. The
portfolio structure ensures that a failed radical architecture does not stop
incremental score progress, while every data and inference improvement remains
useful to both paths.
