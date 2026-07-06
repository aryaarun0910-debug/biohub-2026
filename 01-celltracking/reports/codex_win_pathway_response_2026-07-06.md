# Codex red-team response: the credible pathway to win

**Date:** 2026-07-06  
**Scope:** adversarial review of `codex_win_pathway_2026-07-06.md`, repo/code audit, and parallel cross-domain research sweep.  
**Decision standard:** private-board win probability, not public-board cosmetics.

## Executive verdict

The `0.94 = 0.88 edge + 0.1 × 0.60 division` arithmetic is numerically correct and strategically misleading. It treats two extreme subproblems as if they were ordinary engineering milestones:

- At 95% node recall, independent endpoint availability is only `0.95² = 0.9025`. Edge-J `0.88` would then require approximately 97.2% precision on predicted judged edges and recovery of almost every edge whose endpoints exist.
- Division-J `0.60` requires, for example, about 80% recall at 70.6% precision, across only 151 highly imbalanced annotated divisions (26 in `44b6`, 125 in `6bba`).

`0.94` is therefore a moonshot, not the planning base case. On present evidence:

- **Central private range:** `0.84–0.89`.
- **Credible stretch:** `0.90–0.92`, conditional on a high candidate-edge oracle and successful target-domain association adaptation.
- **0.94 probability:** below 5% until the oracle proves substantial headroom.

The proposed pathway is also aimed partly at the wrong component. NIS3D is static detection data. Detection is already reported at 90–95% held-out recall; NIS3D cannot plausibly contribute the proposed `+0.08` edge gain. The data investment should move from **detector pretraining** to **association/motion pretraining and target-embryo adaptation**.

### The single highest-EV permitted action

Run a **lineage-constrained candidate-edge oracle**, then replace the generic ILP objective with a **calibrated expected-Jaccard fractional objective**. This tells us whether `0.88` is reachable before months of training and, if it is, makes the solver optimize the competition metric rather than generic tracking likelihood.

### The strongest fast external-model experiment

Run the official **Trackastra `ctc` 2D/3D pretrained association checkpoint** on frozen detections, then pass its edge scores into the existing SCIP ILP. Its checkpoint was trained on all available 2D+3D Cell Tracking Challenge GT and imperfect-segmentation data, and it is the successor to the ISBI 2024 generalizable-linking winner. This attacks the measured bottleneck directly and is falsifiable in days, not months. [Official Trackastra repository and model documentation](https://github.com/weigertlab/trackastra), [checkpoint manifest](https://github.com/weigertlab/trackastra/blob/main/trackastra/model/pretrained.json), [Trackastra paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf).

### The one potentially decisive wildcard

**Target-embryo self-supervised association adaptation:** adapt only a small association residual/calibration head on the unlabeled hidden embryo using high-confidence track consensus, skipped-frame path consistency, and forward/backward cycle consistency. This directly attacks the actual failure mode—domain shift to a disjoint embryo—without needing its labels. It must be cleared against the exact rules wording for transductive/test-time optimization.

---

## 1. Corrections to the current evidence base

These do not invalidate the work; they prevent us from promoting preliminary evidence into a false ceiling.

1. **The ILP `~0.67–0.70` result is not yet a banked, full-199, both-fold result.** `WIN_PLAN.md` itself calls it local and not yet full-scale. Full-fold exact scoring is Priority 0.
2. **The 45-epoch overfit result is three held-out crops.** It is a serious warning, not proof that all longer or externally pretrained training is negative.
3. **Detection is 90–95%, not uniformly 95%.** The edge ceiling varies steeply across that range.
4. **There are 151 annotated divisions, not ~82:** 26 in `44b6`, 125 in `6bba`. The fivefold domain imbalance makes a single global division threshold unsafe.
5. **Two embryos do not make OOF “the same ruler” as private.** The crops are correlated samples from two acquisition domains. Leave-one-embryo-out is the best available gate, but it cannot estimate third-embryo uncertainty tightly.

The single biggest reason the pathway may fail is **domain-selection overfit disguised as OOF progress**: a method can improve both known embryos by exploiting properties shared by those acquisitions and still fail on a third embryo's stage, density, motion field, mounting, and signal distribution.

---

## 2. The edge-J `0.70 → 0.88` red team

For `G` GT edges, organizer scoring gives:

```text
FN = G - TP
J = TP / (TP + FP + FN) = TP / (G + FP)
```

This identity is visible in `vendor/kaggle-cell-tracking/src/tracking_cellmot/metrics.py` and matters: false positives only enlarge the denominator, while every lost true edge directly reduces the numerator.

### What 0.88 requires

Assume node recall `r=0.95` and approximately independent endpoint misses:

```text
maximum matchable edge recall q ≈ r² = 0.9025
required FP / G at J=0.88 = q / 0.88 - 1 = 0.0256
required judged-edge precision ≈ 0.9025 / (0.9025 + 0.0256) = 97.2%
```

If every available node emits one edge, a simple wrong-link model implies association correctness must rise from roughly 86.8% at `J=0.70` to 98.7% at `J=0.88`. “Tune ILP and add gaps” does not normally remove ~90% of the remaining recoverable error.

The endpoint-independence approximation is not a theorem. Persistent missed tracks can make edge availability closer to `r`, while flickering detection can make it worse. That is exactly why the candidate oracle is non-negotiable.

### Priority 0 experiment: three oracle ceilings

On every crop and both embryo folds, calculate:

1. **Endpoint oracle:** fraction of GT edges for which both endpoints can be matched to current detections.
2. **Candidate oracle:** fraction recoverable from edges already present in the candidate graph.
3. **Lineage-constrained oracle:** best exact edge-J obtainable while obeying parent/child/division constraints.

Interpretation:

- constrained oracle `<0.88`: current graph cannot reach the target; change candidates/representation;
- `0.88–0.92`: technically reachable but too little margin for private shift;
- `≥0.93`: the solver/calibration track deserves immediate priority.

This experiment is CPU-local, exact, and should interrupt all speculative training.

---

## 3. The metric-aligned solver edge

**EV: HIGH · STATUS: permitted · Offline Kaggle: yes · Novelty: high in this field**

The existing SCIP solve minimizes a generic combination of edge probability, appearance, disappearance, and division costs. It does not optimize expected Jaccard.

With calibrated candidate probabilities `pTP(e)` and `pFP(e)`, expected edge-J is approximately:

```text
Σ pTP(e) x_e / [G + Σ pFP(e) x_e]
```

A Dinkelbach-style parametric solve converts the fraction into repeated lineage-constrained ILPs:

```text
maximize Σ [pTP(e) - λ pFP(e)] x_e
```

The constant `-λG` does not affect edge selection, so hidden test `G` is not required for a fixed OOF-calibrated `λ`. This is a standard fractional-programming transformation; see [Dinkelbach's original method](https://doi.org/10.1287/mnsc.13.7.492).

Important qualification: because sparse annotations ignore some wrong-looking edges, `pFP` is not simply `1-pTP`. Train or calibrate separate probabilities for TP, metric-counted FP, and ignored edge, or use a conservative two-class approximation and tune `λ` strictly embryo-held-out.

### First experiment

- Freeze detections and candidate edges.
- Calibrate edge logits out-of-fold (temperature/isotonic calibration).
- Sweep `λ`, appearance, disappearance, and division costs against the **exact adjusted metric**.
- Compare generic MAP ILP versus metric-aligned ILP on both complete folds.
- Go if min-fold edge/adjusted-J rises `≥0.010` with no fold regression.

Expected gain is `+0.005–0.030`, not `+0.18`; its first value is revealing whether the candidate graph, rather than the solver, is the ceiling.

---

## 4. What should actually produce the edge gain

The levers below overlap; their high ends must not be added mechanically.

| Rank | Lever | Honest edge-J delta | Why |
|---:|---|---:|---|
| 1 | Multi-domain association model: Trackastra `ctc` → SCIP | `+0.02–0.07` | Directly replaces a two-embryo, two-frame association head with a diverse pretrained linker. |
| 2 | Metric-aligned calibration + ILP | `+0.005–0.030` | Converts good candidates into the edge set the metric rewards. |
| 3 | Multi-frame motion/path context | `+0.01–0.04` | Current model scores consecutive pairs; velocity, acceleration and skipped-frame consistency disambiguate dense local swaps. |
| 4 | Same-modality association pretraining (DRO/Zebrahub) | `+0.01–0.05` | Learns embryo motion/division, not merely static detection. |
| 5 | Graph-guided detection repair/gaps | `+0.005–0.025` | Recovers missing endpoints only where a track makes the raw-image evidence credible. |
| 6 | NIS3D detector pretraining | `0–0.015` now | Static volumes; marginal once held-out node recall is already high. |
| 7 | Count/centroid/TTA polish | score `+0.005–0.015` | Valuable final calibration, incapable of closing an 0.18 association gap. |

### Why Trackastra should be tested before building a new GNN

Trackastra predicts associations over a temporal window, supports divisions, and its diverse general model outperformed specialized models out of domain. Its official `ctc` checkpoint supports 2D and 3D and was trained on all available CTC GT/ERR_SEG datasets. Its repository is BSD-3-Clause and supports SCIP-backed ILP. [Repository](https://github.com/weigertlab/trackastra), [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf).

First experiment:

1. Freeze current detections.
2. Rasterize each point as a tiny anisotropic ellipsoid instance mask.
3. Run the local `ctc` checkpoint for associations.
4. Feed association probabilities into the existing SCIP solve.
5. Score full held-out embryos; no fine-tuning on the evaluated embryo.

Gate: `≥0.015` edge-J improvement on both folds. Package the wheel and weights in a Kaggle Dataset for internet-off submission.

### The unusually relevant external trajectory data

CTC `Fluo-N3DL-DRO` is a developing Drosophila embryo acquired with SIMView light-sheet at `0.406 × 0.406 × 2.03 µm` and 30-second cadence—remarkably close to this competition's sampling. It tracks selected nervous-system lineages, so it is motion/association supervision rather than a count prior. [Official CTC dataset page](https://celltrackingchallenge.net/3d-datasets/).

Public Zebrahub/DaXi trajectories remain the best same-organism motion/division source if used as noisy pretraining, never as test-crop identity/label transfer. Randomize physical scale, orientation, time cadence, developmental speed, density, missing detections, false detections, and localization noise.

---

## 5. Organizer association-head weaknesses worth ablation

These are hypotheses supported by code, not guaranteed bugs.

### A. Two-frame context

The checkpoint default window is two frames. The edge head cannot directly use established track velocity, acceleration, skipped-frame consistency, or neighbourhood flow. The ILP enforces global structure but cannot invent missing discriminative information.

First experiment: rerank only low-margin edges with three-frame features—forward/backward consistency, acceleration residual, turning angle, track age, and agreement with median kNN tissue velocity. This transfers the observation-centric insight from [OC-SORT](https://github.com/noahcao/OC_SORT) without importing its 2D pedestrian assumptions.

### B. Absolute spatial position and non-physical displacement

The head receives absolute normalized spatial Fourier features and a pairwise delta computed in original voxel indices divided by 100. It does not explicitly convert this delta to physical microns. Because the axes enter separately, the MLP *can* learn the fixed anisotropy; this is not proof of a 4× error. But absolute crop position is a shortcut, and physical/translation-invariant features should generalize better to a third embryo.

First experiment: freeze the U-Net and retrain only the edge head:

1. baseline features;
2. physical-micron displacement/norm/direction;
3. physical features with absolute spatial PE removed, plus reflections/translations and local kNN deformation features.

The particle-tracking literature provides the transferable principle: classify candidate edges using relational geometry rather than global hit location. [Exa.TrkX interaction-network paper](https://arxiv.org/abs/2103.16701), [embedded-space GNN tracking](https://arxiv.org/abs/2007.00149).

### C. Density-dependent all-parent softmax

The implementation softmaxes each target over all source detections, has no explicit birth/dustbin parent, and then applies a fixed probability threshold. Probability scale can therefore change with frame density, while new/unmatched cells must still allocate mass to some source.

First experiment: local physical candidate mask + explicit no-parent dustbin + density-stratified calibration. This is cheap and may matter more than model size on a denser hidden embryo.

---

## 6. The orthogonal edge: target-embryo self-supervised adaptation

**EV: MED-HIGH with decisive upside · STATUS: needs exact rules wording check · Offline Kaggle: yes**

This is the structural blind spot in the current plan. We know the hidden embryo is the domain shift, but the plan adapts only before seeing it. At inference, the complete unlabeled target movie is available. Use it to estimate the target's motion and calibrate association without reconstructing labels through submissions.

There is direct precedent:

- DARTH adapts MOT detection and association representations at test time using consistency and contrastive losses under domain shift. [ICCV 2023 paper](https://openaccess.thecvf.com/content/ICCV2023/html/Segu_DARTH_Holistic_Test-time_Adaptation_for_Multiple_Object_Tracking_ICCV_2023_paper.html).
- Path Consistency learns association without identity labels by requiring consistent matches when frames are skipped. [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Self-Supervised_Multi-Object_Tracking_with_Path_Consistency_CVPR_2024_paper.html).
- CMTT-JTracker applies fully test-time adaptation directly to cell tracking and updates only normalization parameters; its reported target adaptation takes under five minutes on a 3080 Ti. [Paper](https://academic.oup.com/bib/article/doi/10.1093/bib/bbae591/7902784), [code](https://github.com/lynnwahh/CMTTJ).
- Neural Scene Flow Prior fits a continuous motion field at runtime without labelled training data and is explicitly motivated by out-of-domain robustness. [NeurIPS 2021 project/paper](https://lilac-lee.github.io/Neural_Scene_Flow_Prior/).

### Safe design

Do not fine-tune the whole detector on its own confident mistakes. Instead:

1. Build pseudo-positive edges only where the base model, physical nearest neighbour, forward/backward matching, and ILP agree.
2. Build pseudo-negatives from mutually exclusive nearby alternatives.
3. Require direct `t→t+2` association to agree with composed `t→t+1→t+2` association.
4. Fit only a small physical-motion residual, LayerNorm affine parameters, temperature, or final pair MLP.
5. Reset to source weights for each embryo; cap steps and monitor entropy/collapse.
6. Ensemble adapted and unadapted scores before the metric-aligned ILP.

### Falsification protocol

Simulate true test time twice:

- hide all `44b6` labels, adapt only on its images/detections, then score;
- hide all `6bba` labels, repeat independently.

Go only if both improve by `≥0.010` edge-J with no count/division regression. Compute: 1–4 T4 hours per embryo. Before submission, obtain a literal reading or organizer confirmation that optimization on provided unlabeled test images is permitted.

---

## 7. Divisions: weapon, but not free money

For division recall `r` and precision `p`, division Jaccard is:

```text
D = r / [1 + r(1-p)/p]
```

To reach `D=0.60`:

| Recall | Required precision |
|---:|---:|
| 0.60 | 1.000 |
| 0.70 | 0.808 |
| 0.80 | 0.706 |
| 0.90 | 0.643 |
| 1.00 | 0.600 |

Because edge volume (~129k) is enormous relative to 151 divisions, the division term initially makes true forks very valuable. The marginal break-even precision is approximately 0.8% while division-J is zero, rising to ~17% at D=0.2, ~29% at D=0.4, and ~37% at D=0.6. This does **not** mean a low-precision fork generator reaches D=0.6; the table above governs the global operating point.

### Cheapest credible path

1. Benchmark Trackastra `ctc` division probabilities on frozen detections.
2. Fix the organizer loss's division row weight (currently effectively 1.0) and train an explicit fork posterior.
3. Add mother/daughter appearance, near-symmetric daughter displacement, daughter separation, intensity/shape change, and deviation from local tissue flow.
4. Put calibrated fork probability into SCIP and sweep threshold jointly against edge + division score.
5. Use separate embryo-stratified calibration because 26 versus 125 events is a severe imbalance.

First gate: division-J `≥0.25`, edge-J degradation `≤0.005` on both folds. A realistic eventual private division-J is `0.25–0.50`, contributing `+0.025–0.050`. Treat `0.60` as stretch.

---

## 8. Legal metric surface: use, quarantine, discard

### Permitted, modest value

- **Prune isolated/edge-neutral nodes:** raw edge-J is unchanged while the count multiplier improves. At raw J=.70, removing 10% of estimated nodes is worth only about `+0.007`.
- **Exact count-cost sweep:** include node cost in the ILP and gate on adjusted-J, never on count ratio alone.
- **Subvoxel refinement/arbitration:** useful near the 7 µm match gate; measured lost assignment is only 0–2%, so the ceiling is small.
- **Sparse-region over-linking:** edges completely outside annotated interiors are ignored, but any wrong edge touching an annotated interior becomes FP. Use only as an exact-metric-gated recall bias.

### Needs written rules clearance; operationally quarantine

The division evaluator can credit a GT division when stage nodes share a weakly connected component containing an unmatched predicted fork, while that fork can evade division FP because its own node is unmatched. This is the known evaluator defect. Do not build or submit it without explicit written organizer clearance; the prize/DQ and patch risks dominate its EV.

### Dead ends

- duplicate edges are deduplicated;
- NaN/Inf coordinates do not create useful matches;
- dense proposal clouds cannot create extra TP under one-to-one matching and incur count/assignment costs;
- free unmatched edges cannot create edge TP;
- malformed IDs, missing datasets, or schema tricks are not a responsible pathway.

---

## 9. Reordered execution pathway

### Phase A — prove the ceiling (1–2 days, CPU/local)

1. Full-199, both-fold ILP exact score with TP/FP/FN, count ratio, and divisions.
2. Endpoint, candidate, and lineage-constrained edge oracles.
3. Error split after ILP: missing endpoint, missing candidate edge, wrong probability ranking, solver selection.

**Kill gate:** if constrained oracle `<0.88`, stop treating 0.88 edge as an ILP problem.

### Phase B — cheap objective and representation fixes (2–4 days)

1. Probability calibration + metric-aligned fractional ILP.
2. Local parent mask, birth dustbin, density-stratified threshold.
3. Physical/invariant edge-head ablation, U-Net frozen.
4. Three-frame low-margin reranker.

### Phase C — import association intelligence (3–7 days)

1. Trackastra `ctc` zero-shot on frozen detections.
2. Feed its scores to the same SCIP/metric-aligned solve.
3. Fine-tune only if zero-shot shows signal.
4. Pretrain association on Fluo-N3DL-DRO and public Zebrahub trajectories with competition-like corruptions.

### Phase D — divisions and graph repair (parallel)

1. Explicit fork posterior and joint score sweep.
2. Conservative iterative graph-guided redetection at gaps/births/deaths.
3. K-best/perturbed ILP ensemble only on ambiguous crops.

### Phase E — wildcard (after rules check)

1. Per-embryo scene-flow residual.
2. Path-consistent target-time adaptation.
3. Ensemble adapted/unadapted edge probabilities.

No lever is banked until it lifts both held-out embryo folds under the exact combined metric.

---

## 10. Honest score budget

These are conditional ranges, not additive promises:

| Stage | Plausible private contribution |
|---|---:|
| Full-fold ILP/objective calibration | edge `+0.00–0.03` |
| Trackastra/multi-frame association | edge `+0.02–0.07` |
| Same-modality association pretraining | edge `+0.01–0.05` |
| Target-time/path adaptation | edge `+0.005–0.03` |
| Graph-guided repair + polish | score `+0.01–0.03` |
| Divisions at J=.25–.50 | score `+0.025–0.050` |

Central outcome after overlap: edge `0.79–0.85`, division contribution `+0.025–0.050`, polish `+0.005–0.015`, final `0.84–0.89`. A high oracle plus successful Trackastra/target adaptation can move the stretch into `0.90–0.92`. Nothing observed yet supports `0.94` as the likely endpoint.

## Ranked decision table

| Rank | Edge | EV | Status | Offline/T4×2 feasibility | Concrete first experiment |
|---:|---|---|---|---|---|
| 1 | Candidate/lineage oracle → expected-J fractional ILP | High | Permitted | CPU-local; submission inference unchanged | Compute all three oracle ceilings, then OOF-sweep calibrated `λ`. |
| 2 | Trackastra `ctc` association scores → existing SCIP | High | Permitted; verify checkpoint license artifact | Local weights; T4 inference; `<12h` likely | Tiny ellipsoid masks on frozen detections; full two-fold zero-shot score. |
| 3 | Local physical parent mask + birth dustbin + density calibration | High | Permitted | Small code/model change; T4 | Retrain edge head with frozen U-Net; stratify result by cell density. |
| 4 | Physical, translation-invariant edge head | Med-high | Permitted | 2–6 T4 hours | Raw-voxel versus physical-µm versus no-absolute-PE ablation. |
| 5 | Three-frame/path-consistent association residual | Med-high | Permitted for source training | CPU reranker first; T4 only if trained | Rerank only low-margin edges using acceleration and `t→t+2` consistency. |
| 6 | Target-embryo self-supervised association adaptation | Med-high, high upside | **Needs rules check** | 1–4 T4 hours; internet off | Hide each embryo's labels, adapt on its movie, require gains both directions. |
| 7 | Explicit calibrated division posterior in SCIP | Med-high | Permitted | Frozen detections; 1–4 T4 hours | Joint edge/division threshold sweep; first gate D-J `≥.25`. |
| 8 | Fluo-N3DL-DRO/Zebrahub motion pretraining | Medium | Permitted external data; provenance hygiene | Train externally/Kaggle; ship weights only | Corrupt trajectories to measured FN/FP/anisotropy, fine-tune association only. |
| 9 | Per-embryo neural scene-flow feature | Medium, wildcard | Permitted inference; verify transductive wording | 1–4 T4 hours with tiles | Fit flow without labels, append flow-residual edge cost, reverse-fold test. |
| 10 | Isolated-node pruning/count cost/subvoxel refinement | Low-med | Permitted | CPU; minutes | Exact-metric sweep; expected combined gain `<.015`. |

All expected gains above are hypotheses until exact full-fold OOF. The known unmatched-fork division loophole is excluded: **needs written organizer clearance and should be treated as forbidden for planning purposes**.

## The ONE bet I would make

If forced to bet one non-obvious mechanism capable of changing the private-board outcome, I would bet on **path-consistent, target-embryo adaptation of a multi-domain association model, followed by a metric-aligned SCIP solve**.

Why: the contest's private difficulty is embryo domain shift; this uses the actual unlabeled target embryo to estimate its motion/association regime, but constrains adaptation with cycle/path consistency rather than self-confidence alone. Trackastra supplies a diverse starting representation; the fractional ILP converts it into the exact edge trade-off the metric rewards. It attacks generalization and metric alignment simultaneously—the two places the current pipeline is structurally weakest.

The immediate move remains the oracle. It tells us within days whether that bet has a runway or whether 0.88 edge is fantasy on the current detections.
