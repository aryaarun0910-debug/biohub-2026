# Lineage/division models, positive-unlabeled learning, and domain adaptation for microscopy — 2026-07-12

Lane 04 of the brain swarm (see `README.md`). Companion to `../divisions_2026-07-06.md`, which
already covers division *feature engineering* (mask-free ratio features, LOEO calibration plan,
the "add-only fork gate" recipe) in depth — this lane does not re-derive that; it focuses on the
three legs the mission called out that the sibling file under-covers: (1) **lineage solvers**
(ILP/multicut/GNN machinery that a fork posterior plugs into), (2) **PU/sparse-positive-label
theory** (the actual math for training when unlabeled ≠ negative), and (3) **domain
adaptation/generalization** across the disjoint hidden embryo. Confirmed the exact metric
definitions against the organizer's own repo (§0) — this changes some conclusions from casual
metric-reading.

Evidence tags: **[V]** verified by reading the primary source/abstract directly this session;
**[C]** claim as reported in a secondary source or abstract-only fetch; **[I]** my inference for
this competition.

---

## 0. Metric ground-truth (read before anything else)

Fetched `metrics.md` directly from the organizer's repo,
[royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md).
**[V]**

- `division_jaccard = TP / (TP + FP + FN)`, **micro-averaged** (TP/FP/FN summed across all
  samples before dividing) — so one embryo with many divisions dominates the number.
- A **predicted TP** must satisfy four conditions: one-node-stage coverage, both daughter
  lineages touched, single connected component, and contains a predicted fork — i.e. partial
  credit does not exist at the division level, it's a strict match.
- **FP definition is narrower than "any wrong division":** *"A predicted dividing cell whose
  match lands on a GT cell with outgoing edges (so the region is annotated) but that is not
  paired to any GT division."* This means **a predicted fork landing in an *unannotated* region
  of the sparse labels is NOT scored as a false positive** — it's ignored. This is the single
  most important, easy-to-miss fact for this lane: **the sparse, positive-only labeling scheme is
  baked directly into the metric, not just the training data.** A precision-hungry classifier
  does not need to suppress every geometrically-plausible-but-unlabeled fork — only forks that
  land on an annotated (has-outgoing-edges) GT node that turns out to be a non-division.
- Tolerance: ±1 timepoint on the GT split. `overall = adjusted_edge_jaccard + 0.1 * division_jaccard`.
- `adjusted_edge_jaccard = max(0, jaccard · (1 − 0.1·(T_pred − T_true)/T_true))` — over-predicting
  total node count is what's penalized on the edge side, not over-predicting forks per se.

Practical consequence: false division precision only has to beat the ratio of (annotated,
non-dividing GT cells your fork proposals hit) to (true divisions you hit) — not the ratio over
*all* candidate sites in the volume. This is a materially easier bar than "precision over the
whole image," and it argues for **higher-recall proposal generation than you might otherwise
risk**, since unlabeled-region false positives are free.

---

## Executive summary — 5 highest-value methods

1. **Ultrack (Nature Methods 2025)** — candidate-segmentation + temporal-consistency ILP that
   natively supports division as a graph event, proven on zebrafish/fly/worm light-sheet at
   terabyte scale. [V] Maps to: lineage solver. It's already the organizer's own tool family
   (per MEMORY), so the highest-leverage move is reading its division-cost API, not replacing it.
   https://www.nature.com/articles/s41592-025-02778-0 , code https://github.com/royerlab/ultrack

2. **Two-stage proposal→classifier mitosis detection** (YOLO11x low-threshold proposals →
   ConvNeXt-Tiny precision filter, arXiv 2509.02627) — concretely measured **precision 0.762 →
   0.839** (+7.7 pts) at comparable recall by adding a second-stage classifier on top of a
   deliberately over-recalled first stage. [V] This is the exact shape of fix needed for "0 TP
   from 7 geometric proposals": don't fix the proposal generator, add a learned filter after it.
   Maps to: division posterior. https://arxiv.org/abs/2509.02627v2

3. **nnPU — non-negative PU risk estimator** (Kiryo et al., NeurIPS 2017) — the correct loss
   family for "unlabeled ≠ negative," because it prevents the risk estimate from going negative
   and overfitting when using a flexible classifier (any CNN/GBDT capacity, not just linear).
   [V] Maps to: PU loss, IF we retrain a detector/classifier on the sparse labels directly rather
   than only calibrating a post-hoc posterior. https://arxiv.org/abs/1703.00593 , code
   https://github.com/kiryor/nnPUlearning

4. **linajea** (Malin-Mayor et al., Nature Biotechnology 2022, zebrafish/mouse/fly light-sheet) —
   whole-embryo lineage reconstruction trained from **sparse point annotations**, using an ILP
   over a candidate graph where node/edge costs are *learned* scores, solved block-wise for
   scale. [V] Directly analogous problem shape (sparse labels + lineage + light-sheet), same data
   modality class as our zebrafish task. Maps to: lineage solver + PU-ish sparse training signal.
   https://www.nature.com/articles/s41587-022-01427-7 , code via
   https://funkelab.github.io/projects/linajea/

5. **DINOCell-style self-supervised domain adaptation** (arXiv 2604.10609) — DINOv2
   self-distillation continued-pretraining on ~130k unlabeled microscopy images across 63 sources
   before fine-tuning, reporting 36–285% zero-shot OOD gains over random init. [V, abstract-level]
   Maps to: domain adaptation. The lesson for us isn't "run DINOCell" (wrong modality/detector
   task) but "spend unlabeled pretraining budget on OUR OWN embryos/timepoints before touching
   the 2 labeled ones," since we have far more unlabeled frames than labeled divisions.
   https://arxiv.org/html/2604.10609v1

---

## Findings table

| Method | Source | Core idea | Why it fits our metric/constraints | Bet | First experiment + kill-gate | Cost |
|---|---|---|---|---|---|---|
| Ultrack division-native ILP | [Nat. Methods 2025](https://www.nature.com/articles/s41592-025-02778-0), [code](https://github.com/royerlab/ultrack) | Candidate segmentations from multiple params + temporal consistency, solved as ILP where division is a native graph event, not a bolt-on | Same modality/scale (zebrafish light-sheet), already in our stack per MEMORY; a learned division-cost plugged into its ILP is the least-integration-risk path | Lineage solver | Read the ILP's division-cost hook; swap constant cost for the fork classifier's score; LOEO on edge-J delta | Low (config only) if already wired; Med if not |
| Two-stage proposal→ConvNeXt classifier | [arXiv 2509.02627](https://arxiv.org/abs/2509.02627v2) | Deliberately low-threshold, high-recall stage-1 proposals; stage-2 CNN classifier removes false positives; measured +7.7pt precision | Directly matches "0 TP from 7 proposals" failure mode — the fix is a learned filter, not a better geometric rule | Division posterior | Reuse existing geometric proposal generator (accept its low precision) and bolt a small classifier (GBDT, not CNN — no learned image features needed if points+patches used) after it; LOEO precision/recall curve | Low (1-2 days), reuses sibling lane's feature list |
| nnPU non-negative risk estimator | [Kiryo NeurIPS 2017](https://arxiv.org/abs/1703.00593), [code](https://github.com/kiryor/nnPUlearning) | Clips the negative-risk term at 0 to stop a flexible classifier overfitting when trained on P+U data instead of P+N | If we ever train the fork/detection classifier directly against unlabeled cells as "U" (not "N"), plain uPU will overfit at our data volumes (151 divisions); nnPU is the safe drop-in loss | PU loss | Swap standard BCE for nnPU loss in the fork classifier with class-prior π estimated per embryo (§ recipe below); compare LOEO precision at fixed recall vs BCE baseline | Low (loss-function swap only) |
| Object detection as PU | [arXiv 2002.04672](https://arxiv.org/abs/2002.04672) (BMVC 2020) | Reframes detector training loss so unlabeled proposals are not forced to be negative; shows gains specifically when annotations are incomplete | Same shape as our "unlabeled cells are not annotated as non-dividing" problem, applied to detection heads generally, not just PU classifiers | PU loss | Only relevant if retraining a full detector (not our current point-based pipeline); low priority unless the geometric-proposal approach is abandoned entirely | Med (requires a trainable detector) |
| OOD-SEG (PU segmentation via OOD detection) | [arXiv 2411.09553](https://arxiv.org/abs/2411.09553) (MedIA) | Formulates sparse multi-class positive-only segmentation as pixel-wise PU; reuses OOD-detection scoring machinery instead of a custom PU loss | Conceptually elegant reframe of "sparse, positive-only" that generalizes past nnPU; but built for pixel segmentation, we're point/graph-based | PU loss (alternative framing) | Skip implementing; useful as a second citation if nnPU underperforms and a reviewer asks "did you consider alternatives" | n/a (reference only) |
| SparseDet pseudo-positive mining | [arXiv 2201.04620](https://arxiv.org/abs/2201.04620) | Learns to separate detector proposals into "confidently labeled" vs "unlabeled" subsets and mines pseudo-positives from the latter | Matches our metric's own leniency (§0): forks landing in unannotated regions aren't penalized, so mining pseudo-positive divisions from unlabeled frames is metric-safe by construction | PU loss / self-training | Only if extra positives are needed to fight the 151-division data scarcity; run AFTER the two-stage gate is working, as a data-augmentation step, not a first move | Med |
| Class-prior / mixture-proportion estimation (Elkan-Noto 2008, du Plessis 2015, Ramaswamy 2016) | [arXiv 1611.01586](https://arxiv.org/abs/1611.01586) et al. | Estimates the true positive fraction π in the unlabeled set from P+U data alone, needed by nnPU and by threshold calibration | Division rate genuinely differs 26-vs-125 between the two labeled embryos — π is NOT transferable as a constant; must be (re-)estimated, and this is the literature's standard tool for doing that without extra labels | PU loss / calibration | Estimate π per training embryo via penalized-Pearson-divergence method; check whether the two embryos' estimated π actually differ 5x like the raw counts suggest (sanity check for whether π should be treated as a soft per-embryo prior, per mission ask) | Low (closed-form/analytic solution exists) |
| Teacher-student consistency regularization (PLMT-style) | [PLOS ONE 2024](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0300039) | Self-training pipeline: pseudo-label unlabeled data with a first-stage model, then refine with a teacher-student + consistency loss | Could turn our huge pool of *unlabeled* frames/cells into extra soft supervision for the fork classifier; but risk of teacher hallucinating confident-but-wrong divisions that then get "confirmed" by the student is real with only 151 positives to anchor confidence gating | Domain adaptation / PU | Only pursue after the supervised two-stage gate hits its LOEO gate (division-J≥0.25); use it to close remaining gap, not as first attempt | Med-High |
| Moral Lineage Tracing (multicut + division constraint) | [Jug et al. CVPR 2016](https://arxiv.org/abs/1511.05512), [Rempfler ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Rempfler_Efficient_Algorithms_for_ICCV_2017_paper.pdf) | Joint segmentation+tracking as a minimum-cost multicut with "morality" constraints (cells divide, never merge) enforced via path-cut inequalities in the ILP | Formal, hard structural constraints (no merges, max 2 children) that are stage/rate-independent and hence DO transfer, unlike a soft division-cost weight | Lineage solver | If not already enforced by Ultrack/motile's constraint set, verify MaxChildren=2/MaxParents=1/no-merge is hard-constrained, not just cost-penalized, in our ILP config | Low (config check) |
| linajea sparse-annotation lineage ILP | [Nature Biotechnology 2022](https://www.nature.com/articles/s41587-022-01427-7), [code](https://funkelab.github.io/projects/linajea/) | Learns node/edge costs from sparse point tracks, feeds them into an ILP that reconstructs full lineages; block-wise parallel solve | Nearly identical problem statement to ours (sparse point labels, zebrafish/light-sheet, full lineage target); reconstructed 75.8% of 1-hr lineages vs 31.8% prior SOTA on their hardest (mouse) dataset [C] | Lineage solver / sparse-label training | Read their loss for handling sparse tracks (how do they avoid treating unlabeled cells as negative examples for the edge-cost network?) and port whichever choice they made to our fork classifier | Med (reading + selective porting) |
| GNN edge-classification tracking | [arXiv 2202.04731](https://arxiv.org/abs/2202.04731) | Cells as graph nodes, GNN message-passing predicts frame-to-frame edges including divisions | Precedent for a learned-graph approach if the classical proposal+classifier plateaus; heavier to build/debug for a 2-embryo dataset | Lineage solver / division posterior | Low priority — parameter count vs. 151-division supervision budget is a bad ratio; only revisit if GBDT features saturate | High |
| C. elegans StarryNite scoring heuristic | [Bao et al. PNAS 2006](https://www.pnas.org/doi/full/10.1073/pnas.0511111103) | Classic (pre-deep-learning) division scorer: penalizes short-cycle mothers, non-spherical ("not yet mitotic-rounded") mothers, and dissimilar daughter pairs | Confirms, from a 20-year-old but battle-tested pipeline, that the sibling lane's ratio features (daughter similarity, mitotic rounding) are exactly the right long-standing feature family, not a novel guess | Division posterior (feature validation) | No new experiment — already covered by sibling lane §3; cite as independent confirmation | n/a |
| DINOCell self-supervised domain adaptation | [arXiv 2604.10609](https://arxiv.org/html/2604.10609v1) | Continued DINOv2 self-distillation pretraining on unlabeled cross-source microscopy before supervised fine-tuning; 36-285% zero-shot OOD gains | We have far more unlabeled frames/timepoints per embryo than labeled divisions — self-supervised pretraining on OUR OWN unlabeled data is a cheap way to get features that aren't overfit to 2 embryos' idiosyncrasies | Domain adaptation | Only relevant if training an image-feature (CNN) branch for the classifier; if staying with handcrafted ratio features (sibling lane §3), this bet is lower priority — ratio features are already scale-free by construction | High if pursued (needs a pretraining pipeline) |
| Instance normalization / whitening domain generalization | [survey, Springer 2024](https://link.springer.com/article/10.1007/s10462-024-10817-z) | Strips style/domain-specific statistics from features via instance norm/whitening so the model can't key on embryo-specific intensity scale | Directly addresses "division-rate/density/velocity scales differ per embryo" for any learned image-feature component; cheap architectural change (a norm layer) vs. DINOCell's heavy pretraining | Domain adaptation | If any CNN features are used at all, add instance-norm layers and re-run LOEO; compare to batch-norm baseline | Low |
| Self-supervised Z-slice / anisotropy handling | [arXiv 2503.04843](https://arxiv.org/html/2503.04843) | Learns axial (z) super-resolution from unlabeled volumes via knowledge distillation, addressing PSF-driven anisotropy without new labels | Our voxel spacing is ~4x anisotropic (z coarser); any 3D geometric/patch feature computed in raw voxel units rather than physical µm will silently fail to transfer — this is a data-representation fix, not a model fix | Domain adaptation | Cheaper fix available: always convert to physical µm before computing features (already the plan per sibling lane §3) — treat this method as a fallback only if µm-normalization alone proves insufficient | Low (µm fix) / High (full method) |
| Calibration: isotonic vs Platt for rare/imbalanced events | [MachineLearningMastery summary](https://machinelearningmastery.com/probability-calibration-for-imbalanced-classification/), [imbalance calibration study](https://arxiv.org/pdf/2202.00386) | Isotonic regression corrects arbitrary monotonic miscalibration and empirically outperforms Platt scaling under class imbalance/rare events, at the cost of needing more calibration data than Platt | With only 26-125 divisions per embryo for LOEO calibration, isotonic's extra data appetite is a real risk; Platt (2-parameter sigmoid) may be the safer choice given how few positives are available for calibration-set fitting | Division posterior (calibration step) | Fit both on the held-out LOEO embryo's scores; compare stability of the chosen threshold (bootstrap CI) — pick whichever has the tighter CI, not the better point estimate | Low |

---

## Concrete recipe for a precise division posterior (target division-J ≥ 0.25, Δedge-J ≥ −0.005)

This extends the sibling lane's "add-only fork gate" (already the right shape) with the PU/lineage
/domain-adaptation angle this lane was asked to add.

**Features** (reuse sibling lane §3's scale-free ratios — daughter-displacement symmetry, distance
ratio, mass-conservation ratio, mitotic-rounding Z-score, NN-normalized daughter separation,
local-flow divergence). Add one new feature this lane's search surfaced: **local tissue-flow
divergence** computed from a smoothed neighbor-velocity field (candidate division sites behave as
local sources — PIV/optical-flow literature on morphogenesis, e.g.
https://pmc.ncbi.nlm.nih.gov/articles/PMC11072672/), normalized by local mean speed to stay
scale-free.

**Class-prior handling (the PU-specific addition):** Before calibrating a decision threshold,
explicitly estimate the class prior π (fraction of true divisions among candidate proposals) *per
training embryo* using a closed-form class-prior estimator (du Plessis/Ramaswamy family, no extra
labels needed beyond the sparse positives — https://arxiv.org/abs/1611.01586). Do not assume π is
shared across embryos: the 26-vs-125 division-count disparity between 44b6 and 6bba is exactly the
kind of prior-shift the "positive-unlabeled classification under class-prior shift" literature
addresses (https://link.springer.com/article/10.1007/s10994-022-06190-z). Feed the estimated π
into the decision rule as a **soft prior on the acceptance threshold**, not a hard cutoff: shift
the operating point toward higher precision on embryos where the estimated π is low, and allow
more recall where π is high. This operationalizes the mission's "division rate as a soft prior"
requirement.

**Training/loss:** If the classifier is trained end-to-end on candidate triplets (not just
hand-fit thresholds on features), use the nnPU risk estimator
(`R_pu = π·R_p⁺ + max(0, R_u⁻ − π·R_p⁻)`, https://arxiv.org/abs/1703.00593) instead of plain
BCE/cross-entropy, treating "unlabeled candidate triplets" as U rather than forcing them to
label 0. Given the metric's own leniency toward unlabeled-region false positives (§0), a GBDT
with class-weighted BCE is likely sufficient in practice and lower-risk to debug than nnPU's
non-standard gradient (clip-and-flip on the negative term) with only ~150 positive examples total
— **prefer BCE+class-weights as the default, keep nnPU as the fallback if BCE overfits the
majority-negative triplets** (test both under LOEO, pick by the held-out embryo's bootstrap CI).

**Decision rule:** Two-stage — (1) permissive geometric/velocity proposal generator (recall-first,
reuse existing one that produced "0 TP from 7"; the earlier failure was in using it directly as a
decision, not in the proposals themselves being unusable as *candidates*), (2) GBDT scores each
triplet on the feature set above, (3) accept only if score exceeds a **per-embryo isotonic- or
Platt-calibrated threshold at the target precision implied by the current π estimate**, favoring
Platt if the LOEO calibration set has too few positives for isotonic's flexibility to be trustworthy
(rule of thumb: <50 held-out positives → Platt), (4) accepted forks are added as a *second* daughter
edge only, never replacing an existing edge from the frozen linker, preserving the edge-J-neutral
guarantee from the sibling lane. Enforce MaxChildren=2/MaxParents=1/no-merge as **hard ILP
constraints** (moral-lineage-tracing style, https://arxiv.org/abs/1511.05512), not soft costs, since
those are stage-independent and the literature treats them as structural, not learned.

**Calibration protocol:** embryo-stratified LOEO both directions; report division-J with a
bootstrap CI (few positives per embryo mean high variance); kill this whole approach only if the
CI excludes 0.15 on both held-out directions after feature-importance pruning.

---

## PU/sparse-label loss stack, if we retrain (with the overfitting risk flagged)

If a decision is made to train a learned component (not just calibrate a threshold on handcrafted
features) directly against the sparse positive-only division/detection labels:

1. **Estimate π (class prior) per embryo first**, via a closed-form penalized-divergence estimator
   (Elkan & Noto 2008 baseline; du Plessis 2015/Ramaswamy 2016 refinements,
   https://arxiv.org/abs/1611.01586) — do this before writing any training loop, since every PU
   loss below needs π as an input.
2. **Use nnPU, not uPU**, as the risk estimator
   (https://arxiv.org/abs/1703.00593, code https://github.com/kiryor/nnPUlearning): clip the
   negative-risk term at zero and flip its gradient sign when it goes negative, which is precisely
   the overfitting-prevention mechanism needed with a high-capacity model and few positives.
3. **If detection-style (not classification-style) training is attempted**, adopt the
   "object detection as PU" reframing (https://arxiv.org/abs/2002.04672): unlabeled candidate
   regions are not forced to be background/negative in the loss.
4. **Mine extra pseudo-positives cautiously** via SparseDet-style separation of
   confidently-labeled vs. unlabeled proposals (https://arxiv.org/abs/2201.04620) or a
   teacher-student consistency loop (PLMT, https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0300039)
   — only after the base PU classifier is stable, and only using LOEO-held-out embryo performance
   to decide whether pseudo-positive mining helps or just amplifies the teacher's own biases.
5. **Explicit overfitting risk (per mission brief):** with 2 labeled embryos, ANY retrained
   component (even nnPU-regularized) can and will pick up embryo-specific nuisance correlations
   (absolute density, absolute division rate, acquisition-specific intensity scale) unless every
   input feature is scale-free/self-normalized (sibling lane §3) and unless LOEO is used as the
   *only* validation protocol — a random train/val split across both embryos' data will look great
   and generalize badly to the truly disjoint hidden test embryo. Retraining is a real risk
   multiplier for exactly the reason the mission flags it; the two-stage GBDT-on-ratio-features
   approach with only threshold/prior calibration retrained is the lower-variance default, and full
   PU retraining should be treated as an upgrade attempted only after that baseline is validated
   and only if its LOEO CI is unsatisfying.

---

## What will NOT transfer across embryos, and why

- **Absolute division rate, absolute cell-cycle length, mitotic synchrony/wave timing** — zebrafish
  divisions desynchronize with developmental stage (already established in sibling lane §3); a
  class prior π fit on one embryo is not the true π of another, confirmed independently here by
  the PU literature's own "class-prior shift" problem framing
  (https://link.springer.com/article/10.1007/s10994-022-06190-z) — this is exactly why π must be
  re-estimated per embryo rather than hard-coded, and why the decision threshold is a *soft* prior,
  not a constant.
- **Absolute geometric thresholds** (µm gates, absolute daughter separation, absolute local
  density) — same root cause, restated from the PU angle: any feature or threshold implicitly
  encoding the local density/rate scale is a proxy for π, and π moves across embryos.
- **A single global ILP division-cost weight** — as the sibling lane argues, a shared soft cost
  either floods the high-division embryo or starves the low-division one; the multicut/moral-
  lineage literature's answer is to make the *structural* constraints (no merge, ≤2 children) hard
  and stage-independent, while letting the *cost* be re-calibrated per embryo — consistent with
  treating rate as a soft prior rather than a hard rule.
- **Any CNN image-feature branch trained on raw voxel-unit inputs** — anisotropy (~4x coarser z)
  is embryo/acquisition-specific in degree if imaging settings vary at all; features must be
  computed in physical µm, and if learned image features are used at all, instance
  normalization/whitening (domain-generalization survey,
  https://link.springer.com/article/10.1007/s10462-024-10817-z) is needed to strip acquisition-
  specific intensity statistics — otherwise the classifier keys on scanner/embryo signature rather
  than division morphology.
- **Full end-to-end retrained detectors/GNNs at our label budget** — with 151 total divisions
  across 2 embryos, any model with enough capacity to need nnPU's overfitting protection is already
  in a regime where LOEO variance will be large; the CTC/MIDOG domain-generalization literature
  consistently shows that models trained on one lab/species/domain degrade on unseen ones
  ([2309.15589], abstract-level only — could not confirm full findings this session, flagged as
  [C] and worth a follow-up fetch), and 2 training embryos is a worse generalization regime than
  the multi-institution challenges that literature studies.

---

## Open questions

1. Does Ultrack's ILP (or motile, if that's the actual linker in use — sibling lane mentions both)
   expose a per-triplet division-cost hook that can directly accept the GBDT fork-classifier score,
   or does it only accept a single global division-cost parameter? This determines whether "Bet 1"
   in the executive summary is a config change or a fork of the library. **Not verified this
   session** — requires reading the actual `ultrack`/`motile` API in this repo, out of scope for a
   literature-only lane.
2. What class-prior π does the du Plessis/Ramaswamy estimator actually recover for 44b6 vs 6bba
   when run on real candidate triplets — does it match the raw 26-vs-125 label ratio, or diverge
   (which would suggest the *labeling*, not just the biology, differs in density between the two
   embryos)? Needs an actual run, not literature.
3. Full text of the MIDOG 2022 Domain Generalization Challenge (arXiv 2309.15589) could not be
   parsed this session (binary/PDF fetch failure) — its concrete cross-lab generalization
   recommendations (stain/intensity augmentation specifics, which architectures generalized best)
   remain unconfirmed beyond the abstract-level framing and should be re-fetched via a
   text-extraction route (e.g. ar5iv.org mirror) before relying on it further.
4. Is there enough public Zebrahub/Biohub lineage data (per MEMORY: public 50-epoch weights, public
   embryos) to actually run the class-prior estimator and nnPU pretraining on *thousands* of real
   division examples rather than 151 — turning this from a 2-embryo overfitting risk into a
   properly powered pretraining set? The sibling lane already proposes mining triplets from
   public lineage CSVs for the GBDT; the same public data should be used to fit a more stable π
   and, if full retraining is attempted, an nnPU pretraining stage. Not yet attempted.
5. Given the metric's leniency toward unlabeled-region false positives (§0, a genuinely new finding
   this session), does the sibling lane's "70% precision operating point" target need revision? If
   FPs only count against annotated cells, the effective precision bar to hit division-J ≥ 0.25 may
   be easier than an unconditional 70%-precision framing suggests — worth recomputing the D formula
   with the corrected FP definition before finalizing the operating point.

---

## Sources (this session)

- Metric ground truth: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
- Two-stage mitosis (YOLO11x→ConvNeXt): https://arxiv.org/abs/2509.02627v2
- Ultrack (Nature Methods 2025): https://www.nature.com/articles/s41592-025-02778-0 ,
  code https://github.com/royerlab/ultrack
- linajea (Nature Biotechnology 2022, sparse-annotation lineage ILP):
  https://www.nature.com/articles/s41587-022-01427-7 ,
  https://funkelab.github.io/projects/linajea/ ,
  preprint https://www.biorxiv.org/content/10.1101/2021.07.28.454016v1.full
- Moral Lineage Tracing: https://arxiv.org/abs/1511.05512 ; efficient algorithms follow-up
  https://openaccess.thecvf.com/content_ICCV_2017/papers/Rempfler_Efficient_Algorithms_for_ICCV_2017_paper.pdf
- Tracking by weakly-supervised learning + graph optimization, whole-embryo C. elegans:
  https://arxiv.org/abs/2208.11467
- C. elegans StarryNite division scoring heuristic (PNAS 2006):
  https://www.pnas.org/doi/full/10.1073/pnas.0511111103
- GNN for cell tracking in microscopy videos: https://arxiv.org/pdf/2202.04731
- Trackastra (ECCV 2024, division-aware transformer, CTC 2024 winner):
  https://arxiv.org/abs/2405.15700 , code https://github.com/weigertlab/trackastra
- nnPU non-negative risk estimator (Kiryo, NeurIPS 2017): https://arxiv.org/abs/1703.00593 ,
  code https://github.com/kiryor/nnPUlearning
- Object Detection as a Positive-Unlabeled Problem (BMVC 2020): https://arxiv.org/abs/2002.04672
- OOD-SEG (PU segmentation via OOD detection, MedIA): https://arxiv.org/abs/2411.09553
- SparseDet (pseudo-positive mining for sparse detection): https://arxiv.org/abs/2201.04620
- Class-prior estimation for PU learning: https://arxiv.org/abs/1611.01586 ; class-prior shift:
  https://link.springer.com/article/10.1007/s10994-022-06190-z
- Teacher-student consistency regularization, PLMT (PLOS ONE 2024):
  https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0300039
- DINOCell self-supervised microscopy pretraining: https://arxiv.org/html/2604.10609v1
- Domain generalization for semantic segmentation survey (instance norm/whitening, randomization):
  https://link.springer.com/article/10.1007/s10462-024-10817-z
- Self-supervised Z-slice augmentation for anisotropic 3D bio-imaging: https://arxiv.org/html/2503.04843
- Mitosis Domain Generalization Challenge 2022 (abstract-level only, full text unreadable this
  session): https://arxiv.org/abs/2309.15589
- Tissue flow / divergence at morphogenesis (PIV/optical flow review): https://pmc.ncbi.nlm.nih.gov/articles/PMC11072672/
- Calibration under class imbalance: https://machinelearningmastery.com/probability-calibration-for-imbalanced-classification/ ;
  https://arxiv.org/pdf/2202.00386
- Deep learning for rapid cell-division analysis in epithelial morphogenesis (fetch blocked,
  403 — cited from search snippet only, [C]): https://www.biorxiv.org/content/10.1101/2023.03.20.533343
- CHOTA higher-order cell-tracking accuracy metric (context for lineage-aware evaluation design):
  https://arxiv.org/pdf/2408.11571
