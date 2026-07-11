# Sparse-Label Learning + Developmental-Biology Priors + Imaging-Physics
### Research lane deliverable — Biohub Cell Tracking During Development
Author lane: sparse/PU + dev-bio priors + DaXi imaging physics
Date: 2026-07-06

---

## 0. How to read this doc

Every claim is tagged:
- **[V] verified** — stated in a primary paper/repo I read this session (citation inline).
- **[C] claim** — asserted by a source but not independently cross-checked, or a secondary/news source.
- **[I] inference** — my reasoning from verified facts to this competition. Not in any paper. Treat as a hypothesis to falsify.

Competition facts I take as given (from task brief, not re-verified here): metric = weighted adjacent-edge Jaccard + 0.1·division-Jaccard on a **disjoint hidden embryo**; 2 labelled embryos (44b6, 6bba); ~1% of visible nuclei annotated; unannotated visible nuclei are **unlabelled, not negative**; Kaggle T4×2, internet-off, ≤12 h.

The single most consequential fact for this lane: **the labels are positive-only and radically incomplete (~1%).** Any pixel/voxel loss that says "no annotation here ⇒ background" is training the model to suppress ~99% of the real signal. The entire loss design below exists to avoid that one mistake.

---

## 1. SUB-FRONT 1 — Sparse-label / positive-unlabelled loss stack

### 1.1 The core problem, stated precisely

We have detections/heatmap supervision where positives P are the ~1% annotated nuclei and everything else is **unlabelled U = (real unannotated nuclei) ∪ (true background)**. Naive supervised training minimises risk assuming U = negative. That is the biased/censored-positive regime. The clean theory for it is PU learning.

- **[V]** nnPU (Kiryo et al., NeurIPS 2017): the unbiased PU risk estimator `R = π·R_p^+ − (π·R_p^− − R_u^−)` goes **negative** and overfits badly with flexible models (deep nets); the fix is to clamp the second bracket at 0 (non-negative risk estimator), which is provably more robust and is the standard for deep PU. Reference implementation `pu_loss.py` exists. (arxiv.org/pdf/1703.00593; github.com/kiryor/nnPUlearning)
- **[V]** PU learning has already been applied to *exactly our failure mode* — **cell detection in histopathology with incomplete annotations**: treat partially-labelled cells as P, all other detected positions as U, then select reliable pseudo-labels from U with PU/P-classification (Zhou/Wang et al., MICCAI 2021, "Positive-Unlabeled Learning for Cell Detection in Histopathology Images with Incomplete Annotations", link.springer.com/chapter/10.1007/978-3-030-87237-3_49). This is the closest published analogue to our situation.
- **[V]** Partial-points nuclei work (Qu et al., 2020, arxiv.org/abs/2007.05448) shows the practical recipe that *works in microscopy*: extended-Gaussian mask from the few points to seed an initial model, then **self-training with background propagation** to exploit unlabelled regions and *suppress false positives*. Confirms the two-stage seed→self-train pattern is field-standard.

### 1.2 The critical hidden parameter: the class prior π

- **[V]** All unbiased/nnPU risk estimators require the positive class prior π = P(y=1) (fraction of U that is actually positive). (Kiryo 2017)
- **[I]** Here π is effectively "fraction of candidate voxels/detections that are real nuclei." We cannot know it, but we can *bound* it: because only ~1% of visible nuclei are annotated, the annotated P set is a tiny biased sample, and the effective positive density among unlabelled candidates is high. Mis-setting π is the number-one way nnPU silently fails. **Do not** try to hit π exactly; instead use it as a tunable knob and select it on the tracking metric (Section 4).

### 1.3 RECOMMENDED LOSS STACK (ranked, concrete)

The recommendation is a **staged stack**, not a single loss. Each stage is independently ablatable.

**Stage A — Detection/heatmap head with masked + PU loss (the load-bearing choice).**
1. Build supervision as a small-radius Gaussian blob at each annotated centroid (StarDist/CARE-style; **[V]** extended-Gaussian mask, Qu 2020). Positives = inside-blob voxels.
2. **Do NOT** apply background loss over the whole volume. Two acceptable options, in priority order:
   - **A1 (primary): masked positive-only + nnPU on candidates.** Foreground loss on annotated blobs. For the "negative" term, do **not** use all non-blob voxels; sample a *candidate set* U (e.g. local-maxima of a blob-detector or all voxels above an intensity floor) and apply the **nnPU non-negative risk estimator** over P (annotated) vs U (candidates), with π as a swept hyperparameter. **[V]** estimator form + clamp from Kiryo 2017.
   - **A2 (fallback if nnPU is unstable): masked BCE + hard-negative exclusion.** Compute background loss only on voxels that are confidently non-nucleus (low intensity AND far from any predicted peak), explicitly **masking out** ambiguous medium-confidence voxels so unannotated nuclei never receive a "background" gradient. This is the "uncertainty-masked loss" fallback; weaker theory, but numerically bulletproof and impossible to get catastrophically wrong.
3. **[I]** A1 is higher-ceiling but has the π failure mode; A2 is lower-variance. Ship A2 as the safe baseline on day 1, race A1 against it on the tracking metric. Keep whichever wins OOF on the *held-out embryo split* (train on 44b6, validate on 6bba, and vice-versa) — this is the only split that mimics the disjoint-embryo metric.

**Stage B — Dense-external-pretrain → sparse-finetune (biggest expected recall win).**
- **[V]** StarDist-3D handles anisotropic voxels and densely-packed nuclei and beats watershed/U-Net baselines (Weigert et al., WACV 2020, arxiv.org/abs/1908.03636). **[V]** The organiser ecosystem itself (Ultrack/Zebrahub, Lange/Royer) produced *dense* whole-embryo zebrafish nuclear tracks (pmc.ncbi.nlm.nih.gov/articles/PMC11398427). 
- **[I]** Pretrain the detector on a densely-labelled external nuclear dataset (StarDist-3D demo nuclei, or any public densely-segmented embryo volume), then finetune on the sparse P set. Dense pretraining teaches "what a nucleus looks like" from complete labels, so the sparse finetune only has to *domain-adapt*, not *discover the object class from 1% of examples*. This directly attacks the "unannotated=background" problem by importing a prior that unannotated blobs are still nuclei.

**Stage C — Teacher–student temporal self-training (converts unlabelled voxels into supervision).**
- **[V]** Mean-Teacher = EMA teacher, consistency/pseudo-label loss to student; temporal self-training exploits that adjacent video frames are highly correlated to densify pseudo-labels (Tarvainen & Valpola; and cell/video variants, e.g. arxiv.org/pdf/2412.07072). **[V]** Background-propagation self-training already boosts sparse nuclei detection and suppresses FPs (Qu 2020).
- **[I]** Concrete recipe: (1) train Stage-A model; (2) run it on *all frames* to get high-confidence peaks; (3) **temporally verify** each pseudo-nucleus — keep it only if it is linkable to a detection in t−1 and t+1 (a nucleus that appears for a single frame and vanishes is almost certainly a false positive; a persistent track is almost certainly real); (4) add temporally-verified peaks to P and retrain. Temporal persistence is a *free, physics-grounded* pseudo-label filter that background-propagation alone doesn't have.

**Stage D — Contrastive/representation pretrain (optional, lowest priority).**
- **[C]** Contrastive representation learning from PU data is theoretically motivated (arxiv.org/pdf/2402.06038) but I found no cell-tracking result that makes it clearly worth the Kaggle compute budget. **[I]** Deprioritise unless Stages A–C plateau.

### 1.4 What NOT to do
- **[I]** Do not use plain Dice/BCE over the full volume. Do not use focal loss "to handle imbalance" — it does not fix the *label-noise* problem (unannotated nuclei labelled 0); it just reweights a wrong target.
- **[I]** Do not trust a validation number computed on the *same* embryo the model trained on; the metric is cross-embryo, so validate cross-embryo.

---

## 2. SUB-FRONT 2 — Developmental-biology priors → constraints/features/regularisers

Legend for TRANSFER: **T** = transfers across embryos (safe to bake in as a hard prior for the hidden embryo); **S** = stage-specific / embryo-specific (use only as a soft feature or test-time-fit parameter, never a hard constraint).

| # | Biology prior (evidence) | Translate to | Concrete mechanism | Transfer |
|---|---|---|---|---|
| 1 | **Cell-cycle length lengthens through development**: cleavage cycles ~15 min, cycles 13–16 average 54/78/151/240 min during gastrulation **[V]** (journals.biologists.com Dev 135:2065; Kimmel staging via zeclinics). | Prior on **inter-division interval** per track. | In the linker's division model, penalise two divisions on the same lineage closer than a stage-dependent floor (≥~45–60 min at gastrula). Reject/down-weight candidate division edges that violate it. | **T** as a *floor* (lower bound is biologically hard); the exact value is **S** (fit to observed division intervals in the two training embryos, transfer the floor not the mean). |
| 2 | **MBT at cycle 10 (~3 hpf): divisions become asynchronous & metasynchronous from cycle 8** **[V]** (PMC3131289; Kimmel). | Prior on **division synchrony vs. stage**. | Do NOT assume a global division wave after gastrulation; treat divisions as spatially local events. Early (pre-MBT) frames, if present, expect near-synchronous waves. | Synchrony structure is **S** (stage-dependent); the *rule* "later = more asynchronous" is **T**. |
| 3 | **Division rate varies in space**; proliferation is patterned 80%-epiboly→bud, oriented divisions (AP axis, contrary to long axis, non-canonical Wnt) **[V/C]** (Mendieta-Serrano 2013 Anat Rec; Concha & Adams 1998 PMID 9463345; Gong strain maps PMC8481280). | **Spatial prior on division probability** + **division-orientation prior**. | Feature per candidate: local region → learned/empirical division-rate multiplier. Daughter pair should straddle a plane roughly aligned to AP / tissue surface; score division candidates by daughter-axis orientation. | Division-rate *map* is **S** (depends on embryo registration/stage). Orientation tendency (divisions in tissue-surface plane, daughters separate along a consistent axis) is **T** as a soft prior. |
| 4 | **Interkinetic nuclear migration (IKNM)**: in neuroepithelium nuclei oscillate apico-basally; M-phase always apical, G1/S/G2 basal; motion is diffusive/stochastic (persistent random walk) **[V]** (Leung 2011 Dev Cell PMID 25600237; Azizi/Norden eLife 58635, arxiv 1903.05414). | **Motion model** for a subpopulation + **division-location prior**. | Where neuroepithelium is present, expect large *coherent apical-ward* displacement immediately before division → use "moving toward a tissue surface" as a pre-division feature. Model nuclear motion as persistent random walk (not constant velocity) for these cells. | IKNM presence is **S** (tissue- and stage-specific; may be absent at epiboly). The *diffusive/persistent-random-walk motion prior* is **T** and is a good default motion model generally. |
| 5 | **Convergence–extension & epiboly produce coherent, low-divergence flow fields**: epiboly=posterior translocation, convergence=lateral, extension=anterior; strain maps are smooth **[V/C]** (Gong PMC8481280; time-lapse to 3-somite). | **Global motion regularizer** / optical-flow prior. | Fit a smooth per-region velocity field per frame-pair; bias linking toward the local flow vector; penalise links whose displacement deviates strongly from the local coherent field. | **T** — coherent, spatially-smooth tissue flow during gastrulation is a robust, embryo-general property. Strong bet for the linker. |
| 6 | **Neighboring cells share velocity (local rigidity/coherence of tissue flow)** — corollary of #5, motion is locally correlated. | **Local-motion smoothness regularizer** (neighbours move alike). | In linking cost, add a term: a candidate link's velocity should match the median velocity of already-linked neighbours within radius r. | **T** — strongest single transferable motion prior; tissue moves as a sheet. |
| 7 | **Clone/lineage spatial coherence ("clonal strings")** — sister/related cells stay spatially clustered **[V]** (Kimmel & Warga 1994, "Cell cycles and clonal strings", journals.biologists.com/dev/120/2/265). | **Division/lineage regularizer**. | After a division, the two daughters should remain mutually near for several frames; use daughter proximity + shared subsequent motion to validate division events; penalise "divisions" whose daughters immediately diverge far. | **T** — physical constraint of cytokinesis; daughters start adjacent everywhere. |
| 8 | **Density ↔ division-timing relations** (crowding modulates IKNM/nonlinear diffusion; proliferation patterned with density) **[C]** (Azizi eLife 58635; Mendieta-Serrano 2013). | **Soft feature**, not a constraint. | Local nuclear density as a covariate for division-probability model. | **S** — relationship is stage/tissue-specific and noisy; use as a weak learned feature only. |

**Cross-cutting verdict [I]:** the *transferable* priors are **motion-coherence (#5,#6)**, **daughter-proximity after division (#7)**, **a cell-cycle inter-division floor (#1)**, and **diffusive/persistent motion default (#4-motion)**. These are the ones safe to hard-code for the hidden embryo. The *stage/space-specific* quantities (division-rate maps, synchrony, absolute cycle length, IKNM presence) must be **fit at test time** on the hidden embryo's own statistics or left out — hard-coding them from 44b6/6bba risks overfitting to the wrong stage.

---

## 3. SUB-FRONT 3 — Imaging physics (DaXi/light-sheet) as inverse problem

### 3.1 Verified instrument facts
- **[V]** DaXi resolution: **~450 nm lateral, ~2 µm axial** over 3000×800×300 µm (Yang, Lange, Millett-Sikking… Royer, Nat Methods 19:461, 2022). ⇒ **axial PSF ≈ 4.4× coarser than lateral** — the "z ~4× coarser" premise is confirmed. Anisotropy is the dominant localisation challenge and dictates using **anisotropy-aware** models (StarDist-3D **[V]**).
- **[V]** It is oblique-plane single-objective light-sheet ⇒ depth-dependent illumination, and (general light-sheet **[C]**) attenuation/scattering with depth, photobleaching over time.

### 3.2 Opportunity list, ranked by expected value (EV = P(helps) × magnitude, offline-Kaggle-feasible, no hallucinated cells)

| Rank | Physics-aware step | Why it helps recall/localisation without inventing cells | First falsification experiment | Citation |
|---|---|---|---|---|
| **1** | **Anisotropy-aware detection** (train detector with true z:xy voxel ratio; z-elongated blobs; StarDist-3D anisotropic mode). | Matches the model's spatial prior to the PSF so axially-overlapping nuclei are separated by *shape*, not resolution. Pure representation choice — cannot hallucinate; it only re-parameterises real detections. | Ablate: isotropic vs anisotropic detector on cross-embryo split; measure recall on annotated nuclei + division-Jaccard. If anisotropic ≤ isotropic, drop. | **[V]** Weigert WACV 2020 (arxiv 1908.03636); DaXi Nat Methods 2022. |
| **2** | **Temporal denoising / self-supervised restoration (Noise2Void / Noise2Noise / CARE) as a preprocessing step** to lift shot-noise-limited low-SNR nuclei above detection threshold. | Shot noise is the recall killer in dim/late frames; N2V/N2N need *no clean ground truth* (crucial: none exists here) and provably denoise from noisy data alone → dim real nuclei become detectable. Denoising a real signal ≠ inventing one, IF you validate recall doesn't come with FP inflation. | Denoise → detect; compare recall AND false-positive rate vs raw. Falsify if FP rate rises faster than recall (denoiser is smoothing noise into fake blobs). Use temporal-persistence filter (Stage C) as the FP guard. | **[V]** N2V (arxiv 1906.00651); CARE (Weigert Nat Methods 2018); N2N; blind zero-shot denoiser (Nat Mach Intell 2022, PMC9674521). |
| **3** | **Frame-pair regime classification** (detect irregular Δt, frame freezes, sudden specimen jumps before linking). | The linker's motion model assumes ~constant Δt and smooth motion; a frozen/duplicated frame or a big rigid jump breaks it and manufactures wrong edges. Classifying each frame-pair (normal / frozen / large-motion / large-Δt) and switching linking cost accordingly prevents *spurious* links — improves precision, invents nothing. | Compute per-pair global displacement + image-diff; flag outliers; on flagged pairs relax/re-register before linking; measure adj-edge Jaccard with vs without regime switch. Falsify if flagged pairs are <1% and metric unchanged. | **[I]** (physics reasoning); global-motion registration is standard. |
| **4** | **Depth-attenuation / photobleaching normalisation** (per-z and per-t intensity normalisation, e.g. rolling background / percentile per depth per frame) before detection. | Deep and late nuclei are dimmer purely from attenuation/bleaching, not biology; a global threshold under-detects them. Normalising restores a *stationary* detection threshold → recovers real deep/late nuclei. Monotone intensity remap can't create structure. | Fit intensity vs z and vs t on training embryos; apply normalisation; measure recall stratified by depth and time. Falsify if depth/time recall is already flat (no attenuation) or if normalisation amplifies background into detections. | **[V]** attenuation/bleaching are established light-sheet artifacts (DaXi & general LSFM); **[I]** normalisation recipe. |
| **5** | **Axial-overlap deblending via mild deconvolution** (Richardson–Lucy with measured/approx PSF, *few* iterations) to sharpen axially-merged nuclei before detection. | Can split two nuclei merged along z into two peaks. BUT deconvolution is the **highest hallucination risk** here — over-iteration creates ringing/ghost peaks that look like cells. | Run 3–10 RL iterations; count detected nuclei vs raw and vs annotation density; falsify (and abandon) the moment detected-count inflates beyond plausible density or temporal-persistence filter rejects a rising share of new peaks. | **[C]** RL deconvolution standard; **[I]** risk assessment. Deprioritise vs #2 which is safer. |

**EV ordering rationale [I]:** #1 is free and pure-upside (parameterisation). #2 is the biggest *recall* lever and, critically, needs no ground truth — but must be paired with the temporal-persistence FP guard. #3–#4 are precision/robustness plays that protect the linker. #5 is last because it is the one step that can genuinely fabricate cells; only pursue it if #1–#2 plateau and always gated by temporal persistence.

**Non-negotiable safety rail [I]:** every physics step that could raise recall (#2, #4, #5) must be scored on **both** recall and false-positive rate, and every new detection it produces must survive the **temporal-persistence filter** (appears in ≥3 consecutive linkable frames). This is what separates "recovering a real dim nucleus" from "hallucinating a cell." It ties Sub-front 3 back to Sub-front 1 Stage C.

---

## 4. Validation protocol (applies to all three sub-fronts) [I]

The metric is cross-embryo. Therefore the ONLY trustworthy offline signal is a **leave-one-embryo-out** evaluation:
- Fold 1: train on 44b6, evaluate tracking metric on 6bba.
- Fold 2: train on 6bba, evaluate on 44b6.
- A change ships only if it improves the *mean* of the two folds. Same-embryo validation is forbidden — it rewards memorising this embryo's 1% and predicts nothing about the hidden embryo. All hyperparameters that are hard to set from theory (π in nnPU, denoiser strength, cycle-length floor, flow-smoothness weight) are selected on this split.

---

## 5. THE ONE BET

**Build the detector as: dense-external-pretrain → sparse nnPU/masked-finetune → self-supervised temporal denoising upstream → temporal-persistence pseudo-label self-training, and hand its persistent detections to a motion-coherence + daughter-proximity linker; select every knob on leave-one-embryo-out.**

Rationale [I]: the competition is lost or won on the *positive-only, 1%-labelled* problem, and the single highest-leverage move is to stop treating unannotated nuclei as background. Three independent mechanisms attack that: (a) dense pretraining imports the "unannotated blob is still a nucleus" prior **[V-grounded]**; (b) nnPU/masked loss removes the false-background gradient **[V]**; (c) temporal-persistence self-training converts unlabelled-but-real nuclei into supervision using a *physics-true* filter **[V/I]**. Denoising **[V]** feeds recall in; the persistence filter keeps hallucinations out. This is fully offline, T4×2-feasible, and — because its core priors (motion coherence, daughter proximity, anisotropy) are the *transferable* ones — it is built to generalise to the disjoint hidden embryo rather than memorise 44b6/6bba.

---

## 6. Twelve-line summary

1. The whole lane reduces to one fact: labels are positive-only, ~1% complete — any "no annotation ⇒ background" loss trains the model to suppress ~99% of real nuclei.
2. Loss stack: (A) masked positive loss + nnPU over a candidate set [V: Kiryo 2017], with A2 = uncertainty-masked BCE as the bulletproof fallback.
3. (B) Dense-external-pretrain (StarDist-3D, anisotropy-aware) → sparse-finetune imports "unannotated blob = still a nucleus" — biggest recall win [V].
4. (C) Teacher–student temporal self-training, gated by temporal persistence (≥3 linkable frames), turns unlabelled-but-real nuclei into supervision [V/I].
5. Class prior π is the #1 nnPU failure mode — never fix it by theory; sweep it on the metric.
6. PU-for-incomplete-cell-detection is already published (MICCAI 2021) — our exact analogue [V].
7. Transferable bio priors (hard-code for hidden embryo): motion coherence / neighbour-shared velocity, daughter-proximity after division, a cell-cycle inter-division floor, diffusive default motion [V].
8. Stage/space-specific (fit at test time, never hard-code): absolute cycle length, division-rate maps, synchrony, IKNM presence [V].
9. DaXi PSF is ~4.4× coarser in z (450 nm vs 2 µm) [V] → anisotropy-aware detection is free upside, rank-1 physics step.
10. Rank-2 physics: self-supervised temporal denoising (N2V/N2N/CARE, no clean GT needed) [V] to recover dim/late nuclei — paired with the persistence FP-guard.
11. Deconvolution (rank-5) is the only step that can fabricate cells — gate hard on temporal persistence or skip.
12. THE BET: dense-pretrain → nnPU/masked-finetune → temporal denoise → persistence self-training → motion-coherence+daughter-proximity linker, all selected on leave-one-embryo-out (the only split that mimics the disjoint-embryo metric).

---

### Sources
- Kiryo et al., Positive-Unlabeled Learning with Non-Negative Risk Estimator, NeurIPS 2017 — https://arxiv.org/pdf/1703.00593 ; code https://github.com/kiryor/nnPUlearning
- PU Learning for Cell Detection in Histopathology with Incomplete Annotations, MICCAI 2021 — https://link.springer.com/chapter/10.1007/978-3-030-87237-3_49
- Qu et al., Weakly Supervised Deep Nuclei Segmentation Using Partial Points Annotation, 2020 — https://arxiv.org/abs/2007.05448
- Weigert et al., StarDist-3D (Star-convex Polyhedra), WACV 2020 — https://arxiv.org/abs/1908.03636
- Trackastra (transformer linker), ECCV 2024 — https://arxiv.org/abs/2405.15700 ; https://github.com/weigertlab/trackastra
- Ultrack, 2024 — https://pmc.ncbi.nlm.nih.gov/articles/PMC11398427 ; https://github.com/royerlab/ultrack
- Yang, Lange, Millett-Sikking… Royer, DaXi, Nat Methods 19:461, 2022 — https://www.nature.com/articles/s41592-022-01417-2 ; https://github.com/royerlab/daxi
- Krull et al., Noise2Void, 2019 — https://arxiv.org/pdf/1906.00651 (+ CARE Weigert 2018; blind zero-shot denoiser, Nat Mach Intell 2022, https://pmc.ncbi.nlm.nih.gov/articles/PMC9674521)
- Contrastive learning from PU data, 2024 — https://arxiv.org/pdf/2402.06038
- Cell-cycle length / MBT: Dev 135:2065 https://journals.biologists.com/dev/article/135/12/2065 ; PMC3131289
- Kimmel & Warga, cell cycles & clonal strings, Dev 120:265 — https://journals.biologists.com/dev/article/120/2/265
- Mendieta-Serrano, Cell Proliferation Patterns in Early Zebrafish, Anat Rec 2013 — https://anatomypubs.onlinelibrary.wiley.com/doi/10.1002/ar.22692
- Oriented cell divisions in zebrafish gastrula, PMID 9463345 ; strain maps PMC8481280
- IKNM: Leung et al. Dev Cell 2011 PMID 25600237 ; Azizi et al. eLife 58635 / arxiv 1903.05414
- Mean-Teacher / temporal self-training in video detection — https://arxiv.org/pdf/2412.07072
