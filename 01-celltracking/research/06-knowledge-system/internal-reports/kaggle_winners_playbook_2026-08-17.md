# Kaggle winners' playbook — what actually broke analogous plateaus (agent report) — 2026-08-17

> **EDITOR'S NOTE (added by the host session, same day, after this report was written).**
> This report's "hard gate on levers 2, 3, 9" (§ near line 440/469) assumes Zebrahub imaging is
> not on disk. **That premise is now superseded.** `kkunizaw/biohub-zh001r` was downloaded and
> audited the same day: `(72, 20, 64, 64, 64) uint8` real ZSNS001 imaging, labels aligned to the
> imaging (2.29× intensity contrast), geometry measured at **1.677 µm/voxel (1.032× our deployed
> 1.625 µm grid)** — the same geometry family as our detector input. See
> `../experimental-records.md` (2026-08-17 entries) and `scripts/win_bet/h1r_zh001r_audit.py`.
> **Caveat that partially restores the gate:** those packaged nodes are `(N,4)=[t,z,y,x]` with
> **no track identity**, so they supervise the DETECTOR half only. Levers needing *association*
> supervision remain gated on our own `h1r_fetch_imaging.py` level-1 stream, which carries track ids.
> The rest of this report is unmodified raw agent output.

Mandate: find **winning solution write-ups** from structurally analogous Kaggle competitions, extract what
separated winners from the plateau (not "ensemble + TTA"), and map each pattern onto our situation.

Does not duplicate `competitive_frontier_2026-08-16`, `competitive_refresh_2026-08-17`,
`methods_frontier_2026-08-16`, `novel_crossdomain_2026-08-17`, `quickwins_internal_2026-08-17`,
`redteam_blindspots_2026-08-17`. Where this report touches a lever those cover, it is marked
**CONFIRMATORY** or **CORRECTIVE** and says which of our prior conclusions it supports or contradicts.

---

## Source-access limitation (read this before trusting any citation)

**Kaggle discussion and `/writeups/` pages are client-rendered.** Direct fetch returns a ~5.9 KB SPA shell
with zero solution text (verified: `curl` on the HuBMAP-vasculature 1st-place write-up → HTTP 200,
5907 bytes, no "RTMDet" token in body). The authenticated Kaggle CLI exposes **kernels but not
discussions**. Consequently:

- **DIRECTLY FETCHED AND READ IN FULL** (highest confidence): CryoET 1st-place `WRITEUP.md`, SIIM-ACR
  1st-place README, HuBMAP-vasculature 3rd-place README, NeurIPS CellSeg Nature Methods analysis,
  HPA Nature Methods analysis, HuBMAP+HPA Nature Methods analysis, TrackML arXiv paper, CZII organizer
  abstract, POPSICLE CryoET critique, ML Contests 2022/2024 reports, Vesuvius write-up index.
- **SEARCH-ENGINE EXTRACTION ONLY** (medium confidence — the quoted phrases are attributed to the named
  page by the search index, but I did not read the page myself): tascj HuBMAP-vasculature 1st, Tom
  HuBMAP-kidney 1st, nvnn NFL 1st (README read, write-up not), junkoda contrails 1st, DSB2018 #54741,
  Sartorius #297988. Every such claim below is tagged **[extracted, not read]**.
- **COULD NOT VERIFY AT ALL** (bioRxiv served HTTP 429/403 on five attempts): the CZII lessons-learned
  preprint (biorxiv 2025.11.03.686153) and the DSB2018 analysis preprint (biorxiv 580605). The Nature
  Methods DSB2018 paper is paywalled/captcha'd (`pmc.ncbi.nlm.nih.gov/articles/PMC6919559` → reCAPTCHA;
  `nature.com` → IdP redirect).

---

## Ranked summary — actionable levers for us, by (expected value × cheapness)

| # | Lever | Documented origin | GPU? | One-line falsification test | Status vs our prior reports |
|---|---|---|---|---|---|
| 1 | **Calibrate the accept-threshold as a *quantile of the expected count*, not an absolute score** | Vesuvius 2nd (percentile 0.93); CryoET 1st (per-class OOF F-beta threshold curves) | **No** | If the LOEO-optimal quantile-of-`N_est` operating point coincides with our current absolute-threshold point on **both** embryo directions, the lever is empty | **NEW framing** of an old knob (`quickwins` node-budget). Robustness-under-shift, not more pruning |
| 2 | **Two-stage: pretrain on abundant NOISY labels, then fine-tune on scarce CLEAN labels — different LR and aug schedules per stage** | HuBMAP-vasculature 3rd: "**2-3% boost on validation and 4-6% improvement in leaderboard scores**"; Sartorius 1st (LIVECell→comp); NeurIPS CellSeg T1 (4 external corpora) | **Yes** | Zebrahub-pretrain → 2-embryo-finetune must beat 2-embryo-only on LOEO adj_edge_J in **both** directions; if either direction regresses, kill | **Sharpens H1.** Our reports say "retrain on Zebrahub"; the *documented* mechanism is the **staging**, not joint training |
| 3 | **Make the detector's training-time GT-assignment rule match the metric's matching rule** (radius-aware point assignment, not voxel-wise loss) | CryoET 1st: point-point "IoU" = `exp(-mse(x,y)/(2·radius²))` for GT↔prediction assignment | **Yes** | Same backbone + same data: a 7 µm-radius-matched anchor-free point head must beat the voxel-wise head on LOEO node-F1 at 7 µm | **NEW.** Not in any prior report. Nearest-domain evidence available (3D bio volumes, distance-matched metric) |
| 4 | **Cheap pre-filter → expensive model → GBDT post-stage cascade on candidate pairs** | NFL Player Contact 1st (nvnn): XGB drops easy negatives → 3D CSN on pair crops → **XGB post-processing** | **No** (for the GBDT stages) | A GBDT trained on embryo A's existing edge features must lift LOEO adj_edge_J on B above the frozen linker at matched N_pred | **CONFIRMATORY + upgrade** of `novel_crossdomain` #2 (MetaDetect). Kaggle-winning instance beats the LiDAR-paper evidence |
| 5 | **Architecturally *diverse* independent models, not seeds of one** | CryoET 1st (SegResNet + DynUNet, "essential"); DSB2018 top-3 each used a *different* family (U-Net ens. / FPN / Mask R-CNN); ML Contests: >½ of 2022 winners ensembled | **Yes** | Two different backbones trained on the same data must beat two seeds of one backbone by >2× the seed-to-seed spread on LOEO | **CONFIRMATORY** of `redteam` Claim 3's "under-scoped multi-seed ensembling" — and corrects it: *diversity of architecture*, not seed count |
| 6 | **Learned sub-voxel offset regression as part of the head (not post-hoc interpolation)** | CryoET 1st: stride-2 class maps **+ offsets to object centres** | **Yes** (learned) / **No** (post-hoc) | If our post-hoc parabolic refine already lands centroids <1.5 µm RMS, the learned version cannot pay | **CONFIRMATORY** of `competitive_refresh` CW1, plus a learned upgrade path |
| 7 | **Domain normalization as a standalone lever** | HuBMAP+HPA: **4th place (0.827) used heavy stain normalization *alone*, no external data, no pseudo-labels**, vs winner 0.835 | **No** (preprocessing) / **Yes** (to revalidate) | Per-embryo intensity/contrast standardization must shrink the 44b6↔6bba LOEO node_recall gap (0.985 vs 0.855) | **NEW.** Our reports treat the cross-family gap as a model problem; Kaggle says try normalization first |
| 8 | **Trust LOEO CV, ignore public LB** | Great Barrier Reef winner was **#121 on the public LB**; CryoET 1st lists 8 techniques that raised CV but *lowered* LB | **No** | n/a — this is a decision rule, not an experiment | **CONFIRMATORY** of CLAUDE.md rules 5 and 8 |
| 9 | **DOWN-RANK pseudo-labelling** | NeurIPS CellSeg: "**none of the employed methods demonstrated a notable enhancement**"; HuBMAP-vasc 3rd: "I didn't find any boost"; *contra* HuBMAP-kidney 1st where it did help | **Yes** | Pseudo-labels must lift LOEO in the direction they were *not* mined from | **CORRECTIVE.** `novel_crossdomain` #6 ranks cycle-consistency pseudo-labels highly; the cell-segmentation Kaggle record says this is the least reliable plateau-breaker |
| 10 | **For pure association, structured combinatorics + ambiguity resolution beat learned linkers** | TrackML: 1st = classical seed→extend→consolidate→**resolve by "least polluting hits"**; the deep-learning prize (LSTM) placed **12th** | **No** | A global "least-polluting-track" resolution pass over our existing candidate graph must beat greedy/ILP on LOEO adj_edge_J | **CORRECTIVE counterweight** to the whole "learn the linker" thesis in `methods_frontier`/`novel_crossdomain` |

Levers 1, 4, 7, 10 are **CPU-only and testable on cached OOF graphs**. Levers 2, 3, 5, 6 require the
Zebrahub retrain lane and should be arms of one bet, not four bets.

---

## Detail by competition

### A. CZII — CryoET Object Identification (Kaggle, Nov 2024 – Feb 2025) — *the closest analogue found*

<https://www.kaggle.com/competitions/czii-cryo-et-object-identification>

Why it is the closest: 3D volumetric **point detection** in noisy biological volumes, **distance-based
one-to-one matching to ground-truth centres**, tiny labelled training set (7 tomograms) evaluated against
hundreds of unseen ones, organized by a Chan Zuckerberg institute. Our competition is this plus linking.

**Metric (DOCUMENTED).** F-beta with β=4 — recall-weighted, so it penalizes misses far more than
false positives. *This is the opposite tilt from ours*: our node-count multiplier penalizes
over-prediction. Mechanisms transfer; the **direction of the operating-point sweep does not**.
Sources: competition page; TopCUP model card, <https://virtualcellmodels.cziscience.com/model/topcup>.

**1st place = Christof Henkel ("Dieter") + Eugene Khvedchenya ("BloodAxe").**
Repos: <https://github.com/ChristofHenkel/kaggle-cryoet-1st-place-segmentation>,
<https://github.com/BloodAxe/Kaggle-2024-CryoET> (`WRITEUP.md`, read in full).

DOCUMENTED specifics from `WRITEUP.md`:
- Ensemble of **10 object-detection models**: 5 SegResNet-based + 5 DynUNet-based (MONAI 3D), *plus*
  Henkel's separate segmentation ensemble (2×ResNet34, 2×ResNet34-downsampled, 2×EfficientNetB3).
- **Anchor-free point detection, YOLO-style**: class probability maps + **offsets to object centres** at
  **stride 2** (`[B,C,D/2,H/2,W/2]`). Stride-2 gave "massive impact on throughput" at no accuracy cost.
- **Point-point IoU surrogate**: `exp(-mse(x,y) / (2·radius²))` used in place of box IoU for GT↔prediction
  assignment and for NMS. Loss = varifocal (classification) + IoU-based distance regression, PP-YOLO-style
  assignment with top-K predictions per GT.
- 5-fold CV (2 studies validation / 5 training per fold); **validation extended with flipped and rotated
  volumes specifically to reduce CV noise**.
- Post-processing chain: CenterNet-style NMS → top-16K per class → confidence threshold → greedy NMS on
  pairwise point-IoU → unit conversion.
- **Per-class thresholds selected from OOF F-beta curves** to maximize mean F-beta(4).
- **Explicitly did not work** (raised CV, lowered LB): mixup, copy-paste augmentation, random erasing,
  **2.5D models**, 3D HRNet / ConvNeXt variants, Gaussian noise, **anisotropic scaling**, knowledge
  distillation.

**Organizer post-mortem (DOCUMENTED).** "Lessons Learned from CZII's Kaggle CryoET Object Identification
Challenge", *Microscopy and Microanalysis* 31(S1), ozaf048.496,
<https://academic.oup.com/mam/article/31/Supplement_1/ozaf048.496/8212398>: 1,135 contestants; the field
"has not coalesced behind a single approach", which the organizers read as evidence "that these algorithms
do not generalize sufficiently well to serve the community's diverse needs".

**Independent critique (DOCUMENTED).** POPSICLE, arXiv:2606.10255,
<https://arxiv.org/html/2606.10255v1>: top solutions are "highly specialized pipelines combining custom
inference heuristics, and dataset-specific tuning (e.g., thresholding, clustering, or class-wise
post-processing) that are tightly coupled to the Phantom dataset"; the authors **exclude** them from their
benchmark and treat them as "the upper bound achievable with extensive task-specific optimization".

**MY INFERENCE (not documented):** the CryoET winner's edge was not a novel architecture — it was
(i) making the *training-time assignment* mirror the *scoring-time matching*, (ii) sweeping the operating
point on OOF curves of the *actual metric*, and (iii) architectural diversity in the ensemble. All three are
available to us; (i) is the one we have never tried.

---

### B. NeurIPS 2022 Multi-modality Cell Segmentation Challenge — *the strongest negative result*

Nature Methods 21, 1103–1113 (2024), <https://www.nature.com/articles/s41592-024-02233-6>;
full text read at <https://arxiv.org/html/2308.05864v2>.

Metric: **F1 with IoU≥0.5 one-to-one matching**, boundary cells excluded — same matched-detection family
as our adjusted edge Jaccard.

| Rank | Team | What they did (DOCUMENTED) | Median F1 |
|---|---|---|---|
| 1 | T1-osilab (Lee et al.) | SegFormer encoder + MA-Net decoder (Mish); cell-probability + **gradient-flow regression**; BCE+MSE; **pretrained on external TissueNet, Omnipose, Cellpose, LiveCell**; cell-wise intensity perturbation, boundary exclusion, minority-modality oversampling; gradient tracking + small-cell exclusion + TTA | 89.7% |
| 2 | T2-sribdmed (Lou et al.) | **Unsupervised clustering of images into 4 groups → group-specific models**; distance+semantic maps for round cells, gradient maps for irregular; NMS + watershed | 84.5% |
| 3 | T3-cells (Upschulte et al.) | ResNeXt-101 + Contour Proposal Network, 4 heads incl. **uncertainty**; uncertainty-aware NMS | 84.4% |

**The load-bearing finding for us (DOCUMENTED):** all three top teams tried semi-supervised use of the
unlabelled pool — consistency regularization, reconstruction heads, pseudo-labels, uncertainty-aware
Listen2Student — and *"Despite these joint efforts, none of the employed methods demonstrated a notable
enhancement in segmentation performance."*

Second finding (DOCUMENTED): Cellpose/Omnipose **retrained from scratch** showed "substantial decline" on
unseen cell types, while the top three "maintained their superiority" across four held-out modalities. The
generalization came from **breadth of pretraining data**, not from architecture.

**CORRECTIVE for us.** `novel_crossdomain` #6 (cycle-consistency pseudo-labels) and #3 (PU learning on the
97% unannotated) are ranked highly there. The single best-controlled cell-segmentation experiment on
record says pseudo-labelling the unlabelled pool is the *weakest* of the plateau-breakers in this exact
domain. It should be demoted below external-data pretraining and operating-point calibration.

---

### C. HuBMAP — Hacking the Human Vasculature (Kaggle 2023) — *the clearest "which part of the metric" lesson*

<https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature>

**1st place, tascj** [extracted, not read] — write-up at
<https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/writeups/tascj-1st-place-solution>,
code <https://github.com/tascj/kaggle-hubmap-hacking-the-human-vasculature>. RTMDet-x, input 768. Stated
insight: for instance segmentation "AP mainly relies on bounding box prediction, while mask prediction
precision has a minor impact", so effort went into **bbox accuracy**; the mask head was added *to improve
the bbox*, and mask information was used only indirectly (random rotation, then recomputing bboxes).

**3rd place, Nischay Dhankhar** (DOCUMENTED — README read in full at
<https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution>):
- **Stage 1**: COCO-pretrained models further trained on the **noisy dataset-2 annotations**, ~10 epochs,
  aggressive LR (0.02+), *light* augmentation, cosine schedule with min-LR ~0.01.
- **Stage 2**: fine-tune on the **clean dataset-1** annotations, 15–25 epochs, *heavy* augmentation, low LR.
- Quoted: *"Multi stage approach gave around consistent 2-3% boost on validation and 4-6% improvement in
  leaderboard scores which is quite huge."*
- **Pseudo-labels on the unannotated dataset-3**: *"although I didn't find any boost using them in the
  leaderboard scores, I will still talk about it as they were used in final solution."*
- Single-model public/private: ViT-Adapter 0.600/0.589; CBNetV2 0.567; DetectoRS-X101 0.573; DetectoRS-R50
  0.558.

**Mapping.** The noisy/clean two-stage split is the single best-quantified plateau-break in this whole
survey (+4–6% LB), and our data situation is an exact structural match: Zebrahub `*_tracks.csv` are
**Ultrack-derived, i.e. noisy, abundant**; the 2 embryos' 2.8% annotations are **clean, scarce**. The
documented recipe is *not* "mix them" — it is two stages with **inverted augmentation and LR schedules**.

---

### D. HuBMAP + HPA — Hacking the Human Body (Kaggle 2022) — *domain shift solved by normalization alone*

Nature Methods analysis, <https://pmc.ncbi.nlm.nih.gov/articles/PMC9881902/> (read in full).

Setup: train on HPA, test on **HuBMAP** — different pixel size, tissue thickness and staining protocol.
Structurally our "train on 2 embryos, test on a hidden one".

DOCUMENTED: 1st 0.835 Dice (264 submissions), 2nd 0.833 (100), 3rd 0.832 (255) — top three combined 619
submissions. Winners used ensembles including vision transformers (SegFormer, CoaT), heavy geometric/colour
augmentation, **Vahadane stain normalization**, and pseudo-label loops; team 3 additionally used "pixel size
adaptation and histogram matching". Public and private leaderboard scores "remained similar throughout",
i.e. no shakeup.

**The finding that matters:** *"the 4th place finisher (Dice: 0.827) achieved strong results using 'heavy
stain normalization' alone without external data or pseudo-labeling"* — 99.0% of the winner's score from
one normalization mechanism, with none of the expensive machinery.

**Mapping.** We have a documented 44b6/6bba asymmetry (node_recall 0.985 vs 0.855, `redteam` §Claim 2). We
have never tested whether **per-embryo intensity/contrast/background standardization** closes it. That is
the cheapest thing on this list that attacks the cross-family trap directly, and it is inference-time
applicable (no retrain needed to *measure* it on cached graphs' upstream volumes).

---

### E. HuBMAP — Hacking the Kidney (Kaggle 2021) — *the one place pseudo-labelling did win*

**1st place, "Tom"** [extracted, not read] —
<https://www.kaggle.com/competitions/hubmap-kidney-segmentation/writeups/tom-1st-place-solution>, summarized
at <https://hubmapconsortium.github.io/ccf/pages/kaggle.html>. Generated pseudo-labels for train data,
**public test data**, the hubmap-portal and `dataset_a_dib`; added **external GTEx** images (~140 by the
end), chosen because GTEx is H&E-stained like HuBMAP; "avoiding edge effects consistently boosted CV and
LB, using only the center part of tiles for prediction".

**Why it worked here and not in B/C (MY INFERENCE):** pseudo-labels helped when they were **dense masks in
the same modality** and when the external corpus was chosen for **stain/protocol match**. They failed in
NeurIPS CellSeg (heterogeneous modalities) and HuBMAP-vasculature (sparse instance annotations). Our case is
sparse instance annotations across two families — nearer to the failure cases.

The **"predict only from tile centres, discard borders"** trick is directly portable: our 199 crops are
overlapping, and border detections are exactly where centroid error and duplicate detections concentrate
(`competitive_refresh` CW1/CW2). I found no evidence we have audited this.

---

### F. 2018 Data Science Bowl (Kaggle) — nuclei, and the metric family we live in

<https://www.kaggle.com/c/data-science-bowl-2018>; analysis in Nature Methods 16, 1247–1253 (2019),
<https://www.nature.com/articles/s41592-019-0612-7>.

**Metric (DOCUMENTED, from the competition evaluation page as mirrored at
<https://github.com/williamgrimes/nuclei_segmentation/>):** mean average precision over IoU thresholds
0.50:0.05:0.95; at each threshold TP/FP/FN are counted from one-to-one IoU matching — *every unmatched
prediction is a direct penalty*, so this is the **matched-detection Jaccard family** our adjusted edge
Jaccard belongs to.

**Winner: `[ods.ai] topcoders` — A. Buslaev, V. Durnov, S. Seferbekov**, discussion 54741
[extracted, not read], code <https://github.com/selimsef/dsb2018_topcoders>. Per the Nature Methods
description [extracted, not read]:
- FPN-based, with **two custom output layers producing multichannel relative-position masks encoding each
  nucleus pixel's distance to its boundary in four directions** (vertical, horizontal, 45°, 135°); background
  set to zero.
- Those masks are converted to boundaries, **refined with watershed, and the final non-overlapping mask set
  is chosen by ranking on the consistency between local and global scores**.
- ImageNet-pretrained encoders, **24 augmentation routines** including channel shuffling, colour inversion
  and **object copying**.
- **External data**: additional microscopy images from public databases, including **manually annotated
  Wikimedia images**.
- The top three used **three different architectures** (U-Net ensemble / FPN / Mask R-CNN).

**Stage-2 structure (DOCUMENTED, <https://bbbc.broadinstitute.org/BBBC038>):** the final test set contained
"experimental conditions not present in the first stage" and was deliberately unlabelled to deter manual
annotation — an explicit unseen-domain generalization test, exactly our hidden-embryo setup.

**Mapping.** Two things port. (i) The winning instance representation was a **geometric distance encoding
plus a consistency-ranked selection step** — a *ranker over candidate masks*, which is the shape our own
"re-acceptance must be a ranker" conclusion already has. (ii) The winners built external training data by
**hand-annotating scraped images**; our analogue is Zebrahub, which is legal and already annotated.

---

### G. Sartorius — Cell Instance Segmentation (Kaggle 2021–22)

<https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation>. Metric: MAP over IoU 0.50:0.95,
same family as DSB2018.

The widely-cited 1st-place code release is <https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution>
(README read: setup only). Documented pipeline from that README: **train the detector on the external
LIVECell dataset, then fine-tune on the competition data**, starting from COCO-pretrained **YOLOX-x**;
segmentor is **UperNet Swin-T**. This is again *external-corpus pretraining then in-domain fine-tune*.

**UNVERIFIED:** team-name attribution. One search result names "Takumi Okoshi and Jian Chen" as grand-prize
winners, which does not obviously reconcile with the `tascj` handle. Do not cite a team name from this
report. **UNVERIFIED:** the frequently repeated claim that "3rd place used CellPose" — I saw it only in a
third-party blog (<https://marquis08.github.io/competition/cell/segmentation/Sartorius/>), which itself
links to a Kaggle discussion I could not read.

---

### H. Human Protein Atlas — Weakly Supervised Single-Cell Classification (Kaggle 2021)

Nature Methods 19, 1221–1229 (2022), <https://pmc.ncbi.nlm.nih.gov/articles/PMC9550622/> (read in full).

DOCUMENTED: 1st **bestfitting (Shubin Dai)**, mAP 0.5667 — a "Fair Cell Activation Network" modifying
Puzzle-CAM that "forced the network to pay attention to single cells because the training patches contain a
mix of original crops from full images or masked out single cells". 2nd `[red.ai]` 0.55328 (cell-level +
image-level models, custom losses). 3rd MPWARE & ZFTurbo & Dieter 0.54995 (Puzzle-CAM + image-level + OOF).
4th MILIMED 0.54389 — a **data-centric** approach with manual labels for rare classes.

DOCUMENTED separators: "the key to high performance in this competition was to find approaches that spread
attention evenly to every cell, not only the most discriminative one in the image"; and **top teams made
3.3 submissions/day versus a 0.2/day average**. External data (HPAv20, 82,495 images) was permitted and
used. Only "a very minor shake-up" — attributed to careful proportional class balance across public/private.

**Mapping.** The weak-label pathology here — a model that concentrates on the *easy, discriminative*
instances and ignores the rest — is the same failure shape as our detector producing high recall on obvious
nuclei and a junk pool elsewhere. The documented fix was an **architectural/loss intervention that forces
uniform attention**, not more data and not post-processing. That is a distinct idea from anything in our
reports and is genuinely relevant to a retrained detector's loss design (**GPU: yes**).

---

### I. TrackML Particle Tracking Challenge (Kaggle 2018) — *the only pure-association analogue*

<https://www.kaggle.com/c/trackml-particle-identification>; paper arXiv:1904.06778 (read via ar5iv).

Task shape: connect ~100,000 3D points per event into ~10,000 tracks — candidate-graph association with a
purity-based score, i.e. our linker problem stripped of everything else.

DOCUMENTED, from the paper:
- **1st, "top-quarks"** (Johan Sokrates Wind / *icecuber*, Erling Solberg): **seed generation from hit
  pairs → extension to triplets → track following by helix extrapolation → track consolidation adding
  overlapping-module hits → ambiguity resolution selecting the candidates with the least polluting hits.**
  Mostly C++11; "track following strategy is similar to that of several tracking algorithms currently used
  in production".
- **2nd, "outrunner"**: a neural network predicting the **adjacency matrix over all hit pairs from 27
  constructed features**, then navigating the result with helix-compatibility checks.
- **3rd, Sergey Gorbunov**: classical seeding + helix-fit candidate building + selection with a
  **data-driven magnetic-field estimate**.
- The **jury's deep-learning prize went to "finnies" (LSTM), who placed 12th**.
- The paper's own framing: "the competition was difficult with a dozen front-runners well ahead of a pack",
  with clear statistical separation between top candidates.

**Score discrepancy — FLAGGED.** The challenge results page
(<https://sites.google.com/site/trackmlparticle/results>) reports winning score **0.921** (2nd 0.903, 3rd
0.893); the arXiv paper's ranking figure shows ~**0.96 / 0.945 / 0.941**. I could not resolve which phase or
scoring variant each refers to. Do not quote either number as *the* score.

**CORRECTIVE mapping.** Our `methods_frontier` and `novel_crossdomain` reports lean heavily toward *learned*
association (HOCT, MoTT, D2D, OT, GNNs). The one Kaggle competition that was purely association was won by
**structured combinatorics with an explicit global ambiguity-resolution stage**, and the best learned
pairwise-edge model came **second**. The transferable, cheap piece is the winner's final stage: a global
pass that, among mutually inconsistent track candidates, keeps the ones with the fewest *polluting* hits.
We run greedy/ILP link selection; we have no documented equivalent of "resolve ambiguity by minimizing
pollution across whole tracks". **CPU, testable on cached OOF graphs.**

---

### J. NFL 1st and Future — Player Contact Detection (Kaggle 2023) — *the candidate-pair cascade*

<https://www.kaggle.com/competitions/nfl-player-contact-detection>. 1st place **nvnn (nvnnghia)**;
README read in full at <https://github.com/nvnnghia/nfl3_1st>; write-up
<https://www.kaggle.com/competitions/nfl-player-contact-detection/writeups/nvnn-1st-place-solution>
[extracted, not read].

DOCUMENTED pipeline (from README):
1. Cache helmet boxes + tracking metadata; **"train xgb preprocessing to filter easy negative sample"**,
   separate models for ground-contact and player-contact.
2. Build 3-channel video clips **centred on tracked player pairs**.
3. **3D CSN (ResNet-50 IR-CSN)**, ImageNet-pretrained, 6 checkpoints across configurations.
4. **"Train xgb post processing"** on held-out folds 0–4; separate GBDTs for ground vs player contact;
   ensemble over all CNN checkpoints.

**Mapping — this is our architecture, already validated at 1st place in a Kaggle association task.** Our
pipeline is: candidate graph → transformer edge head → post-processing. The winning shape adds a **cheap
learned pre-filter before the expensive model** (kills the junk candidate pool early, which is our
documented failure mode) and a **learned post-stage that re-ranks the expensive model's outputs together
with geometry** (our "re-acceptance must be a ranker"). Both stages are GBDTs on hand-crafted features:
CPU, minutes to train, testable on cached OOF graphs. This is stronger evidence for
`novel_crossdomain` #2 than the MetaDetect/LMD automotive papers it currently cites.

---

### K. SIIM-ACR Pneumothorax (Kaggle 2019) — *threshold structure replacing a model*

1st place **Aimoldin Anuar (sneddy)**; README read in full at
<https://github.com/sneddy/pneumothorax-segmentation>.

DOCUMENTED: instead of a separate empty/non-empty classifier, the solution used a **triplet
`(top_score_threshold, min_contour_area, bottom_score_threshold)`**: binarize at the high threshold; if the
surviving contour area is below `min_contour_area`, declare the image negative; otherwise **re-binarize the
image at the *lower* threshold**. Best validation triplet `(0.75, 2000, 0.3)`; best public-LB triplet
`(0.7, 600, 0.3)`. Training used a **sliding positive-sample rate 0.8 → 0.4** across the schedule.
Ensemble = top-3 checkpoints averaged per fold per pipeline over AlbuNet / SCSEUnet / ResNet50. Best single
model public 0.8871; private ensembles 0.8679 / 0.8641.

**Mapping.** The mechanism is a **two-threshold hysteresis with a size gate** — commit to an object only on
strong evidence, then *expand* it under weak evidence. Our detector uses a single threshold (0.96875
deployed). A hysteresis analogue for nuclei — high threshold to seed a detection, low threshold to accept
neighbouring peaks belonging to an already-seeded region — is CPU-cheap and has never appeared in our
ledger. Note the *documented* val-vs-LB divergence in the optimal triplet: this is a lever that must be
frozen on LOEO, not tuned on LB.

---

### L. Vesuvius Challenge — Ink Detection (Kaggle 2023) — *percentile thresholding*

Write-up index read at <https://kaggle.curtischong.me/competitions/Vesuvius-Challenge---Ink-Detection>
(third-party mirror of the Kaggle write-ups; content DOCUMENTED there, originals not read).

- **1st**: stochastic weight averaging; **removed easy examples** (blank papyrus layers); BCE+Dice; cosine
  annealing; **hold-out cross-validation with fragment 1 as the validation set**; ensemble of SegFormer and
  U-Net models on crops.
- **2nd (tattaka, mipypf, yukke42)**: reduced resolution to 1/32 to spend capacity on the encoder;
  **percentile thresholding at 0.93**; flip TTA; Swin-Unet decoder; cutout/cutmix/mixup.
- **3rd (traptinblur)**: multiple small-resolution U-Nets; subdivided fragment 2 into extra folds; 24 slices;
  224×224 patches; "inference used smaller cropping strides **without threshold tuning**".

**Mapping — the highest-EV cheap idea in this report.** *Percentile* thresholding fixes the **predicted
positive fraction** rather than an absolute score cut. Our metric multiplies by `1 − 0.1·(N_pred−N_est)/N_est`
and the competition **hands us `N_est`**. That means the natural operating point is not "probability > τ" but
"accept the top `q · N_est` candidates per crop". Under domain shift (2 embryos → hidden embryo) a score
threshold drifts with calibration; a **count quantile does not**. `quickwins`/`redteam` treat node budget as
a post-hoc pruning knob applied after linking; this is the *detector-side* version, and it is the shape the
metric was written in.

---

## Cross-cutting: how a 0.91-style plateau actually gets broken

Counting the documented cases above by mechanism:

| Mechanism | Cases where it demonstrably broke the plateau | Cases where it demonstrably did **not** |
|---|---|---|
| **External-corpus pretraining → in-domain fine-tune** | Sartorius 1st (LIVECell); NeurIPS CellSeg T1 (4 corpora); HuBMAP-vasc 3rd (**+4–6% LB**); DSB2018 winner (scraped + hand-annotated); HuBMAP-kidney 1st (GTEx) | — |
| **Operating-point / threshold structure tuned on OOF for the exact metric** | CryoET 1st (per-class F-beta curves); SIIM-ACR 1st (threshold triplet replaced a whole classifier); Vesuvius 2nd (percentile) | — |
| **Domain normalization** | HuBMAP+HPA 4th reached 0.827 vs winner 0.835 **with normalization alone** | — |
| **Architecturally diverse ensembling** | CryoET 1st ("essential"); DSB2018 top-3 all different families; ML Contests: >½ of 2022 winners | — |
| **Optimizing the metric's true driver, not the salient output** | HuBMAP-vasc 1st (bbox not mask) [extracted] | — |
| **Structured global ambiguity resolution (association)** | TrackML 1st | Learned LSTM association placed 12th |
| **Pseudo-labelling / semi-supervised** | HuBMAP-kidney 1st (dense masks, matched stain) | **NeurIPS CellSeg (all three top teams, explicitly null)**; HuBMAP-vasc 3rd ("didn't find any boost") |
| **Post-processing alone on a shared public model** | *no case found* | our own plateau; the CryoET/POPSICLE critique treats heavy post-proc as dataset-specific overfitting |
| **Metric exploitation** | not observed in any of these; our own competition's `-10000` hub-fork hack was patched 2026-07-20 | — |

**The pattern, stated plainly:** in every biomedical-imaging Kaggle competition surveyed, the plateau was
broken by **training a better model on more or better-matched data**, and the *second* biggest lever was
**calibrating the operating point to the exact metric on trustworthy OOF folds**. Post-processing on a
shared public checkpoint broke a plateau in **zero** of the surveyed competitions. Pseudo-labelling is the
single most over-rated lever relative to its documented hit rate in this domain.

This is independent corroboration of our own `competitive_frontier` H1 thesis, reached from a completely
different evidence base (past competitions rather than this competition's discussions).

**Discipline findings (DOCUMENTED, ML Contests reports,
<https://mlcontests.com/state-of-competitive-machine-learning-2022/> and
<https://mlcontests.com/state-of-machine-learning-competitions-2024/>):** public-LB rank predicted private
rank in >80% of competitions, but the Great Barrier Reef winner was **#121 on the public LB**; >½ of 2022
winning solutions used explicit ensembles; pretrained CV models were "essential"; PyTorch 96% of DL winners;
80%+ of 2024 winners used NVIDIA GPUs, most commonly A100, with top teams on 8×H100 nodes.

---

## Mapping onto our situation — availability, honestly assessed

**AVAILABLE AND UNDER-EXPLOITED**

1. **Quantile/percentile operating point keyed to `N_est`** (Vesuvius 2nd; CryoET 1st). We are handed
   `N_est` by the metric. CPU. Composes with the node-budget work already in `quickwins`.
   *Falsification:* LOEO-optimal `q` must differ from our current absolute-threshold operating point on both
   directions, and must transfer A→B and B→A. **GPU: No.**
2. **Two-stage noisy→clean training with inverted LR/aug schedules** (HuBMAP-vasc 3rd, +4–6% LB). Requires
   Zebrahub **imaging**, which per our own memory is *not* on disk (tracks-only). The Kaggle dataset
   `kkunizaw/biohub-zmnscrops` (3.66 GB npz, noted in `competitive_refresh` §4) is the plausible unblock.
   *Falsification:* must beat 2-embryo-only training on LOEO adj_edge_J in both directions. **GPU: Yes.**
3. **Radius-matched anchor-free point head** (CryoET 1st). Our metric matches at 7 µm one-to-one; our
   detector is trained with a voxel-wise objective whose optimum is not the metric's optimum.
   *Falsification:* same backbone, same data — radius-matched head must beat voxel-wise head on LOEO node-F1
   at 7 µm. **GPU: Yes** (folds into the same retrain lane as #2; not a separate bet).
4. **GBDT pre-filter + GBDT post-ranker cascade** (NFL 1st). Trains in seconds on existing edge features.
   *Falsification:* GBDT trained on embryo A must lift LOEO adj_edge_J on B at matched N_pred. **GPU: No.**
5. **Per-embryo intensity/contrast normalization** (HuBMAP+HPA 4th). *Falsification:* must shrink the
   44b6/6bba node_recall gap (0.985 vs 0.855). **GPU: No** to measure; **Yes** to bank via retrain.
6. **Global ambiguity resolution over whole tracks, "least polluting hits"** (TrackML 1st).
   *Falsification:* must beat our greedy/ILP selection on LOEO adj_edge_J on cached OOF graphs. **GPU: No.**
7. **Centre-crop-only prediction, discard tile borders** (HuBMAP-kidney 1st). Our 199 crops overlap.
   *Falsification:* restricting each crop's emitted nodes to its interior (and letting neighbours cover the
   margin) must not lose recall while reducing duplicates. **GPU: No.**
8. **Two-threshold hysteresis with a size gate** (SIIM-ACR 1st). *Falsification:* a (high-seed, low-expand)
   pair must beat the single 0.96875 threshold on LOEO at matched N_pred. **GPU: No.**
9. **Architecturally diverse ensemble** (CryoET 1st; DSB2018). Only meaningful once ≥1 retrain exists;
   should be an arm of the retrain bet, per `redteam` Claim 3. *Falsification:* two different backbones must
   beat two seeds of one by >2× the seed-to-seed spread. **GPU: Yes.**

**NOT AVAILABLE / NOT APPLICABLE TO US**

- **Recall-tilted operating points.** CryoET's F-beta(4) rewards over-prediction; our node-count multiplier
  punishes it. Import CryoET's *machinery* (OOF threshold curves, point-IoU assignment), never its
  *direction*.
- **Large permissive same-modality labelled corpora.** LIVECell / TissueNet / HPAv20 have no 3D+t
  light-sheet zebrafish equivalent except Zebrahub, whose imaging is 100s GB–TB and (per our memory)
  not on disk. This is a **hard gate on levers 2, 3, 9**.
- **Heavy compute.** 8×H100 nodes and 264-submission campaigns are not reachable at T4×2 / ~30 h per week /
  5 submissions per day.
- **LB-probing strategies.** HPA winners at 3.3 submissions/day is not a template we can copy, and the GBR
  result argues it is not one we should want.
- **Stain-normalization literally.** Vahadane/histogram matching is an H&E-specific method; the *principle*
  (normalize the acquisition nuisance before modelling) transfers, the *method* does not.
- **Metric exploitation.** The hub-fork hack was patched and re-scored ~2026-07-20 (`competitive_frontier`).
  Deliberately under-predicting node count is *inside* the published metric (no upper clip on the bonus,
  `competitive_refresh` CW5) and is not a hack — but anything that games the *matching* rather than the
  *scoring* should be treated as rule risk, not a lever.

---

## Corrections this report makes to our existing conclusions

1. **Pseudo-labelling should be demoted.** `novel_crossdomain` ranks cycle-consistency pseudo-labels (#6) as
   "the cheapest XFAM/sparse-annotation lever" and PU learning (#3) high. The controlled evidence from
   NeurIPS CellSeg (all three medallists, explicitly null) and HuBMAP-vasculature 3rd ("no boost") says this
   family under-delivers *specifically in cell segmentation with sparse instance labels*. Keep it, but below
   external pretraining and operating-point calibration.
2. **"Learn the linker" needs a classical counterweight.** TrackML — the only pure-association Kaggle
   competition — was won by structured combinatorics with a global pollution-minimizing resolution stage;
   the best learned pairwise-edge model came second and the LSTM came 12th. Before building HOCT/MoTT/OT
   machinery, test the *cheap* classical stage we are missing.
3. **"Multi-seed ensembling" should be "multi-architecture ensembling."** `redteam` Claim 3 flags multi-seed
   as an unswept lever. CryoET's winner attributes the margin to **SegResNet vs DynUNet diversity**, and
   DSB2018's top three each used a different architecture family. Seeds of one backbone are the weaker form.
4. **The retrain bet has a documented recipe, not just a direction.** Our reports say "retrain on Zebrahub".
   The documented, quantified mechanism is **two-stage noisy→clean with inverted LR and augmentation
   schedules** (+4–6% LB), plus **assignment-rule/metric alignment** in the detector head. Those are
   specifications, and they change what the first pilot should look like.

---

## Licence and rule flags

- **Zebrahub** external data: host-confirmed permitted 2026-08-13 (#734330) per `competitive_frontier`. Not
  re-verified here.
- **LIVECell, TissueNet, Omnipose, Cellpose, HPAv20, GTEx** — cited as *evidence of a pattern*, not as
  assets for us. None is same-modality. If any were ever considered, licences must be checked individually;
  I did not check them.
- **MONAI** (SegResNet, DynUNet, FlexibleUNet — the CryoET winner's components) is Apache-2.0
  (**UNVERIFIED** — stated from general knowledge, not checked this session). The *mechanisms* (anchor-free
  point head, point-IoU assignment, varifocal loss) are reimplementable from the write-up regardless.
- **`kaggle.curtischong.me`** is a third-party mirror of Kaggle write-ups. Vesuvius content is quoted from
  the mirror; originals not read. Treat rank/author attributions there as medium confidence.
- **Competition rules**: notebook-only, internet-off rerun. Every CPU lever above must run inside the
  submission kernel. The GBDT stages (#4) and threshold/quantile logic (#1) are trivially compatible;
  anything needing `tracksdata`+torch at scoring time is not (`competitive_refresh` §5 tooling note).
- No lever in this report requires redistributing a licence-restricted model. HOCT's CC-BY-NC-ND issue
  (`methods_frontier`) is untouched here.

---

## What I could not verify

- The full text of **any** Kaggle discussion or `/writeups/` page (client-rendered; CLI has no discussions
  endpoint). Six solution summaries are therefore search-extracted, tagged **[extracted, not read]** above.
- The **CZII lessons-learned preprint** (biorxiv 2025.11.03.686153) and the **DSB2018 analysis preprint**
  (biorxiv 580605): bioRxiv returned HTTP 429/403 on five attempts across the session. The DSB2018 claims
  here rest on the Nature Methods article as surfaced by search indexing, not on my own read of it.
- **TrackML winning score**: 0.921 (challenge results site) vs ~0.96 (arXiv figure) — unresolved.
- **Sartorius 1st-place team attribution** and the "3rd place used CellPose" claim.
- Whether any surveyed competition used a **multiplicative cardinality penalty** like ours. Targeted
  searching for Kaggle metrics that penalize prediction count returned **our own competition's `metrics.md`**
  (<https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md>) as the top hit. My
  working conclusion is that the `(1 − 0.1·(N_pred−N_est)/N_est)` multiplier has **no close precedent on
  Kaggle**, and the nearest structural analogues are the `TP/(TP+FP+FN)` mAP family (DSB2018, Sartorius) and
  matched-F1 (NeurIPS CellSeg). Absence of search evidence is not proof of absence.
