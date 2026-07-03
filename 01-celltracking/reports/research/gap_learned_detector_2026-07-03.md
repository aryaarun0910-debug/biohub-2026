# Gap Analysis: Learned 3D Nucleus Detector for Biohub Cell Tracking

Date: 2026-07-03
Scope: SOTA 2024-2026 learned 3D nucleus/cell **detection** (centroids/instances) for developmental light-sheet fluorescence, with a bias toward methods that ship **downloadable offline weights** we can upload to Kaggle and run on 2x T4 in <=12h, in a sparse-label (~1% annotated, 2 embryos), anisotropic (Z ~4x coarser) regime.

Our baseline: classical Difference-of-Gaussians @ 0.807 public; classical ceiling ~0.854. Target lever: learned detector to ~0.865-0.885.

---

## 1. Landscape — what is actually new (2024-2026)

The field has bifurcated into (a) **generalist foundation-style segmenters** with strong pretrained weights (Cellpose-SAM, Cellpose3, CellSAM), (b) **purpose-built anisotropic light-sheet detectors** (STAR-3D, PAC-MAP), and (c) **self-/weakly-supervised** methods that thrive on sparse labels (CellSeg3D/WNet3D, point-supervised pseudo-labeling). Two facts reshaped my recommendation:

- **Trackastra is NOT a detector.** It is a transformer *linker* that consumes an existing segmentation/detection and predicts associations across time. It provides no detection weights. It is relevant later (tracking stage), not for the detection lever. https://github.com/weigertlab/trackastra
- **NIS3D is a benchmark dataset (NeurIPS 2023), not a pretrained detector.** 6 densely-annotated 3D embryonic volumes (zebrafish/drosophila/etc, 22k+ cells) — extremely useful as **external training/validation data**, but ships no drop-in detector weights. https://github.com/yu-lab-vt/NIS3D

The single closest match to our problem is **STAR-3D**: StarDist-3D retrained specifically on *highly anisotropic light-sheet developing-embryo* nuclei.

---

## 2. Comparison table

| Method | Task / output | Offline weights? | License | Anisotropy handling | Centroid? | Kaggle T4x2 <=12h feasible? |
|---|---|---|---|---|---|---|
| **STAR-3D** (StarDist-3D, anisotropic light-sheet embryo) | 3D instance (star-convex) | Yes — `pip install star3d`, `StarDist3D.from_pretrained('STAR-3D')` | GPL-3.0 | **Trained for it**: anisotropic kernels (3,4,4), pooling (1,2,2); trained on mouse embryo 0.174 in-plane / 2 µm z (~11x) | Instance -> centroid trivially (region centroid) | Yes. StarDist-3D is patch-based; ~5 min high-end, ~1.5h on weak CPU/8GB. On T4 a ~100-frame volume set is comfortably <12h. |
| **PAC-MAP** (proximity-adjusted centroid map) | **Direct 3D centroid heatmap** | Yes — Zenodo (pretrained + finetuned + scratch) | **CC BY-NC-SA 4.0 (NonCommercial)** | 3D U-Net; radius/proximity set per-dataset; expects near-isotropic input (resample recommended) | **Native centroid output** (local maxima of proximity-adjusted map) | Yes. Single 3D U-Net forward per patch; light. |
| **Cellpose-SAM** (2025) | 3D instance (flows) via SAM backbone | Yes — auto-download or Google Drive zip to `~/.cellpose/models` | BSD-3-Clause | Explicitly trained robust to **anisotropic blur, downsampling**; `anisotropy` rescale param | Instance -> centroid | Borderline. ViT/SAM backbone is heavy; 3D done slice-wise + stitch or ortho. Feasible with fp16 + tiling but watch the 12h budget on ~100 volumes. |
| **Cellpose3** (cyto3 + restoration) | 3D instance (flows) | Yes — auto-download model zoo | BSD-3-Clause | `anisotropy` param (e.g. 2.0), `flow3D_smooth`, `pretrained_model_ortho`, `stitch_threshold` for 2D->3D | Instance -> centroid | Yes. Lighter than SAM; well-trodden 3D path (`do_3D` or stitch). |
| **EmbedSeg** (3D) | 3D instance (embedding to medoid) | Weights per-dataset (trainable; some released) | BSD-3-Clause | Handles anisotropy via training; small GPU footprint | **Medoid = centroid-like** by design | Yes — low GPU memory, but needs training on our data (no generalist embryo weight). |
| **CellSeg3D / WNet3D** (eLife 2025) | 3D semantic + instance, **self-supervised** (no labels) | Yes — pretrained WNet3D released (repo/HuggingFace, Colab auto-DL) | (repo license — verify; MIT/BSD family) | 3D (SwinUNETR / WNet3D); resample for strong anisotropy | Instance -> centroid | Yes. Attractive because self-supervised matches our sparse-label pain. |
| **3D LSFM Foundation Model** (arXiv 2605.26026, 2026) | Masked-reconstruction + image-text pretrain; few-shot seg/classify/deblur | Yes — code + weights public (repo linked in paper) | (verify in repo) | LSM-native pretraining; anisotropy not documented | Downstream head required (not turnkey centroids) | Unknown/heavier. Newest, highest ceiling, but most integration risk. |
| **CellSAM** | 2D-centric SAM cell foundation model | Yes | (verify) | Not a native 3D detector | — | Not a fit (2D). |
| **NIS3D** | Benchmark **dataset** | Data only (no detector) | dataset license | — (data) | N/A | Use as external train/val data. |
| **Trackastra** | Tracking **linker** (not detection) | Yes (linker weights) | BSD-family | N/A | N/A | Tracking stage only. |

Notes: Cellpose/StarDist/EmbedSeg core licenses are BSD-3-Clause (permissive). **PAC-MAP weights are NonCommercial (CC BY-NC-SA 4.0)** — fine for a research/Kaggle submission, but the ShareAlike/NC terms mean don't reuse commercially and attribute; verify competition rules permit NC-licensed assets.

---

## 3. Top-3 ranked by payoff-per-effort (under our constraints)

### #1 — STAR-3D (highest payoff-per-effort)
Closest domain match on Earth to our task: StarDist-3D **already trained on anisotropic light-sheet developing-embryo nuclei**. Anisotropy is baked into the architecture (anisotropic kernels/pooling), so we sidestep the Z-4x problem instead of engineering around it. Star-convex instances -> centroids trivially, and StarDist has mature fine-tuning. Drop-in `from_pretrained('STAR-3D')`, patch-based inference fits T4 easily.
- Effort: low (install, run, threshold-tune, optional light finetune on our sparse points).
- Risk: GPL-3.0 (our submission code would inherit copyleft if we redistribute — acceptable for Kaggle, note it). Trained anisotropy (~11x) differs from ours (~4x) — validate/adjust `anisotropy`/grid.
- URL: https://github.com/akarsa/star-3d

### #2 — PAC-MAP (best "native centroid" fit for sparse labels)
Purpose-built for exactly our output (nucleus **centroids** in dense 3D) and — critically — its published recipe **is** the sparse-label recipe: weak-supervised pretraining on baseline detections, then **finetune on few expert annotations** (F1 0.793 scratch -> 0.817 with pretrain+few labels). We already have a classical DoG detector to generate the weak pretraining targets. Direct heatmap -> local maxima, light 3D U-Net.
- Effort: low-medium (resample to ~isotropic, set radius, run their pretrain+finetune loop with DoG as weak teacher).
- Risk: NonCommercial license (CC BY-NC-SA 4.0) — confirm Kaggle-permissible; expects roughly isotropic input so resampling needed.
- URLs: https://github.com/DeVosLab/PAC-MAP · weights/data Zenodo https://zenodo.org/records/14138806 · paper https://www.sciencedirect.com/science/article/pii/S0010482524016469

### #3 — Cellpose3 (cyto3) / Cellpose-SAM (safest permissive generalist)
BSD-licensed, huge community, robust generalist weights, explicit `anisotropy` handling, restoration to fight light-sheet noise/undersampling, and human-in-the-loop finetuning. Cellpose3 is the pragmatic pick (lighter, proven 3D path); Cellpose-SAM has the higher ceiling but heavier ViT backbone — watch the 12h budget across ~100 volumes (use fp16 + tiling + `stitch_threshold` 2D->3D if `do_3D` is too slow).
- Effort: low to stand up; medium to tune 3D flow/stitch params for our anisotropy.
- Risk: generalist (not embryo-specific) -> may need finetuning to beat classical ceiling; SAM variant compute.
- URLs: Cellpose3 https://www.nature.com/articles/s41592-025-02595-5 · Cellpose-SAM https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1 · code https://github.com/MouseLand/cellpose

---

## 4. Sparse-label fine-tuning recipe (~1% annotated, 2 embryos)

The literature converges on a **weak-supervision -> pseudo-label self-training** loop, which PAC-MAP operationalizes for 3D nuclei specifically:

1. **Soft mask from sparse points.** Convert each annotated centroid into a Gaussian blob target (StarDist/PAC-MAP style). PAC-MAP goes further: kernel amplitude & sigma scale with **proximity to nearest neighbor** — encodes crowding, which matters for dense embryo nuclei.
2. **Weak-supervised pretraining.** Use our classical **DoG detector as the weak teacher** to label the ~99% unannotated volume, pretrain the net on those noisy targets (this is exactly PAC-MAP's proven step; +F1 over scratch).
3. **Finetune on the sparse expert points** (the real ~1%). Small LR, the true labels correct the teacher's bias.
4. **Self-training / pseudo-label refresh.** Iterate: predict on unlabeled crops, keep **high-confidence** local maxima as new pseudo-GT, retrain. Recent point-supervised work (Dynamic Pseudo-Label Optimization, MICCAI 2024; point-guided attention + self-supervised pseudo-labeling, Bioengineering 2025) shows dynamic pseudo-label filtering beats using static initial pseudo-labels.
5. **Positive-unlabeled framing.** Treat annotated points as positives, everything else as *unlabeled* (not negative) — avoids punishing the net for detecting real-but-unannotated nuclei (critical at 1% labels). Weight/ignore loss on unlabeled background accordingly.
6. **Partner: CellSeg3D/WNet3D self-supervised** — can pretrain a 3D representation with **zero labels** on the raw embryo volumes, then attach a light detection head trained on the sparse points.

Refs: PAC-MAP https://www.biorxiv.org/content/10.1101/2024.07.18.602066v1 · Dynamic Pseudo-Label Optimization https://arxiv.org/pdf/2406.16427 · point-guided self-supervised pseudo-labeling https://www.mdpi.com/2306-5354/12/1/85 · CellSeg3D https://elifesciences.org/articles/99848

---

## 5. Anisotropy best practices (Z ~4x coarser: z=1.625, xy=0.40625 µm)

- **Prefer anisotropic architecture over blind resampling.** Use anisotropic conv/pool kernels (e.g. (1,3,3) or (3,4,4) conv, (1,2,2) pool) so the receptive field is physically balanced without wasting compute upsampling Z. This is exactly what STAR-3D does and why it's the top pick.
- **If a method assumes isotropy (PAC-MAP, generalist Cellpose 3D flows):** resample Z to match XY *or* set the tool's `anisotropy` factor (Cellpose `anisotropy≈4`; StarDist `grid`/`anisotropy` from voxel spacing). Downsample-high-axis-then-upsample avoids redundant data blow-up vs naive isotropic upsampling.
- **Hybrid 2D->3D** (segment XY slices, stitch via `stitch_threshold`) is a valid, cheap fallback that sidesteps 3D-conv cost and handles anisotropy gracefully — useful if `do_3D` blows the 12h budget.
- **Optional**: self-supervised isotropic super-resolution (deep, reference-free) to synthesize denser Z before detection — higher effort, usually unnecessary if the detector is anisotropy-aware.

Refs: STAR-3D architecture (above); anisotropic-kernel & resample guidance from EM/medical segmentation surveys and slice-imputation work https://arxiv.org/pdf/2203.10773 ; reference-free isotropic SR https://pmc.ncbi.nlm.nih.gov/articles/PMC9178036/

---

## 6. Concrete next steps (with URLs)

1. **STAR-3D smoke test (day 1).** `pip install star3d`; load `StarDist3D.from_pretrained('STAR-3D')`; run on a few of our 3D frames with `anisotropy` derived from (1.625/0.40625≈4). Score centroids vs our labeled crops. → https://github.com/akarsa/star-3d
2. **PAC-MAP weak+finetune (day 1-3).** Clone https://github.com/DeVosLab/PAC-MAP ; pull weights from https://zenodo.org/records/14138806 ; generate weak targets from our DoG detector; run pretrain->finetune on our ~1% points. Confirm license OK for Kaggle.
3. **Cellpose3 baseline (parallel).** `pip install cellpose`; cyto3/nuclei with `anisotropy≈4`, tune `flow_threshold`/`stitch_threshold`; use as permissive fallback. → https://github.com/MouseLand/cellpose
4. **Grab NIS3D as external train/val data** (embryo nuclei, densely annotated) to pretrain/validate any of the above. → https://github.com/yu-lab-vt/NIS3D
5. **Package for Kaggle offline:** upload chosen weights as a Kaggle Dataset; fp16 + tiling/patch inference; verify a single ~100-frame volume runs well under 12h on one T4 (leave headroom for T4x2).
6. **Stretch:** evaluate CellSeg3D/WNet3D self-supervised pretraining and the 3D LSFM foundation model (arXiv 2605.26026) if top-3 stall below ~0.86.

---

## Sources
- STAR-3D — https://github.com/akarsa/star-3d
- PAC-MAP — https://github.com/DeVosLab/PAC-MAP ; https://www.biorxiv.org/content/10.1101/2024.07.18.602066v1 ; https://www.sciencedirect.com/science/article/pii/S0010482524016469 ; Zenodo https://zenodo.org/records/14138806
- Cellpose-SAM — https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1
- Cellpose3 — https://www.nature.com/articles/s41592-025-02595-5 ; https://github.com/MouseLand/cellpose
- EmbedSeg — https://github.com/juglab/EmbedSeg ; https://arxiv.org/abs/2101.10033
- CellSeg3D / WNet3D — https://elifesciences.org/articles/99848 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC12187128/
- 3D LSFM Foundation Model — https://arxiv.org/abs/2605.26026
- NIS3D — https://github.com/yu-lab-vt/NIS3D
- Trackastra (linker, not detector) — https://github.com/weigertlab/trackastra
- Sparse-label: Dynamic Pseudo-Label Optimization https://arxiv.org/pdf/2406.16427 ; point-guided self-supervised pseudo-labeling https://www.mdpi.com/2306-5354/12/1/85
- Anisotropy: slice imputation https://arxiv.org/pdf/2203.10773 ; reference-free isotropic SR https://pmc.ncbi.nlm.nih.gov/articles/PMC9178036/
