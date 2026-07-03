# External Public Datasets for Pretraining a 3D Nucleus Detector

Research date: 2026-07-03
Task: find densely-annotated 3D(+t) nucleus/cell datasets to pretrain a learned 3D nucleus
detector for the Kaggle "Biohub - Cell Tracking During Development" competition (Royer Group,
zebrafish light-sheet, uint16 3D volumes, anisotropic z=1.625µm / xy=0.40625µm), then fine-tune
on the 2 sparse competition embryos.

> LICENSE WARNING UP FRONT: Not everything here is safely usable. The **Cell Tracking Challenge**
> datasets are the biggest domain match but carry a **restrictive custom license** (commercial use
> and non-CTC use require explicit permission; redistribution of annotations forbidden). Treat them
> as **flagged / verify-before-use** for a non-CTC Kaggle competition. See caveats.

---

## 1. Master comparison table

| Dataset | Organism | Modality | Annotation | Size | License | Domain-match | URL |
|---|---|---|---|---|---|---|---|
| **NIS3D** | zebrafish + Drosophila + mouse embryos | fluorescence 3D (embryonic nuclei) | **Dense manual instance masks**, 22k+ cells, 6 volumes, per-cell confidence (3 annotators) | ~3.3 GB | **CC-BY-4.0** (commercial OK w/ attribution) | High (zebrafish embryo nuclei, dense) | github.com/yu-lab-vt/NIS3D ; zenodo.org/records/11456029 |
| **CTC Fluo-N3DL-DRO** | *Drosophila* embryo | **Light-sheet** (SIMView) | Gold tracking/lineage, **centroids/points** (nervous-system subset only, SPARSE) | 5.8 GB train / 5.9 GB test | **CTC custom** (restrictive – see caveat) | High (light-sheet, **voxel 0.406×0.406×2.03 ≈ our anisotropy**) | celltrackingchallenge.net/3d-datasets |
| **CTC Fluo-N3DL-TRIF** | *Tribolium* beetle embryo | **Light-sheet** | Gold tracking/lineage, blastoderm subset (SPARSE points) | 320 GB train / 467 GB test | CTC custom (restrictive) | Med-High (light-sheet dev. embryo; huge, sparse) | celltrackingchallenge.net/3d-datasets |
| **CTC Fluo-N3DL-TRIC** | *Tribolium* beetle embryo | **Light-sheet** | Gold tracking/lineage, blastoderm subset (SPARSE) | 20.6 / 19.9 GB | CTC custom (restrictive) | Med-High | celltrackingchallenge.net/3d-datasets |
| **CTC Fluo-N3DH-CE** | *C. elegans* embryo | Confocal | **Gold tracking/lineage + fuller nuclei coverage** | 3.1 / 1.7 GB | CTC custom (restrictive) | Med (dev. embryo nuclei, but confocal, isotropic-ish 0.09×0.09×1.0) | celltrackingchallenge.net/3d-datasets |
| **CTC Fluo-N3DH-SIM+** | simulated HL60 nuclei | simulated confocal | **Perfect dense masks** (synthetic) | 3.1 / 5.9 GB | CTC custom | Low-Med (synthetic, good for warm-start) | celltrackingchallenge.net/3d-datasets |
| **Zebrahub (DaXi/OpenSiMView)** | **zebrafish** whole embryo | **DaXi single-objective light-sheet** (same lab/instrument as competition) | Nuclei segmentations + tracks, but **machine-generated (Ultrack), not manual gold** | large (multi-TB, per-timelapse) | Verify (CZ Biohub public; likely permissive but not confirmed on page) | **Highest domain match** (same lab, same microscope, zebrafish) | zebrahub.sf.czbiohub.org/imaging ; public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ |
| **Linajea zebrafish** | **zebrafish** embryo | Light-sheet | **Sparse point/track ground truth** + trained nets + predicted tracks (imaging via link) | dataset small; imaging separate | figshare (likely CC BY 4.0 – verify) | High (zebrafish light-sheet) but sparse | janelia.figshare.com/articles/dataset/24968724 |
| **Linajea Drosophila** | *Drosophila* embryo | Light-sheet | Sparse point/track GT + nets + tracks; imaging via Dropbox (n5/zarr) | tracks small; images ~GBs | figshare (likely CC BY 4.0 – verify) | Med-High | janelia.figshare.com/articles/dataset/24937092 |
| **Linajea mouse** | mouse embryo | Light-sheet | Sparse point/track GT | — | figshare (verify) | Low-Med | janelia.figshare.com/articles/dataset/24768798 |
| **EmbryoNet** | zebrafish embryos | brightfield/widefield whole-embryo | Phenotype **classification** labels (not 3D nuclei) | large | check repo | Low (not nucleus detection) | github.com/mueller-lab/EmbryoNet |
| **MitoEM** | mammalian brain | EM | Dense mitochondria instances (not nuclei) | ~ large | CC (BBBC-style) | Low (wrong modality/target) | mitoem benchmark |

Notes: CTC voxel/anisotropy and sizes are quoted from the CTC 3D+time datasets page. CTC "N3DL"
datasets are annotated only on a biologically-relevant SUBSET of cells (sparse), similar in spirit
to our competition's sparsity — they are lineage/detection benchmarks, not dense per-voxel masks.

---

## 2. Ranked shortlist (best 2-3 for pretraining under external-data rules)

### #1 — NIS3D  (recommended primary pretraining set)
- **Why:** The only set here that is BOTH **densely + manually annotated 3D nuclei** AND **cleanly
  CC-BY-4.0** (commercial use allowed with attribution → safe for a competition model). Contains
  **zebrafish embryo** nuclei plus Drosophila and mouse — directly the target object class.
- **Content:** 6 large volumes, 22,000+ instance-annotated cells, per-cell confidence scores.
- **Use:** Convert instance masks → centroids/Gaussian heatmaps for detector pretraining (below).
- **License:** CC-BY-4.0 (Zenodo record 11456029). Safe. Just cite the NeurIPS 2023 Datasets paper.
- **URLs:** https://github.com/yu-lab-vt/NIS3D , https://zenodo.org/records/11456029

### #2 — Zebrahub imaging (DaXi / OpenSiMView)  (best domain match, pretrain on pseudo-labels)
- **Why:** **Exact domain** — same lab (Royer/CZ Biohub), same DaXi single-objective light-sheet
  microscope, same organism (developing zebrafish), same anisotropic uint16 volumes as the
  competition. Ideal for self-/weakly-supervised pretraining and domain adaptation.
- **Caveat:** Annotations are **machine-generated (Ultrack pipeline), not manual gold** — treat as
  pseudo-labels, not clean ground truth. **License not stated on the imaging page — VERIFY before
  training a submission model** (CZ Biohub data is generally openly shared but confirm CC terms;
  also confirm the timelapses are NOT the same embryos as the competition test set to avoid leakage).
- **URLs:** https://zebrahub.sf.czbiohub.org/imaging ,
  https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/

### #3 — CTC Fluo-N3DL-DRO (+ optionally Fluo-N3DH-CE)  (best geometric match — but license-flagged)
- **Why:** **Light-sheet** developmental embryo with **voxel size 0.406×0.406×2.03 µm**, almost
  identical anisotropy to our competition data (0.40625 xy / 1.625 z). Excellent for teaching the
  detector our exact z-vs-xy sampling. Fluo-N3DH-CE (C. elegans) adds denser lineage coverage.
- **Caveat (IMPORTANT):** CTC uses a **custom, restrictive license** — "Any CTC-related use…does
  not require explicit consent," but **any non-CTC use, and any commercial use, requires explicit
  permission from organizers + data providers," and "cloning of datasets or annotations is strictly
  forbidden."** A different Kaggle competition is **not** CTC-related → usage is **ambiguous/likely
  restricted**. Do NOT assume it's allowed; either seek permission, use it only for private
  research/ablation, or prefer NIS3D/Zebrahub for the actual submission model.
- **URL:** https://celltrackingchallenge.net/3d-datasets/ , conditions: celltrackingchallenge.net/datasets/

---

## 3. License caveats (read before using)

- **NIS3D — CC-BY-4.0:** Safe, commercial OK, attribution required. Cleanest option.
- **Cell Tracking Challenge (ALL Fluo-* sets) — FLAGGED:** custom non-CC license. CTC-participation
  use is free; **non-CTC and commercial use need explicit written permission**; redistributing the
  annotations is forbidden. For a *different* Kaggle competition this is **not clearly permitted** —
  verify with organizers or avoid for the submitted model. (This overrides the common claim online
  that "CTC is CC-BY" — the dataset conditions page is stricter.)
- **Zebrahub / DaXi imaging — VERIFY:** no explicit license on the imaging page. DaXi *code* is BSD-3.
  Ultrack code is MIT. Confirm the *data* license and check for **train/test leakage** with the
  competition (same lab/instrument raises overlap risk).
- **Linajea (Janelia figshare) — LIKELY CC BY 4.0 (figshare default) but VERIFY per record.**
  Annotations are **sparse** (point/track GT from sparse annotations paper), plus trained nets +
  predicted tracks. Code (funkelab/linajea) separate license.
- **EmbryoNet:** phenotype classification, not 3D nucleus detection — not useful for this task.
- **MitoEM:** mitochondria in EM — wrong target/modality; exclude.

---

## 4. Practical: format, targets, anisotropy

- **Formats:** NIS3D = `Data.tif` + `GroundTruth.tif` (instance labels, 0=bg) + confidence tif per
  volume. CTC = TIFF stacks + `TRA`/`SEG` gold-truth (label masks + `man_track.txt` lineage).
  Zebrahub/Linajea = OME-Zarr / n5. All are readable with `tifffile` / `zarr` / `ome-zarr`.
- **Instance masks → detector targets:** compute per-instance **centroid** (center of mass of each
  label), render a **3D Gaussian heatmap** (or seed/offset maps) at each centroid for a
  detection/counting head; or keep instance masks for a segmentation warm-start. For sparse sets
  (CTC N3DL, Linajea) only annotated cells have targets — mask the loss to labeled regions (matches
  our own ~1%-annotated competition setup, so the sparse-loss recipe transfers directly).
- **Anisotropy handling:** our data is z=1.625 / xy=0.40625 → ~**4× coarser in z**. Options:
  (a) resample all pretraining volumes to a common physical spacing before training; or
  (b) keep native spacing and use **anisotropic 3D kernels / strides** (less pooling in z) plus
  physically-sized Gaussian sigmas (σ_z ≈ σ_xy · (xy/z spacing)). **Fluo-N3DL-DRO's 0.406×0.406×2.03
  spacing is the closest external match**, so it's the best set to validate the anisotropic pipeline
  — license permitting.
- **Domain gap:** NIS3D gives clean dense supervision but varied modality/spacing; Zebrahub gives
  exact modality/spacing but noisy labels. A sensible recipe: **pretrain on NIS3D (dense, clean,
  CC-BY) → domain-adapt on Zebrahub pseudo-labels (exact modality) → fine-tune on the 2 competition
  embryos.** Use CTC-DRO only for anisotropy validation/ablation unless license is cleared.

---

## Sources
- NIS3D: https://github.com/yu-lab-vt/NIS3D , https://zenodo.org/records/11456029 ,
  paper https://proceedings.neurips.cc/paper_files/paper/2023/file/0f2cd3d09a132757555b602e2dd43784-Paper-Datasets_and_Benchmarks.pdf
- Cell Tracking Challenge 3D+time: https://celltrackingchallenge.net/3d-datasets/ ;
  conditions of use: https://celltrackingchallenge.net/datasets/ ;
  10-yr paper: https://www.nature.com/articles/s41592-023-01879-y
- Zebrahub: https://zebrahub.sf.czbiohub.org/imaging ; Cell 2024 paper
  https://www.cell.com/cell/fulltext/S0092-8674(24)01147-4
- DaXi: https://github.com/royerlab/daxi , https://www.nature.com/articles/s41592-022-01417-2 ;
  Ultrack: https://royerlab.github.io/ultrack/
- Linajea: https://github.com/funkelab/linajea ;
  zebrafish data https://janelia.figshare.com/articles/dataset/24968724 ;
  Drosophila https://janelia.figshare.com/articles/dataset/24937092 ;
  paper https://www.nature.com/articles/s41587-022-01427-7
- EmbryoNet: https://github.com/mueller-lab/EmbryoNet ,
  https://www.nature.com/articles/s41592-023-01873-4
