# Codex lateral sweep — asymmetric edges

**Date:** 2026-07-03  
**Objective:** find legal, orthogonal moves that can change the private-set ceiling, not another DoG/LB tuning cycle.

## Executive verdict

The best immediate edge is **motion-compensated track-before-detect**, because it attacks the measured no-candidate bottleneck and can be falsified in hours. The conditional jackpot is dense pretraining from Biohub's public March-22 exact-scale embryo and 122 MB `zoo/Zebrafish` tracking bundle. Both artifacts are verified, but they are **not yet proven to describe the same embryo**; provenance and coordinate alignment must pass before treating the bundle's approximately 11.85 million memberships as image-aligned supervision.

Track-before-detect accumulates raw DoG/PSF response along plausible short trajectories before thresholding. It is conceptually absent from the public field, which still thresholds each frame before linking.

The external-data clause on the authenticated rules page states: “You may use data other than the Competition Data (‘External Data’) to develop and test your Submissions.” Public-data pretraining is therefore permitted. **Do not transfer labels into a suspected test crop or try to identify private labels without written organizer confirmation.** [Official rules](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules)

## Ranked edges

1. **Motion-compensated track-before-detect** — expected upside roughly +0.005 to +0.020.
2. **March-22 provenance/alignment, then dense pretraining** — conditional upside roughly +0.010 to +0.040.
3. **NIS3D pretraining, with nnPU as a separate ablation** — roughly +0.005 to +0.030.
4. **Raw + FM2S-denoised proposal union** — roughly 0 to +0.005.
5. **Target-volume self-supervised temporal denoising** — uncertain 0 to +0.010.
6. **FGW neighborhood association** — roughly +0.001 to +0.006.
7. **CPD/GLMB ambiguity handling** — roughly 0 to +0.005.

### Motion-compensated track-before-detect

**EV: HIGH · Novelty: VERY HIGH · Feasibility: HIGH for diagnostic · STATUS: permitted**

Current pipelines discard weak nuclei at a per-frame threshold and ask linking to repair the damage. Track-before-detect does the reverse: integrate low-threshold PSF/DoG evidence along a 3–7-frame motion-consistent path, then declare the node/track. Astronomy and radar use this to recover objects invisible in individual frames; microscopy literature explicitly notes temporal coherence as crucial for weak fluorescence.

- [Track-before-detect via low-threshold candidate plots](https://doi.org/10.1016/j.dsp.2022.103458)
- [Learn-to-Track-Before-Detect paper](https://doi.org/10.1109/ICASSP48485.2024.10448128) and [code](https://github.com/TslilTap/Learn-to-Track-Before-Detect)
- [Microscopy review: temporal integration for weak motion](https://imagescience.org/meijering/publications/download/spm2006.pdf)

**Why orthogonal:** this is not ordinary gap closing. A missing frame contributes subthreshold image evidence before any detection decision; several weak observations jointly cross a calibrated null.

**First experiment (interrupt execution):** on the 20 worst-recall crops, retain the raw multiscale response volume. Develop the temporal score and null threshold on one fold only; freeze them, then evaluate GT no-candidate recovery on the other fold and reverse. Compare (a) single-frame response percentile, (b) a 5-frame Viterbi sum around local tissue velocity, and (c) matched background paths. **Go** if temporal accumulation recovers at least 20% of DoG misses at ≥70% marginal edge precision or raises exact min-fold adjusted-J by ≥0.003. This falsification test is CPU-feasible in hours.

**Offline Kaggle:** stream 3–7 frames, use the existing low-threshold proposal bank plus local raw-response samples, and beam-search only around reliable endpoints/collisions. No new package is required.

### Exact-scale March-22 provenance/alignment, then dense pretraining

**EV: VERY HIGH · Novelty: HIGH · Feasibility: MED-HIGH · STATUS: permitted external pretraining**

Biohub hosts a separate 522-frame embryo at `(1.625, 0.40625, 0.40625) µm`, exactly matching the competition spacing. It also hosts a 122 MB public `zoo/Zebrafish` track bundle with roughly 11.85 million track-point memberships; the parallel attribute store includes divisions, displacement, density, structure strength, anisotropy, and orientation. Metadata identifies the image as `2024_03_22_dorado/stabilized.zarr`, distinct from ZSNS003–005's February/March DaXi acquisitions. **The current evidence does not establish that this bundle belongs to the March-22 image.**

- [Exact-scale image](https://public.czbiohub.org/royerlab/ultrack/zebrafish_embryo.ome.zarr/)
- [Track bundle](https://public.czbiohub.org/royerlab/zoo/Zebrafish/tracks_zebrafish_bundle.zarr.zip)
- [Track attributes](https://public.czbiohub.org/royerlab/zoo/Zebrafish/tracks_zebrafish_attributes_bundle.zarr/)

**Why orthogonal:** it replaces the central weakness—1–6% spatial annotation—with dense supervision from the same imaging ecosystem. It can pretrain both detection and motion/division heads without treating real cells as negatives.

**Critical caveat:** provenance comes first. If timestamps/count trajectories support identity, bundle points still appear transformed into an isotropic/rotated visualization coordinate system. The edge then lives or dies on recovering the exporter transform or fitting a reliable affine.

**First experiment (interrupt execution):** first compare frame count, valid-point count trajectory, timestamps/metadata, and coarse spatial occupancy. Only then align ten frames by fitting axis permutation/sign/scale/translation and affine refinement against bright nuclear peaks. **Go** if identity is plausible, median point-to-peak error is below 3 µm, and at least 90% of sampled points land in nuclear signal; otherwise stop before training. Then pretrain a compact Gaussian heatmap/point detector on 50–100 aligned frames and gate on both embryo-held-out folds: no-candidate recall +5 points and exact min-fold adjusted-J +0.005 minimum.

**Offline Kaggle:** preprocess aligned patches and labels into a versioned Kaggle Dataset; notebook inference needs no internet. Training fits T4×2/12 h if patches are prepared beforehand.

### NIS3D zebrafish dense pretraining, then nnPU fine-tuning

**EV: HIGH · Novelty: HIGH · Feasibility: HIGH · STATUS: permitted public external data**

NIS3D is a 3.296 GB CC-BY-4.0 benchmark with 22,000+ completely annotated embryonic nuclei across six volumes, including two zebrafish volumes. It includes per-cell annotator-confidence maps and reports zebrafish sampling near the competition laterally. The repository is MIT-licensed.

- [NIS3D paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/0f2cd3d09a132757555b602e2dd43784-Paper-Datasets_and_Benchmarks.pdf)
- [Official repository](https://github.com/yu-lab-vt/NIS3D)
- [Dataset record](https://zenodo.org/records/11456029)
- [Cell-detection nnPU paper](https://arxiv.org/abs/2302.08050) and [code](https://github.com/zipeizhao/PU-learning-for-cell-detection)

**Why orthogonal:** NIS3D teaches crowded embryonic morphology and dense negatives within its fully annotated source volumes; non-negative positive-unlabelled risk can prevent competition fine-tuning from relearning “unannotated = background.” Transfer still requires PSF, intensity, scale, and morphology normalization.

**First experiment:** pretrain on 10–20k NIS3D patches with physical resampling plus measured PSF/intensity/morphology randomization; convert instance masks to centroid Gaussians and weight by annotation confidence. First ablate ordinary sparse fine-tuning. Then separately add nnPU candidate risk over the NMS=1.0 proposal bank. **Go** if each held-out embryo gains ≥5 points of DoG-missed-cell recall and min-fold exact score gains ≥0.005. If March-22 alignment succeeds, NIS3D becomes complementary morphology diversification rather than the primary pretraining source.

**Offline Kaggle:** ship preprocessed patches/weights in a Kaggle Dataset. Expected training is under 12 h on T4×2.

### Raw + FM2S-denoised proposal union

**EV: MED · Novelty: MED-HIGH · Feasibility: HIGH · STATUS: permitted**

FM2S is a fluorescence-specific zero-shot denoiser with about 3.5k parameters and Apache-2.0 code. Use it only as a complementary proposal source; never replace raw-image detections because smoothing can shift or merge nuclei.

- [FM2S paper](https://arxiv.org/abs/2412.10031)
- [Official code](https://github.com/Danielement321/FM2S)

**First experiment:** denoise XY slices for the 20 worst crops, run identical detection on raw and denoised inputs, union only novel candidates, and pass them through conflict arbitration. **Go** at ≥2% recovery of currently missed nodes, median localization degradation <0.5 µm, and exact min-fold +0.002.

**Offline Kaggle:** vendor the tiny codebase and process slices in a streaming notebook. Benchmark projected whole-test runtime before integrating.

### Fused Gromov–Wasserstein neighborhood association

**EV: MED · Novelty: HIGH · Feasibility: MED · STATUS: permitted**

FGW matches points using both features and relational geometry. For crowded nuclei that look identical, a cell's local kNN distance constellation can disambiguate identity even when displacement-only Hungarian costs cannot.

- [FGW paper](https://proceedings.mlr.press/v97/titouan19a.html)
- [Python Optimal Transport](https://github.com/PythonOT/POT)

**First experiment:** only on association-error crops, solve entropic FGW over overlapping 50–150-cell tiles using physical kNN geometry plus intensity/scale features. Add coupling probability to the existing link cost. **Go** at exact min-fold +0.002 with no fold regression >0.001. Do not spend time on it until the detection experiments run; association is the minority failure mode.

**Offline Kaggle:** vendor POT; cache frame-pair couplings. Expected CPU feasibility is acceptable only with spatial tiling.

### Target-volume self-supervised temporal denoising

**EV: MED-LOW · Novelty: MED · Feasibility: MED · STATUS: permitted, rules recheck for transductive fitting**

DeepInterpolation trains from raw noisy sequences without clean targets and reported substantially more recovered neuronal segments and higher SNR. A 2025 microscopy method further adapts denoising to motion with weighted spatiotemporal sampling. This can adapt to each hidden embryo without labels, but motion blur/hallucination risk is real.

- [DeepInterpolation paper](https://www.nature.com/articles/s41592-021-01285-2)
- [Official code](https://github.com/AllenInstitute/deepinterpolation)
- [Motion-aware microscopy denoising](https://openaccess.thecvf.com/content/CVPR2025W/CVMI/papers/Aiyetigbo_Generalizable_Unsupervised_Microscopy_Video_Denoising_via_Weighted_SpatioTemporal_Sampling_CVPRW_2025_paper.pdf)

**First experiment:** train/adapt on one training embryo without labels, score the other embryo, then reverse. Require recovery of true DoG misses and <0.5 µm localization shift. This is later than FM2S because integration and training cost are higher.

### CPD population-flow proposals / local GLMB ambiguity tracking

**EV: LOW-MED · Novelty: MED · Feasibility: MED · STATUS: permitted**

Coherent Point Drift supplies a soft non-rigid population deformation; labelled multi-Bernoulli tracking preserves several short hypotheses around collisions or missed detections. Both should be restricted to ambiguity subsets, never run embryo-wide.

- [CPD paper](https://doi.org/10.1109/TPAMI.2010.46) and [code](https://github.com/siavashk/pycpd)
- [GLMB cell-tracking paper](https://doi.org/10.1109/ICCAIS.2017.8217576)

**First experiment:** fit CPD on high-confidence detections and measure association FN change without changing nodes. Try top-5 short GLMB hypotheses only if CPD/FGW identifies a repeatable collision subset. Expected upside is ≤0.005; these do not interrupt detector work.

## Live-field evidence used as negative space

The July-3 discussion index still shows the field concentrated on DoG, Hungarian/ILP, and small 3D U-Nets. One public rule-based disclosure reports multi-scale DoG at 0.826 and a failed-to-generalize 3D U-Net; this reinforces that temporal evidence accumulation and dense external pretraining are not yet the plateau recipe. A separate shared fold report claims 0.927 aggregate detection recall from a Gaussian-CenterNet detector, but this is competitor-reported and not independently verified. [Rule-based discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716952), [shared technical report thread](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/717109)

The host also clarified that test embryo IDs do not overlap training IDs and test size is roughly similar. That makes embryo-held-out gates non-negotiable and further weakens any plan based on memorizing 44b6/6bba appearance. [Host clarification](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716793)

## Quarantine — do not execute without written host clearance

An exact-code audit found a structural division-scoring loophole: a distant unmatched fork can sit inside a weakly connected predicted component, qualify a GT division as recovered, and avoid division-FP counting because FP counting considers matched predicted division nodes. This is an evaluator defect, not a biological method. It may be technically parse-valid, but it is contrary to the metric's stated intent and carries prize/disqualification risk.

**Decision:** do not submit it. If ever considered, disclose a minimal reproducer privately to the host and obtain written permission first. Do not build a strategy around it.

## What interrupts execution now

1. **Track-before-detect response diagnostic** — hours, not days; immediately tells us whether missed nuclei contain coherent latent signal.
2. **March-22 provenance/alignment test** — conditional jackpot; stop if identity evidence or ten-frame alignment fails.
3. **Prepare NIS3D + nnPU fallback in parallel** — starts the learned-detector path even if March-22 alignment fails.

FM2S is a one-day screen after those. FGW, self-supervised temporal denoising, CPD, and GLMB are later unless the error taxonomy changes.

## One bet

If forced to bet one non-obvious edge **today**, it is **motion-compensated track-before-detect**. It attacks the dominant no-candidate bottleneck, has no unresolved provenance dependency, and can be proved or killed in hours. If the March-22 provenance/alignment gate passes, switch the bet: dense same-modality pretraining has the larger ceiling and remains legal without identifying the private test source.
