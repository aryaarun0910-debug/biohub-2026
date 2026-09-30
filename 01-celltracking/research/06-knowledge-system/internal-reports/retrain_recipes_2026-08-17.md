# Retrain recipe frontier (agent report) — 2026-08-17

> **EDITOR'S NOTE (host session, same day). Defects F1/F2/F5/F6/F7 were INDEPENDENTLY VERIFIED
> in the vendored source — this report's core mechanism claims are confirmed, not merely asserted:**
> - **F1** effective `det_neg_weight = 1e-2` — confirmed at `train_unet_transformer.py:1239`
>   (argparse default) and `:1015` (train signature). NB the *inner* defaults at `:532`/`:788` are
>   `0.1`; the caller overrides them, so 1e-2 is what actually runs. A shallow grep reads 0.1 and
>   is wrong.
> - **F2** `score = test_acc * test_recall` — confirmed at `:1180`. No precision term.
> - **F5** `F.binary_cross_entropy(probs, ...)` — confirmed at `:64` (raises under autocast).
> - **F6** division upweight is a literal no-op — confirmed at `:68-70`
>   (`weight = torch.ones_like(loss)` then `weight[div_rows] = 1.0`).
> - **F7** `coords[:, 1:] *= ds_arr` with no grid-centre term — confirmed at
>   `predict_unet_transformer.py:495`.
>
> **The report's "highest-value zero-GPU pre-flight" (verify the packaged crops' scale) is ALREADY
> CLOSED**, by the nuclear-diameter + NN-distance method it recommends: `kkunizaw/biohub-zh001r`
> measures at **1.677 µm/voxel isotropic, 1.032× our deployed 1.625 µm grid** (see
> `../experimental-records.md` 2026-08-17 and `scripts/win_bet/h1r_zh001r_audit.py`).
> This report's complementary fact — our own level-1 store is `scale_zyx = [0.62, 0.2195, 0.2195]`
> µm (verified in `data/external/zebrahub/imaging/ZSNS003_L1.zarr` attrs) — means reaching the
> deployed grid from *our* stream needs a real resample (2.6× z, 7.4× xy), whereas the packaged
> crops arrive near-deployment-ready. **Counterweight:** the packaged nodes carry no track identity,
> so they serve the DETECTOR half only; the edge half still needs our stream.
> Remainder is unmodified raw agent output.


Mandate: the **training recipe** for the Zebrahub detector/edge retrain (`bet-zebrahub-retrain`),
under Kaggle T4×2 / 12 h-per-session / ≥30 h-per-week. Not a literature tour — a build order.

Deliberately does **not** re-derive: the competitive read (`competitive_frontier_2026-08-16`,
`competitive_refresh_2026-08-17`), the post-processing/linker mechanisms
(`methods_frontier_2026-08-16`, `novel_crossdomain_2026-08-17`), the CPU config archaeology
(`quickwins_internal_2026-08-17`), or the substrate/kill audit (`redteam_blindspots_2026-08-17`).
Where those reports named a lever (PU learning, TTA/SSL, multi-seed ensembling, sub-voxel refine),
this report supplies only the **training-side implementation and its cost**, and says so.

Evidence labels used throughout:
**[DOC]** = documented in cited literature · **[MEAS]** = measured by me this session on our repo/data,
command given · **[INF]** = my arithmetic/inference · **[UNVERIFIED]** = no source found.

---

## 0. The five facts that set the whole recipe (all [MEAS], all this session)

I read the actual training code we would be retraining (`vendor/kaggle-cell-tracking/`, BSD-3-Clause,
© 2026 Thibaut Goldsborough) and measured our own data. Five structural facts dominate every
recipe decision below, and four of them are one-line fixes.

**F1 — The detection loss gives the background exactly 1 % of the gradient mass, by construction.**
`train_unet_transformer.py::compute_detection_loss` sets `w_pos = 1/n_pos` on GT voxels and
`w_neg = neg_weight/n_neg` elsewhere, then reduces with `sum`. That is algebraically
`L_det = mean_pos(BCE) + neg_weight · mean_neg(BCE)`, and the deployed `--det-neg-weight` default is
**1e-2**. So background suppression carries 1 % of the loss. **[MEAS]** (read the code; the
normalisation makes it exact and independent of `n_pos`.)

**F2 — Checkpoint selection has no precision term.** `train(...)` keeps the checkpoint maximising
`score = test_acc * test_recall`, where `test_recall` is `gt_matched/gt_total` (one-to-one greedy,
so ≤ 1) and `test_acc` is edge-classification accuracy on the masked entries only. Nothing in the
selection score falls when the detector emits more junk. **[MEAS]**

F1 + F2 are two *independent* structural pressures toward over-detection. They are the mechanical
explanation of the field's own diagnosis — "detector finds real cells but produces an enormous junk
candidate pool" (#734604) — and of our `bet-learned-ranker` premise that the failure is precision,
not recall. **[INF]**

**F3 — The competition detector is trained on 2.8–8.8 positive voxels per 262,144-voxel volume.**
Measured over all 199 crops:

| family | crops | annotated GT nodes | per-frame mean | median | max | frames with 0 nodes |
|---|---:|---:|---:|---:|---:|---:|
| 44b6 | 71 | 20,197 | **2.84** | 2 | 16 | **10.0 %** |
| 6bba | 128 | 113,121 | **8.84** | 8 | 33 | 2.0 % |

Positive rate ≈ 1.1e-5 – 3.4e-5 of voxels on the deployed 64³ grid. 10 % of 44b6 frames are
**all-negative** samples in which every real nucleus is labelled background. **[MEAS]**
(`zarr.open_group('data/train/*.geff')['nodes']['props']['t']['values']`.)

**F4 — Zebrahub gives ~100–300× more positives per volume and ~50–100× more divisions.**
`data/external/zebrahub/ZSNS003_tracks.csv`: 4,057,611 nodes over 515 frames = **7,879 nodes/frame**
(whole embryo), 169,108 tracks, **31,736 division daughters** in one embryo. The packaged Kaggle
crop set records ~900 nodes/frame *per 64³ crop* (`research/04-data/data-acquisition.md`) vs our
**2.84 / 8.84**. Competition total divisions ≈ 304. **[MEAS]**
This — not "more data" in the abstract — is the mechanism by which the retrain breaks the plateau:
**dense labels are what let you turn F1's background weight up without punishing true nuclei.**
On our own data you cannot raise `det_neg_weight`, because 97 % of real nuclei live in the negative
set. On Zebrahub you can. **[INF]**

**F5 — AMP is currently *blocked by an exception*, not merely unused.** There is no `autocast`,
no `GradScaler`, no LR scheduler and no gradient checkpointing anywhere in the training script;
`lr=1e-4` AdamW is constant for 50 epochs. And `compute_loss` calls
`F.binary_cross_entropy(probs, target)` on post-softmax probabilities — PyTorch documents that
`binary_cross_entropy` / `BCELoss` **"raise an error in autocast-enabled regions"**
([PyTorch AMP docs](https://docs.pytorch.org/docs/2.13/amp.html)). **[MEAS] + [DOC]**
The detection loss uses `binary_cross_entropy_with_logits`, which is on the autocast-to-fp32 list
and is safe. So enabling AMP costs exactly one wrapper: compute the edge loss inside
`with torch.autocast(enabled=False):` on `logits.float()`.

Two secondary [MEAS] findings, both one-liners:

- **F6 — the division upweight in the edge loss is a no-op.** `compute_loss` computes
  `div_rows = target.sum(dim=1) > 1` then executes `weight[div_rows] = 1.0` into an
  already-`ones_like` tensor. Dividing rows get **no** extra weight.
- **F7 — a systematic half-cell centroid bias exists, and it is smaller than feared.**
  `predict_unet_transformer.py:495` does `coords[:,1:] *= ds_arr` with no `+0.5` grid-centre term,
  then `.astype(np.int16)`. GT coords in the geffs are **int64 full-resolution voxel indices**
  ([MEAS]: `integer_frac 1.000` on z/y/x), and `downsample=(1,4,4)`, so **z carries zero
  quantisation error** and only y/x are floored by 4. Error budget: per-axis residual
  ∈ {0, 0.406, 0.813, 1.219} µm uniform → **systematic bias 0.609 µm in each of y and x
  (‖bias‖ = 0.861 µm), RMS radial error 1.075 µm**. An optimal constant shift of +1.5 full-res
  voxels in y and x zeroes the bias and takes RMS radial to **0.642 µm**. **[INF]** from [MEAS] inputs.

  This *de-risks the running sub-voxel lane rather than duplicating it*: against the community-measured
  cliff (σ = 1.5 µm free, 2.5 µm → −16 %, recorded in `competitive_refresh_2026-08-17` §CW1) we are at
  1.075 µm, i.e. **already inside the free zone**. Expect a small gain from the constant shift and
  **near-zero additional gain from a learned offset head**. Size the lane accordingly.

---

## 1. THE RECIPE WE SHOULD ACTUALLY RUN

Ranked by (expected value × cheapness). "GPU-h" = Kaggle T4×2 session-hours. Every item has a
one-line falsification. Items marked *(free rider)* cost no additional GPU time because they ride
inside another item's run.

| # | Lever | GPU-h | Expected effect | Falsification (one line) |
|---|---|---:|---|---|
| **A1** | Unblock + enable AMP fp16; measure s/step | 0.3 | gates whether ANY retrain fits 12 h | log `t_data/t_forward/t_backward` for 50 steps with and without `autocast`; if speed-up < 1.3× the retrain is I/O-bound, go to A2 first |
| **A2** | Pre-materialise the 64³ training tensor as one uint8 memmap | 0 (CPU) | removes strided-zarr I/O on 4 vCPU | the script already prints `t_data` vs `t_forward`; if `t_data` < 20 % of step time, skip A2 |
| **A3** | Fix F2: select checkpoints on precision-aware score | 0.2 | stops selecting the junkiest detector | re-score two checkpoints (best-`acc*recall` vs best-`F1`) with `scripts/core/score_oof.py` LOEO; if `acc*recall` wins on both families, F2 is not binding |
| **A4** | Fix F7: add **+1.5 full-res voxels in y and x** (= `(coords + 0.375) * ds`) at export, drop the `int16` cast | 0 | RMS radial 1.075 → **0.642 µm**, small metric gain | re-export existing OOF detections with the shift and re-score LOEO both directions; require bilateral ≥ 0 |
| **A5** | Fix F6: real division weight in the edge loss | *(free rider on B1)* | division-edge recall | ablate `λ_div ∈ {1, 3, 11}` inside B1's run; read division_jaccard per family |
| **B1** | **Dense-Zebrahub pretrain → competition fine-tune, initialised from the public 50-ep checkpoint** | **8–12** | the bet | LOEO both directions on the patched scorer vs the P3 anchor (44b6 0.759549 / 6bba 0.648965); require bilateral ≥ +0.005 |
| **B2** | Gaussian heatmap target + penalty-reduced focal + raise `det_neg_weight` **on the dense stage only** | *(free rider on B1)* | the actual FP-suppression mechanism | 2-arm ablation inside B1: single-voxel/1e-2 vs Gaussian-focal/1e-1; read node-count ratio `N_pred/N_est` and adj-edge-J separately |
| **B3** | Augmentation upgrade (nnU-Net v2 intensity set + in-plane rotation) | *(free rider on B1)* | cross-embryo generalisation | hold out a Zebrahub embryo; measure detection F1 on it with/without |
| **B4** | Cosine/poly LR decay + `--max-iters` budget | *(free rider on B1)* | convergence in fewer epochs | same wall-clock budget, constant-LR vs poly; compare best val F1 |
| **C1** | BN-statistics recalibration (AdaBN) at inference on the target embryo | 0.5 | free domain adaptation; BN3d is used throughout | recompute BN running stats on the target volumes with `model.train()` + `torch.no_grad()`, no gradient step; re-score LOEO |
| **C2** | Ignore-mask / nnPU loss for the **competition** fine-tune stage | *(free rider on B1)* | stops the fine-tune re-teaching "97 % of nuclei are background" | 3-arm: plain BCE vs ignore-radius-mask vs nnPU; cross-embryo detection precision at fixed recall |
| **C3** | Surgical fine-tuning: which blocks to unfreeze | 1.5 | possibly matches full FT at a fraction of the cost | 4 arms (first-block only / decoder+head only / norm-params only / full) at equal steps; LOEO |
| **C4** | 2-seed logit blend of retrained detectors | +1× B1 | the one un-swept public lever (per `redteam` Claim 3) | only after B1 clears; blend two seeds' det logits, re-score LOEO |
| **C5** | Teacher pseudo-labelling of unannotated competition nuclei | 6–10 | high variance, confirmation-bias risk | pseudo-label 44b6 with a 6bba-trained model, retrain, score on 44b6; if the gain does not survive the *reverse* direction, kill |

### Prescriptions

**A1 — Unblock AMP.** Wrap the edge loss:
`with torch.autocast('cuda', enabled=False): loss = compute_loss(logits.float(), target)`, put
`autocast(dtype=torch.float16)` around `model.encode(...)` and `predict_edges`, add `GradScaler`.
**Use fp16, not bf16**: the T4 is Turing / compute capability 7.5 and has fp16 tensor cores but
**no bf16** (bf16 requires CC ≥ 8.0) **[DOC]** ([NVIDIA Turing whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf); [T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)). `conv3d` autocasts to fp16;
`softmax`, `sum` and `binary_cross_entropy_with_logits` autocast to fp32, so the loss numerics stay
safe **[DOC]** ([PyTorch AMP](https://docs.pytorch.org/docs/2.13/amp.html)). Mixed-precision training
with loss scaling is the standard result of Micikevicius et al., *Mixed Precision Training*, ICLR 2018,
[arXiv:1710.03740](https://arxiv.org/abs/1710.03740) **[DOC]**.

**A2 — Kill the dataloader.** Kaggle T4×2 gives **4 CPU cores / 29 GB RAM / 12 h per session /
≥30 h per week** **[DOC]** ([Kaggle notebook hardware](https://www.kaggle.com/product-feedback/361104),
[weekly quota](https://www.kaggle.com/general/108481)). The script defaults `--num-workers 8` —
oversubscribed 2:1 — and each `__getitem__` does a *strided* read `raw[t:t+W, ::1, ::4, ::4]` from
chunked zarr, which decompresses whole chunks to keep 1/16 of the voxels. Materialise once:
199 crops × 100 t × 64³ = 5.22e9 voxels = **5.2 GB as uint8 / 10.4 GB as uint16** **[INF]** — fits
RAM and a Kaggle dataset. Precedent that this works: the third-party `kkunizaw/biohub-zh001r`
Zebrahub pack is exactly this (`(72,20,64,64,64) uint8`). Set `--num-workers 3`.

**A3 — Selection criterion.** Replace `score = test_acc * test_recall` with a precision-aware score
computed at the *deployed* `det_threshold`, e.g. detection F1 × edge accuracy, or (better) the
scorer's own node-count ratio as a penalty. F2 is the reason the public checkpoint over-proposes;
retraining without fixing it reproduces the defect on new data. **[INF]**

**B1 — The retrain itself, in three stages, one checkpoint chain.**

1. *Stage 0 (init).* Load the public 50-epoch UNet with `--unet-weights` (`strict=False` is already
   supported). **Do not train from scratch.** With ~1e4 windows and ≤ 12 h this is not a close call;
   see §6.
2. *Stage 1 (dense).* Train on packaged Zebrahub crops with **dense** Ultrack nodes. This is the
   only stage where B2's raised background weight is legitimate, because the labels are (near-)complete.
3. *Stage 2 (sparse fine-tune).* Fine-tune on the 199 competition crops with C2's ignore-mask/PU
   loss and `det_neg_weight` returned to ~1e-2, so the fine-tune does not undo Stage 1's
   FP-suppression by re-teaching that unannotated nuclei are background.

**Batch composition matters and is nearly free:** interleave Stage-2 batches with a fraction of
Stage-1 data rather than fine-tuning purely on the sparse set. Arazo et al. found that *"setting a
minimum number of labeled samples per mini-batch"* is one of the two effective regularisers against
confirmation bias in self-training **[DOC]**
([arXiv:1908.02983](https://arxiv.org/abs/1908.02983), IJCNN 2020) — the same mechanism protects a
BatchNorm network from having its running statistics captured by one domain **[INF]**.

**Compute arithmetic for B1 [INF] — replace with A1's measurement before committing.**
TemporalUNet3D `[32,64,128]` on 64³ ≈ 1.18e11 FLOPs/volume forward; batch 16 × window 2 = 32 volumes;
×3 for forward+backward ⇒ **≈ 11.3 TFLOP/step**. One fold ≈ 159 crops × 99 windows ≈ 15.7k windows
≈ 984 steps/epoch.

| precision | assumed effective TFLOPS (2×T4) | s/step | min/epoch | 50 epochs |
|---|---:|---:|---:|---:|
| fp32 (today) | ~4.0 | ~2.8 | ~46 | **~38 h — does not fit** |
| fp16 + AMP | ~19 | ~0.6 | ~10 | **~8 h — fits one session** |

Conclusion: **AMP is not an optimisation, it is the feasibility gate.** If A1 measures < 1.3×,
the run is I/O- or Python-loop-bound (the script has per-sample Python loops in
`compute_batch_loss`, `compute_detection_loss` and `detect_and_match`, plus a
`torch.cuda.synchronize()` per iteration) and A2 + batching those loops becomes the higher lever.
Either way, **budget the run with `--max-iters` and checkpoint-resume across two sessions** rather
than betting a 12 h wall on one epoch count.

---

## 2. Sparse / incomplete annotation — what actually works

Our regime (F3) is the extreme end of "sparsely annotated object detection": ~2.8 % of instances
labelled, 10 % of 44b6 frames with zero labels.

**What the baseline already does right.** The *edge* loss is already a proper ignore-region loss:
`mask = active_rows | active_cols` supervises only matrix entries touching an annotated cell and
ignores the rest. Because annotation is per-*track*, an annotated source's true successor is also
annotated, so the union mask is defensible; its residual failure mode is when the true successor is
*undetected*, which teaches an all-zero row. **[MEAS] + [INF]**

**What it does wrong.** The *detection* loss has **no ignore mechanism at all** — every non-GT voxel
is a negative at weight `1e-2/n_neg` (F1). Under 2.8 % annotation that is ~97 % of true nuclei
pushed toward background. The host's choice of `neg_weight = 1e-2` is best read as a *deliberate
mitigation* of exactly this — it makes the wrong labels cheap — at the price of a detector that
barely suppresses background. **[INF]**

Ranked options, all cheap to implement:

1. **Ignore-radius masking (do this first).** Zero the loss weight in a small ball around every
   *detected but unmatched* peak, i.e. convert likely-unlabelled-true-positives into ignore regions
   rather than negatives. This is the "convert unlabeled instances into ignored regions" family;
   surveys of SAOD describe it as the standard first-line treatment alongside gradient re-weighting
   **[DOC]** ([Sparsely Annotated Object Detection survey material](https://arxiv.org/html/2408.16247);
   [MonoSAOD](https://arxiv.org/html/2604.01646v2)). Cost: ~20 lines in `compute_detection_loss`.
2. **Background Recalibration Loss (BRL).** Re-calibrates the background loss signal for one-stage
   detectors, treating unlabelled instances as hard negatives whose loss is recalibrated rather than
   fully applied — Zhang et al., *Solving Missing-Annotation Object Detection with Background
   Recalibration Loss*, ICASSP 2020, [arXiv:2002.05274](https://arxiv.org/abs/2002.05274) **[DOC]**.
   **Effect size unavailable**: I could not extract the mAP-vs-sparsity table (arXiv HTML is
   abstract-only; the PDF is image-encoded). The paper claims "outperforms the baseline and other
   state-of-the-arts by a large margin" on curated VOC/COCO — treat as mechanism, not magnitude.
3. **nnPU loss.** Already carried as `novel_crossdomain_2026-08-17` #3 for the *CPU re-scorer*; the
   same correction applies to the *detector head*. Zhao et al., MICCAI 2021,
   [arXiv:2106.15918](https://arxiv.org/abs/2106.15918); binary+multi-class extension MELBA 2022,
   [arXiv:2302.08050](https://arxiv.org/abs/2302.08050); base method Kiryo et al., NeurIPS 2017.
   **Effect size unavailable from the abstracts** — the paper states it "improves the performance of
   cell detection given incomplete annotations" without a number I could verify. It needs a class
   prior π; ours is estimable directly as annotated/total ≈ 0.028 **[MEAS-adjacent]**.
   **Honest ranking: nnPU is the *most principled* and the *least cheap*, because a mis-estimated π
   is worse than an ignore mask.** Run it as arm 3 of C2, not as the default.
4. **Self-training / pseudo-labelling.** See §7. It is the highest-variance option and belongs last.

**The strategic point that outranks all four:** with dense Zebrahub supervision (F4) the PU problem
*disappears for the pretraining stage*. The cheapest correct answer to "how do you train a detector
when 2.8 % of objects are labelled" is **don't** — pretrain where the labels are dense, then fine-tune
under an ignore mask. **[INF]**

---

## 3. Point supervision — heatmaps, sigma, offsets

The baseline's detection target is a **single voxel** set to 1.0 (F1). Every point-supervised
detector in the modern literature uses a *spread* target instead. Two directly citable recipes:

**CenterNet** (Zhou, Wang, Krähenbühl, *Objects as Points*,
[arXiv:1904.07850](https://arxiv.org/abs/1904.07850)) **[DOC]**:
- Gaussian target `Y = exp(-((x-p̃)²+(y-p̃)²)/2σ_p²)`, σ_p **object-size-adaptive** (radius rule
  inherited from CornerNet, Law & Deng, ECCV 2018, [arXiv:1808.01244](https://arxiv.org/abs/1808.01244)).
- Penalty-reduced pixelwise logistic regression (the "CornerNet focal"): positives weighted
  `(1-Ŷ)^α log Ŷ`, negatives `(1-Y)^β Ŷ^α log(1-Ŷ)`, with **α = 2, β = 4 in all experiments**.
  Crucially, `(1-Y)^β` *down-weights negatives near a true centre* — the exact behaviour we want for
  near-duplicate suppression without punishing a slightly-off peak.
- **Local offset head**, regressing `p/R − p̃` with **L1 loss**, shared across classes, explicitly to
  recover output-stride discretisation error; loss weights **λ_off = 1, λ_size = 0.1**.
- **No ablation isolating the offset head is reported in the paper** — I checked Table 3 and it
  ablates resolution, size weight, regression loss and schedule, not the offset. So "offset heads
  help" is a *design argument*, not a measured effect size, in this source.

**Spotiflow** (Dominguez-Mantes, Weigert et al., *Spotiflow: accurate and efficient spot detection
for fluorescence microscopy with deep stereographic flow regression*, **Nature Methods 22:1495–1504,
2025**, [doi:10.1038/s41592-025-02662-x](https://www.nature.com/articles/s41592-025-02662-x);
code [weigertlab/spotiflow](https://github.com/weigertlab/spotiflow), **BSD-3-Clause**) **[DOC]**.
This is the closest published tool to our task (subpixel point detection in 3D fluorescence) and its
defaults are directly transplantable. From `spotiflow/model/config.py`:

| Spotiflow default | value |
|---|---|
| heatmap σ | **1.0** (output-grid units) |
| heatmap loss | **BCE**, `pos_weight = 10.0` |
| subpixel head loss | **L1** (stereographic flow) |
| multiscale levels | 4 |
| optimizer / LR | **AdamW / 3e-4** |
| schedule | ReduceLROnPlateau, patience 10 |
| epochs / batch | 200 / 4 |
| 3D | `is_3d` flag; `grid` tuple carries anisotropy; `crop_size_depth = 32` |

Note the contrast with our baseline: Spotiflow uses `pos_weight = 10` on an otherwise **unweighted**
BCE, i.e. background carries ~1/10 of positive mass per voxel-pair — versus our **1/100 of total
loss** (F1). **[INF]** The prescription for B2 is to move toward the Spotiflow/CenterNet regime *only
on the dense stage*.

**Sigma choice — the literature genuinely disagrees.** The landmark-localisation community has run
the ablation we would otherwise run ourselves: fixed σ from 3 to 20 produces > 0.5 mm swings for some
landmarks, with **no clear monotone relation between σ and accuracy**, and per-landmark optima that
differ — motivating *learnable* σ (Thaler, Payer, Urschler, Štern, *Modeling Annotation Uncertainty
with Gaussian Heatmaps in Landmark Localization*, MELBA 2021,
[arXiv:2109.09533](https://arxiv.org/abs/2109.09533)) **[DOC]**. Practical reading for us: **do not
sweep σ — pick σ ≈ 1 grid voxel (Spotiflow's default, ≈ 1.625 µm here, comfortably inside the 7 µm
match radius) and spend the compute elsewhere.** [INF]

**Sub-voxel / offset head: size it with F7, don't assume it.** Our residual localisation error is
1.075 µm RMS with **zero** error in z, against a metric cliff that starts around 1.5–2 µm. The
constant-shift fix (A4) takes it to 0.642 µm for free. A CenterNet-style offset head would recover the
remaining ~0.6 µm — i.e. it is competing for a channel that is nearly closed. **Recommendation:
implement A4; do not add an offset head to B1.** [INF]

---

## 4. Domain adaptation / transfer — external embryo → target embryo

**Geometry first, and it is not free.** Our local Zebrahub level-1 store records
`scale_zyx = [0.62, 0.2195, 0.2195]` µm **[MEAS]**
(`data/external/zebrahub/imaging/ZSNS003_L1.zarr` attrs). The deployed input grid is **1.625 µm
isotropic**. So reaching it needs a **resample by (2.62, 7.40, 7.40) from level-0**, or
(1.31, 3.70, 3.70) from level-1 — *not* a stride. Level-2 is already coarser than 1.625 µm in z and
is unusable. **[INF]** The packaged Kaggle crops (`kkunizaw/biohub-zh001r`, `…/biohub-zmnscrops`) are
pre-resampled by a third party **at an unverified scale** — `data-acquisition.md` already flags this.
**Verify the effective µm/voxel of any packaged pack before it enters B1**, e.g. by matching the
nuclear-diameter histogram to our own crops; a scale error here silently poisons the whole retrain.

**Intensity normalisation — the baseline's choice is already the right one.** Per-video
0.1 %/99.9 % quantile min–max with `clamp(0)` (`load_dataset_windows`, `q_low`/`q_high`) **[MEAS]**.
That is the standard robust percentile scheme and it is the correct default for cross-instrument
transfer; multi-site comparisons find percentile/z-score normalisation to be the reliable general
choice, with **histogram matching preferred specifically when a consistent reference exists**
(reported to cut white-matter intensity variation from 7.5 % to 2.5 % in the MRI setting)
**[DOC]** ([Comparison of Image Normalization Methods for Multi-Site Deep Learning, Applied Sciences
13:8923, 2023](https://www.mdpi.com/2076-3417/13/15/8923); [histogram-based normalisation,
PMC4517549](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4517549/)). **Recommendation: keep the
quantile scheme; add *randomised* histogram matching of Zebrahub crops toward competition crops as an
augmentation, not as a fixed preprocessing step** — the randomised variant is the documented form
that improves generalisation rather than overfitting one reference **[DOC]** (same survey material;
also [Qian et al., histogram-matching-enhanced adversarial UDA, Medical Physics 2025](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.17757)).

**AdaBN is nearly free here and we have the right architecture for it.** `_conv_block` is
`Conv3d → BatchNorm3d → ReLU` throughout **[MEAS]**
(`src/tracking_cellmot/models/temporal_unet.py`). Adaptive Batch Normalization — recompute BN running
statistics on target-domain data, no gradients, no extra parameters — is the canonical parameter-free
domain-adaptation baseline (Li, Wang, Shi, Liu, Hou, *Revisiting Batch Normalization for Practical
Domain Adaptation*, [arXiv:1603.04779](https://arxiv.org/abs/1603.04779)) **[DOC]**. **Effect sizes
not extractable from the arXiv abstract page** — cite the mechanism, measure the magnitude ourselves.
Cost: one forward-only pass over the target volumes with `model.train()` under `no_grad`. This is
C1 and it is the cheapest DA lever we have.

**Test-time SSL / entropy minimisation** is already carried as `methods_frontier_2026-08-16` #4
(SELMA3D, [arXiv:2501.03880](https://arxiv.org/abs/2501.03880); collapse-safe entropy-min). One
training-side addition: SELMA3D is a *challenge report* — 35 large cleared-brain volumes, 84
participants, pretext tasks spanning masked-volume inpainting, BYOL and SimCLR **[DOC]**. It
establishes that SSL pretraining closes an unseen-domain gap in LSM; it does **not** hand us a
recipe with an effect size we can budget against. Treat SSL pretraining as a *short adapter* (see §6
kills), and prefer AdaBN first because it costs 0.5 GPU-h instead of 8.

**Denoising: prefer the joint form, not a separate network.** Noise2Void (Krull, Buchholz, Jug,
CVPR 2019) learns denoising from single noisy images via blind-spot masking **[DOC]**; DenoiSeg
(Buchholz, Prakash, Schmidt, Krull, Jug, ECCV-W 2020,
[arXiv:2005.02987](https://arxiv.org/abs/2005.02987), code [juglab/DenoiSeg](https://github.com/juglab/DenoiSeg))
extends it to predict segmentation *jointly in the same network*, and the documented finding is that
**it outperforms baselines mainly when segmentation ground truth is very limited — down to 10, 2 and 2
annotated images on DSB / Fly Wing / Mouse Nuclei** **[DOC]**. That is exactly our sparse-label
regime, and the joint form costs one extra output channel rather than a second network.
**Recommendation: if we test denoising at all, test it as a DenoiSeg-style auxiliary reconstruction
head on the existing UNet during Stage 2 — never as a preprocessing pass.** [INF]

---

## 5. Augmentation for 3D light-sheet

**The baseline uses two augmentations. Total.** `augmentations.py` provides only
`brightness_augment` (additive shift, ±0.1) and `flip_augment` (the 8 axis-aligned flips) **[MEAS]**.

**The documented reference set** — nnU-Net v2's `get_training_transforms`, verified against source
**[DOC]** ([MIC-DKFZ/nnUNet, `nnUNetTrainer.py`](https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py);
method paper: Isensee et al., *nnU-Net*, **Nature Methods 18:203–211, 2021**,
[doi:10.1038/s41592-020-01008-z](https://www.nature.com/articles/s41592-020-01008-z)):

| transform | p | range |
|---|---:|---|
| SpatialTransform — rotation | 0.2 | ±π/12 |
| SpatialTransform — scaling | 0.2 | 0.7–1.4 |
| GaussianNoise | 0.10 | var 0–0.1 |
| GaussianBlur | 0.20 | σ 0.5–1.0, per-channel p 0.5 |
| MultiplicativeBrightness | 0.15 | 0.75–1.25 |
| Contrast | 0.15 | 0.75–1.25 |
| **SimulateLowResolution** | **0.25** | zoom 0.5–1, per-channel p 0.5 |
| Gamma (inverted) | 0.10 | 0.7–1.5 |
| Gamma | 0.30 | 0.7–1.5 |
| Mirror | — | given axes |

Three things to read off this list:

1. **Elastic deformation is absent.** nnU-Net v2's default 3D pipeline contains **no elastic
   transform** **[DOC]** (same source). For a framework whose entire premise is "the default recipe
   that wins across 20+ 3D datasets", that is the strongest available evidence that elastic
   deformation is *not* worth its cost in 3D medical/biological volumes. **Kill it for us.**
2. **`SimulateLowResolution` at p = 0.25 is the highest-probability *spatial-degradation* transform
   in the set**, and it is precisely the anisotropic-blur/axial-resolution-loss augmentation our
   domain needs: light-sheet degradation worsens with depth, and synthetic PSF/aberration degradation
   is an established training strategy in LSM **[DOC]**
   ([deep-learning aberration compensation](https://www.researchgate.net/publication/387672783_Deep_learning-based_aberration_compensation_improves_contrast_and_resolution_in_fluorescence_microscopy);
   [deep-learning-enhanced LSFM, Light Sci Appl 2025](https://www.nature.com/articles/s41377-024-01710-z)).
   **Take `SimulateLowResolution` and the intensity block (noise, blur, brightness, contrast, gamma)
   essentially verbatim.**
3. **Anisotropy rule for rotations.** nnU-Net restricts spatial augmentation to in-plane when the
   data is anisotropic — explicitly, for ACDC, *"spatial augmentations for the 3D U-Net (such as
   scaling and rotation) are done in-plane only to prevent resampling of imaging information across
   slices"*, and `do_dummy_2d_data_aug` triggers on `max(patch)/patch[0] > ANISO_THRESHOLD` **[DOC]**
   ([arXiv:1904.08128](https://arxiv.org/abs/1904.08128) and the source above).

**The subtle trap in our case [INF].** Our *grid* is isotropic after `downsample=(1,4,4)`, so the
naive conclusion is "full 48-element octahedral symmetry is now available — 6× more augmentation for
free". **That is wrong.** The grid is isotropic; the **PSF is not** — the optical axial resolution is
still far worse than lateral, so a z↔x axis swap produces a volume no microscope could have
produced. Correct group: **4 in-plane 90° rotations × 3 axis flips = 16 elements**, versus the 8
the baseline uses — a 2× enlargement, not 6×. Anything that permutes z with y or x should be
**killed**. Falsification: train two arms (flips-only vs flips + xy-90°) at equal steps and compare
cross-embryo detection F1.

**Also worth adding (cheap, our-domain-specific) [INF, UNVERIFIED as an effect size]:** a
z-dependent SNR/blur ramp (blur σ and noise variance increasing with z index) to mimic depth-dependent
degradation, and randomised histogram matching (§4). Both are one-function additions to
`augmentations.py`. Falsification: hold out one Zebrahub embryo; measure detection F1 with/without.

---

## 6. Small-compute training — what fits, what does not

**Fits inside T4×2 / 12 h.** The memory budget is not the binding constraint. Activation estimate for
`[32,64,128]` at 64³ with B=16, W=2: ~6–8 GB in fp32, ~3–4 GB in fp16, split across two 16 GB cards
**[INF]**. **Time is binding** (§1 table).

Consequences, in order:

- **Gradient checkpointing is the wrong lever here — kill it.** It trades ~30 % extra compute for
  sub-linear memory (Chen, Xu, Zhang, Guestrin, *Training Deep Nets with Sublinear Memory Cost*,
  [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)) **[DOC]**. We are compute-bound with memory
  headroom, so it costs time we do not have to buy memory we do not need. Falsification: print
  `torch.cuda.max_memory_allocated()` after 50 steps; if < 12 GB per card, checkpointing is
  strictly negative. **If** we later want a wider UNet, spend the headroom on channels or batch first.
- **Transfer, don't restart.** nnU-Net's reference schedule is 1000 epochs × 250 iterations =
  **250,000 steps** **[DOC]** (source above). At our best-case 0.6 s/step that is **42 h** — 1.4× our
  entire weekly quota, for one fold. **A from-scratch nnU-Net-scale retrain is dead on arrival and
  should not be proposed again.** Initialise from the public 50-epoch weights via the existing
  `--unet-weights` path.
- **Partial fine-tuning is a live option, not a compromise.** *Surgical fine-tuning* (Lee, Chen,
  Tajwar, Kumar, Yao, Liang, Finn, ICLR 2023,
  [arXiv:2210.11466](https://arxiv.org/abs/2210.11466)) documents that tuning a **subset** of layers
  *matches or outperforms* full fine-tuning across seven real-world distribution-shift tasks, that
  **for image-corruption-type shifts the first few layers are the right ones**, and that tuning more
  parameters on a small target set can erase pretraining **[DOC]**. Our shift (different microscope,
  different intensity statistics, same biology) is input-level, which points at **early layers +
  BN parameters**. Relatedly, LP-FT — linear-probe then fine-tune — is documented to avoid distorting
  pretrained features and underperforming OOD (Kumar, Raghunathan, Jones, Ma, Liang, ICLR 2022,
  [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)) **[DOC]**; the drop-in analogue for us is to
  train the `detect_head` (a `Conv3d(32,1,1)`) alone for a short warm-up before unfreezing the UNet.
  This is C3, and 4 arms at equal steps costs ~1.5 GPU-h.
- **LR schedule.** Constant `1e-4` for 50 epochs is leaving convergence on the table. The two
  documented reference points: nnU-Net's **SGD-Nesterov μ=0.99, lr 1e-2, poly `(1-e/E)^0.9`,
  wd 3e-5** **[DOC]**, and Spotiflow's **AdamW 3e-4 + ReduceLROnPlateau(patience 10)** **[DOC]**.
  Given we are fine-tuning, not training from scratch, **take Spotiflow's side** (AdamW, lower LR,
  plateau or cosine decay) and keep the existing AdamW. Warm restarts (Loshchilov & Hutter, SGDR,
  ICLR 2017, [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)) are an option if we chain sessions.
- **Multi-GPU.** The script wraps only the UNet in `nn.DataParallel`, leaving the detection head and
  transformer on `cuda:0` **[MEAS]**. `DataParallel` replicates the module every iteration and is
  single-process (GIL-bound); `DistributedDataParallel` is the documented recommendation even on one
  machine **[DOC]** (PyTorch DDP documentation — *I could not retrieve the exact quote through the
  docs redirect this session; treat the specific wording as unverified, the recommendation itself is
  long-standing*). **Practical judgement: do not port to DDP inside a Kaggle kernel as part of B1.**
  The 2-GPU DataParallel path is already working and audited; a DDP port risks the run. Revisit only
  if A1's measurement shows GPU-1 idling.
- **Batch size / patch size.** Do **not** switch to random sub-patches. The input is already exactly
  one crop (64³) and both the detection head and the edge transformer consume whole-frame node sets;
  patching would break `detect_and_match`'s within-volume matching semantics. Keep whole-volume
  batches and tune only `--batch-size`. **[INF]**

---

## 7. Edge / temporal model under sparse supervision

**Keep joint training — but fix the weighting.** The baseline already trains detection and edges
jointly (`det_loss_weight = 1.0`) with gradients flowing from the edge head back into the UNet
(our own `h1r_train_smoke.py` verifies `UNet received grad=True`) **[MEAS]**. The closest documented
analogue for "joint helps" is CenterTrack (Zhou, Koltun, Krähenbühl, ECCV 2020,
[arXiv:2004.01177](https://arxiv.org/abs/2004.01177)) **[DOC]**. I found **no source that measures
joint-vs-separate for a detector + association-transformer on 3D microscopy** — so "joint helps here"
is **[UNVERIFIED]**. The cheap falsification is an arm of B1: freeze the UNet for the edge-head
epochs and compare.

**Trackastra is the transplantable recipe** (Gallusser & Weigert, ECCV 2024,
[arXiv:2405.15700](https://arxiv.org/abs/2405.15700), code
[weigertlab/trackastra](https://github.com/weigertlab/trackastra)) **[DOC]**. What to take:

- **Division upweighting**, the fix for F6: Trackastra weights **dividing cells by 1+λ_div = 11×**
  and continuing tracks by 1+λ_cont = 2×. Our code intends this and does nothing.
- **Parental softmax**: normalise so each cell has at most one parent, combined with a secondary
  sigmoid term at **λ = 0.01**. Our `softmax(logits, dim=0)` is already the parental normalisation;
  the auxiliary sigmoid term is not present. **Measured ablation: parental softmax gives ≈ 20 %
  error reduction for dividing objects** **[DOC]**.
- **Temporal window s = 6 frames** with overlapping-window score averaging at inference, versus our
  `window_size = 2`. This is the single biggest *architectural* difference and it is not free —
  it multiplies the UNet forward cost per step. **Do not put it in B1.**
- **Feature set**: learned Fourier positional encodings + shallow morphology (mean intensity, area,
  inertia tensor), **no image crops**, d = 256, batch 8, single consumer GPU **[DOC]**.
- **Generalisation result worth the most to us**: on held-out HeLa, the DeepCell-*specialised* model
  scored AOGM = 190 with ILP while the **general** model trained on many datasets scored **AOGM = 96**
  — "the importance of a large, diverse training set for out-of-domain tracking performance"
  **[DOC]**. That is a direct, measured argument for training the edge head on Zebrahub + competition
  jointly rather than competition alone. It is also the same finding our own `redteam` Claim 3
  reached by elimination.
- Linking ablation: greedy AOGM 36 → ILP 23 on bacteria; DeepCell greedy 11.9 → ILP 7.9 **[DOC]** —
  i.e. a *good* learned edge head still benefits substantially from ILP, which is what we already run.

**Contrastive / metric-learning alternatives.** I found no source showing a contrastive association
embedding beating a supervised pairwise head *for dividing cells* — divisions break the
"same identity ⇒ same embedding" premise, which is also the structural reason
`methods_frontier_2026-08-16` recorded HOCT's non-homophily finding. **Recommendation: do not spend
B1 compute on a contrastive edge objective.** [INF]

---

## 8. Curriculum / pseudo-labelling — and its documented failure mode

**The teacher we would use already exists, and it made our external labels.** Zebrahub's `*_tracks.csv`
are Ultrack outputs (Bragantini et al., *Ultrack: pushing the limits of cell tracking across biological
scales*, **Nature Methods 2025**,
[doi:10.1038/s41592-025-02778-0](https://www.nature.com/articles/s41592-025-02778-0), code
[royerlab/ultrack](https://github.com/royerlab/ultrack)) **[DOC]**, reported at state-of-the-art / "near-perfect"
Cell-Tracking-Challenge accuracy on zebrafish neuromast. So **Stage 1 of B1 is already a
pseudo-label training run** — with a strong, published teacher, on the organisers' own tooling.
State that plainly in any write-up rather than treating Zebrahub tracks as clean GT. **[INF]**

**Documented gains from teacher-student on sparse labels.** Unbiased Teacher (Liu et al., ICLR 2021,
[arXiv:2102.09480](https://arxiv.org/abs/2102.09480)) reports **+6.8 absolute mAP over the prior SOTA
at 1 % labelled COCO, and ≈ +10 mAP over the supervised baseline at 0.5/1/2 % labels** **[DOC]**;
Soft Teacher (Xu et al., ICCV 2021, [arXiv:2106.09018](https://arxiv.org/abs/2106.09018)) weights each
unlabelled box's classification loss by the teacher's score and reports large margins at 1/5/10 %
**[DOC]** (I did not extract Soft Teacher's per-ratio table). Noisy Student (Xie, Luong, Hovy, Le,
CVPR 2020, [arXiv:1911.04252](https://arxiv.org/abs/1911.04252)) is the canonical demonstration that
the student must be **noised** (augmentation/dropout) for self-training to help **[DOC]**.
Our label fraction is 2.8 % — squarely in the band where these methods report their largest gains.

**The failure mode, documented.** Arazo, Ortego, Albert, O'Connor, McGuinness,
*Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning*, IJCNN 2020,
[arXiv:1908.02983](https://arxiv.org/abs/1908.02983): naive pseudo-labelling **overfits its own
incorrect labels**, and the two effective regularisers are **mixup** and **a minimum number of
labelled samples per mini-batch** **[DOC]**. For us this maps to: keep real annotations in every
batch, and never train a round of pseudo-labels without a held-out *reverse-direction* check.

**Our own prior evidence bounds the upside.** `redteam_blindspots_2026-08-17` records
synthetic→real transfer AUC of **0.664** for divisions — i.e. a teacher signal that did not transfer.
That is a warning specifically about *division* pseudo-labels, not about *node* pseudo-labels.
**[MEAS, prior]**

**Verdict — C5, last.** Cost 6–10 GPU-h for one round, and the honest falsification is bidirectional:
pseudo-label 44b6 with a 6bba-trained model *and* the reverse; if the gain does not appear in both
directions, it is confirmation bias, not learning. Do not run C5 before B1 has cleared its bar.

---

## 9. Explicitly killed under our budget

| Killed | Why |
|---|---|
| **From-scratch 3D retrain at nnU-Net scale** (1000 ep × 250 it) | 250k steps ≈ 42 h best case = 1.4× the *weekly* quota, for one fold **[INF from DOC schedule]** |
| **SSL pretraining on full Zebrahub imaging** | one embryo at level-1 is 129×965×1019×515 ≈ **131 GB** **[MEAS from zarr shape]**; cannot be staged, and the pretrain alone exceeds the quota. Use AdaBN (C1) instead; keep masked-volume SSL only as a ≤ 1 h adapter |
| **Gradient checkpointing** | buys memory we already have, costs ~30 % of the time we don't **[DOC + INF]** |
| **Elastic deformation** | absent from nnU-Net v2's default 3D pipeline **[DOC]**; expensive in 3D |
| **z↔xy axis-permutation augmentation** | grid is isotropic, PSF is not — produces physically impossible volumes **[INF]** |
| **Full-resolution (64,256,256) training** | 16× the voxels of the deployed grid; also breaks input compatibility with the deployed detector **[INF]** |
| **Separate denoiser (Noise2Void) as a preprocessing network** | second network + second inference pass; the joint DenoiSeg form gets the documented low-label benefit at ~zero extra cost **[DOC]** |
| **Trackastra's 6-frame temporal window inside B1** | multiplies UNet forward cost per step; park as a later bet **[INF]** |
| **DDP port inside B1** | working audited DataParallel path; porting risks the run for an unmeasured gain **[INF]** |
| **Contrastive/metric-learning edge objective** | no source shows it beating a supervised pairwise head *with divisions* **[INF]** |
| **Joint DETR-style detect+track retrain (Cell-TRACTR)** | already killed in `methods_frontier_2026-08-16` #5; memory-heavy, unproven on T4×2 |

---

## 10. Licence flags

- `vendor/kaggle-cell-tracking` (the training code we would modify) — **BSD-3-Clause**, © 2026
  Thibaut Goldsborough. **[MEAS]** Permissive; safe to patch and ship.
- **Spotiflow** — **BSD-3-Clause** **[MEAS via repo]**. Safe to vendor code *or* reimplement the
  heatmap/flow recipe.
- **Trackastra** — the paper carries only the arXiv non-exclusive licence and states no repo licence;
  **verify [weigertlab/trackastra](https://github.com/weigertlab/trackastra) before vendoring code**.
  The λ_div / parental-softmax *mechanism* is reimplementable from the paper. **[DOC/flag]**
- **Ultrack** — [royerlab/ultrack](https://github.com/royerlab/ultrack); verify licence before
  vendoring. We consume its *outputs* (Zebrahub tracks), which is the host-cleared path.
- **Zebrahub data** — **CC BY-NC**, host-cleared (#734330). Already recorded in
  `research/04-data/data-acquisition.md`; **record the licence in any shipping notebook.**
- **nnU-Net** — Apache-2.0 (MIC-DKFZ). We are copying *augmentation parameter values*, which is not a
  derivative work, but if we vendor `batchgenerators` code, carry the notice.
- **Third-party packaged Zebrahub crops** (`kkunizaw/*`) — a *derivative* of CC BY-NC Zebrahub with
  **unverified resampling scale**. Verify scale before use (§4) and attribute.
- **Freitas synthetic** — CC0, no restriction.

---

## 11. Where the literature disagrees, and what we don't know

1. **Heatmap σ.** Landmark-localisation ablations find **no monotone σ–accuracy relation** and
   per-target optima, arguing for learnable σ **[DOC]**; CenterNet uses a size-adaptive σ; Spotiflow
   fixes σ = 1. Three defensible answers. **Our position: fix σ = 1 grid voxel and don't sweep** —
   the 7 µm match radius is 4.3 grid voxels wide, so the metric is far less σ-sensitive than
   millimetre-precision landmark tasks. **[INF]**
2. **Whether joint detect+link training helps.** No source measured it in our configuration.
   **[UNVERIFIED]** — falsify as an arm of B1, don't assume.
3. **Effect sizes I could not extract**, and which must therefore not be quoted as numbers:
   BRL's mAP-vs-sparsity table (image-encoded PDF); the PU-learning cell-detection F1 deltas
   (abstract-only); AdaBN's benchmark gains (abstract-only); CenterNet's offset-head isolation
   (**not reported in the paper at all**); Spotiflow's flow-head ablation and few-annotation transfer
   numbers (bioRxiv rate-limited this session, and the Nature Methods full text is not open here).
   Each is flagged inline above.
4. **The biggest unknown is not in the literature — it's the packaged Zebrahub scale.** If
   `kkunizaw`'s isotropic resampling is not 1.625 µm, B1 trains on the wrong nuclear size and the
   whole retrain silently fails while looking healthy. **This is the single highest-value pre-flight
   check and it costs no GPU time.** Falsification: compare nuclear-diameter and nearest-neighbour
   distance histograms between a packaged Zebrahub crop and a competition crop; require agreement
   within ~10 %.
5. **Whether F1/F2 are actually load-bearing** is an *inference* from code structure, not a
   measurement. The cheapest test is A3 + B2's two-arm ablation, and both fit inside B1's session.
   If raising `det_neg_weight` on dense data does **not** move `N_pred/N_est` on the deployed
   substrate, the "structural over-detection" story is wrong and the retrain must be re-motivated.

---

## 12. One-paragraph build order

Do A1–A4 first (≈ 0.5 GPU-h total, mostly CPU): unblock AMP, materialise the training tensor, fix the
selection score, fix the half-cell offset, and **verify the packaged Zebrahub voxel scale**. Then run
**B1 as a single chained job**: init from the public 50-epoch weights → dense-Zebrahub stage with
Gaussian-focal targets and raised background weight (B2), nnU-Net intensity augmentations +
`SimulateLowResolution` + in-plane 90° rotations (B3), AdamW with plateau/cosine decay (B4), and a
real division weight (A5) → sparse competition fine-tune under an ignore mask (C2), with Zebrahub
batches interleaved. Gate on LOEO, both embryo directions, patched scorer, bar +0.005 bilateral. Only
if that clears: C1 (AdaBN, 0.5 h), C3 (surgical-FT arms, 1.5 h), C4 (second seed, +1× B1). C5
(pseudo-labelling) is last and must pass a bidirectional check.
