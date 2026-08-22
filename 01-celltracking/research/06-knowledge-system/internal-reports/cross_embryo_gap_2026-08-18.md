# Cross-embryo gap forensics: 44b6 vs 6bba — 2026-08-18

Question posed: *is the 44b6 0.9033 / 6bba 0.7042 gap a domain/calibration problem
(intensity, contrast, density, stage) rather than a model-capacity problem, and is closing
it worth more than any post-processing lever?*

**Answer: no, and the premise of the question is partly built on a contaminated number.**
On the only LOEO-clean substrate we hold, the two families' *detection-stage* profiles are
nearly identical, the intensity/contrast domain shift is real but already absorbed by the
pipeline's existing per-crop normaliser, and the detection-threshold lever is worth
essentially nothing on 6bba under the deployed weights. Details, with the counter-evidence
that would overturn each claim, below.

Labels: **[CODE]** read off source with `file:line`; **[MEASURED]** computed this session
against `data/train` / `_evidence/`; **[INFERENCE]** derived, assumptions stated;
**[UNVERIFIED]** could not settle, with what would settle it.

Scratch scripts (not committed, outside Git):
`…/scratchpad/{charA,charB,attrib,sweep,paired,fixes,density}.py`.

---

## 0. The three findings that matter

1. **[MEASURED] Under the deployed primary detector weights, 6bba's threshold-limited
   miss class is 0.1% of GT (10 of 9,604).** The detection-threshold lever is *dead* on
   6bba, not undervalued. Lowering τ from 0.96875 to 0.01 buys +0.10 pp of GT reach for
   +167% nodes.
2. **[MEASURED] Under those same weights the deployed detector scores *worse* on the
   family it trained on (6bba, node recall 0.8637) than on the family it has never seen
   (44b6, 0.8888).** Memorisation does not rescue 6bba. That falsifies "domain shift" as
   the mechanism for the residual 6bba deficit.
3. **[INFERENCE, high confidence] The headline `node_recall 0.9871 vs 0.8695` is inflated
   on the 44b6 side by the all-199-train secondary detector.** On the LOEO-strict arm the
   detection gap between families is **2.5 pp**, not 11.8 pp. See §5 — this is a
   measurement-integrity finding and it bears directly on the broken LOEO→LB instrument.

---

## 1. What the pipeline actually does to the pixels

Established by reading source, because every "normalisation fix" proposal depends on it.

| step | `file:line` | what it does |
|---|---|---|
| frame load | `vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py:215` | `raw = zarr_arr[t, ::dz, ::dy, ::dx].astype(np.float32)` — **no normalisation at load** |
| downsample | same file `:157` | `"downsample": [1, 4, 4]` → native `(1.625, 0.40625, 0.40625)` µm becomes **1.625 µm isotropic**, `(64,256,256)` → `(64,64,64)` |
| quantile source | `:319-324` | `q_low = ds.quantiles["0.001"]`, `q_high = ds.quantiles["0.999"]`, read from **each crop's own zarr attrs** |
| normalisation | `:368-369` | `imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)` — affine, clamp-at-zero, **no gamma, no upper clip** |
| train parity | `scripts/train_unet_transformer.py:334` | identical expression — train and inference agree |
| peak rule | `:283-285` | `is_peak = (logits == pooled) & (sigmoid(logits) > det_threshold)` |
| pool kernel | `:70`, `:229-251`, `:335` | `pool_kernel_um = 3.0` over a 1.625 µm isotropic grid → **(3,3,3) voxels = 1.625 µm suppression radius** |
| deployed τ | `notebooks/kaggle_p0b_clean913_revtime/…ipynb:67`, `notebooks/kaggle_p3_armb/biohub-p3-armb.ipynb:74` | `BIOHUB_DET_THRESHOLD = 0.96875` |

**[CODE] Consequence for the brief's task 3: "per-crop intensity standardisation" is
already the deployed baseline.** The normaliser is per-crop (quantiles live in each
`*.zarr` `image_statistics.quantiles`), applied identically in training and inference. That
lever is spent. What is *not* done anywhere in the path: per-frame renormalisation, gamma
(`src/tracking_cellmot/img_proc.py:19` implements it but the deployed path never calls
`quantile_normalize`), local contrast equalisation, or any cross-family histogram match.

---

## 2. How far apart the two families actually are (all 199 crops)

### 2.1 Raw ADU, from the stored quantile ladder — **[MEASURED]**

Per-crop medians, `n=71` (44b6) / `128` (6bba):

| statistic | 44b6 | 6bba | family AUC |
|---|---|---|---|
| q0.001 | 35.0 | 20.0 | 0.624 |
| q0.1 (background proxy) | 121.0 | 49.0 | 0.729 |
| q0.9 | 1294.0 | 451.9 | 0.847 |
| q0.99 | 1958.3 | 836.1 | 0.862 |
| q0.999 | 2661.0 | 1437.5 | 0.860 |
| **span = q0.999 − q0.001** (the normaliser's denominator) | **2579.3** | **1415.0** | **0.863** |

6bba is roughly **2× dimmer in raw ADU and has 1.8× less dynamic range**. But the
normaliser divides by exactly that span, so most of this is removed before the model.

### 2.2 What survives normalisation — the input the model sees — **[MEASURED]**

Reproducing `:215` + `:369` exactly, 10 frames × 199 crops (`charB.py`):

| normalised statistic | 44b6 | 6bba | ratio |
|---|---|---|---|
| image mean | 0.1940 | 0.0892 | 0.46 |
| image p50 | 0.1552 | 0.0517 | 0.33 |
| image p90 | 0.4252 | 0.2013 | 0.47 |
| image p99 | 0.6804 | 0.5333 | 0.78 |
| **image p99.9** | **0.8803** | **0.8084** | **0.92** |
| voxel fraction > 0.2 | 0.4153 | 0.1012 | 0.24 |
| **normalised value at GT nuclei (median)** | **0.5785** | **0.5276** | **0.91** |
| normalised value at GT nuclei (p10) | 0.5186 | 0.3721 | 0.72 |

Pooled input histograms over the 19 D1-pilot crops (`fixes.py`):
**total-variation distance 0.1621, Kolmogorov distance 0.1555.**

**[MEASURED] The residual shift is in the *bulk/background*, not in the nuclei.** The bright
tail is matched to within 8% (p99.9 0.880 vs 0.808) and the value *at annotated nuclei* to
within 9% (0.579 vs 0.528). The 2–3× differences are all in the low percentiles — i.e. how
much of the 104 µm crop is filled with dim tissue. A histogram match would therefore move
mostly background, which is the part the detector does not key on.

### 2.3 Density — **[MEASURED], and it goes the opposite way to expectation**

Native-resolution local maxima with a physically-sized (~5 µm) suppression window, 3 frames
× 199 crops (`density.py`), two scale-invariant level anchors:

| level | family | peaks/frame (median) | NN median (µm) | implied mean spacing (µm) | crop-level AUC on count |
|---|---|---|---|---|---|
| p95 | 44b6 | 244 | 7.63 | 16.64 | 0.808 |
| p95 | 6bba | **97** | **8.72** | **22.64** | |
| 0.25·p99.9 | 44b6 | 353 | 7.57 | 14.72 | 0.796 |
| 0.25·p99.9 | 6bba | **117** | **8.53** | **21.26** | |

Accepted detector nodes per frame, same direction (`sweep.py`): **44b6 406.4/frame, 6bba
263.2–269.2/frame.**

**6bba's image is *sparser* in nucleus-scale peaks and its peaks are *farther apart*.**
Whatever 6bba's difficulty is, max-pool NMS merging under crowding is not it.

The contrast is annotation, not imaging: GT nodes per frame **44b6 2.14, 6bba 8.35**
(AUC 0.058 — the single most family-separating statistic we measured). GT-to-nearest-GT
distance in the pilot crops: **44b6 median 17.99 µm (p10 9.88), 6bba median 24.51 µm
(p10 14.97, min 5.22); ≤0.3% of 6bba GT pairs are within 7 µm.** So 6bba carries 4× the annotation over a
*sparser* peak field.

### 2.4 Nucleus size ruler — **[MEASURED]**

Mean radial intensity profile around GT centres on the native grid, physical 0.8125 µm bins
(same construction as `scripts/win_bet/h1r_zh001r_audit.py:48,58-88`):

- 44b6: `1.00 0.94 0.82 0.63 0.49 0.32 0.29 0.18 …` → **half-max radius ≈ 3.57 µm**
- 6bba: `1.00 0.95 0.89 0.73 0.64 0.50 0.53 0.39 …` → **half-max radius ≈ 4.47 µm**

**6bba nuclei are ~25% larger in radius (~2× in volume) and the profile does not return to
background inside 8 µm.** Consistent with an earlier developmental stage. This is the one
genuine *biological* family difference we measured that the normaliser cannot touch.

### 2.5 Depth and time — **[MEASURED], weak at family level**

z-attenuation (bottom-quarter p99 ÷ top-quarter p99), crop means: 44b6 median 1.088
[IQR 0.963–1.218], 6bba 1.018 [0.742–1.355]. **Crop-level AUC 0.548 — not a family
separator.** Depth behaviour is per-crop in both families.

Temporal: normalised p90 at t=99 ÷ t=0, per crop — 44b6 median **1.183** (brightening),
6bba median **0.907** (dimming). Wide spread; directional only.

---

## 3. Attributing the misses — the D1 M/T/C/L/D taxonomy

Taxonomy definition: `src/biotrack/d1_partition.py:41-62`
(`M` matched · `T` unmatched with an *unaccepted* local max ≤7 µm · `C` unmatched with an
*accepted* peak ≤7 µm · `L` no local max ≤7 µm but one ≤15 µm · `D` none ≤15 µm).

Four D1 pilot exports exist, all at the deployed τ = 0.96875, `match_authority = pregraph`,
LOEO arm `strict` (secondary + DeepCenter **off**, `scripts/kaggle_edits/loeo_retarget.py:14-25`).
Checkpoint provenance, read off `loeo_retarget.py:16-18,87`:

> *"fold 0 = held-out 44b6 (71 crops); fold 1 = held-out 6bba (128 crops). the support pack
> ships ONLY weights/unet_transformer/split_0 — **trained on 6bba, held out 44b6**. It is
> therefore LOEO-CLEAN on fold 0 and LEAKY on fold 1."*

So **`split_0` is both the deployed primary detector and the 6bba-trained model**, and
`split_1` is our own 44b6-trained checkpoint.

### 3.1 The census — **[MEASURED]**

| arm | checkpoint | family | crops | n_GT | M (recall) | T | C | L | D |
|---|---|---|---|---|---|---|---|---|---|
| `p3_d1_pilot_f0_v1` | split_0 (6bba-trained) | **44b6, never seen** | 9 | 2,366 | 2,103 (**0.8888**) | **0 (0.00%)** | 168 (7.10%) | 95 (4.01%) | 0 |
| `p3_d1_pilot_f0_src_v1` | split_0 (6bba-trained) | **6bba, in training** | 10 | 9,604 | 8,295 (**0.8637**) | **10 (0.10%)** | 700 (7.29%) | 599 (6.24%) | 0 |
| `p3_d1_pilot_f1_src_v1` | split_1 (44b6-trained) | 44b6, in training | 9 | 2,366 | 2,087 (0.8821) | 203 (8.58%) | 62 (2.62%) | 14 (0.59%) | 0 |
| `p3_d1_pilot_f1_v1` | split_1 (44b6-trained) | 6bba, never seen | 10 | 9,604 | 7,173 (0.7469) | 1,098 (11.43%) | 736 (7.66%) | 597 (6.22%) | 0 |

Read the first two rows — **both are the deployed weights**:

> **The deployed detector's miss profile on 6bba (T 0.1%, C 7.3%, L 6.2%) is almost the
> same as on 44b6 (T 0.0%, C 7.1%, L 4.0%). The recall difference is 2.5 pp.**

**[MEASURED] The brief's hypothesis — "if the D1 taxonomy shows 6bba's misses are dominated
by T, the detection-threshold lever is far more valuable than believed" — is falsified for
the deployed configuration. T is 0.1%.** T only appears at scale (11.4%) when a checkpoint
that never saw 6bba is used, which is not the deployed configuration.

### 3.2 The discriminating profile of a 6bba miss — **[MEASURED]** (`attrib.py`)

Raw-image features at each GT node on the detector's own input view. `contrast` =
node value ÷ 11³ local mean; `n_pk12` = image peaks within 12 µm.

Deployed weights on 6bba (`f0_src`, n=9,604):

| class | n | norm. value | local bg | **contrast** | excess | n_pk12 | dist to nearest image peak | z (median) |
|---|---|---|---|---|---|---|---|---|
| M | 8,295 | 0.481 | 0.232 | **1.765** | 0.199 | 5 | 2.82 µm | 31 |
| C | 700 | 0.481 | **0.393** | 1.238 | 0.096 | **8** | 3.63 µm | 30 |
| L | 599 | 0.414 | 0.361 | **1.123** | 0.043 | 7 | **5.14 µm** | 35 |
| T | 10 | 0.121 | 0.120 | 1.006 | 0.001 | 1 | 10.95 µm | 48 |

- **C is not dim.** Its node intensity equals M's (0.481). Its signature is a **50% higher
  local background** and **60% more image peaks within 12 µm** — it sits in denser tissue.
- **L is contrast-poor and has no image peak nearby.** At *native* resolution 83.6% of
  held-out-6bba L nodes have no image peak within 3.5 µm (`fixes.py`, §4.1).
- **T (when it appears) is simply dark**: contrast 1.16, excess 0.027 against M's 0.225.

Depth and time strata, deployed weights on 6bba: recall by z-quartile
**0.877 / 0.868 / 0.848 / 0.861** — flat. By t-quintile **0.853 / 0.865 / 0.869 / 0.876 /
0.854** — flat. **Neither depth nor frame index discriminates under the deployed weights.**
(Under the 44b6-trained checkpoint they do: recall collapses to 0.660 at z48-63 with
T = 19.3%. That gradient is a property of the *checkpoint*, not the data — see §5.2.)

### 3.3 What C actually is — **[MEASURED] + [INFERENCE]**

For every C node the detector **did fire an accepted peak within 7 µm**: median
`n_acc_7um = 1`, and `best7_prob` for the held-out-6bba C class has **min 0.9698, median
0.99927** (`paired.py`) — every one of the 736 is above τ. And it is not GT-pair
competition: only 9.4% of C nodes have another GT within 14 µm (M: 7.8%), and ≤0.3% of 6bba
GT pairs are within 7 µm at all.

**[INFERENCE] C is a graph-construction loss, not a detection loss.** An accepted detector
peak existed within the match radius and the node did not survive into the matched
pre-graph. Candidate mechanisms in the deployed path, none yet isolated: ILP node
selection; `OUTPUT_PRUNE_ISOLATED`; `OUTPUT_MIN_TRACK_LEN = 6`.
**[UNVERIFIED]** — settled by re-running the D1 export with the accepted-peak set dumped
*before* and *after* the ILP and diffing node ids. That is a cheap kernel edit, no retrain.

This reclassifies **7.3% of 6bba's GT (700 nodes) from the detection budget into the
linking budget**, which is consistent with the host's decomposition (6bba loses 0.150 to
linking, 0.146 to detection).

---

## 4. Testing the cheap fixes

### 4.1 Resolution — is the 4× xy stride the bottleneck? **[MEASURED] NO**

`fixes.py` compares image local maxima at native `(1.625, 0.40625, 0.40625)` µm against
the detector's 1.625 µm isotropic view, matched physical suppression, held-out 6bba:

| class | has image peak ≤3.5 µm, **native** | has image peak ≤3.5 µm, **detector grid** |
|---|---|---|
| M | 0.774 | 0.723 |
| C | 0.548 | 0.548 |
| T | 0.377 | 0.289 |
| L | **0.164** | **0.145** |

The stride costs 0–9 pp of image-peak availability, uniformly across classes. **The signal
that L and T need is missing at native resolution too.** Sub-voxel / full-resolution work
cannot recover them. This also rules out max-pool merging: the pool kernel is 1.625 µm and
6bba's peak NN median is 8.5 µm.

### 4.2 Detection threshold — **[MEASURED] the lever is dead on the deployed weights**

Full offline sweep from the raw D1 audit rows (`sweep.py`). GT side: `best7_prob`, the
strongest local max within 7 µm of each GT centre. FP side: `subthr_localmax` rows are a
32-per-frame sample of the `n_subthr_localmax_in_frame` sub-threshold maxima, re-weighted
to the true count.

**Deployed weights (split_0) on 6bba** — reach at τ=0.96875 is **0.9338**:

| τ | GT reach | Δreach | node budget × | extra nodes per GT gained |
|---|---|---|---|---|
| 0.96875 | 0.9338 | — | 1.000 | — |
| 0.90 | 0.9343 | +0.0005 | 1.148 | 7,951 |
| 0.50 | 0.9347 | +0.0009 | 1.325 | 9,729 |
| 0.01 | 0.9348 | **+0.0010** | **2.671** | 44,987 |

**Deployed weights on 44b6**: reach is **0.9582 at every τ from 0.99 down to 0.01** — the
curve is perfectly flat. Nothing to gain.

**44b6-trained checkpoint on held-out 6bba** (the only arm where τ does anything):

| τ | GT reach | Δreach | node budget × | extra nodes per GT gained |
|---|---|---|---|---|
| 0.96875 | 0.8210 | — | 1.000 | — |
| 0.50 | 0.8400 | +0.0190 | 1.099 | 143.5 |
| 0.01 | 0.8667 | +0.0457 | 1.361 | 216.6 |

Even there it buys +4.6 pp of reach for +36% nodes, and it *cannot* reach the 11.4% T class:
of the 1,098 T nodes only **39.8% have `best7_prob > 0.01`** and **56.7% > 1e-3** — median
`best7_prob = 2.37e-3`. Those local maxima are response-poor, not borderline.

**Falsification of "threshold is undervalued": run the τ sweep on the deployed weights and
show a reach gain > 0.5 pp at any node budget under 1.1×. Measured: +0.10 pp at 2.67×.**

### 4.3 Normalisation / histogram matching / CLAHE — **[MEASURED] + [INFERENCE]**

The decisive observation is a control that costs nothing: **both checkpoints consumed
byte-identical input.** Same zarr, same `:215` stride, same `:369` affine, same τ. Yet on
the 1,098 GT nodes the 44b6-trained checkpoint calls T, `best7_prob` is **median 2.37e-3**
under that checkpoint and **median 0.999 (88.0% above τ)** under the 6bba-trained one
(`paired.py`).

**[INFERENCE] The information those nodes need is already present in the input as
normalised.** The bottleneck is the decision function, not the input representation.
Therefore:

| candidate fix | mechanism it would need | status |
|---|---|---|
| per-crop intensity standardisation | — | **already deployed** (`:319-324`, `:369`). Lever spent. |
| histogram match 6bba→44b6 | move the input distribution into the region the weights handle | **[INFERENCE] near-useless.** TV distance is 0.1621 and it is concentrated in the *bulk*; the nuclei already land within 9% (§2.2). And the deployed weights *are* the 6bba-trained ones — there is no 44b6-trained decision surface in the deployed path to match toward. |
| CLAHE / local contrast | raise the 1.12–1.16 contrast of L/T nodes toward M's 1.77–1.89 | **[UNVERIFIED], and train/inference-mismatched.** No local-contrast step exists in either path (`:369`, `train:334`), so applying it at inference alone is out-of-distribution. Only valid as a *retrain* variant. |
| gamma | same | `img_proc.py:19` implements it but the deployed path never calls it. Same retrain-coupling. |
| per-frame renormalisation | absorb the 9% median temporal decline in 6bba (§2.5) | untested; effect size bounded by 9% against a residual family shift of 0.16 TV. **[INFERENCE] too small to matter.** |

---

## 5. The measurement-integrity finding

### 5.1 The headline gap does not reproduce on the clean substrate — **[INFERENCE, high confidence]**

| substrate | 44b6 node recall | 6bba node recall | gap |
|---|---|---|---|
| deployed (host, 199 crops, secondary + DeepCenter on) | 0.9871 | 0.8695 | **11.8 pp** |
| D1 pilot, LOEO arm `strict`, deployed primary alone (19 crops) | 0.8888 | 0.8637 | **2.5 pp** |

The two configurations differ by: the `unet_transformer_alltrain_seed314159_v1` secondary
(`"train_datasets": 199` — it has seen every train crop,
`loeo_retarget.py:19-21`), DeepCenter, and the wrapper's synthetic node additions. Adding
them lifts 44b6 by **~9.8 pp** and 6bba by **~0.6 pp**.

**[INFERENCE] The asymmetry is a memorisation-capacity artefact, not a capability.** 44b6 is
71 crops × ~284 GT ≈ 20k annotated nodes; 6bba is 128 crops × ~884 ≈ 113k. A model that
memorised all 199 crops recovers a far larger fraction of the smaller set.
**Assumption stated:** the pilot is 19 of 199 crops. §5.3 bounds that.

**Falsification:** run the deployed `strict` arm (secondary off) over all 199 crops and
report node recall per family. If the gap stays near 11.8 pp, this inference dies and the
detection gap is real. **Cost: one GPU kernel, no retrain, no submission.** This is the
single highest-value next experiment in this report.

### 5.2 Each checkpoint carries its own depth bias — **[MEASURED]**

T-class rate by z-quartile:

| checkpoint / family | z0-15 | z16-31 | z32-47 | z48-63 |
|---|---|---|---|---|
| split_1 (44b6-trained) on **held-out 6bba** | 11.2% | 6.1% | 9.5% | **19.3%** |
| split_1 on its own 44b6 | **14.8%** | 11.3% | 7.8% | 1.3% |
| split_0 (deployed) on 6bba | 0.0% | 0.1% | 0.1% | 0.2% |
| split_0 on held-out 44b6 | 0.0% | 0.0% | 0.0% | 0.0% |

Opposite gradients on the two families, from the *same* checkpoint — a textbook learned-prior
signature — and **entirely absent from the deployed weights**. Reported because it is the
cleanest cross-family transfer artefact we have; it is not a deployed problem.

### 5.3 Pilot representativeness — **[MEASURED]**

The 19 pilot crops vs the rest of their family (medians): 44b6 `n_gt` 209 vs 215, `img_p90`
0.509 vs 0.406; 6bba `n_gt` 922 vs 814, `img_p90` 0.299 vs 0.198, `n_peaks` 358 vs 211. The
pilot crops are **slightly brighter and slightly denser than family average in both
families** — i.e. mildly easier, and mildly *pro*-44b6-looking. This does not explain a
9.8 pp lift.

---

## 6. Conclusions, and the falsification test for each

**The gap is not fixable by normalisation or calibration.** [MEASURED, §2.2 §4.3] The
per-crop affine normaliser is already deployed and already aligns the bright tail (8%) and
the value at nuclei (9%). The residual shift is in the background bulk. Both checkpoints saw
identical input and differed by 2.5 orders of magnitude in response on the same nodes —
the input is not the bottleneck.
*Falsification:* apply CLAHE or histogram matching **as a retrain variant** and show
held-out node recall on 6bba above 0.8637. Cost: one full retrain (~H1 scale). **Not
recommended** — the deployed weights already train on 6bba and top out at 0.8637.

**The gap is not a threshold effect.** [MEASURED, §4.2] T = 0.1% under deployed weights;
full τ sweep to 0.01 yields +0.10 pp for +167% nodes.
*Falsification:* stated in §4.2. Cost: zero, already run offline.

**The gap is not a resolution or NMS effect.** [MEASURED, §4.1, §2.3] The missing image
peaks are missing at native resolution too; 6bba's peak field is *sparser* than 44b6's
(NN 8.5 vs 7.6 µm) and the pool radius is 1.625 µm.
*Falsification:* a native-resolution detector recovering >5 pp on the L class. Our measured
headroom is 1.9 pp (0.164 → 0.145 availability difference applied to 6.2% of GT).

**Retraining (H1) will not close it either — the deployed weights already trained on 6bba
and still score 0.8637 there, below the 0.8888 they achieve on a family they have never
seen.** [MEASURED, §3.1] More 6bba-like training data is not the lever.
*Falsification:* a checkpoint reaching >0.90 node recall on 6bba at ≤1.1× node budget.

**What the residual 13.5% actually is:** C = 7.3% (accepted peak fired, node lost before
the matched graph — a **linking/pruning** loss, §3.3) and L = 6.2% (no local maximum within
7 µm at any threshold under either checkpoint, and no image peak within 3.5 µm at native
resolution in 83.6% of cases — **structurally unreachable by the current detector head**).

### Ranked next experiments

1. **[§5.1] Deployed-`strict` node recall over all 199 crops, per family.** Settles whether
   the 11.8 pp headline gap is real or a leakage artefact. One GPU kernel, no retrain, no
   submission. Everything downstream depends on the answer.
2. **[§3.3] Dump the accepted-peak set before and after the ILP in the D1 export and diff
   node ids.** Converts 7.3% of 6bba's GT from an unexplained miss into a named,
   attackable post-processing loss. Kernel edit only.
3. Do **not** spend on threshold sweeps, histogram matching, per-crop standardisation, or
   sub-voxel/native-resolution detection for the purpose of closing this gap. Each is
   measured dead above.

### Caveats

- **[UNVERIFIED]** All D1 numbers are 19 of 199 crops (9 × 44b6, 10 × 6bba), `strict` arm,
  `match_authority = pregraph`. §5.3 bounds the selection effect; the *paired* contrasts
  (§3.1 rows 1–2, §4.3) compare identical row sets and are immune to it.
- **[UNVERIFIED]** The C-class mechanism (§3.3) is inferred from "accepted peak existed,
  node absent from the matched graph". The specific stage that drops it is not isolated.
- Division-adjacent GT nodes are recalled worse in 6bba (0.778 vs 0.864 overall, C-heavy)
  but n=36 in the pilot — **under-powered, directional only.**
- The `frac` and `contrast` image features use a crude percentile-anchored peak finder; they
  are used only for *relative* comparison between classes on identical images, never as
  absolute nucleus counts.
