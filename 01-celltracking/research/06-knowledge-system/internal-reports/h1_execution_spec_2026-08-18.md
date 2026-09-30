# H1 execution spec — detector + edge retrain, evaluation design, staged calendar (2026-08-18)

**Status: LIVE DOCUMENT — sections 0–4 and 6 settled; sections 5 (edge-half manufacturing) and 7
(calendar) in progress.** Written to survive session death; re-checkpointed after each milestone.

Evidence labels, used on every claim:
**[MEAS]** measured by me this session (command/file given) · **[DOC]** documented in-repo or in a
cited source · **[VERIFIED-CODE]** read directly in the named file at the named line ·
**[INF]** my arithmetic or inference · **[OPEN]** not established.

---

## 0. Files created by this session

All are **inert** — nothing is pushed, launched, submitted or committed.

| file | what it is | tested? |
|---|---|---|
| `scripts/kaggle_edits/h1r_trainer_patch.py` | 13 exact-string patches to a **copy** of the vendored `train_unet_transformer.py`: AMP unblock, GradScaler, precision-aware checkpoint score, real division weight, sparse-annotation ignore mask, cosine LR, last-checkpoint save | **YES** — applied to a scratch copy (13/13 exact matches, compile-checked) then run: synthetic `train_epoch` + `evaluate` on CPU **[MEAS]** |
| `scripts/kaggle_edits/h1r_det_train.py` | detector-half training driver on `kkunizaw/biohub-zh001r`: dataset, quantile normalisation, augmentation, train loop, deployed-threshold detection-P/R/F1 validation, resume, deploy-compatible checkpoint + `config.json` | **YES** — full run on synthetic zh001r-shaped data, 2 epochs + resume epoch, checkpoints and `metrics.json` written **[MEAS]** |
| `scripts/win_bet/h1r_l1_scale_audit.py` | voxel-scale audit of **our own** streamed level-1 imaging: nucleus ruler + nuclear spacing + embryo extent, exits non-zero on disagreement | **YES** — run against `ZSNS003_L1.zarr`; **it FAILS its gate**, see §5.1 **[MEAS]** |
| *(not written)* `scripts/kaggle_edits/h1r_edge_data.py` | edge-half data manufacturing | **deliberately deferred** — §5.1 invalidated its main input; spec below, build after the metadata re-audit |
| *(not written)* `scripts/kaggle_specs/h1r_*.json` | Kaggle-side kernel specs | deferred to stage 0, §7 |

Scratch test harnesses (outside Git, in the session scratchpad, not part of the deliverable):
`…/scratchpad/h1r_test/test_patched_trainer.py`, `…/scratchpad/h1r_test/test_det_driver.py`.

---

## 1. Compute budget — REVISED, and it changes the plan

**[DOC, coordinator 2026-08-18]** Two distinct pools, and they are not interchangeable:

| pool | size | internet | use |
|---|---|---|---|
| **Colab Pro** | **~45 GPU-h/week** | **yes** | **TRAINING only** |
| Kaggle T4×2 | ~30 h/week, 12 h/kernel | no (attach datasets) | inference, export, LOEO scoring, **all submissions** |

Submissions must remain Kaggle notebooks — the competition accepts nothing else
(`scripts/core/kaggle_factory.py` module docstring: a locally produced `submission.csv` returns
`FAILED_PRECONDITION`) **[VERIFIED-CODE]**.

**Three consequences, all of which change the recipe report's conclusions:**

1. **AMP is demoted from feasibility gate to throughput multiplier.** `retrain_recipes_2026-08-17`
   §1 concluded "AMP is not an optimisation, it is the feasibility gate", because fp32 ≈ 38 h did
   not fit a 12 h Kaggle kernel. With 45 h/week of Colab and no 12 h wall, a fp32 run is merely
   *expensive*, not impossible. AMP still buys an estimated ~4.75× (their 4.0 → 19 effective
   TFLOPS estimate) **[INF]**, which is the difference between one arm per week and four. Keep the
   patch; stop treating it as a gate. **Do not delete the fp32 fallback** — if `GradScaler`
   misbehaves on the real data we can still finish, just slower.
2. **The parked ablation arms become affordable.** Recipe C3 (surgical fine-tuning, 4 arms,
   ~1.5 GPU-h), C1 (AdaBN, 0.5 h) and the B2 two-arm background-weight ablation now fit inside one
   week alongside the main run instead of competing with it.
3. **Colab has internet.** The edge half no longer needs the imaging pre-staged as a Kaggle
   dataset; `scripts/win_bet/h1r_fetch_imaging.py` can stream OME-Zarr level-1 directly in the
   training environment **[VERIFIED-CODE — the script uses plain `urllib` + `numcodecs`, no
   fsspec]**. Kaggle-side legs still need everything attached as datasets.

**Colab caveat [OPEN]:** Colab Pro allocates T4/L4/A100 non-deterministically, and only the T4 is
guaranteed to match the fp16-only assumption. On an A100 or L4 (compute capability ≥ 8.0), bf16 is
available and is strictly safer than fp16 + GradScaler. The patch reads `H1R_AMP` and hard-codes
`torch.float16`; if the allocated device reports CC ≥ 8.0, switch the dtype. **Measure
`torch.cuda.get_device_capability()` in the first cell of every Colab run and record it** — a run
whose device class is unrecorded is not reproducible.

---

## 2. Verified defect inventory (file:line, re-read this session)

Every claim below was re-read in `vendor/kaggle-cell-tracking/scripts/` today, not inherited.

| id | defect | location | status |
|---|---|---|---|
| F1 | effective `det_neg_weight = 1e-2` — background carries 1 % of detection loss mass; the *inner* defaults at `:532`/`:788` read `0.1` and are overridden by the caller | `train_unet_transformer.py:1239` (argparse), `:1016` (train signature) | **PATCHED via env** (`H1R_NEG_WEIGHT`, default 0.1 on dense data) |
| F2 | checkpoint selection `score = test_acc * test_recall` — **no precision term** | `train_unet_transformer.py:1180` | **PATCHED** (P2, below) |
| F5 | `F.binary_cross_entropy(probs, target)` on post-softmax probabilities — raises inside an autocast region, so AMP is *blocked by an exception*, not merely unused | `train_unet_transformer.py:64` | **PATCHED** (P1, below) |
| F6 | division upweight is a literal no-op: `weight = torch.ones_like(loss)` then `weight[div_rows] = 1.0` | `train_unet_transformer.py:68-70` | **PATCHED** (P3) |
| F7 | export coord bias — `coords[:, 1:] *= ds_arr` with no grid-centre term, then `.astype(np.int16)` | `predict_unet_transformer.py:495-496` | **NOT patched — deliberately.** See §6.4 |
| F8 (new, [MEAS]) | `nn.DataParallel` wraps **only** the UNet (`model.unet = nn.DataParallel(model.unet)`); the detect head and transformer stay on `cuda:0` | `train_unet_transformer.py:1147` | consequence drives the AMP patch design, §3 P1 |
| F9 (new, [MEAS]) | training reads zarr with a **strided** slice `z[t:t+W, ::dz, ::dy, ::dx]`, decompressing whole chunks to keep 1/16 of voxels | `train_unet_transformer.py:333` | avoided entirely on zh001r (npy memmap, §4) |
| F10 (new, [MEAS]) | `torch.cuda.synchronize()` is called **three times per training iteration** for the timing printout | `train_unet_transformer.py:824, 884, 893` | left alone; it is the instrument that answers "I/O- or compute-bound" |

**One nuance worth recording, because it inverts an obvious approach.** F5's usual fix — wrap
`model.encode(...)` in `torch.autocast` and exclude the edge loss — **does not work on the 2-GPU
path.** `torch.autocast` state is thread-local, and `nn.DataParallel` executes replica forwards in
worker threads (`train_unet_transformer.py:1147` wraps the UNet, so the UNet forward is exactly
what runs in those threads). An autocast context established in the main thread would silently not
apply. **[INF from DOC — PyTorch documents autocast as thread-local and DataParallel as
multi-threaded.]** The patch therefore autocasts *inside* `TemporalUNet3D.forward` instead, which
executes in every replica thread. This is a real trap that a naive patch would hit and that would
produce a run that looks fine and is 4× slower than it should be.

---

## 3. `h1r_trainer_patch.py` — the 13 patches, verified

Applies to a **copy** of the trainer (never in place under `vendor/`). Every patch asserts an exact
occurrence count, mirroring the `kaggle_factory` edit discipline, then compile-checks the result.

| # | patch | mechanism |
|---|---|---|
| P1a | insert knobs + `_h1r_enable_amp_unet()` after the last vendored import | env-driven: `H1R_AMP`, `H1R_DIV_WEIGHT`, `H1R_SELECT`, `H1R_COSINE`, `H1R_IGNORE_RADIUS_VOX` |
| P1b | monkeypatch `TemporalUNet3D.forward` to run under `torch.autocast("cuda", float16)` and cast the output back to `.float()` | DataParallel-safe (see §2); leaves the detect head, transformer, `compute_loss`'s `F.binary_cross_entropy` and `compute_detection_loss`'s BCE-with-logits all in fp32 |
| P1c | `GradScaler` created in `train_epoch` | `torch.amp.GradScaler("cuda", enabled=…)` |
| P1d | `loss.backward()` → `scaler.scale(loss).backward()`; `scaler.unscale_(optimizer)` **before** `clip_grad_norm_`; `scaler.step` / `scaler.update` | unscaling before clipping is required or the clip threshold is applied to scaled gradients |
| P2a–d | `evaluate` additionally accumulates `det_total`/`det_matched` and returns a **4-tuple** `(loss, acc, recall, precision)`; `train` computes detection F1 and selects on `test_acc × F1` (`H1R_SELECT=acc_f1`, default) or the legacy `acc × recall` (`acc_recall`); the per-epoch print gains `prec=` | the F2 fix; precision comes free from `detect_and_match`'s existing match array — matched detections / all detections |
| P3 | `weight[div_rows] = 1.0` → `= _H1R_DIV_WEIGHT` | the F6 fix; default 3.0 (Trackastra uses 11 — see §5 on why we do not start there) |
| P4 | ignore-radius mask in `compute_detection_loss`: zero the negative weight in a ball of radius `H1R_IGNORE_RADIUS_VOX` around confident **non-GT** peaks | recipe C2. **Default 0 = OFF.** Legitimate only on the sparse competition fine-tune; on dense Zebrahub labels it would mask real background |
| P5 | `CosineAnnealingLR(T_max=n_epochs)` | recipe B4 |
| P6 | save `edge_predictor_last.pth` every epoch + `scheduler.step()` | cross-session resume; the vendored code saves only on improvement, so a killed session loses everything since the last best |

### What "verified" means, concretely **[MEAS]**

```
.venv/Scripts/python.exe scripts/kaggle_edits/h1r_trainer_patch.py --trainer <scratch copy>
  -> h1r_trainer_patch: 13 patches applied
```
Then a synthetic CPU run through the patched module (batch 2, window 2, 32³ volumes, 6 nodes/frame,
one deliberate division row):

```
train_epoch OK: edge=0.0000 det=0.8024
  [timing] data: 1.2s (4%) | forward: 10.8s (40%) | backward: 15.2s (56%)
evaluate OK: loss=0.0000 acc=0.0000 recall=0.0000 precision=0.0000   <- 4-tuple, P2 live
division weight OK: w=3 loss 0.4339 vs w=1 loss 0.1484               <- P3 no longer a no-op
```

Read these numbers correctly: **this is plumbing, not learning** (CLAUDE.md rule 5). An untrained
net at the deployed threshold detects nothing, so P/R/F1 = 0 is the *expected* output and confirms
the metric path executes rather than that it works. The load-bearing results are: the 4-tuple
returns, the division weight now changes the loss (3.0 vs 1.0 give different values — F6 is fixed),
the ignore-mask branch executes without shape errors at `H1R_IGNORE_RADIUS_VOX=2`, and the
timing instrument still prints (F10 survives, so the I/O-vs-compute question stays answerable).

**Not yet verified [OPEN]:** the AMP path itself. `H1R_AMP=0` was set for the CPU test because
autocast requires CUDA. The first Colab run must, before anything else, confirm (a) no exception
from `F.binary_cross_entropy` under the patched arrangement, (b) a measured s/step ratio fp16 vs
fp32 — recipe A1's falsification: **if the speed-up is < 1.3×, the run is I/O- or Python-loop-bound
and A2 (pre-materialised tensor) is the higher lever**, and the trainer's own `[timing]` line is
the discriminator.

---

## 4. DETECTOR-HALF pilot spec — `h1r_det_train.py` on `kkunizaw/biohub-zh001r`

### 4.1 Substrate facts (all prior-session [MEAS], re-confirmed from the audit script)

- `zh001r_iso.npy` = **(72, 20, 64, 64, 64) uint8** ZSNS001 imaging; `zh001r_nodes.npz` = 1440
  arrays `f{crop*20+t}`, each `(N,4)` float32 `[t, z, y, x]`, ~900 nodes/frame
  (`research/04-data/data-acquisition.md:31-33`) **[DOC]**.
- Voxel scale **1.677 µm isotropic = 1.032× our deployed 1.625 µm grid**, measured by the
  nucleus-size radial-profile ruler (`scripts/win_bet/h1r_zh001r_audit.py::audit_geometry`) **[MEAS,
  prior session]**.
- Labels are aligned to the imaging: 2.29× intensity contrast at nodes vs random voxels; the audit
  hard-fails below 1.15× (`h1r_zh001r_audit.py:165`) **[MEAS, prior]**.
- **No track identity** — node arrays are 4 columns, so no GT transition matrix can be built.
  Detector supervision only (`h1r_zh001r_audit.py:126-137`) **[MEAS, prior]**.

### 4.2 The 1.032× scale question — **ignore it, do not resample**

**Decision: train at their native grid, no resampling.** Three reasons, in order of weight:

1. **The mismatch is inside the augmentation the deployed model already tolerates.** nnU-Net's
   reference 3D pipeline applies random scaling in **0.7–1.4** at p = 0.2 **[DOC,
   `retrain_recipes_2026-08-17` §5]**. A fixed 1.032× offset is 3 % — an eighth of the way to that
   augmentation's *lower* edge. A model that cannot absorb 3 % isotropic scale could not survive
   its own training distribution.
2. **Resampling costs more than it fixes.** Trilinear resampling 64³ → 62³ and back would blur the
   nuclei (the very signal the detector reads), and 62³ is not a clean multiple for the UNet's
   3 pooling levels ([32,64,128] ⇒ /8 ⇒ 7.75). Padding back to 64³ then introduces edge artefacts.
   Cost: a real degradation to remove a 3 % geometric offset. **[INF]**
3. **The measurement's own error bar exceeds the correction.** The audit prints the ruler's
   confound explicitly (`h1r_zh001r_audit.py:218-220`): it assumes comparable physical nucleus size
   across embryos and stages, and their denser (later-stage) nuclei bias the estimate **upward**.
   The prior session's record spans 1.6–1.8 µm (median variously 1.677 and 1.762 across runs). We
   would be correcting by 3 % using an instrument whose uncertainty is ~10 %. **[MEAS + INF]**

**Falsification, if this decision is wrong:** train one short arm with a 1.032× resample and
compare deployed-threshold detection F1 on the held-out zh001r crops. Cost ~1 GPU-h. Run it only
if the main arm's detection F1 is inexplicably poor — not pre-emptively.

**What we must NOT ignore** is the *deployed* input scale for inference: whatever we train, the
retrained detector will be run by `predict_unet_transformer.predict_video`, which reads competition
zarr strided by `downsample=(1,4,4)` (`predict_unet_transformer.py:215`) giving the 1.625 µm grid
**[VERIFIED-CODE]**. Train-on-1.677 / infer-on-1.625 is the whole transfer premise, and it is
exactly what the LOEO leg in §6 measures.

### 4.3 Intensity: uint8 → the deployed normalisation, not a re-standardisation

The deployed loader uses **per-video 0.1 %/99.9 % quantile min–max with `clamp(0)`**
(`train_unet_transformer.py:334`, `predict_unet_transformer.py:369`) **[VERIFIED-CODE]**.
`h1r_zh001r_smoke.py:82` used a per-window mean/std standardisation instead — correct for a
plumbing smoke, **wrong for the pilot**, because it would train the detector on a different
intensity statistic than the one inference will supply.

`Zh001rWindows` therefore computes per-crop `q_low = quantile(0.001)`, `q_high = quantile(0.999)`
over the crop's whole (20, 64³) volume and applies `clip((v - q_low)/(q_high - q_low + 1e-6), 0)` —
byte-for-byte the deployed transform, just with the quantiles computed rather than read from zarr
attrs (the npy pack has no attrs). uint8 vs uint16 becomes irrelevant: the quantile map takes both
to roughly [0, 1]. **[MEAS — implemented and exercised in the synthetic run.]**

### 4.4 Init: from the public 50-epoch checkpoint, not scratch

`H1R_INIT_WEIGHTS` loads the deployed full checkpoint with `strict=False`. Two reasons:

- **Scratch is not affordable even at 45 h/week.** nnU-Net's reference schedule is 250,000 steps;
  at the recipe's best-case 0.6 s/step that is ~42 h for one fold **[DOC + INF,
  `retrain_recipes_2026-08-17` §6]** — a whole week's Colab for one arm with no ablation.
- **We must not void the anchor.** Our entire 0.915 platform is that checkpoint. Initialising from
  it means a failed retrain degrades gracefully toward the known-good weights instead of landing in
  an unrelated basin, and it makes "how far did the weights move" a measurable quantity.

Loading the *full* `edge_predictor_best.pth` (not just the UNet) matters: it carries the transformer
edge head, which this half never trains. The saved checkpoint therefore stays a complete, deployable
model — **`h1r_det_train.py` writes `edge_predictor_best.pth` plus a matching `config.json`**, which
is what `predict_unet_transformer.load_model` reads (`predict_unet_transformer.py:172-200`)
**[VERIFIED-CODE]**. The retrained detector drops into the deployed pipeline with a path change and
nothing else.

**A caveat that must be stated, not buried [INF]:** the edge head was trained *against the old
detector's* feature maps. Moving the UNet moves those features under a frozen head. This is exactly
why the evaluation in §6 measures the **end-to-end scored metric**, not detection F1 alone — a
detector that improves in isolation while breaking its own edge head is a net loss, and only the
scorer sees that.

### 4.5 Training configuration

| knob | value | why |
|---|---|---|
| split | crop index `% 6 == 0` → val (12 crops), else train (60 crops) | held-out crops, not held-out frames — frames within a crop are near-duplicates at 20 t |
| val windows | stride 2 (non-overlapping) | each frame counted exactly once in P/R |
| `det_neg_weight` | **0.1** (`H1R_NEG_WEIGHT`), vs the deployed 1e-2 | the F4 mechanism: dense labels are what *permit* raising the background weight. On competition data 97 % of true nuclei are unlabelled so a raised weight teaches nuclei-are-background; on zh001r the labels are (near-)complete |
| batch / LR / epochs | 16 / 1e-4 AdamW / 20, cosine | fine-tune, not from-scratch — Spotiflow's side of the LR argument **[DOC]** |
| augmentation | brightness ±0.1 + 8 axis flips (vendored, verbatim from `augmentations.py:7-63`) **+ in-plane 90° rotations** | the recipe's 16-element group: 4 in-plane rotations × flips. **z↔xy permutations are excluded** — the grid is isotropic but the PSF is not, so a z↔x swap makes a volume no microscope could produce **[INF, `retrain_recipes_2026-08-17` §5]** |
| workers | 3 (`H1R_WORKERS`) | Kaggle/Colab give 4 vCPU; the vendored default of 8 oversubscribes 2:1 **[DOC]** |
| I/O | `np.load(mmap_mode="r")` on the npy pack | recipe A2 comes free — the pack *is* the pre-materialised uint8 tensor, so F9's strided-zarr decompression never happens |

### 4.6 The checkpoint score, spelled out

Selection is **detection F1 at the deployed operating point**, computed on held-out crops:

- threshold = **logit(0.96875) = 3.434**, because `BIOHUB_DET_THRESHOLD = 0.96875` in the deployed
  P0-B kernel env cell **[MEAS — read from
  `notebooks/kaggle_p0b_clean913_revtime/biohub-p0b-clean913-revtime.ipynb` cell 2]**;
- `pool_kernel_um = 3.0`, matching the deployed `PredictConfig.pool_kernel_um`
  (`predict_unet_transformer.py:74`) **[VERIFIED-CODE]**;
- matching = the trainer's own greedy one-to-one `detect_and_match` at 5 µm.

A second threshold (`p50`, logit 0) is logged every epoch as a free diagnostic: if F1 at p50 is
strong while F1 at the deployed threshold is near zero, the retrained detector is **miscalibrated
rather than bad**, and the fix is a threshold sweep, not more training. That distinction is
otherwise invisible and would be misread as "the retrain failed".

### 4.7 GPU-hour estimate — detector half

Arithmetic from the recipe **[INF]**: TemporalUNet3D [32,64,128] on 64³ ≈ 1.18e11 FLOPs/volume
forward; batch 16 × window 2 = 32 volumes; ×3 for fwd+bwd ⇒ ~11.3 TFLOP/step.
zh001r train set: 60 crops × 19 windows = **1,140 windows ≈ 72 steps/epoch** — far smaller than the
competition fold (~984 steps/epoch), because zh001r has 20 timepoints, not 100.

| stage | steps | fp16 @ ~0.6 s/step | fp32 @ ~2.8 s/step |
|---|---:|---:|---:|
| 1 epoch | 72 | ~0.7 min | ~3.4 min |
| 20 epochs + val | 1,440 | **~0.5 GPU-h** | ~2 GPU-h |

**The detector half is cheap.** Even the fp32 fallback fits comfortably. This is the single most
important budgeting fact in the document: it means **the detector arm can be run several times with
different `det_neg_weight` / init / augmentation settings inside one Colab week** — the B2 two-arm
background-weight ablation costs ~1 GPU-h total, not a session.

**Validate this estimate before trusting it [OPEN]:** the numbers are FLOP-model inferences, not
measurements. The first Colab run reports real s/step; if it exceeds ~1.5 s/step at fp16, the run is
Python-loop-bound (`compute_detection_loss` and `detect_and_match` both contain per-sample Python
loops, `train_unet_transformer.py:557`, `:695`) **[VERIFIED-CODE]** and the batching of those loops
becomes the lever.

### 4.8 "Done" for the detector half

1. Deployed-threshold detection F1 on held-out zh001r crops improves over the same metric measured
   with the **unmodified public weights** on the same crops (the zero-epoch row of the same table —
   this is a free baseline and must be recorded before training starts).
2. The retrained checkpoint loads into `predict_unet_transformer.load_model` unchanged.
3. `N_pred/N_est` on the deployed substrate moves in the predicted direction (down) — the recipe's
   own falsification for the "structural over-detection" story
   (`retrain_recipes_2026-08-17` §11.5). **If raising `det_neg_weight` on dense data does not move
   the node-count ratio, the mechanism story is wrong and the retrain must be re-motivated.**
4. Only then does it earn a Kaggle inference leg.

---

## 5. EDGE-HALF pilot spec

### 5.1 STOP-THE-LINE FINDING: our recorded level-1 voxel scale is wrong by ~4.5× laterally

**[MEAS this session — `scripts/win_bet/h1r_l1_scale_audit.py`, exit code 1]**

`retrain_recipes_2026-08-17` §11.4 named a packaged-crop scale error as the highest-value zero-GPU
pre-flight — *"a scale error here silently poisons the whole retrain"*. That warning was aimed at a
third party. **I pointed the same instrument at our own fetch, which had never been checked, and it
fails.**

Our fetcher records `scale_zyx = [0.62, 0.2195, 0.2195]` µm, read out of the OME-Zarr multiscales
metadata by `h1r_fetch_imaging.py:50-59` and written at `:88` **[VERIFIED-CODE]**. Three
independent measurements say that attribute does not describe the array we actually fetched:

| instrument | under recorded scale | under ruler scale | verdict |
|---|---|---|---|
| **nucleus half-width** (lateral, vs competition where voxel size is exactly known) | implies 0.2195 µm/vox | **0.984 µm/vox** | **4.48× disagreement** |
| **median nuclear NN spacing** (tracks only, no imaging) | **1.89 µm** — smaller than a nucleus, biologically impossible | **6.45 µm** | ruler correct |
| **embryo extent** for 9,638 tracked nuclei | 77 × 160 × 173 µm — too small to hold them | **176 × 717 × 774 µm** | ruler correct |

Per-axis implied voxel size: **z 1.411 µm (2.28× the record), y 0.950 µm (4.33×), x 1.018 µm
(4.64×)**. Lateral is the trustworthy axis — lateral PSF is near diffraction-limited in both
instruments — so the headline is **~4.5× laterally**. The axial figure is PSF-confounded and is
reported but not used for the verdict. Note the *anisotropy* also disagrees (record 2.82:1 z:xy,
ruler 1.41:1), so this is **not** a single clean global factor; the most likely mechanism is an
**off-by-two in pyramid indexing** between the array path we fetch and the `datasets` entry we read
the scale from (0.2195 → 0.439 → 0.878 lands near the measured 0.98) **[INF, not established]**.

**Consequences — these are large:**

1. **The recipe report's resample factors are wrong.** `retrain_recipes_2026-08-17` §4 states the
   resample is "(2.62, 7.40, 7.40) from level-0, or (1.31, 3.70, 3.70) from level-1". Two errors:
   the quoted (2.62, 7.40, 7.40) is arithmetically `1.625 / [0.62, 0.2195, 0.2195]`, i.e. derived
   from the **level-1** numbers while labelled level-0; and the underlying scale is itself wrong.
2. **The true resample is mild and cheap.** Measured: **z ×0.87, xy ×0.61** to reach the deployed
   1.625 µm grid — a gentle downsample, not a 7.4× reduction. A whole level-1 frame
   (129, 965, 1019) becomes ≈ (112, 584, 617) on the deployed grid, from which many 64³ crops can be
   cut. The trilinear path already exists in the deployed loader
   (`train_unet_transformer.py:336-340`) **[VERIFIED-CODE]**, so no new resampling machinery is
   needed. **This makes the edge half substantially cheaper than the recipe assumed** — its
   headline cost item was the 7.4× resample.
3. **It also retires a strategic claim.** `directional-updates.md` (2026-08-17) justified choosing
   level-1 on the basis that retraining there "**voids the deployed 0.915 detector anchor**"
   because it is a different resolution lineage. At ~0.98 µm lateral, level-1 is 1.65× the deployed
   grid, not 7.4× — the same geometry family, reachable by a gentle resample. **The premise that
   the edge half must void the anchor is not supported by the measurement.**

**Required action before any edge-half GPU hour [GREEN-LIGHT NOT REQUIRED — it is a metadata
fetch]:** re-fetch `.zattrs` and the per-level `.zarray` shapes for ZSNS003 and check the
`multiscales.datasets` list against the array actually served at each path. This needs internet, so
it runs locally or on Colab, costs no GPU, and takes minutes. **Until it is done, do not build the
resample on the recorded attribute — calibrate it empirically with the ruler**, which is the method
the zh001r audit already validated.

**Why this did not poison the detector half:** zh001r arrives as a plain `.npy` at a scale we
measured directly from image content (1.677 µm, 1.032× deployed) rather than from a metadata
attribute. The detector spec in §4 is unaffected.

### 5.2 Substrate and supervision

- **Imaging:** our own stream (`h1r_fetch_imaging.py`), level-1, resample calibrated per §5.1.
  Colab's internet removes the need to pre-stage it as a Kaggle dataset (§1).
- **Tracks:** on disk for ZSNS001/003/004/005, columns
  `track_id, t, z, y, x, id, parent_track_id, parent_id` **[MEAS — CSV header]**; the prepared
  parquet drops the node-level ids and keeps `track_id, t, z, y, x, parent_track_id` **[MEAS]**.
  4,057,611 nodes over 515 frames for ZSNS003 = 7,879 nodes/frame **[MEAS]**.
- **Track coords are level-0 voxel indices; ÷2 maps them to level-1** (`h1r_train_smoke.py:32`).
  §5.1's ruler independently confirms this mapping — profiles centred on the mapped coordinates are
  sharply peaked (half-width ~3 voxels), which a mis-mapping would have washed out **[MEAS]**.
- **Nuclei are shell-distributed; crops must be node-centred.** `h1r_train_smoke.py:48-57` scans
  3,000 node-centred origins and keeps the densest, precisely because a naive centre crop lands in
  the hollow interior **[MEAS, prior]**.
- **The supervision is Ultrack OUTPUT, not human GT** (BC(i) ≈ 0.47) **[DOC]**.

### 5.3 Embryo and timepoint selection

| role | embryo | why |
|---|---|---|
| edge training | **ZSNS003** | tracks on disk, 2 frames already fetched and audited |
| cross-embryo val | **ZSNS004** | held out entirely — the only honest test of edge generalisation |
| reserve | ZSNS005 | third fold if 003/004 disagree |
| **excluded** | ZSNS001 | it is the embryo packaged as `zh001r` and used by the **detector** half. Training both halves on the same embryo would make a joint result uninterpretable |

**Timepoints: 100 per embryo for the pilot.** Sized against download, not GPU: level-1 frames cost
~51 MB compressed on the wire **[MEAS — 2 frames = 102 MB on disk]**, and the fetcher measured
~4 MB/s previously **[DOC, prior session]**, so 100 frames ≈ 5 GB ≈ 21 min of wall clock, no GPU.
Store resampled 64³ crops, not raw frames — the raw pyramid level for a whole embryo at 515 frames
is ~131 GB **[MEAS, prior]** and must never be staged.

### 5.4 Data manufacturing pipeline

1. **Fetch** level-1 frames `t ∈ [0, 100)` for ZSNS003 (and ZSNS004 for validation).
2. **Resample** each frame to the deployed 1.625 µm isotropic grid with trilinear interpolation,
   factors from §5.1's empirical calibration — **not** from the zarr attribute until it is re-audited.
3. **Scale the track coordinates by the same factors** so nodes and voxels stay in one frame of
   reference. This is the step where a silent error is unrecoverable: assert afterwards that the
   nucleus ruler on the resampled volume reproduces the competition half-width in µm.
4. **Cut node-centred 64³ crops.** Sample crop origins from node positions (shell distribution),
   greedily reject origins overlapping an accepted crop by more than 50 %, keep ~60 crops/embryo.
5. **Cap nodes per frame at 256** by random subsample, re-drawn each epoch (see §5.6 — this is a
   hard performance requirement, and it doubles as augmentation).
6. **Build GT transition matrices** exactly as `h1r_train_smoke.py:87-93`: for each node *j* at
   t+1, set `gt[i, j] = 1` where node *i* at t has `track_id == track_id(j)` (continuation) **or**
   `track_id == parent_track_id(j)` (division daughter). This reproduces the vendored
   `compute_gt_transition_matrix` semantics from track ids rather than an explicit edge list.
7. **Persist** as an npz pack mirroring the zh001r layout, so `Zh001rWindows` in
   `h1r_det_train.py` is reusable with a different node loader.

### 5.5 Noise handling for Ultrack-derived links

The supervision is a strong published teacher's *output*, and it must be treated as such
(`retrain_recipes_2026-08-17` §8 makes the same point: Stage 1 is already a pseudo-label run).

- **Divisions are the least reliable part, so they get LESS weight, not more.** Trackastra's
  documented **11×** division upweight **[DOC]** assumes trustworthy division labels. Ours are
  Ultrack's, and our own prior measurement bounds the risk: synthetic→real division transfer AUC
  **0.664** **[MEAS, prior — `redteam_blindspots_2026-08-17`]**. **Recommendation: `λ_div = 1.0` on
  Zebrahub-supervised edge training** (i.e. fix F6's no-op but do not exploit it), and reserve the
  raised weight for competition GT, which is human-annotated. The ablation `λ_div ∈ {1, 3, 11}` is
  an **arm**, not a setting; the patch exposes it as `H1R_DIV_WEIGHT`.
- **Cleanest variant, worth an arm of its own:** mask division rows out of the Zebrahub edge loss
  entirely — *learn continuation from Ultrack, learn division from competition GT only*. Given that
  our deployed pipeline recovers essentially zero divisions (613 division FPs against 5 TPs across
  199 crops **[MEAS]**), corrupting continuation learning to chase noisy division labels is a bad
  trade.
- **No per-edge confidence is available.** The tracks carry no score column **[MEAS — CSV header]**,
  so confidence weighting in the Soft-Teacher style is not implementable from this input.
- **Usable proxy:** displacement. Downweight links whose frame-to-frame displacement exceeds the
  frame's 95th percentile — long jumps are where a linker's errors concentrate. This is the same
  geometric intuition as our own motion gate, which is pure geometry and never reads `prob`
  **[DOC, prior]**.
- **Keep real annotations in every batch.** Arazo et al. find that *"a minimum number of labeled
  samples per mini-batch"* is one of only two effective regularisers against confirmation bias in
  self-training **[DOC]**. Interleave competition windows into Zebrahub batches rather than training
  in clean phases — this also protects the BatchNorm running statistics from being captured by one
  domain **[INF]**.

### 5.6 Cost — and the one thing that will actually bite

**A Python-loop wall the recipe did not cost.** `detect_and_match` performs greedy one-to-one
matching in a **Python** loop over detections (`train_unet_transformer.py:713-718`) **[VERIFIED-CODE]**,
and `compute_batch_loss` (`:84-88`), `_index_features` (`:463-470`) and `build_matched_edge_targets`
(`:759-773`) all loop per batch element. On competition data this never mattered: **2.84–8.84 GT
nodes per volume** **[MEAS, F3]**. Zebrahub crops carry **~900 nodes/frame** **[DOC]** — a ~100–300×
increase — so the greedy loop alone becomes ~900 Python iterations per sample per frame, i.e. ~29k
per step at batch 16 × window 2 **[INF]**.

**This, not FLOPs, is the edge half's binding constraint**, and it is why step 5 of §5.4 caps nodes
at 256 — the cap is a performance requirement first and an augmentation second. Both existing smoke
scripts already cap for the same reason (`h1r_train_smoke.py:64` at 256, `h1r_zh001r_smoke.py:47` at
512) **[VERIFIED-CODE]**. If profiling shows the loop still binding, vectorise the greedy match
before buying more GPU hours — the trainer's own `[timing]` line separates data from forward from
backward and will show it.

| item | quantity | cost |
|---|---|---|
| fetch 100 frames × 2 embryos | ~10 GB | ~40 min wall, **0 GPU-h** |
| resample + crop + pack | CPU | ~1 h wall, **0 GPU-h** |
| edge training, 60 crops × 39 windows = 2,340 windows ≈ 146 steps/epoch, 20 epochs | ~2,900 steps | **~1.5–3 GPU-h** at ~2 s/step (dominated by the Python loops, not the UNet) |
| λ_div ablation, 3 arms | | +2× the above ≈ **3–6 GPU-h** |

**Edge half total: ~5–9 Colab GPU-h** including ablations — comfortably inside one 45 h week.

### 5.7 Fold structure

- **Internal:** leave-one-embryo-out over ZSNS003 / ZSNS004 (/005 as third). Train on one, validate
  on the other. This measures *edge generalisation across embryos* and nothing about the
  competition.
- **External (the one that decides anything):** the competition LOEO in §6.5 Option A, then a
  leaderboard slot. Cross-embryo Zebrahub agreement is a **gate to proceed**, never a promotion.
- **Do not train edges and detector on the same embryo** (§5.3), or a joint gain cannot be
  attributed to either half.

---

## 6. EVALUATION DESIGN under the validation crisis

This is the section the mandate calls load-bearing, and it has to start by taking the failure
seriously rather than routing around it.

### 6.1 What actually failed

Arm B (motion-gate) measured **+0.0144 (44b6) / +0.0090 (6bba)** on a *clean paired
deployment-substrate LOEO* — official `tracking_cellmot` scorer, identical crops, a single env
toggle — and scored **+0.000 on the leaderboard** (submission `55585140`, 0.915, identical to P3
alone) **[MEAS, `research/06-knowledge-system/experimental-records.md:477-498`]**. That instrument
was adopted *specifically* to cure the substrate mismatch that had discredited the earlier
E0c-anchored numbers. It is our best instrument and it still failed. This was the **fourth
consecutive optimistic central estimate** **[DOC, `research/07-outputs/submissions.md`]**.

The +0.005 bilateral LOEO bar is therefore **falsified as a sufficient gate**. Any H1 plan that
proposes "LOEO ≥ +0.005 ⇒ promote" is proposing the gate that just failed.

### 6.2 The churn evidence — and why it does *not* automatically exonerate a weights change

**[MEAS, coordinator, `c:/temp/armb_diag/churn_diff.json`]** the arm-B submission churned **3–10 %
of edges** versus P3-harmonic and scored **identically**. Combined with the P3 result — 4.368 %
churn, +0.001 — and the sub-voxel result — a real coordinate change, −0.0004/−0.0009 — the pattern
is: *graph-edit levers move structure without moving the metric.*

It is tempting to conclude "so a weights change is different". **That conclusion has to be earned,
and here is the honest case for and against.**

**The mechanistic argument that it is a different lever class [INF]:**
every failed lever so far operates on a **fixed candidate set**. Arm B re-links edges among
detections the frozen detector already produced; harmonic re-scores those same edges; sub-voxel
nudges those same coordinates. The scorer's structure explains why that is nearly free: ignored
edges cost nothing, and division FPs are only counted on annotated cells
**[DOC, memory: verified metric semantics]**. If the candidate pool's *content* is what bounds the
score, then permuting how you choose within that pool moves churn and not score. A retrained
detector changes the pool itself — different nodes exist. That is the first lever we would have
tried that is not a within-pool permutation.

**The argument against, which must not be suppressed:** this is a *post-hoc* story fitted to four
failures. We have **no measured instance** of a weights change moving our LB score — the deployed
weights have never been swapped. Until one exists, "weights are a different class" is a hypothesis
of exactly the kind that has already burned us four times.

**Which is precisely why the programme now opens with a zero-training test of that hypothesis.**

### 6.3 STAGE 0 — the xiaoleilian A/B: buy the answer to the central H1 question for inference cost

**[DOC, coordinator / live-surface agent]** Kaggle user **xiaoleilian** (rank 86, LB 0.918)
publishes a from-scratch retrained UNet3D stack with **public attachable weights** — datasets
`biohub-m001-ens3-sm6-sim2` and `biohub-unet3d-weights-v2models` — plus a working division patch.

This is a **zero-training A/B reference** and it answers the H1 premise directly:

> *Does swapping a retrained detector into our pipeline move our score?*

Design: our deployed P3 pipeline, unchanged, with only the detector weights path repointed at their
checkpoint. Everything downstream — threshold, pooling, ILP, output filters — held fixed.

- **Cost:** one Kaggle inference kernel (~2–4 GPU-h), **zero training hours**, and — for the LB
  leg — **one submission slot**.
- **Why it dominates every other first move:** it tests the *class* claim of §6.2 with real weights
  and a real score, before a single Colab hour is spent. If a retrained detector from a team **above
  us on the LB** cannot move our number, the H1 premise is in serious trouble and we learn it for
  the price of an inference run instead of a month.
- **Expected obstacles, stated up front [OPEN]:** their architecture may not match
  `TemporalUNet3D([32,64,128])`, in which case `load_state_dict(strict=False)` will report a large
  missing/unexpected count. **That count is itself the first measurement** — record it. If the
  architectures are incompatible, the fallback is to run *their* detector to produce a node set and
  feed those nodes into our linker, which tests the same question one level up at slightly more
  integration cost.
- **Second free artefact:** their division patch. Our division numbers are catastrophic — 613
  division FPs against 5 TPs across 199 crops **[MEAS,
  `research/06-knowledge-system/experimental-records.md`]** — and a working public division patch
  from a higher-ranked team is worth reading regardless of whether we adopt it.

**Licensing note (factual, per instruction):** public Kaggle datasets carry the uploader's chosen
licence; record it in the shipping notebook. No exclusion on licence grounds.

### 6.4 Why F7 stays unpatched, as a worked example of the discipline

The recipe recommends A4: add +1.5 full-res voxels in y/x at export to zero the 0.609 µm bias,
predicted to cut RMS radial 1.075 → 0.642 µm **[INF from MEAS]**. **We do not do this as part of
H1.** The sub-voxel-refine lever tested the same physical quantity and came back
**−0.0004 / −0.0009 bilaterally** — because 1.075 µm is *already inside* the metric's free zone
(the measured scorer cliff starts at ~1.5–2 µm) **[MEAS,
`research/06-knowledge-system/experimental-records.md` §B]**. Moving an error that is already free
cannot pay. Bundling A4 into H1 would add a confound to the one measurement we actually need. It
stays on the shelf as a separately testable residual.

### 6.5 The evaluation options, costed

Four candidate instruments for "did H1 work?".

**Option A — LOEO on the deployment substrate (the instrument that just failed).**
Cost: 2 Kaggle kernels/arm (~4–6 GPU-h), 0 submissions.
*Is it more meaningful for a weights change than it was for a graph edit?* **Partly, and the reason
is specific.** LOEO holds out an embryo family and scores with per-fold weights. For a graph-edit
lever, the fold weights are a nuisance parameter — the lever is weight-independent, so LOEO's
per-fold weights inject variance unrelated to the thing measured. For a **weights** lever, the
per-fold weights *are* the object under test, so LOEO is measuring the right kind of object.
**But** the open diagnostic recorded after the arm-B failure still stands: whether the LOEO→LB
signal dies on the *crop population* (train vs test crops) or on the *per-fold weights* is **not
established** **[DOC, experimental-records 2026-08-18]**. Until that is answered, LOEO is
necessary-not-sufficient for H1 too.
*Failure mode:* passes a lever worth 0.000, again.

**Option B — direct LB A/B with submission slots.**
Cost: 1 slot per arm, ~2–4 Kaggle GPU-h per submission kernel. The only instrument with **zero**
transfer risk, because it is the target metric.
*Failure modes:* slots are finite (12 consumed) and the LB has resolution limits — our last three
distinct pipelines all read 0.915, so a change below ~0.001 is invisible; and LB variance across
submissions is unmeasured, so a single ±0.001 read is not a measurement.

**Option C — intermediate signals: detection P/R on held-out competition crops, deployed vs
retrained weights.**
Cost: ~1 GPU-h, 0 slots. Already implemented (`eval_detection` in `h1r_det_train.py`, §4.6).
*Value:* it is the only instrument that can distinguish "the retrain did nothing" from "the retrain
worked and the threshold is now wrong". Given F1/F2, a retrained detector's calibration will
**almost certainly** shift, and running it at the inherited 0.96875 threshold would misreport a
working retrain as a dead one. **This is not optional.**
*Failure mode:* detection F1 is not the scored metric; sparse annotation means "false positives"
here include real unlabelled nuclei. It cannot promote anything on its own.

**Option D — the diagnostic that resolves the crisis rather than working around it.**
Score the **deployed full-weight** pipeline on the same LOEO crops and compare its LOEO-vs-LB
relationship to the per-fold one **[DOC — this is the open diagnostic named in
experimental-records 2026-08-18]**. Cost ~2–4 Kaggle GPU-h, 0 slots. If full-weight LOEO tracks the
LB and per-fold LOEO does not, the fault is the *fold weights* and LOEO becomes trustworthy for
weights levers evaluated full-weight. If neither tracks, the fault is the *crop population* and
**no train-crop instrument can gate a submission** — which would be the single most valuable
negative result available to us, because it would redirect all remaining effort to Option B.

### 6.6 RECOMMENDATION

**A staged gate, ordered by cost-to-information, with Stage 0 first.**

| stage | instrument | cost | gate |
|---|---|---|---|
| **0** | **xiaoleilian retrained weights A/B** (§6.3) | ~2–4 Kaggle GPU-h; +1 slot for the LB leg | If a *higher-ranked team's* retrained detector cannot move our LB, **pause H1 before spending Colab hours** and reconsider. This is the cheapest evidence on the central question |
| **0b** | **Option D discriminator**, run in the same session | ~2–4 Kaggle GPU-h, 0 slots | Establishes whether *any* train-crop instrument can gate. Runs alongside stage 0 — same kernel family, no extra week |
| **1** | Option C detection P/R, deployed vs retrained, incl. threshold sweep | ~1 GPU-h, 0 slots | If detection F1 does not beat the public weights at *any* threshold, kill before inference |
| **2** | Option A LOEO, both directions, on the retrained weights | ~4–6 Kaggle GPU-h, 0 slots | Necessary, **not sufficient**. A bilateral gain is permission to spend a slot; it is not evidence of a gain |
| **3** | Option B, one slot | 1 slot | The only promotion evidence |

**Total to a scored H1 answer: ~10–15 Kaggle GPU-h + ~2–4 Colab GPU-h + 2 slots** (one at stage 0,
one at stage 3). Both fit inside a single week of either budget.

**The bar changes, and this is the important part.** Because LOEO ≥ +0.005 has been falsified as a
gate, stage 2 no longer *promotes*; it only decides whether a slot is worth spending. **The
promotion decision moves to stage 3, on the LB, and nowhere else.** Any plan that promotes an H1
weights change on LOEO evidence alone is repeating the arm-B mistake with a more expensive lever.

**Prediction discipline [DOC, `research/07-outputs/submissions.md`]:** four consecutive central
estimates have been optimistic, and the log's own retrospective identified **churn magnitude
against a known-null reference** as the predictor that worked while **net cardinality change**
carried no signal. Every H1 submission must carry a written prediction band recorded *before* the
score, and the arm-B lesson says to weight the low band heavily.

---

## 7. STAGED CALENDAR

**Every GPU launch — Kaggle and Colab alike — needs explicit host green-light and is marked
`[GL]`.** Everything unmarked is CPU, metadata or download work that needs none. No submission is
automatic; the factory prints a `submitcmd` and a human runs it
(`scripts/core/kaggle_factory.py` docstring) **[VERIFIED-CODE]**.

### Before 2026-08-24 — no GPU, no green-light needed

| # | task | cost | kill criterion |
|---|---|---|---|
| P0 | ✅ **DONE** — trainer patch (13/13), detector driver, both runtime-verified on synthetic data | 0 | — |
| P1 | ✅ **DONE** — level-1 scale audit; **it fails**, §5.1 | 0 | — |
| P2 | **Re-audit the OME-Zarr multiscales metadata** for ZSNS003: fetch `.zattrs` + each level's `.zarray`, check the `datasets` list against the array served at each path | minutes, internet, **0 GPU** | If the metadata is self-consistent and still says 0.2195 µm, then the ruler is wrong and §5.1 must be re-opened before any edge work. **This is the highest-priority open item.** |
| P3 | Download `kkunizaw/biohub-zh001r` (~363 MB) and run the existing `h1r_zh001r_audit.py` + `h1r_zh001r_smoke.py` against the real bytes | ~15 min, **0 GPU** | audit gate fail ⇒ the packaged crops are unusable, detector half reverts to our own stream |
| P4 | Build the **stage-0 kernel spec** (xiaoleilian weights A/B) + the **Option D discriminator** spec under `scripts/kaggle_specs/`; `kaggle_factory build` and verify locally (build is offline and inert) | ~1 h, **0 GPU** | any edit failing its exact-match assertion ⇒ fix before pushing |
| P5 | Write `h1r_edge_data.py` once P2 settles the resample | ~1 h, **0 GPU** | blocked on P2 |
| P6 | Dry-run `h1r_det_train.py` against the real zh001r pack on CPU with `H1R_MAX_STEPS=2` | ~20 min, **0 GPU** | crash ⇒ fix before spending a Colab hour |

**P2 is the gating item for the edge half and it costs nothing.** Do it first.

### From 2026-08-24 — GPU, all `[GL]`

| # | stage | pool | GPU-h | slots | kill criterion |
|---|---|---|---|---|---|
| S0 | **`[GL]` xiaoleilian retrained-weights A/B** (§6.3) — their detector into our pipeline, everything downstream frozen | Kaggle | 2–4 | 0 (LOEO leg) | Architecture incompatible **and** the node-level fallback also fails ⇒ record the negative and go to S1 anyway, but downgrade H1's prior |
| S0b | **`[GL]` Option D discriminator** (§6.5) — deployed full-weight pipeline on the LOEO crops, to separate "crop population" from "per-fold weights" | Kaggle | 2–4 | 0 | If neither full-weight nor per-fold LOEO tracks the LB, **no train-crop instrument can gate a submission** — the most valuable available negative; all gating moves to slots |
| S0c | **`[GL]` S0 leaderboard leg** — only if S0's LOEO is non-trivially positive | Kaggle | ~2 | **1** | LB +0.000 from a higher-ranked team's retrained detector ⇒ **pause H1**, reconsider the whole lane before spending Colab |
| S1 | **`[GL]` detector arm** on zh001r, init from public 50-ep weights, `det_neg_weight` 0.1 (§4) | **Colab** | 0.5–2 | 0 | Deployed-threshold detection F1 does not beat the zero-epoch public-weights baseline **at any threshold** ⇒ kill the detector half |
| S1b | **`[GL]` AMP + s/step measurement**, first 50 steps of S1 | Colab | 0.3 | 0 | fp16 speed-up < 1.3× ⇒ the run is I/O- or Python-loop-bound; fix that before more arms (recipe A1) |
| S2 | **`[GL]` B2 background-weight ablation**, 2 arms (1e-2 vs 1e-1) | Colab | ~1 | 0 | `N_pred/N_est` does not move on the deployed substrate ⇒ the structural-over-detection story is wrong and the retrain must be re-motivated (§4.8) |
| S3 | **`[GL]` competition LOEO on the retrained detector**, both directions, patched scorer | Kaggle | 4–6 | 0 | Bilateral gain ⇒ permission to spend a slot. **Not** promotion (§6.6) |
| S4 | **`[GL]` submission** | Kaggle | ~2 | **1** | The only promotion evidence. Record a written prediction band *before* the score |
| S5 | **`[GL]` edge half** (§5) — only after S1–S4 resolve, and only if P2 settled the resample | Colab | 5–9 | 0 | No cross-embryo edge generalisation (ZSNS003→004) ⇒ kill before touching the competition |
| S6 | **`[GL]` C1 AdaBN / C3 surgical-FT arms** — the recipe's cheap domain-adaptation levers, now affordable (§1) | Colab | ~2 | 0 | run only if S1 cleared |

**Budget check.** Kaggle legs S0+S0b+S0c+S3+S4 ≈ **12–18 h** against ~30 h/week — fits, with room
for a re-run. Colab legs S1+S1b+S2 ≈ **2–3.5 h**, and with S5+S6 ≈ **9–15 h** against ~45 h/week —
fits comfortably. **The programme is not compute-bound. It is bound by submission slots (12
consumed) and by the unresolved question of whether any offline instrument predicts the LB.**

### Ordering rationale, stated plainly

The expensive thing here is not GPU time, it is **being wrong for a month**. S0 exists because a
higher-ranked team has already published retrained weights, which converts the central H1 question —
*does a retrained detector move our score?* — from a month of work into one inference run and one
slot. Running S1 before S0 would spend Colab hours to answer a question we can buy cheaper.

---

## 8. Open items, ranked

1. **`[P2]` The OME-Zarr metadata re-audit** — zero cost, blocks the entire edge half, and until it
   is done §5.1 leaves us knowing the recorded scale is wrong without knowing the right one.
2. **`[S0]` Whether a retrained detector moves our LB at all** — the central premise, now cheaply
   testable.
3. **`[S0b]` Whether *any* train-crop instrument predicts the LB** — the methodology question the
   arm-B failure opened and that nothing since has closed.
4. **The AMP path is unverified on GPU** (§3) — CPU tests forced `H1R_AMP=0`.
5. **Colab device class is non-deterministic** (§1) — on CC ≥ 8.0, bf16 beats fp16 + GradScaler;
   record `get_device_capability()` every run.
6. **The frozen edge head under a moved detector** (§4.4) — the reason evaluation must be
   end-to-end, and an unquantified risk until S3.

## 9. Licence tags (factual, no exclusions)

- `vendor/kaggle-cell-tracking` — BSD-3-Clause, © 2026 Thibaut Goldsborough; permissive, safe to patch.
- Zebrahub imaging + tracks — **CC BY-NC**, host-cleared (#734330); record in any shipping notebook.
- `kkunizaw/biohub-zh001r` — third-party derivative of CC BY-NC Zebrahub; attribute.
- `xiaoleilian/biohub-m001-ens3-sm6-sim2`, `.../biohub-unet3d-weights-v2models` — public Kaggle
  datasets under the uploader's chosen licence; record it in the shipping notebook.
- Ultrack, Trackastra, Spotiflow, nnU-Net — mechanisms reimplemented from published descriptions;
  verify each repo's licence before vendoring any code.

---

*Sections 0–9 complete. Written 2026-08-18; re-checkpointed after each milestone.*
