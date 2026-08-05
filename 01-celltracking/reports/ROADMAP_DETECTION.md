# Roadmap — the detection route

> ## AMENDED — see `reports/ROADMAP_DETECTION_ADDENDUM.md` before acting on this file.
>
> Three claims below are corrected there:
> 1. **`neg_weight=0.1` is WRONG.** The baseline path is `det_neg_weight=0.01` (`train()` and the
>    CLI both), and `reports/ENVIRONMENT_TRAPS.md` had already documented the 0.1 default as a
>    trap. Aggregate loss mass is 1.0 positive against 0.01 negative, so the background term is
>    **1% of the detection loss**, not the dominant force this file implies. `det_loss_weight`
>    is separately **unresolved** (1.0 via CLI vs 10.0 via `train()`, help text contradicts itself).
> 2. **D1 class D does not imply an encoder problem.** `detect_head` is
>    `Conv3d(32, 1, kernel_size=1)` — **33 parameters**. D1-F, a frozen-feature linear probe, is
>    the decisive gate before any encoder retraining.
> 3. **Linajea mask-only is not a safe deployment loss.** It leaves background unconstrained by
>    design; our metric charges surplus nodes and assignment stealing. It is an **ablation (H1)**,
>    never the final design.
>
> The primary mechanism is now **M2-CTPU** (masked + training-only count prior + temporal
> pseudo-positives), not mask-only.

**Written:** 2026-08-05 · **Platform:** P3 harmonic, public **0.915** · leader 0.948 · gap **0.033**
· slots consumed 11. No deadline constraint. Overnight CPU and submission authority granted.

---

## 1. The thesis, and a natural experiment that supports it

`compute_detection_loss` sets one voxel per annotated cell to 1.0 and applies a `neg_weight=0.1`
background gradient to **everything else**, with no ignore mask. Annotation covers 0.655% of cells
on 44b6 and 8.529% on 6bba. So **91.5–99.3% of real nuclei are explicitly supervised as
background** — the detector is trained to suppress the class it then fails to detect.

**The LOEO folds are a natural experiment on exactly this, and it was already in our data.**

`UNetNodeTransformer` is a single model loaded from a single `state_dict`; the file named
`edge_predictor_best_split_{0,1}.pth` carries the **detection head too**. Our LOEO measurement is
therefore fold-honest on the detector — fold 1's detector never saw the 6bba crops it was scored on.

| fold | detector trained on | annotation density of training family | tested on | **missed GT nodes** |
|---|---|---:|---|---:|
| 0 | 6bba (128 crops, 113,121 nodes) | **8.529%** | 44b6 | **1.37%** |
| 1 | 44b6 (71 crops, 20,197 nodes) | **0.655%** | 6bba | **13.28%** |

**The fold trained on the sparser supervision generalises ~10× worse.** That is the predicted
direction of sparse-supervision damage, measured on held-out data, with no extra compute.

**Confound, stated honestly:** 44b6 also has fewer crops and 5.6× fewer annotated nodes, so this
conflates annotation *density* with training-set *size*. Both point the same way — less usable
positive signal — and the masked-loss fix addresses either. But the experiment does **not** isolate
density, and should not be quoted as if it does.

**Ceiling:** +0.103322 on actual FN edges, −0.008633 node-ratio headwind, **net ≈ +0.095**.
Association's entire oracle ceiling is +0.033. Detection is the only asset that can close 0.033.

---

## 2. What is NOT the problem (already measured, do not re-derive)

| ruled out | evidence |
|---|---|
| NMS / peak merging | grid is isotropic at 1.625 µm; pool suppresses ±1.625 µm vs ~6.5 µm cell spacing |
| target parameterisation | perfect-heatmap recall ceiling **1.0000 at every sigma** tested |
| downsample target collapse | **0 collisions** in 133,318 GT nodes |
| wrapper removing candidates | class C ≈ 0 — missing nodes were never in the pre-wrapper set |
| localisation / recentering | already closed: oracle +0.009, deployed −0.0088 |

---

## 3. Phase 0 — gates. Nothing downstream starts until these land.

### P0.1 · Re-base the census onto P3 (CPU, overnight)
Every Lane B/C number — the 72.2% detection share, the +0.0329 association ceiling, the 0.040245
starvation base rate — is computed on P0-B. P3 changes the edge logits and therefore the edge-FN
population. **Re-run `edge_fn_census.py --substrate prewrapper` against a P3 pregraph.**
*Blocker:* we have no P3 pre-wrapper dump. Needs one GPU run with the pregraph export edit
(`loeo_pregraph_export.py`), which already exists.

### P0.2 · D1 detector-response audit (1 GPU run, ~30 min) — **the gating measurement**
Classify each of the 15,296 missing nodes: **A** local max but under threshold · **B** over
threshold but pool-suppressed · **D** no usable response. Exactly computable from
`is_peak = (logits == pooled) & (sigmoid(logits) > threshold)`; design is in `D1_DESIGN.md`.

- **A dominates** → the signal is there and we threshold it away. Fix threshold/calibration.
  **Cheap. May need no retraining at all.**
- **D dominates** → representation failure. Phase 1 is authorised.
- Predicted before measurement: A + D dominate, B small, C ≈ 0.

### P0.3 · Training preflight (1 GPU run)
Can we reproduce the 50-epoch baseline at all? One-epoch wall time, memory, deterministic resume,
and whether `kms111201/biohub-cell-tracking-data` is sufficient. **We have never trained this
model.** Everything in Phase 1 is speculative until this number exists.

---

## 4. Phase 1 — the confirmed fix

Ordered cheapest-first. Each is an independent arm, measured LOEO on held-out folds.

**M2-A · masked / ignore-radius loss.** Compute detection BCE only inside a radius of each
annotated centre; unannotated voxels contribute **zero** gradient instead of a negative one.
Source: Linajea, *Nat Biotechnol* 2022, **MIT**, validated in exactly our modality (3D light-sheet
whole-embryo centre detection from sparse annotations). This is a change to one function.

*Gate:* held-out 6bba node recall must improve from the current 86.72%. Recovering a quarter of the
15,296 missing nodes is ≈ +0.024 composite at oracle — already more than double the association
ceiling's realistic yield.

**M2-B · density-balanced sampling**, over-representing dense 6bba-like scenes. Cheap, orthogonal.

**M2-C · Gaussian soft targets** — *demoted*. The recall-ceiling test says this cannot be justified
as a recall fix. Keep only as a possible inference-time separation aid, and combine kernels by
**maximum, not sum**.

**M2-D · nnPU** — last, and blocked: every source treats the class prior `π` as a
validation-searched hyperparameter and we have no densely annotated region to search it on. Also
note our 0.655–8.529% is the **labelling frequency**, not `π`; conflating them is the standard
implementation failure.

---

## 5. Phase 2 — representation learning, and where JEPA actually fits

**Conditional. Only if Phase 1 improves recall but plateaus short of the ceiling.**

The distinction that matters: **M2 fixes a supervision defect; JEPA fixes a representation
deficit.** They are different failure modes, and we have *confirmed* the first and only
*hypothesised* the second. Fix the confirmed one first.

**If we do it, the right variant is temporal (V-JEPA-shaped), not I-JEPA.** Our data is a
time-lapse: ~199 crops × 100 frames of volume, against only 133,318 annotated nodes. The unlabelled
signal is enormous and free.

Concrete design:
- **Encoder:** the existing 3D U-Net trunk, so the pretrained weights drop into the current model.
- **Pretext:** predict the *latent* of a masked spatiotemporal block at `t+1` from context at `t`,
  EMA target encoder, no pixel reconstruction.
- **Why it fits this data specifically:** light-sheet volumes are shot-noise dominated. A masked
  autoencoder would spend most of its capacity modelling noise. Predicting in representation space
  is precisely the right call here — this is JEPA's strongest argument and it applies cleanly.
- **Second-order benefit:** a temporal pretext teaches "what this cell looks like *and where it is
  going*", which is what the **edge head** needs too, not just detection.

**Honest risks.**
1. **Validation is sparse.** We can only measure on annotated cells, so "did the representation
   improve" is hard to see directly. We'd be tuning a pretraining objective against a noisy
   downstream signal.
2. **Collapse.** JEPA's anti-collapse mechanism is BYOL's asymmetry trick. It works empirically;
   there's no guarantee, and debugging a collapsed encoder burns GPU weeks.
3. **Capacity.** 133k annotated nodes is not a tiny label set. Self-supervision helps most when
   labels are scarce *relative to model capacity*; the U-Net here is small. The masked loss may
   simply be sufficient.
4. **It is a multi-week infrastructure build**, not a loss-function edit.

**My honest read:** JEPA is the right *second* move and the wrong *first* one. If M2-A moves 6bba
recall materially, we may never need it. If M2-A helps and stalls, temporal JEPA is the best-argued
next lever in the whole project — better than anything left in association.

---

## 6. Phase 3 — recompose

A better detector invalidates the association analysis. The +0.0329 ceiling, the 4.02% starvation
base rate and the 72.2%/19.8% split are all properties of the *current* FN population. **Re-run
Lane B and C0/C1 against the new detector before spending anything on association.**

---

## 7. Standing rules carried into this roadmap

- **Submission bar ≥ +0.020 pooled**, research promotion ≥ +0.015 — but note P3 scored +0.001 on
  the public board for an unmeasured OOF, so magnitude transfer remains unmodelled.
- **Trust OOF ranking, not magnitude.** Measured transfer so far: arm B +0.0085 OOF → 0.000 public.
- **Predict with churn-vs-null, not cardinality.** Arm B churned 7.148% → 0.000; P3 churned 4.368%
  → +0.001. Net cardinality change carried no signal.
- **Substrate transfer has failed 7 times.** Re-measure every constant on the substrate it is
  applied to. Never inherit.
- Oracle-clears/selector-fails has occurred 7 times. Every selector needs a base-rate argument
  *before* it is built.

---

## 8. What I need

1. **GPU budget per run, stated up front.** Phase 0 needs ~3 runs; Phase 1 training is unbounded
   until P0.3 gives a per-epoch number.
2. **Standing submission rule I can apply without asking** — the ≥+0.020 bar would have forbidden
   P3, which was still correct to send.
3. **A sanity check when I reverse my own conclusion.** I have retracted a headline twice here.
