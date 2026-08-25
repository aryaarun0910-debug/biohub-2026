---
id: 00-system/handoff
title: Handoff
area: 00-system
status: active
updated: '2026-08-24'
owner: biohub
links: []
tags: [handoff, entry-point]
---

# Current handoff

**Status:** ACTIVE 2026-08-24. Branch `master`.
This file is the live entry point. Full direction: [directional-updates.md](../01-research-direction/directional-updates.md).
System map: [README.md](../README.md); architecture: [system-design.md](system-design.md); contract: [CLAUDE.md](../../CLAUDE.md).

## 🎯 LIVE 2026-08-25 — P9 SCORED 0.925 (rank 207/2,693). NEXT ACTIONS ARE CPU-ONLY.

**P9 coupled division transplant scored 0.925** (submission 55753516), **+0.010** — the largest gain of
the campaign and the **first division-class change ever to score positive**. Recorded prediction band
(central 0.919-0.925) HIT. Detail: `../07-outputs/submissions.md`; seven-agent cycle:
`../06-knowledge-system/experimental-records.md` (2026-08-25 entries).

**Public frontier is 0.927 and SATURATED** — 202 teams at >=0.926, **106 at exactly 0.926** (one forked
artifact). Top-3 = **0.953**, leader 0.962, gap **+0.028**. Only **6 teams** are at >=0.947.

### FIVE MEASURED FACTS THAT DRIVE THE NEXT CYCLE

1. **Divisions CAN reach top-3 — my earlier doubt is REFUTED.** Perfect division Jaccard = **+0.0993**
   pooled (3.5x the gap); reach-limited oracle **+0.0583**. BUT **division FP suppression alone caps at
   +0.0027** — P9 was a precision fix; **the next division move must raise TP** (~44 TP at 0 FP, or 72
   at <=100 FP, against 5 today).
2. **The nominator is a hard per-target ARGMAX** — in-degree 1 for all 2,162,040 candidate edges, zero
   with in-degree 2. **70.6% of the 17,001 missing GT edges are a RANKING error** (a wrong parent is
   already nominated); only 29.4% are thresholding. kNN-3 recovers **92.5%** of the misses while
   retaining **99.4%** of today's nominations, at 3.5x set size.
3. **BLOCKER on any candidate widening:** the ILP ranks by `edge_prob` = the same column softmax, and
   **`BIOHUB_ILP_APPEARANCE_WEIGHT = "0.0"` means there is NO abstention price** — every feasible
   candidate is accepted. **"Over-nominate and let the ILP prune" DOES NOT HOLD here.** Restoring
   appearance weight is a PRECONDITION, not an independent lever.
4. **The NMS/peak rule is EXONERATED** — 0 collisions on competition GT, 0.12% on dense Zebrahub,
   ceiling >=99.88%. **Pool-kernel tuning is dead.** The recall loss is the **one-hot delta target**
   (`target[b,zi,yi,xi]=1.0` at a TRUNCATED index), which forces broad low-amplitude blobs — **so the
   monotone-decreasing F1-vs-threshold curve is a SYMPTOM of the target, not a separate bug.**
5. **Two of the top three say the gap is NOT the edge model.** TWEAK (#3, 0.953): a *"universal plugin
   ... gains ranging from 0.030, 0.040, to 0.050"*, no new weights, explicitly not division work
   (discussion/735352). Soheil Ayati (#2, 0.959): *"many 'linking' issues actually originated earlier
   during **node selection**"* (discussion/737101). mikelou1 retrained from scratch -> only **0.928**.

### DO THESE FIRST — all CPU-only or <1 GPU-h, all gate the 21-33 GPU-h spend

| # | action | cost | kill criterion |
|---|---|---|---|
| 1 | **Push `p4_detsweep_export_f0.json`** — built, never pushed. The local-max test is threshold-independent, so the whole [0.5,1.0] PR curve is a CPU replay. Measured recall 0.445 -> 0.563 at p0.5; `edge recall ~ node_recall^2` => **x1.60** | CPU replay | curve does not beat 0.96875 on the official scorer |
| 2 | **`BIOHUB_ILP_APPEARANCE_WEIGHT` 0.0 -> 0.1** on the CURRENT candidate set | 1 kernel | does not hold or improve adjusted Jaccard |
| 3 | **M1 duplicated-source LAP replay** on `C:/temp/preilp_f1_v2/preilp_split1.parquet` | CPU | **<8 of 19 contested daughters flip at any delta_div** |
| 4 | **T3.1 drift gate** — 200 steps detection FT, no drift control, measure association AUC drop | <1 GPU-h | **<1% drop => the whole drift apparatus is unnecessary; do not build it** |
| 5 | **Arm-A no-op control** (LR=0, model still in `train()`) attached to any training claim | 0.1 GPU-h | any gain in Arm A is pure AdaBN |
| 6 | **TTA union instead of mean** — `predict_unet_transformer.py:375-388` averages 4 flipped logit volumes and divides by 4; **averaging annihilates peaks that disagree by +-1 voxel** | 1 kernel | union does not raise node count / edge TP |

### STANDING CORRECTIONS
- **Tune EARLY layers, not the head.** PES (VERIFIED): later layers memorise noise first. Under
  appearance shift + noisy labels the reflex *"freeze the backbone, train the head"* is the WORST choice.
  **Do not apply layer-wise LR decay** — it pushes the opposite way.
- **`assert optimizer.state[p]['step'] > 0`** would have caught the phantom +0.104 on its own.
- **Never select a checkpoint on Zebrahub validation.** Select on the target metric via
  `scripts/core/score_oof.py`, LOEO, both embryo directions separately. Use **WiSE-FT** weight-space
  interpolation to get ~10 candidates per training run for CPU cost.
- **Our LOEO split the CROPS but never the MODEL** — all nine specs attach the public support pack.
  Fold 0 is public-weights-only; fold 1 attaches `aryaarun07/biohub-oof-weights`. **The two folds are not
  the same instrument and must not be pooled** until this is resolved (attachment verified, USE is not).
- **Trust Ultrack's detections, distrust Ultrack's divisions** — it is a deterministic ILP, so its errors
  are structured and input-determined: the case where noise-averaging fails.

### STILL BROKEN — neither GPU lane may launch
**S5** crashes on step 1 (`F.binary_cross_entropy` under CUDA autocast — a trap `h1r_trainer_patch.py:12-13`
already documents) and **deletes every division label** (`h1r_edge_train.py:95-97`, `:165-167`), discarding
the 65,741-link asset that is the lane's entire rationale. **S1** carries three SEV-1 defects: selection at
max-F1 rather than the deployed operating point; single-forward eval against 8-view deployed TTA; and a
fine-tuned trunk that silently rewrites every edge feature.

## LIVE UPDATE 2026-08-24 — S1 smoke v2 passed; full run awaits host green-light

This section supersedes the stale P7/S0/blocker text later in this handoff.

- Live position: **0.915, rank 515 / 2,686**; leader 0.962, top-3 boundary 0.953, gap +0.038.
- **P7 is dead and was not submitted:** 400 nodes / 368 edges, local 0.1090 vs the 0.8907 gate.
  The checkpoint replaced the detector as well as the edge head and is catastrophically
  miscalibrated at the inherited 0.96875 detector threshold.
- **S0 is dead without GPU:** xiaoleilian's checkpoint has 0/106 key-name overlap with our model,
  four pooling levels vs three, and no temporal-attention analogue. Its 6–10 GPU-h return to H1.
- The identity blocker is cleared: private Kaggle dataset
  `aryaarun07/biohub-zh001r-identity` exists. The new executable loader reproduces the authority
  exactly: **1,192,441 continuation + 65,741 division-daughter = 1,258,182 associations**.
- S1 now has reproducible smoke/full specs. Pre-launch defects fixed: exact 1.625 µm geometry,
  global validation padding, strict initialization, zero-epoch threshold sweep, full
  optimizer/scheduler/scaler/RNG resume, and paired fp32/fp16 DataParallel telemetry with a 1.3×
  abort gate. Smoke v1 failed before model/data execution because the vendored trainer's `src/`
  was absent from `sys.path`; the runner now inserts both import roots. Corrected v2 completed on
  Kaggle T4x2: AMP was **3.894x** faster, selected-threshold validation F1 moved **0.6335 ->
  0.7375** after the two-step plumbing run, and the deployed-threshold F1 moved **0.5538 ->
  0.7008**. The smoke never submitted. The full S1 run is technically cleared but awaits the
  host's explicit next-stage authorization under `CLAUDE.md` rule 8.
- The embedding lane is concrete: opt-in 64-D normalized appearance projector over existing U-Net
  node features, zero-gated bounded cosine residual, continuation-only spatial-KNN hard triplets,
  and a top-k pre-threshold candidate sidecar. Legacy logits/graph selection remain byte-identical
  when disabled. A runnable S5 trainer now joins adjacent Zh001r windows, freezes and bypasses the
  detector, masks division daughters, reports top-1/candidate recall, and supports full resume.
  Both a two-step smoke spec and a full S5 spec now attach the public Zh001r pack plus the private
  identity sidecar and emit no submission. S5 remains gated behind S1 evidence rather than being
  promoted speculatively.

## 🎯 HOST DECISION 2026-08-23: **TOP-3 OR NOTHING.** Bronze is not the win condition.

Operator, verbatim: *"top 3 or nothing - im willing to go all out."*

**Consequence — this reorders everything.** Bronze (0.918) is +0.003 away and cheap; top-3 (0.947)
is +0.032 away and needs the retrain. They compete for the same 37 days. The decision is that
**post-processing work is no longer a phase**. The four free config levers
(`OUTPUT_MIN_TRACK_LEN` 6→4, `ADAPTIVE_SHORT_TRACK_RESCUE`, the `GAP_CLOSE_MAX_GAP` clamp, the
det-threshold sweep) are worth running only because they are CPU-cheap and ride along — **they are
not the plan and must not consume calendar.**

**The plan is H1: retrain, on Kaggle, starting immediately.** (Colab is gone -- host, 2026-08-24. See the compute section below: this does *not* change the plan's substance.)

**Why nothing else reaches 0.947 — measured, not argued:**
- Wrapping a public notebook can only put us AT the public frontier, never ahead of it. When
  kimi-v17 (0.923) appeared, 155 teams reached 0.917 within days. 665 teams now sit at ≥0.915.
- The leader moved **0.951 → 0.962 in four days**. Chasing +0.002 increments against +0.01/week
  is losing while running.
- `error_atlas_2026-08-19`: the loss is NOT concentrated (worst decile only 1.62×
  over-represented); abstention is CLOSED (58.3% vs a 58.88% bar); **the candidate generator
  withholds 17,067 detectable GT edges (15.65%)** while a perfect solver over today's candidates
  is worth **+0.0012**. Those are facts about the detector and linker, and no post-processing
  config changes them.

**The one asset the field cannot fork from a public notebook:** `data/external/zebrahub/zh001r_identity.npz`
— **1,258,182 GT association edges** incl. 65,741 division-daughter links, registered onto ZSNS001
(72/72 crops, median residual 4e-5 µm) at exactly the deployed 1.625 µm geometry. We own 151
division events; this is ~436× more. Nobody else appears to have registered those crops.

**Budget:** 36 days to 2026-09-29. **ONE pool: Kaggle GPU, host-stated ~45 h/week** (Kaggle's published
quota is 30 h/wk -- plan against 30, it still fits). Covers training AND inference AND submission.
~165 submission slots at 5/day. **Calendar is still the binding constraint, not slots and not compute.**

## 💻 COMPUTE CHANGE 2026-08-24 — Colab is gone. H1 survives; the *ordering* changes.

Host, verbatim: *"we have only 45 hours on Kaggle GPUs a week no Colab"*. The two-pool budget in
`h1_execution_spec_2026-08-18` §1 is void. **Do not re-plan H1 away — re-read its own numbers.**

**H1 was never compute-bound, and the spec says so.** Its §7 budget check, restated against one pool:

| leg | GPU-h | slots |
|---|---|---|
| S0 xiaoleilian A/B + S0b Option-D discriminator + S0c LB leg | 6-10 | 1 |
| S1 detector arm + S1b AMP measurement + S2 background-weight ablation | 2-3.5 | 0 |
| S3 competition LOEO on retrained detector + S4 submission | 6-8 | 1 |
| S5 edge half + S6 AdaBN / surgical-FT | 7-11 | 0 |
| **whole programme** | **21-33** | **2** |

That is **under one week of the new single pool** even at Kaggle's conservative 30 h. Compute did
not become the constraint; calendar still is.

### What actually changes — five items, and one of them inverts a decision

1. **Device is now deterministic: Kaggle T4x2, CC 7.5.** This **closes spec open-item #5**
   (non-deterministic Colab T4/L4/A100). fp16 + `GradScaler` is now *correct* rather than a gamble,
   and the patch's hard-coded `torch.float16` needs no bf16 branch. A simplification, not a cost.
2. **The DataParallel/autocast trap goes from conditional to guaranteed.** On Colab a single-GPU
   allocation would have made spec F5/F8 moot. On T4x2 every run is `nn.DataParallel`, so the
   patch's "autocast *inside* `TemporalUNet3D.forward`" design is load-bearing on **every** run.
   It is written that way already but is **still unverified on GPU (open item #4)**. S1b is
   therefore no longer optional: measure the fp16 speed-up on the first 50 steps or the run is
   silently ~4x slow. **Escape hatch: Kaggle P100 is single-GPU** and sidesteps DataParallel
   entirely if AMP misbehaves -- pin `machine_shape` explicitly either way.
3. **Internet is available but not free.** `kaggle_factory` supports it
   (`scripts/core/kaggle_factory.py:266` `enable_internet`), but an internet-on kernel cannot
   submit. Pattern: train internet-on -> save weights as a dataset -> attach to an internet-off
   submission kernel. **The streaming lane is not needed at all** -- see item 4.
4. **The edge half's data is ALREADY a Kaggle dataset, which removes a blocker.**
   `kkunizaw/biohub-zh001r` is live and attachable (verified 2026-08-24):
   `zh001r_iso.npy` 377 MB, `zh001r_tgt.npy` 377 MB, `zh001r_nodes.npz` 9.4 MB. So the OME-Zarr
   stream lane is off the critical path, and with it **spec open-item #1 (the P2 metadata
   re-audit) and the `h1r_fetch_imaging.py` 3x4-chunk bug**. Losing Colab's internet costs nothing
   here; moving to Kaggle actively helps, because the training data no longer has to be
   re-downloaded per session.
5. **NEW HARD PREREQUISITE, zero GPU: upload `data/external/zebrahub/zh001r_identity.npz`
   (8.7 MB) as a Kaggle dataset.** Verified 2026-08-24: it is **not** among our datasets. Colab
   would have read it off local disk; a Kaggle kernel cannot. **Nothing in H1 runs until this is
   pushed.** It is minutes of work and it is now on the critical path.

### The inversion — S0-before-S1 was calibrated for a budget that no longer exists

Spec §6.6 puts S0 first so that a cheap proxy (a higher-ranked team's retrained detector) can
**pause H1 before Colab hours are spent**. Restate the costs in one pool:

- **S0 + S0c: 6-10 GPU-h and 1 slot.**
- **S1 detector arm: 0.5-2 GPU-h and 0 slots.**

**The gate now costs 3-5x more than the thing it was protecting, and a slot on top.** "Run S0
first to avoid wasting training hours" was sound when training sat in a separate, scarcer pool. It
is no longer a saving -- sequencing S1 behind S0 buys nothing and **spends calendar, the one
binding constraint.**

**Therefore: run S1 in the same week as S0, not after it.** Keep S0 -- it still answers the
central premise question (*does any retrained detector move OUR LB?*) and it is the only leg that
produces LB evidence, which per the transfer law and the LOEO suspension is the only evidence that
counts. But it no longer gates S1. Only S3/S4 -- the legs that cost 6-8 h and a slot -- stay behind
a gate.

**Unchanged and still binding:** every GPU launch needs explicit host green-light; no submission is
automatic; checkpoint-resume (patch P6, `edge_predictor_last.pth` every epoch) is now genuine
insurance rather than convenience, because a preempted Kaggle kernel loses everything since the
last save.

## Mission

Aggressive climb toward **top-3** (private-set-honest). Deployed **P3 harmonic 0.915** public;
leader **0.950**, top-3 boundary **0.948**, gap **+0.035**. 12 submission slots consumed.
See [scientific-mission.md](../01-research-direction/scientific-mission.md).

## The two facts that now dominate everything (2026-08-18)

1. **VALIDATION CRISIS — LOEO→LB transfer failed.** The arm-B motion-gate measured
   **+0.0144/+0.0090** on our best instrument (clean paired deployment-substrate LOEO, official
   scorer) and scored **+0.000 on the LB** (submission `55585140` = 0.915 = P3 alone). The
   "+0.005 bilateral LOEO" promotion gate is **falsified as sufficient**; no validated preflight
   instrument currently exists. `bet-motion-gate` CLOSED. `bet-subvoxel-refine` also CLOSED
   (bilaterally negative −0.0004/−0.0009). Fourth consecutive optimistic central estimate.
   Diagnosis: [internal-reports/loeo_lb_gap_2026-08-18.md](../06-knowledge-system/internal-reports/loeo_lb_gap_2026-08-18.md).
2. **H1 IS FULLY UNBLOCKED — both halves, from one small attachable dataset (2026-08-18).**
   Registration of the packaged `kkunizaw/biohub-zh001r` crops onto the ZSNS001 Ultrack tracks
   **succeeded exactly**: 72/72 crops, all 1,357,051 nodes, median residual 4e-5 µm. The sidecar
   `data/external/zebrahub/zh001r_identity.npz` (8.7 MB, host-verified: 0/1440 mismatches,
   116,320 tracks) yields **1,258,182 GT association edges** (1,192,441 continuation +
   65,741 division-daughter). **This overturns the 2026-08-17 "edge/association NOT SUPPORTED"
   limit.** Voxel size is **exactly 1.625 µm — identical to the deployed detector input** (the
   earlier 1.677 µm ruler estimate is superseded), so nothing is voided on geometry grounds.
   Transform: `global_um = origin + 1.625 * crop_coord`, isotropic (the tracks CSV is already in
   microns — both the "level-0 voxels" premise and the host's anisotropic-scale guidance were wrong).
   **Bounding caveat:** those labels are Ultrack's automated output, so an edge model trained on
   them learns to imitate Ultrack — a ceiling as well as a floor.
   Local scaffold passes on CPU: `scripts/win_bet/h1r_zh001r_{audit,smoke,register}.py`.
   `zmnscrops` is characterised and carries **no labels** (multi-view ZMNS001/2, no public lineage
   table) — zh001r + sidecar dominates it; don't spend on it.
   **Blocker if the stream lane is ever needed:** `h1r_fetch_imaging.py` assumes 1 y/x chunk per
   frame; ZSNS001 level-1 has 3×4, so it raises. Also `kaggle.exe` is blocked by Windows App
   Control — use `python -m kaggle`.

## 🎯 THE TRANSFER LAW — measured 2026-08-19. This is the operating rule.

The degraded control settled the measurement question: **the LB responds.** It scored **0.883**,
moving −0.032. So the three-way 0.915 tie is REAL, and the arm-B and division kills both STAND.

**What transfers, measured on identical substrates with the official scorer:**

| lever class | example | local Δ | LB Δ | transfer |
|---|---|---|---|---|
| **detection surface / candidate set** | control, 28% node cut | −0.0091 | **−0.0320** | **3.5× AMPLIFIED** |
| division term | divfix (703 divisions) | +0.0071 | **0.000** | none |
| edge permutation | arm B (~6% churn) | — | **0.000** | none |

**Why:** the placeholder crops hold **3 GT divisions between them** (2 of 4 have zero; two sit in
the 1st–2nd percentile of annotation density). The `0.1×divJ` term has almost no hidden-set
headroom; `adj_edge_jaccard` responds hard.

**=> ONLY levers that change the detection surface or the candidate set have demonstrated LB
transfer.** Everything else has three zero-scoring submissions behind it.

**Independently corroborated:** `error_atlas_2026-08-19` measured **17,067 detectable GT edges
(15.65%) never nominated as candidates**, while a perfect solver over today's candidates is worth
**+0.0012**. The candidate generator is the bottleneck; the solver is nearly optimal already.

**Bonus instrument:** the placeholder substrate is now a calibrated directional gauge for
detection-class levers (**understates by ~3.5×**), costing zero slots.

## ⏳ IN FLIGHT — `p7_cleanedge` v3 (decision rule pre-registered)

Kernel `aryaarun07/biohub-p7-cleanedge` v3. P3 harmonic with the edge predictor rebound to
`leevvin/biohub-movie-heldout-edge-predictor-v1` (CC0) — a 195-movie held-out retrain,
host-verified `strict=True` 136/136 into our exact `UNetNodeTransformer`. **Zero training cost**,
and it is the candidate-set class (the edge head's column softmax > 0.5 defines every candidate).

*v1 and v2 failed and were fixed:* the mount is not reliably `/kaggle/input/<slug>/`, and
`REPO_DIR/weights/...` is a **read-only** filesystem. v3 searches `/kaggle/input` for a `.pth` of
exactly 8,355,927 bytes and **rebinds `predict_cmd`'s `--weights`** rather than copying. Injection
ordering verified: `predict_cmd` built at char 25159, anchor 30363, first `subprocess.run` 31871.

**PRE-REGISTERED DECISION (fixed before the result, in `../07-outputs/submissions.md`):**
score locally on the four placeholder crops vs **P3's 0.8907** on the identical substrate.
**SUBMIT if local ≥ 0.8907; DO NOT SUBMIT if below.** Audit must PASS 10/10 regardless.
leevvin held these four crops out, so this is the first checkpoint that is **uncontaminated on the
scoring substrate** — though the detector and secondary are unchanged, so read the **delta**, not
the absolute.

```powershell
.\.venv\Scripts\python.exe scripts\core\kaggle_factory.py status --spec scripts\kaggle_specs\p7_cleanedge.json
.\.venv\Scripts\python.exe scripts\core\kaggle_factory.py fetch  --spec scripts\kaggle_specs\p7_cleanedge.json --files submission.csv run_stats.csv --dest c:\temp\p7_final
.\.venv\Scripts\python.exe scripts\core\score_loeo_submission.py --csv c:\temp\p7_final\submission.csv --gt-dir data\train
.\.venv\Scripts\python.exe scripts\core\kaggle_factory.py audit  --spec scripts\kaggle_specs\p7_cleanedge.json --dest c:\temp\p7_final
```

## 📍 POSITION FOR 2026-08-24

Rank **449 / 2,659**; deadline **2026-09-29** (~5 weeks). Slots are NOT the constraint, **calendar is**.
Compute: **Kaggle only, ~30-45 GPU-h/wk, one pool for everything** (see the compute section).

**Wrapping is exhausted — measured, not judged.** Nine levers dead; three whole classes closed this
week (division post-processing, detection threshold, abstention at 58.3% vs a 58.88% bar).

**The competitive edge above the frontier is retraining, and it is documented:** `mikelou1`
(rank 21, 0.935) went divJ **0.03 → 0.30** between 2026-08-10 and 08-18 by retraining. The 0.917
public stack is fully readable and classical — that is commodity, not edge. TWEAK's "universal
plugin +0.03–0.05" was posted once, never substantiated, never answered the disambiguating
question, and they are rank 3 with 178 submissions while a 9-submission team beats them.

**What we hold that others likely do not:**
1. **1,258,182 GT association edges** (incl. 65,741 division-daughter links) in an 8.7 MB
   attachable sidecar at exactly the deployed 1.625 µm geometry — **436×** our own 151 events.
2. **The transfer law** above — bought for one slot.
3. **The candidate-generator diagnosis** — 15.65% of GT edges never nominated.

**First moves on the 24th:** (i) finish the p7 decision; (ii) start the retrain aimed at
**candidate generation + divisions**, the two places with measured reward; (iii) note the honest
tension to resolve first — `error_atlas` says the worst crops differ by **detection rate**
(0.785 vs 0.955), while the candidate-generation gap is a **linking-stage** loss. Both can be
true; they imply different first moves, and that should be settled before GPU is committed.

## 🚨 THE AUDIT FINDING THAT EXPLAINS THE OTHERS (2026-08-22, host-verified)

**`src/biotrack/wrapper.py` and the deployed notebook are TWO DIFFERENT PROGRAMS, and neither is a
superset of the other.** Measured:

| symbol | `wrapper.py` | deployed notebook |
|---|---|---|
| `DEEPCENTER` | **0** | **73** |
| `SAFE_DIV_VETO` | **0** | 5 |
| `GAP_VETO` | **0** | 7 |
| `VOLUME_GUARD` | **4** | **0** |

`CLAUDE.md` names `src/biotrack/` as "the deployed wrapper". It is not — it is a partial mirror
that is missing the entire DeepCenter veto family AND carries a safety mechanism the deployed
artifact lacks. **Anyone auditing the division path in `src/biotrack/` finds no filter stage and
correctly concludes there is none.** That is exactly how a discriminator sat disabled for weeks.
**Fix `CLAUDE.md` to name the built notebook as the audited artifact — it is a one-line change and
it is the highest-leverage item in this section.**

### Consequences already confirmed

- **The division funnel was structurally blind.** `loeo_retarget.py:131-141` sets
  `DEEPCENTER_SAFE_DIV_VETO = False` in the strict arm; `div_proposal_funnel.py` ran on
  `loeo_split1_strict.csv.gz`. The instrument that closed `bet-division-proposal` measured a
  pipeline where the discriminator **could not fire**.
- **The ILP is structurally inert.** `motion_relink_edges` discards **99.87%** of solver output
  (164,470/164,677) and rebuilds a 1:1 matching. Measured `division_like_sources` =
  `safe_divisions_added` **exactly** (557=557; 703=703 on divfix). **100% of output divisions are
  post-processor artifacts** → all four ILP knobs inert → explains three 0.000 division lanes.
- **`p8_loosefilter` is an incomplete port.** kimi-v17 has `SAFE_DIV_REQUIRE_DIVERGENCE`,
  `SAFE_DIV_DIVERGE_UM`, `SAFE_DIV_REQUIRE_MUTUAL_NN`; we have **none** of them (wrapper: diverge 0,
  mutual 0, orphan 0). p8 copied their loose gates without their filters → expect flat/negative.
  **The real port must implement the divergence + mutual-NN tests.**

### Free levers in the TRANSFERRING class (no GPU, no slot)

1. **`ADAPTIVE_SHORT_TRACK_RESCUE = 0`** while the short-track filter deletes **11,028 nodes (5.9%
   of raw)**; its trigger fires at `removed_frac >= 0.10` but the measured rate is **0.029–0.085 on
   every crop — it could not fire even if enabled**, and is capped at ~1.6% of what was removed.
2. **`GAP_CLOSE_MAX_GAP = "2"` is silently clamped to 1** (`wrapper.py:482` `min(...,1)`) in every
   deployed kernel — the 2-frame bridge never runs, threshold evaluates at 11.6 not 17.4 µm.
3. **`DET_THRESHOLD = 0.96875` never selected**; `p4_detsweep_export` specs are **built and never
   pushed**.
4. **`DUAL_SEED_EDGE_THRESHOLD = 0.48`** vs vendor `cfg.threshold = 0.5` — defines the candidate
   set, zero ablation in 33 reports.

### p8 branch test — no precondition, run it the moment the score lands

`deepcenter_safe_div_accepted` / `_rejected` are in the main stats dict (`C6:1374-1375`) and
already fetched: **14,188 checked, 8,288 accepted, 5,900 rejected (41.6%)**. So the veto DID
discriminate. Therefore **a flat p8 means the division class does not transfer, NOT that the veto
was a no-op.**

### The process failure, stated plainly

103 knobs, 55 levers, 13 bets, 33 reports, a claims table with drift detection — **and the finding
that mattered was six env-var comparisons against a public notebook, by hand, in an afternoon.
The machine produces new measurements and has no routine that re-reads its own configuration.**
Proposed guards: instruments declare `disabled_mechanisms` in their manifest; closing a bet
requires an `already_owned:` grep; delete the "already shipped" label (INHERITED / SELECTED /
MEASURED-WIN only); make the deployed artifact the audited artifact; and a `kaggle_factory` build
assertion that any enabled `*_VETO` must set its `*_THRESHOLD` — which would have caught p8's
untuned 0.12 before it was pushed.

## Bet screening — run BEFORE any lever earns effort (2026-08-18)

~55 levers have been tried; **4 survive** a systematic screen
(`internal-reports/bet_consolidation_2026-08-18.md`). Nine of thirteen live bets pruned (69%).
The failure taxonomy is by ROOT CAUSE, not topic — and each class has a cheap ex-ante test that
would have caught it before the work:

| class | what it is | ex-ante test | cost incurred |
|---|---|---|---|
| **A** | post-hoc surgery over a **frozen detection surface** | does it change the candidate set, or only permute it? | ~13 levers, **2 slots for +0.000 twice**, 4 GPU kernels |
| **B** | discriminability shortfall (signal absent from the features) | required-AUC vs measured-AUC | ≥4 correct zero-GPU kills |
| **E** | optimising inside the metric's **dead zone** (signal present, metric doesn't charge) | scorer unit economics | sub-voxel refine |
| **C** | premise false in code | verify at `file:line` first | 12+ unverified premises; wastes *direction* |
| **D** | measured on a broken instrument | instrument validity + noise floor | the whole LOEO era |
| **F/G** | asset/licence unreality; surrogate-corpus shift | asset check; surrogate fidelity | **G is the untested risk on H1's Ultrack labels** |

**Screening rules R1-R8:** detection surface / separability budget / unit economics /
premise-at-file:line / instrument validity / noise floor / surrogate fidelity / asset reality.
**Note the boundary is "over a frozen detection surface", NOT "downstream of the model"** — the
looser phrasing would wrongly prune the threshold-superset export, our cheapest live lever.

## ⚠ PREDICTION DISCOUNT — apply to every internal estimate

Predicted Δ across scored submissions totals **+0.014**; delivered **−0.001**. Realisation ratio
**≈0.083**. The LOEO→LB slope measured **0.00 (n=2)** against a projected 0.56 — that is a
**suspension of LOEO as a promotion instrument, not a discount on it**.

**Operational rule: promote the stated LOW band to the central estimate.** Four consecutive
optimistic central estimates; the low band contained the outcome where the modal band did not.
Two further lessons from the ledger:
- For P3+armB the **churn heuristic was available and correct and was overridden by the LOEO
  number** — we discarded our best predictor for our worst, and it cost a slot.
- **Code-level reasoning is asymmetric: 1/1 for kills, 0/2 for opportunities.** A kill needs one
  necessary condition to fail; an opportunity needs all of them to hold. Trust it to close doors,
  not to open them.

## Guardrails (prize-critical)

- **No lever gets a submission slot on LOEO evidence alone** (see fact 1). LB A/B is the only
  trusted instrument; slots are the scarce resource.
- Never infer hidden-set quality from the four visible placeholder movies.
- The unmatched-fork division-evaluator pathology is diagnostic ONLY.
- Exact public-trajectory transfer into an identified hidden crop needs written host clearance.
- Preserve `.claude/settings.json`. Stage explicit paths; never `git add -A`.
- Licence tags are recorded as facts in the reports (host decision 2026-08-18: they do not
  filter research or design; they matter only at ship time).

## ✅ ENVIRONMENT REGRESSION 2026-08-18 — RESOLVED, no action needed

The Windows Application Control block on Numba's native DLL
(`DLL load failed while importing _typeconv`) **cleared on its own** — typical of Smart App
Control, which blocks an unrecognised binary until its reputation resolves, then admits it.
**No reinstall was performed**; the venv is untouched.

Verified end-to-end 2026-08-18:

| check | result |
|---|---|
| `import numba`, `llvmlite.binding`, `numba.core.typeconv._typeconv` | all OK |
| numba **JIT compile + execute** (not just import) | OK, correct result in 1.5 s |
| `scripts/core/score_loeo_submission.py` import | OK, `DEFAULT_SCALE = (1.625, 0.40625, 0.40625)` |
| **full scorer run** on `loeo_split0_strict.csv.gz` | OK — **SCORE 0.9033**, divJ 0.0152 (TP=2/FP=106/FN=24), node_recall 0.9842 — reproduces the pre-regression number exactly |
| `pytest -q` | **530 passed, 3 skipped**, 29 failures confined to `test_d1_factorial_smoke.py` (12) + `test_d1_v6_export.py` (17) — **0 new; baseline restored** |
| `.venv\Scripts\kaggle.exe` (also previously blocked) | **now works** — CLI 2.2.4 |

**Note the JIT check specifically:** numba can import and still fail when it compiles, which is
what scoring actually needs — so import alone was not sufficient evidence. Both were tested.

**If it recurs:** it is transient and reputation-based, so retry first. Only if it persists,
`uv pip install --python .venv --force-reinstall numba llvmlite`. `python -m kaggle` remains a
safe substitute for `kaggle.exe` regardless.

## Verification

```powershell
git status
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\core\claims_table.py --check
.\.venv\Scripts\python.exe scripts\core\validate_research_tree.py
```
Baseline: 530 passed + 3 skipped; 29 known failures confined to
`test_d1_factorial_smoke.py` / `test_d1_v6_export.py` — 0 new is the gate.
