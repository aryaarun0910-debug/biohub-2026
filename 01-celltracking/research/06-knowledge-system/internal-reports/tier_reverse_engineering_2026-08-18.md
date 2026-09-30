# Tier reverse-engineering: what separates 0.915 from 0.95

Date 2026-08-18. Scope: harvest and decode every public artifact aimed at the 0.94+ tier, and
rank the candidate mechanisms by evidence. No GPU launch, no push, no submission, no commit.

**Headline.** The hypothesis "the jump is one discrete mechanism" is **refuted**. The evidence
supports **two separable, roughly additive mechanisms** with different owners, and the leader has
*publicly stated on the forum* that his is the edge term, not divisions. Separately, the single
most valuable artifact recovered is not a checkpoint — it is `edge_prune_hgb.npz`, which decodes
completely and hands us a 0.918 team's **13-feature edge-quality contract in full**.

---

## 0. TL;DR ranking

| # | Mechanism | Evidence class | Worth | Confidence |
|---|---|---|---|---|
| 1 | **Association/linking layer on frozen public detections** | Leader's own forum post, quantified | +0.030–0.050 | HIGH |
| 2 | **Division term via retrained, division-aware model** | Competitor forum post, quantified | +0.020–0.030 | HIGH |
| 3 | Retrained detector as a drop-in | Weights public; arch incompatible; recall low | ~0 alone | LOW |
| 4 | Ensembling / TTA | Public, measured, plateaued at 0.910 | +0.002 | REFUTED as the gap |
| 5 | Division **metric exploit** | Patched 2026-07-18, rescored, now *loses* score | negative | DEAD |

Mechanisms 1 and 2 together account for the full 0.915 → 0.951 gap with room to spare. Neither is
the detector.

---

## 1. MEASURED: the harvested artifacts

Downloaded to `data/external/public_weights/` (confirmed gitignored via
`git check-ignore -v` → `.gitignore:2:data/`).

```
.venv/Scripts/python.exe -m kaggle datasets download \
  -d xiaoleilian/biohub-unet3d-weights-v2models -p data/external/public_weights/ --unzip
```

| File | Bytes |
|---|---|
| `unet3d_v2_tophat_b32.pt` | 22,453,556 |
| `unet3d_v4_bright40.pt` | 22,453,336 |
| `unet3d_v4_bright40s1.pt` | 22,453,556 |
| `edge_prune_hgb.npz` | 340,276 |

Licence tag (factual): dataset `xiaoleilian/biohub-unet3d-weights-v2models` declares **CC0-1.0**
(`kaggle datasets metadata` → `info.licenses[0].name`). Recorded only; nothing is excluded or
deranked on licence grounds.

### 1.1 MEASURED: the checkpoints are a different architecture, not a retrain of ours

Each `.pt` is a dict with a real config header — unusually well documented for a competition drop:

| key | `v2_tophat_b32` | `v4_bright40` | `v4_bright40s1` |
|---|---|---|---|
| `base` | 32 | 32 | 32 |
| `pool` | 4 | 4 | 4 |
| `norm` | `p50-p99.5` | `p50-p99.5` | `p50-p99.5` |
| `aug` | `flip` | `bright` | `bright` |
| `preproc` | `tophat` | `` (none) | `` (none) |
| `epochs` / `epoch_best` | 120 / 8 | 40 / 12 | 40 / 12 |
| `n_train_movies` | 171 | 171 | 171 |
| `fpm` (frames per movie) | 0 (all) | 64 | 64 |
| `seed` | 0 | 0 | 1 |
| **`val_recall`** | **0.5789** | **0.6745** | — |

106 tensors, 5,605,359 parameters each. Layer names `e1/e2/e3/bott/u3/d3/u2/d2/u1/d1/out`, with
`e1.0.weight = (32,1,3,3,3)` and `out.weight = (1,32,1,1,1)`.

This is a **plain single-frame 3D U-Net with a 1-channel detection output**. The class definition
is public verbatim in `xiaoleilian/biohub-ct-mix-divaug` (`class UNet3D(nn.Module)`), so the
state_dict loads into that class with a 100% key match.

**Compatibility with our stack — the measurement asked for:**

```python
from tracking_cellmot.models.temporal_unet import TemporalUNet3D
r = TemporalUNet3D().load_state_dict(ck["state_dict"], strict=False)
```
→ `missing: 64  unexpected: 106  intersection: 0  shape-compatible shared keys: 0`

**Zero overlap.** Our `TemporalUNet3D` uses `encoder_blocks.* / decoder_blocks.* /
temporal_blocks.* / head.*`, takes `(B,T,C,Z,Y,X)`, carries per-stage temporal attention, and
emits **32 feature channels** consumed by `UNetNodeTransformer`'s edge head. Theirs is temporally
blind and emits **1 detection channel**. These are not the same model family; there is no
partial-load story, and no fine-tune story either.

**INFERENCE.** The reported `val_recall` of 0.58–0.67 is not obviously better than what our
deployed detector achieves, and there is no edge/association head in these files at all. As a
*detector transplant*, these weights are a weak bet on their own merits, independent of the
architecture mismatch.

### 1.2 MEASURED: `edge_prune_hgb.npz` decodes completely — the feature contract

This is the highest-information artifact in the harvest. It is a raw dump of a scikit-learn
`HistGradientBoostingClassifier` predictor tree array, with the feature names shipped alongside.

```
files: value, feature_idx, num_threshold, missing_go_to_left, left, right,
       is_leaf, offsets, baseline, feats
n_trees 162   n_nodes 9882   nodes/tree 61 (constant, i.e. fixed max_depth)
baseline (log-odds) 2.6057   -> prior P(keep) = 0.931
leaf value range [-0.4459, +0.0750]
```

**The feature contract, in order:**

```
0 dist_um    1 sA       2 sB        3 logitdiff  4 vel_cos   5 back_a   6 fwd_b
7 outdeg_a   8 indeg_b  9 alt_a    10 alt_b     11 margin_a 12 margin_b
```

Split counts and root-split counts (4,860 internal nodes total):

| feature | splits | % | root splits | median threshold |
|---|---|---|---|---|
| `dist_um` | 592 | 12.2% | 8 | 1.716 |
| `margin_b` | 589 | 12.1% | **51** | 7.621 |
| `margin_a` | 585 | 12.0% | **26** | 7.351 |
| `vel_cos` | 541 | 11.1% | 4 | 0.9226 |
| `sB` | 495 | 10.2% | 6 | 0.3998 |
| `logitdiff` | 480 | 9.9% | 15 | 0.1144 |
| `sA` | 470 | 9.7% | 17 | 0.4007 |
| `back_a` | 456 | 9.4% | 14 | 14.5 |
| `fwd_b` | 454 | 9.3% | 9 | 16.5 |
| `alt_a` | 105 | 2.2% | 6 | 1.5 |
| `alt_b` | 93 | 1.9% | 6 | 2.5 |
| **`outdeg_a`** | **0** | 0.0% | 0 | — |
| **`indeg_b`** | **0** | 0.0% | 0 | — |

I reimplemented the predictor exactly (tree walk + `baseline`, sigmoid) and swept one feature at a
time about a plausible centre point (`dist_um=3, sA=sB=0.5, logitdiff=0, vel_cos=0.9,
back_a=fwd_b=10, margin_a=margin_b=5, degrees=1`), which gives P(keep) = 0.9574:

```
logitdiff  -3:0.957  -1:0.957  0:0.957  0.3:0.633  1:0.181  3:0.181   <- dominant
sA         0.05:0.763  0.15:0.763  0.3:0.853  0.5:0.957  0.8:0.957
sB         0.05:0.872  0.15:0.872  0.3:0.884  0.5:0.957  0.8:0.957
vel_cos    -1:0.871  0:0.904  0.5:0.902  0.9:0.957  0.99:0.996  1:0.996
margin_b   0:0.941  2:0.943  5:0.957  10:0.983  20:0.995
margin_a   0:0.942  2:0.948  5:0.957  10:0.977  20:0.976
alt_a      0:0.962  1:0.957  2:0.944  3:0.938  5:0.932
dist_um    0.5:0.975  2:0.961  4:0.953  8:0.956  12:0.956   <- nearly flat at centre
back_a/fwd_b  ~flat (0.95-0.97 across 0..50)
outdeg_a / indeg_b   exactly flat (never split)
```

**What this tells us, and it is worth more than the checkpoints:**

1. **`logitdiff` is the decision.** A one-unit margin between this edge and its best competitor
   drops P(keep) from 0.96 to 0.18 — a 5× swing, larger than every other feature combined. The
   pruner is a *competition* model, not a *plausibility* model: it asks "is there a clearly better
   rival for this endpoint?", not "is this edge good?".
2. **`dist_um` is nearly inert once margins are known.** Raw geometric distance, the thing our own
   post-processing leans on hardest, carries almost no independent signal in a 0.918 team's model.
   It has the most splits but the flattest response — it is being used as a *context* variable
   deep in trees, not as a gate. This directly corroborates our own finding
   (`experimental-records.md:885`) that geometric proposal cannot separate good from bad.
3. **Low detector confidence on either endpoint is a strong veto** (`sA` 0.5→0.15 costs 0.19 of
   probability), and it saturates above ~0.5 — confidence *below* threshold matters, confidence
   above it does not.
4. **`vel_cos` near 1 is a strong positive** (0.996 at `vel_cos≥0.99`), i.e. motion coherence is
   used as a confirmer rather than a filter.
5. **`outdeg_a` and `indeg_b` are dead features.** They are in the contract but receive zero splits
   across 162 trees. INFERENCE: the author wired in degree awareness — the natural hook for
   division handling — and the training signal never used it. That is consistent with this being a
   **pure edge pruner with no division mechanism**, and consistent with the team sitting at 0.918
   rather than 0.94+.

**INFERENCE on feature semantics** (names are shipped, definitions are not): `sA`/`sB` = detector
scores at the two endpoints; `logitdiff` = best-rival logit minus this edge's logit (sign
confirmed by the sweep: *higher* → pruned); `vel_cos` = cosine between this edge's displacement
and the source's incoming velocity; `back_a`/`fwd_b` = backward/forward context distances in µm
(thresholds ~14–16 µm); `alt_a`/`alt_b` = count of alternative candidates at each endpoint
(integer thresholds 1.5/2.5); `margin_a`/`margin_b` = per-endpoint assignment margins in µm
(thresholds ~7.4/7.6, suspiciously close to the 8.4 µm 99%-recall linking radius the forum
reports). These are reconstructible on our side from quantities the pipeline already computes.

---

## 2. MEASURED: the division metric exploit is dead

`xiaoleilian/biohub-ct-mix-divaug` (223 votes, the top-voted public kernel) ends with a cell the
author labels, verbatim:

> `OPTIONAL FINAL CELL — division-term augmentation (metric hack)` …
> `Validated offline on host_metric replica: VAL 0.8388 -> 0.9203 (+0.0815).`
> `This is a METRIC EXPLOIT (fake out-of-volume / negative-time hub + forks that game
> division_jaccard). Public score; does NOT commit to final-2. Patchable.`

It appends a hub node at `t=-1000, xyz=-10000` plus 5 forks per dataset.

**It no longer works.** Forum topic 727154 (host, 2026-07-18, 35 votes): a patch was published to
`github.com/royerlab/kaggle-cell-tracking-competition` and **every submission was rescored**
(topic 728324, "COMPLETED: Rescore Underway", 2026-07-22). Topic 733877 measures the before/after
on identical graphs: appending a hub and 40 fake forks moved a one-to-one linker
**0.9839 → 1.0639 under the old metric and 0.9839 → 0.9794 under the new one**. The exploit is now
*score-negative*. The patch (commit `aa65e90`, 17 July) closed three holes at once: divisions now
require directed local topology rather than shared weak connectivity, edges spanning more than one
frame are dropped, and merged edges are collapsed.

**Conclusion: the current leaderboard is entirely post-patch and no part of the 0.94+ tier is
explained by this.** The `divaug` kernel is a July artifact and is a trap, not a lead.

---

## 3. MEASURED: the leader described his own mechanism on the forum

This is the single most decisive piece of evidence in the harvest. Forum topic **735352**
("Possible big leaderboard shakeup"), 2026-08-15, poster **TWEAK** — currently rank 1 at 0.951:

> "I can't speak as to what others in the top 10 are doing, but **we are not focused on 0.0001 or
> Division J**; we are working on a **universal plugin** that the bio cell team can plug into their
> current pipeline with minimal changes. We have tested our plugin with **every available unique
> public notebook and model, with gains ranging from 0.030, 0.040, to 0.050 instantly just
> attaching our plugin**. We've seen gains from a single public model reach a score of **0.940
> untuned**."

Read literally: the leader's gain is (a) **not** the division term, (b) **model-agnostic**, (c)
applied **on top of existing public detections**, (d) worth **+0.030 to +0.050**, (e) enough to
take *one* public model to 0.940 with no tuning. The only layer that satisfies all five is the
**association/linking layer between detection and graph export**.

In the same thread, 2026-08-18, a different competitor states the complementary case:

> "I trained mine from scratch since **my division score is quite good [0.3]** and **edge is really
> bad [it's ~0.01 points below public notebooks]** so I'm trying to improve that."

**Arithmetic check.** `SCORE = adj_edge_jaccard + 0.1 × division_jaccard`. That team:
`0.9144 − 0.01 + 0.1 × 0.3 = 0.934`. The leaderboard has a dense cluster at 0.933–0.936 dated
2026-08-17/18. Self-consistent.

**Our own decomposition.** We print 0.915 with divJ ≈ 0.0065, so our `adj_edge ≈ 0.9144`. The gap
to the leader is **0.0366**. Two ways to close it:

- *All divisions*: needs divJ = 0.366. Our own measured post-processing ceiling is
  ≈ 0.1 × (35/151) = **+0.023**, i.e. divJ ≈ 0.23 at *perfect* precision
  (`experimental-records.md:885`). So divisions alone **cannot** close the gap.
- *All edge*: needs `adj_edge = 0.951`. TWEAK claims exactly this magnitude, model-agnostically.

**INFERENCE: the tier is edge-first, division-second, and the two are additive.** 0.9144 + 0.036
(edge plugin) is the leader; 0.9144 + 0.023 (divisions, near-perfect precision) is ~0.937, which
is where the mid-0.93s cluster sits. Both routes are occupied by real teams who have said so.

---

## 4. MEASURED: the rest of the public surface, and what it does *not* contain

Commands: `kaggle kernels list -s biohub --sort-by scoreDescending`,
`kaggle datasets list -s biohub --sort-by updated`,
`kaggle competitions leaderboard -c biohub-cell-tracking-during-development -s`,
`kaggle competitions topics list -c biohub-cell-tracking-during-development` (pages 1–3).

**Leaderboard shape — the brief's prior is wrong.** There is no "thin 0.919–0.930 band". Measured
2026-08-18: 0.951, 0.950, 0.948, 0.947, 0.945×3, 0.943×2, 0.942, 0.941, 0.940, 0.939×2, 0.936,
0.935×3, 0.934×3, 0.933×5, 0.932, 0.931, 0.930×5, 0.929 … down through a *continuously populated*
0.921–0.930 range with 40+ teams. The cliff is real only at the 0.915 public-notebook pile-up; the
tier above it is a smooth gradient, which is itself evidence *against* a single discrete unlock and
*for* an incremental mechanism applied with varying skill.

**Top public kernels, by votes:**

| Kernel | Votes | What it is |
|---|---|---|
| `xiaoleilian/biohub-ct-mix-divaug` | 223 | 2-model UNet3D heatmap **average** + Hungarian + gap-close + short-track filter + linefit, **plus the dead exploit cell** |
| `kaiwalyaatulraut/biohub-cell-tracking-solution` | 201 | derivative |
| `pilkwang/biohub-cell-tracking-two-seeds-logit-blend` | 191 | the 0.911 two-seed blend — our substrate's ancestor |
| `pilkwang/…learned-graph-w-gap-recovery` | 192 | learned graph + gap recovery |
| `yusuketogashi/no-hack-biohub-cell-another-approch-3rd` | 146 | preset-tuned 0.916 ceiling, explicitly `_nohack` |
| `raykkretzschmar/biohub-harmonic-bidirectional-association-v1` | 83 | source of our already-deployed harmonic rule |

**Nothing public scores above ~0.916.** The `no-hack` kernel's own experiment tag is
`biohub_162_forward_acceleration_lookahead_target0916_nohack` — the public ceiling, self-declared.
Its knob surface (`BIOHUB_SAFE_DIV_*`, `BIOHUB_LOCAL_RANKER_*`, `BIOHUB_EDGE_TTA_*`) is already
present in our deployed notebook; we are at parity with the public frontier, not behind it.

**Division-specific content in public artifacts: essentially none.** The only division mechanisms
visible anywhere public are (i) the dead exploit, and (ii) `add_safe_divisions_postlink` with
frac-caps — which we already run at `FRAME_FRAC_CAP=0.0076 / GLOBAL_FRAC_CAP=0.00375`. The 0.94+
tier's division work is **not public**.

**Hard metric facts recovered from the forum** (topic 733877, nekkon, measured against the host's
own evaluation code on hand-built graphs — no competition data needed):

- Over-detection is nearly free: the penalty line is exactly `1 − 0.1 × over-prediction`, so 10%
  more nodes need buy only 1% relative edge-Jaccard gain to break even. **Recall is worth far more
  than precision**, and "most people's thresholds are too high". Our deployed
  `BIOHUB_DET_THRESHOLD` is **0.96875–0.99**.
- *Duplicating* a detection costs ~9% and buys nothing — node matching is one-to-one bipartite, so
  a twin one voxel away matches nothing. Taking the union of two detectors without a merge pass
  pays this in full. (This is exactly why the `divaug` author's inline comment insists on
  **averaging heatmaps, never unioning detections** — "DoG-union U-Net measured 0.661".)
- A Hungarian one-to-one linker forfeits the **entire** division term by construction: no node ever
  has two outgoing edges, so divJ ≡ 0.000. That is 0.1 of the available 1.1 gone before the tracker
  sees an image.

And from topic 733973 (same author, 199 training movies, 128,883 GT links):
displacement median 1.82 µm / p95 5.34 / p99 8.38; an 8.4 µm NN radius reaches 99% of true links;
**151 divisions total, one link in 853, and 112 of 199 movies contain none at all**. GT follows
~22 tracks/movie and 100% of its edges span exactly one frame.

That last number is the shakeup risk the forum is worried about, and it is real: divJ is estimated
from ~151 events, so a divJ-driven score is high-variance between public and private split.

---

## 5. Mechanism ranking with confirmation tests

### 1 — Association layer on frozen detections (+0.030–0.050) — HIGH

*Evidence:* leader's own quantified forum statement; model-agnostic by his description; the public
plateau is uniformly at 0.9144 edge across many different detectors, which localises the deficit
*after* detection; `edge_prune_hgb`'s feature set is a competing team's independent bet on the same
layer; our own records show the deployed chain is post-processing-saturated.

*Cheap confirmation on our side:* we already export scored `*.geff`. Build the 13-feature table
from an existing LOEO export and run the decoded HGB over our own edges — no training, no GPU. If
the pruner assigns confidently-low P(keep) to a **non-trivial fraction** of edges we currently
keep, that is a directly actionable, already-trained second opinion on our association layer. If it
keeps ~93% of everything (the baseline prior), it is uninformative and we learn that for free.
`scripts/core/score_oof.py` then measures the delta against `data/train`.

### 2 — Division term via a *learned* fork ranker (+0.020–0.023) — HIGH that it exists, MEDIUM that we can get it

*Evidence:* a competitor at ~0.934 states divJ = 0.3, from-scratch trained. Our measured state is
divJ 0.0152 (44b6: TP 2 / FP 106 / FN 24) and 0.0047 (6bba: TP 3 / FP 507 / FN 122) —
**613 division FPs against 5 TPs across 199 crops**. Precision, not recall, is the failure.

*The arithmetic that matters:* `divJ = TP/(TP+FP+FN)`. On 44b6, holding TP=2 and driving FP to 0
takes divJ from 0.0152 to 2/26 = **0.077** — a 5× gain **from suppression alone, recovering nothing
new**. Our current forks are pure metric poison. The `experimental-records.md:885` conclusion
stands: geometric proposal ranks true (wide, 7.4–8.9 µm) divisions *last* among thousands of tight
false candidates, so the existing `parent_dist + 0.15*sister_dist` ranker is sorting in the wrong
direction. That is a *ranker* problem, and the competitor's result says a trained one solves it.

*Cheap confirmation:* re-score an existing LOEO export with `BIOHUB_OUTPUT_SAFE_DIVISIONS=0`. If
divJ rises (2/26 > 2/132), we have bought +0.006 by *deleting code*, at zero inference cost. That
is the cheapest experiment named in this report and it needs no new artifact.

### 3 — Retrained detector as a drop-in — LOW

*Evidence for:* forum topic 734604 has multiple voices saying "retrain, the public ckpt has hit a
wall". Weights are public and attachable.
*Evidence against:* zero state_dict overlap (§1.1); `val_recall` 0.58–0.67 is unremarkable; the
owner of these exact weights sits at **0.918**, i.e. this detector demonstrably does *not* reach
the tier; and TWEAK's claim is explicitly that the same plugin lifts *any* public model, which
locates the gain elsewhere. A better detector plausibly helps, but it is not what separates the
tier.

### 4 — Ensembling / TTA — REFUTED as the gap

Topic 730924 is a controlled public log: single seed 0.908 → parameter probes 0.908 → two-seed
logit blend 0.910 → edge Top-K/feature-TTA 0.885. We already run 4-view edge TTA and 8-view
detection TTA. This lever is measured and spent.

### 5 — Metric exploit — DEAD (§2). Score-negative post-patch.

---

## 6. The zero-training A/B, specified

**Framing first, because it changes the recommendation.** The brief proposes attaching
xiaoleilian's detector to test "does a retrained detector help OUR stack?". The compatibility
measurement (§1.1) makes this **not a drop-in**: 0 of 106 keys load. It can only be run as a
*second, architecturally independent detector fused at the heatmap*, which is a strictly larger
change than the brief assumes, on a mechanism ranked **3rd** with LOW confidence. I would run the
§5.2 zero-cost suppression check and the §5.1 offline pruner replay *first* — both need no kernel
at all. The spec below is written out regardless, so the option is costed and ready.

### 6.1 Spec design

Write `scripts/kaggle_specs/p4_xldet_fuse_f0.json`, derived from `p3_base_loeo_f0.json` (the
paired LOEO baseline — so the comparison is against a run that differs in exactly one thing).

**Changes to the base spec:**

1. `datasets`: append `"xiaoleilian/biohub-unet3d-weights-v2models"` (62.5 MB; well inside quota).
2. `slug` → `biohub-p4-xldet-fuse-f0`; `out_dir` → `notebooks/kaggle_p4_xldet_fuse_f0`;
   `name`/`title`/`code_file` renamed to match. `base_notebook` / `base_sha256` unchanged.
3. `expects_submission: false`, `enable_gpu: true`, `machine_shape: "NvidiaTeslaT4"`,
   `enable_internet: false` — all inherited unchanged.
4. Keep **every** LOEO edit from the base spec byte-identical (`BIOHUB_LOEO_FOLD=0`,
   `BIOHUB_LOEO_ARM=strict`, the 71-stem `BIOHUB_LOEO_STEMS` list, `loeo_retarget.py`,
   `loeo_export.py`). The measurement is only valid paired against `p3_base_loeo_f0`.
5. Add one `env` edit on `cell_match: "BIOHUB_PRESET"`:
   `BIOHUB_XLDET_FUSE_WEIGHT` = `"0.35"`, `BIOHUB_XLDET_DIR` =
   `"/kaggle/input/biohub-unet3d-weights-v2models"`, `BIOHUB_XLDET_MODELS` =
   `"unet3d_v4_bright40.pt"`. Weight 0 must be a hard no-op so the arm can be disabled without
   rebuilding.
6. Add one `insert_before` edit carrying a new `scripts/kaggle_edits/xldet_fuse.py`, anchored on
   the detection-TTA patch site already present in cell 5 of the base notebook — the block that
   rewrites `if cfg.det_tta:` in `REPO_DIR/scripts/predict_unet_transformer.py` and normalises
   `det_logits[f]`. That is the single point where all detection views are already being pooled,
   so it is the correct and only fusion seam.

**What `xldet_fuse.py` must do:**

- Define `UNet3D(base=32)` exactly as published in `xiaoleilian/biohub-ct-mix-divaug`
  (`e1/e2/e3/bott/u3/d3/u2/d2/u1/d1/out`), `load_state_dict(ck["state_dict"])` **strict=True** —
  strict, so a silent mismatch fails the run instead of producing garbage.
- Reproduce their preprocessing **exactly as recorded in the checkpoint header**, because a
  detector run under the wrong normalisation is worse than no detector: `pool_xy` mean-pool by
  `POOL=4` in y/x only, then `norm='p50-p99.5'` → `clip((p−p50)/(p99.5−p50+1e-6), −0.5, 6.0)`, and
  `preproc=''` for `bright40` (no top-hat). The divaug notebook's own comment is emphatic that a
  model must see the preproc it was trained with.
- Resample their `(Z, Y/4, X/4)` sigmoid heatmap onto our `det_logits` grid, convert to logit, and
  **average in logit space** at `BIOHUB_XLDET_FUSE_WEIGHT`:
  `det = (1−w)·det_ours + w·det_theirs`.
- **Never union detections.** Fuse the heatmap and let our single existing peak-finder run. Topic
  733877 measures union-without-merge at ~9% cost for zero gain, and the divaug author independently
  measured a DoG-union at 0.661.
- Print fused-vs-baseline node counts per crop. `N_pred = N_est` is the anchor
  (`biohub-metric-semantics` memory); a fusion that moves node count materially has confounded the
  adjusted term and the run is uninterpretable regardless of score.

### 6.2 Compatibility risks

| Risk | Why | Mitigation |
|---|---|---|
| Grid mismatch | Their `POOL=4` mean-pool in y/x only vs our downsample path | Assert shapes at fuse time; abort on mismatch rather than interpolate silently |
| Normalisation mismatch | Their p50–p99.5 clip to `[−0.5, 6.0]` is not ours | Recompute their norm from raw volume independently; do not reuse our normalised tensor |
| Calibration mismatch | Their sigmoid output vs our raw logits | Fuse in logit space; sweep `w` ∈ {0.2, 0.35, 0.5} only after `w=0.35` shows signal |
| Node-count drift | Breaks the adjusted term | Report `N_pred` per crop; treat any material move as a failed measurement |
| Runtime | A second full 3D U-Net pass per frame | 5.6 M params on T4 is minor next to the existing UNet+transformer; still smoke-first |
| Anchor drift | `cfg.det_tta` text must match byte-exact | `expect: 1` on the edit; `kaggle_factory verify` fails the build otherwise |

### 6.3 Fallback if the state_dict does not load

Already resolved by measurement, and this is worth stating plainly: **it does not load into our
architecture** (0/106). There is no `strict=False` fallback worth having — a 0-key partial load is
a randomly initialised network. The two live options are:

1. **Fallback A (the actual design above):** instantiate their published `UNet3D` class and load
   strict — a 100% key match by construction, since the class is the one the weights were saved
   from. This is not a fallback so much as the only correct path.
2. **Fallback B (if fusion shows nothing):** stop. Do not attempt fine-tuning or key remapping.
   The evidence in §5.3 says this detector is not the tier's mechanism; the owner of these weights
   is at 0.918. Redirect to mechanisms 1 and 2.

---

## 7. What I would do next, in order

1. **Zero cost, no kernel:** re-score an existing LOEO export with safe divisions **off**. Expected
   divJ 0.0152 → 0.077 on 44b6 purely by removing 106 FPs. If it holds, that is ~+0.006 for a
   deleted code path.
2. **Zero cost, no kernel:** reconstruct the 13-feature table from an existing `*.geff` export and
   replay the decoded `edge_prune_hgb` over our edges. We get a 0.918 team's trained second opinion
   on our association layer without training anything. The `logitdiff` finding alone — that edge
   *competition margin* dominates edge *plausibility* — is testable against our own ranker's
   ordering today.
3. **Only then** consider the §6 detector-fusion kernel, and only as smoke → LOEO fold 0 paired
   against `p3_base_loeo_f0`.

The thing to internalise from this harvest: we have been tuning a post-processing surface that the
public frontier has also tuned to 0.916, while the leader states his +0.03–0.05 comes from a
model-agnostic layer between detection and export, and a mid-tier competitor gets +0.03 from
divisions we are actively poisoning with a 122:1 false-positive ratio. Neither of those is a
detector problem.

---

## Appendix: commands and sources

```
kaggle datasets download -d xiaoleilian/biohub-unet3d-weights-v2models -p data/external/public_weights/ --unzip
kaggle datasets metadata -d xiaoleilian/biohub-unet3d-weights-v2models
kaggle kernels list -s biohub --sort-by scoreDescending -v
kaggle datasets list -s biohub --sort-by updated
kaggle kernels pull -m xiaoleilian/biohub-ct-mix-divaug
kaggle kernels pull -m yusuketogashi/no-hack-biohub-cell-another-approch-3rd
kaggle kernels pull -m raykkretzschmar/biohub-harmonic-bidirectional-association-v1
kaggle kernels pull -m pilkwang/biohub-cell-tracking-two-seeds-logit-blend
kaggle kernels pull -m nekkon/your-linker-cannot-score-a-single-division
kaggle competitions leaderboard -c biohub-cell-tracking-during-development -s --page-size 60
kaggle competitions topics list -c biohub-cell-tracking-during-development           # pages 1-3
kaggle competitions topic-messages biohub-cell-tracking-during-development <id> -n -1 --format json
```

Forum topics cited:
- 727154 "Division Metric exploit and patch." (host, 2026-07-18) — patch + rescore announcement
- 728324 "COMPLETED: Rescore Underway" (2026-07-22)
- 728551 "Post-patch: is the 0.91+ frontier separated by the edge term or by divisions?" (2026-07-23)
- 730924 "What model/feature diversity helped beyond the two-seed logit-blend plateau (~0.91)?" (2026-07-30)
- 733877 "A one-to-one linker scores 0.000 on divisions — four measurements on the metric itself" (2026-08-08)
- 733973 "The linking radius is 8.4 µm, and divisions are one link in 853" (2026-08-09)
- 734604 "What is the best model for this domain so far?" (2026-08-12)
- **735352 "Possible big leaderboard shakeup" (2026-08-15) — contains TWEAK's own description of the rank-1 mechanism, and the divJ=0.3 competitor post of 2026-08-18**
- 735531 "so to get a score above the public baseline is training?" (2026-08-16)
- 723655 "simple idea: Your Affinity Field Tells Your Fate" (2026-07-07) — where the exploit was first surfaced

External: `github.com/royerlab/kaggle-cell-tracking-competition` (patched scorer, commit `aa65e90`);
`github.com/royerlab/hoct` + `arxiv.org/pdf/2607.11754` (host's higher-order tracker — topic 728551
reports it under-performs a tuned ILP on point detections and runs ~45 min/movie).

Licence tags, factual, recorded only: `xiaoleilian/biohub-unet3d-weights-v2models` → CC0-1.0.
Kernel metadata returned by `kaggle kernels pull` does not carry a `license_name` field for any of
the five kernels pulled; the previously recorded CC0 tag for
`yusuketogashi/no-hack-biohub-cell-another-approch-3rd` in `p3_base_loeo_f0.json` provenance is
unchanged by this report.

Local analysis scripts used (scratch, not committed): checkpoint inspector, npz decoder, HGB
structure/sensitivity reimplementation, and the `TemporalUNet3D` strict=False compatibility probe.
