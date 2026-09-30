# Public code teardown — reading competitors' actual source

Date of harvest **2026-08-21** (file named `_2026-08-19` per the brief). Scope: pull and read every
reachable public kernel end-to-end, read every reachable discussion thread in full, decode every new
attachable artifact, and separate what is DOCUMENTED (I read the code / the raw thread body) from
INFERRED. No kernel pushed, no submission, no commit.

Prior reads assumed and **not** redone: `tier_reverse_engineering_2026-08-18.md`,
`live_surface_2026-08-18.md`. This report supersedes their *live surface* section — the board and the
public code frontier both moved materially in the three days since.

---

## 0. Mechanism table — what the tier above us does that we do not

Ranked by evidence strength. "We have it?" is measured against
`notebooks/kaggle_p0b_clean913_revtime/biohub-p0b-clean913-revtime.ipynb` (103 `BIOHUB_*` knobs,
enumerated this session).

| # | Mechanism | Evidence class | Where I read it | Measured worth | We have it? |
|---|---|---|---|---|---|
| 1 | **Loose-geometry division *proposal* + topology/dynamics *filter*** — propose second daughters at `d(p,q) ≤ 12 µm`, `d(c1,q) ≤ 15 µm`, then require (a) sisters are **mutual nearest orphans** and (b) they **diverge**: `d(succ(c1),succ(q)) − d(c1,q) ≥ 2.25 µm` | DOCUMENTED code + author's own VAL-24 numbers (div TP/FP/FN **6/31/8**, base 0/0/14) | `reyhanksatria/graph-patches-…-0-917-lb` cell 4; identical in 5 other kernels | VAL-24 0.8387 → **0.8516** (+0.0129) | **NO.** We propose on *tight* geometry (`SAFE_DIV_MAX_UM=4.7`, `SISTER=7.2`) with **no** divergence gate and **no** mutual-NN test — `grep -c diverge\|mutual\|orphan` = 0 |
| 2 | **Sub-threshold candidate reservoir + snap-only gap closing** — keep peaks with `0.05 ≤ score < 0.15` as a *non-node* candidate pool; a 1-frame gap is bridged **only** by snapping to an unused candidate within 3.2 µm, never by synthesising a point | DOCUMENTED code + measured (`+0.0002` on VAL-24, i.e. ~neutral locally but node-count-safe) | same kernel, `_gap_close_1f_snap` | +0.0002 VAL-24 | **NO.** `BIOHUB_GAP_CLOSE_REUSE_EXISTING` reuses *above-threshold* nodes only; we have no sub-threshold reservoir |
| 3 | **Appearance-consistency term in the linker** — `cost += 2.0 × \|logit(s_src) − logit(s_tgt)\|` in µm, added to the velocity-predicted distance inside the Hungarian gate | DOCUMENTED code (`LINK_SIM_W = 2.0`, `_link_sim`) | same kernel | not separately ablated by the author | **NO.** We have `MOTION_RELINK_LEARNED_BONUS` (learned-edge prob bonus) — a *different* signal. Detector-confidence *consistency* is absent |
| 4 | **Retrained edge head, architecture-identical to ours** | DOCUMENTED — **I loaded it**: `model.load_state_dict(sd)` **strict=True OK, 136/136 keys** into our `UNetNodeTransformer` | `leevvin/biohub-movie-heldout-edge-predictor-v1` (CC0) | owner sits at 0.915 (unconverted) | **NO** — and this is the first public checkpoint that is a genuine drop-in for our stack (xiaoleilian's was 0/106) |
| 5 | **Learned edge linker + learned mitosis ranker as 13-feature MLPs** (Linear13→128, 3 residual LayerNorm blocks, head 64→1) | DOCUMENTED — **I reconstructed both, 0 missing / 0 unexpected keys** | `zhongbapapapa/biohub-physics-informed-models` (author rank 88, **0.921**) | author's self-reported proxy `best_div_jaccard 0.987` (**not** the competition metric) | **NO** — but the **feature contract is not shipped**, unlike `edge_prune_hgb`. Not portable as-is |
| 6 | Heterogeneous detector ensemble (3 branches, each with *its own* trained preproc) + flip-quartet logit-TTA + intensity-weighted sub-voxel refine + 4.0 µm **physical** NMS | DOCUMENTED code, six kernels, LB **0.917** verified on the board | `hitoshisaito`, `kunaldesale2408`, `reyhanksatria`, `navazshfathi`, `wuwenmin`, `mtoshidesu` | LB 0.916–0.919 | **Partially** — we have 8-view det TTA + a sub-voxel refine lane in flight; we do **not** run a heterogeneous 3-branch detector |
| 7 | **Shared synthetic-node budget across all repair stages** (one monotone allowance both gap stages draw from, instead of independent per-stage caps) | DOCUMENTED code + explicit metric argument | `lonnieqin/biohub-gap2-joint-node-budget` (author rank 46, **0.929**) | not isolated | **NO.** `GAP_CLOSE_MAX_ADDED_*` and `GAP2_MAX_LINKS_*` are independent caps in our notebook |
| 8 | Single-child division repair (recover the asymmetric 1→1 mitosis) | DOCUMENTED knob, 328 vs 307 divisions claimed | `anhadmahajan06` (0.916), `backtracking` (0.916) | claim, not ablated | **Knob exists** (`BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR`) but is **`0`** in our deployed base |
| 9 | "Universal model-agnostic plugin, +0.030–0.050 on any public model" | **CLAIM ONLY.** Single forum post, never substantiated, direct probe never answered, zero public artifacts | thread 735352 | claimed 0.940 from one public model untuned | n/a — see §4 |
| 10 | Division **metric exploit** (far-away hub + fake forks) | DEAD — patched `aa65e90`, all submissions rescored, now score-**negative** | host posts 727154 / 728324; `yakizakana629` kernel is a July fossil of it | negative | n/a |

**The one-line read.** The public frontier is no longer at 0.915. It is at **0.917–0.919**, it is a
*fully classical* pipeline (no ILP, no transformer, no organizer checkpoint), and the part that
actually moves it is **mechanism 1**: division proposal on *loose* geometry filtered by *topology and
post-event dynamics*. Our own `experimental-records.md:885` already concluded that geometric
proposal ranks true divisions last among thousands of tight false candidates — the public 0.917 wave
is the working counter-design to exactly that failure, and we do not run it.

---

## 1. Method and coverage

```
kaggle kernels list -s {biohub,cell-tracking,zebrafish,tracking,ultrack,trackastra} \
    --page-size 100 --sort-by {voteCount,dateRun,scoreDescending}
kaggle kernels pull <ref> -p <dir> -m          # 41 of 43 targets succeeded
kaggle competitions topics list -c biohub-cell-tracking-during-development --page 1..4
kaggle competitions topic-messages <comp> <id> -n -1 --format json    # 40 threads, FULL bodies
kaggle competitions leaderboard <comp> -d      # 2,591 teams, 2026-08-21
kaggle datasets list -s biohub --sort-by updated
```

**Discussion access is UNBLOCKED.** `topic-messages … --format json` returned 403 in every prior
session; it works now and returns complete thread bodies with dates, votes and message ids. Its one
defect is `authorName: ""` on every message. Author attribution was recovered for two threads via
`https://r.jina.ai/<kaggle-url>` (which also renders the poster's live leaderboard rank) before that
proxy rate-limited. **This changes our standing procedure: use the CLI for content, spend jina only
where attribution is load-bearing.** `WebFetch` on any kaggle discussion URL returns the SPA
`<title>` only — do not retry it.

**41 kernels pulled and read.** Two 403'd — `maulikgajera/biohub-cell-tracking-truncated-track-rescue-patch`
(28 votes) and `chiranjithdharma/replace-midpoint-insertion-with-weighted-interpola` (21 votes) — both
are listed but unpublished, the same pattern as liyansen on 08-18. **72 forum topics indexed, 40 read
in full.**

---

## 2. DOCUMENTED: the public frontier moved from 0.915 to 0.917, and it is a *classical* stack

This is the single largest change since `live_surface_2026-08-18.md` and it was not predicted there.

### 2.1 The lineage

`xiaoleilian/biohub-m001-ens3-sm6-sim2` (68 votes, author rank 127 at 0.918) spawned a fork wave that
propagated hard in three days. I pulled six of them and **diffed the source**: they are the same file
modulo the kernel id.

| Kernel ref | Author LB | Subs | Votes | Last run |
|---|---|---|---|---|
| `hitoshisaito/biohub-kunal-0917-exact-repro` | **0.919** | 23 | 16 | 08-18 |
| `wuwenmin/biohub-kunal-0917-exact-repro` | **0.917** | 9 | 4 | 08-21 |
| `reyhanksatria/graph-patches-for-cell-tracking-0-917-lb` | **0.917** | 10 | 11 | 08-18 |
| `kunaldesale2408/biohub-cell-tracking` | **0.917** | **3** | 63 | 08-18 |
| `navazshfathi/best-score` | **0.917** | 21 | 37 | 08-19 |
| `mtoshidesu/test-biohub-cell-tracking-test` | 0.915 | — | 3 | 08-18 |

`kunaldesale2408` reached 0.917 on **three submissions**. This is a free escalator and 155 teams are
now standing on it (see §2.4).

Licence tags, factual, recorded only: the two weight datasets it needs
(`xiaoleilian/biohub-unet3d-weights`, `…-v2models`) declare CC0-1.0; the kernels themselves carry no
`license_name` field in `kernel-metadata.json`. Nothing is excluded or deranked on licence grounds.

### 2.2 The full pipeline, as read from source

No ILP. No node transformer. No organizer checkpoint. No `tracksdata`. Pure numpy/scipy/torch,
~450 lines, runs on T4.

**Detection.** Three branches, each a `UNet3D(base=24)` — `unet3d_bright.pt` (preproc `''`),
`unet3d_traintophat.pt` (preproc `tophat`), `unet3d_v2_tophat_b32.pt` (preproc `tophat`). Each branch
sees **the preprocessing it was trained with**; the author's inline comment is emphatic about this.
Per branch, 4 flip views `{id, flipY, flipX, flipY·flipX}` — **Z is never flipped (anisotropic)** —
logits inverse-transformed, averaged, **sigmoid after averaging**. Branch heatmaps then averaged
equally. Inline comment, verbatim:

> `# (never union the detections: DoG-union U-Net measured 0.661 -- over-detection`
> `#  blows past T_true and the node-count adjustment punishes it)`

`peak_local_max(min_distance=1, threshold_abs=0.05)`, upsample pooled Y/X by `POOL=4`, then
`_refine()` — intensity-weighted centroid over a `(rz=2, ryx=5)` window with background subtracted —
then **physical** NMS at 4.0 µm using the true `(1.625, 0.40625, 0.40625)` scale.

**Linking.** Two-pass Hungarian on velocity-extrapolated positions
(`pred = P + 0.5·v_prev`), gate 6.0 µm then 10.0 µm for the leftovers, cost
`‖pred_i − c_j‖ + 2.0·|logit(s_i) − logit(s_j)|`, hard-gated on the **raw** (not predicted) distance.

**Post-link, in exact order** — this ordering is load-bearing and the author annotates it:
1. `_gap_close_1f_snap` (mechanism 2) — **before** the short filter.
2. `_short_filter` — drop weakly-connected components with `< 6` nodes **unless the component
   contains a division** (out-degree ≥ 2 protects it).
3. `_linefit` — local degree-1 line fit over ±2 neighbours, blended `0.2·orig + 0.8·fit`.
4. `_add_safe_divisions` (mechanism 1) — **after** linefit, i.e. the divergence test is computed on
   *smoothed* coordinates.

### 2.3 Hyperparameter table (byte-identical across all six kernels)

| Constant | Value | Note |
|---|---:|---|
| `UNET_THRESH` (seed) | **0.15** | sigmoid of a purpose-trained detector — not comparable to our `BIOHUB_DET_THRESHOLD=0.99` on a different head |
| `CAND_THR` (reservoir floor) | 0.05 | mechanism 2's pool is `0.05 ≤ s < 0.15` |
| `NMS_UM` | 4.0 | physical, µm |
| `POOL` | 4 | mean-pool in Y/X only |
| refine window | `rz=2, ryx=5` | intensity-weighted, background-subtracted |
| `TIGHT_UM` / `MAX_LINK_UM` | 6.0 / 10.0 | two-pass Hungarian gates |
| `LINK_SIM_W` | **2.0** | appearance-consistency weight (mechanism 3) |
| `SHORT_MIN` | **6** | overridden from 4 in the final config |
| `LINEFIT_WEIGHT` / `WINDOW` | 0.8 / 2 | identical to our deployed 0.8/2 |
| `GAP1_GATE_UM` / `GAP1_SNAP_UM` | 9.0 / 3.2 | end@t ↔ start@t+2, Hungarian |
| `GAP1_MIN_CAND_SCORE` / `GAP1_CAP_FRAC` | 0.10 / 0.003 | |
| `DIV_PARENT_UM` / `DIV_SISTER_UM` / `DIV_CHILD_UM` | **12.0 / 15.0 / 10.0** | vs our 4.7 / 7.2 |
| `DIV_DIVERGE_UM` | **2.25** | the filter that makes the loose gates safe |
| `DIV_W_SISTER` | 0.15 | ranking score `d(p,q) + 0.15·d(c1,q)` |
| `DIV_FRAME_CAP` / `DIV_GLOBAL_CAP` | 0.0076 / 0.00375 | **identical to ours** — confirms the shared origin |
| `GAP_DT` | 0 | the base multi-frame gap closer is **disabled**; only the snap patch runs |

Author's own measured ladder, verbatim from the notebook:

> `VAL-24, official tracking_cellmot metric (24 held-out movies):`
> `  base 0.8387  ->  +safe divisions 0.8516  ->  +gap(snap-only) 0.8518`
> `  division TP/FP/FN = 6/31/8   (base M001: 0/0/14 — predicts no divisions)`

### 2.4 What the board looks like now (MEASURED, 2026-08-21, 2,591 teams)

| # | Team | Score | Subs | Last sub |
|---|---|---|---|---|
| 1 | **Soheil Ayati** | **0.957** | 33 | 08-19 19:15 |
| 2 | **z7777** (songqizhou) | **0.955** | **9** | 08-20 03:19 |
| 3 | **TWEAK** | 0.952 | **178** | 08-21 00:54 |
| 4 | Mark Cooper | 0.950 | 96 | 08-20 16:42 |
| 5 | yuto083 | 0.947 | 61 | 08-20 15:01 |
| 6 | enddl22 | 0.947 | 124 | 08-21 04:05 |
| 7 | Tang | 0.946 | 48 | 08-20 23:52 |
| 8 | Matt Goldfield | 0.945 | 133 | 08-20 18:14 |
| 9 | **Dylan Gallagher** | **0.944** | **7** | 08-19 14:35 |
| 13 | Dorian Ren | 0.941 | 21 | 08-19 00:13 |
| 15 | fromage _ | 0.939 | 27 | 08-20 00:16 |

Band counts, and the delta against `live_surface_2026-08-18.md`:

| Band | 08-18 | 08-21 | Δ |
|---|---:|---:|---:|
| ≥ 0.950 | 2 | 4 | +2 |
| ≥ 0.945 | 7 | 8 | +1 |
| ≥ 0.940 | 12 | 14 | +2 |
| ≥ 0.930 | 31 | 42 | +11 |
| ≥ 0.920 | 77 | 106 | +29 |
| ≥ 0.918 | 94 | **142** | **+48** |
| **= 0.917** | ~42 | **155** | **+113** |
| **= 0.915** | 281 | **230** | **−51** |

Three readings:
1. **The 0.918 cliff broke.** It was the hard wall on 08-18 (13 teams at 0.918, 4 at 0.919). It is
   now a shelf of 142.
2. **The escalator is the classical fork.** 0.917 went from ~42 to 155 in three days, and the fork
   wave (§2.1) is dated exactly across that window. `dariushafshar/biohub-local-cv-pack` (CC0,
   08-20) independently records the bronze display line crossing **0.916 → 0.917** and notes the
   0.915 block "fell rank 360 → 531 between 08-19 and 08-20 while its score never moved."
3. **We are below the medal line.** `competitions list` reports our rank **369** at 0.915; the
   bronze boundary sits at rank ~258 / displayed 0.917. Our 0.915 is now a queue position that
   decays daily without us doing anything.

---

## 3. DOCUMENTED: the critical prior — verdict on TWEAK

The brief asked me to verify or refute it. **I read thread 735352 in full, every message.**

**The quote is accurate.** TWEAK (`tweakai`), 2026-08-15 13:23, +6 votes, verbatim:

> "I can see why it may appear that way. I can't speak as to what others in the top 10 are doing, but
> we are not focused on 0.0001 or Division J; we are working on a universal plugin that the bio cell
> team can plug into their current pipeline with minimal changes. We have tested our plugin with
> every available unique public notebook and model, with gains ranging from 0.030, 0.040, to 0.050
> instantly just attaching our plugin. We've seen gains from a single public model reach a score of
> 0.940 untuned. We are not focused on the 0.0001 or tuning to the hidden."

**Everything the prior read attached to it is weaker than it looked.**

1. **TWEAK is no longer rank 1.** They are **3rd at 0.952**, behind Soheil Ayati (0.957) and z7777
   (0.955). The jina render of the thread labels the poster "*3rd in this Competition*".
2. **TWEAK posted exactly once in that thread and never came back.** `sersasj` (now rank 98, 0.920)
   asked the precise disambiguating question on 2026-08-15 18:12:
   > "Hi @tweakai, by plugin do you mean something that optimizes/refines the tracking graph, or do
   > you mean you take the public notebook detections and apply your own tracking method on top? Got
   > curious about it, but no worries if you can't share more"

   **It was never answered.** `Moawiz` followed up 2026-08-16 — "interesting to know if the
   calibration was for keeping the track or reassigning" — also unanswered. The only two replies
   after 2026-08-18 are content-free (a thank-you from `farshidamira`, and `komilparmar` saying
   "Trained from scratch").
3. **TWEAK has zero public biohub artifacts.** `kernels list --user {antonoof,tweakai,varianceofx}`
   and `datasets list` for all three members: **nothing** for this competition. The claim is
   unfalsifiable from code.
4. **The framing is a deliverable pitch, not a benchmark.** "a universal plugin that *the bio cell
   team* can plug into their current pipeline with minimal changes" is positioning for a research
   competition's writeup/prize criteria. It is not a leaderboard claim.
5. **178 submissions.** The highest submission volume on the board by a wide margin, and they are
   being beaten by a 9-submission team and a 33-submission team.

**Verdict: the quote is real, the mechanism is UNVERIFIED and INFERRED-ONLY, and the "leader" framing
is now factually wrong.** Treating "+0.030–0.050 model-agnostic association layer" as an established
fact was the weakest load-bearing assumption in `tier_reverse_engineering_2026-08-18.md`. It should be
demoted from HIGH to a *plausible but unsupported* claim, kept alive only because the arithmetic
happens to work (public models sit at 0.907–0.919; +0.030 lands at ~0.940).

**The complementary post is real and stronger.** `mikelou1` — **rank 21, 0.935, 34 subs** — 2026-08-18
00:00:

> "I trained mine from scratch since my division score is quite good [0.3] and edge is really bad
> [its ~0.01 points below public notebooks] so I'm trying to improve that."

Arithmetic: `0.9144 − 0.01 + 0.1×0.3 = 0.934`. They display 0.935. **Self-consistent.** And thread
734192 (2026-08-10) shows the same person at "Division Jaccard roughly 0.03 and edge 0.898" eight
days earlier — so **divJ 0.03 → 0.30 in eight days, by retraining, is a documented trajectory of a
real team.** That is 10× on the division term and it is the best-evidenced single mechanism on the
whole forum.

---

## 4. DOCUMENTED: two new attachable artifacts, both decoded this session

### 4.1 `leevvin/biohub-movie-heldout-edge-predictor-v1` — a genuine drop-in for OUR stack

CC0. 7.7 MB. Published 2026-08-18. 16 downloads. **Author sits at 0.915 (rank 407, 49 subs) — it has
not converted for them.**

**MEASURED compatibility.** I instantiated our vendored model at the default config and loaded strict:

```python
unet  = TemporalUNet3D(in_channels=1, out_channels=32, layers=[32,64,128])
model = UNetNodeTransformer(unet=unet, unet_out_channels=32, pos_feat_dim=4*_POS_EMBED_DIM)
model.load_state_dict(torch.load('edge_predictor_best.pth', weights_only=True))
```
→ **`STRICT LOAD: OK — 136/136 keys`**, 2,076,706 parameters. Prefixes `unet.encoder_blocks.*`,
`detect_head.*`, `transformer.pair_mlp.*`. This is byte-compatible with
`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py::load_model`.

**This is the first public retrained checkpoint that is a drop-in for us.** xiaoleilian's weights were
0/106 (`tier_reverse_engineering_2026-08-18.md` §1.1). This is 136/136.

Author's README, verbatim:
> "UNet warm-started from the public support-pack checkpoint; the node transformer trained from
> scratch. 3 epochs, batch 8, AdamW lr 1e-4, fp16 inputs, 2x T4, ~2.6 h/epoch. Final train losses:
> edge 0.0012, detection 0.0144. Held-out detection accuracy 0.9995, node recall 0.924."

**Caveat I verified from `splits_movie_heldout.json`:** train = 195 movies, test = exactly the four
public-test twins (`44b6_0113de3b`, `44b6_0b24845f`, `6bba_05db0fb1`, `6bba_05b6850b`). Both embryos
are in train. So it is clean against *the public twins* but **is not embryo-disjoint and does not
test cross-embryo generalisation** — the exact failure mode our LOEO harness exists to catch. Three
epochs of transformer-from-scratch on 195 movies is also thin.

### 4.2 `zhongbapapapa/biohub-physics-informed-models` — a 0.921 team's learned edge + mitosis rankers

Published 2026-08-21. Author **rank 88, 0.921, 69 subs**. Three files, 900 KB total. No public kernel
from this user, so the training code is not available.

**MEASURED — I reconstructed both networks exactly (0 missing / 0 unexpected keys):**

```
in_proj : Linear(13→128) + LayerNorm(128)                    # LayerNorm, not BatchNorm —
blocks  : 3 × residual[Linear(128,128), LN, ReLU, Dropout,   #   confirmed: BN gave 14 missing keys
                       Linear(128,128), LN]
head    : Linear(128→64) + ReLU + Linear(64→1)
```

Two heads, same shape: `best_physics_edge_linker.pt` and `best_physics_mitosis_ranker.pt`.
`gpu_normalization_stats.json` ships 13 means + 13 stds per head, plus
`best_edge_jaccard 0.9645`, `best_div_jaccard 0.9865`, `training_time_sec 270.5` on an RTX 6000 Ada.

**Read the 270 seconds.** A 4.5-minute train and a claimed div-Jaccard of 0.987 cannot be the
competition metric — the whole train set has 151 divisions. These are classifier scores on the
author's own constructed pair dataset. Do not import them as competition numbers.

**Feature sensitivity (my sweep, z-scored inputs, one axis at a time from −2σ to +2σ):**

*Edge linker*, P at the population mean = **0.803**:

| feature | µ | σ | P swing | direction |
|---|---:|---:|---:|---|
| f1 | 2.948 | 2.734 | **0.93** | 0.978 → 0.047, monotone **down** |
| f10 | 2.535 | 2.368 | **0.92** | 0.079 → 0.994, monotone **up** |
| f2 | 34.96 | 67.18 | **0.89** | 0.996 → 0.102, monotone **down** |
| f0 | 2.486 | 2.867 | 0.82 | flat-high then falls to 0.181 |
| f9 | 1.037 | 1.747 | 0.65 | non-monotone |

*Mitosis ranker*: P at the population mean = **0.0000**, and only **0.79 %** of random
`z ~ N(0, 1.5)` points exceed 0.5 — it is a hard, sparse discriminator, exactly the *precision*
device our division term needs. Mean z-score of its 400 highest-firing points:

```
f11 z=-2.24 (→ -2.69 vs pop mean +2.795)   <- dominant, and a SIGNED quantity that must go negative
f0  z=-1.30 (→ 1.31 vs 3.081)
f4  z=-1.03 (→ 1.65 vs 4.410)
f6  z=+1.08 (→ 29.71 vs 20.150)
f3  z=+0.98 (→ 6.29 vs 3.746)
```

**INFERENCE (flagged as such).** Feature 11's mean +2.795 / σ 2.448 with firing at −2.69 is the
signature of a *change* quantity — a separation-increase or intensity-change term — that must be
strongly negative. That is structurally the same family as the 0.917 wave's `DIV_DIVERGE_UM` gate,
approached from the opposite sign convention, and independently arrived at by a different 0.921 team.
**Two independent teams above us both filter divisions on a post-event dynamics term.** That
convergence is the strongest indirect evidence in this report.

**Not portable as-is.** Unlike `edge_prune_hgb.npz`, which shipped a `feats` array naming all 13
columns, this artifact ships **no feature names**. We can reconstruct the architecture but not the
input contract. Recorded, not actionable.

Licence tags, factual: `leevvin/…` declares CC0-1.0 in its README; `zhongbapapapa/…` metadata was not
separately queried.

---

## 5. DOCUMENTED: metric and data facts recovered from the forum

Everything here is quoted from raw thread bodies I read this session. Facts already in
`biohub-metric-semantics` memory are not repeated; these are the ones that are new, sharper, or
correct something we hold.

### 5.1 The scorer, probed against the official implementation

`sleepymegacat/the-metric-decides-your-architecture-8-measured` (08-19, 7 votes, author rank 788 at
0.912) reimplements the metric in ~80 lines of numpy/scipy and **validates it against the official
implementation on real ground truth across 14 perturbation families, 6 datasets — "84 cases, 0
mismatches on TP/FP/FN."** The measured facts:

- **Matching is threshold-first, then *maximum-weight* bipartite matching on `1/(1+d)`** — **not**
  minimum total distance. Different objective, different result in dense regions.
- A predicted edge is scored **only** if `(source matched a GT node with out-degree > 0) or (target
  matched a GT node with in-degree > 0)`. Everything else is invisible.
- **`dt != 1` edges are dropped, not penalised.** The scorer filters `t_target − t_source == 1`
  before counting. `metrics.md` and the organisers' own `test_skip_connection_scores_zero` docstring
  say otherwise; **the code wins**. Consequence: a gap *edge* is worth exactly nothing, and the only
  way to recover a missed frame is to **interpolate the NODE** and emit two `dt=1` edges.
- **One duplicate detection 0.4 µm away takes J from 1.000 to 0.500.** NMS quality outranks
  everything.
- **The centroid-error cliff is at σ ≈ 2 µm**, far below the 7 µm match radius, because matching is
  one-to-one and neighbours sit ~9–10 µm apart, so adjacent cells *steal each other's match* — an FP
  and an FN at once. Below 1.5 µm it is free; **σ=2.5 costs 16 %, σ=3 costs 41 %, σ=4 costs 74 %.**
  The curve is identical across both embryos despite nuclei radii of 3.8 vs 5.8 µm, because it is
  driven by *spacing*, not size.
- **Node ids decode as `(t + offset) × BASE + label`**, with `BASE = 1e6` for `6bba` and `1e9` for
  `44b6`, verified on all 199 files with zero exceptions. Decoding them shows **6,206 of the 8,128
  `6bba` file pairs share annotated cells, up to 29 % of the smaller file** — all 128 `6bba` crops
  start at the same frame of one movie and are spatial tiles of it. **"199 samples" is not 199
  independent samples; a random per-video split leaks at the *cell* level.**
- GT is **2.82 % dense** (133,318 annotated nodes vs ~4,725,117 estimated real cells). "A voxel far
  from every annotation is *not* background — it is most likely an unlabelled real cell. Dense binary
  cross-entropy over the volume actively teaches the network to suppress real detections."

**This directly corroborates our sub-voxel refine lane** and puts a number on it: the σ≈2 µm cliff is
1.2 voxels in z and 5 voxels in xy. `dariushafshar/biohub-local-cv-pack` independently reports
**"sub-voxel refinement (+0.0075 CV)"** measured on the official scorer over a 195-volume two-fold
prefix holdout.

### 5.2 The exact-split trick — how to decompose our own submission in ONE probe

Thread 734192, `arulprasadsp` (rank 24, 0.934), 2026-08-10:

> "`summarise()` **drops the division term entirely when a submission contains no divisions at all**
> (score = edge_jaccard if not has_divisions). So a fork-free submission returns your pure adjusted
> edge Jaccard, and the division term follows by subtraction — **one submission for an exact split.**"

We currently infer our decomposition from LOEO. This buys an **exact** public-LB split of our 0.915
into `adj_edge` and `0.1·divJ` for the price of one submission with
`BIOHUB_OUTPUT_SAFE_DIVISIONS=0`. It also happens to be the *same* run as the §7.1 experiment.

### 5.3 The CV-direction trap — the most operationally dangerous post on the forum

Thread 730160, 2026-08-03:

> "The public weights were trained on **all 199 annotated videos**. Their `split_manifest.json` lists
> 199 under train, and the 40 in its own test list are all inside that same set."
> "This cost me a week. I ablated the post-processing in the public pipeline and measured **+0.0184
> from turning one stage off**. … Submitted it and got **0.909 against my 0.912 baseline. The sign
> flipped.**"
> "Those stages are **corrective** — they exist to repair model errors. On memorized videos there
> isn't much left to repair, so they only perturb predictions that were already right. … **A
> training-set harness will keep telling you to delete the components that matter most at test
> time.**"

Same thread, earlier: **"movie-to-movie variability is large: ±0.14, with 18 % coefficient of
variation. Worst movie scores 0.460 and the best is at 0.984."** And `tomasa2` independently: "The
same configuration measured 0.921 on one six-movie sample and 0.824 on another; the 16-movie figure
was 0.882."

**This is a direct warning against the cheapest experiment in the prior report** (§7.1 of
`tier_reverse_engineering_2026-08-18.md`: "re-score an existing LOEO export with safe divisions off,
expected +0.006 by deleting code"). Deleting a corrective stage is exactly the move that flipped sign
for this competitor. It must be run **paired, LOEO, on the official scorer**, and the result read as
"our current forks are net-negative *on unseen embryos*" — never as "delete the code."

### 5.4 Ground truth is Ultrack pseudo-labels, and it is systematically wrong about divisions

Thread 732474:

> "The tracks are algorithm-generated, not manually curated. The `*_tracks.csv` lineages were
> produced with **Ultrack** (Bragantini et al., Royer Lab / CZ Biohub, Nature Methods 2025) … they're
> effectively **pseudo-labels**: they carry Ultrack's own systematic biases and error modes,
> **especially around cell divisions and densely packed regions.** Only small subsets were manually
> corrected for validation, not the full atlas."

And, 2026-08-18 — the sharpest quantified complaint on the forum:

> "each cell's volume after a split should be 0.5 of the volume of the parent before the split.
> However, in GT this is not the case. On average, volume after split in GT is closer to **0.75**. …
> I labelled my own dataset and measure this same measurement at 0.5."
> "I find that I agree with my algorithms 95 % of the time, and GT only 5 % of the time in these
> cases … In some cases GT does not seem to see that the cell has split."

**Implication for the division lane, and it is not comfortable:** our target is a *biased detector's*
opinion of divisions, not biology. A physically-correct mitosis model can be *penalised*. The 0.917
wave's design is consistent with this — it does not model mitosis; it recovers the specific case
Ultrack's own 1:1 assignment drops (mutual-nearest orphans that diverge), which is a model of
**Ultrack's error mode**, not of cell division. That reframes mechanism 1 and makes it more
attractive, not less.

### 5.5 Frozen frames — embryo-specific, and it makes fold-0 vs fold-1 non-comparable

Thread 724283 (+38), quantified reply:

> "**44b6: 71 videos, 0 affected, 0 duplicate pairs, 0 %. 6bba: 128 videos, 114 affected, 947
> duplicate pairs, 7.47 %.** … **89.1 % of 6bba videos are affected.** Approximately one in every 13
> adjacent pairs in 6bba is frozen. **No exact duplicates occur in 44b6.**"
> "Several different 6bba samples have **exactly the same freeze schedule** … 6bba_05b6850b,
> 6bba_07477033, 6bba_5b28472a all freeze after the same frame indices: 4, 12, 27, 42, 52, 57, 59,
> 62, 66, 76."

And thread 729082 (+9): in `6bba_fc516dc6`, frames 81 and 82 are `np.array_equal` **True**, yet the
GT edge moves **8.90 µm** — beyond the 7 µm match radius. **Copying a detection across a duplicated
frame fails to match.**

This is a load-bearing fact for our LOEO harness that I do not believe we have recorded: **fold 0
(44b6) has zero frozen frames and fold 1 (6bba) has 7.47 % frozen pairs.** Any motion-based
mechanism will behave differently across our two folds *for a reason that has nothing to do with the
mechanism*, and 6bba is 113k annotated nodes against 44b6's 20k.

### 5.6 Other verified facts

- **Host, 734330, 2026-08-13:** *"Yes you are free to use the data and all resources in Zebrahub for
  this competition! There is no overlap with the test set."* — H1 is explicitly sanctioned.
- **Host, 723921:** the four visible test movies are byte-identical placeholders, replaced at rerun;
  internet is off in submission mode *"so you can't exfiltrate the real data."*
- **Host, 724386:** *"all the data in this competition followed the same protocol (e.g., instrument,
  developmental stage, etc), each embryo being acquired in a separate imaging session."*
- **Thread 734237, 2026-08-20**, quoting the Overview: *"The size of the hidden test set is
  approximately the same size as the training dataset."*
- **`hengck23` (rank 761, 0.912), thread 726924:** *"It is the lineage (cell division) that will
  decide the winner. … my suggestion is to **learn track without cell division first** … cell
  division is then handled at **post-processing or stage 2** (e.g. classifier to decide if there is a
  split based on appearance changes and longer track cues). it is difficult even for humans to decide
  if there is cell division just based on two frames."*
- **`FOYSAL` (rank 104), thread 730924, 2026-07-30** — the best frontier hint on the forum:
  *"it may be worth examining ambiguous parent links from more than one temporal or model view
  instead of applying a fixed global logit average everywhere. … The next improvement may depend less
  on adding another seed and more on **identifying where the current models disagree — and only using
  the extra view in those uncertain cases.**"*
- **Thread 734604, 2026-08-16 (+3):** *"the current ckpt has kind of hit a wall, it's hard to get
  more gain from post-processing alone."* And 2026-08-12 (+4): *"**Edge recall ≈ node recall² ×
  conditional linking accuracy.**"*

---

## 6. DOCUMENTED: negative results published by others — do not re-run these

`tomasa2/biohub-what-worked-and-what-didnt-for-me` (14 votes, author 0.912) is a paired ablation of
eleven changes on the *learned* stack — our stack's family. Their structural decomposition over six
movies and 1,178 GT edges is the most useful single table in the public corpus:

| | count | share of losses |
|---|---:|---:|
| correct | 1084 | — |
| linked to the **wrong** partner | 52 | **57 %** |
| both endpoints found, **no link made** | 29 | 32 % |
| an endpoint **not detected** | 10 | 11 % |
| division semantics wrong | 3 | 3 % |

> "**Detection is effectively solved** — 0.8 % of edges lost to a missing endpoint, node match rate
> 0.988–1.000. Everything left is association. And the dominant failure is an *active wrong choice*,
> not an omission. … the correct target sits a median **6.08 µm** away against a typical displacement
> of **1.8 µm** — these are cells that *suddenly accelerated*. Extrapolation predicts from past
> velocity, so a cell with no history of moving fast is unreachable by any motion model. **That is
> why every cost variant came back null.**"

Their eight nulls, each paired across movies: divisions (**−0.007**), gap repair (−0.003, better on
0/6), per-movie adaptive gate (+0.001, p=0.85), ILP instead of Hungarian (−0.001, p=0.84), learned
edges as the edge set (worse on 6/10), cost-weight tuning (null; `raw_distance_weight` **bit-for-bit
inert**), harmonic mutual-support on the hints (**provably zero** — the hints are 100 % exclusive so
the fused value is identically 1.0 at every λ), bidirectional fusion inside inference (null across
λ ∈ [0.35, 1.0], non-monotone, carried by one movie).

Also: *"Two of our diagnostics were themselves broken. … gap repair's evidence check, gating on a
frame intensity percentile, rejected 1 candidate in 2,400: an empty frame still has a 92nd
percentile. Replacing it with confirmation from an independent detector rejected 76–87 % and turned a
+0.022 gain into a −0.003 loss."*

Corroborating nulls from `keremelik` (0.912): frozen-CSV surgery **CLOSED** (`min_track ≥7/8/10` =
−0.002…−0.010; gap retune ≈0; false-div sister no-op), and ILP `c_d 1.5→1.4` on a dual-seed stack
**REJECT** (−0.00078, nodes +516 / edges +491 — *"denser tracks → adj-factor tax + hard-clip matched
FP↑. Denser ≠ better."*). From `isakatsuyoshi`'s rule-based CV/LB table (45 votes): adding **division
edges hurt, 0.784 → 0.778**; gap closing roughly neutral; multi-scale DoG was the biggest lever.

**Read together with §2.3 this is the sharpest thing in the report.** Adding divisions naively is
measured *negative* by three independent public teams. The 0.917 wave adds them and gains +0.0129.
The difference is not the geometry — it is the **mutual-nearest-orphan + divergence filter**.

---

## 7. What the 0.93–0.957 tier is actually doing — ranked hypotheses

Our decomposition: 0.915 with divJ ≈ 0.0065, so `adj_edge ≈ 0.9144`. Gap to rank 1 (0.957) is
**0.0426**; gap to the top-3 boundary (0.952) is **0.037**.

### H1 — a *learned, precision-first* division mechanism (worth +0.02 to +0.03). CONFIDENCE: HIGH

*Evidence.* `mikelou1` (rank 21, 0.935) states divJ = 0.3 from a scratch-trained model, and the
forum records the same person at divJ 0.03 eight days earlier — a **documented 10× trajectory**.
`hengck23` names divisions as the deciding factor and prescribes a *stage-2 classifier*.
`zhongbapapapa` (0.921) publishes a **13-feature mitosis-ranker MLP** whose dominant axis is a signed
post-event dynamics term (§4.2). The 0.917 wave gets **+0.0129 on VAL-24** from a hand-coded version
of the same idea (§2.3). Four independent teams, one mechanism family.

*The arithmetic that makes it decisive.* `divJ = TP/(TP+FP+FN)`. Our measured state is
divJ 0.0152 (44b6: TP 2 / FP 106 / FN 24) and 0.0047 (6bba: TP 3 / FP 507 / FN 122) — **613
division FPs against 5 TPs**. Holding TP fixed and driving FP to zero takes 44b6 from 0.0152 to
**2/26 = 0.077**, a 5× gain **recovering nothing new**. Precision, not recall, is our failure, and
every public number says precision is what the mechanism buys.

*Why it is not a contradiction that three teams measured divisions as negative* (§6): they all added
divisions on *geometric* proposal with no dynamics filter. That is what we do too.

### H2 — a retrained edge head, validated leave-one-embryo-out (worth the edge term). CONFIDENCE: HIGH that it is necessary, MEDIUM on size

*Evidence.* Thread 734604, 2026-08-16: *"the current ckpt has kind of hit a wall, it's hard to get
more gain from post-processing alone."* Thread 730160 proves the public checkpoint memorised all 199
movies, and that a training-set harness points the **wrong way** on corrective stages.
`tomasa2`'s decomposition puts **89 % of the remaining loss in association**, of which 57 % is
*actively choosing the wrong partner* on suddenly-accelerating cells — a discrimination failure, not
a gate-width failure, and therefore a model problem. Our own `edge_prune_hgb` decode said the same
thing from the other side: `logitdiff` (competition margin) dominates and `dist_um` is nearly inert.

*Why it is not settled.* Every public retrain lane is **failing to convert**: leevvin 0.915,
kkunizaw 0.887, horaz0 0.914, `rudispresence` (HOCT hard-negative) 0.831, g1en114 0.915. Retraining
is necessary but plainly not sufficient, and the public evidence is that it is easy to do badly.

### H3 — the 0.917 classical post-processing stack, as a set of portable operators. CONFIDENCE: HIGH (it is measured on the live board)

Not a hypothesis about the top tier — a hypothesis about **us**. Three of its four operators are
absent from our 103-knob surface (§0, mechanisms 1–3), it is worth **+0.002 displayed** over our
current position on the board, and it costs no GPU training.

### H4 — a model-agnostic association plugin worth +0.03–0.05. CONFIDENCE: LOW (demoted)

One unsubstantiated forum post, no artifacts, direct probe unanswered, poster now rank 3 with the
board's highest submission count. See §3. Kept alive only by arithmetic plausibility.

### H5 — ensembling / TTA / knob tuning. REFUTED as the gap

Measured publicly and repeatedly: single seed 0.908 → parameter probes 0.908 → two-seed blend 0.910
→ edge Top-K/feature-TTA 0.885 (`mige551`, thread 730924). `tomasa2`'s eight nulls. `keremelik`'s
two CLOSED waves. We already run 4-view edge TTA, 8-view detection TTA, harmonic bidirectional
fusion, dual-seed blending and a retention guard. This lever is spent.

### The low-submission signal, which cuts across all of the above

**z7777: 0.955 on 9 submissions. Dylan Gallagher: 0.944 on 7. Soheil Ayati: 0.957 on 33.**
Against TWEAK's 0.952 on 178. A mechanism that lands at 0.955 in nine probes was not tuned in — it
was **carried in**. That is the signature of a structural change (a model, or a stage-2 classifier),
not a post-processing sweep. It is also, bluntly, evidence against the plugin narrative.

---

## 8. Directly portable to us — five implementations, costed, each with a falsification test

Ordered by (evidence × portability) / cost. All five are specified against
`scripts/kaggle_specs/*.json` + `scripts/kaggle_edits/*.py`, paired against
`p3_base_loeo_f0` / `_f1` so exactly one thing differs. **Nothing here is authorised to launch.**

### P1 — Division proposal/filter inversion (mechanism 1). Cost: one edit file, two LOEO folds. HIGHEST PRIORITY

**What.** Replace the proposal/accept logic in our safe-division stage. Today we propose on tight
geometry and accept on frac-caps. Instead: propose on **loose** geometry, filter on **topology and
post-event dynamics**.

**Sketch.** New `scripts/kaggle_edits/div_divergence_gate.py`, inserted at the existing
`filter_output_graph` anchor already used by `degree_invariants.py`, wrapping our safe-division pass:

1. Widen the proposal gates behind new env vars, defaulting to the 0.917 wave's measured values —
   `BIOHUB_SAFE_DIV_MAX_UM 4.7 → 12.0`, `BIOHUB_SAFE_DIV_SISTER_MAX_UM 7.2 → 15.0`, and a new
   `BIOHUB_SAFE_DIV_CHILD_MAX_UM = 10.0` sanity gate on `d(parent, c1)`.
2. Add **two hard filters** (both new, both must pass):
   - **Mutual nearest orphan.** Build a per-frame KD-tree over in-degree-0 nodes at `t+1`. Candidate
     `q` is admissible only if `q` is the nearest orphan to `p` *and* `c1`'s nearest orphan is `q`.
   - **Post-division divergence.** Both `c1` and `q` must have exactly one successor at `t+2`, and
     `d(succ(c1), succ(q)) − d(c1,q) ≥ BIOHUB_SAFE_DIV_DIVERGE_UM` (default **2.25 µm**).
3. Rank survivors by `d(p,q) + 0.15·d(c1,q)`, accept greedily 1:1, keep our existing frac-caps
   (`0.0076` / `0.00375` — already numerically identical to theirs).
4. Run it **after** linefit smoothing, as they do — the divergence test is on smoothed coordinates.
5. `BIOHUB_SAFE_DIV_DIVERGE_UM = 0` **must** be an exact no-op reproducing today's behaviour, so the
   arm can be disabled without rebuilding.

**Cost.** ~120 lines. No new dataset. No GPU training. Two paired LOEO folds.

**Falsification test.** Paired LOEO fold 0 and fold 1 against `p3_base_loeo_f0/f1`, official scorer,
both embryo directions reported separately. **Report division TP/FP/FN, not just divJ.** The
mechanism is *precision*: it predicts FP collapses (44b6: 106 → tens) while TP holds or rises
(2 → ≥2). **It is refuted if FP does not fall by ≥50 % on fold 0**, or if node count moves materially
(this stage adds edges, not nodes, so `N_pred` must be *unchanged* — any drift means a bug, and the
adjusted term is confounded). Their VAL-24 reference is 6/31/8 on 24 movies.

**Prior risk, stated:** §5.4 says GT divisions are Ultrack pseudo-labels with a documented systematic
bias. This mechanism is a model of Ultrack's 1:1-assignment error mode, which is *why* it should
work — but it also means the ceiling is Ultrack's opinion, not biology.

### P2 — Sub-threshold candidate reservoir + snap-only gap closing (mechanism 2). Cost: one edit file, one fold

**What.** Retain peaks below the node threshold as a *non-node* candidate pool, and let 1-frame gap
closing bridge **only** by snapping to an unused candidate.

**Sketch.** Two touch points.
1. **Export side** — in `predict_unet_transformer.py`, at the peak-extraction site our detection-TTA
   patch already rewrites: emit a second array of peaks in `[BIOHUB_CAND_THR_LO,
   BIOHUB_DET_THRESHOLD)` with their scores, carried alongside the graph (they are **not** nodes and
   **never** enter `N_pred`).
2. **Repair side** — a new `scripts/kaggle_edits/cand_snap_gap.py` replacing the synthetic-point
   branch of `close_single_frame_gaps`: end@t ↔ start@t+2 Hungarian under a 9 µm gate, both endpoints
   required to have their own predecessor/successor (no fragment tips), bridge **only** if an unused
   candidate lies within `BIOHUB_GAP_SNAP_UM = 3.2` of the midpoint with score ≥ 0.10. Never
   synthesise.

**Why it matters more than their +0.0002 suggests.** §5.1 proves a `dt=2` edge is scored as *nothing*
and the only recovery is an interpolated **node**; §5.1 also proves a node 2 µm off costs 16 % and
4 µm off costs 74 %. A midpoint-interpolated synthetic node is a coin flip at that precision. A
snapped real detection is not. **This is a precision upgrade to a stage we already run.**

**Falsification test.** Paired LOEO fold 0. Instrument `n_snapped` vs `n_synthetic_rejected`. **It is
refuted if the snap rate is < 10 % of current synthetic insertions** (the reservoir is empty where
gaps are, so the stage cannot help), or if `adj_edge` does not rise while `N_pred` is held. Cross-check
against `tomasa2`'s broken-diagnostic warning: an evidence gate that rejects 1 in 2,400 is measuring
nothing — **log the rejection rate and require it to be non-trivial before believing any gain.**

### P3 — Appearance-consistency term in the motion relink (mechanism 3). Cost: ~15 lines, one fold

**What.** Add `+ BIOHUB_MOTION_RELINK_SIM_W × |logit(s_src) − logit(s_tgt)|` (in µm) to the relink
assignment cost, inside the gate, gated on the raw distance exactly as they do.

**Why it is not what we already have.** `BIOHUB_MOTION_RELINK_LEARNED_BONUS` consumes the *edge*
model's probability. This consumes the *detector's* confidence at the two endpoints and asks whether
they are the same kind of object. It is the cheapest available proxy for appearance, and
`edge_prune_hgb`'s decode independently found `sA`/`sB` (endpoint detector scores) to be a strong
veto below ~0.5 while saturating above it.

**Cost.** Requires the detector score to be carried onto the node, which our export may already do —
**check before writing anything.** If it does not, this becomes a P2-sized change and drops below P4.

**Falsification test.** Sweep `w ∈ {0, 1.0, 2.0}` on LOEO fold 0. Refuted if `adj_edge` is flat
(`|Δ| < 0.001`) across the sweep — which is the *expected* outcome given `tomasa2` measured
`raw_distance_weight` bit-for-bit inert and every cost-weight variant null on a similar stack.
**This is the weakest of the five and I would run it only bundled with P1/P2, never as a standalone
GPU run.**

### P4 — Exact public decomposition of our own 0.915. Cost: one submission

**What.** Submit our current champion with `BIOHUB_OUTPUT_SAFE_DIVISIONS=0`. Per §5.2, `summarise()`
drops the division term entirely when a submission has no divisions, so the returned score **is** our
pure adjusted edge Jaccard, and `0.1·divJ` follows by subtraction.

**Why it is worth a submission.** Every plan in this report is priced against our estimate
`adj_edge ≈ 0.9144, divJ ≈ 0.0065`, which is inferred from LOEO. This makes it exact on the same
distribution the leaderboard uses, and it is the *same run* as the "safe divisions off" control the
prior report wanted. Budget is not scarce (5/day, 39 days).

**Falsification test.** This is a measurement, not a hypothesis. Two outcomes matter: if the fork-free
score is **> 0.915**, our divisions are net-negative on the hidden set and P1 is worth strictly more
than its VAL-24 number suggests. If it is **< 0.915**, our divisions are already net-positive and P1's
upside is capped at the FP-suppression arithmetic (≈ +0.006).
**Read §5.3 first:** a training-set harness will tell you to delete corrective stages. This test is
run on the *public LB*, precisely so it is not subject to that failure mode — and even so, the public
LB is four twin movies and is itself unreliable. Treat the number as one bit, not as a decision.

### P5 — `leevvin` retrained edge head as a paired detector/edge A/B. Cost: one dataset attach, two folds

**What.** Attach `leevvin/biohub-movie-heldout-edge-predictor-v1` (7.7 MB, CC0) as an alternative
`BIOHUB_PRIMARY_ARTIFACT_MANIFEST` target. **Strict-load compatibility is already MEASURED (136/136,
§4.1)** — there is no integration risk, only a scientific one.

**Sketch.** New spec `scripts/kaggle_specs/p4_leevvin_edge_loeo_f0.json` derived byte-for-byte from
`p3_base_loeo_f0.json`; append the dataset; add one `env` edit pointing the primary weights at the
new `.pth`; keep **every** LOEO edit identical. `enable_internet: false`, `machine_shape:
"NvidiaTeslaT4"` inherited.

**Cost.** No training. One inference run per fold.

**Falsification test.** Paired LOEO fold 0 **and** fold 1, both directions reported. The honest prior
is **negative**: the owner sits at 0.915 on 49 submissions, the split is not embryo-disjoint
(§4.1), and the transformer had three epochs from scratch. **Refuted if `adj_edge` does not exceed
the paired baseline on *both* folds** — a single-fold win here is worthless given fold 1's 7.47 %
frozen-frame contamination (§5.5) and the 113k-vs-20k annotation imbalance.
**The real value of this experiment is not the checkpoint — it is that it prices "does a retrained
edge head help at all on our substrate" for one inference run, before we spend GPU on H1's own
retrain.** That is the cheapest possible probe of H2.

### Explicitly NOT recommended

- **Any frozen-CSV post-processing surgery.** CLOSED by `keremelik` with numbers, and by our own
  records.
- **ILP disappearance-cost retuning.** `c_d 1.5→1.4` measured −0.00078 on a dual-seed stack.
- **More TTA, more seeds, more blending.** Publicly measured to plateau at 0.910.
- **Porting `zhongbapapapa`'s MLPs.** Architecture decoded, feature contract not shipped (§4.2).
- **The division metric exploit.** Dead, score-negative, rescored.

---

## 9. What I could not access

| Item | Failing method | Note |
|---|---|---|
| `maulikgajera/biohub-cell-tracking-truncated-track-rescue-patch` (28 votes) | `kernels pull` → 403 | Listed but unpublished. Same pattern as liyansen on 08-18. Author at 0.915. |
| `chiranjithdharma/replace-midpoint-insertion-with-weighted-interpola` (21 votes) | `kernels pull` → 403 | Unpublished. Title alone suggests a P2-adjacent mechanism. |
| Author names on 30+ threads | `topic-messages --format json` returns `authorName: ""` | Structural API defect. Attribution for 735352 and 734192 recovered via `r.jina.ai`; that proxy then rate-limited. |
| Any discussion body via `WebFetch` | Returns `<title>` only | Kaggle is a client-rendered SPA. Do not retry. |
| Images inside forum posts | Rendered `[IMG]` | Several posts (730924's submission screenshot, 728324's score-drop screenshots, 724283's freeze-schedule table) carry data only in attachments. |
| `zhongbapapapa` / `leevvin` / `horaz0` / `ed2608` / `samuelx1a` training code | `kernels list --user` → Not found | All five publish datasets with no accompanying kernel. |
| `ed2608/biohub-cell-tracking-runtime-public` (678 MB), `rudispresence/…-hoct-code` (826 MB) | Not downloaded | Size vs expected value; HOCT is already closed out for us. |
| `horaz0/biohub-conv4d-joint-e20-artifact` | Not decoded | A "Conv4D" retrain lane; author at 0.914 (unconverted). Flagged for a future pass. |
| Number of selectable final submissions | Not exposed by the API | Still **UNVERIFIED**, carried over from 08-18. Kaggle default is 2. Confirm in a browser once. |

---

## Appendix A — kernels pulled and read, cross-referenced to the live board

41 kernels. `LB` is the *author's* leaderboard score on 2026-08-21, not a claim about the kernel.

| Kernel | LB | Subs | What it is |
|---|---:|---:|---|
| `hitoshisaito/biohub-kunal-0917-exact-repro` | 0.919 | 23 | **the 0.917 classical stack** (§2) |
| `xiaoleilian/biohub-m001-ens3-sm6-sim2` | 0.918 | 30 | its origin |
| `yusuketogashi/no-hack-…-3rd` | 0.918 | 164 | learned stack, self-declared 0.916 ceiling |
| `kunaldesale2408/biohub-cell-tracking` | 0.917 | **3** | 0.917 fork, 63 votes |
| `reyhanksatria/graph-patches-…-0-917-lb` | 0.917 | 10 | 0.917 fork, best-documented variant |
| `navazshfathi/best-score` | 0.917 | 21 | 0.917 fork |
| `wuwenmin/biohub-kunal-0917-exact-repro` | 0.917 | 9 | 0.917 fork |
| `dalloliogm/biohub-exp196-deepcenter-gap-confirmed` | 0.917 | 105 | learned stack + DeepCenter gap confirm + 3-frame accel lookahead |
| `dalloliogm/biohub-exp203-classical-three-model-ensemble` | 0.917 | — | classical branch, explicitly "not claimed to reproduce" |
| `dalloliogm/biohub-exp110-ilp-birth-death-cost` | 0.917 | — | ILP birth 0.0 / death 1.4, gap 5.8 µm density-adaptive, min-track 6 |
| `tangai1/biohub-clean-joint-recall-rescue-v1` | 0.917 | 40 | "Biohub 162" three-frame forward-acceleration lookahead |
| `lonnieqin/biohub-gap2-joint-node-budget` | **0.929** | 123 | **shared synthetic-node budget** (mechanism 7) |
| `hiranorm/new-lb-0-916-infer-ensemble-lf-exp002` | 0.923 | 60 | pre-registered paired CI study; **det 0.90 calibrated to preserve node count**; `division_fork_topk` 50→20 |
| `anhadmahajan06/biohub-track-your-cells-development` | 0.916 | 15 | single-child division repair, 328 vs 307 divisions |
| `backtracking/biohub-medal-v1 / -general-v2 / -v3` | 0.916 | 10 | env-knob variants of the learned stack |
| `saitejabandaruin/biohub-masterpiece-tracker-version-16` | 0.913 | 37 | dual-seed + retention guard, sealed 2-graph ledger |
| `prvsiyan/biohub-clean-public-frontier-lineage-tracker` | 0.913 | 4 | same lineage, 0.913 artifact with SHA evidence |
| `raunakdey07/biohub-harmonic-fusion-dual-seed-pipeline` | 0.915 | 2 | harmonic fusion w=0.20 + DeepCenter veto + density-adaptive gap |
| `sleepymegacat/the-metric-decides-your-architecture-8-measured` | 0.912 | 11 | **the validated metric reimplementation** (§5.1) |
| `tomasa2/biohub-what-worked-and-what-didnt-for-me` | 0.912 | 35 | **the eight-null ablation + error decomposition** (§6) |
| `keremelik/biohub-negative-results-…` | 0.912 | 11 | published negatives: C0 CLOSED, ILP c_d REJECT |
| `keremelik/biohub-metric-map-open-questions-clean-play` | 0.912 | 11 | open/closed ledger |
| `g1en114/biohub-loo-train-val44`, `-finetune-val44` | 0.915 | 12 | **public LOEO training script** (train=44b6/test=6bba and reverse), 50 ep, AdamW 1e-4, bs 16 |
| `yakizakana629/biohub-metric-aware-lineage-completion` | 0.909 | 9 | **July hub-exploit fossil** (`max_components`, `forks`); its "baseline 0.950" is pre-patch |
| `aaaa1597/s1-06-stardist-btrack-pipeline` | 0.648 | 6 | StarDist + btrack, only non-U-Net architecture in the corpus |
| `web3cainiao/biohub-ct-route2-divgap-2p` | 0.144 | 5 | classical DoG route, documents the 1:1-linker division blind spot |
| `deepakjnath/biohub-climb-notes` | 0.915 | — | method card, no new technique |
| plus: `salemali7`, `indarkarhana`, `amerhu`, `romanrozen`, `beicicc`, `ericwang03`, `mtoshidesu`, `dariushafshar` | 0.898–0.917 | | forks / knob sweeps / meta; no new mechanism |

## Appendix B — forum topics read in full

**Priority:** 735352 (TWEAK, §3), 735531, 734604, 734330 (Zebrahub sanctioned), 734237, 734192
(exact-split trick, §5.2), 734093, 734053, 733973, 733877, 733389, 732474 (GT is Ultrack, §5.4),
732345, 732103, 730924, 730160 (CV-direction trap, §5.3), 729082, 729057, 728551, 728324, 728300,
727154, 726924, 726521, 724917, 724582, 724386, 724283 (frozen frames, §5.5), 723921, 723655, 716952.

**Host posts, all verbatim in §5.6.** No host post since 2026-08-14. **No new topic created since
2026-08-16** — the forum has gone quiet while the board moved +0.006 at the top.

## Appendix C — commands

```
kaggle kernels list -s biohub --page-size 100 --sort-by {voteCount,dateRun,scoreDescending}
kaggle kernels pull <ref> -p <dir> -m
kaggle competitions topics list -c biohub-cell-tracking-during-development --page 1..4
kaggle competitions topic-messages biohub-cell-tracking-during-development <id> -n -1 --format json
kaggle competitions leaderboard biohub-cell-tracking-during-development -d
kaggle datasets list -s biohub --sort-by updated --page-size 40
kaggle datasets download -d leevvin/biohub-movie-heldout-edge-predictor-v1 --unzip
kaggle datasets download -d zhongbapapapa/biohub-physics-informed-models --unzip
kaggle datasets download -d dariushafshar/biohub-local-cv-pack --unzip
```

Local analysis (scratch, not committed): notebook→text converter, markdown extractor, the
`UNetNodeTransformer` strict-load probe, and the 13-feature MLP reconstruction + sensitivity sweep.
No repo file was modified other than this report. No kernel pushed, no submission, no commit.
