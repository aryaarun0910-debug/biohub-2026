# Kaggle Live Competitive Intelligence — Biohub Cell Tracking During Development

**Date:** 2026-07-12
**Analyst:** competitive-intelligence research agent
**Scope:** public kernels (pulled + read), discussion/forum, related past competitions, external-data licenses.
**Method note:** Kaggle discussion and rules pages render via JS/behind auth and returned only page titles to WebFetch; primary evidence below therefore leans on (a) the kernels' own written commentary and configs (pulled via Kaggle CLI and read in full), (b) the organizer's public metric repo `royerlab/kaggle-cell-tracking-competition`, and (c) the image.sc/FEBS announcements. Kernel refs are given as `owner/slug` and are directly reproducible with `kaggle kernels pull`.

---

## Executive summary — 5 highest-value items

1. **The entire top of the public board is ONE pipeline, forked ~1000×.** Every high-vote notebook (pilkwang, yusuketogashi/LB897, yaroslav V4, beicicc/exp0xx, abhijith, udit 0.893) is the *same* `TemporalUNet3D` center detector + node cross-attention edge transformer + ILP graph selection + deterministic graph-repair stack. They differ only by **single-axis tweaks to ~40 `BIOHUB_*` env vars**. The public artifact ships **50-epoch weights** (`biohub-tracking-support-pack-50ep-v1`); the stronger forks patch in a **400-epoch "DeepCenter" model + 8-way D4 detection TTA**. Our 400-epoch model at 0.889 is already at/above the public frontier's *architecture*; our edge is training depth + private data discipline, not novelty. Ref: `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery`, `pilkwang/biohub-cell-tracking-blend-preprocessings`.

2. **A converged "public recipe" now dominates and is worth A/B-matching exactly.** Across the strongest forks the settled knobs are: `DET_THRESHOLD=0.97` (down from 0.985), `GAP_CLOSE_MAX_GAP=2`, `OUTPUT_MIN_TRACK_LEN=6` (7 being probed), `KEEP_DIVISION_COMPONENTS=1`, `GAP2_RECOVERY=0`, `MOTION_RELINK_LEARNED_BONUS=0.75–0.90`, tight safe-division caps (`SAFE_DIV_MAX_UM≈4.66`, `SISTER≈7.05`, global frac cap ≈0.0037), plus **D4 detection TTA** and an **edge-policy veto at τ=0.35**. Refs: `beicicc/biohub-exp056-division-prior09`, `yaroslavkholmirzayev/high-upside-min7-short-track-filter-risk-a-b`, `yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction`.

3. **Short-track pruning is the current highest-ROI "free" lever and it directly games the count penalty.** Filtering tracks shorter than 6–7 frames (while *keeping* division components) removes low-confidence isolated detections that add FP edges AND inflate `N_pred`, improving both the raw Jaccard and the `(1 − 0.1·(N_pred−N_est)/N_est)` multiplier at once. This is the single most-copied recent change. Ref: `yaroslavkholmirzayev/high-upside-min7-short-track-filter-risk-a-b`, `tamerlanomralinov/biohub-high-upside-min7-short-track-filter`.

4. **Divisions are deliberately being *suppressed*, not grown, by the pack — and that's rational given the metric.** Division is weighted only 0.1, and the organizer stack over-predicts it, so forks lower `ILP_DIVISION_WEIGHT 1.0→0.9` and gate "safe divisions" behind very tight geometry (parent–daughter ≤4.5–5.0µm, sister ≤6.8–7.5µm, global frac cap ~0.003). Net division ROI is small; do **not** over-invest here. Ref: `beicicc/biohub-exp056-division-prior09`.

5. **The 0.968 outlier is most consistent with public-split overfit, not leakage.** The public forks openly reference a **"0.9700 anchor"** they calibrate every graph parameter to (`pilkwang/biohub-cell-tracking-blend-preprocessings`), meaning ~0.97 is reachable by threshold/cap sweeping on the *visible* 29% public split. With a hidden disjoint 71% private set and positive-only sparse labels, a lone 0.968 above a 0.90 pack is best explained as **calibration to the public split (and possibly the 4 visible movies)**, which should regress privately. Treat it as a mirage; do not chase its exact caps.

---

## Public-code teardown

All scores below are **public LB** and are the authors' own claims (in titles/markdown). "Backbone" = pilkwang `TemporalUNet3D` + node-transformer + ILP + graph-repair unless noted.

| Kernel (owner/slug) | Votes | Claimed | Technique / what's distinctive | What to steal | What to avoid |
|---|---|---|---|---|---|
| `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` | 157 | ~0.89 | **Canonical backbone.** WBCE detection (α=0.01 neg weight), softmax-over-parents edge loss (allows 1→2 daughters), ILP (`edge=−1.0, appear=0.1, disappear=0.1, div=1.0`), motion relink (λv=0.52, β=0.78, tight 6.2/relaxed 10.4µm), 1-frame gap close (≤5.75µm), gap2 (≤9.2µm total, ≤3.9µm step), safe-div, line-fit smoothing (w=0.76). `DET_THRESHOLD=0.985`. Voxel µm = (1.625, 0.40625, 0.40625). | The full repair taxonomy + exact µm gates as a checklist to diff against ours. Softmax-over-parents edge normalization. | 0.985 threshold is now stale (pack moved to 0.97). |
| `pilkwang/biohub-cell-tracking-blend-preprocessings` | 117 | highest pilkwang | **400-epoch `DeepCenterUNet3D` used as an add-only "Center gate"** to confirm marginal *observed* gap nodes (`g_obs=1[8≤d≤12µm]`, needs Center conf ≥0.20); **D4 spatial detection TTA** patched into `predict_unet_transformer.py`; calibrated to a **"0.9700 anchor"**, `DET=0.97, MIN_TRACK_LEN=6, GAP_CLOSE_MAX_GAP=2`. | The 400ep center model as a *second-opinion gate on repairs only* (never deletes backbone). D4 detection TTA. | The "0.9700 anchor" framing — it is a public-split target, not a private one. |
| `yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction` | 123 | ~0.89 | **Edge-policy veto:** a tiny logistic on features {P(u→v), d, Δz/Δy/Δx, n_t, n_t+1, local density ρ7(u/v), degrees} removes a learned edge only if `p_ψ(e)<τ_veto=0.35`, capped at 1% of edges. Never creates edges. | Cheap independent edge re-ranker as a veto-only cleanup; density features ρ7 (neighbors within 7µm). | Veto fraction cap >1% risks deleting true edges. |
| `yaroslavkholmirzayev/high-upside-min7-short-track-filter-risk-a-b` | 20 | ~0.897+ | **Short-track filter min-len 6→7** + `MOTION_RELINK_LEARNED_BONUS=0.90` (vs 0.75). Pure post-processing probe. | Short-track pruning sweep {4,5,6,7} with division-component protection. | Aggressive min-len can prune true short lineages on the smallest embryo (yusuke restores min6 there). |
| `yusuketogashi/lb897-baseline` | 147 | **0.897** | The **canonical fork base** ("LB897") the board rallied around. Per-dataset min-track rule (min7 global, min6 on smallest dataset). Otherwise backbone. | Per-dataset (not global) filter thresholds — treat each embryo's node budget separately. | Nothing egregious; it's a careful tune. |
| `beicicc/biohub-exp056-division-prior09` | 12 | ~0.90 | Single-axis: `ILP_DIVISION_WEIGHT 1.0→0.9` on the 400ep+D4 graph; DeepCenter veto thresholds `gap=0.06, safe_div=0.08` (disabled in this run). | The exact division-prior probe and DeepCenter veto thresholds. | Marginal; div weight in metric is only 0.1. |
| `abhijithneilabraham/solution` | 81 | ~0.89 | **300-epoch backbone + DeepCenter add-only gate** (rejects new gap nodes/daughters, never deletes). `DET=0.97, GAP=2, MIN_TRACK=6`. | "Add-only" gating philosophy: repairs must pass a second model; backbone is sacrosanct. | — |
| `uditjain13/0-893-lb-best-score-full-code-explained` | 13 | **0.893** | Best *explanatory* writeup of the backbone; confirms "most of the score comes from the repair layer, not the net." | Use as the plain-English map of the repair layer. | Nothing new algorithmically. |
| `jirkaborovec/biohub-celltrack-dog-trackastra-graph-trans` | 16 | mid | **Only real alternative architecture:** multi-scale **DoG** blob detection (3 σ scales, union peaks) + intensity-weighted COM refine + **physical-space NMS (4µm kd-tree)** → paint ball masks → **`Trackastra.from_pretrained("ctc").track(mode="greedy")`**. Isotropic resample first. | Trackastra "ctc" pretrained as an *independent linker* for ensembling/diversity; DoG+COM+physical-NMS detector as a cheap detection cross-check. | Trackastra ctc greedy alone underperforms the learned ILP stack; needs division handling bolted on. |
| `romanrozen/strong-start-dog-band-pass` | 78 | 0.73+ | DoG band-pass starter. | Reference DoG params only. | Below frontier. |
| `seshurajup/lb-0-857-rule-based-v14` / `isakatsuyoshi/biohub-rule-based-baseline` | 92/67 | 0.826→0.857 | Pure classical: gaussian/maximum-filter detection + Hungarian linking, no learned net. Contains a **clean local re-implementation of the scoring** (edge+div Jaccard). | The standalone metric code for offline CV. | As a tracker it caps ~0.857. |
| `amanatar/biohub-0-855-ema-intensity-cost-tracking` | 16 | 0.855 | Classical but adds **EMA velocity smoothing (α=0.35)** + **multi-feature Hungarian cost (distance + intensity similarity, weight 0.12)**; 4-scale DoG. Also drives backbone with `DET=0.992` and loosened gap caps. | **Intensity-similarity term in the linking cost** — an appearance feature the learned edge model may under-use. EMA velocity is already in our motion relink. | Classical detector ceiling. |
| `inversion/cell-tracking-getting-started-w-nearest-neighbor` | 253 | low | Official getting-started; nearest-neighbor linker. | Baseline sanity only. | — |

**Structural read:** the board has *no* architectural diversity at the top — it is a monoculture of the pilkwang stack. That is both a risk (everyone shakes together) and an opportunity (any genuinely independent second model gives ensemble diversity nobody else has).

---

## Discussion intelligence

Note: Kaggle discussion pages did not scrape (JS/auth; WebFetch returned title-only, and site-scoped WebSearch surfaced only kernels/landing pages). The items below are sourced from kernel commentary (primary author statements), the organizer's metric repo, and the official announcements.

- **Competition framing / provenance.** Hosted by CZ Biohub (Royer Group), built on **Ultrack** (Nature Methods 2025). Data are real zebrafish light-sheet (DaXi) movies; billed as "the largest publicly available cell-tracking dataset by annotation count," released **CC0**. Sources: [image.sc announcement](https://forum.image.sc/t/biohub-cell-tracking-during-development-kaggle-competition/121671), [FEBS Network](https://network.febs.org/posts/biohub-calls-on-ai-community-to-transform-3d-cell-tracking), [Kaggle competition page](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).
- **Exact metric (organizer repo).** `adjusted_jaccard = max(0, J·(1 − 0.1·(T_pred − T_true)/T_true))`; **T_true is a *coarse* estimate of node count**; node matching by optimal bipartite assignment up to **7µm**; division TP needs pre-split matched node + both daughter lineages matched + single connected component + a predicted dividing node, with **±1 timepoint tolerance**; `score = adjusted_edge_jaccard + 0.1·division_jaccard`; **micro-averaged across videos**. Source: [royerlab/kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md).
  - Implication that the pack has internalized: because T_true is *coarse* and the penalty is linear in `(N_pred−N_est)/N_est`, trimming spurious nodes (short-track filter) is nearly free upside.
- **"0.9700 anchor" is openly discussed in-code.** `pilkwang/biohub-cell-tracking-blend-preprocessings` states every graph parameter "matches the 0.9700 anchor." This is the community's shared calibration target on the public split — strong evidence the ~0.97 region is a *public-LB* construct reachable by post-processing sweeps.
- **Convergent belief across authors:** "most of the score comes from the repair layer, not the network" (`uditjain13/...`, `pilkwang/...`). The community treats detection as solved and pours effort into graph repair / linking heuristics — consistent with our own read that linking is the headroom.
- **External pretrained models are being used in public without objection:** Trackastra `from_pretrained("ctc")` (`jirkaborovec/...`) and references to Ultrack/CTC data. No visible organizer takedown or rule flag observed in kernels. (Rules page itself did not scrape — verify the external-data clause directly before relying on this.)
- **No credible leakage claim was found in scraped material.** Absence of evidence, not evidence of absence — the forum did not render. Flagged as an open question below.

---

## Past-competition winning tricks (transfer analysis)

| Competition | Winning trick | Source | Transfers? Why |
|---|---|---|---|
| **Cell Tracking Challenge (ISBI) — KIT-GE / KIT-Sch-GE** | U-Net predicting **neighbor/cell distance transforms** (not binary masks) → cleaner separation of touching cells; **min-cost-flow graph linking with motion estimation** to re-link across missed frames. | [arXiv 2004.01486](https://arxiv.org/abs/2004.01486), [PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243219), [KIT-GE repo](https://github.com/TimScherr/KIT-GE-3-Cell-Segmentation-for-CTC) | **Partial.** Distance-transform detection head could sharpen centers in dense regions (our detection ceiling ~0.885). Motion-aware gap re-linking is already in our stack. Distance-head worth a probe. |
| **CTC 2024 (7th) — Trackastra** | **Transformer that links pre-segmented cells** via learned pairwise association; pretrained "ctc"/"general_2d" generalize out-of-the-box; handles divisions. | [Trackastra repo (BSD-3)](https://github.com/weigertlab/trackastra), [TrackMate-Trackastra](https://imagej.net/plugins/trackmate/trackers/trackmate-trackastra) | **Yes, as ensemble diversity.** It's an *independent* learned linker → different error modes than our node-transformer. Use its edge probabilities as a second opinion / veto, or as a diversity member. Legit (BSD-3, pretrained on public CTC). |
| **Ultrack (Nature Methods 2025)** | **Candidate-hypothesis selection:** generate many candidate segmentations across params, pick a temporally consistent subset via ILP; robust under segmentation uncertainty; strong on dense 3D embryos. | [Nature Methods](https://www.nature.com/articles/s41592-025-02778-0), [Ultrack repo](https://github.com/royerlab/ultrack) | **Yes, conceptually.** Multi-hypothesis detection (several thresholds/TTA variants) fed to ILP is exactly the untapped lever vs the pack's single-threshold detection. This is the organizer's own method — strong prior it works on this data. |
| **Sartorius Cell Instance Segmentation (2021)** | Pretrain detector on large external same-domain corpus (**LIVECell**) then fine-tune; strong instance models (YOLOX / Swin-UperNet); heavy TTA + ensembling. | [Kaggle comp](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation), [tascj solution](https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution) | **Partial.** "Pretrain on external same-domain data then fine-tune" maps to pretraining detection on **CTC/Zebrahub** before fine-tuning on the 2 labelled families — a data lever the pack hasn't pulled. TTA/ensemble already standard here. |
| **HuBMAP (2022–23)** | Large-encoder U-Nets, **multi-fold ensembling**, tiled inference at multiple scales, **heavy augmentation for cross-donor generalization**, TTA. | [3rd place](https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution), [hippocampus-garden writeup](https://hippocampus-garden.com/kaggle_hubmap_hpa/) | **High-value transfer for the hidden-embryo problem.** The private set is *disjoint embryos* = a domain-shift/generalization problem exactly like cross-donor HuBMAP. Heavy augmentation + multi-fold ensembling to generalize to unseen embryos is directly on point and under-exploited by the monoculture. |

---

## The 0.968 outlier — best-supported explanation

**Best-supported reading: public-split overfit / calibration mirage, not leakage and not (only) a bigger model.**

Evidence:
- The public forks *name* a **"0.9700 anchor"** they tune every graph parameter toward (`pilkwang/biohub-cell-tracking-blend-preprocessings`). ~0.97 is demonstrably reachable on the *visible* split by post-processing sweeps alone — no secret model required.
- The scoring rewards trimming nodes (coarse T_true, linear count penalty) and the visible surface is only the **29% public** split with **positive-only sparse labels**. That is an ideal setup to over-fit thresholds/caps to the public rows and print a number that won't hold on the **71% hidden disjoint** private set.
- The pack sits ~0.90 using the *same* backbone; a single 0.968 is far likelier a tuning artifact than a genuine +0.07 method gap that nobody else can reproduce.

**Less likely but possible:** a legitimate 400ep(+)+full-D4-TTA + independent second linker (Trackastra/Ultrack) ensemble. If real, it would still be an *ensemble/data* edge, not a new algorithm.

**Implication for us:** Do **not** chase the outlier's exact caps or treat 0.968 as the target. Optimize for **private-set robustness**: trust local min-fold CV over the 2 labelled families, keep node-count discipline, and prefer changes that help *generalization to unseen embryos* (augmentation, ensembling, multi-hypothesis detection) over public-split micro-tuning. Expect a shakeup that punishes public-split fitters.

---

## New actionable ideas we are NOT already doing

Each: idea → first experiment → cheap kill-gate.

1. **Multi-hypothesis (multi-threshold) detection → ILP selection (the Ultrack lever).**
   The whole pack runs a *single* `DET_THRESHOLD`. Instead, generate candidate nodes at several thresholds (e.g. 0.95/0.97/0.99) + D4 TTA, feed the *union* as ILP candidates and let the solver + count penalty pick the temporally consistent subset.
   - First experiment: on the 2 labelled families, run detection at 3 thresholds, union candidates, re-solve ILP, measure min-fold edge Jaccard vs single-threshold.
   - Kill-gate: if min-fold adj-edge-Jaccard doesn't beat single-threshold by ≥+0.003, drop it (added nodes will otherwise trip the count penalty).

2. **Trackastra "ctc" as an independent second linker for ensemble/veto diversity.**
   Everyone uses one linker (our node-transformer). Add Trackastra edge probabilities as a second opinion: agreement → keep, disagreement → down-weight, à la the τ=0.35 edge-policy veto but with a genuinely *independent* model.
   - First experiment: run Trackastra `from_pretrained("ctc")` over our detections on both families; compute edge-level agreement with our transformer; measure Jaccard of intersection vs union vs ours-alone.
   - Kill-gate: if intersection/agreement-weighting doesn't raise min-fold Jaccard, ship it only if it at least reduces private-set variance; else drop.

3. **Pretrain the detector on CTC + Zebrahub before fine-tuning on the 2 families (the Sartorius/LIVECell lever).**
   Only 2 labelled embryo families is the core constraint; the private set is disjoint embryos. External same-domain pretraining is the classic fix and nobody public has done it.
   - First experiment: pretrain `TemporalUNet3D` detection head on CTC 3D + Zebrahub dense trajectories, fine-tune on 44b6/6bba, compare min-fold node ceiling vs current ~0.885.
   - Kill-gate: if the min-fold node ceiling doesn't move, external pretrain isn't helping detection; stop.

4. **Intensity/appearance term in the linking cost (from `amanatar`).**
   Add a nucleus-intensity-similarity feature to edge scoring / motion-relink cost (weight ~0.1). Our learned edge model may under-use appearance; classical trackers found it cheap and useful.
   - First experiment: add intensity-similarity to the motion-relink Hungarian cost on ambiguous close ties; measure Jaccard on high-density frames.
   - Kill-gate: no improvement on dense-region edges → drop.

5. **Short-track filter sweep with per-dataset budgets (from yusuke/yaroslav).**
   If we're not already sweeping `MIN_TRACK_LEN` *per embryo* with division-component protection, do it — it's the pack's highest-ROI free lever and directly improves the count multiplier.
   - First experiment: sweep min-len {4,5,6,7} per family, keep division components, measure adj-edge-Jaccard incl. the count term.
   - Kill-gate: if best per-family setting ≤ our current global setting, keep global.

6. **Distance-transform / neighbor-distance detection head (KIT-GE) for dense regions.**
   Add a second detection head predicting cell-center distance transform to sharpen separation of touching nuclei where detection currently plateaus.
   - First experiment: add the aux head, evaluate detection recall/precision in the densest frames only.
   - Kill-gate: no gain in dense-frame detection F1 → not worth the training cost.

---

## Contradictions / de-risking of our current plan

- **We are NOT behind on architecture.** The public frontier is our architecture with fewer epochs. Our 400ep=0.889 is competitive; the reframe (top ~0.896, meta baseline) holds. **Don't** rebuild the pipeline — the marginal wins are in linking, multi-hypothesis detection, ensembling and generalization, exactly where we already suspected.
- **De-risks the "divisions are the headroom" temptation.** Division is weighted 0.1 and the pack is actively *suppressing* over-prediction with tight caps. Our note that "divisions are badly over-predicted by the organizer stack" is corroborated — the fix is conservative caps + lower ILP div prior, not a division model. Low ceiling; cap the investment.
- **Public-LB is an actively-gamed 29% surface with a named 0.97 anchor.** This strongly validates the team's discipline of never tuning on the 4 visible movies and trusting min-fold CV. **Do not** import the outlier's caps. Expect the private board to reward generalization, not public-split fitting.
- **Confirms detection is near-solved / linking is the lever** (multiple authors state it explicitly). Aligns with current strategy; reallocate effort from detection micro-tuning to linking + ensemble diversity.
- **One genuine risk to our plan:** the board monoculture means *our* pipeline shares failure modes with everyone. A truly independent second model (Trackastra/Ultrack ensemble member) is the cheapest way to buy uncorrelated error before the shakeup — worth prioritizing over another threshold sweep.

---

## Open questions

1. **External-data rule text.** The rules page did not scrape. Confirm directly on Kaggle whether Trackastra/Ultrack pretrained weights and CTC/Zebrahub data are explicitly permitted (public kernels use them un-challenged, but verify the clause before building a submission on them).
2. **What actually is the 0.968 submission?** Is it a single team, and does its author post any writeup? The forum didn't render — check the discussion tab manually for any claim/refutation.
3. **Private/public split mechanics.** Confirm the 29/71 split and whether the 4 visible movies overlap the public split (affects how much public LB signal to trust at all).
4. **T_true "coarse estimate" definition.** The metric repo says T_true is a *coarse* node-count estimate — how coarse, and per-video or global? This determines exactly how aggressive short-track pruning can be before the count term flips sign.
5. **Does Trackastra have a usable 3D pretrained model, or only 2D/ctc?** jirka uses "ctc"; confirm whether a 3D checkpoint exists or whether we'd link slice-wise / on our own detections.
6. **Zebrahub imaging + March-22 dense trajectories license/URL** for legitimate pretraining — competition data is CC0, but confirm the auxiliary Zebrahub imaging release terms and the exact download path before relying on it.

---

### Source list
- Kaggle competition & metric: [competition](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) · [royerlab metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- Announcements: [image.sc](https://forum.image.sc/t/biohub-cell-tracking-during-development-kaggle-competition/121671) · [FEBS Network](https://network.febs.org/posts/biohub-calls-on-ai-community-to-transform-3d-cell-tracking)
- Pulled kernels (owner/slug): pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery · pilkwang/biohub-cell-tracking-blend-preprocessings · yusuketogashi/lb897-baseline · yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction · yaroslavkholmirzayev/high-upside-min7-short-track-filter-risk-a-b · beicicc/biohub-exp056-division-prior09 · abhijithneilabraham/solution · uditjain13/0-893-lb-best-score-full-code-explained · jirkaborovec/biohub-celltrack-dog-trackastra-graph-trans · seshurajup/lb-0-857-rule-based-v14 · amanatar/biohub-0-855-ema-intensity-cost-tracking · inversion/cell-tracking-getting-started-w-nearest-neighbor
- Related methods: [Ultrack (Nature Methods 2025)](https://www.nature.com/articles/s41592-025-02778-0) · [Ultrack repo](https://github.com/royerlab/ultrack) · [Trackastra (BSD-3)](https://github.com/weigertlab/trackastra) · [KIT-GE](https://github.com/TimScherr/KIT-GE-3-Cell-Segmentation-for-CTC) · [KIT-GE paper](https://arxiv.org/abs/2004.01486) · [Sartorius](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation) · [tascj Sartorius](https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution) · [HuBMAP 3rd](https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution) · [Cell Tracking Challenge](https://celltrackingchallenge.net/) · [Zebrahub](https://zebrahub.sf.czbiohub.org/)
