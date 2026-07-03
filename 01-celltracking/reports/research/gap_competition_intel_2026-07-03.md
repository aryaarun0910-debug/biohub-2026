# Gap / Competition Intel — Biohub Cell Tracking During Development

**Date:** 2026-07-03 · **Source of truth:** live Kaggle CLI (leaderboard pulled 2026-07-03T19:02Z), pulled public notebooks, organizer GitHub repo, image.sc/FEBS. **Author:** research agent.

Competition: `biohub-cell-tracking-during-development` (CZ Biohub, Royer Group). 3D+time light-sheet zebrafish. Deadline 29 Sep 2026. Notebook-only, internet OFF, ≤12h, T4×2. Metric = adjusted edge-Jaccard + 0.1·division-Jaccard, embryo-held-out.

---

## 0. HEADLINE — the number that changes everything

The "classical ceiling ~0.854" we believed in is **not** the ceiling. It is roughly the **90th percentile** of a 624-team leaderboard. The top of the board is 0.896, and **62 teams are already above 0.854.** The gap is not a tuning gap — it is an **architecture gap**: the top of the board runs the *organizer's learned baseline* (UNet detector + Transformer edge model), which we have not adopted. We are running classical DoG only.

---

## 1. LIVE LEADERBOARD (pulled 2026-07-03, 624 teams)

Top of board:

| Rank | Team | Score |
|---|---|---|
| 1 | Rahul Parmeshwar | **0.896** |
| 2 | doheon114 | 0.893 |
| 3 | Pathik Patel | 0.890 |
| 4 | Kaushik Ramayya Chikkala | 0.875 |
| 5 | Matt Goldfield | 0.874 |
| 6 | Kevin | 0.872 |
| 7 | Mendrika Ramarlina | 0.870 |
| … | (cluster) | 0.860–0.867 |

Distribution (public LB, 624 teams):

| Stat | Value |
|---|---|
| max | 0.896 |
| p99 | 0.869 |
| p95 | 0.856 |
| p90 | **0.854**  ← our supposed "ceiling" |
| p75 | 0.839 |
| p50 | 0.810 |
| teams > 0.88 | 3 |
| teams > 0.87 | 6 |
| teams > 0.86 | 17 |
| teams > 0.854 | **62** |
| teams > 0.84 | 140 |
| teams > 0.807 (our best) | 327 |

**Read:** our reproduced DoG at 0.807 sits at ~p52 (median). Public LB uses ~29% of test, so exact ranks will shift, but the *shape* is unambiguous: a learned method is required to be competitive, and it is freely available (see §4). Note our most recent submission (0.727, 2026-07-03) is a **regression** vs 0.807 — whatever changed in that run hurt.

Kaggle CLI:
```
.venv/Scripts/python.exe -m kaggle competitions leaderboard -c biohub-cell-tracking-during-development --download
```

---

## 2. THE WINNING STACK (organizer baseline, open source)

Everyone above ~0.83 is running the **organizer's own baseline**, published at:

- **Repo:** https://github.com/royerlab/kaggle-cell-tracking-competition — Python package `tracking_cellmot`, BSD-3.

Architecture:
1. **Detector:** `TemporalUNet3D` — 3D U-Net with temporal attention. Per-voxel features + single-channel detection map; centres via local-max suppression.
2. **Linker:** `SimpleNodeTransformer` — node features pooled at detected centres, fed to a cross-attention transformer that scores every (t, t+1) node pair.
3. **Training:** *sparse supervision* — only GT-annotated edges backprop; background/unannotated cells ignored.
4. **Output:** GEFF graphs via the `tracksdata` library (`td.graph.IndexedRXGraph.from_geff`), one `.geff` per test video → flattened to `submission.csv` (node rows `t,z,y,x`; edge rows `source_id,target_id`; a division = one source node with two outgoing edges).

**Critical:** the repo ships weights trained for only **3 epochs** (~LB 0.81) and explicitly says *"expect gains from training longer."* The community did exactly that — 50-epoch retrains hit **LB 0.856**, and the top 0.89 adds ILP + gap recovery + safe division on top.

---

## 3. PUBLIC NOTEBOOKS — ranked by signal (approach + role)

| Notebook (ref) | Votes | Approach | Role / score |
|---|---|---|---|
| `inversion/cell-tracking-getting-started-w-nearest-neighbor` | 161 | Official NN starter | baseline |
| `pilkwang/biohub-cell-tracking-data-model-eda-baseline` | 120 | **Canonical EDA** + rule-based DoG tracker (micron-space linking, robust centroids) | data/metric reference |
| `yusuketogashi/lb839-learned-graph-tracker-micro-safe-divisi` | 90 | **Learned graph** (50ep UNet+Transformer) + motion relinker + micro safe-division | ~LB 0.856+ |
| `seshurajup/lb-0-856-rule-based-v13` | 83 | Rule-based (built on isaka) tuned to **LB 0.856** | classical SOTA |
| `romanrozen/strong-start-dog-band-pass-lb-0-73` | 77 | DoG band-pass | LB 0.73 |
| `yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction` | 75 | **UNet+ILP reproduction** wrapper of thibaut's baseline (offline artifact route) | ~LB 0.81 |
| `xiaoleilian/biohub-cell-tracking-3d-u-net` (+ `-training`, `-ensemble`) | 47/10/2 | 3D U-Net train+infer, ensembling | learned |
| `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` | 57 | Learned detections + two-pass Hungarian (6µm/10µm gates) + 1-frame gap recovery + synthetic node insertion | ~LB 0.856 |
| `isakatsuyoshi/biohub-rule-based-baseline` | 56 | Rule-based baseline (parent of seshu V13) | LB 0.82–0.84 |
| `lucifer19/biohub-cell-lineage-tracker` | 43 | Lineage tracker | — |
| `thibautgoldsborough/unet-baseline-inference-submission` | 17 | **Official inference wrapper** for the learned baseline (offline, from artifacts dataset) | reference |
| `beicicc/*` (Kun Zhang, ~20 exp notebooks) | 1–14 | Systematic tuning: `det_threshold=0.995`, `division_weight=0.8`, gap distance sweeps (gap50/gap525/gap55) | ablation goldmine |

**Pattern of the 0.86→0.896 recipe** (assembled from pilkwang/yusuke math cells):
- Learned UNet detections, threshold `p > τ` (τ≈0.995) + local-max NMS.
- **Link in physical microns**, not voxels: anisotropic scale `(1.625·Δz, 0.40625·Δy, 0.40625·Δx)`; gates are ellipsoids in voxel space.
- Two-pass Hungarian/ILP: tight gate **6 µm**, relaxed **10 µm** for leftovers; cost from predicted next position `r̂ = r_t + 0.5·(r_t − r_{t−1})`.
- ILP objective `min Σ w_e x_e + c_a·appear + c_d·disappear + c_m·division`, `w_e = −edge_prob`.
- **1-frame gap recovery**: connect end@t to start@t+2 when `d_µm ≤ 2g` (g=6µm); reuse a near-midpoint isolated node or insert a synthetic node with intensity-weighted local centroid.
- **Capped "safe" division recovery**: add a second daughter only if parent has 1 child, both daughters close to parent and each other, daughter unused, capped per-frame & globally.

---

## 4. FREE WEIGHTS — we can skip training entirely

Public Kaggle datasets (attach as offline input; internet-OFF compatible):

| Dataset | Size | What |
|---|---|---|
| `pilkwang/biohub-tracking-support-pack-50ep-v1` | 349 MB | **50-epoch** UNet+Transformer + repo + offline wheels (the LB856 artifact) |
| `thibautgoldsborough/cellmot-baseline-artifacts` | — | Official baseline artifacts (repo+weights+wheels) used by thibaut's inference notebook |
| `addisonhoward/biohub-unet-weights` | 1.3 MB | UNet weights — **`addisonhoward` is the Kaggle competitions admin account** → effectively official mirror |
| `subinium/biohub-trackastra-public-weights-mirror` | 251 MB | Trackastra public weights mirror (alt learned linker) |
| `kms111201/biohub-cell-tracking-data` | 79 GB | Full training embryos+GT (for retraining longer) |

The whole offline route is standardized: notebooks search for a mounted dir containing `repo/`, `weights/`, `wheels/`, `pip install` from wheels only, `PYTHONPATH=src`, run `tracking_cellmot` prediction → `.geff` → CSV.

---

## 5. METRIC — exact definition & exploitable quirks

From `royerlab/kaggle-cell-tracking-competition/metrics.md`:

- **Node matching:** optimal bipartite assignment on centroid distance, **max 7 µm**; each pred node matches ≤1 GT node.
- **Edge TP:** both endpoints match GT nodes that are GT-connected.
- **Edge FP:** only when a pred edge *"steals"* a matched node — its source/target matches a GT node connected to a *different* node. **Edges between two unmatched (false/background) nodes are IGNORED — not FP.**
- **Adjusted edge Jaccard (over-prediction penalty):**
  `adjusted = max(0, jaccard · (1 − 0.1 · (T_pred − T_true)/T_true))`
  where `T_pred` = total predicted nodes, `T_true` = estimated true node count.
- **Division Jaccard:** GT division is TP when the prediction has a matching **fork** meeting all 4 criteria (one-node-stage coverage, both daughter lineages touched, single connected component, contains a predicted fork), tolerance **±1 timepoint**.
- **Final:** `score = adjusted_edge_jaccard + 0.1 · division_jaccard`, **micro-averaged** (sum TP/FP/FN across embryos before dividing).

**Exploitable quirks (things we may not be using):**
1. **Over-prediction is directly taxed** (a=0.1). Detecting 10% more nodes than truth multiplies your Jaccard by 0.99; 30% over → ×0.97. This is exactly the "6bba over-detection ratio 0.88→1.06" regression noted in our own commit log — the metric confirms count calibration, not continuity, is the lever.
2. **Unmatched↔unmatched edges are free.** Aggressively linking low-confidence detections does not create edge FPs; it only costs via the node-count tax. So recall-heavy linking is safe *if* node count stays calibrated.
3. **Node match is 7 µm.** Sub-voxel centroid refinement (intensity-weighted, background-subtracted) directly lifts match rate — cheap, precision-preserving. Anisotropy matters: z-voxel is 1.625 µm so a 7 µm gate is only ~4 z-planes.
4. **Divisions are cheap points.** Only 0.1 weight, but a capped safe-fork recovery adds TP forks with near-zero FP risk (±1 frame tolerance is generous).

---

## 6. DATA & RULES

- **License:** dataset is **CC0 / open** ("largest publicly available cell-tracking dataset by number of annotations"). Full train+GT is on Kaggle (79 GB, §4).
- **Format:** OME-Zarr `(T,Z,Y,X)`, voxel `1.625 / 0.40625 / 0.40625` µm (z/y/x). Test embryos held out. Test file example: `test/44b6_0113de3b.zarr/...` (chunked zarr, multi-channel `c/…`).
- **External data / pretrained models:** could not load the JS-gated rules page directly, but the *de facto* rule is settled in practice — the entire top of the LB attaches public Kaggle weight datasets as offline inputs, and organizer/admin accounts (`thibautgoldsborough`, `addisonhoward`) publish them. Attaching public Kaggle datasets is standard-allowed. **Action item:** confirm the exact wording on the rules page before relying on external ImageNet-pretrained backbones, but community-shared competition-derived weights are clearly in-bounds.
- **Announcements:** image.sc thread `forum.image.sc/t/…/121671` (WebFetch 403 — JS/anti-bot gated; open in browser). FEBS article: network.febs.org/posts/biohub-calls-on-ai-community-to-transform-3d-cell-tracking.

---

## 7. GAPS WE ARE MISSING (ranked)

1. **We have no learned detector.** The organizer's `tracking_cellmot` (UNet+Transformer) is the price of entry above p90. Our DoG tops out where the median sits. This is the whole game. → **Adopt the baseline.**
2. **We are training-blind but training is optional.** 50-epoch weights are public (`pilkwang/…-50ep-v1`, 349 MB) and attachable offline. We can be at ~0.856 by *inference only*, no GPU training budget spent.
3. **Our "0.854 classical ceiling" proxy is likely miscalibrated.** Real classical rule-based SOTA is **0.856** (seshu V13), and it is beaten by learned methods to 0.896. Our internal proxy may be scoring on a different held-out split than the real metric — we should compute the *real* metric locally using the open-source `tracksdata` matcher + `metrics.md`, not our proxy.
4. **Linking space.** If we link in voxel (not micron/anisotropic) space, our z-gates are wrong by 4× (z voxel 1.625 vs xy 0.40625 µm). Every top team uses micron ellipsoid gates.
5. **No sub-voxel centroid refinement.** 7 µm match window + intensity-weighted centroid = free node-match recall. pilkwang's whole EDA baseline is *just* this on top of DoG.
6. **No gap recovery / safe division.** 1-frame gap closing (synthetic node insertion) and capped safe-fork recovery are the 0.856→0.89 deltas.
7. **We may be over-detecting.** The metric's a=0.1 node-count tax + our own 6bba finding say: calibrate `T_pred ≈ T_true`. Kun Zhang's `beicicc` notebooks show `det_threshold=0.995` is where the community tuned it.

---

## 8. RECOMMENDED ACTIONS (in order)

1. **Today: run the free learned baseline.** Fork `thibautgoldsborough/unet-baseline-inference-submission` or `yusuketogashi/lb839-…`, attach `pilkwang/biohub-tracking-support-pack-50ep-v1`, run offline, submit. Expect ~0.85. This immediately jumps us from p52 to ~p88 and gives a real learned starting point.
2. **Stand up the real metric locally.** Clone `royerlab/kaggle-cell-tracking-competition`, use `tracksdata` GEFF matching + `metrics.md` to score on our held-out embryos. Retire the 0.854 proxy. Verify the 0.727 regression cause.
3. **Retrain longer.** Attach `kms111201/biohub-cell-tracking-data` (79 GB) or use the 50ep pack; push epochs past 50 within the 12h T4×2 budget. The baseline explicitly leaves gains on the table here.
4. **Port the 0.86→0.89 post-processing to our pipeline** regardless of detector: micron anisotropic linking, two-pass Hungarian (6/10 µm), 1-frame gap recovery with synthetic nodes + intensity centroid, capped safe-division. These stack on *any* detector, including our DoG.
5. **Calibrate node count** to `T_true` (tune `det_threshold`/NMS) to avoid the a=0.1 tax — the single change most consistent with our own regression logs.
6. **Confirm external-data wording** on the rules page (browser) before adding non-competition pretrained backbones.

### Key URLs
- Repo: https://github.com/royerlab/kaggle-cell-tracking-competition
- Metric: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
- Comp: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development (rules: `/rules`)
- Weights: kaggle datasets `pilkwang/biohub-tracking-support-pack-50ep-v1`, `thibautgoldsborough/cellmot-baseline-artifacts`, `addisonhoward/biohub-unet-weights`, `subinium/biohub-trackastra-public-weights-mirror`, `kms111201/biohub-cell-tracking-data`
- Notebooks: `thibautgoldsborough/unet-baseline-inference-submission`, `yusuketogashi/lb839-learned-graph-tracker-micro-safe-divisi`, `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery`, `seshurajup/lb-0-856-rule-based-v13`, `pilkwang/biohub-cell-tracking-data-model-eda-baseline`
- Ecosystem: https://github.com/royerlab/ultrack · https://royerlab.github.io/ultrack/ · FEBS: https://network.febs.org/posts/biohub-calls-on-ai-community-to-transform-3d-cell-tracking
