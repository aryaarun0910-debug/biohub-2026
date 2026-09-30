# Competitive refresh — 2026-08-17

Follow-up to `competitive_frontier_2026-08-16.md`. Live via authenticated Kaggle CLI
(leaderboard CSV `2026-08-17T07:48`, discussion topics API JSON, pulled notebook sources,
dataset/model search). Evidence in session scratchpad. Window covered: ~08-15 → 08-17
(≈36 h of new activity; the prior scan's cutoff was 08-16).

Bottom line up front: **the field barely moved in 11 days.** No public technique has broken
the plateau. The public-reproducible ceiling nudged from ~0.916 to **0.918** (liyansen) via a
motion/division tweak, not a new mechanism. The only genuinely new *intel* asset is a rigorous
measured-scorer notebook (`sleepymegacat`) that hands us 3–4 cheap, portable detector-side levers
we are likely NOT fully exploiting. External Zebrahub crops have now materialised as a Kaggle npz
dataset (the H1 enabler), but no one has yet posted a score from training on them.

---

## 1. Leaderboard delta (2441 teams now vs 2426 on 08-16; $60k; notebook-only)

| Band | 08-16 count | 08-17 count | Δ |
|---|---|---|---|
| ≥ 0.950 | 1 | 1 | – |
| ≥ 0.948 (top-3 boundary) | 3 | 3 | – |
| ≥ 0.945 | 7 | 7 | – |
| ≥ 0.940 | 10 | 11 | +1 |
| ≥ 0.935 | 16 | 16 | – |
| ≥ 0.930 (top ~1.1%) | 27 | 28 | +1 |
| ≥ 0.920 | 73 | 74 | +1 |
| ≥ 0.915 | 447 | 462 | +15 |
| 0.915 exact (the plateau) | ~302 | 300 | ≈flat |

**Read:** distribution is essentially frozen. The plateau is still ~300 teams stacked on 0.915.
We are rank **378** (score 0.915), i.e. still inside the plateau block.

**Top-20 movement (new/changed since 08-16):**
- **yuto083 — 0.947, rank 4** (submitted 2026-08-17 01:05). NEW near-top entrant; not on the 08-16 top list. 45 subs.
- **enddl22 — 0.945, rank 7** (2026-08-17 00:02). NEW near-top; 109 subs (heavy prober).
- Unchanged leaders: Mark Cooper 0.950, TWEAK 0.949, Soheil Ayati 0.948, z7777 0.945, Matt Goldfield 0.945.
- Tang now 0.943 (rank 9, still climbing, active 08-17). mikelou1 0.935.

The 0.945–0.950 tier is getting *more crowded* (two fresh arrivals in 24 h), consistent with
"several teams now hold a retrained/generalising edge model." No movement at the very top (0.950 leader static since 08-16).

---

## 2. New / active discussion threads since ~08-15

Only three threads are new or freshly-commented in-window. All three **reinforce the prior
scan's thesis** (post-proc exhausted → retrain / external data / trustworthy CV). No new trick disclosed.

- **#735352 "Possible big leaderboard shakeup"** (mikelou1, 08-15). Argues division-Jaccard has
  so few samples (~151–304 events) that 0.001 top-of-board gaps are overfit → private shakeup likely.
  - **TWEAK (rank 2) reply — CLAIM, not measured:** "we are working on a universal plugin the bio-cell
    team can plug in with minimal changes… tested with every unique public notebook/model, gains of
    **0.030 / 0.040 / 0.050 instantly** just attaching our plugin… a single public model reached
    **0.940 untuned**." This is an AI-agent company advertising; unreproducible; treat as marketing.
  - **Tang:** "Build a reliable CV and trust it. Use external/synthetic data to make the model more
    robust on **division** cells." (= our H1/H3.)
  - **Bharath Varma:** "training past ep400 has diminishing returns… maybe wait for the private shake."
- **#734604 "What is the best model for this domain so far?"** (Moawiz, 08-12; fresh 08-16 comments).
  - **Mendrika Ramarlina (high-signal):** decompose before switching models. `Edge recall ≈ node_recall²
    × conditional-linking-accuracy` → **detector misses are quadratically expensive.** Measure 4 things
    separately (detection recall & count; conditional linking where both GT endpoints detected; oracle
    detection ceiling; division TP/FP/FN). "Pilkwang UNet/node-transformer is still a very strong backbone…
    focus on **detector calibration/precision and division ranking** before a new linker."
  - **Moawiz (repeats known gap):** "detector finds real cells but produces an enormous junk candidate
    pool; linker chooses wrong." **Tang:** "retrain instead of public ckpt… ckpt has hit a wall, hard to
    get more from post-proc alone."
- **#735531 "so to get a score above the public baseline is training?"** (Bharath Varma, 08-16).
  Community consensus in-thread: yes, post-proc is tapped out; wait for GM guidance. No technique.

---

## 3. New public notebooks since ~08-15 (with measured LB scores)

Author→score cross-referenced against the live LB CSV. **Nothing public beats 0.918.**

| Notebook (slug) | Author LB score / rank | What it is | New vs our levers? |
|---|---|---|---|
| `liyansen/biohub-v16-ranker-persistent-divisions` & `…-v13-ranker-recall-finetune` | **0.918 / rank 79** | Public "no-hack 159B" stack (Togashi fork) + **3-frame forward-acceleration lookahead** + **persistent-division gate** (mother-history + persistent-diverging-daughters) + ranker **recall fine-tune**. Current public-reproducible ceiling. | Lookahead & division-gate = partially new (see §5). |
| `anhadmahajan06/biohub-track-your-cells-development` | 0.916 / rank 113 (56 votes) | Same P3-harmonic stack exposed as env-var knobs: reverse-harmonic 0.20, velocity relink 0.65, **boundary-track rescue** (t=0/t=Tmax), sister-div radius 9.0 µm, 2-seed consensus. | Mostly our closed levers. Boundary-track rescue = minor. |
| `dalloliogm/biohub-exp196-deepcenter-gap-confirmed` | 0.915 (exp196 pending) / rank 175 | Public stack + **DeepCenter center-prior UNet3D used only to CONFIRM marginal 1-frame gaps** + bounded 3-frame forward-accel lookahead. Well-documented, submission-shaped. | DeepCenter-as-gate = moderately new (§5). |
| `sleepymegacat/the-metric-decides-your-architecture-8-measured` | **not submitted (pure EDA)** — 7 votes | **The most valuable new asset.** 8 measured facts probed against the organisers' own scorer + an 80-line numpy/scipy scorer reimpl (84 perturbation cases, 0 mismatches vs official). See §5. | Yes — several portable detector levers. |
| `saitejabandaruin/biohub-masterpiece-tracker` | 0.913 / rank 565 (19 votes) | Popular but sub-plateau. Votes ≠ score. | No. |
| `yakizakana629/biohub-metric-aware-lineage-completion` | 0.890 / rank 1231 | Below plateau. | No. |

Note: high vote counts (anhad 56, saiteja 19) are popularity of *baseline reproductions*, not score —
several sit at or below the plateau. The only public author *above* 0.916 is **liyansen at 0.918**.

---

## 4. New datasets / model artifacts

- **`kkunizaw/biohub-zmnscrops` (3.66 GB, uploaded 2026-08-16 19:18)** — two npz files
  `zmns001_crops.npz` (5.76 GB) + `zmns002_crops.npz` (7.13 GB). "zmns" = **Zebrahub external
  crops packaged for Kaggle** — this is the H1 external-training-data enabler *materialising as a
  ready-to-attach dataset*, six days after the host opened Zebrahub (#734330). Uploader kkunizaw sits
  at only 0.887, so **no strong score has yet been posted from training on it** — the lever is unclaimed.
- **`ideaplatsteven/biohub-celltrack-pipeline` (326 MB, 08-16)** — the royerlab `tracking_cellmot`
  baseline **source tree** packaged (predict_unet_transformer.py, division_metrics.py, csv_to_geffs.py,
  augmentations.py…). Host baseline code, not new science; convenient for offline retrain harnesses.
- **DeepCenter center-prior UNet3D** (`pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, 2096 dls;
  epoch-400 snapshots by beicicc/wasbornlazy) — *not new* (July), but newly **integrated into the
  0.915–0.916 public post-proc as an independent gap-confirmation signal** (dalloliogm, anhad, tangai).
- Individual retrained 3D-UNet weights continue to appear (sdeogade/biohub-3dunet, emreakpnr/unet3d-biohub,
  ouedraogosomkieta/biohub-unet-weights, jirkaborovec Trackastra artifact) = ongoing retrain activity, no shared scores.

No new synthetic dataset beyond Freitas 18.5 GB (#732103, already known).

---

## 5. Cheap wins we should try (portable; not on our closed-lever list)

Our closed levers: scalar re-acceptance, component selector, node-budget pruning, split/merge,
edge-TTA, HOCT drop-in, CoTracker. The following are NEW and portable.

**CW1 — Sub-voxel centroid refinement of detector peaks. [HIGHEST EV, cheapest]**
`sleepymegacat` fact 7 (measured against the scorer): centroid error has a **cliff at σ≈2 µm** — well
below the 7 µm match radius — because matching is one-to-one and nearest neighbours are only ~9–10 µm
apart, so a slightly-off centroid steals a neighbour's match (costs 1 FP + 1 FN at once). σ=2.5 µm → −16 %;
3 µm → −41 %; 4 µm → −74 %. Below 1.5 µm it is free.
- *Expected benefit:* directly proportional to how far our current peak centroids sit above ~1.5–2 µm.
  Potentially several points if our peaks are integer-voxel or coarse argmax (z voxel = 1.625 µm, so
  integer-z rounding alone can put us near the cliff).
- *How to test (offline, no GPU):* take our current detections, add parabolic/intensity-weighted sub-voxel
  refinement around each peak, re-score with LOEO on the patched scorer. If our peaks are already sub-voxel,
  measure the residual and stop. One afternoon; no retrain.

**CW2 — Duplicate-detection (NMS) audit. [cheap correctness check]**
Fact 6: a *single* duplicate detection on one cell **halves** that cell's Jaccard (1.000→0.500) because
only one twin wins the one-to-one match and the loser's edges are graded FP. "NMS quality outranks everything."
- *Expected benefit:* bounded but potentially large in dense regions if our NMS radius admits near-duplicates
  (~0.4 µm apart already halves). Distinct from our count-based node-budget pruning.
- *How to test:* histogram nearest-neighbour distance among our emitted nodes per frame; count pairs below
  ~2 µm; tighten NMS min-distance and re-score LOEO. Cheap.

**CW3 — Verify gap repair inserts an interpolated NODE, not a dt=2 skip edge. [free correctness]**
Facts 4+5: the scorer **silently drops every dt≠1 edge** (not even an FP) — so a "gap edge" bridging a
missed frame is *worthless*. Full credit is only recovered by **interpolating a node** at the missing frame
and emitting two dt=1 edges (median step 1.8 µm vs 7 µm radius → linear interpolation lands inside easily).
liyansen's 0.918 stack does insert an in-volume intermediate node; we must confirm *ours* does too.
- *Expected benefit:* zero if we already interpolate nodes; a real free gain if our gap repair currently
  draws skip edges. Pure audit of `filter_output_graph`-equivalent in our stack.

**CW4 — 3-frame forward-acceleration lookahead in association. [the convergent public trick]**
Three independent 0.915→0.918 notebooks (liyansen v13/v16, dalloliogm exp196) added the *same* mechanism:
a bounded continuation bonus that rewards a candidate link only when the displacement is followed by a
**physically-gated next-frame displacement with low acceleration residual** (smooth t→t+1→t+2). This is a
longer-temporal-context extension of one-frame motion relink (which we already do).
- *Expected benefit:* the measured public delta from 0.915/0.916 → 0.918 (~+0.002–0.003). Modest, portable,
  shakeup-robust (it is a geometry prior, not a hidden-set tune).
- *How to test:* add the acceleration-residual gate to our motion-relink cost and re-score LOEO both directions.

**CW5 — Under-predict node count for the free bonus. [tune last]**
Fact 8: the node-count penalty **has no upper clip** — *under*-predicting nodes earns a bonus up to **1.1×**
(part of why scores exceed 1.0). Our node-budget pruning already trims; the new insight is the *direction* and
the ceiling — a slight deliberate under-shoot vs `estimated_number_of_nodes` may be strictly beneficial.
- *Expected benefit:* small (mild linear term); tune last. *Test:* sweep node budget below the estimate on LOEO.

**Tooling win — adopt `sleepymegacat`'s 80-line numpy/scipy scorer** for offline LOEO CV: it reproduces the
official TP/FP/FN (84 cases, 0 mismatches) *without* needing `tracksdata`+torch (which won't install in the
offline rerun). Also confirms our CV design: **leave-one-embryo-out is the only honest split** — the 199
"samples" are overlapping crops of just 2 movies (6,206 of 8,128 `6bba` file-pairs share annotated cells), so
per-video CV leaks at the cell level. This validates our existing LOEO-strict harness (`loeo_f0/f1_strict.json`).

---

## 6. Cross-check against our knowledge (what is NOT new)

Confirmed already-closed / already-known, do **not** re-flag: 2-seed detector blend, edge-TTA (community
also finds it can hurt), ILP appearance/disappearance/division costs, harmonic bidirectional fusion, motion
relink (one-frame), short-track pruning, line-fit smoothing, DeepCenter model existence, Zebrahub legality
(#734330), Freitas synthetic (#732103), HOCT/CoTracker as drop-ins (underperform). The retrain-on-external
thesis (H1/H2/H3) is unchanged and still the top-tier's likely edge — but still *unclaimed in public* as of 08-17.
