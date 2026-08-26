---
id: 00-system/handoff
title: Handoff
area: 00-system
status: active
updated: '2026-08-26'
owner: biohub
links: []
tags: [handoff, entry-point]
record_kind: state
---

# Current handoff

**Status:** ACTIVE 2026-08-26. Branch `master`.
This file is the live entry point. Full direction: [directional-updates.md](../01-research-direction/directional-updates.md).
System map: [README.md](../README.md); architecture: [system-design.md](system-design.md); contract: [CLAUDE.md](../../CLAUDE.md).

> ## 🔴 2026-08-26 (late) — LEVER-0021 IS KILLED, the deployed DeepCenter is dead weight, and Colab is back
>
> **The priority-1 lane closed in one CPU session, both folds, both mechanisms, zero slots** (`PKT-0017`).
> Sub-voxel refinement around the detection moved the honest-fold median residual **−1.3%** against a −10%
> bar and passed only on the fold where DeepCenter is in-sample — and even there the score was flat, because
> the *mean* residual never moved (`FACT-0314`). The pre-registered successor, snapping each detection to
> the nearest heatmap peak, was **worse in sign** on both folds and lost −0.015 / −0.011 through the scorer
> (`FACT-0316`). Root cause, measured: **the prior does not localise** — its peaks sit a median 2.6 µm from
> the annotated centre and its centroid recovers ~20% of an imposed offset (`FACT-0317`). The 0.75 µm
> "self-localisation" I recorded mid-session was the window returning its own centre; it is superseded.
>
> **What this does NOT kill:** `FACT-0270`. Mislinks are still geometrically wrong endpoints. What died is
> the idea that the pipeline's own centre prior can fix them. `FACT-0272`'s dose-response is refuted as a
> median-based model — the score follows the mean and the >3 µm tail. A successor needs a **different
> localiser** and a falsifier written on those two statistics, then a scorer round-trip; do not reopen on
> DeepCenter.
>
> **Three substrate facts worth more than the lever:**
> 1. **The deployed DeepCenter checkpoint is collapsed** (`FACT-0311`): validation loss rose 7× from epoch 2
>    to the deployed epoch 500, and on unseen embryos its heatmap is ~0 — so the deployed gap veto
>    **rejects 100% of checked gap points** on the visible movies (`FACT-0312`). `best.pt` (epoch 2) accepts
>    ~18%. The P10 A/B that would have measured this was built and never submitted. Whether accepting gap
>    repairs helps the LB is a one-submission question; it is not a lever until someone writes its falsifier.
> 2. **DeepCenter was trained on all of 44b6 and validated on all of 6bba** (`FACT-0310`) — any fold-0 number
>    that touches it is in-sample.
> 3. **Fold 1's −0.67-voxel z offset belongs to the LOEO detector** (`FACT-0313`, `FACT-0317`), not to the
>    annotation — one more reason fold 1 is a crippled configuration (`FACT-0261`), not an embryo.
>
> **Compute changed (host, 2026-08-26): Colab Pro is available again** (`FACT-0318`, ~200 compute units).
> This reverses the 2026-08-24 "Kaggle only" framing below. Nothing runs there automatically (rule 8). The
> first use that pays: the zero-submission GPU work queued behind Kaggle's two slots (`FACT-0061`) —
> calibration stages 1–2 and the fold-1 LOEO configuration repair.
>
> **The Colab lane is BUILT (same evening): `scripts/core/colab_factory.py` + `colab_relay.py`, worker at
> `notebooks/colab_worker/biohub-colab-worker.ipynb`, private relay repo `aryaarun0910-debug/biohub-colab-relay`
> (clone at `C:/temp/colab_relay`).** Host decisions recorded 2026-08-26: Drive job-queue worker (not a remote
> shell — Colab's terms disallow those), GitHub relay for status/metrics, **per-session unit budget** as the
> rule-8 mechanism. Flow: host adds Colab Secrets (`GH_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`) once; host
> states a budget → `colab_factory.py session --units N --hours H --gpu L4 --scope smoke,notebook
> --host-approved "<their words>"`; `smoke` first (~0.1 units), then `queue --spec <kaggle spec>`; host opens
> the worker on the named GPU and presses Run all; `status` / `fetch` from here. The worker refuses any job
> outside the recorded budget and exits after 30 idle minutes. Built notebooks run unmodified because the
> worker recreates `/kaggle/input/<slug>` via `kagglehub`; the competition tree (~79 GB) is re-downloaded per
> runtime (Drive quota unknown, so it is not cached there). Unit rates are UNVERIFIED defaults rounded up
> (`FACT-0318`) — the host should pass `--rate L4=<panel value>` once read off the resources panel.
> **Smoke PASSED 2026-08-26 21:43 UTC on an L4 for ~0.011 units (`FACT-0319`).** Next in the same
> 10-unit session: `p21_colab_parity_f0` - the EXP-0022 champion control arm on the first 20 fold-0
> crops (`BIOHUB_LOEO_LIMIT=20`, sweep `-1` only), to check crop-for-crop that Colab reproduces Kaggle
> before any Colab number is trusted, and to measure L4 minutes per crop for unit planning. A full
> 71-crop fold took 2.71 h on Kaggle T4x2 (`_evidence/exports/loeo_f0_strict/kernel.log`), so a full
> fold needs a bigger session than 10 units. Known accounting gap: the worker's worst-case estimate
> covers the notebook's `max_hours` only, not the ~79 GB competition download that precedes it.
> **Four pilot attempts failed on Colab platform facts, none on science (`FACT-0320`):** read-only
> `/kaggle/input` (fixed: input-root rewrite), Kaggle's API rejecting username+key (fixed: `KAGGLE_API_TOKEN`
> = the laptop's `KGAT_` access token), and Python 3.13 vs the pack's cp312 wheels (fixed: uv-built 3.12
> kernel). The competition tree downloads in <14 min on Colab. Worker at commit `4837783`; attempt 5 pending.
>
> **Next actions, re-ranked:** (1) read `EXP-0021` / `EXP-0022` when they land — check the −1 control
> reproduces the champion before anything else; (2) repair the fold-1 LOEO configuration (`FACT-0261`);
> (3) calibration stages 1–2, now schedulable on Colab; (4) the two free division findings (`FACT-0295`);
> (5) `LEVER-0019` at 4.5, env-gated. The localisation *mechanism* stays on the board without an owner or
> an instrument.

> ## 🔴 READ THIS FIRST — 2026-08-26. The campaign's central belief was leak-manufactured.
>
> **`EXP-0019` evaluated LOEO fold 1 (6bba) with weights trained on 6bba.** Its spec set no
> `BIOHUB_LOEO_WEIGHTS_GLOB`, so it fell back to the pack's `split_0`. All nine other fold-1
> LOEO specs set the override. Worth **+0.203 of score**, and it manufactured **25 of our 26
> division true positives** (`FACT-0160`). `tests/test_loeo_weights_hygiene.py` now makes this
> a build failure.
>
> **The honest numbers** (`FACT-0210`, official scorer, `_evidence/exports/loeo_f1_strict/`):
> fold 1 score **0.7026**, adj_edge **0.7025**, node recall **0.8547**, divisions **TP 1 / FP 491
> / FN 124**, reach **66/125**. Fold 0 (clean): score 0.9002, adj_edge 0.8986.
>
> ### 🔵 CYCLE-2 REVERSAL — our deployed division term is NOT broken (`FACT-0260`)
>
> `EXP-0015` emits no forks, so `divJ` is **zero by construction** and its **LB 0.906 is a direct
> leaderboard reading of our `adj_edge` term**. Therefore our deployed decomposition is:
>
> ```
> adj_edge ≈ 0.906     divJ ≈ 0.16 – 0.19     total 0.925
> ```
>
> **I had this backwards.** `FACT-0101` offered reading (a) `adj_edge 0.924 / divJ 0.01` and
> reading (b) `adj_edge 0.906 / divJ 0.19`. I chose (a). **(b) is right, by ~6×** — dropping all
> forks offline costs only −0.00075 `adj_edge`, so the edge channel can explain at most ~16% of
> the −0.019 LB drop.
>
> **Our LOEO substrate understates the deployed division term by ~11× (fold 0) and ~112× (fold
> 1).** Every "our divisions are catastrophically broken" conclusion in the banner below was
> measured on a substrate that cannot see them — including "we fork on 0 of 106 GT division
> parents", which was measured on fold 1.
>
> **And fold 1 is a crippled CONFIGURATION, not the 6bba embryo** (`FACT-0261`). Its node-recall
> collapse (0.8547, min 0.112) belongs to the LOEO ablations — pack primary only, secondary and
> DeepCenter OFF. The *same crops* reach 0.994–0.999 under the deployed chain. Annotation density
> and cell density were both tested and killed as explanations. **Do not quote fold-1 numbers as
> performance.**
>
> Also: no pooling reconciles offline with the LB (required fold-0 weight is 1.1257, outside
> [0,1]; best single fold is still 0.025 short, `FACT-0262`), and offline **levels** carry MDE
> ±0.026/±0.041 — larger than the whole rank-148-to-rank-1 spread. **Paired deltas** are ±0.0036
> / ±0.0006 and are the only usable currency (`FACT-0263`).
>
> ### 🟢 THE EDGE MECHANISM, AND THE ~~NEW PRIORITY 1~~ (`LEVER-0021` — **KILLED the same evening, see the block above**)
>
> **The linker is correct. The geometry is wrong.** Mislinked cells barely moved — GT
> displacement median **2.33 / 2.44 µm** — yet our two detections sit **7.68 / 7.93 µm** apart.
> Endpoint residual is **3.35 / 3.68 µm on mislinks** against **1.68 / 2.30 µm on TPs**, and a
> *neighbouring* cell sits 1.82 µm from the child, exactly the right displacement. The assignment
> picks the geometrically better answer and is wrong because the coordinates are (`FACT-0270`).
>
> **Reducing endpoint residual 25% is worth +0.014…+0.026 (f0) and +0.021…+0.037 (f1)** —
> 2–6× the best measured division gain, and the only lever this cycle that moves **both folds
> the same way** (`FACT-0272`).
>
> **And it is completely unexploited.** Every coordinate in both honest exports is **100%
> integer — including the export labelled "subvoxel"**. Sub-voxel localisation has never reached
> a submission. Meanwhile **DEEPCENTER**, a trained UNet3D centre-prior heatmap, is already in the
> deployed pipeline but used *only as a boolean veto* for synthetic gap points — it never moves a
> primary detection (`FACT-0273`).
>
> **Cheapest test: no retraining, no Kaggle, CPU-hours.** Run the DeepCenter heatmap over the
> existing export, replace integer `(z,y,x)` with local sub-voxel centres, re-score both folds.
> Two-stage falsifier pre-registered: kill it if the residual does not drop ≥10%; kill it if a
> ~25% residual drop does not lift the score ≥+0.010 on **both** folds.
>
> **Killed alongside it:** widening the relink gate (−0.036 @9µm, −0.19 @12µm on f0),
> dedup/merge before assignment (**duplicate-stealing REFUTED** — 0 provable steals of 6,047
> mislinks; the competitor is a *neighbouring cell* at ~8.4 µm ≈ the cell spacing, and we
> UNDER-predict nodes so there is nothing to dedup), and both-sides-free repair (+0.0002/+0.0009 —
> the assignment is saturated). Also: a pure-distance LAP at gate 6 reproduces the deployed
> linker to within 0.001, so **the learned edge model contributes almost nothing to linking**
> (`FACT-0271`, `FACT-0274`, `FACT-0275`).
>
> ### Three things every future session must know — READ THE REVERSAL ABOVE FIRST
>
> 1. **WE NEVER PROPOSE A DIVISION.** Our fork generator's *maximum ever* sibling separation is
>    **8.97 µm**; the 25th percentile of real divisions is **9.19 µm**. We place a fork on **0 of
>    106** detected GT division-parent nodes (`FACT-0180`, `FACT-0200`). All 5,426 forks are in
>    the wrong places. **Every lever that selects, ranks, filters or vetoes among current forks
>    is capped near zero** — deleting all 491 false divisions is worth **+0.00064** (`FACT-0211`).
> 2. **THE PROBLEM CLASS IS WRONG.** Daughter antipodality separates at **AUC 0.785**, but it is
>    a *pairwise* interaction and the relink cost is a *sum over edges* (`FACT-0182`).
>    Pre-registered prediction: b-matching, min-cost flow and degree-constrained subgraph all
>    remain linear per-edge and will fail identically. The correct class scores
>    **(mother, daughter-pair) triples**.
> 3. **EDGES ARE THE CAMPAIGN, ON BOTH FOLDS.** Perfect edges **+0.2975** vs perfect divisions
>    **+0.0342** (`FACT-0213`). And the long-standing "all headroom is in fold-1 NODES" was a
>    **unit error** — `FACT-0031`'s +0.136 is an edge-*recall* delta, not a score delta. In score
>    units the LINKING oracle beats the node oracle on **both** folds (`FACT-0231`).
>
> ### ✅ AND ONE MEASURED GAIN — the only thing in this banner that is good news
>
> **DEFLATED 2026-08-26 (`FACT-0300`) — it is real but ~5× smaller than first measured.**
> Fold 1 end-to-end: **+0.0003**. Vintage-matched fold 0: **+0.0036** (the original +0.0062 came
> from a *different notebook vintage*). **Pooled across 199 crops — the objective the contract
> requires — +0.00085 at T=4.5, at most +0.00111 at 5.0.** Positive on both embryo directions,
> free at inference, no GPU. **Never quote it as +0.0062 again.** `LEVER-0019`, now priority 3.
>
> **Deploy at 4.5, not the pooled optimum 5.0**: fold 0's lowest TP fork sits at dxy 5.45, so
> 4.5 leaves 0.95 µm of margin against 0.45 µm. Two priced risks: the margin is the same size as
> an unmeasured **coordinate distortion of up to 0.74 µm** (exports are linefit-*smoothed*; the
> gate would see pre-smoothing coords, `FACT-0301`), and it must be a **post-emission removal** —
> a proposal-time reject frees the cap slot and creates new forks at higher dxy, and the
> `safe_division` tag decides which edge drops, with the fold-1 **sign** depending on that choice
> (`FACT-0303`). Also: the AUC 0.994 rests on **four** positives, and it **anti-composes with any
> ceiling raise** — dxy falls to ~0.51 once long-separation candidates enter (`FACT-0292`).
>
> **Why it works, CORRECTED (`FACT-0290`, `FACT-0302`):** the binding gate is
> `SAFE_DIV_MAX_UM = 4.66` — the **PARENT** distance — not the sister gate. The deployed sister
> ceiling is **8.5**, not the 7.2 code default I first recorded. The linker takes the *nearer*
> daughter, so the leftover orphan is systematically the *farther* one at median parent distance
> **9.21 µm against a 4.66 gate** — which is why raising sister alone to 22 µm admits **zero**
> true divisions on both folds. And the frame caps **do** bind (`safe_division_skipped_cap=137`),
> contrary to what I recorded.
>
> **Before promoting:** rescore the clean FOLD-1 export with the same rule (killed mid-run at
> session close) and report both embryo directions, per the contract.
>
> ### Open, and genuinely important
>
> - ~~**Two scorers exist**~~ **RESOLVED 2026-08-26 (`FACT-0250`).** All **199/199** GT geffs carry
>   `estimated_number_of_nodes`, which ONLY `summarise()` consumes — `evaluate_datasets()` never
>   reads it and takes graph objects, not a submission. The node term is LIVE. Strong evidence,
>   not proof (we cannot read Kaggle's scorer), but the campaign should treat it as real.
> - **0 of 7 LB-scored experiments have an offline score** (`FACT-0184`). The offline apparatus
>   has never been calibrated against the leaderboard, so no LOEO number can be compared to a
>   competitor's LB number — the error that produced the retracted `FACT-0141`.
> - **`LEVER-0012` is REOPENED** (`FACT-0221`): the p19 penalty sweep was scored entirely on the
>   leaked substrate, where our precision already exceeded break-even. On honest folds we
>   over-accept by **3.1x (f0)** and **16.7x (f1)**. Falsifier pre-registered: re-run the sweep on
>   the honest export; score should rise monotonically until accepted-fork precision hits ~3.4%.
> - **The one division action the numbers justify** is sequencing, not modelling: removing the
>   dead generator drops the precision bar for any future proposer **5x, from 0.384% to 0.078%**
>   (`FACT-0212`) — and the best measured geometric gate already achieves 0.090%.
>
> ### Registry gap — FIXED 2026-08-26, with its remaining holes named
>
> `facts.yaml` and `levers.yaml` are now **schema v2**. Validity is a SECOND axis, orthogonal to
> provenance, because the leaked facts were `MEASURED` and *deserved* it — they were correctly
> computed **from an invalid run**, and no single axis can say that:
>
> ```
> provenance   how strong is the derivation?            VERIFIED .. UNVERIFIED
> validity     was the run it derives from legitimate?  VALID | SUSPECT | INVALID | UNKNOWN
> ```
>
> Four new rules in `validate_registry.py`. **R8 is the retraction mechanism**: void an
> experiment once and every fact naming it must declare itself. R9 forbids closing or
> supporting a lever on an INVALID fact. R10 reports adoption gaps as notes. Backfilled:
> 5 INVALID, 3 SUSPECT against `EXP-0019`. 20 tests, each planting a violation and asserting
> reject-then-recover.
>
> **R9 immediately caught four real defects in committed data**, including `LEVER-0012` being
> `status: killed` AND `reopened: true` simultaneously — a self-contradiction no gate could see.
>
> **What it still does NOT do, stated plainly:**
> 1. **It does not DETECT leaks** — R8 fires only after a human sets `status: void`. This is a
>    propagation mechanism, not a sensor. Detection is `test_loeo_weights_hygiene.py`'s job and
>    only for that one defect.
> 2. **`derived_from` does not propagate transitively** — R8 walks one hop.
> 3. **17 experiment-derived facts still name no experiment**, so a void cannot reach them.
> 4. **`protocol.weights` is unverified free text** — nothing cross-checks it against the run's
>    `loeo_manifest.json`, so a *wrong* protocol string is as invisible as a missing one.
>    25 held-out-fold facts carry none.
> 5. **Absence of `validity` means VALID** for 84 facts — "no defect on record", not "audited clean".
> 6. **A SUSPECT fact can still close a lever** — `LEVER-0015` does, and the caveat lives only in
>    gate stdout. That is a science call, not an infrastructure one.


> ## 📌 READ `../../AGENTS.md` FIRST — the record system changed on 2026-08-25
>
> **Numbers now live in `registry/`, and prose cites ids instead of restating values.**
> Measured that day: the superseded score 0.915 appeared **244 times across 36 files** while
> the live score appeared 31 times across 7 — an agent reading a random doc was **8x more
> likely to hit a dead number than a live one**. That, not agent coordination, was the main
> engine of second-guessing.
>
> - `registry/facts.yaml` — every value once, with provenance AND validity. As of 2026-08-26:
>   92 facts, 4 UNVERIFIED, **5 INVALID and 3 SUSPECT** (all from the `EXP-0019` leak).
>   `FACT-0030`..`FACT-0033` are now PINNED to an instrument and VINDICATED — but their unit is
>   an edge-RECALL delta, not a score delta (`FACT-0230`).
> - `registry/levers.yaml` — every hypothesis and its status. A lever may only be closed by
>   evidence **about itself**.
> - `registry/packets/` — one lever, one owner. `validate_registry.py` fails on a double claim.
> - `record_kind:` frontmatter — `state` docs must be current; `ledger`/`archive` numbers are
>   frozen by design.
>
> `NEXT_SESSION_PROMPT.md` is deleted; it duplicated this file and had already gone wrong once.

## 🎯 LIVE 2026-08-25 (evening) — 0.925, rank 207/2,693. THE "ONLY RETRAINING WINS" THESIS IS REFUTED.

**Deployed: P9 coupled division transplant, LB 0.925** (submission 55753516, +0.010 over the superseded 0.915, FACT-0002) — the
largest gain of the campaign and the first division-class change ever to score positive. Prediction
band (central 0.919-0.925) HIT. Committed at `51365e7`.

**Public frontier 0.927 and SATURATED** — 202 teams >=0.926. Top-3 = **0.953**, leader 0.962, gap
**+0.028**. Only **6 teams** at >=0.947.

---

### ⚠️ STRATEGIC REVISION — "the jump to 0.950+ is solely heavy retraining" NO LONGER STANDS

That was the operating thesis from 2026-08-23. **Four independent pieces of evidence now contradict
it, and one of them is our own leaderboard result:**

1. **We gained +0.010 by POST-PROCESSING** (P9), from the superseded 0.915 (FACT-0002) to 0.925, with no retraining at all. The
   "wrapping is exhausted" claim was false — it was exhausted for *our* ideas, not for the field's.
2. **mikelou1 retrained from scratch and reached only 0.928** (forum, VERIFIED). Retraining alone
   demonstrably does NOT reach 0.947+.
3. **TWEAK is at 0.953 claiming a model-agnostic plugin with NO new weights** — "gains ranging from
   0.030, 0.040, to 0.050 instantly just attaching our plugin ... a single public model reach a score
   of 0.940 untuned" (discussion/735352, VERIFIED). That is a post-processing/linker claim AT the
   top-3 boundary.
4. **Soheil Ayati (rank 2, 0.959) points at NODE SELECTION, not model capacity** — "many 'linking'
   issues actually originated earlier during node selection" (discussion/737101, VERIFIED).

**REVISED THESIS: the gap is ARCHITECTURAL, not capacity.** It sits in candidate generation, node
selection, and the association/solver structure — none of which necessarily require a bigger or
better-trained model. Tang (0.946) still says "retrain instead of using the public ckpt", so
retraining is likely *necessary* for the last stretch; but it is demonstrably **not sufficient**, and
it is no longer the first thing to spend on.

**Our own strongest evidence for the architectural read:** the deployed pipeline's division Jaccard
ceiling is **structurally ZERO** (below), and fixing that needs no retraining whatsoever.

---

### THE SIX MEASURED FACTS THAT NOW DRIVE EVERYTHING

1. **THE BIJECTION FORFEITS THE ENTIRE DIVISION TERM.** ~~Both association stages~~ **CORRECTED
   2026-08-25 (`FACT-0093`, VERIFIED): the two `linear_sum_assignment` calls are
   `motion_relink_edges` (cell 6:487) and `close_single_frame_gaps` (cell 6:697) — POST-PROCESSES
   we fully control, NOT the primary association, which is the ILP in the prediction subprocess.
   Calling them "association stages" made this sound like a model problem. It is not.**
   Measured directly: of the 24 divisions present pre-ILP, **14 are fork-collapsed by the relink**
   (`FACT-0090`) with every endpoint still detected and one daughter edge already correct.
   These are bare `linear_sum_assignment` calls. Out-degree <= 1 by construction, so **a division
   is not merely rare, it is INEXPRESSIBLE**. Measured: divJ ceiling **0.0000 at out-degree<=1 vs
   0.2320 at out-degree<=2**, i.e. **+0.0232 of SCORE is structurally unreachable** — for the price
   of 29 edges out of 81,055. Independently confirmed in public by Luka Duvanov (discussion/733877,
   "40 FN out of 40"). **Our standing "+0.0012 perfect solver" oracle was computed UNDER that
   constraint and never bounded this.**
2. **BUT naive relaxation is HARMFUL.** Blanket out-degree<=2 scored **-0.0081** end-to-end
   (6 TP bought at 2,160 FP plus -0.0084 adjacency). exp041-style gating gave **+0.0018**, and
   entirely through ADJACENCY — zero divisions recovered. **The oracle ceiling needs a SELECTOR, not
   a relaxation.**
3. **~~THE METRIC PAYS FOR RECALL~~ — REFUTED 2026-08-25 BY MEASUREMENT (`FACT-0102`).**
   Lowering the threshold scored **0.94 -> 0.924** and **0.90 -> 0.922** against 0.96875 -> 0.925:
   monotone DOWN. All four supporting lines below were wrong about the SIGN. The optimum is at or
   slightly ABOVE the deployed threshold. Kept for provenance — the reasoning was sound and the
   conclusion was still false, which is why arms get scored. The line is
   **`1 - 0.1 x over-prediction`**, and **`edge recall ~ node_recall^2 x conditional linking
   accuracy`** — recall enters SQUARED, over-prediction is taxed 0.1x linearly
   (discussion/733877, /734604, VERIFIED). Our deployed threshold 0.96875 is inherited and never
   selected. Caveat: duplicates cost ~9% for nothing, so any union of detectors needs a merge pass.
4. **THE NOMINATOR IS A PER-TARGET ARGMAX.** In-degree is **1 for all 2,162,040** candidate edges;
   13.42% of targets get ZERO candidates. **70.6% of the 17,001 missing GT edges are a RANKING error**
   (a wrong parent already nominated), only 29.4% thresholding. kNN-3 recovers 92.5% of misses and
   retains 99.4% of today's; **divisions need k=5 (93.2%) where continuations need k=3 (92.7%)**.
5. **WE COULD SEE ONLY 9.5% OF THE PUBLIC CORPUS.** Of 719 public kernels, **68 have a score in the
   title** — and all three mechanism-bearing notebooks we found have NO number in their names. Score-
   keyed searching was structurally blind to 90.5% of the field.
6. **ZEBRAHUB IS EXPLICITLY HOST-CLEARED** — "Yes you are free to use the data and all resources in
   Zebrahub ... **There is no overlap with the test set**" (Thibgolds, discussion/734330, VERIFIED).
   The 1.26M GT association edges are legitimate for the term that separates the frontier.

---

### ⏳ IN FLIGHT (do NOT poll scoring; it can take ~7 h)

| ref / kernel | what it settles | state |
|---|---|---|
| 55768476 **p15 fork-free** | pure **adj_edge** by subtraction | ✅ **0.906** — below both bands; confounded, see below |
| 55768483 **p16 det 0.90** | first LB reading on **adding** nodes | ✅ **0.922** (−0.003) — band HIT |
| **p17 det 0.94** (55769398) | pairs with p16 for SLOPE | ✅ **0.924** (−0.001) — band HIT |

#### ✅ 2026-08-25 evening — ALL THREE PROBES SCORED. The detection lane is REFUTED.

**Prediction record: 2 of 3 bands hit.** Four points now bracket the detection optimum:

```
0.90 -> 0.922 | 0.94 -> 0.924 | 0.96875 -> 0.925 (deployed) | 0.999 -> 0.883 (p6)
```

**Monotone decreasing as the threshold drops.** The pre-registration said both arms down
*"retires the whole recall-push thesis"* — both went down. Its four supporting lines (the
`1 − 0.1x` charge, recall-squared, the 17%-cut datapoint, our S1 sweep) were wrong about the
SIGN. Lowering is CLOSED (`FACT-0102`). The live remnant is whether a **slightly higher**
threshold pays — the detpeak superset answers that offline for zero slots, since it is exported
at T=0.5 and every higher threshold is a subset.

**p15 does NOT pin divJ — do not quote the subtraction.** It gives divJ 0.19, contradicting this
session's direct 1-of-125 measurement (`FACT-0080`) by **24x**. The probe lost ~984
true-positive edges beyond its 406 divisions, enough to explain the whole drop. Preferred
reading: **adj_edge ≈ 0.924, divJ ≈ 0.01** — our edge term is near top-3 level and the deficit
IS divisions — but that is INFERRED (`FACT-0101`), not measured. A clean read needs a fork-free
probe that does not trip the short-track filter.

**Net: the lane is divisions, and `LEVER-0002` owns it.**
| **p4 detpeak f0** | ❌ **RETURNED NOTHING** — empty `detpeaks/`, see below | COMPLETE, VOID |
| **p4 detpeak f1** | Same defect; still yields a current-substrate fold-1 LOEO export | RUNNING (left to finish) |

**p15 caveat:** it lost 578 edges / 230 nodes, not just its 406 divisions, because the short-track
filter reacts to the changed components. `divJ = (0.925 - probe)/0.1` is therefore APPROXIMATE and
biased to read divJ high.

#### ❌ 2026-08-25 — the p4 detection curve was NOT bought; the export exported nothing

Fold 0 ran 9,439 s and returned `detpeaks/` as an **empty directory** for all 71 crops. The log has
the patch banner but **neither** the per-crop success line **nor** its own `no peaks buffered`
warning — the flush never ran, and nothing raised.

**Cause (verified at `file:line`).** The notebook launches the predictor as a **subprocess**
(`subprocess.Popen([...], env=shard_env)`, base notebook cell 5:388). The old patch passed its sink
and flush through the parent's `builtins`, which a fresh interpreter does not inherit; both guarded
call sites saw `None` and no-oped silently. Rewriting the source file DID cross; the in-memory hooks
did not. **Same class as the DataParallel/autocast trap — this is the second instance.**

**Repaired** in `scripts/kaggle_edits/detpeak_export.py`: sink and flush are now module-level inside
the patched source, gated on `BIOHUB_DETPEAK_ENABLE` / `_EXPORT_T` / `_DIR` (env vars DO cross).
Verified against the real vendor source — 3 anchors 1x each, patched module compiles, 5 hook sites
present, 0 `builtins` refs; a fresh-interpreter test writes and round-trips the npz, while the old
version under the identical test writes 0 files. `pytest` 769 passed.

The child now announces `detpeak: export ACTIVE in pid N`; its absence is the alarm.

**The curve is still unbought.** Re-running needs a fresh push — **not automatic** (rule 8). Fold 1
was left running because its LOEO export is on the current substrate (the disk copy in
`_evidence/exports/loeo_f1_strict/` predates P9). Raw output: `_evidence/p4_detpeak_f0/`.

Replay harness for p4 is built and tested: `scripts/win_bet/detpeak_curve.py` — it has never had
input.

---

### ✅ BOTH TRAINING LANES REPAIRED (commit `51365e7`) — launchable, not launched

**S5** had six defects, all fixed: the CUDA-autocast crash on step 1; **all division supervision was
being deleted** (discarding the 65,741-link asset the lane exists for); a ~9x background-prior error
from independent source/target sampling; a precision-free selection criterion; a "frozen detection"
that froze only a 1x1 conv; and a tautological guard, now replaced by a real `detection_drift()`
instrument that doubles as the pre-flight gate.

**S1**'s three SEV-1 defects were **already fixed by an earlier session** — verified at file:line,
not redone. One remained and is fixed: resume was local-only dead code, plus a latent corruption
where resuming without `metrics.json` relabelled a trained model as the zero-epoch baseline.

---

### 📌 CORRECTIONS TO STANDING BELIEFS
- **`ILP_APPEARANCE_WEIGHT` is a CLOSED lever.** 0.0 is the deliberate tuned public consensus (code
  default is 0.1; all 26 pulled kernels set 0.0). I had queued flipping it as a "precondition" —
  that was wrong.
- **The `leevvin` leak claim does NOT hold as stated.** The vendored trainer's default split is
  90/10 and strictly DISJOINT (`train_unet_transformer.py:1036-1051`); `test`-inside-`train` needs a
  custom splits file the support pack does not ship. **But the practical conclusion survives for a
  simpler reason:** 90/10 over 199 movies still puts ~179 in training, so scoring against
  `data/train` is contaminated anyway. State the right mechanism.
- **HOCT is closed** — measured under-performing a tuned ILP at ~45 min/movie by Arul
  (discussion/728551); the Ultrack author confirms they never tried it on this dataset.
- **`ILP_DIVISION_WEIGHT` 0.3/1.0/2.0/3.0 ALL scored 0.915** on someone else's LB (their number, not FACT-0002). Closed, free.
- **Kaggle permits a MAXIMUM OF 2 CONCURRENT GPU BATCH SESSIONS.** Pushes must be pipelined.

---

### ❌ CLOSED 2026-08-25 — TEMPORAL POSITION SMOOTHING, the former #1 ranked action

Killed on its own pre-registered criterion, CPU-only, zero slots. **Six of six measurements moved the
statistic DOWN**, across two substrates and both estimators. On the unsmoothed pre-ILP frame (17,462
mislinked GT edges): baseline **26.08%**, tracklet-mean **20.44%** (-5.64 pp), tracklet-linefit
**24.66%** (-1.41 pp), GT ceiling 78.85%. The scored f1 frame agrees in sign.

The kill survives the anchor problem below, because it depends only on the SIGN of the change against
each run's own baseline.

**Mechanism.** The mean was predicted to be worse before it ran — a zeroth-order average pulls nodes
toward the track centroid, injecting the motion bias already recorded as *"a static distance prior is
BACKWARDS"* — and it was, by 4x, in the predicted direction. But the line fit loses too, because
**smoothing follows PREDICTED tracks and predicted tracks are what is broken**; denoising along a
wrong link propagates that link's error. Only 35-39% of localisation error is independent per frame,
so that is the ceiling on what any smoother can remove.

**Do NOT reorder the deployed smoother.** `linefit_smooth_output_graph` runs at cell 6:1514, last
before `return`, so the linker never sees a smoothed coordinate. Moving it earlier — the actual
proposal — would COST ~1.4 pp of nearest-parent accuracy. Its current position is accidentally right.

**The 3.44% anchor does not reproduce and is now unverified.** Rebuilt as
`scripts/win_bet/nearest_parent_oracle.py` (the old one was ad-hoc and never committed). Calibration
gate 1 PASSES (GT displacement median **1.817 um** vs known 1.82) and the mislinked count reproduces
exactly (6,861), but the oracle reads **12.24%** vs recorded 3.44% and **73.24%** vs 68.39%. The
coordinate trap was tested as the cause and REFUTED (isotropic and raw-voxel both give 11.15%; only
the correct anisotropic scale differs, at 12.24%). Cite neither number until reconciled.

### NEXT, gated on the in-flight readings
1. **p15 decides the lane.** ~0.923-0.924 => our edge term is already at 0.953 level and the whole
   deficit is divisions. ~0.915-0.920 (a probe band, not our score) => the gap is genuinely edge, as the forum arithmetic implies.
2. **p4's curve picks the detection threshold** without spending a slot per point — but the export
   was BROKEN and bought nothing; see the IN FLIGHT section. Repaired and re-smoked 2026-08-25.
3. **The association ranker** (`scripts/kaggle_edits/ranker_block.py`, vendored + validated) is the
   rerank half of retrieve-then-rerank. Remaining work is the ~100-line feature context builder.
   **Its local evaluation is CONTAMINATED, so a slot is the only honest instrument for it.**
4. **A division SELECTOR, not a relaxation** — the +0.0232 ceiling is real but unreachable by
   loosening gates.


## 🗄️ SUPERSEDED BELOW (kept for provenance; read the LIVE section above first)

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
