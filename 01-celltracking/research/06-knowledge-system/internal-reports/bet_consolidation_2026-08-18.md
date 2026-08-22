# Bet consolidation: failure taxonomy, a-priori pruning rules, and the reduced portfolio — 2026-08-18

Labels used throughout. **[MEASURED]** = a number produced by the official scorer or a count over
real data, cited to its record. **[CODE]** = read off source, `file:line` given. **[INFERENCE]** =
derived by me from measured inputs, assumptions stated. **[UNVERIFIED]** = asserted somewhere in
the record but never checked.

Sources read in full: `research/01-research-direction/bets.yaml`,
`research/06-knowledge-system/failed-experiments.md`,
`research/06-knowledge-system/experimental-records.md`,
`research/07-outputs/submissions.md`, `research/06-knowledge-system/claims-table.md`,
`research/01-research-direction/directional-updates.md`, and the 2026-08-18 internal reports
`operating_point`, `loeo_lb_gap`, `live_surface`, `division_lane`, plus section maps of
`endgame_strategy`, `edge_training_frontier`, `h1_execution_spec`, `zebrahub_data_engineering`.

The six sibling reports the brief anticipated (`linker_division_capacity`, `cross_embryo_gap`,
`tier_reverse_engineering`, `instrument_repair`, `operating_point_v2`, `edge_loss_structure`) were
**not present on disk** at write time. Nothing here duplicates them because nothing here could
read them.

---

## 0. THE REDUCED PORTFOLIO — read this first

**We opened at least 55 distinct levers. Four survive.**

| # | Survivor | Class it belongs to | Cost | Falsification (count- or artifact-based, not LOEO) | Why it survives |
|---|---|---|---|---|---|
| **S1** | **Instrument calibration over the 5 existing LB anchors** — fetch P0-A / P0-B / P0-CR outputs, score locally, regress local vs LB (0.906 / 0.913 / 0.914 / 0.915 / 0.915) | measure the measurer | **0 GPU, 0 slots, ~30 min** (`loeo_lb_gap_2026-08-18.md` §6) | rank correlation ≈ 0 ⇒ no preflight instrument we hold is validated; that *is* the result | Precondition for every other bet. Every gate we own is currently unfalsified. |
| **S2** | **H1 Zebrahub retrain as a SUBSTITUTION for the contaminated `seed314159` secondary** (`bet-zebrahub-retrain`) | change the model | high (staged; `h1_execution_spec_2026-08-18.md`) | no gain over the public 50-ep weights once S1 gives a validated instrument; or Ultrack-imitation ceiling shows up as flat division recall | Only bet that creates new features rather than mining existing ones. Also the only one that **repairs the instrument as a by-product**. |
| **S3** | **Threshold-superset export + CPU threshold sweep** (`p4_detsweep_export_f{0,1}.json` + `detpeak_export.py`, written, **not pushed**) | change the detection surface | 1 GPU export/fold, then unlimited CPU replay | score monotone-decreasing as `T` falls ⇒ extra peaks are duplicates, not cells ⇒ dead, and it settles the `N_est` lever for free | The only *downstream* lever whose mechanism is node recall rather than edge allocation (`operating_point_2026-08-18.md` §1.2, L3). Break-even `p* ≈ 0.500` vs deployed `0.96875`. |
| **S4** | **Stage-0 A/B on `xiaoleilian/biohub-unet3d-weights-v2models`** — swap a *publicly retrained* detector into our stack | change the model, zero training | inference only, no Colab hours | naive swap loses on normalisation/grid ⇒ cheap to establish, and that itself is an H1 design input | Buys the answer to H1's central question ("does a retrained detector help *our* stack?") before any training spend. **Not currently in `bets.yaml` — it should be.** |

**What the four survivors have in common — and it is the single load-bearing generalisation of
this report:** every one of them either changes *which nodes exist* (S3, S2, S4), changes *the
model that produces them* (S2, S4), or *measures whether we can measure anything at all* (S1).
**Not one of them re-allocates edges among an already-committed node set.** Every lever we
killed in the last four weeks did exactly that, and the class is 0-for-2 on the leaderboard at
~7% edge churn.

Secondary commonality, equally important: **every survivor is falsifiable by a count, an artifact
fetch, or a monotonicity check — none of them requires a LOEO delta to be believed.** That is
deliberate. Per `loeo_lb_gap_2026-08-18.md` §4, LOEO deltas currently carry zero measured
predictive value (§4 of this report).

### Verdict on the host's working thesis

The thesis — *"everything downstream of the model is exhausted, so the survivors collapse to
(i) repair the instrument, (ii) change the model, (iii) at most one or two detection-surface
levers"* — is **CONFIRMED in substance, with three amendments.**

**Amendment 1 — the boundary is not "downstream of the model", it is "over a frozen detection
surface".** S3 is downstream of the model and survives, because it changes how many nodes exist.
The sharp form of the rule is: *anything that re-allocates structure over a fixed set of detected
nodes is exhausted; the one downstream knob that changes the size of that set is not.* Stating it
as "downstream" would wrongly prune S3, which is the cheapest live lever we hold.

**Amendment 2 — (i) and (ii) are not co-equal items, they are the same item viewed twice.**
`bet-zebrahub-retrain`'s own falsification in `bets.yaml:65` is *"No LOEO gain over the public
50ep weights"* — and that test is **currently unrunnable**, because the deployed pipeline cannot
be validly measured on any labelled data we hold (`loeo_lb_gap_2026-08-18.md` §4: the deployed
secondary `unet_transformer_alltrain_seed314159_v1` declares `"train_datasets": 199`). H1
designed as a *substitution* is what removes the contamination and restores the instrument. So
S1 is not merely "do this first" — H1's design is *constrained* by it.

**Amendment 3 — the host's list is missing S4.** A publicly retrained detector with published
weights already exists at LB 0.918 (`live_surface_2026-08-18.md` §3.2). It answers H1's central
question for inference cost. Omitting it would mean spending GPU hours to learn something we can
buy for free.

**Not amended:** the host's item (iii) resolves to exactly one lever, not two. Detection threshold
survives; every other candidate in that space (duplicate filter, node budget, N_est quantile,
grid shift, sub-voxel) is dead by measurement.

---

## 1. FAILURE TAXONOMY — classified by root cause

### 1.0 What I did to the host's four classes

The host's (A), (C), (D) survive with sharpened definitions. **(B) is refuted as a single class**
— it merges two mechanically distinct failures that have *different ex-ante tests*, and merging
them is why one of them (E) kept recurring after the other (B) had been solved. I also add two
classes the record contains but the host's list omits.

| host's class | verdict | what I did |
|---|---|---|
| (A) post-hoc graph surgery | **KEEP, renamed** | → **A: Zero-sum re-allocation over a frozen detection surface.** The scorer's per-frame matching is one-to-one, so this is not "post-hoc" as an incidental property — it is *structurally* zero-sum. |
| (B) hand-crafted geometry can't discriminate | **SPLIT** | → **B: Discriminability shortfall** (the signal is not in the features) and **E: Optimising inside the metric's dead zone** (the signal is there but the metric does not charge for it). Sub-voxel refine, which the host placed in B, is actually E. |
| (C) premise false in code | **KEEP, broadened** | → **C: Unverified load-bearing premise** (code *or* data *or* units). Half the instances are unit/metadata errors, not code errors. |
| (D) broken instrument | **KEEP, escalated** | → **D: Instrument invalidity.** Not an occasional accident — `loeo_lb_gap` §4 shows it is the *generic* behaviour of every instrument we hold. |
| — | **ADD** | **F: Asset unavailable** (licence / weights that do not exist). |
| — | **ADD** | **G: Surrogate-corpus distribution shift.** Distinct from B: the features *are* discriminative, they are discriminative of the wrong distribution. |

**Effort accounting caveat, stated once.** Token- or hour-level effort is **not recorded anywhere
in this repo** [UNVERIFIED by construction]. I account in the three units the record does hold:
**submission slots** (12 consumed, `submissions.md:402`), **GPU LOEO/kernel runs**, and **distinct
levers closed**. Where I say "cheap" or "expensive" I mean in those units.

---

### Class A — Zero-sum re-allocation over a frozen detection surface

**Definition.** The lever changes which edges exist, or which nodes own which edges, without
increasing the number of predicted nodes that can match a GT node. Because the scorer's per-frame
assignment is one-to-one (`metrics.py:276` match radius; `_ctc_metrics.py:174` `_match_single_frame`),
every gain has a symmetric loss available to it.

**Levers killed: ~13.**

| lever | measured outcome | record |
|---|---|---|
| motion gate arm B, solo | 7.148% edge churn → **LB +0.000** | `submissions.md:248-261` |
| motion gate arm B, on P3 | 6.42%/7.69%/7.19% churn → **LB +0.000**; LOEO said +0.0144/+0.0090 | `submissions.md:381`, `loeo_lb_gap` §1b |
| node-budget pruning (arm A / arm D / P0-B / 199-crop sweep) | +0.00099 pooled at optimum; arm D −0.00088; P0-B −0.0000103 | `bets.yaml:171`, `experimental-records.md:173` |
| duplicate-topology filter | 91.3%/90.7% of duplicates already free; ceiling **+0.003**; merge half has **0 in-degree≥2 across 3.8 M nodes** | `operating_point` §2 |
| division FP filter | **+0.0027** perfect-play, **+0.0007** honestly fitted | `division_lane` §3, `experimental-records.md:622` |
| high-precision fork selector (`bet-division-selector`) | `fork_at_matched_mother = 0` in **both** folds — FPs and FNs are **disjoint populations** | `division_lane` §0 |
| split/merge arbitration | +0.0006, below noise | `failed-experiments.md:27` |
| GT-free component selector | −0.008 | `failed-experiments.md:25` |
| association repair via bipartite competition | +0.00099/+0.00002; transformer picks the true parent **9.36%** vs ~50% break-even | `experimental-records.md:175` |
| orphan-target swap | blind −0.01667; LOFO −0.00019/+0.00006 | `experimental-records.md:176` |
| blanket short-component retention | edge TP *falls*; adjJ 0.8821→0.8766 / 0.8068→0.7924 | `experimental-records.md:177` |
| pre-ILP candidate breadth (10 µm) | −0.1596/−0.1496, **and still −0.1319/−0.1281 under a PERFECT ORACLE edge probability** | `experimental-records.md:39` |
| breadth candidate reranking | beat the wrapper ranking on neither held-out family | `experimental-records.md:31` |

**Effort consumed. The most expensive class in the project.** 2 of 12 submission slots
(`55181562`, `55585140`) spent for **exactly +0.000 twice**; 4 GPU LOEO kernels
(`biohub-p3-armb-loeo-f{0,1}`, `biohub-p3-base-armb-off-loeo-f{0,1}`); the full 199-crop
node-budget sweep; a full duplicate census over 3.8 M nodes; and the entire 2026-07-31-late
research block (six closed methods in one cycle). Arm B alone consumed a promotion decision, a
submission build, a diagnostic 2×2, and a host-verified churn analysis.

**The cheap ex-ante test that would have caught all thirteen.**
> Count the predicted nodes that newly become GT-matchable. If that count is **zero**, the lever
> is class A and its ceiling is bounded by whatever fraction of edge FPs are (a) on annotated
> cells and (b) contested. Both quantities were already on record before arm B was built:
> ~2,000 annotated GT edges over the four scored movies (`submissions.md:266`) and
> **1 net-correct repair = 1.095e-05 pooled, so +0.002 needs 183 net-correct repairs**
> (`experimental-records.md:211`).

That arithmetic was derived on **2026-07-31** and then not applied to arm B, which was promoted on
2026-08-17 and cost a slot on 2026-08-18. The single most expensive failure in the record is not
a wrong lever — it is a correct calculation that existed and was not consulted.

**The decisive datum for the whole class,** and it deserves more weight than it has been given:
pre-ILP candidate breadth was **−0.13 under a perfect oracle edge probability**
(`experimental-records.md:39`). A perfect ranker over a widened candidate set is catastrophically
negative. That is not a statement about ranker quality; it is a statement that the class has a
*negative ceiling* when it widens, and a ≈+0.003 ceiling when it does not.

---

### Class B — Discriminability shortfall: the signal is not in the features

**Definition.** A classifier/ranker is asked to separate a rare positive class using features the
model never encoded for that purpose. The required AUC or precision is computable from the base
rate *before* any model is built.

**Levers killed: ~10.**

| lever | required | measured | record |
|---|---|---|---|
| counterfactual image critic | AUC **0.9386** | appearance 0.657 / 0.496, 7–232× dependent | `experimental-records.md:253` |
| flat mother classification | AUC **0.983–0.9992** (92 true among 4,957,806) | 0.86–0.92 | `experimental-records.md:251` |
| branch-emergence proposer (D1) | AUC **0.9695** at K=20,000 | 0.856 / 0.777; a **GT-oracle in-sample logit is still 19.5× over budget** | `experimental-records.md:252` |
| division proposal via relaxed geometric gates | separate 1 true from **2,400–3,300** false candidates | ranker sorts *tightest-first*; true divisions are *wide* (7.4–8.9 µm) ⇒ rank last | `experimental-records.md:874-886` |
| learned division posterior | — | recall@precision 0.9 ≈ **0.045**, ≈0 on 3/4 embryos | `experimental-records.md:32` |
| laneD learned division pair-ranker | beat frozen geometry | MLP **≈ or < frozen geometry** | `directional-updates.md:128` |
| H1-T conditional pair ranker | — | **b = 0** in 44b6 — its correct set is a strict SUBSET of geometry's | `experimental-records.md:118` |
| edge-level selection among short-component deletions | separate deleted-true from selected | prob **0.785 vs 0.786**, raw_um **2.30 vs 2.30** — statistically identical | `experimental-records.md:178` |
| appearance × appearance stacking | independence | FPs concentrate on the **same** mothers (7–232× dependence), lift **0.00** | `experimental-records.md:115` |
| shape-aware localisation | separate misses | SMD ≤ 0.122; oracle +0.009125 → deployed **−0.008778** | `experimental-records.md:249` |

**Effort consumed: moderate, and the trend is genuinely good.** The two cheapest kills in the
entire project are in this class: the image critic was *"foreclosed by arithmetic, never
launched"* (`experimental-records.md:253`) and flat mother was killed on a base-rate count. The
project **learned** the ex-ante test for class B and applied it correctly at least four times.
The residual cost is the earlier instances (learned division posterior, D1, laneD) that were
built before the rule existed.

**The cheap ex-ante test — and it is already proven to work here.**
> From the base rate, compute the AUC (or precision at full recall) the classifier must reach for
> the *metric* delta to clear the bar. Then measure the AUC of the features you already have on a
> held-out family. If required > measured, the lever is dead and costs zero GPU to kill.
> The break-evens are already tabulated: detection **40.6%** precision, a division action
> **10.15%** at full recall (`experimental-records.md:146`, `claims-table.md:136`).

---

### Class E (NEW — split out of the host's B) — Optimising inside the metric's dead zone

**Definition.** The lever improves a real quantity, but the scorer does not charge for the error
it removes. Mechanically distinct from B: the signal *is* present and the intervention *does*
work; the metric is simply indifferent.

**Levers killed: 4.**

| lever | measured | why the metric didn't care |
|---|---|---|
| sub-voxel centroid refinement | LOEO 0.9037→**0.9033** (44b6), 0.7051→**0.7042** (6bba) — bilaterally negative | localisation RMS already inside the metric's free zone; scorer cliff starts ~1.5–2 µm; `OUTPUT_LINEFIT_SMOOTH` dilutes the rest (`experimental-records.md:500-521`) |
| duplicate-topology filter | ceiling **+0.003** | **91%** of duplicate pairs are parallel-disjoint components — free (`operating_point` §2) |
| split/merge arbitration | +0.0006 | below the noise floor (`failed-experiments.md:27`) |
| grid-centre shift (also class C) | −0.00163 / −0.00012 at the predicted optimum | the bias it corrects does not exist (`operating_point` §0.2) |

**Effort consumed: 2 GPU kernels** (`biohub-p3-subvoxel-loeo-f{0,1}`) plus the scaffold
`h1r_subvoxel_refine.py`, plus a full duplicate census. **Notably, the sub-voxel kill was
correctly predicted in advance from code** — see §4; the spend happened anyway.

**The cheap ex-ante test.**
> Price the intervention in the scorer's own units *before* building it. Multiply the number of
> **annotated** GT edges/nodes it touches by the per-unit value (1.095e-05 pooled per net-correct
> repair). If the resulting ceiling does not exceed the LB quantum (0.001) with a factor of 2+ of
> headroom, do not build it. Corollary: measure the *free zone* first — for localisation that is
> the 7 µm match radius (`metrics.py:276`) and the measured cliff at ~1.5–2 µm.

**Why splitting E out of B matters practically.** B's test is a *separability* calculation; E's
test is a *metric-economics* calculation. They have different inputs and different owners. Merging
them is why the project solved B (four correct zero-GPU kills) while continuing to pay for E
(sub-voxel spent 2 GPU kernels *after* its own mechanism had been derived from code).

---

### Class C — Unverified load-bearing premise (code, data, or units)

**Definition.** A factual claim that the lever depends on was asserted and never checked at
`file:line` or against the data. This class produces no measured negatives — it produces *wasted
direction*.

**Instances: 12+.** This is the largest class by count and the cheapest to prevent.

| premise asserted | truth | record |
|---|---|---|
| the loader block-averages, so `coords *= ds_arr` omits a +0.5-voxel grid-centre term | loader is **pure striding** — `raw = zarr_arr[t, ::dz, ::dy, ::dx]`, `predict_unet_transformer.py:215`; estimator is coarse but **unbiased** | `operating_point` §0.2 |
| F7: "a systematic 0.609 µm y/x bias, RMS 1.075 → 0.642 µm from a constant shift" | **WITHDRAWN.** Host verification had confirmed only that the *line exists*, not the interpretation | `experimental-records.md:727-732` |
| `N_est` is a usable operating-point target | `N_est` is **GT metadata** (`src/biotrack/metric.py:37-47`), unreadable at test time — **and inverted**: we under-produce (pooled 0.70/0.93), so forcing `N_pred=N_est` costs **−0.018 / −0.005** | `operating_point` §3 |
| the **sister** gate (7.2 µm) is what blocks divisions | the **parent** gate (4.7 µm) is binding; it kills 34 of 35 survivors, after which the sister gate removes **zero** more | `experimental-records.md:866-871` |
| ~89 of 151 divisions are proposable ⇒ ceiling ~+0.058 | genuinely proposable pool is **35**; ceiling **+0.023** at perfect precision | `experimental-records.md:888-892` |
| F3: forbid a second in-edge | **0** in-degree ≥ 2 nodes across 3.8 M — nothing to act on | `operating_point` §2.4 |
| node budget's sign "tracks the node ratio" | **WRONG.** Count cost is `0.1·tp/N_est`, *exactly invariant* to over/under-prediction; the real driver is `d_tp/d_fp` of what is deleted | `experimental-records.md:187-204` |
| "P0-B has 8 forks" | **units error** — 8 is its metric division-FP count; P0-B has **305** graph forks | `experimental-records.md:270` |
| a level-1 retrain "voids the deployed 0.915 anchor" | unsupported — zh001r is **exactly 1.625 µm**, identical to the deployed detector input | `experimental-records.md:794-798` |
| our own `scale_zyx = [0.62, 0.2195, 0.2195]` for ZSNS003_L1 | **wrong by ~4.5× laterally**; three independent instruments agree; likely off-by-two in pyramid indexing at `h1r_fetch_imaging.py:50-59` | `experimental-records.md:688-701` |
| the zh001r→Zebrahub transform needs an anisotropic ~4.1× y/x scale (host guidance) | transform is **isotropic**: `global_um = origin + 1.625 * crop_coord`; tracks CSV is already in microns | `experimental-records.md:790-796` |
| "temporal NMS is free" | property of *one* score; under other rankers it destroys **38–48 of 92** true forks | `experimental-records.md:274` |
| "annotation-coverage inflation" for division features | **FALSE** — full-denominator AUC matches the annotated subpopulation | `experimental-records.md:275` |
| `zh001r` cannot supervise the edge half (no track identity) | **OVERTURNED** — registration recovered 1,258,182 GT association edges incl. 65,741 division-daughter links | `experimental-records.md:773-788` |

**Effort consumed: not GPU, but *steering*.** The level-1 acquisition plan budgeted 150–232 GB and
a full resample engineering pass on a scale claim now known to be wrong by 4.5×. The sub-voxel
lane's stated residual came from F7, now withdrawn. The division lane spent a full report on the
sister gate before the parent gate was identified as binding. Class C does not lose experiments —
it aims them at the wrong target.

**The cheap ex-ante test.**
> For every load-bearing premise, write the `file:line` (for code) or the exact measurement (for
> data) that establishes it, **in the bet, before any work**. "The code says X" is not
> verification; "I read line N and it does X" is. Note the failure mode the record actually
> exhibits: host verification of `predict_unet_transformer.py:495` confirmed *that the line
> exists* and this was recorded as confirming *the interpretation of what it does*. Verification
> must name the claim it discharges, not the line it read.

---

### Class D — Instrument invalidity

**Definition.** The measurement is sound in its own frame but the frame is not the one that
scores us. Escalated from the host's framing: `loeo_lb_gap_2026-08-18.md` §4 establishes this is
not an occasional accident.

> **There is no configuration in which the deployed pipeline can be validly measured against
> ground truth.** The deployed secondary `unet_transformer_alltrain_seed314159_v1` declares
> `"train_datasets": 199` — trained on every labelled crop. LOEO must ablate it (leakage), so
> **LOEO measures a different pipeline**. The four `data/test` movies are **byte-identical** to
> train crops (SHA-256 over 102 files each), so scoring there is contaminated by memorisation —
> quantified at **3.8× inflation** (deployed chain +0.0365 vs LOEO-strict +0.0097 on identical
> crops).

**Instances: the whole promotion history.**

| instance | measured |
|---|---|
| motion gate arm B | LOEO **+0.0144/+0.0090** → LB **+0.000**. Best instrument we had, single toggle, paired, official scorer, bilaterally positive, min-fold above the bar. Transferred nothing. |
| the "five instances" pattern | node budget · H0d live-filter cross-term · reverse-time · the 22/26 substrate · suppress-all — **every one transferred badly across substrate or family** (`experimental-records.md:280-287`). All five were measured on the ablated pipeline and deployed on the full one. |
| fold-0 noise floor | fold 0 total weight **W = 21,210** vs fold 1's **123,413** (5.8×). Measured noise floors **±0.003 / ±0.004**. `44b6_2f31fc2f` swings **+0.066** in its own Jaccard off a **+4 edge-TP** change. **Fold 0 is our weakest instrument and the source of our most exciting deltas.** (`operating_point` §0.1b) |
| node budget "closed" | prior kill used the **forbidden placeholder-movie substrate**; honestly reopened, then re-closed as sub-bar (`bets.yaml:162-173`) |
| H1-M | pooled ~+0.0023 → **+0.00007/−0.00131** — an in-family CV ceiling quoted as cross-family, **~30×** (`experimental-records.md:135`) |
| node budget headline | +0.00822 → **+0.00157**, 5.2× (`experimental-records.md:136`) |
| FN association share | 63.5% → **43.3%**, 1.5× (`experimental-records.md:137`) |
| reverse-time | **base-dependent**: +0.001 on clean913, **−0.002** on v122 — same mechanism, same `w=0.20`, opposite sign (`submissions.md:156-165`) |

**Effort consumed: 2 submission slots, 4 GPU LOEO kernels, and the credibility of every gate we
own.** The `+0.005 bilateral LOEO` promotion bar is now **falsified as a sufficient gate** — it
passed a lever worth exactly zero (`submissions.md:395-397`).

**The cheap ex-ante test.**
> Before measuring, name the instrument and answer two questions in writing: **(1)** does this
> instrument run the *same pipeline* the leaderboard runs? **(2)** is the predicted effect larger
> than that instrument's measured noise floor by ≥2×, on **both** folds?
> As of today, question (1) has the answer **"no, for every instrument we hold"** — which is why
> S1 is the top of the portfolio and why the surviving levers are falsified by **counts** rather
> than by score deltas. The template already exists: `div_proposal_funnel.py` measured the
> division thesis as *counts*, "deliberately immune to the LOEO→LB transfer problem"
> (`experimental-records.md:846`), and it produced the cleanest kill of the month.

---

### Class F (NEW) — Asset unavailable

**Levers killed: 3.** CTC replacement corpus (*"cloning of datasets or their parts, including
reference annotations, is strictly forbidden"*); CAP / track-as-point (**no licence at all**, plus
CC BY-NC upstream, plus CTC training data, plus **no weights exist despite the abstract claiming
they do**); OrganoidTracker marginalisation code (GPL-2, reimplement never vendor).
(`experimental-records.md:230-241`)

**Effort: "substantial reading" before the block was found, on two of three.**

**The cheap ex-ante test — already a standing rule, and it works:** *audit the licence, and verify
the advertised checkpoints actually exist, BEFORE reading the code for reuse.*

---

### Class G (NEW) — Surrogate-corpus distribution shift

**Definition.** Distinct from B: the features *are* discriminative — of the wrong distribution.

**Levers killed: 4.**

| lever | measured |
|---|---|
| Freitas synthetic divisions | sister separation **7.24 µm** vs our measured GT median **10.57 µm** — 32% below, and *below the 8.5 µm cap measured to retain only 29.1% of true pairs* (`experimental-records.md:303-313`) |
| synthetic split patches | real-vs-synth CV AUC **0.9888** (trivially separable); synth-trained→real AUC **0.664** vs 0.867 real-trained (`experimental-records.md:119`) |
| M1 domain randomisation | same-family +0.0074; **cross-family +0.0010, CI [−0.0102,+0.0132]**; made raw held-out linking *worse* 0.6499→0.6421 (`experimental-records.md:38,59`) |
| Zebrahub as a division corpus | 5.7–11.7 terminations per division; oracle over 30 anchor×stride pairs still **wrong-signed** (`experimental-records.md:116`) |

**The cheap ex-ante test.**
> **(1)** Match the *load-bearing distributional parameter* — for divisions that is inter-daughter
> separation; measure it on both corpora and compare medians and p90. **(2)** Train a
> real-vs-surrogate discriminator; if its AUC is far from chance, transfer will fail.
> The red-team caveat is essential: **per-feature SMD audits drastically understate multivariate
> separability** — synthetic patches passed at mean |SMD| 0.302 while being **98.9% separable**
> (`experimental-records.md:141`). Use a discriminator, not an SMD table.

**Live relevance:** this class governs the one real caveat on S2. The zh001r sidecar labels are
**Ultrack's automated output**, so an edge model trained on them learns to *imitate Ultrack* — a
ceiling as well as a floor (`experimental-records.md:812-814`). The G-test applies to H1 and has
not been run.

---

### 1.9 Taxonomy roll-up

| class | levers killed | heaviest recorded cost | cheap ex-ante test | is the test already in use? |
|---|---:|---|---|---|
| **A** zero-sum re-allocation | ~13 | **2 submission slots for +0.000 twice**, 4 GPU LOEO kernels | count newly GT-matchable nodes; if 0, bound the ceiling by annotated-edge economics | **No** — the arithmetic existed 2026-07-31 and was not applied |
| **B** discriminability shortfall | ~10 | early instances (D1, laneD, learned posterior) | required AUC from base rate vs measured cross-family AUC | **Yes**, ≥4 correct zero-GPU kills |
| **E** metric dead zone | 4 | 2 GPU kernels (sub-voxel) | price in scorer units before building; measure the free zone | **Partly** — derived, then ignored |
| **C** unverified premise | 12+ | mis-aimed acquisition + division + localisation lanes | `file:line` or measurement per premise, in the bet, before work | **No** |
| **D** instrument invalidity | the promotion history | 2 slots, 4 kernels, the promotion gate itself | name the instrument; same pipeline? ≥2× its noise floor bilaterally? | **Now, yes** — `div_proposal_funnel` is the template |
| **F** asset unavailable | 3 | substantial reading ×2 | licence + weights-exist audit first | **Yes**, standing rule |
| **G** surrogate shift | 4 | Freitas evaluation, M1 seed | match the load-bearing parameter; real-vs-surrogate discriminator (not SMD) | **Yes** — but **not yet applied to H1** |

---

## 2. THE A-PRIORI PRUNING RULE SET

Run these **before** a bet earns effort. Any single FAIL is sufficient to prune; a bet that cannot
answer a question is NEEDS-CHEAP-CHECK, not KEEP.

| # | Rule | Question | Kills class | Evidentiary basis |
|---|---|---|---|---|
| **R1** | **Detection surface** | Does this increase the count of predicted nodes that newly become GT-matchable, or does it only permute/filter structure over nodes already committed? | A | arm B: 6–7% churn twice → LB +0.000 twice; breadth −0.13 *under a perfect oracle* |
| **R2** | **Separability budget** | Compute the required AUC / precision from the base rate. Is it ≤ the measured cross-family AUC of features we already hold? | B | image critic needed 0.9386 vs 0.657 measured — killed with **zero GPU** |
| **R3** | **Metric unit economics** | Priced in scorer units (1 net repair = 1.095e-05 pooled; +0.002 = 183 repairs), does the *perfect-play ceiling* exceed 2× the LB quantum? | E | division FP filter: +0.0027 perfect / **+0.0007** fitted; duplicates: +0.003 |
| **R4** | **Premise verification** | Is every load-bearing premise pinned to a `file:line` or a measurement, and does that citation discharge *the claim* rather than merely confirm a line exists? | C | grid-shift premise refuted at `predict_unet_transformer.py:215`; F7 withdrawn |
| **R5** | **Instrument validity** | Which instrument measures this, and does it run the pipeline the leaderboard runs? If not, is the falsification expressible as a **count** instead of a score delta? | D | `loeo_lb_gap` §4: currently **no** instrument runs the deployed pipeline honestly |
| **R6** | **Noise floor** | Does the predicted effect exceed the measuring fold's noise floor (f0 ±0.003, f1 ±0.004) by ≥2×, on **both** folds? | D | fold 0 W=21,210; `44b6_2f31fc2f` swings +0.066 on 4 edges |
| **R7** | **Surrogate fidelity** | If it trains on a surrogate corpus: is the load-bearing distributional parameter matched, and is real-vs-surrogate discriminability near chance? (Discriminator, **not** an SMD table.) | G | Freitas 7.24 vs 10.57 µm; real-vs-synth AUC 0.9888 |
| **R8** | **Asset reality** | Licence audited and weights verified to exist, *before* reading the code? | F | CAP: no licence at all, and the advertised checkpoints do not exist |

**Meta-rule (derived in §4, and it is the one with the best track record):** where a structural
argument (churn magnitude against a known-null reference; a count-based funnel) and a LOEO delta
disagree, **the structural argument wins**. It is 2-for-2; LOEO→LB is 0-for-2.

### 2.1 Verdict table — every `proposed` / `open` / `parked` bet in `bets.yaml`

`bets.yaml` holds 12 bets: 4 `closed`, 2 `lost`, 1 `parked`, **5 `proposed`**. All six
non-terminal bets are screened below.

| bet | status | verdict | rule(s) fired | evidence |
|---|---|---|---|---|
| **`bet-zebrahub-retrain`** | proposed | **KEEP** — the portfolio's spine | R1 ✅ R2 ✅ R3 ✅ R4 ⚠️ R5 ⚠️ **R7 ✱ unrun** | Changes the model, so it is the only bet that *creates* features rather than mining them. Supervisable end-to-end from `kkunizaw/biohub-zh001r` + `data/external/zebrahub/zh001r_identity.npz`: **1,258,182 GT association edges incl. 65,741 division-daughter links**, 116,320 track ids, at **exactly 1.625 µm** = the deployed detector input. R4 ⚠️: its history is class-C-dense (scale wrong 4.5×, "voids the anchor" refuted, "no track identity" overturned) — every remaining premise needs a citation. R5 ⚠️: its stated falsification (`bets.yaml:65`, "no LOEO gain") is **currently unrunnable** — see S1. **R7 is the live risk and has not been tested**: Ultrack labels ⇒ the model learns to imitate Ultrack. |
| **`bet-learned-ranker`** | proposed | **PRUNE as a standalone lane; FOLD INTO `bet-zebrahub-retrain`** | **R1 FAIL, R2 FAIL** | R1: "over-propose candidates then re-score" permutes *edge* candidates over a fixed detection surface. R2: refuted three ways — pre-ILP breadth at 10 µm was −0.1596/−0.1496 **and still −0.13 under a perfect oracle edge probability** (`experimental-records.md:39`); H1-T conditional pair ranker measured **b = 0**, its correct set a strict *subset* of geometry's; breadth reranking beat the wrapper on neither held-out family. A ranker over *new* features (i.e. inside H1) is a different bet and is already in S2's scope. |
| **`bet-synthetic-division`** | proposed | **PRUNE** | **R7 FAIL, and dominated** | Freitas sister separation **7.24 µm** vs GT median **10.57 µm** — below the 8.5 µm cap measured to retain only 29.1% of true pairs, i.e. mis-calibrated *in the one parameter that governs reachability*; real-vs-synth CV AUC **0.9888**; synth→real **0.664** vs 0.867. Independently **dominated**: the zh001r sidecar supplies **65,741 real division-daughter links** vs our 151 events. Reopening condition already on record (`experimental-records.md:321-323`) and unmet. |
| **`bet-ot-linker`** | proposed | **PRUNE** | **R1 FAIL, R4 FAIL ×2** | Two of its three advertised benefits are refuted *in its own mechanism statement* (`bets.yaml:139`). "one→two mass split = division" is geometric division proposal — refuted: every gate that admits real divisions opens **2,400–3,300 false candidates per true one**. "mass knob = `N_pred` vs `N_est`" — `N_est` is **GT metadata** (`metric.py:37-47`), unreadable at test time, **and inverted**: we under-produce, so targeting it costs **−0.018/−0.005** before any edge gain. The residual (a better linker over a fixed candidate set) is pure class A. |
| **`bet-meta-ranker`** | proposed | **PRUNE as a lane** — with one free observation attached | **R1 FAIL, R2 FAIL, R3 FAIL** | R1: "filters FP edges" over committed candidates. R2: the directly analogous measurement found deleted-true and selected edges **statistically identical** (prob 0.785 vs 0.786; raw_um 2.30 vs 2.30). R3: its best-characterised instance (division FP filter) caps at **+0.0027** perfect / **+0.0007** fitted. "ranks toward `N_est`" fails R4 with `bet-ot-linker`. **Free observation, ~30 min, no lane:** `xiaoleilian/biohub-unet3d-weights-v2models` ships `edge_prune_hgb.npz` — a working instance of this exact idea from a team at LB 0.918. Inspect it while doing S4; do not build our own. |
| **`bet-division-selector`** | parked | **PRUNE — formally dead, with a mechanism** | **R1 FAIL, R3 FAIL** | `fork_at_matched_mother = 0` in **both** folds: the 613 FP forks and the 146 FN divisions are **disjoint populations**, so re-ranking existing forks recovers **exactly zero** divisions. Its +0.06 GT-oracle ceiling is real but unreachable by selection. It survives **only** as an acceptance criterion for S2. |

### 2.2 Live levers not in `bets.yaml` — screened for completeness

| lever | source | verdict | rule(s) |
|---|---|---|---|
| **Threshold-superset export + CPU sweep (L3)** | `operating_point` §1.2 | **KEEP = S3** | R1 ✅ (only node-recall lever in that report), R3 ✅ (`p* ≈ 0.500` vs deployed `0.96875`), R5 ✅ (falsified by monotonicity, not a delta), R6 ⚠️ (must clear ±0.003/±0.004 bilaterally) |
| **LB-anchor instrument calibration** | `loeo_lb_gap` §6 | **KEEP = S1** | R5 — this rule *is* the lever. 0 GPU, 0 slots. |
| **xiaoleilian stage-0 detector A/B** | `live_surface` §3.2, `h1_execution_spec` §6.3 | **KEEP = S4** | R1 ✅, R8 ✅ (weights public, 62 MB, verified via the Kaggle API) |
| **D1 "re-gate the proposer to measured GT geometry"** | `division_lane` §4 (D1, "do this first") | **PRUNE** | **R2 FAIL.** Two internal reports disagree and the later, count-based, host-verified one wins: `div_proposal_funnel.py` walked all 151 GT divisions through `add_safe_divisions_postlink` and admitted **1**. The linker consumes the second daughter in **61/96 (64%)** of detected divisions (`candidate_ids` requires no incoming edge, `wrapper.py:806`) — unproposable at *any* gate. Relaxation opens 2,400–3,300 false per true and the ranker sorts tightest-first, i.e. backwards. Standing decision stands: **"close the post-processing division lane. Do NOT relax the gates."** (`experimental-records.md:910`) |
| **Duplicate relink filter (L4)** | `operating_point` §2 | **PRUNE** | R1 FAIL, R3 FAIL (+0.003 ceiling, arm-B class) — the report itself recommends against spending |
| **Count-quantile to `N_est` (L5)** | `operating_point` §3 | **PRUNE** | R4 FAIL (GT metadata), and sign-inverted |
| **Grid-centre shift (L1/L2)** | `operating_point` §0 | **DEAD, measured** | R4 FAIL at `predict_unet_transformer.py:215`; R6 FAIL (+1 flips sign across embryos) |

### 2.3 One infrastructure repair that is not a bet but blocks the bets

`GATE_STATS` is **structurally unrecoverable from every LOEO run** — `loeo_export.py:117-124`
deletes `submission.csv` before the provenance cell reads it (`armb_provenance.py:21`), so all
four kernels died with `FileNotFoundError`. Fix is an edit-ordering swap in the spec. Without it
**no LOEO lane can report what its lever actually did** — which is precisely the diagnostic that
would have caught arm B earlier. Do it before S3's GPU export.

---

## 3. THE REDUCED PORTFOLIO — the count

| stage | count | basis |
|---|---:|---|
| formal bets ever opened (`bets.yaml`) | **12** | 4 closed · 2 lost · 1 parked · 5 proposed |
| distinct levers closed in the graveyards | **≥43** | `failed-experiments.md` (7) + `experimental-records.md` closed-method tables (8 + 7 + 6 + 7 + 8) |
| live non-bet levers surfaced by the 08-18 reports | **7** | L1–L5, D1, stage-0 A/B |
| **total distinct levers ever opened** | **≈55** | deduplicated across the three sources |
| screened this session (`proposed`/`open`/`parked` + live non-bet) | **13** | §2.1 + §2.2 |
| **survivors** | **4** | S1–S4 |
| of which formal bets | **1** | `bet-zebrahub-retrain` |
| of which not currently in `bets.yaml` | **3** | S1, S3, S4 — **all three should be added** |

**Prune rate this session: 9 of 13 (69%).** Four of the five `proposed` bets die, and the one
`parked` bet dies. That is the correct outcome: five of six were written before the three
findings that now govern everything — the disjointness of division FPs and FNs, the 0-for-2
churn record, and the instrument-invalidity finding.

**What the survivors have in common — restated as a testable rule for the next cycle:**

1. **Each changes what exists, not what owns what.** S2/S4 change the model; S3 changes the node
   count; S1 changes what we can know. None re-allocates edges.
2. **Each is falsifiable without a LOEO delta.** S1 by a regression over 5 existing artifacts, S2
   by counts + a restored instrument, S3 by a monotonicity check, S4 by an inference run. This is
   not stylistic — it is forced by `loeo_lb_gap` §4.
3. **Each has a measured ceiling above the LB quantum by a wide margin.** The gap to leader is
   0.036 (0.915 → 0.951, `live_surface` §1). Every pruned lever's perfect-play ceiling was
   ≤ +0.023 and realistically ≤ +0.003. Only a model change is in the right order of magnitude.
4. **Three of four cost approximately nothing.** S1 is 30 minutes; S4 is inference; S3 is one GPU
   export per fold and then unlimited CPU. Only S2 is expensive, and S4 partially de-risks it.

**Blunt statement of the wasted effort, because the point is to stop repeating it.** The project
spent **2 of 12 submission slots and 4 GPU LOEO kernels** on one class-A lever that churned ~7% of
edges twice for **exactly +0.000 twice** — after having derived, on 2026-07-31, the arithmetic
that bounds that class (`+0.002 needs 183 net-correct repairs` over ~2,000 annotated edges). A
further 2 GPU kernels went to sub-voxel refinement **after** its own mechanism had been derived
from source and predicted the kill. In both cases the correct answer existed in this repository
before the spend. The taxonomy above is not new knowledge; it is knowledge we already had, filed
under the wrong index.

---

## 4. PREDICTION-CALIBRATION AUDIT

### 4.1 Every recorded prediction vs outcome

| # | arm / claim | prediction, recorded **before** the result | actual | verdict |
|---|---|---|---|---|
| 1 | **P0-A** | exact reproduction of public 0.913 | **0.913** | **HIT** (reproduction, not an effect prediction) |
| 2 | **P0-B** | historical public effect 0.912→0.914 | **0.914** (+0.001) | roughly on |
| 3 | **P0-CR** | **0.908–0.912**, explicitly *"a causal probe, not a score climb"* | **0.906** | **MISS — below the stated range** |
| 3b | P0-CR *structural pre-registration* | divisions **+10** on v122 vs **−9** on clean913; "the arm whose divisions move up is the one that loses score" | v122 arm lost 0.002 | **HIT** — direction called before any score existed |
| 4 | **P1 (node budget)** | measured −0.0000103; withheld, slot not spent | not submitted | **HIT** (a correct decision to *not* spend) |
| 5 | **arm B solo** | **≈ +0.0045**, from an "OOF→public slope of ~0.56" | **+0.000** | **MISS — optimistic by 0.0045; slope measured 0.00** |
| 5b | arm B *structural pre-registration* | divisions **+13 (up)** ⇒ by the P0-CR precedent, should lose | went **flat** | ledger's own read: *"not confirmed and not refuted"* |
| 6 | **P3 harmonic** | range 0.913–0.928; **modal band 0.916–0.921 @55%**; low band 0.913–0.914 @25% | **0.915** | **MISS on the central estimate** — in range, **below the modal band**; **low band ≈ correct** |
| 6b | P3 *churn argument* | "arm B churned 7.148% for 0.000; P3 churns 4.368%, so hard to argue +0.016" | +0.001 | **HIT** |
| 6c | P3 *net-cardinality argument* | +1% nodes and edges ⇒ net-additive ⇒ push estimate up | +0.001 | **MISS** — ledger: *"carried no predictive signal"* |
| 7 | **P3 + arm B** | **0.916–0.920 central**; low band 0.915–0.916 @40% | **0.915** | **MISS on the central estimate** — landed **at the bottom of the low band** |
| 8 | **sub-voxel refine** | predicted *in advance from code* that localisation error was already inside the metric's free zone ⇒ cannot pay | **−0.0004 / −0.0009** | **HIT** — a code-derived **negative** prediction |
| 9 | **grid-centre shift** | +2 predicted as the optimum | **−0.00163 / −0.00012**; premise refuted at `:215` | **MISS** — a code-derived **positive** prediction |
| 10 | **counterfactual image critic** | foreclosed by arithmetic (needs AUC 0.9386), never launched | never launched, 0 GPU | **HIT** |
| 11 | **flat mother / D1 proposer** | required AUC exceeds measured ⇒ dead | closed | **HIT ×2** |
| 12 | **division funnel** | counts predicted the thesis would fail on proposal, not ranking | 1 of 151 admitted | **HIT** |

### 4.2 The bias, quantified

Take the four arms with a numeric central estimate of an *effect* (excluding reproductions):

| arm | anchor | predicted Δ (central) | actual Δ |
|---|---|---:|---:|
| P0-CR | v122 0.908 | +0.002 | **−0.002** |
| arm B solo | P0-B 0.914 | +0.0045 | **+0.000** |
| P3 harmonic | P0-B 0.914 | +0.0045 | **+0.001** |
| P3 + arm B | P3 0.915 | +0.003 | **+0.000** |
| **totals** | | **+0.014** | **−0.001** |

- **Realisation ratio, all four: −0.07.** Excluding P0-CR (explicitly labelled a probe, not a
  climb): predicted **+0.012**, actual **+0.001** → **realisation ≈ 0.083**.
- **Directional bias: 4 consecutive optimistic central estimates** (`submissions.md:383-386`).
  Honest statistics: a pure sign test on n=4 gives **p = 0.125**, which is *not* conventionally
  significant. **The magnitude carries the finding, not the sign count** — an 8% realisation
  ratio is not a small-sample artifact.
- **Measured LOEO→LB slope: 0.00 on n = 2.** The projected slope was 0.56 (`submissions.md:255`).
  The two levers where both numbers exist are (OOF +0.0088 → LB +0.000) and (LOEO +0.0090/+0.0144
  → LB +0.000). The one arm with a *positive* measured LB delta (P3, +0.001) had **no LOEO number
  at all**. There is currently no evidence that the LOEO→LB slope is distinguishable from zero.

### 4.3 The discount factor to apply

Three forms, in increasing order of how much I trust them:

1. **Multiplicative discount on internal central estimates: ×0.08** (≈ **÷12**). Derived from the
   3-arm realisation ratio above, excluding the probe. Blunt, single-number, applies to any
   internally-projected LB delta.
2. **LOEO deltas: multiply by 0.00 until S1 returns a slope.** Not a discount — a **suspension**.
   A LOEO delta is currently a statement about a *different pipeline* (`loeo_lb_gap` §4) and
   should be treated as zero evidence about the leaderboard. This supersedes the "+0.005 bilateral"
   bar, which is falsified as a sufficient gate.
3. **The best-calibrated correction, and the one I recommend actually adopting: promote the stated
   LOW band to the central estimate.** Scored against the record: P3's low band was 0.913–0.914
   and the actual was 0.915 (one quantum above); P3+armB's low band was 0.915–0.916 and the actual
   was **0.915** (dead on). **When a low band was stated, the outcome landed in it or within one
   quantum of it — 2 for 2 — while the modal band was wrong 2 for 2.** We already know how to
   produce a well-calibrated number; we then label it "low band" and quote the other one.

### 4.4 What class of reasoning has actually been predictive — VERIFIED against the record

The ledger's claim (`submissions.md:357-359`) is that **structural-churn arguments beat
net-cardinality arguments**. **Verified, and it is understated.** Full scoreboard:

| reasoning class | hits | misses | verdict |
|---|---:|---:|---|
| **count-based funnel / oracle / base-rate arithmetic → a KILL** | **5** (image critic, flat mother, D1, division funnel, node-budget withhold) | 0 | **Best in the project. Costs ~zero GPU.** |
| **code-level mechanism → a KILL** | **1** (sub-voxel, predicted from `predict_unet_transformer.py:495`) | 0 | reliable |
| **structural churn vs a known-null reference** | **1 used** (P3) + **1 available and ignored** (P3+armB) | 0 | **2-for-2 when consulted** |
| **division-count direction pre-registration** | 1 (P0-CR) | 0 (1 push: arm B) | promising, n=2 |
| **code-level mechanism → an OPPORTUNITY** | 0 | **2** (grid-shift F7 bias; `N_est` quantile) | **unreliable** |
| **net cardinality change** | 0 | 1 (P3) | no signal — ledger's own verdict |
| **LOEO delta → LB delta** | 0 | **2** | **zero measured predictive value** |
| **OOF→public slope ≈ 0.56** | 0 | 1 | refuted; measured 0.00 |

**Three findings fall out of this table, and they are the practical payload of §4.**

**(a) The sharpest indictment in the record.** For P3+armB, the churn heuristic was **available
and would have been correct** — the arm churned ~6% of edges, and the directly analogous prior
(arm B solo, 7.148% churn → +0.000) was already on the books. It was **overridden by the LOEO
number**. Our best-performing predictor was discarded in favour of our worst-performing one, and
it cost a submission slot. Rule: *where structure and LOEO disagree, structure wins.*

**(b) Code-level reasoning is asymmetric — reliable for kills, unreliable for opportunities.**
Same report (`retrain_recipes_2026-08-17`) produced both the correct sub-voxel kill and the
withdrawn F7 bias claim. The asymmetry has a mechanism: killing requires showing *one* necessary
condition fails, which a single line can do; claiming an opportunity requires showing *all*
conditions hold, which a single line cannot. Treat a code-derived opportunity as a hypothesis
requiring measurement; treat a code-derived kill as decisive.

**(c) Counts beat scores.** Every one of the five zero-GPU kills was a *count* — required AUC vs
measured AUC, candidates per true positive, forks at matched mothers, divisions admitted through
a funnel. Not one was a score delta. This is why all four survivors in §0 are specified with
count-based or artifact-based falsifications.

---

## 5. What I did not verify

- **Effort in tokens or hours is not recorded anywhere in this repo.** §1 accounts in submission
  slots, GPU kernels, and levers closed. Any hour figure would be fabricated.
- **The ≈55 lever count is a deduplicated read of three ledgers**, not an audited registry. Some
  entries (e.g. node budget) appear in more than one cycle under different arms; I counted those
  once. The count is a lower bound on distinct *attempts*, not an exact figure.
- **The six sibling 2026-08-18 reports the brief anticipated do not exist on disk.** If any of
  them lands a finding that contradicts a PRUNE above — particularly `linker_division_capacity`
  (which could bear on `bet-division-selector`) or `edge_loss_structure` (which could bear on
  `bet-learned-ranker`) — the relevant row should be re-screened rather than assumed settled.
- **The realisation ratio of 0.083 rests on n = 3** (n = 4 including the probe). It is a working
  discount, not an estimate with a confidence interval.
- **S4's premise that a naive weight swap is even loadable** into our stack is
  `h1_execution_spec` §6.3's claim, not independently verified by me at `file:line`. Per R4 it
  needs one before it consumes an inference run.
