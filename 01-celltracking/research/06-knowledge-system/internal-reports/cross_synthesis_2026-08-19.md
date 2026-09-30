# Cross-synthesis — what no single report can see (2026-08-19)

**Standing document. Regenerated wholesale on every update.** Do not edit in place; re-run the
cross-analyst with the new reports and replace this file.

| field | value |
|---|---|
| corpus at this revision | 27 reports in `internal-reports/` (2026-08-16 ×2, 08-17 ×9, 08-18 ×16) |
| ledgers read | `experimental-records.md`, `07-outputs/submissions.md`, `bets.yaml`, `00-system/handoff.md`, `failed-experiments.md` |
| still awaited | `public_code_teardown_2026-08-19`, `architecture_training_2026-08-19`, `biology_priors_2026-08-19`, `error_atlas_2026-08-19`, and the `p6_control_degraded` LB result |
| new measurements made by this analyst | 7 (§6) — all CPU, no GPU, no kernel, no slot, no commit |

Evidence tags used throughout: **[M]** measured (by whom, with the artifact) · **[C]** verified at
`file:line` · **[D]** documented external source · **[I]** inference/arithmetic ·
**[A]** asserted, no stated basis.

---

## 1. TOP 3 CONVERGENCES, ranked by strength of independent support

### C-1 — The column softmax destroys the only calibrated quantity the edge head produces, and the one artifact in the field that works uses exactly that destroyed quantity

**Rank 1: five independent evidence bases, three of them measured on different substrates, plus
one external artifact that was decoded byte-by-byte. This is the strongest signal in the corpus.**

The seed convergence in the mandate is real, but it is stated one level too shallow. The four
findings are not four symptoms of "the edge head is barely used". They are four symptoms of a
single arithmetic fact:

> **A bare column softmax is shift-invariant, so the absolute level of `edge_prob` is unconstrained
> by the loss; the only quantity a column softmax *does* determine is the within-column ordering
> and the log-odds MARGIN between candidates. We consume the share and throw away the margin.**

The supporting lines, each from a different evidence base:

| # | finding | source | tag |
|---|---|---|---|
| 1 | `compute_loss` normalises with a bare column softmax (`train_unet_transformer.py:55-72`, `:63`); shifting a whole column ±5 changes the loss by **0.000e+00**; with Trackastra's `1+Σexp` background term the same shift changes it by 0.0359 | `edge_loss_structure_2026-08-18.md` §0.1, self-test T2a/T2b | **[C]+[M]** |
| 2 | Turning the learned probability **off entirely** costs the deployed pipeline **1 reachable GT edge in 9,020** (8,263 vs 8,264); quadrupling its weight buys six. At bonus 0 the geometry linker already agrees with the head on 164,247/172,344 = **95.3%** of edges | `edge_loss_structure_2026-08-18.md` §0.2, verified port `sym_diff = 0` vs vendored `motion_relink_edges` | **[M]** |
| 3 | Every exported `edge_prob` lies in **[0.500, 1.000]** — the floor is the 0.5 candidate threshold, not model confidence. Best TP/FP **AUC 0.701** (better than motion residual 0.641 and raw distance 0.630) but **worst operating point, 49.6%** deletion precision against a 59.0% break-even | `edge_loss_structure_2026-08-18.md` §1.1, §3.3 | **[M]** |
| 4 | The 0.918-tier team's shipped `edge_prune_hgb.npz` decodes to a 13-feature GBM in which **`logitdiff` dominates** (P(keep) 0.957 → 0.181 over one unit) and `dist_um` is nearly inert; `outdeg_a`/`indeg_b` receive **0 splits across 162 trees** | `tier_reverse_engineering_2026-08-18.md` §1.2 — decoded and the predictor reimplemented locally | **[M]** |
| 5 | Independently, `xiaoleilian` (rank 86, LB **0.918**, weights public) builds its Hungarian cost from a **"appearance (logit-difference) cost"** | `live_surface_2026-08-18.md` §L-2, notebook read | **[C]** |

**Why this is one mechanism, not four.** `p_top1 = exp(l_1)/Σ_j exp(l_j)` depends on the *whole
column*, i.e. on the token count, which varies (measured mean 301.6 detections/frame, p95 493,
max 537 — `edge_loss_structure` §1.3 **[M]**) and which differs systematically between training
(threshold on raw logits at 0.3 ≈ sigmoid 0.574) and inference (sigmoid > 0.96875) — a train/deploy
mismatch flagged at `edge_loss_structure` §1.3 **[C]** and nowhere else in the corpus. The margin
`l_1 − l_2 = log(p_1/p_2)` is invariant to both the shift and the column length. **"Good AUC, bad
operating point" is exactly the signature of using a column-length-contaminated share where a
column-length-invariant margin is the well-posed quantity.** The rival's working pruner uses the
margin. Our abstention sweep used the share. Same head, same information, different coordinate.

**The margin is not recoverable from any artifact we hold — I checked [M, §6.5].** In the pre-ILP
fold-1 candidate export (`c:/temp/preilp_f1_v2/preilp_split1.parquet`, 2,162,040 candidate edges)
the **in-degree histogram is `{1: 2,162,040}`** — every target column has exactly one candidate,
because `probs[i,j] > 0.5` on a column that sums to 1 makes in-degree ≤ 1 a theorem
(`linker_division_capacity_2026-08-18.md` §1.1 **[C]**). The second-best parent's probability is
thresholded away before anything is written to disk.

**Consequence — the single highest-value zero-GPU zero-slot action in the corpus.**
`edge_loss_structure_2026-08-18.md` §7 Experiment 0 lists "the *margin* between the head's top-1
and top-2 parents per column (available from the same forward pass but not currently exported)" as
one of several composite signals to try. On this synthesis it is not one option among several; it
is **the** option, it is the only quantity the loss actually constrains, and the field's only
working edge pruner is built on it. The export is the same one-line kernel edit as the pre-ILP
export that already shipped (`~+2% wall-clock, no extra GPU` — `linker_division_capacity` §6.1),
and it re-runs `edge_loss_structure` §3.3's deletion sweep on CPU against a hard, pre-registered
bar (**59.0% deletion precision on 6bba / 53.1% on 44b6**).

**What would falsify C-1.** Export top-2 column probabilities, recompute the §3.3 sweep on
`log(p_1/p_2)` and on `p_1 − p_2`; if neither reaches 59.0% at ≥2% deletion, the abstention lane
genuinely requires a retrain and E1+E2 (`h1r_edge_loss_patch.py`) become the only route.

**Standing caveat that survives either way.** `edge_loss_structure` §5.1 is right that loss
structure and inference assignment are **not separable**: E7 (`wrapper.py:1146-1157` builds
`learned_edge_probs` only from edges the greedy pre-selector already emitted) means the linker sees
a learned score on ~1 candidate pair per source out of ~300 in gate. A margin exported but not
consumed is worth zero.

---

### C-2 — Geometry cannot discriminate anything the model did not encode, and this is now measured five separate ways

**Rank 2: five measurements, four different scripts, three different substrates, zero
counter-evidence anywhere in the corpus.**

| # | finding | source | tag |
|---|---|---|---|
| 1 | Relaxing the division proposer's gates opens **~2,400–3,300 false candidates per true division**; the tightest-first ranker puts true (wide, 7.4–8.9 µm) divisions near last | `division_lane_2026-08-18.md`, `div_proposal_funnel.py` (counts, instrument-free) | **[M]** |
| 2 | Full 199-crop CPU replay of `add_safe_divisions_postlink` at **seven** alternative gate settings: every setting raises TP and FP by the same factor, precision pinned near **0.03%**, best Δscore **≤ +0.0004** — an order inside the ±0.003 fold-0 floor | `linker_division_capacity_2026-08-18.md` §5.4, `safe_div_gate_replay.py` | **[M]** |
| 3 | The earlier learned division pair-ranker **tied frozen geometry**: 0/100 true dividers in the top-100 mothers on both families; on 44b6 the MLP's correct set is a **strict subset** of geometry's (b=0), McNemar p ≥ 0.39 | `redteam_blindspots_2026-08-17.md` Claim 1 | **[M]** |
| 4 | The rival's shipped pruner finds **`dist_um` nearly inert** — a team at 0.918 independently discovered that raw distance carries almost no TP/FP information | `tier_reverse_engineering_2026-08-18.md` §1.2 | **[M]** |
| 5 | In our own abstention sweep raw displacement is the **worst** of three signals (AUC 0.630) and the best geometric operating point (motion residual, 57.3%) still loses score (ΔJ = −0.0002) | `edge_loss_structure_2026-08-18.md` §3.3 | **[M]** |

**The mechanism, which only the cross-view gives.** `linker_division_capacity` §5.1 shows GT
divisions *are* geometrically well separated from ordinary motion (parent→farther-daughter median
**7.13 µm** vs continuation median **1.82 µm**) and a band gate cuts continuation background
**11.7×**. So why does re-gating fail? §5.4 answers it: **the proposer draws from orphan targets,
not continuation targets, and orphans are precisely the anomalous detections whose geometry
relative to a nearby mother is division-shaped by accident.** Geometry separates divisions from
*motion*; it cannot separate divisions from *detector noise*. That distinction is stated once, in
one report, and it is the load-bearing one — it generalises immediately to every hand-crafted
feature we might try, and it is why C-1 (a learned, appearance-derived margin) is the only lane
left.

**Confirmed-not-contradicted by the one apparent counter-example.** `live_surface` §L-3 reports
xiaoleilian's division patch measuring TP/FP/FN = **6/31/8** on their VAL-24, and reads this as
"division recall is demonstrably recoverable". That is **16.2% precision** — the report does not
compute it. Against a division-Jaccard denominator this is a recall mechanism paying 31 FPs for 6
TPs, exactly the regime `linker_division_capacity` §5.4 measured as worth ≤ +0.0004. It corroborates
C-2; it does not challenge it.

---

### C-3 — The leader's "+0.03–0.05 model-agnostic plugin" claim and our own measured headroom point at the SAME layer, and the numbers are compatible, not in conflict

**Rank 3: the two numbers agree to within our own error bars, from completely disjoint evidence
bases. But one side is marketing, so this is a directional convergence, not a quantitative one.**

The mandate frames these as a tension. They are not.

| side | claim | tag |
|---|---|---|
| rival | TWEAK (rank 1, 0.951, 170 subs), forum #735352, 2026-08-15: an association/linking layer over **frozen public detections**, model-agnostic, worth **+0.030–0.050**; "a single public model reached 0.940 untuned" | **[D]** verbatim quote, `tier_reverse_engineering_2026-08-18.md` §3; classified **MARKETING / unreproducible** by `proprietary_incumbents_2026-08-17.md` §6 and `competitive_refresh_2026-08-17.md` §2 |
| us | On already-detected nodes, deleting every false-positive edge is worth **+0.0617 (44b6) / +0.0920 (6bba)**; linking every reachable-but-unlinked edge is worth +0.0659 / +0.1000. Realistic abstention band **+0.010 to +0.040 on 6bba** | **[M]** `edge_loss_structure_2026-08-18.md` §3, §3.3 |

**+0.030–0.050 sits squarely inside our own measured +0.010…+0.092 window on exactly the layer the
rival names.** Two parties with no shared evidence converged on "the association layer over a
frozen detection surface is where 0.03–0.05 lives". That is the strongest strategic corroboration
in the corpus and it directly contradicts the corpus's own dominant planning assumption.

**And that is the contradiction worth naming.** `endgame_strategy_2026-08-18.md` §2.2/§4.2 hard-codes
a slot-eligibility gate requiring a lever that "changes the detection surface, the trained weights,
or the division base rate", on the reasoning that post-hoc surgery over a frozen surface is 0-for-2
on the LB. `h1_execution_spec_2026-08-18.md` §6.2 makes the same argument and honestly labels it
"a post-hoc story fitted to four failures… we have no measured instance of a weights change moving
our LB score". **The rival at 0.951 says the opposite of the gate, and our own §3 decomposition
agrees with the rival.** The gate should be struck: it would have pruned the abstention main line,
which is the largest measured number we hold.

**Also note the asymmetry the corpus keeps missing.** "Frozen detection surface" is 0-for-2 on the
LB — but both of those two levers (arm B, divfix) were **permutations** of the edge set, not
**deletions** from it. No deletion lever has ever been submitted. The taxonomy in
`bet_consolidation_2026-08-18.md` class A ("does it change the candidate set, or only permute it?")
already draws exactly this line, and the abstention lane falls on the *right* side of it. The
handoff's own note — "the boundary is 'over a frozen detection surface', NOT 'downstream of the
model'" — is the correct reading and is being over-applied downstream.

---

## 2. LIVE CONTRADICTION LIST

Ordered by how much downstream work rests on the wrong side.

### OPEN — needs adjudication

| # | contradiction | verdict and why |
|---|---|---|
| **X-1** | `edge_training_frontier_2026-08-18.md` E2 claims the mask + column softmax means "every column in that row — including unmatched-FP columns — **is pushed to 0 through the softmax**" **[M, self-labelled]**. `edge_loss_structure_2026-08-18.md` §1.1 measures the opposite: a 64×4 block of logits all at −12 still gives **column sums of exactly 1.000000** **[M, self-test T1a]**. | **`edge_loss_structure` is right; E2 is arithmetically impossible.** A softmax over `dim=0` normalises *down each column*; the entries of a column sum to 1 and cannot all be driven to 0. **This is not a framing dispute — the report's stated FP-suppression mechanism does not exist.** Everything downstream inherits it: E-3/§1.3 banks Trackastra's largest single ablation win (AOGM 136→23, parental softmax) as "already ours" when we hold the *shape* without the `1+Σ` background term that is the actual mechanism; R4's node-dropout rationale ("a dropped true parent teaches 'no parent'") teaches nothing under a bare column softmax. **~30 GPU-h of proposed ablation programme rests on this.** |
| **X-2** | `edge_training_frontier_2026-08-18.md` §3.2 recommends *lowering* training `det_threshold` 0.3 → 0.1 for "negative-richness". `edge_loss_structure_2026-08-18.md` §3 measures the gap as **59% false-positive on 6bba / 76% on 44b6**. | **`edge_loss_structure` is right and §3.2 is the wrong sign.** Enriching FP tokens optimises against the error mode that is *not* dominant. Also note the interaction with C-1: changing the training token count under a shift-invariant loss changes nothing about the probability scale — it only widens the train/deploy column-length mismatch. |
| **X-3** | `live_surface_2026-08-18.md` §5(1): "our sub-voxel refine lane is **corroborated again** — xiaoleilian's `_refine` … and 4 µm physical NMS are **load-bearing** in the only surviving public 0.918" **[A]**. Against: paired deployment-substrate LOEO measured **−0.0004 / −0.0009 bilaterally** (`bets.yaml: bet-subvoxel-refine`, closed) **[M]**, and `h1_execution_spec` §6.4 gives the mechanism (RMS radial 1.075 µm is already inside the scorer's free zone; the cliff starts ~1.5–2 µm). | **The kill stands.** Presence-in-a-good-pipeline is not evidence of contribution; no ablation isolates `_refine` in that notebook. "Corroborated" and "load-bearing" are unsupported. **This is the third instance of the same reasoning error in the corpus** (see X-6) and it should be named as a class: *co-occurrence in a rival artifact is not an effect size.* |
| **X-4** | `cross_embryo_gap_2026-08-18.md` §5.1 asserts that on the LOEO **strict** arm 44b6 node recall is **0.8888**, attributing 0.9871 entirely to secondary + DeepCenter being on. `operating_point_2026-08-18.md` §0.1 reports **node_recall 0.9842** over 71 44b6 crops on `loeo_split0_strict.csv.gz`. | **Unresolved, and it is load-bearing.** `cross_embryo_gap`'s entire memorisation-artefact inference (and therefore the "domain shift is falsified / 2.5 pp not 11.8 pp" headline now in the handoff) rests on the 0.8888 figure. The two are probably measuring different things — `cross_embryo` uses `match_authority = pregraph` at 7 µm, `operating_point` uses the scorer's own `_match_single_frame` (`metrics.py:276`) — but nobody has reconciled them. **Cheapest fix: re-run `node_recall` from `metrics.py` on `c:/temp/subvoxel_f0/loeo_split0_strict.csv.gz` and report both authorities side by side. Zero GPU, minutes.** Until then the handoff's "domain shift is falsified" line is provisional. |
| **X-5** | `endgame_strategy_2026-08-18.md` §2.1/§4.2 retains bilateral LOEO as a **kill** gate ("bilateral LOEO non-negative" is a slot requirement), on the [REASONED] asymmetry that a broken promote-gate can still reject. `instrument_repair_2026-08-18.md` §5d states flatly: **"No LOEO delta may promote anything again"**, and demotes LOEO to a build-correctness smoke. | **`instrument_repair` is right and `endgame`'s asymmetry is unearned.** If LOEO measures a different pipeline (`loeo_lb_gap` §4: the deployed secondary saw all 199 crops, so LOEO *must* ablate it), then a lever that is genuinely +LB and flat-or-negative on LOEO gets killed by the gate. `endgame` self-flags this as "the load-bearing assumption of the whole plan" and never validates it. **The asymmetry holds only for kills based on a *necessary condition failing in code*, which is a different instrument** — and the handoff already records the right rule: "code-level reasoning is 1/1 for kills, 0/2 for opportunities". LOEO is not code-level reasoning. |
| **X-6** | `foundation_models_2026-08-17.md` §7.2 ranks a Spotiflow-style subpixel head "HIGHEST EV in this report" **[I]**; `proprietary_incumbents_2026-08-17.md` #4 sizes an anisotropic z-fit lever from the σ≈2 µm cliff **[D+I]**. Both refuted by `retrain_recipes_2026-08-17.md` F7 **[C+M]**: residual is 1.075 µm RMS with **z error exactly zero** (loader is pure striding, `predict_unet_transformer.py:215`; `downsample=(1,4,4)`). | **F7's *measurement* wins; F7's *conclusion* was itself partly withdrawn** (the "0.609 µm bias" claim, handoff 2026-08-18). The surviving fact is that the residual sits inside the metric's free zone, which kills both. Recorded so nobody re-derives the lever from the vendor-default side a third time. |
| **X-7** | `instrument_repair_2026-08-18.md` §1b headline: excluding the two arm-B configs, the placeholder-4 substrate is **anti-informative, Pearson −0.199** (n=4). | **Not robust.** Adding one anchor — P3+divfix, which I scored today at **0.8978** local vs 0.915 LB (§6.1) — moves that same statistic to **Pearson +0.319** (n=5), and the full-set Spearman from +0.500 (p=0.333, n=6) to **+0.524 (p=0.227, n=7)** **[M, §6.7]**. The honest statement is not "anti-informative"; it is **"n is too small to establish anything, in either direction"**. The operational conclusion (retire the substrate for selection) is unchanged, because MDE ±0.0354 alone settles it — but the *reason* stated in the handoff should be the power analysis, not the sign of a Pearson coefficient that flips on one point. |

### RESOLVED — kept so they are not relitigated

| # | contradiction | resolution |
|---|---|---|
| R-1 | `tier_reverse_engineering` "+0.006 from deleting safe divisions" vs `division_lane` **−0.0007 measured** | division_lane. Every fork comes from the patch, so its TPs die with it. *(host, 2026-08-18)* |
| R-2 | `retrain_recipes` F7 "0.609 µm systematic bias" | Withdrawn; loader is pure striding, so `coords *= ds_arr` is unbiased. *(host, 2026-08-18)* |
| R-3 | 304 vs **151** GT divisions | 151 (125 6bba / 26 44b6), read directly from `.geff` metadata — `metric_forensics_2026-08-17.md` §0; `competitive_frontier`'s 304 was **[A]** |
| R-4 | "A duplicate detection halves that cell's Jaccard / NMS quality outranks everything" (`competitive_refresh` CW2, via a public source) | Refuted by `metric_forensics_2026-08-17.md` §1.3 **[M]**: true only when the duplicate's edges touch matched nodes; an isolated duplicate costs only node budget. *The penalised object is the spurious **link**, not the spurious detection.* |
| R-5 | zh001r geometry 1.677 µm / "detector half only, no track identity" | Both overturned by `zebrahub_data_engineering_2026-08-18.md` **[M]**: exactly **1.625 µm**, identity **100.000%** recovered, 1,258,182 GT association edges |
| R-6 | `kkunizaw/biohub-zmnscrops` as the Zebrahub unblock (`kaggle_winners_playbook` lever 2) | It contains **no labels of any kind** — `zebrahub_data_engineering` §4 **[M]** |
| R-7 | Trackastra licence Apache-2.0 (`edge_training_frontier` §1/§8, "verified from the repo LICENSE file") | **BSD-3-Clause** — `scientific_incumbents_2026-08-17.md` §8, GitHub licence API. The "verified" claim is wrong. |
| R-8 | `tier_reverse_engineering` §4: recall is worth far more than precision, our thresholds are too high | Measured dead: τ→0.01 on deployed weights buys **+0.0010 reach for 2.671× nodes** on 6bba, and 44b6 reach is **flat across every τ from 0.99 to 0.01** (`cross_embryo_gap` §4.2 **[M]**); forcing `N_pred = N_est` costs **−0.018 / −0.005** (`operating_point` §3.3 **[M]**) |

---

## 3. THE MEASUREMENT SPINE — the three-way tie, re-examined

**Headline: on the evidence now available, there is no residual anomaly requiring a broken
measurement chain. The corpus has over-read the tie. Run the control anyway — it is one slot and
the downside of being wrong is enormous — but the prior should be strongly "the chain is fine".**

Three arguments, in order of force. Each dissolves one leg of the "three materially different
graphs, identical score" claim.

### 3.1 Leg three (divfix) was never a 45× prediction. It was a conditional prediction whose condition the same ledger entry measured as failing, before the score landed.

The `+0.0227` in the handoff and in `submissions.md` is a **fold-1 division-term** figure: `L1 +
patch-OFF` from `linker_division_capacity_2026-08-18.md` §5.3 / `h1_execution_spec` §L1. It requires
`divJ` to reach ≈0.227 on the scored population. `divJ = TP/(TP+FP+FN)`, and the same ledger entry
recorded, **before the score**, that we were emitting **703 divisions across 4 crops ≈ 176/crop
against an estimated true ~27/crop — roughly 6×**.

I scored the divfix artifact locally with the official scorer today **[M, §6.1]**:

```
divfix on the placeholder-4:  adj_edge_jaccard 0.8911 · divJ 0.0667 (TP=1 FP=12 FN=2) · SCORE 0.8978
P3     on the placeholder-4:  adj_edge_jaccard 0.8907 · divJ 0.0000 (TP=0 FP= 8 FN=3) · SCORE 0.8907
```

Achieved `divJ` is **0.0667, not 0.227** — a factor of 3.4 short, and it is 3 GT division events on
one crop. Of the total local delta (+0.0071), **+0.0067 is the division term and +0.0004 is the
edge term**. So the honest pre-registered expectation, using the pipeline's own measured precision
rather than the ceiling arithmetic, was **+0.007, not +0.023** — and even that rests on a 3-event
sample. **`submissions.md`'s "roughly 45× the rounding threshold" should read "roughly 7× the
rounding threshold, at n=3 division events, on a substrate whose MDE is ±0.0354".**

I also measured the divfix graph's actual departure from P3 **[M, §6.2]**:

| crop | edge churn vs P3 | node churn | edge-mass share of the visible set |
|---|---:|---:|---:|
| `44b6_0113de3b` | 1.34% | 0.26% | 2.28% |
| `44b6_0b24845f` | **8.10%** | 2.33% | 2.28% |
| `6bba_05b6850b` | 0.97% | 0.36% | 37.81% |
| `6bba_05db0fb1` | **2.77%** | 0.62% | **57.64%** |
| pooled | **3.22%** | 0.81% | — |

**divfix churns less than half of what arm B churned (7.19%), and its largest churn (8.10%) lands
on a crop carrying 2.28% of the edge mass.** Calling P3 / P3+armB / P3+divfix "three materially
different graphs" overstates the third one by a factor of ~2 in the only currency the metric reads.

### 3.2 Leg two (arm B) is a delta measured inside its own instrument's noise floor.

`loeo_lb_gap_2026-08-18.md` §3 Read 3 calls it decisive: "two submissions the LB scores identically
differ by **+0.0365** when scored locally … a **36× discrepancy** against the LB quantum."

`instrument_repair_2026-08-18.md` §3a, three hours later, measured that same substrate's noise
floor: **placeholder-4 crop bootstrap, 20,000 resamples, h₉₅ = 0.0248, MDE @80% power = ±0.0354.**

> **+0.0365 against an MDE of ±0.0354 is not a 36× discrepancy. It is one marginal observation on
> an instrument that cannot resolve it.** The two reports are by the same team on the same day and
> nobody put the two numbers next to each other.

The structural reason is visible in the per-crop table above: **one crop (`6bba_05db0fb1`) carries
57.6% of the visible edge mass and two crops carry 95.4%**; total annotated edge mass over the four
movies is **2,285**. Arm B's local gain is essentially the movement of one crop's Jaccard.

### 3.3 The leaderboard demonstrably responds to us. It has responded four times.

| ref | config | LB | Δ | local, placeholder-4 |
|---|---|---:|---:|---:|
| — | P0-CR | **0.906** | — | 0.890430 |
| — | P0-A | **0.913** | +0.007 | 0.890444 |
| `55136908` | P0-B | **0.914** | +0.001 | 0.889225 |
| `55181562` | arm B solo | 0.914 | +0.000 | 0.927784 |
| `55274582` | P3 harmonic | **0.915** | +0.001 | 0.890692 |
| `55585140` | P3 + arm B | 0.915 | +0.000 | 0.927230 |
| `55616685` | P3 + divfix | 0.915 | +0.000 | **0.897800** ← measured today |

The LB moved **7 quanta** between P0-CR and P0-A and **1 quantum** twice after that. Any
"measurement chain is broken" hypothesis must explain why the chain worked through submission 11
and silently failed from submission 12 onward, with no build change between them — `p3_armb` is the
same factory, same spec family, and its artifact hash and gate census were both fetched and verified
(`loeo_lb_gap` §1a). **There is no such story on offer in the corpus.**

Note also the pattern in that table: **the three LB movements all coincide with a change in the
node set or the base pipeline; the three flat results are all edge/division permutations at a
near-fixed node set.** That is consistent with C-3's refinement — permutation levers are cheap to
the metric, deletion and detection levers are not — and it does *not* require a chain fault. It is,
however, the observation the `endgame` "change the detection surface" gate was reaching for; the
gate is right about the *evidence* and wrong about the *inference* (§C-3).

### 3.4 What actually needs explaining, and it is not the LB

The one genuinely significant, un-transferred measurement in the record is the **LOEO** arm-B delta:
**+0.0144 (44b6) / +0.0090 (6bba)** against LOEO's own measured MDE of **±0.0043 / ±0.0057**
(`instrument_repair` §3a). That *is* above its noise floor and it produced +0.000. And
`loeo_lb_gap_2026-08-18.md` §4 already gives the sufficient explanation without any chain fault:

> **LOEO strict must ablate the secondary model (trained on all 199 crops, verified from
> `split_manifest.json`: `train` length 199, all four placeholder stems named, `|train ∩ test| = 40`)
> and DeepCenter. So LOEO measures a pipeline the leaderboard never runs.** `instrument_repair` §2d
> adds that the contamination is layered by family in *opposite* directions, so no labelled subset
> is clean. **[M]**

That is the whole finding. It is sufficient. It does not need the tie.

### 3.5 → THE SURVIVAL LIST: which conclusions die if the control comes back flat

Maintained per the mandate. If `p6_control_degraded` (`BIOHUB_DET_THRESHOLD` 0.96875 → 0.999)
returns 0.915 or barely moves:

**Dies immediately (LB-only evidence, no independent corroboration):**
- `bet-motion-gate` CLOSED — the *only* evidence for the kill is `55585140` = 0.915. LOEO said +0.0144/+0.0090; local-deployed said +0.0365. **Everything else points the other way.** This is the single most exposed conclusion in the corpus.
- `bet-ilp-division-economics` PARKED-toward-closed — same, though §3.1 above gives it an independent non-LB reason to be unpromising (achieved divJ 0.0667, over-division 6×).
- "The plateau is real / our levers are genuinely sub-0.0005 on the hidden set" — the entire premise of `endgame_strategy`'s Phase 1.
- `instrument_repair` §5d's whole protocol (**"the leaderboard is the instrument"**, 120 lever tests, ≥+0.002 adopted). If the LB cannot resolve us, we have **no instrument at all** and the correct move is `instrument_repair` §5a (drop the contaminated components from the *shipped* pipeline, pay real score, regain a clean fold-0 at MDE ±0.0043).
- The 0.001-resolution assumption behind every pre-registered band in `submissions.md`.

**Survives regardless (measured by counts or code, not by any score):**
- **Everything in C-1.** The shift-invariance is a numerical identity (T2a); the 1-edge-in-9,020 ablation is a CPU replay validated `sym_diff = 0`; the [0.500, 1.000] `edge_prob` range is read off exported parquet; the 59.0%/53.1% break-even is arithmetic on measured TP/FP counts.
- **Everything in C-2.** Counts, not scores, by explicit design (`division_lane`, `linker_division_capacity` §0).
- The ILP division proof (`ilp_division_probe.py`, local `tracksdata`/`ilpy`, 0 divisions at p=0.99 under deployed weights) **[M]**.
- The contamination map (`instrument_repair` §2a–2d) — read from artifact manifests.
- 151 GT divisions; annotation density 0.771% / 5.371% / 2.821%; the closed-form metric (`metric_forensics` §1, verified to 1e-12).
- Detection-threshold refutation: 0/28,800 subthreshold local maxima within 7 µm of GT on 44b6 is label-free and score-free.
- The Zebrahub sidecar (1,258,182 GT edges at exactly 1.625 µm, 0/1440 mismatches).

**Becomes undecidable (was already weak, would lose its last support):**
- Any statement of the form "lever X is worth +0.00N" — every instrument would be uncalibrated.
- The public/private shakeup arithmetic in `endgame_strategy` §5.

### 3.6 Two things to check about the control BEFORE reading its result

1. **The adjustment term partially compensates node loss, and nobody has said so.**
   `J_adj = max(0, J·(1 − 0.1·(N_pred − N_est)/N_est))` (`metrics.py:440-448` **[C]**) has **no upper
   clamp**. We already under-produce: measured `N_est` over the four placeholder crops is
   **134,712** against P3's 122,083 nodes **[M, §6.3]**, and one crop (`44b6_0b24845f`) runs at
   `total_node_ratio = −0.393`, giving it `adj_edge_jaccard = 1.0179 > 1` under P0-CR **[M, §6.4]**.
   Raising `det_threshold` to 0.999 drives every ratio sharply negative and **raises** the multiplier
   even as `J` collapses. The pre-registered "≤0.90, most likely 0.80–0.89" should hold anyway
   (node-recall collapse dominates), but **an outcome in 0.905–0.914 is an ambiguous band, not a
   clean "sound"**, and the interpretation table in `submissions.md` does not have a row for it.
   **Pre-commit to reading 0.905–0.914 as "chain responds, magnitude smaller than expected" and
   re-deriving from the run's `num_pred_nodes`, not as a pass.**

2. **The reference scorer silently drops datasets, it does not zero them.**
   `vendor/kaggle-cell-tracking/scripts/evaluate.py:73-76` **[C, verified by me]**:
   `names = sorted(pred_names & gt_names)` — only datasets present in **both** are scored — plus a
   bare `except Exception: skipped.append(name); continue` on any unreadable geff. And
   `metrics.summarise` skips rows with NaN `edge_tp` rather than scoring them zero
   (`metrics.py:483`). **If the hidden rerun's population is larger than the visible four and our
   kernel drops or fails a crop, that crop leaves the average rather than penalising it.** This is
   a live, unexamined route by which the scored population could be both smaller and more
   concentrated than assumed — and concentration is exactly what makes a 3% churn invisible. It is
   also the only mechanism I can find that *would* produce a genuinely insensitive score. **Check
   the control kernel's `run_stats.csv` row count and dataset names against the four expected
   stems before interpreting anything.**

---

## 4. CLAIM PROVENANCE LEDGER — load-bearing claims still at ASSERTED or INFERENCE

The host has twice caught agents whose arithmetic was right and whose premise was false. These are
the remaining premises that carry planning weight and have not been verified.

| # | claim | where it is load-bearing | status | cheapest verification |
|---|---|---|---|---|
| **P-1** | "A model-agnostic association plugin over frozen detections is worth +0.030–0.050" | The whole strategic case for the abstention lane (C-3); `tier_reverse_engineering` §3's entire tier ladder | **[D] but the document is a competitor's marketing post** (forum #735352). Classified MARKETING by two reports. **Effectively ASSERTED for our stack.** | Not verifiable externally. But it is *dimensionally corroborated* by our own [M] +0.062/+0.092 FP-deletion bound. Treat as a direction, never as a number. |
| **P-2** | "The primary model (`split_0`) was trained on 6bba with 44b6 held out" | Every claim that LOEO fold 0 is clean; the 2.5 pp vs 11.8 pp cross-embryo conclusion; the fallback in `instrument_repair` §5a | **ASSERTED.** `instrument_repair` §2c verified the artifact ships **no `training_config.json`, no `split_manifest.json`, no `history.csv`** — the only evidence is the directory name `split_0`. | Unverifiable from the artifact. **Record it as an assumption in `research/04-data/` and stop quoting it as fact** (`instrument_repair` §6.3 already asks for this; it has not been done). |
| **P-3** | "The four `data/test` movies are placeholders and the hidden test is a different, unobservable population" | `loeo_lb_gap` §3 Read 3; the retirement of the placeholder substrate; the entire "population mismatch" hypothesis at ~0.45 residual probability | **[M] for byte-identity** (102 files, identical sha256 per stem). **[I] for "the hidden set is a different population."** The alternative — same movies, *denser* organiser GT — is never considered anywhere in the corpus and would explain the local↔LB level gap (0.8907 vs 0.915) and the flat deltas together. | The degraded control discriminates: under a denser-GT-same-movies hypothesis the control still drops hard, so it does not settle it. **What settles it: the ratio of our local score to LB across the seven anchors.** If a single affine map fits all seven within ±0.001, the populations are related; if not, they are not. n=7 is now available (§3.3 table) — **30 minutes, zero cost, and it has not been done.** |
| **P-4** | "703 divisions / 4 crops ≈ 176/crop against a true ~27/crop" | The over-division explanation for divfix; the parked `DIVISION_WEIGHT=0.75` follow-up | **[I]**, from 151 GT divisions / 199 crops ÷ 2.8% annotation. | Now partly **[M]** by my local scoring: on the annotated subset divfix reads TP=1 / FP=12 across 4 crops, so its FP rate *in annotated territory* is ~3/crop, not 176. The 6× claim is about total forks, not scored forks. **The two must not be conflated.** Restate as: total forks are ~6× the true rate; scored division FPs are 12. |
| **P-5** | "The +0.0227 fold-1 division gain" | Drove one submission slot | **[I] on a counterfactual.** Now superseded by the measured **+0.0067** achieved division term (§3.1). | Done. Retire the number. |
| **P-6** | "Turning off the safe-division patch removes 507 FPs that dominate the denominator" | The L1 + patch-OFF coupling argument | **[I]** from fold-1 counts on the leaky arm. Contradicted at the deployed scale: P3's placeholder-4 division FP is **8**, not 507, and divfix's is **12** **[M]**. The 507 is a fold-1-LOEO-substrate number. | Already resolved by the local scoring; note the substrate every time the number is quoted. |
| **P-7** | "Levers over a frozen detection surface are 0-for-2 on the LB, therefore a lever must change the detection surface" | `endgame_strategy` §4.2 gate 2 — a hard slot-eligibility rule | **[I], and the converse of a 2-observation pattern.** `h1_execution_spec` §6.2 says so explicitly; `endgame` drops the caveat. Both flat levers were **permutations**; no **deletion** lever has ever been submitted. | **Strike the gate.** Replace with `bet_consolidation`'s class-A test as written ("does it change the candidate set, or only permute it?"), which admits the abstention lane. |
| **P-8** | "The public/private split is random over crops, so shakeup ≈ 0.002 over 40 decisions" | `instrument_repair` §5d failure mode 1; `endgame` §5 | **[I] with the assumption named by its own author**, and `endgame` §S-6 records **no host statement of the split was found — by-crop vs by-movie vs by-embryo is [UNVERIFIED]**. If it is by embryo, the estimate is worthless. | Ask the host / search the forum once discussions are reachable again (`live_surface` L-7: CLI, API, WebFetch and forum.image.sc all 403 — **this blocker is itself unresolved and blocks P-1 and P-8 both**). |
| **P-9** | "`edge_prob` median 0.9188 on declined second-daughter edges means the ILP is discarding divisions the model is 92% confident in" | The whole L1 case that earned a slot | **[M] for the number, [I] and now known-misleading for the reading.** The handoff already carries the correction (`edge_prob` is a share). Under C-1 the correct reading is weaker still: 0.9188 is a share over a column whose length we did not record. | Export column length alongside `edge_prob` in any future pre-ILP dump. One extra integer. |
| **P-10** | "~280 teams" (the mandate's framing) | Sets the perceived competitive density | **Wrong by an order of magnitude.** `team_count = 2,486` (Kaggle API, `instrument_repair` App. A) / **2,483** (`live_surface`, CSV 08:31Z 08-18). What is ~280 is the number of teams **tied with us at exactly 0.915** (281 on 08-18, 288 on 08-17, 302 on 08-16). | Fixed here. The distinction matters: we are rank ~229 of ~2,483, inside a 281-team plateau block. |

---

## 5. WHAT NOBODY HAS ASKED

Nine levers died from three root causes (`bet_consolidation_2026-08-18.md`): post-hoc surgery over
a frozen detection surface; hand-crafted geometry discriminating what the model never encoded;
premises false in code. Those are three *answers*. The question the whole corpus avoids is upstream
of all of them.

> ### **Every lever in 27 reports proposes to CHANGE something. Not one asks what the pipeline should DECLINE to do.**

Twenty-seven reports, ~55 levers, and the taxonomy is: add candidates, re-rank candidates, re-gate
proposals, re-weight the loss, retrain the weights, swap the detector, restore divisions the solver
declined, relink edges the gate excluded. **Every one is additive or permutative.** The corpus has
never proposed, costed, or submitted a lever whose action is *deletion*.

That is not an aesthetic observation. It is the measured shape of the problem:

- **59% of the 6bba linking gap and 76% of the 44b6 gap are false positives, not missing links**
  (`edge_loss_structure` §3, **[M]**). Deleting every FP is worth **+0.0617 / +0.0920**. Linking
  every reachable-but-unlinked edge is worth **+0.0659 / +0.1000** — nearly the same — but the FP
  bucket is the one with a *mechanism* attached.
- **Division precision is 5 TP against 613 FP across 199 crops** (`linker_division_capacity` §5.3,
  **[M]**) — a 122:1 poison ratio. Every division lever proposed has been about *proposing more*.
- **6bba's C class is 7.3% of GT (700 nodes) where an accepted peak fired within 7 µm (min prob
  0.9698) and the node never reached the matched graph** (`cross_embryo_gap` §3.3, **[M]**) — a
  deletion the pipeline is *already* performing, wrongly, and nobody has asked which stage does it.
- The metric itself is a deletion metric: an omitted edge costs one FN; an emitted wrong edge costs
  one FN **and** one FP (`metrics.py:314-317`, `edge_loss_structure` D1 **[C]**). It strictly
  rewards abstention and our objective cannot express it.
- And the field's only decoded working artifact from above the plateau is **a pruner**
  (`edge_prune_hgb.npz` — `tier_reverse_engineering` §1.2 **[M]**). Not a detector, not a linker.
  A thing that deletes.

Three sub-questions nobody has asked, each cheap:

**5.1 What is the pipeline already deleting, and is it right?** `motion_relink_edges` *replaces*
the ILP's entire edge set (`wrapper.py:1158-1163` **[C]**), running on 100% of crops and replacing
100% of learned edges (`run_stats.csv`, all 9 crops **[M]**). `OUTPUT_MIN_TRACK_LEN = 6` deletes
short tracks. `cross_embryo_gap`'s 700 C-class nodes vanish somewhere in there. **No report has
audited the deletions the deployed wrapper already performs.** That audit is a CPU replay on
artifacts we already hold, and it is the direct route to the 41%-missing half of the gap as well as
the 59% FP half.

**5.2 Why did we spend the abstention experiment on the one signal the loss provably does not
constrain?** §C-1. The sweep used `edge_prob` (a column-length-contaminated share) and geometry.
The margin — the only quantity a column softmax determines — was never exported, is unrecoverable
from every artifact we hold (in-degree histogram `{1: 2,162,040}` **[M, §6.5]**), and is the feature
the 0.918 pruner is built on.

**5.3 If we cannot measure, why are we still trying to measure?** `loeo_lb_gap` §4 proved no
configuration measures the deployed pipeline honestly. `instrument_repair` §5a costs the fix
(ablate the contaminated components from the *shipped* pipeline: 1 kernel + 1 slot + an unknown
permanent score loss) and rejects it as "not worth it as a primary strategy" — on the grounds that
the surviving primary's provenance is still undocumented (P-2) and that MDE ±0.0043 is worse than
the LB's 0.001. **That reasoning is only valid while the LB is trusted.** The control probe exists
precisely because it may not be. **Nobody has written the contingency: if the control comes back
flat, §5a is not a fallback, it is the only remaining instrument, and its cost should be priced
now rather than after the slot is spent.** Price it before the 24th.

---

## 6. NEW MEASUREMENTS MADE BY THIS ANALYST

All CPU, on artifacts already on disk. No GPU, no kernel push, no submission, no commit.

| # | measurement | result |
|---|---|---|
| **6.1** | Official patched scorer on the divfix artifact vs `data/train` (`scripts/core/score_loeo_submission.py --csv c:/temp/p5_divfix/submission.csv`) — never previously scored locally | adj_edge_jaccard **0.8911** · divJ **0.0667** (TP=1 FP=12 FN=2) · node_recall 0.9841 · **SCORE 0.8978**. vs P3 0.8907. Delta **+0.0071**, of which **+0.0067 is the division term**. Achieved divJ is 3.4× below the 0.227 the +0.0227 prediction required. |
| **6.2** | Edge/node set diff, divfix vs P3, per crop | pooled edge churn **3.22%** (vs arm B's 7.19%), node churn 0.81%. Per crop: 1.34% / **8.10%** / 0.97% / 2.77%. The 8.10% lands on a crop holding 2.28% of the edge mass. |
| **6.3** | `estimated_number_of_nodes` for the four placeholder crops, read from GT geff metadata | 25,755 / 32,795 / 6,362 / 69,800 = **134,712** total, against P3's 122,083 predicted nodes. We under-produce by ~9% pooled and by **39%** on `44b6_0b24845f`. |
| **6.4** | Per-crop edge mass `w_i = TP+FP+FN` and multiplier behaviour, from the scorer's own per-crop output | `6bba_05db0fb1` = **57.64%** of visible edge mass, `6bba_05b6850b` = 37.81%, the two 44b6 crops = 4.55% combined (total W = **2,285**). `adj_edge_jaccard` is **unclamped above 1**: P0-CR scores **1.0179** on `44b6_0b24845f` at `total_node_ratio = −0.387`. |
| **6.5** | Degree histograms of the fold-1 pre-ILP candidate graph (`c:/temp/preilp_f1_v2/preilp_split1.parquet`, 2,162,040 candidate edges) | **in-degree `{1: 2,162,040}` — exactly one candidate parent per target column, universally.** Out-degree `{1: 1,795,844, 2: 165,148, 3: 10,911, 4: 715, 5: 59, 6: 2}` → 176,835 sources with out-degree ≥2, reproducing the handoff's figure exactly. **The top-2 column margin is unrecoverable from every artifact we hold.** |
| **6.6** | Isolated-node census on the P3 and P3+armB submissions | **0 isolated nodes in all 4 crops, both arms.** Kills the "prune edgeless nodes for a free adjustment-multiplier gain" idea before anyone proposes it. Also: P3 → P3+armB changes the **node** count by +131 (122,083 → 122,214), so arm B is not a pure edge permutation. |
| **6.7** | LB↔local calibration re-run with divfix as a 7th anchor | n=7 Spearman **+0.524 (p = 0.227)**, Pearson +0.330, OLS slope +0.0605. Excluding the two arm-B configs (n=5): Pearson **+0.319** — versus `instrument_repair` §1b's **−0.199** at n=4. The "anti-informative" headline is not robust to one point. |
| **6.8** | Code check, `vendor/kaggle-cell-tracking/scripts/evaluate.py:73-76` | `names = sorted(pred_names & gt_names)` — datasets missing from the prediction are **silently skipped, not scored zero**; a bare `except Exception` also drops unreadable crops; `metrics.summarise` skips NaN rows. A route to a smaller-and-more-concentrated scored population than assumed. Never examined in the corpus. |

---

## 7. WHAT THIS DOCUMENT ASKS FOR, IN ORDER

Not a plan — a list of the cheapest things that would change what this synthesis says.

1. **Export the top-2 column probability from the inference kernel** (one line beside the existing
   pre-ILP export, ~+2% wall-clock, no GPU) and re-run `edge_loss_structure` §3.3's deletion sweep
   on `log(p₁/p₂)`. Bar: **59.0% deletion precision at ≥2% deletion on 6bba**. This is C-1's
   falsification and it is the only test in the corpus that a rival has already passed.
2. **Fit an affine map from local placeholder-4 score to LB across all seven anchors** (§3.3 table,
   30 minutes, zero cost). Settles P-3 — whether the LB population is related to ours at all —
   independently of the control probe.
3. **Reconcile X-4** (44b6 node recall 0.8888 vs 0.9842) by running `metrics.node_recall` on
   `loeo_split0_strict.csv.gz`. The handoff's "domain shift is falsified" line depends on it.
4. **Pre-commit the 0.905–0.914 reading for the control probe** (§3.6.1) and check the rerun's crop
   count and stem names (§3.6.2) before interpreting the number.
5. **Strike `endgame_strategy` §4.2's "must change the detection surface" gate** (P-7) and replace
   it with `bet_consolidation`'s permutation-vs-candidate-set test, which admits the abstention lane.
6. **Correct `edge_training_frontier` X-1 in place** before any of its ~30 GPU-h programme is
   costed further.
7. **Price `instrument_repair` §5a now** as the contingency instrument, not as a rejected option
   (§5.3).

---

## Regeneration protocol

When new reports land, re-read: this file's §1–§5 headings, the four ledgers, and the new reports.
Then rewrite this file wholesale. Preserve across revisions: the RESOLVED contradiction table (§2,
lower half), the survival list (§3.5), and the measurement log (§6) — appending, never deleting,
and marking any entry that a later measurement overturns.

Constraints honoured: no experiment launched, no kernel pushed, no submission, no commit, no GPU.
Local CPU scoring of an already-fetched artifact (§6.1) and read-only artifact inspection only.
`.claude/settings.json` untouched. This file carries no YAML frontmatter, per the standing
convention for `internal-reports/`.
