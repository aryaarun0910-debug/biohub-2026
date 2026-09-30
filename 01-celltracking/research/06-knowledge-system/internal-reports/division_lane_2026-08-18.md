# Division lane — forensics, break-even, and design — 2026-08-18

Scope: (1) characterise the measured 5 TP / 613 FP / 146 FN division population on the full
deployment substrate; (2) design the division lane; (3) refresh the 2024–2026 external state of
the art; (4) deliver break-even arithmetic derived from the vendored scorer.

Labels: **[MEASURED]** = run this session against real data with the official scorer;
**[CODE]** = read off vendored source with `file:line`; **[INFERENCE]** = derived, assumptions
stated; **[UNVERIFIED]** = could not confirm, with what would settle it.

Companion: `metric_forensics_2026-08-17.md` (objective derivation — not repeated here),
`scientific_incumbents_2026-08-17.md` (§3.2 linajea cell-state classifier).

---

## 0. THE HEADLINE — read this first

**[MEASURED]** Across both LOEO folds (199 crops, official scorer, per-fork classification via
`tracking_cellmot.division_metrics.score_divisions`):

| | forks emitted | div TP | div FP | div FN | GT divisions | divJ |
|---|---|---|---|---|---|---|
| fold 0 (44b6) | 5,279 | 2 | 106 | 24 | 26 | 0.0152 |
| fold 1 (6bba) | 6,303 | 3 | 507 | 122 | 125 | 0.0047 |
| **pooled** | **11,582** | **5** | **613** | **146** | **151** | **0.0065** |

Reconciles exactly with the brief's independently-measured counts, and the reachable census
(89/151) reproduces `scripts/core/score_loeo_submission.py:88-103`'s definition exactly
(21/26 + 68/125 = 89/151). Both pipelines agree, so the decomposition below is trustworthy.

### The decisive finding

**[MEASURED] Of the 146 FN divisions, ZERO have a fork emitted at the matched mother.**
`fork_at_matched_mother = 0` in *both* folds, and `edge_m_d1 & edge_m_d2` present = 0 in both.

> **The 613 FP forks and the 146 FN divisions are disjoint populations. Re-ranking, filtering,
> or re-scoring the forks we already emit recovers exactly ZERO divisions — the fork is never
> in the right place to begin with.**

This kills the framing "precision, not recall, is the failure" as a *route to points*. Precision
is indeed where the FP mass is, but a pure precision filter cannot add a single TP, and the
arithmetic in §3 shows FP suppression alone caps at **+0.0027** and realistically delivers
**≈ +0.0007** at honest cross-family separation. **The binding constraint is fork PROPOSAL at
the correct mothers, not fork ranking.**

The good news is that the substrate supports it: **99/146 FN divisions have both daughters
already present as predicted nodes within 7 µm**, and **84/146 are fully metric-reachable**
(mother + both daughters matched to distinct predicted nodes). Detection is largely not the
blocker; linking topology is.

### The root cause, quantified

**[MEASURED] Only 22 of 151 GT divisions (14.6 %) can pass the deployed fork proposer's geometry
gates at all — and that is a generous upper bound.** Of the 89 metric-reachable divisions, just
**10** are admissible under the current gates. Widening the two binding gates to match the
measured GT geometry raises that to **81 of 89**.

> **We are not ranking true divisions badly. We are refusing to propose them.** The proposer's
> sister-distance gate (7.2 µm) sits *below* the GT median inter-daughter distance in **both**
> embryos (8.98 µm in 44b6, 11.47 µm in 6bba). This is one constant in one file
> (`src/biotrack/wrapper.py:124`), it is CPU-only to change, and it fully explains
> `fork_at_matched_mother = 0`.

---

## 1. LOCAL FORENSICS [MEASURED]

Method: `<scratchpad>/div_forensics.py` — for every crop in
`c:/temp/subvoxel_f{0,1}/loeo_split{0,1}_strict.csv.gz`, reconstruct the graph via
`biotrack.submission.submission_to_graphs`, run the official
`division_metrics.score_divisions(pred, gt, scale=(1.625,0.40625,0.40625), max_distance=7.0)`
for the authoritative TP/FP fork sets, and `division_metrics._match_full` for the authoritative
node matching. All distances are physical microns (voxel scale applied before the norm), matching
the scorer (`tracksdata/metrics/_matching.py:270-279`). Outputs: `forks_pooled.csv` (11,582 rows),
`gtdiv_pooled.csv` (151 rows).

### 1.1 Why each of the 146 FN divisions was missed

| FN cause | 44b6 | 6bba | pooled | interpretation |
|---|---|---|---|---|
| `reachable_one_edge_missing` | 18 | 62 | **80** | mother + both daughters matched to distinct pred nodes; **at most one of the two mother→daughter edges exists**. Pure linking failure. |
| `daughter_undetected` | 1 | 28 | **29** | a GT daughter has **no** predicted node within 7 µm. Needs detection. |
| `mother_unmatched` | 0 | 20 | **20** | the GT mother matched no predicted node. Needs detection. |
| `daughter_unmatched_detected` | 4 | 9 | **13** | a pred node exists within 7 µm of the daughter but the optimal per-frame assignment gave it to another GT cell. Contention, not absence. |
| `reachable_no_edges` | 1 | 3 | **4** | all three matched, neither mother→daughter edge present. |
| (TP) | 2 | 3 | 5 | — |

**Reachability ladder** (pooled, of 151 GT divisions):

| Tier | count | fraction | what it needs |
|---|---|---|---|
| Already TP | 5 | 3.3 % | — |
| Metric-reachable (mother + 2 distinct daughters matched) | **84** | 55.6 % | **linking only — no new detections** |
| Both daughters have a pred node ≤ 7 µm (matched or not) | 99 | 65.6 % | linking + contention resolution |
| Needs new detections (`daughter_undetected` + `mother_unmatched`) | 49 | 32.5 % | detector work / retrain |
| *of the 89 reachable (84 FN + 5 TP): admissible under current proposer gates* | *10* | *11 %* | *§3.2 — the actual bottleneck* |

**[MEASURED] 84 of 151 divisions (55.6 %) are recoverable by post-processing alone**, i.e. by
emitting the second mother→daughter edge at a mother whose daughters we already detected and
matched. Combined with the 5 existing TPs, the **post-processing ceiling is 89/151 = 0.589 divJ**.

### 1.2 Geometry of GT divisions vs the forks we actually emit

Physical µm. Median [IQR] unless noted.

| Population | n | inter-daughter dist | parent→daughter (near/far) | cos(angle) | daughter separation at t+2 |
|---|---|---|---|---|---|
| **GT divisions, 44b6** | 26 | **8.98** (p10 5.46, p90 11.87) | 4.12 / 5.50 | **−0.71** | — |
| **GT divisions, 6bba** | 125 | **11.47** (p10 6.58, p90 14.48) | 5.70 / 6.36 | **−0.76** | — |
| Emitted forks — TP | 5 | 5.69 (min 4.14, max 7.73) | 2.60 / 2.63 | **−0.88** | 9.48 |
| Emitted forks — FP | 613 | **3.72** [3.28, 4.45] | 2.03 / 2.63 | **−0.33** | **5.28** |
| Emitted forks — unscored | 10,964 | 3.85 [3.30, 5.01] | 2.07 / 2.87 | −0.32 | 5.54 |

**Three structural mismatches, all pointing the same way:**

1. **Our forks are far too tight.** Real divisions separate their daughters by **9–11.5 µm**
   (median), our emitted forks by **3.7 µm**. The deployed proposer's gate
   `SAFE_DIV_MAX_UM = 4.7` (`src/biotrack/wrapper.py:123`) caps the parent→new-daughter distance
   at 4.7 µm, while the GT median parent→daughter distance is **5.7–6.4 µm** and p90 is
   **8.4–9.9 µm**. **The proposer's primary gate excludes the median true division.** This is the
   mechanical explanation for `fork_at_matched_mother = 0`.
2. **Our forks are not angularly opposed.** True divisions push daughters apart
   (cos ≈ −0.71…−0.76, i.e. ~135–140°); our FP forks sit at cos ≈ −0.33 (~110°), consistent with
   grabbing an unrelated neighbouring track rather than a sister.
3. **Our forks do not persist.** True sisters keep diverging (TP forks reach 9.5 µm at t+2);
   FP forks stay at 5.3 µm — they are transient mis-associations.

### 1.3 Are the 613 FPs separable from the 5 TPs on existing features? [MEASURED]

Fraction of FPs killed by the loosest one-sided cut that retains **all 5** TPs (pooled):

| feature | TP range | cut | FPs killed |
|---|---|---|---|
| `cos_angle` | [−0.99, −0.51] | ≤ −0.51 | **63.7 %** |
| `nn_parent` (dist to nearest same-frame node) | [10.32, 13.41] | ≥ 10.32 | 33.0 % |
| `mother_step` | [0.81, 3.37] | ≤ 3.37 | 13.7 % |
| `d_parent_c2` | [2.07, 3.68] | ≤ 3.68 | 12.9 % |
| `inter_daughter` | [4.14, 7.73] | ≥ 4.14 | 0.3 % (wrong direction — FPs are *tighter*) |
| `daughters_next_dist` | [7.32, 11.84] | ≥ 7.32 | 0.5 % (same) |

`cos_angle` is the only feature with real one-sided power. **But this is fitted on 5 positives
(2 in 44b6, 3 in 6bba) and is therefore not honestly fittable** — any cut chosen to retain 5
points has a standard error on the retained-TP rate of roughly ±0.2. §4.1 states the only
honest fitting protocol available.

### 1.4 Frame and depth distribution

**[MEASURED]** Divisions and FP forks are both essentially uniform over t (FP median t = 39,
IQR [17, 59] of a 0–99 window; TP median 41) and over z (FP median 52 µm, TP 53.6 µm). **There is
no temporal or depth-based gating opportunity** — no early/late or shallow/deep enrichment to
exploit. This closes off the cheapest possible filter.

---

## 2. HOW divJ ENTERS THE COMPOSITE — and whether a division gain is LB-visible

This is the analytically decisive question, given the two negative transfer results in hand
(the post-processing lever at +0.009..+0.014 LOEO → +0.000 LB, and armB churning 3–10 % of edges
for an identical 0.915).

### 2.1 The formula [CODE]

```
score = adj_edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard   # metrics.py:522
SCORE_DIVISION_WEIGHT = 0.1                                           # metrics.py:34
division_jaccard = dTP / (dTP + dFP + dFN)                            # metrics.py:519-521
```

`division_jaccard` is **micro-pooled across all crops** (`metrics.py:508-521` sums
`COUNT_COLUMNS` first, then divides once), unlike `adj_edge_jaccard` which is per-sample and
weight-averaged (`metrics.py:498-506`). Confirmed at `metrics.md:131-149`.

Because `dTP + dFN = G` (total GT divisions) is a constant, the denominator is
**`D = dFP + G`** and the whole term reduces to a two-parameter function:

> **score_div = 0.1 · dTP / (dFP + G)**, with `G = 151` on our substrate.

### 2.2 Why the division term is structurally different from the edge term

The armB result — 3–10 % of edges churned, LB identical — is explained by the edge term's
**offsetting structure**: `adj_edge_jaccard ≈ 0.915` over `W ≈ 128,555` scored edge events, so
swapping one mislink for another is TP-neutral, and the numerator barely moves. Churn is free
*and worthless*.

**The division term has no offsetting mode.** We currently score `divJ = 0.0065` out of a
possible 1.0. Every FP removed strictly decreases the denominator; every TP gained strictly
increases the numerator. There is no way to churn divisions neutrally. Per-event leverage:

```
∂score/∂(1 division TP) at dFP=0  = 0.1/151      = 6.6e-4
∂score/∂(1 edge TP)               = m/W ≈ 1/128555 = 7.8e-6
ratio                                              = 85×
```

(at the current dFP=613 operating point the ratio is 0.1/764 ÷ 7.8e-6 = **16.8×**, reproducing
`metric_forensics_2026-08-17.md` §0.2 row 6's "+17 edge-TP equivalents" exactly.)

**So yes — a division gain is visible where the edge gain was not, *in kind*.** The question is
magnitude, and the honest answer is below.

### 2.3 Break-even arithmetic — how much must be recovered for +0.001 [INFERENCE from CODE]

`Δscore = 0.1 · [ dTP/(dFP+151) − 0.00654 ]`. Solving for `Δscore = +0.001`:

| Regime | dFP | dTP needed | **net new divisions needed** |
|---|---|---|---|
| No FP work | 613 | 12.6 | **+8** |
| Half the FPs killed | 307 | 7.6 | **+3** |
| All FPs killed | 0 | 2.5 | **already met by our 5 TPs** |

**Scenario table** (Δscore vs deployed 0.915):

| Scenario | dTP | dFP | divJ | **Δscore** | LB reads |
|---|---|---|---|---|---|
| Deployed today | 5 | 613 | 0.0065 | — | 0.915 |
| **FP filter only, perfect (keeps 5 TP, kills 613)** | 5 | 0 | 0.0331 | **+0.0027** | 0.918 |
| **FP filter only, honest (cos_angle cut, −63.7 %)** | 5 | 222 | 0.0134 | **+0.0007** | 0.915–0.916 |
| Suppress all forks (`OUTPUT_SAFE_DIVISIONS=0`) | 0 | ~0 | 0.000 | **−0.0007** | 0.914–0.915 |
| Perfect precision + half the reachable 84 | 47 | 0 | 0.311 | **+0.030** | 0.945 |
| **Perfect precision + all 84 reachable (post-proc ceiling)** | **89** | **0** | **0.589** | **+0.058** | **0.973** |
| Theoretical max (all 151, no FP) | 151 | 0 | 1.000 | +0.099 | 1.014 |

### 2.4 The honest verdict on visibility

1. **A pure FP filter is NOT worth doing on its own.** Its perfect-play ceiling is +0.0027, and
   the only honestly-fittable version of it (§1.3) delivers **+0.0007 — below the LB's 3-decimal
   resolution.** It would replicate the exact failure mode of the last two levers: a real but
   sub-resolution LOEO gain that reads +0.000 on the LB.
2. **Turning divisions off entirely is mildly negative** (−0.0007) — the caps are currently
   buying us 5 TPs for 613 FPs, and at divJ≈0 that trade is barely positive. This confirms
   `metric_forensics` §1.5's "quadratic FP cost" trap: at our operating point the 613 FPs cost
   almost nothing, which is precisely why no local search has ever escaped.
3. **The lane is only worth entering if it recovers divisions.** The threshold for LB visibility
   is roughly **+8 net divisions with no FP work, or +3 with the FP mass halved**. Against 84
   metric-reachable FNs — of which §3.2 shows re-gating makes **81 admissible where only 10 are
   today** — that is a **10 % conversion rate of the newly-admissible pool**: a genuinely low
   bar, and the first lever on this project whose break-even sits *below* its measured substrate
   rather than above it.
4. **Caveat on the hidden test [INFERENCE].** `G` for the hidden test is unknown; our 151/199
   crops implies ≈0.76 divisions/crop. The public LB is 29 % of the test, so public-LB division
   counts are ~1/3 of the private ones and correspondingly noisier — a +0.001-scale division
   move may not resolve cleanly on the public LB even when real on the private one. **All
   division decisions should be made on the LOEO substrate, not on public-LB feedback.**

---

## 3. THE PROPOSER IS THE BUG — root cause [MEASURED + CODE]

`add_safe_divisions_postlink` (`src/biotrack/wrapper.py:771-865`) is the only fork source in the
deployed pipeline. Its gates (`wrapper.py:123-127`) versus the measured GT geometry (§1.2):

| Gate | value | GT reality | verdict |
|---|---|---|---|
| `SAFE_DIV_MAX_UM` (parent → new daughter) | **4.7** | GT median 5.7–6.4, p90 8.4–9.9 | **excludes >50 % of true divisions outright** |
| `SAFE_DIV_SISTER_MAX_UM` (daughter ↔ daughter) | 7.2 | GT median **8.98 / 11.47**, p90 11.9 / 14.5 | **excludes >50 %, and ~90 % in 6bba** |
| `SAFE_DIV_EXISTING_CHILD_MAX_UM` | 7.8 | — | not binding |
| `SAFE_DIV_FRAME_FRAC_CAP` | 0.008 | — | secondary |
| `SAFE_DIV_GLOBAL_FRAC_CAP` | 0.004 | — | secondary |

Scoring function `score = parent_dist + 0.15 * sister_dist` (`wrapper.py:831`) is **monotonically
increasing in both distances — it ranks the tightest pairs first, i.e. exactly backwards** with
respect to the measured GT geometry, where true divisions are the *wide, opposed, persistently
diverging* ones. Combined with the caps, the proposer spends its entire budget on the tightest
candidates in the volume, which are overwhelmingly adjacent unrelated tracks.

**This is a single, mechanically-explained, CPU-only bug with a measured target geometry.** It is
also the reason `fork_at_matched_mother = 0`: the gates make it structurally impossible for a
fork to appear at a true mother whose daughters are 9–11 µm apart.

> **[MEASURED, decisive] The deployed division proposer's sister-distance gate (7.2 µm) sits
> below the GT median inter-daughter distance in both embryos (8.98 / 11.47 µm). We are not
> ranking true divisions badly — we are refusing to propose them.**

### 3.1 The gates are empirically binding [MEASURED]

Across all 11,582 emitted forks, the geometry distribution is **hard-truncated at exactly the
gate constants**, confirming that the proposer is the sole fork source and that its gates bind:

| statistic | measured | gate |
|---|---|---|
| nearer parent→daughter distance | 99.95 % ≤ 4.7 µm, max 5.92 | `SAFE_DIV_MAX_UM = 4.7` |
| farther parent→daughter distance | max **7.84** | `SAFE_DIV_EXISTING_CHILD_MAX_UM = 7.8` |
| inter-daughter distance | 99.2 % ≤ 7.2 µm (only 93 of 11,582 exceed it), p99 = 7.13 | `SAFE_DIV_SISTER_MAX_UM = 7.2` |

### 3.2 The proposer's structural ceiling [MEASURED]

Fraction of **GT** divisions whose geometry could pass the gates at all. Computed with the
*most favourable* assignment (nearer daughter treated as the proposed one), so these are
**upper bounds** on what the proposer could ever emit:

| | GT divisions | pass sister ≤ 7.2 | pass parent ≤ 4.7 | **pass ALL gates** |
|---|---|---|---|---|
| 44b6 | 26 | 30.8 % | 76.9 % | **8 / 26 = 30.8 %** |
| 6bba | 125 | 14.4 % | 53.6 % | **14 / 125 = 11.2 %** |
| **pooled** | **151** | **17.2 %** | **57.6 %** | **22 / 151 = 14.6 %** |

**Of the 89 metric-reachable divisions, only 10 are admissible under the current gates.**
Widening both gates recovers admissibility rapidly:

| sister gate | parent gate 4.7 (current) | parent gate 10 |
|---|---|---|
| 7.2 (current) | **10** | 13 |
| 9.0 | 22 | 26 |
| 11.0 | 30 | 43 |
| 13.0 | 33 | 67 |
| 15.0 | 33 | **81** |
| 18.0 | 33 | **84** |

**Both gates must move.** Raising the sister gate alone saturates at 33/89 because the parent
gate then binds; raising both to (15, 10) admits **81 of 89 — an 8× increase in what is even
proposable.**

Note this also explains why the public 0.918 stack's identical constants
(`metric_forensics_2026-08-17.md` §2.4) leave the whole public field at divJ≈0: the entire
lineage inherited the same mis-set gate.

---

## 4. THE DIVISION LANE — ranked plan

### D1 — Re-gate the proposer to the measured GT geometry (**do this first**)

**Mechanism.** Replace the geometry gates and the ranking score in
`add_safe_divisions_postlink` with values fitted to §1.2's measured GT distributions:
raise `SAFE_DIV_MAX_UM` 4.7 → **10** and `SAFE_DIV_SISTER_MAX_UM` 7.2 → **15**
(§3.2: this pair admits **81 of the 89** metric-reachable divisions, versus 10 today), and
**replace the ranking score with one that is decreasing in "divisionness"** — the deployed
`score = parent_dist + 0.15*sister_dist` (`wrapper.py:831`) ranks tightest-first, which §1.2
shows is exactly backwards. Use `score = −cos_angle + λ·(divergence persistence)` so wide,
angularly-opposed, persistently-diverging pairs rank first. Raise the frame/global caps
proportionally, since the proposal pool becomes both larger and better.

- **Why this is not a closed lever.** `failed-experiments.md` closes *scalar-threshold
  re-acceptance* (a detection-recovery threshold) and the H1-T *conditional pair ranker*
  (`experimental-records.md:118`). This is neither: it is a correction of a proposer gate that is
  **measurably mis-set against ground truth**, with a mechanical explanation for
  `fork_at_matched_mother = 0`. New mechanism, stated falsification below.
- **Cost.** CPU-only, hours. No retrain, no GPU, no new data.
- **Falsification test.** Replay both LOEO folds with the re-gated proposer. **Kill unless
  `fork_at_matched_mother` rises from 0 to ≥ 25 across the 89 metric-reachable divisions** —
  i.e. at least a third of the 81 now-admissible divisions actually receive a fork. Composite is
  the second-order check; the first-order check is whether forks appear at true mothers at all.
  If forks land at true mothers but divJ does not rise, the fault has moved to the
  `_is_strongly_connected_division` topology check (`division_metrics.py:227-271`) and D2 becomes
  necessary. Report **both** score terms, not just divJ (see §7.5).
- **Expected.** Break-even is +8 net divisions (§2.3). Admissibility goes 10 → 81 of the 89
  reachable; converting even **10 % of the newly-admissible pool clears break-even**, and 50 %
  (≈40 divisions) is ≈ **+0.026** at moderate FP levels. **[INFERENCE]**
- **Risk.** Widening the gates will also multiply FP forks. §2.3 shows this is nearly free at our
  operating point (dFP 613 → 1200 costs only −0.0003), and the non-linearity only bites once
  precision is high. **Widen first, filter second** — this is the correct order given the
  quadratic FP cost structure.

### D2 — Fork-precision filter (**only as D1's second stage, never alone**)

**[MEASURED] Do not run this before D1.** §2.4 shows a filter on today's forks caps at +0.0027
perfect / +0.0007 honest. It becomes valuable only once D1 has raised dTP, because the FP cost
`0.1·dTP/D²` grows with dTP (`metric_forensics` §0.2 row 7).

- **Features** (all present in `forks_pooled.csv`, all CPU, all computable from the graph alone):
  `cos_angle` (best single: 63.7 % FP kill at full TP retention), `daughters_next_dist` /
  `inter_daughter` ratio (divergence persistence), `nn_parent` (local crowding),
  `mother_step`, plus the existing-child and sister distances the proposer already computes.
- **Honest fitting protocol given 26 vs 125 events.** Never fit on pooled events and report
  pooled. **Fit on 44b6's 26 divisions, evaluate on 6bba's 125, and separately the reverse; report
  both directions and take the min.** This is the same paired-LOEO discipline that the motion-gate
  arm B win was measured under, and it is the only protocol that survives the 5-positive problem
  — after D1 the positive count should be in the tens, at which point a 2–3 parameter monotone
  rule (not an MLP — that is the closed H1-T lever) is fittable.
- **Falsification test.** Rank all emitted forks by the fitted score, sweep top-k, re-score LOEO.
  **If divJ does not rise monotonically as k falls, the FP mass is not rank-separable and D2 is
  dead** (this is `metric_forensics` F1's test, still un-run and still correct — but it must be
  run on D1's output, not today's).
- **Cost.** CPU replay on cached graphs, hours.

### D3 — linajea-style cell-state classifier with hard constraints

**Mechanism** (`scientific_incumbents_2026-08-17.md` §3.2; Hirsch et al., MICCAI 2022,
arXiv:2208.11467, doi:10.1007/978-3-031-16440-8_3; code `github.com/funkelab/linajea`, **MIT**):
a 4-class cell-state head (normal / dividing-parent / daughter / background) whose **parent-side
and daughter-side predictions are coupled as hard ILP constraints**, so a division is admitted
only when the parent-at-t head and the daughter-at-t+1 head agree *and* the edge is selected.
Measured: **17× and 7.5× reduction in division FPs** on mskcc-confocal / nih-ls (FPdiv
0.89→0.053, 1.5→0.20 per 1000 GT edges) at the cost of ~+50 % FNdiv — it buys division
*precision*, the currency §2 is denominated in.

- **Clears the closed-lever bar** for the reason already recorded: it is a two-sided classifier
  coupled by hard constraints, not the conditional pair-ranker that was closed structurally
  (`experimental-records.md:118`).
- **What trains it.** Our own 151 GT divisions are too few for a patch classifier. Candidates:
  (a) **Freitas synthetic** — see §4.4, verification required first; (b) Zebrahub — **explicitly
  flagged L8 in `scientific_incumbents_2026-08-17.md`: its divisions are Ultrack OUTPUT with
  BC(i)≈0.47, so training on them risks distilling Ultrack's division errors.** Do not use
  Zebrahub divisions without the agreement pre-check in §4.4.
- **Cost.** One small GPU session (linajea's own note: "for anisotropic data a smaller GPU is
  sufficient"); plus ILP integration work.
- **Falsification test** (unchanged from L4): train the 4-class head on fold A's cached candidate
  patches; on fold B require **precision ≥ 4.07 % / 6.38 % at 50 % recall among metric-visible
  forks**; kill otherwise.
- **Ranking rationale.** D3 is a *precision* mechanism, and §1 proves our deficit is *proposal*.
  **D3 is therefore correctly sequenced after D1**, not before it — it would currently be applied
  to a fork population containing zero recoverable divisions.

### D4 — Detection work for the 49 unreachable divisions

29 `daughter_undetected` + 20 `mother_unmatched` = 49/151 (32.5 %) need new detections and are
untouchable by post-processing. This is the H1 retrain lane's territory, not the division lane's.
Recorded here only to bound D1+D2's ceiling at **89/151 = 0.589 divJ (+0.058)**.

### 4.4 Does Freitas synthetic actually resemble competition divisions? — **NOT VERIFIED, and never has been**

**[MEASURED] Prior-work check.** Freitas appears in-repo **only as a plan, never as an
experiment**: `research/01-research-direction/bets.yaml:66-74` (`bet-synthetic-division`, status
= a bet), `research-themes.md:27-30` (Theme C), `research-landscape.md:44` (H3),
`papers.md:26` (marked "program"), `existing-knowledge.md:44`, `data-acquisition.md:28-29`,
`data-governance.md:35` (CC0), `retrain_recipes_2026-08-17.md:600` (licence line only).
It is **absent from `failed-experiments.md`** and absent from `experimental-records.md` results
rows. **No closed work is being redone — but equally, no evidence exists that it transfers.**

**[UNVERIFIED] What is claimed and what is actually checked:**

| Claim | Status |
|---|---|
| 18.5 GB, CC0, 165k labelled divisions (~540× real) | documented from discussion #732103; **file-level unverified — not present in `data/`** |
| "pooling-matched to the evaluator's `vol[:, ::4, ::4]` stride" | documented; **this matches the *pooling*, not the physics** |
| Divisions resemble competition divisions | **NEVER TESTED** |

**The specific risk, now quantifiable.** §1.2 gives the exact target statistic: competition
divisions have inter-daughter distance **8.98 µm (44b6) / 11.47 µm (6bba)** and cos(angle)
**−0.71 / −0.76**, under an anisotropic voxel scale of (1.625, 0.40625, 0.40625) µm. A synthetic
generator matched on *pooling stride* says nothing about whether its divisions reproduce that
geometry or that anisotropy. **[INFERENCE]** Also note the two real embryos differ from each
other by 28 % on inter-daughter distance — if a synthetic corpus cannot even span that real
spread, it will not transfer.

**Gate before any Freitas spend (cheap, CPU, hours):** download, extract the division events,
and compute the §1.2 table on them — inter-daughter distance, parent→daughter distances,
cos(angle), all in physical µm. **Abort the Freitas lane unless the synthetic inter-daughter
median falls inside the real [8.98, 11.47] µm range and cos(angle) median is ≤ −0.6.** This is a
one-table test that costs a download and settles a bet that has sat open since the project began.

The same gate applies to Zebrahub divisions (L8's pre-check): measure agreement against our
annotated divisions before distilling; abort if precision < 0.7.

---

## 5. EXTERNAL STATE OF THE ART, 2024–2026

| Method | Division mechanism | Numbers | Portability |
|---|---|---|---|
| **Trackastra** (ECCV 2024, arXiv:2405.15700, BSD-3) | Transformer over pairwise associations in a temporal window + **parental softmax**: permits one-to-many (a parent → two children) but **forbids >1 parent per detection** by construction | parental softmax alone **reduces errors ~20 %** for both greedy and ILP linkers on dividing data | **Mechanism is directly portable and cheap.** The constraint "≤1 parent, ≤2 children" is a *normalisation*, not a threshold — reimplementable over our edge probabilities without vendoring |
| **OrganoidTracker 2.0** (Betjes et al., *Nature Methods* 22:2400–2410, 2025, doi:10.1038/s41592-025-02845-6; GPL-2.0 ⚠) | **A dedicated CNN predicts link *and division* likelihoods from image crops**, then a global max-likelihood solve; error probabilities per step from comparison with alternative solutions | <0.5 % error per cell per frame pre-curation; best BC(i) 0.930 on the closest CTC dataset | **Mechanism only** (GPL-2.0). Confirms the field's answer is a *separate division head*, not geometry |
| **HOCT** (Bragantini, Theodoro, Royer — the organisers — arXiv:2607.11754, 2026-07) | Edge-centric transformer: candidate links attend to each other under a 3D geometric prior, explicitly because **"cell divisions entangle distinct lineage paths in node embedding space"** | 59 % tracking-error reduction with 400 annotations; 6.75 % over transformer baselines. **No division-specific metric published** | Already community-tested as a drop-in linker and **underperformed a tuned ILP** (#728551) — `failed-experiments.md`. Diagnosis of *why* divisions are hard is the takeaway |
| **ITEC** (bioRxiv 2026.03.12.711203) | Fully unsupervised iterative tracking with error correction; exposes an explicit **division threshold** trading FP/FN | zebrafish, 18.5 M cells, claimed >99.7 % lineage accuracy | **[UNVERIFIED]** — full text not retrieved (bioRxiv rate-limited). Same organism/modality as ours; worth a follow-up read |
| **BiologicalNeeds** (LUH-GE, TMI 2025, arXiv:2403.15011, MIT) | Mitosis-aware multi-hypothesis tracking with aleatoric uncertainty; Erlang mitosis cost | **9 datasets, all 2D** | Mechanism only; no 3D evidence |
| **linajea + cell-state classifier** (JAN-US) | §4.3 | **17× / 7.5× FPdiv reduction** | MIT; the strongest *precision* mechanism available |
| **SynCellFactory** (MICCAI 2024, arXiv:2404.16421) | Generative (ControlNet) augmentation for cell tracking; boosts trackers when real data is sparse | improves established trackers on sparse-data regimes | Relevant precedent that synthetic *can* transfer — but 2D CTC, and it conditions on real data rather than being free-standing |

**Cross-cutting reading.** Every 2024–2026 method that does divisions well uses **a dedicated,
separately-supervised division signal** (a division head, a 4-class state classifier, or a
structural constraint like the parental softmax) — **none** relies on geometric gates over a
linker's output, which is exactly what our deployed pipeline does. The field's BC(i) on dense
light-sheet embryos remains 0.00–0.67, and Kaiser et al. note the standard metrics *under*-reward
mitosis, so incumbents are weakly optimised for precisely the term our metric weights at 0.1.
This remains the strongest available argument that division recovery is where an unclaimed edge
sits.

**[INFERENCE]** The cheapest external mechanism to port is **Trackastra's parental softmax**
(a normalisation over our existing edge probabilities, CPU, no retrain, BSD-3) — worth pairing
with D1 as the structural half of the fix.

---

## 6. RANKED PLAN WITH COSTS AND FALSIFICATION

| # | Lever | EV | Cost | GPU? | Falsification test |
|---|---|---|---|---|---|
| **D1** | **Re-gate + re-rank the fork proposer to measured GT geometry** (`wrapper.py:123-124,831`): sister 7.2→15, parent 4.7→10, ranking score sign-flipped | **Highest — the only lever that can add a TP.** Raises admissible reachable divisions 10 → 81 of 89 | CPU, hours | **No** | Replay both LOEO folds; **kill unless `fork_at_matched_mother` rises 0 → ≥25 of the 89 reachable divisions** |
| **D2** | Fork-precision filter on D1's output, `cos_angle` + divergence persistence, fitted 44b6→6bba and reverse, min of both | High **only after D1** | CPU replay | No | Sweep top-k; **kill if divJ is not monotone in k** |
| **F-gate** | **Freitas geometry gate** — compute §1.2's table on synthetic divisions | Decides a bet open since project start | download + hours CPU | No | **Abort Freitas unless synthetic inter-daughter median ∈ [8.98, 11.47] µm and cos ≤ −0.6** |
| **D3** | linajea 4-class cell-state head + hard coupling constraints | High, but a *precision* mechanism — sequence after D1 | 1 small GPU session + ILP work | Yes | Train on A, test on B; **precision ≥ 4.07 %/6.38 % at 50 % recall** |
| **PS** | Trackastra parental-softmax normalisation over our edge probabilities | Medium, very cheap | CPU | No | Apply and re-score; kill unless bilateral min-fold ≥ +0.002 |
| **D4** | Detector work for the 49 unreachable divisions | — | GPU retrain | Yes | Belongs to H1, not this lane |

**Recommended immediate action: D1 alone**, on CPU, judged on `fork_at_matched_mother` before
any composite claim. It is the only lever measured to address the actual defect, it needs no new
data and no GPU, and its falsification test is a count, not a score difference — so it cannot be
confounded by the LOEO→LB transfer problem that killed the last two levers.

**Do not run a fork-precision filter as a standalone lever.** §2.4 shows its honest ceiling is
+0.0007, which would reproduce the exact +0.000 LB outcome already seen twice.

---

## 7. WHAT I COULD NOT VERIFY

1. **Freitas synthetic's actual division geometry** — the dataset is not in `data/` and was not
   downloaded this session. §4.4 gives the exact one-table gate that settles it.
2. **ITEC's division mechanism in detail** — bioRxiv returned HTTP 429 repeatedly; only the
   abstract-level description (curvature threshold + division threshold) was obtained.
3. **HOCT's division-specific numbers** — the abstract publishes no branching/division metric;
   would need the full PDF.
4. **The hidden test set's division count `G`** — unknowable; §2.3's break-even is stated in net
   divisions, which is `G`-invariant to first order, precisely so it survives that ignorance.
5. **The edge-term side-effect of removing or adding forks** — each fork carries a second
   out-edge whose own TP/FP status was not measured here. Widening the proposer (D1) adds edges;
   at `d_ann ≈ 2.8 %` most land in unannotated territory and are free
   (`metric_forensics` §1.3), but **D1's replay must report both terms**, not just divJ.
   Cheapest way to settle it without a full rescore: record, per fork, whether each of its two
   out-edges is an edge-TP under `_match_full` — one matching pass per crop, far cheaper than
   `score_divisions`' per-division matching.
6. **Whether the 89-reachable census is stable under a re-gated proposer.** Reach is measured
   against *the current* prediction's node matching; adding forks does not change node positions,
   so reach should be invariant, but **[UNVERIFIED]** — D1's replay re-measures it for free.

---

## Appendix — reproduction

- Forensics: `<scratchpad>/div_forensics.py` (per-fork classification via the official
  `score_divisions`; per-GT-division cause decomposition via `_match_full`), run over
  `c:/temp/subvoxel_f{0,1}/loeo_split{0,1}_strict.csv.gz` against `data/train/*.geff`.
- Aggregation: `<scratchpad>/analyse.py`. Row-level outputs: `forks_pooled.csv` (11,582 forks),
  `gtdiv_pooled.csv` (151 GT divisions), `summary_f{0,1}.jsonl` (per-crop).
- Cross-check: pooled reach 89/151 (21/26 + 68/125) reproduces
  `scripts/core/score_loeo_submission.py:88-103` exactly; pooled 5 TP / 613 FP / 146 FN
  reproduces the brief's independent scorer run exactly.
- Scorer citations read from `vendor/kaggle-cell-tracking/src/tracking_cellmot/`
  (`metrics.py`, `division_metrics.py`) at merge `075fc5f`.
