# Endgame strategy: timeline, shakeup, slots, and the resumption battle plan — 2026-08-18

Strategy-layer report commissioned at the 2026-08-18 pause. Scope: (1) hard timeline facts and
resource arithmetic; (2) public/private shakeup analysis; (3) submission-slot policy under the
validation crisis; (4) the 08-24 → deadline battle plan; (5) risk register. Written after reading
all 12 internal reports (2026-08-16/17), `bets.yaml`, `failed-experiments.md`,
`experimental-records.md`, `submissions.md`, and live Kaggle API pulls made this session
(2026-08-18). Companion 08-18 agent specs are referenced by filename; none had landed at time of
writing — reconcile on resumption per `research/00-system/handoff.md` item 0.

Labels: **[DOCUMENTED]** = read off the Kaggle API / competition artifacts / cited file this
session; **[REASONED]** = derived, assumptions stated; **[UNVERIFIED]** = could not confirm, with
what would settle it.

---

## 1. TIMELINE FACTS (the numbers that gate everything)

All API rows pulled live 2026-08-18 via `kaggle.api` (competition id 136605,
`https://www.kaggle.com/competitions/biohub-cell-tracking-during-development`). The `kaggle.exe`
CLI is blocked by a Windows Application Control policy on this machine; the Python module path
(`.venv\Scripts\python.exe -c "from kaggle.api.kaggle_api_extended import KaggleApi; ..."`)
works and is the reproduction route.

| Fact | Value | Status |
|---|---|---|
| Final submission deadline | **2026-09-29 23:59 UTC** | [DOCUMENTED] API `deadline` |
| Team merger deadline | **2026-09-22 23:59 UTC** | [DOCUMENTED] API `mergerDeadline` |
| New-entrant (entry) deadline | 2026-09-22 23:59 UTC | [DOCUMENTED] API `newEntrantDeadline` |
| Daily submission limit | **5 / day** | [DOCUMENTED] API `maxDailySubmissions` |
| Max team size | 5 | [DOCUMENTED] API `maxTeamSize` |
| Prize pool | **$60,000 USD** total | [DOCUMENTED] API `reward`; per-place breakdown **[UNVERIFIED]** — read the Prizes tab in a browser (SPA, not scriptable here) |
| Format | Kernels-submissions-only (notebook, internet-off rerun) | [DOCUMENTED] API `isKernelsSubmissionsOnly=True`; rules summary in `competitive_frontier_2026-08-16.md` §Rules |
| Category | Research (awards points) | [DOCUMENTED] API |
| Teams | **2,480** (2026-08-18; was 2,471 on 08-17 pm) | [DOCUMENTED] API `teamCount` |
| Our submissions consumed | **12** (all COMPLETE; list matches `research/07-outputs/submissions.md` exactly; `privateScore=None` on all — normal pre-deadline) | [DOCUMENTED] API `competition_submissions` |
| Our best public | **0.915** (`55274582` P3 harmonic; `55585140` P3+armB identical) | [DOCUMENTED] |
| Leaderboard top (2026-08-18) | **TWEAK 0.951** (new leader, up from 0.949), Mark Cooper 0.950, Soheil Ayati 0.948, yuto083 0.947, z7777/Matt Goldfield/enddl22 0.945 | [DOCUMENTED] API `competition_leaderboard_view` this session |
| Top-3 boundary | **0.948**; our gap **+0.033** | [DOCUMENTED] |
| Our rank | API `userRank=229`; the 08-17 full-CSV read put us at 403 of 2,471 at 0.915 (`metric_forensics_2026-08-17.md` §2.1). Treat "~rank 230–410 inside the 0.915 plateau, drifting down" as the honest statement | [DOCUMENTED, two sources disagree on the exact integer] |
| Final-submission selection | Kaggle standard is **2 selected final submissions** (auto-select = best public if none chosen) | **[UNVERIFIED for this competition]** — confirm on the submissions page in a browser on day 1 of resumption; the whole final-pair strategy in §4 assumes 2 |
| Kaggle GPU quota | **~30 T4×2-session-hours/week**, 12 h/session cap — reserved for inference / export / LOEO kernels and submission runs (submissions must come from Kaggle notebooks) | [DOCUMENTED] `retrain_recipes_2026-08-17.md` §A2; reservation policy = host directive 2026-08-18 |
| Colab Pro GPU quota | **~45 GPU-hours/week** (host-stated 2026-08-18), internet available, no 12 h-cap concern at the same severity — the **training** budget | [DOCUMENTED as a host statement]; GPU type mix and effective throughput vs T4×2 **[UNVERIFIED]** — measure s/step in the first Colab session |

### 1.1 Resource arithmetic, 2026-08-24 → 2026-09-29

- **Calendar:** Aug 24–31 = 8 days, Sep 1–29 = 29 days → **37 working days**, host full-time.
  Hard interior date: 09-22 merger/entry deadline (no action needed — we are entered; merging is
  a strategic option that expires then).
- **Submission slots:** 5/day × 37 days = **185 slot ceiling** [REASONED from documented 5/day].
  Slots are *not* the numerically binding constraint — see §3 for what actually is.
- **GPU — two separate budgets (host directive 2026-08-18):**
  - **Kaggle: ~30 T4×2-session-hours/week**, 12 h max/session (pin
    `machine_shape: NvidiaTeslaT4` per memory note, else P100) [DOCUMENTED in
    `retrain_recipes_2026-08-17.md` §A2, citing Kaggle product pages]. The window touches ~6
    weekly reset periods → **ceiling ≈ 180 h; plan against ~150 usable** [REASONED — reset
    day/UTC boundary not verified for this account; measure it the first week].
    **Reserved for inference, export, LOEO scoring kernels, and submission runs** — submissions
    must come from Kaggle notebooks (unchanged constraint).
  - **Colab Pro: ~45 GPU-hours/week** (host-stated), internet available, session caps less
    binding → **ceiling ≈ 270 h over ~6 weekly periods; plan against ~220 usable**
    [REASONED from the host's weekly figure; Colab's GPU mix (T4/L4/A100) and effective
    s/step vs Kaggle T4×2 are **[UNVERIFIED]** — benchmark in the first session before
    budgeting runs against it]. **This is the training budget** (H1 retrains, ablation arms,
    seeds).
  - **Combined plan-against figure: ~370 GPU-hours (~2.5× the pre-update assumption).**
    Consequence: *training feasibility is no longer the gate it was.* `retrain_recipes`' "AMP
    is the feasibility gate" softens to "AMP is a 3–6× throughput multiplier worth 0.3 h to
    enable"; B1-scale retrains, the C3 surgical-FT arms, C4 second seeds, B2/B3 ablations, and
    even a C5 pseudo-labelling round all fit without cannibalising inference capacity.
    What Colab does **not** change: submissions and LOEO export kernels still consume Kaggle
    quota, and any Colab-trained weights must be uploaded as a Kaggle dataset and reproduced
    through the internet-off kernel path before they can score (adds a packaging/parity step —
    see §5 risk 3).
  - For scale: one full H1 retrain stage (B1) is budgeted 8–12 h at Kaggle-T4×2 throughput
    (`retrain_recipes` §1) — now a *Colab* line item; a LOEO fold-pair eval ≈ 2 Kaggle sessions
    ≈ 6–8 h (observed for the arm-B kernels); one submission-kernel run ≈ 0.5–1.5 h (P0-A
    completed in 1,522 s).
- **Scoring latency:** observed 1.5–4.5 h from submit to score (P0-CR ~4.5 h;
  `55585140` >1.5 h pending) [DOCUMENTED in `submissions.md`]. Budget half a day per LB answer;
  never plan a same-day decide-and-resubmit chain on the final two days.

---

## 2. RESUMPTION BATTLE PLAN (08-24 → 09-29)

### 2.0 Prior facts the plan is built on

1. **Validation crisis.** Arm-B motion-gate: paired deployment-substrate LOEO +0.0144/+0.0090
   (official scorer, single toggle) → **LB +0.000** (`55585140` = 0.915 = P3 alone). The
   "+0.005 bilateral LOEO" promotion gate is falsified *as sufficient*. Fourth consecutive
   optimistic central estimate. [DOCUMENTED: `experimental-records.md` 2026-08-18 §A,
   `submissions.md`.]
   **LOEO retains one honest role: a kill gate.** A bilaterally-negative or sub-bar LOEO result
   still kills a lever cheaply (sub-voxel refine died this way, −0.0004/−0.0009). What LOEO can
   no longer do is *promote to a submission* on its own. [REASONED — this asymmetry is the load-
   bearing assumption of the whole plan; nothing in the arm-B failure contradicts using LOEO to
   reject.]
2. **The gap is weights + divisions.** The 0.915→0.948 gap is almost entirely edge term
   (retrained weights) + the division pool (`redteam_blindspots` Claim 3; `metric_forensics`
   §0.3: division term worth 0.100, we claim ≈0.000; full-substrate divJ 0.0152/0.0047, TP=5
   FP=613 FN=146; 1 recovered division ≈ +17 edge-TP). Post-processing on the shared checkpoint
   broke a plateau in **zero** of the analogous competitions surveyed
   (`kaggle_winners_playbook` §Cross-cutting).
3. **H1 is split.** Detector half: `kkunizaw/biohub-zh001r`, geometry-verified 1.677 µm ≈
   1.032× deployed grid, labels aligned, **no track ids**. Edge half: our own level-1 stream
   (`scripts/win_bet/h1r_fetch_imaging.py`), track ids present, Ultrack-noisy divisions
   BC(i)≈0.47. Scaffolds pass on CPU (`h1r_zh001r_{audit,smoke}.py`, `h1r_train_smoke.py`).
   [DOCUMENTED: `experimental-records.md` 2026-08-17 entries.]
4. **A rival is executing the same retrain** (kkunizaw uploads 08-16/17, currently 0.887) and
   the 0.945+ tier gained two entrants in 24 h on 08-17. The plateau will move. [DOCUMENTED:
   `competitive_refresh`, `metric_forensics` §2.]

Where a sibling 08-18 spec exists, this plan states only the budget, sequencing, and kill
criteria, and defers design detail by filename: `loeo_lb_gap_2026-08-18.md`,
`h1_execution_spec_2026-08-18.md`, `zebrahub_data_engineering_2026-08-18.md`,
`division_lane_2026-08-18.md`, `operating_point_2026-08-18.md`, `live_surface_2026-08-18.md`,
`edge_training_frontier_2026-08-18.md`.

Budget notation below: **C** = Colab hours (training), **K** = Kaggle hours
(inference/export/LOEO/submission kernels), per the §1.1 split.

### 2.1 Phase 0 — Instrument repair (Aug 24–26). Budget: ≤8 K GPU-h, ≤2 slots.

Nothing GPU-heavy launches until the LOEO→LB gap has a working theory. The open diagnostic
(stated in `experimental-records.md` 2026-08-18 §A): LOEO scores `data/train` held-out-embryo
crops with **per-fold weights**; the LB scores `data/test` crops with **full weights**. Which
difference kills the signal?

- **P0.1 (Day 1, CPU + ≤4 GPU-h):** score the *deployed full-weight* pipeline on the same LOEO
  crops (the cheap discriminator already named in the ledger). If full-weight LOEO also shows
  arm-B ≈ +0.009 where the LB showed +0.000 → the **crop population** is the killer and no local
  instrument can promote; if full-weight LOEO shows ≈ 0 → **per-fold weights** were the killer
  and full-weight LOEO becomes a usable (bias-known) preflight. Detailed design: defer to
  `loeo_lb_gap_2026-08-18.md`.
- **P0.2 (Day 1–2, CPU):** confirm the 2 final-selection count, the prize breakdown, and the
  GPU reset day in a browser (the three [UNVERIFIED] cells in §1).
- **P0.3 (Day 2):** reconcile this plan against the seven sibling specs; update
  `handoff.md`.

**Branch D1:**
- *Weights-were-the-killer* → adopt full-weight LOEO as preflight (still with the §3 slot rules);
  proceed to Phase 1 with more local trust.
- *Population-is-the-killer or inconclusive* → LB A/B is the only promotion instrument; tighten
  §3 further (only lever classes with expected ≥ +0.005 ever reach a slot); proceed to Phase 1
  unchanged — the H1 falsification is LOEO-as-kill-gate + LB-as-promotion either way.

### 2.2 Phase 1 — H1 pilots, both halves (Aug 26 – Sep 8). Budget: ~70 C + ~20 K GPU-h, 2–4 slots.

The top-3 path. Execution detail: `h1_execution_spec_2026-08-18.md` and
`zebrahub_data_engineering_2026-08-18.md`; training-side recipe already fixed in
`retrain_recipes_2026-08-17.md` (A1–A4 preflights ≈ 0.5 GPU-h; B1 = init-from-public-50ep →
dense-Zebrahub stage → sparse fine-tune under ignore mask, 8–12 GPU-h at T4×2 throughput;
B2/B3/B4/A5 ride inside B1). **Colab update:** training runs on Colab (internet on, data pulls
direct), so `retrain_recipes`' "AMP is the feasibility gate" is downgraded to a throughput
multiplier — still enable it first (0.3 h), but a slow first B1 no longer kills the lane.
The Colab budget also un-parks arms `retrain_recipes` sized as luxuries: run **C3 (surgical-FT,
4 arms, ~1.5 h)** and **C4 (second seed, +1× B1)** inside Phase 1 rather than deferring them,
and hold **C5 (pseudo-labelling, 6–10 h, bidirectional check)** as a Phase-3 option instead of
a cut. The from-scratch nnU-Net-scale retrain (≈ 42 h/fold) moves from "dead on arrival" to
"affordable but still dominated by init-from-50ep" — do not reopen it without a measured reason
[REASONED]. LOEO evaluation of any trained checkpoint stays on **Kaggle** (paired export
kernels, official scorer path), as does everything that must match the internet-off runtime.

- **P1.1 Detector half** on `zh001r` (+ `zmnscrops` after audit): B1 with the two-stage
  noisy→clean discipline (`kaggle_winners_playbook` lever 2 — the best-quantified plateau-break
  in the survey, +4–6 % LB in its origin) and the radius-matched selection fix (A3).
  **Kill:** no bilateral LOEO gain over the public 50-ep weights (bets.yaml falsification,
  unchanged), assessed with the Phase-0-repaired instrument. Two checkpoint seeds saved from one
  run (free rider) for the C4 blend test.
- **P1.2 Edge half** on the level-1 identity-carrying stream: the plateau thesis lives here.
  Defer training design to `edge_training_frontier_2026-08-18.md`; Trackastra's λ_div=11× /
  parental-softmax fixes (F6) ride along. Colab's internet access also removes the level-1
  streaming friction (`h1r_fetch_imaging.py` can pull Zebrahub chunks directly in-session
  instead of staging through local disk or Kaggle datasets) [REASONED]. **Kill:** same
  bilateral rule.
- **P1.3 First LB A/B** only when a retrained system clears bilateral LOEO **and** the Phase-0
  instrument theory says the delta class should transfer (weights-level changes are exactly the
  class the arm-B failure does *not* implicate — arm B was a post-proc gate; a retrained model
  changes the detection surface, which is the mechanism class the five-instance transfer-failure
  pattern in `experimental-records.md` says is required). Expected-effect floor +0.005. 1 slot;
  prediction registered before the score, as always.

**Branch D2 (by Sep 8):** at least one H1 half bilaterally positive → Phase 3 composes it.
Both halves dead → **pivot declaration**: top-3 is no longer credible; retarget "best private
finish from 0.915 + divisions + composition" and hand all remaining GPU to Phase 2/3.

### 2.3 Phase 2 — Division lane (Sep 1 – Sep 16, parallel CPU-first). Budget: ~10 C + ~8 K GPU-h, 2–3 slots.

Largest unclaimed pool; precision, not recall, is the failure (613 FP / 5 TP; recall on the
division-rich sample measured 32 %). Design detail: `division_lane_2026-08-18.md`. Sequencing
constraints from the existing corpus:

- F1/F2 CPU replays first (`metric_forensics` Part 3): rank forks by confidence, keep-top-k
  curve; raise `SAFE_DIV_*` caps 4× to test whether the caps or the proposal stage bind.
  Requires **P3 OOF graphs persisted once** (~4 GPU-h, `quickwins` Candidate 2) — fold into the
  first Phase-1 session.
- Respect the closures: the pair-ranker and mother-gate are dead on both families
  (`redteam` Claim 1 — reopening needs AUC ≥ 0.97 evidence); the linajea-style cell-state
  classifier (17× FPdiv cut, `scientific_incumbents` #4) and Erlang/MHT mitosis cost
  (`novel_crossdomain` #5) are the two mechanisms not yet falsified here.
- The FP-cost non-linearity (`metric_forensics` §0.2 row 7) means divJ gains are self-reinforcing
  above ~0.05 and near-worthless below: **only a system projecting pooled ≥ +0.01 (divJ ≈ 0.10)
  earns a slot.** Division is also the shakeup-fragile term (§3.2b) — it is a *second-slot*
  (upside) candidate, never the safe pick.

### 2.4 Phase 3 — Composition + cheap residuals (Sep 8 – Sep 22). Budget: ~50 C + ~20 K GPU-h, 4–6 slots.

- Compose surviving H1 checkpoint(s) with the deployed post-proc; re-measure, never assume
  additivity (the harmonic×armB lesson). 2-seed / 2-architecture blend if B1 cleared
  (`redteam` Claim 3 as corrected by `kaggle_winners_playbook` — diversity of architecture, not
  seeds). Under the Colab budget the 2-architecture arm (e.g. a SegResNet-family second
  backbone per the CryoET precedent) is affordable in this phase if — and only if — the first
  retrain cleared its bar; C5 pseudo-labelling likewise, with its bidirectional check.
- Cheap residuals, only those measured positive by their specs: constant +0.5-voxel grid-centre
  shift (RMS 1.075 → 0.642 µm, zero inference cost — `operating_point_2026-08-18.md`);
  detection-threshold superset export (one GPU run → CPU threshold curve); quantile-of-`N_est`
  operating point (`kaggle_winners_playbook` lever 1, CPU). Each gets at most one shared A/B
  slot, bundled where mechanisms are independent and separable by pre-registered structural
  deltas.
- Monitor the public surface twice weekly (`live_surface_2026-08-18.md`); any adopted public
  mechanism goes through the P0-A-style audit pipeline (exploit scan, hash, structural audit)
  — never re-run "the top public notebook" raw (hub-node exploit family, `submissions.md`
  §Quarantined).

### 2.5 Phase 4 — Lock (Sep 22 – Sep 29). Budget: ~20 C + ~15 K GPU-h reserve, 4–6 slots.

- **Sep 22:** merger option expires (no merge planned; revisit only if a complementary
  0.94-tier partner materialises — [REASONED] EV low, identity/IP friction high).
- **Sep 22–26:** final A/Bs of the two candidate pairs; re-anchor P3 once if anything drifted.
- **Sep 27:** **lock the two final selections** (rule in §3.3). Two-day buffer covers the
  observed 4.5 h scoring latency, kernel failures, and one full contingency rebuild.
- **Sep 28–29:** no new science. Verify selections are marked, kernels reproducible, licences
  recorded in the shipping notebook (§6), evidence archived.

### 2.6 Budget roll-up

| Lane | Colab-h (train) | Kaggle-h (infer/export/LOEO/submit) | Lever slots | Kill-by |
|---|---:|---:|---:|---|
| Phase 0 diagnosis | 0 | ≤8 | ≤2 | Aug 26 (D1) |
| H1 detector (incl. C3/C4 arms) | ~30 | ~10 | 1–2 | Sep 8 (D2) |
| H1 edge | ~40 | ~10 | 1–2 | Sep 14 (D3, inside D2 pivot logic) |
| Division lane | ~10 | ~8 | 2–3 | Sep 16 |
| Residuals + composition (incl. 2nd-arch / C5 options) | ~50 | ~20 | 4–6 | Sep 22 |
| Final lock + reserve | ~20 | ~15 | 4–6 | Sep 29 |
| **Total** | **~150 of ~220** | **~71 of ~150** | **~14–21 of 185** | |

Both budgets carry deliberate slack — the five-instance transfer-failure pattern says most
levers die, and the reserve is what lets a late positive be exploited instead of merely
observed. Note the shape of the plan after the Colab update: **training compute is no longer a
gating constraint anywhere in it.** The gates that remain are the validation instrument
(Phase 0), Kaggle inference/export throughput, calendar kill-dates, and the §3/§4 slot
discipline — Colab hours widen the ablation menu (more arms per bet), not the number of bets.

---

## 3. PRIVATE-LB / SHAKEUP ANALYSIS

### 3.1 What is DOCUMENTED about the split

Very little, and it is worth being precise about how little:

- The competition is kernels-only with an internet-off rerun; `data/test` visibly contains
  **four placeholder movies whose names duplicate train crops** (`test/44b6_0113de3b.zarr` etc.
  — confirmed via `competition_list_files` this session), and the real hidden test replaces them
  at rerun. `N_est` is **not supplied** for hidden test movies
  (`research/02-theory/concepts-and-definitions.md`).
- **No host statement of a public/private percentage split was found** in the repo's captured
  rules text, the vendored scorer repo (`vendor/kaggle-cell-tracking/README.md` says only
  "test/ … no ground truth"), or web search this session. The rules page is SPA-gated. Whether
  the public/private division is by-crop percentage, by-movie, or by-embryo is **[UNVERIFIED]**
  — a browser read of the rules "Leaderboard" clause on day 1 is the single cheapest
  intelligence action remaining.
- Community awareness of shakeup risk is documented: discussion #735352 "Possible big
  leaderboard shakeup" (mikelou1, 08-15) argues the tiny division-event count makes top-of-board
  0.001 gaps overfit (`competitive_refresh` §2).
- Our own submissions' `privateScore` fields are all `None` (normal pre-deadline; confirms a
  private set exists and is withheld) [DOCUMENTED, API].

### 3.2 What can be REASONED (assumptions named)

**The public set is small and coarse.** The arm-B post-mortem arithmetic (`submissions.md`):
GT annotation density (0.655 % of estimated cells on 44b6, 8.529 % on 6bba — measured) implies
the public score is computed over roughly **~2,000 annotated edges**, so one edge ≈ 0.05 % and a
+0.008 effect needs ~16 net-correct annotated edges. Two independent observations fit this
model: arm B churned 7.148 % of edges and moved the public score by exactly 0.000, and the whole
plateau is 288 teams frozen at exactly 0.915. Assumption: the hidden public movies have
annotation statistics like the train crops. The ledger's further claim that the private set has
*more movies* than the public set is plausible (that is the standard Kaggle construction, and
the organisers held out material for post-deadline evaluation) but is **[REASONED, UNVERIFIED]**.

**(a) Plateau sitters at 0.915.** Within the plateau, submissions are near-byte-identical
(shared public checkpoint), so private ordering among them is decided by noise on a set nobody
has measured. For a *prize* objective this is irrelevant: staying at 0.915 loses with
certainty, shakeup or no shakeup. The correct posture is therefore **risk-seeking on the second
final slot** — we are 0.033 behind with nothing to defend. The only thing a plateau sitter can
lose in a shakeup is leaderboard vanity, not prize probability.

**(b) Our levers.**
- *Division lever:* highest variance per point. Train has 151 events over two embryos; the
  private set's count is unknown but the same order of magnitude per embryo. `division_jaccard`
  is micro-pooled with D ≈ a few hundred, so a handful of hidden-set events swing the term by
  0.01+. A division system that looks +0.02 on public could plausibly land anywhere in
  [−0.01, +0.05] privately [REASONED from D-size arithmetic]. Mitigation: only promote a
  division system whose gain is bilateral across both train families and rests on ≥ tens of
  TP events, not single-digit; and carry it only in the upside slot.
- *Retrained weights:* lowest shakeup exposure of any lever class we hold. A generalisation
  gain is exactly the thing a disjoint-embryo private set rewards, and the HuBMAP+HPA precedent
  (public/private "remained similar throughout" for generalisation-driven solutions —
  `kaggle_winners_playbook` §D) supports it. The symmetric caution: a retrained model that
  *reads flat on the tiny public set* may still be genuinely better privately — which is why
  §3.3 requires public **non-regression**, not public gain, for the upside slot.
- *Post-proc micro-tunes:* the arm-B result says the public set cannot even see +0.009-class
  post-proc effects; the private set may see them, in either direction. Anything tuned to
  3 significant figures on our substrates (the `SAFE_DIV_*` caps) is private-set-fragile.

**(c) The 0.94+ tier.** Submission counts are bimodal: TWEAK 166+, Matt Goldfield 124, enddl22
110 (probing-compatible) versus z7777 at 0.945 on **7** submissions and Soheil Ayati 0.948 on
29 (structural-edge-compatible) [DOCUMENTED counts, `metric_forensics` §2.1]. If the private
set is materially larger/different, the heavy probers are the most exposed cohort above us and
a 1–3 rank compression at the top is plausible; the 7-submission 0.945 is the score to treat as
real. Net for us: **the effective private top-3 bar may be slightly softer than 0.948 public,
but planning should assume 0.948** [REASONED].

**Overall implication:** the public-private structure *rewards* exactly the two risks the
battle plan takes (retrained weights, measured divisions) and *punishes* the two things we are
already disciplined against (public-set micro-tuning, placeholder-movie inference). No change
of direction is implied — the structure is an argument for the current programme, executed with
the slot rules below.

---

## 4. SUBMISSION-SLOT STRATEGY under the validation crisis

### 4.1 The real scarcity — a correction

Raw slots are plentiful: **185 remaining** (5/day × 37 days) against 12 consumed in the entire
campaign. The binding constraints are:

1. **The public LB's resolution as an instrument.** Display quantum 0.001; demonstrated
   insensitivity to a +0.009-LOEO-class post-proc change; ~2,000-edge denominator [REASONED,
   §3.2]. The LB can answer only *large* questions (≥ +0.003–0.005 expected), and every question
   asked and answered "flat" both spends a day of latency and adds a multiple-comparisons debt
   (§5 risk 7). **The scarce resource is trustworthy LB experiments, not slots.**
2. **Kaggle inference/export hours** (~150 usable — the half of the compute budget that
   submissions and LOEO scoring must come from; Colab's ~220 training hours cannot substitute
   for them) and **calendar** (37 days) — see §1.1.
3. **Candidate production**: every submission needs a built, audited, hash-logged kernel
   (`submissions.md` six-field discipline). That pipeline produces at most ~1–2 quality
   candidates/day flat out.

The existing standing rule ("never spend a submission to resolve an effect smaller than
~0.005", `submissions.md` 2026-08-02) is retained and extended below.

### 4.2 Slot allocation policy (08-24 → 09-29)

**Eligibility gate for any lever slot (all four required):**
1. Expected LB effect ≥ +0.005 under the pre-registered central estimate — *and*, given four
   consecutive optimistic central estimates, the recorded prediction must state the honest
   flat-outcome probability first.
2. Mechanism class changes the detection surface, the trained weights, or the division base
   rate — not a re-selection over existing candidates (the five-instance transfer-failure
   pattern, `experimental-records.md`).
3. Bilateral LOEO non-negative (kill-gate use of LOEO) on the deployment substrate.
4. Structural audit PASS 10/10 + pre-registered structural delta (churn %, division-count
   direction — the one preflight signal with a predictive win on record, P0-CR).

**A/B cost:** the P3 = 0.915 anchor stands (two independent confirmations), so a minimal A/B is
**1 slot**. Re-anchor (1 extra slot) only if the base kernel is rebuilt or Kaggle's environment
shifts. Bundling: independent mechanisms may share one slot only when their structural deltas
are disjoint enough to attribute the result; otherwise 1 lever = 1 slot.

**Ordering of lever classes into slots** (mirrors §2 phases):
1. Retrained-weights candidates (H1 detector, then edge, then blend) — the class the private
   set rewards and the arm-B failure does not implicate.
2. Division systems clearing the pooled ≥ +0.01 projection bar.
3. Bundled cheap residuals (grid shift + operating point) — one confirmatory slot.
4. Composite / final-pair confirmations.

**Tempo:** ≤ 2 lever submissions/day through Sep 21 (discipline, latency, attribution), rising
to the full 5/day only in the final week for operating-point and final-pair confirmation. Target
total spend ~15–20 lever slots; leaving ~165 unused is correct, not wasteful.

**Stop-exploring rule:** no *new* mechanism enters the LB pipeline after **Sep 20**; Sep 20–26
is confirmation and composition only.

### 4.3 Final-pair decision rule (lock Sep 27)

Assuming 2 selections (confirm §1 [UNVERIFIED] on day 1):

- **Slot 1 — the anchor (max-min):** the highest-public-scoring candidate that is structurally
  audited, bilateral-non-regressive on LOEO, and free of single-family or division-term
  fragility. Today that is P3 harmonic (0.915) + any confirmed composed residuals. This slot's
  job is to make the floor equal to our best verified state.
- **Slot 2 — the upside (max-EV-private):** the candidate maximising expected *private* score:
  a retrained-weights system with bilateral LOEO gain and public ≥ anchor − 0.001 (non-regression
  suffices — §3.2b), else the best division system clearing its bar, else a composite. If
  nothing qualifies, select the anchor's nearest structurally-diverse sibling (e.g. anchor +
  division suppression variant) rather than a duplicate — two identical selections waste the
  variance hedge.
- Tie-break between upside candidates: (i) larger bilateral min-fold, (ii) mechanism class
  rank from §4.2, (iii) smaller reliance on the division term.
- Both selections locked and verified marked in the UI by **Sep 27 23:59 UTC**; nothing after
  Sep 27 changes them except a catastrophic defect discovery in a selected kernel.

---

## 5. RISK REGISTER — top 8 failure modes

| # | Failure mode | Class | Early-warning signal | Mitigation |
|---|---|---|---|---|
| 1 | **No validated instrument ever emerges**: Phase-0 diagnosis is inconclusive and every promotion decision stays a 1-slot coin-flip at 0.001 resolution | methodological | D1 (Aug 26) produces neither a weights- nor a population-attribution | Pre-committed: treat LB A/B as the only promotion gate, effect floor +0.005, LOEO as kill-only; the plan already functions in this worst case (§2.1 branch) |
| 2 | **H1 does not transfer** — the retrain reproduces the transfer-failure pattern at the weights level | technical | Stage-1 dense pretrain fails to lift detection F1 on a held-out Zebrahub embryo (measurable before any competition eval); or detector LOEO flat by Sep 5 | Hard kill dates (D2 Sep 8, D3 Sep 14) + declared pivot to divisions/composition; two-stage recipe and A1–A4 preflights de-risk the run itself |
| 3 | **GPU budget burns on failed walls, or the Colab→Kaggle seam breaks**: sessions dying at export; or Colab-trained weights failing to reproduce through the internet-off Kaggle path (torch/env version skew, dataset packaging, GPU-type numerical drift) so training throughput never converts into scoreable artifacts | execution | >30 % of a week's *Kaggle* quota consumed by runs that produced no scored artifact; first Colab-trained checkpoint's Kaggle-kernel inference not bit-compatible / not parity-checked by Aug 29 | `--max-iters` + checkpoint-resume (mandated in `retrain_recipes` §1); smoke → pilot → full staging (CLAUDE.md rule 5); the arm-B "recovered from failed kernel with zero GPU" playbook; **prove the Colab→Kaggle round-trip once in week 1** (train a token checkpoint on Colab, upload as dataset, run the deployed inference kernel on it, parity-check outputs) before any full B1 relies on it |
| 4 | **Division-term shakeup**: an adopted division lever's public gain evaporates or inverts on the private division sample | competitive/statistical | Gain concentrated in one family or in < ~20 TP events; divJ projection sensitive to single crops | Division systems ride only in the upside slot; bilateral + event-count floors (§2.3); anchor slot carries no new division behaviour |
| 5 | **The plateau moves under us**: rival Zebrahub retrains (kkunizaw et al.) or a new public notebook resets the public ceiling above 0.918 while we are heads-down | competitive | `live_surface` monitoring: any public author > 0.918; zh001r/zmnscrops download counts jumping; new 0.93+ entrants clustering | Twice-weekly surface scans budgeted; audit-pipeline adoption path for any public mechanism (never raw-fork); our own H1 lane is the same lever, so rival success validates rather than obsoletes the direction |
| 6 | **Calendar compression**: heavy lanes serialize late, and the final week absorbs science instead of confirmation | calendar | H1 B1 not launched by Aug 28; any Phase-2 CPU replay still unrun by Sep 5; new mechanism proposed after Sep 20 | Phase kill-dates are calendar-triggered, not result-triggered; Sep 20 exploration freeze; Sep 27 lock with 2-day buffer sized to observed 4.5 h scoring latency |
| 7 | **Public-LB overfit by accumulation**: many 1-slot A/Bs each read +0.001–0.002 and the composed system's public gain is selection optimism (D-04: best-of-T manufactures ~+0.010 at T=5) | methodological | Sum of adopted individual public deltas exceeds the composed system's measured delta; adopted levers that were never re-confirmed in composition | One composite re-confirmation submission before lock; pre-registered predictions with flat-probability stated; effect floor keeps T small |
| 8 | **Ship-time compliance failure**: final kernel fails the hidden rerun (over-strict assertions, env drift), or a licence/reproducibility defect taints a winning entry (Zebrahub CC BY-NC-derived weights; reproducibility-if-win requirement) | technical/rules | Any export assertion firing on the placeholder run; any shipped asset lacking a licence line in the notebook; unanswered CC BY-NC-weights question | Soft-fail guards around non-critical assertions in submission kernels (keep hard guards in LOEO kernels); licence table maintained (§6) and printed in the shipping notebook; ask the host the weights-derived-from-NC-data question early (free, asynchronous), not at lock time |

---

## 6. Licence facts (one-line tags; never used to exclude research)

Host decision 2026-08-18 (handoff): licences are recorded as facts and matter at ship time only.

- Zebrahub imaging + tracks: **CC BY-NC 4.0**, host-cleared for competition use (#734330).
- `kkunizaw/biohub-zh001r`, `biohub-zmnscrops`: third-party derivatives of CC BY-NC Zebrahub.
- `vendor/kaggle-cell-tracking` (training/scorer code): **BSD-3-Clause**.
- Spotiflow **BSD-3**; Trackastra **BSD-3** (repo); ByoTrack **MIT**; linajea **MIT**;
  Ultrack **BSD-3**; nnU-Net **Apache-2.0**.
- Freitas synthetic dataset: **CC0**.
- uotod **LGPL-3.0** (prefer POT, permissive); CELLECT / OrganoidTracker **GPL-2.0**;
  PAC-MAP **CC BY-NC-SA**; CAP **no licence**; HOCT paper **CC BY-NC-ND**; CTC data
  **cloning forbidden**; DINOv3 weights **custom gated Meta licence**.
- Public notebook lineage of P3 harmonic: Togashi rule **CC0**; P0-A source licence
  **UNVERIFIED-DEFAULT-APACHE-2.0** (`submissions.md`).

---

## 7. Reproduction of this session's API pulls

```powershell
# competition metadata (CLI exe is Application-Control-blocked; use the module)
.\.venv\Scripts\python.exe -c "from kaggle.api.kaggle_api_extended import KaggleApi; api=KaggleApi(); api.authenticate(); r=api.competitions_list(search='biohub'); d=r.competitions[0].to_dict(); print({k:d.get(k) for k in ['ref','deadline','mergerDeadline','newEntrantDeadline','maxDailySubmissions','maxTeamSize','reward','teamCount','isKernelsSubmissionsOnly','userRank']})"
# our submissions
.\.venv\Scripts\python.exe -c "from kaggle.api.kaggle_api_extended import KaggleApi; api=KaggleApi(); api.authenticate(); [print(s.to_dict().get('date'), s.to_dict().get('ref'), s.to_dict().get('publicScore'), s.to_dict().get('privateScore')) for s in api.competition_submissions('biohub-cell-tracking-during-development')]"
# leaderboard top-20
.\.venv\Scripts\python.exe -c "from kaggle.api.kaggle_api_extended import KaggleApi; api=KaggleApi(); api.authenticate(); [print(r.to_dict().get('teamName'), r.to_dict().get('score')) for r in api.competition_leaderboard_view('biohub-cell-tracking-during-development')]"
```
