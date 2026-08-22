---
id: 00-system/handoff
title: Handoff
area: 00-system
status: active
updated: '2026-08-18'
owner: biohub
links: []
tags: [handoff, entry-point]
---

# Current handoff

**Status:** PAUSED 2026-08-18; host resumes **full-time 2026-08-24**. Branch `master`.
This file is the live entry point. Full direction: [directional-updates.md](../01-research-direction/directional-updates.md).
System map: [README.md](../README.md); architecture: [system-design.md](system-design.md); contract: [CLAUDE.md](../../CLAUDE.md).

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

Rank **370 / 2,588**; bronze ≈ 258. Deadline **2026-09-29** (~5.5 weeks). **171 slots** left at
5/day — slots are NOT the constraint, **calendar is**. Compute: Colab Pro ~45 GPU-h/wk (training,
internet) + Kaggle ~30 h/wk (inference/submission only).

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
