# Why the arm-B motion-gate measured +0.0144 / +0.0090 on LOEO and +0.000 on the leaderboard

**Status: COMPLETE.** All four cells of the diagnostic 2×2 are measured. All three
pre-specified mechanisms are **refuted**. The residual explanation is different from, and more
consequential than, any of them.

**Question.** Submission `55585140` (P3 harmonic + arm-B flow gate) scored **0.915** — identical
to P3 alone (`55274582`, 0.915). The lever had been promoted on a clean *paired*
deployment-substrate LOEO (official `tracking_cellmot` scorer, identical crops, only
`BIOHUB_ARMB_FLOW_GATE` differing): 44b6 0.9037 → 0.9181, 6bba 0.7051 → 0.7141. Where does the
signal die?

Pre-specified candidates: **(a)** weights/chain redundancy, **(b)** crop population,
**(c)** pooling/rounding.

**Headline.** None of them. The measurement chain is sound and every train-observable substrate
agrees the lever is worth **+0.0097 to +0.0365**. The leaderboard scores a population we cannot
observe, and — the load-bearing discovery — **the deployed pipeline contains a model trained on
all 199 labelled crops, so the deployed system cannot be validly measured on any labelled data we
hold.** That, not arm B, is the finding that gates H1.

---

## ⚠️ GOVERNANCE NOTICE — binding, read before using any number below

`data/test` contains four movies — `44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`,
`6bba_05db0fb1` — and I verified each is **byte-for-byte identical** to its `data/train`
counterpart (102 files, identical total bytes, identical SHA-256 over the concatenated chunks;
see §3). They are placeholders, and their ground truth is in `data/train`.

**CLAUDE.md rule 2 and `research/04-data/data-governance.md` forbid using these four movies for
model or threshold selection.** They are used here for one authorised purpose: **diagnosing the
instrument**. Binding constraints:

1. **No lever may be selected, tuned, thresholded, or promoted on this substrate** — not arm B,
   not H1, not anything.
2. **Every score computed on these four crops is in-sample and biased.** §4 shows the bias is not
   hypothetical: the deployed pipeline's secondary model was *trained on these exact crops*.
3. **The local scores do NOT reproduce the leaderboard** (§3). Quantified: two submissions the LB
   scores identically differ by **+0.0365** locally. So these numbers are not, and must never be
   presented as, an estimate of hidden-set quality.

---

## 1. MEASURED — the gate fired on the test population, at the same rate as on train

### 1a. Gate census from inside the actual submission kernel

```
PYTHONUTF8=1 .venv/Scripts/python.exe scripts/core/kaggle_factory.py fetch \
  --spec scripts/kaggle_specs/p3_armb.json --files armb_provenance.json run_stats.csv \
  --dest c:/temp/armb_diag
```

`armb_provenance.json` (sha256 `632e2b7657eefccbb66b7460cc3b97db1ce64b98fd922d608c56d9ac2dd08ea1`),
written by `scripts/kaggle_edits/armb_provenance.py:147` from the `GATE_STATS` counters defined at
`scripts/kaggle_edits/armb_flow_gate.py:37-46` and incremented at
`scripts/kaggle_specs/p3_armb.json:120`:

| counter | tight (6.0 µm) | relaxed (10.0 µm) |
|---|---:|---:|
| `pairs` | 63,100,692 | 183,673 |
| `admit_raw` (arm A predicate) | 133,103 | 1,360 |
| `admit_arm` (arm B predicate) | 131,880 | 1,344 |
| `newly_admitted` | 8,150 | 169 |
| `newly_excluded` | 9,373 | 185 |
| `net_delta` | **−1,223 (−0.919%)** | −16 (−1.18%) |
| `churn_frac` | **0.1316** | 0.2603 |

`arm: B_flow_compensated_relink_gate`, `armb_flow_gate_enabled: true`. Constants confirmed
unchanged (tight 6.0, relaxed 10.0, `armb_knn_k` 16, `det_threshold` 0.96875,
`output_edge_max_um` 14.0). **The gate fired hard: 13.2% of the arm-A admitted set was re-aimed.**

### 1b. Graph-level churn — and it refutes "the gate didn't fire on test"

`GATE_STATS` is **unrecoverable from every LOEO run**. All four LOEO kernels emitted
`armb_provenance_FAILED.txt` with `FileNotFoundError: '/kaggle/working/submission.csv'`.
**Root cause (a real instrumentation defect):** `scripts/kaggle_edits/loeo_export.py:117-124`
deletes every working-dir entry outside `_LOEO_KEEP` — including `submission.csv` — and the
appended provenance cell (`scripts/kaggle_specs/p3_armb_loeo_f0.json:136`) runs *afterwards* and
opens that file first (`scripts/kaggle_edits/armb_provenance.py:21`). **Fix: order the
`append_cell` provenance edit before the `replace_cell` LOEO-export edit.**

So I reconstructed the census by direct edge-set diffing of the paired artifacts
(`c:/temp/armb_diag/churn_diff.py`, results in `churn_diff.json`):

| contrast | crops | base edges | removed | added | **churn** | net |
|---|---:|---:|---:|---:|---:|---:|
| **TEST**: P3 vs P3+armB | 4 | 117,803 | 4,073 | 4,396 | **7.19%** | +323 |
| LOEO f0: base vs armB (44b6) | 71 | 1,847,798 | 58,287 | 60,282 | **6.42%** | +1,995 |
| LOEO f1: base vs armB (6bba) | 128 | 1,869,905 | 71,872 | 71,978 | **7.69%** | +106 |

Per-crop scale is comparable (test 29,451 edges/crop; f0 26,025; f1 14,609). Per-crop test churn
4.00% / 9.89% / 3.02% / 7.97% sits inside the per-crop range of both train folds.

> **The "gate is silent on test" form of mechanism (b) is REFUTED.** The intervention magnitude is
> statistically the same on test as on train.

Artifact hashes confirm two genuinely different submissions: P3
`3e98739f5c46f2ddd4bf1cc6e12aaf362bed1c4a1a8cb70053dc42e0ff5aa1e0` (12,486,220 B) vs P3+armB
`5b37524d96f522667b0786811e04349f1dc93c9d050cf6827b7d325b8cf52d80` (12,510,115 B), matching
`research/07-outputs/submissions.md:315` and the fetched provenance.

---

## 2. MEASURED — the LOEO chain is not the deployment chain, and the brief's (a) is factually wrong

Comparing `scripts/kaggle_specs/p3_armb.json` against `p3_armb_loeo_f0.json` / `p3_armb_loeo_f1.json`
edit-by-edit, plus `scripts/kaggle_edits/loeo_retarget.py`:

**Identical across all three:** the harmonic `_bi_new` literal swap, `degree_invariants.py`,
all four `add_safe_divisions_postlink` / `close_single_frame_gaps` patches, both
`assert_degree_invariants` calls, `armb_flow_gate.py`, the gate-predicate replacement, the
`GATE_COMPENSATION` wiring, and `BIOHUB_ARMB_FLOW_GATE=1`. **The lever and the harmonic base are
transcription-exact between LOEO and submission.**

**Differences:**

| component | deployment (`p3_armb`) | LOEO strict | source |
|---|---|---|---|
| primary edge weights, **fold 0** | pack `split_0` | **pack `split_0` — SAME** | `loeo_retarget.py:124` |
| primary edge weights, **fold 1** | pack `split_0` | overridden → `edge_predictor_best_split_1.pth` | `p3_armb_loeo_f1.json` env; `loeo_retarget.py:106-122` |
| **secondary model** (`seed314159`, `"train_datasets": 199`) | **ON** | **OFF** | `loeo_retarget.py:127-131` |
| **DeepCenter veto stack** | **ON** | **OFF** | `loeo_retarget.py:132-141` |
| crops | `data/test` (4) | `data/train` fold stems, `TEST_DIR` rebound | `loeo_retarget.py:83` |

> **The brief's mechanism (a) — "LOEO uses per-fold weights, deployment uses public 50-epoch
> weights trained on both embryos" — is FALSE for fold 0.** The support pack ships only
> `weights/unet_transformer/split_0` (`loeo_retarget.py:16-18`), which deployment also uses. Fold 0
> and deployment run the **same primary edge weights**. Only fold 1 overrides them (so the 6bba
> +0.0090 carries an extra confound the 44b6 +0.0144 does not).

The real chain difference is **two ablated components**: the all-199-train secondary model and
the DeepCenter add-only gate, both ON at deployment and OFF in every LOEO strict arm. That
restates (a) as:

> **(a′) REDUNDANCY.** Arm B recovers relinks the secondary model and/or DeepCenter already
> recover at deployment; LOEO strict removes both, opening a gap arm B fills that does not exist
> in the deployed pipeline.

§3 tests (a′) directly and **refutes it in the opposite direction.**

---

## 3. MEASURED — the 2×2, and it refutes every pre-specified mechanism

Because the four `data/test` movies are byte-identical copies of train movies, all four cells can
be scored locally with the official scorer (`scripts/core/score_loeo_submission.py`, GT
`data/train`). Inputs `loeo_{base,armb}_pub4.csv.gz` were assembled by taking the two 44b6 crops
from each fold-0 artifact and the two 6bba crops from each fold-1 artifact, so **every crop is
scored under the fold configuration in which it was held out.**

### Zarr identity check (§ governance)

| stem | files | bytes | sha256(concat) test == train |
|---|---:|---:|:--:|
| `44b6_0113de3b` | 102 | 456,757,564 | ✅ `11c72882d56cdec99269` |
| `44b6_0b24845f` | 102 | 547,662,847 | ✅ `064ed4f5c7b2648bbd86` |
| `6bba_05b6850b` | 102 | 361,668,669 | ✅ `489b7c70a7baeabebedc` |
| `6bba_05db0fb1` | 102 | 540,242,928 | ✅ `00d63fb943731d963afe` |

### The 2×2

|  | **the 4 placeholder crops** | **full LOEO fold (71 / 128)** |
|---|---|---|
| **deployed chain** (secondary + DeepCenter ON, `split_0`) | P3 **0.8907** → armB **0.9272** · **Δ = +0.0365** | *unmeasurable — see §4* |
| **LOEO strict chain** (both ablated, per-fold weights) | base **0.7969** → armB **0.8066** · **Δ = +0.0097** | 44b6 +0.0144 · 6bba +0.0090 |
| **LEADERBOARD** (hidden population) | — | P3 0.915 → armB 0.915 · **Δ = +0.000** |

Per-crop detail (`w_i = edge_tp+edge_fp+edge_fn`, the scorer's pooling weight):

| cell | crop | w | tp | fp | fn | adjJ |
|---|---|---:|---:|---:|---:|---:|
| dep P3 | `6bba_05db0fb1` | 1317 | 1104 | 134 | 79 | 0.8374 |
| dep armB | `6bba_05db0fb1` | 1266 | 1132 | 83 | 51 | **0.8931** |
| dep P3 | `44b6_0113de3b` | 52 | 47 | 2 | 3 | 0.9049 |
| dep armB | `44b6_0113de3b` | 50 | 48 | 0 | 2 | **0.9610** |
| loeo base | `6bba_05db0fb1` | 1309 | 1012 | 126 | 171 | 0.7831 |
| loeo armB | `6bba_05db0fb1` | 1294 | 1022 | 111 | 161 | 0.8000 |

Total annotated edge mass over the four movies ≈ **2,285** — matching the ~2,000 figure already on
record (`research/07-outputs/submissions.md:266`). 44b6 carries only **4.55%** of it.

### The three reads

**Read 1 — POPULATION within train is exonerated.** LOEO-strict on the 4 placeholder crops gives
**+0.0097**; LOEO-strict on the full 71/128 folds gives **+0.0144 / +0.0090**. Same chain,
wildly different crop samples, same answer. Crop sampling within `data/train` does not destroy
the lever.

**Read 2 — CHAIN redundancy (a′) is REFUTED, decisively and in the opposite direction.** On
*identical crops*, the deployed chain shows **+0.0365** versus LOEO-strict's **+0.0097** — the
deployed chain **amplifies** arm B by 3.8×, it does not make it redundant. The secondary model and
DeepCenter do not absorb arm B's contribution. (§4 explains why the amplification is itself an
artifact.)

**Read 3 — the leaderboard does not score this.** Two submissions the LB scores **identically**
(0.915 / 0.915) differ by **+0.0365** when scored locally on the four movies whose images the
submission was generated from. That is a 36× discrepancy against the LB quantum. Confirmed
independently: `is_kernels_submissions_only = True` (Kaggle API — this is a Code Competition), and
the four visible movies are byte-identical train copies, i.e. placeholders. **The scored
population is hidden and unobservable to us.**

### Mechanism (c) — the arithmetic, for completeness

From `vendor/kaggle-cell-tracking/src/tracking_cellmot/metrics.py`: `ADJUSTMENT_ALPHA = 0.1`
(:30), `SCORE_DIVISION_WEIGHT = 0.1` (:34); `summarise` (:471) pools as an edge-mass-weighted mean
of per-crop `J_adj`, `adj_edge_jaccard = Σ w_i·J_adj_i / Σ w_i` (:498-505), and
`score = adj_edge_jaccard + 0.1·division_jaccard` (:522).

Pooling is a **convex combination**, so with per-family deltas +0.0144 (44b6) and +0.0090 (6bba):

| assumed composition | 44b6 edge-mass share | implied Δ_pooled |
|---|---:|---:|
| all 6bba (worst case) | 0.000 | **+0.00900** |
| placeholder-set composition (measured) | 0.0455 | +0.00925 |
| corpus ratio (`experimental-records.md:167`) | 0.1494 | +0.00981 |
| equal mass | 0.500 | +0.01170 |
| all 44b6 | 1.000 | +0.01440 |

> **Mechanism (c) is REFUTED.** The minimum implied delta is **+0.0090 — nine LB quanta.** A true
> +0.0090 on 0.9150 prints `0.924`, not `0.915`. No pooling composition and no rounding boundary
> converts these LOEO deltas into a flat print. For the LB to print `0.915` twice, the true delta
> must be |Δ| < 0.001, at least 9× smaller than the LOEO numbers permit.

**Division sub-mechanism (d), tested and dismissed on this substrate.** Arm B moved test forks
307 → 328 (`armb_provenance.json` `structure.divisions` vs `submissions.md:316`), which I expected
to cost score. It did not: on the annotated division metric both deployed arms score **TP=0 FP=8
FN=3, divJ = 0.0000 — identical**. The division term contributes exactly zero to the delta here.

---

## 4. THE FINDING — the deployed pipeline is unmeasurable on labelled data

The `+0.0365` in the top-left cell is not good news; it is the diagnosis.

The deployed pipeline's secondary model is `unet_transformer_alltrain_seed314159_v1`, whose own
`training_config.json` declares `"train_datasets": 199` — **it was trained on every labelled crop**
(`scripts/kaggle_edits/loeo_retarget.py:19-22`). The four placeholder movies *are* four of those
199 crops. So in the top-left cell, the deployed pipeline is being scored on crops its own
secondary model memorised. The 3.8× amplification over LOEO-strict is that memorisation.

This closes a loop that the project has been circling for weeks:

- **LOEO-strict is clean but is not the deployed system.** It must ablate the secondary model and
  DeepCenter precisely *because* the secondary model saw all the labels
  (`loeo_retarget.py:14-25` states this explicitly, for leakage reasons).
- **The deployed system is measurable on train crops only in a contaminated way.**
- **The hidden test is the only place the deployed system runs honestly — and it returns one
  rounded number per submission slot.**

> **There is no configuration in which the deployed pipeline can be validly measured against
> ground truth.** Every preflight instrument we possess measures a *different pipeline*
> (LOEO-strict) or the *right pipeline on memorised data* (placeholders). The arm-B failure is not
> a fluke of one lever; it is the generic behaviour of this instrument set.

This also reframes the ledger's standing puzzle — five levers that "transferred badly across
substrate or family" (`experimental-records.md:280-287`). They were all measured on the ablated
pipeline and deployed on the full one.

---

## 5. Verdict table

| mechanism | evidence FOR | evidence AGAINST | residual probability |
|---|---|---|---|
| **(a′) chain redundancy** — secondary model + DeepCenter already recover what arm B recovers | Both ON at deployment, OFF in LOEO strict (`loeo_retarget.py:127-141`); they act on the same add/keep decisions | **REFUTED:** on identical crops the deployed chain gives **+0.0365** vs LOEO-strict **+0.0097** — amplification, not redundancy | **< 0.05** |
| **(b) population — "gate doesn't fire on test"** | — | **REFUTED:** churn 7.19% (test) vs 6.42% / 7.69% (folds) | **< 0.05** |
| **(b′) population — hidden test differs from every observable substrate** | LB scores a hidden set (`is_kernels_submissions_only=True`; the 4 visible movies are byte-identical train copies); local scoring of the two submissions gives +0.0365 where the LB gives +0.000 | Cannot be inspected directly; within-train crop sampling was exonerated (Read 1) | **~0.45** |
| **(c) score mechanics** — pooling / rounding hide the gain | LB prints 3 decimals | **REFUTED:** pooling is convex ⇒ Δ_pooled ≥ +0.0090 for any composition = 9 LB quanta | **< 0.05** |
| **(d) division-term sign flip** | forks 307 → 328 at deployment | **REFUTED on this substrate:** both arms score divJ = 0.0000 (TP0/FP8/FN3), contributing exactly 0 | **< 0.05** |
| **(e) INSTRUMENT INVALIDITY — the deployed pipeline cannot be measured on labelled data at all** (§4) | Secondary model trained on all 199 crops ⇒ LOEO must ablate it ⇒ LOEO measures a different pipeline; the placeholder cell is contaminated by memorisation (3.8× inflation) | None found | **~0.50** |

(b′) and (e) are not exclusive — they are the two halves of one situation: the only honest
measurement surface is hidden, and every visible surface is either the wrong pipeline or
memorised. Together they carry ~0.95.

---

## 6. THE cheapest next experiment

### Step 1 — instrument calibration over the existing LB anchors. **Zero GPU, zero slots, ~30 min.**

We hold **five** completed kernels with known LB scores. Fetching a completed kernel's output is
free (no GPU, no slot). **P3 and P3+armB are already measured** (0.8907 and 0.9272), so three
points remain:

| arm | LB | spec / kernel slug | local score |
|---|---:|---|---:|
| P0-A | 0.913 | *(spec deleted)* slug `biohub-p0a-clean-913-repro` | to measure |
| P0-B | 0.914 | `scripts/kaggle_specs/live_p0b.json` → `biohub-p0b-clean-913-reverse-time` | to measure |
| P0-CR | 0.906 | *(spec deleted)* slug `biohub-p0cr-v122-revtime-volguard` | to measure |
| P3 | 0.915 | `scripts/kaggle_specs/p3_harmonic.json` | **0.8907** ✅ |
| P3+armB | 0.915 | `scripts/kaggle_specs/p3_armb.json` | **0.9272** ✅ |

P0-B runs directly off an existing spec:

```
PYTHONUTF8=1 .venv/Scripts/python.exe scripts/core/kaggle_factory.py fetch \
    --spec scripts/kaggle_specs/live_p0b.json --files submission.csv --dest c:/temp/lbcal/p0b
PYTHONUTF8=1 .venv/Scripts/python.exe scripts/core/score_loeo_submission.py \
    --csv c:/temp/lbcal/p0b/submission.csv --gt-dir data/train --json-out c:/temp/lbcal/p0b.json
```

P0-A and P0-CR have no surviving spec file (only `notebooks/kaggle_p0a_clean913/` remains, and
P0-CR is gone entirely). `cmd_fetch` reads only `spec["slug"]`
(`scripts/core/kaggle_factory.py:422-424`), so each needs a two-line throwaway spec
`{"slug": "<slug>"}` — cheaper than reconstructing the build. P0-CR is the most valuable of the
three because it is the only **negative** anchor (0.906), which is what pins the slope.

Then regress local-placeholder score on LB score across the five points.

**What each outcome means:**

- **Rank correlation ≈ 0** (which the arm-B point alone already suggests) → the placeholder movies
  carry **no** LB information. Combined with the LOEO failure, **no currently-available preflight
  instrument is validated**, and the answer to "can any preflight instrument be trusted?" is *no,
  none of the ones we have*. H1 must then be gated as in Step 3.
- **High rank correlation, slope ≪ 1** → instruments are directionally usable with a measured
  discount factor, and arm B is an outlier to be explained rather than a systemic failure. This
  would be the best available outcome and would let H1 keep a LOEO gate, with the slope applied.
- **High correlation, slope ≈ 1** → then the arm-B submission itself is suspect (e.g. the rerun
  failed to apply the gate), and the cheap follow-up is to re-examine `55585140`'s rerun rather
  than the methodology.

This is decisive about the only surviving hypotheses and costs nothing.

### Step 2 — conditional, if Step 1 is uninformative. **2 GPU kernels.**

`loeo_retarget.py:35` **already implements** an arm `hybrid` (`:127`) that disables the
contaminated secondary model but **keeps DeepCenter ON**. Running fold 0 armB-on/off under
`hybrid` instead of `strict` isolates DeepCenter's share of the LOEO↔deployment chain gap at the
cost of one env edit per spec. It cannot fix the secondary-model contamination — nothing can — but
it narrows the unmeasurable residue from two components to one.

### Step 3 — what this means for the H1 retrain design

The instinct would be to gate H1 on a bigger, better LOEO. **That is the wrong lesson.** LOEO
cannot be fixed by more crops or tighter pairing; it is measuring a pipeline that is missing the
two components deployment runs, and it must stay that way because one of them saw every label.

The strategic consequence is the opposite, and it is an argument *for* H1 that has not yet been
made in the portfolio:

> **H1 retrains the model on Zebrahub — external, non-competition data.** If the retrained model
> *replaces* the contaminated `seed314159` secondary rather than sitting alongside it, then for
> the first time the **entire deployed pipeline contains no component trained on the 199 labelled
> crops** — and the whole system becomes honestly measurable on all 199 of them.

That restores the instrument as a by-product of the retrain. It should be weighed alongside H1's
raw score potential, and it argues for designing H1 as a *substitution* of the secondary model,
not an addition. Note the standing limit (`experimental-records.md:455-470`): `zh001r` carries no
track identity and can supervise only the **detector** half; the edge half needs our own
`h1r_fetch_imaging.py` level-1 stream, which does carry track ids.

**Until Step 1 returns, no LOEO-only projection in `bets.yaml` should be treated as calibrated,
and the "+0.005 bilateral LOEO" bar should be regarded as measuring the wrong pipeline rather
than merely being set too low.**

---

## 7. Instrumentation defects to fix before the next lane

1. **`GATE_STATS` is unrecoverable from any LOEO run.** `loeo_export.py:117-124` deletes
   `submission.csv`; the provenance cell runs after it and reads that file first
   (`armb_provenance.py:21`) → `FileNotFoundError`, every time. **Fix:** order the `append_cell`
   provenance edit *before* the `replace_cell` LOEO-export edit, or have provenance read the gzip.
   Without this no LOEO lane can report what its lever actually did.
2. **The LOEO strict ablation is documented but its transfer consequence was never stated.**
   `loeo_retarget.py:14-25` correctly explains *why* the secondary model and DeepCenter must be
   disabled (leakage). It should also record that **any lever measured this way is measured on a
   pipeline the leaderboard never runs**, so a LOEO delta is a lower-bound statement about a
   different system, not a prediction.
3. **Fold 1 confounds weights with the ablation.** `p3_armb_loeo_f1.json` overrides primary
   weights to `split_1` while fold 0 uses the deployed `split_0`. Bilateral claims should note
   that the two folds are not the same kind of contrast.

---

## Appendix — artifacts

All under `c:/temp/armb_diag/` (outside Git, per CLAUDE.md rule 6):

| file | contents |
|---|---|
| `armb_provenance.json` | test-set gate census from kernel `biohub-p3-armb` |
| `run_stats.csv` | per-crop wrapper counters, submission kernel |
| `churn_diff.py` / `churn_diff.json` | per-crop edge-set diffs, test + both folds |
| `p3_harmonic/submission.csv`, `p3_armb/submission.csv` | the two fetched submissions |
| `loeo_base_pub4.csv.gz`, `loeo_armb_pub4.csv.gz` | 4-placeholder-crop subsets of the LOEO artifacts |
| `score_deployed_p3.json`, `score_deployed_armb.json` | 2×2 top row |
| `score_loeo_base_pub4.json`, `score_loeo_armb_pub4.json` | 2×2 bottom row |

Source LOEO artifacts (previous session scratchpad, still present):
`.../eb16434d-b670-4ac0-844c-cbfad2c4dd82/scratchpad/{loeo_f0,loeo_f1,base_f0,base_f1}/`.
