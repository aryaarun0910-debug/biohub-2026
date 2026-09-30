# Instrument repair — calibrating every preflight instrument we hold against the leaderboard

**Verdict: no preflight instrument we hold is validated. The leaderboard is the only calibrated
measurement surface, and it is also the cheapest one per decision.**

Everything in §1–§4 is MEASURED this session (2026-08-18, zero GPU, zero submission slots).
§5–§7 are INFERENCE and are labelled as such.

---

## 1. CALIBRATION — the headline result

Six configurations whose true leaderboard scores are known were re-fetched from their completed
kernels and re-scored locally with the official patched scorer
(`scripts/core/score_loeo_submission.py`, GT `data/train`) on the four `data/test` placeholder
crops. Four were newly fetched and scored today; two were already on record from
`loeo_lb_gap_2026-08-18.md`.

| config | LB (public) | local, placeholder-4 | submission sha256 | bytes |
|---|---:|---:|---|---:|
| P0-CR | **0.906** | 0.890430 | `435bf5d19d85563c…` | 12,343,722 |
| P0-A | **0.913** | 0.890444 | `8c1605b5944d25e4…` | 12,350,156 |
| P0-B | **0.914** | 0.889225 | `4c285cae0c220a11…` | 12,359,118 |
| arm B solo | **0.914** | 0.927784 | `83498f9e27d44532…` | 12,382,751 |
| P3 harmonic | **0.915** | 0.890692 | `3e98739f5c46f2dd…` | 12,486,220 |
| P3 + arm B | **0.915** | 0.927230 | `5b37524d96f522667…` | 12,510,115 |

Every fetched hash reproduces `research/07-outputs/submissions.md` exactly. `p2_armb_baseline`
was fetched as a control and is **byte-identical** to P0-B (`4c285cae…`), independently
re-confirming the ledger's "GATE 1" claim.

### 1a. Regression, n = 6

```
Spearman rho (LB vs local)      = +0.500   exact two-sided permutation p = 0.333
Kendall tau-b                   = +0.358   (9 concordant, 4 discordant, 15 pairs)
Pearson r                       = +0.371
OLS  LB ~ local  : slope = +0.0659,  intercept = 0.8533
OLS  local ~ LB  : slope = +2.08
LB range = 0.009        local range = 0.0386
```

**Read: rank correlation is not distinguishable from zero at n = 6 (p = 0.33).** Even taking the
point estimate at face value, the slope is **0.066** — a local move of +0.038 buys +0.0025 on the
leaderboard, a 15:1 discount. That is not "usable with a measured discount"; it is a coin-flip
with a discount attached.

### 1b. The structure of the failure is worse than the summary statistic

Split the six points:

| subset | n | Spearman | Pearson | local range | LB range |
|---|---:|---:|---:|---:|---:|
| all | 6 | +0.500 | +0.371 | 0.0386 | 0.009 |
| **excluding the two arm-B configs** | 4 | +0.400 | **−0.199** | **0.0015** | **0.009** |

The four non-arm-B configs span the **entire** known LB range (0.906 → 0.915, nine quanta) and the
local instrument moves **0.0015** across them — with a *negative* Pearson correlation. The whole of
the apparent positive rank correlation at n = 6 comes from the two arm-B configs happening to sit
at the top of the LB ordering (0.914 / 0.915) while sitting +0.038 high locally. They are +0 to +1
quantum on the LB and +38 quanta locally. That is a coincidence of ordering, not signal.

### 1c. Paired contrasts — the form the instrument is actually used in

An instrument is used to gate a *lever*, i.e. to sign a paired delta on a fixed base. Five such
contrasts exist across the six anchors. 95 % CI is a crop-level paired bootstrap, 20,000 resamples
over the four placeholder crops, recomputing the full edge-mass-weighted pooled statistic each time.

| contrast | Δ LB | Δ local | 95 % CI (crop bootstrap) | sign call |
|---|---:|---:|---|---|
| P0-A → P0-B (reverse-time on clean913) | **+0.001** | −0.00122 | [−0.00135, +0.00029] | **WRONG SIGN** |
| P0-A → P0-CR (base swap to v122) | **−0.007** | −0.00001 | [−0.01312, +0.03802] | vacuous (500× too small, CI spans zero) |
| P0-B → arm B solo | **+0.000** | +0.03856 | [+0.00603, +0.05917] | **false positive, "significant"** |
| P0-B → P3 harmonic | **+0.001** | +0.00147 | [−0.00080, +0.00197] | right sign, CI spans zero |
| P3 → P3 + arm B | **+0.000** | +0.03654 | [+0.00619, +0.05575] | **false positive, "significant"** |

```
paired-delta OLS  dLB ~ dLOCAL : slope = +0.0448   Pearson r = +0.271   (n = 5)
```

**One of five contrasts is called correctly and it is the one whose CI spans zero. Two are
confidently-signed false positives. One is confidently wrong. One is vacuous.** The placeholder-4
local score is not a weak instrument; on paired contrasts it is anti-informative.

### 1d. Calibration of the *other* instrument, LOEO — n = 1

The only paired LOEO measurement with a matching LB outcome is arm B:

| instrument | Δ measured | Δ LB |
|---|---|---:|
| paired deployment-substrate LOEO, official scorer | +0.0144 (44b6) / +0.0090 (6bba) | **+0.000** |

n = 1, and it cannot be increased without GPU kernels (P0-A / P0-B / P0-CR have never been run
through LOEO, and doing so costs two kernels each). **LOEO has one calibration point and it is a
miss.**

> **Conclusion of §1: the outcome the brief called "~zero rank correlation ⇒ no instrument we hold
> is validated" is the one that obtains.** The placeholder-4 substrate is retired as an instrument
> effective immediately. LOEO retains one calibration point, which is a failure.

---

## 2. CONTAMINATION — verified, and two findings the diagnosis did not have

The claim under test: `pilkwang/biohub-temporal-unet3d-seed314159-v1` was trained on all 199
labelled crops. **Confirmed, and by a stronger artifact than the one previously cited.**

### 2a. The secondary model — VERIFIED, with the placeholder crops named explicitly

`weights/unet_transformer/split_0/training_config.json`:

```json
{ "method": "unet_transformer_alltrain_seed314159_v1",
  "splits_file": "dataset_splits_alltrain.json",
  "splits_file_sha256": "d1d52843503f014400c22541d27db992ef94c42dfd71b91beb6e09f660acc179",
  "train_datasets": 199, "validation_datasets": 40,
  "base_seed": 314159, "epochs_target": 500 }
```

`weights/unet_transformer/split_0/split_manifest.json` goes further — it enumerates the training
stems by name:

| check | measured |
|---|---|
| `train` list length | **199** (71 × `44b6`, 128 × `6bba`) — i.e. every labelled crop |
| `44b6_0113de3b` in `train` | **True** |
| `44b6_0b24845f` in `train` | **True** |
| `6bba_05b6850b` in `train` | **True** |
| `6bba_05db0fb1` in `train` | **True** |
| `test` list length | 40 |
| `|train ∩ test|` | **40** — the "validation" set is a strict subset of the training set |

`SNAPSHOT_MANIFEST.json`: `best_epoch 381`, `best_score 0.9779747766406395`, snapshot at epoch 400.

**The diagnosis in `loeo_lb_gap_2026-08-18.md` §4 is correct and is now sourced to the artifact
rather than to a code comment.** All four placeholder movies are named in this model's training
manifest, and the model's own reported validation score of 0.978 was measured on 40 crops it had
trained on.

### 2b. NEW — DeepCenter is trained on `44b6` only, and is CLEAN on `6bba`

`pilkwang/biohub-deepcenter-unet3d-center-prior-v1`,
`weights/full_frame_center/split_manifest.json`:

| field | measured |
|---|---|
| `all` | 199 |
| `train` | **71 — all `44b6`**, includes `44b6_0113de3b` and `44b6_0b24845f` |
| `val` | **128 — all `6bba`**, includes `6bba_05b6850b` and `6bba_05db0fb1` |
| `val_fraction` | 0.10 · `seed` 2026 |

Confirmed against the packaged trainer, `source_scripts/train_full_frame_center_detector.py:416-419`:
the split is **by embryo prefix**, `n_val = max(1, round(n_embryos × 0.10)) = max(1, 0) = 1`, so one
whole embryo becomes validation. That embryo was `6bba`.

> `loeo_retarget.py:23` states "DeepCenter has no fold variant either" and ablates it on **both**
> folds. That is over-conservative on fold 1: **DeepCenter never saw a `6bba` crop**, so on fold 1
> (held-out `6bba`) it is out-of-sample and could legitimately stay ON. Ablating it there removes a
> deployed component from the measurement for no leakage reason.

### 2c. NEW — the PRIMARY model's training set is undocumented in its own artifact

Full file listing of `pilkwang/biohub-tracking-support-pack-50ep-v1` under
`weights/unet_transformer/split_0/` (87 files total in the dataset):

```
weights/unet_transformer/split_0/checkpoint_last.pth
weights/unet_transformer/split_0/config.json
weights/unet_transformer/split_0/edge_predictor_best.pth
```

**No `training_config.json`. No `split_manifest.json`. No `history.csv`.** Its `ARTIFACT_MANIFEST.json`
records `artifact_name: "biohub-tracking-support-pack-400ep-snapshot-v1"` (the dataset slug says
50ep), `source: "public learned baseline artifact, repackaged locally"`, and weight sha256
`12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771` (8,363,159 B) — and nothing
about which crops it saw. The seed314159 pack's `edge_predictor_best.pth` is a *different* weight
(`9bac2fa0dadc4a6f…`, same 8,363,159 B, same architecture config), so these are genuinely two
models.

`lab-notebooks.md:3174` and `:3332` already flag this ("the only evidence that 44b6 was held out is
the directory name `split_0`"). This audit confirms it from the artifact side: **the claim
"split_0 was trained on 6bba with 44b6 held out" has no documentary basis whatsoever.**

### 2d. Consequence — there is no labelled subset on which the deployed chain is clean

| component | deployed? | training set | clean on `44b6`? | clean on `6bba`? |
|---|---|---|---|---|
| primary — pack `split_0` | ON | **undocumented** (assumed `6bba`) | assumed yes | assumed **no** |
| secondary — `seed314159` | ON | **199 = all crops (verified)** | **no** | **no** |
| DeepCenter — `full_frame_center` | ON | **71 = all `44b6` (verified)** | **no** | **yes** |

Every row of `data/train` is contaminated for at least one deployed component. **§4 of
`loeo_lb_gap_2026-08-18.md` stands: the deployed pipeline cannot be validly measured against any
ground truth we hold.** The new detail is that the contamination is *layered by family in opposite
directions*, so it cannot be dodged by picking crops.

---

## 3. POWER ANALYSIS — the minimum detectable effect on every substrate

Convention: two-sided α = 0.05, 80 % power ⇒ **MDE = (z₀.₉₇₅ + z₀.₈₀) × SE = 2.8016 × SE**, and
SE = h₉₅ / 1.95996 where h₉₅ is the half-width of a measured 95 % bootstrap CI.

### 3a. Measured noise floors and their MDEs

| substrate | source of h₉₅ | h₉₅ | SE | **MDE @ 80 %** |
|---|---|---:|---:|---:|
| LOEO fold 0 (`44b6`, 71 crops) | crop bootstrap, 4,000 resamples (`operating_point_2026-08-18.md` §0.1b) | 0.0030 | 0.00153 | **±0.0043** |
| LOEO fold 1 (`6bba`, 128 crops) | same | 0.0040 | 0.00204 | **±0.0057** |
| LOEO pooled by edge mass | analytic from the two above, folds independent | — | 0.00176 | **±0.0049** |
| **bilateral (must clear BOTH folds)** | governed by the weaker fold | — | — | **±0.0057** |
| **placeholder-4 (4 crops, deployed chain)** | crop bootstrap, 20,000 resamples, **measured today** | 0.0248 | 0.01264 | **±0.0354** |

### 3b. Why pooling folds does not rescue power

Fold edge-mass and crop-weight structure, measured from
`research/06-knowledge-system/inventory/loeo_f{0,1}_strict.json`:

| fold | crops | W = Σwᵢ | median wᵢ | min | max | top-10 crops' share of W | Kish effective n |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (`44b6`) | 71 | **21,222** | 226 | 49 | 1,407 | **37.4 %** | **42.4** of 71 |
| 1 (`6bba`) | 128 | **122,779** | 899 | 227 | 2,054 | 14.4 % | 111.5 of 128 |

(Handoff quotes 21,210 / 123,413 for a slightly different arm; the difference is < 0.6 % and does
not move any conclusion.)

Pooling the folds by edge mass gives MDE **±0.0049 — worse than fold 0 alone (±0.0043)**, because
fold 1 carries 85 % of the mass and has the *larger* measured per-crop delta dispersion. And
CLAUDE.md rule 3 requires both embryo directions reported separately, so the operative bar is the
bilateral one: **±0.0057**.

Fold 0's weakness is not fixable by adding crops — it already uses all 71 `44b6` crops. It is
annotation sparsity (`44b6` is 0.771 % annotated, 19,826 GT edges over 71 crops), so W is fixed at
~21 k and the top 10 crops carry 37 % of it. **There is no statistical fix available.**

### 3c. THE HARD GATE — put this number in the handoff

> **The standing "+0.005 bilateral LOEO" promotion bar sits BELOW the fold-1 detection limit of
> +0.0057.** It was never a valid gate. A lever printing exactly +0.005 on fold 1 has under 80 %
> chance of being distinguishable from zero on the substrate that measured it — before any question
> of whether that substrate predicts the leaderboard.
>
> - **LOEO, if used at all: bilateral MDE = +0.0057.** Nothing below this is measurable.
> - **Placeholder-4 local score: MDE = ±0.0354.** Every difference in §1 except the two arm-B
>   contrasts is inside this instrument's own noise. **Retired.**
> - **Leaderboard: resolution 0.001, no sampling noise** (fixed hidden set; see §5a caveat).

---

## 4. MEASURED — the sensitivity comparison that decides the design

| instrument | measures | MDE / resolution | cost per paired decision |
|---|---|---:|---|
| placeholder-4 local | deployed chain, on memorised data | ±0.0354 | free — and anti-informative (§1c) |
| LOEO bilateral | a chain missing 2 deployed components | ±0.0057 | 2–4 GPU kernels ≈ 8 GPU-h of a 30 h/wk quota ⇒ **3–4 tests/week** |
| **leaderboard** | **the shipped pipeline on the scored population** | **0.001** | **1 slot of 5/day ⇒ 35 tests/week** |

**The leaderboard is 5.7× more sensitive than the bilateral LOEO bar, measures the right pipeline,
and delivers ~10× more decisions per week.** The standing rule in `submissions.md` — *"never spend
a submission to resolve an effect smaller than ~0.005 … the LB is a smoke test for large effects;
OOF is the instrument"* — is inverted by this table and by §1. It should be struck.

---

## 5. INFERENCE — the four design options, evaluated

### (a) Drop the contaminated components from the SHIPPED pipeline

Mechanically cheap: `loeo_retarget.py:127-141` already shows the toggle is a plain env set
(`BIOHUB_SECONDARY_WEIGHTS=""`, secondary edge/detection weights → 0, four `DEEPCENTER_*` flags →
0). Applying the same env to the deployment spec makes shipped == LOEO-strict, so fold 0 would
measure exactly what ships.

- **Cost:** 1 Kaggle GPU kernel (P0-A ran in 1,522 s on T4×2) + **1 slot** to re-anchor, plus an
  unknown permanent loss of absolute score — we are deleting two components that the public
  baseline ships because they help.
- **Fatal objection (from §2c):** even after both ablations, the surviving primary's training set is
  **undocumented**. We would pay real score for an instrument that is still only *presumptively*
  clean, resting on a directory name. And we would still be capped at MDE ±0.0043 / ±0.0057 — worse
  than the leaderboard's 0.001.
- **Verdict: not worth it as a primary strategy.** Keep it in reserve as the fallback if §5d's
  step-0 determinism probe fails.

### (b) Leave-crop-out within an embryo

**Invalid, and provably so.** The secondary's `split_manifest.json` names all 199 crops in `train`
(§2a). Holding out any crop leaves a model that has already memorised it. Leave-crop-out cannot
decontaminate a model whose training set is the universe. It would be valid only for a model with a
documented held-out family — which is DeepCenter on `6bba` (§2b) and, presumptively, the primary on
`44b6`. **Reject as an instrument; adopt only the narrow correction that DeepCenter may stay ON in
LOEO fold 1.**

### (c) Statistical power fixes

- **Pooling folds: measured to make it worse** (§3b, ±0.0049 vs fold 0's ±0.0043), and it violates
  the bilateral reporting rule.
- **Paired bootstrap / blocking by crop: already in use** — the ±0.003/±0.004 floors are already
  paired crop bootstraps. There is no unexploited variance reduction here.
- **More crops: impossible.** 199 is the entire labelled corpus and both folds already use all of it.
- **Best achievable ≈ ±0.004**, on the wrong pipeline. **Reject as a solution;** the numbers are
  worth keeping only as the gate in §3c.

### (d) Accept the leaderboard as the only instrument — **RECOMMENDED**

The leaderboard is a **deterministic** function of the submitted kernel over a **fixed** hidden set.
Between two submissions there is no sampling noise — only quantisation to 0.001. Contrast LOEO,
where every delta carries ±0.0057 of crop-resampling noise *and* a chain mismatch of unknown sign.

**Protocol.**

| step | action | cost |
|---|---|---|
| **0** | **Determinism probe.** Resubmit the reigning best kernel (`biohub-p3-armb` or `biohub-p3-harmonic`) unchanged. If it returns 0.915 exactly, every subsequent ΔLB is exact to ±1 quantum and the instrument is characterised. If it does not, we have measured the rerun variance and the whole protocol must widen. | **1 slot** |
| 1 | One lever per submission, single toggle against the reigning best. No compound arms. | 1 slot each |
| 2 | Before each slot, record: artifact sha256, node/edge/division counts, structural audit, and **edge churn %** against the base. Churn ≠ 0 proves the lever fired; churn is a *build* check, never a score claim. | free |
| 3 | Pre-register a predicted sign and band. LOEO/local numbers may inform the prediction but **may not gate it**. | free |
| 4 | Promotion: **≥ +0.002 (two quanta) = adopted. +0.001 = provisional**, must reproduce on a second base before adoption (the P0-B/P0-CR precedent: reverse-time was +0.001 on one base and −0.002 on another). **0.000 = killed.** | — |
| 5 | Reserve ≥ 25 slots for the final week; never run the queue below that. | — |

**Budget fit.** ~185 slots remain to 2026-09-29 at 5/day. At 1 lever per slot with ~30 % spent on
reruns, sanity checks and build failures, that is **≈ 120 lever tests** — an order of magnitude more
than LOEO's 3–4/week (≈ 25 total). Kaggle GPU quota (30 h/wk) is then spent on *building* candidate
kernels rather than on measuring them.

**Failure modes, named.**

1. **Public/private overfit.** Selecting on public across N decisions inflates public over private.
   *Quantification (INFERENCE, stated assumption):* the three recorded structurally-large contrasts
   moved the public LB by ≤ 1 quantum (arm B churned 7.19 % of edges for 0.000; P3+armB 7.19 % for
   0.000; P0-A→P0-B for +0.001), which bounds the public-minus-private noise σ at roughly ≤ 0.001.
   Over ~40 selection decisions, expected inflation ≈ σ·E[max of 40 standard normals] ≈ 0.001 × 2.16
   ≈ **0.002**, i.e. about two quanta of shakedown. Acceptable, and small next to the 0.035 gap to
   the leader. This assumes the split is random over crops; if it is split by embryo the estimate is
   worthless and the risk is larger.
2. **No diagnostics.** A 0.000 tells you nothing about *why*. Mitigation: step 2's churn/provenance
   census, which is what `GATE_STATS` exists for — and which is currently broken (see §6).
3. **Latency, not slots, may bind.** P0-CR was PENDING ~4.5 h; `55585140` ≥ 1.5 h. Five slots a day
   is achievable only if kernels are queued in parallel and results are not serialised behind each
   other.
4. **A rerun failure burns a slot with no information.** Mitigation: every kernel must pass the
   structural audit locally on its `data/test` output before submission — the audit is free and has
   already caught one out-of-volume node (P0-C).

**What LOEO is demoted to.** A *build-correctness smoke*: did the lever fire, are the degree
invariants intact, is the churn in the expected range. Those are software contracts (CLAUDE.md rule
4) and belong in a smoke, not a promotion gate. **No LOEO delta may promote anything again.**

---

## 6. Instrumentation defects still open

1. **`GATE_STATS` is unrecoverable from any LOEO run.** `loeo_export.py:117-124` deletes
   `submission.csv`; the provenance cell appended afterwards
   (`p3_armb_loeo_f0.json:136` → `armb_provenance.py:21`) opens that file first and dies. **Fix:
   order the `append_cell` provenance edit before the `replace_cell` LOEO-export edit.** This is now
   load-bearing for step 2 of the §5d protocol, not a nicety.
2. **`loeo_retarget.py:23` over-ablates DeepCenter on fold 1** (§2b). DeepCenter never saw `6bba`.
   The comment "DeepCenter has no fold variant either" should be corrected to record the measured
   `train` = 71 × `44b6` / `val` = 128 × `6bba` split.
3. **The primary's provenance is undocumented** (§2c) and should be recorded as such in
   `research/04-data/` so it stops being quoted as a fact. Any claim of "LOEO-clean on fold 0"
   inherits this assumption.
4. **Fold 1 confounds primary weights with the ablation** — `p3_armb_loeo_f1.json` mounts
   `aryaarun07/biohub-oof-weights` and overrides to `split_1`, while fold 0 uses the deployed pack
   `split_0`. The two folds are not the same kind of contrast and bilateral claims must say so.

---

## 7. What to change in the research machine

| target | change |
|---|---|
| `research/00-system/handoff.md` | Replace the "+0.005 bilateral LOEO" bar with the §3c hard gate. Record that no preflight instrument is validated (§1) and that the LB is the promotion gate (§5d). |
| `research/07-outputs/submissions.md` | Strike the standing rule *"never spend a submission to resolve an effect smaller than ~0.005 … OOF is the instrument."* §4 inverts it. Replace with the §5d step-4 promotion ladder. |
| `research/01-research-direction/bets.yaml` | Every LOEO-only projection is uncalibrated (§1d, n = 1, a miss). Mark them as such rather than discounting them by a slope — there is no measured slope to apply. |
| `research/06-knowledge-system/experimental-records.md` | One row for this calibration: n = 6, Spearman +0.500 (p = 0.33), paired slope +0.045, placeholder-4 substrate RETIRED. |
| `research/04-data/data-governance.md` | Record §2d: the layered, family-opposed contamination map, and that no labelled subset is clean for the deployed chain. |

**Standing consequence for H1.** The argument in `loeo_lb_gap_2026-08-18.md` §6 step 3 survives and
strengthens: if an H1 Zebrahub retrain **substitutes** the `seed314159` secondary rather than
sitting alongside it, the deployed chain loses its only verified all-199 component. It would still
carry DeepCenter (contaminated on `44b6`) and a primary of undocumented provenance, so it does not
by itself restore a clean instrument — but it removes the one contamination we have *proved*. H1
should still be designed as a substitution.

---

## Appendix A — provenance of every number

**Fetched today** (free; `cmd_fetch` reads only `spec["slug"]`, so throwaway specs were used for the
kernels whose specs no longer exist):

```
PYTHONUTF8=1 .venv/Scripts/python.exe scripts/core/kaggle_factory.py fetch \
    --spec c:/temp/lbcal/spec_<n>.json --files submission.csv --dest c:/temp/lbcal/<n>
PYTHONUTF8=1 .venv/Scripts/python.exe scripts/core/score_loeo_submission.py \
    --csv c:/temp/lbcal/<n>/submission.csv --gt-dir data/train \
    --json-out c:/temp/lbcal/score_<n>.json
```

| n | slug |
|---|---|
| `p0a` | `aryaarun07/biohub-p0a-clean-913-repro` |
| `p0b` | `aryaarun07/biohub-p0b-clean-913-reverse-time` |
| `p0cr` | `aryaarun07/biohub-p0cr-v122-revtime-volguard` |
| `armb_solo` | `aryaarun07/biohub-p2-armb-flowgate` |
| `armb_base` | `aryaarun07/biohub-p2-armb-baseline` (control; byte-identical to `p0b`) |

Artifacts, all outside Git per CLAUDE.md rule 6:

| path | contents |
|---|---|
| `c:/temp/lbcal/{p0a,p0b,p0cr,armb_solo,armb_base}/submission.csv` | fetched submissions |
| `c:/temp/lbcal/score_{p0a,p0b,p0cr,armb_solo}.json` | official-scorer output, per-crop |
| `c:/temp/lbcal/ds/seed314159/{training_config,split_manifest,SNAPSHOT_MANIFEST,ARTIFACT_MANIFEST,README}.json` | §2a evidence |
| `c:/temp/lbcal/ds/dc/{split_manifest,config,SNAPSHOT_MANIFEST}.json`, `train_full_frame_center_detector.py` | §2b evidence |
| `c:/temp/lbcal/ds/pack/{ARTIFACT_MANIFEST,config}.json` | §2c evidence |
| `c:/temp/armb_diag/score_deployed_{p3,armb}.json` | pre-existing, from `loeo_lb_gap_2026-08-18.md` |

**Reused from the record:** LB scores from `research/07-outputs/submissions.md`; noise floors
±0.003 / ±0.004 from `research/06-knowledge-system/internal-reports/operating_point_2026-08-18.md`
§0.1b (crop bootstrap, 4,000 resamples); per-crop weights from
`research/06-knowledge-system/inventory/loeo_f{0,1}_strict.json`; chain composition from
`scripts/kaggle_edits/loeo_retarget.py:100-141`.

**Competition metadata** (Kaggle API, 2026-08-18): `max_daily_submissions = 5`,
`deadline = 2026-09-29 23:59`, `is_kernels_submissions_only = True`, `team_count = 2486`,
`reward = 60,000 USD`. Kaggle does not expose the public/private split ratio through the API; §5d
failure mode 1 is therefore INFERENCE.

## Appendix B — constraints honoured

No GPU launched. No submission made. No commit. `.claude/settings.json` untouched. The four
placeholder movies were used **only** to calibrate the instrument, never to select, tune, threshold
or promote anything — and the result of that calibration is that they must never be used for
selection at all.
