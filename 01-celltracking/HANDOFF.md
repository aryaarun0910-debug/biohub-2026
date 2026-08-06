# Current handoff — cold start

**Updated:** 2026-08-06 · **Branch:** `master` · **Best public score: 0.915** (P3 harmonic)
· leader **0.948** · gap **0.033** · slots consumed **11**

## Verify state first

```powershell
git status                                                  # only .claude/settings.json + .gitignore dirty
git rev-list --left-right --count origin/master...master    # expect 0 0
.\.venv\Scripts\python.exe -m pytest -q                     # expect 90
.\.venv\Scripts\python.exe scripts\claims_table.py --check  # expect 53
```

## Read in order

1. This file
2. `reports/ROADMAP_DETECTION_ADDENDUM.md` — **the live plan.** Supersedes `ROADMAP_DETECTION.md`
3. `reports/CYCLE3_LANE_O_AND_D0.md` — why the submission bar is +0.020
4. `reports/CYCLE2_SUBSTRATE_CORRECTION.md` — the substrate rule
5. `reports/ENVIRONMENT_TRAPS.md` — 24 defects; **I walked into trap 1 anyway**

---

## 1. THE ONE THING TO DO NEXT

**Build the v6 TTA-consistent feature export. Read `reports/D1_V5_STATUS.md` first.**

v5 is **not** clean. Three blockers were found on 2026-08-06; the full-199 export was **not**
launched and `scripts/d1f_probe.py` was **never run**. Corrected triage:

| area | status |
|---|---|
| kernel wiring, immutable manifests, aggregation, raw statistics, feature serialization | **PASS** |
| M/C/T/L/D partition | **NOT RUN in v5** — now derived CPU-side by `scripts/d1_postprocess.py` |
| D1-F representation-vs-head | **INVALID / UNPROVEN** |

**The fatal blocker (B3):** the deployed TTA loop averages **only detection logits** and
**discards every flipped feature map**; `unet_out` is bound once from the identity view
(`predict_unet_transformer.py:372-388`). v5 therefore paired post-TTA logits with
identity-view features, so `checkpoint_detect_head(feat)` cannot reproduce the deployed
logit and **H0 parity is mathematically unavailable**. Damage is bounded: neighbourhood
statistics come from the post-TTA logits, so **the partition is sound and only the 32-D
features are contaminated.**

**The deployed view count is 8, not 4.** The vendored `/4` block is *replaced at build time*
by a full D4 set (identity + 3 flips + rot90 k=1,3 + transpose + anti-transpose, `/_nv`).
**Read the generated notebook, never `vendor/` — the base file is not the run.** The patch
guard is a `print`, not a `raise`, and the view count is in no manifest field. There are
**two** such blocks (primary + secondary detection model).

**A second launch-blocker (B4), found by the red team and reproduced:** `_d1_flush` used
polars' default 100-row schema inference on heterogeneous dicts. With 96 non-GT rows emitted
per frame, any crop whose first GT lands at frame ≥ 2 **silently loses every gt-only column**
while still reporting `gt_rows` and `status: complete`. **28 of 199 crops (14.1%)** would
have been destroyed; all three smoke crops start at frame 0 so the smoke could not catch it.
Fixed with `infer_schema_length=None` + a hard column contract.

The derived partition over the 3-crop smoke (pregraph authority, reproduces **52/52** on
`44b6_0113de3b`): GT 3,027 · **M 1,469 · C 289 · T 849 · L 420 · D 0**. Of 1,558 unmatched:
C 18.5%, T 54.5%, L 27.0%, **D 0.0%**. Basis: 3-crop smoke, IN-FAMILY, **diagnostic only** —
one crop was deliberately chosen as extreme. Class counts still may never authorise encoder
retraining on their own.

Order of work: v6 export → rewrite `d1f_probe.py` with a capability registry → re-run the
3-crop smoke as v6 → **only then** one combined full-199 launch. Do not run full v5 and then
repeat 199 crops for v6.

---

## 2. What the instrument now is

`M/C/T/L/D` over all GT, in `src/biotrack/d1_partition.py`:

- **M** matched by the **scorer's own** bipartite matching (`MATCH_UM` asserted == `MAX_DISTANCE`)
- then for unmatched only: **C** accepted peak ≤7 µm · **T** unaccepted local max ≤7 µm ·
  **L** maximum only in (7,15] µm · **D** nothing within 15 µm

The old exact-GT-voxel A/B/D is **demoted** to `voxel_*` columns and drives nothing. v4 settled
this empirically: `44b6_0113de3b` matched **52/52** GT while the exact-voxel rule called **26 of
those same GT "suppressed"**. An accepted peak one grid voxel away matches correctly.

---

## 3. Five smoke versions, five defects — all now locked by tests

| v | defect | lock |
|---|---|---|
| 1 | `det_logits[f_idx]` is (1,1,Z,Y,X); passing it un-indexed gave max_pool3d a 6D input | injection sim |
| 2 | LOEO mounts only `.zarr` — kernel was label-blind by design, `gt_rows=0` | GT resolver + aggregator raises on zero GT |
| 3 | `tracksdata.io.load_geff` **does not exist** | `tests/test_external_api_contract.py` |
| 4 | 15 µm **cube** → 25.98 µm corners; shared mutable manifest lost a crop | sphere mask + `tests/test_d1_manifest_aggregation.py` |
| 4 | exact-voxel A/B/D was the wrong causal unit | `tests/test_d1_partition.py` (14 cases) |

**The rule that came out of this: no external API may be pushed to a kernel until it has been
executed locally against `data/train`.** v3 cost a GPU run on a call verifiable in seconds.
`tests/test_external_api_contract.py` is now a mandatory pre-push gate.

---

## 4. Numbers that are settled — do not re-derive

| | |
|---|---|
| P3 harmonic public | **0.915** (+0.001 over P0-B) |
| arm B OOF, correct substrate | **+0.0084877** (44b6 +0.0139, 6bba +0.0077, CI [+0.00745, +0.01365]) |
| arm B public | **0.914** — flat |
| detection share of edge FN | **72.2%**, oracle **+0.10332** |
| association ceiling | **+0.032931** |
| node-ratio bonus at risk | **−0.008633** |
| missing GT nodes | 15,296 (11.47%); 44b6 **1.37%** vs 6bba **13.28%** |
| detector loss | `det_loss_weight=1.0`, `det_neg_weight=0.01` (retained training_config `4f29349439e133ad`) |
| target-voxel collisions | **0 of 133,318** |
| perfect-heatmap recall ceiling | **1.0000** — Gaussian-target line is dead |
| detector grid | isotropic **1.625 µm**; pool (3,3,3) = ±1.625 µm; `det_threshold` 0.96875 = logit 3.434 |

---

## 5. Standing rules that were each learned the hard way

- **Substrate transfer has failed 7×.** Re-measure every constant on the substrate it is applied
  to. `p0strict_cache` is POST-wrapper and cannot carry a pre-wrapper change
  (`wsf_ROUTE1_VERDICT.json`) — I ignored that and lost a cycle.
- **Trust OOF ranking, not magnitude.** arm B: +0.0085 OOF → 0.000 public.
- **Predict with churn-vs-null, not cardinality.** arm B churned 7.148% → 0.000; P3 4.368% → +0.001.
- **Oracle-clears/selector-fails has occurred 7×.** Every selector needs a base-rate argument first.
- Research promotion **≥ +0.015**; submission **≥ +0.020**.
- **Never `git add -A` or `git add -u`** — stage explicit paths. I did this once and swept
  `.claude/settings.json`, `.gitignore` and a Kaggle output into a commit.
- Kaggle outputs are archived to `../Biohub-CellTracking-2026_RESEARCH/artifacts/`, never committed.

## 6. Blocked / not started

- **C0-FULL / C1** — association. Must re-base onto P3, not P0-B. `reports/OVERNIGHT_RUNBOOK.md`.
- **M2-CTPU** — masked loss + training-only count prior + temporal pseudo-positives. Design in the
  addendum; H5 (positive-exposure control) is mandatory because density and training-set size are
  confounded.
- **JEPA** — deferred to JEPA-lite, only if CTPU plateaus.
