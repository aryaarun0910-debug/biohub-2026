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

**The D1 smoke passed. Audit it, then run D1-F. Do not launch the full corpus first.**

v5 smoke (`biohub-p3-d1-smoke-f0` / `-f1`, both version 5, COMPLETE):

| fold | crop | GT rows | rows | features |
|---|---|---:|---:|---|
| 0 | `44b6_0113de3b` (parity) | 52 | 9,652 | 9,652 × 32 finite |
| 1 | `6bba_57b7cc1e` (stress) | 1,659 | 11,259 | 11,259 × 32 finite |
| 1 | `6bba_6feb10f0` (extreme) | 1,368 | 10,968 | 10,968 × 32 finite |

**Both folds `COMPLETE=True`, 3/3, zero aggregator problems.**

Remaining acceptance checks (not yet run): M+C+T+L+D == all GT · C+T+L+D == scorer-unmatched
· all sphere distances ≤ 15 µm · exact peak/pregraph parity · graph invariants · no unexplained
files. The parity gate (52/52, `node_recall 1.000000`) was already proven locally against v4.

Then **D1-F** (`scripts/d1f_probe.py`) is the actual encoder-vs-head verdict. **Class counts are
diagnostic only and must never authorise encoder retraining on their own.**

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
