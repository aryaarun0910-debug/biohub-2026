# Corpus C/T/L/D census and the re-acceptance ceiling

**Date:** 2026-08-07 · **Lane:** R2-P1 · **Basis:** `exact-pooled-OOF`, 199/199 crops, both
families, CPU only. Full verdict and raw outputs live outside Git at
`<RESEARCH>/agent_runs/lean_2026-08-07/census/` (`VERDICT.md`, `code/`, `out/`).

Supersedes every 3-crop and 60-crop ceiling figure quoted before this date.

## Reachability

Corpus-wide **`T`** (unmatched GT with a *rejected* local maximum inside 7 µm) is **not
CPU-reachable**: it needs the detector heatmap, and no logit/probability volume exists on disk
for any crop. `T` is therefore reported only on the 3 GPU-audited crops. Everything else below
is corpus-wide and scorer-exact.

## Method

Matching is the scorer's own rule, taken from `biotrack.metric_numpy._match_timepoint`
(`MAX_DISTANCE = 7.0` µm, `DEFAULT_SCALE = (1.625, 0.40625, 0.40625)`): per timepoint, a
**maximum-weight** one-to-one assignment on `w = 1/(1+d)` over in-gate pairs. This is **not**
minimum-total-distance, which is what the prior Gate A `missing_mask` used; that was corrected
here. The census matcher agrees node-for-node with `biotrack.metric_numpy.match_nodes` on 20
randomly drawn crops (0 mismatches). Partition sums are asserted per crop and corpus-wide.

Substrate: `artifacts/kaggle/coupled_cache/det/<stem>__tta-4view__det-0.969.npz`, `coords` in
`(t, z, y, x)` full-resolution voxel units (verified by y/x swap test and against the known
`44b6_0113de3b` 52/52 result).

**`clean903_wrapper_oof_cache` is from a superseded detector.** Its own status files declare
`"clean903 wrapper on frozen 0.990 detections"`; the deployed threshold is 0.969 and the 0.99
node set is a strict subset on all 199 crops. Not usable for metric claims — used here only as
the second point on the acceptance curve.

## Corpus partition (133,318 annotated GT nodes)

| class | 44b6 (71) | 6bba (128) | pooled (199) |
|---|---:|---:|---:|
| annotated GT nodes | 20,197 | 113,121 | **133,318** |
| **C** matched | 19,356 (95.836%) | 103,138 (91.175%) | **122,494 (91.881%)** |
| unmatched | 841 (4.164%) | 9,983 (8.825%) | **10,824 (8.119%)** |
| **`L_assign`** accepted peak ≤7 µm, lost the arbitration | 41 | 18 | **59 (0.044%)** |
| **`NA7`** no accepted peak ≤7 µm (= `T ∪ L_disp ∪ D`) | 800 | 9,965 | **10,765** |
| · nearest accepted peak in (7,15] µm | 800 | 4,406 | 5,206 |
| · none within 15 µm | 0 | 5,559 | 5,559 |

At-stake GT edges (≥1 endpoint unmatched): **12,394** (44b6 1,339 / 6bba 11,055); **65.44%**
have *both* endpoints missing. Scorer weight share `w = tp+fp+fn`: 44b6 **14.86%** / 6bba
**85.14%**. Detection owns **44.7%** of edge FN.

## Ceiling (exact scorer counts, `summarise` aggregation, arm A baseline 0.664909)

Full detection recovery — 10,824 nodes, 12,394 at-stake edges. `φ` = counted-FP edges per
admitted false node.

| p | φ = 0 | φ = 2.0 | 44b6 @ φ=2 | 6bba @ φ=2 |
|---:|---:|---:|---:|---:|
| 1.0 | **+0.08088** | +0.08088 | +0.05804 | +0.08487 |
| 0.8 | +0.08074 | +0.05504 | +0.04306 | +0.05741 |
| 0.7 | +0.08064 | **+0.03764** | +0.03269 | +0.03901 |
| 0.6 | +0.08050 | +0.01571 | +0.01927 | +0.01592 |
| 0.5 | +0.08031 | **−0.01280** | +0.00122 | −0.01393 |
| 0.4 | +0.08002 | −0.05138 | −0.02433 | −0.05403 |

Break-even at **p ≈ 0.55** (φ=2). Nodes needed for +0.020 at p=0.7: **6,695** coherent /
**8,257** random at φ=2 (3,042 / 4,394 at φ=0) — confirming the "5,400–7,600, not 2,961"
correction and showing it is still ~20% optimistic. Arm D is ~7% larger throughout.

## Measured re-acceptance precision

Acceptance 0.99 → 0.969, all 199 crops (0.99 set verified a strict subset, zero GT lost):

| | 44b6 | 6bba | pooled |
|---|---:|---:|---:|
| extra nodes admitted | 425,636 | 163,130 | **588,766** |
| GT nodes recovered | 51 | 1,605 | **1,656** |
| marginal precision | 0.00012 | 0.00984 | **0.00281** |

Priced through the scorer with the empirical `φ = 0.0044`: **−0.0234 pooled**. The threshold
lever delivers p ≈ 0.003 where the ceiling needs p ≈ 0.55 — **two orders of magnitude short**.
Re-acceptance cannot be a threshold; it must be a ranker.

## T/L/D where observable — 3 crops only (2.31% of corpus, 0.26% of 44b6)

C 1,781 · **T 839 (64.6% of unmatched)** · L-assignment 57 · L-displaced (7–15 µm) 402 ·
**D 0**. The two 6bba crops disagree on `T`/unmatched by 2.4× (0.287 vs 0.684). Corpus
extrapolation **3,100–7,400 nodes — an extrapolation, with zero 44b6 basis.**

## Verdict on the three conflicting numbers

Prior: **66 nodes / +0.00045**, **184 / +0.0012**, **3-crop C class ⇒ ~2,830 / +0.019**.

**Measured: 59 nodes → +0.000581 pooled (arm A) / +0.000628 (arm D)**; 44b6 +0.00296,
6bba +0.00017; precision-insensitive at this scale.

- **+0.00045 and +0.0012 survive as a bracket** — the truth lies between them.
- **+0.019 is refuted, by ~31×**, on both detection substrates.

**Cause identified.** The 3-crop partition's `M = 1,521` is the **post-wrapper** scorer node
match of a smoke submission (exact scorer on that submission gives recall 0.9423 / 0.788 /
0.109 → 1,505), differenced against a **pre-wrapper** accepted-peak count of 1,794. The
resulting ~273-node "C class" measured **wrapper/solver node loss in a degraded smoke run**
(adj-J 0.33 on 6bba vs deployed 0.648; `det_threshold 0.96875`), not detector arbitration. Like
for like, the arbitration class on those crops is 0–57, never 273.

**Split/merge arbitration is closed:** ≤59 nodes and ~+0.0006 pooled corpus-wide.
**Detection is not closed:** +0.0809 at p=1, +0.0376 at p=0.7/φ=2, bilaterally positive to
p ≈ 0.6.

## Robustness and caveats

Repeating the census on the P0-B pre-wrapper pregraph substrate reproduces A7 exactly (17,327
at-stake, 65.83% both-missing, 2.42% 44b6 share) and gives `L_assign` = **50** — so the verdict
is substrate-independent.

- **"44b6 holds 2.42% of at-stake edge mass" is substrate-specific**, not a corpus property: it
  is **10.80%** on the deployed `det-0.969` export. The stable per-family quantity is the
  scorer weight share, 14.86% / 85.14%.
- `φ` is the largest remaining uncertainty and has never been measured for a *targeted*
  re-acceptance; the sign of the ceiling below p ≈ 0.6 is entirely a `φ` argument.
- The ceiling assumes perfect association on recovered endpoints and holds divisions fixed.
- Crop heterogeneity is severe: 6bba per-crop miss-rate CV **1.495**, max 0.860; the worst 20%
  of crops hold **71.5%** of all missed nodes.
- No P3-harmonic per-crop detection export exists on disk; the census is anchored to
  `coupled_cache det-0.969`.
- `+0.020` appears above only as the yardstick the prior estimates used. It is retracted policy.

## Provenance

Input fingerprint — sha256 over the sorted sha256s of all 199
`det/*__tta-4view__det-0.969.npz`: `9411f0d0be21dd267219acb5a93df947`.

| artifact | sha256 (first 32) |
|---|---|
| `out/c1_gt_nodes.parquet` | `72e431889635782096b8ea0a2f0f238b` |
| `out/c1_crops.parquet` | `729c1bf0217a0b8be15c09d747cf7845` |
| `out/c1_summary.json` | `37fd002374b014c0190eed44155f23e9` |
| `out/c2_ceiling.json` | `346d813536f0d0b56406b78e05ccfc3d` |
| `out/c3_threshold_slope.json` | `d69bd358f050b0179a206b906992d216` |
| `out/c4_precision_curve.json` | `9096bfa3ccd001f9c507fa19785a998d` |
| `out/c5_substrate_check.json` | `095da12890193e7a00b6b9fae497224a` |
| `out/c6_tld_subset.json` | `e517dd31d0c37374187e67fbb93de9e0` |

No test was added: `tests/test_d1_partition.py` already locks the M/T/C/L/D invariants, and the
census asserts its partition sums inline.
