# Synthetic corpus (Kaggle discussion 732103) — REJECTED

**Audited:** 2026-08-07 · **Verdict: REJECT. No training authorised. Do not download at scale.**
Raw research, scripts, 61 MB hashed sample and full mismatch report:
`..\Biohub-CellTracking-2026_RESEARCH\agent_runs\v7_diagnostic_20260807\gateB\`

---

## Identity and licence — this alone closes it

*"[Free Dataset] 18.5 GB of fully-labelled synthetic 3D microscopy — 165k labelled divisions"*,
José Freitas, 2026-08-01.
`https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset` · accessed 2026-08-07.

**Licence claimed CC0 in the post text only.** The artifact carries **no licence field anywhere
machine-reachable** — not in kernel metadata, not in the Kaggle kernels API (the schema has no
`license_name`), not on the rendered page. It is a **kernel output**, not a Kaggle Dataset; the
dataset slug 404s/403s while a control query succeeds.

Under this project's standing discipline — unlicensed `naivete5656/SCDTC` already closed a lane
on exactly this basis — **an unverifiable licence closes the lane regardless of distributional
fit.** Everything below is therefore confirmatory, not load-bearing.

18 files / 61.13 MB sampled (0.33% of corpus), all SHA-256'd in the lane's
`out/SAMPLE_SHA256.json`. Byte cap enforced in code.

---

## The gate: mismatch is LARGER than the 0.9888 that closed this line before

> **A single fit-free univariate rank statistic separates the corpora perfectly:
> saturated-voxel fraction AUC = 1.0000**, 120 real frames vs 75 synthetic volumes, **zero
> overlap.** No model was fitted anywhere.

Intensity CV AUC 0.9351 · local-background ratio AUC 0.8299.

The prior synthetic line was closed at real-vs-synth CV AUC **0.9888** with a *fitted*
classifier. This is worse, and it needs no classifier at all.

## Seven axes, lane-7 estimators verbatim, denominators stated

| axis | real | synthetic | mismatch |
|---|---|---|---|
| NN ÷ own axial FWHM | 0.918 (**inside**) | 1.241 (**outside**) | crosses the merge boundary |
| frac. NN < own axial FWHM | 0.622 | **0.055** | **11.4×** |
| nucleus anisotropy | 1.738 / 1.686 | **1.159** | excess aniso = 22% of real |
| local bg p95/p5 | 11.50 | 6.42 | 1.79× flatter |
| axial share of motion energy | 0.419 | **0.872** | **direction inverted** |
| sister separation p50 / σ | 10.570 / 3.165 µm (n=151) | 7.129 / **1.617** (n=829) | 33% low, 1.96× narrow |
| division rate per node | 0.1133% (151/133,318) | 4.036% | **35.6×** |
| band-pass recall | 0.8769 (919/1,048) | 0.9840 | misses 12.31% vs **1.60%** |
| deployed detector, peaks ÷ true cells | **1.463** × N_est/T | **1.010** | **no FP pressure exists** |

**Clean axes, and they are the only ones:** division label timing (all 151 real events Δt = 1,
same as synthetic — this independently confirms our GT convention) and density/occupancy
(219 vs 248 cells/frame).

## Resampling or reweighting fixes exactly one axis of seven

- **Division rate — YES.** Scalar 1/35.64 per event, or 1/17.82 per daughter node.
- **Sister separation — PROVABLY NO.** Finite-variance importance weighting requires
  `σ_synth > σ_real/√2 = 2.238`; it is **1.617**, so the χ² divergence is **infinite**.
  Empirical KDE **ESS 49.6/829 = 5.98%**, top 1% of samples carry 26.6% of the weight, **49% of
  real divisions lie above the synthetic p99 and 30.5% above its maximum.**
- **Motion direction — NO, with the root cause found at source.**
  `vel = rng.normal(0, s_step, (n,3))` samples **one isotropic σ in VOXEL units** while the
  voxels are 4:1 anisotropic. Axial share is degenerate corpus-wide (0.860–0.895, σ = 0.0103)
  against our 0.419; **no sequence in 2,174 comes near it.**
- **Crowding / PSF / contrast / detector hardness — NO.** These are rendering *constants*, not
  sampled attributes, so no reweighting can reach them. Note that domain randomisation over
  exactly these axes has already failed in this project.

## Three findings that propagate beyond this lane

1. **The released 18.5 GB comes from the parametric `gen_volume_native` path, not the
   template/real-background hybrid.** The notebook's headline validation (mean KS 0.225) belongs
   to a **different generator mode**. Anyone citing those KS numbers for this corpus is citing
   the wrong experiment.
2. **The post's detectability claim reverses under the deployed estimator** (real 0.8769 vs
   synthetic 0.9840). The error is that it measures real recall on the **2.8% annotated
   subset**, whose NN is 22.44 µm against a true local 9.68 µm — **our own recorded trap**, made
   independently by someone else. Further evidence that the annotated subset is not
   representative of the field the detector actually faces.
3. **Undisclosed defect in the artifact:** in the sequences, node coordinates are native voxel
   indices while the volumes are stride-pooled 4× in XY and `voxel_um_pooled` is declared
   isotropic 1.625. Taken at face value, **essentially every label falls outside its own
   volume.**

## Reopening conditions

Quantified as 8 numeric gates in the lane report §7. The binding ones: **σ_sister ≥ 2.24 µm**
(below this, reweighting is mathematically impossible, not merely inefficient), **anisotropy
≥ 1.60**, **best single-statistic AUC ≤ 0.75**, and a **machine-verifiable licence**.

Also incidental: `kaggle kernels files` reports every output file as ~924 B. That is a **Kaggle
API defect** — true sizes via ranged GET are 3.18 MB (sequences) / 8.39 MB (static). Do not size
a download from that endpoint.
