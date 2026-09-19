---
tags:
  - infra
  - key
---

# The Deployed Pipeline Runs On The Mac — Bit-Identical

**The single biggest capability change in the project.** Kaggle is no longer
needed to measure anything.

The kernel ships its own source tree, and it was sitting in
`artifacts/s05_output/tracking_repo/` the whole time: the temporal UNet3D, the
node transformer, and `predict_unet_transformer.py`.

**Everything installs on macOS arm64.** `tracksdata` and `ilpy` ship as
*pure-Python* wheels in the weights datasets; the only native dependency is
`pyscipopt` (the SCIP ILP solver), and PyPI has **6.2.1** for arm64 — the exact
version the kernel uses.

**Two patches, both trivial:**
1. `device = cuda if available else cpu` → add an `mps` branch. The original
   falls silently to CPU on a Mac.
2. Two hardcoded `Path('/kaggle/working')` → `$BIOHUB_WORKING_DIR`.

Only **16** `BIOHUB_*` vars are read by the pipeline (the other 49 the notebook
sets are post-processing, which the [[Local Harness]] already replicates).

## The verification that matters

Run on `44b6_341df25f`, against the graph the Kaggle T4 produced:

| | nodes | edges | max coord diff | edge sets |
|---|---|---|---|---|
| LOCAL (MPS) | 8523 | 8189 | **0** | **identical** |
| KERNEL (T4) | 8523 | 8189 | | |

Identical `J`, identical tp/fp/fn. **Not a rebuild — the same graphs.**

## What it changes

- **96 s/film on the M5 Pro against the T4's ~236 s — 2.5× faster than Kaggle.**
- All 199 films in ~5.3 h single-process, less when sharded (`--slice N::M`).
- No 9 h kernel limit, no 2-session cap, no submission cost.
- [[Evidence Tiers]]' REBUILD-199 row — greedy, weaker, direction-only — is
  **obsolete**. It becomes a *deployed-quality* 199-film tier.
- Both Kaggle slots are freed for what they are actually for: scoring.

`scripts/local_predict_199.sh`

Related: [[s11]], [[The 32-Film Verdict]]
