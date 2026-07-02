# Biohub Cell Tracking 2026

Kaggle: *Cell Tracking During Development* (zebrafish 3D+time). Detect cell
centroids, link across time, find divisions, reconstruct lineages.
Metric: `weighted_avg(adj_edge_jaccard) + 0.1 * division_jaccard`.
Deadline 29 Sep 2026. Notebook-only, internet off, <=12 h runtime.

Strategy dossier: `../Codex/2026-06-30/.../biohub_intelligence_sweep_2026-06-30.md`.

## Operating model
- **Local machine = dev + exact scoring + full-data CPU work.** GPU is an MX350 (2 GB) —
  cannot train/infer 3D U-Nets. Heavy TRAINING/GPU inference runs on Kaggle/cloud (T4x2).
- **Full 87 GB dataset IS mirrored locally** (`data/train`, `data/test`) for full-data EDA,
  CPU DoG sweeps, and exact embryo-held-out scoring across all crops. Kaggle also mounts it
  at `/kaggle/input/...` for the final inference notebook.

## Environment
- `.venv` = Python 3.12 (host metric stack needs >=3.11,<3.14).
  Has: `tracking-cellmot` (editable, organizer pkg), `tracksdata`, `geff`,
  `polars`, CPU `torch`, `pyscipopt` (SCIP ILP — license-safe vs Gurobi),
  `kaggle`, `dask`.
- Default `py` = 3.14 (too new for tracksdata) — use `.venv` for anything
  touching the metric/host code.

## Data facts (verified from LOCAL extracted data)
- **199 train CROPS from only 2 embryos**: families `44b6` (71 crops) and `6bba`
  (128 crops). (The earlier "129" came from an INCOMPLETE API manifest; ground truth is the
  199 local zarr/geff pairs.) Totals: 128,883 annotated edges, 151 divisions. "Dataset names
  begin with the embryo identifier" (prefix); each crop is a ~100-frame sub-volume. 4 visible
  test crops (all `44b6`) are debug copies; hidden test is a DIFFERENT embryo.
- **CV must be leave-one-embryo(-family)-out**, NOT k-fold over crops — splitting crops
  leaks an embryo into train+val (the "validation trap"). See scripts/build_splits.py.
- Each crop: `<id>.zarr` image (T,Z,Y,X uint16) + `<id>.geff` sparse label graph.
- Labels EXTREMELY sparse: per-crop label_fraction median ~1%, min 0.13%, max ~18%
  (annotated nodes / estimated_number_of_nodes). Count adjustment uses the ESTIMATED total.
- Fixed voxel scale (z,y,x) = (1.625, 0.40625, 0.40625) µm; Z is 4x coarser.

## Layout
- `vendor/kaggle-cell-tracking/` — cloned organizer repo (metric + baseline model).
- `src/biotrack/metric.py` — LB-faithful local scorer (wraps host `evaluate`).
- `scripts/fetch.py` — targeted downloader (`--kind geff|zmeta|zarr`).
- `reports/inventory/file_manifest.txt` — cached list of all 16,200 comp files.
- `data/` — gitignored; only fetched embryos land here.

## Tools
- `src/biotrack/metric.py` — LB-faithful scorer. `score_one/score_many` (geff pairs),
  `score_pred_graph` (in-memory graph vs GT geff), `score_submission` (submission.csv vs GT dir).
- `src/biotrack/submission.py` — lossless graph <-> sample_submission.csv converter
  (round-trip verified: GT -> CSV -> graph -> edge_jaccard 1.0).
- `scripts/fetch.py` — targeted downloader.
- `scripts/build_inventory.py` — pull all 96 GEFF labels (retry/resume) -> embryo_stats.csv.
- `scripts/build_splits.py` — embryo-grouped K-fold balanced by edge volume -> dataset_splits.json.
- Methods research: `reports/research/methods_research_2026-06-30.md`.

## Status
- [x] Full 87 GB dataset mirrored + extracted locally (199 train + test).
- [x] Inventory + leave-one-embryo-out split over all 199 (fold0 test=44b6: 71 crops/19.8k edges/26 div;
      fold1 test=6bba: 128 crops/109.1k edges/125 div).
- [x] Exact NUMPY edge metric (src/biotrack/metric_numpy.py) — validated == tracksdata incl. adversarial
      assignment-conflict cases (tests/). Division term routes through tracksdata harness (biotrack.metric).
- [x] Lossless graph <-> submission.csv converter; score_submission charges omitted datasets as all-FN.
- [x] DAXI U-Net I/O verified; anisotropic NMS fixed (coarsest-axis pre-filter + physical NMS).
- [x] V3 DoG notebook (0.842 recipe) + 3-way edge taxonomy (no-candidate / lost-assignment / association).
- [x] Repo repair (Codex audit): stale 129->199 fixed, tests added, unsafe rm perm removed, deps locked.
- [ ] Phase 0: close V3 repro gap (precomputed-quantile norm) -> confirm ~0.84 anchor on both folds.
- [ ] Phase 1: matching-aware arbitration + track-conditioned redetection (the core weapon).

## Plans
- **reports/EXECUTION_PLAN.md** — phased plan to >=0.88 private (v2, red-team corrected).
- **reports/experiment_plan.md** — ranked experiment queue + crystallized direction.
- **reports/research/** — methods + ecosystem + competitive deep-dives.
