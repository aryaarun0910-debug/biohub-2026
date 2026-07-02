# Scientific-Methods Research Report — Biohub Cell Tracking

Produced 2026-06-30 by a research pass. All concrete claims cited to source URL.
Unverifiable items flagged with ⚠️. HTTP-verified facts noted inline.

---

## ★ PRIORITY 1A — Biohub DAXI U-Net weights + public embryo

### DAXI U-Net weights (highest-leverage asset)
Directory: `https://public.czbiohub.org/royerlab/ultrack/unet_weights/`

| File | URL | Size |
|---|---|---|
| `unet-daxi.pt` | `https://public.czbiohub.org/royerlab/ultrack/unet_weights/unet-daxi.pt` | **20,420,030 bytes (~19.5 MiB)**, HTTP 200 verified |
| `unet-simview.pt` | `https://public.czbiohub.org/royerlab/ultrack/unet_weights/unet-simview.pt` | ~84 MiB |

- **TorchScript-compiled** → load with `torch.jit.load("unet-daxi.pt")`, no architecture class needed.
- Output = **2 channels: foreground prob + contour/boundary prob** — exactly what Ultrack's `Tracker.track(foreground=…, edges=…)` consumes.
- `unet-daxi.pt` = trained for **DAXI light-sheet** zebrafish (same modality + voxel scale as this competition). **Use this one.**
- ⚠️ Exact input patch size / channel count / normalization not verifiable from listing. Strong inference: single-channel, intensity-normalized to [0,1] via percentile clip `lower_q=0.001, upper_q=0.9999` + gamma. Confirm empirically.

### Public dense-labelled embryo zarr
- `https://public.czbiohub.org/royerlab/ultrack/zebrafish_embryo.ome.zarr/`
- **Verified**: `shape=[522,1,505,2217,2170]` (T,C,Z,Y,X), `dtype="<u2"` (uint16), chunks `[1,1,128,2217,2170]`.
- **Verified `.zattrs`**: OME-NGFF v0.4, level-0 `scale=[1.0,1.0,1.625,0.40625,0.40625]` — **exact competition voxel scale**. Origin `2024_03_22_dorado/stabilized.zarr`.

### Additional Zebrahub embryos + tracking CSVs
Index: `https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/`
- Zarrs: `ZSNS001`, `ZSNS001_tail`, `ZSNS002`, `ZSNS003`, `ZSNS004`, `ZSNS005`, + `tracks_benchmark/`.
- Lineage CSVs (track_id,t,z,y,x): `ZSNS001_tracks.csv` 849 MiB, `ZSNS001_tail` 464 MiB, `ZSNS003` 199 MiB, `ZSNS004` 360 MiB, `ZSNS005` 326 MiB. (No CSV for ZSNS002.)

---

## ★ PRIORITY 1B — Ultrack (Nature Methods 2025)
Repo `https://github.com/royerlab/ultrack` · Docs `https://royerlab.github.io/ultrack/`

```python
from ultrack import MainConfig, Tracker
config = MainConfig()
config.segmentation_config.min_area   = ...   # ~half smallest cell volume
config.segmentation_config.max_area   = ...   # ~1.25-1.5x largest
config.segmentation_config.min_frontier = 0.0 # raise in 0.05 steps to merge weak boundaries
config.segmentation_config.threshold  = 0.5
config.linking_config.max_distance    = 15.0  # ~1.5x expected per-frame displacement
config.linking_config.max_neighbors   = 5
config.tracking_config.appear_weight     = -0.001
config.tracking_config.disappear_weight  = -0.001
config.tracking_config.division_weight   = -0.001
tracker = Tracker(config)
tracker.track(foreground=foreground, edges=contours)   # edges == contour/boundary map
tracks_df, graph = tracker.to_tracks_layer()           # track_id,t,z,y,x + parent graph
```

**Verified defaults**: SegmentationConfig `min_area=100, max_area=1_000_000, min_frontier=0.0,
threshold=0.5, min_area_factor=4.0`. LinkingConfig `max_distance=15.0, max_neighbors=5`.
TrackingConfig `appear/disappear/division_weight=-0.001, power=4, solution_gap=0.001`.

**Solver (verified `mip_solver.py`)**: uses Python `mip`; `solver_name=""` → Gurobi if present
else **CBC (bundled, offline-OK)**. ⚠️ No SCIP/HiGHS backend in ultrack. Also a numba
heuristic solver for full-embryo scale when CBC too slow.

**Export (verified `ultrack/core/export/`)**: `to_geff(config, filename)` writes GEFF; also
CTC, NetworkX, TrackMate, DataFrame exporters.

**Preprocessing (verified `ultrack/imgproc/intensity.py`)**: `normalize(img, gamma,
lower_q=0.001, upper_q=0.9999)`; `robust_invert(...)`; `labels_to_contours()`.
⚠️ No built-in torchscript loader for `unet-daxi.pt` — apply via user `torch.jit.load` + sliding window.

---

## ★ PRIORITY 1C — PAC-MAP (proximity-adjusted heatmap)
Repo `https://github.com/DeVosLab/PAC-MAP` (code read from `weak_targets.py`, `points2prob.py`).

**Target kernel** (anisotropic 3D Gaussian, σ = radius/4 per axis, in voxels):
```python
gkern1d_z = gaussian(r_z*2+1, std=r_z/4); ...  # r_axis = int(r_um/voxelsize_axis)
gauss = np.einsum('i,j,k', gz, gy, gx)
gauss = (gauss-gauss.min())/(gauss.max()-gauss.min())  # -> [0,1]
```
**Proximity adjustment** (`points2prob.py`): amplitude = distance to nearest OTHER nucleus
(physical, via `kdtree.query(points*voxelsize, k=2)[:,1]`); isolated points dropped;
**kernels combine by `np.maximum`**, not sum. Network learns centroid AND local crowding.

**Training** (`configs/18_boutin_et_al_train.py`): **loss = MSE** (no final sigmoid),
patch `[46,256,256]`, channels=1, U-Net `f_maps=16`, lr 1e-4, batch 12, 150 epochs,
percentile-norm `[0.1,99.9]`. Backend `pytorch-3dunet` (submodule). Weak-pretrain → finetune.
Pretrained (other domains): Zenodo `https://zenodo.org/records/14138806`.

---

## PRIORITY 2A — Sparse/PU detection loss (Linajea, Nat. Biotech 2023)
Repo `https://github.com/funkelab/linajea`.
- Target = sum of Gaussians (max 1 at annotation, decreasing by σ).
- **Key trick**: build a **training mask of small radius around each annotation** (radius <
  nearest-neighbor spacing so it never touches a neighbor); **MSE loss computed ONLY inside
  the masks.** Unannotated pixels are left **unconstrained**, not forced to background.
- ⚠️ σ and mask radius are per-dataset; practical: mask radius ≈ one nuclear radius, σ smaller.

→ This is the principled fix for 52 labelled / 25,755 true cells. Combine with PAC-MAP kernel.

---

## PRIORITY 2B — Trackastra (ECCV 2024)
Repo `https://github.com/weigertlab/trackastra` (from `README.md`, `pretrained.json`).
- **Pretrained, offline-usable**: only **`ctc`** model is 3D →
  `https://github.com/weigertlab/trackastra-models/releases/download/v0.3.0/ctc.zip`.
  Load offline: `Trackastra.from_folder('ctc_model_folder/', device='cuda')`.
- **Linker, NOT a detector** — needs `imgs` AND instance `masks` (both `time,(z),y,x`).
  `model.track(imgs, masks, mode="ilp")` (`ilp` needs `motile`, offline-OK; `greedy` has divisions).
- Export: `write_to_geff(track_graph, masks_tracked, outdir=...)` — GEFF, ILP-compatible.

---

## Bottom line — recommended pipeline
1. **Detect**: `unet-daxi.pt` (offline, 19.5 MiB) → `(foreground, contours)`; OR train PAC-MAP-style
   3D U-Net (MSE on proximity-adjusted anisotropic Gaussians, σ=r/4, amp=NN-distance, max-combine)
   with **Linajea masked loss** on the 52 sparse points.
2. **Link/solve**: Ultrack `Tracker.track(foreground, edges)` with CBC (no Gurobi) → `to_geff`.
   Cross-check Trackastra `ctc` + `motile` ILP on the same detections.
3. **Calibrate** min/max area + division weights; validate on dense `zebrafish_embryo.ome.zarr`
   (scale matches) and the five `ZSNS00x_tracks.csv` lineages.

Verified: DAXI weight HTTP 200 = 20,420,030 bytes; embryo zarr shape/dtype confirmed.
Flagged ⚠️: DAXI exact input patch/normalization; per-dataset σ/radius numbers.
