# Trackastra zero-shot bridge

This bridge evaluates Trackastra **association only** on frozen learned detections. Existing
learned OOF outputs are one predicted GEFF per crop (`pred_geffs_split_<fold>/<crop>.geff`),
containing `t,z,y,x` nodes and organizer edges. The bridge ignores those edges, creates
anisotropic ellipsoid instance masks, runs the offline Trackastra `ctc` model, and restores the
original coordinates in its output GEFF for `scripts/score_oof.py`.

## Offline inputs not currently in this repository

1. Download both fold OOF GEFF output directories from their Kaggle kernels.
2. Stage Trackastra 0.5.2 (wheel or source plus dependencies) as an offline Kaggle dataset.
3. Stage the extracted official `ctc` model folder. It must contain `config.yaml`, `model.pt`,
   and `train_config.yaml`. Do not call `Trackastra.from_pretrained` in a no-internet kernel;
   this bridge uses `Trackastra.from_folder`.

The official model registry describes `ctc` as a 2D+3D model trained on all available Cell
Tracking Challenge GT and ERR_SEG data. Trackastra's public API consumes images and instance
masks. Candidate association scores are only exposed through `_predict`, so this adapter pins
0.5.2 and validates the private API before inference.

## Commands

Validate paths, shape, detections, and memory without allocating a mask or importing Trackastra:

```powershell
.venv\Scripts\python.exe -m scripts.trackastra_zero_shot.run `
  --detections <pred_geffs_split_1\6bba_x.geff> `
  --images <data\train\6bba_x.zarr> `
  --model-dir <offline\trackastra_ctc> --dry-run
```

Prepare the point-to-mask conversion before Trackastra is installed:

```powershell
.venv\Scripts\python.exe -m scripts.trackastra_zero_shot.run `
  --detections <prediction.geff> --images <image.zarr> `
  --model-dir <future-model-folder> --prepare-only `
  --mask-npy <work\instances.npy> --mapping-csv <work\mapping.csv>
```

Run zero-shot linking, preserving all frozen nodes and changing only edges:

```powershell
.venv\Scripts\python.exe -m scripts.trackastra_zero_shot.run `
  --detections <prediction.geff> --images <image.zarr> `
  --model-dir <offline\trackastra_ctc> --device cuda --mode greedy `
  --mask-npy <work\instances.npy> --output-geff <out\crop.geff> `
  --edge-scores <out\crop_edges.csv>
```

Score a completed fold with the existing exact scorer:

```powershell
.venv\Scripts\python.exe scripts\score_oof.py --pred-dir <out-fold> --gt-dir data\train
```

Start with `greedy`, not Trackastra ILP: its ILP objective is not this competition metric and
adds an offline `motile/ilpy/SCIP` dependency. The emitted candidate-score CSV is the intended
input to the repository's metric-aligned solver/ensemble front.

## Important constraints

- A 100x64x256x256 uint16 mask is about 0.78 GiB; crops must run sequentially.
- Full images are materialised for Trackastra normalisation, so expect roughly 3--5 GiB CPU RAM
  per crop before model workspace.
- Default ellipsoid radius is 2.5 µm, equivalent to radii `(1.54, 6.15, 6.15)` voxels. Radius is
  an OOF ablation, not a trusted biological constant.
- Trackastra sees voxel-coordinate centroids. The `ctc` model supports 3D and was trained across
  heterogeneous CTC data, but DaXi anisotropy/domain shift is still the central falsification risk.
