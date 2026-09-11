## Dataset Description

Each sample in this dataset is a short 3D+time video of fluorescently labeled zebrafish embryo cells, stored as a [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) volume. Your task is to detect cells in each timepoint and link them across time, producing a tracking graph of nodes (cell detections) and edges (temporal links between cells).

### Data Format

Image volumes are stored as `.zarr` directories. Each contains a single array at path `0/` with shape `(T, Z, Y, X)` — typically `(100, 64, 256, 256)` in `uint16` format. Chunks are one timepoint each
: `(1, 64, 256, 256)`, compressed with blosc/zstd. The chunk for timepoint `t` is located at `0/c/{t}/0/0/0`. Array metadata (shape, dtype, codecs) is in `0/zarr.json`.

The physical voxel scale is z=1.625, y=0.40625, x=0.40625 µm/voxel.

### Ground Truth (Training Only)

Ground-truth annotations are provided as `.geff` directories (a graph exchange format also built on Zarr v3). Each `.geff` contains:

- **`nodes/ids`** — node ID array
- **`nodes/props/{t,z,y,x}/values`** — integer centroid coordinates per node (in voxels)
- **`edges/ids`** — edge array of shape `(N, 2)` with columns `(source_id, target_id)`

Annotations are **sparse** — not every cell in every frame is labeled. The `estimated_number_of_nodes` field in the `.geff` metadata (`zarr.json`) provides an estimate of the true total cell count per sample.

All arrays within `.geff` use zstd compression.

### Embryo Identity

Folder names follow the pattern `{embryo_id}_{field_of_view}` (e.g., `44b6_0049_0438_1330_1273`). The first segment identifies which embryo the sample comes from. Multiple samples may share the same embryo. **Train and test sets are embryo-disjoint** — no embryo appears in both.

## Files

- **train/** - Training samples. Each sample has a paired `.zarr` (image volume) and `.geff` (ground-truth tracking graph).
- **test/** - Example test samples (copies from train). Contains `.zarr` image volumes only — no ground truth is provided. When a notebook is submitted for rerun, a new hidden test set is swapped in. The size of the hidden test set is approximately the same size as the training dataset.
- **sample_submission.csv** - A valid submission file demonstrating the correct format.