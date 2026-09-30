# Biohub Tracking Support Pack V1

This artifact contains the learned inference code and weights required by the
UNet + node-transformer + ILP submission notebooks. When present, it can also
contain a separately trained full-frame center detector used only as an
auxiliary node-rescue prior.

Included:

- `repo/`: inference/training source used by the notebook.
- `weights/`: `unet_transformer/split_0/edge_predictor_best.pth` and `config.json`.
- `source_scripts/`: local training/packaging scripts used to create the support pack.
- `requirements-unet-ilp.txt`: Python dependencies to install separately.
- `requirements-unet-ilp-kaggle-predownload.txt`: quote-free package list for Kaggle dependency helpers.
- `kaggle_dependency_install_command.txt`: one-line Kaggle dependency input command.
- `ARTIFACT_MANIFEST.json`: generated provenance and weight checksum.
- `wheels/`: offline dependency wheels for code-submission reruns.


The manifest includes a `compatibility` section that defines the shared
coordinate contract:

```text
coordinate order: z, y, x
coordinate unit:  original voxel
voxel scale:      z=1.625, y=x=0.40625 microns/voxel
```

If `weights/full_frame_center/` exists, the detector outputs are calibrated by:

```text
gate_summary.json
gate_threshold_metrics.csv
gate_frame_metrics.csv
gate_peak_samples.csv
```

These files are sparse-label calibration features. They are intended for
conservative node-rescue gates, not as complete-cell leaderboard estimates.

Kaggle dependency input command:

```bash
pip install tracksdata zarr>=3.0.10,<4 pyscipopt geff ilpy polars blosc2 dask imagecodecs pyarrow rustworkx sqlalchemy
```

Do not add quotes around `zarr>=3.0.10,<4` in Kaggle's dependency input.
