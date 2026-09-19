# Biohub Full-Frame Center Detector Pack

This dataset contains a standalone DeepCenterUNet3D center-heatmap detector.
It is intended as an auxiliary gated/blended prior for Biohub cell tracking
submissions, not as a replacement for the primary graph model.

Important paths:

```text
weights/full_frame_center/best.pt
weights/full_frame_center/checkpoint_last.pt
weights/full_frame_center/config.json
weights/full_frame_center/history.csv
weights/full_frame_center/split_manifest.json
weights/full_frame_center/gate_summary.json
weights/full_frame_center/gate_threshold_metrics.csv
weights/full_frame_center/gate_frame_metrics.csv
weights/full_frame_center/gate_peak_samples.csv
ARTIFACT_MANIFEST.json
```

Coordinate contract:

```text
coordinate order: z, y, x
coordinate unit:  original voxel
voxel scale:      z=1.625, y=x=0.40625 microns/voxel
```

The detector runs on an XY-pooled image volume. A heatmap peak at pooled
coordinate `(z, y, x)` maps back to original voxel coordinates as:

```text
z_orig = z
y_orig = y * pool_factor + (pool_factor - 1) / 2
x_orig = x * pool_factor + (pool_factor - 1) / 2
```

The gate diagnostic files use sparse labels, so their precision/recall values
are calibration signals rather than complete-cell metrics. Use the model as a
high-confidence rescue source near graph gaps, short components, or unmatched
motion endpoints.

Kaggle input path:

```text
/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/ARTIFACT_MANIFEST.json
```
