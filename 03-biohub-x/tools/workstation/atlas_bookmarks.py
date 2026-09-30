"""Bookmark the miss atlas: one napari capture per representative miss, per class, with digests.

Objective: turn each representative miss of the atlas into a visual record a
person can open and argue with: the frame it was missed in, the plane it lies
on, the frames either side, the ground truth around it and the proposals the
source did place. Falsifier for the visual step itself: a capture that renders
nothing (one colour) or a bookmark whose coordinates do not land on the cell.

The default visual procedure of D-0043: volume read-only, ground truth and
proposals as separate named layers, axis order and physical scale confirmed,
the miss ringed in red, the previous and following frames captured, dataset
id, frame, coordinates, layer configuration and screenshot digests recorded.
Nothing here is a finding; the atlas classes are hypotheses until a probe
reproduces them.

Runs inside biohub-visual with BIOHUB_DATA_ROOT set. Reads the atlas JSON the
`biohubx evaluate miss-atlas` command wrote.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
_data_root = os.environ.get("BIOHUB_DATA_ROOT")
if not _data_root:
    raise SystemExit("BIOHUB_DATA_ROOT is unset; give the competition data root the way biohubx does")
DATA = Path(_data_root)
ATLAS = (Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "artifacts" / "miss-atlas-6bba.json").resolve()
SESSION = sys.argv[2] if len(sys.argv) > 2 else "WS-VISUAL-02"
OUT = REPO / "artifacts" / "tool-sessions" / SESSION
PER_CLASS = int(os.environ.get("BIOHUBX_ATLAS_PER_CLASS", "2"))
HALF_UM = 30.0
"""Half-width of the bookmark crop in micrometres, in y and x; z is taken whole."""

os.environ.pop("QT_QPA_PLATFORM", None)
sys.path.insert(0, str(REPO / "src"))

import napari
import numpy as np
from qtpy.QtWidgets import QApplication

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE
from biohubx.data.competition import WindowSelection, load_ground_truth, load_window
from biohubx.proposals import dog


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


atlas = json.loads(ATLAS.read_text(encoding="utf-8"))
source = atlas["source"]
scale = (OFFICIAL_VOXEL_SCALE.z_um, OFFICIAL_VOXEL_SCALE.y_um, OFFICIAL_VOXEL_SCALE.x_um)
started = time.time()
OUT.mkdir(parents=True, exist_ok=True)

record: dict = {
    "schema_version": 1,
    "session_id": SESSION,
    "profile": "biohub-visual",
    "provenance_status": "integration_only",
    "objective": "bookmark representative misses of every atlas class as verifiable captures with digests",
    "falsifier": "a capture rendering one colour, or a bookmark whose coordinates do not land on the annotated cell",
    "atlas": {
        "path": str(ATLAS.relative_to(REPO)).replace("\\", "/"),
        "sha256": sha256(ATLAS),
        "source": source,
    },
    "layers": "volume (gray, additive), ground truth (lime points), proposals (magenta points), the miss (red ring); scale (1, 4.0, 1.0, 1.0) um on (t, z, y, x)",
    "read_scope": "competition root read-only; the atlas JSON",
    "write_scope": str(OUT),
    "versions": {"napari": napari.__version__, "numpy": np.__version__},
    "bookmarks": [],
}

viewer = napari.Viewer(show=True)
app = QApplication.instance()


def capture(path: Path) -> tuple[str, int]:
    for _ in range(5):
        if app is not None:
            app.processEvents()
        viewer.window._qt_viewer.canvas.native.repaint()
        if app is not None:
            app.processEvents()
    frame = viewer.screenshot(str(path), canvas_only=True, flash=False)
    colours = len(np.unique(frame.reshape(-1, frame.shape[-1]), axis=0))
    if colours < 2:
        raise SystemExit(f"{path.name} rendered a single colour; the canvas did not draw")
    return sha256(path), colours


loaded: dict[str, object] = {}
for cls, examples in atlas["representatives"].items():
    for example in examples[:PER_CLASS]:
        dataset_id = example["dataset"]
        frame = int(example["frame"])
        vz, vy, vx = (float(v) for v in example["voxel_zyx"])
        truth = load_ground_truth(DATA, dataset_id)
        first = max(0, frame - 1)
        frames = 3 if frame + 1 < truth.frames else truth.frames - first
        import zarr

        group = zarr.open(str(DATA / "train" / f"{dataset_id}.zarr"), mode="r")
        depth, height, width = group["0"].shape[1:]
        half_y = int(HALF_UM / OFFICIAL_VOXEL_SCALE.y_um)
        half_x = int(HALF_UM / OFFICIAL_VOXEL_SCALE.x_um)
        y0, y1 = max(0, int(vy) - half_y), min(height, int(vy) + half_y)
        x0, x1 = max(0, int(vx) - half_x), min(width, int(vx) + half_x)
        window = load_window(
            DATA, WindowSelection(dataset_id, first, frames, 0, depth, y0, y1, x0, x1), split="train"
        )
        instances = dog.detect_instances(
            window.volume,
            dataset=window.annotated.dataset,
            radii_um=tuple(float(r) for r in source["radii_um"]),
            response_quantile=float(source["response_quantile"]),
            suppression_radius_um=float(source["suppression_radius_um"]),
            local_maxima_only=bool(source["local_maxima_only"]),
            per_scale_union=bool(source["per_scale_union"]),
        )
        viewer.layers.clear()
        viewer.add_image(
            np.asarray(window.volume),
            name=f"{dataset_id} volume",
            scale=(1.0, *scale),
            colormap="gray",
            blending="additive",
        )
        gt = np.array(
            [[n.frame - first, n.voxel.z, n.voxel.y - y0, n.voxel.x - x0] for n in window.annotated.nodes],
            dtype=float,
        )
        pr = np.array([[i.frame, i.voxel.z, i.voxel.y, i.voxel.x] for i in instances.instances], dtype=float)
        if len(gt):
            viewer.add_points(
                gt,
                name="ground truth",
                size=6,
                face_color="lime",
                scale=(1.0, *scale),
                out_of_slice_display=True,
            )
        if len(pr):
            viewer.add_points(
                pr,
                name="proposals (A2)",
                size=4,
                face_color="magenta",
                scale=(1.0, *scale),
                out_of_slice_display=True,
            )
        miss = np.array([[frame - first, vz, vy - y0, vx - x0]], dtype=float)
        viewer.add_points(
            miss,
            name=f"MISS {cls}",
            size=10,
            face_color="transparent",
            border_color="red",
            border_width=0.3,
            scale=(1.0, *scale),
            out_of_slice_display=True,
        )
        viewer.dims.ndisplay = 2
        shots: dict[str, dict] = {}
        for label, t_local in (
            ("previous", frame - first - 1),
            ("missed", frame - first),
            ("following", frame - first + 1),
        ):
            if not 0 <= t_local < frames:
                continue
            # dims points are world coordinates: with scale (1, 4, 1, 1) the z plane index
            # must be multiplied by the z spacing, or the capture shows plane z/4. The first
            # run did exactly that; current_step is checked so it cannot happen silently.
            viewer.dims.set_point(0, t_local)
            viewer.dims.set_point(1, vz * scale[0])
            landed = int(viewer.dims.current_step[1])
            if landed != round(vz):
                raise SystemExit(f"asked for plane {vz}, the viewer landed on {landed}")
            name = f"{cls}-{dataset_id}-f{frame}-n{example['node_id']}-{label}.png"
            digest, colours = capture(OUT / name)
            shots[label] = {
                "file": name,
                "digest": f"raw_artifact_sha256:sha256:{digest}",
                "distinct_colours": colours,
                "frame": first + t_local,
                "z_plane": vz,
            }
        viewer.dims.ndisplay = 3
        viewer.dims.set_point(0, frame - first)
        viewer.camera.angles = (0.0, 30.0, 130.0)
        name = f"{cls}-{dataset_id}-f{frame}-n{example['node_id']}-3d.png"
        digest, colours = capture(OUT / name)
        shots["3d"] = {
            "file": name,
            "digest": f"raw_artifact_sha256:sha256:{digest}",
            "distinct_colours": colours,
        }
        record["bookmarks"].append(
            {
                "class": cls,
                "dataset": dataset_id,
                "frame": frame,
                "node_id": example["node_id"],
                "voxel_zyx": [vz, vy, vx],
                "physical_um_zyx": example["physical_um_zyx"],
                "crop_voxels": {"z": [0, int(depth)], "y": [y0, y1], "x": [x0, x1]},
                "window_first_frame": first,
                "intensity_percentile": example["intensity_percentile"],
                "nearest_proposal_um": example["nearest_proposal_um"],
                "flags": example["flags"],
                "layers": [layer.name for layer in viewer.layers],
                "captures": shots,
            }
        )
        print(f"{cls:22} {dataset_id} f{frame} node {example['node_id']} captures={len(shots)}", flush=True)

viewer.close()
record["elapsed_seconds"] = round(time.time() - started, 3)
record["bookmark_count"] = len(record["bookmarks"])
record["conclusion"] = "captures recorded with digests; see the bookmarks for what each class looks like"
record["unresolved"] = (
    "the classes remain hypotheses; each becomes a probe with a falsifier before it can support an arm"
)
target = REPO / "artifacts" / "tool-sessions" / f"{SESSION}.json"
target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("record:", target, "bookmarks:", len(record["bookmarks"]))
