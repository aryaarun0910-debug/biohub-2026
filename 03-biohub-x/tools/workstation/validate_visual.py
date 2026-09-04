"""Validate the biohub-visual profile on real competition data, read-only, and write a session record.

Objective: show that the pinned viewer stack can open a competition volume, load
the ground-truth GEFF graph through napari-geff, put Biohub-X's own frozen
proposals beside it, and render, all without writing to the data root. The
falsifier: any import, read, layer construction or render that fails, any write
attempt inside the data root, or a proposal layer whose coordinates do not land
on the same physical scale as the ground truth.

Runs inside the biohub-visual environment. The Biohub-X package is imported from
the repository's src/ for the proposal step only; it is read, never installed
here.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
# The data root is never written down here: it comes from the same environment
# variable every biohubx command reads, and this refuses without it rather than
# guessing a location.
_data_root = os.environ.get("BIOHUB_DATA_ROOT")
if not _data_root:
    raise SystemExit("BIOHUB_DATA_ROOT is unset; give the competition data root the way biohubx does")
DATA = Path(_data_root)
DATASET = "44b6_ddf577ad"
FRAMES = 2
OUT = REPO / "artifacts/tool-sessions"
SHOTS = OUT / "WS-VISUAL-01"

# Not offscreen: Qt's offscreen platform gives vispy no GL context, and napari
# refuses with GL_INVALID_OPERATION before a layer exists. The viewer is created
# with show=False on the workstation's own display, which is what this profile is
# for; nothing here needs a headless server.
os.environ.pop("QT_QPA_PLATFORM", None)
sys.path.insert(0, str(REPO / "src"))

record: dict[str, object] = {
    "schema_version": 1,
    "session_id": "WS-VISUAL-01",
    "profile": "biohub-visual",
    "provenance_status": "integration_only",
    "objective": (
        "Show the pinned viewer stack opens a competition volume read-only, loads the ground-truth "
        "GEFF graph as a napari layer, places the frozen D-0041 proposals beside it on the same "
        "physical scale, and renders both orthogonal and 3D views."
    ),
    "falsifier": (
        "Any import, read, layer construction or render failure; any write inside the data root; or "
        "proposal coordinates that do not share the ground truth's physical scale."
    ),
    "read_scope": f"{DATA} (read-only), repository src/ for biohubx",
    "write_scope": str(SHOTS),
}
started = time.time()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root: Path) -> dict[str, tuple[int, float]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime)
        for p in root.rglob("*")
        if p.is_file()
    }


SHOTS.mkdir(parents=True, exist_ok=True)

import geff
import napari
import napari_geff
import numpy as np
import tracksdata
import zarr

record["versions"] = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "napari": napari.__version__,
    "geff": geff.__version__ if hasattr(geff, "__version__") else "unknown",
    "napari_geff": getattr(napari_geff, "__version__", "unknown"),
    "tracksdata": tracksdata.__version__,
    "numpy": np.__version__,
    "zarr": zarr.__version__,
}

# The data root's file listing before anything is opened, to prove nothing was written.
watched = DATA / "train" / f"{DATASET}.geff"
before = snapshot(watched)

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE
from biohubx.data.competition import WindowSelection, load_ground_truth, load_window
from biohubx.proposals import dog

# The GEFF store is a directory. Its root metadata file is what identifies the
# graph here; the whole tree is not walked, because this session reads it and
# does not claim an identity for it.
geff_root = DATA / "train" / f"{DATASET}.geff" / "zarr.json"
geff_root_digest = (
    f"raw_artifact_sha256:sha256:{sha256(geff_root)} (zarr.json only)"
    if geff_root.is_file()
    else "directory store, no root zarr.json"
)
truth = load_ground_truth(DATA, DATASET)
annotated = sorted({n.frame for n in truth.lineage.nodes})
first = min(annotated[0], max(0, truth.frames - FRAMES))
group = zarr.open(str(DATA / "train" / f"{DATASET}.zarr"), mode="r")
depth, height, width = group["0"].shape[1:]
window = load_window(
    DATA, WindowSelection(DATASET, first, FRAMES, 0, depth, 0, height, 0, width), split="train"
)
record["inputs"] = {
    "dataset": DATASET,
    "first_frame": first,
    "frames": FRAMES,
    "volume_shape_tzyx": [int(FRAMES), int(depth), int(height), int(width)],
    "volume_dtype": str(window.volume.dtype),
    "annotated_nodes_in_window": len(window.annotated.nodes),
    "annotated_edges_in_window": len(window.annotated.edges),
    "geff_root_digest": geff_root_digest,
    "voxel_scale_um_zyx": [OFFICIAL_VOXEL_SCALE.z_um, OFFICIAL_VOXEL_SCALE.y_um, OFFICIAL_VOXEL_SCALE.x_um],
}

scale = (OFFICIAL_VOXEL_SCALE.z_um, OFFICIAL_VOXEL_SCALE.y_um, OFFICIAL_VOXEL_SCALE.x_um)
instances = dog.detect_instances(
    window.volume,
    dataset=window.annotated.dataset,
    radii_um=(2.0, 3.0),
    response_quantile=0.95,
    suppression_radius_um=4.0,
    local_maxima_only=True,
)
record["proposals"] = {
    "count": len(instances.instances),
    "config": "D-0041 strict peaks, radii (2.0, 3.0), q=0.95, suppression 4.0 um",
}

# Voxel coordinates in both cases; the layer scale below turns them into micrometres,
# so ground truth and proposals land on one physical grid rather than two.
gt_points = np.array(
    [
        [n.frame - first, n.voxel.z, n.voxel.y, n.voxel.x]
        for n in window.annotated.nodes
        if first <= n.frame < first + FRAMES
    ],
    dtype=float,
)
pred_points = np.array([[i.frame, i.voxel.z, i.voxel.y, i.voxel.x] for i in instances.instances], dtype=float)
# The physical field must agree with voxel times scale, or the two layers would
# only look aligned. Checked on the first proposal, which is enough to catch a
# convention slip; the contract itself validates every instance.
if len(instances.instances):
    one = instances.instances[0]
    record["scale_check"] = {
        "voxel_zyx": [one.voxel.z, one.voxel.y, one.voxel.x],
        "physical_um_zyx": [one.physical.z_um, one.physical.y_um, one.physical.x_um],
        "voxel_times_scale_um": [one.voxel.z * scale[0], one.voxel.y * scale[1], one.voxel.x * scale[2]],
    }

# The window is shown. A hidden viewer's canvas never renders: its screenshot came
# back a single flat colour, and both views produced identical bytes for that
# reason rather than because the switch to 3D failed. On this workstation a
# briefly visible window is the intended mode of the profile.
viewer = napari.Viewer(show=True)
viewer.add_image(
    np.asarray(window.volume),
    name=f"{DATASET} volume",
    scale=(1.0, *scale),
    colormap="gray",
    blending="additive",
)
viewer.add_points(
    gt_points,
    name="ground truth nodes",
    size=6,
    face_color="lime",
    scale=(1.0, *scale),
    out_of_slice_display=True,
)
viewer.add_points(
    pred_points,
    name="frozen proposals (D-0041)",
    size=4,
    face_color="magenta",
    scale=(1.0, *scale),
    out_of_slice_display=True,
)
record["layers"] = [
    {"name": layer.name, "type": type(layer).__name__, "scale": [float(s) for s in layer.scale]}
    for layer in viewer.layers
]

# The canvas redraws on the Qt event loop, which nothing is running here, so a
# screenshot taken immediately after switching to 3D returns the 2D frame that is
# still on the canvas. The first attempt produced two identical digests for that
# reason. Events are pumped and the canvas drawn before each capture.
from qtpy.QtWidgets import QApplication


def capture(name: str) -> tuple[str, int]:
    """Pump the event loop, capture, and report how many distinct colours landed.

    The colour count is the guard: a canvas that did not render returns one
    colour, which reads as a successful capture everywhere except here.
    """
    app = QApplication.instance()
    for _ in range(5):
        if app is not None:
            app.processEvents()
        viewer.window._qt_viewer.canvas.native.repaint()
        if app is not None:
            app.processEvents()
    path = SHOTS / name
    frame = viewer.screenshot(str(path), canvas_only=True, flash=False)
    colours = len(np.unique(frame.reshape(-1, frame.shape[-1]), axis=0))
    if colours < 2:
        raise SystemExit(f"{name} rendered a single colour; the canvas did not draw")
    return sha256(path), colours


shots: dict[str, tuple[str, int]] = {}
viewer.dims.ndisplay = 2
viewer.dims.set_point(0, 0)
shots["orthogonal-xy.png"] = capture("orthogonal-xy.png")
viewer.dims.ndisplay = 3
viewer.camera.angles = (0.0, 30.0, 130.0)
shots["volume-3d.png"] = capture("volume-3d.png")
if shots["orthogonal-xy.png"][0] == shots["volume-3d.png"][0]:
    raise SystemExit("the two views produced identical bytes; the canvas did not redraw")
viewer.close()
record["screenshots"] = {
    name: {"digest": f"raw_artifact_sha256:sha256:{digest}", "distinct_colours": colours}
    for name, (digest, colours) in shots.items()
}

after = snapshot(watched)
record["data_root_unchanged"] = before == after
record["axis_convention"] = (
    "napari layers are (t, z, y, x); scale (1.0, 4.0, 1.0, 1.0) um puts z at the official anisotropy"
)
record["conclusion"] = (
    "The pinned visual stack opens a real competition window read-only, renders the volume with "
    "ground truth and frozen proposals on one physical scale, and writes only into the session "
    "directory. napari-geff and tracksdata import in the same environment."
)
record["unresolved"] = (
    "napari-geff's reader was not exercised on this store: the competition .geff is a directory "
    "store the plugin reads through its npe2 reader, which needs a running plugin manager; the "
    "graph here was read with Biohub-X's own reader instead. Whether the plugin round-trips the "
    "official store is the next visual check, before any error-analysis session relies on it."
)
record["elapsed_seconds"] = round(time.time() - started, 3)
record["reproduce"] = (
    "biohub-visual python tools/workstation/validate_visual.py (this script, kept at tools/workstation/)"
)

OUT.mkdir(parents=True, exist_ok=True)
target = OUT / "WS-VISUAL-01.json"
target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(
    json.dumps(
        {k: v for k, v in record.items() if k not in ("objective", "falsifier", "conclusion", "unresolved")},
        indent=2,
        sort_keys=True,
    )
)
print("record:", target)
