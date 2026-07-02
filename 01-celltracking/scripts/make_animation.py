"""Time-lapse GIFs of real crops (for Codex + humans to inspect data dynamics).

anim_xy_timelapse.gif : XY max-projection over time with GT cells overlaid ->
    shows cell motion, density, and how few cells are annotated vs how many are visible.
anim_3d_rotate.gif    : the 3D GT lineage rotating -> shows lineage structure / divisions.

Uses matplotlib -> PIL frames -> GIF (no external tools). Writes to reports/figures/.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TRAIN = ROOT / "data" / "train"


def _fig_to_rgba(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf.copy()


def save_gif(frames, path, fps=6):
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, disposal=2)


def pick_crop():
    import csv
    rows = list(csv.DictReader((ROOT / "reports/inventory/embryo_stats.csv").open(encoding="utf-8")))
    rows = [r for r in rows if (TRAIN / f"{r['embryo']}.zarr").exists()]
    return max(rows, key=lambda r: int(r["n_div"]))["embryo"]


def timelapse_xy(name, stride=4):
    import zarr
    import tracksdata as td
    grp = zarr.open_group(str(TRAIN / f"{name}.zarr"), mode="r")
    arr = grp["0"]
    T = arr.shape[0]
    g = td.graph.IndexedRXGraph.from_geff(TRAIN / f"{name}.geff")
    g = g[0] if isinstance(g, tuple) else g
    na = g.node_attrs(attr_keys=["t", "y", "x"])
    tt = np.asarray(na["t"].to_list()); yy = na["y"].to_numpy(); xx = na["x"].to_numpy()
    vmax = float(np.quantile(np.asarray(arr[T // 2]), 0.999))

    frames = []
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    for t in range(0, T, stride):
        ax.clear()
        ax.imshow(np.asarray(arr[t]).max(axis=0), cmap="gray", vmax=vmax, origin="upper")
        m = tt == t
        ax.scatter(xx[m], yy[m], s=42, facecolors="none", edgecolors="#E69F00", linewidths=1.5)
        ax.set_title(f"{name}  XY max-proj  t={t}/{T-1}   ({int(m.sum())} GT cells)",
                     fontsize=11, color="#222")
        ax.set_xticks([]); ax.set_yticks([])
        frames.append(_fig_to_rgba(fig))
    plt.close(fig)
    save_gif(frames, FIG / "anim_xy_timelapse.gif", fps=6)
    print("anim_xy_timelapse.gif ok", len(frames), "frames")


def rotate_3d(name):
    import tracksdata as td
    g = td.graph.IndexedRXGraph.from_geff(TRAIN / f"{name}.geff")
    g = g[0] if isinstance(g, tuple) else g
    na = g.node_attrs(attr_keys=["t", "z", "y", "x"])
    t = np.asarray(na["t"].to_list())
    X = na["x"].to_numpy() * 0.40625; Y = na["y"].to_numpy() * 0.40625; Z = na["z"].to_numpy() * 1.625

    frames = []
    fig = plt.figure(figsize=(5.4, 5.0), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    for az in range(0, 360, 8):
        ax.clear()
        ax.scatter(X, Y, Z, c=t, cmap="viridis", s=10, alpha=0.7)
        ax.set_title(f"{name}  3D GT lineage", fontsize=11, color="#222")
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)"); ax.set_zlabel("z (um)")
        ax.view_init(elev=22, azim=az)
        frames.append(_fig_to_rgba(fig))
    plt.close(fig)
    save_gif(frames, FIG / "anim_3d_rotate.gif", fps=12)
    print("anim_3d_rotate.gif ok", len(frames), "frames")


def main():
    name = pick_crop()
    print("crop:", name)
    timelapse_xy(name)
    rotate_3d(name)
    print("animations ->", FIG)


if __name__ == "__main__":
    main()
