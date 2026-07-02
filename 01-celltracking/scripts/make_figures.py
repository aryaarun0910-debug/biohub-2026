"""Generate research-journal figures from REAL data into reports/figures/.

Palette: Okabe-Ito (a published colorblind-safe categorical scheme) -> the two embryos
get fixed hues (44b6=blue, 6bba=orange), never cycled. Clean thin marks, direct labels,
recessive axes. Figures:
  fig1_label_sparsity   - annotated/estimated fraction per crop (the sparse-label problem)
  fig2_edges_per_crop   - annotated edge volume per crop, per embryo (fold imbalance)
  fig3_recall_vs_adjj   - node recall -> adjusted edge Jaccard (recall drives score)
  fig4_fn_taxonomy      - 3-way missed-edge split per embryo (no-candidate dominates)
  fig5_norm_ablation    - per-frame vs precomputed normalization (no meaningful difference)
  fig6_crop_2d3d        - a real crop: XY/XZ max projections + 3D GT lineage scatter
"""

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
STATS = ROOT / "reports" / "inventory" / "embryo_stats.csv"

# Okabe-Ito colorblind-safe categorical hues (fixed per entity)
C = {"44b6": "#0072B2", "6bba": "#E69F00"}
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_stats():
    rows = list(csv.DictReader(STATS.open(encoding="utf-8")))
    for r in rows:
        for k in ("n_nodes", "n_edges", "n_div"):
            r[k] = int(r[k])
        r["label_fraction"] = float(r["label_fraction"])
        r["fam"] = r["embryo"].split("_")[0]
    return rows


def _strip(ax, groups, values, ylabel, title, logy=False):
    for i, (g, vals) in enumerate(zip(groups, values)):
        x = np.random.default_rng(i).normal(i, 0.06, len(vals))
        ax.scatter(x, vals, s=14, c=C[g], alpha=0.55, edgecolors="white", linewidths=0.4, zorder=3)
        med = np.median(vals)
        ax.plot([i - 0.28, i + 0.28], [med, med], color=C[g], lw=2.5, zorder=4)
        ax.text(i, max(vals) if not logy else max(vals), f"  n={len(vals)}", va="bottom", ha="center",
                color=MUTED, fontsize=9)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"embryo {g}" for g in groups])
    ax.set_ylabel(ylabel); ax.set_title(title, color=INK, fontweight="bold", loc="left")
    if logy:
        ax.set_yscale("log")


def fig1_sparsity(rows):
    fams = ["44b6", "6bba"]
    vals = [[r["label_fraction"] * 100 for r in rows if r["fam"] == f] for f in fams]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _strip(ax, fams, vals, "annotated / estimated cells (%)",
           "Label sparsity per crop  (median ~1-2% of cells annotated)")
    ax.axhline(np.median([v for vv in vals for v in vv]), color=MUTED, ls="--", lw=1)
    fig.tight_layout(); fig.savefig(FIG / "fig1_label_sparsity.png"); plt.close(fig)


def fig2_edges(rows):
    fams = ["44b6", "6bba"]
    vals = [[r["n_edges"] for r in rows if r["fam"] == f] for f in fams]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _strip(ax, fams, vals, "annotated edges per crop (log)",
           "Edge volume per crop  (6bba dominates: 128 crops / 109k edges)", logy=True)
    fig.tight_layout(); fig.savefig(FIG / "fig2_edges_per_crop.png"); plt.close(fig)


def _parse_taxonomy_txt(path):
    """Parse per-crop rows: crop fam Npred Nest ratio recall adjJ ..."""
    out = []
    for ln in Path(path).read_text().splitlines():
        m = re.match(r"^(\S+)\s+(44b6|6bba)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", ln)
        if m:
            out.append({"fam": m.group(2), "ratio": float(m.group(5)),
                        "recall": float(m.group(6)), "adjJ": float(m.group(7))})
    return out


def fig3_recall_adjj():
    src = ROOT / "reports" / "v3_taxonomy_20crops.txt"
    if not src.exists():
        return
    d = _parse_taxonomy_txt(src)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for f in ["44b6", "6bba"]:
        pts = [(r["recall"], r["adjJ"]) for r in d if r["fam"] == f]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=42, c=C[f], alpha=0.8, edgecolors="white", linewidths=0.6,
                   label=f"embryo {f}", zorder=3)
    ax.set_xlabel("node recall (fraction of GT cells detected within 7 um)")
    ax.set_ylabel("adjusted edge Jaccard")
    ax.set_title("Recall drives score  (V3 DoG, per crop)", color=INK, fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig3_recall_vs_adjj.png"); plt.close(fig)


def fig4_taxonomy():
    # 3-way FN split, per-embryo aggregates from the corrected taxonomy (10 crops/embryo)
    data = {"44b6": (79, 0, 21), "6bba": (70, 1, 29)}
    cats = ["no-candidate\n(undetected)", "lost-assignment\n(arbitration)", "association\n(linking)"]
    seg = ["#0072B2", "#009E73", "#CC79A7"]  # Okabe-Ito blue/green/purple by FAILURE MODE
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    fams = list(data)
    bottom = np.zeros(len(fams))
    for k, cat in enumerate(cats):
        vals = np.array([data[f][k] for f in fams], float)
        bars = ax.bar([f"embryo {f}" for f in fams], vals, bottom=bottom, color=seg[k],
                      width=0.6, label=cat, edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            if v >= 6:
                ax.text(b.get_x() + b.get_width() / 2, b.get_y() + v / 2, f"{v:.0f}%",
                        ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        bottom += vals
    ax.set_ylabel("share of missed GT edges (%)"); ax.set_ylim(0, 100)
    ax.set_title("Why edges are missed", color=INK, fontweight="bold", loc="left")
    ax.text(0, 104, "no-candidate (undetected) dominates -> raise recall", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=9)
    fig.subplots_adjust(right=0.72, top=0.86)
    fig.savefig(FIG / "fig4_fn_taxonomy.png"); plt.close(fig)


def fig5_norm():
    modes = ["per-frame\n(V3)", "precomputed\n(host)"]
    adjJ = [0.7466, 0.7417]; recall = [0.822, 0.820]
    x = np.arange(len(modes)); w = 0.35
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    b1 = ax.bar(x - w / 2, adjJ, w, color="#0072B2", label="adjusted edge Jaccard", edgecolor="white", linewidth=2)
    b2 = ax.bar(x + w / 2, recall, w, color="#56B4E9", label="node recall", edgecolor="white", linewidth=2)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}",
                    ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(modes); ax.set_ylim(0, 1.0)
    ax.set_title("Normalization ablation  (no meaningful difference)", color=INK, fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig5_norm_ablation.png"); plt.close(fig)


def fig6_crop(rows):
    import zarr
    import tracksdata as td
    # pick the crop with the most divisions for an interesting lineage
    r = max(rows, key=lambda r: r["n_div"])
    name = r["embryo"]
    grp = zarr.open_group(str(ROOT / "data/train" / f"{name}.zarr"), mode="r")
    arr = grp["0"]
    T = arr.shape[0]
    t_mid = T // 2
    vol = np.asarray(arr[t_mid]).astype(np.float32)      # (Z,Y,X)
    mip_xy = vol.max(axis=0)                              # (Y,X)
    mip_xz = vol.max(axis=1)                              # (Z,X)

    g = td.graph.IndexedRXGraph.from_geff(ROOT / "data/train" / f"{name}.geff")
    g = g[0] if isinstance(g, tuple) else g
    na = g.node_attrs(attr_keys=["t", "z", "y", "x"])
    t = np.asarray(na["t"].to_list()); z = na["z"].to_numpy()
    y = na["y"].to_numpy(); x = na["x"].to_numpy()

    fig = plt.figure(figsize=(12.5, 4.4))
    vmax = np.quantile(vol, 0.999)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(mip_xy, cmap="gray", vmax=vmax, origin="upper")
    m = t == t_mid
    ax1.scatter(x[m], y[m], s=30, facecolors="none", edgecolors="#E69F00", linewidths=1.3)
    ax1.set_title(f"{name}  t={t_mid}: XY max-proj + GT cells", fontsize=10, loc="left", color=INK)
    ax1.set_xlabel("x (vox)"); ax1.set_ylabel("y (vox)"); ax1.grid(False)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(mip_xz, cmap="gray", vmax=vmax, aspect=1.625 / 0.40625, origin="upper")
    ax2.scatter(x[m], z[m], s=30, facecolors="none", edgecolors="#E69F00", linewidths=1.3)
    ax2.set_title("XZ max-proj (anisotropy: Z 4x coarser)", fontsize=10, loc="left", color=INK)
    ax2.set_xlabel("x (vox)"); ax2.set_ylabel("z (vox)"); ax2.grid(False)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    sc = ax3.scatter(x * 0.40625, y * 0.40625, z * 1.625, c=t, cmap="viridis", s=10, alpha=0.7)
    # draw lineage edges
    ea = g.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
    na_ids = g.node_ids()
    idpos = {int(nid): (x[k] * 0.40625, y[k] * 0.40625, z[k] * 1.625)
             for k, nid in enumerate(na_ids)}
    for s_, tg_ in zip(ea[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(),
                       ea[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()):
        if int(s_) in idpos and int(tg_) in idpos:
            p, q = idpos[int(s_)], idpos[int(tg_)]
            ax3.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#888888", lw=0.5, alpha=0.5)
    ax3.set_title(f"3D GT lineage ({g.num_nodes()} nodes, {r['n_div']} divisions)", fontsize=10, color=INK)
    ax3.set_xlabel("x (um)"); ax3.set_ylabel("y (um)"); ax3.set_zlabel("z (um)")
    cb = fig.colorbar(sc, ax=ax3, shrink=0.6, pad=0.1); cb.set_label("timepoint", color=MUTED)
    fig.tight_layout(); fig.savefig(FIG / "fig6_crop_2d3d.png"); plt.close(fig)


def main():
    rows = load_stats()
    fig1_sparsity(rows); print("fig1 ok")
    fig2_edges(rows); print("fig2 ok")
    fig3_recall_adjj(); print("fig3 ok")
    fig4_taxonomy(); print("fig4 ok")
    fig5_norm(); print("fig5 ok")
    fig6_crop(rows); print("fig6 ok")
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
