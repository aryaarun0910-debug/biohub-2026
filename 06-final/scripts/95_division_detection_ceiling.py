"""G95: is there a hard ceiling on division recall from the detector's peak suppression?

Three questions, all answerable from ground truth + artifacts/cache (no prediction
graphs, no re-running the tracker):

  A. sister separation at the instant of division, over all 199 films / 151 events
  B. what the max_pool3d suppression radius ACTUALLY is after quantisation, and how
     many divisions fall under it
  C. what a smaller suppression radius would cost in predicted node count

Coordinate conventions (mixing these is the classic error here):
  GT geff          -> ORIGINAL voxels, anisotropic scale (1.625, 0.40625, 0.40625) um
  artifacts/cache  -> DOWNSAMPLED (1,4,4) grid, ISOTROPIC 1.625 um/voxel
Everything geometric in this file happens in micrometres.

Usage:
    python scripts/95_division_detection_ceiling.py              # A + B + C (cache only)
    python scripts/95_division_detection_ceiling.py --kernels 24 # + GPU kernel sweep
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub import io  # noqa: E402

CACHE = ROOT / "artifacts" / "cache"
OUT = ROOT / "artifacts"
GRID_UM = 1.625          # downsampled grid pitch, isotropic
MATCH_RADIUS_UM = 7.0    # the metric's node-matching radius
POOL_KERNEL_UM = 5.0     # config value in weights/.../config.json
SHELLS_UM = (3.25, 4.0, 5.0, 6.0, 7.0, 8.0)


def pool_kernel(um: float, pitch: float = GRID_UM) -> int:
    """Exactly biohub.model.pool_kernel, one axis (the grid is isotropic)."""
    k = max(1, round(um / pitch))
    return k + 1 if k % 2 == 0 else k


def min_separation_um(kernel: int, pitch: float = GRID_UM) -> float:
    """Two voxels can both survive max_pool3d(kernel) only if they are at least
    (kernel+1)//2 voxels apart in Chebyshev distance."""
    return ((kernel + 1) // 2) * pitch


# --------------------------------------------------------------------------
# per-film worker
# --------------------------------------------------------------------------

def one_film(film: str) -> dict:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    gp = io.dataset_root() / "train" / f"{film}.geff"
    gt = io.read_geff(gp)
    embryo = film.split("_")[0]

    n = len(gt["ids"])
    idx = {int(v): i for i, v in enumerate(gt["ids"])}
    zyx_vox = np.stack([gt["z"], gt["y"], gt["x"]], 1).astype(np.float64)
    zyx_um = io.to_um(zyx_vox)                       # ORIGINAL voxels -> um
    gt_grid = zyx_um / GRID_UM                       # -> downsampled isotropic grid
    gt_t = gt["t"].astype(np.int64)

    children: dict[int, list[int]] = {}
    parent_of: dict[int, int] = {}
    for a, b in gt["edges"]:
        ia, ib = idx[int(a)], idx[int(b)]
        children.setdefault(ia, []).append(ib)
        parent_of[ib] = ia

    # ---- detections from the local cache (downsampled grid) ----
    det_um = det_t = det_p = None
    npz = CACHE / f"{film}.npz"
    film_row = {"film": film, "embryo": embryo, "n_gt_nodes": n,
                "est_nodes": float(gt["estimated_number_of_nodes"])}
    p2g: dict[int, int] = {}
    g2p: dict[int, int] = {}
    if npz.exists():
        d = np.load(npz)
        det_t = d["det_t"].astype(np.int64)
        det_um = d["det_zyx"].astype(np.float64) * GRID_UM
        det_p = d["det_p"].astype(np.float32)
        film_row["n_det"] = int(len(det_t))

        # detection crowding: NN distance per detection, within its own frame
        nn_all, min_pair = [], np.inf
        pairs_in = {r: 0 for r in SHELLS_UM}
        for t in np.unique(det_t):
            sel = np.where(det_t == t)[0]
            if len(sel) < 2:
                continue
            tree = cKDTree(det_um[sel])
            dd, _ = tree.query(det_um[sel], k=2)
            nn_all.append(dd[:, 1])
            min_pair = min(min_pair, float(dd[:, 1].min()))
            for r in SHELLS_UM:
                c = tree.count_neighbors(tree, r)
                pairs_in[r] += int((c - len(sel)) // 2)
        nn_all = np.concatenate(nn_all) if nn_all else np.zeros(0)
        film_row["det_nn_min"] = float(min_pair) if np.isfinite(min_pair) else np.nan
        film_row["det_nn_median"] = float(np.median(nn_all)) if len(nn_all) else np.nan
        film_row["det_nn_p01"] = float(np.percentile(nn_all, 1)) if len(nn_all) else np.nan
        film_row["det_nn_p05"] = float(np.percentile(nn_all, 5)) if len(nn_all) else np.nan
        for r in SHELLS_UM:
            film_row[f"det_pairs_lt_{r}"] = pairs_in[r]
            film_row[f"det_frac_nn_lt_{r}"] = (
                float((nn_all < r).mean()) if len(nn_all) else np.nan)

        # metric-style per-frame Hungarian matching GT <-> detections at 7 um
        for t in np.unique(gt_t):
            gi = np.where(gt_t == t)[0]
            pi = np.where(det_t == t)[0]
            if not len(gi) or not len(pi):
                continue
            dm = np.linalg.norm(gt_grid[gi][:, None] * GRID_UM - det_um[pi][None], axis=-1)
            r, c = linear_sum_assignment(dm)
            for i, j in zip(r, c):
                if dm[i, j] <= MATCH_RADIUS_UM:
                    g2p[int(gi[i])] = int(pi[j])
                    p2g[int(pi[j])] = int(gi[i])
        film_row["gt_node_recall"] = len(g2p) / n if n else np.nan

    # ---- divisions ----
    rows = []
    for par, ch in children.items():
        if len(ch) < 2:
            continue
        c = sorted(ch, key=lambda i: -np.linalg.norm(zyx_um[i] - zyx_um[par]))
        a, b = c[0], c[1]
        dt_a = int(gt_t[a] - gt_t[par])
        dt_b = int(gt_t[b] - gt_t[par])
        dvec = zyx_um[a] - zyx_um[b]
        sep = float(np.linalg.norm(dvec))
        # per-axis separation expressed in DOWNSAMPLED grid voxels: the max_pool
        # constraint is Chebyshev, not Euclidean
        dvox = np.abs(dvec) / GRID_UM
        cheb_float = float(dvox.max())
        ga = np.rint(gt_grid[a]).astype(int)
        gb = np.rint(gt_grid[b]).astype(int)
        cheb_int = int(np.abs(ga - gb).max())

        r = {
            "film": film, "embryo": embryo,
            "parent_id": int(gt["ids"][par]), "t_parent": int(gt_t[par]),
            "n_children": len(ch), "dt_a": dt_a, "dt_b": dt_b,
            "sister_sep_um": sep,
            "sister_dz_um": float(abs(dvec[0])), "sister_dy_um": float(abs(dvec[1])),
            "sister_dx_um": float(abs(dvec[2])),
            "sister_cheb_vox": cheb_float, "sister_cheb_vox_rounded": cheb_int,
            "pd_a_um": float(np.linalg.norm(zyx_um[a] - zyx_um[par])),
            "pd_b_um": float(np.linalg.norm(zyx_um[b] - zyx_um[par])),
            "grandparent": int(par in parent_of),
        }
        # detection-stage fate of this division
        if det_um is not None:
            pa, pb = g2p.get(a), g2p.get(b)
            r["parent_matched"] = int(par in g2p)
            r["a_matched"] = int(pa is not None)
            r["b_matched"] = int(pb is not None)
            r["both_matched"] = int(pa is not None and pb is not None)
            for who, gidx, mine, other in (("a", a, pa, pb), ("b", b, pb, pa)):
                selt = np.where(det_t == gt_t[gidx])[0]
                if len(selt):
                    dd = np.linalg.norm(det_um[selt] - gt_grid[gidx] * GRID_UM, axis=-1)
                    k = int(np.argmin(dd))
                    r[f"{who}_nearest_det_um"] = float(dd[k])
                    r[f"{who}_nearest_is_sisters"] = int(
                        other is not None and int(selt[k]) == other)
                else:
                    r[f"{who}_nearest_det_um"] = np.nan
                    r[f"{who}_nearest_is_sisters"] = 0
        rows.append(r)

    film_row["divisions"] = len(rows)
    return {"film_row": film_row, "div_rows": rows}


# --------------------------------------------------------------------------
# GPU stage: what the peak kernel actually costs in node count
# --------------------------------------------------------------------------

def kernel_sweep(films: list[str], kernels=(1, 3, 5, 7), device="mps") -> list[dict]:
    import torch
    from biohub import model as M

    m, _cfg = M.load("primary", device)
    out = []
    for film in films:
        zp = io.dataset_root() / "train" / f"{film}.zarr"
        gp = io.dataset_root() / "train" / f"{film}.geff"
        q = io.quantiles(zp)
        q_low, q_high = float(q["0.001"]), float(q["0.999"])
        T = io.image_meta(zp)["shape"][0]
        est = float(io.read_geff(gp)["estimated_number_of_nodes"])
        counts = {k: 0 for k in kernels}
        supra = 0
        for ws in range(0, T - M.WINDOW + 1):
            ts = list(range(ws, ws + M.WINDOW))
            imgs = torch.stack([M.normalise(io.read_frame(zp, t), q_low, q_high) for t in ts])
            with torch.no_grad():
                _unet, det = m.encode(imgs.unsqueeze(0).to(device))
            for i, _t in enumerate(ts):
                if ws > 0 and i == 0:
                    continue
                lg = det[i][0]
                supra += int((torch.sigmoid(lg) > M.DET_THRESHOLD).sum())
                for k in kernels:
                    counts[k] += len(M.peaks(lg, kernel=(k, k, k)))
        row = {"film": film, "embryo": film.split("_")[0], "est_nodes": est,
               "suprathreshold_voxels": supra}
        for k in kernels:
            row[f"n_k{k}"] = counts[k]
            row[f"ratio_k{k}"] = counts[k] / est
        out.append(row)
        print(f"  {film}  " + "  ".join(f"k{k}={counts[k]}" for k in kernels)
              + f"  est={est:.0f}", flush=True)
    return out


# --------------------------------------------------------------------------

def qs(x: np.ndarray) -> dict:
    p = [0, 5, 10, 25, 50, 75, 90, 100]
    v = np.percentile(x, p)
    return {"n": int(len(x)), "min": float(v[0]), "p5": float(v[1]), "p10": float(v[2]),
            "p25": float(v[3]), "median": float(v[4]), "p75": float(v[5]),
            "p90": float(v[6]), "max": float(v[7]), "mean": float(x.mean())}


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", type=int, default=0,
                    help="run the GPU peak-kernel sweep on this many films (0 = skip)")
    ap.add_argument("--procs", type=int, default=18)
    args = ap.parse_args()

    films = io.films("train")
    t0 = time.perf_counter()
    with mp.Pool(args.procs) as pool:
        res = pool.map(one_film, films, chunksize=2)
    print(f"{len(films)} films in {time.perf_counter() - t0:.1f}s on {args.procs} procs\n")

    film_rows = [r["film_row"] for r in res]
    div_rows = [d for r in res for d in r["div_rows"]]
    write_csv(OUT / "division_ceiling_divisions.csv", div_rows)
    write_csv(OUT / "division_ceiling_films.csv", film_rows)

    sep = np.array([d["sister_sep_um"] for d in div_rows])
    emb = np.array([d["embryo"] for d in div_rows])
    pd_um = np.array([d["pd_a_um"] for d in div_rows] + [d["pd_b_um"] for d in div_rows])
    cheb = np.array([d["sister_cheb_vox"] for d in div_rows])
    cheb_i = np.array([d["sister_cheb_vox_rounded"] for d in div_rows])
    dt_ok = np.array([d["dt_a"] == 1 and d["dt_b"] == 1 for d in div_rows])

    k_real = pool_kernel(POOL_KERNEL_UM)
    floor = min_separation_um(k_real)

    summary: dict = {
        "n_films": len(films), "n_divisions": len(div_rows),
        "frame_convention_ok": {"daughters_at_t_plus_1": int(dt_ok.sum()),
                                "other": int((~dt_ok).sum())},
        "kernel": {"pool_kernel_um_config": POOL_KERNEL_UM, "grid_pitch_um": GRID_UM,
                   "kernel_voxels": k_real,
                   "hard_min_separation_um": floor,
                   "kernel_um_range_giving_same_kernel": [
                       round((k_real - 1) / 2 * GRID_UM + 1e-9, 4),
                       round((k_real + 1) / 2 * GRID_UM, 4)],
                   "next_smaller_kernel": 1,
                   "next_smaller_kernel_min_sep_um": min_separation_um(1)},
        "sister_separation_um": qs(sep),
        "parent_daughter_um": qs(pd_um),
        "fraction_below": {f"{r}": float((sep < r).mean()) for r in
                           (3.25, 4.0, 5.0, 6.0, 7.0)},
        "count_below": {f"{r}": int((sep < r).sum()) for r in
                        (3.25, 4.0, 5.0, 6.0, 7.0)},
        "chebyshev_voxels": {
            "frac_max_axis_below_2vox": float((cheb < 2.0).mean()),
            "count_max_axis_below_2vox": int((cheb < 2.0).sum()),
            "frac_rounded_cheb_lt_2": float((cheb_i < 2).mean()),
            "count_rounded_cheb_lt_2": int((cheb_i < 2).sum()),
        },
        "by_embryo": {},
    }
    for e in sorted(set(emb)):
        s = sep[emb == e]
        c = cheb[emb == e]
        summary["by_embryo"][e] = {
            **qs(s),
            "frac_below_3.25": float((s < 3.25).mean()),
            "frac_below_4.0": float((s < 4.0).mean()),
            "frac_below_5.0": float((s < 5.0).mean()),
            "frac_below_6.0": float((s < 6.0).mean()),
            "frac_max_axis_below_2vox": float((c < 2.0).mean()),
        }

    # what each candidate kernel would structurally forbid.  max_pool3d(k) lets two
    # peaks coexist only at Chebyshev >= (k+1)//2 voxels, so the per-axis separation
    # -- not the Euclidean one -- is the test.
    summary["kernel_floor_vs_divisions"] = {}
    for k in (1, 3, 5, 7, 9):
        need = (k + 1) // 2
        summary["kernel_floor_vs_divisions"][f"k{k}"] = {
            "min_sep_um": min_separation_um(k),
            "divisions_unresolvable": int((cheb < need).sum()),
            "fraction": float((cheb < need).mean()),
            "by_embryo": {e: int((cheb[emb == e] < need).sum()) for e in sorted(set(emb))},
        }

    # detection-stage fate of divisions, straight from the cache
    if div_rows and "both_matched" in div_rows[0]:
        bm = np.array([d["both_matched"] for d in div_rows])
        am = np.array([d["a_matched"] for d in div_rows])
        bb = np.array([d["b_matched"] for d in div_rows])
        steal = np.array([d["a_nearest_is_sisters"] or d["b_nearest_is_sisters"]
                          for d in div_rows])
        lost = (am + bb) < 2
        summary["detection_stage"] = {
            "both_daughters_detected": int(bm.sum()),
            "one_or_both_lost": int(lost.sum()),
            "lost_and_nearest_det_is_sisters": int((lost & (steal > 0)).sum()),
            "sister_sep_when_both_detected_median": float(np.median(sep[bm == 1]))
            if bm.sum() else np.nan,
            "sister_sep_when_lost_median": float(np.median(sep[lost]))
            if lost.sum() else np.nan,
            "lost_with_sep_below_floor": int((lost & (sep < floor)).sum()),
        }

    # ---- cost side: detection crowding across all 199 films ----
    have = [r for r in film_rows if "n_det" in r]
    n_det = sum(r["n_det"] for r in have)
    n_est = sum(r["est_nodes"] for r in have)
    crowd = {"n_det_total": int(n_det), "est_nodes_total": float(n_est),
             "det_over_est": n_det / n_est,
             "det_nn_min_over_corpus": float(np.nanmin([r["det_nn_min"] for r in have])),
             "det_nn_median_of_film_medians": float(
                 np.nanmedian([r["det_nn_median"] for r in have]))}
    for r in SHELLS_UM:
        # NOTE cKDTree.count_neighbors is INCLUSIVE (d <= r); frac_det_with_neighbour
        # is STRICT (d < r).  The gap between the two at r=3.25 is the population of
        # pairs sitting exactly on the suppression floor.
        pairs = sum(x[f"det_pairs_lt_{r}"] for x in have)
        crowd[f"pairs_le_{r}um"] = int(pairs)
        crowd[f"pairs_le_{r}um_per_1000_det"] = 1000.0 * pairs / n_det
        crowd[f"frac_det_with_neighbour_strictly_lt_{r}"] = float(
            np.average([x[f"det_frac_nn_lt_{r}"] for x in have],
                       weights=[x["n_det"] for x in have]))
    summary["detection_crowding"] = crowd

    kcsv = OUT / "division_ceiling_kernels.csv"
    krows = None
    if args.kernels:
        pick = ([f for f in films if f.startswith("44b6")][: args.kernels // 2]
                + [f for f in films if f.startswith("6bba")][: args.kernels - args.kernels // 2])
        print(f"kernel sweep on {len(pick)} films ...", flush=True)
        krows = kernel_sweep(pick)
        write_csv(kcsv, krows)
    elif kcsv.exists():
        import csv as _csv
        krows = [{k: (v if k in ("film", "embryo") else float(v)) for k, v in r.items()}
                 for r in _csv.DictReader(open(kcsv))]
    if krows:
        ks = {"n_films": len(krows)}
        for k in (1, 3, 5, 7):
            tot = sum(r[f"n_k{k}"] for r in krows)
            ke = sum(r["est_nodes"] for r in krows)
            ks[f"k{k}"] = {"kernel": k, "min_sep_um": min_separation_um(k),
                           "n_peaks": int(tot), "ratio_to_est": tot / ke}
        ks["vs_current"] = {f"k{k}": ks[f"k{k}"]["n_peaks"] / ks["k3"]["n_peaks"]
                            for k in (1, 3, 5, 7)}
        summary["kernel_sweep"] = ks

    # ---- the trade, in competition score units ----
    # score = adj_J_edge + 0.1 * div_jaccard ; div_jaccard is micro-averaged over
    # the whole eval set, so one FN->TP is worth ~0.1/D.  The node penalty scales
    # the EDGE term only: adj = J * (1 - 0.1*(n_pred-n_est)/n_est).
    D = len(div_rows)
    J_EDGE = 0.83   # measured on this pipeline, artifacts/baseline_repair.csv
    trade = {
        "D_labelled_divisions": D,
        "score_per_division": 0.1 / D,
        "J_edge_assumed": J_EDGE,
        "score_per_1pct_extra_nodes": 0.1 * 0.01 * J_EDGE,
        "divisions_equivalent_of_1pct_nodes": (0.1 * 0.01 * J_EDGE) / (0.1 / D),
        "current_node_overprediction": crowd["det_over_est"] - 1.0,
        "current_node_penalty_score": 0.1 * (crowd["det_over_est"] - 1.0) * J_EDGE,
    }
    if krows:
        for k in (1, 5, 7):
            dn = ks[f"k{k}"]["ratio_to_est"] - ks["k3"]["ratio_to_est"]
            need = (k + 1) // 2
            gained = int((cheb >= (3 + 1) // 2).sum()) - int((cheb >= need).sum())
            trade[f"k3_to_k{k}"] = {
                "node_ratio_delta": dn,
                "node_penalty_score_delta": -0.1 * dn * J_EDGE,
                "divisions_delta": gained,
                "division_score_delta": gained * 0.1 / D,
                "net_score_delta": gained * 0.1 / D - 0.1 * dn * J_EDGE,
            }
    summary["trade"] = trade

    (OUT / "division_ceiling_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
