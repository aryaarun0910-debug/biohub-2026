r"""H1-R -- recover TRACK IDENTITY for the packaged Zebrahub crops (`kkunizaw/biohub-zh001r`).

WHY THIS EXISTS
---------------
`h1r_zh001r_audit.py` established the limit that narrows the H1 claim: the packaged node arrays
are (N, 4) = [t, z, y, x] with NO track identity, so the dataset supervises a DETECTOR retrain
and NOT the edge/association half -- which is precisely the half our strategic thesis says is
the plateau. We separately hold the full ZSNS001 Ultrack lineage table
(`data/external/zebrahub/ZSNS001_tracks.csv`, 21.7 M rows, t = 0..790, with track_id and
parent_track_id). If the 72 crops can be registered back onto that global track cloud, identity
is recovered by nearest-neighbour lookup and the edge half unlocks from a 363 MB attachable
Kaggle dataset.

THE RECIPE (measured, not assumed)
----------------------------------
1. UNITS. `ZSNS001_tracks.csv` z/y/x are **microns already**, not level-0 voxel indices. Verified
   against the published OME-Zarr pyramid: level-0 is (791, 1, 448, 2174, 2423) at scale
   (1.24, 0.439, 0.439) um -> physical extent 555 x 954 x 1063 um, and the CSV ranges are
   z 27.3-527.0, y 41.7-927.2, x 7.9-1043.5. Interpreting the CSV as voxels would place z at
   812 um, outside the volume. This single wrong assumption flattens every registration score;
   it cost the first pass of this work.
2. MODEL. global_um = origin_um + 1.625 * crop_coord, per-axis, identity axis order, no flips,
   no rotation. The crops are an axis-aligned window on an isotropic 1.625 um grid.
3. SEARCH. Geometric-hash (Hough) vote over the timepoint offset and the origin jointly: take k
   seed nodes from crop frame 0, form every offset to every global node at candidate t, quantise
   offsets to the 1.625 um lattice, and take the modal bin. At the true t the seeds all vote for
   one identical bin; at a wrong t votes scatter. Per-frame node COUNTS are a weak filter here
   (a 64^3 crop holds ~900 of ~27000 nodes/frame), so the vote is used instead.
4. VERIFY. Least-squares per-axis refit against nearest neighbours, then the acceptance gate:
   fraction of crop nodes within 2 um of a global node, over all 20 frames.

Usage:
  .venv\Scripts\python.exe scripts\win_bet\h1r_zh001r_register.py --root <dir-with-zh001r> --crops 0-3
  .venv\Scripts\python.exe scripts\win_bet\h1r_zh001r_register.py --root <dir> --crops all \
      --out data/external/zebrahub/zh001r_identity.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())

TRACKS_CSV = ROOT / "data" / "external" / "zebrahub" / "ZSNS001_tracks.csv"
TRACKS_CACHE = ROOT / "data" / "external" / "zebrahub" / "ZSNS001_tracks_cache.npz"
VOX_UM = 1.625        # measured crop voxel size (exact; see module docstring)
N_T_CROP = 20         # timepoints per packaged crop
ACCEPT_FRAC = 0.90    # >=90% of nodes within ACCEPT_TOL um => registration accepted
ACCEPT_TOL = 2.0


# --------------------------------------------------------------------------- global track cloud
def load_tracks() -> dict:
    """Load the ZSNS001 Ultrack table, t-sorted, with an index into each timepoint."""
    if TRACKS_CACHE.exists():
        d = np.load(TRACKS_CACHE)
        t, z, y, x, tid, pid = (d["t"], d["z"], d["y"], d["x"], d["tid"], d["pid"])
    else:
        import pandas as pd
        if not TRACKS_CSV.exists():
            raise SystemExit(f"missing {TRACKS_CSV}")
        df = pd.read_csv(TRACKS_CSV, engine="pyarrow",
                         usecols=["track_id", "t", "z", "y", "x", "parent_track_id"])
        o = np.argsort(df["t"].to_numpy(), kind="stable")
        t = df["t"].to_numpy(np.int32)[o]
        z = df["z"].to_numpy(np.float32)[o]
        y = df["y"].to_numpy(np.float32)[o]
        x = df["x"].to_numpy(np.float32)[o]
        tid = df["track_id"].to_numpy(np.int64)[o]
        pid = df["parent_track_id"].to_numpy(np.int64)[o]
        TRACKS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(TRACKS_CACHE, t=t, z=z, y=y, x=x, tid=tid, pid=pid)
    P = np.stack([z, y, x], 1).astype(np.float32)   # already microns
    n_t = int(t.max()) + 1
    starts = np.searchsorted(t, np.arange(n_t + 1))
    return {"P": P, "tid": tid, "pid": pid, "starts": starts, "n_t": n_t}


def pts_at(G: dict, t: int) -> np.ndarray:
    return G["P"][G["starts"][t]:G["starts"][t + 1]]


def ids_at(G: dict, t: int) -> tuple[np.ndarray, np.ndarray]:
    s, e = G["starts"][t], G["starts"][t + 1]
    return G["tid"][s:e], G["pid"][s:e]


# --------------------------------------------------------------------------------- packaged pack
def load_nodes(root: Path):
    p = root / "zh001r_nodes.npz"
    if not p.exists():
        raise SystemExit(f"missing zh001r_nodes.npz under {root}")
    return np.load(p, allow_pickle=True)


def crop_pts(nodes, ci: int, ti: int) -> np.ndarray:
    """Crop-box coordinates [z, y, x] (sub-voxel, fractional) for crop `ci`, local frame `ti`."""
    return nodes[f"f{ci * N_T_CROP + ti}"][:, 1:].astype(np.float64)


# ------------------------------------------------------------------------------------- the search
def hough_scan(G: dict, Q: np.ndarray, t_list, k: int = 16,
               seed: int = 0) -> list[tuple[int, int, np.ndarray]]:
    """Vote for (timepoint, origin) jointly. Returns [(votes, t, origin_um), ...] sorted desc.

    At the true timepoint every seed node offsets to the SAME origin, so the modal quantised
    offset collects ~k votes; at a wrong timepoint the votes scatter over distinct bins.
    """
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(Q), size=min(k, len(Q)), replace=False)
    S = Q[sel] * VOX_UM                                   # (k, 3) um, crop-frame
    out = []
    # lattice index range: +-1024 voxels = +-1664 um, comfortably covers the embryo; keep
    # BIG**3 well inside int64 (2048**3 = 2**33) -- a larger BIG silently overflows the hash.
    BIG = np.int64(2048)
    HALF = BIG // 2
    for t in t_list:
        P = pts_at(G, t)
        if len(P) == 0:
            continue
        diff = P[None, :, :] - S[:, None, :]              # (k, M, 3) candidate origins
        q = np.clip(np.rint(diff / VOX_UM), -HALF + 1, HALF - 1).astype(np.int64)
        h = (q[..., 0] + HALF) * BIG * BIG + (q[..., 1] + HALF) * BIG + (q[..., 2] + HALF)
        u, c = np.unique(h.ravel(), return_counts=True)
        j = int(np.argmax(c))
        hv = u[j]
        ix = hv % BIG - HALF
        iy = (hv // BIG) % BIG - HALF
        iz = (hv // (BIG * BIG)) - HALF
        out.append((int(c[j]), t, np.array([iz, iy, ix], float) * VOX_UM))
    out.sort(key=lambda r: -r[0])
    return out


def refine(G: dict, nodes, ci: int, t0: int, origin: np.ndarray, frames=(0, 9, 19),
           iters: int = 6, r0: float = 6.0) -> dict | None:
    """Per-axis least-squares refit of (scale, origin) against nearest neighbours."""
    s = np.full(3, VOX_UM, float)
    off = np.asarray(origin, float).copy()
    r = r0
    for _ in range(iters):
        A_l, B_l = [], []
        for f in frames:
            Q = crop_pts(nodes, ci, f)
            P = pts_at(G, t0 + f)
            d, j = cKDTree(P).query(off + Q * s, distance_upper_bound=r)
            m = np.isfinite(d)
            A_l.append(Q[m]); B_l.append(P[j[m]])
        A, B = np.vstack(A_l), np.vstack(B_l)
        if len(A) < 50:
            return None
        for k in range(3):
            a, b = A[:, k], B[:, k]
            am, bm = a.mean(), b.mean()
            denom = ((a - am) ** 2).sum()
            if denom <= 0:
                return None
            s[k] = ((a - am) * (b - bm)).sum() / denom
            off[k] = bm - s[k] * am
        r = max(2.0, r * 0.7)
    return {"s": s, "off": off}


def evaluate(G: dict, nodes, ci: int, t0: int, s: np.ndarray, off: np.ndarray) -> dict:
    """Acceptance gate over ALL 20 frames: NN residual distribution in microns."""
    res = []
    for f in range(N_T_CROP):
        Q = crop_pts(nodes, ci, f)
        d, _ = cKDTree(pts_at(G, t0 + f)).query(off + Q * s)
        res.append(d)
    r = np.concatenate(res)
    return {"n": int(r.size), "frac_lt_tol": float((r < ACCEPT_TOL).mean()),
            "med": float(np.median(r)), "p90": float(np.percentile(r, 90)),
            "rms": float(np.sqrt((r ** 2).mean())), "max": float(r.max())}


def assign_identity(G: dict, nodes, ci: int, t0: int, s: np.ndarray, off: np.ndarray,
                    tol: float = ACCEPT_TOL):
    """Nearest-neighbour track_id / parent_track_id per crop node, -1 where beyond `tol`."""
    tids, pids = [], []
    for f in range(N_T_CROP):
        Q = crop_pts(nodes, ci, f)
        P = pts_at(G, t0 + f)
        d, j = cKDTree(P).query(off + Q * s, distance_upper_bound=tol)
        gt_id, gp_id = ids_at(G, t0 + f)
        ti = np.full(len(Q), -1, np.int64)
        pi = np.full(len(Q), -1, np.int64)
        m = np.isfinite(d)
        ti[m] = gt_id[j[m]]
        pi[m] = gp_id[j[m]]
        tids.append(ti); pids.append(pi)
    return tids, pids


def _try_t_list(G: dict, nodes, ci: int, t_list, n_cand: int) -> dict | None:
    Q0 = crop_pts(nodes, ci, 0)
    votes = hough_scan(G, Q0, t_list)
    best = None
    for v, t0, org in votes[:n_cand]:
        if t0 + N_T_CROP > G["n_t"]:
            continue
        fit = refine(G, nodes, ci, t0, org)
        if fit is None:
            continue
        ev = evaluate(G, nodes, ci, t0, fit["s"], fit["off"])
        cand = {"crop": ci, "t0": t0, "votes": v, "scale_um": fit["s"].tolist(),
                "origin_um": fit["off"].tolist(), **ev}
        if best is None or cand["frac_lt_tol"] > best["frac_lt_tol"]:
            best = cand
        if cand["frac_lt_tol"] >= ACCEPT_FRAC:
            break
    return best


def register_crop(G: dict, nodes, ci: int, t_lo: int, t_hi: int, n_cand: int = 4,
                  t_prior=(), verbose: bool = True) -> dict:
    """Register one crop. `t_prior` is a cheap candidate list tried first; on failure the
    search falls back to the exhaustive t-scan, so the prior can never hide a real solution."""
    t_start = time.time()
    best = _try_t_list(G, nodes, ci, list(t_prior), n_cand) if t_prior else None
    used_prior = bool(best and best["frac_lt_tol"] >= ACCEPT_FRAC)
    if not used_prior:
        best = _try_t_list(G, nodes, ci, range(t_lo, t_hi), n_cand) or best
    if best is None:
        return {"crop": ci, "ok": False, "reason": "no candidate refined", "sec": time.time() - t_start}
    best["via_prior"] = used_prior
    best["ok"] = bool(best["frac_lt_tol"] >= ACCEPT_FRAC)
    best["sec"] = round(time.time() - t_start, 1)
    if verbose:
        print(f"  crop {ci:3d}: t0={best['t0']:4d} votes={best['votes']:3d} "
              f"frac<{ACCEPT_TOL}um={best['frac_lt_tol']:.4f} med={best['med']:.4f} "
              f"p90={best['p90']:.3f} scale={np.round(best['scale_um'], 6).tolist()} "
              f"origin={np.round(best['origin_um'], 3).tolist()} "
              f"{'OK' if best['ok'] else 'FAIL'} [{best['sec']}s]", flush=True)
    return best


def parse_crops(spec: str, n_max: int) -> list[int]:
    if spec == "all":
        return list(range(n_max))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [c for c in out if c < n_max]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory holding zh001r_nodes.npz")
    ap.add_argument("--crops", default="0-3", help="'all', '0-5', or '0,7,31'")
    ap.add_argument("--t-lo", type=int, default=0)
    ap.add_argument("--t-hi", type=int, default=0, help="0 = all timepoints minus the crop span")
    ap.add_argument("--t-prior", default="",
                    help="comma-separated t0 candidates tried before the exhaustive scan "
                         "(e.g. '158,316,474,632'); failures still fall back to the full scan")
    ap.add_argument("--out", default="", help="npz to write recovered identity into")
    ap.add_argument("--report", default="", help="json summary path")
    args = ap.parse_args()

    nodes = load_nodes(Path(args.root))
    n_crop = len(nodes.files) // N_T_CROP
    t_load = time.time()
    G = load_tracks()
    print(f"tracks: {len(G['P']):,} nodes, t=0..{G['n_t'] - 1} "
          f"({time.time() - t_load:.1f}s); crops: {n_crop}")
    t_hi = args.t_hi or (G["n_t"] - N_T_CROP + 1)

    t_prior = [int(v) for v in args.t_prior.split(",") if v.strip()] if args.t_prior else []
    results, ident = [], {}
    for ci in parse_crops(args.crops, n_crop):
        r = register_crop(G, nodes, ci, args.t_lo, t_hi, t_prior=t_prior)
        results.append(r)
        if r.get("ok") and args.out:
            tids, pids = assign_identity(G, nodes, ci, r["t0"],
                                         np.array(r["scale_um"]), np.array(r["origin_um"]))
            for f in range(N_T_CROP):
                ident[f"tid_{ci}_{f}"] = tids[f]
                ident[f"pid_{ci}_{f}"] = pids[f]

    ok = [r for r in results if r.get("ok")]
    print(f"\nREGISTERED {len(ok)}/{len(results)} crops at >={ACCEPT_FRAC:.0%} within {ACCEPT_TOL} um")
    if ok:
        fr = np.array([r["frac_lt_tol"] for r in ok])
        md = np.array([r["med"] for r in ok])
        print(f"  frac within tol : min {fr.min():.4f} median {np.median(fr):.4f} max {fr.max():.4f}")
        print(f"  median residual : min {md.min():.5f} median {np.median(md):.5f} max {md.max():.5f} um")
        print(f"  t0 values       : {sorted(r['t0'] for r in ok)}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **ident)
        print(f"  identity written: {args.out} ({len(ident) // 2} crop-frames)")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
