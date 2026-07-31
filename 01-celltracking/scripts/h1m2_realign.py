"""H1-M2: repair the EXTERNAL (Zebrahub) division-time anchor.

Lane C proved the Zebrahub appearance route fails because the LABEL TIME is wrong, not
because the imagery or the representation is wrong (mean SMD 0.21, no feature over 1 SD,
coordinate audit AUC 0.941, yet competition transfer AUC 0.494 / 0.532 = chance).

This module does the label-side half of the repair, using TRACK GEOMETRY ONLY -- no
imagery is read here, so the appearance statistic that later judges the repair
(forward core-mass collapse) is never used to choose the anchor. That separation is the
whole point: a realignment chosen by maximising core-mass collapse would prove nothing.

Two modes:

  census   Measure, on both corpora, the physical geometry of a division as a function
           of frames-after-the-annotated-fork:
             * daughter-daughter separation  sep(k)
             * annotated-mother -> daughter-midpoint drift  drift(k)
             * the per-frame displacement CLOCK of ordinary (non-dividing) cells
           The competition curve defines the target configuration; the Zebrahub curve
           says how many frames after its own annotation that configuration is reached.

  build    Emit a realigned external event table: for every Zebrahub fork, the anchor
           frame t_a and stride s such that (t_a, t_a + s) reproduces the competition's
           (mother frame, mother frame + 1) configuration, plus full provenance
           (original t0, shift, sep at both ends, the rule that moved it).

Usage:
  .venv\\Scripts\\python.exe scripts\\h1m2_realign.py --mode census --out census.json
  .venv\\Scripts\\python.exe scripts\\h1m2_realign.py --mode build  --census census.json \
      --embryo ZSNS003 --out realigned_ZSNS003.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h1m_features as F  # noqa: E402

ROOT = Path(r"c:\Users\aryaa\Documents\Biohub-CellTracking-2026")
KMAX = 10                      # frames after the annotated fork to follow
PREP = Path(r"c:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH"
            r"\agent_runs\agent4\zebrahub_prep")


# --------------------------------------------------------------- competition side
def _geff(crop: str) -> tuple[pd.DataFrame, np.ndarray]:
    import zarr
    g = zarr.open(str(ROOT / "data" / "train" / f"{crop}.geff"), mode="r")
    ids = np.asarray(g["nodes/ids"][:])
    nodes = pd.DataFrame({
        "id": ids,
        "t": np.asarray(g["nodes/props/t/values"][:], dtype=np.int64),
        "z": np.asarray(g["nodes/props/z/values"][:], dtype=np.float64),
        "y": np.asarray(g["nodes/props/y/values"][:], dtype=np.float64),
        "x": np.asarray(g["nodes/props/x/values"][:], dtype=np.float64),
    })
    edges = np.asarray(g["edges/ids"][:])
    return nodes, edges


def _follow(succ: dict, node: int, kmax: int) -> list[int]:
    """Unique-successor chain from `node`, stopping at a fork or a track end."""
    out, cur = [node], node
    for _ in range(kmax):
        nxt = succ.get(cur)
        if nxt is None or len(nxt) != 1:
            break
        cur = nxt[0]
        out.append(cur)
    return out


def competition_census(crops: list[str], kmax: int = KMAX) -> dict:
    vox = F.VOX_COMPETITION
    sep = {k: [] for k in range(0, kmax + 1)}
    drift = {k: [] for k in range(0, kmax + 1)}
    clock: list[float] = []
    n_fork = 0
    for crop in crops:
        try:
            nodes, edges = _geff(crop)
        except Exception:                                   # noqa: BLE001
            continue
        if len(edges) == 0:
            continue
        P = nodes[["z", "y", "x"]].to_numpy(float) * vox[None, :]
        pos = {int(i): P[j] for j, i in enumerate(nodes.id.to_numpy())}
        succ: dict[int, list[int]] = {}
        for u, v in edges:
            succ.setdefault(int(u), []).append(int(v))
        # clock: one-step displacement of every out-degree-1 node
        for u, vs in succ.items():
            if len(vs) == 1:
                clock.append(float(np.linalg.norm(pos[vs[0]] - pos[u])))
        for u, vs in succ.items():
            if len(vs) != 2:
                continue
            n_fork += 1
            c1 = _follow(succ, int(vs[0]), kmax)
            c2 = _follow(succ, int(vs[1]), kmax)
            m = pos[int(u)]
            for k in range(1, kmax + 1):
                if k > len(c1) or k > len(c2):
                    break
                a, b = pos[c1[k - 1]], pos[c2[k - 1]]
                sep[k].append(float(np.linalg.norm(a - b)))
                drift[k].append(float(np.linalg.norm(0.5 * (a + b) - m)))
    return {"n_fork": n_fork,
            "sep": {k: _q(v) for k, v in sep.items() if v},
            "drift": {k: _q(v) for k, v in drift.items() if v},
            "clock": _q(clock)}


def _q(v) -> dict:
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    return {"n": int(a.size), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
            "mean": float(a.mean())}


# ------------------------------------------------------------------ zebrahub side
def zebrahub_forks(emb: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (tracks, forks). `forks` has one row per (parent, d1, d2)."""
    tr = pd.read_parquet(PREP / f"{emb}.parquet")
    tr = tr.sort_values(["track_id", "t"], kind="stable").reset_index(drop=True)
    ch = tr[tr.parent_track_id >= 0].groupby("track_id").agg(
        parent=("parent_track_id", "first"), tstart=("t", "min"))
    ch = ch.reset_index()
    grp = ch.groupby("parent").track_id.apply(list)
    pend = tr.groupby("track_id").t.max()
    rows = []
    for parent, kids in grp.items():
        if len(kids) != 2:
            continue
        t0 = int(pend.get(parent, -1))
        if t0 < 0:
            continue
        rows.append((int(parent), int(kids[0]), int(kids[1]), t0))
    forks = pd.DataFrame(rows, columns=["parent", "d1", "d2", "t0"])
    return tr, forks


def zebrahub_census(emb: str, kmax: int = KMAX, tr=None, forks=None) -> dict:
    vox = F.VOX_ZEBRAHUB_L0
    if tr is None:
        tr, forks = zebrahub_forks(emb)
    # position lookup keyed by (track_id, t)
    key = tr.track_id.to_numpy(np.int64) * 100000 + tr.t.to_numpy(np.int64)
    P = tr[["z", "y", "x"]].to_numpy(float) * vox[None, :]
    order = np.argsort(key, kind="stable")
    key_s, P_s = key[order], P[order]

    def look(tid: np.ndarray, t: np.ndarray):
        k = tid.astype(np.int64) * 100000 + t.astype(np.int64)
        j = np.searchsorted(key_s, k)
        ok = (j < key_s.size)
        j = np.clip(j, 0, key_s.size - 1)
        ok &= (key_s[j] == k)
        out = np.full((tid.size, 3), np.nan)
        out[ok] = P_s[j[ok]]
        return out

    # forks whose daughters really start at t0+1 (the tracker's convention)
    t0 = forks.t0.to_numpy()
    mpos = look(forks.parent.to_numpy(), t0)
    sep, drift = {}, {}
    for k in range(1, kmax + 1):
        a = look(forks.d1.to_numpy(), t0 + k)
        b = look(forks.d2.to_numpy(), t0 + k)
        s = np.linalg.norm(a - b, axis=1)
        d = np.linalg.norm(0.5 * (a + b) - mpos, axis=1)
        sep[k] = _q(s[np.isfinite(s)])
        drift[k] = _q(d[np.isfinite(d)])
    # clock: one-step displacement inside a track
    d = tr.groupby("track_id")[["z", "y", "x"]].diff().to_numpy(float) * vox[None, :]
    dt = tr.groupby("track_id").t.diff().to_numpy(float)
    step = np.linalg.norm(d, axis=1)
    step = step[np.isfinite(step) & (dt == 1)]
    return {"n_fork": int(len(forks)), "sep": sep, "drift": drift,
            "clock": _q(step), "n_tracks": int(tr.track_id.nunique()),
            "frames": [int(tr.t.min()), int(tr.t.max())]}


# ------------------------------------------------------------------ realign build
def realign(emb: str, sep_target: float, kmax: int = KMAX,
            tr=None, forks=None) -> pd.DataFrame:
    """One row per Zebrahub fork with the REALIGNED anchor.

    Rule (track geometry only, deterministic, no imagery, no fitting):

      k*   = smallest k in 1..kmax with sep(t0+k) >= sep_target, where sep_target is the
             competition's own median daughter separation one frame after an annotated
             division.  This is the first frame at which the pair occupies the
             competition's post-division configuration.
      s    = k*                      (the competition-scale transition length)
      t_a  = t0                      (the last frame at which the annotation says the
                                      mother is a single object)
      shift = s - 1                  (frames by which the effective forward step moved)

    So the realigned event is: mother at t_a, evaluated over t_a -> t_a + s, with the
    5-frame block resampled at stride s.  When k* does not exist within kmax the fork is
    kept but flagged `realigned=False` -- these are the forks whose daughters never reach
    a competition-like separation and are the prime suspects for tracker fragmentation.
    """
    vox = F.VOX_ZEBRAHUB_L0
    if tr is None:
        tr, forks = zebrahub_forks(emb)
    key = tr.track_id.to_numpy(np.int64) * 100000 + tr.t.to_numpy(np.int64)
    P = tr[["z", "y", "x"]].to_numpy(float)
    order = np.argsort(key, kind="stable")
    key_s, P_s = key[order], P[order]

    def look(tid, t):
        k = np.asarray(tid, np.int64) * 100000 + np.asarray(t, np.int64)
        j = np.searchsorted(key_s, k)
        ok = j < key_s.size
        j = np.clip(j, 0, key_s.size - 1)
        ok &= key_s[j] == k
        out = np.full((len(k), 3), np.nan)
        out[ok] = P_s[j[ok]]
        return out

    t0 = forks.t0.to_numpy()
    mvox = look(forks.parent.to_numpy(), t0)
    seps = np.full((len(forks), kmax + 1), np.nan)
    mids = np.full((len(forks), kmax + 1, 3), np.nan)
    for k in range(1, kmax + 1):
        a = look(forks.d1.to_numpy(), t0 + k)
        b = look(forks.d2.to_numpy(), t0 + k)
        seps[:, k] = np.linalg.norm((a - b) * vox[None, :], axis=1)
        mids[:, k] = 0.5 * (a + b)

    reach = seps[:, 1:] >= sep_target
    has = reach.any(axis=1)
    kstar = np.where(has, reach.argmax(axis=1) + 1, -1)

    out = pd.DataFrame({
        "embryo": emb,
        "mother_track_id": forks.parent.to_numpy(),
        "d1": forks.d1.to_numpy(), "d2": forks.d2.to_numpy(),
        "t_orig": t0,
        "z_vox": mvox[:, 0], "y_vox": mvox[:, 1], "x_vox": mvox[:, 2],
        "stride": kstar,
        "realigned": has,
        "sep_p1_um": seps[:, 1],
        "sep_target_um": sep_target,
    })
    out["t_anchor"] = out.t_orig                      # anchor stays; the STEP is stretched
    out["shift_frames"] = np.where(has, kstar - 1, 0)
    out["sep_at_stride_um"] = [seps[i, k] if k > 0 else np.nan
                               for i, k in enumerate(kstar)]
    # drift of the daughter midpoint away from the annotated mother, at the stride
    dm = np.full(len(out), np.nan)
    for i, k in enumerate(kstar):
        if k > 0 and np.isfinite(mids[i, k]).all() and np.isfinite(mvox[i]).all():
            dm[i] = float(np.linalg.norm((mids[i, k] - mvox[i]) * vox))
    out["drift_at_stride_um"] = dm
    out["max_sep_um"] = np.nanmax(seps[:, 1:], axis=1)
    return out


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["census", "build"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--census", default=None)
    ap.add_argument("--embryos", default="ZSNS003,ZSNS004,ZSNS005")
    ap.add_argument("--crops-file", default=None)
    ap.add_argument("--kmax", type=int, default=KMAX)
    ap.add_argument("--sep-target", type=float, default=None)
    a = ap.parse_args()
    t_start = time.time()

    if a.mode == "census":
        rep = {"kmax": a.kmax}
        if a.crops_file:
            crops = [ln.strip() for ln in Path(a.crops_file).read_text().split()
                     if ln.strip()]
            rep["competition"] = competition_census(crops, a.kmax)
            rep["competition"]["n_crops"] = len(crops)
        for emb in a.embryos.split(","):
            if not emb:
                continue
            rep[emb] = zebrahub_census(emb, a.kmax)
            print(json.dumps({emb: rep[emb]}), flush=True)
        rep["seconds"] = round(time.time() - t_start, 1)
        Path(a.out).write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2)[:4000])
        return

    cen = json.loads(Path(a.census).read_text()) if a.census else {}
    tgt = a.sep_target
    if tgt is None:
        tgt = cen["competition"]["sep"]["1"]["median"]
    frames = []
    for emb in a.embryos.split(","):
        if not emb:
            continue
        df = realign(emb, float(tgt), a.kmax)
        frames.append(df)
        print(json.dumps({
            "embryo": emb, "forks": int(len(df)),
            "realigned": int(df.realigned.sum()),
            "stride_hist": {int(k): int(v) for k, v in
                            df.loc[df.realigned, "stride"].value_counts().items()},
            "median_stride": float(df.loc[df.realigned, "stride"].median()),
            "sep_p1_median": float(np.nanmedian(df.sep_p1_um)),
            "sep_at_stride_median": float(np.nanmedian(df.sep_at_stride_um)),
        }), flush=True)
    allf = pd.concat(frames, ignore_index=True)
    allf["sep_target_um"] = float(tgt)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    allf.to_parquet(a.out, compression="zstd", index=False)
    print(json.dumps({"rows": int(len(allf)), "out": a.out,
                      "seconds": round(time.time() - t_start, 1)}))


if __name__ == "__main__":
    main()
