"""Price the deployed 0.947 notebook's POST-PROCESSING env vars against ground truth.

The provenance audit found 33 of 36 non-default knobs were inherited unexamined
down a chain of forks. This script rebuilds the deployed post-processing chain on
the kernel's own ILP prediction graphs for the 8 TRAIN films (ground truth
exists), then moves ONE knob at a time off its deployed value and reports what
the competition proxy actually does.

Chain order, transcribed from filter_output_graph() in cell 2 of the notebook:

    edge prefilter (enforce_next_frame, edge_max_um)
      -> motion relink            (replaces ALL edges)
      -> single-parent repair -> single-child repair
      -> close_single_frame_gaps  (gap close, t -> t+2)
      -> recover_strict_gap2      (t -> t+3, 2 synthetic nodes)
      -> add_safe_divisions_postlink
      -> division geometry filter
      -> prune isolated
      -> filter_short_track_components (+ adaptive rescue)
      -> linefit_smooth_output_graph

Stage code is REUSED, not rewritten:
  scripts/24_keep_ilp_edges.py -> load_pred, load_gt, safe_div
  scripts/91_other_stages.py   -> gap_close, gap2, short_track, linefit,
                                  prune_isolated, _remap, _components, make_refine
  scripts/25_relink_control.py -> motion_relink
Only the knobs those ports do not expose are added here (gap-close density
adaptation, safe-div mutual-NN / divergence / tau switches, short-track rescue,
the division geometry filter, the edge prefilter and the single-parent/child
repairs). gap_close2() is verified byte-identical to 91's gap_close() when the
density adaptation is off -- see --selftest.

Metric: src/biohub/metric2.py (SCALE = 1.625, 0.40625, 0.40625).
NOT src/biohub/metric.py and NOT src/biohub/postprocess.py: both assume an
isotropic downsampled grid and are 4x wrong in y/x on these coordinates.

Reading the output:
  * the multiplier is 1 - 0.1*(n_pred - n_est)/n_est and is UNCAPPED above 1, so
    deleting nodes pays without bound. Any row whose proxy rises while J is flat
    or falling is flagged <<MULT-ONLY: that is a metric exploit, not tracking.
  * there are only 12 ground-truth divisions across all 8 films. One division
    event is worth ~0.0083 of proxy. divJ differences below that are noise.

Outputs: artifacts/envvar_pricing.csv, artifacts/envvar_pricing_inventory.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import multiprocessing as mp
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub import io, metric2 as M2  # noqa: E402

SCALE = M2.SCALE
PRED = ROOT / "artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0"
OUT_CSV = ROOT / "artifacts/envvar_pricing.csv"
OUT_INV = ROOT / "artifacts/envvar_pricing_inventory.csv"

# ---------------------------------------------------------------- reused ports
_h91 = (ROOT / "scripts/91_other_stages.py").read_text().split(
    "# ---------------------------------------------------------------- harness")[0]
exec(compile(_h91, "91_other_stages.py(head)", "exec"), globals())
# -> load_pred, load_gt, safe_div, G_of, _remap, _components,
#    prune_isolated, short_track, gap_close, gap2, linefit

_h25 = (ROOT / "scripts/25_relink_control.py").read_text().split(
    "PRED = Path")[0].split("SCALE = M2.SCALE")[1]
exec(compile(_h25, "25_relink_control.py(motion_relink)", "exec"), globals())
# -> motion_relink


# ---------------------------------------------------------------- frame access
class FrameLRU:
    """Bounded frame cache. gap_close walks t upward, gap2 jumps around."""

    def __init__(self, stem, maxsize=8):
        self.path = io.dataset_root() / "train" / f"{stem}.zarr"
        self.maxsize = maxsize
        self.d = OrderedDict()

    def __contains__(self, t):
        return t in self.d

    def __getitem__(self, t):
        if t in self.d:
            self.d.move_to_end(t)
            return self.d[t]
        f = io.read_frame(self.path, t)
        self.d[t] = f
        while len(self.d) > self.maxsize:
            self.d.popitem(last=False)
        return f

    def get(self, t, default=None):
        try:
            return self[t]
        except Exception:
            return default


def make_refine2(stem, win_z=1, win_yx=3, max_shift=3.2):
    """refine_synthetic_midpoint() from the notebook, bounded cache."""
    cache = FrameLRU(stem)

    def refine(t, vox):
        try:
            f = cache[int(t)]
            z, y, x = [int(round(v)) for v in vox]
            z0, z1 = max(0, z - win_z), min(f.shape[0], z + win_z + 1)
            y0, y1 = max(0, y - win_yx), min(f.shape[1], y + win_yx + 1)
            x0, x1 = max(0, x - win_yx), min(f.shape[2], x + win_yx + 1)
            patch = f[z0:z1, y0:y1, x0:x1].astype(np.float64)
            if patch.size == 0:
                return vox
            wgt = np.maximum(patch - float(np.percentile(patch, 20.0)), 0.0)
            tot = float(wgt.sum())
            if tot <= 0:
                return vox
            g = np.array([float((wgt * a).sum() / tot)
                          for a in np.ogrid[z0:z1, y0:y1, x0:x1]])
            if np.linalg.norm((g - np.asarray(vox, float)) * SCALE) > max_shift:
                return vox
            return g
        except Exception:
            return vox

    return refine


# ---------------------------------------------------------------- new stages
def edge_prefilter(t, pos, edges, probs, edge_max_um, enforce_next_frame):
    oe, op = [], []
    for (s, d), p in zip(edges, probs):
        if enforce_next_frame and int(t[d]) != int(t[s]) + 1:
            continue
        if edge_max_um > 0 and float(np.linalg.norm(pos[d] - pos[s])) > edge_max_um:
            continue
        oe.append((s, d))
        op.append(float(p))
    return oe, op


def _sort_key(pos, s, d, p):
    """edge_sort_key() from the notebook: (edge_prob, -distance_um)."""
    return (p, -float(np.linalg.norm(pos[d] - pos[s])))


def unique_repair(pos, edges, probs, on_target):
    """Keep the single best edge per target (single-parent) or per source."""
    best = {}
    for i, (s, d) in enumerate(edges):
        k = d if on_target else s
        key = _sort_key(pos, s, d, probs[i])
        if k not in best or key > best[k][0]:
            best[k] = (key, i)
    idx = sorted(v[1] for v in best.values())
    return [edges[i] for i in idx], [probs[i] for i in idx]


def relink_stage(t, zyx, edges, probs, tight, relaxed, vel, bonus, max_frame_nodes):
    """motion_relink() from script 25, plus the MAX_FRAME_NODES tripwire."""
    counts = np.bincount(np.asarray(t, np.int64))
    if counts.size and counts.max() > max_frame_nodes:
        return None                                    # notebook falls back to raw
    P = dict(t=t, zyx=zyx, edges=edges, prob=probs)
    e = motion_relink(P, tight=tight, relaxed=relaxed, vel=vel, beta=bonus)
    if not e:
        return None
    look = {}
    for (a, b), p in zip(edges, probs):
        look[(int(a), int(b))] = float(p)
    return e, [look.get((int(a), int(b)), 0.0) for a, b in e]


def gap_close2(G, max_um=10.0, reuse_um=3.2, allow_synth=True, synth_frac=0.05,
               synth_abs=2000, refine=None, density=False, dens_ref=6.5,
               dens_gain=0.040, dens_max_delta=0.125, dens_neighbors=3, gap=1):
    """91's gap_close() with the GAP_DENSITY_ADAPTIVE threshold added.

    `max_um` is GAP_CLOSE_UM * (gap + 1) -- the notebook scales the gate by the
    span, so the deployed GAP_CLOSE_UM = 5.0 is a 10.0 um gate at gap = 1.
    """
    t, zyx, edges = G["t"], G["zyx"], list(G["edges"])
    pos = zyx * SCALE
    outgoing = {int(s) for s, _ in edges}
    incoming = {int(d) for _, d in edges}
    incident = outgoing | incoming
    ends, starts, iso, allt = {}, {}, {}, {}
    for i, tt in enumerate(t):
        tt = int(tt)
        allt.setdefault(tt, []).append(i)
        if i not in outgoing:
            ends.setdefault(tt, []).append(i)
        if i not in incoming:
            starts.setdefault(tt, []).append(i)
        if i not in incident:
            iso.setdefault(tt, []).append(i)

    n_base = len(t)
    cap = min(synth_abs, max(1, round(n_base * synth_frac))) if (allow_synth and synth_frac > 0) else 0
    new_nodes, added = [], []
    used_start, used_iso = set(), set()
    n_reuse = n_synth = n_outside = 0

    spacing_cache = {}

    def spacing(tt):
        if tt in spacing_cache:
            return spacing_cache[tt]
        ids = allt.get(tt, [])
        if len(ids) <= 1:
            r = {i: dens_ref for i in ids}
            spacing_cache[tt] = r
            return r
        pts = pos[ids]
        k = min(len(ids), max(2, dens_neighbors + 1))
        dd, _ = cKDTree(pts).query(pts, k=k)
        if dd.ndim == 1:
            dd = dd[:, None]
        r = {}
        for j, i in enumerate(ids):
            nb = dd[j, 1:]
            nb = nb[np.isfinite(nb)]
            r[i] = float(np.median(nb)) if nb.size else dens_ref
        spacing_cache[tt] = r
        return r

    for tt in sorted(ends):
        E = [i for i in ends[tt] if i not in outgoing]
        S = [j for j in starts.get(tt + gap + 1, []) if j not in incoming and j not in used_start]
        if not E or not S:
            continue
        D = np.linalg.norm(pos[S][None] - pos[E][:, None], axis=-1)
        thr = np.full_like(D, max_um)
        if density:
            ss, ts = spacing(tt), spacing(tt + gap + 1)
            sv = np.array([ss.get(i, dens_ref) for i in E])[:, None]
            tv = np.array([ts.get(j, dens_ref) for j in S])[None, :]
            delta = np.clip(dens_gain * (0.5 * (sv + tv) - dens_ref), -dens_max_delta, dens_max_delta)
            thr = max_um + delta * (gap + 1)
        base_ok = D <= max_um
        ok = D <= thr
        if not ok.any():
            continue
        cost = np.where(ok, D, float(np.max(thr)) * 1000.0 + 1.0)
        for r, c in zip(*linear_sum_assignment(cost)):
            if not ok[r, c]:
                continue
            if not base_ok[r, c]:
                n_outside += 1
            e, s = E[r], S[c]
            if e in outgoing or s in incoming or s in used_start:
                continue
            mid_t = tt + gap
            mid_vox = (zyx[e] + zyx[s]) / 2.0
            mid_um = mid_vox * SCALE
            pick = None
            free = [k for k in iso.get(mid_t, []) if k not in used_iso and k not in incident]
            if reuse_um > 0 and free:
                dd = np.linalg.norm(pos[free] - mid_um, axis=1)
                if dd.min() <= reuse_um:
                    pick = free[int(np.argmin(dd))]
                    used_iso.add(pick)
                    n_reuse += 1
            if pick is None:
                if len(new_nodes) >= cap:
                    continue
                p = refine(mid_t, mid_vox) if refine is not None else mid_vox
                pick = n_base + len(new_nodes)
                new_nodes.append((mid_t, p))
                n_synth += 1
            added += [(e, pick), (pick, s)]
            outgoing.add(e); outgoing.add(pick)
            incoming.add(pick); incoming.add(s)
            incident.add(e); incident.add(pick); incident.add(s)
            used_start.add(s)

    if new_nodes:
        t = np.concatenate([t, np.array([a for a, _ in new_nodes], t.dtype)])
        zyx = np.concatenate([zyx, np.array([b for _, b in new_nodes], zyx.dtype)])
    return (dict(t=t, zyx=zyx, edges=edges + added),
            {"gap_pairs": len(added) // 2, "gap_reuse": n_reuse,
             "gap_synth": n_synth, "gap_dens_outside": n_outside})


def safe_div2(t, zyx, edges, parent_max=9.0, sister_max=14.0, child_max=10.0,
              tau=0.6, diverge=2.25, require_div=True, mutual_nn=True,
              frame_cap=0.0076, glob_cap=0.00375):
    """24's safe_div() with the three deployed switches the port hard-codes.

    tau <= 0 disables the symmetry gate (the notebook guards it with
    `if SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0`), require_div toggles
    SAFE_DIV_REQUIRE_DIVERGENCE, mutual_nn toggles SAFE_DIV_REQUIRE_MUTUAL_NN
    (off = every orphan inside the gates is a candidate, not just C's NN).
    """
    pos = zyx * SCALE
    succ, indeg = {}, {}
    for s, d in edges:
        succ.setdefault(int(s), []).append(int(d))
        indeg[int(d)] = indeg.get(int(d), 0) + 1
    by_t = {}
    for i, tt in enumerate(t):
        by_t.setdefault(int(tt), []).append(i)
    dist = lambda a, b: float(np.linalg.norm(pos[a] - pos[b]))
    added = []
    for tt in sorted(by_t):
        nxt = by_t.get(tt + 1)
        if not nxt:
            continue
        orph = [j for j in nxt if indeg.get(j, 0) == 0]
        if not orph:
            continue
        opos = pos[orph]
        cands = []
        for Pn in [i for i in by_t[tt] if len(succ.get(i, ())) == 1]:
            C = succ[Pn][0]
            if int(t[C]) != tt + 1:
                continue
            dpc = dist(Pn, C)
            if dpc > child_max:
                continue
            if mutual_nn:
                pool = [orph[int(np.argmin(np.linalg.norm(opos - pos[C], axis=1)))]]
            else:
                pool = orph
            for Q in pool:
                if Q == C:
                    continue
                dpq, dcq = dist(Pn, Q), dist(C, Q)
                if dpq > parent_max or dcq > sister_max:
                    continue
                if require_div:
                    sc, sq = succ.get(C, []), succ.get(Q, [])
                    if len(sc) != 1 or len(sq) != 1:
                        continue
                    if int(t[sc[0]]) != tt + 2 or int(t[sq[0]]) != tt + 2:
                        continue
                    if dist(sc[0], sq[0]) - dcq < diverge:
                        continue
                if tau > 0.0 and abs(dpc - dpq) / max((dpc + dpq) / 2.0, 1e-6) > tau:
                    continue
                cands.append((dpq + 0.15 * dcq, Pn, Q))
        cap = max(1, round(frame_cap * len(by_t[tt])))
        n = 0
        for _, Pn, Q in sorted(cands):
            if n >= cap:
                break
            if indeg.get(Q, 0) or len(succ.get(Pn, ())) >= 2:
                continue
            added.append((Pn, Q))
            succ.setdefault(Pn, []).append(Q)
            indeg[Q] = 1
            n += 1
    cap = max(1, round(glob_cap * max(1, len(edges))))
    added = added[:cap]
    return list(edges) + added, len(added)


def div_geom_filter(t, zyx, edges, probs, parent_max=10.5, sister_max=8.0,
                    drop_single=True):
    """OUTPUT_DIVISION_GEOMETRY_FILTER (deployed OFF)."""
    pos = zyx * SCALE
    by_src = {}
    for i, (s, d) in enumerate(edges):
        by_src.setdefault(int(s), []).append(i)
    keep, dropped = [], 0
    for s, idxs in by_src.items():
        if len(idxs) <= 1:
            keep += idxs
            continue
        ranked = sorted(idxs, key=lambda i: _sort_key(pos, edges[i][0], edges[i][1], probs[i]),
                        reverse=True)
        i1, i2 = ranked[0], ranked[1]
        a, b = edges[i1][1], edges[i2][1]
        d1 = float(np.linalg.norm(pos[a] - pos[s]))
        d2 = float(np.linalg.norm(pos[b] - pos[s]))
        sis = float(np.linalg.norm(pos[a] - pos[b]))
        ok = (max(d1, d2) <= parent_max and sis <= sister_max
              and int(t[a]) == int(t[s]) + 1 and int(t[b]) == int(t[s]) + 1)
        if ok:
            keep += [i1, i2]
            dropped += max(0, len(ranked) - 2)
        elif drop_single:
            keep += [i1]
            dropped += len(ranked) - 1
        else:
            keep += ranked
    keep.sort()
    return [edges[i] for i in keep], [probs[i] for i in keep], {"divgeo_dropped": dropped}


def remap_keep(t, zyx, edges, probs, keep):
    idx = np.where(keep)[0]
    rm = -np.ones(len(t), np.int64)
    rm[idx] = np.arange(len(idx))
    oe, op = [], []
    for (s, d), p in zip(edges, probs):
        if keep[s] and keep[d]:
            oe.append((int(rm[s]), int(rm[d])))
            op.append(p)
    return t[idx], zyx[idx], oe, op


def prune_isolated2(t, zyx, edges, probs):
    inc = np.zeros(len(t), bool)
    for s, d in edges:
        inc[s] = inc[d] = True
    n = int((~inc).sum())
    if n == 0:
        return t, zyx, edges, probs, {"pruned": 0}
    t, zyx, edges, probs = remap_keep(t, zyx, edges, probs, inc)
    return t, zyx, edges, probs, {"pruned": n}


def short_track2(t, zyx, edges, probs, min_len=6, keep_forks=True, rescue=False,
                 rescue_trigger=0.10, rescue_min_len=4, rescue_min_prob=0.88,
                 rescue_max_dist=3.0, rescue_frac=0.012, rescue_abs=120):
    """filter_short_track_components() including ADAPTIVE_SHORT_TRACK_RESCUE."""
    n = len(t)
    if min_len <= 1 or not edges:
        return t, zyx, edges, probs, {}
    comp = _components(n, edges)
    sizes = np.bincount(comp, minlength=n)
    outdeg = {}
    for s, _ in edges:
        outdeg[int(s)] = outdeg.get(int(s), 0) + 1
    fork = {comp[s] for s, k in outdeg.items() if k >= 2} if keep_forks else set()
    keep = sizes[comp] >= min_len
    if fork:
        keep |= np.isin(comp, list(fork))
    if not keep.any():
        return t, zyx, edges, probs, {"st_skipped_all": 1}
    removed = int((~keep).sum())
    if removed <= 0:
        return t, zyx, edges, probs, {}

    st = {"st_removed_pre_rescue": removed, "st_removed_frac_max": 0}
    st["st_removed_frac_max"] = removed / max(n, 1)
    n_resc_nodes = n_resc_comp = 0
    if rescue and removed / max(n, 1) >= rescue_trigger:
        budget = min(rescue_abs, max(0, round(n * rescue_frac)))
        pos = zyx * SCALE
        ce, cd = {}, {}
        for (s, d), p in zip(edges, probs):
            r = int(comp[s])
            ce.setdefault(r, []).append(float(p))
            cd.setdefault(r, []).append(float(np.linalg.norm(pos[d] - pos[s])))
        props = []
        for r in set(int(x) for x in comp[~keep]):
            sz = int(sizes[r])
            if sz < rescue_min_len or sz >= min_len:
                continue
            if r not in ce:
                continue
            mp = float(np.mean(ce[r]))
            md = float(np.mean(cd[r])) if cd.get(r) else float("inf")
            if mp < rescue_min_prob or md > rescue_max_dist:
                continue
            props.append((mp - 0.02 * md + 0.004 * sz, sz, r))
        props.sort(reverse=True)
        rescued = []
        for _, sz, r in props:
            if budget <= 0 or n_resc_nodes + sz > budget:
                continue
            rescued.append(r)
            n_resc_nodes += sz
            n_resc_comp += 1
        if rescued:
            keep |= np.isin(comp, rescued)
        st["st_rescue_triggered"] = 1
        st["st_rescue_budget"] = budget
    st["st_rescued_nodes"] = n_resc_nodes
    st["st_rescued_comps"] = n_resc_comp
    removed = int((~keep).sum())
    st["st_removed"] = removed
    if removed <= 0:
        return t, zyx, edges, probs, st
    t, zyx, edges, probs = remap_keep(t, zyx, edges, probs, keep)
    return t, zyx, edges, probs, st


# ---------------------------------------------------------------- the chain
DEPLOYED = dict(
    # edge prefilter
    enforce_next_frame=True, edge_max_um=14.0,
    single_parent_repair=True, single_child_repair=False,
    # motion relink
    motion_relink=True, relink_tight=6.0, relink_relaxed=10.0,
    relink_vel=0.5, relink_bonus=1.0, relink_max_frame_nodes=2600,
    # gap close
    gap_close=True, gap_close_max_gap=2, gap_close_um=5.0,
    gap_density_adaptive=True, gap_density_ref=6.5, gap_density_gain=0.040,
    gap_density_max_delta=0.125, gap_density_neighbors=3,
    gap_reuse_existing=True, gap_reuse_um=3.2,
    gap_added_frac=0.05, gap_added_abs=2000,
    gap_refine=True, gap_refine_win_z=1, gap_refine_win_yx=3, gap_refine_max_shift=3.2,
    # gap2
    gap2=True, gap2_total=10.2, gap2_step=4.4, gap2_frac=0.0045, gap2_abs=180,
    gap2_context=True, gap2_frame_frac=0.006,
    # safe divisions
    safe_div=True, sd_parent=9.0, sd_sister=14.0, sd_tau=0.6, sd_child=10.0,
    sd_diverge=2.25, sd_require_div=True, sd_mutual_nn=True,
    sd_frame_cap=0.0076, sd_glob_cap=0.00375,
    # division geometry filter (deployed OFF)
    div_geom=False, div_parent=10.5, div_sister=8.0, div_drop_single=True,
    # tail
    prune_isolated=True,
    short_tracks=True, min_track_len=6, keep_div_comps=True,
    st_rescue=True, st_rescue_trigger=0.10, st_rescue_min_len=4,
    st_rescue_min_prob=0.88, st_rescue_max_dist=3.0,
    st_rescue_frac=0.012, st_rescue_abs=120,
    linefit=True, linefit_w=0.8, linefit_window=2,
)


def run_chain(P, c, stem):
    t = P["t"].copy()
    zyx = P["zyx"].copy()
    pos = zyx * SCALE
    stats = {}

    edges, probs = edge_prefilter(t, pos, P["edges"], P["prob"],
                                  c["edge_max_um"], c["enforce_next_frame"])
    stats["prefilter_dropped"] = len(P["edges"]) - len(edges)

    if c["motion_relink"]:
        r = relink_stage(t, zyx, edges, probs, c["relink_tight"], c["relink_relaxed"],
                         c["relink_vel"], c["relink_bonus"], c["relink_max_frame_nodes"])
        if r is None:
            stats["relink_fallback_raw"] = 1
        else:
            edges, probs = r
            stats["relink_edges"] = len(edges)

    if c["single_parent_repair"] and edges:
        k = len(edges)
        edges, probs = unique_repair(pos, edges, probs, on_target=True)
        stats["sp_dropped"] = k - len(edges)
    if c["single_child_repair"] and edges:
        k = len(edges)
        edges, probs = unique_repair(pos, edges, probs, on_target=False)
        stats["sc_dropped"] = k - len(edges)

    refine = None
    if c["gap_refine"]:
        refine = make_refine2(stem, c["gap_refine_win_z"], c["gap_refine_win_yx"],
                              c["gap_refine_max_shift"])

    eff_gap = min(int(c["gap_close_max_gap"]), 1)
    if c["gap_close"] and eff_gap >= 1 and edges:
        G = dict(t=t, zyx=zyx, edges=edges)
        G, st = gap_close2(G, max_um=c["gap_close_um"] * (eff_gap + 1),
                           reuse_um=c["gap_reuse_um"] if c["gap_reuse_existing"] else 0.0,
                           allow_synth=True, synth_frac=c["gap_added_frac"],
                           synth_abs=c["gap_added_abs"], refine=refine,
                           density=c["gap_density_adaptive"], dens_ref=c["gap_density_ref"],
                           dens_gain=c["gap_density_gain"],
                           dens_max_delta=c["gap_density_max_delta"],
                           dens_neighbors=c["gap_density_neighbors"], gap=eff_gap)
        probs = probs + [0.0] * (len(G["edges"]) - len(edges))
        t, zyx, edges = G["t"], G["zyx"], G["edges"]
        stats.update(st)

    if c["gap2"] and edges:
        G = dict(t=t, zyx=zyx, edges=edges)
        G, st = gap2(G, max_total=c["gap2_total"], max_step=c["gap2_step"],
                     require_context=c["gap2_context"], frac_cap=c["gap2_frac"],
                     abs_cap=c["gap2_abs"], frame_frac=c["gap2_frame_frac"],
                     refine=refine)
        probs = probs + [0.0] * (len(G["edges"]) - len(edges))
        t, zyx, edges = G["t"], G["zyx"], G["edges"]
        stats.update(st)

    if c["safe_div"] and edges:
        k = len(edges)
        edges, n = safe_div2(t, zyx, edges, parent_max=c["sd_parent"],
                             sister_max=c["sd_sister"], child_max=c["sd_child"],
                             tau=c["sd_tau"], diverge=c["sd_diverge"],
                             require_div=c["sd_require_div"], mutual_nn=c["sd_mutual_nn"],
                             frame_cap=c["sd_frame_cap"], glob_cap=c["sd_glob_cap"])
        probs = probs + [0.0] * (len(edges) - k)
        stats["sd_added"] = n

    if c["div_geom"] and edges:
        edges, probs, st = div_geom_filter(t, zyx, edges, probs, c["div_parent"],
                                           c["div_sister"], c["div_drop_single"])
        stats.update(st)

    if c["prune_isolated"]:
        t, zyx, edges, probs, st = prune_isolated2(t, zyx, edges, probs)
        stats.update(st)

    if c["short_tracks"] and edges:
        t, zyx, edges, probs, st = short_track2(
            t, zyx, edges, probs, min_len=c["min_track_len"],
            keep_forks=c["keep_div_comps"], rescue=c["st_rescue"],
            rescue_trigger=c["st_rescue_trigger"], rescue_min_len=c["st_rescue_min_len"],
            rescue_min_prob=c["st_rescue_min_prob"], rescue_max_dist=c["st_rescue_max_dist"],
            rescue_frac=c["st_rescue_frac"], rescue_abs=c["st_rescue_abs"])
        stats.update(st)

    if c["linefit"] and c["linefit_w"] > 0 and c["linefit_window"] > 0 and edges:
        G, st = linefit(dict(t=t, zyx=zyx, edges=edges), w=c["linefit_w"],
                        window=c["linefit_window"])
        t, zyx, edges = G["t"], G["zyx"], G["edges"]
        stats.update(st)

    return t, zyx, edges, stats


# ---------------------------------------------------------------- sweep space
def S(env, stage, dep, default, values, **fixed):
    """One knob: (env var, stage, deployed value, notebook default, values)."""
    return dict(env=env, stage=stage, dep=dep, default=default, values=values, fixed=fixed)


SWEEPS = [
    # ---- edge prefilter
    S("BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME", "edge prefilter", 1, 1,
      [("enforce_next_frame", v) for v in (False, True)]),
    S("BIOHUB_OUTPUT_EDGE_MAX_UM", "edge prefilter", 14.0, 14.0,
      [("edge_max_um", v) for v in (0.0, 6.0, 8.0, 10.0, 12.0, 14.0, 20.0, 40.0)]),
    S("BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR", "edge prefilter", 1, 1,
      [("single_parent_repair", v) for v in (False, True)]),
    S("BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR", "edge prefilter", 0, 0,
      [("single_child_repair", v) for v in (False, True)]),
    # ---- motion relink
    S("BIOHUB_OUTPUT_MOTION_RELINK", "motion relink", 1, 1,
      [("motion_relink", v) for v in (False, True)]),
    S("BIOHUB_MOTION_RELINK_TIGHT_UM", "motion relink", 6.0, 6.0,
      [("relink_tight", v) for v in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)]),
    S("BIOHUB_MOTION_RELINK_RELAXED_UM", "motion relink", 10.0, 10.0,
      [("relink_relaxed", v) for v in (6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 18.0)]),
    S("BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT", "motion relink", 0.5, 0.5,
      [("relink_vel", v) for v in (0.0, 0.25, 0.5, 0.75, 1.0)]),
    S("BIOHUB_MOTION_RELINK_LEARNED_BONUS", "motion relink", 1.0, 0.75,
      [("relink_bonus", v) for v in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)]),
    S("BIOHUB_MOTION_RELINK_MAX_FRAME_NODES", "motion relink", 2600, 2600,
      [("relink_max_frame_nodes", v) for v in (80, 300, 520, 2600)]),
    # ---- gap close
    S("BIOHUB_OUTPUT_GAP_CLOSE", "gap close", 1, 1,
      [("gap_close", v) for v in (False, True)]),
    S("BIOHUB_GAP_CLOSE_MAX_GAP", "gap close", 2, 1,
      [("gap_close_max_gap", v) for v in (0, 1, 2, 3)]),
    S("BIOHUB_GAP_CLOSE_UM", "gap close", 5.0, 6.0,
      [("gap_close_um", v) for v in (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)]),
    S("BIOHUB_GAP_DENSITY_ADAPTIVE", "gap close", 1, 0,
      [("gap_density_adaptive", v) for v in (False, True)]),
    S("BIOHUB_GAP_DENSITY_GAIN", "gap close", 0.040, 0.040,
      [("gap_density_gain", v) for v in (0.0, 0.04, 0.2, 1.0)]),
    S("BIOHUB_GAP_DENSITY_REFERENCE_UM", "gap close", 6.5, 6.5,
      [("gap_density_ref", v) for v in (2.0, 4.0, 6.5, 9.0, 14.0)]),
    S("BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM", "gap close", 0.125, 0.125,
      [("gap_density_max_delta", v) for v in (0.0, 0.125, 0.5, 2.0)]),
    S("BIOHUB_GAP_DENSITY_NEIGHBORS", "gap close", 3, 3,
      [("gap_density_neighbors", v) for v in (1, 3, 8)]),
    S("BIOHUB_GAP_CLOSE_REUSE_EXISTING", "gap close", 1, 1,
      [("gap_reuse_existing", v) for v in (False, True)]),
    S("BIOHUB_GAP_CLOSE_REUSE_UM", "gap close", 3.2, 3.2,
      [("gap_reuse_um", v) for v in (0.8, 1.6, 3.2, 6.4, 12.0)]),
    S("BIOHUB_GAP_CLOSE_MAX_ADDED_FRAC", "gap close", 0.05, 0.05,
      [("gap_added_frac", v) for v in (0.0, 0.002, 0.01, 0.025, 0.05, 0.15)]),
    S("BIOHUB_GAP_CLOSE_MAX_ADDED_ABS", "gap close", 2000, 2000,
      [("gap_added_abs", v) for v in (50, 250, 1000, 2000, 8000)]),
    S("BIOHUB_GAP_REFINE_SYNTHETIC", "gap close", 1, 1,
      [("gap_refine", v) for v in (False, True)]),
    S("BIOHUB_GAP_REFINE_WIN_YX", "gap close", 3, 3,
      [("gap_refine_win_yx", v) for v in (1, 3, 6)]),
    S("BIOHUB_GAP_REFINE_WIN_Z", "gap close", 1, 1,
      [("gap_refine_win_z", v) for v in (0, 1, 2)]),
    S("BIOHUB_GAP_REFINE_MAX_SHIFT_UM", "gap close", 3.2, 3.2,
      [("gap_refine_max_shift", v) for v in (0.5, 3.2, 12.0)]),
    # ---- gap2
    S("BIOHUB_OUTPUT_GAP2_RECOVERY", "gap2", 1, 0,
      [("gap2", v) for v in (False, True)]),
    S("BIOHUB_GAP2_MAX_TOTAL_UM", "gap2", 10.2, 10.2,
      [("gap2_total", v) for v in (4.0, 6.0, 8.0, 10.2, 14.0, 20.0)]),
    S("BIOHUB_GAP2_MAX_STEP_UM", "gap2", 4.4, 4.4,
      [("gap2_step", v) for v in (1.5, 2.5, 3.4, 4.4, 6.0, 10.0)]),
    S("BIOHUB_GAP2_MAX_LINKS_FRAC", "gap2", 0.0045, 0.0045,
      [("gap2_frac", v) for v in (0.0, 0.0005, 0.002, 0.0045, 0.02)]),
    S("BIOHUB_GAP2_MAX_LINKS_ABS", "gap2", 180, 180,
      [("gap2_abs", v) for v in (10, 60, 180, 600, 5000)]),
    S("BIOHUB_GAP2_REQUIRE_CONTEXT", "gap2", 1, 1,
      [("gap2_context", v) for v in (False, True)]),
    S("BIOHUB_GAP2_FRAME_FRAC_CAP", "gap2", 0.006, 0.006,
      [("gap2_frame_frac", v) for v in (0.001, 0.006, 0.03, 0.2)]),
    # ---- safe divisions
    S("BIOHUB_OUTPUT_SAFE_DIVISIONS", "safe div", 1, 1,
      [("safe_div", v) for v in (False, True)]),
    S("BIOHUB_SAFE_DIV_MAX_UM", "safe div", 9.0, 4.7,
      [("sd_parent", v) for v in (4.7, 6.0, 7.5, 9.0, 11.0, 13.0, 16.0)]),
    S("BIOHUB_SAFE_DIV_SISTER_MAX_UM", "safe div", 14.0, 7.2,
      [("sd_sister", v) for v in (7.2, 9.0, 11.0, 14.0, 18.0, 24.0)]),
    S("BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU", "safe div", 0.6, 0.0,
      [("sd_tau", v) for v in (0.0, 0.2, 0.4, 0.6, 0.9, 1.35)]),
    S("BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM", "safe div", 10.0, 7.8,
      [("sd_child", v) for v in (5.0, 6.5, 7.8, 10.0, 13.0, 18.0)]),
    S("BIOHUB_SAFE_DIV_DIVERGE_UM", "safe div", 2.25, 2.25,
      [("sd_diverge", v) for v in (0.0, 0.5, 1.25, 2.25, 3.5, 6.0)]),
    S("BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE", "safe div", 1, 1,
      [("sd_require_div", v) for v in (False, True)]),
    S("BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN", "safe div", 1, 1,
      [("sd_mutual_nn", v) for v in (False, True)]),
    S("BIOHUB_SAFE_DIV_FRAME_FRAC_CAP", "safe div", 0.0076, 0.008,
      [("sd_frame_cap", v) for v in (0.001, 0.004, 0.0076, 0.02, 0.10)]),
    S("BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP", "safe div", 0.00375, 0.004,
      [("sd_glob_cap", v) for v in (0.0005, 0.0015, 0.00375, 0.008, 0.02, 0.05)]),
    # ---- division geometry filter
    S("BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER", "div geometry", 0, 0,
      [("div_geom", v) for v in (False, True)]),
    S("BIOHUB_DIV_PARENT_MAX_UM", "div geometry", 10.5, 10.5,
      [("div_parent", v) for v in (7.0, 10.5, 16.0)], div_geom=True),
    S("BIOHUB_DIV_SISTER_MAX_UM", "div geometry", 8.0, 8.0,
      [("div_sister", v) for v in (5.0, 8.0, 14.0, 25.0)], div_geom=True),
    S("BIOHUB_DIV_DROP_TO_SINGLE_IF_BAD", "div geometry", 1, 1,
      [("div_drop_single", v) for v in (False, True)], div_geom=True),
    # ---- prune / short tracks
    S("BIOHUB_OUTPUT_PRUNE_ISOLATED", "prune isolated", 1, 1,
      [("prune_isolated", v) for v in (False, True)]),
    S("BIOHUB_OUTPUT_FILTER_SHORT_TRACKS", "short track", 1, 1,
      [("short_tracks", v) for v in (False, True)]),
    S("BIOHUB_OUTPUT_MIN_TRACK_LEN", "short track", 6, 6,
      [("min_track_len", v) for v in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 40)]),
    S("BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS", "short track", 1, 1,
      [("keep_div_comps", v) for v in (False, True)]),
    S("BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE", "short track rescue", 1, 0,
      [("st_rescue", v) for v in (False, True)]),
    S("BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC", "short track rescue", 0.10, 0.10,
      [("st_rescue_trigger", v) for v in (0.0, 0.02, 0.10, 0.5)]),
    S("BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN", "short track rescue", 4, 4,
      [("st_rescue_min_len", v) for v in (2, 3, 4, 5)], st_rescue_trigger=0.0),
    S("BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB", "short track rescue", 0.88, 0.82,
      [("st_rescue_min_prob", v) for v in (0.0, 0.5, 0.82, 0.88, 0.95)], st_rescue_trigger=0.0),
    S("BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM", "short track rescue", 3.0, 3.25,
      [("st_rescue_max_dist", v) for v in (1.0, 3.0, 3.25, 10.0)], st_rescue_trigger=0.0),
    S("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC", "short track rescue", 0.012, 0.018,
      [("st_rescue_frac", v) for v in (0.0, 0.012, 0.018, 0.10)], st_rescue_trigger=0.0),
    S("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS", "short track rescue", 120, 180,
      [("st_rescue_abs", v) for v in (10, 120, 180, 20000)], st_rescue_trigger=0.0),
    # ---- linefit
    S("BIOHUB_OUTPUT_LINEFIT_SMOOTH", "linefit", 1, 1,
      [("linefit", v) for v in (False, True)]),
    S("BIOHUB_OUTPUT_LINEFIT_WEIGHT", "linefit", 0.8, 0.8,
      [("linefit_w", v) for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]),
    S("BIOHUB_OUTPUT_LINEFIT_WINDOW", "linefit", 2, 2,
      [("linefit_window", v) for v in (1, 2, 3, 4, 5, 6, 8)]),
]

UNTESTABLE = [
    # (env var, deployed, default, why)
    ("BIOHUB_DET_THRESHOLD", "0.965", "0.99", "detection threshold, inference-side"),
    ("BIOHUB_UNET_BATCH_SIZE", "4", "4", "inference throughput only"),
    ("BIOHUB_USE_ILP", "1", "1", "association solver, upstream of these graphs"),
    ("BIOHUB_ILP_EDGE_WEIGHT", "-1.0", "-1.0", "ILP objective, baked into the graphs"),
    ("BIOHUB_ILP_APPEARANCE_WEIGHT", "0.0", "0.1", "ILP objective, baked into the graphs"),
    ("BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "2", "0.1", "ILP objective, baked into the graphs"),
    ("BIOHUB_ILP_DIVISION_WEIGHT", "1.2", "1.0", "ILP objective, baked into the graphs"),
    ("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0.15", "0", "edge-probability fusion, inference-side"),
    ("BIOHUB_BIDIRECTIONAL_FUSION_MODE", "harmonic_probability", "(unset)", "edge-probability fusion, inference-side"),
    ("BIOHUB_EDGE_FEATURE_TTA", "1", "0", "test-time augmentation, inference-side"),
    ("BIOHUB_DEEPCENTER_TTA", "1", "0", "DeepCenter heatmap TTA; needs the DeepCenter net"),
    ("BIOHUB_SECONDARY_EDGE_FEATURE_TTA", "1", "0", "secondary model TTA, inference-side"),
    ("BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT", "0.75", "0.75/1.0", "secondary model blend, inference-side"),
    ("BIOHUB_SECONDARY_EDGE_WEIGHT", "0.15", "0", "dual-seed fusion, inference-side"),
    ("BIOHUB_SECONDARY_DETECTION_WEIGHT", "0.80", "0", "dual-seed fusion, inference-side"),
    ("BIOHUB_SECONDARY_LINK_MODE", "low_margin_consensus", "raw", "dual-seed fusion, inference-side"),
    ("BIOHUB_SECONDARY_MIX_TEMPERATURE", "1", "1", "dual-seed fusion, inference-side"),
    ("BIOHUB_SECONDARY_LOW_MARGIN_MAX", "0.35", "0.2", "dual-seed fusion, inference-side"),
    ("BIOHUB_DUAL_SEED_EDGE_THRESHOLD", "0.48", "cfg.threshold", "candidate generation, inference-side"),
    ("BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION", "0.90", "0.90", "frame retention guard, inference-side"),
    ("BIOHUB_USE_DEEPCENTER_VETO", "1", "1", "post-proc gate, needs the DeepCenter net (priced in script 92)"),
    ("BIOHUB_REQUIRE_DEEPCENTER_VETO", "1", "1", "hard-fail if the net is missing; not a scoring knob"),
    ("BIOHUB_DEEPCENTER_GAP_VETO", "1", "1", "post-proc gate, needs the DeepCenter net"),
    ("BIOHUB_DEEPCENTER_GAP_THRESHOLD", "0.25", "0.10", "post-proc gate, needs the DeepCenter net"),
    ("BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM", "8.5", "0", "post-proc gate, needs the DeepCenter net"),
    ("BIOHUB_DEEPCENTER_SAFE_DIV_VETO", "1", "1", "post-proc gate, priced in script 92"),
    ("BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD", "0.20", "0.12", "post-proc gate, priced in script 92"),
    ("BIOHUB_DEEPCENTER_EXPECTED_EPOCH", "2", "0", "checkpoint integrity assertion"),
    ("BIOHUB_DEEPCENTER_SCORE_WIN_Z", "1", "1", "DeepCenter scoring window"),
    ("BIOHUB_DEEPCENTER_SCORE_WIN_YX", "2", "2", "DeepCenter scoring window"),
    ("BIOHUB_DEEPCENTER_SCORE_CACHE_MAX_FRAMES", "8", "8", "memory only"),
    ("BIOHUB_VALIDATOR_ENABLE", "1", "1", "held-out candidate selection, not a graph knob"),
    ("BIOHUB_VALIDATOR_N_PER_TYPE", "4", "(unset)", "held-out sample size for selection"),
    ("BIOHUB_PPSWEEP_SELECT_MARGIN", "0.001", "(unset)", "selection gate on the validator"),
    ("BIOHUB_PPSWEEP_MAX_ADJ_LOSS", "0.0005", "(unset)", "selection gate on the validator"),
    ("BIOHUB_RUN_OUTPUT_DIAGNOSTICS", "0", "1", "printing only"),
    ("BIOHUB_DIAGNOSTIC_ARM", "harmonic_association_production", "(unset)", "logging label"),
    ("BIOHUB_MODEL_ARTIFACTS / _TARGET_ARTIFACT_SLUG / _ALLOW_ARTIFACT_FALLBACK",
     "(paths)", "(paths)", "artifact discovery"),
    ("BIOHUB_DEEPCENTER_CHECKPOINT / _MANIFEST / _RELATIVE", "(paths)", "(paths)", "artifact discovery"),
    ("BIOHUB_SECONDARY_ARTIFACT_MANIFEST / _WEIGHTS", "(paths)", "(paths)", "artifact discovery"),
    ("BIOHUB_ALLOW_PIP_INSTALL", "0", "0", "dependency install"),
    ("BIOHUB_GPU_SHARD", "(runtime)", "single", "sharding"),
]


# ---------------------------------------------------------------- harness
def load_all():
    stems = sorted(p.stem for p in PRED.glob("*.geff"))
    return {s: (load_pred(PRED / f"{s}.geff"), load_gt(s)) for s in stems}


DATA = {}


def _task(arg):
    key, cfg, stem = arg
    P, GT = DATA[stem]
    t, zyx, edges, stats = run_chain(P, cfg, stem)
    r = M2.score(t, zyx, edges, GT["t"], GT["zyx"], GT["edges"], GT["n_est"])
    return key, stem, r, stats


def aggregate(rows, stats):
    a = M2.aggregate(rows)
    a["ratio"] = sum(r["n_pred"] for r in rows) / sum(r["n_est"] for r in rows)
    a["min_ratio"] = min(r["n_pred"] / r["n_est"] for r in rows)
    acc = {}
    for st in stats:
        for k, v in st.items():
            acc[k] = acc.get(k, 0) + (v if isinstance(v, (int, float)) else 0)
    a["stats"] = acc
    return a


HDR = (f"{'knob value':<44}{'proxy':>9}{'dprox':>9}{'J':>9}{'dJ':>9}"
       f"{'mult':>9}{'n/nest':>8}{'divJ':>7}{'TP/FP/FN':>9}  flags")


def fmt(label, a, base=None, mark=""):
    if base is None:
        dp = dj = ""
        flag = ""
    else:
        dp = f"{a['proxy'] - base['proxy']:+.5f}"
        dj = f"{a['J'] - base['J']:+.5f}"
        flag = ""
        if a["proxy"] > base["proxy"] + 1e-6 and a["J"] <= base["J"] + 1e-6:
            flag = "<<MULT-ONLY"
    return (f"{label:<44}{a['proxy']:>9.5f}{dp:>9}{a['J']:>9.5f}{dj:>9}"
            f"{a['mult']:>9.5f}{a['ratio']:>8.4f}{a['divJ']:>7.4f}"
            f"{f'{a[chr(100)+chr(116)+chr(112)]}/{a[chr(100)+chr(102)+chr(112)]}/{a[chr(100)+chr(102)+chr(110)]}':>9}"
            f"  {mark}{flag}")


def selftest():
    """gap_close2(density=False) must equal 91's gap_close(); safe_div2 must
    equal 24's safe_div() at the deployed gates."""
    stem = sorted(DATA)[0]
    P = DATA[stem][0]
    G = dict(t=P["t"].copy(), zyx=P["zyx"].copy(), edges=list(P["edges"]))
    A, _ = gap_close(G, max_um=8.0, allow_synth=True)
    B, _ = gap_close2(G, max_um=8.0, allow_synth=True, density=False)
    assert A["edges"] == B["edges"] and len(A["t"]) == len(B["t"]), "gap_close2 diverged"
    assert np.allclose(A["zyx"], B["zyx"]), "gap_close2 positions diverged"
    a, _ = safe_div(P)
    b, _ = safe_div2(P["t"], P["zyx"], P["edges"])
    assert a == b, f"safe_div2 diverged: {len(a)} vs {len(b)}"
    print("selftest OK: gap_close2 == 91.gap_close, safe_div2 == 24.safe_div")


def main():
    global DATA
    t0 = time.time()
    DATA = load_all()
    stems = sorted(DATA)
    print(f"{len(stems)} films, "
          f"{sum(len(P['t']) for P, _ in DATA.values())} pred nodes, "
          f"{sum(len(G['edges']) for _, G in DATA.values())} GT edges, "
          f"{sum(1 for _, G in DATA.values() for s in [0])} ... loaded in {time.time()-t0:.1f}s")
    selftest()

    # ---- build the job list ------------------------------------------------
    jobs, meta = [], {}
    def add(key, cfg):
        meta[key] = cfg
        for s in stems:
            jobs.append((key, cfg, s))

    add(("__base__", "DEPLOYED", None), dict(DEPLOYED))

    anchor = dict(DEPLOYED)
    anchor.update(motion_relink=False, gap_close=False, gap2=False, div_geom=False,
                  prune_isolated=False, short_tracks=False, linefit=False,
                  single_parent_repair=False, edge_max_um=0.0,
                  enforce_next_frame=False, gap_refine=False)
    add(("__anchor__", "raw ILP + safe_div only", None), anchor)

    raw = dict(anchor)
    raw["safe_div"] = False
    add(("__raw__", "raw ILP, nothing applied", None), raw)

    for sw in SWEEPS:
        for pkey, val in sw["values"]:
            cfg = dict(DEPLOYED)
            cfg.update(sw["fixed"])
            cfg[pkey] = val
            add((sw["env"], f"{pkey}={val}", val), cfg)

    ncfg = len(meta)
    print(f"{ncfg} configurations x {len(stems)} films = {len(jobs)} film-runs")

    # ---- run ---------------------------------------------------------------
    nproc = min(18, max(1, (os.cpu_count() or 8)))
    ctx = mp.get_context("fork")
    res_rows, res_stats = {}, {}
    t0 = time.time()
    done = 0
    with ctx.Pool(nproc) as pool:
        for key, stem, r, st in pool.imap_unordered(_task, jobs, chunksize=1):
            res_rows.setdefault(key, []).append(r)
            res_stats.setdefault(key, []).append(st)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} film-runs  {time.time()-t0:.0f}s", flush=True)
    print(f"all {len(jobs)} film-runs in {time.time()-t0:.0f}s on {nproc} processes")

    AGG = {k: aggregate(res_rows[k], res_stats[k]) for k in res_rows}
    base = AGG[("__base__", "DEPLOYED", None)]

    # ---- report ------------------------------------------------------------
    print("\n" + "=" * 128)
    print("REFERENCE POINTS")
    print("=" * 128)
    print(HDR)
    print(fmt("raw ILP, nothing applied", AGG[("__raw__", "raw ILP, nothing applied", None)], base))
    print(fmt("raw ILP + safe_div only (the ANCHOR)",
              AGG[("__anchor__", "raw ILP + safe_div only", None)], base))
    print(fmt("FULL DEPLOYED CHAIN (baseline)", base))
    st = base["stats"]
    print("  deployed-chain stage counts: "
          + " ".join(f"{k}={v:g}" for k, v in sorted(st.items()) if v))

    out = []
    print("\n" + "=" * 128)
    print("ONE KNOB AT A TIME, OFF THE DEPLOYED CONFIGURATION")
    print("=" * 128)
    inert, matters = [], []
    for sw in SWEEPS:
        rows = [(v, AGG[(sw["env"], f"{k}={v}", v)]) for k, v in sw["values"]]
        spread = max(a["proxy"] for _, a in rows) - min(a["proxy"] for _, a in rows)
        tag = "INERT" if spread < 1e-9 else ""
        (inert if spread < 1e-9 else matters).append((sw, rows, spread))
        print(f"\n--- {sw['env']}   [{sw['stage']}]   deployed={sw['dep']}  "
              f"notebook default={sw['default']}"
              + (f"  (held: {sw['fixed']})" if sw["fixed"] else "")
              + f"   spread={spread:.5f} {tag}")
        print(HDR)
        for v, a in rows:
            mark = "*DEPLOYED* " if v == sw["dep"] or (isinstance(v, bool) and int(v) == sw["dep"]) else ""
            print(fmt(f"  {v}", a, base, mark))
            out.append(dict(env=sw["env"], stage=sw["stage"], deployed=sw["dep"],
                            notebook_default=sw["default"], value=v,
                            proxy=a["proxy"], d_proxy=a["proxy"] - base["proxy"],
                            adj=a["adj"], J=a["J"], d_J=a["J"] - base["J"],
                            mult=a["mult"], ratio=a["ratio"], min_ratio=a["min_ratio"],
                            divJ=a["divJ"], dtp=a["dtp"], dfp=a["dfp"], dfn=a["dfn"],
                            mult_only=int(a["proxy"] > base["proxy"] + 1e-6
                                          and a["J"] <= base["J"] + 1e-6),
                            spread=spread))

    # ---- summary table -----------------------------------------------------
    print("\n" + "=" * 128)
    print("SUMMARY: best value per knob (delta vs the deployed chain)")
    print("=" * 128)
    print(f"{'env var':<48}{'deployed':>10}{'best':>10}{'d_proxy':>10}{'d_J':>10}"
          f"{'d_mult':>10}{'divJ':>7}  verdict")
    order = sorted(matters, key=lambda x: -max(a["proxy"] for _, a in x[1]) + base["proxy"])
    for sw, rows, spread in order:
        v, a = max(rows, key=lambda r: r[1]["proxy"])
        dJ = a["J"] - base["J"]
        dm = a["mult"] - base["mult"]
        dp = a["proxy"] - base["proxy"]
        if dp <= 1e-6:
            verdict = "deployed value is already best"
        elif dJ <= 1e-6:
            verdict = "MULTIPLIER-ONLY (metric exploit, not tracking)"
        elif abs(a["divJ"] - base["divJ"]) > 1e-9 and abs(dp - 0.1 * (a["divJ"] - base["divJ"])) < 1e-5:
            verdict = "division-driven -- 12 GT divisions, not resolvable"
        else:
            verdict = "real edge-Jaccard gain"
        print(f"{sw['env']:<48}{str(sw['dep']):>10}{str(v):>10}{dp:>+10.5f}"
              f"{dJ:>+10.5f}{dm:>+10.5f}{a['divJ']:>7.4f}  {verdict}")
    print(f"\nINERT (no effect at any value tested, {len(inert)} knobs):")
    for sw, rows, _ in inert:
        print(f"  {sw['env']:<50} deployed={sw['dep']}  default={sw['default']}  "
              f"[{sw['stage']}]  values tested: {[v for _, v in sw['values']]}")

    print(f"\nNOT TESTABLE on these graphs ({len(UNTESTABLE)} entries):")
    for e, d, dd, why in UNTESTABLE:
        print(f"  {e:<62} deployed={d:<26} default={dd:<14} {why}")

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow(r)
    with open(OUT_INV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["env", "stage", "deployed", "notebook_default", "testable",
                    "inert", "note"])
        inert_envs = {sw["env"] for sw, _, _ in inert}
        for sw in SWEEPS:
            w.writerow([sw["env"], sw["stage"], sw["dep"], sw["default"], 1,
                        int(sw["env"] in inert_envs), ""])
        for e, d, dd, why in UNTESTABLE:
            w.writerow([e, "untestable", d, dd, 0, "", why])
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_INV}")


if __name__ == "__main__":
    main()
