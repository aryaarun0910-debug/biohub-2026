"""Do the OTHER post-processing stages help on the RAW ILP graph (no relink)?

Motion relink is off (that was worth +0.0295 proxy on its own). The question
left over is whether the rest of the deployed chain -- gap closing, gap2
recovery, prune-isolated, short-track filtering, linefit smoothing -- are
carrying their weight once the ILP's own global assignment is kept.

Every stage is measured ONE AT A TIME against the raw + safe_div anchor.
Each stage sits at its DEPLOYED position in the chain:

    gap_close -> gap2 -> safe_div -> prune_isolated -> short_track -> linefit

so "gap close" means gap-close-then-safe_div, because gap closing eats the
orphan pool that safe_div draws its daughters from. Anything after safe_div is
a pure post-hoc edit of the same divided graph.

Stage implementations are ported from the 0.947 notebook (cell 2:
close_single_frame_gaps, recover_strict_gap2, filter_short_track_components,
linefit_smooth_output_graph, the OUTPUT_PRUNE_ISOLATED block) onto
ORIGINAL-VOXEL coordinates with the anisotropic SCALE. src/biohub/postprocess.py
has lookalikes but they use an isotropic GRID_UM = 1.625, which is 4x wrong in
y and x on these graphs -- do not use them here.

Node counts: the metric multiplies edge Jaccard by 1 - 0.1*(n_pred - n_est)/n_est.
Deleting nodes RAISES that multiplier, so every row prints n_pred/n_est and
J separately. A row where mult moves and J does not is a bookkeeping gain,
not a tracking gain.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub import io, metric2 as M2

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0"
OUT = ROOT / "artifacts/other_stages.csv"
SCALE = M2.SCALE

# ---------------------------------------------------------------- loaders
# (load_pred / load_gt / safe_div are the ones from scripts/24_keep_ilp_edges.py)
exec(open(ROOT / "scripts/24_keep_ilp_edges.py").read().split("stems = sorted")[0]
     .split('"""', 2)[2])


def G_of(P):
    return dict(t=P["t"].copy(), zyx=P["zyx"].copy(), edges=list(P["edges"]))


def _remap(G, keep):
    """Drop nodes where keep is False, renumbering edges."""
    idx = np.where(keep)[0]
    remap = -np.ones(len(G["t"]), np.int64)
    remap[idx] = np.arange(len(idx))
    return dict(t=G["t"][idx], zyx=G["zyx"][idx],
                edges=[(int(remap[s]), int(remap[d])) for s, d in G["edges"]
                       if keep[s] and keep[d]])


def _components(n, edges):
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for s, d in edges:
        ra, rb = find(int(s)), find(int(d))
        if ra != rb:
            par[ra] = rb
    return np.array([find(i) for i in range(n)])


# ---------------------------------------------------------------- stages
def prune_isolated(G):
    inc = np.zeros(len(G["t"]), bool)
    for s, d in G["edges"]:
        inc[s] = inc[d] = True
    return _remap(G, inc), {"pruned": int((~inc).sum())}


def short_track(G, min_len=6, keep_forks=True):
    n = len(G["t"])
    if not G["edges"]:
        return G, {"st_removed": 0}
    comp = _components(n, G["edges"])
    sizes = np.bincount(comp, minlength=n)
    outdeg = {}
    for s, _ in G["edges"]:
        outdeg[s] = outdeg.get(s, 0) + 1
    fork_comps = {comp[s] for s, k in outdeg.items() if k >= 2} if keep_forks else set()
    keep = sizes[comp] >= min_len
    if fork_comps:
        keep |= np.isin(comp, list(fork_comps))
    if not keep.any():
        return G, {"st_removed": 0}
    return _remap(G, keep), {"st_removed": int((~keep).sum()),
                             "st_comps": int(len({comp[i] for i in range(n) if not keep[i]}))}


def gap_close(G, max_um=12.0, reuse_um=3.2, allow_synth=True, synth_frac=0.05,
              refine=None):
    """Track ends at t rejoined to track starts at t+2, Hungarian per frame.

    Prefers an existing UNLINKED node near the midpoint (adds no node, removes
    an orphan) over inventing a synthetic one.  `refine` is an optional
    callable(t, voxel_midpoint) -> voxel point for image-based recentring.
    """
    t, zyx, edges = G["t"], G["zyx"], list(G["edges"])
    pos = zyx * SCALE
    outgoing = {int(s) for s, _ in edges}
    incoming = {int(d) for _, d in edges}
    incident = outgoing | incoming
    ends, starts, iso = {}, {}, {}
    for i, tt in enumerate(t):
        tt = int(tt)
        if i not in outgoing:
            ends.setdefault(tt, []).append(i)
        if i not in incoming:
            starts.setdefault(tt, []).append(i)
        if i not in incident:
            iso.setdefault(tt, []).append(i)

    n_base = len(t)
    cap = min(2000, max(1, round(n_base * synth_frac))) if allow_synth else 0
    new_nodes, added = [], []
    used_start, used_iso = set(), set()
    n_reuse = n_synth = 0

    for tt in sorted(ends):
        E = [i for i in ends[tt] if i not in outgoing]
        S = [j for j in starts.get(tt + 2, []) if j not in incoming and j not in used_start]
        if not E or not S:
            continue
        D = np.linalg.norm(pos[S][None] - pos[E][:, None], axis=-1)
        ok = D <= max_um
        if not ok.any():
            continue
        cost = np.where(ok, D, 1e6)
        for r, c in zip(*linear_sum_assignment(cost)):
            if not ok[r, c]:
                continue
            e, s = E[r], S[c]
            if e in outgoing or s in incoming or s in used_start:
                continue
            mid_t = tt + 1
            mid_vox = (zyx[e] + zyx[s]) / 2.0
            mid_um = mid_vox * SCALE
            pick = None
            free = [k for k in iso.get(mid_t, [])
                    if k not in used_iso and k not in incident]
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
            {"gap_pairs": len(added) // 2, "gap_reuse": n_reuse, "gap_synth": n_synth})


def gap2(G, max_total=10.2, max_step=4.4, require_context=True,
         frac_cap=0.0045, abs_cap=180, frame_frac=0.006, refine=None):
    """Two-frame recovery: end at t joined to start at t+3 via 2 synthetic nodes."""
    t, zyx, edges = G["t"], G["zyx"], list(G["edges"])
    pos = zyx * SCALE
    outgoing = {int(s) for s, _ in edges}
    incoming = {int(d) for _, d in edges}
    src_of, dst_of = {}, {}
    for s, d in edges:
        src_of.setdefault(int(d), []).append(int(s))
        dst_of.setdefault(int(s), []).append(int(d))
    pred = {k: v[0] for k, v in src_of.items() if len(v) == 1}
    succ = {k: v[0] for k, v in dst_of.items() if len(v) == 1}
    ends, starts = {}, {}
    for i, tt in enumerate(t):
        tt = int(tt)
        if i not in outgoing:
            ends.setdefault(tt, []).append(i)
        if i not in incoming:
            starts.setdefault(tt, []).append(i)

    cap = min(abs_cap, max(1, round(len(edges) * frac_cap)))
    props = []
    for tt in sorted(ends):
        S = starts.get(tt + 3, [])
        if not S:
            continue
        for e in ends[tt]:
            for s in S:
                step_v = (pos[s] - pos[e]) / 3.0
                dist = float(np.linalg.norm(pos[s] - pos[e]))
                if dist > max_total or dist / 3.0 > max_step:
                    continue
                pen, okc = 0.0, not require_context
                if require_context:
                    for nb, vec in ((pred.get(e), lambda p: pos[e] - pos[p]),
                                    (succ.get(s), lambda p: pos[p] - pos[s])):
                        if nb is None:
                            continue
                        v = vec(nb)
                        nv, ns = np.linalg.norm(v), np.linalg.norm(step_v)
                        if nv <= 0.01 or ns <= 0.01:
                            okc = True
                            continue
                        cos = float(np.dot(v, step_v) / (nv * ns + 1e-9))
                        if cos > -0.25 and np.linalg.norm(v - step_v) <= 6.0:
                            okc = True
                        pen += max(0.0, 0.25 - cos)
                    if not okc:
                        continue
                props.append((dist + 2.0 * pen, e, s, tt))
    props.sort()
    sel, ue, us, per_t = [], set(), set(), {}
    for score, e, s, tt in props:
        if len(sel) >= cap:
            break
        if e in ue or s in us:
            continue
        fc = max(1, round(len(ends.get(tt, [])) * frame_frac))
        if per_t.get(tt, 0) >= fc:
            continue
        sel.append((e, s, tt))
        ue.add(e); us.add(s); per_t[tt] = per_t.get(tt, 0) + 1

    new_nodes, added = [], []
    n_base = len(t)
    for e, s, tt in sel:
        prev = e
        for k in (1, 2):
            p = zyx[e] + (zyx[s] - zyx[e]) * (k / 3.0)
            if refine is not None:
                p = refine(tt + k, p)
            nid = n_base + len(new_nodes)
            new_nodes.append((tt + k, p))
            added.append((prev, nid))
            prev = nid
        added.append((prev, s))
    if new_nodes:
        t = np.concatenate([t, np.array([a for a, _ in new_nodes], t.dtype)])
        zyx = np.concatenate([zyx, np.array([b for _, b in new_nodes], zyx.dtype)])
    return (dict(t=t, zyx=zyx, edges=edges + added),
            {"gap2_pairs": len(sel), "gap2_nodes": len(new_nodes)})


def linefit(G, w=0.8, window=2):
    """Move each node toward a line fit over +-window frames of its unique chain.

    Topology and node count are untouched -- this can only move J, never mult.
    """
    if w <= 0 or not G["edges"]:
        return G, {"lf_moved": 0}
    t, zyx = G["t"], G["zyx"]
    pred, succ = {}, {}
    for s, d in G["edges"]:
        s, d = int(s), int(d)
        if int(t[d]) != int(t[s]) + 1:
            continue
        succ.setdefault(s, []).append(d)
        pred.setdefault(d, []).append(s)
    out = zyx.copy()
    moved = 0
    for i in range(len(t)):
        nbh = [(0, i)]
        cur = i
        for step in range(1, window + 1):
            p = pred.get(cur, [])
            if len(p) != 1:
                break
            cur = p[0]
            nbh.append((-step, cur))
        cur = i
        for step in range(1, window + 1):
            q = succ.get(cur, [])
            if len(q) != 1:
                break
            cur = q[0]
            nbh.append((step, cur))
        if len(nbh) < 3:
            continue
        dts = np.array([a for a, _ in nbh], float)
        co = zyx[[b for _, b in nbh]]
        fit = np.array([np.polyval(np.polyfit(dts, co[:, k], 1), 0.0) for k in range(3)])
        if not np.isfinite(fit).all():
            continue
        out[i] = (1.0 - w) * zyx[i] + w * fit
        moved += 1
    return dict(t=t, zyx=out, edges=list(G["edges"])), {"lf_moved": moved}


# ---------------------------------------------------------------- harness
stems = sorted(p.stem for p in PRED.glob("*.geff"))
DATA = {s: (load_pred(PRED / f"{s}.geff"), load_gt(s)) for s in stems}


def make_refine(stem, win_z=1, win_yx=3, max_shift=3.2):
    """Image-centroid recentring of a synthetic point, as the notebook does it."""
    path = io.dataset_root() / "train" / f"{stem}.zarr"
    cache = {}

    def refine(t, vox):
        try:
            f = cache.get(t)
            if f is None:
                f = cache[t] = io.read_frame(path, t)
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
            g = np.array([float((wgt * a).sum() / tot) for a in np.ogrid[z0:z1, y0:y1, x0:x1]])
            if np.linalg.norm((g - vox) * SCALE) > max_shift:
                return vox
            return g
        except Exception:
            return vox

    return refine


def run(build, label, rows_out=None):
    """build(G, P, stem) -> (G, stats). Returns an aggregate row."""
    rows, agg_stats, npred, nest = [], {}, 0.0, 0.0
    for stem, (P, GT) in DATA.items():
        G, st = build(G_of(P), P, stem)
        for k, v in st.items():
            agg_stats[k] = agg_stats.get(k, 0) + v
        rows.append(M2.score(G["t"], G["zyx"], G["edges"],
                             GT["t"], GT["zyx"], GT["edges"], GT["n_est"]))
        npred += len(G["t"]); nest += GT["n_est"]
    r = M2.aggregate(rows)
    r["label"] = label
    r["ratio"] = npred / nest
    r["min_ratio"] = min(x["n_pred"] / x["n_est"] for x in rows)
    r["stats"] = agg_stats
    if rows_out is not None:
        rows_out.append(r)
    return r


HDR = (f"{'configuration':<40}{'proxy':>9}{'dprox':>9}{'J':>9}{'dJ':>9}"
       f"{'mult':>8}{'dmult':>9}{'n/n_est':>8}{'divJ':>8}{'TP/FP/FN':>9}  notes")


def show(r, base=None):
    if base is None:
        dp = dj = dm = ""
        flag = ""
    else:
        dp = f"{r['proxy'] - base['proxy']:+.5f}"
        dj = f"{r['J'] - base['J']:+.5f}"
        dm = f"{r['mult'] - base['mult']:+.5f}"
        # false gain: proxy up, but the edge Jaccard did not move up with it
        flag = ("  <<FALSE GAIN: all multiplier"
                if r["proxy"] > base["proxy"] + 1e-6 and r["J"] <= base["J"] + 1e-6 else "")
    notes = " ".join(f"{k}={v}" for k, v in sorted(r["stats"].items()) if v)
    div = f"{r['dtp']}/{r['dfp']}/{r['dfn']}"
    print(f"{r['label']:<40}{r['proxy']:>9.5f}{dp:>9}{r['J']:>9.5f}{dj:>9}"
          f"{r['mult']:>8.5f}{dm:>9}{r['ratio']:>8.4f}{r['divJ']:>8.4f}{div:>9}"
          f"  {notes}{flag}")


ALL = []
print("graph shape: no isolated nodes, min component size 4 in every film "
      "-- the ILP already enforces this.")
print("n_pred/n_est per film runs 0.73 .. 1.38; the metric multiplier is UNCAPPED, "
      "so deleting nodes pays forever.\n")
print(HDR)
raw = run(lambda G, P, s: (G, {}), "raw ILP, nothing applied", ALL)
show(raw)
base = run(lambda G, P, s: (dict(G, edges=safe_div(dict(P, edges=G["edges"]))[0]), {}),
           "ANCHOR raw + safe_div", ALL)
show(base, raw)
print()


def SD(G, P):
    """safe_div on a (possibly node-augmented) graph."""
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    e, a = safe_div(Q)
    return dict(G, edges=e), len(a)


# ---- 1. prune isolated -------------------------------------------------
print("--- stage 1: prune isolated nodes (after safe_div) ---")
def _prune(G, P, s):
    G, _ = SD(G, P)
    return prune_isolated(G)
show(run(_prune, "  + prune isolated", ALL), base)
print()

# ---- 2. short-track component filter -----------------------------------
print("--- stage 2: short-track component filter (after safe_div) ---")
for L in (4, 6, 8):
    for kf in (True, False):
        def _st(G, P, s, L=L, kf=kf):
            G, _ = SD(G, P)
            return short_track(G, L, kf)
        show(run(_st, f"  + short_track L={L} keep_forks={int(kf)}", ALL), base)
print("  (monotonicity probe -- if the gain were real it would peak somewhere)")
for L in (12, 20, 40):
    def _stx(G, P, s, L=L):
        G, _ = SD(G, P)
        return short_track(G, L, True)
    show(run(_stx, f"  + short_track L={L} keep_forks=1", ALL), base)
print()

# ---- 3. single-frame gap closing (before safe_div) ---------------------
print("--- stage 3: single-frame gap closing, t -> t+2 (before safe_div) ---")
for thr in (5.0, 6.0, 8.0, 10.0, 12.0, 14.0):
    def _gc(G, P, s, thr=thr):
        G, st = gap_close(G, max_um=thr, allow_synth=True)
        G, _ = SD(G, P)
        return G, st
    show(run(_gc, f"  + gap_close thr={thr:g}um (reuse+synth)", ALL), base)
for thr in (6.0, 10.0, 12.0):
    def _gcr(G, P, s, thr=thr):
        G, st = gap_close(G, max_um=thr, allow_synth=False)
        G, _ = SD(G, P)
        return G, st
    show(run(_gcr, f"  + gap_close thr={thr:g}um REUSE-ONLY", ALL), base)
for thr in (6.0, 12.0):
    def _gcs(G, P, s, thr=thr):
        G, st = gap_close(G, max_um=thr, reuse_um=0.0, allow_synth=True)
        G, _ = SD(G, P)
        return G, st
    show(run(_gcs, f"  + gap_close thr={thr:g}um SYNTH-ONLY", ALL), base)
print("  (ordering probe: gap close AFTER safe_div, so it cannot eat daughters)")
for thr in (8.0, 12.0):
    def _gca(G, P, s, thr=thr):
        G, _ = SD(G, P)
        return gap_close(G, max_um=thr, allow_synth=True)
    show(run(_gca, f"  + safe_div then gap_close thr={thr:g}um", ALL), base)
print("  (image-refined synthetic midpoints, deployed GAP_REFINE_* settings)")
for thr in (8.0,):
    def _gcref(G, P, s, thr=thr):
        G, st = gap_close(G, max_um=thr, allow_synth=True, refine=make_refine(s))
        G, _ = SD(G, P)
        return G, st
    show(run(_gcref, f"  + gap_close thr={thr:g}um IMAGE-REFINED", ALL), base)
print()

# ---- 3b. two-frame gap recovery ---------------------------------------
print("--- stage 3b: 2-frame gap recovery, t -> t+3 (before safe_div) ---")
for tot, stp in ((10.2, 4.4), (14.0, 6.0)):
    def _g2(G, P, s, tot=tot, stp=stp):
        G, st = gap2(G, max_total=tot, max_step=stp)
        G, _ = SD(G, P)
        return G, st
    show(run(_g2, f"  + gap2 total={tot:g} step={stp:g}", ALL), base)
def _g2nc(G, P, s):
    G, st = gap2(G, require_context=False)
    G, _ = SD(G, P)
    return G, st
show(run(_g2nc, "  + gap2 no-context-gate", ALL), base)
print("  (ordering probe: gap2 AFTER safe_div)")
def _g2a(G, P, s):
    G, _ = SD(G, P)
    return gap2(G)
show(run(_g2a, "  + safe_div then gap2", ALL), base)
print()

# ---- 4. linefit smoothing ---------------------------------------------
print("--- stage 4: linefit smoothing (after safe_div; cannot move mult) ---")
for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
    def _lf(G, P, s, w=w):
        G, _ = SD(G, P)
        return linefit(G, w=w)
    show(run(_lf, f"  + linefit w={w:g} window=2", ALL), base)
for win in (1, 3, 4, 5):
    for w in (0.3, 0.4):
        def _lfw(G, P, s, win=win, w=w):
            G, _ = SD(G, P)
            return linefit(G, w=w, window=win)
        show(run(_lfw, f"  + linefit w={w:g} window={win}", ALL), base)
print("  (linefit BEFORE safe_div -- smoother positions feed the division gates)")
for w in (0.3, 0.4):
    def _lfb(G, P, s, w=w):
        G, st = linefit(G, w=w)
        G, _ = SD(G, P)
        return G, st
    show(run(_lfb, f"  + linefit w={w:g} then safe_div", ALL), base)
print()

# ---- 5. combinations ---------------------------------------------------
print("--- best combinations (J-honest first, then with the node-deleting filter) ---")
def _c1(G, P, s):
    G, st = gap_close(G, max_um=8.0, allow_synth=True)
    G, _ = SD(G, P)
    G2, st2 = linefit(G, w=0.4)
    return G2, {**st, **st2}
show(run(_c1, "  gap_close8 + safe_div + linefit0.4", ALL), base)

def _c2(G, P, s):
    G, stl = linefit(G, w=0.4)
    G, st = gap_close(G, max_um=8.0, allow_synth=True)
    G, _ = SD(G, P)
    return G, {**st, **stl}
show(run(_c2, "  linefit0.4 + gap_close8 + safe_div", ALL), base)

def _c3(G, P, s):
    G, st = gap_close(G, max_um=8.0, allow_synth=True)
    G, _ = SD(G, P)
    G, st2 = short_track(G, 6, True)
    G, st3 = linefit(G, w=0.4)
    return G, {**st, **st2, **st3}
show(run(_c3, "  gap_close8+safe_div+short6+linefit0.4", ALL), base)

def _c4(G, P, s):
    G, st = gap_close(G, max_um=8.0, allow_synth=True)
    G, _ = SD(G, P)
    G, st2 = short_track(G, 8, True)
    G, st3 = linefit(G, w=0.4)
    return G, {**st, **st2, **st3}
show(run(_c4, "  gap_close8+safe_div+short8+linefit0.4", ALL), base)


def BEST(G, P, s, lw=0.4, lwin=3, use_gap2=True):
    G, st = gap_close(G, max_um=8.0, allow_synth=True)
    G, _ = SD(G, P)
    if use_gap2:
        G, st2 = gap2(G)
        st.update(st2)
    G, st3 = linefit(G, w=lw, window=lwin)
    return G, {**st, **st3}


for lw, lwin, g2 in ((0.4, 3, True), (0.3, 3, True), (0.4, 3, False),
                     (0.4, 4, True), (0.3, 4, True)):
    show(run(lambda G, P, s, a=lw, b=lwin, c=g2: BEST(G, P, s, a, b, c),
             f"  BEST gc8+sd+{'gap2+' if g2 else ''}lf{lw:g}/w{lwin}", ALL), base)
print()

# ---- per-film consistency of the winner -------------------------------
print("--- per-film: is the winner's gain broad or one film? ---")
print(f"{'film':<24}{'anchor J':>10}{'best J':>10}{'dJ':>10}"
      f"{'anchor adj':>12}{'best adj':>10}{'dadj':>10}{'n/n_est':>9}")
nwin = 0
for stem, (P, GT) in DATA.items():
    Ga = dict(G_of(P))
    Ga["edges"] = safe_div(dict(P, edges=Ga["edges"]))[0]
    ra = M2.score(Ga["t"], Ga["zyx"], Ga["edges"], GT["t"], GT["zyx"], GT["edges"], GT["n_est"])
    Gb, _ = BEST(G_of(P), P, stem)
    rb = M2.score(Gb["t"], Gb["zyx"], Gb["edges"], GT["t"], GT["zyx"], GT["edges"], GT["n_est"])
    nwin += rb["J_edge"] > ra["J_edge"]
    print(f"{stem:<24}{ra['J_edge']:>10.5f}{rb['J_edge']:>10.5f}{rb['J_edge']-ra['J_edge']:>+10.5f}"
          f"{ra['adj']:>12.5f}{rb['adj']:>10.5f}{rb['adj']-ra['adj']:>+10.5f}"
          f"{rb['n_pred']/rb['n_est']:>9.4f}")
print(f"films where edge J improved: {nwin}/8")
print()

import csv
with open(OUT, "w", newline="") as fh:
    wtr = csv.writer(fh)
    wtr.writerow(["label", "proxy", "delta", "adj", "J", "mult", "ratio",
                  "min_ratio", "divJ", "dtp", "dfp", "dfn", "notes"])
    for r in ALL:
        wtr.writerow([r["label"], f"{r['proxy']:.6f}", f"{r['proxy']-base['proxy']:+.6f}",
                      f"{r['adj']:.6f}", f"{r['J']:.6f}", f"{r['mult']:.6f}",
                      f"{r['ratio']:.4f}", f"{r['min_ratio']:.4f}", f"{r['divJ']:.4f}",
                      r["dtp"], r["dfp"], r["dfn"],
                      " ".join(f"{k}={v}" for k, v in sorted(r["stats"].items()) if v)])
print(f"wrote {OUT}")
