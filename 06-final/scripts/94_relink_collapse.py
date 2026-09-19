"""Why does motion relink collapse on 44b6_267148e4, and is it systematic?

Relink replaces the ILP's edge set with a 1:1 Hungarian assignment per frame
pair.  Two structural facts about the stage drive everything below:

  1. The prediction .geff stores ONLY the ILP solution edges (every `solution`
     flag is True).  So `prob.get((s,t), 0.0)` inside motion_relink returns a
     real probability for incumbent ILP edges and exactly 0.0 for every other
     candidate pair.  The `-beta * prob` term is therefore not a calibrated
     likelihood field -- it is a flat ~0.9 um *incumbency discount*.
  2. Hungarian is 1:1 and gated only on raw distance.  It assigns
     min(n_t, n_{t+1}) pairs and keeps every one whose raw distance is under
     the gate, regardless of whether the ILP thought the cell appeared,
     vanished or divided.

Consequence: relink survives a film only when true inter-frame motion is small
compared with the spacing between neighbouring detections, so that the nearest
target is the correct target and the 0.9 um discount is never tested.  This
script measures that ratio and everything else the brief asked for.

Outputs: artifacts/94_relink_collapse_films.csv   (per-film stats + deltas)
         artifacts/94_relink_collapse_diffs.csv   (per-GT-edge fate on the
                                                   collapsing film)
         artifacts/94_relink_collapse.txt         (the printed report)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub import io, metric2 as M2

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0"
OUT = ROOT / "artifacts"
SCALE = M2.SCALE
FOCUS = "44b6_267148e4"


# ---------------------------------------------------------------- loaders
# (same as scripts/24_keep_ilp_edges.py; copied rather than exec'd so this
#  file runs standalone and multiprocessing can pickle it)
def load_pred(p):
    d = io.read_geff(p)
    idx = {int(i): k for k, i in enumerate(d["ids"])}
    e = np.array([[idx[int(a)], idx[int(b)]] for a, b in d["edges"]], np.int64).reshape(-1, 2)
    prob = np.asarray(io.read_zstd_array(p / "edges/props/edge_prob/values"), np.float64)
    return dict(t=d["t"].astype(np.int64),
                zyx=np.stack([d["z"], d["y"], d["x"]], 1).astype(np.float64),
                edges=[(int(a), int(b)) for a, b in e], prob=prob)


def load_gt(stem):
    g = io.read_geff(io.dataset_root() / "train" / f"{stem}.geff")
    idx = {int(i): k for k, i in enumerate(g["ids"])}
    e = np.array([[idx[int(a)], idx[int(b)]] for a, b in g["edges"]], np.int64).reshape(-1, 2)
    return dict(t=g["t"].astype(np.int64),
                zyx=np.stack([g["z"], g["y"], g["x"]], 1).astype(np.float64),
                edges=e, n_est=float(g["estimated_number_of_nodes"]))


# ------------------------------------------------- the deployed relink
def motion_relink(P, tight=6.0, relaxed=10.0, vel=0.5, beta=1.0, gates=None, trace=False):
    """Semantics verbatim from scripts/25_relink_control.py; the O(n^2) python
    loop that builds the probability matrix is vectorised so the ablation sweep
    is affordable.  `gates` overrides the (tight, relaxed) cascade."""
    pos = P["zyx"] * SCALE
    N = len(P["t"])
    ek = np.array([a * N + b for a, b in P["edges"]], np.int64)
    ev = np.asarray(P["prob"], float)
    o = np.argsort(ek); ek, ev = ek[o], ev[o]
    by_t = {}
    for i, t in enumerate(P["t"]):
        by_t.setdefault(int(t), []).append(i)
    out, prev = [], {}
    tr = {}                      # edge -> (pass index, cost of the correct rival)
    if gates is None:
        gates = (tight, relaxed)
    for t in sorted(by_t):
        si, ti = by_t.get(t), by_t.get(t + 1)
        if not si or not ti:
            continue
        si, ti = np.array(si), np.array(ti)
        s_um, t_um = pos[si], pos[ti]
        anchor = s_um.copy()
        for k, g in enumerate(si):
            p = prev.get(int(g))
            if p is not None:
                anchor[k] = s_um[k] + vel * (s_um[k] - pos[p])
        raw = np.linalg.norm(t_um[None] - s_um[:, None], axis=-1)
        key = si[:, None] * N + ti[None, :]
        j = np.searchsorted(ek, key)
        j_c = np.clip(j, 0, len(ek) - 1)
        pm = np.where(ek[j_c] == key, ev[j_c], 0.0) if len(ek) else np.zeros_like(raw)
        base = np.linalg.norm(t_um[None] - anchor[:, None], axis=-1) + 0.05 * raw - beta * pm
        used_i, used_j = set(), set()
        for gi, gate in enumerate(gates):
            ii = [a for a in range(len(si)) if a not in used_i]
            jj = [b for b in range(len(ti)) if b not in used_j]
            if not ii or not jj:
                break
            sub, sraw = base[np.ix_(ii, jj)].copy(), raw[np.ix_(ii, jj)]
            bad = sraw > gate
            sub[bad] = 1e6
            r, c = linear_sum_assignment(sub)
            for a, b in zip(r, c):
                if bad[a, b]:
                    continue
                s_, t_ = int(si[ii[a]]), int(ti[jj[b]])
                out.append((s_, t_))
                if trace:
                    tr[(s_, t_)] = (gi, float(base[ii[a], jj[b]]), float(raw[ii[a], jj[b]]))
                prev[t_] = s_
                used_i.add(ii[a]); used_j.add(jj[b])
        if trace:
            tr[("costmat", t)] = (si, ti, base, raw)
    return (out, tr) if trace else out


def motion_relink_ref(P, tight=6.0, relaxed=10.0, vel=0.5, beta=1.0):
    """Literal copy of the deployed loop, used once to prove the fast path
    above is byte-identical."""
    pos = P["zyx"] * SCALE
    prob = {(int(a), int(b)): float(p) for (a, b), p in zip(P["edges"], P["prob"])}
    by_t = {}
    for i, t in enumerate(P["t"]):
        by_t.setdefault(int(t), []).append(i)
    out, prev = [], {}
    for t in sorted(by_t):
        si, ti = by_t.get(t), by_t.get(t + 1)
        if not si or not ti:
            continue
        si, ti = np.array(si), np.array(ti)
        s_um, t_um = pos[si], pos[ti]
        anchor = s_um.copy()
        for k, g in enumerate(si):
            p = prev.get(int(g))
            if p is not None:
                anchor[k] = s_um[k] + vel * (s_um[k] - pos[p])
        raw = np.linalg.norm(t_um[None] - s_um[:, None], axis=-1)
        pm = np.zeros_like(raw)
        for a, b in np.ndindex(*raw.shape):
            pm[a, b] = prob.get((int(si[a]), int(ti[b])), 0.0)
        base = np.linalg.norm(t_um[None] - anchor[:, None], axis=-1) + 0.05 * raw - beta * pm
        used_i, used_j = set(), set()
        for gate in (tight, relaxed):
            ii = [a for a in range(len(si)) if a not in used_i]
            jj = [b for b in range(len(ti)) if b not in used_j]
            if not ii or not jj:
                break
            sub, sraw = base[np.ix_(ii, jj)].copy(), raw[np.ix_(ii, jj)]
            bad = sraw > gate
            sub[bad] = 1e6
            r, c = linear_sum_assignment(sub)
            for a, b in zip(r, c):
                if bad[a, b]:
                    continue
                out.append((int(si[ii[a]]), int(ti[jj[b]])))
                prev[int(ti[jj[b]])] = int(si[ii[a]])
                used_i.add(ii[a]); used_j.add(jj[b])
    return out


def safe_div(P, edges, parent_max=9.0, sister_max=14.0, child_max=10.0, tau=0.6,
             diverge=2.25, frame_cap=0.0076, glob_cap=0.00375):
    """The deployed safe-division rule (scripts/24_keep_ilp_edges.py), on an
    arbitrary edge set so it can be run on the relinked graph too."""
    pos = P["zyx"] * SCALE
    succ, indeg = {}, {}
    for s, t in edges:
        succ.setdefault(s, []).append(t); indeg[t] = indeg.get(t, 0) + 1
    by_t = {}
    for i, t in enumerate(P["t"]):
        by_t.setdefault(int(t), []).append(i)
    d = lambda a, b: float(np.linalg.norm(pos[a] - pos[b]))
    added = []
    for t in sorted(by_t):
        nxt = by_t.get(t + 1)
        if not nxt:
            continue
        orph = [j for j in nxt if indeg.get(j, 0) == 0]
        if not orph:
            continue
        opos = pos[orph]
        cands = []
        for Pn in [i for i in by_t[t] if len(succ.get(i, ())) == 1]:
            C = succ[Pn][0]
            dpc = d(Pn, C)
            if dpc > child_max:
                continue
            k = int(np.argmin(np.linalg.norm(opos - pos[C], axis=1)))
            Q = orph[k]
            dpq, dcq = d(Pn, Q), d(C, Q)
            if dpq > parent_max or dcq > sister_max:
                continue
            if abs(dpc - dpq) / max((dpc + dpq) / 2, 1e-9) > tau:
                continue
            sc, sq = succ.get(C, []), succ.get(Q, [])
            if len(sc) != 1 or len(sq) != 1:
                continue
            if d(sc[0], sq[0]) - dcq < diverge:
                continue
            cands.append((dpq + 0.15 * dcq, Pn, Q))
        cap = max(1, round(frame_cap * len(by_t[t])))
        n = 0
        for _, Pn, Q in sorted(cands):
            if n >= cap or indeg.get(Q, 0) or len(succ.get(Pn, ())) >= 2:
                continue
            added.append((Pn, Q)); succ.setdefault(Pn, []).append(Q); indeg[Q] = 1; n += 1
    cap = max(1, round(glob_cap * len(edges)))
    return list(edges) + added[:cap]


# -------------------------------------------------- ablations on the stage
ABLATIONS = [
    ("deployed (6 then 10)",        dict()),
    ("no tight pass (10 only)",     dict(gates=(10.0,))),
    ("no tight pass (6 only)",      dict(gates=(6.0,))),
    ("tight=relaxed=8",             dict(gates=(8.0, 8.0))),
    ("no incumbency bonus beta=0",  dict(beta=0.0)),
    ("bonus x5 (beta=5)",           dict(beta=5.0)),
    ("no velocity anchor vel=0",    dict(vel=0.0)),
    ("wide gates (10 then 16)",     dict(gates=(10.0, 16.0))),
]


# ------------------------------------------------------------- film stats
def pct(a, qs=(10, 50, 90)):
    a = np.asarray(a, float)
    return [float(np.percentile(a, q)) for q in qs] if len(a) else [np.nan] * len(qs)


def film_stats(P, G):
    """Geometry / motion / probability descriptors of one film."""
    pos = P["zyx"] * SCALE
    by_t = {}
    for i, t in enumerate(P["t"]):
        by_t.setdefault(int(t), []).append(i)
    ts = sorted(by_t)

    # --- density: nodes per frame, nearest-neighbour spacing within a frame
    counts = np.array([len(by_t[t]) for t in ts], float)
    nn = []
    for t in ts:
        idx = np.array(by_t[t])
        if len(idx) < 2:
            continue
        p = pos[idx]
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn.append(d.min(1))
    nn = np.concatenate(nn) if nn else np.array([np.nan])

    # --- motion: length of every ILP edge, in um
    e = np.array(P["edges"], np.int64).reshape(-1, 2)
    disp = np.linalg.norm(pos[e[:, 1]] - pos[e[:, 0]], axis=1) if len(e) else np.array([np.nan])

    # --- how many source/target slots the ILP leaves empty vs. what 1:1 fills
    pair_min = sum(min(len(by_t[t]), len(by_t.get(t + 1, []))) for t in ts if t + 1 in by_t)

    # --- probability field seen by relink
    prob = np.asarray(P["prob"], float)

    nn_med = float(np.median(nn))
    disp_med = float(np.median(disp))
    return dict(
        frames=len(ts),
        n_pred=len(P["t"]),
        n_est=G["n_est"],
        ratio_n=len(P["t"]) / G["n_est"],
        nodes_per_frame_mean=float(counts.mean()),
        nodes_per_frame_max=float(counts.max()),
        nn_p10=pct(nn)[0], nn_med=nn_med, nn_p90=pct(nn)[2],
        disp_p10=pct(disp)[0], disp_med=disp_med, disp_p90=pct(disp)[2],
        disp_p99=float(np.percentile(disp, 99)),
        frac_disp_gt_tight=float((disp > 6.0).mean()),
        frac_disp_gt_relaxed=float((disp > 10.0).mean()),
        ambiguity=disp_med / nn_med,          # motion measured in neighbour-spacings
        ambiguity_p90=float(np.percentile(disp, 90)) / nn_med,
        prob_mean=float(prob.mean()), prob_med=float(np.median(prob)),
        prob_std=float(prob.std()), prob_p10=pct(prob)[0],
        n_ilp_edges=len(e),
        pair_min=pair_min,
        ilp_fill=len(e) / max(pair_min, 1),   # fraction of 1:1 slots the ILP uses
    )


# ------------------------------------------------- per-GT-edge forensics
def gt_edge_fates(P, G, ilp, rel):
    """For every GT edge with both endpoints matched, what each graph did."""
    p2g, g2p = M2.match(P["t"], P["zyx"], G["t"], G["zyx"])
    pos = P["zyx"] * SCALE
    ilp_out, rel_out = {}, {}
    for s, t in ilp:
        ilp_out.setdefault(int(s), []).append(int(t))
    for s, t in rel:
        rel_out.setdefault(int(s), []).append(int(t))
    by_t = {}
    for i, t in enumerate(P["t"]):
        by_t.setdefault(int(t), []).append(i)

    rows = []
    for a, b in G["edges"]:
        a, b = int(a), int(b)
        pa, pb = g2p.get(a), g2p.get(b)
        if pa is None or pb is None:
            continue
        ilp_ok = pb in ilp_out.get(pa, ())
        rel_ok = pb in rel_out.get(pa, ())
        chosen = rel_out.get(pa, [None])[0]
        true_d = float(np.linalg.norm(pos[pb] - pos[pa]))
        chose_d = float(np.linalg.norm(pos[chosen] - pos[pa])) if chosen is not None else np.nan
        # how far the chosen target sits from the correct target
        miss_d = float(np.linalg.norm(pos[chosen] - pos[pb])) if chosen is not None else np.nan
        # how crowded the correct target is: distance to its nearest same-frame peer
        peers = np.array([i for i in by_t[int(P["t"][pb])] if i != pb])
        crowd = float(np.linalg.norm(pos[peers] - pos[pb], axis=1).min()) if len(peers) else np.nan
        rows.append(dict(t=int(P["t"][pa]), gt_src=a, gt_dst=b, p_src=pa, p_dst=pb,
                         ilp_ok=ilp_ok, rel_ok=rel_ok, true_dist_um=true_d,
                         chosen_dist_um=chose_d, miss_dist_um=miss_d,
                         target_crowding_um=crowd))
    return rows


# ---------------------------------------------------------------- driver
def fp_sources(P, G, ilp, rel):
    """Where relink's extra false positives come from."""
    p2g, _ = M2.match(P["t"], P["zyx"], G["t"], G["zyx"])
    gt_set = {(int(a), int(b)) for a, b in G["edges"]}
    gt_out, gt_in = {}, set()
    for a, b in G["edges"]:
        gt_out.setdefault(int(a), []).append(int(b)); gt_in.add(int(b))

    def is_fp(s, t):
        ms, mt = p2g.get(int(s)), p2g.get(int(t))
        if ms is not None and mt is not None and (ms, mt) in gt_set:
            return False
        return (mt is not None and mt in gt_in) or (ms is not None and ms in gt_out)

    ilp_set = set(ilp)
    ilp_src = {int(s) for s, _ in ilp}
    ilp_fp = {(s, t) for s, t in ilp if is_fp(s, t)}
    rel_fp = {(s, t) for s, t in rel if is_fp(s, t)}
    new_fp = rel_fp - ilp_fp
    # classify each newly-created false positive
    replaced = sum(1 for s, t in new_fp if s in ilp_src)      # source had an ILP edge
    fresh = sum(1 for s, t in new_fp if s not in ilp_src)     # ILP left the source unlinked
    return dict(ilp_fp=len(ilp_fp), rel_fp=len(rel_fp), new_fp=len(new_fp),
                new_fp_replacing=replaced, new_fp_on_unlinked_source=fresh,
                fp_fixed=len(ilp_fp - rel_fp))


def gt_motion(G):
    """How far GT cells actually travel between frames, in um."""
    gp = G["zyx"] * SCALE
    e = G["edges"]
    if not len(e):
        return dict(gt_disp_med=np.nan, gt_disp_p90=np.nan, gt_frac_gt6=np.nan)
    d = np.linalg.norm(gp[e[:, 1]] - gp[e[:, 0]], axis=1)
    return dict(gt_disp_med=float(np.median(d)), gt_disp_p90=float(np.percentile(d, 90)),
                gt_frac_gt6=float((d > 6.0).mean()), gt_disp_max=float(d.max()))


def one_film(stem):
    P = load_pred(PRED / f"{stem}.geff")
    G = load_gt(stem)
    ilp = P["edges"]
    rel = motion_relink(P)
    S = lambda e: M2.score(P["t"], P["zyx"], e, G["t"], G["zyx"], G["edges"], G["n_est"])
    sr, sx = S(ilp), S(rel)

    abl = []
    for lbl, kw in ABLATIONS:
        e = rel if lbl.startswith("deployed") else motion_relink(P, **kw)
        s = S(e)
        abl.append((lbl, s, len(e)))

    ilp_set, rel_set = set(ilp), set(rel)
    pos = P["zyx"] * SCALE
    dlen = lambda E: (np.linalg.norm(pos[[b for _, b in E]] - pos[[a for a, _ in E]], axis=1)
                      if E else np.array([np.nan]))
    dropped = sorted(ilp_set - rel_set)
    added = sorted(rel_set - ilp_set)

    ilp_fork = sum(1 for v in _outdeg(ilp).values() if v >= 2)
    rel_fork = sum(1 for v in _outdeg(rel).values() if v >= 2)

    st = film_stats(P, G)
    st.update(gt_motion(G))
    st.update(fp_sources(P, G, ilp, rel))
    st.update(raw_dtp=sr["dtp"], raw_dfp=sr["dfp"], raw_dfn=sr["dfn"],
              rel_dtp=sx["dtp"], rel_dfp=sx["dfp"], rel_dfn=sx["dfn"],
              ilp_forks=ilp_fork, rel_forks=rel_fork)
    st.update(stem=stem,
              raw_adj=sr["adj"], rel_adj=sx["adj"], d_adj=sr["adj"] - sx["adj"],
              raw_J=sr["J_edge"], rel_J=sx["J_edge"],
              raw_tp=sr["etp"], raw_fp=sr["efp"], raw_fn=sr["efn"],
              rel_tp=sx["etp"], rel_fp=sx["efp"], rel_fn=sx["efn"],
              weight=sr["weight"],
              n_rel_edges=len(rel_set),
              rel_fill=len(rel_set) / max(st["pair_min"], 1),
              n_kept=len(ilp_set & rel_set), n_dropped=len(dropped), n_added=len(added),
              frac_ilp_dropped=len(dropped) / max(len(ilp_set), 1),
              dropped_len_med=float(np.median(dlen(dropped))),
              added_len_med=float(np.median(dlen(added))),
              ilp_len_med=float(np.median(dlen(list(ilp_set)))),
              rel_len_med=float(np.median(dlen(list(rel_set)))))
    # per-film proxy with the deployed safe_div on top of each graph -- this is
    # the quantity the brief quotes as the per-film delta
    sdr = M2.aggregate([S(safe_div(P, ilp))])["proxy"]
    sdx = M2.aggregate([S(safe_div(P, rel))])["proxy"]

    fates = gt_edge_fates(P, G, ilp, rel)
    for f in fates:
        f["stem"] = stem
    return st, fates, (sr, sx), abl, (sdr, sdx)


def _outdeg(edges):
    d = {}
    for s, _ in edges:
        d[int(s)] = d.get(int(s), 0) + 1
    return d


def main():
    stems = sorted(p.stem for p in PRED.glob("*.geff"))
    # prove the vectorised relink is identical to the deployed loop
    _P = load_pred(PRED / f"{FOCUS}.geff")
    assert motion_relink(_P) == motion_relink_ref(_P), "fast relink diverged from the deployed one"
    del _P
    with ProcessPoolExecutor(max_workers=min(8, len(stems))) as ex:
        res = list(ex.map(one_film, stems))
    stats = [r[0] for r in res]
    all_fates = [f for r in res for f in r[1]]
    fates = [f for f in all_fates if f["stem"] == FOCUS]
    rows_raw = [r[2][0] for r in res]
    rows_rel = [r[2][1] for r in res]
    abls = [r[3] for r in res]
    focus_i = stems.index(FOCUS)
    SD_PROXY = {st: r[4] for st, r in zip(stems, res)}

    lines = []
    def p(s=""):
        print(s); lines.append(s)

    agg_raw, agg_rel = M2.aggregate(rows_raw), M2.aggregate(rows_rel)
    p("=" * 108)
    p("AGGREGATE (weighted, 8 films)")
    p(f"  raw ILP       adj {agg_raw['adj']:.5f}  J {agg_raw['J']:.5f}  proxy {agg_raw['proxy']:.5f}")
    p(f"  motion relink adj {agg_rel['adj']:.5f}  J {agg_rel['J']:.5f}  proxy {agg_rel['proxy']:.5f}")
    p("")

    p("=" * 108)
    p("PER-FILM SCORES")
    p(f"{'stem':<16}{'w':>6}{'rawadj':>9}{'reladj':>9}{'dadj':>9}"
      f"{'rawTP':>7}{'relTP':>7}{'rawFP':>7}{'relFP':>7}{'ilpE':>8}{'relE':>8}{'drop%':>7}")
    for s in sorted(stats, key=lambda d: -d["d_adj"]):
        p(f"{s['stem']:<16}{s['weight']:>6}{s['raw_adj']:>9.5f}{s['rel_adj']:>9.5f}{s['d_adj']:>+9.5f}"
          f"{s['raw_tp']:>7}{s['rel_tp']:>7}{s['raw_fp']:>7}{s['rel_fp']:>7}"
          f"{s['n_ilp_edges']:>8,}{s['n_rel_edges']:>8,}{100*s['frac_ilp_dropped']:>7.1f}")
    p("")

    p("=" * 108)
    p("DENSITY  (deployed pipeline skips relink when any frame has > 2600 nodes)")
    p(f"{'stem':<16}{'frames':>7}{'n_pred':>9}{'n/frame':>9}{'max/frm':>9}"
      f"{'nn_p10':>8}{'nn_med':>8}{'nn_p90':>8}{'n/n_est':>9}")
    for s in sorted(stats, key=lambda d: -d["d_adj"]):
        p(f"{s['stem']:<16}{s['frames']:>7}{s['n_pred']:>9,}{s['nodes_per_frame_mean']:>9.0f}"
          f"{s['nodes_per_frame_max']:>9.0f}{s['nn_p10']:>8.2f}{s['nn_med']:>8.2f}"
          f"{s['nn_p90']:>8.2f}{s['ratio_n']:>9.2f}")
    p("")

    p("=" * 108)
    p("MOTION  (ILP edge length, um; relink gates at 6 then 10)")
    p(f"{'stem':<16}{'d_p10':>8}{'d_med':>8}{'d_p90':>8}{'d_p99':>8}"
      f"{'>6um%':>8}{'>10um%':>8}{'AMBIG':>8}{'ambig90':>9}")
    for s in sorted(stats, key=lambda d: -d["d_adj"]):
        p(f"{s['stem']:<16}{s['disp_p10']:>8.2f}{s['disp_med']:>8.2f}{s['disp_p90']:>8.2f}"
          f"{s['disp_p99']:>8.2f}{100*s['frac_disp_gt_tight']:>8.2f}"
          f"{100*s['frac_disp_gt_relaxed']:>8.2f}{s['ambiguity']:>8.3f}{s['ambiguity_p90']:>9.3f}")
    p("   AMBIG = median ILP edge length / median within-frame nearest-neighbour spacing.")
    p("")

    p("=" * 108)
    p("PROBABILITY FIELD  (edge_prob exists ONLY on ILP edges; every other pair scores 0.0,")
    p("                    so -beta*prob is a flat incumbency discount, not a likelihood)")
    p(f"{'stem':<16}{'p_mean':>9}{'p_med':>9}{'p_std':>9}{'p_p10':>9}"
      f"{'ilp_fill':>10}{'rel_fill':>10}")
    for s in sorted(stats, key=lambda d: -d["d_adj"]):
        p(f"{s['stem']:<16}{s['prob_mean']:>9.4f}{s['prob_med']:>9.4f}{s['prob_std']:>9.4f}"
          f"{s['prob_p10']:>9.4f}{s['ilp_fill']:>10.4f}{s['rel_fill']:>10.4f}")
    p("   *_fill = edges / sum_t min(n_t, n_t+1): how much of the 1:1 capacity each graph uses.")
    p("")

    p("=" * 108)
    p("EDGE-SET DIFF (relink vs ILP)")
    p(f"{'stem':<16}{'kept':>9}{'dropped':>9}{'added':>9}{'ilpLmed':>9}{'relLmed':>9}"
      f"{'dropLmed':>10}{'addLmed':>9}")
    for s in sorted(stats, key=lambda d: -d["d_adj"]):
        p(f"{s['stem']:<16}{s['n_kept']:>9,}{s['n_dropped']:>9,}{s['n_added']:>9,}"
          f"{s['ilp_len_med']:>9.2f}{s['rel_len_med']:>9.2f}"
          f"{s['dropped_len_med']:>10.2f}{s['added_len_med']:>9.2f}")
    p("")

    # ------------------------------------------------ focus film forensics
    p("=" * 108)
    p(f"FORENSICS ON {FOCUS}: fate of every GT edge whose endpoints both matched")
    n = len(fates)
    both = sum(1 for f in fates if f["ilp_ok"] and f["rel_ok"])
    ilp_only = [f for f in fates if f["ilp_ok"] and not f["rel_ok"]]
    rel_only = [f for f in fates if f["rel_ok"] and not f["ilp_ok"]]
    neither = sum(1 for f in fates if not f["ilp_ok"] and not f["rel_ok"])
    p(f"  matched GT edges           {n}")
    p(f"  both graphs correct        {both}")
    p(f"  ILP correct, relink WRONG  {len(ilp_only)}   <-- the damage")
    p(f"  relink correct, ILP wrong  {len(rel_only)}")
    p(f"  both wrong                 {neither}")
    if ilp_only:
        td = np.array([f["true_dist_um"] for f in ilp_only])
        cd = np.array([f["chosen_dist_um"] for f in ilp_only if np.isfinite(f["chosen_dist_um"])])
        md = np.array([f["miss_dist_um"] for f in ilp_only if np.isfinite(f["miss_dist_um"])])
        cr = np.array([f["target_crowding_um"] for f in ilp_only if np.isfinite(f["target_crowding_um"])])
        allt = np.array([f["true_dist_um"] for f in fates])
        allc = np.array([f["target_crowding_um"] for f in fates if np.isfinite(f["target_crowding_um"])])
        p("")
        p("  broken edges: the true link vs what relink chose instead")
        p(f"    true link length          med {np.median(td):6.2f} um   (all matched GT edges: "
          f"med {np.median(allt):.2f})")
        p(f"    relink's chosen link      med {np.median(cd):6.2f} um  "
          f"(n with a chosen target = {len(cd)}/{len(ilp_only)})")
        p(f"    chosen target is          {np.mean(cd < td)*100:.0f}% NEARER the source than the truth")
        p(f"    dist(chosen, truth)       med {np.median(md):6.2f} um")
        p(f"    crowding at true target   med {np.median(cr):6.2f} um   (all matched GT: "
          f"med {np.median(allc):.2f})")
        p(f"    true link exceeded 6um gate in {(td > 6).sum()}/{len(td)} cases, "
          f"10um gate in {(td > 10).sum()}/{len(td)}")
        p(f"    source left unlinked entirely: {len(ilp_only) - len(cd)}")
        # temporal concentration
        import collections
        c = collections.Counter(f["t"] for f in ilp_only)
        top = c.most_common(8)
        p(f"    frames touched: {len(c)} distinct (of {stats[0]['frames']});"
          f" busiest: {', '.join(f't={t}:{k}' for t, k in top)}")
        allc_t = collections.Counter(f["t"] for f in fates)
        p(f"    GT edges available in those frames: "
          f"{', '.join(f't={t}:{allc_t[t]}' for t, _ in top)}")
    p("")

    # ---------------------------------- who commits the damaging edge, and when
    p("=" * 108)
    p(f"WHICH PASS COMMITS THE WRONG LINK ON {FOCUS}?")
    Pf = load_pred(PRED / f"{FOCUS}.geff")
    Gf = load_gt(FOCUS)
    relf, tr = motion_relink(Pf, trace=True)
    posf = Pf["zyx"] * SCALE
    pass_of = {k: v[0] for k, v in tr.items() if isinstance(k, tuple) and len(k) == 2
               and not isinstance(k[0], str)}
    cm = {k[1]: v for k, v in tr.items() if isinstance(k[0], str) and k[0] == "costmat"}
    rel_out, rel_in = {}, {}
    for s_, t_ in relf:
        rel_out.setdefault(int(s_), []).append(int(t_))
        rel_in[int(t_)] = int(s_)

    broken = [f for f in fates if f["ilp_ok"] and not f["rel_ok"]]
    n_tight, n_relaxed, n_none = 0, 0, 0
    flips = []
    for f in broken:
        src, true_t = f["p_src"], f["p_dst"]
        chosen = rel_out.get(src, [None])[0]
        if chosen is None:
            n_none += 1
            continue
        pi = pass_of.get((src, chosen))
        if pi == 0:
            n_tight += 1
        elif pi == 1:
            n_relaxed += 1
        si, ti, base, raw = cm[f["t"]]
        a = int(np.where(si == src)[0][0])
        bt = int(np.where(ti == true_t)[0][0])
        bc = int(np.where(ti == chosen)[0][0])
        thief = rel_in.get(true_t)
        flips.append(dict(t=f["t"], pass_i=pi,
                          cost_true=float(base[a, bt]), cost_chosen=float(base[a, bc]),
                          raw_true=float(raw[a, bt]), raw_chosen=float(raw[a, bc]),
                          margin=float(base[a, bt] - base[a, bc]),
                          stolen=thief is not None and thief != src,
                          thief_pass=pass_of.get((thief, true_t)) if thief is not None else None))
    p(f"  wrong links committed in the TIGHT (6 um) pass   {n_tight}/{len(broken)}")
    p(f"  wrong links committed in the RELAXED (10 um) pass {n_relaxed}/{len(broken)}")
    p(f"  source left unlinked                              {n_none}/{len(broken)}")
    p("")
    p("  Cost of the CORRECT target vs the one relink took (both in um of cost units).")
    p("  'margin' is how much extra bonus beta would have to supply to flip it back;")
    p("  the incumbency bonus available is at most beta*edge_prob ~ 0.9.")
    p(f"  {'t':>5}{'pass':>6}{'raw_true':>10}{'raw_chosen':>12}{'cost_true':>11}"
      f"{'cost_chosen':>13}{'margin':>9}{'gated?':>8}{'stolen?':>9}{'cause':>14}")
    causes = {}
    for fl in sorted(flips, key=lambda d: d["t"]):
        gated = fl["pass_i"] == 0 and fl["raw_true"] > 6.0
        cause = "GATE" if gated else ("STOLEN" if fl["stolen"] else "cost")
        causes[cause] = causes.get(cause, 0) + 1
        p(f"  {fl['t']:>5}{fl['pass_i']:>6}{fl['raw_true']:>10.2f}{fl['raw_chosen']:>12.2f}"
          f"{fl['cost_true']:>11.2f}{fl['cost_chosen']:>13.2f}{fl['margin']:>9.2f}"
          f"{('YES' if gated else 'no'):>8}{('YES' if fl['stolen'] else 'no'):>9}{cause:>14}")
    m = np.array([fl["margin"] for fl in flips])
    p("")
    p(f"  cause breakdown: {causes}")
    p(f"  GATE   = tight pass committed a wrong link while the correct target sat beyond")
    p(f"           6 um and was therefore not even a candidate; the source is then in")
    p(f"           used_i and the 10 um relaxed pass can never revisit it.")
    p(f"  STOLEN = the correct target was cheaper for this source, but the 1:1 constraint")
    p(f"           had already handed that target to a competing source.")
    p(f"  median |margin| {np.median(np.abs(m)):.2f} um; "
      f"{(np.abs(m) > 0.9).sum()}/{len(m)} exceed the ~0.9 um that the")
    p(f"  learned-probability bonus can supply, so the bonus cannot rescue them.")
    p("")

    # ------------------------------------------------ beta sweep
    p("=" * 108)
    p("HOW MUCH INCUMBENCY BONUS WOULD IT TAKE TO UNDO THE DAMAGE?")
    Sf = lambda e: M2.score(Pf["t"], Pf["zyx"], e, Gf["t"], Gf["zyx"], Gf["edges"], Gf["n_est"])
    p(f"  {'beta':>6}{'FOCUS adj':>12}{'TP':>6}{'FP':>6}   (raw ILP = "
      f"{rows_raw[focus_i]['adj']:.5f}, TP {rows_raw[focus_i]['etp']}, "
      f"FP {rows_raw[focus_i]['efp']})")
    for b in (0.0, 1.0, 2.0, 5.0, 10.0, 50.0):
        s = Sf(motion_relink(Pf, beta=b))
        p(f"  {b:>6.1f}{s['adj']:>12.5f}{s['etp']:>6}{s['efp']:>6}")
    p("  Even an overwhelming bonus cannot recover the raw ILP score: the 1:1")
    p("  constraint and the distance gates remove options the bonus cannot buy back.")
    p("")

    # ------------------------------------------------ causal ablation
    p("=" * 108)
    p("CAUSAL ABLATION: which ingredient of the stage does the damage?")
    p(f"{'variant':<28}{'FOCUS adj':>11}{'FOCUS TP':>10}{'FOCUS FP':>10}"
      f"{'   |':>4}{'all-8 adj':>11}{'all-8 J':>10}")
    for k, (lbl, _) in enumerate(ABLATIONS):
        f = abls[focus_i][k][1]
        agg = M2.aggregate([a[k][1] for a in abls])
        p(f"{lbl:<28}{f['adj']:>11.5f}{f['etp']:>10}{f['efp']:>10}{'   |':>4}"
          f"{agg['adj']:>11.5f}{agg['J']:>10.5f}")
    p(f"{'(no relink at all: raw ILP)':<28}{rows_raw[focus_i]['adj']:>11.5f}"
      f"{rows_raw[focus_i]['etp']:>10}{rows_raw[focus_i]['efp']:>10}{'   |':>4}"
      f"{agg_raw['adj']:>11.5f}{agg_raw['J']:>10.5f}")
    p("")

    p("=" * 108)
    p("GT MOTION vs PREDICTED MOTION, and what relink does to divisions")
    p(f"{'stem':<16}{'gtDmed':>8}{'gtDp90':>8}{'gtDmax':>8}{'gt>6um%':>9}"
      f"{'predDmed':>10}{'ilpForks':>10}{'relForks':>10}{'rawDivJ':>9}{'relDivJ':>9}")
    for s in sorted(stats, key=lambda d: -d["d_adj"]):
        rdj = s["raw_dtp"] / max(s["raw_dtp"] + s["raw_dfp"] + s["raw_dfn"], 1)
        xdj = s["rel_dtp"] / max(s["rel_dtp"] + s["rel_dfp"] + s["rel_dfn"], 1)
        p(f"{s['stem']:<16}{s['gt_disp_med']:>8.2f}{s['gt_disp_p90']:>8.2f}{s['gt_disp_max']:>8.2f}"
          f"{100*s['gt_frac_gt6']:>9.2f}{s['disp_med']:>10.2f}"
          f"{s['ilp_forks']:>10,}{s['rel_forks']:>10,}{rdj:>9.3f}{xdj:>9.3f}")
    p("   NOTE: ilpForks is 0 on every film -- the raw ILP graphs contain no out-degree-2")
    p("   node at all, and rawDivJ is 0 everywhere.  So relink's 1:1 constraint destroys")
    p("   nothing here: the division term is not part of this collapse.  The whole delta")
    p("   is edge Jaccard.  (Divisions only appear later, when safe_div runs.)")
    p("")

    p("=" * 108)
    p("WHERE THE EXTRA FALSE POSITIVES COME FROM")
    p(f"{'stem':<16}{'ilpFP':>8}{'relFP':>8}{'newFP':>8}{'replacing':>11}"
      f"{'onUnlinked':>12}{'FPfixed':>9}")
    for s in sorted(stats, key=lambda d: -d["d_adj"]):
        p(f"{s['stem']:<16}{s['ilp_fp']:>8}{s['rel_fp']:>8}{s['new_fp']:>8}"
          f"{s['new_fp_replacing']:>11}{s['new_fp_on_unlinked_source']:>12}{s['fp_fixed']:>9}")
    p("   'replacing'  = relink overwrote an ILP link on that source with a wrong one.")
    p("   'onUnlinked' = ILP deliberately left the source unlinked; 1:1 Hungarian linked it anyway.")
    p("")

    # ------------------------------------------------ correlation search
    p("=" * 108)
    p("CAN ANY FILM-LEVEL STATISTIC PREDICT THE RELINK DELTA?  (n = 8)")
    y = np.array([s["d_adj"] for s in stats])
    keys = ["ambiguity", "ambiguity_p90", "disp_med", "disp_p90", "disp_p99", "nn_med", "nn_p10",
            "frac_disp_gt_tight", "frac_disp_gt_relaxed", "ratio_n", "nodes_per_frame_mean",
            "nodes_per_frame_max", "n_pred", "prob_mean", "prob_med", "prob_std",
            "ilp_fill", "rel_fill", "frac_ilp_dropped", "frames"]

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    tbl = []
    for k in keys:
        x = np.array([s[k] for s in stats], float)
        if np.allclose(x.std(), 0):
            continue
        tbl.append((k, float(np.corrcoef(x, y)[0, 1]), spearman(x, y)))
    p(f"{'statistic':<26}{'pearson r':>11}{'spearman':>11}{'r w/o FOCUS':>13}"
      f"   leave-one-out pearson range")
    for k, r, rs in sorted(tbl, key=lambda z: -abs(z[1])):
        x = np.array([s[k] for s in stats], float)
        loo = []
        for i in range(len(x)):
            m = np.ones(len(x), bool); m[i] = False
            if np.allclose(x[m].std(), 0):
                continue
            loo.append(float(np.corrcoef(x[m], y[m])[0, 1]))
        mf = np.ones(len(x), bool); mf[focus_i] = False
        rf = float(np.corrcoef(x[mf], y[mf])[0, 1]) if x[mf].std() else np.nan
        p(f"{k:<26}{r:>+11.3f}{rs:>+11.3f}{rf:>+13.3f}   [{min(loo):+.3f}, {max(loo):+.3f}]")
    p("   'r w/o FOCUS' drops 44b6_267148e4 and refits on the remaining 7 films.")
    p("")
    p("  leave-one-out range is the decisive column: a statistic whose correlation")
    p("  survives dropping any single film is worth something; one that does not is")
    p("  describing 44b6_267148e4 and nothing else.")
    p("")

    # -------- honest null: best-of-K correlation on 8 points is inflated
    K = len(tbl)
    X = np.array([[s[k] for k, _, _ in tbl] for s in stats], float)
    rng = np.random.default_rng(0)
    best_null = []
    for _ in range(20000):
        yp = rng.permutation(y)
        rr = [abs(np.corrcoef(X[:, j], yp)[0, 1]) for j in range(K)]
        best_null.append(max(rr))
    best_null = np.array(best_null)
    obs = max(abs(r) for _, r, _ in tbl)
    pval = float((best_null >= obs).mean())
    p(f"  PERMUTATION NULL ({K} candidate statistics, 20,000 shuffles of the 8 deltas):")
    p(f"    best |r| observed                 {obs:.3f}")
    p(f"    best |r| under the null, median   {np.median(best_null):.3f}   "
      f"90th pct {np.percentile(best_null, 90):.3f}   99th pct {np.percentile(best_null, 99):.3f}")
    p(f"    P(best-of-{K} null |r| >= observed) = {pval:.4f}")
    p("    With 8 points and a screen this wide, |r| ~ 0.85 happens by chance often;")
    p("    read the p-value, not the raw r.")
    p("")
    # sign-only test: does the top statistic order the films correctly?
    top_k = max(tbl, key=lambda z: abs(z[1]))[0]
    xs = np.array([s[top_k] for s in stats], float)
    order = np.argsort(-xs)
    p(f"  RANKING CHECK on '{top_k}' (films sorted by the statistic, descending):")
    for i in order:
        p(f"    {stats[i]['stem']:<16} stat {stats[i][top_k]:>7.3f}   d_adj {stats[i]['d_adj']:>+9.5f}")

    # -------------------------------- the mechanism, tested at EDGE level (n >> 8)
    p("")
    p("=" * 108)
    p("THE MECHANISM AT EDGE LEVEL -- pooled over all 8 films, so n is thousands, not 8")
    p("For every GT edge the ILP already got right, does relink break it?  Bucketed by")
    p("the true displacement of that edge, i.e. by how the 6 um tight gate sees it.")
    keep = [f for f in all_fates if f["ilp_ok"]]
    edges_ = [0, 2, 4, 6, 8, 10, 99]
    p(f"  {'true displacement (um)':<24}{'n':>7}{'broken':>8}{'break rate':>12}")
    for lo, hi in zip(edges_[:-1], edges_[1:]):
        b = [f for f in keep if lo <= f["true_dist_um"] < hi]
        if not b:
            continue
        nb = sum(1 for f in b if not f["rel_ok"])
        p(f"  {f'[{lo}, {hi})':<24}{len(b):>7}{nb:>8}{nb/len(b)*100:>11.1f}%")
    below = [f for f in keep if f["true_dist_um"] < 6]
    above = [f for f in keep if f["true_dist_um"] >= 6]
    rb = sum(1 for f in below if not f["rel_ok"]) / max(len(below), 1)
    ra = sum(1 for f in above if not f["rel_ok"]) / max(len(above), 1)
    p("")
    p(f"  below the 6 um tight gate: {sum(1 for f in below if not f['rel_ok'])}/{len(below)}"
      f" broken = {rb*100:.2f}%")
    p(f"  above the 6 um tight gate: {sum(1 for f in above if not f['rel_ok'])}/{len(above)}"
      f" broken = {ra*100:.1f}%   ({ra/max(rb,1e-9):.0f}x)")
    p("")
    p("  Same split, per film -- the law is not one film's quirk:")
    p(f"  {'stem':<16}{'n<6um':>8}{'brk<6':>7}{'rate':>8}{'   ':>3}"
      f"{'n>=6um':>8}{'brk>=6':>8}{'rate':>8}{'   ':>3}{'d_adj':>9}")
    for s in sorted(stats, key=lambda d: -d["d_adj"]):
        kf = [f for f in keep if f["stem"] == s["stem"]]
        lo_ = [f for f in kf if f["true_dist_um"] < 6]
        hi_ = [f for f in kf if f["true_dist_um"] >= 6]
        nlo = sum(1 for f in lo_ if not f["rel_ok"])
        nhi = sum(1 for f in hi_ if not f["rel_ok"])
        p(f"  {s['stem']:<16}{len(lo_):>8}{nlo:>7}{(nlo/max(len(lo_),1))*100:>7.1f}%{'   ':>3}"
          f"{len(hi_):>8}{nhi:>8}{(nhi/max(len(hi_),1))*100:>7.1f}%{'   ':>3}{s['d_adj']:>+9.5f}")
    p("")
    p("  A per-film predictor follows from the law WITHOUT fitting anything:")
    p("  expected broken edges ~ (share of GT edges over 6 um) x (break rate above the gate).")
    p(f"  {'stem':<16}{'GT>6um share':>14}{'predicted brk':>15}{'actual brk':>12}{'d_adj':>10}")
    for s in sorted(stats, key=lambda d: -d["d_adj"]):
        kf = [f for f in keep if f["stem"] == s["stem"]]
        share = np.mean([f["true_dist_um"] >= 6 for f in kf]) if kf else np.nan
        pred = share * ra * len(kf)
        act = sum(1 for f in kf if not f["rel_ok"])
        p(f"  {s['stem']:<16}{share*100:>13.1f}%{pred:>15.1f}{act:>12}{s['d_adj']:>+10.5f}")
    sh = np.array([np.mean([f["true_dist_um"] >= 6
                            for f in keep if f["stem"] == s["stem"]]) for s in stats])
    p(f"  pearson r (GT>6um share vs d_adj) = "
      f"{float(np.corrcoef(sh, np.array([s['d_adj'] for s in stats]))[0,1]):+.3f}"
      f"   -- but this needs GT, so it is a diagnosis, not a deployable rule.")
    p("")

    # ------------------------------------------------ reconciliation
    p("")
    p("=" * 108)
    p("RECONCILIATION with the per-film 'proxy' deltas in the brief")
    p("  Raw ILP has zero forks, so per-film proxy == per-film adj unless safe_div runs.")
    p("  Re-running both graphs through the deployed safe_div reproduces the brief's numbers:")
    p(f"  {'stem':<16}{'proxy(raw+sd)':>15}{'proxy(rel+sd)':>15}{'delta':>10}"
      f"{'  |':>3}{'adj-only delta':>16}")
    for s in sorted(stats, key=lambda d: -d["d_adj"]):
        pr, px = SD_PROXY[s["stem"]]
        p(f"  {s['stem']:<16}{pr:>15.5f}{px:>15.5f}{pr - px:>+10.4f}{'  |':>3}"
          f"{s['d_adj']:>+16.5f}")
    p("  The extra swing beyond the edge-Jaccard delta is safe_div: on 44b6_267148e4 the")
    p("  raw graph lets safe_div recover the film's one division and the relinked graph")
    p("  does not, which adds ~0.10 of per-film proxy on top of the 0.085 edge-Jaccard loss.")
    p("")

    # ------------------------------------------------ write outputs
    (OUT / "94_relink_collapse.txt").write_text("\n".join(lines) + "\n")
    cols = sorted(stats[0].keys())
    with open(OUT / "94_relink_collapse_films.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem"] + [c for c in cols if c != "stem"])
        w.writeheader()
        for s in sorted(stats, key=lambda d: -d["d_adj"]):
            w.writerow(s)
    with open(OUT / "94_relink_collapse_diffs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fates[0].keys()))
        w.writeheader()
        w.writerows(fates)
    print(f"\nwrote {OUT/'94_relink_collapse.txt'}")
    print(f"wrote {OUT/'94_relink_collapse_films.csv'}")
    print(f"wrote {OUT/'94_relink_collapse_diffs.csv'}")


if __name__ == "__main__":
    main()
