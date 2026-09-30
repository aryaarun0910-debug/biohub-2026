"""Isolate the graph effect from the implementation effect.

"raw + my safe_div" beat "relinked + their safe_div" -- but that changes two
things at once. This runs MY safe_div on BOTH graphs, so the only difference is
whether motion relink ran.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.optimize import linear_sum_assignment
from biohub import metric2 as M2
exec(open("scripts/24_keep_ilp_edges.py").read().split("stems = sorted")[0]
     .split('"""', 2)[2])                      # reuse load_pred/load_gt/safe_div

SCALE = M2.SCALE


def motion_relink(P, tight=6.0, relaxed=10.0, vel=0.5, beta=1.0):
    """The deployed relink: 1:1 Hungarian, tight then relaxed, velocity anchor,
    learned-probability bonus. Replaces the ILP edges entirely."""
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


PRED = Path("artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0")
stems = sorted(p.stem for p in PRED.glob("*.geff"))
DATA = {s: (load_pred(PRED / f"{s}.geff"), load_gt(s)) for s in stems}
S = lambda e, P, G: M2.score(P["t"], P["zyx"], e, G["t"], G["zyx"], G["edges"], G["n_est"])

print(f"{'graph':<26}{'safe_div':<10}{'proxy':>9}{'adj':>9}{'J':>9}{'divJ':>8}"
      f"{'TP':>4}{'FP':>4}{'FN':>4}{'edges':>9}")
for gname in ("raw ILP", "motion-relinked"):
    for sd in (False, True):
        rows, ne = [], 0
        for P, G in DATA.values():
            e = P["edges"] if gname == "raw ILP" else motion_relink(P)
            Q = dict(P); Q["edges"] = e
            if sd:
                e, _ = safe_div(Q)
            ne += len(e)
            rows.append(S(e, P, G))
        r = M2.aggregate(rows)
        print(f"{gname:<26}{str(sd):<10}{r['proxy']:>9.5f}{r['adj']:>9.5f}{r['J']:>9.5f}"
              f"{r['divJ']:>8.4f}{r['dtp']:>4}{r['dfp']:>4}{r['dfn']:>4}{ne:>9,}")
print("\nreference -- the kernel's own pipeline on these same 8 films:")
print("  unmodified   proxy 0.9491  adj 0.9260  divJ 0.2308 (3/1/9)")
print("  diverge=0    proxy 0.93694 adj 0.92583 divJ 0.1111 (3/15/9)")
