"""Can we get divisions WITHOUT paying motion-relink's edge cost?

On the kernel's own graphs: raw ILP output scores adj 0.93358, the deployed
post-processed output 0.92583. The chain costs 0.0078 of edge Jaccard and
repays it with divisions (unmodified divJ 0.2308 = +0.0231, net +0.0155).

If safe_div can run on the RAW graph, the score becomes 0.9336 + 0.1*divJ.

Caveat on the record: the one published offline-vs-board ledger contains
"turning off a relink stage: +0.0187 offline, board 0.943 -> 0.939". This exact
family of idea has failed on the board before. Local evidence here is necessary
but not sufficient.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from biohub import io, metric2 as M2

PRED = Path("artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0")
SCALE = M2.SCALE


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


def safe_div(P, parent_max=9.0, sister_max=14.0, child_max=10.0, tau=0.6,
             diverge=2.25, mutual_nn=True, frame_cap=0.0076, glob_cap=0.00375):
    """The deployed safe-division rule, on original-voxel coordinates."""
    pos = P["zyx"] * SCALE
    succ, indeg = {}, {}
    for s, t in P["edges"]:
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
            if diverge > 0 or True:
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
    cap = max(1, round(glob_cap * len(P["edges"])))
    return P["edges"] + added[:cap], added[:cap]


stems = sorted(p.stem for p in PRED.glob("*.geff"))
DATA = {s: (load_pred(PRED / f"{s}.geff"), load_gt(s)) for s in stems}
S = lambda e, P, G: M2.score(P["t"], P["zyx"], e, G["t"], G["zyx"], G["edges"], G["n_est"])

print(f"{'config':<34}{'proxy':>9}{'adj':>9}{'J':>9}{'divJ':>8}{'TP':>4}{'FP':>4}{'FN':>4}{'added':>7}")
raw = M2.aggregate([S(P["edges"], P, G) for P, G in DATA.values()])
print(f"{'raw ILP, nothing applied':<34}{raw['proxy']:>9.5f}{raw['adj']:>9.5f}{raw['J']:>9.5f}"
      f"{raw['divJ']:>8.4f}{raw['dtp']:>4}{raw['dfp']:>4}{raw['dfn']:>4}{0:>7}")

for lbl, kw in [("raw + safe_div (deployed gates)", {}),
                ("  diverge 0", dict(diverge=0.0)),
                ("  tau 1.35", dict(tau=1.35)),
                ("  parent 12", dict(parent_max=12.0)),
                ("  global cap 1%", dict(glob_cap=0.01)),
                ("  global cap 0.15%", dict(glob_cap=0.0015)),
                ("  frame cap 0.3%", dict(frame_cap=0.003))]:
    rows, na = [], 0
    for P, G in DATA.values():
        e, a = safe_div(P, **kw); na += len(a); rows.append(S(e, P, G))
    r = M2.aggregate(rows)
    print(f"{lbl:<34}{r['proxy']:>9.5f}{r['adj']:>9.5f}{r['J']:>9.5f}{r['divJ']:>8.4f}"
          f"{r['dtp']:>4}{r['dfp']:>4}{r['dfn']:>4}{na:>7}")
print(f"\ndeployed pipeline on these films (unmodified): proxy 0.9491  adj 0.9260  divJ 0.2308")
