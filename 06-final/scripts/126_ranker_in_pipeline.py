"""Replace safe_div's GATE BLOCK with the learned ranker, in the real chain.

scripts/125: on real GT triples the ranker beats the gates by +0.1907 AUC,
CI [+0.0907,+0.2740], excluding zero. That was GT-derived triples. Deployment
runs on PREDICTED graphs, so this is the test that counts.

BASE IS THE 0.947 NOTEBOOK, RELINK ON. Today's board said relink OFF is 0.945
and every variant on it scored below baseline, so s05/s08/s10 are abandoned and
the incumbent is the unmodified chain.

ONE CHANGE: inside safe_div, the three geometric gates
    max(d_parent_daughter) <= 9.0 | d_sister <= 14.0 | symmetry <= 0.6
are replaced by `ranker_score >= threshold`, and candidates are ranked by score
instead of (dpq + 0.15*dcq). Everything else -- candidate generation, the
mutual-NN test, the divergence test, the frame and global caps -- is untouched.

Reported on BOTH tiers (the scored films are the instrument that passed the s08
test; the validator films are kept only for contrast).
"""
import sys, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch, torch.nn as nn

exec((ROOT / "scripts/91_other_stages.py").read_text().split("\nALL = []")[0])
_r = (ROOT / "scripts/25_relink_control.py").read_text()
exec(_r.split("def motion_relink", 1)[1].join(["def motion_relink", ""])
     .split("\nstems =")[0].split("\nDATA =")[0].split("\nif __name__")[0])

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
UM = 1.625
TEST_PRED = ROOT / ("artifacts/s05_output/tracking_repo/predictions/unknown/"
                    "unet_transformer/split_0")
SCORED = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]


class M(nn.Module):
    def __init__(s):
        super().__init__()
        s.h = nn.Sequential(nn.Linear(7, 96), nn.ReLU(True), nn.Dropout(0.2),
                            nn.Linear(96, 48), nn.ReLU(True), nn.Linear(48, 1))
    def forward(s, g): return s.h(g).squeeze(-1)


def train_ranker():
    d = np.load(ROOT / "artifacts/div_triples.npz"); G, Y = d["G"], d["Y"]
    rng = np.random.default_rng(0); i = rng.permutation(len(Y)); G, Y = G[i], Y[i]
    mu, sd = G.mean(0), G.std(0) + 1e-6
    net = M().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=DEV))
    for _ in range(12):
        net.train(); o = rng.permutation(len(Y))
        for k in range(0, len(o), 1024):
            b = o[k:k+1024]; opt.zero_grad()
            lf(net(torch.from_numpy((G[b]-mu)/sd).to(DEV)),
               torch.from_numpy(Y[b]).float().to(DEV)).backward(); opt.step()
    net.eval()
    return net, mu, sd


NET, MU, SD = train_ranker()


def geom_batch(P3):
    out = np.empty((len(P3), 7), np.float32)
    for k, (pp, pa, pb) in enumerate(P3):
        dpa = np.linalg.norm(pa-pp); dpb = np.linalg.norm(pb-pp)
        dab = np.linalg.norm(pa-pb); mean = (dpa+dpb)/2 + 1e-6
        va, vb = pa-pp, pb-pp
        cos = float(np.dot(va, vb)/((np.linalg.norm(va)+1e-6)*(np.linalg.norm(vb)+1e-6)))
        out[k] = (dpa, dpb, dab, abs(dpa-dpb)/mean, cos, max(dpa, dpb), min(dpa, dpb))
    return out


def safe_div_learned(P, thr, frame_cap=0.0076, glob_cap=0.00375, child_max=10.0):
    """safe_div with the gate block swapped for the ranker. Same candidate
    generation, same caps, same mutual-NN structure."""
    pos = P["zyx"] * SCALE
    succ, indeg = {}, {}
    for s_, t_ in P["edges"]:
        succ.setdefault(s_, []).append(t_); indeg[t_] = indeg.get(t_, 0) + 1
    by_t = {}
    for i, t in enumerate(P["t"]):
        by_t.setdefault(int(t), []).append(i)
    d = lambda a, b: float(np.linalg.norm(pos[a]-pos[b]))
    added = []
    for t in sorted(by_t):
        nxt = by_t.get(t+1)
        if not nxt: continue
        orph = [j for j in nxt if indeg.get(j, 0) == 0]
        if not orph: continue
        opos = pos[orph]
        trip, meta = [], []
        for Pn in [i for i in by_t[t] if len(succ.get(i, ())) == 1]:
            C = succ[Pn][0]
            if d(Pn, C) > child_max: continue
            Q = orph[int(np.argmin(np.linalg.norm(opos - pos[C], axis=1)))]
            trip.append((pos[Pn], pos[C], pos[Q])); meta.append((Pn, Q))
        if not trip: continue
        with torch.no_grad():
            sc = NET(torch.from_numpy((geom_batch(trip)-MU)/SD).to(DEV)).cpu().numpy()
        cands = [(-sc[k], meta[k][0], meta[k][1]) for k in range(len(meta))
                 if sc[k] >= thr]
        cap = max(1, round(frame_cap * len(by_t[t]))); n = 0
        for _, Pn, Q in sorted(cands):
            if n >= cap or indeg.get(Q, 0) or len(succ.get(Pn, ())) >= 2: continue
            added.append((Pn, Q)); succ.setdefault(Pn, []).append(Q)
            indeg[Q] = 1; n += 1
    cap = max(1, round(glob_cap * len(P["edges"])))
    return P["edges"] + added[:cap]


def chain(G, P, sd_fn):
    G = dict(G, edges=motion_relink(dict(P, edges=G["edges"])))   # BASELINE: relink ON
    G, _ = gap_close(G, max_um=5.0, reuse_um=3.2, allow_synth=True)
    G, _ = gap2(G, max_total=10.2, max_step=4.4)
    Q = dict(P); Q["t"] = G["t"]; Q["zyx"] = G["zyx"]; Q["edges"] = G["edges"]
    G = dict(G, edges=sd_fn(Q))
    G, _ = prune_isolated(G); G, _ = short_track(G, 6, True)
    G, _ = linefit(G, w=0.8, window=2)
    return G


SETS = {"SCORED (4 films)": {s: (load_pred(TEST_PRED/f"{s}.geff"), load_gt(s)) for s in SCORED},
        "VALIDATOR (8)": DATA}

print(__doc__.strip()); print("=" * 96)
for name, data in SETS.items():
    print(f"\n### {name}")
    print(f"{'safe_div variant':<34}{'proxy':>10}{'dproxy':>10}{'divJ':>8}{'TP/FP/FN':>11}")
    base = None
    arms = [("GATES (deployed)", lambda Q: safe_div(Q)[0])]
    arms += [(f"RANKER thr={t:+.1f}", (lambda tt: (lambda Q: safe_div_learned(Q, tt)))(t))
             for t in (2.0, 1.0, 0.0, -1.0)]
    for label, fn in arms:
        rows = []
        for stem, (P, GT) in data.items():
            G = chain(G_of(P), P, fn)
            rows.append(M2.score(G["t"], G["zyx"], G["edges"],
                                 GT["t"], GT["zyx"], GT["edges"], GT["n_est"]))
        r = M2.aggregate(rows)
        if base is None: base = r
        dd = "" if label.startswith("GATES") else f"{r['proxy']-base['proxy']:+.5f}"
        print(f"{label:<34}{r['proxy']:>10.5f}{dd:>10}{r['divJ']:>8.4f}"
              f"{f'{r[chr(100)+chr(116)+chr(112)]}/{r[chr(100)+chr(102)+chr(112)]}/{r[chr(100)+chr(102)+chr(110)]}':>11}")
