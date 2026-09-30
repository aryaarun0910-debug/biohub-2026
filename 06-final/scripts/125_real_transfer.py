"""Does the synthetic-trained GEOMETRY ranker transfer to REAL divisions?

scripts/124 on synthetic candidate triples:
    deployed GATES      AUC 0.7126   precision 0.171   recall 0.153
    learned GEOM        AUC 0.9233   precision 0.641   recall 0.572
    learned GEOM+IMAGE  AUC 0.9231   -- the image adds -0.0002, i.e. NOTHING

So the gates are the wrong FUNCTIONAL FORM, not badly tuned -- which explains
why section 5's sweep could not improve any threshold in any direction.

Geometry should transfer far better than appearance: the generator calibrates
sister separation (7.24um) and motion (1.86um/frame) against the real lineage
edges, and micrometres are micrometres. This tests it on the 151 REAL GT
divisions, with candidates built exactly as safe_div builds them.

No real label is ever fit here -- the model is trained purely on synthetic -- so
this is an honest test, the same property that made the s08 instrument test
meaningful.
"""
import sys, glob, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch, torch.nn as nn
from biohub import io

UM, R_UM = 1.625, 14.0
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l)
    pos, neg = s[l == 1], s[l == 0]
    if not len(pos) or not len(neg): return float("nan")
    r = np.concatenate([pos, neg]).argsort().argsort().astype(float) + 1
    return (r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

def geom(pp, pa, pb):
    dpa = np.linalg.norm(pa-pp)*UM; dpb = np.linalg.norm(pb-pp)*UM
    dab = np.linalg.norm(pa-pb)*UM; mean = (dpa+dpb)/2 + 1e-6
    va, vb = pa-pp, pb-pp
    cos = float(np.dot(va, vb)/((np.linalg.norm(va)+1e-6)*(np.linalg.norm(vb)+1e-6)))
    return np.array([dpa, dpb, dab, abs(dpa-dpb)/mean, cos,
                     max(dpa, dpb), min(dpa, dpb)], np.float32)

def real_triples():
    G, Y = [], []
    rng = random.Random(0)
    for gp in sorted((io.dataset_root()/"train").glob("*.geff")):
        g = io.read_geff(gp)
        idx = {int(i): k for k, i in enumerate(g["ids"])}
        out = {}
        for a, b in g["edges"]:
            out.setdefault(idx[int(a)], []).append(idx[int(b)])
        t = g["t"].astype(int)
        zyx = np.stack([g["z"], g["y"], g["x"]], 1).astype(float)
        by_t = {}
        for i in range(len(t)): by_t.setdefault(int(t[i]), []).append(i)
        for p, ch in out.items():
            nxt = np.array(by_t.get(int(t[p])+1, []))
            if len(nxt) < 2: continue
            dist = np.linalg.norm(zyx[nxt]-zyx[p], axis=1)*UM
            cand = nxt[dist <= R_UM]
            if len(cand) < 2: continue
            true = set(ch)
            if len(ch) >= 2:
                a, b = sorted(list(true))[:2]
                if a in cand and b in cand:
                    G.append(geom(zyx[p], zyx[a], zyx[b])); Y.append(1)
            others = [(x, y) for ii, x in enumerate(cand) for y in cand[ii+1:]
                      if set((int(x), int(y))) != true]
            rng.shuffle(others)
            for x, y in others[:4]:
                G.append(geom(zyx[p], zyx[x], zyx[y])); Y.append(0)
    return np.asarray(G, np.float32), np.asarray(Y, np.int64)

class M(nn.Module):
    def __init__(s):
        super().__init__()
        s.h = nn.Sequential(nn.Linear(7, 96), nn.ReLU(True), nn.Dropout(0.2),
                            nn.Linear(96, 48), nn.ReLU(True), nn.Linear(48, 1))
    def forward(s, g): return s.h(g).squeeze(-1)

print(__doc__.strip()); print("=" * 84)
d = np.load(ROOT/"artifacts/div_triples.npz"); G, Y = d["G"], d["Y"]
rng = np.random.default_rng(0); idx = rng.permutation(len(Y)); G, Y = G[idx], Y[idx]
mu, sd = G.mean(0), G.std(0)+1e-6
net = M().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=DEV))
for ep in range(12):
    net.train(); o = rng.permutation(len(Y))
    for i in range(0, len(o), 1024):
        b = o[i:i+1024]
        opt.zero_grad()
        l = lf(net(torch.from_numpy((G[b]-mu)/sd).to(DEV)),
               torch.from_numpy(Y[b]).float().to(DEV))
        l.backward(); opt.step()
print("trained on synthetic geometry only\n")

Gr, Yr = real_triples()
print(f"REAL candidate triples: {len(Yr):,}  positives {int(Yr.sum())} "
      f"(the GT divisions)  negatives {len(Yr)-int(Yr.sum()):,}")
net.eval()
with torch.no_grad():
    sr = net(torch.from_numpy((Gr-mu)/sd).to(DEV)).cpu().numpy()
gate = ((Gr[:,5] <= 9.0) & (Gr[:,2] <= 14.0) & (Gr[:,3] <= 0.6)).astype(float)
def pr(sel, name, a):
    tp = int(((sel==1)&(Yr==1)).sum()); fp = int(((sel==1)&(Yr==0)).sum())
    fn = int(((sel==0)&(Yr==1)).sum())
    print(f"{name:<26}{a:>8.4f}{tp:>7}{fp:>7}{fn:>7}"
          f"{tp/max(tp+fp,1):>11.3f}{tp/max(tp+fn,1):>9.3f}")
    return tp+fp
print(f"\n{'ARM (on REAL data)':<26}{'AUC':>8}{'TP':>7}{'FP':>7}{'FN':>7}{'precision':>11}{'recall':>9}")
k = pr(gate, "deployed GATES", auc(gate, Yr))
thr = np.quantile(sr, 1 - k/len(sr))
pr((sr >= thr).astype(int), "learned GEOM (synth)", auc(sr, Yr))
print("\n(matched operating point: both accept the same number of candidates)")

# ---- the rule s10 taught us: a positive point estimate is not enough --------
# Require the bootstrap CI on the DIFFERENCE to exclude zero. s10 shipped on
# +4 edges with CI [-3,+12] and lost on the board; do not repeat it.
def boot_auc_diff(a, b, y, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(n):
        p = rng.choice(pos, len(pos), replace=True)
        q = rng.choice(neg, len(neg), replace=True)
        i = np.concatenate([p, q]); yy = y[i]
        out.append(auc(b[i], yy) - auc(a[i], yy))
    return np.array(out)

d_ = boot_auc_diff(gate, sr, Yr)
lo, hi = np.percentile(d_, [2.5, 97.5])
print(f"\n--- bootstrap, {int(Yr.sum())} positives / {len(Yr)-int(Yr.sum())} negatives ---")
print(f"AUC(learned) - AUC(gates) = {auc(sr,Yr)-auc(gate,Yr):+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
print("=> " + ("CI EXCLUDES ZERO -- the improvement is resolved at this sample size"
               if lo > 0 else
               "CI SPANS ZERO -- NOT resolved. Under the post-s10 rule this does NOT ship."))
