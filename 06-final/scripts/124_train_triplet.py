"""Three arms on the same candidate triples: deployed GATES vs learned GEOM vs GEOM+IMAGE.

The gates are the incumbent. safe_div accepts a triple when
    max(d_parent_daughter) <= SAFE_DIV_MAX_UM (9.0)
    d_sister              <= SAFE_DIV_SISTER_MAX_UM (14.0)
    |dpa-dpb|/mean        <= SAFE_DIV_SISTER_SYMMETRY_TAU (0.6)
and section 5's sweep could not improve any of those three in any direction.

So the question is not "can a model do this" but "can a model beat THOSE RULES
on the same inputs". Two ways it might:
  GEOM        same information, better functional form -- the gates are
              axis-aligned thresholds on what is really a joint distribution.
  GEOM+IMAGE  information the gates structurally cannot see.
The gap between the learned arms is the measured value of the image.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch, torch.nn as nn

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l)
    pos, neg = s[l == 1], s[l == 0]
    r = np.concatenate([pos, neg]).argsort().argsort().astype(float) + 1
    return (r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

d = np.load(ROOT / "artifacts/div_triples.npz")
G, I, Y = d["G"], d["I"], d["Y"]
rng = np.random.default_rng(0); idx = rng.permutation(len(Y))
G, I, Y = G[idx], I[idx], Y[idx]
nv = len(Y)//5
Gv, Iv, Yv = G[:nv], I[:nv], Y[:nv]
Gt, It, Yt = G[nv:], I[nv:], Y[nv:]
print(__doc__.strip()); print("=" * 84)
print(f"train {len(Yt):,} | val {len(Yv):,} | {100*Y.mean():.1f}% positive\n")

# ---- arm 0: the deployed gates, as a rule -------------------------------
# G columns: [dpa, dpb, dab, sym, cos, max, min]
gate = ((Gv[:, 5] <= 9.0) & (Gv[:, 2] <= 14.0) & (Gv[:, 3] <= 0.6)).astype(float)
tp = int(((gate == 1) & (Yv == 1)).sum()); fp = int(((gate == 1) & (Yv == 0)).sum())
fn = int(((gate == 0) & (Yv == 1)).sum())
print(f"{'ARM':<26}{'AUC':>8}{'TP':>7}{'FP':>7}{'FN':>7}{'precision':>11}{'recall':>9}")
print(f"{'0 deployed GATES (rule)':<26}{auc(gate, Yv):>8.4f}{tp:>7}{fp:>7}{fn:>7}"
      f"{tp/max(tp+fp,1):>11.3f}{tp/max(tp+fn,1):>9.3f}")

mu, sd = Gt.mean(0), Gt.std(0) + 1e-6

def run(name, use_img, epochs=6):
    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.use = use_img
            if use_img:
                s.c = nn.Sequential(
                    nn.Conv3d(3, 24, 3, padding=1), nn.BatchNorm3d(24), nn.ReLU(True),
                    nn.MaxPool3d(2),
                    nn.Conv3d(24, 48, 3, padding=1), nn.BatchNorm3d(48), nn.ReLU(True),
                    nn.AdaptiveAvgPool3d(1), nn.Flatten())
            n = 7 + (48 if use_img else 0)
            s.h = nn.Sequential(nn.Linear(n, 96), nn.ReLU(True), nn.Dropout(0.2),
                                nn.Linear(96, 48), nn.ReLU(True), nn.Linear(48, 1))
        def forward(s, g, im):
            z = s.c(im) if s.use else None
            return s.h(torch.cat([g, z], 1) if s.use else g).squeeze(-1)

    net = M().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=DEV))
    best = 0.0
    for ep in range(epochs):
        net.train(); order = rng.permutation(len(Yt))
        for i in range(0, len(order), 512):
            b = order[i:i+512]
            g = torch.from_numpy((Gt[b]-mu)/sd).to(DEV)
            im = torch.from_numpy(It[b]).to(DEV) if use_img else torch.zeros(1)
            y = torch.from_numpy(Yt[b]).float().to(DEV)
            opt.zero_grad(); l = lf(net(g, im), y); l.backward(); opt.step()
        net.eval(); sc = []
        with torch.no_grad():
            for i in range(0, len(Yv), 1024):
                g = torch.from_numpy((Gv[i:i+1024]-mu)/sd).to(DEV)
                im = torch.from_numpy(Iv[i:i+1024]).to(DEV) if use_img else torch.zeros(1)
                sc.append(net(g, im).cpu().numpy())
        a = auc(np.concatenate(sc), Yv); best = max(best, a)
    sc = np.concatenate(sc)
    # operating point matched to the gates' recall, so precision is comparable
    thr = np.quantile(sc, 1 - (tp+fp)/len(Yv))
    pr = (sc >= thr).astype(int)
    t2 = int(((pr==1)&(Yv==1)).sum()); f2 = int(((pr==1)&(Yv==0)).sum())
    n2 = int(((pr==0)&(Yv==1)).sum())
    print(f"{name:<26}{best:>8.4f}{t2:>7}{f2:>7}{n2:>7}"
          f"{t2/max(t2+f2,1):>11.3f}{t2/max(t2+n2,1):>9.3f}")
    return best

a1 = run("1 learned GEOM", False)
a2 = run("2 learned GEOM+IMAGE", True)
print(f"\nvalue of the image = {a2-a1:+.4f} AUC over geometry alone")
