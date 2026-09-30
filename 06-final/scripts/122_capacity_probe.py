"""Is the underfit a CAPACITY problem or an ILL-POSED-TASK problem?

scripts/121 plateaued at BCE 0.5636 against a prior-only baseline of 0.5623,
with synthetic AUC (0.65) equal to real AUC (0.66). Transfer is fine; the model
never learned the task on its own data.

Two explanations, and they imply different next steps:
  CAPACITY   -> a bigger net / lower LR fixes it. Cheap.
  ILL-POSED  -> no patch model can win, because a 26um window at t+1 in a dense
                embryo contains ~3 nuclei either way, and the question "did the
                centre nucleus split" is genuinely ambiguous without knowing
                WHICH neighbours are its daughters. Then the fix is to stop
                asking that question and score CANDIDATE TRIPLES instead --
                which is also what safe_div actually needs.

This probe settles it. If synthetic AUC moves well above 0.70 with more capacity,
it was capacity. If it stays ~0.65, the task is ill-posed from a patch.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch, torch.nn as nn

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def auc(s, l):
    s, l = np.asarray(s), np.asarray(l)
    pos, neg = s[l == 1], s[l == 0]
    r = np.concatenate([pos, neg]).argsort().argsort().astype(float) + 1
    return (r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

class Big(nn.Module):
    def __init__(s, c=(32, 64, 128, 256)):
        super().__init__()
        L, prev = [], 2
        for i, ch in enumerate(c):
            L += [nn.Conv3d(prev, ch, 3, padding=1), nn.BatchNorm3d(ch), nn.ReLU(True),
                  nn.Conv3d(ch, ch, 3, padding=1), nn.BatchNorm3d(ch), nn.ReLU(True)]
            if i < 3: L += [nn.MaxPool3d(2)]
            prev = ch
        s.f = nn.Sequential(*L)
        s.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                               nn.Linear(prev, 128), nn.ReLU(True),
                               nn.Dropout(0.2), nn.Linear(128, 1))
    def forward(s, x): return s.head(s.f(x)).squeeze(-1)

d = np.load(ROOT / "artifacts/div_synth_patches.npz")
X, Y = d["X"], d["Y"]
rng = np.random.default_rng(0); idx = rng.permutation(len(Y)); X, Y = X[idx], Y[idx]
nv = len(Y)//10; Xv, Yv, Xt, Yt = X[:nv], Y[:nv], X[nv:], Y[nv:]
r = np.load(ROOT / "artifacts/div_real_patches.npz"); Xr, Yr = r["X"], r["Y"]

print(__doc__.strip()); print("=" * 80)
net = Big().to(DEV)
print(f"parameters: {sum(p.numel() for p in net.parameters()):,} "
      f"(scripts/121 used ~60k)\n")
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
lossf = nn.BCEWithLogitsLoss()
Xr_t = torch.from_numpy(Xr).to(DEV)
print(f"{'epoch':>6}{'train loss':>12}{'synth AUC':>11}{'REAL AUC':>10}")
for ep in range(1, 9):
    net.train(); tot=n=0
    order = rng.permutation(len(Yt))
    for i in range(0, len(order), 256):
        b = order[i:i+256]
        xb = torch.from_numpy(Xt[b]).to(DEV); yb = torch.from_numpy(Yt[b]).float().to(DEV)
        opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
        tot += float(l)*len(b); n += len(b)
    net.eval()
    with torch.no_grad():
        sv = np.concatenate([net(torch.from_numpy(Xv[i:i+512]).to(DEV)).cpu().numpy()
                             for i in range(0, len(Yv), 512)])
        sr = net(Xr_t).cpu().numpy()
    print(f"{ep:>6}{tot/n:>12.4f}{auc(sv,Yv):>11.4f}{auc(sr,Yr):>10.4f}")
print("\nprior-only BCE baseline = 0.5623")
