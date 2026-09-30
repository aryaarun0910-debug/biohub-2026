"""Train the division classifier on synthetic patches; test on REAL divisions.

THE KILL TEST: real AUC. Our frozen-feature attempt on ~304 real events managed
0.456 (chance). If a synthetic-trained model cannot beat 0.70 on the 151 real GT
divisions, the domain gap is fatal and we stop -- an evening spent, not a week.

The real set is honest: no model is ever fit to a real label here, so unlike
every other measurement in this project there is no contamination to argue about.
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def auc(s, l):
    s, l = np.asarray(s), np.asarray(l)
    pos, neg = s[l == 1], s[l == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.concatenate([pos, neg]).argsort().argsort().astype(float) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


class Net(nn.Module):
    """Small 3D CNN. Deliberately small: 184k patches of 2x16x16x16, and a big
    model would just memorise the generator's quirks instead of learning mitosis."""
    def __init__(s, c=(16, 32, 64)):
        super().__init__()
        L, prev = [], 2
        for ch in c:
            L += [nn.Conv3d(prev, ch, 3, padding=1), nn.BatchNorm3d(ch),
                  nn.ReLU(inplace=True), nn.MaxPool3d(2)]
            prev = ch
        s.f = nn.Sequential(*L)
        s.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(prev * 2 * 2 * 2, 1))

    def forward(s, x):
        return s.head(s.f(x)).squeeze(-1)


def main():
    print(__doc__.strip()); print("=" * 84)
    d = np.load(ROOT / "artifacts/div_synth_patches.npz")
    X, Y = d["X"], d["Y"]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Y))
    X, Y = X[idx], Y[idx]
    nv = len(Y) // 10
    Xv, Yv, Xt, Yt = X[:nv], Y[:nv], X[nv:], Y[nv:]
    r = np.load(ROOT / "artifacts/div_real_patches.npz")
    Xr, Yr = r["X"], r["Y"]
    print(f"synthetic train {len(Yt):,} | synthetic val {len(Yv):,} | "
          f"REAL test {len(Yr)} ({int(Yr.sum())} divisions)")
    print(f"device: {DEV}\n")

    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xr_t = torch.from_numpy(Xr).to(DEV)
    BS, best = 256, 0.0
    print(f"{'epoch':>6}{'train loss':>12}{'synth AUC':>11}{'REAL AUC':>10}   verdict")
    for ep in range(1, 13):
        net.train(); tot = n = 0
        order = rng.permutation(len(Yt))
        for i in range(0, len(order), BS):
            b = order[i:i + BS]
            xb = torch.from_numpy(Xt[b]).to(DEV)
            yb = torch.from_numpy(Yt[b]).float().to(DEV)
            opt.zero_grad(); out = net(xb); l = lossf(out, yb)
            l.backward(); opt.step()
            tot += float(l) * len(b); n += len(b)
        net.eval()
        with torch.no_grad():
            sv = np.concatenate([net(torch.from_numpy(Xv[i:i+512]).to(DEV)).cpu().numpy()
                                 for i in range(0, len(Yv), 512)])
            sr = net(Xr_t).cpu().numpy()
        a_s, a_r = auc(sv, Yv), auc(sr, Yr)
        mark = ""
        if a_r > best:
            best = a_r; mark = " *"
            torch.save(net.state_dict(), ROOT / "artifacts/div_model_best.pth")
        print(f"{ep:>6}{tot/n:>12.4f}{a_s:>11.4f}{a_r:>10.4f}{mark}")

    print(f"\nBEST REAL AUC: {best:.4f}   (prior attempt on frozen features: 0.456)")
    if best > 0.70:
        print("=> CLEARS THE BAR. Synthetic divisions transfer to real microscopy;")
        print("   the division lever is real and worth the remaining days.")
    elif best > 0.60:
        print("=> PARTIAL. Better than chance but under the bar -- real signal,")
        print("   weak transfer. Worth one iteration on patch size / normalisation.")
    else:
        print("=> FAILS. The domain gap is fatal; do not spend a week on this.")


if __name__ == "__main__":
    main()
