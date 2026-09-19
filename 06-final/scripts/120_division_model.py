"""A division classifier trained on SYNTHETIC data, evaluated on REAL divisions.

WHY THIS AND NOT ANYTHING ELSE (docs/HANDOFF.md, The Division Lever):
the gap to first place is a division problem. proxy = adj_edge + 0.1*divJ, we sit
at divJ 0.2308, and 0.0769 of the metric is simply unclaimed. Reaching 0.975 by
edges needs +0.026 against an oracle ceiling of +0.049 for eliminating EVERY edge
error. Reaching it by divisions needs divJ 0.23 -> 0.49.

WHY IT FAILED BEFORE: our classifier hit AUC 0.456 -- chance -- fit to ~304 real
division events on frozen features. A data problem, not a method problem. The CC0
synthetic set has ~160,000 time-resolved divisions, ~526x more.

WHY THE PATCHES TRANSFER: synthetic volumes are (64,64,64) uint16 at 1.625um
ISOTROPIC; real frames pooled the way the evaluator pools them (vol[:, ::4, ::4])
are also (64,64,64) uint16 at 1.625um isotropic. Same grid, same dtype, no
rescaling. The generator was built to match.

THE TEST, and it is deliberately a kill-test: does a model that has never seen a
single real label beat AUC 0.70 on the real GT divisions? Our frozen-feature
attempt managed 0.456. This evaluation is also the cleanest measurement in the
project -- no training labels are involved on the real side at all, so there is
no contamination argument to have.
"""
import sys, time, glob, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

SYN = ROOT / "artifacts/synthetic/biohub_synthetic/sequences"
P = 16                      # patch edge in voxels; 16*1.625 = 26um, and sister
                            # separation is 7.24um, so a split fits comfortably
NEG_PER_POS = 3


def crop(vol, zyx, p=P):
    """Centred crop with zero padding at the border."""
    z, y, x = [int(round(v)) for v in zyx]
    h = p // 2
    out = np.zeros((p, p, p), np.float32)
    z0, z1 = max(0, z - h), min(vol.shape[0], z - h + p)
    y0, y1 = max(0, y - h), min(vol.shape[1], y - h + p)
    x0, x1 = max(0, x - h), min(vol.shape[2], x - h + p)
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return out
    out[z0 - (z - h):z1 - (z - h), y0 - (y - h):y1 - (y - h),
        x0 - (x - h):x1 - (x - h)] = vol[z0:z1, y0:y1, x0:x1]
    return out


def norm(a):
    """Per-patch normalisation -- the ONLY defence against the intensity domain
    gap between a synthetic generator and a real microscope."""
    m, s = float(a.mean()), float(a.std())
    return (a - m) / (s + 1e-6)


def synth_patches(files, cap=None):
    """(N,2,P,P,P) float32 and labels. Channel 0 = frame t, channel 1 = t+1."""
    X, Y = [], []
    rng = random.Random(0)
    for f in files:
        d = np.load(f, allow_pickle=True)
        vols, nodes, divs = d["volumes"], d["nodes"], set(d["divisions"].tolist())
        t = nodes[:, 0].astype(int)
        pos = [i for i in range(len(nodes)) if i in divs and t[i] + 1 < vols.shape[0]]
        neg_pool = [i for i in range(len(nodes))
                    if i not in divs and t[i] + 1 < vols.shape[0]]
        neg = rng.sample(neg_pool, min(len(pos) * NEG_PER_POS, len(neg_pool)))
        for i in pos + neg:
            a = crop(vols[t[i]], nodes[i, 1:4])
            b = crop(vols[t[i] + 1], nodes[i, 1:4])
            X.append(np.stack([norm(a), norm(b)]))
            Y.append(1 if i in divs else 0)
        if cap and len(Y) >= cap:
            break
    return np.asarray(X, np.float32), np.asarray(Y, np.int64)


def real_patches():
    """Real GT divisions vs real GT non-dividing nodes, from the pooled volumes.

    No model has ever been fit to these labels for this task, so this is an
    honest test set in a way nothing else in this project has been.
    """
    from biohub import io
    X, Y, films = [], [], 0
    for gp in sorted((io.dataset_root() / "train").glob("*.geff")):
        stem = gp.stem
        g = io.read_geff(gp)
        idx = {int(i): k for k, i in enumerate(g["ids"])}
        out = {}
        for a, b in g["edges"]:
            out.setdefault(idx[int(a)], []).append(idx[int(b)])
        gt_t = g["t"].astype(int)
        gzyx = np.stack([g["z"], g["y"], g["x"]], 1).astype(float)
        divs = [n for n, ch in out.items() if len(ch) >= 2]
        ones = [n for n, ch in out.items() if len(ch) == 1]
        if not divs:
            continue
        rng = random.Random(hash(stem) & 0xFFFF)
        negs = rng.sample(ones, min(len(divs) * NEG_PER_POS, len(ones)))
        need = sorted({gt_t[n] for n in divs + negs} |
                      {gt_t[n] + 1 for n in divs + negs})
        zarr_path = io.dataset_root() / "train" / f"{stem}.zarr"
        frames = {}
        for fr in need:
            try:
                frames[fr] = io.read_frame(zarr_path, fr)[:, ::4, ::4]
            except Exception:
                pass
        for n in divs + negs:
            t0, t1 = gt_t[n], gt_t[n] + 1
            if t0 not in frames or t1 not in frames:
                continue
            a = crop(frames[t0], gzyx[n])
            b = crop(frames[t1], gzyx[n])
            X.append(np.stack([norm(a), norm(b)]))
            Y.append(1 if n in set(divs) else 0)
        films += 1
    print(f"  real eval set: {films} films with divisions, "
          f"{int(sum(Y))} positives, {len(Y) - int(sum(Y))} negatives")
    return np.asarray(X, np.float32), np.asarray(Y, np.int64)


def auc(scores, labels):
    s, l = np.asarray(scores), np.asarray(labels)
    pos, neg = s[l == 1], s[l == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


if __name__ == "__main__":
    print(__doc__.strip())
    print("=" * 88)
    t0 = time.time()
    files = sorted(glob.glob(str(SYN / "*.npz")))
    # hikaggler found 497 sequences beat all 2,174 for pretraining -- more is
    # not better here, and it keeps the build fast.
    files = files[:600]
    print(f"\nbuilding synthetic patches from {len(files)} sequences ...")
    Xs, Ys = synth_patches(files)
    print(f"  {len(Ys):,} patches, {int(Ys.sum()):,} positive "
          f"({100*Ys.mean():.1f}%), shape {Xs.shape[1:]}  [{time.time()-t0:.0f}s]")
    np.savez_compressed(ROOT / "artifacts/div_synth_patches.npz", X=Xs, Y=Ys)
    print(f"  cached -> artifacts/div_synth_patches.npz")

    print("\nbuilding REAL evaluation patches ...")
    t1 = time.time()
    Xr, Yr = real_patches()
    np.savez_compressed(ROOT / "artifacts/div_real_patches.npz", X=Xr, Y=Yr)
    print(f"  cached -> artifacts/div_real_patches.npz  [{time.time()-t1:.0f}s]")
