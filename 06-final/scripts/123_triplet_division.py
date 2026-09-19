"""Score CANDIDATE TRIPLES (parent, daughter_a, daughter_b), not patches.

scripts/121 asked "does this nucleus divide?" from a patch and got AUC 0.65 on
synthetic AND 0.66 on real -- the same, so transfer was never the issue. The
reason is now measured (scripts/122 probe): inside a 16^3 window at t+1 a
dividing parent has on average 1.82 nuclei and a non-dividing one has 1.38. A
+0.44 difference. The patch does not contain a decidable answer, so no capacity
fixes it.

Given the CANDIDATE PAIR the ambiguity disappears: the question stops being "are
there two nuclei nearby" and becomes "is THIS pair a mother-daughter split".
That is also exactly what safe_div needs -- it proposes triples and then applies
hand-tuned geometric gates that section 4 showed sit at a local optimum and
section 5's sweep could not improve in any direction.

Two arms, because the comparison is the point:
  GEOM        the same information the gates already use. If a learned model on
              geometry alone beats the gates, the gates were simply the wrong
              functional form (axis-aligned thresholds on a joint distribution).
  GEOM+IMAGE  adds appearance, which the gates structurally cannot see. The
              difference between the two arms IS the value of the image.
"""
import sys, glob, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

SYN = ROOT / "artifacts/synthetic/biohub_synthetic/sequences"
UM = 1.625
R_UM = 14.0          # candidate radius; safe_div's sister gate is 14um
P = 10               # patch edge around each of the three points


def crop(vol, zyx, p=P):
    z, y, x = [int(round(v)) for v in zyx]; h = p // 2
    out = np.zeros((p, p, p), np.float32)
    z0, z1 = max(0, z-h), min(vol.shape[0], z-h+p)
    y0, y1 = max(0, y-h), min(vol.shape[1], y-h+p)
    x0, x1 = max(0, x-h), min(vol.shape[2], x-h+p)
    if z1 > z0 and y1 > y0 and x1 > x0:
        out[z0-(z-h):z1-(z-h), y0-(y-h):y1-(y-h), x0-(x-h):x1-(x-h)] = vol[z0:z1, y0:y1, x0:x1]
    m, s = out.mean(), out.std()
    return (out - m) / (s + 1e-6)


def geom(pp, pa, pb):
    """The geometry safe_div's gates read, in um."""
    dpa = np.linalg.norm(pa - pp) * UM
    dpb = np.linalg.norm(pb - pp) * UM
    dab = np.linalg.norm(pa - pb) * UM
    mean = (dpa + dpb) / 2 + 1e-6
    sym = abs(dpa - dpb) / mean                       # sister symmetry tau
    va, vb = (pa - pp), (pb - pp)
    na, nb = np.linalg.norm(va) + 1e-6, np.linalg.norm(vb) + 1e-6
    cos = float(np.dot(va, vb) / (na * nb))           # divergence of daughters
    return np.array([dpa, dpb, dab, sym, cos, max(dpa, dpb), min(dpa, dpb)], np.float32)


def build(files, neg_per_pos=4, seed=0):
    rng = random.Random(seed)
    G, I, Y = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        vols, nodes = d["volumes"], d["nodes"]
        divs = set(d["divisions"].tolist())
        t = nodes[:, 0].astype(int); zyx = nodes[:, 1:4].astype(float)
        out = {}
        for s_, d_ in d["edges"]:
            out.setdefault(int(s_), []).append(int(d_))
        by_t = {}
        for i in range(len(nodes)):
            by_t.setdefault(t[i], []).append(i)
        for p in range(len(nodes)):
            if t[p] + 1 >= vols.shape[0]:
                continue
            nxt = np.array(by_t.get(t[p] + 1, []))
            if len(nxt) < 2:
                continue
            dist = np.linalg.norm(zyx[nxt] - zyx[p], axis=1) * UM
            cand = nxt[dist <= R_UM]
            if len(cand) < 2:
                continue
            true = set(out.get(p, []))
            is_div = p in divs and len(true) == 2
            pairs = []
            if is_div:
                a, b = sorted(true)
                if a in cand and b in cand:
                    pairs.append(((a, b), 1))
            others = [(x, y) for ii, x in enumerate(cand) for y in cand[ii+1:]
                      if not (is_div and {int(x), int(y)} == true)]
            rng.shuffle(others)
            pairs += [(pr, 0) for pr in others[:neg_per_pos]]
            for (a, b), lab in pairs:
                G.append(geom(zyx[p], zyx[a], zyx[b])); Y.append(lab)
                I.append(np.stack([crop(vols[t[p]], zyx[p]),
                                   crop(vols[t[p]+1], zyx[a]),
                                   crop(vols[t[p]+1], zyx[b])]))
    return np.asarray(G, np.float32), np.asarray(I, np.float32), np.asarray(Y, np.int64)


if __name__ == "__main__":
    print(__doc__.strip()); print("=" * 84)
    files = sorted(glob.glob(str(SYN / "*.npz")))[:300]
    print(f"building candidate triples from {len(files)} sequences "
          f"(radius {R_UM}um, the safe_div sister gate) ...")
    G, I, Y = build(files)
    print(f"  {len(Y):,} triples, {int(Y.sum()):,} true divisions "
          f"({100*Y.mean():.1f}%)")
    print(f"  geometry {G.shape}, image {I.shape}")
    np.savez_compressed(ROOT / "artifacts/div_triples.npz", G=G, I=I, Y=Y)
    print(f"  cached -> artifacts/div_triples.npz")
