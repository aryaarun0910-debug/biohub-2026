#!/usr/bin/env python3
"""Build the detector training cache so Monday is training, not data engineering.

WHY THIS SHAPE. Annotation is 2.82% dense overall and 7x asymmetric between embryos (EXP-11),
so a dense "unannotated = background" target is wrong by construction -- it would teach the
detector to suppress 97% of real cells. Every sample therefore carries BOTH:

    heat   a Gaussian target rendered ONLY at annotated centres
    mask   1 where the loss may be taken, 0 where we simply do not know

A first attempt masked negatives purely by DISTANCE from annotated centres. Measured, that was
wrong: on 44b6 (0.77% annotated) "far from any annotation" is 100% of the volume, so every
unannotated real cell became a confident negative -- the exact failure the mask exists to stop.

The fix is INTENSITY, and it is measured, not assumed: annotated centres sit above the 99th
percentile of the volume, and 99.1-99.2% of voxels are dimmer than the DIMMEST annotated cell
(44b6 1574 vs p99 1520; 6bba 790 vs p99 756). So the mask is three-way:

    positive   at annotated Gaussian blobs
    negative   where intensity < the 5th percentile of this dataset's annotated centres
    UNKNOWN    the bright-but-unannotated remainder, ~0.9% of voxels -- excluded from the loss,
               because that is exactly where the unannotated cells live

The threshold is per-dataset: the two embryos differ ~2x in absolute intensity.

Output is one .npz per dataset under data/cache/, written atomically so an interrupted run
resumes instead of corrupting. It stores TARGETS ONLY -- heat, mask and the threshold used.
The volume stays in its zarr: re-storing it drove the cache toward ~104GB for nothing, when
zarr already streams at 211 MB/s (EXP-12), and all of this has to be copied to the Mac.
"""
from __future__ import annotations
import sys, os, argparse, time
from pathlib import Path
import numpy as np, zarr

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
from biohub.contracts import SCALE
import tracksdata as td

IMG = Path("data/images/train"); GT = Path("data/train_geff"); OUT = Path("data/cache")
STEM = (1, 4, 4)                      # detect.py's stem stride: targets live on this grid
SIGMA_UM = 2.0                        # Gaussian radius, physical
NEG_PCT = 5.0                         # negatives must be dimmer than this pct of annotated peaks


def grid_scale():
    """Physical size of one voxel ON THE STEM GRID."""
    return SCALE * np.array(STEM)


def downsample(v):
    """Volume -> stem grid by MAX pooling. Cells are bright; max preserves them, mean blurs."""
    z, y, x = v.shape
    return v.reshape(z // STEM[0], STEM[0], y // STEM[1], STEM[1],
                     x // STEM[2], STEM[2]).max(axis=(1, 3, 5))


def render(zyx_um, shape, vs, neg_thr):
    """Gaussian heat + THREE-WAY loss mask on the stem grid. `vs` is the stem-grid volume."""
    gs = grid_scale()
    heat = np.zeros(shape, np.float32)
    idx = np.empty((0, 3), int)
    if len(zyx_um):
        idx = np.rint(zyx_um / gs).astype(int)
        idx = idx[np.all((idx >= 0) & (idx < np.array(shape)), 1)]
    if len(idx):
        r = np.ceil(2.5 * SIGMA_UM / gs).astype(int)
        zz, yy, xx = [np.arange(-r[i], r[i] + 1) for i in range(3)]
        dz, dy, dx = np.meshgrid(zz * gs[0], yy * gs[1], xx * gs[2], indexing="ij")
        blob = np.exp(-(dz**2 + dy**2 + dx**2) / (2 * SIGMA_UM**2)).astype(np.float32)
        for c in idx:
            lo = np.maximum(c - r, 0); hi = np.minimum(c + r + 1, shape)
            bl = blob[tuple(slice(l - (c[i] - r[i]), l - (c[i] - r[i]) + (hi[i] - lo[i]))
                            for i, l in enumerate(lo))]
            sl = tuple(slice(lo[i], hi[i]) for i in range(3))
            np.maximum(heat[sl], bl, out=heat[sl])
    pos = heat > 0.01
    neg = vs < neg_thr                       # confidently background: dimmer than any real cell
    mask = (pos | neg).astype(np.uint8)      # everything else is UNKNOWN and carries no loss
    return heat, mask


def load_nodes(p):
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs()
    t = np.array([r["t"] for r in n.iter_rows(named=True)])
    zyx = np.array([[r["z"], r["y"], r["x"]] for r in n.iter_rows(named=True)], float)
    return t, zyx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0, help="frames per dataset (0 = all)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    geffs = sorted(GT.glob("*.geff"))
    if a.limit: geffs = geffs[:a.limit]
    t0 = time.time(); done = skipped = 0
    for gp in geffs:
        zp = IMG / f"{gp.stem}.zarr"
        op = OUT / f"{gp.stem}.npz"
        if op.exists(): skipped += 1; continue
        if not zp.exists(): continue                    # not extracted yet
        z = zarr.open(str(zp), mode="r"); arr = z["0"] if "0" in z else z
        T = arr.shape[0] if not a.frames else min(a.frames, arr.shape[0])
        gshape = tuple(arr.shape[1 + i] // STEM[i] for i in range(3))
        t, zyx_vox = load_nodes(gp)
        zyx_um = zyx_vox * SCALE
        # per-dataset negative threshold: the embryos differ ~2x in absolute intensity
        peaks = []
        for fi in range(0, T, max(1, T // 10)):
            vs = downsample(np.asarray(arr[fi], np.float32))
            sel = zyx_um[t == fi]
            if not len(sel): continue
            ix = np.rint(sel / grid_scale()).astype(int)
            ix = ix[np.all((ix >= 0) & (ix < np.array(gshape)), 1)]
            if len(ix): peaks.append(vs[ix[:, 0], ix[:, 1], ix[:, 2]])
        neg_thr = float(np.percentile(np.concatenate(peaks), NEG_PCT)) if peaks else np.inf
        heats, masks, ts = [], [], []
        for fi in range(T):
            vs = downsample(np.asarray(arr[fi], np.float32))
            h, m = render(zyx_um[t == fi], gshape, vs, neg_thr)
            heats.append(h); masks.append(m); ts.append(fi)
        tmp = op.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, heat=np.stack(heats).astype(np.float16),
                            mask=np.stack(masks), t=np.array(ts), zarr=str(zp),
                            sigma_um=SIGMA_UM, neg_pct=NEG_PCT, neg_thr=neg_thr,
                            stem=np.array(STEM))
        os.replace(tmp, op); done += 1
        if done % 10 == 0:
            print(f"    {done} written, {time.time()-t0:.0f}s", flush=True)
    print(f"  cache: {done} written, {skipped} already present, {time.time()-t0:.0f}s")
    if done:
        d = np.load(sorted(OUT.glob('*.npz'))[0])
        pos = (d['heat'] > 0.01).mean(); m = d['mask'].mean()
        print(f"  sample: heat {d['heat'].shape} {d['heat'].dtype} (volume stays in the zarr)")
        print(f"    positive {pos:.4%} | loss taken on {m:.2%} | UNKNOWN (excluded) {1-m:.2%}")


if __name__ == "__main__":
    main()
