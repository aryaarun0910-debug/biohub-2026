r"""H1-R pilot -- audit the packaged Zebrahub crops (`kkunizaw/biohub-zh001r`).

WHY THIS EXISTS
---------------
The H1 retrain was gated on acquiring Zebrahub imaging (150-232 GB at level-1 against 383 GB
free). A competitor has published the imaging pre-cropped as a public Kaggle dataset, which
collapses that acquisition to ~750 MB and is attachable to a kernel with no internet. Before any
GPU is spent on someone else's preprocessing, three things must be established from the bytes:

  1. GEOMETRY  -- what voxel scale is their "iso" grid, relative to our deployed detector input
                  (64^3 isotropic at 1.625 um)? If it is a different resolution lineage, a
                  retrain voids the deployed 0.915 anchor; if it is the same family, it does not.
  2. ALIGNMENT -- do their node coordinates actually land on nuclei in their own volumes? A
                  packaged dataset whose labels are offset from its images is worthless and the
                  failure is silent (training just converges to nothing).
  3. SUPERVISION -- what can actually be trained from it. This is the load-bearing one: the node
                  arrays are (N, 4) = [t, z, y, x] with NO track identity, so the dataset
                  supports a DETECTOR retrain and does NOT by itself support an edge/association
                  retrain. Discovering that after building a training lane would be expensive.

This script measures all three and exits non-zero if an integrity gate fails. It is the
prerequisite for `h1r_zh001r_smoke.py`, which runs one training step on the audited data.

Geometry is measured with a nucleus-size ruler, not nuclei spacing: spacing is strongly
stage-dependent (our own frames drift 11.7 -> 8.9 um within five frames) whereas the radial
intensity profile of a nucleus is comparatively stable. The ruler compares the mean radial
profile around real-label nuclei in their volumes against the same profile around competition GT
nuclei in ours, both on 64^3 isotropic grids, and reads the scale off the radius ratio.

Usage:
  .venv\Scripts\python.exe scripts\win_bet\h1r_zh001r_audit.py --root <dir-with-zh001r-files>
  .venv\Scripts\python.exe scripts\win_bet\h1r_zh001r_audit.py --root /kaggle/input/biohub-zh001r
  # add --compare-competition to run the geometry ruler (needs data/train locally)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())

DEPLOYED_VOX_UM = 1.625      # our detector input: 64^3 isotropic, xy downsampled 4x
DEPLOYED_XY_DS = 4           # competition zarr is (t, 64, 256, 256); detector reads [:, ::4, ::4]
PROFILE_R = 8                # radial profile out to 8 voxels
_RAD_IDX = None              # lazily built radius->bin index for the profile window


def _profile_index(r: int) -> np.ndarray:
    zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    rad = np.sqrt(zz ** 2 + yy ** 2 + xx ** 2).ravel()
    return np.digitize(rad, np.arange(0, r + 1)) - 1


def radial_profile(vol: np.ndarray, pts: np.ndarray, r: int = PROFILE_R,
                   acc: np.ndarray | None = None, cnt: np.ndarray | None = None):
    """Accumulate mean intensity vs integer radius around `pts` on an isotropic grid."""
    global _RAD_IDX
    if _RAD_IDX is None or len(_RAD_IDX) != (2 * r + 1) ** 3:
        _RAD_IDX = _profile_index(r)
    if acc is None:
        acc, cnt = np.zeros(r + 1), np.zeros(r + 1)
    vol = vol.astype(np.float32)
    for p in pts:
        z, y, x = (int(round(float(v))) for v in p)
        if not (r <= z < vol.shape[0] - r and r <= y < vol.shape[1] - r
                and r <= x < vol.shape[2] - r):
            continue
        np.add.at(acc, _RAD_IDX, vol[z - r:z + r + 1, y - r:y + r + 1, x - r:x + r + 1].ravel())
        np.add.at(cnt, _RAD_IDX, 1)
    return acc, cnt


def normalised(acc: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    p = acc / np.maximum(cnt, 1)
    p = p - p.min()
    return p / max(p[0], 1e-9)


def radius_at(prof: np.ndarray, frac: float) -> float | None:
    """Interpolated radius at which the normalised profile drops to `frac`."""
    for i in range(1, len(prof)):
        if prof[i] <= frac:
            return (i - 1) + (prof[i - 1] - frac) / max(prof[i - 1] - prof[i], 1e-9)
    return None


def load_pack(root: Path):
    iso_p, nodes_p = root / "zh001r_iso.npy", root / "zh001r_nodes.npz"
    if not iso_p.exists() or not nodes_p.exists():
        raise SystemExit(f"missing zh001r_iso.npy / zh001r_nodes.npz under {root}")
    iso = np.load(iso_p, mmap_mode="r")
    nodes = np.load(nodes_p, allow_pickle=True)
    tgt_p = root / "zh001r_tgt.npy"
    tgt = np.load(tgt_p, mmap_mode="r") if tgt_p.exists() else None
    return iso, nodes, tgt


def audit_structure(iso, nodes, tgt) -> dict:
    print("== STRUCTURE ==")
    print(f"  iso   : shape={iso.shape} dtype={iso.dtype}")
    if iso.ndim != 5:
        raise SystemExit(f"GATE FAIL: expected (crops, T, Z, Y, X), got {iso.shape}")
    n_crop, n_t = iso.shape[0], iso.shape[1]
    keys = list(nodes.keys())
    print(f"  nodes : {len(keys)} arrays, expect crops*T = {n_crop}*{n_t} = {n_crop * n_t}")
    if len(keys) != n_crop * n_t:
        raise SystemExit(f"GATE FAIL: node array count {len(keys)} != {n_crop * n_t}")

    widths = {nodes[k].shape[1] for k in keys[:64]}
    counts = [nodes[k].shape[0] for k in keys]
    print(f"  nodes : column widths {widths}, per-frame count "
          f"min/median/max = {min(counts)}/{int(np.median(counts))}/{max(counts)}, "
          f"total {sum(counts)}")
    if tgt is not None:
        print(f"  tgt   : shape={tgt.shape} dtype={tgt.dtype}")
        s = np.asarray(tgt[0, 0])
        print(f"          crop0/t0 min={s.min()} max={s.max()} mean={s.mean():.3f} "
              f"nonzero_frac={(s > 0).mean():.4f}")
    else:
        print("  tgt   : ABSENT (not downloaded) -- optional")

    # --- SUPERVISION: the load-bearing check -------------------------------------------
    print("\n== SUPERVISION ==")
    has_identity = widths != {4}
    print(f"  node columns = {sorted(widths)}  -> interpreted as [t, z, y, x]")
    print(f"  track identity present: {has_identity}")
    print("  => DETECTOR retrain  : SUPPORTED (point supervision, matches "
          "train_unet_transformer.compute_detection_loss)")
    print(f"  => EDGE/ASSOC retrain: {'SUPPORTED' if has_identity else 'NOT SUPPORTED'} "
          "(no track_id/parent_id => no GT transition matrix can be built)")
    if not has_identity:
        print("     Association supervision would need identity recovered by registering these")
        print("     crops back onto the ZSNS001 Zebrahub tracks we already hold on disk.")
    return {"n_crop": n_crop, "n_t": n_t, "has_identity": has_identity, "counts": counts}


def audit_alignment(iso, nodes, meta, n_probe: int = 12) -> np.ndarray:
    """Do their labels sit on nuclei in their own volumes? Peak-vs-background contrast."""
    print("\n== ALIGNMENT (their nodes vs their imaging) ==")
    n_t = meta["n_t"]
    acc = cnt = None
    rng = np.random.default_rng(0)
    ctr, bg = [], []
    for ci in range(0, min(meta["n_crop"], 24), max(1, 24 // n_probe)):
        for ti in (0, n_t // 2, n_t - 1):
            vol = np.asarray(iso[ci, ti])
            a = nodes[f"f{ci * n_t + ti}"]
            sel = a[:: max(1, len(a) // 60)]
            acc, cnt = radial_profile(vol, sel[:, 1:], acc=acc, cnt=cnt)
            zi = np.clip(sel[:, 1].astype(int), 0, vol.shape[0] - 1)
            yi = np.clip(sel[:, 2].astype(int), 0, vol.shape[1] - 1)
            xi = np.clip(sel[:, 3].astype(int), 0, vol.shape[2] - 1)
            ctr.append(vol[zi, yi, xi].astype(np.float32))
            bg.append(vol[rng.integers(0, vol.shape[0], len(sel)),
                          rng.integers(0, vol.shape[1], len(sel)),
                          rng.integers(0, vol.shape[2], len(sel))].astype(np.float32))
    prof = normalised(acc, cnt)
    c, b = np.concatenate(ctr).mean(), np.concatenate(bg).mean()
    print(f"  mean intensity at nodes = {c:.1f}   at random voxels = {b:.1f}   ratio = {c / max(b, 1e-6):.2f}x")
    print(f"  normalised radial profile: {np.round(prof, 3)}")
    if c <= b * 1.15:
        raise SystemExit("GATE FAIL: node positions show no intensity contrast -- labels are "
                         "not aligned to the imaging; do not train on this.")
    print("  GATE PASS: labels are aligned to the imaging.")
    return prof


def audit_geometry(their_prof: np.ndarray, n_crops: int = 18) -> None:
    """Nucleus-size ruler: their voxel size relative to our deployed 1.625 um grid."""
    import zarr  # local import: only needed for this optional half
    print("\n== GEOMETRY (nucleus-size ruler vs competition data) ==")
    acc = cnt = None
    used = 0
    for gp in sorted(glob.glob(str(ROOT / "data" / "train" / "44b6_*.geff")))[:n_crops]:
        stem = gp[:-5]
        try:
            arr = zarr.open(stem + ".zarr", mode="r")["0"]
            g = zarr.open(gp, mode="r")
        except Exception:
            continue
        t = np.asarray(g["nodes/props/t/values"][:])
        gz = np.asarray(g["nodes/props/z/values"][:])
        gy = np.asarray(g["nodes/props/y/values"][:])
        gx = np.asarray(g["nodes/props/x/values"][:])
        for tt in np.unique(t)[::5]:
            m = t == tt
            vol = np.asarray(arr[int(tt), :, ::DEPLOYED_XY_DS, ::DEPLOYED_XY_DS])
            pts = np.stack([gz[m], gy[m] / DEPLOYED_XY_DS, gx[m] / DEPLOYED_XY_DS], 1)
            acc, cnt = radial_profile(vol, pts, acc=acc, cnt=cnt)
            used += int(m.sum())
    if acc is None or used < 50:
        print("  SKIPPED: not enough competition GT nodes found under data/train")
        return
    ours = normalised(acc, cnt)
    print(f"  ours   ({used} competition GT nuclei, 64^3 @ {DEPLOYED_VOX_UM} um): {np.round(ours, 3)}")
    print(f"  theirs                                                    : {np.round(their_prof, 3)}")
    ests = []
    for f in (0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        a, b = radius_at(ours, f), radius_at(their_prof, f)
        if a and b:
            ests.append(a * DEPLOYED_VOX_UM / b)
    if not ests:
        print("  SKIPPED: profiles too flat to read a radius")
        return
    ests = np.array(ests)
    med = float(np.median(ests))
    print(f"  implied THEIR voxel size: median {med:.3f} um  "
          f"(range {ests.min():.3f}-{ests.max():.3f}), ratio {med / DEPLOYED_VOX_UM:.3f}x ours")
    print(f"  implied crop extent: {64 * med:.1f} um cube (ours {64 * DEPLOYED_VOX_UM:.1f} um)")
    verdict = ("SAME geometry family as the deployed input"
               if 0.75 <= med / DEPLOYED_VOX_UM <= 1.35
               else "DIFFERENT resolution lineage -- a retrain would void the deployed anchor")
    print(f"  => {verdict}")
    print("  CONFOUND: the ruler assumes comparable physical nucleus size across embryos/stages.")
    print("            Their nuclei are denser (later stage), which biases this estimate UPWARD.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="directory holding zh001r_iso.npy / zh001r_nodes.npz [/ zh001r_tgt.npy]")
    ap.add_argument("--compare-competition", action="store_true",
                    help="run the geometry ruler against data/train (needs the competition data)")
    args = ap.parse_args()

    iso, nodes, tgt = load_pack(Path(args.root))
    meta = audit_structure(iso, nodes, tgt)
    prof = audit_alignment(iso, nodes, meta)
    if args.compare_competition:
        audit_geometry(prof)
    else:
        print("\n(geometry ruler skipped; pass --compare-competition to run it)")
    print("\nAUDIT OK")


if __name__ == "__main__":
    sys.exit(main())
