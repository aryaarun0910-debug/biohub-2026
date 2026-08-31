r"""CALIBRATION GATE for the reversed ceiling ladder (PKT-0046).

Refuses to let any derived quantity be trusted until the coordinate convention is proved.

WHY THIS IS A SEPARATE, MANDATORY STEP
--------------------------------------
GT coords are FULL-RES ``(z, y, x)`` and take ``(1.625, 0.40625, 0.40625)``, NOT isotropic
1.625. AGENTS.md section 4 records that two of three distance analyses in one day started with
the wrong convention. ``FACT-0040`` is the anchor: the median true inter-frame displacement of
GT edges on fold 1. If this module does not reproduce it, nothing downstream means anything and
the packet halts here.

The gate also reports the median under the ISOTROPIC convention - the mistake that has actually
been made twice - so the check demonstrably DISCRIMINATES rather than merely asserting a pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
ISO_SCALE = np.array([1.625, 1.625, 1.625], dtype=np.float64)

# FACT-0040, fold 1, VERIFIED. Tolerance is the +/- 0.15 um the committed
# nearest_parent_oracle.py calibration gate uses.
ANCHOR_UM = 1.817
TOLERANCE_UM = 0.15


def gt_voxel_deltas(geff: Path) -> np.ndarray:
    """Per-GT-edge (dz, dy, dx) in VOXELS, consecutive frames only."""
    from biotrack.metric import load_graph

    g = load_graph(geff)
    n = g.node_attrs().to_pandas()
    ids = n["node_id"].to_numpy().astype(np.int64)
    pos = n[["z", "y", "x"]].to_numpy().astype(np.float64)
    t = n["t"].to_numpy().astype(np.int64)
    row = {int(v): i for i, v in enumerate(ids)}
    e = g.edge_attrs().to_pandas()
    out = []
    for s, d in zip(e["source_id"], e["target_id"]):
        i, j = row.get(int(s)), row.get(int(d))
        if i is None or j is None or t[j] - t[i] != 1:
            continue
        out.append(pos[j] - pos[i])
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--prefix", default="6bba", help="fold-1 embryo; FACT-0040 is a fold-1 anchor")
    ap.add_argument("--max-crops", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    geffs = sorted(args.gt_dir.glob(f"{args.prefix}_*.geff"))
    if args.max_crops:
        geffs = geffs[: args.max_crops]
    if not geffs:
        raise SystemExit(f"no GT geffs under {args.gt_dir} with prefix {args.prefix}")

    chunks = []
    for i, g in enumerate(geffs, 1):
        chunks.append(gt_voxel_deltas(g))
        if i % 25 == 0 or i == len(geffs):
            print(f"  [{i}/{len(geffs)}] {g.stem}", flush=True)
    delta = np.concatenate(chunks)

    d_um = np.linalg.norm(delta * SCALE, axis=1)
    d_iso = np.linalg.norm(delta * ISO_SCALE, axis=1)
    median = float(np.median(d_um))
    iso_median = float(np.median(d_iso))
    ok = abs(median - ANCHOR_UM) <= TOLERANCE_UM
    iso_would_pass = abs(iso_median - ANCHOR_UM) <= TOLERANCE_UM

    result = {
        "schema_version": 1,
        "heartbeat": "REVLADDER_CALIBRATION_COMPLETE" if ok else "REVLADDER_CALIBRATION_FAILED",
        "anchor_fact": "FACT-0040",
        "anchor_um": ANCHOR_UM,
        "tolerance_um": TOLERANCE_UM,
        "scale_um_zyx": SCALE.tolist(),
        "prefix": args.prefix,
        "n_crops": len(geffs),
        "n_gt_edges": int(len(delta)),
        "median_um": median,
        "mean_um": float(d_um.mean()),
        "p25_um": float(np.percentile(d_um, 25)),
        "p75_um": float(np.percentile(d_um, 75)),
        "isotropic_median_um": iso_median,
        "isotropic_would_pass": bool(iso_would_pass),
        "passes": bool(ok),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{result['heartbeat']}")
    print(f"  anisotropic (1.625, 0.40625, 0.40625) median = {median:.3f} um"
          f"   anchor {ANCHOR_UM} +/- {TOLERANCE_UM} (FACT-0040) -> {'PASS' if ok else 'FAIL'}")
    print(f"  isotropic  (1.625, 1.625, 1.625)      median = {iso_median:.3f} um"
          f"   -> would {'PASS (gate does NOT discriminate!)' if iso_would_pass else 'FAIL'}")
    print(f"  n_gt_edges={len(delta):,} over {len(geffs)} {args.prefix} crops")
    if not ok:
        print("  !! COORDINATE CONVENTION IS WRONG - every derived quantity is void. HALT.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
