r"""Score a submission CSV WITHOUT numba -- the fallback when Smart App Control blocks llvmlite.

WHY THIS EXISTS
---------------
Windows Smart App Control is ENFORCED on this machine (verified:
HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy\VerifiedAndReputablePolicyState = 1). It blocks
`llvmlite.dll` (an unsigned 120 MB native DLL), so `import numba` raises and every scoring path
that goes through `tracksdata` -> `biotrack.metric` dies at import. A `--force-reinstall` of
numba/llvmlite does NOT fix it: the fresh copy is the same unsigned binary and is blocked
identically. SAC has no per-file allow-list, and disabling it is irreversible without reinstalling
Windows -- so the fix is to route around it, not to weaken the machine.

`src/biotrack/metric_numpy.py` is a pure-numpy reimplementation of the EDGE term, validated exact
against the organiser implementation (see `scripts/validate_numpy_metric.py`,
`tests/test_numpy_metric.py`). It imports with no numba. This script drives it over a submission
CSV plus `data/train/*.geff`, reproducing `score_loeo_submission.py`'s edge numbers.

SCOPE / HONESTY
---------------
- `adj_edge_jaccard`, `node_recall` and the count ratio are EXACT (validated module).
- The **division term is computed here separately in plain numpy** (fork match within
  `max_distance`), and is a CLOSE reimplementation, not the validated one. Treat divJ from this
  script as indicative; confirm with the official scorer once numba is available.
- `N_est` is GT metadata and is unavailable at test time; like the official harness we take it
  from the GT node count per crop.

Usage:
  .venv\Scripts\python.exe scripts\core\score_submission_nonumba.py \
      --csv c:\temp\p7_final\submission.csv --gt-dir data\train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
from biotrack.metric_numpy import Sample, score_sample, summarise  # noqa: E402

SCALE = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 7.0


def load_gt(geff: Path) -> Sample:
    g = zarr.open(str(geff), mode="r")
    nid = np.asarray(g["nodes/ids"][:]).astype(np.int64)
    t = np.asarray(g["nodes/props/t/values"][:]).astype(np.int64)
    zyx = np.stack([np.asarray(g["nodes/props/z/values"][:]),
                    np.asarray(g["nodes/props/y/values"][:]),
                    np.asarray(g["nodes/props/x/values"][:])], axis=1).astype(float)
    e = np.asarray(g["edges/ids"][:])
    e = e.astype(np.int64) if e.ndim == 2 and len(e) else np.zeros((0, 2), np.int64)
    return Sample(node_ids=nid, t=t, zyx=zyx, edges=e)


def division_counts(pred: Sample, gt: Sample) -> tuple[int, int, int]:
    """Indicative division TP/FP/FN: a fork matches if mother and both daughters match."""
    from biotrack.metric_numpy import match_nodes
    m = match_nodes(pred, gt, SCALE, MATCH_UM)

    def forks(s: Sample) -> dict[int, set]:
        out: dict[int, set] = {}
        for a, b in s.edges:
            out.setdefault(int(a), set()).add(int(b))
        return {k: v for k, v in out.items() if len(v) >= 2}

    gt_f, pr_f = forks(gt), forks(pred)
    gt_ann = set(gt.node_ids.tolist())
    tp = 0
    matched_gt: set[int] = set()
    for src, kids in pr_f.items():
        gsrc = m.get(src)
        if gsrc is None or gsrc not in gt_f:
            continue
        gkids = {m[k] for k in kids if k in m}
        if len(gkids & gt_f[gsrc]) >= 2:
            tp += 1
            matched_gt.add(gsrc)
    # FP counted only where the mother lands on an annotated GT node (host behaviour)
    fp = sum(1 for src in pr_f if m.get(src) in gt_ann and m.get(src) not in matched_gt)
    fn = len(gt_f) - len(matched_gt)
    return tp, fp, fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--gt-dir", default="data/train")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rows, dtp, dfp, dfn = [], 0, 0, 0
    for ds, g in df.groupby("dataset"):
        geff = Path(args.gt_dir) / f"{ds}.geff"
        if not geff.exists():
            print(f"  {ds}: no GT, skipped")
            continue
        gt = load_gt(geff)
        n = g[g.row_type == "node"]
        e = g[g.row_type == "edge"]
        pred = Sample(
            node_ids=n["node_id"].to_numpy().astype(np.int64),
            t=n["t"].to_numpy().astype(np.int64),
            zyx=n[["z", "y", "x"]].to_numpy().astype(float),
            edges=(e[["source_id", "target_id"]].to_numpy().astype(np.int64)
                   if len(e) else np.zeros((0, 2), np.int64)),
        )
        r = score_sample(pred, gt, n_est=float(len(gt.node_ids)))
        rows.append(r)
        a, b, c = division_counts(pred, gt)
        dtp, dfp, dfn = dtp + a, dfp + b, dfn + c
        print(f"  {ds:18s} nodes={len(pred.node_ids):7d} adjJ={r['adj_edge_jaccard']:.4f} "
              f"recall={r.get('node_recall', float('nan')):.3f}")

    s = summarise(rows)
    divj = dtp / max(dtp + dfp + dfn, 1)
    print(f"\n===== NON-NUMBA COMPOSITE ({len(rows)} crops) =====")
    print(f"  adj_edge_jaccard (weighted) = {s['adj_edge_jaccard']:.4f}    [EXACT: validated module]")
    print(f"  division_jaccard            = {divj:.4f} (TP={dtp} FP={dfp} FN={dfn})"
          f"    [INDICATIVE reimplementation]")
    print(f"  SCORE (approx)              = {s['adj_edge_jaccard'] + 0.1 * divj:.4f}")
    print("\n  NOTE: the edge term is exact; the division term is indicative. Re-confirm with")
    print("  scripts/core/score_loeo_submission.py once Smart App Control stops blocking llvmlite.")


if __name__ == "__main__":
    main()
