"""H1-N follow-up diagnostic -- TEMPORAL SNAP.

The forensic taxonomy found 26 of 202 admitted false positives to be TEMPORAL_PHASE:
H1-M picked the right predicted TRACK but the wrong FRAME, with a real GT division
within +/-2 frames on that same track. That is a decomposable error -- one signal chooses
the CELL, another chooses the FRAME -- and it is the only error class in the taxonomy
that can turn a false positive into a true positive rather than merely deleting it.

This script measures the size of that prize:

  ORACLE ceiling   move every admission to the frame on its track that maximises the
                   GT outcome. Upper bound only, never deployable.
  GT-FREE snap     move every admission to the frame on its track (within +/-W) that
                   maximises signal S. One arm per signal. Deployable in principle.

Both keep the admission COUNT fixed (deduplicating collisions), so any gain comes from
re-timing, not from a larger budget.

Usage:
  .venv\\Scripts\\python.exe scripts\\h1n_temporal_snap.py ^
      --forensics <...>\\forensics.all.parquet ^
      --breakeven reports\\inventory\\pooled_breakeven.json --out <...>\\snap.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h1n_prereg as P                              # noqa: E402
from h1n_compose import Pooled                      # noqa: E402

SIGNALS = ["S_H1M", "S_H1I", "S_RTD_A", "S_RTD_G", "S_GEOM"]
WINDOW = P.TEMPORAL_PHASE_WINDOW


def snap(D: pd.DataFrame, by: str, w: int) -> pd.DataFrame:
    """Re-time every admitted event to the best frame on its own predicted track within
    +/-w. `by` == 'ORACLE' uses the label; otherwise it is a signal column."""
    adm = D[D.admitted]
    idx = {}
    for r in D.itertuples(index=False):
        idx.setdefault((r.crop, int(r.mother_track_id)), []).append(r)
    picked, seen = [], set()
    for r in adm.itertuples(index=False):
        cands = [x for x in idx[(r.crop, int(r.mother_track_id))]
                 if abs(int(x.t) - int(r.t)) <= w]
        if by == "ORACLE":
            best = max(cands, key=lambda x: (bool(x.realisable), getattr(x, "S_H1M")))
        else:
            best = max(cands, key=lambda x: (getattr(x, by) if np.isfinite(
                getattr(x, by)) else -np.inf))
        if best.key in seen:
            continue                       # collision: two admissions snap to one frame
        seen.add(best.key)
        picked.append({"key": best.key, "family": best.family,
                       "realisable": bool(best.realisable)})
    return pd.DataFrame(picked)


def score(sel: pd.DataFrame, PL: Pooled) -> dict:
    o = {}
    for fam in ("44b6", "6bba"):
        g = sel[sel.family == fam]
        o[fam] = (int(g.realisable.sum()), int((~g.realisable).sum()))
    k44, m44 = o["44b6"]
    k6, m6 = o["6bba"]
    K = k44 + m44 + k6 + m6
    return {"K": K, "TP": k44 + k6, "FP": m44 + m6,
            "TP_44b6": k44, "FP_44b6": m44, "TP_6bba": k6, "FP_6bba": m6,
            "precision": (k44 + k6) / max(K, 1),
            "composite_delta_SURROGATE": PL.corrected(k44, m44, k6, m6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forensics", required=True)
    ap.add_argument("--breakeven", required=True)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D = pd.read_parquet(a.forensics)
    PL = Pooled(json.loads(Path(a.breakeven).read_text()))
    base = score(D[D.admitted][["key", "family", "realisable"]], PL)
    res = {"prereg_hash": P.prereg_hash(), "window": a.window,
           "no_snap": base, "arms": {}}
    print(f"  no snap   : {base['TP']:3d} TP / {base['FP']:3d} FP  K={base['K']:3d}  "
          f"prec={base['precision']:.4f}  delta={base['composite_delta_SURROGATE']:+.5f}")
    for by in ["ORACLE"] + SIGNALS:
        r = score(snap(D, by, a.window), PL)
        res["arms"][by] = r
        tag = "  [CEILING, GT-informed]" if by == "ORACLE" else ""
        print(f"  {by:9s}: {r['TP']:3d} TP / {r['FP']:3d} FP  K={r['K']:3d}  "
              f"prec={r['precision']:.4f}  delta={r['composite_delta_SURROGATE']:+.5f}"
              f"{tag}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
