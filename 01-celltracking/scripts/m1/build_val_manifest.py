"""Build the frozen M1 inner-validation manifests (12 crops per LOEO direction).

Replaces the biased "first six sorted crop names" split, which on 44b6 strongly favours
very sparsely labelled crops and is not representative.

Selection is stratified over a 3x2x2 grid of DEPLOYMENT-OBSERVABLE statistics taken from
the cached pre-ILP detections (density x candidate load x motion), giving exactly 12 cells.
Within each cell the crop with the most annotated GT edges is chosen, so the inner-
validation score is as stable as the cell allows.

  * inner validation is drawn ONLY from the training family for that direction;
  * the held-out family is never touched here;
  * GT edge counts are used purely to pick the most stably scorable crop inside a cell,
    and only ever for the TRAINING family -- never the held-out one;
  * near-duplicates are dropped by correlation of per-frame detection-count series.

Directions (matching data/dataset_splits.json):
  split 1 -> train 44b6 (71 crops), hold out 6bba   [M1 runs this FIRST: min-fold]
  split 0 -> train 6bba (128 crops), hold out 44b6

The manifests are frozen and committed before any training.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

DET = ROOT / "artifacts/kaggle/coupled_cache/det"
OUT = ROOT / "scripts/m1/val_manifests.json"
TRAIN_FAMILY = {1: "44b6", 0: "6bba"}   # direction -> family trained on
HELD_OUT = {1: "6bba", 0: "44b6"}
N_VAL = 12
DUP_CORR = 0.995


def crop_stats(crop: str) -> dict:
    """Deployment-observable statistics from the cached pre-ILP detections."""
    z = np.load(DET / f"{crop}__tta-4view__det-0.969.npz")
    c = z["coords"]
    n = int(c.shape[0])
    t = c[:, 0].astype(int)
    tmax = int(t.max()) + 1 if n else 1
    per_frame = np.bincount(t, minlength=tmax).astype(float)
    src, tgt = z["edge_src"], z["edge_tgt"]
    # median inter-frame displacement in um (motion proxy)
    sz, sy, sx = 1.625, 0.40625, 0.40625
    if len(src):
        p = c[:, 1:].astype(float) * np.array([sz, sy, sx])
        d = np.linalg.norm(p[tgt.astype(int)] - p[src.astype(int)], axis=1)
        motion = float(np.median(d))
    else:
        motion = 0.0
    return {"n_det": n, "det_per_frame": n / tmax,
            "cand_per_node": len(src) / max(n, 1), "motion_um": motion,
            "per_frame": per_frame}


def gt_edges(crop: str) -> int:
    from biotrack.metric import load_graph
    g = load_graph(str(ROOT / "data" / "train" / f"{crop}.geff"))
    return int(g.num_edges())


def drop_near_duplicates(crops: list[str], stats: dict) -> tuple[list[str], list[tuple]]:
    keep, dropped = [], []
    for c in crops:
        dup = None
        for k in keep:
            a, b = stats[c]["per_frame"], stats[k]["per_frame"]
            m = min(len(a), len(b))
            if m < 5:
                continue
            r = float(np.corrcoef(a[:m], b[:m])[0, 1])
            if r >= DUP_CORR:
                dup = (k, r); break
        if dup:
            dropped.append((c, dup[0], round(dup[1], 4)))
        else:
            keep.append(c)
    return keep, dropped


def select(direction: int) -> dict:
    fam = TRAIN_FAMILY[direction]
    crops = sorted(p.name.split("__")[0] for p in DET.glob(f"{fam}_*__tta-4view__det-0.969.npz"))
    stats = {c: crop_stats(c) for c in crops}
    kept, dropped = drop_near_duplicates(crops, stats)

    dens = np.array([stats[c]["det_per_frame"] for c in kept])
    cand = np.array([stats[c]["cand_per_node"] for c in kept])
    mot = np.array([stats[c]["motion_um"] for c in kept])
    d_edges = np.quantile(dens, [1 / 3, 2 / 3])
    c_med, m_med = np.median(cand), np.median(mot)

    def cell(c):
        s = stats[c]
        di = 0 if s["det_per_frame"] <= d_edges[0] else (1 if s["det_per_frame"] <= d_edges[1] else 2)
        return (di, int(s["cand_per_node"] > c_med), int(s["motion_um"] > m_med))

    buckets: dict[tuple, list[str]] = {}
    for c in kept:
        buckets.setdefault(cell(c), []).append(c)

    ge = {c: gt_edges(c) for c in kept}
    chosen, cells = [], {}
    for cl in sorted(buckets):                       # 12 cells max; pick most-scorable
        best = max(buckets[cl], key=lambda c: ge[c])
        chosen.append(best); cells[str(cl)] = best
    # If the grid yielded fewer than N_VAL cells, top up with the most scorable remainder.
    if len(chosen) < N_VAL:
        rest = sorted(set(kept) - set(chosen), key=lambda c: -ge[c])
        chosen += rest[:N_VAL - len(chosen)]
    chosen = sorted(chosen[:N_VAL])

    return {
        "direction": direction, "train_family": fam, "held_out_family": HELD_OUT[direction],
        "n_train_family_crops": len(crops), "n_after_dedup": len(kept),
        "near_duplicates_dropped": dropped,
        "stratification": {"density_tercile_edges": [float(x) for x in d_edges],
                           "cand_per_node_median": float(c_med),
                           "motion_um_median": float(m_med),
                           "cells_filled": len(buckets)},
        "inner_val_crops": chosen,
        "coverage": [{"crop": c, "det_per_frame": round(stats[c]["det_per_frame"], 1),
                      "cand_per_node": round(stats[c]["cand_per_node"], 3),
                      "motion_um": round(stats[c]["motion_um"], 3),
                      "gt_edges": ge[c], "cell": str(cell(c))} for c in chosen],
        "train_crops": sorted(set(crops) - set(chosen)),
    }


def main() -> None:
    man = {"note": ("Frozen M1 inner-validation manifests. Inner validation is drawn ONLY "
                    "from the training family; the held-out family is untouched until a "
                    "checkpoint has been selected."),
           "n_val": N_VAL, "dup_corr_threshold": DUP_CORR, "directions": {}}
    for d in (1, 0):
        man["directions"][str(d)] = select(d)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(man, indent=2))

    for d in (1, 0):
        e = man["directions"][str(d)]
        print(f"\n=== direction {d}: train {e['train_family']} "
              f"({e['n_train_family_crops']} crops) -> hold out {e['held_out_family']} ===")
        print(f"  dedup: {e['n_train_family_crops']} -> {e['n_after_dedup']} "
              f"({len(e['near_duplicates_dropped'])} near-duplicates dropped)")
        print(f"  cells filled: {e['stratification']['cells_filled']}/12 | "
              f"val={len(e['inner_val_crops'])} train={len(e['train_crops'])}")
        print(f"  {'crop':<18}{'det/frame':>10}{'cand/node':>11}{'motion_um':>11}{'gt_edges':>10}{'cell':>10}")
        for r in e["coverage"]:
            print(f"  {r['crop']:<18}{r['det_per_frame']:>10.1f}{r['cand_per_node']:>11.3f}"
                  f"{r['motion_um']:>11.3f}{r['gt_edges']:>10}{r['cell']:>10}")
        g = [r["gt_edges"] for r in e["coverage"]]
        print(f"  GT edges: min={min(g)} median={int(np.median(g))} max={max(g)} total={sum(g)}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
