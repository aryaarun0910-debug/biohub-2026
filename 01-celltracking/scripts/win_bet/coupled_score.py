"""Authoritative scoring + decomposition for the coupled arms over the full OOF population.

Scores A/B/B'/C/D per crop with the pinned patched scorer, then reports:
  * per-fold composite, raw edge Jaccard, count multiplier, division contribution;
  * node/edge survival through greedy -> ILP -> wrapper, and node-recall change;
  * regime slices (density, count ratio, displacement);
  * crop-bootstrap intervals on every step delta;
  * the preregistered sequential path B-A, B'-B, C-B', D-C and total D-A.

The decomposition is ORDER-DEPENDENT and its steps contain interactions: it is one
sequential path through a non-commutative configuration space, not an orthogonal split.

Per-crop scores are cached to disk so re-aggregation is instant.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

CACHE = ROOT / "artifacts" / "kaggle" / "coupled_cache"
ARMS_DIR = CACHE / "arms"
E0C = ROOT / "artifacts" / "kaggle" / "e0c_cache" / "graphs"
SCORES = CACHE / "scores"
ARMS = ["A", "B", "Bp", "C", "D"]
FOLD_NAME = {0: "44b6", 1: "6bba"}
STEPS = [("B-A", "A", "B", "detector population (greedy downstream)"),
         ("B'-B", "B", "Bp", "introduce global ILP at default 0.1/0.1"),
         ("C-B'", "Bp", "C", "ILP survival costs -> 0.0/1.5"),
         ("D-C", "C", "D", "wrapper/min-track/gap -> v122"),
         ("D-A", "A", "D", "TOTAL coupled effect")]


def _graph_path(arm: str, split: int, crop: str) -> Path:
    return (E0C / str(split) / f"{crop}.parquet") if arm == "A" \
        else (ARMS_DIR / arm / str(split) / f"{crop}.parquet")


def score_one(task) -> dict | None:
    arm, split, crop = task
    sys.path.insert(0, str(ROOT / "src"))
    from biotrack.metric import estimated_nodes, per_sample_metrics, score_pred_graph
    from biotrack.submission import submission_to_graphs
    from tracking_cellmot.metrics import EvaluationResult

    p = _graph_path(arm, split, crop)
    if not p.exists():
        return None
    df = pl.read_parquet(p)
    gdf = df.with_columns(pl.lit(crop).alias("dataset")).with_row_index("id")
    graph = submission_to_graphs(gdf)[crop]
    gt = str(ROOT / "data" / "train" / f"{crop}.geff")
    r = score_pred_graph(graph, gt)
    er = EvaluationResult(r["edge_tp"], r["edge_fp"], r["edge_fn"],
                          r["division_tp"], r["division_fp"], r["division_fn"],
                          r["num_pred_nodes"])
    n_est = estimated_nodes(gt)
    m = per_sample_metrics(er, n_est, r["node_recall"])

    nodes = df.filter(pl.col("row_type") == "node")
    edges = df.filter(pl.col("row_type") == "edge")
    n_frames = max(1, nodes["t"].n_unique())
    # Output graphs carry only geometry, not edge_dist, so derive displacement in um
    # from the endpoint coordinates (voxel -> um via the competition scale).
    disp = float("nan")
    if edges.height:
        sz, sy, sx = 1.625, 0.40625, 0.40625
        pos = {int(r["node_id"]): (r["z"] * sz, r["y"] * sy, r["x"] * sx)
               for r in nodes.iter_rows(named=True)}
        ds = []
        for s, t in zip(edges["source_id"].to_list(), edges["target_id"].to_list()):
            a, b = pos.get(int(s)), pos.get(int(t))
            if a and b:
                ds.append(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5)
        if ds:
            disp = float(np.median(ds))
    return {"arm": arm, "split": split, "crop": crop, **m,
            "n_est": n_est, "ratio": r["num_pred_nodes"] / n_est if n_est else float("nan"),
            "density": nodes.height / n_frames, "displacement": disp,
            "n_nodes_out": nodes.height, "n_edges_out": edges.height}


def collect(workers: int, refresh: bool) -> list[dict]:
    SCORES.mkdir(parents=True, exist_ok=True)
    cached, tasks = [], []
    for arm in ARMS:
        for split in (0, 1):
            src = E0C / str(split) if arm == "A" else ARMS_DIR / arm / str(split)
            if not src.exists():
                continue
            for p in sorted(src.glob("*.parquet")):
                crop = p.stem
                sp = SCORES / f"{arm}__{split}__{crop}.json"
                if sp.exists() and not refresh:
                    cached.append(json.loads(sp.read_text()))
                else:
                    tasks.append((arm, split, crop))
    print(f"[score] cached={len(cached)} to_score={len(tasks)}")
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rec in ex.map(score_one, tasks):
                if rec is None:
                    continue
                (SCORES / f"{rec['arm']}__{rec['split']}__{rec['crop']}.json").write_text(
                    json.dumps(rec))
                cached.append(rec)
    return cached


def agg(rows: list[dict]) -> dict:
    """Edge-volume-weighted aggregate, matching tracking_cellmot.metrics.summarise."""
    from tracking_cellmot.metrics import summarise
    return summarise(rows)


def boot(pairs: list[tuple[float, float]], n: int = 10000, seed: int = 20260728):
    """Crop-level bootstrap of a paired delta."""
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    d = np.array([b - a for a, b in pairs], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    rows = collect(a.workers, a.refresh)
    by = {}
    for r in rows:
        by.setdefault((r["arm"], r["split"]), []).append(r)

    # CRITICAL: every aggregate and every delta must be computed over the SAME crops.
    # Arm A is available for all 199 crops from the frozen E0c cache, while the other arms
    # only cover whatever has been replayed so far (and could lose a crop to a solver
    # failure). Comparing A-on-199 against D-on-N is a population mismatch that silently
    # fabricates a delta, so restrict every arm to the per-fold intersection.
    for split in (0, 1):
        present = [set(r["crop"] for r in by.get((arm, split), [])) for arm in ARMS]
        present = [s for s in present if s]
        if len(present) < len(ARMS):
            missing = [arm for arm in ARMS if not by.get((arm, split))]
            print(f"[score] fold {FOLD_NAME[split]}: arms with no data {missing} -- skipping fold")
            for arm in ARMS:
                by.pop((arm, split), None)
            continue
        common = set.intersection(*present)
        dropped = max(len(s) for s in present) - len(common)
        if dropped:
            print(f"[score] fold {FOLD_NAME[split]}: restricting to {len(common)} crops "
                  f"common to all arms ({dropped} dropped -- PARTIAL COVERAGE)")
        for arm in ARMS:
            by[(arm, split)] = [r for r in by.get((arm, split), []) if r["crop"] in common]

    print("\n" + "=" * 100)
    print("AGGREGATE BY FOLD (edge-volume weighted, patched authoritative scorer)")
    print("=" * 100)
    print(f"{'arm':<5}{'fold':<7}{'n':>5}{'composite':>11}{'adjJ':>9}{'rawJ':>9}"
          f"{'divJ':>8}{'mult':>8}{'recall':>8}{'nodes':>10}{'edges':>10}")
    print("-" * 100)
    comp = {}
    for arm in ARMS:
        for split in (0, 1):
            rs = by.get((arm, split), [])
            if not rs:
                continue
            s = agg(rs)
            mult = float(np.average([r["adj_edge_jaccard"] / r["edge_jaccard"]
                                     for r in rs if r["edge_jaccard"]],
                                    weights=[r["edge_tp"] + r["edge_fp"] + r["edge_fn"]
                                             for r in rs if r["edge_jaccard"]]))
            rec = float(np.mean([r["node_recall"] for r in rs]))
            comp[(arm, split)] = s["score"]
            dj = s["division_jaccard"]
            print(f"{arm:<5}{FOLD_NAME[split]:<7}{len(rs):>5}{s['score']:>11.4f}"
                  f"{s['adj_edge_jaccard']:>9.4f}{s['edge_jaccard']:>9.4f}"
                  f"{(dj if dj == dj else 0):>8.4f}{mult:>8.4f}{rec:>8.4f}"
                  f"{sum(r['n_nodes_out'] for r in rs):>10}"
                  f"{sum(r['n_edges_out'] for r in rs):>10}")

    print("\n" + "=" * 100)
    print("SEQUENTIAL DECOMPOSITION -- ORDER-DEPENDENT, CONTAINS INTERACTIONS")
    print("=" * 100)
    for label, x, y, why in STEPS:
        print(f"\n{label:<7} {why}")
        for split in (0, 1):
            rx = {r["crop"]: r for r in by.get((x, split), [])}
            ry = {r["crop"]: r for r in by.get((y, split), [])}
            common = sorted(set(rx) & set(ry))
            if not common:
                continue
            pairs = [(rx[c]["adj_edge_jaccard"], ry[c]["adj_edge_jaccard"]) for c in common]
            m, lo, hi = boot(pairs)
            agg_d = comp.get((y, split), float("nan")) - comp.get((x, split), float("nan"))
            pos = sum(1 for p, q in pairs if q > p)
            print(f"   {FOLD_NAME[split]:<6} n={len(common):>3}  aggregate_d={agg_d:+.4f}"
                  f"   percrop_mean={m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]"
                  f"   crops_up={pos}/{len(pairs)}")

    print("\n" + "=" * 100)
    print("PROMOTION GATE (D vs A)")
    print("=" * 100)
    deltas = {}
    for split in (0, 1):
        if ("A", split) in comp and ("D", split) in comp:
            deltas[split] = comp[("D", split)] - comp[("A", split)]
            print(f"   {FOLD_NAME[split]}: A={comp[('A',split)]:.4f} D={comp[('D',split)]:.4f} "
                  f"delta={deltas[split]:+.4f}")
    if len(deltas) == 2:
        mn = min(deltas.values())
        both = all(v > 0 for v in deltas.values())
        print(f"\n   both folds improve : {both}")
        print(f"   min-fold delta     : {mn:+.4f}  (gate: >= +0.005)")
        print(f"   VERDICT            : {'PASS' if (both and mn >= 0.005) else 'FAIL'}")

    print("\n" + "=" * 100)
    print("REGIME SLICES (D-A by tercile of each deployment-observable feature)")
    print("=" * 100)
    for feat in ("density", "ratio", "displacement"):
        print(f"\n  {feat}")
        for split in (0, 1):
            ra = {r["crop"]: r for r in by.get(("A", split), [])}
            rd = {r["crop"]: r for r in by.get(("D", split), [])}
            common = sorted(set(ra) & set(rd))
            if len(common) < 6:
                continue
            vals = np.array([ra[c][feat] for c in common], float)
            qs = np.nanpercentile(vals, [33.3, 66.7])
            for name, sel in (("low", vals <= qs[0]),
                              ("mid", (vals > qs[0]) & (vals <= qs[1])),
                              ("high", vals > qs[1])):
                cs = [c for c, k in zip(common, sel) if k]
                if not cs:
                    continue
                d = np.mean([rd[c]["adj_edge_jaccard"] - ra[c]["adj_edge_jaccard"] for c in cs])
                print(f"    {FOLD_NAME[split]:<6}{name:<5} n={len(cs):>3}  mean_dJ={d:+.4f}")


if __name__ == "__main__":
    main()
