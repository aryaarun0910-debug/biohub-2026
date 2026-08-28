"""Pair full-chain replay JSONs against the actual P28 control export."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.win_bet import div_reach_steal as drs  # noqa: E402


def paired_bootstrap(control: list[dict], candidate: list[dict], summarise, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        indices = rng.integers(0, len(control), len(control))
        c = summarise([control[i] for i in indices])["score"]
        o = summarise([candidate[i] for i in indices])["score"]
        deltas.append(float(o - c))
    values = np.asarray(deltas)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "draws": draws,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-csv", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    result_paths = sorted(args.results_dir.glob("candidate_3p0_*.json"))
    if not result_paths:
        raise SystemExit("no candidate_3p0 replay JSONs found")
    candidate_json = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    crops = [item["crop"] for item in candidate_json]
    if len(crops) != len(set(crops)):
        raise SystemExit("duplicate candidate crops")

    from biotrack.submission import read_submission

    champion = read_submission(args.champion_csv).filter(pl.col("dataset").is_in(crops))
    found = set(champion["dataset"].unique().to_list())
    missing = sorted(set(crops) - found)
    if missing:
        raise SystemExit(f"candidate crops absent from champion CSV: {missing}")
    ea, scorer = drs._ea_atlas(), drs._scorer()
    summarise = scorer[-1]
    controls: list[dict] = []
    candidates: list[dict] = []
    per_crop = []
    by_crop = {item["crop"]: item for item in candidate_json}
    for crop in crops:
        control = drs.score_crop(
            champion.filter(pl.col("dataset") == crop), args.train_dir / f"{crop}.geff", ea, scorer
        )
        candidate = by_crop[crop]["score"]
        controls.append(control)
        candidates.append(candidate)
        per_crop.append(
            {
                "crop": crop,
                "control_adj_edge_jaccard": float(control["adj_edge_jaccard"]),
                "candidate_adj_edge_jaccard": float(candidate["adj_edge_jaccard"]),
                "delta_adj_edge_jaccard": float(candidate["adj_edge_jaccard"] - control["adj_edge_jaccard"]),
                "delta_edge_jaccard": float(candidate["edge_jaccard"] - control["edge_jaccard"]),
                "delta_node_recall": float(candidate["node_recall"] - control["node_recall"]),
                "delta_nodes": int(candidate["num_pred_nodes"] - control["num_pred_nodes"]),
                "delta_division_fp": int(candidate["division_fp"] - control["division_fp"]),
            }
        )
    control_summary = drs._arm_summary(controls, summarise)
    candidate_summary = drs._arm_summary(candidates, summarise)
    keys = ("score", "adj_edge_jaccard", "edge_jaccard", "division_jaccard", "edge_tp", "edge_fp", "edge_fn")
    delta = {
        key: float(candidate_summary[key] - control_summary[key])
        for key in keys
        if isinstance(candidate_summary.get(key), (int, float))
        and isinstance(control_summary.get(key), (int, float))
        and candidate_summary[key] == candidate_summary[key]
        and control_summary[key] == control_summary[key]
    }
    result = {
        "schema_version": 1,
        "heartbeat": "P28_FULL_CHAIN_PANEL_COMPLETE",
        "selection": "five smallest controls plus fifteen fixed raw-node-rank probes",
        "n_crops": len(crops),
        "control": control_summary,
        "candidate": candidate_summary,
        "delta": delta,
        "paired_bootstrap": paired_bootstrap(controls, candidates, summarise, args.draws, args.seed),
        "signs": {
            "positive": sum(row["delta_adj_edge_jaccard"] > 0 for row in per_crop),
            "negative": sum(row["delta_adj_edge_jaccard"] < 0 for row in per_crop),
            "zero": sum(row["delta_adj_edge_jaccard"] == 0 for row in per_crop),
        },
        "per_crop": per_crop,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("heartbeat", "n_crops", "delta", "paired_bootstrap", "signs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
