"""Measure fixed-detection edge ceilings on true held-out-embryo OOF GEFFs.

Example (both folds are required by default):
  python scripts/oracle_redetect/measure_oracles.py \
    --pred-dir artifacts/oof/fold_44b6 --pred-dir artifacts/oof/fold_6bba \
    --gt-dir data/train --out-csv artifacts/oracles.csv

Optional candidate CSVs are named <crop>.csv and have source_id,target_id.
Without --candidate-dir, the candidate-support oracle is explicitly measured on
the prediction's already-selected edge support, not the linker's pre-ILP pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from biotrack.metric import estimated_nodes  # noqa: E402
from biotrack.metric_numpy import summarise  # noqa: E402
from core import load_sample, measure_oracles  # noqa: E402


def discover(pred_dirs: list[Path], folds: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in pred_dirs:
        if not root.exists():
            raise FileNotFoundError(f"prediction directory does not exist: {root}")
        for path in root.rglob("*.geff"):
            if not path.is_dir():
                continue
            stem = path.stem
            if stem.split("_", 1)[0] not in folds:
                continue
            if stem in found:
                raise ValueError(f"duplicate prediction for {stem}: {found[stem]} and {path}")
            found[stem] = path
    missing_folds = sorted(f for f in folds if not any(s.startswith(f + "_") for s in found))
    if missing_folds:
        raise FileNotFoundError(
            "no OOF prediction GEFFs found for fold(s) " + ", ".join(missing_folds)
            + ". Download the Kaggle predict-score outputs first; training checkpoints alone are insufficient."
        )
    return found


def read_candidate_edges(candidate_dir: Path | None, stem: str, pred_edges: np.ndarray) -> tuple[np.ndarray, str]:
    if candidate_dir is None:
        return pred_edges, "selected_prediction_edges"
    path = candidate_dir / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing candidate CSV: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    required = {"source_id", "target_id"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    edges = np.asarray([(int(r["source_id"]), int(r["target_id"])) for r in rows], dtype=np.int64).reshape(-1, 2)
    return edges, "pre_solver_candidate_csv"


def flat_row(stem: str, source: str, result: dict) -> dict:
    row = {
        "crop": stem,
        "fold": stem.split("_", 1)[0],
        "candidate_source": source,
        "gt_edges": result["gt_edges"],
        "candidate_tp_edges": result["candidate_tp_edges"],
        "assignment_endpoint_edges": result["assignment_endpoint_edges"],
        "existential_endpoint_edges": result["existential_endpoint_edges"],
        "candidate_edge_gap": result["candidate_edge_gap"],
        "lost_assignment_edges": result["lost_assignment_edges"],
        "no_candidate_edges": result["no_candidate_edges"],
        "assignment_endpoint_recall": result["assignment_endpoint_recall"],
        "existential_endpoint_recall": result["existential_endpoint_recall"],
    }
    for name in ("baseline", "candidate_oracle", "endpoint_oracle"):
        for key in ("edge_tp", "edge_fp", "edge_fn", "num_pred_nodes", "total_node_ratio", "edge_jaccard", "adj_edge_jaccard", "node_recall"):
            row[f"{name}_{key}"] = result[name][key]
    return row


def summary(rows: list[dict]) -> dict:
    by_fold: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_fold[row["fold"]].append(row)
    output = {}
    for fold, group in sorted(by_fold.items()):
        output[fold] = {name: summarise([r[name] for r in group]) for name in ("baseline", "candidate_oracle", "endpoint_oracle")}
        total_gt = sum(r["gt_edges"] for r in group)
        for key in ("candidate_tp_edges", "assignment_endpoint_edges", "existential_endpoint_edges", "no_candidate_edges", "lost_assignment_edges", "candidate_edge_gap"):
            output[fold][key] = sum(r[key] for r in group)
        output[fold]["gt_edges"] = total_gt
    return output


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pred-dir", action="append", required=True, type=Path, help="repeatable OOF GEFF root")
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--candidate-dir", type=Path, help="optional <crop>.csv pre-solver candidate edges")
    ap.add_argument("--fold", action="append", choices=["44b6", "6bba"], help="default requires both folds")
    ap.add_argument("--out-csv", type=Path)
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args()
    folds = set(args.fold or ["44b6", "6bba"])
    preds = discover(args.pred_dir, folds)
    results, rows = [], []
    for i, (stem, pred_path) in enumerate(sorted(preds.items()), 1):
        gt_path = args.gt_dir / f"{stem}.geff"
        if not gt_path.exists():
            raise FileNotFoundError(f"missing GT: {gt_path}")
        pred, gt = load_sample(pred_path), load_sample(gt_path)
        candidates, source = read_candidate_edges(args.candidate_dir, stem, pred.edges)
        result = measure_oracles(pred, gt, estimated_nodes(gt_path), candidates)
        results.append({"crop": stem, "fold": stem.split("_", 1)[0], **result})
        rows.append(flat_row(stem, source, result))
        print(
            f"[{i:3d}/{len(preds)}] {stem}: base={result['baseline']['adj_edge_jaccard']:.4f} "
            f"candidate={result['candidate_oracle']['adj_edge_jaccard']:.4f} "
            f"endpoint={result['endpoint_oracle']['adj_edge_jaccard']:.4f} "
            f"existential={result['existential_endpoint_recall']:.4f}"
        )
    aggregate = summary(results)
    print("\n=== OOF edge ceilings (competition-weighted within fold) ===")
    for fold, s in aggregate.items():
        print(
            f"{fold}: baseline={s['baseline']['adj_edge_jaccard']:.4f} "
            f"candidate={s['candidate_oracle']['adj_edge_jaccard']:.4f} "
            f"endpoint={s['endpoint_oracle']['adj_edge_jaccard']:.4f} "
            f"edges candidate/assignment/existential/GT="
            f"{s['candidate_tp_edges']}/{s['assignment_endpoint_edges']}/{s['existential_endpoint_edges']}/{s['gt_edges']}"
        )
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(aggregate, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
