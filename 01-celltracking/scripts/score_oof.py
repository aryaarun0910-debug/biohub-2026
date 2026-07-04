"""Score downloaded predicted geffs against GT with the hardened biotrack.metric = the authoritative OOF.

Run AFTER downloading the predict-score kernel's exported geffs (pred_geffs_split_{FOLD}/). Uses our
exact, CPU/GPU-agnostic scorer (numpy<->authoritative parity verified) rather than the pack's --evaluate.
Empty predicted graphs are charged as all-FN (avoids the empty-geff KeyError in division matching).

Usage:
  .venv/Scripts/python.exe scripts/score_oof.py --pred-dir <dir of *.geff> --gt-dir data/train
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biotrack.metric import estimated_nodes, load_graph, score_pred_graph  # noqa: E402
from biotrack.metric import _empty_graph  # noqa: E402
from tracking_cellmot.metrics import summarise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="dir with predicted <crop>.geff")
    ap.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
    args = ap.parse_args()

    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    preds = sorted(pred_dir.glob("*.geff"))
    if not preds:
        sys.exit(f"No *.geff in {pred_dir}")

    rows = []
    for pg in preds:
        name = pg.stem
        gt_geff = gt_dir / f"{name}.geff"
        if not gt_geff.exists():
            print(f"  skip {name}: no GT"); continue
        g = load_graph(str(pg))
        # guard empty predictions (0 nodes -> all-FN; avoids the division-match KeyError on empty geffs)
        pred = g if g.num_nodes() > 0 else _empty_graph()
        row = score_pred_graph(pred, str(gt_geff))
        rows.append(row)
        print(f"  {name:<16} nodes={row['num_pred_nodes']:>6} "
              f"adjJ={row['adj_edge_jaccard']:.4f} recall={row['node_recall']:.3f} "
              f"divTP/FP/FN={row['division_tp']}/{row['division_fp']}/{row['division_fn']}")

    s = summarise(rows)
    fam = preds[0].stem.split("_")[0]
    print(f"\n===== OOF on held-out {fam} ({len(rows)} crops) =====")
    print(f"  adj_edge_jaccard (weighted) = {s['adj_edge_jaccard']:.4f}")
    print(f"  division_jaccard            = {s['division_jaccard']:.4f} "
          f"(TP={s['division_tp']} FP={s['division_fp']} FN={s['division_fn']})")
    print(f"  node_recall                 = {s['node_recall']:.4f}")
    print(f"  SCORE = adj_edge_J + 0.1*div_J = {s['score']:.4f}   <- the clean OOF number")


if __name__ == "__main__":
    main()
