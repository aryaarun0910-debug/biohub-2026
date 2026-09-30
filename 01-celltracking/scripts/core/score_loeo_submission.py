r"""Score a LOEO-analogue submission CSV and measure its DIVISION-REACH SUBSTRATE.

The composite is only half the answer. The blocking question for the division track is
substrate: how many ground-truth divisions are even *reachable* in a given graph, i.e.
the mother and BOTH daughters are matched to distinct predicted nodes. No division
layer, oracle or learned, can ever recover an unreachable one.

Known substrates (reachable / all GT divisions, 44b6 then 6bba):

    E0c        20/26   93/125     -> division layer worth +0.0641 / +0.0646
    clean903   20/26   96/125     -> best measured
    v122       15/26   68/125     -> +0.0419 / +0.0461
    P0-A       ?/26    ?/125      <- this script

Reach uses exactly the definition in scripts/win_bet/phaseb_d0p_proposer.py: the GT
mother and both GT children must each carry a MATCHED_NODE_ID from the metric's own
distance matching, and the two children must map to DIFFERENT predicted nodes.
The matching is the authoritative one -- ``tracking_cellmot.metrics.evaluate`` writes it
onto the predicted graph as a side effect -- so reach and the composite come from a
single pass and cannot disagree.

Usage
-----
  # smoke (<=3 crops, cheap)
  .\.venv\Scripts\python.exe scripts\core\score_loeo_submission.py ^
      --csv <loeo_split0_strict.csv.gz> --gt-dir data\train --max-crops 3

  # full fold (HEAVY CPU -- needs a human go-ahead)
  .\.venv\Scripts\python.exe scripts\core\score_loeo_submission.py ^
      --csv <loeo_split0_strict.csv.gz> --gt-dir data\train ^
      --json-out research\06-knowledge-system\inventory\loeo_f0_strict.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / 'pyproject.toml').exists())
sys.path.insert(0, str(ROOT / "src"))

import tracksdata as td  # noqa: E402

from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph  # noqa: E402
from biotrack.metric import estimated_nodes  # noqa: E402
from biotrack.submission import read_submission, submission_to_graphs  # noqa: E402
from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise  # noqa: E402

NID = td.DEFAULT_ATTR_KEYS.NODE_ID
MID = td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID


def gt_divisions(gt) -> dict[int, tuple[int, int]]:
    """{mother_gt_id: (child_gt_id, child_gt_id)} for every out-degree-2 GT node."""
    ids = [int(n) for n in gt.node_ids()]
    outdeg = dict(zip(ids, gt.out_degree(ids)))
    out: dict[int, tuple[int, int]] = {}
    for n in ids:
        if outdeg[n] >= 2:
            ch = [int(c) for c in gt.successors(int(n))][:2]
            if len(ch) == 2:
                out[n] = (ch[0], ch[1])
    return out


def gt_to_pred_map(pred) -> dict[int, int]:
    """Invert the metric's own pred->GT matching. Collisions keep the lowest pred id."""
    na = pred.node_attrs(attr_keys=[NID, MID])
    inv: dict[int, int] = {}
    for row in na.iter_rows(named=True):
        g = row[MID]
        if g is None or int(g) == -1:
            continue
        p = int(row[NID])
        g = int(g)
        if g not in inv or p < inv[g]:
            inv[g] = p
    return inv


def reach_for_crop(pred, gt) -> dict:
    div = gt_divisions(gt)
    g2p = gt_to_pred_map(pred)
    reach = 0
    unreachable_cause = defaultdict(int)
    for mother, (c1, c2) in div.items():
        pm, p1, p2 = g2p.get(mother), g2p.get(c1), g2p.get(c2)
        if pm is not None and p1 is not None and p2 is not None and p1 != p2:
            reach += 1
        elif pm is None:
            unreachable_cause["mother_unmatched"] += 1
        elif p1 is None or p2 is None:
            unreachable_cause["daughter_unmatched"] += 1
        else:
            unreachable_cause["daughters_collide"] += 1
    return {"gt_divisions": len(div), "reachable": reach, **dict(unreachable_cause)}


def open_csv(path: Path) -> Path:
    """Transparently decompress a .csv.gz into a temp file the reader can take."""
    if path.suffix != ".gz":
        return path
    tmp = Path(tempfile.mkdtemp(prefix="loeo_")) / path.name[:-3]
    with gzip.open(path, "rb") as fin, tmp.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    return tmp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True, help="submission.csv or .csv.gz from the LOEO kernel")
    ap.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
    ap.add_argument("--max-crops", type=int, help="smoke limit")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    csv = open_csv(Path(args.csv))
    gt_dir = Path(args.gt_dir)
    graphs = submission_to_graphs(read_submission(csv))
    names = sorted(graphs)
    if args.max_crops:
        names = names[: args.max_crops]
    print(f"{len(names)} crops from {args.csv}")

    rows, reach_rows = [], []
    for name in names:
        gt_geff = gt_dir / f"{name}.geff"
        if not gt_geff.exists():
            print(f"  skip {name}: no GT")
            continue
        pred = graphs[name]
        gt = load_graph(gt_geff)
        # ONE matching pass drives both the composite and the reach census.
        er = evaluate(pred, gt, scale=DEFAULT_SCALE, max_distance=MAX_DISTANCE)
        recall = node_recall(pred, gt) if pred.num_edges() and pred.num_nodes() else 0.0
        row = {"dataset": name, **per_sample_metrics(er, estimated_nodes(gt_geff), recall)}
        rows.append(row)
        rc = {"dataset": name, "family": name.split("_")[0], **reach_for_crop(pred, gt)}
        reach_rows.append(rc)
        print(f"  {name:<16} nodes={row['num_pred_nodes']:>6} "
              f"adjJ={row['adj_edge_jaccard']:.4f} recall={row['node_recall']:.3f} "
              f"divTP/FP/FN={row['division_tp']}/{row['division_fp']}/{row['division_fn']} "
              f"reach={rc['reachable']}/{rc['gt_divisions']}")

    if not rows:
        sys.exit("nothing scored")

    s = summarise(rows)
    fam_reach: dict[str, dict[str, int]] = defaultdict(lambda: {"reachable": 0, "gt_divisions": 0})
    for rc in reach_rows:
        fam_reach[rc["family"]]["reachable"] += rc["reachable"]
        fam_reach[rc["family"]]["gt_divisions"] += rc["gt_divisions"]

    print(f"\n===== LOEO composite ({len(rows)} crops) =====")
    print(f"  adj_edge_jaccard (weighted) = {s['adj_edge_jaccard']:.4f}")
    print(f"  division_jaccard            = {s['division_jaccard']:.4f} "
          f"(TP={s['division_tp']} FP={s['division_fp']} FN={s['division_fn']})")
    print(f"  node_recall                 = {s['node_recall']:.4f}")
    print(f"  SCORE                       = {s['score']:.4f}")
    print("\n===== SUBSTRATE: reachable GT divisions =====")
    for fam, v in sorted(fam_reach.items()):
        frac = v["reachable"] / v["gt_divisions"] if v["gt_divisions"] else float("nan")
        print(f"  {fam}: {v['reachable']}/{v['gt_divisions']}  ({frac:.3f})"
              "   <- compare E0c 20/26, 93/125 | clean903 20/26, 96/125 | v122 15/26, 68/125")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "csv": str(args.csv), "n_crops": len(rows), "summary": s,
            "family_reach": dict(fam_reach), "per_crop": rows, "per_crop_reach": reach_rows,
        }, indent=2, default=float))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
