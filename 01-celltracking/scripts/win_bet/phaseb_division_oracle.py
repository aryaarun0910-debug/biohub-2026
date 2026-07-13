"""Track-B gate — division oracle: is there reachable division headroom?

Read-only. For each crop: enumerate GT divisions (GT node with >=2 children), then from
the cached labeled candidate surface (which carries source_gt_id/target_gt_id = the
scorer-consistent pred->GT match) count how many GT divisions are REACHABLE — i.e. a
predicted source matches the GT parent AND >=2 of the parent's GT children appear as
candidate targets of that source (so both daughter edges could be added from the existing
candidate pool). ±1-frame tolerance is inherent in the 7um node match.

Reports per fold:
  - GT divisions, wrapper current division-TP (existing forks), reachable divisions
  - EXISTING-FORK oracle: perfectly calibrate current forks -> div-J from current TP only
  - NEW-PROPOSAL oracle (optimistic): add all reachable forks, no FP -> div-J = reach/GT
  - optimistic composite headroom = 0.1 * (oracle div-J - current div-J)  [+ small edge-TP]
GATE: proceed with the breadth division posterior only if optimistic composite headroom
>= +0.01 on BOTH folds; else kill division as main bet.
"""
from __future__ import annotations

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

LABELED = ROOT / "artifacts/kaggle/e0c_cache/candidates_labeled_v2"
# E0c authoritative current division counts (from e0c_score): fold -> (tp, fp, fn)
CURRENT_DIV = {0: (0, 93, 26), 1: (4, 582, 121)}


def cached_crops(split: int):
    import json
    return sorted(json.loads(p.read_text())["crop"]
                  for p in (ROOT / "artifacts/kaggle/e0c_cache/status").glob(f"{split}__*.json")
                  if json.loads(p.read_text()).get("status") == "ok")


def analyze(args) -> dict:
    split, crop = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from biotrack.metric import load_graph
    gt = load_graph(str(ROOT / "data" / "train" / f"{crop}.geff"))
    ids = gt.node_ids()
    outdeg = dict(zip(ids, gt.out_degree(ids)))
    gt_div = {int(n): set(map(int, gt.successors(int(n)))) for n in ids if outdeg[n] >= 2}
    if not gt_div:
        return {"split": split, "crop": crop, "gt_div": 0, "reachable": 0}
    cand = pl.read_parquet(LABELED / str(split) / f"{crop}.parquet")
    # candidate targets grouped by their matched GT source node
    tgt_by_gtsrc: dict[int, set[int]] = {}
    if "source_gt_id" in cand.columns and cand.height:
        sub = cand.filter(pl.col("source_gt_id").is_not_null() & pl.col("target_gt_id").is_not_null())
        for sg, tg in zip(sub["source_gt_id"].to_list(), sub["target_gt_id"].to_list()):
            tgt_by_gtsrc.setdefault(int(sg), set()).add(int(tg))
    reachable = sum(1 for g, children in gt_div.items()
                    if len(children & tgt_by_gtsrc.get(g, set())) >= 2)
    return {"split": split, "crop": crop, "gt_div": len(gt_div), "reachable": reachable}


def main() -> None:
    tasks = [(s, c) for s in (0, 1) for c in cached_crops(s)]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(analyze, tasks))
    df = pl.DataFrame(rows)
    print(f"{'fold':6} {'GT_div':>7} {'cur_TP':>7} {'reach':>7} {'reach%':>7} "
          f"{'cur_divJ':>9} {'oracle_divJ':>12} {'d_composite':>11}")
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        d = df.filter(pl.col("split") == fold)
        gt_div = int(d["gt_div"].sum())
        reach = int(d["reachable"].sum())
        tp, fp, fn = CURRENT_DIV[fold]
        cur_divJ = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        # optimistic new-proposal oracle: add all reachable forks, drop all current FP forks
        oracle_divJ = reach / gt_div if gt_div else 0.0
        dcomposite = 0.1 * (oracle_divJ - cur_divJ)
        print(f"{fam:6} {gt_div:>7} {tp:>7} {reach:>7} {100*reach/max(1,gt_div):>6.1f}% "
              f"{cur_divJ:>9.4f} {oracle_divJ:>12.4f} {dcomposite:>+11.4f}")
    print("\nGATE (Track B divisions): proceed only if d_composite >= +0.01 on BOTH folds.")


if __name__ == "__main__":
    main()
