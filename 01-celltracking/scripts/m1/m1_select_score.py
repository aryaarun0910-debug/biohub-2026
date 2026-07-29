"""M1 checkpoint selection: local cache -> E0c wrapper -> patched scorer, then select ONE.

Selection semantics are FROZEN here, before any candidate is scored:

  * choose the checkpoint maximising the exact aggregate patched composite over the 12-crop
    44b6 inner validation, using the preregistered edge-volume weighting
    (w_i = edge_tp + edge_fp + edge_fn), i.e. tracking_cellmot.metrics.summarise;
  * exact ties resolve to the EARLIER epoch;
  * regime slices are DIAGNOSTIC ONLY and can never act as a tie-break or override.

The held-out 6bba family is never read here. It is evaluated exactly once, afterwards,
using the single selected checkpoint.

`--parity` runs the path end-to-end on the parity-proven oof_clean graph and requires it to
reproduce E0c's recorded per-crop score, so the local scoring path is validated before the
selection GPU block is spent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))
sys.path.insert(0, str(ROOT / "scripts" / "m1"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

CANDIDATE_EPOCHS = [10, 15, 20, 25, 30]
DET = 0.990
MANIFEST_SHA = "ea5fe9b2eb9bd0fee8df513512c93b38ec9268a8da1c6fece57f2dbd5f0324f0"
# Parity reference: E0c's own recorded per-crop composite (reports/inventory/e0c_score_test).
PARITY_CROP = "44b6_0113de3b"
PARITY_EXPECT = 0.8132


def _e0c_wrapper():
    import coupled_arms as CA
    CA.set_e0c_wrapper()


def graph_from_npz(path: Path):
    from coupled_replay import graph_from_cached_detections
    return graph_from_cached_detections(path)


def score_graph(nbi, raw, crop: str) -> dict:
    """E0c wrapper -> patched authoritative scorer for one crop."""
    import copy

    from biotrack import wrapper as W
    from biotrack.metric import estimated_nodes, per_sample_metrics, score_pred_graph
    from biotrack.submission import submission_to_graphs
    from e0c_run import graph_rows
    from tracking_cellmot.metrics import EvaluationResult

    _e0c_wrapper()
    fn, fe, _ = W.filter_output_graph(copy.deepcopy(nbi), list(raw), dataset=crop)
    gdf = (pl.DataFrame(graph_rows(fn, fe))
           .with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))
    graph = submission_to_graphs(gdf)[crop]
    gt = str(ROOT / "data" / "train" / f"{crop}.geff")
    r = score_pred_graph(graph, gt)
    er = EvaluationResult(r["edge_tp"], r["edge_fp"], r["edge_fn"],
                          r["division_tp"], r["division_fp"], r["division_fn"],
                          r["num_pred_nodes"])
    return per_sample_metrics(er, estimated_nodes(gt), r["node_recall"])


def parity_check() -> bool:
    """Validate the local path on the parity-proven oof_clean graph before spending GPU."""
    from biotrack import wrapper as W
    from e0c_run import build_nodes_edges
    from tracking_cellmot.metrics import summarise

    src = ROOT / "artifacts/kaggle/oof_clean/pred_geffs_split_0" / f"{PARITY_CROP}.geff"
    g = W.graph_from_geff(src)
    nbi, raw = build_nodes_edges(g)
    m = score_graph(nbi, raw, PARITY_CROP)
    got = summarise([m])["score"]
    ok = abs(got - PARITY_EXPECT) < 5e-4
    print(f"PARITY {PARITY_CROP}: got {got:.4f} expect {PARITY_EXPECT:.4f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=ROOT / "artifacts/kaggle/m1_select")
    ap.add_argument("--parity", action="store_true", help="run parity check only")
    a = ap.parse_args()

    if a.parity:
        raise SystemExit(0 if parity_check() else 1)
    if not parity_check():
        raise SystemExit("ABORT -- local scoring path failed parity; do not select")

    man = json.loads((ROOT / "scripts/m1/val_manifests.json").read_text())
    if hashlib.sha256((ROOT / "scripts/m1/val_manifests.json").read_bytes()).hexdigest() != MANIFEST_SHA:
        raise SystemExit("ABORT -- validation manifest SHA mismatch")
    val_crops = sorted(man["directions"]["1"]["inner_val_crops"])
    if any(c.startswith("6bba") for c in val_crops):
        raise SystemExit("ABORT -- held-out family in selection set")

    rows, missing = {}, []
    for ep in CANDIDATE_EPOCHS:
        d = a.cache / f"epoch{ep:02d}"
        per = []
        for crop in val_crops:
            p = d / f"{crop}__det-{DET:g}.npz"
            if not p.exists():
                missing.append(str(p))
                continue
            from e0c_run import build_nodes_edges
            nbi, raw = build_nodes_edges(graph_from_npz(p))
            per.append(score_graph(nbi, raw, crop))
        rows[ep] = per
    if missing:
        raise SystemExit(f"ABORT -- incomplete grid, {len(missing)} caches missing: {missing[:4]}")
    for ep, per in rows.items():
        if len(per) != len(val_crops):
            raise SystemExit(f"ABORT -- epoch {ep} scored {len(per)}/{len(val_crops)} crops")

    from tracking_cellmot.metrics import summarise
    agg = {ep: summarise(per) for ep, per in rows.items()}

    print(f"\n{'epoch':>6}{'composite':>12}{'adjJ':>10}{'rawJ':>10}{'divJ':>9}{'crops':>7}")
    print("-" * 54)
    for ep in CANDIDATE_EPOCHS:
        s = agg[ep]
        dj = s["division_jaccard"]
        print(f"{ep:>6}{s['score']:>12.4f}{s['adj_edge_jaccard']:>10.4f}"
              f"{s['edge_jaccard']:>10.4f}{(dj if dj == dj else 0):>9.4f}{len(rows[ep]):>7}")

    # FROZEN RULE: max composite; exact ties -> earlier epoch (CANDIDATE_EPOCHS is ascending)
    best = max(CANDIDATE_EPOCHS, key=lambda e: (agg[e]["score"], -e))
    ties = [e for e in CANDIDATE_EPOCHS if agg[e]["score"] == agg[best]["score"]]
    if len(ties) > 1:
        best = min(ties)
        print(f"\nexact tie among {ties} -> earlier epoch {best}")
    print(f"\nSELECTED CHECKPOINT: epoch {best}  composite {agg[best]['score']:.4f}")
    print("Regime slices are diagnostic only and did not influence this choice.")

    out = ROOT / "reports/inventory/m1_selection.json"
    out.write_text(json.dumps({
        "rule": ("max aggregate patched composite, edge-volume weighted; exact ties -> "
                 "earlier epoch; regime slices diagnostic only"),
        "manifest_sha256": MANIFEST_SHA, "det_threshold": DET,
        "downstream": "E0c greedy + faithful E0c wrapper, no ILP",
        "val_crops": val_crops,
        "per_epoch": {str(e): {"composite": agg[e]["score"],
                               "adj_edge_jaccard": agg[e]["adj_edge_jaccard"],
                               "edge_jaccard": agg[e]["edge_jaccard"]}
                      for e in CANDIDATE_EPOCHS},
        "selected_epoch": best, "selected_composite": agg[best]["score"],
    }, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
