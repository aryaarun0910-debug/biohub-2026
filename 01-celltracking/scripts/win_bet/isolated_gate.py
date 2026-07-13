"""Isolated-miss gate — Stages 2-4: de-novo tracklets -> selection -> exact graph score.

Uses cached DAXI candidate peaks (cache_daxi_candidates.py). Stage 2 builds short de-novo
tracklets by linking peaks across consecutive frames (motion-gated, length>=3) -- the key
filter that collapses the ~40k/frame noise cloud. Stage 3 selects tracklets:
  ORACLE     : tracklets overlapping isolated missed GT nodes (perfect discrimination = CEILING)
  CONF       : tracklets above a confidence threshold (length x mean-response x motion) = a
               realistic deployable proxy (no GT used)
Stage 4 inserts selected tracklets as NEW nodes+edges into a copy of the E0c graph (count
penalty applies to every added node), and runs the authoritative scorer.

Reports per fold: E0c composite, oracle/conf composite delta, isolated recovery %, nodes
added, count delta. Gate: >=20% isolated recovery AND exact min-fold composite >=+0.005.
If even the ORACLE ceiling < +0.005, isolated detection is dead as a winning lever.
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
import polars as pl  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

CACHE = ROOT / "artifacts/kaggle/e0c_cache"
DAXI = ROOT / "artifacts/kaggle/daxi_cand"
SCALE = np.array([1.625, 0.40625, 0.40625])
LINK_GATE_UM = 6.0
MIN_LEN = 3


def isolated_misses(split, crop):
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph
    from biotrack.submission import submission_to_graphs
    gdf = (pl.read_parquet(CACHE / "graphs" / str(split) / f"{crop}.parquet")
           .with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))
    pred = submission_to_graphs(gdf)[crop]
    gt = load_graph(str(ROOT / "data" / "train" / f"{crop}.geff"))
    pred.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    na = pred.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    matched = {int(m) for m in na[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list() if m not in (None, -1)}
    gna = gt.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
    pos = {int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): (int(r["t"]), r["z"], r["y"], r["x"]) for r in gna.iter_rows(named=True)}
    iso = []
    for g in map(int, gt.node_ids()):
        if g in matched:
            continue
        pr = [int(p) for p in gt.predecessors(g)]; su = [int(s) for s in gt.successors(g)]
        if not (any(p in matched for p in pr) or any(s in matched for s in su)):
            iso.append(pos[g])
    return iso


def build_tracklets(peaks):
    """peaks: dict t -> (Nx3 voxel coords, resp). Greedy motion-gated NN linking -> tracklets."""
    times = sorted(peaks)
    nextlink = {}          # (t, i) -> (t+1, j)
    for t in times:
        if t + 1 not in peaks:
            continue
        A, _ = peaks[t]; B, rB = peaks[t + 1]
        if len(A) == 0 or len(B) == 0:
            continue
        tree = cKDTree(B * SCALE)
        d, j = tree.query(A * SCALE, k=1)
        for i in range(len(A)):
            if d[i] <= LINK_GATE_UM:
                nextlink[(t, i)] = (t + 1, int(j[i]))
    # assemble chains (only start where nothing links into a node)
    incoming = set(nextlink.values())
    tracklets = []
    for t in times:
        A, rA = peaks[t]
        for i in range(len(A)):
            if (t, i) in incoming:
                continue
            chain = [(t, i)]; cur = (t, i)
            while cur in nextlink:
                cur = nextlink[cur]; chain.append(cur)
            if len(chain) >= MIN_LEN:
                coords = np.array([peaks[tt][0][ii] for tt, ii in chain], float)
                resp = np.mean([float(peaks[tt][1][ii]) for tt, ii in chain])
                ts = [tt for tt, _ in chain]
                tracklets.append({"t": ts, "coords": coords, "resp": resp, "len": len(chain)})
    return tracklets


def score_with(nodes_df, base_edges, add_nodes, add_edges, crop):
    from biotrack.metric import score_pred_graph
    from biotrack.submission import submission_to_graphs
    nrows = nodes_df.select("row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")
    extra_n = pl.DataFrame(add_nodes) if add_nodes else nrows.head(0)
    erows = [{"row_type": "edge", "node_id": -1, "t": -1, "z": -1.0, "y": -1.0, "x": -1.0,
              "source_id": s, "target_id": t} for s, t in (base_edges + add_edges)]
    edf = pl.DataFrame(erows) if erows else nrows.head(0)
    out = pl.concat([x for x in (nrows, extra_n, edf) if x.height])
    g = submission_to_graphs(out.with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))[crop]
    return score_pred_graph(g, str(ROOT / "data" / "train" / f"{crop}.geff"))


def process(split, crop, mode):
    df = pl.read_parquet(CACHE / "graphs" / str(split) / f"{crop}.parquet")
    nodes = df.filter(pl.col("row_type") == "node")
    base_edges = [(int(r["source_id"]), int(r["target_id"])) for r in df.filter(pl.col("row_type") == "edge").iter_rows(named=True)]
    d = np.load(DAXI / f"{crop}.npz")
    peaks = {}
    for t in np.unique(d["t"]):
        m = d["t"] == t
        peaks[int(t)] = (np.stack([d["z"][m], d["y"][m], d["x"][m]], 1).astype(float), d["resp"][m])
    tracklets = build_tracklets(peaks)
    iso = isolated_misses(split, crop)
    iso_um = np.array([[z, y, x] for (_, z, y, x) in iso]) * SCALE if iso else np.zeros((0, 3))
    iso_t = np.array([t for (t, _, _, _) in iso]) if iso else np.zeros(0)

    # which tracklets cover an isolated miss (same t, <=7um)
    def covers(tr):
        for k, tt in enumerate(tr["t"]):
            c = tr["coords"][k] * SCALE
            sel = iso_t == tt
            if sel.any() and np.linalg.norm(iso_um[sel] - c, axis=1).min() <= 7.0:
                return True
        return False

    if mode == "oracle":
        chosen = [tr for tr in tracklets if covers(tr)]
    else:  # conf
        thr = np.quantile([tr["resp"] * tr["len"] for tr in tracklets], 0.98) if tracklets else 1e9
        chosen = [tr for tr in tracklets if tr["resp"] * tr["len"] >= thr]

    # recovery: isolated misses within 7um of a chosen tracklet node
    recovered = 0
    if len(iso_um):
        chosen_pts = [(tt, tr["coords"][k]) for tr in chosen for k, tt in enumerate(tr["t"])]
        for i in range(len(iso_um)):
            for tt, c in chosen_pts:
                if tt == iso_t[i] and np.linalg.norm(iso_um[i] - c * SCALE) <= 7.0:
                    recovered += 1; break

    # insert chosen tracklets as new nodes + internal edges
    nid = int(nodes["node_id"].max()) + 1 if nodes.height else 1
    add_nodes, add_edges = [], []
    for tr in chosen:
        ids = []
        for k, tt in enumerate(tr["t"]):
            z, y, x = tr["coords"][k]
            add_nodes.append({"row_type": "node", "node_id": nid, "t": int(tt), "z": float(z),
                              "y": float(y), "x": float(x), "source_id": -1, "target_id": -1})
            ids.append(nid); nid += 1
        for k in range(len(ids) - 1):
            add_edges.append((ids[k], ids[k + 1]))
    row = score_with(nodes, base_edges, add_nodes, add_edges, crop)
    return {"split": split, "crop": crop, "iso": len(iso), "recovered": recovered,
            "tracklets": len(tracklets), "chosen": len(chosen), "added_nodes": len(add_nodes),
            **{k: row[k] for k in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp",
                                   "division_fn", "node_recall", "num_pred_nodes")}}


def main():
    from biotrack.metric import estimated_nodes, per_sample_metrics
    from tracking_cellmot.metrics import EvaluationResult, summarise
    crops = {0: [], 1: []}
    for f in DAXI.glob("*.npz"):
        crop = f.stem; split = 0 if crop.startswith("44b6") else 1
        if (CACHE / "graphs" / str(split) / f"{crop}.parquet").exists():
            crops[split].append(crop)
    E0C = {0: 0.7595, 1: 0.6490}
    for mode in ("oracle", "conf"):
        print(f"\n######### ISOLATED GATE mode={mode} #########")
        for fold, fam in ((0, "44b6"), (1, "6bba")):
            if not crops[fold]:
                continue
            rows, iso, rec, add = [], 0, 0, 0
            for c in crops[fold]:
                r = process(fold, c, mode)
                er = EvaluationResult(r["edge_tp"], r["edge_fp"], r["edge_fn"], r["division_tp"],
                                      r["division_fp"], r["division_fn"], r["num_pred_nodes"])
                rows.append(per_sample_metrics(er, estimated_nodes(str(ROOT / "data/train" / f"{c}.geff")), r["node_recall"]))
                iso += r["iso"]; rec += r["recovered"]; add += r["added_nodes"]
            s = summarise(rows)
            print(f"  {fam} ({len(crops[fold])} crops): composite={s['score']:.4f} (dvs E0c {s['score']-E0C[fold]:+.4f}) "
                  f"| isolated {iso} recovered {rec} ({100*rec/max(1,iso):.1f}%) | nodes added {add}")


if __name__ == "__main__":
    main()
