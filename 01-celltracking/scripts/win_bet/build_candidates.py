"""WIN_BET Milestone 0 — scale-free association features on dense lineage data.

Builds candidate frame-to-frame edges from a dense track table and computes
density/velocity-invariant features, then a within-embryo train/test GBDT AUC as
the first kill-gate: do the scale-free features separate true links from
plausible-but-wrong kNN alternatives at all? If not, the transfer thesis is dead.

Node/edge model for a `track_id,t,z,y,x,parent_track_id` table:
  node = one (track_id, t) row. Each node's true predecessor is (track_id, t-1)
  if it exists, else (parent_track_id, t-1) if parent != -1 (birth/division),
  else none (free birth). A node with two successors is a division.
Coordinates are made isotropic per source (competition scale applied elsewhere);
here Zebrahub coords are used as-is and every feature is self-normalized so the
absolute unit/anisotropy of a given embryo cancels out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

K_SPACE = 8   # spatial neighbors used to estimate local flow
K_CAND = 6    # candidate successors per source node
EPS = 1e-6


def load_tracks(path: Path, t_lo: int, t_hi: int) -> pl.DataFrame:
    df = pl.read_csv(path).filter((pl.col("t") >= t_lo) & (pl.col("t") <= t_hi))
    # stable integer node id per (track_id, t)
    return df.with_row_index("node_id")


def true_edges(df: pl.DataFrame) -> dict[int, int]:
    """Map child node_id -> parent node_id (the one true incoming link)."""
    key = {(int(r["track_id"]), int(r["t"])): int(r["node_id"]) for r in df.iter_rows(named=True)}
    parent_of: dict[int, int] = {}
    for r in df.iter_rows(named=True):
        tid, t, ptid, nid = int(r["track_id"]), int(r["t"]), int(r["parent_track_id"]), int(r["node_id"])
        pred = key.get((tid, t - 1))
        if pred is None and ptid != -1:
            pred = key.get((ptid, t - 1))
        if pred is not None:
            parent_of[nid] = pred
    return parent_of


def build_features(df: pl.DataFrame, parent_of: dict[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # successors[src] = set of true child node_ids
    succ: dict[int, set[int]] = {}
    for child, par in parent_of.items():
        succ.setdefault(par, set()).add(child)

    times = sorted(df["t"].unique().to_list())
    by_t = {t: df.filter(pl.col("t") == t) for t in times}
    feats: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    gid = 0

    for t in times[:-1]:
        src = by_t[t]
        dst = by_t.get(t + 1)
        if dst is None or dst.height == 0 or src.height == 0:
            continue
        s_ids = src["node_id"].to_numpy()
        s_xyz = src.select("x", "y", "z").to_numpy().astype(float)
        d_ids = dst["node_id"].to_numpy()
        d_xyz = dst.select("x", "y", "z").to_numpy().astype(float)

        s_tree = cKDTree(s_xyz)
        d_tree = cKDTree(d_xyz)

        # local flow per source: displacement of each source to its nearest dst,
        # then median over the source's K_SPACE spatial neighbors (robust field).
        nn_d_dist, nn_d_idx = d_tree.query(s_xyz, k=1)
        raw_flow = d_xyz[nn_d_idx] - s_xyz  # (Ns,3)
        ksp = min(K_SPACE, len(s_ids))
        _, sp_idx = s_tree.query(s_xyz, k=ksp)
        sp_idx = np.atleast_2d(sp_idx)
        local_flow = np.median(raw_flow[sp_idx], axis=1)          # (Ns,3)
        local_speed = np.median(np.linalg.norm(raw_flow[sp_idx], axis=2), axis=1)  # (Ns,)

        # candidate successors: K_CAND nearest dst per source
        kc = min(K_CAND, len(d_ids))
        cand_dist, cand_idx = d_tree.query(s_xyz, k=kc)
        cand_dist = np.atleast_2d(cand_dist)
        cand_idx = np.atleast_2d(cand_idx)
        # back-nearest: nearest src for each dst (for mutual-NN feature)
        _, back_idx = s_tree.query(d_xyz, k=1)

        gate = np.median(local_speed) * 3.0 + EPS
        for i, sid in enumerate(s_ids):
            ls = local_speed[i] + EPS
            dmean = cand_dist[i].mean() + EPS
            n_within = int(np.sum(cand_dist[i] < gate))
            true_succ = succ.get(int(sid), set())
            for r in range(kc):
                j = cand_idx[i, r]
                disp = d_xyz[j] - s_xyz[i]
                dn = float(np.linalg.norm(disp))
                flow_res = float(np.linalg.norm(disp - local_flow[i])) / ls
                cos_flow = float(
                    np.dot(disp, local_flow[i])
                    / (dn * np.linalg.norm(local_flow[i]) + EPS)
                )
                mutual = 1.0 if back_idx[j] == i else 0.0
                feats.append([
                    dn / ls,            # velocity-normalized distance
                    r / kc,             # density-invariant rank
                    dn / dmean,         # relative distance
                    flow_res,           # residual vs local flow
                    cos_flow,           # direction agreement with local flow
                    mutual,             # back-consistency
                    n_within / kc,      # local candidate density
                ])
                labels.append(1 if int(d_ids[j]) in true_succ else 0)
                groups.append(gid)
            gid += 1

    return np.asarray(feats, float), np.asarray(labels, int), np.asarray(groups, int)


FEATURE_NAMES = ["d_norm", "rank", "d_rel", "flow_res", "cos_flow", "mutual", "n_dens"]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based ROC-AUC, pure numpy."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    tie_mean = np.zeros(len(cnt))
    np.add.at(tie_mean, inv, ranks)
    ranks = (tie_mean / cnt)[inv]
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def fit_logreg(X: np.ndarray, y: np.ndarray, iters: int = 400, lr: float = 0.5, l2: float = 1e-3):
    mu, sd = X.mean(0), X.std(0) + EPS
    Xs = (X - mu) / sd
    w, b = np.zeros(X.shape[1]), 0.0
    n = len(y)
    for _ in range(iters):
        z = Xs @ w + b
        p = 1 / (1 + np.exp(-z))
        g = p - y
        w -= lr * (Xs.T @ g / n + l2 * w)
        b -= lr * g.mean()
    return w, b, mu, sd


def features_for(path: Path, t_lo: int, t_hi: int):
    df = load_tracks(path, t_lo, t_hi)
    return build_features(df, true_edges(df))


def hard_mask(y: np.ndarray, g: np.ndarray, rank_col: np.ndarray) -> np.ndarray:
    """Candidates belonging to sources whose true successor is NOT the nearest
    dst (rank 0) -- i.e. nearest-neighbour linking would be wrong for that source.
    Vectorized: per-group min rank over positive candidates."""
    G = int(g.max()) + 1 if len(g) else 0
    min_pos_rank = np.full(G, np.inf)
    pm = y == 1
    np.minimum.at(min_pos_rank, g[pm], rank_col[pm])
    hard_groups = min_pos_rank > 0.0  # sources whose positive is not rank 0
    return hard_groups[g]


def group_ranking_metrics(
    y: np.ndarray, scores: np.ndarray, groups: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    """Measure successor ranking within each source candidate set.

    Groups with no labelled successor in the pool count against candidate
    coverage and are excluded from top-1/MRR because no ranker can recover them.
    """
    selected_groups = np.unique(groups if mask is None else groups[mask])
    eligible = 0
    top1 = 0
    reciprocal_ranks: list[float] = []
    for gid in selected_groups:
        idx = np.flatnonzero(groups == gid)
        positives = y[idx] == 1
        if not positives.any():
            continue
        eligible += 1
        order = np.argsort(-scores[idx], kind="mergesort")
        first = int(np.flatnonzero(positives[order])[0])
        top1 += int(first == 0)
        reciprocal_ranks.append(1.0 / (first + 1))
    total = len(selected_groups)
    return {
        "groups": float(total),
        "candidate_coverage": eligible / max(1, total),
        "top1": top1 / max(1, eligible),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=20, help="inclusive frame-window width")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    D = Path("data/external/zebrahub")
    # cross-embryo transfer: train on 003/004/005, test on held-out 001 (very
    # different coordinate scale + density -> the real scale-free transfer test).
    W = args.window
    print("building train (ZSNS003/004/005)...")
    parts = [features_for(D / f, 200, 200 + W) for f in
             ("ZSNS003_tracks.csv", "ZSNS004_tracks.csv", "ZSNS005_tracks.csv")]
    Xtr = np.vstack([p[0] for p in parts])
    ytr = np.concatenate([p[1] for p in parts])
    print("building test (held-out ZSNS001)...")
    Xte, yte, gte = features_for(D / "ZSNS001_tracks.csv", 300, 300 + W)
    print(f"train cands {len(ytr)} (pos {int(ytr.sum())}); test cands {len(yte)} (pos {int(yte.sum())})")

    w, b, mu, sd = fit_logreg(Xtr, ytr.astype(float))
    p = 1 / (1 + np.exp(-(((Xte - mu) / sd) @ w + b)))
    print(f"\nCROSS-EMBRYO (train 003/004/005 -> test 001)")
    print(f"  full   AUC={auc(yte, p):.4f}  base-rate(pos)={yte.mean():.3f}")

    hm = hard_mask(yte, gte, Xte[:, 1])  # col 1 = rank
    frac_hard_src = len(np.unique(gte[hm])) / max(1, len(np.unique(gte)))
    print(f"  HARD   AUC={auc(yte[hm], p[hm]):.4f}  "
          f"(sources where nearest != truth: {frac_hard_src:.1%}, {int(yte[hm].sum())} hard pos)")
    # nearest-neighbour-only baseline on the hard subset (does the model beat NN?)
    nn_score = -Xte[:, 0]  # smaller d_norm = more likely; higher score
    print(f"  HARD   NN-baseline AUC={auc(yte[hm], nn_score[hm]):.4f}  (d_norm alone)")
    print(f"  HARD   model-vs-NN lift = {auc(yte[hm], p[hm]) - auc(yte[hm], nn_score[hm]):+.4f}")

    model_all = group_ranking_metrics(yte, p, gte)
    nn_all = group_ranking_metrics(yte, nn_score, gte)
    model_hard = group_ranking_metrics(yte, p, gte, hm)
    nn_hard = group_ranking_metrics(yte, nn_score, gte, hm)
    print("\nGROUPWISE LINK SELECTION (decision-relevant gate)")
    print(f"  candidate recall@{K_CAND}={model_all['candidate_coverage']:.4f}")
    print(f"  all:  model top1={model_all['top1']:.4f} MRR={model_all['mrr']:.4f}; "
          f"NN top1={nn_all['top1']:.4f} MRR={nn_all['mrr']:.4f}")
    print(f"  hard: model top1={model_hard['top1']:.4f} MRR={model_hard['mrr']:.4f}; "
          f"NN top1={nn_hard['top1']:.4f} MRR={nn_hard['mrr']:.4f}")


if __name__ == "__main__":
    main()
