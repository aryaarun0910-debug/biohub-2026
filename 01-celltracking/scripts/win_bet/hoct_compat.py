r"""PKT-0029 STEP 2: the HOCT compatibility evaluation, on CPU, from the publisher's own contract.

WHAT THIS ANSWERS, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------------
``FACT-0390`` established that the publisher's model code is on disk and that
``scripts/win_bet/hoct_head.py`` reproduces its edge and quiet logits exactly. ``FACT-0392``
established that the head cannot be run as a reranker of OUR candidate list - it was trained on a
purely geometric 15 um ball, uncapped, while our deployed rule admits at most one parent per
target by arithmetic (``FACT-0369``). So the only comparable quantity is WITHIN-TARGET RANKING on
a COMMON candidate set, and the common set must be HOCT's own graph, rebuilt from node
coordinates. Absolute probability levels are never compared: HOCT's normalisation carries an
abstain mass ours has no counterpart for.

This module therefore measures exactly one thing: on HOCT's own 15 um graph over OUR UNCHANGED
node surface, does the published head choose the true parent more often than the deployed edge
head does, on the same targets and the same candidates?

THE "GPU-BOUND" PREMISE WAS FALSE, AND THAT IS WHY THIS RUNS AT ALL
--------------------------------------------------------------------
``PKT-0029`` recorded scoring as blocked because "the 32-channel UNet node features are GPU-bound
(``cache_official_hoct_features.py`` refuses to run without CUDA)". That refusal is a bare guard -
``raise RuntimeError("CUDA is required ...")`` at the top of the publisher's ``main`` - and not a
property of the computation. Three facts, each read at source, remove it:

  1. The deployed ``predict_video`` calls ``open_dataset(..., load_image=False)``
     (``vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py:319``), so the
     ``pin_memory()`` accelerator refusal recorded against Gate 1
     (``tracking_cellmot/io.py:199``) is on a code path the feature computation never enters.
  2. That same function already selects ``cpu`` when CUDA is absent (:537).
  3. The trunk runs on a (1, 2, 64, 64, 64) window after the ``[1, 4, 4]`` downsample, which is
     seconds of CPU per frame pair, not hours.

This does NOT satisfy Gate 1, and no claim here is a Gate 1 pass. Gate 1 proves that a SERIALISED
cache reproduces the deployed head inside the deployed process; this is a local reconstruction.
What it does instead is CALIBRATE ITSELF against an independently recorded value, as AGENTS.md
requires: ``trunk-parity`` re-derives the deployed head's own candidate probabilities and
differences them against the P30 ECB sidecars. A reconstruction that cannot reproduce the deployed
export has no business scoring anything, so that check runs first and FAILS CLOSED.

BOTH TRUNKS, IN THE SAME RUN - AND THE NAME COLLISION THAT ALMOST PICKED THE WRONG ONE
---------------------------------------------------------------------------------------
``FACT-0392`` item (3): which detector produced the 32-dim features is a ``--weights`` argument,
absent from the checkpoint, so feeding official-trunk features to a head trained on
StableDet-trunk features would look exactly like a wrong contract. Both are therefore evaluated.

THREE distinct checkpoints in this tree answer to "split_0 edge predictor", and only one of them
is the OFFICIAL LOEO fold-0 detector the publisher's cache script defaults to:

  8,363,159 B  12f6881e...  the SUPPORT PACK's split_0 - the official LOEO fold-0 detector,
                            trained on 6bba, holds out 44b6, and the trunk our own deployed P30
                            fold-0 run used (its kernel log names
                            `weights/unet_transformer/split_0/edge_predictor_best.pth`)
  8,357,783 B  d3e89eb3...  OUR July out-of-fold retrain of the same split, which FAILED recipe
                            parity by -0.162 (``FACT-0339``) and is NOT the official detector
  8,357,783 B  32d80486...  the StableDet bundle's OWN fold-0 detector

The two 8,357,783-byte files have the same size and the same filename stem and are different
models. The sizes and hashes are recorded at ``scripts/kaggle_specs/p25_recipe_parity_loeo_f0.json``.
Running this on the middle one would have measured our failed retrain, not the published stack.

SCOPE, STATED RATHER THAN IMPLIED
---------------------------------
FOLD 0 ONLY. The published bundle ships no fold-1 base edge predictor, and
``fold0_edge_predictor_best.pth`` is a bare state dict with no provenance, so a fold-1 HOCT
evaluation has no clean trunk to stand on. That is a scope statement, not a result.

The fork head is never invoked (``FACT-0347``, ``FACT-0362``) and no number here may be read as
division recovery (``FACT-0371``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor/kaggle-cell-tracking/src"))
sys.path.insert(0, str(ROOT / "vendor/kaggle-cell-tracking/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hoct_head import (  # noqa: E402
    GATE_UM,
    SCALE_ZYX_UM,
    build_edge_neighborhood,
    candidate_graph,
    load_checkpoint,
    parent_probability,
)

#: Deployed candidate rule, for the parity channel only (FACT-0369).
DEPLOYED_SOURCE_SOFTMAX_THRESHOLD = 0.5


def _crop_nodes(pre: pl.DataFrame) -> dict:
    nodes = pre.filter(pl.col("row_type") == "node").sort("node_id")
    if nodes.height == 0:
        raise RuntimeError("no node rows for this crop - refusing to score an empty surface")
    node_id = nodes["node_id"].to_numpy().astype(np.int64)
    if not np.array_equal(node_id, np.arange(len(node_id))):
        raise RuntimeError(
            "node ids are not a contiguous 0..N-1 range; the export's ids are positional indices "
            "into coords_so_far and every index below assumes it",
        )
    return {
        "node_id": node_id,
        "t": nodes["t"].to_numpy().astype(np.int64),
        "zyx": nodes.select(["z", "y", "x"]).to_numpy().astype(np.float64),
    }


def run_crop(
    crop: str,
    trunk: Path,
    hoct_checkpoint: Path,
    pre: pl.DataFrame,
    data_dir: Path,
    ecb: Path | None,
    max_frames: int | None,
    zero_node_features: bool = False,
) -> dict:
    """One crop, one trunk: HOCT and deployed scores on HOCT's graph, plus the parity channel.

    ``zero_node_features`` runs the head on GEOMETRY ALONE - the 32 appearance channels replaced
    by zeros, no trunk and no image touched. It is the ablation that separates "this head
    discriminates weakly on our substrate" from "this head is not reading our appearance features
    at all", and those two have completely different consequences for the lever.
    """
    import predict_unet_transformer as P
    from tracking_cellmot.io import open_dataset
    import zarr

    device = torch.device("cpu")
    if torch.cuda.is_available():   # binding constraint of this packet, asserted not assumed
        raise RuntimeError("this instrument is CPU-only by packet constraint; CUDA is visible")
    model, window_size, downsample = P.load_model(trunk, device)
    head, head_meta = load_checkpoint(hoct_checkpoint)
    gate_um = float(head_meta.get("gate_um", GATE_UM))

    node = _crop_nodes(pre)
    ds_arr = np.asarray(downsample, dtype=np.float64)
    grid = node["zyx"] / ds_arr
    if not np.allclose(grid, np.rint(grid)):
        raise RuntimeError(
            "node coordinates are off the detector grid after dividing by the downsample; "
            "features cannot be indexed at them",
        )
    grid = np.rint(grid).astype(np.float32)
    node_um = (node["zyx"] * SCALE_ZYX_UM).astype(np.float32)

    ds = open_dataset(data_dir / crop, normalize=False, load_image=False, downsample=downsample)
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise RuntimeError(f"{crop}: zarr attrs carry no image_statistics.quantiles")
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low, q_high = float(ds.quantiles["0.001"]), float(ds.quantiles["0.999"])
    n_frames = ds.image_shape[0] if max_frames is None else min(max_frames, ds.image_shape[0])
    target_shape = list(ds.image_shape[1:])
    window_shape = (window_size,) + tuple(ds.image_shape[1:])

    recorded_by_frame: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if ecb is not None and ecb.exists():
        with np.load(ecb, allow_pickle=False) as z:
            rec_s = z["source_id"].astype(np.int64)
            rec_t = z["target_id"].astype(np.int64)
            rec_p = z["edge_prob"].astype(np.float64)
        order = np.argsort(node["t"][rec_s], kind="stable")
        rec_s, rec_t, rec_p = rec_s[order], rec_t[order], rec_p[order]
        frame_of = node["t"][rec_s]
        for f in np.unique(frame_of):
            m = frame_of == f
            recorded_by_frame[int(f)] = (rec_s[m], rec_t[m], rec_p[m])

    rows_src, rows_tgt, rows_t = [], [], []
    hoct_logit, hoct_prob, dep_logit, dep_prob = [], [], [], []
    parity_mine, parity_recorded = [], []
    parity_rank_targets, parity_rank_agree = 0, 0
    frames_compared = 0

    by_frame = {int(t): np.nonzero(node["t"] == t)[0] for t in np.unique(node["t"])}
    for t in range(n_frames - 1):
        si, ti = by_frame.get(t), by_frame.get(t + 1)
        if si is None or ti is None or not len(si) or not len(ti):
            continue
        with torch.no_grad():
            if zero_node_features:
                f_src = torch.zeros(1, len(si), UNET_CHANNELS)
                f_tgt = torch.zeros(1, len(ti), UNET_CHANNELS)
                # The control column becomes the pure-distance ranker, which is the honest
                # reference for a geometry-only arm (FACT-0271 / FACT-0274).
                logits = -torch.cdist(
                    torch.from_numpy(node_um[si])[None], torch.from_numpy(node_um[ti])[None],
                )[0]
                probs = torch.softmax(logits, dim=0)
            else:
                imgs = torch.stack([
                    P._load_frame(zarr_arr, f, target_shape, downsample) for f in (t, t + 1)
                ])
                imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).unsqueeze(0)
                unet_out, _ = model.encode(imgs)

                c_src = torch.from_numpy(grid[si]).unsqueeze(0)
                c_tgt = torch.from_numpy(grid[ti]).unsqueeze(0)
                m_src = torch.ones(1, len(si), dtype=torch.bool)
                m_tgt = torch.ones(1, len(ti), dtype=torch.bool)
                f_src = model._index_features(unet_out[:, 0], c_src, m_src)
                f_tgt = model._index_features(unet_out[:, 1], c_tgt, m_tgt)

                # Window-relative time, exactly as the deployed path builds it.
                rel_src = np.concatenate([np.zeros((len(si), 1)), grid[si]], axis=1)
                rel_tgt = np.concatenate([np.ones((len(ti), 1)), grid[ti]], axis=1)
                p_src = torch.from_numpy(
                    P.extract_pos_features(rel_src, window_shape)).unsqueeze(0)
                p_tgt = torch.from_numpy(
                    P.extract_pos_features(rel_tgt, window_shape)).unsqueeze(0)
                logits = model.predict_edges(
                    f_src, f_tgt,
                    c_src * torch.from_numpy(ds_arr.astype(np.float32)),
                    c_tgt * torch.from_numpy(ds_arr.astype(np.float32)),
                    p_src, p_tgt, m_src, m_tgt,
                )[0]
                probs = torch.softmax(logits, dim=0)   # over the SOURCE axis (FACT-0369)

            src_um, tgt_um = node_um[si], node_um[ti]
            e_s, e_t = candidate_graph(src_um, tgt_um, gate_um)
            if not len(e_s):
                continue
            neighbors = build_edge_neighborhood(e_s, e_t, src_um, tgt_um)
            out = head(
                # float16 is the publisher's own storage dtype for these features
                # (cache_official_hoct_features.py:183); the head never saw full float32.
                source_features=f_src[0].to(torch.float16).to(torch.float32),
                target_features=f_tgt[0].to(torch.float16).to(torch.float32),
                source_um=torch.from_numpy(src_um),
                target_um=torch.from_numpy(tgt_um),
                edge_source=torch.from_numpy(e_s),
                edge_target=torch.from_numpy(e_t),
                neighbors=torch.from_numpy(neighbors),
            )
        el = out["edge_logits"].numpy()
        hp = parent_probability(el, out["quiet_logits"].numpy(), e_t)
        rows_src.append(node["node_id"][si][e_s])
        rows_tgt.append(node["node_id"][ti][e_t])
        rows_t.append(np.full(len(e_s), t, dtype=np.int64))
        hoct_logit.append(el.astype(np.float64))
        hoct_prob.append(hp.astype(np.float64))
        dep_logit.append(logits.numpy()[e_s, e_t].astype(np.float64))
        dep_prob.append(probs.numpy()[e_s, e_t].astype(np.float64))
        frames_compared += 1

        if t in recorded_by_frame:
            r_s, r_t, r_p = recorded_by_frame[t]
            local_src = np.full(len(node["node_id"]), -1, dtype=np.int64)
            local_tgt = np.full(len(node["node_id"]), -1, dtype=np.int64)
            local_src[si] = np.arange(len(si))
            local_tgt[ti] = np.arange(len(ti))
            a, b = local_src[r_s], local_tgt[r_t]
            ok = (a >= 0) & (b >= 0)
            if ok.any():
                mine_p = probs.numpy()[a[ok], b[ok]]
                parity_mine.append(mine_p)
                parity_recorded.append(r_p[ok])
                # THE PARITY THAT MATTERS FOR THIS EXPERIMENT. The measured quantity is
                # WITHIN-TARGET RANKING, so agreement of the argmax parent over each recorded
                # target's own candidates is the fidelity criterion; an absolute probability
                # delta answers a question this instrument does not ask.
                for tgt in np.unique(b[ok]):
                    m = b[ok] == tgt
                    if m.sum() > 1:
                        parity_rank_targets += 1
                        parity_rank_agree += int(
                            np.argmax(mine_p[m]) == np.argmax(r_p[ok][m]),
                        )
        del unet_out

    if frames_compared == 0:
        raise RuntimeError(f"{crop}: ZERO frame pairs compared - refusing to report a pass")
    return {
        "crop": crop,
        "frames_compared": frames_compared,
        "source": np.concatenate(rows_src),
        "target": np.concatenate(rows_tgt),
        "t": np.concatenate(rows_t),
        "hoct_logit": np.concatenate(hoct_logit),
        "hoct_prob": np.concatenate(hoct_prob),
        "deployed_logit": np.concatenate(dep_logit),
        "deployed_prob": np.concatenate(dep_prob),
        "parity_mine": (np.concatenate(parity_mine) if parity_mine
                        else np.empty(0, dtype=np.float64)),
        "parity_recorded": (np.concatenate(parity_recorded) if parity_recorded
                            else np.empty(0, dtype=np.float64)),
        "parity_rank_targets": parity_rank_targets,
        "parity_rank_agree": parity_rank_agree,
    }


def parity_summary(mine: np.ndarray, recorded: np.ndarray,
                   rank_targets: int = 0, rank_agree: int = 0) -> dict:
    """FAIL CLOSED: an empty comparison is a failure, never a silent pass.

    This is NOT Gate 1 and must never be cited as one. Gate 1 proves a SERIALISED cache
    reproduces the deployed head inside the deployed process; this is a local reconstruction of
    the same computation from the same weights, and it is here because AGENTS.md requires a
    derived quantity to be calibrated against an independently known value before it is trusted.
    The criterion is ARGMAX AGREEMENT within each recorded target, because within-target ranking
    is the only quantity this experiment measures.
    """
    if not len(mine):
        return {"pairs": 0, "passed": False,
                "reason": "no recorded pair was reproduced - the parity channel is empty"}
    delta = np.abs(mine - recorded)
    above = recorded > DEPLOYED_SOURCE_SOFTMAX_THRESHOLD
    below = ~above
    return {
        "pairs": int(len(mine)),
        "contested_targets_recorded": int(rank_targets),
        "argmax_agreement": (rank_agree / rank_targets) if rank_targets else None,
        "max_abs_delta": float(delta.max()),
        "median_abs_delta": float(np.median(delta)),
        "p99_abs_delta": float(np.percentile(delta, 99)),
        "band_a_deployed_above_0p5": {
            "n": int(above.sum()),
            "max_abs_delta": float(delta[above].max()) if above.any() else None,
        },
        "band_b_sub_threshold": {
            "n": int(below.sum()),
            "max_abs_delta": float(delta[below].max()) if below.any() else None,
        },
        "passed": bool(rank_targets > 0 and rank_agree / rank_targets >= 0.98),
        "criterion": "argmax agreement >= 0.98 on recorded contested targets",
    }


def label_contract(crop: str, gt_geff: Path, nodes: pl.DataFrame,
                   source: np.ndarray, target: np.ndarray, score: np.ndarray) -> pl.DataFrame:
    """The FROZEN label contract of ``assoc_parent_dataset.build_crop``, computed in linear time.

    WHY THIS IS NOT A SECOND CONTRACT. Every rule is imported or copied from that module and
    ``tests/test_hoct_compat.py`` asserts this function reproduces ``build_crop`` ROW FOR ROW on
    real data. The one thing that changes is complexity: ``build_crop`` decides
    ``true_parent_is_candidate`` with a scan over every candidate edge per row, which is
    quadratic. Our surface is HOCT's uncapped 15 um graph - hundreds of thousands of candidate
    edges per crop against the deployed rule's at-most-one-parent (``FACT-0369``) - so the
    quadratic form does not terminate here. A set membership decides the same predicate.
    """
    from assoc_parent_dataset import MAX_DISTANCE_UM, SCALE, match_one_to_one_pairs
    from biotrack.metric import load_graph

    nodes = nodes.sort("node_id")
    node_id = nodes["node_id"].to_numpy().astype(np.int64)
    node_t = nodes["t"].to_numpy().astype(np.int64)
    node_zyx = nodes.select(["z", "y", "x"]).to_numpy().astype(np.float64)
    pos = {int(v): i for i, v in enumerate(node_id)}

    order = np.lexsort((-score, target))
    ranks = np.empty(len(source), dtype=np.int64)
    best: dict[int, float] = {}
    cur, seen = None, 0
    for p in order:
        if target[p] != cur:
            cur, seen = target[p], 0
            best[int(cur)] = float(score[p])
        seen += 1
        ranks[p] = seen

    n_cand = np.bincount(target, minlength=int(target.max()) + 1)
    out_deg = np.bincount(source, minlength=int(source.max()) + 1)

    gt = load_graph(gt_geff)
    gtn = gt.node_attrs().to_pandas()
    gt_ids = gtn["node_id"].to_numpy().astype(np.int64)
    gt_row = {int(v): i for i, v in enumerate(gt_ids)}
    gtc = gtn[["t", "z", "y", "x"]].to_numpy().astype(np.float64)
    gte = gt.edge_attrs().to_pandas()
    predecessor = {int(t): int(s) for s, t in zip(gte["source_id"], gte["target_id"])}

    matched: dict[int, int] = {}
    for frame in np.unique(gtc[:, 0]).astype(np.int64):
        gm = np.nonzero(gtc[:, 0] == frame)[0]
        pm = np.nonzero(node_t == frame)[0]
        if not len(pm):
            continue
        for g, p, _d in match_one_to_one_pairs(
            node_zyx[pm] * SCALE, gtc[gm, 1:] * SCALE, MAX_DISTANCE_UM,
        ):
            matched[int(gm[g])] = int(node_id[pm][p])
    node_to_gt = {v: k for k, v in matched.items()}

    true_parent: dict[int, int] = {}
    for gt_row_idx, pre_id in matched.items():
        pred_gt = predecessor.get(int(gt_ids[gt_row_idx]))
        if pred_gt is None:
            continue
        pred_row = gt_row.get(pred_gt)
        if pred_row is None or pred_row not in matched:
            continue
        true_parent[pre_id] = matched[pred_row]

    pairs = set(zip(source.tolist(), target.tolist()))
    offered = {t: ((true_parent[t], t) in pairs) for t in true_parent}
    d = (node_zyx[[pos[int(t)] for t in target]] - node_zyx[[pos[int(s)] for s in source]]) * SCALE
    return pl.DataFrame({
        "crop": [crop] * len(source),
        "target": target, "source": source,
        "prob": score, "rank": ranks,
        "margin_to_best": np.asarray([best[int(t)] for t in target]) - score,
        "n_candidates": n_cand[target],
        "dist_um": np.linalg.norm(d, axis=1),
        "dz_um": d[:, 0], "dy_um": d[:, 1], "dx_um": d[:, 2],
        "src_out_degree": out_deg[source],
        "is_true_parent": np.asarray(
            [int(true_parent.get(int(t), -1) == int(s)) for s, t in zip(source, target)]),
        "target_has_true_parent": np.asarray([int(int(t) in true_parent) for t in target]),
        "true_parent_is_candidate": np.asarray(
            [int(offered.get(int(t), False)) for t in target]),
        "target_matched_gt": np.asarray([int(int(t) in node_to_gt) for t in target]),
    })


def build_table(results: list[dict], gt_dir: Path, pre: pl.DataFrame) -> pl.DataFrame:
    """One row per (target, HOCT candidate) with BOTH scores attached to the SAME pair."""
    frames: list[pl.DataFrame] = []
    for res in results:
        crop = res["crop"]
        nodes = pre.filter((pl.col("dataset") == crop) & (pl.col("row_type") == "node"))
        table = label_contract(
            crop, gt_dir / f"{crop}.geff", nodes,
            res["source"], res["target"], res["hoct_prob"],
        )
        if table.height != len(res["source"]):
            raise RuntimeError(f"{crop}: the label contract dropped rows")
        frames.append(table.with_columns(
            hoct_prob=pl.Series(res["hoct_prob"]), hoct_logit=pl.Series(res["hoct_logit"]),
            deployed_prob=pl.Series(res["deployed_prob"]),
            deployed_logit=pl.Series(res["deployed_logit"]),
        ))
    return pl.concat(frames)


def calibration(table: pl.DataFrame, score_col: str, bins: int = 10) -> dict:
    """Reliability of the score as a probability of being the true parent.

    Reported for BOTH models and never compared across them as a level: HOCT's normalisation
    carries an abstain mass ours lacks (``FACT-0392``), so only the shape is informative.
    """
    p = table[score_col].to_numpy()
    y = table["is_true_parent"].to_numpy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin": [float(edges[b]), float(edges[b + 1])], "n": int(m.sum()),
                     "mean_score": float(p[m].mean()), "empirical": float(y[m].mean())})
        ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return {"bins": rows, "ece": float(ece), "base_rate": float(y.mean())}


def deployed_contested_targets(crops: list[str], ecb_dir: Path | None, pre: pl.DataFrame,
                               floor: float = 0.1, rank_cap: int = 4) -> set[tuple[str, int]]:
    """Targets the DEPLOYED pipeline's own widened surface finds contested.

    WHY STRATIFY, AND WHY THIS STRATUM SPECIFICALLY. HOCT's 15 um ball is geometric, so it
    offers a median of seven parents for every target including the trivial ones, where the true
    parent sits ~2.3 um away and the nearest competitor ~11 um. Pooled top-1 on that surface is
    dominated by decisions no model can get wrong. The population the campaign's own numbers are
    measured on is different: ``FACT-0381``'s 4,157 contested fold-0 targets and ``FACT-0386``'s
    kill are both defined by OUR candidate surface being contested at the P30 floor. Reporting
    that stratum as well is what makes this comparable to those numbers rather than to a
    different, easier task. It is a stratification of the SAME ranking, on the SAME HOCT
    candidates - only the target population changes.
    """
    hard: set[tuple[str, int]] = set()
    if ecb_dir is None:
        return hard
    for crop in crops:
        blob = ecb_dir / f"{crop}.npz"
        if not blob.exists():
            continue
        with np.load(blob, allow_pickle=False) as z:
            src = z["source_id"].astype(np.int64)
            tgt = z["target_id"].astype(np.int64)
            prob = z["edge_prob"].astype(np.float64)
        keep = prob > floor
        src, tgt, prob = src[keep], tgt[keep], prob[keep]
        order = np.lexsort((-prob, tgt))
        ranks = np.empty(len(src), dtype=np.int64)
        cur, seen = None, 0
        for p in order:
            if tgt[p] != cur:
                cur, seen = tgt[p], 0
            seen += 1
            ranks[p] = seen
        tgt = tgt[ranks <= rank_cap]
        counts = np.bincount(tgt)
        hard.update((crop, int(t)) for t in np.nonzero(counts > 1)[0])
    return hard


def _mass_summary(table: pl.DataFrame) -> dict:
    """Per target, how much probability HOCT assigns to a real parent at all."""
    mass = (table.group_by("crop", "target").agg(pl.col("hoct_prob").sum())
            ["hoct_prob"].to_numpy())
    p10, median, p90 = np.percentile(mass, [10, 50, 90])
    return {"median": float(median), "mean": float(mass.mean()),
            "p10": float(p10), "p90": float(p90),
            "note": "1 - this is HOCT's abstain (quiet) mass, which our rule has no counterpart "
                    "for (FACT-0392); it is reported, never compared to a deployed level"}


def _binomial_p(smaller: int, discordant: int) -> float | None:
    """Two-sided exact binomial p for a McNemar test at p = 0.5."""
    if discordant == 0:
        return None
    from math import comb
    tail = sum(comb(discordant, k) for k in range(smaller + 1)) / 2 ** discordant
    return float(min(1.0, 2.0 * tail))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="score one trunk over N crops and write the raw result")
    p.add_argument("--trunk", type=Path, required=True)
    p.add_argument("--trunk-name", required=True)
    p.add_argument("--hoct-checkpoint", type=Path, required=True)
    p.add_argument("--preilp", type=Path, required=True)
    p.add_argument("--ecb-dir", type=Path)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    p.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    p.add_argument("--crops", type=int, default=4)
    p.add_argument("--crop-offset", type=int, default=0)
    p.add_argument("--crop-list", nargs="+",
                   help="explicit crops, so the selection rule is in the command, not implied")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--threads", type=int)
    p.add_argument("--cache-dir", type=Path,
                   help="per-crop npz written as each crop finishes, and reused on a rerun")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--out-table", type=Path, required=True)

    args = ap.parse_args()
    if args.command != "run":
        raise SystemExit(2)

    import assoc_parent_dataset as APD

    if args.threads:
        torch.set_num_threads(args.threads)
    pre = pl.read_parquet(args.preilp)
    available = sorted(pre["dataset"].unique().to_list())
    if args.crop_list:
        missing = [c for c in args.crop_list if c not in available]
        if missing:
            raise SystemExit(f"crops absent from the export: {missing}")
        crops = list(args.crop_list)
    else:
        crops = available[args.crop_offset: args.crop_offset + args.crops]
    if not crops:
        raise SystemExit("no crops selected")

    results, started = [], time.time()
    cache = args.cache_dir
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    for i, crop in enumerate(crops, 1):
        t0 = time.time()
        blob = (cache / f"{crop}.npz") if cache else None
        if blob is not None and blob.exists():
            with np.load(blob, allow_pickle=False) as z:
                res = {k: z[k] for k in z.files}
            res["crop"] = crop
            res["frames_compared"] = int(res["frames_compared"])
            res["parity_rank_targets"] = int(res["parity_rank_targets"])
            res["parity_rank_agree"] = int(res["parity_rank_agree"])
            print(f"  [{i}/{len(crops)}] {crop} REUSED from cache "
                  f"pairs={len(res['source']):,}", flush=True)
        else:
            res = run_crop(
                crop, args.trunk, args.hoct_checkpoint,
                pre.filter(pl.col("dataset") == crop), args.data_dir,
                (args.ecb_dir / f"{crop}.npz") if args.ecb_dir else None,
                args.max_frames,
            )
            # Written BEFORE the next crop starts, so a kill or a crash never costs a crop that
            # was already paid for in CPU. Partial evidence beats no evidence.
            if blob is not None:
                np.savez_compressed(blob, **{k: v for k, v in res.items() if k != "crop"})
            print(f"  [{i}/{len(crops)}] {crop} pairs={len(res['source']):,} "
                  f"frames={res['frames_compared']} {time.time() - t0:.0f}s", flush=True)
        results.append(res)

    parity = parity_summary(
        np.concatenate([r["parity_mine"] for r in results]),
        np.concatenate([r["parity_recorded"] for r in results]),
        sum(r["parity_rank_targets"] for r in results),
        sum(r["parity_rank_agree"] for r in results),
    )
    table = build_table(results, args.gt_dir, pre)
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(args.out_table)

    # A THIRD RANKER, AS A SANITY FLOOR RATHER THAN A CANDIDATE. FACT-0271/FACT-0274 measured
    # that a pure-distance assignment reproduces the deployed linker to within 0.001, so nearest
    # source is a known reference point on this task. Its role here is diagnostic: a learned head
    # that ranks BELOW nearest-neighbour on its own candidate graph is more likely receiving
    # out-of-distribution features than failing to discriminate, and that distinction decides
    # whether a null result is about the weights or about the harness.
    table = table.with_columns(neg_dist=-pl.col("dist_um"))
    decidable_rows = table.filter(pl.col("true_parent_is_candidate") == 1)
    if decidable_rows.height == 0:
        raise SystemExit("no decidable target on this surface - refusing to report a ranking")
    hoct = APD.evaluate(table, "hoct_prob")
    deployed = APD.evaluate(table, "deployed_prob")
    nearest = APD.evaluate(table, "neg_dist")
    nearest.pop("per_target_correct", None)
    before = deployed.get("per_target_correct", {})
    after = hoct.get("per_target_correct", {})
    conversions = None
    try:
        from assoc_report import parent_conversions
        conversions = parent_conversions(before, after)
    except Exception as exc:   # a conversion failure must be visible, never swallowed
        conversions = {"error": str(exc)}
    # McNemar on the DISCORDANT pairs. A net top-1 delta over thousands of targets that are
    # correct for both models is not a test; the discordant count is.
    contested_keys = {
        (str(c), int(t))
        for c, t in table.filter(pl.col("true_parent_is_candidate") == 1)
        .group_by("crop", "target").len().filter(pl.col("len") > 1)
        .select(["crop", "target"]).iter_rows()
    }
    b = sum(1 for k in contested_keys if before.get(k) and not after.get(k))
    c_ = sum(1 for k in contested_keys if not before.get(k) and after.get(k))
    mcnemar = {
        "contested_targets": len(contested_keys),
        "deployed_right_hoct_wrong": b,
        "hoct_right_deployed_wrong": c_,
        "discordant": b + c_,
        "exact_binomial_p_two_sided": _binomial_p(min(b, c_), b + c_),
        "direction": ("hoct_better" if c_ > b else "deployed_better" if b > c_ else "tie"),
    }
    hard = deployed_contested_targets(crops, args.ecb_dir, pre)
    hard_keys = {k for k in contested_keys if k in hard}
    hb = sum(1 for k in hard_keys if before.get(k) and not after.get(k))
    hc = sum(1 for k in hard_keys if not before.get(k) and after.get(k))
    hard_stratum = {
        "definition": "targets contested on the DEPLOYED widened P30 surface (floor 0.1, rank "
                      "cap 4) - the FACT-0381 / FACT-0386 population",
        "targets": len(hard_keys),
        "hoct_top1": (sum(1 for k in hard_keys if after.get(k)) / len(hard_keys)
                      if hard_keys else None),
        "deployed_top1": (sum(1 for k in hard_keys if before.get(k)) / len(hard_keys)
                          if hard_keys else None),
        "deployed_right_hoct_wrong": hb,
        "hoct_right_deployed_wrong": hc,
        "discordant": hb + hc,
        "exact_binomial_p_two_sided": _binomial_p(min(hb, hc), hb + hc),
        "direction": ("hoct_better" if hc > hb else "deployed_better" if hb > hc else "tie"),
    }
    per_crop = {}
    for crop in crops:
        sub = table.filter(pl.col("crop") == crop)
        if sub.height == 0:
            continue
        h, d = APD.evaluate(sub, "hoct_prob"), APD.evaluate(sub, "deployed_prob")
        per_crop[crop] = {
            "contested_n": h.get("contested", {}).get("n"),
            "hoct_contested_top1": h.get("contested", {}).get("top1"),
            "deployed_contested_top1": d.get("contested", {}).get("top1"),
        }
    for summary in (hoct, deployed):
        summary.pop("per_target_correct", None)

    report = {
        "schema_version": 1,
        "heartbeat": "HOCT_COMPAT_COMPLETE",
        "fold": 0,
        "trunk": {"name": args.trunk_name, "path": str(args.trunk)},
        "hoct_checkpoint": str(args.hoct_checkpoint),
        "crops": crops,
        "candidate_surface": {
            "rule": "HOCT geometric ball, uncapped, from node coordinates (FACT-0392)",
            "gate_um": GATE_UM, "rows": table.height,
        },
        "trunk_parity_vs_deployed_export": parity,
        "hoct": hoct,
        "deployed_on_the_same_graph": deployed,
        "nearest_source_floor": nearest,
        "mcnemar_contested": mcnemar,
        "hard_stratum_deployed_contested": hard_stratum,
        "per_crop": per_crop,
        "parent_conversions": conversions,
        # Calibration is measured on the DECIDABLE rows only. Over the whole graph the base rate
        # is set by how sparsely this competition annotates - most candidate pairs have no GT
        # cell at either end - so an ECE computed there measures the annotation density, not the
        # model.
        "calibration": {
            "surface": "rows whose target has its true parent among the candidates",
            "hoct": calibration(decidable_rows, "hoct_prob"),
            "deployed": calibration(decidable_rows, "deployed_prob"),
        },
        # FACT-0392's abstain mass, MEASURED on our substrate rather than asserted. Per target,
        # HOCT's candidate probabilities sum to 1 minus the quiet (no-parent) mass. A head fed
        # out-of-distribution features abstains on nearly everything, so this number separates
        # "declines to choose here" from "chooses badly here".
        "hoct_assigned_mass_per_target": _mass_summary(table),
        "wall_seconds": time.time() - started,
        "caveat": (
            "WITHIN-TARGET RANKING ONLY. Absolute levels are not comparable: HOCT's normalisation "
            "carries an abstain mass ours lacks (FACT-0392). The fork head was never invoked and "
            "nothing here is a division result (FACT-0347, FACT-0371)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    hc, dc = hoct.get("contested", {}), deployed.get("contested", {})
    print(
        f"\nHOCT_COMPAT_COMPLETE trunk={args.trunk_name} crops={len(crops)} rows={table.height:,}\n"
        f"  trunk parity vs deployed export  pairs={parity['pairs']:,} "
        f"argmax_agreement={parity.get('argmax_agreement')} "
        f"median|delta|={parity.get('median_abs_delta')} passed={parity['passed']}\n"
        f"  decidable targets                {hoct.get('decidable_targets', 0):,}\n"
        f"  CONTESTED targets                {hc.get('n', 0):,} ({hc.get('share', 0):.1%})\n"
        f"    HOCT      contested top-1      {hc.get('top1')}  errors {hc.get('errors')}\n"
        f"    DEPLOYED  contested top-1      {dc.get('top1')}  errors {dc.get('errors')}\n"
        f"    NEAREST   contested top-1      "
        f"{nearest.get('contested', {}).get('top1')}  (diagnostic floor)\n"
        f"  overall top-1  HOCT {hoct.get('parent_top1')}  DEPLOYED {deployed.get('parent_top1')}\n"
        f"  margin (median) HOCT {hoct.get('true_parent_margin_median')}  "
        f"DEPLOYED {deployed.get('true_parent_margin_median')}\n"
        f"  McNemar (contested)              {mcnemar}\n"
        f"  HARD stratum (deployed-contested){hard_stratum}\n"
        f"  parent conversions               {conversions}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
