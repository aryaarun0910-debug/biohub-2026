r"""Patch candidate export into a COPY of ``predict_unet_transformer.py``.

The deployed predictor keeps only edges whose activated score exceeds its scalar
threshold, then immediately applies greedy degree limits.  That destroys the losing
source logits for every target, including cases where no source exceeds the threshold.
Those alternatives are exactly what an embedding-aware or retrained association ranker
needs.

This patch leaves graph selection untouched.  When ``H1R_EDGE_TOPK`` is positive
(default 5), it additionally writes ``<dataset>.edge_candidates.npz`` beside each GEFF.
The sidecar is the union of:

* every valid, adjacent-frame edge above ``PredictConfig.threshold``; and
* the bounded top-k raw logits for each valid target.

Each row records global source/target indices, source/target frames, raw logit,
activated probability, threshold membership, and top-k rank.  ``H1R_EDGE_TOPK=0``
disables collection and sidecar output; the original threshold/greedy/ILP graph path is
not modified.  Values above 64 fail closed to prevent an accidentally unbounded export.

Apply only to a scratch/kernel COPY of the vendored predictor.  Every source edit is an
exact-string replacement with an asserted occurrence count.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


# This source is executed here so focused unit tests exercise exactly the helper text
# injected into the vendored predictor.  Keep it self-contained: the target already
# imports os and numpy as np.
_HELPERS = r'''
_H1R_EDGE_TOPK_MAX = 64


def _h1r_edge_topk_from_env() -> int:
    try:
        value = int(os.environ.get("H1R_EDGE_TOPK", "5"))
    except ValueError as exc:
        raise ValueError("H1R_EDGE_TOPK must be an integer") from exc
    if not 0 <= value <= _H1R_EDGE_TOPK_MAX:
        raise ValueError(
            f"H1R_EDGE_TOPK must be in [0, {_H1R_EDGE_TOPK_MAX}], got {value}"
        )
    return value


_H1R_EDGE_TOPK = _h1r_edge_topk_from_env()


def _h1r_select_candidate_rows(
    probs,
    logits,
    threshold,
    topk,
    source_mask,
    target_mask,
    source_indices,
    target_indices,
    source_frame,
    target_frame,
):
    """Return threshold-union-top-k rows for one adjacent-frame pair.

    Rows are ``(source_index, target_index, source_frame, target_frame,
    edge_logit, edge_prob, above_threshold, topk_rank)``.  A rank of zero
    means the edge was retained only because it exceeded the threshold.
    """
    if not 0 <= int(topk) <= _H1R_EDGE_TOPK_MAX:
        raise ValueError(f"topk must be in [0, {_H1R_EDGE_TOPK_MAX}], got {topk}")
    if int(target_frame) != int(source_frame) + 1:
        raise ValueError(
            f"candidate export requires adjacent frames, got {source_frame}->{target_frame}"
        )
    if int(topk) == 0:
        return []

    probs = np.asarray(probs)
    logits = np.asarray(logits)
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    source_indices = np.asarray(source_indices)
    target_indices = np.asarray(target_indices)

    if probs.ndim != 2 or logits.shape != probs.shape:
        raise ValueError(
            f"probs/logits must have one identical 2-D shape, got {probs.shape}/{logits.shape}"
        )
    n_src, n_tgt = probs.shape
    if source_mask.shape != (n_src,) or target_mask.shape != (n_tgt,):
        raise ValueError(
            f"mask shapes must be {(n_src,)}/{(n_tgt,)}, got "
            f"{source_mask.shape}/{target_mask.shape}"
        )
    if source_indices.shape != (n_src,) or target_indices.shape != (n_tgt,):
        raise ValueError("global index arrays do not match the probability matrix")
    if not np.all(np.isfinite(probs)) or not np.all(np.isfinite(logits)):
        raise ValueError("candidate probabilities/logits must be finite")

    valid_sources = np.flatnonzero(source_mask)
    rows = []
    for j in np.flatnonzero(target_mask):
        # Raw-logit order is the representation-preserving order requested by H1.
        # lexsort's final key is primary; source index makes ties deterministic.
        order = valid_sources[
            np.lexsort((valid_sources, -logits[valid_sources, j]))
        ]
        top_sources = order[: min(int(topk), len(order))]
        rank_by_source = {int(i): rank + 1 for rank, i in enumerate(top_sources)}
        threshold_sources = valid_sources[probs[valid_sources, j] > float(threshold)]
        retained = sorted(set(map(int, top_sources)) | set(map(int, threshold_sources)))

        for i in retained:
            rows.append((
                int(source_indices[i]),
                int(target_indices[j]),
                int(source_frame),
                int(target_frame),
                float(logits[i, j]),
                float(probs[i, j]),
                bool(probs[i, j] > float(threshold)),
                int(rank_by_source.get(i, 0)),
            ))
    return rows


def _h1r_save_candidate_sidecar(path, rows, *, coords, topk, threshold, activation):
    """Write a self-contained columnar NPZ for future training/relink."""
    path = Path(path)
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"coords must be (N, 4) [t,z,y,x], got {coords.shape}")
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int16),
        edge_topk=np.asarray(topk, dtype=np.int16),
        threshold=np.asarray(threshold, dtype=np.float32),
        activation=np.asarray(activation),
        node_coords_tzyx=coords.astype(np.int16, copy=False),
        source_index=np.asarray([r[0] for r in rows], dtype=np.int64),
        target_index=np.asarray([r[1] for r in rows], dtype=np.int64),
        source_frame=np.asarray([r[2] for r in rows], dtype=np.int32),
        target_frame=np.asarray([r[3] for r in rows], dtype=np.int32),
        edge_logit=np.asarray([r[4] for r in rows], dtype=np.float32),
        edge_prob=np.asarray([r[5] for r in rows], dtype=np.float32),
        above_threshold=np.asarray([r[6] for r in rows], dtype=bool),
        topk_rank=np.asarray([r[7] for r in rows], dtype=np.int16),
    )
'''

exec(_HELPERS, globals())


_LEGACY_CANDIDATE_BLOCK = (
    "            candidates = sorted(\n"
    "                [\n"
    "                    (probs[i, j], i, j)\n"
    "                    for i in range(n_src)\n"
    "                    for j in range(n_tgt)\n"
    "                    if probs[i, j] > cfg.threshold\n"
    "                ],\n"
    "                reverse=True,\n"
    "            )\n"
)

_EXPORT_BLOCK = (
    "            if candidate_rows is not None:\n"
    "                candidate_rows.extend(_h1r_select_candidate_rows(\n"
    "                    probs, raw.float().cpu().numpy(), cfg.threshold, _H1R_EDGE_TOPK,\n"
    "                    p_mask_src[0].cpu().numpy(), p_mask_tgt[0].cpu().numpy(),\n"
    "                    idx_src, idx_tgt, t_src, t_tgt,\n"
    "                ))\n"
    "\n"
)


PATCHES: list[tuple[str, str, int]] = [
    (
        "from tracking_cellmot.metrics import summarise\n",
        "from tracking_cellmot.metrics import summarise\n\n" + _HELPERS,
        1,
    ),
    (
        "    unet_batch_size: int = 4,\n"
        "    downsample: tuple[int, ...] = (1, 4, 4),\n"
        ") -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:\n",
        "    unet_batch_size: int = 4,\n"
        "    downsample: tuple[int, ...] = (1, 4, 4),\n"
        "    candidate_rows: list[tuple] | None = None,\n"
        ") -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:\n",
        1,
    ),
    (
        _LEGACY_CANDIDATE_BLOCK,
        _EXPORT_BLOCK + _LEGACY_CANDIDATE_BLOCK,
        1,
    ),
    (
        "        for old in output_dir.glob(\"*.geff\"):\n"
        "            if old.is_dir():\n"
        "                shutil.rmtree(old)\n"
        "            else:\n"
        "                old.unlink()\n",
        "        for old in output_dir.glob(\"*.geff\"):\n"
        "            if old.is_dir():\n"
        "                shutil.rmtree(old)\n"
        "            else:\n"
        "                old.unlink()\n"
        "        for old in output_dir.glob(\"*.edge_candidates.npz\"):\n"
        "            old.unlink()\n",
        1,
    ),
    (
        "    for name in tqdm(test_names, desc=\"Predicting\", disable=not INTERACTIVE):\n"
        "        ds_path = data_dir / name\n"
        "        coords, edges = predict_video(\n"
        "                model, ds_path, device,\n"
        "                cfg=cfg,\n"
        "                window_size=window_size,\n"
        "                unet_batch_size=unet_batch_size,\n"
        "                downsample=downsample,\n"
        "            )\n",
        "    for name in tqdm(test_names, desc=\"Predicting\", disable=not INTERACTIVE):\n"
        "        ds_path = data_dir / name\n"
        "        candidate_rows = [] if _H1R_EDGE_TOPK > 0 else None\n"
        "        coords, edges = predict_video(\n"
        "                model, ds_path, device,\n"
        "                cfg=cfg,\n"
        "                window_size=window_size,\n"
        "                unet_batch_size=unet_batch_size,\n"
        "                downsample=downsample,\n"
        "                candidate_rows=candidate_rows,\n"
        "            )\n",
        1,
    ),
    (
        "        save_graph(graph, output_dir / f\"{name}.geff\")\n",
        "        save_graph(graph, output_dir / f\"{name}.geff\")\n"
        "        if candidate_rows is not None:\n"
        "            _h1r_save_candidate_sidecar(\n"
        "                output_dir / f\"{name}.edge_candidates.npz\", candidate_rows,\n"
        "                coords=coords, topk=_H1R_EDGE_TOPK, threshold=cfg.threshold,\n"
        "                activation=cfg.edge_activation,\n"
        "            )\n",
        1,
    ),
]


def apply_h1r_candidate_export_patch(predictor_path: Path | str) -> None:
    """Patch *predictor_path* in place, asserting vendor identity and compilation."""
    predictor_path = Path(predictor_path)
    src = predictor_path.read_text(encoding="utf-8")

    if "_H1R_EDGE_TOPK_MAX = 64" in src:
        sentinels = (
            "candidate_rows: list[tuple] | None = None",
            "candidate_rows = [] if _H1R_EDGE_TOPK > 0 else None",
            "            _h1r_save_candidate_sidecar(\n",
            _LEGACY_CANDIDATE_BLOCK,
        )
        missing = [s[:80] for s in sentinels if src.count(s) != 1]
        if missing:
            raise AssertionError(
                f"h1r_candidate_export_patch: partial/corrupt prior patch in {predictor_path}: "
                f"{missing}"
            )
        print(f"h1r_candidate_export_patch: {predictor_path} already patched -- skipping")
        return

    for i, (old, new, expect) in enumerate(PATCHES):
        count = src.count(old)
        if count != expect:
            raise AssertionError(
                f"h1r_candidate_export_patch: patch {i} matched {count} times "
                f"(expected {expect}) in {predictor_path}"
            )
        src = src.replace(old, new, expect)

    compile(src, str(predictor_path), "exec")
    predictor_path.write_text(src, encoding="utf-8")
    print(
        f"h1r_candidate_export_patch: {len(PATCHES)} patches applied to {predictor_path}"
    )


if __name__ == "__main__":
    import os as _os

    if _os.environ.get("H1R_KERNEL") != "1":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--predictor", required=True,
            help="path to a COPY of predict_unet_transformer.py (patched in place)",
        )
        args = parser.parse_args()
        apply_h1r_candidate_export_patch(args.predictor)
