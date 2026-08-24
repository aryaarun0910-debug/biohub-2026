"""Validated Zh001r identity data for association-head training.

The public ``zh001r_nodes.npz`` stores one ``(N, 4)`` array per crop-frame with
columns ``[t, z, y, x]``.  Our registration sidecar stores row-aligned track and
parent-track identities as ``tid_{crop}_{frame}`` / ``pid_{crop}_{frame}``.

This module joins those assets without guessing, validates the complete schema,
and produces dense float32 transition matrices compatible with
``train_unet_transformer.compute_batch_loss``:

* continuation: ``source.track_id == target.track_id``;
* division daughter: ``source.track_id == target.parent_track_id``.

The sidecar is derived data.  Any missing row, duplicate identity, ambiguous
two-parent target, or row-count mismatch is therefore fatal rather than silently
being treated as an unlabelled node.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch


N_FRAMES_PER_CROP = 20
EXPECTED_FULL_CROPS = 72
EXPECTED_CONTINUATION_LINKS = 1_192_441
EXPECTED_DIVISION_DAUGHTER_LINKS = 65_741

_NODE_RE = re.compile(r"^f(\d+)$")
_IDENT_RE = re.compile(r"^(tid|pid)_(\d+)_(\d+)$")


class EdgeDataError(ValueError):
    """The node pack and identity sidecar do not describe one trustworthy corpus."""


@dataclass(frozen=True)
class EdgeAssets:
    nodes: Path
    identity: Path


@dataclass(frozen=True)
class IdentityFrame:
    crop: int
    frame: int
    time: int
    coords: np.ndarray
    track_ids: np.ndarray
    parent_track_ids: np.ndarray


@dataclass(frozen=True)
class EdgePair:
    crop: int
    source_frame: int
    target_frame: int
    source_coords: np.ndarray
    target_coords: np.ndarray
    target: torch.Tensor
    continuation_links: int
    division_daughter_links: int


def _unique_named(search_roots: Iterable[str | Path], name: str) -> Path:
    hits: set[Path] = set()
    for raw_root in search_roots:
        root = Path(raw_root)
        if root.is_file():
            if root.name == name:
                hits.add(root.resolve())
            continue
        direct = root / name
        if direct.is_file():
            hits.add(direct.resolve())
        if root.is_dir():
            hits.update(p.resolve() for p in root.rglob(name) if p.is_file())
    if not hits:
        raise FileNotFoundError(f"could not find {name!r} under: "
                                + ", ".join(map(str, search_roots)))
    if len(hits) != 1:
        raise EdgeDataError(f"ambiguous {name!r}; found {len(hits)} copies: "
                            + ", ".join(map(str, sorted(hits))))
    return next(iter(hits))


def discover_edge_assets(
    search_roots: Iterable[str | Path],
    *,
    nodes_name: str = "zh001r_nodes.npz",
    identity_name: str = "zh001r_identity.npz",
) -> EdgeAssets:
    """Discover exactly one node pack and one identity sidecar.

    The two files may live in different Kaggle input mounts.  Ambiguity is an
    error because silently choosing an older sidecar would poison every target.
    """
    roots = tuple(search_roots)
    if not roots:
        raise ValueError("at least one search root is required")
    return EdgeAssets(
        nodes=_unique_named(roots, nodes_name),
        identity=_unique_named(roots, identity_name),
    )


def load_edge_data(
    search_roots: Iterable[str | Path],
    *,
    n_frames: int = N_FRAMES_PER_CROP,
    assert_published_totals: bool = True,
) -> "Zh001rEdgeData":
    """Discover both assets and return one validated edge-data corpus."""
    assets = discover_edge_assets(search_roots)
    return Zh001rEdgeData(
        assets.nodes,
        assets.identity,
        n_frames=n_frames,
        assert_published_totals=assert_published_totals,
    )


def _as_int_ids(value: np.ndarray, *, key: str, allow_negative: bool) -> np.ndarray:
    a = np.asarray(value)
    if a.ndim != 1:
        raise EdgeDataError(f"{key}: expected a 1-D identity array, got {a.shape}")
    if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise EdgeDataError(f"{key}: identities must be finite numbers")
    rounded = np.rint(a)
    if not np.array_equal(a, rounded):
        raise EdgeDataError(f"{key}: identities must be integer-valued")
    out = rounded.astype(np.int64, copy=False)
    if not allow_negative and np.any(out < 0):
        bad = int(np.sum(out < 0))
        raise EdgeDataError(f"{key}: {bad} nodes have no registered track identity")
    if allow_negative and np.any(out < -1):
        raise EdgeDataError(f"{key}: parent ids may use -1 only as the no-parent sentinel")
    return out


def build_transition_target(
    source_track_ids: np.ndarray,
    target_track_ids: np.ndarray,
    target_parent_track_ids: np.ndarray,
) -> tuple[torch.Tensor, int, int]:
    """Build one adjacent-frame target matrix and return its link-type counts."""
    rows, cols, n_continuation, n_division, shape = _transition_indices(
        source_track_ids, target_track_ids, target_parent_track_ids
    )
    target = torch.zeros(shape, dtype=torch.float32)
    if rows:
        target[rows, cols] = 1.0
    return target, n_continuation, n_division


def _transition_indices(
    source_track_ids: np.ndarray,
    target_track_ids: np.ndarray,
    target_parent_track_ids: np.ndarray,
) -> tuple[list[int], list[int], int, int, tuple[int, int]]:
    """Validate a pair and return sparse positive indices without allocating N x M."""
    src = _as_int_ids(source_track_ids, key="source track ids", allow_negative=False)
    tgt = _as_int_ids(target_track_ids, key="target track ids", allow_negative=False)
    pid = _as_int_ids(target_parent_track_ids, key="target parent ids", allow_negative=True)
    if len(tgt) != len(pid):
        raise EdgeDataError(
            f"target identity length mismatch: tid={len(tgt)} pid={len(pid)}"
        )
    if len(np.unique(src)) != len(src):
        raise EdgeDataError("duplicate track_id in source frame")
    if len(np.unique(tgt)) != len(tgt):
        raise EdgeDataError("duplicate track_id in target frame")

    source_row = {int(track_id): i for i, track_id in enumerate(src)}
    rows: list[int] = []
    cols: list[int] = []
    n_continuation = 0
    n_division = 0
    for j, (track_id, parent_id) in enumerate(zip(tgt, pid, strict=True)):
        cont_i = source_row.get(int(track_id))
        div_i = source_row.get(int(parent_id)) if parent_id >= 0 else None
        if cont_i is not None and div_i is not None:
            raise EdgeDataError(
                "target node has both a continuation parent and a division parent in "
                f"the source frame (track_id={int(track_id)}, parent_track_id={int(parent_id)})"
            )
        if cont_i is not None:
            rows.append(cont_i)
            cols.append(j)
            n_continuation += 1
        elif div_i is not None:
            rows.append(div_i)
            cols.append(j)
            n_division += 1
    return rows, cols, n_continuation, n_division, (len(src), len(tgt))


class Zh001rEdgeData:
    """Eagerly validated, row-aligned Zh001r node identities."""

    def __init__(
        self,
        nodes_path: str | Path,
        identity_path: str | Path,
        *,
        n_frames: int = N_FRAMES_PER_CROP,
        assert_published_totals: bool = True,
    ) -> None:
        self.nodes_path = Path(nodes_path)
        self.identity_path = Path(identity_path)
        self.n_frames = int(n_frames)
        if self.n_frames < 2:
            raise ValueError("n_frames must be at least 2")
        if not self.nodes_path.is_file():
            raise FileNotFoundError(self.nodes_path)
        if not self.identity_path.is_file():
            raise FileNotFoundError(self.identity_path)

        with np.load(self.nodes_path, allow_pickle=False) as node_pack, \
                np.load(self.identity_path, allow_pickle=False) as identity_pack:
            self._frames = self._load_frames(node_pack, identity_pack)

        crops = sorted({crop for crop, _ in self._frames})
        expected_crops = list(range(len(crops)))
        if crops != expected_crops:
            raise EdgeDataError(
                f"crop ids must be contiguous from zero; found {crops[:8]}"
            )
        self.crops = tuple(crops)
        self.n_crops = len(crops)
        self.counts = self.link_counts()
        is_full = (
            self.n_frames == N_FRAMES_PER_CROP
            and self.n_crops == EXPECTED_FULL_CROPS
            and len(self._frames) == EXPECTED_FULL_CROPS * N_FRAMES_PER_CROP
        )
        if assert_published_totals and is_full:
            observed = (
                self.counts["continuation_links"],
                self.counts["division_daughter_links"],
            )
            expected = (
                EXPECTED_CONTINUATION_LINKS,
                EXPECTED_DIVISION_DAUGHTER_LINKS,
            )
            if observed != expected:
                raise EdgeDataError(
                    "full Zh001r association totals disagree with the registered authority: "
                    f"observed continuation/division={observed}, expected={expected}"
                )

    def _load_frames(self, node_pack, identity_pack) -> dict[tuple[int, int], IdentityFrame]:
        node_indices: dict[int, str] = {}
        for key in node_pack.files:
            match = _NODE_RE.fullmatch(key)
            if match is None:
                raise EdgeDataError(f"unexpected node-pack key {key!r}; expected f<index>")
            index = int(match.group(1))
            if index in node_indices:
                raise EdgeDataError(f"duplicate node frame index {index}")
            node_indices[index] = key
        if not node_indices:
            raise EdgeDataError("node pack is empty")
        expected_indices = set(range(max(node_indices) + 1))
        if set(node_indices) != expected_indices:
            missing = sorted(expected_indices - set(node_indices))
            raise EdgeDataError(f"node frame indices are not contiguous; missing {missing[:8]}")
        if len(node_indices) % self.n_frames:
            raise EdgeDataError(
                f"{len(node_indices)} node frames is not divisible by {self.n_frames}"
            )

        identity_keys: dict[tuple[str, int, int], str] = {}
        for key in identity_pack.files:
            match = _IDENT_RE.fullmatch(key)
            if match is None:
                raise EdgeDataError(
                    f"unexpected identity key {key!r}; expected tid_<crop>_<frame> or pid_<crop>_<frame>"
                )
            kind, crop_text, frame_text = match.groups()
            ident = (kind, int(crop_text), int(frame_text))
            if ident in identity_keys:
                raise EdgeDataError(f"duplicate identity key {key!r}")
            identity_keys[ident] = key

        n_crops = len(node_indices) // self.n_frames
        required = {
            (kind, crop, frame)
            for crop in range(n_crops)
            for frame in range(self.n_frames)
            for kind in ("tid", "pid")
        }
        actual = set(identity_keys)
        if actual != required:
            missing = sorted(required - actual)
            extra = sorted(actual - required)
            raise EdgeDataError(
                "identity schema does not exactly cover the node pack; "
                f"missing={missing[:6]} extra={extra[:6]}"
            )

        frames: dict[tuple[int, int], IdentityFrame] = {}
        for crop in range(n_crops):
            previous_time: int | None = None
            for frame in range(self.n_frames):
                node_key = node_indices[crop * self.n_frames + frame]
                nodes = np.asarray(node_pack[node_key])
                if nodes.ndim != 2 or nodes.shape[1] != 4:
                    raise EdgeDataError(f"{node_key}: expected (N, 4), got {nodes.shape}")
                if not np.issubdtype(nodes.dtype, np.number) or not np.isfinite(nodes).all():
                    raise EdgeDataError(f"{node_key}: node rows must be finite numbers")
                times = nodes[:, 0]
                if len(times):
                    rounded_times = np.rint(times)
                    if not np.array_equal(times, rounded_times) or len(np.unique(rounded_times)) != 1:
                        raise EdgeDataError(f"{node_key}: all rows must carry one integer time")
                    time = int(rounded_times[0])
                else:
                    time = frame if previous_time is None else previous_time + 1
                if previous_time is not None and time != previous_time + 1:
                    raise EdgeDataError(
                        f"crop {crop}: node times are not adjacent at frame {frame}: "
                        f"{previous_time} -> {time}"
                    )
                previous_time = time

                tid_key = identity_keys[("tid", crop, frame)]
                pid_key = identity_keys[("pid", crop, frame)]
                tids = _as_int_ids(identity_pack[tid_key], key=tid_key, allow_negative=False)
                pids = _as_int_ids(identity_pack[pid_key], key=pid_key, allow_negative=True)
                if len(nodes) != len(tids) or len(nodes) != len(pids):
                    raise EdgeDataError(
                        f"crop {crop} frame {frame}: row mismatch nodes={len(nodes)} "
                        f"tid={len(tids)} pid={len(pids)}"
                    )
                if len(np.unique(tids)) != len(tids):
                    raise EdgeDataError(f"crop {crop} frame {frame}: duplicate track_id")
                frames[(crop, frame)] = IdentityFrame(
                    crop=crop,
                    frame=frame,
                    time=time,
                    coords=np.asarray(nodes[:, 1:], dtype=np.float32),
                    track_ids=tids.copy(),
                    parent_track_ids=pids.copy(),
                )
        return frames

    def frame(self, crop: int, frame: int) -> IdentityFrame:
        try:
            return self._frames[(int(crop), int(frame))]
        except KeyError as exc:
            raise IndexError(f"unknown crop/frame ({crop}, {frame})") from exc

    def pair(self, crop: int, source_frame: int) -> EdgePair:
        if not 0 <= source_frame < self.n_frames - 1:
            raise IndexError(
                f"source_frame must be in [0, {self.n_frames - 1}), got {source_frame}"
            )
        source = self.frame(crop, source_frame)
        target_frame = self.frame(crop, source_frame + 1)
        matrix, n_cont, n_div = build_transition_target(
            source.track_ids,
            target_frame.track_ids,
            target_frame.parent_track_ids,
        )
        return EdgePair(
            crop=int(crop),
            source_frame=int(source_frame),
            target_frame=int(source_frame + 1),
            source_coords=source.coords,
            target_coords=target_frame.coords,
            target=matrix,
            continuation_links=n_cont,
            division_daughter_links=n_div,
        )

    def iter_pairs(self, crops: Iterable[int] | None = None) -> Iterator[EdgePair]:
        selected = self.crops if crops is None else tuple(int(c) for c in crops)
        for crop in selected:
            if crop not in self.crops:
                raise IndexError(f"unknown crop {crop}")
            for frame in range(self.n_frames - 1):
                yield self.pair(crop, frame)

    def link_counts(self) -> dict[str, int]:
        continuation = division = 0
        for crop in sorted({c for c, _ in self._frames}):
            for frame in range(self.n_frames - 1):
                source = self._frames[(crop, frame)]
                target = self._frames[(crop, frame + 1)]
                _, _, n_cont, n_div, _ = _transition_indices(
                    source.track_ids, target.track_ids, target.parent_track_ids
                )
                continuation += n_cont
                division += n_div
        return {
            "continuation_links": continuation,
            "division_daughter_links": division,
            "association_links": continuation + division,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", action="append", required=True,
                        help="search root; repeat when nodes and identity use separate mounts")
    parser.add_argument("--no-published-assert", action="store_true",
                        help="skip the 72-crop published-total assertion")
    args = parser.parse_args(argv)
    assets = discover_edge_assets(args.root)
    data = Zh001rEdgeData(
        assets.nodes,
        assets.identity,
        assert_published_totals=not args.no_published_assert,
    )
    report = {
        "nodes": str(assets.nodes),
        "identity": str(assets.identity),
        "crops": data.n_crops,
        "frames_per_crop": data.n_frames,
        **data.counts,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if os.environ.get("H1R_KERNEL") == "1":
    # Kaggle factory embeds this module as a notebook cell. Register that cell's
    # namespace so the following h1r_edge_train cell can use its normal import.
    sys.modules.setdefault("h1r_edge_data", sys.modules[__name__])
elif __name__ == "__main__":
    raise SystemExit(main())
