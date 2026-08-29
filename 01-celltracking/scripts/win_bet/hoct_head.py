r"""The published HOCT association head, its DETERMINED feature contract, and the gates that
prove both.

WHAT CHANGED, AND WHY THIS FILE WAS REWRITTEN (PKT-0029 STEP 1, 2026-08-29)
--------------------------------------------------------------------------
The previous version of this module opened with the claim that the publisher "shipped
CHECKPOINTS AND A CONFIG, NO MODEL CODE", and it therefore refused to implement a forward pass
at all - the 3 node extras, 4 pair extras and 13 relation features were to be recovered by
ranking candidate orderings against held-out parent choice.

**That premise is false.** `C:/temp/arch_inventory/hoct/LICENSES.md` names a second Kaggle
dataset, `rudispresence/biohub-stabledet-hoct-code`, which exists and ships the publisher's own
`repo_overlay/methods/stabledet_hoct_fork/hoct_edge_transformer.py` under the MIT licence. Its
class `StableDetHOCTFork(feature_dim=32)` strict-loads BOTH published checkpoints at exactly
447,376 parameters, so the published source IS the source that produced the weights. The feature
contract is therefore READ, not inferred, and the identifiability experiment PKT-0029 STEP 1
specified is moot - see `C:/temp/hoct/contract_manifest.json`, frozen before any scoring.

THE STRICT LOAD WAS NECESSARY AND DEMONSTRABLY NOT SUFFICIENT
-------------------------------------------------------------
The previous reconstruction strict-loaded both checkpoints and still computed the wrong
function, in three ways that change no parameter shape and so cannot be seen by `strict=True`:

    node_encoder activation   was ReLU (torch default)   publisher uses GELU
    node_encoder norm_first   was False (torch default)  publisher uses norm_first=True
    edge-block positions      absent                     publisher applies 3-D RoPE to q and k

The RoPE frequencies are a `persistent=False` buffer, so they never appear in a checkpoint at
all. This is exactly the silent-failure class PKT-0029 falsifier (b) named, arriving through the
trunk rather than through the feature slots. `equivalence` below is the gate that catches it.

THE DETERMINED CONTRACT (every slot, with the publisher site that fixes it)
--------------------------------------------------------------------------
node extras (3)    ``(coords_um - center) / gate_um`` in (z, y, x) order, where ``center`` is the
                   mean over the SOURCE AND TARGET frames POOLED, not per frame.
                   -> hoct_edge_transformer.py, StableDetHOCTFork.forward, src_input/tgt_input
                   The frame tag is ADDED (``frame_embedding.weight[0]`` / ``[1]``), not
                   concatenated, so it consumes none of the 35 input slots.

pair extras (4)    ``displacement`` (3) = ``(target_um[edge_target] - source_um[edge_source]) /
                   gate_um`` in (z, y, x), THEN ``distance`` (1) = its L2 norm. Order matters:
                   distance follows displacement.
                   -> StableDetHOCTFork.forward, edge_projection input; 96 + 96 + 3 + 1 = 196

relation (13)      ``(n_mid - q_mid)/gate``(3), ``(n_start - q_start)/gate``(3),
                   ``(n_end - q_end)/gate``(3), ``cosine``(1), ``line_distance/gate``(1),
                   ``shared_source``(1), ``shared_target``(1)
                   -> edge_relation_features

gate_um            15.0, from the checkpoint metadata, used as the ONLY length normaliser.

THE AUXILIARY HEADS ARE BUILT AND NOT CALLED
--------------------------------------------
``division_head`` and ``fork_head`` exist so the strict load covers the whole checkpoint. PKT-0029
forbids using the fork head: its published division recall is zero (``FACT-0347``) on 110 positive
training pairs (``FACT-0362``), and an association gain must never be read as division recovery
(``FACT-0371``). ``tests/test_hoct_head.py`` asserts this file never invokes it.

ATTRIBUTION
-----------
Reimplemented from ``StableDet-HOCT`` publisher code, MIT licence, Copyright (c) 2026
rudispresence (``PUBLISHER_CODE_LICENSE.txt`` in the code dataset). Checkpoints are CC BY 4.0.
This module is a reimplementation rather than a vendored copy so that the repository keeps one
audit surface; ``equivalence`` is what makes the reimplementation trustworthy.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
import torch.nn.functional as F

D_MODEL = 96
N_HEADS = 4
NODE_IN = 35
UNET_CHANNELS = 32
NODE_EXTRAS = NODE_IN - UNET_CHANNELS      # 3
EDGE_PROJ_IN = 196
PAIR_EXTRAS = EDGE_PROJ_IN - 2 * D_MODEL   # 4
RELATION_IN = 13
FORK_IN = 197
GATE_UM = 15.0
EDGE_NEIGHBORS = 64
LOCAL_MIDPOINT_NEIGHBORS = 16

# Full-resolution voxel -> micrometre scale. Identical to the publisher's SCALE_ZYX_UM
# (profile_predicted_node_supervision.py) and to this campaign's own atlas scale, which AGENTS.md
# warns is NOT isotropic.
SCALE_ZYX_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)

#: The frozen contract, as data, so an instrument can permute a named slot group rather than a
#: magic index range. Values are (start, stop) half-open slices into each feature block.
CONTRACT_SLOTS = {
    "node_extras": (UNET_CHANNELS, NODE_IN),        # 32:35
    "pair_extras": (2 * D_MODEL, EDGE_PROJ_IN),     # 192:196
    "relation": (0, RELATION_IN),                   # 0:13
}


class Rotary3D(nn.Module):
    """Independent rotary position encoding on the z, y and x channels.

    ``inv_freq`` is registered NON-persistently, exactly as the publisher does, which is why no
    checkpoint carries it and why a strict load cannot tell you this module is missing.
    """

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 6:
            raise ValueError("head_dim must be divisible by 6 for 3D RoPE")
        pairs_per_axis = head_dim // 6
        inv_freq = base ** (-torch.arange(pairs_per_axis, dtype=torch.float32) / pairs_per_axis)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.axis_dim = pairs_per_axis * 2

    @staticmethod
    def _rotate(chunk: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        even, odd = chunk[..., 0::2], chunk[..., 1::2]
        cos, sin = angle.cos(), angle.sin()
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        parts = tensor.split(self.axis_dim, dim=-1)
        rotated = []
        for axis, part in enumerate(parts):
            rotated.append(self._rotate(part, positions[..., axis, None, None] * self.inv_freq))
        return torch.cat(rotated, dim=-1)


def _point_segment_distance(
    point: torch.Tensor, start: torch.Tensor, end: torch.Tensor,
) -> torch.Tensor:
    direction = end - start
    denominator = direction.square().sum(dim=-1).clamp_min(1e-8)
    fraction = ((point - start) * direction).sum(dim=-1) / denominator
    closest = start + fraction.clamp(0.0, 1.0).unsqueeze(-1) * direction
    return torch.linalg.vector_norm(point - closest, dim=-1)


def candidate_graph(
    source_um: np.ndarray, target_um: np.ndarray, gate_um: float = GATE_UM,
) -> tuple[np.ndarray, np.ndarray]:
    """HOCT's candidate rule: EVERY source within ``gate_um`` of a target.

    This is NOT our deployed rule and the difference is the largest upstream incompatibility on
    this lever. Ours takes a softmax over the source axis and thresholds at 0.5, which admits at
    most ONE parent per target by arithmetic (``FACT-0369``); HOCT's is a purely geometric ball
    with no cap and no probability filter, so the head is trained to discriminate among many
    competing parents that our pipeline never offers it.
    """
    incoming = cKDTree(source_um).query_ball_point(target_um, r=gate_um)
    source, target = [], []
    for target_index, source_indices in enumerate(incoming):
        for source_index in source_indices:
            source.append(source_index)
            target.append(target_index)
    return (np.asarray(source, dtype=np.int64), np.asarray(target, dtype=np.int64))


def build_edge_neighborhood(
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    source_um: np.ndarray,
    target_um: np.ndarray,
    *,
    local_midpoint_neighbors: int = LOCAL_MIDPOINT_NEIGHBORS,
    max_neighbors: int = EDGE_NEIGHBORS,
) -> np.ndarray:
    """Padded edge-neighbour indices, endpoint-sharing edges FIRST.

    The ordering is load-bearing: edges sharing a target are competing parents and edges sharing
    a source are competing daughters, so they must survive the truncation to ``max_neighbors``
    ahead of merely nearby line segments.
    """
    edge_source = np.asarray(edge_source, dtype=np.int64)
    edge_target = np.asarray(edge_target, dtype=np.int64)
    n_edges = len(edge_source)
    if n_edges == 0:
        return np.empty((0, max_neighbors), dtype=np.int64)

    by_source: dict[int, list[int]] = defaultdict(list)
    by_target: dict[int, list[int]] = defaultdict(list)
    for edge_index, (source, target) in enumerate(zip(edge_source, edge_target)):
        by_source[int(source)].append(edge_index)
        by_target[int(target)].append(edge_index)

    midpoints = (
        np.asarray(source_um, dtype=np.float32)[edge_source]
        + np.asarray(target_um, dtype=np.float32)[edge_target]
    ) / 2.0
    local_k = min(n_edges, local_midpoint_neighbors + 1)
    _, local_indices = cKDTree(midpoints).query(midpoints, k=local_k)
    if local_k == 1:
        local_indices = local_indices[:, None]

    neighbors = np.full((n_edges, max_neighbors), -1, dtype=np.int64)
    for edge_index, (source, target) in enumerate(zip(edge_source, edge_target)):
        ordered = [edge_index]
        ordered.extend(by_target[int(target)])
        ordered.extend(by_source[int(source)])
        ordered.extend(np.asarray(local_indices[edge_index]).reshape(-1).tolist())
        unique: list[int] = []
        seen: set[int] = set()
        for candidate in ordered:
            candidate = int(candidate)
            if candidate not in seen:
                seen.add(candidate)
                unique.append(candidate)
            if len(unique) == max_neighbors:
                break
        neighbors[edge_index, : len(unique)] = unique
    return neighbors


def edge_relation_features(
    source_um: torch.Tensor,
    target_um: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    neighbors: torch.Tensor,
    gate_um: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The 13 relation features and the edge midpoints RoPE is applied at.

    Slot order is the contract and is asserted by ``tests/test_hoct_head.py``.
    """
    valid = neighbors.ge(0)
    safe_neighbors = neighbors.clamp_min(0)
    q_start = source_um[edge_source][:, None, :]
    q_end = target_um[edge_target][:, None, :]
    n_source = edge_source[safe_neighbors]
    n_target = edge_target[safe_neighbors]
    n_start = source_um[n_source]
    n_end = target_um[n_target]

    q_mid = (q_start + q_end) / 2.0
    n_mid = (n_start + n_end) / 2.0
    cosine = F.cosine_similarity(q_end - q_start, n_end - n_start, dim=-1).unsqueeze(-1)
    line_distance = torch.minimum(
        torch.minimum(
            _point_segment_distance(q_start.expand_as(n_start), n_start, n_end),
            _point_segment_distance(q_end.expand_as(n_end), n_start, n_end),
        ),
        torch.minimum(
            _point_segment_distance(n_start, q_start.expand_as(n_start), q_end.expand_as(n_end)),
            _point_segment_distance(n_end, q_start.expand_as(n_end), q_end.expand_as(n_end)),
        ),
    ).unsqueeze(-1)
    features = torch.cat(
        [
            (n_mid - q_mid) / gate_um,
            (n_start - q_start) / gate_um,
            (n_end - q_end) / gate_um,
            cosine,
            line_distance / gate_um,
            n_source.eq(edge_source[:, None]).float().unsqueeze(-1),
            n_target.eq(edge_target[:, None]).float().unsqueeze(-1),
        ],
        dim=-1,
    )
    midpoints = ((source_um[edge_source] + target_um[edge_target]) / 2.0) / gate_um
    return features, valid, midpoints


class EdgeBlock(nn.Module):
    """Pre-norm attention over candidate edges with a learned relation bias and 3-D RoPE."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
        self.scale = 1.0 / sqrt(self.head_dim)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.relation_bias = nn.Sequential(
            nn.Linear(RELATION_IN, D_MODEL // 2), nn.GELU(), nn.Linear(D_MODEL // 2, N_HEADS),
        )
        self.rope = Rotary3D(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(D_MODEL * 2, D_MODEL),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        neighbors: torch.Tensor,
        relation: torch.Tensor,
        valid: torch.Tensor,
        midpoints: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm1(tokens)
        safe = neighbors.clamp_min(0)
        q = self.q(normalized).view(len(tokens), self.n_heads, self.head_dim)
        k = self.k(normalized[safe]).view(
            len(tokens), neighbors.shape[1], self.n_heads, self.head_dim,
        )
        v = self.v(normalized[safe]).view(
            len(tokens), neighbors.shape[1], self.n_heads, self.head_dim,
        )
        q = self.rope(q, midpoints)
        k = self.rope(k, midpoints[safe])
        logits = (q[:, None] * k).sum(dim=-1) * self.scale + self.relation_bias(relation)
        logits = logits.masked_fill(~valid[..., None], torch.finfo(logits.dtype).min)
        attended = (torch.softmax(logits, dim=1)[..., None] * v).sum(dim=1).reshape(
            len(tokens), -1,
        )
        tokens = tokens + self.dropout(self.out(attended))
        return tokens + self.dropout(self.mlp(self.norm2(tokens)))


class HoctAssociationHead(nn.Module):
    """The published HOCT association head, matching its state dict AND its function."""

    def __init__(self, node_feature_dim: int = UNET_CHANNELS, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_um = GATE_UM
        self.node_projection = nn.Linear(node_feature_dim + NODE_EXTRAS, D_MODEL)
        self.frame_embedding = nn.Embedding(2, D_MODEL)
        # activation="gelu" and norm_first=True are NOT visible to a strict load and were both
        # wrong in the previous reconstruction. See the module docstring.
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, D_MODEL * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.node_encoder = nn.TransformerEncoder(layer, 2)
        self.edge_projection = nn.Sequential(
            nn.Linear(D_MODEL * 2 + PAIR_EXTRAS, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL),
        )
        self.edge_blocks = nn.ModuleList(EdgeBlock(dropout) for _ in range(3))
        self.edge_head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, 1))
        self.quiet_head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, 1))
        self.division_head = nn.Sequential(
            nn.LayerNorm(D_MODEL * 3), nn.Linear(D_MODEL * 3, D_MODEL),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(D_MODEL, 1),
        )
        # Built for the strict load, never invoked - FACT-0347 / FACT-0371.
        self.fork_head = nn.Sequential(
            nn.Linear(FORK_IN, D_MODEL), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(D_MODEL, 1),
        )

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_um: torch.Tensor,
        target_um: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        center = torch.cat([source_um, target_um], dim=0).mean(dim=0, keepdim=True)
        src = self.node_projection(
            torch.cat([source_features, (source_um - center) / self.gate_um], dim=-1),
        ) + self.frame_embedding.weight[0]
        tgt = self.node_projection(
            torch.cat([target_features, (target_um - center) / self.gate_um], dim=-1),
        ) + self.frame_embedding.weight[1]
        nodes = self.node_encoder(torch.cat([src, tgt], dim=0).unsqueeze(0)).squeeze(0)
        src, tgt = nodes[: len(src)], nodes[len(src):]

        displacement = (target_um[edge_target] - source_um[edge_source]) / self.gate_um
        distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        edge_tokens = self.edge_projection(
            torch.cat([src[edge_source], tgt[edge_target], displacement, distance], dim=-1),
        )
        relation, valid, midpoints = edge_relation_features(
            source_um, target_um, edge_source, edge_target, neighbors, self.gate_um,
        )
        for block in self.edge_blocks:
            edge_tokens = block(edge_tokens, neighbors, relation, valid, midpoints)
        return {
            "edge_logits": self.edge_head(edge_tokens).squeeze(-1),
            "quiet_logits": self.quiet_head(tgt).squeeze(-1),
            "edge_tokens": edge_tokens,
        }


def parent_probability(
    edge_logits: np.ndarray, quiet_logits: np.ndarray, edge_target: np.ndarray,
) -> np.ndarray:
    """HOCT's own normalisation: a softmax over each target's incoming edges PLUS a quiet logit.

    The abstain mass is the structural difference from our deployed rule, which softmaxes over
    the source axis with no no-parent alternative (``FACT-0369``).
    -> infer_stabledet_hoct.py, edge_probability
    """
    maximum = quiet_logits.copy()
    np.maximum.at(maximum, edge_target, edge_logits)
    denominator = np.exp(quiet_logits - maximum)
    np.add.at(denominator, edge_target, np.exp(edge_logits - maximum[edge_target]))
    return np.exp(edge_logits - maximum[edge_target]) / denominator[edge_target]


def load_checkpoint(path: Path) -> tuple[HoctAssociationHead, dict]:
    """Build the module and load the published weights with ``strict=True``.

    Necessary and NOT sufficient: it proves every published tensor has a home of the right shape
    and says nothing about activation, normalisation order, positional encoding or the feature
    contract. ``equivalence`` is what covers those.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"] if "model" in payload else payload
    model = HoctAssociationHead(int(payload.get("feature_dim", UNET_CHANNELS)))
    model.load_state_dict(state, strict=True)
    model.eval()
    meta = {k: v for k, v in payload.items() if k not in {"model", "optimizer"}}
    return model, meta


# ---------------------------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------------------------

def _seeded_inputs(
    n_source: int, n_target: int, seed: int, feature_dim: int = UNET_CHANNELS,
) -> dict:
    """A small, structured, reproducible frame pair. Coordinates are drawn at a realistic cell
    spacing so the 15 um gate actually produces competing parents rather than a trivial graph."""
    rng = np.random.default_rng(seed)
    source_um = rng.uniform(0.0, 60.0, size=(n_source, 3)).astype(np.float32)
    target_um = (source_um[:n_target] + rng.normal(0.0, 2.4, size=(n_target, 3))).astype(
        np.float32,
    )
    edge_source, edge_target = candidate_graph(source_um, target_um)
    if not len(edge_source):
        raise RuntimeError("seeded geometry produced no candidate edges - cannot gate on it")
    neighbors = build_edge_neighborhood(edge_source, edge_target, source_um, target_um)
    return {
        "source_features": torch.from_numpy(
            rng.normal(0.0, 1.0, size=(n_source, feature_dim)).astype(np.float32),
        ),
        "target_features": torch.from_numpy(
            rng.normal(0.0, 1.0, size=(n_target, feature_dim)).astype(np.float32),
        ),
        "source_um": torch.from_numpy(source_um),
        "target_um": torch.from_numpy(target_um),
        "edge_source": torch.from_numpy(edge_source),
        "edge_target": torch.from_numpy(edge_target),
        "neighbors": torch.from_numpy(neighbors),
    }


def run_equivalence(checkpoint: Path, publisher_source: Path, tolerance: float) -> dict:
    """FAIL CLOSED. Prove this reimplementation computes the publisher's function.

    Imports the publisher module by path, loads the same checkpoint into both, and compares
    ``edge_logits`` and ``quiet_logits`` on identical inputs. A missing publisher source is a
    HARD FAILURE here, because a gate that quietly does nothing is indistinguishable from a gate
    that passed - the failure mode AGENTS.md names explicitly.
    """
    import importlib.util

    if not publisher_source.exists():
        raise SystemExit(
            f"EQUIVALENCE CANNOT RUN: publisher source absent at {publisher_source}. "
            "Fetch it with: python -m kaggle datasets download -d "
            "rudispresence/biohub-stabledet-hoct-code -f "
            "repo_overlay/methods/stabledet_hoct_fork/hoct_edge_transformer.py",
        )
    spec = importlib.util.spec_from_file_location("_hoct_publisher", publisher_source)
    if spec is None or spec.loader is None:
        raise SystemExit(f"EQUIVALENCE CANNOT RUN: cannot import {publisher_source}")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    feature_dim = int(payload["feature_dim"])
    theirs = publisher.StableDetHOCTFork(feature_dim, dropout=0.0)
    theirs.load_state_dict(payload["model"], strict=True)
    theirs.eval()
    ours, _ = load_checkpoint(checkpoint)

    inputs = _seeded_inputs(48, 46, seed=20260829, feature_dim=feature_dim)
    with torch.no_grad():
        mine = ours(**inputs)
        yours = theirs(
            inputs["source_features"], inputs["target_features"],
            inputs["source_um"], inputs["target_um"],
            inputs["edge_source"], inputs["edge_target"], inputs["neighbors"],
        )
    edge_delta = float((mine["edge_logits"] - yours.edge_logits).abs().max())
    quiet_delta = float((mine["quiet_logits"] - yours.quiet_logits).abs().max())
    passed = edge_delta <= tolerance and quiet_delta <= tolerance
    return {
        "checkpoint": checkpoint.name,
        "publisher_source": str(publisher_source),
        "candidate_edges": int(len(inputs["edge_source"])),
        "max_abs_edge_logit_delta": edge_delta,
        "max_abs_quiet_logit_delta": quiet_delta,
        "tolerance": tolerance,
        "passed": passed,
    }


def _permuted_argmax_change(
    model: HoctAssociationHead, inputs: dict, group: str, seed: int,
) -> float:
    """Fraction of CONTESTED targets whose argmax parent moves when one contract slot group is
    permuted. This is the POWER of the ranking experiment PKT-0029 STEP 1 specified: if a wrong
    ordering leaves the ranking untouched, no amount of held-out scoring could ever have
    identified the contract, and `not_identifiable` would have been a statement about the
    instrument rather than about the weights."""
    if group not in CONTRACT_SLOTS:
        raise ValueError(f"unknown contract slot group: {group}")
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        base = model(**inputs)["edge_logits"].numpy()
    start, stop = CONTRACT_SLOTS[group]
    # ALL THREE GROUPS ARE PERMUTED THE SAME WAY - by reordering the WEIGHT COLUMNS that read the
    # slot - so the three numbers are comparable. An earlier version permuted the coordinate AXES
    # for node_extras instead, which is an ISOMETRY: it preserves every distance and simultaneously
    # relabels the displacement and relation channels, so it measured global axis-convention
    # robustness rather than the sensitivity of the 3 node slots. That is a different question and
    # it is not the one this probe exists to answer.
    other = _forward_with_permuted_block(model, inputs, group, rng.permutation(stop - start))
    target = inputs["edge_target"].numpy()
    changed = total = 0
    for t in np.unique(target):
        mask = target == t
        if mask.sum() < 2:
            continue
        total += 1
        changed += int(np.argmax(base[mask]) != np.argmax(other[mask]))
    return changed / max(total, 1)


def _forward_with_permuted_block(
    model: HoctAssociationHead, inputs: dict, group: str, order: np.ndarray,
) -> np.ndarray:
    """Permute the pair-extra or relation slots by permuting the WEIGHT COLUMNS that read them.

    Permuting the input columns and permuting the reading weight columns are the same experiment,
    and doing it on the weights avoids rebuilding the geometry, so the two runs differ in exactly
    one thing.
    """
    index = torch.from_numpy(np.asarray(order, dtype=np.int64))
    start, stop = CONTRACT_SLOTS[group]
    if group == "node_extras":
        layers = [model.node_projection]
    elif group == "pair_extras":
        layers = [model.edge_projection[0]]
    else:
        layers = [block.relation_bias[0] for block in model.edge_blocks]
    if index.numel() != stop - start:
        raise ValueError(f"permutation of size {index.numel()} for slot group {group}")

    originals = [layer.weight.data.clone() for layer in layers]
    try:
        for layer, original in zip(layers, originals):
            layer.weight.data[:, start:stop] = original[:, start:stop][:, index]
        with torch.no_grad():
            return model(**inputs)["edge_logits"].numpy()
    finally:
        for layer, original in zip(layers, originals):
            layer.weight.data.copy_(original)


def real_inputs(
    parquet: Path, frame: int, seed: int, feature_dim: int = UNET_CHANNELS,
) -> dict:
    """Real fold-0 node geometry from the P30 pre-ILP export, one frame pair.

    The stored ``z, y, x`` are FULL-RES VOXEL INDICES, so they are scaled by ``SCALE_ZYX_UM``,
    which is anisotropic. Calibrated per AGENTS.md against a known value: this convention puts
    the nearest-neighbour median at 9.6 um, consistent with the ~8.4 um cell spacing this
    campaign has measured, where leaving the coordinates in raw voxels gives 13.6 um.
    """
    import polars as pl

    # scan_parquet with predicate pushdown: the fold-0 export is millions of rows and
    # materialising all of them competes with whatever else holds this machine's RAM.
    scan = pl.scan_parquet(parquet).filter(pl.col("row_type") == "node")
    datasets = scan.select("dataset").unique().collect()["dataset"].to_list()
    if not datasets:
        raise SystemExit(f"PERMUTATION POWER CANNOT RUN: no node rows in {parquet}")
    dataset = sorted(datasets)[0]
    crop = scan.filter(pl.col("dataset") == dataset).select(["t", "z", "y", "x"]).collect()
    times = sorted(crop["t"].unique().to_list())
    if frame + 1 >= len(times):
        raise SystemExit(f"PERMUTATION POWER CANNOT RUN: frame {frame} out of range")
    t_src, t_tgt = times[frame], times[frame + 1]

    def coords(t: int) -> np.ndarray:
        return (
            crop.filter(pl.col("t") == t).select(["z", "y", "x"]).to_numpy() * SCALE_ZYX_UM
        ).astype(np.float32)

    source_um, target_um = coords(t_src), coords(t_tgt)
    edge_source, edge_target = candidate_graph(source_um, target_um)
    if not len(edge_source):
        raise SystemExit("PERMUTATION POWER CANNOT RUN: real geometry produced no candidates")
    neighbors = build_edge_neighborhood(edge_source, edge_target, source_um, target_um)
    rng = np.random.default_rng(seed)
    return {
        "source_features": torch.from_numpy(
            rng.normal(0.0, 1.0, size=(len(source_um), feature_dim)).astype(np.float32),
        ),
        "target_features": torch.from_numpy(
            rng.normal(0.0, 1.0, size=(len(target_um), feature_dim)).astype(np.float32),
        ),
        "source_um": torch.from_numpy(source_um),
        "target_um": torch.from_numpy(target_um),
        "edge_source": torch.from_numpy(edge_source),
        "edge_target": torch.from_numpy(edge_target),
        "neighbors": torch.from_numpy(neighbors),
        "_dataset": dataset,
        "_frames": (int(t_src), int(t_tgt)),
    }


def run_permutation_power(
    checkpoint: Path, repeats: int, seed: int, coords_parquet: Path | None = None,
    frame: int = 0,
) -> dict:
    """How detectable is a WRONG contract, under the real trained weights?

    Node features are drawn from a standard normal rather than sampled from the UNet, because the
    32-channel node features are GPU-bound (``cache_official_hoct_features.py`` refuses to run
    without CUDA) and PKT-0029's feature-cache budget is gated. This measures the head's
    SENSITIVITY to each contract slot group, NOT its accuracy, and nothing here is a performance
    claim about HOCT.

    With ``--coords-parquet`` the geometry is REAL fold-0 node coordinates, so only the node
    features are synthetic.
    """
    model, meta = load_checkpoint(checkpoint)
    feature_dim = int(meta["feature_dim"])
    if coords_parquet is not None:
        print(f"    loading real geometry from {coords_parquet} ...", flush=True)
        inputs = real_inputs(coords_parquet, frame, seed, feature_dim)
        geometry = {"source": "real", "crop": inputs.pop("_dataset"),
                    "frames": inputs.pop("_frames")}
        print(f"    geometry ready: {len(inputs['source_um'])}x{len(inputs['target_um'])} nodes, "
              f"{len(inputs['edge_source'])} candidate edges", flush=True)
    else:
        inputs = _seeded_inputs(64, 62, seed=seed, feature_dim=feature_dim)
        geometry = {"source": "seeded"}
    result = {
        "checkpoint": checkpoint.name,
        "geometry": geometry,
        "nodes": [int(len(inputs["source_um"])), int(len(inputs["target_um"]))],
        "candidate_edges": int(len(inputs["edge_source"])),
        "repeats": repeats,
        "features": "synthetic N(0,1) - GPU-bound UNet features unavailable; NOT a performance claim",
        "note": "argmax-parent change rate on contested targets under a permuted slot group",
    }
    for group in CONTRACT_SLOTS:
        print(f"    permuting {group} x{repeats} ...", flush=True)
        rates = [
            _permuted_argmax_change(model, inputs, group, seed + offset)
            for offset in range(repeats)
        ]
        result[group] = {
            "mean_argmax_change_rate": float(np.mean(rates)),
            "min": float(np.min(rates)),
            "max": float(np.max(rates)),
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reconstruct", help="strict-load the published checkpoints")
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path)

    p = sub.add_parser("equivalence", help="prove this module computes the publisher's function")
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--publisher-source", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=1e-5)
    p.add_argument("--out", type=Path)

    p = sub.add_parser("permutation-power", help="how detectable is a wrong contract at all")
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260829)
    p.add_argument("--coords-parquet", type=Path,
                   help="P30 pre-ILP export; use REAL node geometry instead of seeded")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", type=Path)

    args = ap.parse_args()
    rows: list[dict] = []
    passed = True

    for path in args.checkpoints:
        if args.command == "reconstruct":
            entry: dict = {"checkpoint": str(path)}
            try:
                model, meta = load_checkpoint(path)
                entry.update(
                    strict_load=True,
                    parameters=int(sum(q.numel() for q in model.parameters())),
                    method=str(meta.get("method")),
                    gate_um=meta.get("gate_um"),
                    edge_neighbors=meta.get("edge_neighbors"),
                    feature_dim=meta.get("feature_dim"),
                )
            except Exception as exc:  # noqa: BLE001 - the failure IS the result here
                entry.update(strict_load=False, error=f"{type(exc).__name__}: {exc}")
            passed &= bool(entry["strict_load"])
            rows.append(entry)
            print(f"  {'OK  ' if entry['strict_load'] else 'FAIL'} {path.name}")
        elif args.command == "equivalence":
            entry = run_equivalence(path, args.publisher_source, args.tolerance)
            passed &= entry["passed"]
            rows.append(entry)
            print(
                f"  {'OK  ' if entry['passed'] else 'FAIL'} {path.name}  "
                f"edge_delta={entry['max_abs_edge_logit_delta']:.3e}  "
                f"quiet_delta={entry['max_abs_quiet_logit_delta']:.3e}  "
                f"edges={entry['candidate_edges']}",
            )
        else:
            entry = run_permutation_power(
                path, args.repeats, args.seed, args.coords_parquet, args.frame,
            )
            rows.append(entry)
            print(f"  {path.name}  edges={entry['candidate_edges']} "
                  f"geometry={entry['geometry']['source']}")
            for group in CONTRACT_SLOTS:
                print(f"      {group:<13} argmax change {entry[group]['mean_argmax_change_rate']:.3f}")

    result = {"schema_version": 1, "command": args.command, "passed": passed, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Positive heartbeat: its ABSENCE is the alarm, per AGENTS.md.
    print(f"\nHOCT_HEAD {args.command.upper()} passed={passed} rows={len(rows)}")
    if not passed:
        raise SystemExit(f"{args.command} FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
