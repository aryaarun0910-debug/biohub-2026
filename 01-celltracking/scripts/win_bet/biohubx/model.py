"""A minimal T=2, graph-owning association model for BIOHUB-X.

This module deliberately keeps three ideas separate:

``temporal_window``
    The number of frames presented together.  ``biohubx_io_v1`` starts at exactly two.
``depth``
    The number of transformer encoder blocks.  The preregistered depth ladder is 4/6/8/10.
``node_feature_dim``
    The representation supplied by the detector/instance encoder.  Whether that producer is
    frozen or trainable is a training-policy decision outside this module.

The matcher owns the fourth arrow in PKT-0049.  It emits candidate probabilities, including one
explicit learned no-parent logit per target, and turns them directly into a lineage graph.  It
does not call the incumbent ILP or motion relink (FACT-0428).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .contract import CLASSES, CONTRACT_VERSION, NO_PARENT


SUPPORTED_DEPTHS: tuple[int, ...] = (4, 6, 8, 10)
TEMPORAL_WINDOW = 2
CONSUMER_CHAIN: tuple[str, ...] = (
    "biohubx.t2_matcher.v1",
    "biohubx.capacity_graph_consumer.v1",
)


@dataclass(frozen=True)
class MatcherConfig:
    """Architecture configuration; temporal extent and network depth are independent fields."""

    node_feature_dim: int
    model_dim: int = 128
    depth: int = 6
    temporal_window: int = TEMPORAL_WINDOW
    num_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.temporal_window != TEMPORAL_WINDOW:
            raise ValueError(
                f"biohubx_io_v1 starts at temporal_window T={TEMPORAL_WINDOW}; "
                f"got T={self.temporal_window}. T is not transformer depth D"
            )
        if self.depth not in SUPPORTED_DEPTHS:
            raise ValueError(
                f"transformer depth D must be one of {SUPPORTED_DEPTHS}; got D={self.depth}"
            )
        if self.node_feature_dim < 1:
            raise ValueError("node_feature_dim must be positive")
        if self.model_dim < 1 or self.feedforward_dim < 1:
            raise ValueError("model_dim and feedforward_dim must be positive")
        if self.num_heads < 1 or self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")


@dataclass(frozen=True)
class MatcherOutput:
    """Raw learned outputs.

    ``pair_logits[edge, class]`` uses the frozen ``CLASSES`` slot order.  Candidate endpoint
    indices are explicit: the model never materialises the dense target-by-source product and
    never invents a score for a pair the candidate generator did not offer.  The
    ``no_parent_logits`` tensor is a genuinely separate learned target-level output; it is not a
    threshold or a softmax denominator reconstructed from the pair logits (FACT-0457).
    """

    pair_logits: Tensor
    no_parent_logits: Tensor
    candidate_source_index: Tensor
    candidate_target_index: Tensor
    n_source: int
    n_target: int

    def __post_init__(self) -> None:
        if self.pair_logits.ndim != 2 or self.pair_logits.shape[-1] != len(CLASSES):
            raise ValueError(
                f"pair_logits must have shape (n_candidate, {len(CLASSES)}), "
                f"got {tuple(self.pair_logits.shape)}"
            )
        n_edges = self.pair_logits.shape[0]
        if self.no_parent_logits.shape != (self.n_target,):
            raise ValueError(
                "no_parent_logits must contain exactly one explicit logit per target; got "
                f"{tuple(self.no_parent_logits.shape)} for pair logits "
                f"{tuple(self.pair_logits.shape)}"
            )
        for name, index in (
            ("candidate_source_index", self.candidate_source_index),
            ("candidate_target_index", self.candidate_target_index),
        ):
            if index.shape != (n_edges,) or index.dtype != torch.long:
                raise ValueError(f"{name} must be int64 with shape ({n_edges},)")
        if self.n_source < 0 or self.n_target < 1:
            raise ValueError("n_source must be non-negative and n_target must be positive")
        if n_edges:
            if int(self.candidate_source_index.min()) < 0 or \
                    int(self.candidate_source_index.max()) >= self.n_source:
                raise ValueError("candidate source index is out of bounds")
            if int(self.candidate_target_index.min()) < 0 or \
                    int(self.candidate_target_index.max()) >= self.n_target:
                raise ValueError("candidate target index is out of bounds")
            keys = self.candidate_target_index * max(self.n_source, 1) + self.candidate_source_index
            if torch.unique(keys).numel() != n_edges:
                raise ValueError("candidate source-target pairs must be unique")


@dataclass(frozen=True)
class GraphDecisions:
    """One direct graph decision per target, before it is materialised as a node table."""

    parent_index: np.ndarray
    relation_index: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class OwnedGraph:
    """The emitted graph and the positive declaration of the consumer that produced it."""

    table: dict[str, np.ndarray]
    consumer_chain: tuple[str, ...]
    decisions: GraphDecisions


class BiohubXMatcher(nn.Module):
    """Contextual T=2 matcher with a direct capacity-constrained graph consumer."""

    def __init__(self, config: MatcherConfig):
        super().__init__()
        self.config = config
        d = config.model_dim

        self.feature_projection = nn.Linear(config.node_feature_dim, d)
        self.position_projection = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.frame_embedding = nn.Embedding(TEMPORAL_WINDOW, d)
        self.input_norm = nn.LayerNorm(d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.depth, enable_nested_tensor=False
        )

        # Pair identity, relative motion and feature agreement are all explicit in the edge token.
        self.pair_head = nn.Sequential(
            nn.Linear(4 * d, config.feedforward_dim),
            nn.GELU(),
            nn.Linear(config.feedforward_dim, len(CLASSES)),
        )
        self.no_parent_head = nn.Sequential(
            nn.Linear(d, config.feedforward_dim),
            nn.GELU(),
            nn.Linear(config.feedforward_dim, 1),
        )

    def _validate_inputs(
        self,
        source_features: Tensor,
        target_features: Tensor,
        source_positions_um: Tensor,
        target_positions_um: Tensor,
    ) -> None:
        expected = self.config.node_feature_dim
        for name, value in (
            ("source_features", source_features),
            ("target_features", target_features),
        ):
            if value.ndim != 2 or value.shape[1] != expected:
                raise ValueError(f"{name} must have shape (N, {expected}), got {tuple(value.shape)}")
            if not torch.is_floating_point(value):
                raise ValueError(f"{name} must be floating point")
        for name, value, n in (
            ("source_positions_um", source_positions_um, source_features.shape[0]),
            ("target_positions_um", target_positions_um, target_features.shape[0]),
        ):
            if value.shape != (n, 3):
                raise ValueError(f"{name} must have shape ({n}, 3), got {tuple(value.shape)}")
            if not torch.is_floating_point(value):
                raise ValueError(f"{name} must be floating point physical (z, y, x) coordinates")
        values = (source_features, target_features, source_positions_um, target_positions_um)
        if any(not torch.isfinite(v).all() for v in values):
            raise ValueError("matcher inputs must be finite; NaN is not a neutral fill")
        if target_features.shape[0] == 0:
            raise ValueError("a T=2 matcher call must contain at least one target")
        if len({v.device for v in values}) != 1:
            raise ValueError("all matcher inputs must be on the same device")

    def forward(
        self,
        source_features: Tensor,
        target_features: Tensor,
        source_positions_um: Tensor,
        target_positions_um: Tensor,
        *,
        candidate_source_index: Tensor,
        candidate_target_index: Tensor,
    ) -> MatcherOutput:
        """Return sparse three-way pair logits and one explicit no-parent logit per target."""
        self._validate_inputs(
            source_features, target_features, source_positions_um, target_positions_um
        )

        features = torch.cat((source_features, target_features), dim=0)
        positions = torch.cat((source_positions_um, target_positions_um), dim=0)
        frame_ids = torch.cat(
            (
                torch.zeros(source_features.shape[0], dtype=torch.long, device=features.device),
                torch.ones(target_features.shape[0], dtype=torch.long, device=features.device),
            )
        )
        tokens = (
            self.feature_projection(features)
            + self.position_projection(positions)
            + self.frame_embedding(frame_ids)
        )
        encoded = self.encoder(self.input_norm(tokens).unsqueeze(0)).squeeze(0)
        n_source = source_features.shape[0]
        source = encoded[:n_source]
        target = encoded[n_source:]

        n_target = target.shape[0]
        for name, index in (
            ("candidate_source_index", candidate_source_index),
            ("candidate_target_index", candidate_target_index),
        ):
            if index.ndim != 1 or index.dtype != torch.long or index.device != target.device:
                raise ValueError(f"{name} must be a one-dimensional int64 tensor on the model device")
        if candidate_source_index.shape != candidate_target_index.shape:
            raise ValueError("candidate source and target index arrays must have equal shape")
        if candidate_source_index.numel():
            if n_source == 0 or int(candidate_source_index.min()) < 0 or \
                    int(candidate_source_index.max()) >= n_source:
                raise ValueError("candidate source index is out of bounds")
            if int(candidate_target_index.min()) < 0 or int(candidate_target_index.max()) >= n_target:
                raise ValueError("candidate target index is out of bounds")
            keys = candidate_target_index * n_source + candidate_source_index
            if torch.unique(keys).numel() != keys.numel():
                raise ValueError("candidate source-target pairs must be unique")
            src = source[candidate_source_index]
            tgt = target[candidate_target_index]
            pair_token = torch.cat((src, tgt, tgt - src, tgt * src), dim=-1)
            pair_logits = self.pair_head(pair_token)
        else:
            pair_logits = target.new_empty((0, len(CLASSES)))
        no_parent_logits = self.no_parent_head(target).squeeze(-1)
        return MatcherOutput(
            pair_logits=pair_logits,
            no_parent_logits=no_parent_logits,
            candidate_source_index=candidate_source_index,
            candidate_target_index=candidate_target_index,
            n_source=n_source,
            n_target=n_target,
        )

    @staticmethod
    def probabilities(output: MatcherOutput) -> tuple[Tensor, Tensor]:
        """Return real-pair and explicit no-parent-row three-way probabilities.

        A no-parent pseudo-pair has two fixed reference logits and the learned no-parent logit in
        the frozen ``neither`` slot.  Softmax therefore produces a valid three-way row while the
        abstention evidence itself remains an independent learned quantity.
        """
        pair = torch.softmax(output.pair_logits, dim=-1)
        refs = torch.zeros(
            (output.no_parent_logits.shape[0], len(CLASSES) - 1),
            dtype=output.no_parent_logits.dtype,
            device=output.no_parent_logits.device,
        )
        no_parent = torch.softmax(torch.cat((refs, output.no_parent_logits[:, None]), dim=-1), dim=-1)
        return pair, no_parent

    @staticmethod
    def _integer_vector(value: Tensor | Sequence[int] | np.ndarray, name: str, n: int) -> np.ndarray:
        a = value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
        if a.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
        if a.dtype.kind not in "iu":
            raise ValueError(f"{name} must be integer, got {a.dtype}")
        return a.astype(np.int64, copy=False)

    def candidate_table(
        self,
        output: MatcherOutput,
        *,
        crop: str,
        t_target: int,
        source_uids: Tensor | Sequence[int] | np.ndarray,
        target_uids: Tensor | Sequence[int] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Materialise every scored pair plus exactly one no-parent row per target.

        The returned table is directly consumable by ``contract.verify_candidates``.  Nothing is
        omitted and no uncovered candidate is filled after the fact.
        """
        n_target, n_source = output.n_target, output.n_source
        src_uid = self._integer_vector(source_uids, "source_uids", n_source)
        tgt_uid = self._integer_vector(target_uids, "target_uids", n_target)
        pair_prob, no_parent_prob = self.probabilities(output)
        pair_np = pair_prob.detach().cpu().numpy()
        no_parent_np = no_parent_prob.detach().cpu().numpy()
        edge_src = output.candidate_source_index.detach().cpu().numpy()
        edge_tgt = output.candidate_target_index.detach().cpu().numpy()

        rows = len(edge_src) + n_target
        crops = np.full(rows, str(crop), dtype=object)
        times = np.full(rows, int(t_target), dtype=np.int64)
        targets = np.empty(rows, dtype=np.int64)
        sources = np.empty(rows, dtype=np.int64)
        probs = np.empty((rows, len(CLASSES)), dtype=np.float64)
        cursor = 0
        for ti, uid in enumerate(tgt_uid):
            edge_rows = np.flatnonzero(edge_tgt == ti)
            if edge_rows.size:
                sl = slice(cursor, cursor + edge_rows.size)
                targets[sl] = uid
                sources[sl] = src_uid[edge_src[edge_rows]]
                probs[sl] = pair_np[edge_rows]
                cursor += edge_rows.size
            targets[cursor] = uid
            sources[cursor] = NO_PARENT
            probs[cursor] = no_parent_np[ti]
            cursor += 1

        table: dict[str, np.ndarray] = {
            "crop": crops,
            "t_target": times,
            "target_uid": targets,
            "source_uid": sources,
        }
        for i, name in enumerate(CLASSES):
            table[f"p_{name}"] = probs[:, i]
        return table

    @staticmethod
    def select_parents(output: MatcherOutput) -> GraphDecisions:
        """Own the T=2 graph with deterministic greedy capacity constraints.

        A target is linked only when a continuation/division class beats both ``neither`` for the
        real pair and the target's explicit no-parent probability.  Continuations consume the
        source completely; division-labelled edges permit at most two children and cannot mix with
        a continuation edge.  This creates one parent per target and out-degree at most two without
        handing the result to the incumbent relink.
        """
        pair, no_parent = BiohubXMatcher.probabilities(output)
        pair_np = pair.detach().cpu().numpy()
        abstain_np = no_parent[:, CLASSES.index("neither")].detach().cpu().numpy()
        n_target, n_source = output.n_target, output.n_source

        parent = np.full(n_target, NO_PARENT, dtype=np.int64)
        relation = np.full(n_target, CLASSES.index("neither"), dtype=np.int64)
        confidence = abstain_np.astype(np.float64, copy=True)
        if not n_source:
            return GraphDecisions(parent, relation, confidence)

        continuation_i = CLASSES.index("continuation")
        division_i = CLASSES.index("division")
        neither_i = CLASSES.index("neither")
        proposals: list[tuple[float, int, int, int]] = []
        edge_src = output.candidate_source_index.detach().cpu().numpy()
        edge_tgt = output.candidate_target_index.detach().cpu().numpy()
        for ei, (si, ti) in enumerate(zip(edge_src, edge_tgt)):
            rel = continuation_i if pair_np[ei, continuation_i] >= pair_np[ei, division_i] \
                else division_i
            score = float(pair_np[ei, rel])
            if score > float(pair_np[ei, neither_i]) and score > float(abstain_np[ti]):
                proposals.append((-score, int(ti), int(si), rel))

        # Negative score makes ordinary tuple ordering descending by confidence.  The explicit
        # target/source tie-break makes output independent of insertion or dataframe row order.
        proposals.sort()
        source_degree = np.zeros(n_source, dtype=np.int64)
        source_mode = np.full(n_source, -1, dtype=np.int64)
        for neg_score, ti, si, rel in proposals:
            if parent[ti] != NO_PARENT:
                continue
            if rel == continuation_i:
                if source_degree[si] != 0:
                    continue
                source_mode[si] = continuation_i
            else:
                if source_mode[si] == continuation_i or source_degree[si] >= 2:
                    continue
                source_mode[si] = division_i
            parent[ti] = si
            relation[ti] = rel
            confidence[ti] = -neg_score
            source_degree[si] += 1
        return GraphDecisions(parent, relation, confidence)

    @staticmethod
    def _integer_positions(
        value: Tensor | Sequence[Sequence[int]] | np.ndarray, name: str, n: int
    ) -> np.ndarray:
        a = value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
        if a.shape != (n, 3):
            raise ValueError(f"{name} must have shape ({n}, 3), got {a.shape}")
        if not np.isfinite(a).all() or not np.array_equal(a, np.rint(a)):
            raise ValueError(f"{name} must contain finite integer voxel coordinates")
        return np.rint(a).astype(np.int64)

    def emit_graph(
        self,
        output: MatcherOutput,
        *,
        crop: str,
        source_uids: Tensor | Sequence[int] | np.ndarray,
        target_uids: Tensor | Sequence[int] | np.ndarray,
        source_zyx_vox: Tensor | Sequence[Sequence[int]] | np.ndarray,
        target_zyx_vox: Tensor | Sequence[Sequence[int]] | np.ndarray,
        t_source: int,
        t_target: int,
    ) -> OwnedGraph:
        """Emit the complete two-frame lineage graph directly from this model's decisions."""
        if int(t_target) - int(t_source) != 1:
            raise ValueError("biohubx_io_v1 T=2 edges must span exactly one frame")
        n_target, n_source = output.n_target, output.n_source
        src_uid = self._integer_vector(source_uids, "source_uids", n_source)
        tgt_uid = self._integer_vector(target_uids, "target_uids", n_target)
        all_uid = np.concatenate((src_uid, tgt_uid))
        if np.unique(all_uid).size != all_uid.size:
            raise ValueError("source_uids and target_uids must be globally unique in the emitted graph")
        src_pos = self._integer_positions(source_zyx_vox, "source_zyx_vox", n_source)
        tgt_pos = self._integer_positions(target_zyx_vox, "target_zyx_vox", n_target)

        decisions = self.select_parents(output)
        target_parent = np.full(n_target, NO_PARENT, dtype=np.int64)
        linked = decisions.parent_index != NO_PARENT
        target_parent[linked] = src_uid[decisions.parent_index[linked]]
        positions = np.concatenate((src_pos, tgt_pos), axis=0)
        table = {
            "crop": np.full(all_uid.size, str(crop), dtype=object),
            "node_uid": all_uid,
            "parent_uid": np.concatenate(
                (np.full(n_source, NO_PARENT, dtype=np.int64), target_parent)
            ),
            "t": np.concatenate(
                (
                    np.full(n_source, int(t_source), dtype=np.int64),
                    np.full(n_target, int(t_target), dtype=np.int64),
                )
            ),
            "z_vox": positions[:, 0],
            "y_vox": positions[:, 1],
            "x_vox": positions[:, 2],
        }
        return OwnedGraph(table=table, consumer_chain=CONSUMER_CHAIN, decisions=decisions)


__all__ = [
    "BiohubXMatcher",
    "CONSUMER_CHAIN",
    "GraphDecisions",
    "MatcherConfig",
    "MatcherOutput",
    "OwnedGraph",
    "SUPPORTED_DEPTHS",
    "TEMPORAL_WINDOW",
    "CONTRACT_VERSION",
]
