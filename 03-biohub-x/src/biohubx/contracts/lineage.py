"""Dataset, node, edge, and legal-lineage contracts.

Consumer: :mod:`biohubx.evaluation.official_metric`.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from biohubx.contracts.coordinates import VoxelCoordinateZYX

Identity = Annotated[StrictInt, Field(ge=0)]
Frame = Annotated[StrictInt, Field(ge=0)]


class EdgeKind(StrEnum):
    """The only two biological edge meanings emitted by Biohub-X."""

    CONTINUATION = "continuation"
    DIVISION = "division"


class DatasetIdentity(BaseModel):
    """Stable logical identity; never a filesystem path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class LineageNode(BaseModel):
    """One cell detection in one dataset and frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: DatasetIdentity
    node_id: Identity
    frame: Frame
    voxel: VoxelCoordinateZYX


class LineageEdge(BaseModel):
    """One directed, one-frame lineage edge with explicit endpoint datasets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_dataset: DatasetIdentity
    source: Identity
    target_dataset: DatasetIdentity
    target: Identity
    kind: EdgeKind


class LineageGraph(BaseModel):
    """A complete legal graph ready for integer export and official scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: DatasetIdentity
    nodes: tuple[LineageNode, ...]
    edges: tuple[LineageEdge, ...]

    @field_validator("nodes")
    @classmethod
    def require_nodes(cls, value: tuple[LineageNode, ...]) -> tuple[LineageNode, ...]:
        if not value:
            raise ValueError("a lineage graph must contain at least one node")
        return value

    @model_validator(mode="after")
    def validate_complete_graph(self) -> LineageGraph:
        ids = [node.node_id for node in self.nodes]
        duplicate_ids = sorted(node_id for node_id, count in Counter(ids).items() if count > 1)
        if duplicate_ids:
            raise ValueError(f"duplicate node identities: {duplicate_ids}")

        nodes = {node.node_id: node for node in self.nodes}
        for node in self.nodes:
            if node.dataset != self.dataset:
                raise ValueError(f"node {node.node_id} crosses dataset identity")

        indegree: Counter[int] = Counter()
        outgoing: dict[int, list[LineageEdge]] = {}
        edge_pairs: set[tuple[int, int]] = set()
        for edge in self.edges:
            if edge.source_dataset != self.dataset or edge.target_dataset != self.dataset:
                raise ValueError("cross-dataset edges are forbidden")
            if edge.source not in nodes or edge.target not in nodes:
                raise ValueError(f"edge endpoint is missing: {edge.source}->{edge.target}")
            pair = (edge.source, edge.target)
            if pair in edge_pairs:
                raise ValueError(f"duplicate edge: {edge.source}->{edge.target}")
            edge_pairs.add(pair)
            if nodes[edge.target].frame != nodes[edge.source].frame + 1:
                raise ValueError(f"edge must advance exactly one frame: {edge.source}->{edge.target}")
            indegree[edge.target] += 1
            outgoing.setdefault(edge.source, []).append(edge)

        illegal_in = sorted(node_id for node_id, degree in indegree.items() if degree > 1)
        if illegal_in:
            raise ValueError(f"illegal in-degree greater than one: {illegal_in}")
        for source, edges in outgoing.items():
            if len(edges) > 2:
                raise ValueError(f"illegal out-degree greater than two: {source}")
            expected = EdgeKind.DIVISION if len(edges) == 2 else EdgeKind.CONTINUATION
            if any(edge.kind is not expected for edge in edges):
                raise ValueError(f"edge kind disagrees with lineage topology at source {source}")
        return self

    def integer_export(self) -> dict[str, list[dict[str, int | str]]]:
        """Return deterministic records with identities and frames kept as integers."""
        return {
            "nodes": [
                {"node_id": node.node_id, "frame": node.frame, "dataset": node.dataset.value}
                for node in sorted(self.nodes, key=lambda item: item.node_id)
            ],
            "edges": [
                {"source": edge.source, "target": edge.target, "kind": edge.kind.value}
                for edge in sorted(self.edges, key=lambda item: (item.source, item.target))
            ],
        }
