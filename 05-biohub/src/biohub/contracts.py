"""Types every stage speaks. Deliberately small — the graph is the currency."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import os, numpy as np

SCALE = np.array([1.625, 0.40625, 0.40625])          # z, y, x micrometres per voxel
BOUNDS = {"t": (0, 99), "z": (0, 63), "y": (0, 255), "x": (0, 255)}

@dataclass
class Graph:
    """Nodes carry MUTABLE coordinates: detection and localisation are not separable."""
    t: np.ndarray                      # (N,) int
    zyx: np.ndarray                    # (N,3) float voxels — any stage may refine these
    score: np.ndarray | None = None    # (N,) detector confidence / refine_logit rank
    edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))
    edge_prob: np.ndarray | None = None
    dataset: str = ""

    def um(self) -> np.ndarray:
        return self.zyx * SCALE

    def out_degree(self) -> np.ndarray:
        d = np.zeros(len(self.t), int)
        if len(self.edges): np.add.at(d, self.edges[:, 0], 1)
        return d

    def forks(self) -> int:
        return int((self.out_degree() == 2).sum())

@dataclass
class StageDelta:
    """What a stage changed. Printed for every run — this is how you see one stage
    destroying another's work, which is the defining bug of the public field."""
    stage: str
    nodes_added: int = 0; nodes_removed: int = 0; nodes_moved: int = 0
    edges_added: int = 0; edges_removed: int = 0
    forks_created: int = 0; forks_destroyed: int = 0
    wall_s: float = 0.0

    @staticmethod
    def between(stage: str, before: Graph, after: Graph, wall_s: float = 0.0) -> "StageDelta":
        fb, fa = before.forks(), after.forks()
        moved = 0
        if len(before.t) == len(after.t):
            moved = int((np.abs(before.zyx - after.zyx) > 1e-9).any(axis=1).sum())
        return StageDelta(stage,
            nodes_added=max(0, len(after.t) - len(before.t)),
            nodes_removed=max(0, len(before.t) - len(after.t)),
            nodes_moved=moved,
            edges_added=max(0, len(after.edges) - len(before.edges)),
            edges_removed=max(0, len(before.edges) - len(after.edges)),
            forks_created=max(0, fa - fb), forks_destroyed=max(0, fb - fa), wall_s=wall_s)

    def line(self) -> str:
        return (f"  {self.stage:<12} nodes {self.nodes_added:+6}/{-self.nodes_removed:<6} "
                f"moved {self.nodes_moved:>6}  edges {self.edges_added:+7}/{-self.edges_removed:<7} "
                f"forks {self.forks_created:+5}/{-self.forks_destroyed:<5} {self.wall_s:6.1f}s")

def _env(name, default, cast=float):
    v = os.environ.get(f"BIOHUB_{name.upper()}")
    return cast(v) if v is not None else default

@dataclass(frozen=True)
class Config:
    """Every constant the pre-Mac experiments touched. Env-overridable, because the entire
    public 0.940-0.947 band is one codebase differing only in BIOHUB_* values."""
    det_threshold:   float = _env("det_threshold", 0.965)
    nms_radius_um:   float = _env("nms_radius_um", 5.0)
    refine_enabled:  bool  = bool(_env("refine_enabled", 1, int))
    edge_prob_min:   float = _env("edge_prob_min", 0.48)
    motion_gate_um:  float = _env("motion_gate_um", 10.0)
    # EXP-1: parent<=12 / sister<=18 admits 150 of 151 GT divisions (99.3%)
    fork_parent_um:  float = _env("fork_parent_um", 12.0)
    fork_sister_um:  float = _env("fork_sister_um", 18.0)
    # EXP-1 discriminators for fork ACCEPTANCE (EXP-5: precision is the binding constraint)
    fork_cos_max:    float = _env("fork_cos_max", -0.30)
    fork_divergence_min_um: float = _env("fork_divergence_min_um", 1.0)
    gap_max_frames:  int   = _env("gap_max_frames", 2, int)
    min_track_len:   int   = _env("min_track_len", 6, int)
