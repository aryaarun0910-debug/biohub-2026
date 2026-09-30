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

    def is_resolved(self) -> bool:
        """A graph still carrying edge_prob is a CANDIDATE set, not a selection."""
        return self.edge_prob is None

    def forks(self) -> int:
        """Only meaningful on a resolved graph. On a candidate set a node simply has several
        possible children, which is not a fork, and counting them produces false alarms in the
        ledger at the score_edges -> resolve seam."""
        if not self.is_resolved(): return -1
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
    def between(stage: str, before: Graph, after: Graph, wall_s: float = 0.0,
                nodes_moved: int | None = None) -> "StageDelta":
        """`nodes_moved` must be passed explicitly by any stage that both MOVES and PRUNES
        nodes — refine does both, and without ids there is no way to infer it here. Silently
        reporting 0 would blind the ledger at the one stage it most needs to see."""
        fb, fa = before.forks(), after.forks()
        if fb < 0 or fa < 0:               # a candidate set is on one side: forks undefined
            fb = fa = 0
        if nodes_moved is not None:
            moved = nodes_moved
        elif len(before.t) == len(after.t):
            moved = int((np.abs(before.zyx - after.zyx) > 1e-9).any(axis=1).sum())
        else:
            moved = -1                     # unknown: shown as '?' rather than a false 0
        return StageDelta(stage,
            nodes_added=max(0, len(after.t) - len(before.t)),
            nodes_removed=max(0, len(before.t) - len(after.t)),
            nodes_moved=moved,
            edges_added=max(0, len(after.edges) - len(before.edges)),
            edges_removed=max(0, len(before.edges) - len(after.edges)),
            forks_created=max(0, fa - fb), forks_destroyed=max(0, fb - fa), wall_s=wall_s)

    def line(self) -> str:
        mv = "     ?" if self.nodes_moved < 0 else f"{self.nodes_moved:>6}"
        return (f"  {self.stage:<12} nodes {self.nodes_added:+6}/{-self.nodes_removed:<6} "
                f"moved {mv}  edges {self.edges_added:+7}/{-self.edges_removed:<7} "
                f"forks {self.forks_created:+5}/{-self.forks_destroyed:<5} {self.wall_s:6.1f}s")

_ENV_SEEN: set[str] = set()


def _env(name, default, cast=float):
    _ENV_SEEN.add(f"BIOHUB_{name.upper()}")
    v = os.environ.get(f"BIOHUB_{name.upper()}")
    return cast(v) if v is not None else default


def check_env() -> None:
    """Fail loudly on a BIOHUB_* variable that matches no field.

    _env upper-cases the name, so `BIOHUB_fork_accept_p=0.4` sets nothing and says nothing --
    a whole threshold sweep once returned four identical rows because of it. A config override
    that is silently ignored is worse than one that crashes.
    """
    allowed = _ENV_SEEN | {"BIOHUB_FORK_MODEL"}      # a path, not a Config field
    unknown = sorted(k for k in os.environ
                     if k.startswith("BIOHUB_") and k not in allowed)
    if unknown:
        raise SystemExit(
            "unrecognised BIOHUB_* override(s): " + ", ".join(unknown) +
            "\n  (names are UPPER_CASE; known: " + ", ".join(sorted(_ENV_SEEN)) + ")")

@dataclass(frozen=True)
class Config:
    """Every constant the pre-Mac experiments touched. Env-overridable, because the entire
    public 0.940-0.947 band is one codebase differing only in BIOHUB_* values."""
    det_threshold:   float = _env("det_threshold", 0.965)
    nms_radius_um:   float = _env("nms_radius_um", 5.0)
    refine_enabled:  bool  = bool(_env("refine_enabled", 1, int))
    refine_prune_quantile: float = _env("refine_prune_quantile", 0.12)  # author pruned 10-14% of peaks
    edge_prob_min:   float = _env("edge_prob_min", 0.48)
    motion_gate_um:  float = _env("motion_gate_um", 14.0)   # EXP-8 LOEO: 10 -> 14 is +0.045 score, fn 69 -> 21
    # EXP-1: parent<=12 / sister<=18 admits 150 of 151 GT divisions (99.3%)
    fork_parent_um:  float = _env("fork_parent_um", 12.0)
    fork_sister_um:  float = _env("fork_sister_um", 18.0)
    # Fork ACCEPTANCE discriminators. EXP-7 swept these leave-one-embryo-out: RECALL binds,
    # not precision, so both were loosened from the EXP-1 angle/divergence distributions.
    fork_cos_max:    float = _env("fork_cos_max", 1.0)          # EXP-7 LOEO: angle gate is monotone loss -> off
    fork_divergence_min_um: float = _env("fork_divergence_min_um", 0.0)  # EXP-7 LOEO: sisters must merely not re-converge
    # EXP-10: both repair ops are net-harmful UNDER ORACLE DETECTION (+0.0042 to disable),
    # because close_gaps only has detector misses to repair and there are none, and prune_short
    # only has spurious tracks to remove and there are none. MUST BE RE-SWEPT once detect is
    # trained -- under a real detector this verdict is expected to flip.
    # EXP-15: learned fork acceptance. >=0 uses fork_model.json; <0 falls back to the
    # hand-tuned divergence threshold above.
    fork_accept_p:   float = _env("fork_accept_p", 0.30)   # EXP-16: mean of the two LOEO folds (0.25, 0.35)
    gap_max_frames:  int   = _env("gap_max_frames", 0, int)
    min_track_len:   int   = _env("min_track_len", 1, int)
