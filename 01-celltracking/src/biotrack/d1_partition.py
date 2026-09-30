"""M/T/C/L/D — the causally correct detection-loss partition.

The exact-GT-voxel A/B/D partition was the WRONG UNIT. Detection success is decided by
bipartite matching of ACCEPTED PEAKS to GT within the scorer's radius, so an accepted peak one
grid voxel away matches the GT correctly even when the exact GT voxel is pool-suppressed. The
v4 smoke proved this: 44b6_0113de3b matched 52/52 GT (node_recall 1.000000) while the
exact-voxel classifier called 26 of those same 52 "B", i.e. suppressed.

Primary partition, over ALL GT nodes:

    M  matched to an accepted peak by the scorer's own bipartite matching
    T  unmatched; an UNACCEPTED local maximum exists within 7 um  -> threshold-limited
    C  unmatched; an ACCEPTED peak exists within 7 um             -> competition/assignment
    L  unmatched; no usable maximum within 7 um but one within 15 um -> displaced/localisation
    D  unmatched; no local maximum within 15 um                   -> genuinely response-poor

Invariants (enforced by tests): M+T+C+L+D == all GT, and T+C+L+D == scorer-unmatched GT.

Distances are PHYSICAL Euclidean in microns and filtered to a SPHERE. A 15 um axis-aligned
box admits corner distances up to 15*sqrt(3) = 25.98 um, which is why the v4 export reported a
p50 of 15.4 um and a max of 24.4 um -- it was selecting cube corners, not a 15 um neighbourhood.
"""
from __future__ import annotations

from dataclasses import dataclass

MATCH_UM = 7.0     # scorer's max_distance
SEARCH_UM = 15.0   # outer search radius for L
CUBE_CORNER_FACTOR = 3 ** 0.5


@dataclass(frozen=True)
class LocalMax:
    """A local maximum near a GT centre. `dist_um` is physical Euclidean."""
    dist_um: float
    logit: float
    prob: float
    accepted: bool          # passed BOTH the local-max test and the detector threshold


def classify_gt(
    matched: bool,
    maxima: list[LocalMax],
    match_um: float = MATCH_UM,
    search_um: float = SEARCH_UM,
) -> str:
    """Classify ONE GT node. `matched` comes from the scorer, never from a re-implementation.

    `maxima` must already be sphere-filtered to <= search_um; anything beyond is dropped here
    as a defence against a caller passing a cube.
    """
    if matched:
        return "M"
    near = [m for m in maxima if m.dist_um <= search_um]
    within_match = [m for m in near if m.dist_um <= match_um]
    if any(m.accepted for m in within_match):
        return "C"
    if any(not m.accepted for m in within_match):
        return "T"
    if near:
        return "L"
    return "D"


def sphere_filter(maxima: list[LocalMax], radius_um: float = SEARCH_UM) -> list[LocalMax]:
    """Drop cube-corner candidates. Keeps only genuine within-radius maxima."""
    return [m for m in maxima if m.dist_um <= radius_um]


def strongest_within(maxima: list[LocalMax], radius_um: float) -> LocalMax | None:
    """Strongest by logit among those inside the SPHERE, not the enclosing cube."""
    inside = [m for m in maxima if m.dist_um <= radius_um]
    return max(inside, key=lambda m: m.logit) if inside else None


def census(rows) -> dict[str, int]:
    """rows: iterable of (matched, maxima). Returns the M/T/C/L/D counts."""
    out = {k: 0 for k in ("M", "T", "C", "L", "D")}
    for matched, maxima in rows:
        out[classify_gt(matched, maxima)] += 1
    return out


def check_invariants(counts: dict[str, int], n_gt: int, n_unmatched: int) -> None:
    total = sum(counts.values())
    if total != n_gt:
        raise ValueError(f"M+T+C+L+D = {total} != {n_gt} GT nodes")
    unm = counts["T"] + counts["C"] + counts["L"] + counts["D"]
    if unm != n_unmatched:
        raise ValueError(f"T+C+L+D = {unm} != {n_unmatched} scorer-unmatched GT")
