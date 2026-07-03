"""Phase-1 step 4-5: same-cell conflict sets + matching-aware arbitration.

Same-cell conflict sets group candidates that are duplicate detections of ONE nucleus. Per the
red-team P0 fix, we do NOT use transitive connected components (a chain of nearby proposals can
bridge two real nuclei). We use **complete-linkage clustering with a hard diameter cap** so no
cluster can span more than r_same_um (no chaining), falling back to singletons when unclustered.
(A watershed/intensity-basin variant can replace this if the guardrail merge-rate is too high.)

GUARDRAIL (phase-critical): two GT nuclei <7 um apart must NOT land in the same conflict set.
`merge_rate` measures this on GT; it must be ~0.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree

SCALE = (1.625, 0.40625, 0.40625)


def same_cell_sets(coords_zyx: np.ndarray, scale=SCALE, r_same_um: float = 3.0) -> np.ndarray:
    """Label candidates into same-cell sets via complete-linkage with a diameter cap.

    Complete linkage + distance threshold r_same_um => every cluster's MAX pairwise physical
    distance <= r_same_um, so a cluster can never span two nuclei > r_same_um apart (no chaining).
    Returns an int label per candidate (0-based).
    """
    n = len(coords_zyx)
    if n == 0:
        return np.zeros(0, np.int64)
    if n == 1:
        return np.zeros(1, np.int64)
    phys = np.asarray(coords_zyx, float) * np.asarray(scale, float)
    Z = linkage(phys, method="complete")
    return fcluster(Z, t=r_same_um, criterion="distance").astype(np.int64) - 1


def sets_per_frame(coords_by_t: list[np.ndarray], scale=SCALE, r_same_um: float = 3.0) -> list[np.ndarray]:
    return [same_cell_sets(c, scale, r_same_um) for c in coords_by_t]


def merge_rate(gt_coords_by_t: list[np.ndarray], scale=SCALE, r_same_um: float = 3.0,
               gate_um: float = 7.0) -> dict:
    """Guardrail: fraction of GT nuclei PAIRS <gate_um apart (same frame) that complete-linkage
    would place in the SAME conflict set. Must be ~0 (real distinct nuclei must not merge).

    Evaluated directly on GT points (worst case: the annotated cells themselves as 'candidates')."""
    sc = np.asarray(scale, float)
    n_pairs = n_merged = 0
    for coords in gt_coords_by_t:
        if len(coords) < 2:
            continue
        labels = same_cell_sets(coords, scale, r_same_um)
        phys = coords * sc
        tree = cKDTree(phys)
        for i, j in tree.query_pairs(gate_um):        # GT pairs within 7 um
            n_pairs += 1
            if labels[i] == labels[j]:
                n_merged += 1
    return {"gt_pairs_within_gate": n_pairs, "merged": n_merged,
            "merge_rate": (n_merged / n_pairs) if n_pairs else 0.0, "r_same_um": r_same_um}
