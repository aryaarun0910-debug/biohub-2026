"""Guardrail tests for same-cell conflict sets (Phase 1 step 4).

The P0 risk: a chain of nearby proposals bridging two real nuclei. Complete-linkage with a hard
diameter cap must prevent this by construction. These tests assert it on synthetic worst cases.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biotrack.arbitrate import same_cell_sets  # noqa: E402

SCALE = (1.625, 0.40625, 0.40625)


def _xy(pts_um):
    """Build (z,y,x) voxel coords from (y_um,x_um) at fixed z, undoing the y/x scale."""
    return np.array([[10.0, y / 0.40625, x / 0.40625] for (y, x) in pts_um], float)


def test_chain_does_not_bridge_two_nuclei():
    # a chain from A(y=0) to B(y=8um) spaced 2um; r_same=3um must NOT put the 8um-apart ends together
    coords = _xy([(0, 0), (2, 0), (4, 0), (6, 0), (8, 0)])
    labels = same_cell_sets(coords, SCALE, r_same_um=3.0)
    assert labels[0] != labels[-1], "chain bridged two nuclei 8um apart (diameter cap failed)"


def test_distinct_nuclei_within_7um_not_merged():
    # two nuclei 5um apart (inside the 7um evaluator gate) must be SEPARATE sets at r_same=3
    coords = _xy([(0, 0), (5, 0)])
    labels = same_cell_sets(coords, SCALE, r_same_um=3.0)
    assert labels[0] != labels[1]


def test_true_duplicates_merged():
    # two detections 1um apart (same nucleus) must share a set
    coords = _xy([(0, 0), (1, 0)])
    labels = same_cell_sets(coords, SCALE, r_same_um=3.0)
    assert labels[0] == labels[1]


def test_diameter_cap_never_exceeded():
    # random cloud: every cluster's max pairwise physical distance must be <= r_same
    rng = np.random.default_rng(0)
    coords = np.column_stack([np.full(40, 10.0), rng.uniform(0, 40, 40), rng.uniform(0, 40, 40)])
    r_same = 3.0
    labels = same_cell_sets(coords, SCALE, r_same_um=r_same)
    phys = coords * np.array(SCALE)
    for lab in np.unique(labels):
        pts = phys[labels == lab]
        if len(pts) > 1:
            d = np.max(np.linalg.norm(pts[:, None] - pts[None], axis=2))
            assert d <= r_same + 1e-6, f"cluster diameter {d:.3f} exceeds cap {r_same}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
