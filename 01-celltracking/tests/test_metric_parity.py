"""Parity: the fast numpy edge gate vs the AUTHORITATIVE tracksdata matcher on a real crop (red-team #10).

Our numpy metric was only validated on 7 hand-built adversarial cases; this confirms it agrees with the
organizer's DistanceMatching on a real crowded crop (dense-frame tie handling). Skips when local train
data is absent (e.g. CI). The full multi-crop check is scripts/metric/validate_metric_parity.py.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TRAIN = ROOT / "data" / "train"
CROP = "6bba_07477033"  # crowded but small: real per-frame collisions, fast enough for a test


@pytest.mark.skipif(not (TRAIN / f"{CROP}.zarr").exists(), reason="needs local train zarr/geff")
def test_numpy_gate_matches_authoritative_on_real_crop():
    from biotrack.metric import estimated_nodes, score_pred_graph
    from biotrack.metric_numpy import score_sample
    from validate_metric_parity import canonical_oof, geff_to_sample, sample_to_graph

    pred_path = canonical_oof(CROP)
    if not pred_path.exists():
        pytest.skip("needs canonical OOF prediction")
    pred = geff_to_sample(pred_path)
    gt = geff_to_sample(str(TRAIN / f"{CROP}.geff"))
    n_est = estimated_nodes(str(TRAIN / f"{CROP}.geff"))

    npm = score_sample(pred, gt, n_est)
    tdm = score_pred_graph(sample_to_graph(pred, CROP), str(TRAIN / f"{CROP}.geff"))

    assert (npm["edge_tp"], npm["edge_fp"], npm["edge_fn"]) == \
           (tdm["edge_tp"], tdm["edge_fp"], tdm["edge_fn"]), "edge TP/FP/FN must match the authoritative matcher"
    assert abs(npm["adj_edge_jaccard"] - tdm["adj_edge_jaccard"]) < 5e-4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
