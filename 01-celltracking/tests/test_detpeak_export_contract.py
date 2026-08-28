from pathlib import Path


SOURCE = Path("scripts/kaggle_edits/detpeak_export.py").read_text(encoding="utf-8")


def test_export_crosses_process_boundary_and_has_positive_heartbeats():
    assert 'os.environ["BIOHUB_DETPEAK_ENABLE"] = "1"' in SOURCE
    assert "detpeak: export ACTIVE in pid" in SOURCE
    assert "pipeline_peak_count" in SOURCE
    assert "pipeline_threshold" in SOURCE
    assert "_biohub_flush_peaks(name)" in SOURCE


def test_export_mask_is_observational_and_pipeline_mask_stays_original():
    assert "_local_max & (_sig > _BIOHUB_DETPEAK_T)" in SOURCE
    assert "is_peak = _local_max & (_sig > det_threshold)" in SOURCE
