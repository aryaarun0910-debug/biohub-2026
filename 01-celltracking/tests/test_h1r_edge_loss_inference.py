from __future__ import annotations

from pathlib import Path

from scripts.kaggle_edits.h1r_edge_loss_patch import (
    apply_h1r_edge_loss_inference_patch,
)


ROOT = Path(__file__).resolve().parents[1]
PREDICT = ROOT / "vendor" / "kaggle-cell-tracking" / "scripts" / "predict_unet_transformer.py"


def test_inference_defaults_to_training_background_normalization(tmp_path) -> None:
    predictor = tmp_path / PREDICT.name
    predictor.write_text(PREDICT.read_text(encoding="utf-8"), encoding="utf-8")
    apply_h1r_edge_loss_inference_patch(predictor)
    source = predictor.read_text(encoding="utf-8")
    assert '"softmax_bg" if os.environ.get("H1R_BG_TERM", "1") == "1"' in source
    assert 'unsupported H1R_EDGE_PROB=' in source
    assert 'elif _h1r_edge_prob == "softmax":' in source
    compile(source, str(predictor), "exec")
