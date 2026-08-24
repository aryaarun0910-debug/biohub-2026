from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from scripts.core import kaggle_factory as KF
from scripts.kaggle_edits.h1r_adabn_detection_only import (
    CALIBRATED_SHA256,
    CALIBRATION_MODE,
    patch_detection_only_predictor,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "scripts/kaggle_specs/p13_p9_adabn_detection_only.json"
P9_SPEC_PATH = ROOT / "scripts/kaggle_specs/p9_coupled_division.json"
PRODUCER_PATH = ROOT / "scripts/kaggle_specs/h1r_det_s1_adabn_control.json"
PATCH_PATH = ROOT / "scripts/kaggle_edits/h1r_adabn_detection_only.py"
P9_NOTEBOOK = ROOT / "notebooks/kaggle_p9_coupled_division/biohub-p9-coupled-division.ipynb"
P13_NOTEBOOK = ROOT / "notebooks/kaggle_p13_p9_adabn_detection_only/biohub-p13-p9-adabn-detection-only.ipynb"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_texts(notebook: dict) -> list[str]:
    return [
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
    ]


def test_p13_is_an_isolated_single_edit_of_exact_built_p9() -> None:
    spec, p9 = _load(SPEC_PATH), _load(P9_SPEC_PATH)
    assert spec["base_notebook"] == (
        "notebooks/kaggle_p9_coupled_division/biohub-p9-coupled-division.ipynb"
    )
    assert spec["base_sha256"] == hashlib.sha256(P9_NOTEBOOK.read_bytes()).hexdigest()
    assert spec["datasets"] == p9["datasets"]
    assert spec["competition_sources"] == p9["competition_sources"]
    assert spec["enable_gpu"] == p9["enable_gpu"]
    assert spec["enable_internet"] == p9["enable_internet"]
    assert spec["machine_shape"] == p9["machine_shape"]
    assert spec["edits"] == [{
        "kind": "insert_before", "cell_match": "start_time = time.time()",
        "anchor": "start_time = time.time()",
        "code_file": "scripts/kaggle_edits/h1r_adabn_detection_only.py", "expect": 1,
    }]


def test_built_p13_equals_p9_after_removing_exact_bridge_bytes() -> None:
    base, built = _load(P9_NOTEBOOK), _load(P13_NOTEBOOK)
    base_cells, built_cells = _cell_texts(base), _cell_texts(built)
    assert len(base_cells) == len(built_cells) == 11
    changed = [i for i, (a, b) in enumerate(zip(base_cells, built_cells)) if a != b]
    assert changed == [5]
    bridge = PATCH_PATH.read_text(encoding="utf-8")
    expected = base_cells[5].replace(
        "start_time = time.time()",
        "\n" + bridge + "\nstart_time = time.time()",
        1,
    )
    assert built_cells[5] == expected

    # These cells carry P9's coupled radii/function and submission postprocessing.
    assert built_cells[2] == base_cells[2]
    assert built_cells[6] == base_cells[6]
    for literal in (
        'os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = \'8.0\'',
        'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = \'11.0\'',
        'os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = \'10.0\'',
        'os.environ["BIOHUB_SAFE_DIV_DIVERGE_UM"] = \'2.25\'',
        "Coupled safe-division transplant for the P3 harmonic deployment.",
    ):
        assert literal in "\n".join(built_cells)


def test_p13_runtime_keeps_all_association_on_p9_and_all_eight_views_on_adabn(
    tmp_path,
) -> None:
    """Replay P9's predictor rewrites before applying the bridge to its exact runtime surface."""
    notebook = _load(P9_NOTEBOOK)
    notebook_code = "\n".join(
        text for cell, text in zip(notebook["cells"], _cell_texts(notebook))
        if cell["cell_type"] == "code"
    )
    wanted = {
        "_old", "_new", "_ensemble_replacements", "_guard_old", "_guard_new",
        "_bi_old", "_bi_new",
    }
    values = {}
    for node in ast.walk(ast.parse(notebook_code)):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            values[target.id] = ast.literal_eval(node.value)

    source = (ROOT / "vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py").read_text(
        encoding="utf-8"
    )
    rewrites = [
        (values["_old"], values["_new"]), *values["_ensemble_replacements"],
        (values["_guard_old"], values["_guard_new"]),
        (values["_bi_old"], values["_bi_new"]),
    ]
    for index, (old, new) in enumerate(rewrites):
        assert source.count(old) == 1, f"P9 predictor rewrite {index} drifted"
        source = source.replace(old, new, 1)

    predictor = tmp_path / "predict_unet_transformer.py"
    predictor.write_text(source, encoding="utf-8")
    patch_detection_only_predictor(predictor)
    patched = predictor.read_text(encoding="utf-8")
    tta = patched[
        patched.index("        if cfg.det_tta:"):
        patched.index("        secondary_unet_out = None")
    ]
    # Five source call sites execute 1 identity + 3 flips + 2 rotations + 2 transposes.
    assert patched.count("adabn_detection_model.encode(") == 5
    assert tta.count("adabn_detection_model.encode(") == 4
    assert 1 + 3 + 2 + 1 + 1 == 8
    assert "_, det_flip = model.encode(" not in tta
    assert "_, det_rot = model.encode(" not in tta
    assert "_, det_t = model.encode(" not in tta
    assert "_, det_at = model.encode(" not in tta
    assert patched.count("model._index_features(\n                unet_out") == 2
    assert "adabn_detection_model._index_features" not in patched
    assert "adabn_detection_model.predict_edges" not in patched
    assert "secondary_model.predict_edges" in patched
    assert "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT" in patched


def test_p13_reciprocal_contract_hashes_and_defect_gate() -> None:
    spec, producer = _load(SPEC_PATH), _load(PRODUCER_PATH)
    consumed = {
        "producer_spec": "scripts/kaggle_specs/h1r_det_s1_adabn_control.json",
        "artifact": "edge_predictor_best.pth",
        "mode": CALIBRATION_MODE,
        "sha256": CALIBRATED_SHA256,
    }
    declared = {
        "artifact": "edge_predictor_best.pth",
        "spec": "scripts/kaggle_specs/p13_p9_adabn_detection_only.json",
        "mode": CALIBRATION_MODE,
        "sha256": CALIBRATED_SHA256,
    }
    assert spec["consumes_artifacts"] == [consumed]
    assert declared in producer["deploy_consumers"]
    assert spec["kernel_sources"] == [
        "aryaarun07/biohub-h1r-detector-s1-adabn-control"
    ]
    assert spec["provenance"]["consumer_patch_sha256"] == hashlib.sha256(
        PATCH_PATH.read_bytes()
    ).hexdigest()
    assert spec["provenance"]["expected_built_notebook_sha256"] == hashlib.sha256(
        P13_NOTEBOOK.read_bytes()
    ).hexdigest()
    applicable, violations = KF.validate_defect_gate(spec, _load(P13_NOTEBOOK))
    assert applicable == ["DG-008-calibration-crosses-into-association"]
    assert violations == []
