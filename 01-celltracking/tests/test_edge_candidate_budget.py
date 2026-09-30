"""Contracts for the LEVER-0037 edge-candidate budget patch.

Software contracts only. Which threshold or top-k to deploy is an experiment result.

The patch rewrites the vendored predictor source, so these tests apply it to a real copy of
that source and then exercise the generated code. Testing a hand-written imitation would
prove nothing about the thing that actually runs on Kaggle.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "scripts" / "kaggle_edits" / "edge_candidate_budget.py"
PREDICTOR = ROOT / "vendor" / "kaggle-cell-tracking" / "scripts" / "predict_unet_transformer.py"


class _Env:
    """Set environment variables for the duration of a block, then restore them.

    The generated config block reads the environment at IMPORT time, so the variables must
    still be set when that block is executed - not merely when the patch is applied.
    """

    def __init__(self, env: dict[str, str] | None):
        self.env = env or {}
        self.previous: dict[str, str | None] = {}

    def __enter__(self):
        self.previous = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        return self

    def __exit__(self, *exc):
        for k, v in self.previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def apply_patch(tmp_path: Path, env: dict[str, str] | None = None) -> Path:
    """Run the patch against a copy of the real predictor, returning the patched file."""
    target = tmp_path / "predict_unet_transformer.py"
    target.write_text(PREDICTOR.read_text(encoding="utf-8"), encoding="utf-8")
    with _Env(env):
        exec(compile(PATCH.read_text(encoding="utf-8"), str(PATCH), "exec"), {"_ps": target})
    return target


def load_select(target: Path, env: dict[str, str] | None = None):
    """Extract the generated `_ecb_select` and its config into a live namespace."""
    namespace: dict = {"np": np}
    source = target.read_text(encoding="utf-8")
    start = source.index("# --- LEVER-0037 edge-candidate budget")
    end = source.index("# --- end LEVER-0037 edge-candidate budget")
    with _Env(env):
        exec(compile(source[start:end], "<ecb>", "exec"), namespace)
    return namespace


def patch_and_load(tmp_path: Path, env: dict[str, str] | None = None):
    """Apply the patch and evaluate its generated config under the SAME environment."""
    return load_select(apply_patch(tmp_path, env), env)


def test_patch_applies_to_the_real_predictor_and_compiles(tmp_path):
    target = apply_patch(tmp_path)
    compile(target.read_text(encoding="utf-8"), str(target), "exec")
    assert "_ecb_select(probs, n_src, n_tgt" in target.read_text(encoding="utf-8")


def test_patch_fails_closed_when_the_primary_path_is_absent(tmp_path):
    """A silent no-op would look exactly like 'the richer candidates do not exist'."""
    target = tmp_path / "predict_unet_transformer.py"
    mangled = PREDICTOR.read_text(encoding="utf-8").replace(
        "if probs[i, j] > cfg.threshold", "if probs[i, j] > cfg.threshold  # moved", 1
    )
    target.write_text(mangled, encoding="utf-8")
    with pytest.raises(AssertionError, match="primary path was NOT patched"):
        exec(compile(PATCH.read_text(encoding="utf-8"), str(PATCH), "exec"), {"_ps": target})


def test_default_selection_is_identical_to_the_deployed_comprehension(tmp_path):
    """Bit-for-bit champion preservation rests on this: unset env == deployed behaviour."""
    ns = load_select(apply_patch(tmp_path))
    assert ns["_ECB_THRESHOLD"] is None
    assert ns["_ECB_TOPK"] is None
    assert ns["_ECB_EXPORT_ON"] is False

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(7, 5))
    probs = np.exp(logits) / np.exp(logits).sum(axis=0, keepdims=True)  # softmax over SOURCES
    deployed = sorted(
        [(probs[i, j], i, j) for i in range(7) for j in range(5) if probs[i, j] > 0.5],
        reverse=True,
    )
    assert ns["_ecb_select"](probs, 7, 5, 0.5, None) == deployed


def test_default_threshold_guarantees_at_most_one_parent_per_target(tmp_path):
    """FACT-0369's arithmetic: a softmax over sources sums to 1, so only one entry per
    column can exceed 0.5. This is the property the default configuration must keep."""
    ns = load_select(apply_patch(tmp_path))
    rng = np.random.default_rng(7)
    for _ in range(50):
        n_src, n_tgt = int(rng.integers(2, 12)), int(rng.integers(2, 12))
        logits = rng.normal(scale=3.0, size=(n_src, n_tgt))
        probs = np.exp(logits) / np.exp(logits).sum(axis=0, keepdims=True)
        picked = ns["_ecb_select"](probs, n_src, n_tgt, 0.5, None)
        per_target: dict[int, int] = {}
        for _p, _i, j in picked:
            per_target[j] = per_target.get(j, 0) + 1
        assert max(per_target.values(), default=0) <= 1


def test_lower_threshold_offers_more_parents_and_topk_bounds_them(tmp_path):
    """The whole point of the lever: below 0.5 a target can have several candidate parents,
    and top-k is what keeps that bounded."""
    ns = load_select(apply_patch(tmp_path))
    rng = np.random.default_rng(11)
    logits = rng.normal(scale=0.2, size=(9, 6))          # deliberately flat -> diluted softmax
    probs = np.exp(logits) / np.exp(logits).sum(axis=0, keepdims=True)

    assert ns["_ecb_select"](probs, 9, 6, 0.5, None) == []      # nothing clears 0.5
    wide = ns["_ecb_select"](probs, 9, 6, 0.05, None)
    assert len(wide) > 0
    counts: dict[int, int] = {}
    for _p, _i, j in wide:
        counts[j] = counts.get(j, 0) + 1
    assert max(counts.values()) > 1                             # more than one parent offered

    for k in (1, 2, 3):
        capped = ns["_ecb_select"](probs, 9, 6, 0.05, k)
        per_target: dict[int, int] = {}
        for _p, _i, j in capped:
            per_target[j] = per_target.get(j, 0) + 1
        assert max(per_target.values()) <= k
        assert len(capped) <= len(wide)


def test_topk_one_reduces_to_one_parent_per_target_at_any_threshold(tmp_path):
    ns = load_select(apply_patch(tmp_path))
    rng = np.random.default_rng(3)
    probs = rng.random((6, 4))                                   # not even a softmax
    picked = ns["_ecb_select"](probs, 6, 4, 0.01, 1)
    per_target: dict[int, int] = {}
    for _p, _i, j in picked:
        per_target[j] = per_target.get(j, 0) + 1
    assert max(per_target.values(), default=0) <= 1


def test_topk_keeps_the_highest_scoring_parents(tmp_path):
    ns = load_select(apply_patch(tmp_path))
    probs = np.array([[0.1], [0.9], [0.5], [0.7]])                # one target, four sources
    picked = ns["_ecb_select"](probs, 4, 1, 0.01, 2)
    assert [i for _p, i, _j in picked] == [1, 3]


def test_export_configuration_is_separate_from_the_treatment(tmp_path):
    """An acquisition run must be able to export WITHOUT changing the consumed graph."""
    ns = patch_and_load(tmp_path, {
        "BIOHUB_EDGE_CANDIDATE_EXPORT_THRESHOLD": "0.05",
        "BIOHUB_EDGE_CANDIDATE_EXPORT_TOPK": "4",
        "BIOHUB_EDGE_CANDIDATE_EXPORT_DIR": str(tmp_path / "ecb"),
    })
    assert ns["_ECB_EXPORT_ON"] is True
    assert ns["_ECB_EXPORT_TOPK"] == 4
    # The treatment knobs stay unset, so the consumed candidate list is the deployed one.
    assert ns["_ECB_THRESHOLD"] is None
    assert ns["_ECB_TOPK"] is None


def test_export_is_off_unless_both_dir_and_threshold_are_given(tmp_path):
    only_dir = patch_and_load(tmp_path, {
        "BIOHUB_EDGE_CANDIDATE_EXPORT_DIR": str(tmp_path / "ecb"),
    })
    assert only_dir["_ECB_EXPORT_ON"] is False
    only_threshold = patch_and_load(tmp_path, {
        "BIOHUB_EDGE_CANDIDATE_EXPORT_THRESHOLD": "0.05",
    })
    assert only_threshold["_ECB_EXPORT_ON"] is False


def test_out_of_range_configuration_raises_rather_than_silently_clamping(tmp_path):
    for value in ("0", "1", "1.5", "-0.2"):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            patch_and_load(tmp_path, {"BIOHUB_EDGE_CANDIDATE_THRESHOLD": value})
    with pytest.raises(ValueError, match=">= 1"):
        patch_and_load(tmp_path, {"BIOHUB_EDGE_CANDIDATE_TOPK": "0"})


def test_sidecar_flush_actually_produces_a_readable_npz(tmp_path):
    """EXECUTE the generated flush, do not merely grep it.

    An earlier version of this test asserted the tmp-write/rename pattern appeared in the
    source and passed, while the real flush crashed every Kaggle run: np.savez_compressed
    APPENDS '.npz' to a path that lacks it, so it wrote '<crop>.npz.tmp.npz' and the rename
    of '<crop>.npz.tmp' raised FileNotFoundError. A string check cannot see that; running the
    code can. The flush is extracted from the patched source and executed against real arrays.
    """
    target = apply_patch(tmp_path, {
        "BIOHUB_EDGE_CANDIDATE_EXPORT_THRESHOLD": "0.05",
        "BIOHUB_EDGE_CANDIDATE_EXPORT_DIR": str(tmp_path / "ecb"),
    })
    source = target.read_text(encoding="utf-8")
    start = source.index("    if _ECB_EXPORT_ON:\n        from pathlib import Path as _EcbPath")
    end = source.index("    return coords, all_edges", start)
    block = "\n".join(
        line[4:] if line.startswith("    ") else line
        for line in source[start:end].split("\n")
    )

    class _DsPath:
        stem = "44b6_testcrop"

    ns = {
        "np": np,
        "_ECB_EXPORT_ON": True,
        "_ECB_EXPORT_DIR": str(tmp_path / "ecb"),
        "_ECB_EXPORT_THRESHOLD": 0.05,
        "_ECB_EXPORT_TOPK": 8,
        "_ECB_BUFFER": [np.array([[1.0, 2.0, 0.9], [3.0, 4.0, 0.6]])],
        "_ECB_CROP": {"pairs": 7, "frame_pairs": 3},
        "_ECB_TOTAL": {"pairs": 0, "exported": 0, "crops": 0},
        "cfg": type("C", (), {"threshold": 0.5})(),
        "ds_path": _DsPath(),
    }
    exec(compile(block, "<flush>", "exec"), ns)

    written = tmp_path / "ecb" / "44b6_testcrop.npz"
    assert written.exists(), sorted(p.name for p in (tmp_path / "ecb").iterdir())
    assert not list((tmp_path / "ecb").glob("*.tmp*")), "temporary file left behind"
    with np.load(written) as z:
        assert z["source_id"].tolist() == [1, 3]
        assert z["target_id"].tolist() == [2, 4]
        assert float(z["deployed_threshold"]) == 0.5
        assert int(z["deployed_candidate_count"]) == 7
    # per-crop state must be cleared for the next crop
    assert ns["_ECB_BUFFER"] == []
    assert ns["_ECB_CROP"] == {"pairs": 0, "frame_pairs": 0}
    assert ns["_ECB_TOTAL"]["crops"] == 1


def test_per_crop_counters_reset_so_the_heartbeat_is_not_cumulative(tmp_path):
    """`deployed_candidate_count` is a per-crop number; leaking it across crops would
    silently inflate every sidecar after the first."""
    source = apply_patch(tmp_path).read_text(encoding="utf-8")
    assert '_ECB_CROP["pairs"] = 0' in source
    assert '_ECB_CROP["frame_pairs"] = 0' in source
    assert "_ECB_BUFFER.clear()" in source


def test_heartbeat_reports_the_four_required_fields(tmp_path):
    source = apply_patch(tmp_path).read_text(encoding="utf-8")
    for field in ("export_threshold=", "export_topk=", "exported_candidates=", "crops_done="):
        assert field in source, field


def test_composes_with_the_preilp_export_patch(tmp_path):
    """PKT-0027 applies BOTH patches to the same predictor source in one kernel.

    They touch different anchors - pre-ILP wraps `build_graph`, this one rewrites the
    candidate list and `predict_video`'s return - but an ordering or anchor collision would
    only surface on Kaggle, after a GPU session had been spent.
    """
    target = tmp_path / "predict_unet_transformer.py"
    target.write_text(PREDICTOR.read_text(encoding="utf-8"), encoding="utf-8")
    preilp = ROOT / "scripts" / "kaggle_edits" / "pre_ilp_export.py"
    env = {
        "BIOHUB_EDGE_CANDIDATE_EXPORT_THRESHOLD": "0.02",
        "BIOHUB_EDGE_CANDIDATE_EXPORT_DIR": str(tmp_path / "ecb"),
    }
    with _Env(env):
        exec(compile(preilp.read_text(encoding="utf-8"), str(preilp), "exec"), {"_ps": target})
        exec(compile(PATCH.read_text(encoding="utf-8"), str(PATCH), "exec"), {"_ps": target})
    source = target.read_text(encoding="utf-8")
    compile(source, str(target), "exec")
    assert "_pi_dir" in source            # pre-ILP export survived
    assert "_ecb_select(probs" in source  # candidate budget survived
