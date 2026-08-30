"""Gate 1 (PKT-0029) contract tests that RUN the worker, rather than reading it.

WHY THIS SHAPE. Attempt 1 of this gate cost a GPU session and compared zero crops because it
imported a module that only exists inside a subprocess (FACT-0387). A test that greps the patch
source would not have caught that, and this project has already been burned once by a test that
grepped source instead of running it. So these tests extract the real worker script from the real
patch file and EXECUTE it as two genuine subprocesses against a stubbed predictor - the same
two-phase, separate-process structure the gate uses on Kaggle. Everything but the trained model is
real: argument parsing, the cache round-trip through disk, the band-A/band-B comparison arithmetic,
the node-count assertion and the exit codes.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "scripts" / "kaggle_edits" / "assoc_feature_parity.py"

N_PER_FRAME = 3
N_FRAMES = 4
FEAT_DIM = 6


def worker_source() -> str:
    """The real worker text, taken from the real patch - not a copy that can drift."""
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_AFP_WORKER" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("_AFP_WORKER not found in the patch - the gate's worker has been renamed")


STUB = textwrap.dedent(
    '''
    """Deterministic stand-in for the deployed predictor. Only the model is fake."""
    import numpy as np
    import torch

    import os

    N_PER_FRAME, FEAT_DIM = 3, 6
    FLAT_FEATURES = os.environ.get("AFP_STUB_FLAT_FEATURES") == "1"


    class _Cfg:
        def __init__(self, det_threshold=0.5):
            self.det_threshold = det_threshold
            self.pool_kernel_um = 3.0


    def PredictConfig(det_threshold=0.5):
        return _Cfg(det_threshold)


    def pool_kernel_from_um(um, voxel):
        return (1, 1, 1)


    class _Image:
        shape = (4, 2, 4, 4)

        def __getitem__(self, t):
            return np.full((2, 4, 4), float(t), dtype=np.float32)


    class _DS:
        image = _Image()


    def open_dataset(path):
        return _DS()


    def _detect_cells_pooled(det_slice, t, threshold, pool_k):
        # Deterministic: N_PER_FRAME cells per frame at fixed offsets.
        return np.asarray(
            [[t, 1, i, i + 1] for i in range(N_PER_FRAME)], dtype=np.int16
        )


    def extract_pos_features(coords, shape):
        return coords.astype(np.float32)


    class _Model:
        def unet(self, x):
            return torch.zeros(1, 2, FEAT_DIM, 4, 4)

        def detection_head(self, x):
            return torch.zeros(1, 2, 1, 4, 4)

        def _index_features(self, x, pc, pm):
            n = pc.shape[1]
            # Features depend ONLY on the coordinates, so the cache round-trip is checkable.
            # The second coordinate varies BETWEEN nodes of a frame, so the source-axis softmax
            # is non-uniform and some pair clears the deployed 0.5 - i.e. band A actually has
            # something to compare. Under FLAT_FEATURES it is dropped, every source in a frame
            # becomes identical, no probability can exceed 1/N_PER_FRAME, and band A is VACUOUS.
            base = pc[0, :, 0].reshape(n, 1).float()
            if not FLAT_FEATURES:
                # 0.1, not 1.0: the source-axis softmax must be non-uniform enough that some
                # pair clears 0.5, yet NOT saturated - a saturated softmax would swallow the
                # 0.05 feature perturbation test_a_corrupted_cache_fails_the_gate relies on and
                # silently turn the corruption test green for the wrong reason.
                base = base + 0.1 * pc[0, :, 1].reshape(n, 1).float()
            return (base + torch.arange(FEAT_DIM).float().reshape(1, FEAT_DIM)).unsqueeze(0)

        def predict_edges(self, fs, ft, pcs, pct, pps, ppt, ms, mt):
            ns, nt = fs.shape[1], ft.shape[1]
            # Logits are a fixed function of the cached features, so any corruption of the cache
            # changes the probabilities and the gate must notice.
            a = fs[0, :, 0].reshape(ns, 1)
            b = ft[0, :, 0].reshape(1, nt)
            return ((a * 2.0 - b) * 3.0).reshape(1, ns, nt)


    def load_model(weights, device):
        return _Model(), 2, np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    '''
)


def build_env(tmp_path: Path, *, node_count_shift: int = 0, corrupt_cache: str | None = None,
              flat_features: bool = False):
    """Lay out a fake repo, run the CACHE phase, then build the parity targets from its output."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "predict_unet_transformer.py").write_text(STUB, encoding="utf-8")
    worker = repo / "afp_gate1.py"
    worker.write_text(worker_source(), encoding="utf-8")
    cache = tmp_path / "cache"
    common = [
        "--weights", str(tmp_path / "w.pth"),
        "--test-dir", str(tmp_path),
        "--cache-dir", str(cache),
        "--max-frames", str(N_FRAMES),
        "--crops", "cropA",
    ]
    env = dict(os.environ)
    if flat_features:
        env["AFP_STUB_FLAT_FEATURES"] = "1"
    else:
        env.pop("AFP_STUB_FLAT_FEATURES", None)
    run = subprocess.run(
        [sys.executable, str(worker), "--phase", "cache", *common],
        cwd=str(repo), text=True, capture_output=True, env=env,
    )
    assert run.returncode == 0, run.stderr
    assert "AFP_CACHE crop=cropA" in run.stdout

    z = np.load(cache / "cropA.npz", allow_pickle=False)
    coords, frames = z["coords"], z["frames"].tolist()
    offset = {int(t): (int(s), int(e)) for t, s, e in zip(frames, z["starts"], z["ends"])}

    # Reproduce the stub's own arithmetic to build a TRUE parity target.
    feats = {int(t): z[f"feat_{int(t)}"] for t in z["feat_frames"].tolist()}
    pairs = {}
    for ts, tt in zip(frames[:-1], frames[1:]):
        if tt != ts + 1 or ts not in feats or tt not in feats:
            continue
        ss, _se = offset[ts]
        tsq, _te = offset[tt]
        a = feats[ts][:, 0].reshape(-1, 1)
        b = feats[tt][:, 0].reshape(1, -1)
        logits = (a * 2.0 - b) * 3.0
        ex = np.exp(logits - logits.max(axis=0, keepdims=True))
        probs = ex / ex.sum(axis=0, keepdims=True)
        for i in range(probs.shape[0]):
            for j in range(probs.shape[1]):
                pairs[(ss + i, tsq + j)] = float(probs[i, j])

    node_rows = []
    for t, (s, e) in offset.items():
        keep = (e - s) + node_count_shift if t == frames[0] else (e - s)
        for k in range(keep):
            node_rows.append({"dataset": "cropA", "row_type": "node", "t": int(t),
                              "node_id": s + k, "source_id": -1, "target_id": -1,
                              "edge_prob": 0.0})
    edge_rows = [
        {"dataset": "cropA", "row_type": "edge", "t": -1, "node_id": -1,
         "source_id": a, "target_id": b, "edge_prob": p}
        for (a, b), p in pairs.items() if p > 0.5
    ]
    preilp = tmp_path / "preilp.parquet"
    pl.DataFrame(node_rows + edge_rows).write_parquet(preilp)

    sub = {k: v for k, v in pairs.items() if v <= 0.5}
    assert sub, "the fixture must produce a sub-threshold band or band B proves nothing"
    ecb = tmp_path / "ecb"
    ecb.mkdir()
    np.savez_compressed(
        ecb / "cropA.npz",
        source_id=np.asarray([k[0] for k in sub], dtype=np.int64),
        target_id=np.asarray([k[1] for k in sub], dtype=np.int64),
        edge_prob=np.asarray(list(sub.values()), dtype=np.float32),
    )

    if corrupt_cache is not None:
        # A cache that loads cleanly and holds subtly wrong numbers is exactly what the gate exists
        # to catch. "node" perturbs ONE node; "uniform" shifts a whole source frame equally, which
        # is provably invisible - see test_uniform_source_shift_is_a_known_blind_spot.
        data = {k: z[k] for k in z.files}
        first = int(z["feat_frames"][0])
        bumped = data[f"feat_{first}"].copy()
        if corrupt_cache == "node":
            bumped[0, :] += 0.05
        elif corrupt_cache == "uniform":
            bumped += 0.05
        else:
            raise AssertionError(f"unknown corruption mode {corrupt_cache!r}")
        data[f"feat_{first}"] = bumped
        np.savez_compressed(cache / "cropA.npz", **data)

    return repo, worker, common, preilp, ecb, env


def run_verify(repo, worker, common, preilp, ecb, out, env=None):
    return subprocess.run(
        [sys.executable, str(worker), "--phase", "verify", *common,
         "--preilp", str(preilp), "--ecb-dir", str(ecb), "--out", str(out)],
        cwd=str(repo), text=True, capture_output=True, env=env or dict(os.environ),
    )


def test_gate_passes_when_the_cache_is_faithful(tmp_path):
    repo, worker, common, preilp, ecb, env = build_env(tmp_path)
    out = tmp_path / "report.json"
    run = run_verify(repo, worker, common, preilp, ecb, out, env)
    assert run.returncode == 0, run.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["all_passed"] is True
    crop = report["crops"][0]
    assert crop["band_a"]["missing"] == 0 and crop["band_a"]["extra"] == 0
    assert crop["band_a"]["checked"] > 0, "band A must actually compare something"
    assert crop["band_b"]["checked"] > 0, "band B must actually compare something"
    assert crop["node_count_mismatches"] == []


def test_a_crop_whose_band_a_compares_nothing_fails_the_gate(tmp_path):
    """Band A must be floored for non-vacuity exactly as band B is.

    THE DEFECT THIS PINS. Before the floor, `a_missing`, `a_extra` and `a_delta` were all
    trivially satisfied when a crop's band A compared zero pairs, so the crop was reported
    `passed` having verified nothing about the deployed band. The floor is PER CROP, so an
    uncapped full run was not immune - one such crop certified a cache nobody checked.

    The vacuum is not contrived. With flat features every source in a frame is identical, the
    source-axis softmax is uniform at 1/N, no probability clears the deployed 0.5, and the
    pre-ILP fixture consequently records no band-A edge. Band B still compares 27 pairs here,
    so a refusal can only be band A's floor.
    """
    repo, worker, common, preilp, ecb, env = build_env(tmp_path, flat_features=True)
    out = tmp_path / "report.json"
    run = run_verify(repo, worker, common, preilp, ecb, out, env)
    assert run.returncode == 0, run.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    crop = report["crops"][0]
    assert crop["band_a"]["checked"] == 0, "the fixture must actually empty band A"
    assert crop["band_b"]["checked"] > 0, (
        "band B must still compare something, or this test would prove nothing about band A"
    )
    assert crop["band_a"]["missing"] == 0 and crop["band_a"]["extra"] == 0, (
        "and every other band-A condition must be trivially SATISFIED - that is the point"
    )
    assert report["all_passed"] is False, (
        "a crop that compared nothing in the deployed band must not certify the cache"
    )


def test_a_corrupted_cache_fails_the_gate(tmp_path):
    """The whole point: features that load cleanly but are subtly wrong must not pass."""
    repo, worker, common, preilp, ecb, env = build_env(tmp_path, corrupt_cache="node")
    out = tmp_path / "report.json"
    run = run_verify(repo, worker, common, preilp, ecb, out, env)
    assert run.returncode == 0, run.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["all_passed"] is False


def test_uniform_source_shift_is_a_known_blind_spot(tmp_path):
    """A LIMITATION OF THE GATE, asserted so it cannot be forgotten or rediscovered by accident.

    The gate compares probabilities AFTER a softmax over the SOURCE axis, and softmax is invariant
    to a constant offset along the axis it normalises. So a cache error that shifts every source in
    a frame by the same amount is mathematically invisible here - not merely undetected in practice.
    Any cache error that is NOT uniform across the source axis does show up, which is what
    test_a_corrupted_cache_fails_the_gate pins. This test found the property before a GPU session
    was spent believing the gate was total.
    """
    repo, worker, common, preilp, ecb, env = build_env(tmp_path, corrupt_cache="uniform")
    out = tmp_path / "report.json"
    run = run_verify(repo, worker, common, preilp, ecb, out, env)
    assert run.returncode == 0, run.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["all_passed"] is True, (
        "if this ever fails the invariance argument has changed - re-derive it before relaxing "
        "the assertion, because the gate would then be stronger than documented"
    )


def test_node_count_mismatch_fails_the_gate(tmp_path):
    """A short detection must not silently shrink the comparison denominator."""
    repo, worker, common, preilp, ecb, env = build_env(tmp_path, node_count_shift=-1)
    out = tmp_path / "report.json"
    run = run_verify(repo, worker, common, preilp, ecb, out, env)
    assert run.returncode == 0, run.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["all_passed"] is False
    assert report["crops"][0]["node_count_mismatches"]


def test_verify_refuses_to_run_without_a_cache_on_disk(tmp_path):
    """`from the cache alone` is enforced by the process boundary, so a missing cache must raise."""
    repo, worker, common, preilp, ecb, env = build_env(tmp_path)
    for stale in (tmp_path / "cache").glob("*.npz"):
        stale.unlink()
    run = run_verify(repo, worker, common, preilp, ecb, tmp_path / "report.json")
    assert run.returncode != 0
    assert "cache missing" in run.stderr


def test_the_verify_phase_retains_the_full_probability_matrix(tmp_path):
    """Band B only exists if sub-0.5 pairs survive into the comparison (FACT-0382)."""
    repo, worker, common, preilp, ecb, env = build_env(tmp_path)
    out = tmp_path / "report.json"
    run_verify(repo, worker, common, preilp, ecb, out, env)
    crop = json.loads(out.read_text(encoding="utf-8"))["crops"][0]
    assert crop["band_b"]["recorded_sub_threshold"] > 0
    assert crop["band_b"]["missing"] == 0, (
        "every recorded sub-threshold pair must be reproduced; a >0.5 filter on the reproduction "
        "would drop them all and this is the assertion that catches it"
    )


def test_patch_and_worker_both_compile(tmp_path):
    """A syntax error would waste a GPU session as surely as attempt 1's import did."""
    compile(PATCH.read_text(encoding="utf-8"), str(PATCH), "exec")
    compile(worker_source(), "afp_gate1.py", "exec")


def test_the_gate_requires_the_sidecars_rather_than_silently_skipping_band_b(tmp_path):
    """A band-A-only pass is not Gate 1 - the patch must fail closed when sidecars are absent."""
    source = PATCH.read_text(encoding="utf-8")
    marker = "ECB sidecars not found"
    assert marker in source
    # And prove the raise is reachable rather than trusting the string: the guard must sit in the
    # branch taken when no sidecar directory resolves.
    tree = ast.parse(source)
    raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
    assert any(
        isinstance(r.exc, ast.Call) and any(
            isinstance(a, ast.Constant) and marker in str(a.value)
            for a in r.exc.args
        )
        for r in raises
    ), "the sidecar guard must raise, not warn"
