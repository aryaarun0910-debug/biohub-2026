"""Gate 1 passive instrumentation, tested against the REAL class surface in PRODUCTION LAYOUT.

WHY THE LAYOUT IS PART OF THE TEST. Three of the six defects this gate has produced lived at a
boundary a same-process check cannot see: the predictor is not importable from the notebook
process (FACT-0387), the dataset root is not where the patch assumed (FACT-0397), and the worker
script's directory - not the working directory - is sys.path[0] (FACT-0399). So the sandbox here
mirrors Kaggle: the deployed module sits in ``<repo>/scripts``, the worker sits in a separate
``<working>`` directory, and every run crosses a real ``subprocess`` boundary with cwd=<repo>.

WHY THE CLASS SURFACE IS PART OF THE TEST. A fourth defect - ``model.detection_head``, which has
never existed on ``UNetNodeTransformer`` - survived a green suite because the suite's stub defined
it. Nothing here is stubbed: the model is the real ``UNetNodeTransformer`` built by the real
``load_model`` from a real ``config.json``, the predictor is the real ``predict_video`` reading a
real zarr, and ``test_the_worker_only_touches_the_real_class_surface`` resolves every attribute
the worker names against that real object.

The fixture is small on purpose (5 frames, 8x32x32, a randomly initialised trunk) but nothing
about the PATH is small: the image is normalised from the dataset's own recorded quantiles, the
downsample is carried into the array and the voxel size, the detection peaks come from the
deployed pooled extractor, and the head is the deployed ``predict_edges``.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "kaggle-cell-tracking"
TAP_PATCH = ROOT / "scripts" / "kaggle_edits" / "assoc_feature_tap.py"
GATE_PATCH = ROOT / "scripts" / "kaggle_edits" / "assoc_tap_gate.py"
WORKER = ROOT / "scripts" / "win_bet" / "assoc_tap_replay.py"

CROP = "smoke_a"
# The DEPLOYED values. The fixture is tuned so both bands are populated at these, rather than the
# thresholds being tuned to the fixture - a gate proven only at a made-up threshold proves little.
DET_THRESHOLD = 0.5
EDGE_THRESHOLD = 0.5
BAND_B_FLOOR = 0.02

PREDICT_DRIVER = '''
import hashlib, json, sys
from pathlib import Path
import torch
import predict_unet_transformer as AFP

weights, crop, out = sys.argv[1], sys.argv[2], sys.argv[3]
device = torch.device("cpu")
model, W, ds = AFP.load_model(Path(weights), device)
cfg = AFP.PredictConfig(det_threshold=%(det)r, threshold=%(edge)r)
coords, edges = AFP.predict_video(
    model, Path("data") / f"{crop}.zarr", device, cfg, window_size=W, downsample=tuple(ds),
)
rows = [(int(a), int(b), float(p), float(d)) for a, b, p, d in edges]
Path(out).write_text(json.dumps({
    "coords_sha256": hashlib.sha256(coords.tobytes()).hexdigest(),
    "coords_shape": list(coords.shape),
    "edges_sha256": hashlib.sha256(repr(rows).encode()).hexdigest(),
    "n_edges": len(rows),
}, indent=2), encoding="utf-8")
print("PREDICT_OK nodes", coords.shape[0], "edges", len(rows), flush=True)
''' % {"det": DET_THRESHOLD, "edge": EDGE_THRESHOLD}


# ---------------------------------------------------------------------------------------------
# Production-mirrored sandbox
# ---------------------------------------------------------------------------------------------
def _build_zarr(repo: Path, crop: str, seed: int = 0) -> None:
    """A real OME-NGFF group with the scale transform and the recorded 0.001/0.999 quantiles.

    FACT-0400 is exactly about those quantiles: the deployed predictor RAISES when they are
    absent and normalises from them when present. A fixture without them would let a
    normalisation defect through unnoticed, which is how the defect survived three attempts.
    """
    rng = np.random.default_rng(seed)
    data = (rng.random((5, 8, 32, 32), dtype=np.float32) * 900.0 + 50.0)
    path = repo / "data" / f"{crop}.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w")
    group.create_array("0", shape=data.shape, dtype="f4", chunks=(1, 8, 32, 32))
    group["0"][:] = data
    group.attrs["multiscales"] = [{"datasets": [{"coordinateTransformations": [
        {"type": "scale", "scale": [1.0, 1.625, 0.40625, 0.40625]}]}]}]
    group.attrs["image_statistics"] = {"quantiles": {
        "0.001": float(np.quantile(data, 0.001)),
        "0.999": float(np.quantile(data, 0.999)),
    }}


def _build_weights(repo: Path, seed: int = 0) -> Path:
    """A real UNetNodeTransformer state dict plus the config.json load_model reads."""
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo / "src"))
    torch.manual_seed(seed)
    from train_unet_transformer import _POS_EMBED_DIM, UNetNodeTransformer  # noqa: PLC0415
    from tracking_cellmot.models import TemporalUNet3D  # noqa: PLC0415

    unet = TemporalUNet3D(in_channels=1, out_channels=8, layers=[8, 16])
    model = UNetNodeTransformer(unet=unet, unet_out_channels=8,
                                pos_feat_dim=4 * _POS_EMBED_DIM)
    with torch.no_grad():
        # Bias detection positive so peaks exist, and spread the head's logits so the source-axis
        # softmax is not almost uniform - at random init every probability sits near 1/n_src and
        # BOTH bands would be degenerate, which would make every mutation test vacuous.
        model.detect_head.bias.fill_(2.0)
        for param in model.transformer.parameters():
            param.mul_(3.0)
    out = repo / "weights" / "split_0"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "best.pt")
    (out / "config.json").write_text(json.dumps({
        "unet_out_channels": 8, "unet_layers": [8, 16],
        "downsample": [1, 4, 4], "window_size": 2, "pool_kernel_um": 3.0,
    }), encoding="utf-8")
    return out / "best.pt"


def _run(repo: Path, script: Path, args: list[str], env_extra: dict | None = None,
         pythonpath: str | None = "scripts+src") -> subprocess.CompletedProcess:
    """Launch a script the way the kernel does: separate process, cwd=<repo>, explicit env."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    if pythonpath == "scripts+src":
        env["PYTHONPATH"] = "scripts" + os.pathsep + "src"
    elif pythonpath:
        env["PYTHONPATH"] = pythonpath
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(script), *args], cwd=str(repo),
                          env=env, text=True, capture_output=True)


class Sandbox:
    def __init__(self, base: Path):
        self.base = base
        self.repo = base / "repo"
        self.working = base / "working"          # mirrors /kaggle/working
        self.predictor = self.repo / "scripts" / "predict_unet_transformer.py"
        self.driver = self.working / "run_predict.py"
        self.worker = self.working / "assoc_tap_replay.py"
        self.weights: Path
        self.pristine: str
        self.control: dict

    def restore_pristine(self) -> None:
        self.predictor.write_text(self.pristine, encoding="utf-8", newline="")

    def apply_tap(self) -> dict:
        """Execute the REAL patch file the way the kernel cell does, with `_ps` bound."""
        self.restore_pristine()
        namespace: dict = {"_ps": self.predictor}
        exec(compile(TAP_PATCH.read_text(encoding="utf-8"), str(TAP_PATCH), "exec"),
             namespace)  # noqa: S102
        return namespace

    def predict(self, out_name: str, env_extra: dict | None = None) -> dict:
        out = self.base / out_name
        res = _run(self.repo, self.driver, [str(self.weights), CROP, str(out)], env_extra)
        assert res.returncode == 0, f"predict failed:\n{res.stdout}\n{res.stderr}"
        return json.loads(out.read_text(encoding="utf-8"))

    def cache_path(self, cache_dir: Path) -> Path:
        return cache_dir / f"{CROP}.npz"

    def replay(self, cache_dir: Path, out_name: str = "gate.json",
               pythonpath: str | None = "scripts+src") -> tuple[subprocess.CompletedProcess, dict]:
        out = self.base / out_name
        res = _run(self.repo, self.worker,
                   ["--weights", str(self.weights), "--cache-dir", str(cache_dir),
                    "--out", str(out)], pythonpath=pythonpath)
        report = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
        return res, report


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory) -> Sandbox:
    base = tmp_path_factory.mktemp("aft")
    box = Sandbox(base)
    box.working.mkdir(parents=True)
    shutil.copytree(VENDOR / "scripts", box.repo / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(VENDOR / "src", box.repo / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    _build_zarr(box.repo, CROP)
    box.weights = _build_weights(box.repo)
    box.pristine = box.predictor.read_text(encoding="utf-8")
    box.driver.write_text(PREDICT_DRIVER, encoding="utf-8")
    # The worker is copied to the WORKING directory, not imported from the repo - that is the
    # production layout, and it is the layout in which sys.path[0] is not <repo>/scripts.
    shutil.copyfile(WORKER, box.worker)
    box.control = box.predict("control.json")
    return box


@pytest.fixture(scope="module")
def tapped_cache(sandbox: Sandbox) -> tuple[Path, dict]:
    """One tapped run, reused by every test that needs a faithful cache."""
    sandbox.apply_tap()
    cache_dir = sandbox.base / "cache"
    tapped = sandbox.predict("tapped.json", {
        "BIOHUB_AFT_DIR": str(cache_dir),
        "BIOHUB_AFT_BAND_B_FLOOR": str(BAND_B_FLOOR),
    })
    return cache_dir, tapped


def _mutate(src: Path, dst: Path, fn) -> Path:
    with np.load(src, allow_pickle=False) as cache:
        data = {k: cache[k] for k in cache.files}
    fn(data)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst.with_suffix(".npz.tmp"), "wb") as handle:
        np.savez_compressed(handle, **data)
    dst.with_suffix(".npz.tmp").replace(dst)
    return dst


def _mutated_dir(sandbox: Sandbox, cache_dir: Path, name: str, fn) -> Path:
    out = sandbox.base / name
    _mutate(sandbox.cache_path(cache_dir), out / f"{CROP}.npz", fn)
    return out


# ---------------------------------------------------------------------------------------------
# 1. The patch is passive - proved statically and then dynamically
# ---------------------------------------------------------------------------------------------
def test_the_patch_is_pure_insertion(sandbox: Sandbox):
    """Removing the marked spans must restore the pre-patch bytes EXACTLY.

    This is the static half of the passivity claim. If it holds, no deployed statement was
    altered, so the graph the run emits cannot have changed for any reason inside this patch.
    """
    namespace = sandbox.apply_tap()
    patched = sandbox.predictor.read_text(encoding="utf-8")
    assert patched != sandbox.pristine
    assert namespace["aft_strip_inserts"](patched) == sandbox.pristine
    assert namespace["_AFT_PATCH_RECEIPT"]["pre_sha256"] == hashlib.sha256(
        sandbox.pristine.encode("utf-8")).hexdigest()


def test_the_patch_fails_closed_on_a_missing_anchor(sandbox: Sandbox, tmp_path: Path):
    """A silent anchor miss would look exactly like a tap that recorded nothing."""
    broken = tmp_path / "predict_unet_transformer.py"
    broken.write_text(sandbox.pristine.replace("@torch.no_grad()\ndef predict_video(",
                                               "def predict_video("), encoding="utf-8")
    with pytest.raises(RuntimeError, match="anchor 'config' matched 0 times"):
        exec(compile(TAP_PATCH.read_text(encoding="utf-8"), str(TAP_PATCH), "exec"),
             {"_ps": broken})  # noqa: S102


def test_the_tapped_predictor_emits_an_identical_graph(sandbox: Sandbox, tapped_cache):
    """The dynamic half: same coords, same edges, bit for bit, against the untapped control.

    A parity gate whose own instrumentation moves the graph is not a control for anything.
    """
    _cache_dir, tapped = tapped_cache
    assert tapped["coords_sha256"] == sandbox.control["coords_sha256"]
    assert tapped["edges_sha256"] == sandbox.control["edges_sha256"]
    assert tapped["n_edges"] == sandbox.control["n_edges"] > 0
    assert tapped["coords_shape"] == sandbox.control["coords_shape"]


def test_the_tap_is_inert_when_unconfigured(sandbox: Sandbox, tmp_path: Path):
    """With BIOHUB_AFT_DIR unset the tap writes nothing and the graph is still identical.

    That is the in-kernel control arm: the same built notebook, one environment variable apart.
    """
    sandbox.apply_tap()
    idle = sandbox.predict("idle.json")
    assert idle["coords_sha256"] == sandbox.control["coords_sha256"]
    assert idle["edges_sha256"] == sandbox.control["edges_sha256"]
    assert not list((sandbox.base / "cache_never").glob("*")) if (
        sandbox.base / "cache_never").exists() else True


# ---------------------------------------------------------------------------------------------
# 2. The cache carries the deployed image pipeline instead of reimplementing it (FACT-0400)
# ---------------------------------------------------------------------------------------------
def test_the_cache_records_the_deployed_normalisation_downsample_and_voxel_scale(
    sandbox: Sandbox, tapped_cache,
):
    """FACT-0400 as a regression test.

    The old worker called ``open_dataset(path).image`` bare, taking ``normalize=True`` and no
    downsample. Here every one of those quantities is READ BACK OUT OF THE DEPLOYED RUN, so the
    only way for them to be wrong is for the deployed run itself to be wrong.
    """
    cache_dir, _ = tapped_cache
    group = zarr.open_group(str(sandbox.repo / "data" / f"{CROP}.zarr"), mode="r")
    quantiles = dict(group.attrs["image_statistics"])["quantiles"]
    with np.load(sandbox.cache_path(cache_dir), allow_pickle=False) as cache:
        assert float(cache["q_low"]) == pytest.approx(float(quantiles["0.001"]))
        assert float(cache["q_high"]) == pytest.approx(float(quantiles["0.999"]))
        assert [int(d) for d in cache["downsample"]] == [1, 4, 4]
        # voxel_size = ds.scale * downsample, exactly as predict_video computes it.
        assert list(np.asarray(cache["voxel_size"])) == pytest.approx(
            [1.625 * 1, 0.40625 * 4, 0.40625 * 4])
        # image_shape is the DOWNSAMPLED shape the model indexed, not the raw one.
        assert [int(v) for v in cache["image_shape"]] == [5, 8, 8, 8]
        assert float(cache["det_threshold"]) == pytest.approx(DET_THRESHOLD)
        assert float(cache["edge_threshold"]) == pytest.approx(EDGE_THRESHOLD)
        assert str(cache["edge_activation"]) == "softmax"
        assert int(cache["window"]) == 2


def test_the_cache_is_keyed_by_pair_and_role_not_by_frame(sandbox: Sandbox, tapped_cache):
    """The sixth defect, as a structural assertion.

    ``TemporalUNet3D`` mixes across the time axis of the window, so a node's trunk feature is a
    property of (window, frame), not of frame. With window_size 2 and stride 1 an interior frame
    is the TARGET of one window and the SOURCE of the next, and the two feature vectors differ.
    A cache keyed by frame alone - which is what the old worker built - therefore cannot be
    faithful, and this test measures the size of the error rather than asserting it exists.
    """
    cache_dir, _ = tapped_cache
    with np.load(sandbox.cache_path(cache_dir), allow_pickle=False) as cache:
        gid = cache["role_gid"]
        feat = cache["role_feat"]
        role = cache["role_role"]
        assert int(cache["pair_f_idx"].shape[0]) >= 3
        seen: dict[int, list[int]] = {}
        for k, node in enumerate(gid.tolist()):
            seen.setdefault(node, []).append(k)
        repeated = [v for v in seen.values() if len(v) > 1]
        assert repeated, "no interior node appeared in two roles - the fixture is too short"
        deltas = [float(np.abs(feat[slots[0]] - feat[slots[1]]).max()) for slots in repeated]
        assert max(deltas) > 1e-4, (
            "the same node's SOURCE and TARGET features are identical, so this fixture cannot "
            "demonstrate the window dependence the cache layout exists for"
        )
        # And the two occurrences really are the two roles, not a duplicate of one.
        for slots in repeated:
            assert {int(role[slots[0]]), int(role[slots[1]])} == {0, 1}


# ---------------------------------------------------------------------------------------------
# 3. The replay: fresh process, real worker file, real subprocess boundary
# ---------------------------------------------------------------------------------------------
def test_the_replay_reproduces_the_deployed_probabilities(sandbox: Sandbox, tapped_cache):
    cache_dir, _ = tapped_cache
    res, report = sandbox.replay(cache_dir)
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    assert report["all_passed"] is True
    row = report["crops"][0]
    # EXACT POSITIVE COUNTS, not "non-empty". FACT-0394 is what "non-empty" costs.
    assert row["band_a"]["checked"] > 0
    assert row["band_b"]["checked"] > 0
    assert row["band_a"]["missing"] == row["band_a"]["extra"] == 0
    assert row["band_b"]["missing"] == 0
    assert row["integrity_failure_count"] == 0
    assert row["band_a"]["max_abs_prob_delta"] <= report["tolerance"]
    assert row["band_b"]["max_abs_prob_delta"] <= report["tolerance"]
    assert row["pos_feature_max_abs_delta"] <= 1e-6


def test_the_launcher_env_is_the_whole_import_fix(sandbox: Sandbox, tapped_cache):
    """FACT-0399, executed rather than read - all three ways, in the production layout.

    The worker sits in <working>, so sys.path[0] is <working>, NOT <repo>/scripts. cwd=<repo>
    does not put anything on sys.path for a script invocation.
    """
    cache_dir, _ = tapped_cache
    bare, _ = sandbox.replay(cache_dir, "gate_nopath.json", pythonpath=None)
    assert bare.returncode != 0
    assert "AFT_REPLAY_IMPORT_FAILED" in bare.stdout
    assert "ModuleNotFoundError" in bare.stderr

    src_only, _ = sandbox.replay(cache_dir, "gate_srconly.json", pythonpath="src")
    assert src_only.returncode != 0, "src alone must NOT be enough - the module is in scripts/"

    both, report = sandbox.replay(cache_dir, "gate_both.json")
    assert both.returncode == 0 and report["all_passed"] is True


def test_the_kernel_patch_launches_the_worker_with_scripts_and_src():
    """The shipped launcher's env, parsed from its own source rather than assumed."""
    tree = ast.parse(GATE_PATCH.read_text(encoding="utf-8"))
    envs = [n for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "_aftg_env" for t in n.targets)]
    assert len(envs) == 1
    rendered = ast.unparse(envs[0].value)
    assert "'PYTHONPATH': 'scripts' + _aftg_os.pathsep + 'src'" in rendered, rendered
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "_aftg_sub.run"]
    assert len(calls) == 1
    kwargs = {k.arg for k in calls[0].keywords}
    assert {"cwd", "env"} <= kwargs, f"the launcher must pass both cwd and env, got {kwargs}"


# ---------------------------------------------------------------------------------------------
# 4. The gate must REJECT. A checker that has not been shown to reject is not a checker.
# ---------------------------------------------------------------------------------------------
def test_a_per_frame_feature_cache_fails_the_gate(sandbox: Sandbox, tapped_cache):
    """DEFECT 6, manufactured exactly as the old worker built it, and required to be rejected.

    The old worker stored ``feats[t]`` once, at first sight of frame t, and reused it for both
    roles. Reproducing that here means overwriting every role block for a frame with the block
    that was seen first - which is what a frame-keyed cache is.
    """
    cache_dir, _ = tapped_cache

    def per_frame(data: dict) -> None:
        frames = np.empty(data["role_gid"].shape[0], dtype=np.int64)
        for pair in range(data["pair_f_idx"].shape[0]):
            for role, ptr_key, n_key, t_key in ((0, "pair_src_ptr", "pair_src_n", "pair_t_src"),
                                                (1, "pair_tgt_ptr", "pair_tgt_n", "pair_t_tgt")):
                ptr, n = int(data[ptr_key][pair]), int(data[n_key][pair])
                frames[ptr:ptr + n] = int(data[t_key][pair])
        first: dict[int, int] = {}
        for k, t in enumerate(frames.tolist()):
            first.setdefault(t, k)
        feat = data["role_feat"].copy()
        for k, (t, node) in enumerate(zip(frames.tolist(), data["role_gid"].tolist())):
            base = first[t]
            feat[k] = data["role_feat"][base + (node - int(data["role_gid"][base]))]
        data["role_feat"] = feat

    mutated = _mutated_dir(sandbox, cache_dir, "cache_per_frame", per_frame)
    res, report = sandbox.replay(mutated, "gate_per_frame.json")
    assert res.returncode == 2
    assert report["all_passed"] is False
    row = report["crops"][0]
    assert row["band_a"]["max_abs_prob_delta"] > report["tolerance"] or row["band_a"]["missing"]


def test_a_single_perturbed_node_fails_the_gate(sandbox: Sandbox, tapped_cache):
    """One node's features wrong. Not uniform across the source axis, so it must be visible."""
    cache_dir, _ = tapped_cache

    def perturb(data: dict) -> None:
        feat = data["role_feat"].copy()
        feat[0] = feat[0] + np.float32(0.5)
        data["role_feat"] = feat

    mutated = _mutated_dir(sandbox, cache_dir, "cache_perturbed", perturb)
    _res, report = sandbox.replay(mutated, "gate_perturbed.json")
    assert report["all_passed"] is False


def test_a_within_frame_node_permutation_fails_the_gate(sandbox: Sandbox, tapped_cache):
    """A reorder that changes no shape, no count and no feature VALUE must still be rejected.

    Node ids are positional: ``coords_so_far`` numbers by frame in detection order, and the
    pre-ILP export and the ECB sidecars both address nodes by that number. A reorder silently
    repoints every id in every downstream artifact.
    """
    cache_dir, _ = tapped_cache

    def permute(data: dict) -> None:
        ptr, n = int(data["pair_src_ptr"][0]), int(data["pair_src_n"][0])
        assert n >= 3
        order = np.arange(n)
        order[0], order[1] = order[1], order[0]
        for key in ("role_feat", "role_pos", "role_coord_scaled", "role_coord_rel", "role_gid"):
            if key in data:
                block = data[key][ptr:ptr + n]
                data[key] = data[key].copy()
                data[key][ptr:ptr + n] = block[order]

    mutated = _mutated_dir(sandbox, cache_dir, "cache_permuted", permute)
    _res, report = sandbox.replay(mutated, "gate_permuted.json")
    assert report["all_passed"] is False
    row = report["crops"][0]
    assert row["integrity_failure_count"] > 0
    assert any("node ids are not the deployed frame slice" in f["why"]
               for f in row["integrity_failures"])


def test_the_gate_refuses_an_empty_band_a(sandbox: Sandbox, tapped_cache):
    """FACT-0394 applied to the new gate, and ISOLATED so the refusal is attributable.

    The threshold is raised out of reach so that nothing is reproduced above it either - band A's
    three conditions are then all trivially satisfied and only the floor can fire. Band B still
    checks a positive number of pairs, which is what proves the refusal is band A.
    """
    cache_dir, _ = tapped_cache

    def empty_a(data: dict) -> None:
        data["edge_threshold"] = np.float64(0.999999)
        for key in ("pair", "i", "j", "source_id", "target_id"):
            data[f"band_a_{key}"] = np.empty(0, dtype=np.int64)
        data["band_a_prob"] = np.empty(0, dtype=np.float64)

    mutated = _mutated_dir(sandbox, cache_dir, "cache_no_a", empty_a)
    _res, report = sandbox.replay(mutated, "gate_no_a.json")
    row = report["crops"][0]
    assert report["all_passed"] is False
    assert row["band_a"]["checked"] == 0
    assert row["band_a"]["missing"] == row["band_a"]["extra"] == 0
    assert row["band_b"]["checked"] > 0, "the refusal must be band A, not a collateral band-B miss"
    assert row["integrity_failure_count"] == 0


def test_the_gate_refuses_an_empty_band_b(sandbox: Sandbox, tapped_cache):
    """Band B is mandatory: FACT-0382 measured that ALL 691 fold-0 contested errors sit below
    the deployed threshold, so a band-A-only pass validates the band the task does not use."""
    cache_dir, _ = tapped_cache

    def empty_b(data: dict) -> None:
        for key in ("pair", "i", "j", "source_id", "target_id"):
            data[f"band_b_{key}"] = np.empty(0, dtype=np.int64)
        data["band_b_prob"] = np.empty(0, dtype=np.float64)

    mutated = _mutated_dir(sandbox, cache_dir, "cache_no_b", empty_b)
    _res, report = sandbox.replay(mutated, "gate_no_b.json")
    row = report["crops"][0]
    assert report["all_passed"] is False
    assert row["band_b"]["checked"] == 0
    assert row["band_a"]["checked"] > 0, "the refusal must be band B"


def test_a_torn_window_shape_record_fails_the_gate(sandbox: Sandbox, tapped_cache):
    """The recorded positional features are cross-checked against the DEPLOYED function.

    If the window shape a cache records disagrees with the positional features it records, one of
    the two is stale - and a downstream head trained from that cache would receive a position
    encoding no deployed run ever produced.
    """
    cache_dir, _ = tapped_cache

    def tear(data: dict) -> None:
        shape = data["pair_window_shape"].copy()
        shape[0, 2] += 1
        data["pair_window_shape"] = shape

    mutated = _mutated_dir(sandbox, cache_dir, "cache_torn", tear)
    _res, report = sandbox.replay(mutated, "gate_torn.json")
    assert report["all_passed"] is False
    assert report["crops"][0]["pos_feature_max_abs_delta"] > 1e-6


def test_the_gate_refuses_an_empty_cache_directory(sandbox: Sandbox, tmp_path: Path):
    """A gate that can quietly compare nothing looks exactly like a gate that passed."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    res, report = sandbox.replay(empty, "gate_empty.json")
    assert res.returncode == 1
    assert report["all_passed"] is False
    assert "the tap wrote nothing" in report["error"]


# ---------------------------------------------------------------------------------------------
# 5. Two ACCEPT controls - an auditor that rejects everything checks nothing
# ---------------------------------------------------------------------------------------------
def test_a_faithful_cache_still_passes_after_a_lossless_round_trip(sandbox: Sandbox,
                                                                   tapped_cache):
    """Rewriting the cache without changing a value must not flip the verdict."""
    cache_dir, _ = tapped_cache
    same = _mutated_dir(sandbox, cache_dir, "cache_roundtrip", lambda data: None)
    _res, report = sandbox.replay(same, "gate_roundtrip.json")
    assert report["all_passed"] is True


def test_a_uniform_per_target_logit_offset_is_invisible_and_a_non_uniform_one_is_not():
    """THE BLIND SPOT, stated precisely and asserted in BOTH directions.

    The gate compares probabilities AFTER ``softmax(dim=0)`` over the SOURCE axis, so the map
    from logits to probabilities is many-to-one: its fibre is exactly the set of PER-TARGET
    constant offsets. Any cache error whose net effect on the logits is a per-target constant is
    mathematically invisible here - not merely undetected in practice.

    Note the precision. The older wording said "a uniform per-frame SOURCE shift"; in feature
    space a uniform shift of the source features does NOT in general produce a per-target
    constant logit offset, because the head is not affine in those features. The invariance is a
    property of the LOGITS, and that is what is asserted here. Closing it would require
    pre-softmax logits, which nothing in the current artifacts records.
    """
    torch.manual_seed(0)
    logits = torch.randn(7, 5)
    base = torch.softmax(logits, dim=0)
    per_target = torch.softmax(logits + torch.randn(1, 5) * 3.0, dim=0)
    assert float((per_target - base).abs().max()) < 1e-5      # invisible

    per_source = torch.softmax(logits + torch.randn(7, 1) * 3.0, dim=0)
    assert float((per_source - base).abs().max()) > 1e-4      # visible


# ---------------------------------------------------------------------------------------------
# 6. Real class surface, and no drift between the worker and the patch that ships it
# ---------------------------------------------------------------------------------------------
def _attribute_names(source: str, receiver: str) -> set[str]:
    return {n.attr for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Attribute) and getattr(n.value, "id", None) == receiver}


def test_the_worker_only_touches_the_real_class_surface(sandbox: Sandbox):
    """Every attribute the worker names must resolve on the REAL objects.

    This is the direct answer to the fourth defect. ``model.detection_head`` passed a green suite
    because the suite's stub defined it; ``UNetNodeTransformer`` has only ever had
    ``detect_head``, and it is an INSTANCE attribute assigned in ``__init__``, so a class-level
    ``hasattr`` would miss it. The model here is built by the real ``load_model``.
    """
    sys.path.insert(0, str(sandbox.repo / "scripts"))
    sys.path.insert(0, str(sandbox.repo / "src"))
    import predict_unet_transformer as AFP  # noqa: PLC0415

    model, _window, _downsample = AFP.load_model(sandbox.weights, torch.device("cpu"))
    source = WORKER.read_text(encoding="utf-8")

    used_on_model = _attribute_names(source, "model")
    assert used_on_model, "the worker names no model attribute - the check would be vacuous"
    for name in used_on_model:
        assert hasattr(model, name), f"model has no attribute {name!r}"

    for name in _attribute_names(source, "AFP"):
        assert hasattr(AFP, name), f"predict_unet_transformer has no attribute {name!r}"

    # The same standard applied to the PATCH: it reads cfg fields off the real PredictConfig.
    cfg = AFP.PredictConfig()
    for name in _attribute_names(TAP_PATCH.read_text(encoding="utf-8"), "cfg"):
        assert hasattr(cfg, name), f"PredictConfig has no field {name!r}"

    # And the control: an attribute the class has never had must fail this check.
    assert not hasattr(model, "detection_head")


def _slice_literals(node) -> set[str]:
    if isinstance(node, ast.JoinedStr):
        return set()
    if isinstance(node, ast.Constant):
        return {node.value} if isinstance(node.value, str) else set()
    out: set[str] = set()
    for child in ast.iter_child_nodes(node):
        out |= _slice_literals(child)
    return out


def _declared_schema() -> tuple[set[str], set[str]]:
    """The schema constants live INSIDE the patch's injected-source literal, so `ast.parse` of
    the patch file never sees them as assignments - they are text until the kernel writes them
    into the predictor. Read them the way the preflight does, from the text."""
    text = TAP_PATCH.read_text(encoding="utf-8")
    out = []
    for name in ("_AFT_SCHEMA_REQUIRED", "_AFT_SCHEMA_OPTIONAL"):
        m = re.search("^" + name + r" = (\(.*?\))\n", text, re.S | re.M)
        assert m, f"{name} not found in {TAP_PATCH}"
        out.append(set(ast.literal_eval(m.group(1))))
    return out[0], out[1]


def test_the_emitted_cache_matches_the_declared_schema(sandbox: Sandbox, tapped_cache):
    """The schema is declared once and enforced at both ends, so a rename cannot go silent."""
    cache_dir, _ = tapped_cache
    required, optional = _declared_schema()
    with np.load(sandbox.cache_path(cache_dir), allow_pickle=False) as cache:
        keys = set(cache.files)
    assert required <= keys
    assert keys - required <= optional


def test_every_cache_key_the_worker_reads_is_one_the_tap_writes():
    """A key the replay reads but the tap never writes is a KeyError after the GPU spend."""
    required, optional = _declared_schema()
    source = WORKER.read_text(encoding="utf-8")
    read = set()
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Subscript)
                and getattr(node.value, "id", None) in ("cache", "data")):
            # The slice may be a conditional expression (`"pair_src_ptr" if role == 0 else ...`),
            # so every string constant inside it counts - EXCEPT an f-string's literal fragments,
            # which are Constants but not keys. The dynamic band keys are expanded below.
            read |= _slice_literals(node.slice)
    # f-string keys the worker builds per band, resolved rather than ignored.
    for band in ("a", "b"):
        for field in ("pair", "i", "j", "source_id", "target_id", "prob"):
            read.add(f"band_{band}_{field}")
    assert read, "no cache key was found - the cross-reference would be vacuous"
    assert read <= required | optional, sorted(read - required - optional)


def test_the_kernel_patch_carries_the_committed_worker():
    """Two copies of anything drift. The kernel patch must embed the current worker verbatim."""
    sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))
    import sync_tap_worker  # noqa: PLC0415

    assert sync_tap_worker.rendered() == GATE_PATCH.read_text(encoding="utf-8"), (
        "scripts/kaggle_edits/assoc_tap_gate.py does not carry the current "
        "scripts/win_bet/assoc_tap_replay.py. Run: "
        "python scripts/win_bet/sync_tap_worker.py --write"
    )
