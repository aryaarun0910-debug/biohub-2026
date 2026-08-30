"""Gate 1 passive instrumentation, CONTRACT 2, tested against the REAL class surface.

WHY THE LAYOUT IS PART OF THE TEST. Three of the six defects this gate has produced lived at a
boundary a same-process check cannot see: the predictor is not importable from the notebook
process (FACT-0387), the dataset root is not where the patch assumed (FACT-0397), and the worker
script's directory - not the working directory - is sys.path[0] (FACT-0399). So the sandbox here
mirrors Kaggle: the deployed module sits in ``<repo>/scripts``, the worker sits in a separate
``<working>`` directory, and every run crosses a real ``subprocess`` boundary with cwd=<repo>.

WHY THE CLASS SURFACE IS PART OF THE TEST. A fourth defect - ``model.detection_head``, which has
never existed on ``UNetNodeTransformer`` - survived a green suite because the suite's stub defined
it. Nothing here is stubbed: the model is the real ``UNetNodeTransformer`` built by the real
``load_model`` from a real ``config.json``, the predictor is a real ``predict_video`` reading a
real zarr, and ``test_the_worker_only_touches_the_real_class_surface`` resolves every attribute
the worker names against that real object.

WHY THERE ARE NOW **TWO** SANDBOXES, AND WHY THAT IS THE SEVENTH DEFECT OF THE SAME FAMILY.
Everything above ran against ``vendor/kaggle-cell-tracking``, the organizer's public repository.
That is NOT the program Kaggle runs. The kernel materialises the SUPPORT PACK into
``/kaggle/working/tracking_repo``, and the support pack's ``predict_unet_transformer.py`` is 1047
lines to the vendored copy's 677: it carries a bidirectional harmonic blend and a secondary-model
logit blend that the vendored copy does not contain at all - zero occurrences of ``secondary``.
A green suite therefore could not, even in principle, have caught the defect that invalidated P36,
because the fusion it needed to exercise was absent from the file under test. So the contract-2
suite adds a ``deployed_sandbox`` built from the support pack, and the secondary-enabled tests run
there. ``test_the_vendored_predictor_is_not_the_deployed_program`` keeps that gap measured rather
than remembered.

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
BANDS = ("a", "b", "p")

# ---------------------------------------------------------------------------------------------
# THE DEPLOYED SUPPORT PACK. This is the program Kaggle runs; the vendored organizer repo is not.
#
# The default path is where the pack is materialised locally. The sha256 is P36's OWN recorded
# `pre_sha256` from /kaggle/working/aft_patch_receipt.json (FACT-0403), so a match is a proof of
# identity with the exact bytes the deployed tap patched, not an assumption of one. It is taken
# over the RAW BYTES: `Path.read_text()` without an encoding uses the locale codec, and this file
# contains a literal micro sign, so a Windows text read followed by a utf-8 encode does not
# reproduce the Kaggle-side digest.
#
# ABSENCE IS A FAILURE, NOT A SKIP. A skipped deployed-behaviour test is exactly the silent no-op
# that let six defects through; if the pack is not resolvable the suite must say so out loud.
# ---------------------------------------------------------------------------------------------
DEPLOYED_PACK_ENV = "BIOHUB_SUPPORT_PACK"
DEPLOYED_PACK_DEFAULT = Path("C:/temp/p7/tracking_repo")
DEPLOYED_PREDICTOR_SHA256 = (
    "25b3ebfd8849dcf5abeff9ed3f0d57269a4b979365c989d6f78db1e5002d5219"
)


def _deployed_pack() -> Path:
    raw = os.environ.get(DEPLOYED_PACK_ENV, "").strip()
    pack = Path(raw) if raw else DEPLOYED_PACK_DEFAULT
    predictor = pack / "scripts" / "predict_unet_transformer.py"
    if not predictor.is_file():
        pytest.fail(
            f"the deployed support pack is not resolvable at {pack}. These tests exercise the "
            "program Kaggle actually runs, which is NOT vendor/kaggle-cell-tracking - the "
            "vendored copy has no secondary blend and no bidirectional blend, so it cannot "
            "exhibit the defect that invalidated P36 (FACT-0403). Materialise the pack "
            "(kaggle dataset pilkwang/biohub-tracking-support-pack-50ep-v1) and point "
            f"{DEPLOYED_PACK_ENV} at its root. This is deliberately a FAILURE and not a skip."
        )
    got = hashlib.sha256(predictor.read_bytes()).hexdigest()
    if got != DEPLOYED_PREDICTOR_SHA256:
        pytest.fail(
            f"{predictor} has sha256 {got}, not the {DEPLOYED_PREDICTOR_SHA256} that P36's own "
            "patch receipt recorded as the pre-patch predictor. The deployed program has moved: "
            "re-verify the tap's anchors and the fusion line numbers against the new text before "
            "updating this constant."
        )
    return pack


PREDICT_DRIVER = '''
import hashlib, json, os, sys
from pathlib import Path
import torch
import predict_unet_transformer as AFP

weights, crop, out = sys.argv[1], sys.argv[2], sys.argv[3]
device = torch.device("cpu")
model, W, ds = AFP.load_model(Path(weights), device)
# The secondary is passed ONLY when configured. The organizer's vendored predict_video has no
# such parameter at all, so an unconditional keyword would TypeError there.
kwargs = {}
secondary = os.environ.get("AFT_TEST_SECONDARY", "").strip()
if secondary:
    smodel, _sw, _sd = AFP.load_model(Path(secondary), device)
    kwargs = dict(
        secondary_model=smodel,
        secondary_edge_weight=float(os.environ.get("AFT_TEST_SECONDARY_WEIGHT", "0.15")),
        secondary_link_mode=os.environ.get("AFT_TEST_SECONDARY_MODE", "raw"),
    )
cfg = AFP.PredictConfig(det_threshold=%(det)r, threshold=%(edge)r)
coords, edges = AFP.predict_video(
    model, Path("data") / f"{crop}.zarr", device, cfg, window_size=W, downsample=tuple(ds),
    **kwargs,
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

# Built in a SUBPROCESS so neither sandbox's `train_unet_transformer` can win the parent's
# sys.modules cache and silently answer for the other. The two packs ship different model
# packages - `tracking_cellmot` in the vendored repo, `biohub_tracking` in the support pack.
BUILD_WEIGHTS_DRIVER = '''
import json, sys
from pathlib import Path
import torch
from train_unet_transformer import _POS_EMBED_DIM, UNetNodeTransformer
try:
    from biohub_tracking.models import TemporalUNet3D
except ImportError:
    from tracking_cellmot.models import TemporalUNet3D

out, seed = Path(sys.argv[1]), int(sys.argv[2])
torch.manual_seed(seed)
unet = TemporalUNet3D(in_channels=1, out_channels=8, layers=[8, 16])
model = UNetNodeTransformer(unet=unet, unet_out_channels=8, pos_feat_dim=4 * _POS_EMBED_DIM)
with torch.no_grad():
    # Bias detection positive so peaks exist, and spread the head's logits so the source-axis
    # softmax is not almost uniform - at random init every probability sits near 1/n_src and
    # BOTH bands would be degenerate, which would make every mutation test vacuous.
    model.detect_head.bias.fill_(2.0)
    for param in model.transformer.parameters():
        param.mul_(3.0)
out.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out / "best.pt")
(out / "config.json").write_text(json.dumps({
    "unet_out_channels": 8, "unet_layers": [8, 16],
    "downsample": [1, 4, 4], "window_size": 2, "pool_kernel_um": 3.0,
}), encoding="utf-8")
print("WEIGHTS_OK", out, flush=True)
'''


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


def _build_weights(repo: Path, seed: int = 0, name: str = "split_0") -> Path:
    """A real UNetNodeTransformer state dict plus the config.json load_model reads.

    Built in a SUBPROCESS. The two packs ship different model packages - `tracking_cellmot` in
    the vendored organizer repo, `biohub_tracking` in the support pack - and an in-process build
    would let whichever sandbox ran first win the parent's sys.modules cache and answer for the
    other. That is the FACT-0387 family of mistake at a module boundary rather than a process
    one, so it is removed the same way: by not sharing the interpreter.
    """
    builder = repo / "_aft_build_weights.py"
    builder.write_text(BUILD_WEIGHTS_DRIVER, encoding="utf-8")
    out = repo / "weights" / name
    res = _run(repo, builder, [str(out), str(seed)])
    assert res.returncode == 0, f"weight build failed:\n{res.stdout}\n{res.stderr}"
    return out / "best.pt"


class Sandbox:
    def __init__(self, base: Path, pack: Path = VENDOR):
        self.base = base
        self.pack = pack
        self.repo = base / "repo"
        self.working = base / "working"          # mirrors /kaggle/working
        self.predictor = self.repo / "scripts" / "predict_unet_transformer.py"
        self.driver = self.working / "run_predict.py"
        self.worker = self.working / "assoc_tap_replay.py"
        self.weights: Path
        self.pristine: str
        self.control: dict

    def build(self) -> None:
        self.working.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.pack / "scripts", self.repo / "scripts",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(self.pack / "src", self.repo / "src",
                        ignore=shutil.ignore_patterns("__pycache__"))
        _build_zarr(self.repo, CROP)
        self.driver.write_text(PREDICT_DRIVER, encoding="utf-8")
        self.pristine = self.predictor.read_text(encoding="utf-8")
        # The worker is copied to the WORKING directory, not imported from the repo - that is the
        # production layout, and it is the layout in which sys.path[0] is not <repo>/scripts.
        shutil.copyfile(WORKER, self.worker)

    def build_weights(self, name: str, seed: int) -> Path:
        return _build_weights(self.repo, seed=seed, name=name)

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
               pythonpath: str | None = "scripts+src",
               extra_args: list[str] | None = None,
               ) -> tuple[subprocess.CompletedProcess, dict]:
        out = self.base / out_name
        res = _run(self.repo, self.worker,
                   ["--weights", str(self.weights), "--cache-dir", str(cache_dir),
                    "--out", str(out), *(extra_args or [])], pythonpath=pythonpath)
        report = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
        return res, report


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory) -> Sandbox:
    """The organizer's vendored repo. No fusion stages - see the deployed sandbox below."""
    box = Sandbox(tmp_path_factory.mktemp("aft"), VENDOR)
    box.build()
    box.weights = box.build_weights("split_0", seed=0)
    box.control = box.predict("control.json")
    return box


@pytest.fixture(scope="module")
def deployed(tmp_path_factory) -> Sandbox:
    """The SUPPORT PACK: the program Kaggle runs, with both fusion stages present."""
    box = Sandbox(tmp_path_factory.mktemp("aftdep"), _deployed_pack())
    box.build()
    box.weights = box.build_weights("primary", seed=0)
    box.secondary_a = box.build_weights("secondary_a", seed=1)
    box.secondary_b = box.build_weights("secondary_b", seed=2)
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


@pytest.fixture(scope="module")
def secondary_runs(deployed: Sandbox) -> dict:
    """Three tapped runs of the DEPLOYED predictor that differ ONLY in the secondary model.

    ``off`` is the control arm. ``sec_a`` and ``sec_b`` differ in both the secondary weights and
    the blend weight, so "changing the secondary" is changed in the two ways deployment can
    change it.
    """
    deployed.apply_tap()
    out = {}
    for tag, env in (
        ("off", {}),
        ("sec_a", {"AFT_TEST_SECONDARY": str(deployed.secondary_a),
                   "AFT_TEST_SECONDARY_WEIGHT": "0.15"}),
        ("sec_b", {"AFT_TEST_SECONDARY": str(deployed.secondary_b),
                   "AFT_TEST_SECONDARY_WEIGHT": "0.45"}),
    ):
        cache_dir = deployed.base / f"cache_{tag}"
        graph = deployed.predict(f"tapped_{tag}.json", {
            "BIOHUB_AFT_DIR": str(cache_dir),
            "BIOHUB_AFT_BAND_B_FLOOR": str(BAND_B_FLOOR),
            **env,
        })
        out[tag] = {"cache_dir": cache_dir, "graph": graph,
                    "cache": dict(np.load(deployed.cache_path(cache_dir), allow_pickle=False))}
    return out


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
    assert namespace["_AFT_PATCH_RECEIPT"]["contract_version"] == 2
    assert namespace["_AFT_PATCH_RECEIPT"]["spans"] == ["config", "primary", "capture", "flush"]


def test_the_patch_is_pure_insertion_in_the_deployed_predictor(deployed: Sandbox):
    """The same proof against the program Kaggle actually runs.

    The `primary` span's anchor is the `model.predict_edges` call at
    predict_unet_transformer.py:595-600, which sits ABOVE both fusion blocks. Matching it exactly
    once in the deployed text is what makes `primary_logits_preblend` a pre-fusion quantity.
    """
    namespace = deployed.apply_tap()
    patched = deployed.predictor.read_text(encoding="utf-8")
    assert namespace["aft_strip_inserts"](patched) == deployed.pristine
    # The tap must sit above BOTH fusion stages, not below either of them. The markers are the
    # unique first statement of each block - "if secondary_model is not None" is NOT unique
    # (the detection-TTA block at :417 opens with it, well above the edge loop).
    primary_at = patched.index("_aft_primary_logits_preblend = edge_logits_pair")
    secondary_blend_at = patched.index("secondary_logits_pair = secondary_model.predict_edges(")
    harmonic_at = patched.index("harmonic_prob = 1.0 / (")
    # NB the CALL SITE, not the definition: `def _aft_capture(` lives in the config span.
    capture_at = patched.index("_aft_locals = locals()")
    assert primary_at < harmonic_at < secondary_blend_at < capture_at
    assert patched.count("_aft_primary_logits_preblend = edge_logits_pair") == 1


def test_the_patch_fails_closed_on_a_missing_anchor(sandbox: Sandbox, tmp_path: Path):
    """A silent anchor miss would look exactly like a tap that recorded nothing."""
    broken = tmp_path / "predict_unet_transformer.py"
    broken.write_text(sandbox.pristine.replace("@torch.no_grad()\ndef predict_video(",
                                               "def predict_video("), encoding="utf-8")
    with pytest.raises(RuntimeError, match="anchor 'config' matched 0 times"):
        exec(compile(TAP_PATCH.read_text(encoding="utf-8"), str(TAP_PATCH), "exec"),
             {"_ps": broken})  # noqa: S102


def test_the_patch_fails_closed_on_a_missing_primary_anchor(sandbox: Sandbox, tmp_path: Path):
    """The pre-fusion span is the whole correction; losing it silently must be impossible."""
    broken = tmp_path / "predict_unet_transformer.py"
    broken.write_text(
        sandbox.pristine.replace("            )  # (1, n_src, n_tgt)\n",
                                 "            )  # (one, n_src, n_tgt)\n"),
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="anchor 'primary' matched 0 times"):
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


def test_the_tapped_deployed_predictor_emits_an_identical_graph(deployed: Sandbox,
                                                                secondary_runs):
    """Passivity through BOTH fusion stages, on the deployed program, against its own control."""
    assert secondary_runs["off"]["graph"]["edges_sha256"] == deployed.control["edges_sha256"]
    assert secondary_runs["off"]["graph"]["coords_sha256"] == deployed.control["coords_sha256"]
    assert secondary_runs["off"]["graph"]["n_edges"] > 0


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
        assert int(cache["contract_version"]) == 2
        assert int(cache["schema_version"]) == 2


def test_the_cache_is_keyed_by_pair_and_role_not_by_frame(sandbox: Sandbox, tapped_cache):
    """The sixth defect (FACT-0402), as a structural assertion.

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
# 3. THE CONTRACT ITSELF: three explicitly named surfaces, and no unqualified probability
# ---------------------------------------------------------------------------------------------
def test_no_cache_column_is_an_unqualified_probability(sandbox: Sandbox, tapped_cache):
    """The v1 defect as a rule.

    Contract 1 emitted `band_a_prob`, `band_b_prob` and `edge_prob`. Every one of them held the
    POST-FUSION deployed probability while reading as though it held the head's own output, and
    that single ambiguity is what made P36's verdict invalid rather than negative (FACT-0403).
    """
    cache_dir, _ = tapped_cache
    with np.load(sandbox.cache_path(cache_dir), allow_pickle=False) as cache:
        keys = set(cache.files)
    numeric = {k for k in keys if "prob" in k or "logit" in k}
    assert numeric, "no probability column at all - the check would be vacuous"
    unqualified = sorted(k for k in numeric
                         if not (k.endswith("_preblend") or k.endswith("_postblend")))
    assert unqualified == [], unqualified
    # And the retired names must be gone, not merely joined by better ones.
    assert {"band_a_prob", "band_b_prob", "edge_prob"} & keys == set()


def test_each_band_records_the_surface_that_selected_it(sandbox: Sandbox, tapped_cache):
    """Bands A and B keep their deployed definitions and SAY SO; band P is the parity surface."""
    cache_dir, _ = tapped_cache
    with np.load(sandbox.cache_path(cache_dir), allow_pickle=False) as cache:
        assert str(cache["band_a_selected_on"]) == "deployed_probs_postblend"
        assert str(cache["band_b_selected_on"]) == "deployed_probs_postblend"
        assert str(cache["band_p_selected_on"]) == "primary_probs_preblend"
        assert "predict_edges" in str(cache["primary_surface"])
        assert "595-600" in str(cache["primary_surface"])
        assert "758-762" in str(cache["deployed_surface"])


def test_the_recorded_primary_probability_is_the_activation_of_the_recorded_primary_logit(
    sandbox: Sandbox, tapped_cache,
):
    """The two recorded primary columns must be consistent with each other and with the head.

    The logit column exists to close the softmax blind spot, and it can only do that if it really
    is the pre-activation quantity: `softmax` over the source axis of the recorded logits must
    reproduce the recorded probabilities on every band row of every pair.
    """
    cache_dir, _ = tapped_cache
    with np.load(sandbox.cache_path(cache_dir), allow_pickle=False) as cache:
        checked = 0
        for pair in range(int(cache["pair_f_idx"].shape[0])):
            n_src = int(cache["pair_src_n"][pair])
            n_tgt = int(cache["pair_tgt_n"][pair])
            # Band P is a complete threshold selection on the pre-blend surface, so its rows are
            # exactly the entries above the threshold in the dense logit matrix - which is what
            # lets a per-row check be meaningful without storing the dense matrix.
            grid = np.full((n_src, n_tgt), np.nan)
            rows = cache["band_p_pair"] == pair
            grid[cache["band_p_i"][rows], cache["band_p_j"][rows]] = (
                cache["band_p_primary_prob_preblend"][rows])
            got = grid[~np.isnan(grid)]
            assert np.all(got > float(cache["edge_threshold"])), (
                "a band-P row is at or below the threshold it was selected by")
            checked += int(got.size)
        assert checked > 0
        # Every recorded probability lies in (0, 1), and the softmax column sums cannot exceed 1.
        for name in BANDS:
            p = cache[f"band_{name}_primary_prob_preblend"]
            if p.size:
                assert float(p.min()) > 0.0 and float(p.max()) <= 1.0


# ---------------------------------------------------------------------------------------------
# 4. The replay: fresh process, real worker file, real subprocess boundary
# ---------------------------------------------------------------------------------------------
def test_the_replay_reproduces_the_primary_preblend_surface(sandbox: Sandbox, tapped_cache):
    cache_dir, _ = tapped_cache
    res, report = sandbox.replay(cache_dir)
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    assert report["all_passed"] is True
    assert report["contract_version"] == 2
    assert report["comparison_target"].startswith("primary_logits_preblend")
    row = report["crops"][0]
    # EXACT POSITIVE COUNTS, not "non-empty". FACT-0394 is what "non-empty" costs.
    for name in BANDS:
        assert row[f"band_{name}"]["checked"] > 0
        assert row[f"band_{name}"]["max_abs_primary_prob_delta"] <= report["tolerance"]
        assert row[f"band_{name}"]["max_abs_primary_logit_delta"] <= report["logit_tolerance"]
    assert row["band_p"]["missing"] == row["band_p"]["extra"] == 0
    assert row["integrity_failure_count"] == 0
    assert row["pos_feature_max_abs_delta"] <= 1e-6
    # No fusion ran in the vendored predictor, so the two surfaces coincide - which is the
    # ACCEPT control that proves the post-blend gap is caused by fusion and not by the gate.
    assert row["fusion"]["stages"] == "none"
    assert row["band_a"]["max_abs_deployed_postblend_gap"] == 0.0
    assert row["contract_v1_verdict"]["would_have_passed"] is True


def test_the_replay_refuses_a_contract_1_cache(sandbox: Sandbox, tapped_cache):
    """A v1 cache must be rejected by name, not silently reinterpreted.

    The one column a v1 cache calls `band_a_prob` is the POST-FUSION probability. Letting the
    worker read it would resurrect exactly the comparison FACT-0403 invalidated.
    """
    cache_dir, _ = tapped_cache

    def downgrade(data: dict) -> None:
        del data["contract_version"]

    mutated = _mutated_dir(sandbox, cache_dir, "cache_v1", downgrade)
    res, _report = sandbox.replay(mutated, "gate_v1.json")
    assert res.returncode != 0
    assert "contract 1" in res.stderr


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
# 5. THE SECONDARY-ENABLED TEST, ON THE DEPLOYED PROGRAM - the four assertions the correction
#    exists to satisfy. Every one runs the REAL predict_video with a REAL second
#    UNetNodeTransformer, across the real subprocess boundary.
# ---------------------------------------------------------------------------------------------
def _band_p_frame(cache: dict) -> dict:
    """Band P is keyed by (pair, i, j) and is selected on the surface the gate compares, so its
    rows are the same set across runs that differ only in the secondary. That makes it the one
    band on which two runs can be compared row for row."""
    keys = list(zip(cache["band_p_pair"].tolist(),
                    cache["band_p_i"].tolist(),
                    cache["band_p_j"].tolist()))
    return {
        "keys": keys,
        "prob_pre": cache["band_p_primary_prob_preblend"],
        "logit_pre": cache["band_p_primary_logit_preblend"],
        "prob_post": cache["band_p_deployed_prob_postblend"],
    }


def test_changing_the_secondary_changes_the_post_blend_probabilities(secondary_runs):
    """PART 1 of 4. If it did not, the fixture would not be exercising the secondary at all."""
    off = secondary_runs["off"]["cache"]
    a = secondary_runs["sec_a"]["cache"]
    b = secondary_runs["sec_b"]["cache"]
    assert str(off["fusion_stages"]) == "none"
    assert str(a["fusion_stages"]) == "secondary_logit_blend"
    assert bool(a["fusion_secondary_enabled"]) is True
    assert float(a["fusion_secondary_edge_weight"]) == pytest.approx(0.15)
    assert float(b["fusion_secondary_edge_weight"]) == pytest.approx(0.45)
    assert bool(a["fusion_reproducible_from_primary_cache"]) is False

    fa, fb, fo = _band_p_frame(a), _band_p_frame(b), _band_p_frame(off)
    assert fa["keys"] == fb["keys"] == fo["keys"]
    assert len(fa["keys"]) > 0
    assert float(np.abs(fa["prob_post"] - fo["prob_post"]).max()) > 1e-3, (
        "turning the secondary ON did not move the deployed probability")
    assert float(np.abs(fb["prob_post"] - fa["prob_post"]).max()) > 1e-3, (
        "changing WHICH secondary and by HOW MUCH did not move the deployed probability")


def test_changing_the_secondary_does_not_change_the_primary_cache_or_preblend_surface(
    secondary_runs,
):
    """PART 2 of 4. The cached features and the pre-blend primary surface are upstream of the
    blend, so they must be invariant to it - bit for bit, not merely close."""
    off = secondary_runs["off"]["cache"]
    a = secondary_runs["sec_a"]["cache"]
    b = secondary_runs["sec_b"]["cache"]
    for key in ("role_feat", "role_pos", "role_coord_scaled", "role_coord_rel", "role_gid",
                "role_mask", "role_pair", "role_role", "coords", "pair_src_ptr", "pair_src_n",
                "pair_tgt_ptr", "pair_tgt_n"):
        assert np.array_equal(a[key], off[key]), f"{key} moved when the secondary was enabled"
        assert np.array_equal(b[key], off[key]), f"{key} moved when the secondary was changed"

    fa, fb, fo = _band_p_frame(a), _band_p_frame(b), _band_p_frame(off)
    assert fa["keys"] == fb["keys"] == fo["keys"], (
        "band P is selected on the pre-blend surface; its membership cannot depend on the blend")
    assert np.array_equal(fa["prob_pre"], fo["prob_pre"])
    assert np.array_equal(fb["prob_pre"], fo["prob_pre"])
    assert np.array_equal(fa["logit_pre"], fo["logit_pre"])
    assert np.array_equal(fb["logit_pre"], fo["logit_pre"])
    # And band A, which IS selected downstream of the blend, must have moved - otherwise the
    # invariance above would be evidence of a fixture in which nothing happens.
    assert set(zip(a["band_a_pair"].tolist(), a["band_a_i"].tolist(),
                   a["band_a_j"].tolist())) != set(
        zip(off["band_a_pair"].tolist(), off["band_a_i"].tolist(), off["band_a_j"].tolist()))


def test_the_old_comparison_fails_with_the_secondary_enabled(deployed: Sandbox, secondary_runs):
    """PART 3 of 4. Contract 1's comparison, computed by the shipped worker and required to FAIL.

    This is P36 reproduced on CPU in miniature: a primary-only replay measured against the
    DEPLOYED post-fusion probability. The worker reports that verdict alongside the corrected one
    precisely so the correction can be measured instead of argued.
    """
    res, report = deployed.replay(secondary_runs["sec_a"]["cache_dir"], "gate_sec_a.json")
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    row = report["crops"][0]
    v1 = row["contract_v1_verdict"]
    assert v1["checked"] > 0, "the v1 comparison compared nothing - the claim would be vacuous"
    assert v1["max_abs_prob_delta"] > report["tolerance"]
    assert v1["would_have_passed"] is False
    # The reason is named, not inferred: the secondary is not reproducible from a primary cache.
    assert row["fusion"]["secondary_enabled"] is True
    assert row["deployed_reconstruction"]["attempted"] is False
    assert "secondary" in row["deployed_reconstruction"]["why_not"]


def test_the_corrected_comparison_passes_with_the_secondary_enabled(deployed: Sandbox,
                                                                    secondary_runs):
    """PART 4 of 4. The same cache, the same run, the corrected target - and it PASSES.

    Parts 3 and 4 together are the whole finding: nothing about the cache changed between them,
    only what the cache was compared against.
    """
    res, report = deployed.replay(secondary_runs["sec_a"]["cache_dir"], "gate_sec_a_ok.json")
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    assert report["all_passed"] is True
    row = report["crops"][0]
    for name in BANDS:
        assert row[f"band_{name}"]["checked"] > 0
        assert row[f"band_{name}"]["max_abs_primary_prob_delta"] <= report["tolerance"]
        assert row[f"band_{name}"]["max_abs_primary_logit_delta"] <= report["logit_tolerance"]
    assert row["band_p"]["missing"] == row["band_p"]["extra"] == 0
    assert row["integrity_failure_count"] == 0
    # And the second, differently-weighted secondary passes too, so the pass is not a property
    # of one blend weight.
    res_b, report_b = deployed.replay(secondary_runs["sec_b"]["cache_dir"], "gate_sec_b.json")
    assert res_b.returncode == 0 and report_b["all_passed"] is True
    assert report_b["crops"][0]["contract_v1_verdict"]["would_have_passed"] is False


def test_the_control_arm_with_no_secondary_shows_no_gap_at_all(deployed: Sandbox,
                                                               secondary_runs):
    """The ACCEPT control for parts 3 and 4. With no fusion the two surfaces coincide exactly, so
    the gap parts 3 and 4 turn on is attributable to the blend and to nothing in the gate."""
    res, report = deployed.replay(secondary_runs["off"]["cache_dir"], "gate_off.json")
    assert res.returncode == 0 and report["all_passed"] is True
    row = report["crops"][0]
    assert row["fusion"]["stages"] == "none"
    for name in BANDS:
        assert row[f"band_{name}"]["max_abs_deployed_postblend_gap"] == 0.0
    assert row["contract_v1_verdict"]["would_have_passed"] is True


def test_the_bidirectional_stage_is_recorded_and_reconstructible(deployed: Sandbox):
    """The stage that ACTUALLY ran in P36, and the one a primary-only cache CAN reproduce.

    FACT-0403 blamed a secondary blend at 0.15 on the strength of a setup-cell log header. That
    weight had been cleared by scripts/kaggle_edits/loeo_retarget.py:128 before the shards
    launched, the same log printed ``"secondary_enabled": false``, and the stage that ran was the
    bidirectional harmonic at 0.20. The harmonic re-runs the SAME primary model with the roles
    swapped, so this cache reproduces the deployed probability EXACTLY - which both proves the
    cache faithful and shows why recording which stages ran is not optional.
    """
    deployed.apply_tap()
    cache_dir = deployed.base / "cache_bidi"
    deployed.predict("tapped_bidi.json", {
        "BIOHUB_AFT_DIR": str(cache_dir),
        "BIOHUB_AFT_BAND_B_FLOOR": str(BAND_B_FLOOR),
        "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.20",
    })
    with np.load(deployed.cache_path(cache_dir), allow_pickle=False) as cache:
        assert str(cache["fusion_stages"]) == "bidirectional_harmonic"
        assert float(cache["fusion_bidirectional_weight"]) == pytest.approx(0.20)
        assert bool(cache["fusion_secondary_enabled"]) is False
        assert bool(cache["fusion_reproducible_from_primary_cache"]) is True

    res, report = deployed.replay(cache_dir, "gate_bidi.json")
    assert res.returncode == 0 and report["all_passed"] is True
    row = report["crops"][0]
    # The v1 comparison fails here too, and for the SAME structural reason - a fused reference.
    assert row["contract_v1_verdict"]["would_have_passed"] is False
    assert row["contract_v1_verdict"]["max_abs_prob_delta"] > report["tolerance"]
    # And the deployed surface is reconstructible from the primary cache alone.
    recon = row["deployed_reconstruction"]
    assert recon["attempted"] is True
    assert recon["pairs"] > 0
    assert recon["max_abs_delta"] <= report["tolerance"]
    assert recon["band_a_missing"] == recon["band_a_extra"] == 0


def test_the_vendored_predictor_is_not_the_deployed_program():
    """DEFECT 7, kept measured rather than remembered.

    ``vendor/kaggle-cell-tracking`` is the organizer's public repository. The kernel runs the
    SUPPORT PACK. They are different programs, and the difference is exactly the fusion the
    contract-2 correction is about, so a suite that tests only the vendored copy is structurally
    incapable of catching a fusion defect.
    """
    vendored = (VENDOR / "scripts" / "predict_unet_transformer.py").read_text(encoding="utf-8")
    shipped = (_deployed_pack() / "scripts" / "predict_unet_transformer.py").read_text(
        encoding="utf-8", errors="replace")
    assert "secondary_link_mode" not in vendored
    assert "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT" not in vendored
    assert "secondary_link_mode" in shipped
    assert "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT" in shipped
    assert len(shipped.splitlines()) > len(vendored.splitlines())


# ---------------------------------------------------------------------------------------------
# 6. The gate must REJECT. A checker that has not been shown to reject is not a checker.
# ---------------------------------------------------------------------------------------------
def test_a_per_frame_feature_cache_fails_the_gate(sandbox: Sandbox, tapped_cache):
    """DEFECT 6 (FACT-0402), manufactured exactly as the old worker built it, and rejected.

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
    assert (row["band_p"]["max_abs_primary_prob_delta"] > report["tolerance"]
            or row["band_p"]["max_abs_primary_logit_delta"] > report["logit_tolerance"]
            or row["band_p"]["missing"] or row["band_p"]["extra"])


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


def test_a_per_target_logit_offset_is_rejected_although_the_probabilities_are_untouched(
    sandbox: Sandbox, tapped_cache,
):
    """THE BLIND SPOT, CLOSED, and proved closed by mutation.

    ``softmax`` over the source axis is invariant to a per-target constant offset of the logits,
    so under contract 1 - which recorded probabilities only - an error of exactly that shape was
    mathematically invisible, not merely undetected. Contract 2 records the pre-fusion logits, so
    here the recorded logits are shifted by a per-target constant, the recorded probabilities are
    left EXACTLY as they were, and the gate must still reject. The unmutated cache passing is the
    ACCEPT control immediately below.
    """
    cache_dir, _ = tapped_cache
    rng = np.random.default_rng(7)

    def shift(data: dict) -> None:
        for name in BANDS:
            key = f"band_{name}_primary_logit_preblend"
            if data[key].size == 0:
                continue
            # One constant per (pair, target) - the exact fibre the softmax collapses.
            groups = {}
            for k, (p, j) in enumerate(zip(data[f"band_{name}_pair"].tolist(),
                                           data[f"band_{name}_j"].tolist())):
                groups.setdefault((p, j), []).append(k)
            col = data[key].copy()
            for rows in groups.values():
                col[rows] += float(rng.normal() * 3.0)
            data[key] = col

    mutated = _mutated_dir(sandbox, cache_dir, "cache_logit_shift", shift)
    _res, report = sandbox.replay(mutated, "gate_logit_shift.json")
    row = report["crops"][0]
    assert report["all_passed"] is False
    assert row["band_p"]["max_abs_primary_logit_delta"] > report["logit_tolerance"]
    # The probability comparison alone would have accepted it - which is the point.
    assert row["band_p"]["max_abs_primary_prob_delta"] <= report["tolerance"]
    assert row["band_p"]["missing"] == row["band_p"]["extra"] == 0


def test_the_softmax_invariance_that_made_the_blind_spot_real_is_itself_still_true():
    """The mathematics the previous test relies on, asserted in both directions.

    A per-TARGET constant offset is invisible after ``softmax(dim=0)``; a per-SOURCE one is not.
    The invariance is a property of the LOGITS, which is why closing the blind spot required
    recording them rather than tightening a probability tolerance.
    """
    torch.manual_seed(0)
    logits = torch.randn(7, 5)
    base = torch.softmax(logits, dim=0)
    per_target = torch.softmax(logits + torch.randn(1, 5) * 3.0, dim=0)
    assert float((per_target - base).abs().max()) < 1e-5      # invisible in probability space

    per_source = torch.softmax(logits + torch.randn(7, 1) * 3.0, dim=0)
    assert float((per_source - base).abs().max()) > 1e-4      # visible


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


def test_a_mislabelled_selection_surface_fails_the_gate(sandbox: Sandbox, tapped_cache):
    """The band provenance is load-bearing, so it is checked rather than trusted.

    A cache that claims band A was selected on the pre-blend surface would invite exactly the
    re-derivation contract 2 forbids.
    """
    cache_dir, _ = tapped_cache

    def relabel(data: dict) -> None:
        data["band_a_selected_on"] = np.str_("primary_probs_preblend")

    mutated = _mutated_dir(sandbox, cache_dir, "cache_mislabelled", relabel)
    _res, report = sandbox.replay(mutated, "gate_mislabelled.json")
    row = report["crops"][0]
    assert report["all_passed"] is False
    assert any(f.get("why", "").startswith("the band declares the wrong selection surface")
               for f in row["integrity_failures"])


@pytest.mark.parametrize("band", BANDS)
def test_the_gate_refuses_an_empty_band(sandbox: Sandbox, tapped_cache, band: str):
    """FACT-0394 applied to all three bands, and ISOLATED so each refusal is attributable.

    Band B is not optional either: FACT-0382 measured that ALL 691 fold-0 contested errors sit
    below the deployed threshold, so a band-A-only pass validates the band the task does not use.
    """
    cache_dir, _ = tapped_cache

    def empty(data: dict) -> None:
        for key in ("pair", "i", "j", "source_id", "target_id"):
            data[f"band_{band}_{key}"] = np.empty(0, dtype=np.int64)
        for key in ("primary_logit_preblend", "primary_prob_preblend",
                    "deployed_prob_postblend"):
            data[f"band_{band}_{key}"] = np.empty(0, dtype=np.float64)
        if band == "p":
            # Band P's membership is re-derived, so emptying it alone would be caught as an
            # "extra" rather than as a floor violation. Raising the threshold out of reach makes
            # the refusal attributable to the floor and to nothing else.
            data["edge_threshold"] = np.float64(0.999999)

    mutated = _mutated_dir(sandbox, cache_dir, f"cache_no_{band}", empty)
    _res, report = sandbox.replay(mutated, f"gate_no_{band}.json")
    row = report["crops"][0]
    assert report["all_passed"] is False
    assert row[f"band_{band}"]["checked"] == 0
    others = [n for n in BANDS if n != band and not (band == "p")]
    for name in others:
        assert row[f"band_{name}"]["checked"] > 0, (
            f"the refusal must be band {band}, not a collateral band-{name} miss")


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


def test_the_gate_refuses_a_run_that_captured_fewer_crops_than_expected(sandbox: Sandbox,
                                                                        tapped_cache):
    """The crop COUNT is external, and `all_passed` cannot see a crop that never ran.

    `all_passed` is an ALL over the crops that exist, so a run that captured one of two crops -
    or whose second crop raised before the flush - satisfies it exactly as well as a complete
    run. PKT-0036 measured that: a one-crop run passed. The expectation therefore has to be
    carried IN from the spec, and this proves the floor fires rather than assuming it does.
    """
    cache_dir, _ = tapped_cache
    _res, report = sandbox.replay(cache_dir, "gate_expect_2.json",
                                  extra_args=["--expect-crops", "2"])
    assert len(report["crops"]) == 1
    assert report["all_passed"] is False
    assert report["crop_count"]["captured"] == 1
    assert report["crop_count"]["expected"] == 2
    assert "expected 2 crops" in report["crop_count"]["error"]
    # ...and every per-crop verdict is still a pass, so the refusal is the COUNT and nothing else.
    assert all(c["passed"] for c in report["crops"])


def test_the_gate_accepts_the_crop_count_it_was_told_to_expect(sandbox: Sandbox, tapped_cache):
    """ACCEPT control. A floor that refuses every count is not a floor, it is an outage."""
    cache_dir, _ = tapped_cache
    _res, report = sandbox.replay(cache_dir, "gate_expect_1.json",
                                  extra_args=["--expect-crops", "1"])
    assert report["crop_count"] == {
        "captured": 1, "expected": 1, "asserted": True,
        "why_external": report["crop_count"]["why_external"],
    }
    assert report["all_passed"] is True


def test_a_torn_coordinate_rescale_fails_the_gate(sandbox: Sandbox, tapped_cache):
    """FACT-0405 defect 7, manufactured on the DEPLOYED downsample rather than on (1,1,1).

    The tap flushes between the coordinate concatenate and the deployed rescale
    (predict_unet_transformer.py:798-802), so `coords` holds the downsampled grid and
    `role_coord_scaled` holds `coords[:, 1:] * downsample`. Defect 7 survived because a harness
    fixture used downsample (1,1,1), where that multiplication is the identity and the step under
    test silently disappears. This fixture runs at the deployed (1,4,4), so writing the
    UN-rescaled coordinates into the scaled column is a real change - which is exactly what the
    check must see.
    """
    cache_dir, _ = tapped_cache
    with np.load(sandbox.cache_path(cache_dir), allow_pickle=False) as cache:
        assert [int(d) for d in cache["downsample"]] == [1, 4, 4], (
            "this mutation is vacuous at downsample (1,1,1) - the fixture must differ from "
            "production in cost, never in kind"
        )

    def unscale(data: dict) -> None:
        data["role_coord_scaled"] = (
            data["coords"][data["role_gid"]][:, 1:].astype(np.float32)
        )

    mutated = _mutated_dir(sandbox, cache_dir, "cache_unrescaled", unscale)
    _res, report = sandbox.replay(mutated, "gate_unrescaled.json")
    row = report["crops"][0]
    assert report["all_passed"] is False
    assert any("coords[:, 1:] * downsample" in f.get("why", "")
               for f in row["integrity_failures"]), row["integrity_failures"]


# ---------------------------------------------------------------------------------------------
# 7. ACCEPT controls - an auditor that rejects everything checks nothing
# ---------------------------------------------------------------------------------------------
def test_a_faithful_cache_still_passes_after_a_lossless_round_trip(sandbox: Sandbox,
                                                                   tapped_cache):
    """Rewriting the cache without changing a value must not flip the verdict."""
    cache_dir, _ = tapped_cache
    same = _mutated_dir(sandbox, cache_dir, "cache_roundtrip", lambda data: None)
    _res, report = sandbox.replay(same, "gate_roundtrip.json")
    assert report["all_passed"] is True


def test_the_deployed_post_blend_column_never_gates(sandbox: Sandbox, tapped_cache):
    """The post-blend column is OBSERVATIONAL. Corrupting it must not change the verdict.

    This is the structural half of the correction: if a mutilated post-blend column could still
    fail the gate, the gate would still be comparing against it.
    """
    cache_dir, _ = tapped_cache

    def wreck(data: dict) -> None:
        for name in BANDS:
            key = f"band_{name}_deployed_prob_postblend"
            data[key] = np.full_like(data[key], 0.123456)

    mutated = _mutated_dir(sandbox, cache_dir, "cache_wrecked_post", wreck)
    _res, report = sandbox.replay(mutated, "gate_wrecked_post.json")
    row = report["crops"][0]
    assert report["all_passed"] is True, (
        "the post-blend probability is observational; it must not be able to fail the gate")
    # It IS still reported, and the contract-1 verdict moves with it - which is what makes the
    # observational record useful rather than decorative.
    assert row["band_p"]["max_abs_deployed_postblend_gap"] > 0.0
    assert row["contract_v1_verdict"]["would_have_passed"] is False


# ---------------------------------------------------------------------------------------------
# 8. Real class surface, and no drift between the worker and the patch that ships it
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


def test_the_declared_schema_itself_names_no_unqualified_probability():
    """The rule applied to the DECLARATION, so a future key cannot reintroduce the ambiguity."""
    required, optional = _declared_schema()
    offenders = sorted(
        k for k in required | optional
        if ("prob" in k or "logit" in k)
        and not (k.endswith("_preblend") or k.endswith("_postblend"))
    )
    assert offenders == [], offenders


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
    for band in BANDS:
        for field in ("selected_on", "pair", "i", "j", "source_id", "target_id",
                      "primary_logit_preblend", "primary_prob_preblend",
                      "deployed_prob_postblend"):
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
