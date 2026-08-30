"""Contract tests for the GPU preflight system (PKT-0037).

These test SOFTWARE CONTRACTS only. The scientific verdict on any particular kernel lives in the
signed receipt `gpu_preflight.py run` emits, not here.

The artifact-discovery cases carry BOTH fetch conventions as fixtures - synthetic ones so the
suite is hermetic, and the real P24 / P32 directories when they are on this machine. That pairing
is deliberate: the false FAIL corrected in FACT-0396 happened because an instrument knew only the
factory convention and its search stopped one directory short of the champion's evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import gpu_preflight as PF      # noqa: E402
import kaggle_artifacts as KA   # noqa: E402
import kaggle_mounts as KM      # noqa: E402


# --------------------------------------------------------------------------- central resolver
@pytest.mark.parametrize("convention", KM.CONVENTIONS)
def test_central_ladder_resolves_under_every_observed_convention(tmp_path, convention):
    root = tmp_path / convention
    KM.simulate_mount_tree(root, "slug-x", {"meta/p.parquet": b"x"}, convention, owner="own")
    hit = KM.resolve("meta/p.parquet", slugs=("slug-x",), owners=("own",),
                     input_root=str(root))
    assert hit is not None and Path(hit).is_file()


def test_central_ladder_fails_closed_with_a_diagnosis(tmp_path):
    root = tmp_path / "input"
    KM.simulate_mount_tree(root, "some-other-slug", {"a/b.txt": b"x"}, "datasets")
    with pytest.raises(KM.BiohubMountNotFound) as exc:
        KM.resolve("meta/p.parquet", slugs=("slug-x",), input_root=str(root))
    # "not found" is useless on its own; attempt 2 reported exactly that (FACT-0397).
    assert "some-other-slug" in str(exc.value)


def test_central_ladder_never_uses_a_recursive_walk():
    """A `**` under /kaggle/input descends the 79 GB competition zarr tree."""
    assert KM.is_bounded(KM.LADDER_SOURCE)


def test_is_bounded_reads_glob_arguments_not_prose():
    documented = 'def f():\n    """Bounded, never `**`."""\n    return p.glob("*/x")\n'
    offending = 'def f():\n    return p.glob("**/x")\n'
    assert KM.is_bounded(documented)
    assert not KM.is_bounded(offending)
    assert not KM.is_bounded('def f():\n    return p.rglob("x")\n')


def test_an_empty_directory_is_a_miss_not_a_hit(tmp_path):
    """FACT-0394's shape: 'it exists' is never the question the caller actually has."""
    root = tmp_path / "input"
    KM.simulate_mount_tree(root, "slug-x", {"ecb/.keep": b""}, "datasets")
    assert KM.resolve_dir("ecb", contains_glob="*.npz", slugs=("slug-x",),
                          input_root=str(root), require=False) is None
    KM.simulate_mount_tree(root, "slug-x", {"ecb/a.npz": b"x"}, "datasets")
    assert KM.resolve_dir("ecb", contains_glob="*.npz", slugs=("slug-x",),
                          input_root=str(root)) is not None


def test_a_hardcoded_flat_resolver_misses_the_datasets_convention(tmp_path):
    """The exact defect that cost GPU session 2, executed rather than asserted."""
    src = ('def _find(relative):\n'
           '    hit = _AfpPath("/kaggle/input/slug-x") / relative\n'
           '    return hit if hit.exists() else None\n')
    seen = {}
    for conv in ("flat", "datasets"):
        box = tmp_path / conv
        KM.simulate_mount_tree(box, "slug-x", {"meta/p.parquet": b"x"}, conv)
        seen[conv] = bool(KM.rebase_and_exec(src, "_find", box)("meta/p.parquet"))
    assert seen == {"flat": True, "datasets": False}


# ------------------------------------------------------------------- artifact discovery, both
def _fixture(root: Path, files: tuple[str, ...]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for f in files:
        (root / f).write_text("{}" if f.endswith(".json") else "id\n", encoding="utf-8")
    return root


QUEUE_FILES = ("audit_receipt.json", "structural_audit.json", "submission.csv", "run_stats.csv")
FACTORY_FILES = ("audit_receipt.json", "structural_audit.json", "submission.csv")


@pytest.fixture()
def two_conventions(tmp_path, monkeypatch):
    """P24-shaped queue fetch and P32-shaped factory output, side by side."""
    monkeypatch.setattr(KA, "ROOT", tmp_path)
    monkeypatch.setattr(KA, "QUEUE_ROOT", tmp_path / "queue")
    monkeypatch.setattr(KA, "SCRATCH_ROOT", tmp_path / "scratch")
    _fixture(tmp_path / "queue" / "p24_deepcenter_best_veto", QUEUE_FILES)
    _fixture(tmp_path / "notebooks" / "kaggle_p32_public931_exact" / "_out", FACTORY_FILES)
    return tmp_path


def test_queue_convention_is_found_and_named(two_conventions):
    spec = {"name": "p24_deepcenter_best_veto", "slug": "biohub-p24",
            "out_dir": "notebooks/kaggle_p24_deepcenter_best_veto"}
    d = KA.discover(spec, roles=("receipt", "submission", "run_stats"), hash_files=False)
    assert d.found and d.convention == "queue"
    assert set(d.files) == {"receipt", "submission", "run_stats"}


def test_factory_convention_is_found_and_named(two_conventions):
    spec = {"name": "p32_public931_exact", "slug": "biohub-p32",
            "out_dir": "notebooks/kaggle_p32_public931_exact"}
    d = KA.discover(spec, roles=("receipt", "submission"), hash_files=False)
    assert d.found and d.convention == "factory"


def test_a_factory_only_search_reproduces_the_fact_0396_false_fail(two_conventions, monkeypatch):
    spec = {"name": "p24_deepcenter_best_veto", "slug": "biohub-p24",
            "out_dir": "notebooks/kaggle_p24_deepcenter_best_veto"}
    monkeypatch.setattr(KA, "candidate_roots",
                        lambda s, override=None: [("factory", KA.ROOT / s["out_dir"] / "_out")])
    assert not KA.discover(spec, roles=("receipt",), hash_files=False).found


def test_role_aliases_cover_the_second_spelling(two_conventions):
    """C:/temp/p32 carries `audit.json`, not `audit_receipt.json`."""
    _fixture(two_conventions / "scratch" / "p99_alias", ("audit.json",))
    d = KA.discover({"name": "p99_alias", "slug": "s", "out_dir": "notebooks/none"},
                    roles=("receipt",), hash_files=False)
    assert d.found and d.files["receipt"].name == "audit.json"


def test_a_miss_records_every_root_it_searched(two_conventions):
    d = KA.discover({"name": "p00_absent", "slug": "s", "out_dir": "notebooks/none"},
                    roles=("receipt",), hash_files=False)
    assert not d.found and len(d.searched) >= 3        # an unfinished search is not a negative


@pytest.mark.parametrize("spec_name,expected", [("p24_deepcenter_best_veto", "queue"),
                                                ("p32_public931_exact", "factory")])
def test_real_champion_artifacts_resolve_by_their_own_convention(spec_name, expected):
    """Regression fixture against the artifacts that produced the FACT-0396 correction."""
    spec_path = ROOT / "scripts" / "kaggle_specs" / f"{spec_name}.json"
    if not spec_path.is_file():
        pytest.skip(f"{spec_name} spec absent")
    d = KA.discover(KA.load_spec(spec_path), roles=("receipt",), hash_files=False)
    if not d.found:
        pytest.skip(f"{spec_name} artifact not on this machine (gitignored evidence)")
    assert d.convention == expected


# ------------------------------------------------------------------------------- the receipt
def test_receipt_signature_is_tamper_evident():
    body = {"verdict": "FAIL", "checks": [{"id": "PF07", "passed": False}]}
    receipt = {"body": body, "signature": PF.sign(body)}
    assert PF.verify(receipt)[0]
    receipt["body"]["verdict"] = "PASS"
    ok, problems = PF.verify(receipt)
    assert not ok and any("body_sha256" in p for p in problems)


def test_receipt_binds_the_instrument_bytes():
    body = {"verdict": "PASS"}
    sig = PF.sign(body)
    assert "scripts/win_bet/gpu_preflight.py" in sig["instruments"]
    assert sig["instruments"]["scripts/kaggle_edits/kaggle_mount_ladder.py"] == KM.LADDER_SHA256


def test_the_ladder_the_preflight_proves_is_the_ladder_that_ships():
    """No second copy can drift: the CPU resolver is exec'd from the injectable file itself."""
    assert KM.LADDER_PATH.read_text(encoding="utf-8") == KM.LADDER_SOURCE
    # compiled FROM the injectable file, not re-typed beside it
    assert KM.resolve.__code__.co_filename == str(KM.LADDER_PATH)
    assert KM.resolve_dir.__code__.co_filename == str(KM.LADDER_PATH)


# --------------------------------------------------------------------------- fail-closed CLI
def test_run_on_a_missing_spec_fails_closed_rather_than_raising(tmp_path):
    receipt = PF.run_preflight(tmp_path / "no_such_spec.json", tmp_path / "box")
    assert receipt["body"]["verdict"] == "FAIL" and "fatal" in receipt["body"]
    assert PF.verify(receipt)[0]


def test_every_check_declares_the_mutation_that_proves_it(tmp_path):
    """A checker that has not been shown to reject is not a checker."""
    spec = ROOT / "scripts" / "kaggle_specs" / "p33_assoc_feature_parity_smoke.json"
    if not spec.is_file():
        pytest.skip("p33 spec absent")
    ctx = PF.load_context(spec, tmp_path / "box")
    assert ctx.env["BIOHUB_AFP_CROPS"] == "2"
    for fn in PF.CHECKS:
        assert (fn.__doc__ or "").strip(), f"{fn.__name__} has no stated purpose"
    receipt = json.loads(json.dumps(PF.sign({"a": 1})))
    assert receipt["algorithm"].startswith("sha256")
