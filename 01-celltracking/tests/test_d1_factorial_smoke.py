"""The smoke build must BE the 2x2, not merely be adjacent to it.

`scripts/d1f_probe.py::run_direction` raises `CROSS-ENCODED` when it is handed a routed-only
export, and it is right to: correction C1 established that the two 32-D bases are independent
(rel-L2 1.41542 ~ sqrt(2), cos(w0,w1) = -0.155), so a head fitted in split-0's basis cannot be
evaluated in split-1's. The 2x2 therefore needs the SAME checkpoint to encode both families,
which is four kernels and six crop-inferences -- and the build previously shipped only two of
them, both TARGET cells. D1-F was unrunnable on the smoke as a matter of arithmetic.

Everything here compares the generated specs and the BUILT notebooks against
`data/d1_factorial/manifest_smoke.json`, which is the single source of truth for the pairing.
A glob is not a manifest and a hand-written stem list is not a manifest; both are recorded
traps in this repo, so several tests below exist purely to make either one fail loudly.

Nothing here needs a GPU, a network, or the dataset.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FACT_DIR = ROOT / "data" / "d1_factorial"
MANIFEST = FACT_DIR / "manifest_smoke.json"
SPEC_DIR = ROOT / "scripts" / "kaggle_specs"
AUDIT_SRC = ROOT / "scripts" / "kaggle_edits" / "d1_response_audit.py"
INJECT_SRC = ROOT / "scripts" / "kaggle_edits" / "d1_inject.py"

sys.path.insert(0, str(ROOT / "scripts"))

import assemble_p3_d1_smoke_spec as ASM  # noqa: E402
import kaggle_factory as KF  # noqa: E402

# The three smoke crops, spelled out ONCE, here, so that a test can assert the assembler
# itself never spells them out.
SMOKE_CROPS = ("44b6_0113de3b", "6bba_57b7cc1e", "6bba_6feb10f0")

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="data/d1_factorial/manifest_smoke.json absent")


def lf(raw: bytes) -> bytes:
    """Normalise the checkout filter away. core.autocrlf=true is set in this environment."""
    return raw.replace(b"\r\n", b"\n")


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shards(manifest) -> list[dict]:
    return sorted(manifest["shards"], key=lambda s: int(s["stage"]))


@pytest.fixture(scope="module")
def specs(shards) -> dict[str, dict]:
    """cell key -> spec JSON, as committed."""
    out = {}
    for s in shards:
        key = ASM.cell_key(s)
        p = SPEC_DIR / f"p3_d1_smoke_{key}.json"
        assert p.exists(), f"shard {s['shard_id']} has no spec at {p}"
        out[key] = json.loads(p.read_text(encoding="utf-8"))
    return out


@pytest.fixture(scope="module")
def notebooks(specs) -> dict[str, list[str]]:
    """cell key -> the built notebook's code cells."""
    out = {}
    for key, spec in specs.items():
        nb = ROOT / spec["out_dir"] / spec["code_file"]
        assert nb.exists(), f"{key}: notebook not built at {nb}"
        out[key] = ["".join(c["source"])
                    for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]]
    return out


def env_of(spec: dict) -> dict[str, str]:
    return next(e for e in spec["edits"] if e["kind"] == "env")["vars"]


def audit_block_source(cells: list[str]) -> str:
    """The D1 audit block travels through the notebook base64-encoded, so its contract can
    only be checked by decoding what the build actually carries."""
    src = "\n".join(cells)
    tree = ast.parse(next(c for c in cells if "_D1_BLOCK_B64 = (" in c))
    b64 = next(
        ast.literal_eval(n.value) for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_D1_BLOCK_B64")
    return base64.b64decode(b64).decode("utf-8")


# --------------------------------------------------------------------------------------
# 1. coverage: exactly the manifest's six crop-inferences, none missing, none duplicated
# --------------------------------------------------------------------------------------
def test_every_shard_has_exactly_one_spec_and_every_spec_one_shard(shards, specs):
    assert len(specs) == len(shards) == 4
    assert {s["shard_id"] for s in shards} == {sp["d1_cell"]["shard_id"]
                                               for sp in specs.values()}
    on_disk = {p.stem for p in SPEC_DIR.glob("p3_d1_smoke_*.json")}
    assert on_disk == {f"p3_d1_smoke_{k}" for k in specs}, (
        "a stale smoke spec is still on disk; every p3_d1_smoke_* spec must be one cell of "
        "the 2x2, or a launcher will run something the manifest does not describe")


def test_the_six_crop_inferences_are_exactly_the_manifests(manifest, specs):
    want = {(int(r["fold"]), r["crop_id"]) for r in manifest["rows"]}
    got = {(int(sp["d1_cell"]["split"]), crop)
           for sp in specs.values() for crop in sp["d1_cell"]["crops"]}
    assert got == want
    assert len(want) == 6 == manifest["n_rows"]


def test_no_crop_inference_is_duplicated_across_cells(specs):
    pairs = [(sp["d1_cell"]["split"], crop)
             for sp in specs.values() for crop in sp["d1_cell"]["crops"]]
    assert len(pairs) == len(set(pairs)) == 6, (
        f"a crop is encoded twice, i.e. paid for twice: {sorted(pairs)}")


def test_three_crops_under_both_checkpoints_which_is_what_makes_d1f_runnable(specs):
    """The whole point of this lane. A routed-only export gives each family its own basis;
    d1f_probe.run_direction then finds 0 source or 0 target rows in a basis and aborts."""
    by_split: dict[int, set[str]] = {0: set(), 1: set()}
    for sp in specs.values():
        by_split[int(sp["d1_cell"]["split"])].update(sp["d1_cell"]["crops"])
    assert by_split[0] == by_split[1] == set(SMOKE_CROPS)
    for split, crops in by_split.items():
        fams = {c.split("_")[0] for c in crops}
        assert fams == {"44b6", "6bba"}, (
            f"checkpoint split_{split} encodes only {fams}; run_direction needs BOTH "
            f"families inside one basis or it raises CROSS-ENCODED")


def test_the_stem_lists_reaching_the_kernels_are_the_manifests(specs, notebooks):
    for key, spec in specs.items():
        declared = json.loads(env_of(spec)["BIOHUB_LOEO_STEMS"])
        assert declared == spec["d1_cell"]["crops"]
        allb = "\n".join(notebooks[key])
        assert f"""os.environ["BIOHUB_LOEO_STEMS"] = {json.dumps(declared)!r}""" in allb


# --------------------------------------------------------------------------------------
# 2. checkpoints
# --------------------------------------------------------------------------------------
def test_each_cell_pins_the_manifest_checkpoint_sha_everywhere(shards, specs, notebooks):
    for s in shards:
        key = ASM.cell_key(s)
        spec = specs[key]
        want = s["checkpoint_sha256"]
        assert spec["d1_cell"]["checkpoint_sha256"] == want
        assert spec["provenance"]["checkpoint_sha256"] == want
        assert env_of(spec)["BIOHUB_D1_CKPT_SHA"] == want
        assert "\n".join(notebooks[key]).count(want) >= 2, (
            f"{key}: the checkpoint hash must appear both as the env pin and inside the "
            f"cell-identity record that the kernel re-verifies against the loaded weights")


def test_checkpoint_sha_is_the_independently_pinned_split_hash(specs):
    for key, spec in specs.items():
        cell = spec["d1_cell"]
        assert cell["checkpoint_sha256"] == ASM.CKPT_SHA[int(cell["split"])], key


def test_the_weights_and_config_globs_agree_with_the_declared_basis(specs):
    for key, spec in specs.items():
        split = int(spec["d1_cell"]["split"])
        env = env_of(spec)
        assert env["BIOHUB_LOEO_WEIGHTS_GLOB"].endswith(
            f"edge_predictor_best_split_{split}.pth"), key
        assert env["BIOHUB_LOEO_CONFIG_GLOB"].endswith(f"config_split_{split}.json"), key
        assert env["BIOHUB_D1_SPLIT"] == str(split), key


# --------------------------------------------------------------------------------------
# 3. roles -- the assertion the routed-only build could not make
# --------------------------------------------------------------------------------------
def test_role_is_a_derived_fact_and_matches_the_manifest(shards, specs):
    for s in shards:
        spec = specs[ASM.cell_key(s)]
        cell = spec["d1_cell"]
        split, family = int(cell["split"]), cell["family"]
        derived = "source" if family == ASM.SPLIT_SOURCE_FAMILY[split] else "target"
        assert derived == cell["role"] == s["role"]


def test_target_cell_never_uses_the_checkpoint_trained_on_its_own_family(specs):
    """A TARGET cell is the measurement. If its checkpoint had trained on that family the
    number would be in-sample and the whole transfer question would be void."""
    for key, spec in specs.items():
        cell = spec["d1_cell"]
        if cell["role"] != "target":
            continue
        split = int(cell["split"])
        assert cell["family"] != ASM.SPLIT_SOURCE_FAMILY[split], key
        assert cell["family"] == ASM.SPLIT_HELDOUT_FAMILY[split], key
        assert cell["family"] == cell["checkpoint_heldout_family"], key


def test_source_cell_uses_exactly_the_checkpoints_training_family(specs):
    """A SOURCE cell exists only to supply fitting rows for a 33-parameter pointwise head
    inside one basis. Fitting on the held-out family would be the target leaking in."""
    for key, spec in specs.items():
        cell = spec["d1_cell"]
        if cell["role"] != "source":
            continue
        split = int(cell["split"])
        assert cell["family"] == ASM.SPLIT_SOURCE_FAMILY[split], key
        assert cell["family"] == cell["checkpoint_trained_on_family"], key


def test_each_checkpoint_has_exactly_one_source_and_one_target_cell(specs):
    seen: dict[tuple[int, str], str] = {}
    for key, spec in specs.items():
        cell = spec["d1_cell"]
        k = (int(cell["split"]), cell["role"])
        assert k not in seen, f"{key} and {seen[k]} are the same cell of the 2x2"
        seen[k] = key
    assert set(seen) == {(0, "source"), (0, "target"), (1, "source"), (1, "target")}


def test_every_cell_is_family_pure(specs):
    for key, spec in specs.items():
        cell = spec["d1_cell"]
        assert {c.split("_")[0] for c in cell["crops"]} == {cell["family"]}, key


# --------------------------------------------------------------------------------------
# 4. role inversion is inexpressible -- the kernel re-derives and refuses
# --------------------------------------------------------------------------------------
class _FakeDigest:
    def __init__(self, hexd: str):
        self._h = hexd

    def hexdigest(self) -> str:
        return self._h


class _FakeHashlib:
    def __init__(self, hexd: str):
        self._h = hexd

    def sha256(self, _data):
        return _FakeDigest(self._h)


def _run_identity(rec: dict, tmp_path, *, stems=None, fold=None, observed_sha=None,
                  env_over: dict | None = None):
    """Execute the generated cell-identity block against fake kernel globals.

    Only one substitution is made to the generated source: the /kaggle/working audit path is
    redirected into tmp_path. Every guard runs verbatim.
    """
    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    code = ASM.identity_code(rec).replace('"/kaggle/working/d1_audit"', repr(str(tmp_path)))
    weights = tmp_path / "edge_predictor.pth"
    weights.write_bytes(b"not the real weights; the digest is faked below")
    env = {
        "BIOHUB_D1_SPLIT": str(rec["split"]),
        "BIOHUB_D1_CKPT_SHA": rec["checkpoint_sha256"],
        "BIOHUB_D1_EXPECTED_CROPS": str(rec["n_crops"]),
    }
    env.update(env_over or {})
    g = {
        "LOEO_STEMS": list(rec["crops"] if stems is None else stems),
        "LOEO_FOLD": rec["fold"] if fold is None else fold,
        "LOEO_ARM": "strict",
        "WEIGHTS_RELATIVE": str(weights),
        "Path": pathlib.Path,
        "json": json,
        "os": types.SimpleNamespace(environ=env),
        "hashlib": _FakeHashlib(observed_sha or rec["checkpoint_sha256"]),
    }
    exec(compile(code, "d1_cell_identity", "exec"), g)
    return json.loads((tmp_path / "d1_cell.json").read_text(encoding="utf-8"))


def test_a_correct_cell_runs_and_writes_its_identity(specs, tmp_path):
    for key, spec in specs.items():
        out = _run_identity(spec["d1_cell"], tmp_path / key)
        cell = spec["d1_cell"]
        for field in ("shard_id", "checkpoint_sha256", "split", "family", "role"):
            assert out[field] == cell[field], (key, field)
        assert out["crops"] == cell["crops"]
        assert out["crops_mounted"] == sorted(cell["crops"])
        assert out["role_rederived"] == cell["role"]
        assert out["family_rederived"] == cell["family"]
        assert out["checkpoint_sha256_observed"] == cell["checkpoint_sha256"]
        assert out["arm"] == "strict"


@pytest.mark.parametrize("role_from,role_to", [("source", "target"), ("target", "source")])
def test_a_flipped_role_label_cannot_run(specs, tmp_path, role_from, role_to):
    """Role inversion must be IMPOSSIBLE TO EXPRESS. Relabelling the record is not enough:
    the kernel recomputes the role from the checkpoint split and the crop family."""
    spec = next(s for s in specs.values() if s["d1_cell"]["role"] == role_from)
    bad = dict(spec["d1_cell"])
    bad["role"] = role_to
    with pytest.raises(RuntimeError, match="ROLE INVERSION"):
        _run_identity(bad, tmp_path)


def test_swapping_the_crops_for_the_other_familys_cannot_run(specs, tmp_path):
    """The other half of the same guard: keep the label, change what is mounted."""
    spec = specs["f0"]                       # split_0 / 44b6 / target
    with pytest.raises(RuntimeError) as exc:
        _run_identity(spec["d1_cell"], tmp_path, stems=["6bba_57b7cc1e"])
    msg = str(exc.value)
    assert "ROLE INVERSION" in msg and "family: mounted 6bba" in msg


def test_a_crop_list_that_is_not_the_declared_one_cannot_run(specs, tmp_path):
    spec = specs["f0_src"]                   # split_0 / 6bba / source, 2 crops
    with pytest.raises(RuntimeError, match="crop list"):
        _run_identity(spec["d1_cell"], tmp_path, stems=["6bba_57b7cc1e"],
                      env_over={"BIOHUB_D1_EXPECTED_CROPS": "2"})


def test_the_wrong_checkpoint_cannot_run(specs, tmp_path):
    spec = specs["f1"]
    with pytest.raises(RuntimeError, match="checkpoint: loaded sha256"):
        _run_identity(spec["d1_cell"], tmp_path, observed_sha="00" * 32)


def test_a_mismatched_basis_env_cannot_run(specs, tmp_path):
    spec = specs["f1_src"]
    with pytest.raises(RuntimeError, match="BIOHUB_D1_SPLIT"):
        _run_identity(spec["d1_cell"], tmp_path, env_over={"BIOHUB_D1_SPLIT": "0"})


def test_the_identity_block_is_present_exactly_once_in_every_build(specs, notebooks):
    for key, spec in specs.items():
        allb = "\n".join(notebooks[key])
        block = ASM.identity_code(spec["d1_cell"])
        assert allb.count(block.rstrip("\n")) == 1, key
        assert allb.count("D1 FACTORIAL CELL IDENTITY") == 1, key
        assert allb.count(spec["d1_cell"]["shard_id"]) >= 1, key


# --------------------------------------------------------------------------------------
# 5. the manifest is the source of truth, not a glob and not a transcription
# --------------------------------------------------------------------------------------
def test_the_assembler_never_spells_out_a_crop_id(specs):
    """`data/d1_factorial/manifest_smoke.json` is authoritative. If the crop ids appear in
    the assembler, someone has transcribed the routing again -- which is exactly how the
    build ended up routed-only in the first place."""
    src = pathlib.Path(ASM.__file__).read_text(encoding="utf-8")
    for crop in SMOKE_CROPS:
        assert crop not in src, f"{crop} is hard-coded in the assembler"


def test_spec_routing_env_is_the_shard_env_verbatim(shards, specs):
    for s in shards:
        env = env_of(specs[ASM.cell_key(s)])
        for name, value in s["env"].items():
            assert env[name] == value, f"{s['shard_id']}: {name} drifted from the manifest"


def test_the_spec_records_where_its_routing_came_from(shards, specs):
    lf_sha = hashlib.sha256(lf(MANIFEST.read_bytes())).hexdigest()
    for s in shards:
        spec = specs[ASM.cell_key(s)]
        assert s["shard_id"] in spec["provenance"]["routing_source"]
        assert spec["d1_cell"]["manifest_sha256_lf"] == lf_sha
        assert spec["d1_cell"]["manifest"] == "data/d1_factorial/manifest_smoke.json"


def test_no_shard_needs_a_kernel_edit(shards):
    """Verified against scripts/kaggle_edits/loeo_retarget.py: it globs every train .zarr and
    selects purely by BIOHUB_LOEO_STEMS membership, with no fold-membership guard anywhere,
    so a SOURCE cell is a configuration change and not a code change."""
    retarget = (ROOT / "scripts" / "kaggle_edits" / "loeo_retarget.py").read_text(
        encoding="utf-8")
    assert "LOEO_STEMS = [s for s in LOEO_STEMS_DECLARED if s in _loeo_by_stem]" in retarget
    tree = ast.parse(retarget)
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)
              and "LOEO_FOLD" in ast.unparse(n)]
    assert not guards, (
        "loeo_retarget.py now compares against LOEO_FOLD; a fold-membership guard would make "
        f"SOURCE cells unrunnable without a kernel edit: {[ast.unparse(g) for g in guards]}")
    for s in shards:
        assert s["kernel_edit_change_required"] is False, s["shard_id"]


# --------------------------------------------------------------------------------------
# 6. determinism
# --------------------------------------------------------------------------------------
def test_regeneration_byte_reproduces_every_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(ASM, "SPECS", tmp_path)
    ASM.main()
    regenerated = sorted(p.name for p in tmp_path.glob("p3_d1_smoke_*.json"))
    committed = sorted(p.name for p in SPEC_DIR.glob("p3_d1_smoke_*.json"))
    assert regenerated == committed
    for name in committed:
        assert lf((tmp_path / name).read_bytes()) == lf((SPEC_DIR / name).read_bytes()), (
            f"{name} is not byte-reproducible from the manifest")


def test_regenerating_twice_is_byte_identical(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        monkeypatch.setattr(ASM, "SPECS", d)
        ASM.main()
    for p in sorted(a.glob("*.json")):
        assert p.read_bytes() == (b / p.name).read_bytes(), p.name


def test_every_notebook_rebuilds_byte_identically_from_its_spec(specs):
    """Replays kaggle_factory's build in memory rather than on disk, so a failure never
    leaves a half-written notebook behind."""
    for key, spec in specs.items():
        nb = KF.load_nb(ROOT / spec["base_notebook"])
        for n, edit in enumerate(spec["edits"]):
            KF.apply_edit(nb, edit, f"{key}/edit{n}")
        rebuilt = json.dumps(nb, indent=1, ensure_ascii=False)
        on_disk = (ROOT / spec["out_dir"] / spec["code_file"]).read_text(encoding="utf-8")
        assert rebuilt == on_disk, f"{key}: the committed notebook is not its spec's output"


def test_kernel_metadata_is_present_private_and_gpu(specs):
    for key, spec in specs.items():
        meta = json.loads(
            (ROOT / spec["out_dir"] / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta == KF.metadata_for({**spec, "_spec_path": ""}), key
        assert meta["is_private"] is True and meta["enable_internet"] is False
        assert meta["enable_gpu"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
        assert spec["expects_submission"] is False


def test_every_cell_is_independently_launchable_under_its_own_slug(specs):
    slugs = {spec["slug"] for spec in specs.values()}
    dirs = {spec["out_dir"] for spec in specs.values()}
    files = {spec["code_file"] for spec in specs.values()}
    assert len(slugs) == len(dirs) == len(files) == 4, (
        "two cells share a slug, directory or notebook filename; one launch would overwrite "
        "the other's kernel")


# --------------------------------------------------------------------------------------
# 7. everything v6 established survives in ALL FOUR builds
# --------------------------------------------------------------------------------------
def test_the_four_builds_are_one_kernel_apart_from_env_and_identity(specs, notebooks):
    """The source cells must be the SAME instrument as the target cells. Strip the env cell
    and the generated identity block and every remaining byte must coincide."""
    stripped: dict[str, list[str]] = {}
    for key, spec in specs.items():
        cells = list(notebooks[key])
        block = ASM.identity_code(spec["d1_cell"]).rstrip("\n")
        hits = [i for i, c in enumerate(cells) if block in c]
        assert len(hits) == 1, key
        cells[hits[0]] = cells[hits[0]].replace(block, "<IDENTITY>")
        env_idx = [i for i, c in enumerate(cells) if "BIOHUB_LOEO_STEMS" in c
                   and "factory env overrides" in c]
        assert len(env_idx) == 1, key
        cells[env_idx[0]] = "<ENV>"
        stripped[key] = cells
    ref_key, ref = next(iter(stripped.items()))
    for key, cells in stripped.items():
        assert cells == ref, f"{key} and {ref_key} are not the same kernel"


def test_seven_distinct_views_over_eight_encode_calls_survive_verbatim(specs, notebooks):
    for key in specs:
        allb = "\n".join(notebooks[key])
        # the collision itself: rot90(k=1) then transpose IS flip(-1), already view 1
        assert "torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)" in allb, key
        assert allb.count("_nv += 1") >= 7, key
        assert 'os.environ["BIOHUB_D1_REQUIRE_ENCODE_CALLS"] = \'8\'' in allb, key
        assert 'os.environ["BIOHUB_D1_REQUIRE_DISTINCT_VIEWS"] = \'7\'' in allb, key
        assert "n_views" not in allb, (
            f"{key}: a conflated view count reached the notebook; 8 encode calls and 7 "
            f"distinct views are two different numbers")
        assert "TTA WARNING" not in allb, f"{key}: the print guard survived"
        assert allb.count("TTA PATCH FAILED") == 1, f"{key}: the raise guard is missing"
        assert allb.index("TTA patch applied") < allb.index("D1 + D1-F INJECTION"), key


def test_unet_out_is_never_reassigned_in_any_build(specs, notebooks):
    """v6 trap, spec section 0. The injector refuses to write a predict script in which
    `unet_out` appears on the left of an assignment; that guard must reach every build, and
    the TTA mean must live in its own tensor rather than overwriting the association
    substrate. Substring scanning of the whole notebook cannot be used here -- the guard
    itself contains the forbidden forms as string literals."""
    for key in specs:
        allb = "\n".join(notebooks[key])
        assert "unet_out is read-only." in allb, f"{key}: the mutation guard is gone"
        assert allb.count('"unet_out.add_", "unet_out.div_",') == 1, key
        assert "_d1_unet_tta = unet_out.clone()" in allb, (
            f"{key}: the TTA mean no longer lives in a separate tensor")
        assert "del unet_out, _d1_unet_tta" in allb, key
        # the two representations reach the audit in the order it binds them
        assert "_d1_unet_tta[0, f_idx],\\n                    unet_out[0, f_idx]," in allb, key


def test_the_parity_gate_and_its_peak_set_equality_survive(specs, notebooks):
    for key in specs:
        block = audit_block_source(notebooks[key])
        assert "peak_set_identical" in block, key
        assert "ACCEPTED-PEAK SETS DIFFER" in block, key
        assert "_D1_PARITY_SLACK" in block, key
        assert 'os.environ["BIOHUB_D1_PARITY_SLACK"] = \'2.0\'' in "\n".join(notebooks[key])


def test_the_nan_sentinel_row_id_and_radii_survive(specs, notebooks):
    for key in specs:
        block = audit_block_source(notebooks[key])
        assert "feat_max_valid" in block, key
        assert "_d1_np.nan" in block, key
        assert '"row_id"' in block, key
        assert "match_um" in block and "search_um" in block, key


def test_every_cell_runs_the_strict_arm(specs, notebooks):
    """strict turns the secondary detector off, which is what makes the 8-encode-call mean
    the ONLY gap between w.x + b and rows.logit -- and is also why these numbers are not
    directly deployable (correction C5)."""
    for key, spec in specs.items():
        assert env_of(spec)["BIOHUB_LOEO_ARM"] == "strict", key
        assert "correction C5" in spec["purpose"], key
        assert 'os.environ["BIOHUB_LOEO_ARM"] = \'strict\'' in "\n".join(notebooks[key]), key


def test_every_biohub_d1_env_name_is_read_by_the_audit_or_the_injector(specs):
    """A spec that sets a name nothing reads is a gate that does not exist. The prior defect
    shipped BIOHUB_D1_REQUIRE_TTA_VIEWS=8, which no code read."""
    audit = AUDIT_SRC.read_text(encoding="utf-8")
    inject = INJECT_SRC.read_text(encoding="utf-8")
    for key, spec in specs.items():
        for name in env_of(spec):
            if not name.startswith("BIOHUB_D1_"):
                continue
            assert f'"{name}"' in audit or f'"{name}"' in inject, f"{key}: {name}"
        assert "BIOHUB_D1_REQUIRE_TTA_VIEWS" not in env_of(spec), key


def test_the_audit_directory_is_retained_by_every_cell(specs, notebooks):
    for key in specs:
        allb = "\n".join(notebooks[key])
        assert '"d1_audit"' in allb, f"{key}: the cleanup rmtree would delete the audit"
        assert "pregraphs_split{LOEO_FOLD}.parquet" in allb, key


# --------------------------------------------------------------------------------------
# 8. budget: every cell must fit its own session
# --------------------------------------------------------------------------------------
def test_every_cell_fits_one_session(manifest, shards):
    cap = float(manifest["budget"]["max_shard_seconds"])
    for s in shards:
        assert float(s["est_wall_hours"]) * 3600.0 <= cap, s["shard_id"]


def test_storage_follows_the_merged_factorial_model(manifest, shards):
    gt = {r["crop_id"]: int(r["gt_nodes"]) for r in manifest["rows"]}
    total = 0
    for s in shards:
        want = sum(2_636_600 + 375 * gt[c] for c in s["stems"])
        assert int(s["est_bytes"]) == want, s["shard_id"]
        total += want
    assert total == int(manifest["budget"]["total"]["bytes"])
    assert int(manifest["budget"]["total"]["crop_inferences"]) == 6


def test_the_four_kernels_fit_the_remaining_gpu_budget(manifest):
    total = float(manifest["budget"]["total"]["wall_hours_incl_fixed"])
    assert total == pytest.approx(1.3232, abs=1e-4)
    assert total < float(manifest["feasibility"]["roadmap_total_remaining_t4_hours"])
    assert manifest["feasibility"]["every_shard_fits_one_session"] is True
