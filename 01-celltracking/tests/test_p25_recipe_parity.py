"""Software contract for scripts/kaggle_specs/p25_recipe_parity_loeo_f0.json (LEVER-0023 sizing, PKT-0019).

P25 is the champion configuration (P9 coupled division on the P3 harmonic base, FACT-0322) retargeted at
LOEO fold 0 with ONE substitution: the primary edge weights are OUR out-of-fold split_0 (trained on 6bba,
aryaarun07/biohub-oof-weights) instead of the pack's split_0. It pairs against the EXP-0022 control on the
same 71 crops with the FACT-0321 instrument. These tests pin the spec to exactly that and nothing else;
whether the July recipe reproduces the public weights is the experiment's job, not a unit test's.

The two traps this guards against, both of which have already cost a run:
  * a copied spec that keeps its sibling's out_dir/slug silently repoints the sibling's kernel
    (AGENTS.md section 5);
  * a LOEO spec whose weights glob resolves the wrong split scores an embryo with a model that saw it
    (tests/test_loeo_weights_hygiene.py, the EXP-0019 defect).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPECS_DIR = ROOT / "scripts" / "kaggle_specs"
SPEC = SPECS_DIR / "p25_recipe_parity_loeo_f0.json"
SIBLING = SPECS_DIR / "deploy_h1r_edge_s5_loeo_f0.json"
D1_PILOT = SPECS_DIR / "p3_d1_pilot_f0.json"

OOF_DATASET = "aryaarun07/biohub-oof-weights"
WEIGHTS_GLOB_KEY = "BIOHUB_LOEO_WEIGHTS_GLOB"
CONFIG_GLOB_KEY = "BIOHUB_LOEO_CONFIG_GLOB"
GLOB_KEYS = {WEIGHTS_GLOB_KEY, CONFIG_GLOB_KEY}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _env(spec: dict) -> dict:
    """Merged env vars across every env edit, as loeo_retarget.py will see them."""
    env: dict = {}
    for edit in spec.get("edits", []):
        if edit.get("kind") == "env":
            env.update(edit.get("vars", {}))
    return env


@pytest.fixture(scope="module")
def spec() -> dict:
    return _load(SPEC)


@pytest.fixture(scope="module")
def sibling() -> dict:
    return _load(SIBLING)


def test_oof_weights_dataset_is_attached(spec, sibling):
    assert OOF_DATASET in spec["datasets"], (
        f"{SPEC.name} points the weights glob at the OOF dataset but does not attach it; "
        "loeo_retarget.py would find zero hits and fail closed (or, worse, a sibling spec's copy "
        "would fall back to the pack weights)"
    )
    # Exactly the champion's inputs plus the OOF weights - nothing else mounted.
    assert spec["datasets"] == list(sibling["datasets"]) + [OOF_DATASET]


def test_both_globs_name_split_0_and_never_split_1(spec):
    env = _env(spec)
    weights, config = env[WEIGHTS_GLOB_KEY], env[CONFIG_GLOB_KEY]
    for glob in (weights, config):
        assert "split_0" in glob, f"{glob!r} does not name split_0"
        assert "split_1" not in glob, f"{glob!r} names split_1 - trained on 44b6, the fold-0 held-out embryo"
        # loeo_retarget.py resolves the declared pattern first, then a bounded depth ladder on the
        # BASENAME under /kaggle/input; the pattern must therefore live under /kaggle/input.
        assert glob.startswith("/kaggle/input/"), glob
    assert Path(weights).name == "edge_predictor_best_split_0.pth"
    assert Path(config).name == "config_split_0.json"
    # Same convention p3_d1_pilot_f0 already used to resolve this dataset's split_0 on fold 0.
    d1_env = _env(_load(D1_PILOT))
    assert weights == d1_env[WEIGHTS_GLOB_KEY]
    assert config == d1_env[CONFIG_GLOB_KEY]


def test_env_is_identical_to_the_deploy_sibling_except_the_two_globs(spec, sibling):
    ours, theirs = _env(spec), _env(sibling)
    differing = {k for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k)}
    assert differing == GLOB_KEYS, (
        f"env drift beyond the weights substitution: {sorted(differing - GLOB_KEYS)}. The safe-division "
        "env and the 71 fold-0 stems define the champion configuration (FACT-0322); changing them makes "
        "the export incomparable with the EXP-0022 control"
    )
    # The env edit itself lands in the same cell, once, exactly as the sibling's does.
    ours_env_edits = [e for e in spec["edits"] if e["kind"] == "env"]
    theirs_env_edits = [e for e in sibling["edits"] if e["kind"] == "env"]
    assert len(ours_env_edits) == len(theirs_env_edits) == 1
    assert ours_env_edits[0]["cell_match"] == theirs_env_edits[0]["cell_match"]
    assert ours_env_edits[0]["expect"] == theirs_env_edits[0]["expect"]
    # Byte-identical stems, not merely set-equal: the export is paired crop-for-crop.
    assert ours["BIOHUB_LOEO_STEMS"] == theirs["BIOHUB_LOEO_STEMS"]
    assert ours["BIOHUB_LOEO_FOLD"] == "0"
    assert ours["BIOHUB_LOEO_ARM"] == "strict"


def test_everything_but_the_env_edit_matches_the_deploy_sibling(spec, sibling):
    """Same base notebook, same sha, same non-env edits: the pipeline is the champion's."""
    assert spec["base_notebook"] == sibling["base_notebook"]
    assert spec["base_sha256"] == sibling["base_sha256"]
    assert spec["edits"][0]["kind"] == "env"
    assert spec["edits"][1:] == sibling["edits"][1:]
    assert spec["competition_sources"] == sibling["competition_sources"]
    assert spec["machine_shape"] == sibling["machine_shape"]
    assert spec["enable_internet"] is False


def test_no_producer_kernel_plumbing(spec):
    """This spec consumes a DATASET, not a producer kernel. Declaring kernel_sources would mount the
    S5 output and its edge_predictor_best_h1r_s5.pth alongside; declaring consumes_artifacts would
    make the defect gate demand a reciprocal producer that does not exist."""
    assert not spec.get("kernel_sources")
    assert "artifact_role" not in spec
    assert "consumes_artifacts" not in spec


def test_expects_no_submission(spec):
    assert spec["expects_submission"] is False


def test_out_dir_slug_and_code_file_are_distinct_from_every_other_spec(spec):
    for other_path in sorted(SPECS_DIR.glob("*.json")):
        if other_path == SPEC:
            continue
        other = _load(other_path)
        for key in ("name", "slug", "title", "out_dir", "code_file"):
            assert spec[key] != other.get(key), (
                f"{SPEC.name} shares {key}={spec[key]!r} with {other_path.name}; a shared out_dir "
                "overwrites the sibling's kernel-metadata.json and repoints its kernel id"
            )


def test_purpose_states_the_pairing_and_the_pre_registered_reading(spec):
    """The falsifier is written before the score (AGENTS.md section 3). Pin that the spec carries
    the control, the instrument, both weight identities and the reading, so a reader of the built
    notebook's provenance cannot rationalise the outcome afterwards."""
    purpose = spec["purpose"]
    for token in (
        "sweep_pen_off.csv.gz",          # the EXP-0022 control it pairs against
        "FACT-0321",                     # the paired instrument
        "FACT-0263",                     # the fold-0 paired MDE
        "FACT-0322",                     # the champion-configuration provenance
        "d3e89eb361eeadef",              # our split_0
        "12f6881ee3620a83",              # the pack's split_0
        "more than 0.01 below",          # the pre-registered negative reading
    ):
        assert token in purpose, f"purpose block is missing {token!r}"
    prov = spec["provenance"]
    assert prov["weights_ours"]["dataset"] == OOF_DATASET
    assert prov["weights_ours"]["file"] == "edge_predictor_best_split_0.pth"
    assert prov["weights_ours"]["bytes"] == 8357783
    assert prov["weights_pack"]["bytes"] == 8363159


def test_built_notebook_carries_the_split_0_override_and_not_the_s5_one(spec):
    built = ROOT / spec["out_dir"] / spec["code_file"]
    if not built.exists():
        pytest.skip("run kaggle_factory build first")
    nb = _load(built)
    source = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    weights_line = "os.environ[\"BIOHUB_LOEO_WEIGHTS_GLOB\"] = '/kaggle/input/*/edge_predictor_best_split_0.pth'"
    config_line = "os.environ[\"BIOHUB_LOEO_CONFIG_GLOB\"] = '/kaggle/input/*/config_split_0.json'"
    assert weights_line in source
    assert config_line in source
    # Exactly one assignment of each glob in the whole notebook - a second one later in the
    # cell order would silently win over the override.
    assert source.count("os.environ[\"BIOHUB_LOEO_WEIGHTS_GLOB\"] =") == 1
    assert source.count("os.environ[\"BIOHUB_LOEO_CONFIG_GLOB\"] =") == 1
    assert "edge_predictor_best_h1r_s5.pth" not in source
    # loeo_retarget.py's own commentary mentions split_1 (fold 1 needs it), so the guard is on
    # the split_1 FILE names, which must appear nowhere in the built notebook.
    assert "edge_predictor_best_split_1.pth" not in source
    assert "config_split_1.json" not in source
    meta = _load(ROOT / spec["out_dir"] / "kernel-metadata.json")
    assert OOF_DATASET in meta["dataset_sources"]
    assert meta.get("kernel_sources", []) == []
    assert meta["id"].endswith("/" + spec["slug"])
