"""Train/test hygiene for LOEO specs — the guard that would have prevented EXP-0019.

WHAT HAPPENED (2026-08-26)
`scripts/kaggle_edits/loeo_retarget.py:15-18` states it plainly: the support pack ships ONLY
``weights/unet_transformer/split_0``, trained on 6bba and holding out 44b6. It is therefore
**LOEO-CLEAN ON FOLD 0 AND LEAKY ON FOLD 1**. Every fold-1 LOEO spec must override it with our
out-of-fold ``split_1`` weights.

`p19_relink_sweep_f1` did not. It was built by copying `p9_coupled_division` — a *test*
submission spec, where the pack weights are correct because test data is not in training — and
bolting LOEO retargeting onto it. The weights override was never added, so
`loeo_retarget.py` silently fell back to the pack default and the run scored the 6bba embryo
with a model trained on 6bba.

Nothing caught it. The build passed, the defect gate reported "0 applicable rules", the
in-run control passed (it reproduced its own leaky primary export exactly), and every summary
statistic we record looked outstanding — node recall 0.9818 against an honest 0.8547, divJ
0.0743 against an honest ~0.002-0.005. The inflated numbers then propagated into the registry
and into eight work packets before an audit caught them.

The failure is invisible in results *by construction*: a leaked run looks BETTER, not broken.
So it has to be caught at build time, from the spec, which is what this does.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPECS = sorted((REPO / "scripts" / "kaggle_specs").glob("*.json"))

# The pack's only shipped weights. Trained on 6bba => clean on fold 0, LEAKY on fold 1.
PACK_WEIGHTS = "split_0"
OOF_WEIGHTS_KEY = "BIOHUB_LOEO_WEIGHTS_GLOB"


def spec_env(spec: dict) -> dict:
    env: dict = {}
    for edit in spec.get("edits", []):
        if edit.get("kind") == "env":
            env.update(edit.get("vars", {}))
    return env


# --- THE INHERITED-SETTING HOLE, AND WHY THE RESOLVER BELOW EXISTS ----------------------------
# This guard originally read the fold from the SPEC's own env edits only. But a spec is built on
# top of an already-built notebook (`base_notebook`), and `kaggle_factory.apply_edit` APPENDS the
# spec's env block after that notebook's existing text (kaggle_factory.py:499-508). Every
# `os.environ[...]` the base notebook already carries therefore stays in force unless the spec
# names the same key. A spec that inherits `BIOHUB_LOEO_FOLD` from its base is a fold-1 LOEO run
# that this guard used to SKIP - so the run it exists to protect was the one shape it could not
# see. Measured 2026-08-30 by the PKT-0036 audit: 12 specs checked, 54 skipped; the effective
# resolver raises that to 14 checked, and both of the newly visible specs really are fold 1
# (p29_p28_detpeak_export_f1, p34_acquisition_f1). The fold-1 feature-cache run is built exactly
# that way.
ENV_ASSIGN = re.compile(
    r"""os\.environ\[\s*["'](BIOHUB_[A-Z0-9_]+)["']\s*\]\s*=\s*r?["'](.*?)["']\s*$""",
    re.M,
)


def notebook_env(path: Path) -> dict:
    """Every BIOHUB_* environment assignment baked into a built notebook. Last write wins."""
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
    source = "\n".join("".join(c.get("source", [])) for c in cells)
    return dict(ENV_ASSIGN.findall(source))


def effective_env(spec: dict) -> dict:
    """Base-notebook environment, overridden by the spec's own env edits.

    FAIL CLOSED. A spec that names a base notebook we cannot read is not "not a LOEO spec" - it
    is a spec whose fold is UNKNOWN, and returning {} there would restore exactly the silent skip
    this resolver removes.
    """
    base = spec.get("base_notebook")
    inherited: dict = {}
    if base:
        path = REPO / base
        if not path.is_file():
            raise AssertionError(
                f"base_notebook {base!r} is not on disk, so this spec's effective LOEO fold "
                "cannot be resolved and its train/test hygiene cannot be checked. Build the "
                "base notebook or correct the path - do not let the guard skip it."
            )
        inherited = notebook_env(path)
    return {**inherited, **spec_env(spec)}


def loeo_fold(spec: dict):
    """The EFFECTIVE LOEO fold this spec evaluates, or None if it is not a LOEO run."""
    fold = effective_env(spec).get("BIOHUB_LOEO_FOLD")
    return None if fold is None else str(fold).strip()


def test_the_pack_weights_are_documented_as_fold1_leaky():
    """Anchor the premise in the source, so this guard cannot drift from reality."""
    text = (REPO / "scripts" / "kaggle_edits" / "loeo_retarget.py").read_text(encoding="utf-8")
    assert "LOEO-CLEAN on fold 0 and LEAKY on fold 1" in text, (
        "loeo_retarget.py no longer documents the fold-1 leak; re-verify which weights the "
        "support pack ships before trusting this test"
    )


def assert_fold1_hygiene(stem: str, spec: dict) -> None:
    """The fold-1 leak contract. Factored out so the mutation test below exercises the REAL
    assertions rather than a paraphrase of them."""
    glob = effective_env(spec).get(OOF_WEIGHTS_KEY, "")
    assert glob, (
        f"{stem} evaluates LOEO fold 1 but sets no {OOF_WEIGHTS_KEY}. "
        "loeo_retarget.py falls back to the pack's split_0 weights, which were TRAINED ON "
        "6bba - the very embryo fold 1 holds out. This is the EXP-0019 defect."
    )
    assert PACK_WEIGHTS not in glob, (
        f"{stem} points {OOF_WEIGHTS_KEY} at {glob!r}, which still names "
        f"{PACK_WEIGHTS!r} - the leaky pack weights."
    )
    assert any("oof" in str(d).lower() for d in spec.get("datasets", [])), (
        f"{stem} sets {OOF_WEIGHTS_KEY} but attaches no out-of-fold weights "
        "dataset, so the glob will match nothing and it will fall back to the pack."
    )


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: p.stem)
def test_fold1_loeo_specs_override_the_leaky_pack_weights(spec_path: Path):
    """A fold-1 LOEO spec that does not override the pack weights trains on its own
    evaluation embryo. The run will look excellent and mean nothing."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if loeo_fold(spec) != "1":
        pytest.skip("not a fold-1 LOEO spec")
    assert_fold1_hygiene(spec_path.stem, spec)


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: p.stem)
def test_fold0_loeo_specs_are_clean_by_construction(spec_path: Path):
    """Fold 0 holds out 44b6, which the pack weights never saw, so the pack default is
    legitimate there. Assert only that a fold-0 spec does not accidentally load split_1."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if loeo_fold(spec) != "0":
        pytest.skip("not a fold-0 LOEO spec")
    glob = effective_env(spec).get(OOF_WEIGHTS_KEY, "")
    assert "split_1" not in glob, (
        f"{spec_path.stem} evaluates fold 0 but loads split_1 weights"
    )


def test_evidence_manifests_declare_their_weights():
    """A fetched LOEO export must record which weights produced it. Two exports of the same
    fold with different train/test hygiene are otherwise indistinguishable - which is exactly
    how the leak survived review."""
    manifests = list((REPO / "_evidence").rglob("loeo_manifest.json"))
    if not manifests:
        pytest.skip("no LOEO manifests fetched locally")
    missing = [
        str(m.relative_to(REPO)) for m in manifests
        if "weights" not in (json.loads(m.read_text(encoding="utf-8")) or {})
    ]
    assert not missing, f"LOEO manifests without a weights field: {missing}"


def test_known_leaky_export_is_labelled_in_the_registry():
    """The EXP-0019 output must stay marked as leak-contaminated so no future session
    mistakes it for an authoritative measurement."""
    facts = (REPO / "research/00-system/registry/facts.yaml").read_text(encoding="utf-8")
    assert re.search(r"leak", facts, re.I), (
        "the registry no longer mentions the EXP-0019 leak; if it was genuinely resolved, "
        "delete this test deliberately rather than letting it lapse"
    )


# --- upstream guard: a LOEO arm must be known to loeo_retarget BEFORE a kernel runs -----------
# The 'champion' arm (2026-08-28) errored 30 s into a Kaggle GPU run with "unknown LOEO arm"
# because the runtime whitelist in loeo_retarget.py had not been updated. That is a build-time
# contract: the arm a spec requests must be one the injected code accepts. This test reads the
# whitelist from the source of truth and validates every LOEO spec against it, so a typo'd or
# newly-added arm fails at pytest, not on a wasted kernel.
def _known_loeo_arms() -> set[str]:
    src = (REPO / "scripts" / "kaggle_edits" / "loeo_retarget.py").read_text(encoding="utf-8")
    m = re.search(r"LOEO_ARM not in \{([^}]*)\}", src)
    assert m, "could not find the LOEO arm whitelist in loeo_retarget.py"
    return {tok.strip().strip("'\"") for tok in m.group(1).split(",") if tok.strip()}


def test_the_arm_whitelist_is_parseable_and_nonempty():
    arms = _known_loeo_arms()
    assert "strict" in arms and "champion" in arms and len(arms) >= 3


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: p.stem)
def test_every_loeo_spec_requests_a_known_arm(spec_path: Path):
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    env = spec_env(spec)
    arm = env.get("BIOHUB_LOEO_ARM")
    if arm is None:
        pytest.skip("not a LOEO spec / no arm requested")
    known = _known_loeo_arms()
    assert arm in known, (
        f"{spec_path.stem} requests BIOHUB_LOEO_ARM={arm!r}, which loeo_retarget.py does not accept "
        f"(known arms: {sorted(known)}). This would raise 'unknown LOEO arm' at runtime on Kaggle - "
        "add the arm to the whitelist AND its handling branch before pushing."
    )


# --- MUTATION PROOFS FOR THE EFFECTIVE-FOLD RESOLVER -----------------------------------------
# A checker that has not been shown to REJECT is not a checker. These manufacture the exact
# defect shape the resolver exists to catch - a spec that inherits fold 1 from its base notebook
# and sets no out-of-fold weights - and assert that the guard fires on it. Without the resolver
# the same spec is silently SKIPPED, which is asserted here too so the two behaviours cannot be
# confused by a future reader.


def _fake_notebook(path: Path, env: dict) -> None:
    body = "".join(f'os.environ["{k}"] = {v!r}\n' for k, v in env.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": [body]}],
                    "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
        encoding="utf-8",
    )


def _spec_inheriting_fold1(tmp_path: Path, monkeypatch, *, spec_env_vars: dict | None = None,
                           datasets: list | None = None) -> dict:
    """A spec whose OWN env never mentions the fold - it comes from the base notebook."""
    nb = tmp_path / "notebooks" / "base" / "base.ipynb"
    _fake_notebook(nb, {"BIOHUB_LOEO_FOLD": "1", "BIOHUB_LOEO_ARM": "champion"})
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    return {
        "name": "inherited_fold1",
        "base_notebook": "notebooks/base/base.ipynb",
        "datasets": datasets if datasets is not None else ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
        "edits": [{"kind": "env", "vars": spec_env_vars or {"BIOHUB_AFP_CROPS": "2"}}],
    }


def test_the_spec_env_alone_cannot_see_an_inherited_fold(tmp_path, monkeypatch):
    """THE HOLE, stated as an executable fact: the old spec-env-only read returns None here."""
    spec = _spec_inheriting_fold1(tmp_path, monkeypatch)
    assert spec_env(spec).get("BIOHUB_LOEO_FOLD") is None
    assert loeo_fold(spec) == "1", (
        "the effective resolver must recover the fold from the base notebook, or the fold-1 "
        "guard skips exactly the specs that are built by inheritance"
    )


def test_the_leak_guard_fires_on_a_spec_that_inherits_fold1_and_sets_no_oof_weights(
        tmp_path, monkeypatch):
    """The EXP-0019 defect, reached through the inheritance route rather than the spec env."""
    spec = _spec_inheriting_fold1(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match="sets no BIOHUB_LOEO_WEIGHTS_GLOB"):
        assert_fold1_hygiene("inherited_fold1", spec)


def test_the_leak_guard_fires_when_the_inherited_glob_is_the_leaky_pack(tmp_path, monkeypatch):
    spec = _spec_inheriting_fold1(
        tmp_path, monkeypatch,
        spec_env_vars={OOF_WEIGHTS_KEY: "/kaggle/input/*/split_0/edge_predictor_best.pth"},
        datasets=["aryaarun07/biohub-oof-weights"],
    )
    with pytest.raises(AssertionError, match="leaky pack weights"):
        assert_fold1_hygiene("inherited_fold1", spec)


def test_the_leak_guard_fires_when_the_oof_dataset_is_not_attached(tmp_path, monkeypatch):
    spec = _spec_inheriting_fold1(
        tmp_path, monkeypatch,
        spec_env_vars={OOF_WEIGHTS_KEY: "/kaggle/input/*/edge_predictor_best_split_1.pth"},
        datasets=["pilkwang/biohub-tracking-support-pack-50ep-v1"],
    )
    with pytest.raises(AssertionError, match="attaches no out-of-fold weights"):
        assert_fold1_hygiene("inherited_fold1", spec)


def test_a_correctly_built_inherited_fold1_spec_passes(tmp_path, monkeypatch):
    """The guard must not simply reject everything - the accept case is part of the proof."""
    spec = _spec_inheriting_fold1(
        tmp_path, monkeypatch,
        spec_env_vars={OOF_WEIGHTS_KEY: "/kaggle/input/*/edge_predictor_best_split_1.pth"},
        datasets=["aryaarun07/biohub-oof-weights"],
    )
    assert_fold1_hygiene("inherited_fold1", spec)


def test_an_unreadable_base_notebook_fails_closed(tmp_path, monkeypatch):
    """A fold we cannot resolve must be an ERROR, not a skip. Returning {} here would restore
    the silent-skip hole through a different door."""
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    spec = {"base_notebook": "notebooks/nope/missing.ipynb", "edits": []}
    with pytest.raises(AssertionError, match="not on disk"):
        loeo_fold(spec)


def test_the_resolver_actually_checks_the_specs_it_claims_to(tmp_path):
    """POSITIVE HEARTBEAT. A guard whose skip rate silently rises to 100% looks identical to a
    guard that passed. Pin the two specs that are ONLY visible through inheritance, by name."""
    inherited = set()
    for spec_path in SPECS:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec_env(spec).get("BIOHUB_LOEO_FOLD") is None and loeo_fold(spec) is not None:
            inherited.add(spec_path.stem)
    assert {"p29_p28_detpeak_export_f1", "p34_acquisition_f1"} <= inherited, (
        "these fold-1 specs inherit their fold from a base notebook and are the reason this "
        f"resolver exists; the resolver currently recovers {sorted(inherited)}"
    )
