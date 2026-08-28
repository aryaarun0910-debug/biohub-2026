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


def loeo_fold(spec: dict):
    """The LOEO fold this spec evaluates, or None if it is not a LOEO run."""
    fold = spec_env(spec).get("BIOHUB_LOEO_FOLD")
    return None if fold is None else str(fold).strip()


def test_the_pack_weights_are_documented_as_fold1_leaky():
    """Anchor the premise in the source, so this guard cannot drift from reality."""
    text = (REPO / "scripts" / "kaggle_edits" / "loeo_retarget.py").read_text(encoding="utf-8")
    assert "LOEO-CLEAN on fold 0 and LEAKY on fold 1" in text, (
        "loeo_retarget.py no longer documents the fold-1 leak; re-verify which weights the "
        "support pack ships before trusting this test"
    )


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: p.stem)
def test_fold1_loeo_specs_override_the_leaky_pack_weights(spec_path: Path):
    """A fold-1 LOEO spec that does not override the pack weights trains on its own
    evaluation embryo. The run will look excellent and mean nothing."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if loeo_fold(spec) != "1":
        pytest.skip("not a fold-1 LOEO spec")

    env = spec_env(spec)
    glob = env.get(OOF_WEIGHTS_KEY, "")
    assert glob, (
        f"{spec_path.stem} evaluates LOEO fold 1 but sets no {OOF_WEIGHTS_KEY}. "
        "loeo_retarget.py falls back to the pack's split_0 weights, which were TRAINED ON "
        "6bba - the very embryo fold 1 holds out. This is the EXP-0019 defect."
    )
    assert PACK_WEIGHTS not in glob, (
        f"{spec_path.stem} points {OOF_WEIGHTS_KEY} at {glob!r}, which still names "
        f"{PACK_WEIGHTS!r} - the leaky pack weights."
    )
    assert any("oof" in str(d).lower() for d in spec.get("datasets", [])), (
        f"{spec_path.stem} sets {OOF_WEIGHTS_KEY} but attaches no out-of-fold weights "
        "dataset, so the glob will match nothing and it will fall back to the pack."
    )


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: p.stem)
def test_fold0_loeo_specs_are_clean_by_construction(spec_path: Path):
    """Fold 0 holds out 44b6, which the pack weights never saw, so the pack default is
    legitimate there. Assert only that a fold-0 spec does not accidentally load split_1."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if loeo_fold(spec) != "0":
        pytest.skip("not a fold-0 LOEO spec")
    glob = spec_env(spec).get(OOF_WEIGHTS_KEY, "")
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
