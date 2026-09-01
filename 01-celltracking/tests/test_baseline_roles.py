"""THE THREE BASELINE ROLES, AND THE DRIFT THAT MADE THEM NECESSARY.

Measured 2026-09-01: no `champion` key existed anywhere in the registry; the only experiment with
`status: deployed` was EXP-0009, many generations stale; and four documents named four different
artifacts. An agent starting from the registry, the README, the directional update or CLAUDE.md
audited a different notebook each time - the exact failure CLAUDE.md records having already made
once, when an agent sent to audit the champion read `p3_harmonic` instead.

One `champion` field would not have fixed it, because the word was doing three incompatible jobs:
the best SCORE, the artifact new work is BUILT ON, and the artifact causal deltas are MEASURED
AGAINST. Those genuinely differ here.

These tests make each role's binding checkable, and fail when prose drifts away from the manifest.
Software contracts only (CLAUDE.md rule 4): nothing here decides which artifact SHOULD hold a role.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "core"))

import baseline_contract as BC  # noqa: E402

ROLES = yaml.safe_load(
    (ROOT / "research" / "00-system" / "registry" / "baseline_roles.yaml").read_text("utf-8"))
REGISTRY = ROOT / "research" / "00-system" / "registry"


def _facts():
    return {f["id"]: f for f in
            yaml.safe_load((REGISTRY / "facts.yaml").read_text("utf-8"))["facts"]}


def _experiments():
    d = yaml.safe_load((REGISTRY / "experiments.yaml").read_text("utf-8"))
    exps = d["experiments"] if isinstance(d, dict) and "experiments" in d else d
    return {e["id"]: e for e in exps}


# ------------------------------------------------------------------------------------------
# 1. NO SCORE VALUE IS DUPLICATED INTO THE MANIFEST
# ------------------------------------------------------------------------------------------
def test_the_role_manifest_states_no_score_value():
    """Roles bind by id. A second copy of a number is a second chance to go stale.

    CLAUDE.md measured the cost of the alternative: the superseded score appeared 244 times
    across 36 files while the live one appeared 31 times across 7.
    """
    text = (REGISTRY / "baseline_roles.yaml").read_text("utf-8")
    # a leaderboard score in this project is 0.8xx-0.9xx; ids and hashes are not
    stripped = re.sub(r"\b[0-9a-f]{16,64}\b", "", text)
    offenders = re.findall(r"(?<![\w.])0\.9[0-9]{2}(?![\w])", stripped)
    assert not offenders, (
        f"the role manifest contains score-shaped literals {sorted(set(offenders))}. Cite the "
        f"FACT/EXP id instead - the registry is the only source of truth for numbers."
    )


# ------------------------------------------------------------------------------------------
# 2. EVERY ROLE BINDING RESOLVES
# ------------------------------------------------------------------------------------------
def test_leaderboard_champion_binds_a_real_scored_submission():
    lc = ROLES["leaderboard_champion"]
    exps, facts = _experiments(), _facts()
    e = exps[lc["experiment"]]
    assert e.get("status") == "scored", f"{lc['experiment']} is {e.get('status')}, not scored"
    assert e.get("submission") == lc["submission"], "submission id disagrees with the experiment"
    assert lc["score_fact"] in facts, "the score fact does not exist"
    assert facts[lc["score_fact"]].get("validity") != "INVALID", (
        "the leaderboard champion cites an INVALID fact")


def test_the_champion_is_the_highest_scored_submission_by_the_stated_rule():
    """The selection rule is applied here, so the manifest cannot quietly disagree with it."""
    exps, facts = _experiments(), _facts()
    scored = [e for e in exps.values()
              if e.get("status") == "scored" and e.get("lb") is not None]
    best = max(e["lb"] for e in scored)
    # submission ids are int in some experiments and str in others, so the tie-break sorts on
    # a normalised key rather than crashing on the comparison.
    top = sorted([e for e in scored if e["lb"] == best],
                 key=lambda e: int(str(e["submission"]).strip()))
    assert ROLES["leaderboard_champion"]["experiment"] == top[0]["id"], (
        f"the manifest names {ROLES['leaderboard_champion']['experiment']} but the stated rule "
        f"(highest score; ties to the EARLIER submission) selects {top[0]['id']}"
    )
    if len(top) > 1:
        assert ROLES["leaderboard_champion"]["tied_with"]["experiment"] == top[1]["id"], (
            "there is a tie and the manifest does not record the tied experiment")


def test_operational_base_is_read_from_the_spec_graph_not_chosen():
    """A treatment spec declares it as `base`; the manifest must agree with the spec."""
    ob = ROLES["operational_base"]
    declaring = []
    for p in sorted((ROOT / "scripts" / "kaggle_specs").glob("*.json")):
        try:
            spec = json.loads(p.read_text("utf-8"))
        except json.JSONDecodeError:
            continue
        declared = spec.get("base_notebook") or spec.get("base") or ""
        if str(declared).replace("\\", "/") == ob["notebook"]:
            declaring.append((p.name, spec.get("base_sha256")))
    assert declaring, (
        f"no spec declares {ob['notebook']} as its base, so it is not the operational base - "
        f"the manifest was chosen rather than read")
    for name, sha in declaring:
        assert sha == ob["notebook_sha256"], (
            f"{name} pins base_sha256 {sha}, the manifest declares {ob['notebook_sha256']}")


def test_the_operational_base_notebook_hash_matches_the_manifest():
    """A receipt/hash that does not match its declared role is a refusal, not a note."""
    ob = ROLES["operational_base"]
    nb = ROOT / ob["notebook"]
    assert nb.is_file(), f"the operational base notebook is absent: {nb}"
    digest = hashlib.sha256(nb.read_bytes()).hexdigest()
    assert digest == ob["notebook_sha256"], (
        f"{nb.name} hashes to {digest}, the manifest declares {ob['notebook_sha256']}. If this "
        f"fires on a non-Windows checkout, read `digest_convention_warning` in "
        f"baseline_roles.yaml: the recorded digest is the CRLF working-tree form."
    )


def test_the_release_receipt_named_by_the_champion_exists():
    p = ROOT / ROLES["leaderboard_champion"]["release_receipt"]
    assert p.is_file(), f"the champion names a release receipt that is absent: {p}"


# ------------------------------------------------------------------------------------------
# 3. PROSE MAY NOT DRIFT AWAY FROM THE MANIFEST
# ------------------------------------------------------------------------------------------
def test_claude_md_does_not_name_a_base_inconsistent_with_the_manifest():
    """CLAUDE.md is the contract agents read first; it must not point at a different notebook."""
    text = (ROOT / "CLAUDE.md").read_text("utf-8")
    ob_dir = Path(ROLES["operational_base"]["notebook"]).parent.name          # kaggle_p35_...
    champ_dir = Path(ROLES["leaderboard_champion"]["notebook"]).parent.name
    assert ob_dir in text, (
        f"CLAUDE.md does not mention the operational base {ob_dir}. An agent reading the contract "
        f"would not know which notebook to build on.")
    # any OTHER notebook directory CLAUDE.md calls the champion is drift
    # Take the FIRST notebook directory after each occurrence of "CHAMPION", which is how a
    # reader parses the sentence. A window-based match is wrong here: CLAUDE.md legitimately
    # names p3_harmonic in the SAME sentence in order to say it is NOT the champion, and the
    # first draft of this test flagged that correct sentence as drift.
    stray = []
    for m in re.finditer(r"CHAMPION", text, re.I):
        nxt = re.search(r"notebooks/(kaggle_[a-z0-9_]+)/", text[m.end():m.end() + 200])
        if nxt and nxt.group(1) not in {ob_dir, champ_dir}:
            stray.append(nxt.group(1))
    assert not stray, (
        f"CLAUDE.md names {sorted(set(stray))} as champion, but the manifest binds {champ_dir}")


def test_scientific_control_states_why_it_is_comparable():
    sc = ROLES["scientific_control"]
    if sc.get("differs_from_operational_base"):
        assert sc.get("comparable_because", "").strip(), (
            "the scientific control differs from the operational base and does not say why the "
            "comparison is still valid - which is how a treatment gets audited against the wrong "
            "control")
        assert sc.get("established_by"), "the control names no establishing fact"


# ------------------------------------------------------------------------------------------
# 4. THE GENERATED CONTRACT IS IN SYNC AND IS ACTUALLY CONSUMED
# ------------------------------------------------------------------------------------------
def test_the_generated_contract_has_no_drift():
    assert BC.main(["--check"]) == 0, (
        "operational_base_contract.json disagrees with the built notebook. Either the notebook "
        "changed and the contract was not regenerated, or the contract was hand-edited.")


def test_the_contract_reports_the_operational_bases_own_geometry():
    geom = BC.safe_division()
    assert geom["BIOHUB_SAFE_DIV_MAX_UM"] != 4.66, (
        "the contract reports p3_harmonic's safe-division radius. That triple was hard-coded in "
        "three committed places while nothing encoded the operational base's.")


@pytest.mark.parametrize("path,token", [
    ("scripts/win_bet/constant_audit.py", "baseline_contract"),
    ("scripts/d1/gt_division_gates.py", "baseline_contract"),
    ("tests/test_lineage_degree_invariants.py", "baseline_contract"),
])
def test_the_formerly_stale_sites_now_consume_the_contract(path, token):
    import ast
    src = (ROOT / path).read_text("utf-8")
    assert token in src, f"{path} does not consume the generated contract"
    # The check is on CODE LITERALS, not on the text. These files explain the repair in prose
    # and must be able to name the stale value they replaced; the first draft of this test read
    # the raw text and flagged its own explanation.
    literals = [n.value for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert 4.66 not in literals, (
        f"{path} still hard-codes p3_harmonic's safe-division radius as a live literal")
