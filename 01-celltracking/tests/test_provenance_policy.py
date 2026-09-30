"""ONE POLICY, TWO GUARDS, AND THEY MUST NOT BE ABLE TO DISAGREE.

``FACT-0418``: ``audit_feature_cache.PKT0029_REQUIRED_TRUNK_ROLES`` was ``('official',
'stabledet')``, so ``audit-pair --require-roles`` would have REFUSED the honest pair and ACCEPTED
the pair with no fold-legitimate arm on either fold. ``FACT-0431``: the newer
``gpu_protection_contract`` DATA-3 clause answered that same question the other way. Two committed
instruments returning opposite verdicts is worse than one wrong guard, because whichever runs last
looks authoritative.

The repair is NOT "edit one constant until they agree" - that buys agreement by coincidence and
hides the next divergence. So this file tests the property, not the coincidence:

  * an AGREEMENT MATRIX over (fold x pair): the two guards must return the same verdict on every
    cell, and the cells are chosen so a naive constant would split them;
  * the FOLD SWAP, in BOTH directions, rejected by BOTH guards - one guard rejecting is not
    enough, because the failure being repaired is precisely that two disagreed;
  * ACCEPT CONTROLS on both folds, because a guard that refuses everything proves nothing;
  * a SINGLE-SOURCE proof: mutate the policy and BOTH guards move. A guard that did not move
    would still be reading a table of its own.

Guard 1 is exercised END TO END on real synthetic caches through ``audit_dual_trunk``; guard 2
through ``section_data``'s DATA-2/DATA-3/DATA-4 clauses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import audit_feature_cache as A        # noqa: E402
import gpu_protection_contract as G    # noqa: E402
import provenance_policy as PP         # noqa: E402

FOLD_CROP = {"0": "44b6_aaaaaaaa", "1": "6bba_bbbbbbbb"}
# A glob that satisfies each fold's pre-existing weights-glob check, so the only thing left for
# these tests to move is the ROLE. Without this the older glob check would fire first and the
# policy would never be reached - a rejection for the wrong reason is not evidence.
FOLD_GLOB = {"0": "/kaggle/input/*/split_0/edge_predictor_best.pth",
             "1": "/kaggle/input/*/edge_predictor_best_split_1.pth"}


# =============================================================================================
# guard 1: real caches, real manifests
# =============================================================================================
@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    """Two caches per fold over ONE node set, with different features - a well-formed pair."""
    root = tmp_path_factory.mktemp("policy")
    nb = root / "notebook.ipynb"
    nb.write_text('{"cells": []}', encoding="utf-8")
    trunks = {}
    dirs = {}
    for fold in ("0", "1"):
        for seed in (1, 7):
            d = root / f"f{fold}_seed{seed}"
            d.mkdir()
            A._synth_cache(d / f"{FOLD_CROP[fold]}.npz", crop=FOLD_CROP[fold], trunk_seed=seed)
            dirs[(fold, seed)] = d
            t = root / f"trunk_{fold}_{seed}.pth"
            t.write_bytes(bytes([seed]) * 4096 + f"trunk-{fold}-{seed}".encode())
            trunks[(fold, seed)] = t
    return {"root": root, "nb": nb, "dirs": dirs, "trunks": trunks}


def manifest_for(caches, fold, seed, role, name=None):
    d = caches["dirs"][(fold, seed)]
    p = d / (name or f"manifest_{role}.json")
    args = A._bind_args(d, caches["trunks"][(fold, seed)], fold=fold, role=role,
                        weights_glob=FOLD_GLOB[fold], notebook=caches["nb"])
    p.write_text(json.dumps(A.build_manifest(args)), encoding="utf-8")
    return p


def guard1_pair(caches, fold, roles):
    """audit_feature_cache's verdict on this pair, as (accepted: bool, reason: str)."""
    man_a = manifest_for(caches, fold, 1, roles[0])
    man_b = manifest_for(caches, fold, 7, roles[1])
    try:
        A.audit_dual_trunk(caches["dirs"][(fold, 1)], caches["dirs"][(fold, 7)],
                           man_a, man_b, require_roles=True)
        return True, ""
    except A.Reject as err:
        return False, str(err)


def guard2_pair(fold, roles, *, claim_role=None, sha="0" * 16):
    """gpu_protection_contract's verdict on the same pair, from DATA-2 and DATA-3."""
    spec = {"edits": [{"vars": {"BIOHUB_LOEO_FOLD": fold}}]}
    trunk = {"role": claim_role or roles[0], "fold_legitimate": True,
             "dual_trunk_pair": list(roles), "contamination": "none",
             "checkpoint_sha256": sha}
    clauses = {c["id"]: c for c in G.section_data(spec, trunk)}
    reasons = []
    for cid in ("DATA-2", "DATA-3"):
        reasons += clauses[cid]["evidence"].get("refusals", [])
    return clauses["DATA-3"]["passed"], "; ".join(reasons), clauses


# =============================================================================================
# 1. THE AGREEMENT MATRIX - the property the repair actually has to have
# =============================================================================================
MATRIX = [
    # (fold, pair, expected_accept, why)
    ("0", ("pack_split0", "stabledet"), True,
     "the honest fold-0 pair: a legitimate claim arm plus the declared comparison arm"),
    ("1", ("oof_split1", "stabledet"), True,
     "the honest fold-1 pair - which the RETIRED constant would have refused (FACT-0418)"),
    ("0", ("official", "stabledet"), False,
     "the FACT-0418 pairing - which the RETIRED constant would have ACCEPTED"),
    ("1", ("official", "stabledet"), False,
     "the same pairing on the other fold; it has no legitimate arm on EITHER"),
    ("0", ("oof_split1", "stabledet"), False,
     "FOLD SWAP: the fold-1 claim arm relabelled onto fold 0"),
    ("1", ("pack_split0", "stabledet"), False,
     "FOLD SWAP: the fold-0 claim arm relabelled onto fold 1 - the EXP-0019 defect class"),
    ("1", ("oof_split1", "official"), False,
     "a split_0 checkpoint inside the fold-1 session, even as the comparison arm"),
    ("0", ("oof_split0", "stabledet"), True,
     "fold 0 has two legitimate trunks; our own OOF split_0 is clean there too"),
]


@pytest.mark.parametrize("fold,roles,expect_accept,why",
                         MATRIX, ids=[f"{f}-{'+'.join(r)}" for f, r, _, _ in MATRIX])
def test_both_guards_return_the_same_verdict(caches, fold, roles, expect_accept, why):
    g1_ok, g1_why = guard1_pair(caches, fold, roles)
    g2_ok, g2_why, _ = guard2_pair(fold, roles)
    assert g1_ok == g2_ok, (
        f"THE GUARDS DISAGREE on fold {fold} {roles} ({why}): "
        f"audit_feature_cache={'accept' if g1_ok else 'reject'} ({g1_why}) vs "
        f"gpu_protection_contract={'accept' if g2_ok else 'reject'} ({g2_why}). "
        "This is the FACT-0431 failure returning."
    )
    assert g1_ok is expect_accept, f"{why}: guard 1 said {g1_ok}, expected {expect_accept} ({g1_why})"
    assert g2_ok is expect_accept, f"{why}: guard 2 said {g2_ok}, expected {expect_accept} ({g2_why})"


# =============================================================================================
# 2. THE FOLD SWAP, BOTH DIRECTIONS, BOTH GUARDS - stated on its own, not only in the matrix
# =============================================================================================
def test_fold_swap_split1_onto_fold0_is_rejected_by_both_guards(caches):
    g1_ok, g1_why = guard1_pair(caches, "0", ("oof_split1", "stabledet"))
    g2_ok, g2_why, clauses = guard2_pair("0", ("oof_split1", "stabledet"))
    assert g1_ok is False and "leaky_checkpoint_in_the_fold" in g1_why
    assert g2_ok is False and "leaky_checkpoint_in_the_fold" in g2_why
    assert clauses["DATA-2"]["passed"] is False, "the claim arm itself is illegitimate here"


def test_fold_swap_split0_onto_fold1_is_rejected_by_both_guards(caches):
    """split_0 was trained on 6bba, the embryo fold 1 holds out - the EXP-0019 defect class."""
    g1_ok, g1_why = guard1_pair(caches, "1", ("pack_split0", "stabledet"))
    g2_ok, g2_why, clauses = guard2_pair("1", ("pack_split0", "stabledet"))
    assert g1_ok is False and "leaky_checkpoint_in_the_fold" in g1_why
    assert g2_ok is False and "leaky_checkpoint_in_the_fold" in g2_why
    assert clauses["DATA-2"]["passed"] is False


def test_the_swap_survives_the_specs_own_declaration_of_legitimacy(caches):
    """A spec asserting fold_legitimate true does not make a swapped fold legitimate."""
    _, _, clauses = guard2_pair("1", ("pack_split0", "stabledet"), claim_role="pack_split0")
    assert clauses["DATA-2"]["evidence"]["fold_legitimate_declared"] is True
    assert clauses["DATA-2"]["passed"] is False


# =============================================================================================
# 3. NEITHER GUARD KEEPS A TABLE OF ITS OWN
# =============================================================================================
def test_the_two_local_tables_are_retired_not_corrected():
    assert not hasattr(A, "PKT0029_REQUIRED_TRUNK_ROLES"), (
        "FACT-0418's constant is still here. Correcting it would have produced agreement by "
        "coincidence; it has to be gone")
    assert not hasattr(G, "INVALID_PAIR")
    assert not hasattr(G, "LEGITIMATE_TRUNKS")
    assert A.TRUNK_ROLES == set(PP.TRUNK_ROLES)


def test_mutating_the_policy_moves_both_guards(caches, monkeypatch):
    """THE SINGLE-SOURCE PROOF. Make stabledet a legitimate fold-1 claim arm in the POLICY only.

    If either guard kept its own table, its verdict would not move - and that is the exact defect
    being repaired, so it is worth a test rather than an inspection of the imports.
    """
    before1, _ = guard1_pair(caches, "1", ("stabledet", "official"))
    before2, _, _ = guard2_pair("1", ("stabledet", "official"), claim_role="stabledet")
    assert before1 is False and before2 is False

    patched = {k: dict(v) for k, v in PP.ROLES.items()}
    patched["stabledet"]["claim_folds"] = {"1"}
    patched["official"]["permitted_folds"] = {"0", "1"}
    monkeypatch.setattr(PP, "ROLES", patched)

    after1, why1 = guard1_pair(caches, "1", ("stabledet", "official"))
    after2, why2, _ = guard2_pair("1", ("stabledet", "official"), claim_role="stabledet")
    assert after1 is True, f"audit_feature_cache did not follow the policy: {why1}"
    assert after2 is True, f"gpu_protection_contract did not follow the policy: {why2}"


# =============================================================================================
# 4. EVERY ROLE BOUND BY SHA, FOLD, EMBRYO AND PROVENANCE
# =============================================================================================
def test_a_role_whose_bytes_are_another_checkpoint_is_refused_by_name():
    """FACT-0418's retraction, as a check: the file recorded 'official' IS our OOF split_0."""
    oof0 = "d3e89eb361eeadef06d18159d834594776174a9f8cabd81f65271abbb42a492f"
    refusals = PP.role_binding_refusals("official", oof0)
    assert any("role_contradicts_its_bytes" in r for r in refusals), refusals
    assert any("oof_split0" in r for r in refusals)
    # the 16-hex prefix the registry writes must resolve identically
    assert PP.known_identity(oof0[:16])[0] == "oof_split0"
    # and the honest declaration of the same bytes passes
    assert PP.role_binding_refusals("oof_split0", oof0) == []


def test_guard2_data4_refuses_a_role_its_digest_contradicts():
    _, _, clauses = guard2_pair("1", ("oof_split1", "stabledet"), claim_role="oof_split1",
                                sha="d3e89eb361eeadef")
    assert clauses["DATA-4"]["passed"] is False
    assert any("role_contradicts_its_bytes" in r
               for r in clauses["DATA-4"]["evidence"]["refusals"])


def test_the_fold1_legitimate_trunk_digest_is_the_one_on_disk():
    """The fold-1 claim arm is bound to bytes, not to a filename (FACT-0418's located trunk)."""
    split1 = "2e4ebf616b3d4fb53881eadf42c7972e04978c28f74f229c5cb30bee835063de"
    assert PP.known_identity(split1)[0] == "oof_split1"
    assert PP.role_binding_refusals("oof_split1", split1) == []
    on_disk = ROOT / "artifacts" / "kaggle" / "weights_dataset" / "edge_predictor_best_split_1.pth"
    if on_disk.is_file():          # artifacts/ is gitignored, so absence is not a failure here
        import hashlib
        assert hashlib.sha256(on_disk.read_bytes()).hexdigest() == split1


def test_the_embryo_must_match_the_fold():
    assert PP.role_binding_refusals("oof_split1", None, fold="1", embryo="6bba") == []
    bad = PP.role_binding_refusals("oof_split1", None, fold="1", embryo="44b6")
    assert any("embryo_does_not_match_the_fold" in r for r in bad)


def test_a_placeholder_provenance_is_refused():
    bad = PP.role_binding_refusals("oof_split1", None, provenance="tbd")
    assert any("trunk_provenance_is_a_placeholder" in r for r in bad)


def test_an_unresolvable_fold_fails_closed():
    assert any("fold_unresolved" in r for r in PP.pair_refusals("2", ("oof_split1", "stabledet")))
    assert any("fold_unresolved" in r for r in PP.claim_arm_refusals(None, "oof_split1"))


def test_an_unknown_role_fails_closed():
    assert any("trunk_role_unknown" in r for r in PP.claim_arm_refusals("1", "whatever"))
    assert any("trunk_role_unknown" in r for r in PP.role_binding_refusals("whatever"))


def test_the_policy_states_the_fold_asymmetry_it_is_there_to_enforce():
    assert PP.legitimate_claim_roles("0") == {"pack_split0", "oof_split0"}
    assert PP.legitimate_claim_roles("1") == {"oof_split1"}
    # no split_0 checkpoint may appear on fold 1 AT ALL, in either arm
    assert PP.permitted_roles("1") == {"oof_split1", "stabledet"}
    assert "official" not in PP.legitimate_claim_roles("0")
    assert "official" not in PP.legitimate_claim_roles("1")
