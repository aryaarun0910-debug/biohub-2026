r"""ONE FOLD-AWARE TRUNK PROVENANCE POLICY. Both guards consume THIS, and nothing else.

WHY THIS MODULE EXISTS - TWO GUARDS RETURNED OPPOSITE VERDICTS
--------------------------------------------------------------
``FACT-0418``: ``audit_feature_cache.PKT0029_REQUIRED_TRUNK_ROLES`` was ``('official',
'stabledet')``, so ``audit-pair --require-roles`` would have REFUSED the honest pair and ACCEPTED
the pair with no fold-legitimate arm on either fold. ``FACT-0431``: the newer
``gpu_protection_contract`` DATA-3 clause independently REFUSES an illegitimate role even when the
spec declares it legitimate. Two committed instruments, one question, opposite answers - which is
worse than one wrong guard, because whichever runs last looks authoritative.

THE FIX IS NOT TO EDIT ONE CONSTANT UNTIL THEY AGREE. That buys agreement by coincidence and
leaves the next divergence invisible. This module is the single source of truth; both guards call
into it and neither keeps its own table. ``PKT0029_REQUIRED_TRUNK_ROLES`` and
``gpu_protection_contract.LEGITIMATE_TRUNKS`` are both RETIRED into it.

WHAT THE PKT-0029 CONCERN ACTUALLY WAS, AND WHY IT SURVIVES
-----------------------------------------------------------
``FACT-0392`` risk three: which trunk produced a feature cache is a ``--weights`` command-line
argument recorded NOWHERE in the checkpoint, so feeding official-trunk features to a StableDet-
trained head looks exactly like a wrong feature contract, and a null result is uninterpretable.
The remedy - cache TWO trunks over ONE detector pass and compare them in the SAME session - is
untouched. What was wrong was the assumption that the two had to be ``official`` and
``stabledet`` specifically. The second arm is a CONTROL, not a claim: it must be a DIFFERENT trunk
over the SAME node set, and it need not be fold-legitimate. What it must never be is READ AS A
RESULT - and what it must never do is put a LEAKY checkpoint inside the fold at all.

THE POLICY
----------
FOLD 0 (holds out 44b6). The legitimate CLAIM arm is a split_0 trunk - the support pack's own
    ``pack_split0``, or our OOF ``oof_split0``, which is equally clean here. ``stabledet`` may
    serve ONLY as the declared comparison arm and is never read as a result.
FOLD 1 (holds out 6bba). The legitimate CLAIM arm is our OOF ``oof_split1``. NO split_0
    checkpoint may appear in the session AT ALL, in either arm: the pack ships only ``split_0``,
    trained on 6bba, so on fold 1 it is LEAKY - the EXP-0019 defect class, mirrored in
    ``tests/test_loeo_weights_hygiene.py`` and in ``audit_feature_cache.PACK_WEIGHTS_TOKEN``.
    ``stabledet`` is again comparison-only.
``official`` IS NEVER A CLAIM ARM ON EITHER FOLD. ``FACT-0418`` retracted its provenance to
    UNVERIFIED after the file recorded as the HOCT publisher default proved BYTE-IDENTICAL to our
    own OOF split_0. Nobody has established what it is, and this module must not re-assert one.
    Because those bytes ARE split_0, the role also inherits split_0's fold-1 leak.

EVERY ROLE IS BOUND BY SHA, FOLD, EMBRYO AND PROVENANCE. The digest binding is the half that
caught the retraction in the first place: the earlier record asserted a documented provenance for
a file whose BYTES said otherwise. So a declared role whose digest is a known OTHER checkpoint is
refused BY NAME here, not merely left unverified.
"""
from __future__ import annotations

# fold <-> held-out embryo. AGENTS.md and upstream audit item U10 both fix this direction.
FOLD_EMBRYO = {"0": "44b6", "1": "6bba"}
FOLDS = tuple(sorted(FOLD_EMBRYO))

# ---------------------------------------------------------------------------------------------
# THE ROLE TABLE. `claim_folds` is where the role may be READ AS A RESULT; `permitted_folds` is
# where it may appear at all, including as a comparison arm. The two differ on purpose: a
# comparison arm need not be legitimate, but a LEAKY checkpoint is not merely illegitimate - it
# contaminates the session it sits in, so it is excluded from `permitted_folds` as well.
# ---------------------------------------------------------------------------------------------
ROLES = {
    "pack_split0": {
        "split": "split_0",
        "claim_folds": {"0"},
        "permitted_folds": {"0"},
        "provenance": "EXTERNAL - the support pack's primary, trained on 6bba (FACT-0378)",
        "why": "the pack ships only split_0; LOEO-clean on fold 0, LEAKY on fold 1",
    },
    "oof_split0": {
        "split": "split_0",
        "claim_folds": {"0"},
        "permitted_folds": {"0"},
        "provenance": "MEASURED - our own out-of-fold split_0",
        "why": "clean on fold 0, and it is the SAME split as the pack, so it leaks on fold 1 too",
    },
    "oof_split1": {
        "split": "split_1",
        "claim_folds": {"1"},
        "permitted_folds": {"1"},
        "provenance": "MEASURED - our own out-of-fold split_1 (FACT-0418)",
        "why": "the ONLY fold-legitimate trunk fold 1 has; leaky on fold 0 by the same argument",
    },
    "official": {
        "split": "split_0",
        "claim_folds": set(),
        "permitted_folds": {"0"},
        "provenance": "UNVERIFIED - RETRACTED (FACT-0418): byte-identical to our OOF split_0, "
                      "not the HOCT publisher default it was recorded as",
        "why": "never a claim arm on either fold - nobody has established what this file is. Its "
               "bytes are split_0, so it also carries split_0's fold-1 leak",
    },
    "stabledet": {
        "split": None,
        "claim_folds": set(),
        "permitted_folds": {"0", "1"},
        "provenance": "EXTERNAL - the StableDet trunk shipped in the HOCT bundle",
        "why": "comparison arm only, on either fold: no LOEO split is established for it, so it "
               "answers FACT-0392's trunk-identity confound without ever being read as a result",
    },
}

TRUNK_ROLES = frozenset(ROLES)

# ---------------------------------------------------------------------------------------------
# KNOWN CHECKPOINT BYTES -> THEIR CANONICAL IDENTITY. Re-derived from the bytes on 2026-08-30 and
# agreeing with _evidence/foundations/p37_trunk_inventory.json. The first entry is the whole
# reason this table exists: two roles pointed at ONE file.
# ---------------------------------------------------------------------------------------------
KNOWN_CHECKPOINTS = {
    "d3e89eb361eeadef06d18159d834594776174a9f8cabd81f65271abbb42a492f": {
        "identity": "oof_split0",
        "bytes": 8357783,
        "found_at": ["artifacts/kaggle/weights_dataset/edge_predictor_best_split_0.pth",
                     "C:/temp/hoct/trunks/official_f0/edge_predictor_best.pth"],
        "note": "FACT-0418: the file recorded as the 'official' publisher trunk IS our OOF "
                "split_0. Declaring these bytes 'official' asserts a provenance nobody has "
                "established",
    },
    "2e4ebf616b3d4fb53881eadf42c7972e04978c28f74f229c5cb30bee835063de": {
        "identity": "oof_split1",
        "bytes": 8357783,
        "found_at": ["artifacts/kaggle/weights_dataset/edge_predictor_best_split_1.pth",
                     "kaggle dataset aryaarun07/biohub-oof-weights"],
        "note": "the only fold-legitimate fold-1 trunk. A `find` over C:/temp misses it because "
                "artifacts/ is gitignored - an empty search is not evidence (FACT-0418)",
    },
    "32d8048692aeb95fecd0245758a6b5b68889a84af4c4afc610414ae6fccb6fe4": {
        "identity": "stabledet",
        "bytes": 8357783,
        "found_at": ["C:/temp/hoct/trunks/stabledet_f0/edge_predictor_best.pth"],
        "note": "shares a byte SIZE with the split checkpoints and a different digest - identity "
                "is the HASH, never the size (FACT-0392)",
    },
    "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771": {
        "identity": "pack_split0",
        "bytes": 8363159,
        "found_at": ["C:/temp/hoct/trunks/pack_official_f0/edge_predictor_best.pth"],
        "note": "the support pack's split_0 primary - a DIFFERENT checkpoint from our OOF "
                "split_0 despite the shared name",
    },
}

# The registry writes 16-hex prefixes, so a caller may hold either form.
MIN_DIGEST_PREFIX = 16


def expected_embryo(fold) -> str | None:
    return FOLD_EMBRYO.get(str(fold).strip())


def known_identity(sha256: str | None) -> tuple[str | None, dict | None]:
    """Resolve a digest, full or >=16-hex prefix, to the identity its BYTES have."""
    if not sha256:
        return None, None
    s = str(sha256).strip().lower()
    if len(s) < MIN_DIGEST_PREFIX:
        return None, None
    for full, rec in KNOWN_CHECKPOINTS.items():
        if full.startswith(s) or s.startswith(full):
            return rec["identity"], rec
    return None, None


def legitimate_claim_roles(fold) -> set[str]:
    f = str(fold).strip()
    return {r for r, spec in ROLES.items() if f in spec["claim_folds"]}


def permitted_roles(fold) -> set[str]:
    f = str(fold).strip()
    return {r for r, spec in ROLES.items() if f in spec["permitted_folds"]}


def _unknown(fold, role) -> list[str]:
    out = []
    if str(fold).strip() not in FOLD_EMBRYO:
        out.append(f"fold_unresolved: {fold!r} is not one of {list(FOLDS)}. A fold that cannot be "
                   "resolved cannot be checked for leakage, so it FAILS CLOSED")
    if role not in ROLES:
        out.append(f"trunk_role_unknown: {role!r}; known roles are {sorted(ROLES)}")
    return out


def claim_arm_refusals(fold, role) -> list[str]:
    """May this role be READ AS A RESULT on this fold?"""
    out = _unknown(fold, role)
    if out:
        return out
    f, spec = str(fold).strip(), ROLES[role]
    if f not in spec["claim_folds"]:
        out.append(
            f"claim_arm_not_fold_legitimate: role {role!r} ({spec['split'] or 'no LOEO split'}) "
            f"may not be read as a result on fold {f} (holds out {FOLD_EMBRYO[f]}) - "
            f"{spec['why']}. Legitimate claim arms here: {sorted(legitimate_claim_roles(f))}")
    return out + comparison_arm_refusals(f, role)


def comparison_arm_refusals(fold, role) -> list[str]:
    """May this role be PRESENT on this fold at all, as the declared comparison arm?"""
    out = _unknown(fold, role)
    if out:
        return out
    f, spec = str(fold).strip(), ROLES[role]
    if f not in spec["permitted_folds"]:
        out.append(
            f"leaky_checkpoint_in_the_fold: role {role!r} is a {spec['split']} trunk and fold {f} "
            f"holds out {FOLD_EMBRYO[f]}, so it saw the held-out embryo in training. A comparison "
            "arm does not have to be fold-legitimate, but it must not be LEAKY - the EXP-0019 "
            f"defect class. {spec['why']}")
    return out


PLACEHOLDER_PROVENANCE = {"", "-", "n/a", "na", "none", "null", "unknown", "tbd", "todo", "?"}


def role_binding_refusals(role, sha256=None, fold=None, embryo=None,
                          provenance=None) -> list[str]:
    """Bind the ROLE to its BYTES, its fold's embryo and an explicit provenance.

    The digest half is what caught the ``official`` retraction: a role was believed because it was
    documented, on a file whose bytes were somebody else's. A digest we do not recognise is
    RECORDED, not refused - synthetic fixtures and future trunks are legitimate - but a digest we
    DO recognise as another checkpoint is refused by name.
    """
    out = []
    if role not in ROLES:
        return [f"trunk_role_unknown: {role!r}; known roles are {sorted(ROLES)}"]
    identity, rec = known_identity(sha256)
    if identity and identity != role:
        out.append(
            f"role_contradicts_its_bytes: this checkpoint is {identity!r}, not {role!r} "
            f"({str(sha256)[:16]}...). {rec['note']}")
    if fold is not None and embryo is not None:
        want = expected_embryo(fold)
        if want is None:
            out.append(f"fold_unresolved: {fold!r} is not one of {list(FOLDS)} - FAILS CLOSED")
        elif str(embryo).strip() != want:
            out.append(f"embryo_does_not_match_the_fold: fold {fold} holds out {want}, the "
                       f"declaration says {embryo!r}")
    if provenance is not None and str(provenance).strip().lower() in PLACEHOLDER_PROVENANCE:
        out.append("trunk_provenance_is_a_placeholder: the checkpoint is a bare state dict, so "
                   "the manifest is the ONLY place its producing trunk can be named")
    return out


def pair_refusals(fold, roles, digests=None) -> list[str]:
    """Is this dual-trunk pair legitimate ON THIS FOLD? The single answer both guards read.

    ``digests`` is optional and maps role -> sha256, so the pair is bound by content as well as
    by name where the caller has the bytes.
    """
    roles = list(roles or [])
    out = []
    if str(fold).strip() not in FOLD_EMBRYO:
        return [f"fold_unresolved: {fold!r} is not one of {list(FOLDS)} - FAILS CLOSED"]
    f = str(fold).strip()
    if len(roles) != 2:
        return [f"dual_trunk_pair_is_not_a_pair: {roles!r}"]
    if roles[0] == roles[1]:
        return [f"dual_trunk_roles_differ: both arms declare role {roles[0]!r}"]

    for role in roles:
        out += comparison_arm_refusals(f, role)
        if digests:
            out += role_binding_refusals(role, (digests or {}).get(role))

    claim = [r for r in roles if r in legitimate_claim_roles(f)]
    if not claim:
        named = " This is the FACT-0418 pairing, which has no legitimate arm on EITHER fold." \
            if set(roles) == {"official", "stabledet"} else ""
        out.append(
            f"pair_has_no_fold_legitimate_claim_arm: {sorted(roles)} on fold {f} (holds out "
            f"{FOLD_EMBRYO[f]}). One arm must be readable as a result; legitimate claim arms here "
            f"are {sorted(legitimate_claim_roles(f))}.{named}")
    return out


POLICY_VERSION = "provenance_policy_v1"


def policy_digest() -> str:
    """A sha256 over the policy's SEMANTIC CONTENT, not over the file's bytes.

    Hashing the file would move on a comment edit and, worse, would NOT move when the policy is
    changed in memory - so two guards reading a mutated policy could still report matching
    "versions" while acting on it. Hashing the decision tables makes the stamp a statement about
    what was ENFORCED. Both guards embed it, so a receipt pair that disagrees is detectable
    without re-running either guard.
    """
    import hashlib
    import json as _json

    payload = {
        "version": POLICY_VERSION,
        "fold_embryo": dict(sorted(FOLD_EMBRYO.items())),
        "roles": {
            role: {
                "split": spec["split"],
                "claim_folds": sorted(spec["claim_folds"]),
                "permitted_folds": sorted(spec["permitted_folds"]),
            }
            for role, spec in sorted(ROLES.items())
        },
        "known_checkpoints": {
            sha: rec["identity"] for sha, rec in sorted(KNOWN_CHECKPOINTS.items())
        },
    }
    return hashlib.sha256(
        _json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def policy_stamp() -> dict:
    """The identity of the policy that actually ran. Required in BOTH guards' receipts."""
    return {"policy_version": POLICY_VERSION, "policy_sha256": policy_digest(),
            "policy_module": "scripts/win_bet/provenance_policy.py"}


def describe(fold) -> dict:
    """What the policy says about one fold, for a receipt."""
    f = str(fold).strip()
    return {
        "fold": f,
        "held_out_embryo": FOLD_EMBRYO.get(f),
        "legitimate_claim_roles": sorted(legitimate_claim_roles(f)),
        "permitted_roles": sorted(permitted_roles(f)),
        "excluded_roles": sorted(set(ROLES) - permitted_roles(f)),
        "source": "scripts/win_bet/provenance_policy.py - the ONE policy both "
                  "audit_feature_cache and gpu_protection_contract consume (FACT-0418, FACT-0431)",
        **policy_stamp(),
    }
