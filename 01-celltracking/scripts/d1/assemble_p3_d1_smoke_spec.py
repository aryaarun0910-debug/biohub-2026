"""Deterministic assembler for the combined P3 + D1 + D1-F smoke specs.

ONE SPEC PER FACTORIAL CELL, DRIVEN BY THE MANIFEST
---------------------------------------------------
`data/d1_factorial/manifest_smoke.json` is the single source of truth for which crops run
under which checkpoint. This module reads its `shards` array and emits one Kaggle spec per
shard. Nothing about the routing is transcribed here: stems, weights glob, config glob, fold,
checkpoint hash and expected-crop count all come from the shard, and the assembler asserts
that it reproduced them byte-for-byte. A glob is not a manifest, and a hand-written stem list
is not a manifest either -- both are recorded traps in this repo.

WHY THIS REPLACED THE HAND-WRITTEN `SMOKE` TABLE. The previous version pinned exactly two
cells -- 44b6 under split_0 and 6bba under split_1 -- i.e. each family under the checkpoint
that held it out. That is a ROUTED-ONLY export, and `scripts/d1/d1f_probe.py::run_direction`
correctly refuses it with `CROSS-ENCODED`: correction C1 established that the two 32-D bases
are independent (rel-L2 1.41542 ~ sqrt(2), cos(w0,w1) = -0.155), so a head fitted in split-0's
basis cannot be evaluated in split-1's. The 2x2 needs the SAME checkpoint to encode both
families, which means two further cells that did not exist:

    smoke__s1__split_1__44b6__source__0   split_1 checkpoint over its OWN family   (NEW)
    smoke__s2__split_1__6bba__target__0   split_1 checkpoint over the held-out one  (was f1)
    smoke__s3__split_0__6bba__source__0   split_0 checkpoint over its OWN family   (NEW)
    smoke__s4__split_0__44b6__target__0   split_0 checkpoint over the held-out one  (was f0)

Three crops x two checkpoints = six crop-inferences, which is exactly the host's Phase 3.

SOURCE CELLS NEED NO KERNEL EDIT. Verified again against
`scripts/kaggle_edits/loeo_retarget.py`: it globs EVERY `train/*.zarr` under /kaggle/input
(l.42-49), then selects purely by `BIOHUB_LOEO_STEMS` membership (l.62-65). `LOEO_FOLD` is
used only for the scratch directory name, the printed manifest and the export filenames --
there is NO fold-membership guard anywhere in the block, and the primary weights come solely
from `BIOHUB_LOEO_WEIGHTS_GLOB` (l.106-122). Running a checkpoint over its own training
family is therefore a pure configuration change.

ROLE INVERSION IS INEXPRESSIBLE, NOT MERELY DISCOURAGED. `role` is never a free parameter:
it is a function of (checkpoint split, crop family) via SPLIT_SOURCE_FAMILY, checked against
the manifest at assembly time, and RE-DERIVED INSIDE THE KERNEL from the family prefix of the
stems that were actually mounted and the split of the checkpoint that was actually loaded
(edit E02b). A cell whose declared role disagrees with its own arithmetic raises before any
GPU work happens, and every kernel writes `d1_audit/d1_cell.json` naming its shard_id, split,
family, role, crop list and the OBSERVED sha256 of the weights it loaded.

WHY FOUR KERNELS AND NOT ONE. The LOEO retarget block binds a single fold at module level --
it reads BIOHUB_LOEO_FOLD once, mounts only the declared crops and REBINDS the notebook global
TEST_DIR, which the predict command, the wrapper's read_test_frame and the final audit all
read. Two checkpoints in one kernel would mean restructuring the notebook's single-pass
predict/wrapper/export flow, i.e. editing its most load-bearing cell. Each crop-inference is
still paid for exactly once; the four manifests are aggregated only after all four complete.

Determinism: sorted keys, no timestamps, no randomness. Generating twice must be byte-identical.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import pathlib
import re
import sys

ROOT = next(_p for _p in pathlib.Path(__file__).resolve().parents if (_p / 'pyproject.toml').exists())
SPECS = ROOT / "scripts" / "kaggle_specs"
MANIFESTS = {
    "smoke": ROOT / "data" / "d1_factorial" / "manifest_smoke.json",
    "pilot": ROOT / "data" / "d1_factorial" / "manifest_pilot.json",
}
MANIFEST = MANIFESTS["smoke"]  # Back-compatible import surface for existing tests.

BACKBONE = SPECS / "p3_harmonic.json"
LOEO = ROOT / "scripts" / "kaggle_templates" / "loeo_pregraph_edits.json"
LOEO_OUTPUT_PROVENANCE = (
    "loeo_f1_strict_pregraph.json sha256 "
    "0dc68c81e554828e5719bd6cc70fcc408c5439b74833a178cccbaa169ea01055"
)

# LOEO semantics. Fold 0 holds out 44b6, so the split_0 checkpoint TRAINED on 6bba; fold 1
# holds out 6bba, so split_1 trained on 44b6. These two dicts are the only definition of
# "role" anywhere in this module, and they agree with scripts/d1/d1f_probe.py:145-146.
SPLIT_SOURCE_FAMILY = {0: "6bba", 1: "44b6"}    # the family this checkpoint trained on
SPLIT_HELDOUT_FAMILY = {0: "44b6", 1: "6bba"}   # the family it has never seen

# Checkpoint hashes, verified locally from artifacts/kaggle/weights_dataset. Kept as an
# independent cross-check on the manifest rather than as the source of truth.
CKPT_SHA = {
    0: "d3e89eb361eeadef06d18159d834594776174a9f8cabd81f65271abbb42a492f",
    1: "2e4ebf616b3d4fb53881eadf42c7972e04978c28f74f229c5cb30bee835063de",
}

# A TARGET cell is the measurement and keeps the plain `f{fold}` identity the v6 build already
# published; a SOURCE cell is new and takes the `_src` suffix. The suffix is derived from the
# role, never chosen.
ROLE_SUFFIX = {"target": "", "source": "_src"}

# The routing keys the manifest owns. Anything else in a shard env is drift.
ROUTING_KEYS = (
    "BIOHUB_LOEO_FOLD",
    "BIOHUB_LOEO_ARM",
    "BIOHUB_LOEO_LIMIT",
    "BIOHUB_LOEO_STEMS",
    "BIOHUB_LOEO_WEIGHTS_GLOB",
    "BIOHUB_LOEO_CONFIG_GLOB",
    "BIOHUB_D1_FOLD",
    "BIOHUB_D1_CKPT_SHA",
    "BIOHUB_D1_EXPECTED_CROPS",
    "BIOHUB_D1_N_UNIFORM",
    "BIOHUB_D1_N_SUBTHR",
)


def sha_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_lf(p: pathlib.Path) -> str:
    """Checkout-filter-independent hash. core.autocrlf=true rewrites JSON on checkout, so a
    raw byte hash of a committed manifest is not portable; the LF-normalised one is."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_edit(e: dict) -> str:
    return hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest()


def tag(edit: dict, edit_id: str, source: str, order: int, intent: str) -> dict:
    """Attach audit metadata. kaggle_factory ignores unknown keys; our tests do not."""
    out = dict(edit)
    out["_audit"] = {
        "edit_id": edit_id,
        "source": source,
        "canonical_sha256": sha_edit(edit),
        "intent": intent,
        "order": order,
    }
    return out


def load_shards(tier: str = "smoke") -> list[dict]:
    """Every shard of a frozen tier, in the manifest's own stage order."""
    manifest = MANIFESTS[tier]
    man = json.loads(manifest.read_text(encoding="utf-8"))
    assert man["kind"] == "d1_factorial_manifest", man["kind"]
    assert man["tier"] == tier, man["tier"]
    shards = sorted(man["shards"], key=lambda s: int(s["stage"]))
    assert [int(s["stage"]) for s in shards] == list(range(1, len(shards) + 1)), (
        f"{tier} stages are not a contiguous 1..N run")
    return shards


def cell_key(shard: dict) -> str:
    return f"f{int(shard['fold'])}{ROLE_SUFFIX[shard['role']]}"


def cell_record(shard: dict, tier: str = "smoke") -> dict:
    """The immutable identity of one factorial cell.

    Every value is a str/int/list of str so the dict is simultaneously valid JSON and a valid
    Python literal -- it is embedded verbatim in the generated kernel cell.
    """
    split = declared_split(shard)
    # `role` is DERIVED, never read. There is no parameter through which a caller could
    # declare a source cell to be a target one; the manifest's value is only ever checked.
    role = "source" if shard["family"] == SPLIT_SOURCE_FAMILY[split] else "target"
    assert role == shard["role"], (
        f"{shard['shard_id']}: manifest says role={shard['role']} but split {split} trained on "
        f"{SPLIT_SOURCE_FAMILY[split]} and the crops are {shard['family']} => {role}")
    assert bool(shard["stems_are_in_checkpoint_training_set"]) is (role == "source"), (
        f"{shard['shard_id']}: training-set flag contradicts the derived role")
    return {
        "kind": "d1_factorial_cell",
        "tier": tier,
        "shard_id": shard["shard_id"],
        "stage": int(shard["stage"]),
        "split": split,
        "fold": int(shard["fold"]),
        "family": shard["family"],
        "role": role,
        "checkpoint_file": shard["weights_glob"].rsplit("/", 1)[-1],
        "checkpoint_sha256": shard["checkpoint_sha256"],
        "checkpoint_trained_on_family": SPLIT_SOURCE_FAMILY[split],
        "checkpoint_heldout_family": SPLIT_HELDOUT_FAMILY[split],
        "crops": list(shard["stems"]),
        "n_crops": len(shard["stems"]),
        "gt_nodes_total": int(shard["gt_nodes_total"]),
        "est_bytes": int(shard["est_bytes"]),
        "est_wall_hours": float(shard["est_wall_hours"]),
        "manifest": f"data/d1_factorial/manifest_{tier}.json",
        "manifest_sha256_lf": sha_lf(MANIFESTS[tier]),
        "spec": f"p3_d1_{tier}_{cell_key(shard)}",
    }


def declared_split(shard: dict) -> int:
    """The checkpoint BASIS, read off the weights the shard actually pins.

    The fold number and the checkpoint number are two different facts that happen to coincide
    here; deriving the basis from the weights filename and then asserting the coincidence is
    what keeps a future re-pairing honest.
    """
    m = re.search(r"edge_predictor_best_split_(\d+)\.pth$", shard["weights_glob"])
    assert m, f"{shard['shard_id']}: cannot read a split out of {shard['weights_glob']!r}"
    split = int(m.group(1))
    assert split == int(shard["fold"]), (
        f"{shard['shard_id']}: weights are split_{split} but fold is {shard['fold']}")
    assert shard["config_glob"].endswith(f"config_split_{split}.json"), (
        f"{shard['shard_id']}: config glob {shard['config_glob']!r} is not split {split}")
    assert shard["checkpoint_sha256"] == CKPT_SHA[split], (
        f"{shard['shard_id']}: checkpoint hash is not the pinned split-{split} one")
    return split


def identity_code(rec: dict) -> str:
    """The kernel-side identity block: re-derive the role, then write d1_cell.json.

    Placed immediately after the LOEO retarget's own manifest print, so it runs before the
    predict script is patched and long before any GPU work. Every check is a raise.
    """
    literal = json.dumps(rec, indent=4, sort_keys=True)
    return "\n".join([
        "# ---------------------------------------------------------- D1 FACTORIAL CELL IDENTITY",
        "# This kernel is ONE cell of the 2x2 checkpoint x family factorial (correction C1).",
        "# A SOURCE cell runs a checkpoint over the family it was TRAINED on and supplies the",
        "# fitting rows; a TARGET cell runs the SAME checkpoint over the family it has NEVER",
        "# seen and is the measurement. Both live in one 32-D basis -- that is the only reason a",
        "# head fitted on the source can be applied to the target at all.",
        "#",
        "# The role is not a free parameter. It is re-derived below from the split of the",
        "# checkpoint that was actually loaded and the family prefix of the stems that were",
        "# actually mounted, then compared with the value assembled from the manifest. A cell",
        "# that has been mislabelled cannot run.",
        f"_D1C = {literal}",
        '_D1C_TRAINED_ON = {0: "6bba", 1: "44b6"}',
        '_D1C_HELD_OUT = {0: "44b6", 1: "6bba"}',
        "",
        '_d1c_split = int(_D1C["split"])',
        '_d1c_fams = sorted({_s.split("_")[0] for _s in LOEO_STEMS})',
        "if len(_d1c_fams) != 1:",
        "    raise RuntimeError(",
        '        f"{_D1C[\'shard_id\']}: stems span families {_d1c_fams}; a factorial cell is "',
        '        "family-pure by construction")',
        "_d1c_family = _d1c_fams[0]",
        '_d1c_role = "source" if _d1c_family == _D1C_TRAINED_ON[_d1c_split] else "target"',
        "",
        "_d1c_bad = []",
        'if _d1c_family != _D1C["family"]:',
        '    _d1c_bad.append(f"family: mounted {_d1c_family} != declared {_D1C[\'family\']}")',
        'if _d1c_role != _D1C["role"]:',
        '    _d1c_bad.append(f"ROLE INVERSION: split {_d1c_split} trained on "',
        '                    f"{_D1C_TRAINED_ON[_d1c_split]} and held out "',
        '                    f"{_D1C_HELD_OUT[_d1c_split]}, so {_d1c_family} crops are "',
        '                    f"{_d1c_role}, but this cell declares {_D1C[\'role\']}")',
        'if sorted(LOEO_STEMS) != sorted(_D1C["crops"]):',
        '    _d1c_bad.append(f"crop list: mounted {sorted(LOEO_STEMS)} != declared "',
        '                    f"{sorted(_D1C[\'crops\'])}")',
        'if int(LOEO_FOLD) != int(_D1C["fold"]):',
        '    _d1c_bad.append(f"fold: env {LOEO_FOLD} != declared {_D1C[\'fold\']}")',
        'if os.environ.get("BIOHUB_D1_SPLIT") != str(_d1c_split):',
        '    _d1c_bad.append(f"basis: BIOHUB_D1_SPLIT="',
        '                    f"{os.environ.get(\'BIOHUB_D1_SPLIT\')!r} != {_d1c_split}")',
        'if os.environ.get("BIOHUB_D1_CKPT_SHA") != _D1C["checkpoint_sha256"]:',
        '    _d1c_bad.append("basis: BIOHUB_D1_CKPT_SHA is not this cell\'s checkpoint")',
        'if os.environ.get("BIOHUB_D1_EXPECTED_CROPS") != str(_D1C["n_crops"]):',
        '    _d1c_bad.append("crop budget: BIOHUB_D1_EXPECTED_CROPS disagrees with the manifest")',
        "",
        "# The weights the kernel actually resolved, hashed in full -- the retarget only prints",
        "# the first 16 hex digits, which is a display, not a gate.",
        "_d1c_obs = hashlib.sha256(Path(WEIGHTS_RELATIVE).read_bytes()).hexdigest()",
        'if _d1c_obs != _D1C["checkpoint_sha256"]:',
        '    _d1c_bad.append(f"checkpoint: loaded sha256 {_d1c_obs} != pinned "',
        '                    f"{_D1C[\'checkpoint_sha256\']}")',
        "",
        "if _d1c_bad:",
        "    raise RuntimeError(",
        '        f"D1 FACTORIAL CELL {_D1C[\'shard_id\']} IS NOT WHAT IT CLAIMS TO BE:\\n  "',
        '        + "\\n  ".join(_d1c_bad))',
        "",
        '_d1c_dir = Path("/kaggle/working/d1_audit")',
        "_d1c_dir.mkdir(parents=True, exist_ok=True)",
        "_d1c_rec = dict(_D1C)",
        "_d1c_rec.update({",
        '    "checkpoint_sha256_observed": _d1c_obs,',
        '    "weights_path": str(WEIGHTS_RELATIVE),',
        '    "family_rederived": _d1c_family,',
        '    "role_rederived": _d1c_role,',
        '    "crops_mounted": sorted(LOEO_STEMS),',
        '    "arm": LOEO_ARM,',
        "})",
        '(_d1c_dir / "d1_cell.json").write_text(',
        "    json.dumps(_d1c_rec, indent=2, sort_keys=True), encoding=\"utf-8\")",
        'print(f"D1 CELL {_D1C[\'shard_id\']}: split {_d1c_split} basis, family "',
        '      f"{_d1c_family}, role {_d1c_role.upper()}, {len(LOEO_STEMS)} crop(s), "',
        '      f"checkpoint {_d1c_obs[:16]} VERIFIED", flush=True)',
        "# --------------------------------------------------------------------------------------",
    ]) + "\n"


def build(shard: dict, tier: str = "smoke") -> dict:
    backbone = json.loads(BACKBONE.read_text(encoding="utf-8"))
    loeo = json.loads(LOEO.read_text(encoding="utf-8"))
    b_edits, l_edits = backbone["edits"], loeo["edits"]

    # --- assert imported sources are the ones we validated -------------------------------
    # Pinned per-edit hashes. If a source spec is edited later, assembly FAILS LOUDLY rather
    # than silently composing a different kernel.
    PINNED = {
        1: "d8e8bd4dac7f38676e4b8b0dbb2b8f0d",   # loeo retarget
        2: "a546af45c3057302",                    # test_stems routing
        3: "7e2712e00a82f54c",                    # export cell
        4: "5cbb94850ba34054",                    # pregraph export
        5: "b6a6c809e5d33dbf",                    # keep list
    }
    for idx, want in PINNED.items():
        got = sha_edit(l_edits[idx])
        assert got.startswith(want[:16]), (
            f"loeo edit {idx} drifted: expected {want[:16]}..., got {got[:16]}..."
        )
    imported_sha = {
        "loeo_retarget": sha_edit(l_edits[1]),
        "loeo_test_stems": sha_edit(l_edits[2]),
        "loeo_export_cell": sha_edit(l_edits[3]),
        "loeo_pregraph": sha_edit(l_edits[4]),
        "loeo_keep": sha_edit(l_edits[5]),
    }

    harmonic = [e for e in b_edits if e["kind"] == "replace" and "_bi_new" in e.get("old", "")]
    assert len(harmonic) == 1, f"expected exactly one harmonic edit, got {len(harmonic)}"
    invariants = [e for e in b_edits if e is not harmonic[0]]
    assert len(invariants) == 9, f"expected 9 invariant edits, got {len(invariants)}"

    rec = cell_record(shard, tier=tier)
    key = cell_key(shard)
    split, fold, role = rec["split"], rec["fold"], rec["role"]
    stems = rec["crops"]

    # --- environment: routing comes from the manifest, the v6 contract comes from here ----
    m_env = dict(shard["env"])
    assert set(m_env) == set(ROUTING_KEYS), (
        f"{rec['shard_id']}: shard env keys drifted from the routing contract: "
        f"{sorted(set(m_env) ^ set(ROUTING_KEYS))}")
    assert json.loads(m_env["BIOHUB_LOEO_STEMS"]) == stems, "shard env stems != shard stems"
    assert m_env["BIOHUB_LOEO_ARM"] == "strict", (
        "strict is what turns the secondary detector off, which is what makes the 8-encode-call "
        "mean the ONLY gap between w.x + b and rows.logit -- i.e. what makes the parity gate "
        "achievable. It is also why these results are not directly deployable (correction C5).")

    env_vars = {
        # --- routing, verbatim from data/d1_factorial/manifest_smoke.json ----------------
        "BIOHUB_LOEO_FOLD": m_env["BIOHUB_LOEO_FOLD"],
        "BIOHUB_LOEO_ARM": m_env["BIOHUB_LOEO_ARM"],
        "BIOHUB_LOEO_LIMIT": m_env["BIOHUB_LOEO_LIMIT"],
        "BIOHUB_LOEO_STEMS": m_env["BIOHUB_LOEO_STEMS"],
        "BIOHUB_LOEO_WEIGHTS_GLOB": m_env["BIOHUB_LOEO_WEIGHTS_GLOB"],
        "BIOHUB_LOEO_CONFIG_GLOB": m_env["BIOHUB_LOEO_CONFIG_GLOB"],
        "BIOHUB_D1_FOLD": m_env["BIOHUB_D1_FOLD"],
        # The manifest must name the checkpoint BASIS, not just the fold: with the routed-only
        # export 44b6 sat in split-0's 32-D basis and 6bba in split-1's, and the two bases are
        # independent (rel-L2 1.41542 ~ sqrt(2)). A head fitted in one cannot be applied in the
        # other, so every exported feature row has to carry the split it was encoded in. In this
        # factorial the fold and the basis are deliberately decoupled: a SOURCE cell runs fold
        # N's checkpoint over fold N's OWN training family.
        "BIOHUB_D1_SPLIT": str(split),
        "BIOHUB_D1_CKPT_SHA": m_env["BIOHUB_D1_CKPT_SHA"],
        "BIOHUB_D1_EXPECTED_CROPS": m_env["BIOHUB_D1_EXPECTED_CROPS"],
        "BIOHUB_D1_N_UNIFORM": m_env["BIOHUB_D1_N_UNIFORM"],
        "BIOHUB_D1_N_SUBTHR": m_env["BIOHUB_D1_N_SUBTHR"],
        # v6 view-set contract. All RAISE gates, not preferences.
        #
        # THE 8/7 SPLIT IS TWO NUMBERS AND MUST STAY TWO NUMBERS. The deployed TTA makes
        # EIGHT encode calls over SEVEN distinct spatial permutations, because
        # `torch.rot90(imgs, 1, dims=(-2,-1)).transpose(-1,-2)` -- written as the
        # anti-transpose -- is exactly `imgs.flip(-1)`, already view 1. So flip(-1) carries
        # weight 2/8, the true anti-transpose 0/8, and the divisor is still 8. A single
        # conflated view count cannot express that and would let a future agent "fix" the
        # collision silently. Fixing it is a DETECTOR change: it moves the node population
        # (correction C3) and voids the 0.889 anchor. v6 replicates it verbatim.
        #
        #   REQUIRE_ENCODE_CALLS    refuse to export unless the divisor reached 8, so
        #                           identity-view features can never ship under the post-TTA
        #                           name (blocker B3).
        #   REQUIRE_DISTINCT_VIEWS  refuse unless exactly 7 of those 8 are distinct
        #                           permutations -- counted by applying each named view to an
        #                           index grid, not asserted from a comment.
        #   REQUIRE_NOTEBOOK_TTA    the injector refuses to proceed if only the vendored
        #                           4-view block is present, i.e. if the notebook's own TTA
        #                           patch silently failed -- its guard only prints.
        #   PARITY_SLACK            multiplier on the DERIVED float32 bound
        #                           `n_encode_calls * 2**-24 * max|logit|`. Pinned in the spec
        #                           so the gate that hard-aborts the run is recorded, not
        #                           inherited from a default. See _d1_parity_bound.
        "BIOHUB_D1_REQUIRE_ENCODE_CALLS": "8",
        "BIOHUB_D1_REQUIRE_DISTINCT_VIEWS": "7",
        "BIOHUB_D1_REQUIRE_NOTEBOOK_TTA": "1",
        "BIOHUB_D1_PARITY_SLACK": "2.0",
    }

    # keep-list: derive from the imported edit rather than transcribing it, then extend so the
    # cleanup's rmtree cannot delete the audit directory.
    keep = dict(l_edits[5])
    anchor = '"pregraphs_split{LOEO_FOLD}.json"}'
    assert keep["new"].count(anchor) == 1, "keep-list shape changed; refusing to guess"
    keep["new"] = keep["new"].replace(
        anchor, '"pregraphs_split{LOEO_FOLD}.json",\n              "d1_audit"}', 1)

    # v6 REQUIREMENT: the TTA patch guard RAISES instead of printing.
    #
    # The base notebook ends its TTA patch with
    #     else:
    #         print("TTA WARNING: block not found - using default 4-way")
    # so a failed patch left the vendored FOUR-view average in place and carried on. That is
    # not silent in practice -- the NEXT patch anchors on text containing `_nv`, so a failed
    # TTA patch already hard-fails one cell later -- but the failure is reported a step away
    # from its cause. This converts it into a local, self-naming abort. It is a LOCALITY fix,
    # NOT the repair of a correctness hole; recorded that way so nobody credits it with more.
    #
    # Scoped to the D1 smoke specs only. The base notebook is not this lane's file to change.
    tta_guard = {
        "kind": "replace",
        "cell_match": "TTA patch applied",
        "old": '    print("TTA WARNING: block not found - using default 4-way")',
        "new": chr(10).join([
            "    raise RuntimeError(",
            '        "TTA PATCH FAILED: the eight-encode-call TTA block was not found in "',
            '        + str(_ps) + ". The deployed view set would fall back to the vendored "',
            '        "four-view average, so any feature exported under the post-TTA name "',
            '        "would describe a detector that is not in production. Refusing."',
            "    )",
        ]),
        "expect": 1,
    }

    cell_identity = {
        "kind": "insert_after",
        "cell_match": "_ps = REPO_DIR",
        "anchor": "print(json.dumps(LOEO_MANIFEST, indent=2)[:800])",
        "code": identity_code(rec),
        "expect": 1,
    }

    d1_inject = {
        "kind": "insert_before",
        "anchor": "def list_test_stems() -> list[str]:",
        "cell_match": "_ps = REPO_DIR",
        "code_file": "scripts/kaggle_edits/d1_inject.py",
        "expect": 1,
    }

    # --- FROZEN SEMANTIC ORDER -----------------------------------------------------------
    ordered: list[dict] = []
    n = 0

    def add(edit, edit_id, source, intent):
        nonlocal n
        ordered.append(tag(edit, edit_id, source, n, intent))
        n += 1

    add({"kind": "env", "cell_match": "BIOHUB_PRESET", "vars": env_vars, "expect": 1},
        "E01_env", f"assembler+manifest_{tier}.json",
        "1. environment: LOEO fold routing + D1 knobs")
    add(l_edits[1], "E02_loeo_retarget", "loeo_f1_strict_pregraph#1",
        "1. LOEO retarget: mount fold crops, rebind TEST_DIR, resolve fold weights")
    add(cell_identity, "E02b_cell_identity", f"assembler+manifest_{tier}.json",
        "1. factorial cell identity: re-derive the role, verify the loaded checkpoint, "
        "emit d1_cell.json")
    add(l_edits[2], "E03_test_stems", "loeo_f1_strict_pregraph#2",
        "1. route stem discovery at the LOEO fold")
    add(harmonic[0], "E04_harmonic", "p3_harmonic#harmonic",
        "2. harmonic edge-logit fusion (edge_logits_pair ONLY)")
    for i, e in enumerate(invariants):
        add(e, f"E05_{i:02d}_invariant", "p3_harmonic#invariant",
            "3. degree-invariant fixes and export assertions")
    add(tta_guard, "E05b_tta_guard", "assembler",
        "4. TTA patch guard raises instead of printing (locality, not correctness)")
    add(d1_inject, "E06_d1_inject", "scripts/kaggle_edits/d1_inject.py",
        "4. D1 audit + frozen-feature injection into the predict script")
    add(l_edits[4], "E07_pregraph", "loeo_f1_strict_pregraph#4",
        "5. pre-wrapper prediction-graph export")
    add(l_edits[3], "E08_export_cell", "loeo_f1_strict_pregraph#3",
        "6. LOEO export cell replaces the submission guard")
    add({"kind": "append_cell", "code_file": "scripts/kaggle_edits/d1_aggregate.py"},
        "E10_aggregator", "scripts/kaggle_edits/d1_aggregate.py",
        "6. parent aggregator: read immutable per-crop records, fail hard on any gap")
    add(keep, "E09_keep", "loeo_f1_strict_pregraph#5 (+d1_audit)",
        "6. artifact retention: keep pregraphs AND the d1_audit directory")

    role_word = "SOURCE" if role == "source" else "TARGET"
    role_gloss = (
        "the checkpoint runs over the family it was TRAINED on, so these rows are the "
        "fitting/selection substrate and are never a measurement"
        if role == "source" else
        "the checkpoint runs over the family it has NEVER seen, so these rows are the "
        "measurement and are opened exactly once after the head is frozen")
    purpose_prefix = (
        "STRUCTURAL SMOKE" if tier == "smoke" else "PILOT FACTORIAL EXPORT"
    )

    return {
        "name": f"p3_d1_{tier}_{key}",
        "slug": f"biohub-p3-d1-{tier}-{key.replace('_', '-')}",
        "title": f"Biohub P3 D1 {tier.title()} {key.upper().replace('_', ' ')}",
        "code_file": f"biohub-p3-d1-{tier}-{key.replace('_', '-')}.ipynb",
        "out_dir": f"notebooks/kaggle_p3_d1_{tier}_{key}",
        "datasets": loeo["datasets"],
        "competition_sources": loeo["competition_sources"],
        "enable_gpu": True,
        "enable_internet": False,
        "is_private": True,
        "machine_shape": "NvidiaTeslaT4",
        "expects_submission": False,
        "base_notebook": backbone["base_notebook"],
        "base_sha256": backbone["base_sha256"],
        "d1_cell": rec,
        "purpose": (
            f"{purpose_prefix}. Factorial cell {rec['shard_id']} (stage {rec['stage']} of 4): "
            f"the split_{split} checkpoint over {len(stems)} {rec['family']} crop(s), role "
            f"{role_word} -- {role_gloss}. The four smoke cells together encode all three "
            "crops under BOTH checkpoints, which is what makes the C1 2x2 runnable at all: a "
            "routed-only export puts each family in its own 32-D basis and d1f_probe refuses "
            "it as CROSS-ENCODED. Combined P3 + D1 + D1-F export: one encoder pass per crop "
            "emits the P3 pre-wrapper graph (harmonic association), accepted detector peaks, "
            "the exact A/B/D classification of every GT node with D stratified by distance to "
            "the nearest local maximum, and FOUR 32-D frozen feature arrays: the TTA MEAN over "
            "the deployed view set -- 8 encode calls over 7 DISTINCT views, divisor 8, "
            "replicated verbatim -- which is the post-TTA detector representation, and the "
            "IDENTITY VIEW, which is the association representation that predict_edges "
            "actually reads; each at the sampled voxel and at the strongest nearby local "
            "maximum. Combinable because the harmonic patch touches edge_logits_pair only and "
            "never det_logits. Runs under BIOHUB_LOEO_ARM=strict, so the secondary detector is "
            "off and the result is NOT directly deployable (correction C5: the public path "
            "blends 0.525 primary + 0.475 aligned secondary behind a per-frame retention "
            "guard). NOT a scientific verdict and NOT a submission."
        ),
        "provenance": {
            "backbone": f"p3_harmonic.json sha256 {sha_file(BACKBONE)}",
                # Preserve the generated spec's original source identity byte-for-byte. The
                # complete source is archived at pre-lean-2026-08-07; the active tree retains
                # only its five hash-pinned edits in kaggle_templates/.
                "loeo_source": LOEO_OUTPUT_PROVENANCE,
            "imported_edit_sha256": imported_sha,
            "d1_inject_sha256": sha_file(ROOT / "scripts/kaggle_edits/d1_inject.py"),
            "d1_block_sha256": sha_file(ROOT / "scripts/kaggle_edits/d1_response_audit.py"),
            "checkpoint_sha256": rec["checkpoint_sha256"],
            "stems": stems,
            "routing_source": (
                f"data/d1_factorial/manifest_{tier}.json shard {rec['shard_id']} "
                f"(LF sha256 {rec['manifest_sha256_lf']}). Stems, weights glob, config glob, "
                "fold, checkpoint hash and expected-crop count are copied from the shard and "
                "asserted equal; nothing here is transcribed and no glob stands in for a "
                "manifest."
            ),
            "four_spec_rationale": (
                "The LOEO retarget binds one fold at module level and rebinds TEST_DIR "
                "globally, so two checkpoints in one kernel would require restructuring the "
                "notebook's single-pass predict/wrapper/export flow. The 2x2 needs four "
                "kernels: each of the two checkpoints must encode BOTH families, because the "
                "two 32-D bases are independent (rel-L2 1.41542 ~ sqrt(2), cos(w0,w1) = "
                "-0.155) and a head fitted in one cannot be applied in the other. Six "
                "crop-inferences total; each is paid for exactly once and the four manifests "
                "are aggregated only after all four runs complete."
            ),
            "source_cells_need_no_kernel_edit": (
                "scripts/kaggle_edits/loeo_retarget.py mounts EVERY train .zarr under "
                "/kaggle/input and selects purely by BIOHUB_LOEO_STEMS membership; LOEO_FOLD "
                "only names the scratch directory, the printed manifest and the export files. "
                "There is no fold-membership guard, and the primary weights come solely from "
                "BIOHUB_LOEO_WEIGHTS_GLOB. Running a checkpoint over its own training family "
                "is therefore a configuration change, not a code change."
            ),
        },
        "edits": ordered,
    }


def main(argv: list[str] | None = ()) -> None:
    ap = argparse.ArgumentParser(description="Assemble cross-encoded P3+D1 factorial specs")
    ap.add_argument("--tier", choices=sorted(MANIFESTS), default="smoke")
    args = ap.parse_args(argv)
    tier = args.tier
    for shard in load_shards(tier):
        spec = build(shard, tier=tier)
        out = SPECS / f"{spec['name']}.json"
        text = json.dumps(spec, indent=2, sort_keys=False) + "\n"
        out.write_text(text, encoding="utf-8")
        rec = spec["d1_cell"]
        print(f"wrote {out.name:26s} stage={rec['stage']} shard={rec['shard_id']:38s} "
              f"split={rec['split']} family={rec['family']} role={rec['role']:6s} "
              f"crops={len(rec['crops'])} edits={len(spec['edits'])} "
              f"sha256={hashlib.sha256(text.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    main(sys.argv[1:])
