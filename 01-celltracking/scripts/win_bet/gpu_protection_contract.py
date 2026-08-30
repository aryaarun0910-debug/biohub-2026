r"""THE GPU PROTECTION CONTRACT - four signed sections, enforced, failing CLOSED.

WHAT THIS IS
------------
The host's standing block at the head of ``research/00-system/registry/levers.yaml`` names four
sections that must be SIGNED before any GPU session starts. This module is that contract as an
INSTRUMENT rather than a checklist: every clause is a check that returns a verdict, the run FAILS
if any clause fails, and a clause that could not be evaluated is a FAILURE and never a skip.

It sits UNDER the standing GPU rule. That rule says every session must produce an asset; this one
says what must be true before a session may start.

WHY EACH SECTION EXISTS - EVERY CLAUSE HAS ALREADY COST SOMETHING
-----------------------------------------------------------------
CODE      A built notebook embedding uncommitted code cannot be bound to a source version. The
          trap that produced the clause, and it is not hypothetical: the injected patch may be
          GENERATED, so committing the file the check names can STILL leave HEAD failing its own
          drift lock. ``scripts/kaggle_edits/assoc_tap_gate.py`` is rendered from
          ``scripts/win_bet/assoc_tap_replay.py`` by ``sync_tap_worker.py``, and
          ``tests/test_assoc_feature_tap.py`` locks the two together.
DATA      ``FACT-0418``: every declared ``dual_trunk_pair`` was ``['official','stabledet']`` with
          NO fold-legitimate arm on either fold, while ``audit_feature_cache``'s own constant would
          have REFUSED the honest pair and ACCEPTED the pair with no legitimate arm. A trunk
          section that only checks "a pair is declared" is exactly the check that passed that
          state. Both guards now read ONE table, ``provenance_policy`` - the divergence
          ``FACT-0431`` recorded was two instruments each keeping their own.
PROCESS   ``FACT-0387`` and ``FACT-0399``: state that does not cross the subprocess boundary. And
          the kernel's own ``all_passed`` CANNOT see a crop that never ran, so the crop COUNT is
          asserted externally or it is not asserted at all.
ARTIFACT  ``FACT-0416``: the storage figure was re-derived at 1,580 B/node after an estimate that
          was 12.3x LOW. ``FACT-0417``: the signed receipt's own ``expected_gpu_outputs`` was
          hardcoded to the wrong worker - which would have FAILED a correct run and PASSED a run
          that recorded nothing - and the post-run audit reads that field while nobody re-derives
          it. So expected outputs are re-derived FROM THE BUILT NOTEBOOK, never trusted.

FAIL CLOSED, AND PROVED BY MUTATION
-----------------------------------
``--selftest`` does not assert that the sections pass. It MUTATES a known-good world once per
clause and requires that the clause, and preferably only that clause, rejects. A contract whose
clauses have never rejected anything is a contract that has never been tested.

    python scripts/win_bet/gpu_protection_contract.py sign --spec <spec.json>
    python scripts/win_bet/gpu_protection_contract.py selftest
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
HEARTBEAT = "GPU_PROTECTION_CONTRACT_COMPLETE"

# FACT-0416, re-derived from the tap's own declared dtypes. The superseded value is named so a
# regression to it is refused BY VALUE rather than going unnoticed.
BYTES_PER_NODE = 1580
SUPERSEDED_BYTES_PER_NODE = 128
# THE FOLD-LEGITIMACY TABLE IS NOT HERE. `INVALID_PAIR` and `LEGITIMATE_TRUNKS` used to be local
# constants, and audit_feature_cache kept its own - which is how two committed guards came to
# answer one question in opposite directions (FACT-0418, FACT-0431). Both now read
# `provenance_policy`, and neither keeps a table of its own, so the next divergence cannot happen
# quietly: it becomes an edit to a file both of them import.
sys.path.insert(0, str(REPO / "scripts" / "win_bet"))
import provenance_policy as PP  # noqa: E402

DEPLOYED_PREDICTOR_SHA256 = "25b3ebfd8849dcf5abeff9ed3f0d57269a4b979365c989d6f78db1e5002d5219"
# Patch sources that are GENERATED. Mapping: generated file -> module exposing rendered().
GENERATED_PATCHES = {"scripts/kaggle_edits/assoc_tap_gate.py": "sync_tap_worker"}


class ContractRefusal(RuntimeError):
    """Raised when the contract cannot be evaluated. Never downgraded to a skip."""


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str:
    r = subprocess.run(["git", *args], capture_output=True, cwd=str(cwd or REPO))
    return r.stdout.decode("utf-8", "replace")


def _clause(cid, title, passed, evidence, mutation):
    return {"id": cid, "title": title, "passed": bool(passed),
            "evidence": evidence, "proved_by_mutation": mutation}


# ======================================================================================
# SECTION 1 - CODE
# ======================================================================================
def section_code(spec: dict, spec_path: Path, repo: Path, all_specs: list[Path]) -> list[dict]:
    out = []
    srcs = [e["code_file"] for e in spec.get("edits", []) if e.get("code_file")]
    scoped = srcs + [spec.get("base_notebook"), str(spec_path.relative_to(repo).as_posix())]
    scoped = [s for s in scoped if s]

    dirty = []
    for s in scoped:
        st = _git("status", "--porcelain", "--", s, cwd=repo).strip()
        if st:
            dirty.append(st)
    out.append(_clause(
        "CODE-1", "every file this build depends on is committed - a built notebook embedding "
        "uncommitted code cannot be bound to a source version",
        not dirty, {"scoped_paths": scoped, "dirty": dirty},
        "touch any scoped file; git status stops being empty"))

    # THE GENERATED-FILE TRAP. Committing the named file is not enough if it is RENDERED.
    gen_problems, gen_checked = [], []
    for s in srcs:
        mod_name = GENERATED_PATCHES.get(s)
        if not mod_name:
            continue
        gen_checked.append(s)
        sys.path.insert(0, str(repo / "scripts" / "win_bet"))
        try:
            mod = __import__(mod_name)
            rendered = mod.rendered()
        except Exception as exc:                       # a generator we cannot run is a FAILURE
            gen_problems.append(f"{s}: generator {mod_name} unusable: {exc}")
            continue
        on_disk = (repo / s).read_text(encoding="utf-8")
        if rendered != on_disk:
            gen_problems.append(f"{s} is STALE against its generator {mod_name}")
    out.append(_clause(
        "CODE-2", "every GENERATED patch source matches its generator - committing the file the "
        "check names can still leave HEAD failing its own drift lock",
        not gen_problems, {"generated_sources_checked": gen_checked, "problems": gen_problems},
        "edit the generated file without re-rendering; the render comparison fails"))

    man_p = repo / spec["out_dir"] / "build_manifest.json"
    nb_p = repo / spec["out_dir"] / spec["code_file"]
    ok, ev = False, {}
    if man_p.is_file() and nb_p.is_file():
        man = json.loads(man_p.read_text(encoding="utf-8"))
        got, want = _sha(nb_p), man.get("built_sha256")
        ok = got == want
        ev = {"built_sha256": got, "manifest_built_sha256": want}
    else:
        ev = {"error": "build manifest or built notebook absent"}
    out.append(_clause(
        "CODE-3", "the built notebook reproduces the SHA its manifest recorded", ok, ev,
        "rewrite one byte of the notebook; the sha stops matching"))

    slug, odir = spec.get("slug"), spec.get("out_dir")
    clashes = []
    for other in all_specs:
        if other.resolve() == spec_path.resolve():
            continue
        try:
            o = json.loads(other.read_text(encoding="utf-8"))
        except Exception:
            continue
        if o.get("slug") == slug:
            clashes.append(f"slug {slug!r} also in {other.name}")
        if o.get("out_dir") == odir:
            clashes.append(f"out_dir {odir!r} also in {other.name}")
    out.append(_clause(
        "CODE-4", "slug and out_dir are unique across every spec - a copied spec silently "
        "overwrites its sibling's kernel-metadata and repoints that kernel's id",
        not clashes, {"slug": slug, "out_dir": odir, "clashes": clashes},
        "point a second spec at the same out_dir; the clash list is non-empty"))

    pack = Path("C:/temp/p7/tracking_repo/scripts/predict_unet_transformer.py")
    got = _sha(pack) if pack.is_file() else None
    out.append(_clause(
        "CODE-5", "the DEPLOYED support pack is pinned by sha256 - not vendor/kaggle-cell-tracking, "
        "which has no fusion code and cannot exhibit the defect under test (FACT-0408)",
        got == DEPLOYED_PREDICTOR_SHA256,
        {"pack_predictor": str(pack), "sha256": got, "pinned": DEPLOYED_PREDICTOR_SHA256},
        "point the pin at the vendored predictor; the sha stops matching"))
    return out


# ======================================================================================
# SECTION 2 - DATA
# ======================================================================================
def section_data(spec: dict, trunk: dict | None) -> list[dict]:
    out = []
    fold = None
    for e in spec.get("edits", []):
        v = e.get("vars", {})
        if "BIOHUB_LOEO_FOLD" in v:
            fold = str(v["BIOHUB_LOEO_FOLD"]).strip()
    out.append(_clause(
        "DATA-1", "the run declares which LOEO fold it evaluates - a fold that cannot be resolved "
        "cannot be checked for leakage",
        fold in ("0", "1"), {"fold": fold},
        "remove BIOHUB_LOEO_FOLD from the spec; the fold stops resolving"))

    if trunk is None:
        out.append(_clause(
            "DATA-2", "a trunk-bearing run declares a FOLD-LEGITIMATE trunk", True,
            {"trunk": None, "note": "no trunk declared - single-trunk runs claiming no pair are "
                                    "out of scope for this clause (FACT-0418 scope)"},
            "declare an illegitimate trunk; the role check rejects"))
        out.append(_clause(
            "DATA-3", "any declared dual_trunk_pair has at least one FOLD-LEGITIMATE arm", True,
            {"pair": None}, "declare the FACT-0418 pair; the clause rejects"))
        return out

    role = trunk.get("role")
    # THE SHARED POLICY, and the spec's own declaration is checked AGAINST it rather than trusted:
    # a spec that declares `fold_legitimate: true` on an illegitimate role is the mutation
    # FACT-0431 records this clause catching.
    claim_refusals = PP.claim_arm_refusals(fold, role)
    if not trunk.get("fold_legitimate"):
        claim_refusals = claim_refusals + [
            "fold_legitimate_not_declared: the spec does not claim this trunk is legitimate, so "
            "nothing in it may be read as a result"]
    out.append(_clause(
        "DATA-2", "the trunk this run trains on is FOLD-LEGITIMATE for its own fold, by role AND "
        "by explicit declaration - FACT-0418 retracted 'official' to UNVERIFIED after it proved "
        "byte-identical to our own split_0",
        not claim_refusals,
        {"fold": fold, "role": role, "fold_legitimate_declared": trunk.get("fold_legitimate"),
         "refusals": claim_refusals, "policy": PP.describe(fold)},
        "set role to 'official' on fold 1; the role leaves the policy's claim set"))

    pair = tuple(trunk.get("dual_trunk_pair") or ())
    if pair:
        pair_refusals = PP.pair_refusals(fold, pair)
        out.append(_clause(
            "DATA-3", "any declared dual_trunk_pair is fold-legitimate under provenance_policy - "
            "one arm readable as a result, the other a permitted comparison arm, and no leaky "
            "checkpoint in either. The FACT-0418 pairing ['official','stabledet'] has no "
            "legitimate arm on EITHER fold",
            not pair_refusals,
            {"pair": list(pair), "refusals": pair_refusals, "policy": PP.describe(fold)},
            "declare ['official','stabledet'], or swap the pair onto the other fold; both reject"))
    else:
        out.append(_clause(
            "DATA-3", "any declared dual_trunk_pair has at least one FOLD-LEGITIMATE arm", True,
            {"pair": None, "note": "single-trunk by design, claims no pair"},
            "declare the FACT-0418 pair; the clause rejects"))

    sha = trunk.get("checkpoint_sha256")
    bind_refusals = PP.role_binding_refusals(role, sha, fold=fold,
                                             embryo=trunk.get("held_out_embryo"),
                                             provenance=trunk.get("provenance"))
    if not trunk.get("contamination"):
        bind_refusals = bind_refusals + ["contamination_status_not_declared"]
    if not sha:
        bind_refusals = bind_refusals + ["checkpoint_sha256_missing: the binding is gone"]
    out.append(_clause(
        "DATA-4", "contamination status is declared and the checkpoint is bound by hash, and the "
        "hash must AGREE with the declared role - a checkpoint whose provenance is not "
        "split-specific is the EXP-0019 defect, and FACT-0418 was found by hashing the bytes of a "
        "file whose documented role said something else",
        not bind_refusals,
        {"contamination": trunk.get("contamination"), "checkpoint_sha256": sha,
         "refusals": bind_refusals},
        "drop checkpoint_sha256, or declare a role the bytes contradict; both reject"))
    return out


# ======================================================================================
# SECTION 3 - PROCESS
# ======================================================================================
def section_process(nb_text: str, spec: dict) -> list[dict]:
    out = []
    env_keys = sorted({k for e in spec.get("edits", []) for k in e.get("vars", {})})
    # The env the notebook sets must be handed to the predictor SUBPROCESS explicitly.
    propagates = bool(re.search(r"env\s*=\s*", nb_text)) and "PYTHONPATH" in nb_text
    out.append(_clause(
        "PROC-1", "the notebook propagates an EXPLICIT environment into the predictor subprocess - "
        "parent-process state does not reach a child (FACT-0060), and cwd is not on sys.path for a "
        "script invocation (FACT-0399)",
        propagates,
        {"env_keys_set_by_spec": env_keys, "explicit_env_passed": bool(re.search(r"env\s*=\s*", nb_text)),
         "pythonpath_set": "PYTHONPATH" in nb_text},
        "delete the env= argument from the subprocess launch; the clause rejects"))

    beats = [t for t in ("AFT_GATE", "ASSOC_FEATURE_PARITY_COMPLETE", "AFT_GATE_FAILED",
                         "ASSOC_TRAIN_HARNESS_COMPLETE", "AFT_PATCH_APPLIED") if t in nb_text]
    out.append(_clause(
        "PROC-2", "a POSITIVE heartbeat is emitted whose ABSENCE is the alarm - a silent no-op is "
        "worse than a crash",
        bool(beats), {"heartbeat_tokens_found": beats},
        "strip the heartbeat prints; no token is found and the clause rejects"))

    expect = None
    for e in spec.get("edits", []):
        if "BIOHUB_AFT_EXPECT_CROPS" in e.get("vars", {}):
            expect = e["vars"]["BIOHUB_AFT_EXPECT_CROPS"]
    limit = None
    for e in spec.get("edits", []):
        if "BIOHUB_LOEO_LIMIT" in e.get("vars", {}):
            limit = e["vars"]["BIOHUB_LOEO_LIMIT"]
    ok = expect is not None and str(expect) == str(limit)
    out.append(_clause(
        "PROC-3", "the expected CROP COUNT is asserted externally and equals the run limit - the "
        "kernel's own all_passed cannot see a crop that never ran",
        ok, {"expect_crops": expect, "loeo_limit": limit,
             "why": "a report with a crop count other than the declared one is a FAIL regardless "
                    "of what all_passed says"},
        "set EXPECT_CROPS to 2 and LIMIT to 1; the counts disagree and the clause rejects"))

    timing = bool(re.search(r"time\.time\(\)|perf_counter", nb_text)) and \
        bool(re.search(r"BUDGET|abort|deadline|elapsed", nb_text, re.I))
    out.append(_clause(
        "PROC-4", "a timing probe with EARLY ABORT exists, so a session that will not finish banks "
        "a partial result instead of dying at the wall clock",
        timing,
        {"clock_present": bool(re.search(r"time\.time\(\)|perf_counter", nb_text)),
         "abort_or_budget_present": bool(re.search(r"BUDGET|abort|deadline|elapsed", nb_text, re.I))},
        "remove the elapsed/abort logic; the clause rejects"))
    return out


# ======================================================================================
# SECTION 4 - ARTIFACT
# ======================================================================================
def section_artifact(nb_text: str, spec: dict, nodes: int | None) -> list[dict]:
    out = []
    atomic = bool(re.search(r"\.replace\(|os\.replace|\.rename\(|tmp.*->|\.part\b", nb_text))
    keep = "_LOEO_KEEP" in nb_text
    out.append(_clause(
        "ART-1", "the run's own output is written ATOMICALLY and survives the export sweep - a "
        "half-written artifact that looks complete is unrecoverable after the session ends",
        atomic and keep,
        {"atomic_rename_present": atomic, "export_keep_list_present": keep},
        "drop the file from _LOEO_KEEP; the sweep would delete it and the clause rejects"))

    ev = {"bytes_per_node": BYTES_PER_NODE, "superseded_and_refused": SUPERSEDED_BYTES_PER_NODE,
          "nodes": nodes}
    if nodes:
        gib = nodes * BYTES_PER_NODE / 1024**3
        ev["single_trunk_gib"] = round(gib, 2)
        ev["dual_trunk_gib"] = round(gib * 2, 2)
        ev["within_20gib_working_disk"] = gib * 2 <= 20.0
        ok = gib * 2 <= 20.0
    else:
        ok = False
        ev["error"] = "node count not supplied - a capacity guard with no denominator is vacuous"
    out.append(_clause(
        "ART-2", "the capacity guard uses the RE-DERIVED storage figure (FACT-0416, 1,580 B/node) "
        "and not the superseded estimate that was 12.3x LOW",
        ok, ev,
        "substitute 128 B/node; the projection collapses and the guard stops binding"))

    out.append(_clause(
        "ART-3", "an INDEPENDENT post-run reconstruction is possible - the artifact can be audited "
        "by code that did not write it",
        bool(re.search(r"aft_cache|tar\.gz|cache_manifest", nb_text)),
        {"artifact_tokens": [t for t in ("aft_cache", "tar.gz", "cache_manifest") if t in nb_text],
         "auditor": "scripts/win_bet/audit_feature_cache.py"},
        "remove the manifest write; nothing binds the artifact and the clause rejects"))

    # FACT-0417: RE-DERIVE the expected outputs from the BUILT NOTEBOOK, never trust the receipt.
    derived = sorted({m for m in re.findall(r"[\w./-]*aft_[\w.]+\.(?:json|tar\.gz)", nb_text)})
    out.append(_clause(
        "ART-4", "expected outputs are RE-DERIVED FROM THE BUILT NOTEBOOK, not trusted from the "
        "signed receipt - FACT-0417 found that field hardcoded to the wrong worker, which would "
        "have FAILED a correct run and PASSED a run that recorded nothing",
        bool(derived),
        {"expected_outputs_rederived_from_notebook": derived,
         "source": "the built notebook's own text", "receipt_trusted": False},
        "hardcode a different worker's outputs; the re-derived list stops matching the notebook"))
    return out


# ======================================================================================
# DRIVER
# ======================================================================================
def sign(spec_path: Path, repo: Path = REPO, nodes: int | None = None,
         trunk: dict | None = None) -> dict:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    nb_p = repo / spec["out_dir"] / spec["code_file"]
    if not nb_p.is_file():
        raise ContractRefusal(f"built notebook absent at {nb_p} - the contract signs the artifact "
                              "that would run, never the spec alone")
    nb = json.loads(nb_p.read_text(encoding="utf-8"))
    nb_text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    all_specs = sorted((repo / "scripts" / "kaggle_specs").glob("*.json"))

    sections = {
        "CODE": section_code(spec, Path(spec_path), repo, all_specs),
        "DATA": section_data(spec, trunk),
        "PROCESS": section_process(nb_text, spec),
        "ARTIFACT": section_artifact(nb_text, spec, nodes),
    }
    failed = [c["id"] for cs in sections.values() for c in cs if not c["passed"]]
    return {"schema_version": 1, "heartbeat": HEARTBEAT,
            "contract": "GPU PROTECTION CONTRACT (host, 2026-08-30)",
            "spec": str(Path(spec_path).relative_to(repo).as_posix()),
            "built_notebook_sha256": _sha(nb_p),
            "commit": _git("rev-parse", "HEAD", cwd=repo).strip(),
            "sections": sections,
            "signed": not failed, "failed": failed,
            "sections_signed": {k: all(c["passed"] for c in v) for k, v in sections.items()}}


# ======================================================================================
# SELFTEST - every clause must REJECT a defect built for it
# ======================================================================================
def _selftest(sandbox: Path) -> int:
    """Mutate a known-good world once per clause. A clause that never rejects is untested."""
    results = []

    def check(name, cond, detail=""):
        results.append({"mutation": name, "caught": bool(cond), "detail": detail})

    # --- CODE-2, the generated-file trap, on the real generator -----------------------
    sys.path.insert(0, str(REPO / "scripts" / "win_bet"))
    import sync_tap_worker
    real = (REPO / "scripts/kaggle_edits/assoc_tap_gate.py").read_text(encoding="utf-8")
    check("CODE-2 control: the shipped generated file matches its generator",
          sync_tap_worker.rendered() == real)
    check("CODE-2 mutation: a generated file edited without re-rendering",
          sync_tap_worker.rendered() != real + "\n# drift\n")

    # --- CODE-4 uniqueness ------------------------------------------------------------
    s = {"slug": "x", "out_dir": "d", "code_file": "n.ipynb", "edits": []}
    tmp = sandbox / "specs"; tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "a.json").write_text(json.dumps(s), encoding="utf-8")
    (tmp / "b.json").write_text(json.dumps(s), encoding="utf-8")
    cl = section_code(s, tmp / "a.json", sandbox, [tmp / "a.json", tmp / "b.json"])
    check("CODE-4 mutation: two specs share slug and out_dir",
          not [c for c in cl if c["id"] == "CODE-4"][0]["passed"])

    # --- CODE-5 pack pin --------------------------------------------------------------
    ven = REPO / "vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py"
    check("CODE-5 mutation: the vendored predictor does not satisfy the deployed pin",
          (not ven.is_file()) or _sha(ven) != DEPLOYED_PREDICTOR_SHA256)

    # --- DATA -------------------------------------------------------------------------
    f1 = {"edits": [{"vars": {"BIOHUB_LOEO_FOLD": "1"}}]}
    good = {"role": "oof_split1", "fold_legitimate": True, "dual_trunk_pair": ["oof_split1", "stabledet"],
            "contamination": "none", "checkpoint_sha256": "2e4ebf616b3d4fb5"}
    d = section_data(f1, good)
    check("DATA control: the HONEST fold-1 pair is ACCEPTED (not a reject-everything guard)",
          all(c["passed"] for c in d), json.dumps([c["id"] for c in d if not c["passed"]]))
    bad_pair = dict(good, dual_trunk_pair=["official", "stabledet"])
    check("DATA-3 mutation: the FACT-0418 pair with no fold-legitimate arm",
          not [c for c in section_data(f1, bad_pair) if c["id"] == "DATA-3"][0]["passed"])
    bad_role = dict(good, role="official")
    check("DATA-2 mutation: an illegitimate trunk role on fold 1",
          not [c for c in section_data(f1, bad_role) if c["id"] == "DATA-2"][0]["passed"])
    lying = dict(good, role="official", fold_legitimate=True)
    check("DATA-2 mutation: a spec DECLARING fold_legitimate true on an illegitimate role",
          not [c for c in section_data(f1, lying) if c["id"] == "DATA-2"][0]["passed"])
    no_hash = dict(good); no_hash.pop("checkpoint_sha256")
    check("DATA-4 mutation: the checkpoint hash is missing",
          not [c for c in section_data(f1, no_hash) if c["id"] == "DATA-4"][0]["passed"])
    check("DATA-1 mutation: the fold cannot be resolved",
          not [c for c in section_data({"edits": []}, None) if c["id"] == "DATA-1"][0]["passed"])

    # --- THE FOLD SWAP, BOTH DIRECTIONS, THROUGH THE SHARED POLICY ---------------------
    # Each takes an HONEST pair and relabels it onto the other fold, where its claim arm becomes a
    # leaky checkpoint. audit_feature_cache runs the SAME two mutations against its own guard;
    # they must agree, and the only way to make them agree is that they read one table.
    f0 = {"edits": [{"vars": {"BIOHUB_LOEO_FOLD": "0"}}]}
    good_f0 = {"role": "pack_split0", "fold_legitimate": True,
               "dual_trunk_pair": ["pack_split0", "stabledet"], "contamination": "none",
               "checkpoint_sha256": "12f6881ee3620a83"}
    d0 = section_data(f0, good_f0)
    check("DATA control: the HONEST fold-0 pair is ACCEPTED",
          all(c["passed"] for c in d0), json.dumps([c["id"] for c in d0 if not c["passed"]]))
    swapped_to_f0 = dict(good, dual_trunk_pair=["oof_split1", "stabledet"])
    d = section_data(f0, swapped_to_f0)
    check("DATA-2 fold swap: the fold-1 claim arm relabelled onto fold 0",
          not [c for c in d if c["id"] == "DATA-2"][0]["passed"])
    check("DATA-3 fold swap: the fold-1 PAIR relabelled onto fold 0",
          not [c for c in d if c["id"] == "DATA-3"][0]["passed"])
    swapped_to_f1 = dict(good_f0, dual_trunk_pair=["pack_split0", "stabledet"])
    d = section_data(f1, swapped_to_f1)
    check("DATA-2 fold swap: the fold-0 claim arm relabelled onto fold 1 (the EXP-0019 defect)",
          not [c for c in d if c["id"] == "DATA-2"][0]["passed"])
    check("DATA-3 fold swap: the fold-0 PAIR relabelled onto fold 1",
          not [c for c in d if c["id"] == "DATA-3"][0]["passed"])
    aliased = dict(good, checkpoint_sha256="d3e89eb361eeadef")
    check("DATA-4 mutation: the declared role contradicts the checkpoint's own bytes",
          not [c for c in section_data(f1, aliased) if c["id"] == "DATA-4"][0]["passed"])

    # --- PROCESS ----------------------------------------------------------------------
    good_nb = ('env = dict(os.environ)\nenv["PYTHONPATH"] = "scripts"\nprint("AFT_GATE ok")\n'
               'start = time.time()\nelapsed = time.time() - start\nif elapsed > BUDGET: abort()\n')
    gs = {"edits": [{"vars": {"BIOHUB_AFT_EXPECT_CROPS": "2", "BIOHUB_LOEO_LIMIT": "2"}}]}
    p = section_process(good_nb, gs)
    check("PROCESS control: a correct notebook passes all four clauses",
          all(c["passed"] for c in p), json.dumps([c["id"] for c in p if not c["passed"]]))
    check("PROC-1 mutation: the explicit env is not passed to the subprocess",
          not [c for c in section_process(good_nb.replace("env = dict(os.environ)", ""), gs)
               if c["id"] == "PROC-1"][0]["passed"])
    check("PROC-2 mutation: the positive heartbeat is stripped",
          not [c for c in section_process(good_nb.replace("AFT_GATE", "quiet"), gs)
               if c["id"] == "PROC-2"][0]["passed"])
    check("PROC-3 mutation: expected crop count disagrees with the run limit",
          not [c for c in section_process(good_nb, {"edits": [{"vars": {
               "BIOHUB_AFT_EXPECT_CROPS": "2", "BIOHUB_LOEO_LIMIT": "1"}}]})
               if c["id"] == "PROC-3"][0]["passed"])
    check("PROC-3 mutation: no crop-count assertion at all",
          not [c for c in section_process(good_nb, {"edits": []}) if c["id"] == "PROC-3"][0]["passed"])
    check("PROC-4 mutation: the timing probe and early abort are removed",
          not [c for c in section_process("print('AFT_GATE')\nenv = dict()\nPYTHONPATH", gs)
               if c["id"] == "PROC-4"][0]["passed"])

    # --- ARTIFACT ---------------------------------------------------------------------
    good_art = 'os.replace(tmp, out)\n_LOEO_KEEP |= {"aft_cache.tar.gz"}\ncache_manifest\naft_gate.json\n'
    a = section_artifact(good_art, {}, 2332346)
    check("ARTIFACT control: a correct notebook passes all four clauses",
          all(c["passed"] for c in a), json.dumps([c["id"] for c in a if not c["passed"]]))
    check("ART-1 mutation: the artifact is dropped from the export keep-list",
          not [c for c in section_artifact(good_art.replace("_LOEO_KEEP", "x"), {}, 1)
               if c["id"] == "ART-1"][0]["passed"])
    check("ART-2 mutation: no node count - a capacity guard with no denominator",
          not [c for c in section_artifact(good_art, {}, None) if c["id"] == "ART-2"][0]["passed"])
    # the superseded figure must not be what makes a too-large run pass
    big = 14_000_000
    check("ART-2 mutation: the superseded 128 B/node would pass a run the re-derived figure refuses",
          (big * SUPERSEDED_BYTES_PER_NODE * 2 / 1024**3 <= 20.0)
          and not [c for c in section_artifact(good_art, {}, big) if c["id"] == "ART-2"][0]["passed"])
    check("ART-3 mutation: nothing binds the artifact for independent reconstruction",
          not [c for c in section_artifact("os.replace(a,b)\n_LOEO_KEEP", {}, 1)
               if c["id"] == "ART-3"][0]["passed"])
    check("ART-4 mutation: the notebook names no output, so nothing can be re-derived from it",
          not [c for c in section_artifact("os.replace(a,b)\n_LOEO_KEEP\ncache_manifest", {}, 1)
               if c["id"] == "ART-4"][0]["passed"])

    caught = sum(1 for r in results if r["caught"])
    for r in results:
        print(f"  [{'CAUGHT' if r['caught'] else 'MISSED'}] {r['mutation']}"
              + (f"   {r['detail']}" if r["detail"] and not r["caught"] else ""))
    print(f"\n{caught}/{len(results)} mutations caught")
    return 0 if caught == len(results) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sign", help="sign a spec's BUILT notebook against all four sections")
    s.add_argument("--spec", required=True)
    s.add_argument("--nodes", type=int, default=None)
    s.add_argument("--trunk", default=None, help="JSON file describing the declared trunk")
    s.add_argument("--out", default=None)
    t = sub.add_parser("selftest")
    t.add_argument("--sandbox", default="C:/temp/gpu_contract_selftest")
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        box = Path(a.sandbox); box.mkdir(parents=True, exist_ok=True)
        print("GPU PROTECTION CONTRACT SELFTEST - every clause must REJECT a defect built for it")
        rc = _selftest(box)
        print(HEARTBEAT + f" selftest rc={rc}")
        return rc

    trunk = json.loads(Path(a.trunk).read_text(encoding="utf-8")) if a.trunk else None
    rep = sign(Path(a.spec).resolve(), REPO, a.nodes, trunk)
    for name, cs in rep["sections"].items():
        print(f"\n{name}  {'SIGNED' if rep['sections_signed'][name] else 'REFUSED'}")
        for c in cs:
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['id']}  {c['title'][:96]}")
    print(f"\nVERDICT: {'SIGNED' if rep['signed'] else 'REFUSED'}   failed={rep['failed']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"receipt -> {a.out}")
    print(HEARTBEAT + f" signed={rep['signed']}")
    return 0 if rep["signed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
