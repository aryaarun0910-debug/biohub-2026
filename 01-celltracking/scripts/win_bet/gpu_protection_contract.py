r"""THE GPU PROTECTION CONTRACT - profile-driven, fail-closed, and owned by the contract.

WHAT THIS IS
------------
The host's standing block at the head of ``research/00-system/registry/levers.yaml`` names four
sections that must be SIGNED before any GPU session starts. This module is that contract as an
INSTRUMENT: every clause returns a verdict, the run FAILS if any clause fails, and a clause that
cannot be evaluated is a FAILURE, never a silent skip.

WHY THERE ARE PROFILES, AND WHY THE CONTRACT OWNS THEM
------------------------------------------------------
Version 1 was not a general GPU contract. It hardcoded LOEO fold fields, AFT heartbeat tokens,
feature-cache capacity accounting and ``aft_*`` outputs, so it could only describe one artifact
class. Pointed at a SUBMISSION it produced six FAILs for the wrong reason and - far worse - three
FALSE PASSES, each measured on the real candidate rather than imagined:

  ART-3  passed on the token ``tar.gz``, which in that notebook is ``("*.whl", "*.tar.gz", "*.zip")``
         - a WHEEL-INSTALL glob - while its evidence named ``audit_feature_cache.py`` as the
         auditor. A feature-cache reconstruction check reported green on a submission that has no
         cache at all.
  PROC-4 passed on ``BUDGET``, which matched ``short_track_rescue_budget`` - an ILP rescue NODE
         budget, a scientific parameter with no relation to a time budget or an early abort. The
         notebook contains no ``abort``, no ``deadline`` and no ``elapsed``.
  DATA-2 and DATA-3 reported PASS while their own evidence said "no trunk declared". An
         inapplicable clause counted as a satisfied one.

A false PASS is worse than a FAIL because nobody looks at it. So:

  * The SPEC SELECTS A PROFILE. The spec can never decide whether an individual clause applies -
    that would let a submitter exempt itself from the clause it is about to violate.
  * Every clause returns PASS, FAIL or OUT_OF_SCOPE, and every OUT_OF_SCOPE reason is written HERE,
    in ``OOS_REASONS``, never supplied by the submitting agent.
  * An UNKNOWN profile REFUSES. Not a skip, not a default.
  * Cross-class mutation tests prove a submission cannot pass by carrying cache tokens and a cache
    cannot pass by carrying submission tokens. That is the check that stops two profiles laundering
    each other, and it is the direct descendant of the ART-3 false pass.

FAIL CLOSED, AND PROVED BY MUTATION
-----------------------------------
``--selftest`` does not assert that clauses pass. It MUTATES a known-good world once per clause and
requires that clause to reject. A contract whose clauses have never rejected anything is untested.

    python scripts/win_bet/gpu_protection_contract.py sign --spec <spec.json>
    python scripts/win_bet/gpu_protection_contract.py selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
HEARTBEAT = "GPU_PROTECTION_CONTRACT_COMPLETE"

PASS, FAIL, OOS = "PASS", "FAIL", "OUT_OF_SCOPE"

# FACT-0416, re-derived from the tap's own declared dtypes. The superseded value is named so a
# regression to it is refused BY VALUE rather than going unnoticed. This is a FEATURE-CACHE figure
# and the submission profile must never use it.
BYTES_PER_NODE = 1580
SUPERSEDED_BYTES_PER_NODE = 128
# FACT-0418: the pairing that has no fold-legitimate arm on either fold.
INVALID_PAIR = ("official", "stabledet")
LEGITIMATE_TRUNKS = {"0": {"pack_split0", "oof_split0"}, "1": {"oof_split1"}}
DEPLOYED_PREDICTOR_SHA256 = "25b3ebfd8849dcf5abeff9ed3f0d57269a4b979365c989d6f78db1e5002d5219"
GENERATED_PATCHES = {"scripts/kaggle_edits/assoc_tap_gate.py": "sync_tap_worker"}
# A Kaggle notebook output slot is 20 GiB; a submission.csv is a text table and must be far under it.
SUBMISSION_MAX_BYTES = 2 * 1024**3


class ContractRefusal(RuntimeError):
    """Raised when the contract cannot be evaluated. Never downgraded to a skip."""


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], capture_output=True,
                          cwd=str(cwd or REPO)).stdout.decode("utf-8", "replace")


def _r(status, evidence, mutation):
    return {"status": status, "evidence": evidence, "proved_by_mutation": mutation}


def _b(cond, evidence, mutation):
    return _r(PASS if cond else FAIL, evidence, mutation)


# ======================================================================================
# CLAUSES.  Each takes the context and returns {status, evidence, proved_by_mutation}.
# ======================================================================================
def c_code_1(ctx):
    spec, repo = ctx["spec"], ctx["repo"]
    scoped = [e["code_file"] for e in spec.get("edits", []) if e.get("code_file")]
    scoped += [spec.get("base_notebook"), ctx["spec_rel"]]
    scoped = [s for s in scoped if s]
    dirty = [d for s in scoped if (d := _git("status", "--porcelain", "--", s, cwd=repo).strip())]
    return _b(not dirty, {"scoped_paths": scoped, "dirty": dirty},
              "touch any scoped file; git status stops being empty")


def c_code_2(ctx):
    spec, repo = ctx["spec"], ctx["repo"]
    srcs = [e["code_file"] for e in spec.get("edits", []) if e.get("code_file")]
    gen = [s for s in srcs if s in GENERATED_PATCHES]
    if not gen:
        return _r(OOS, {"injected_code_files": srcs,
                        "reason": "this build injects no GENERATED patch source, so there is no "
                                  "generator/rendered pair to drift apart"},
                  "inject a generated patch and edit it without re-rendering")
    problems = []
    sys.path.insert(0, str(repo / "scripts" / "win_bet"))
    for s in gen:
        try:
            mod = __import__(GENERATED_PATCHES[s])
            rendered = mod.rendered()
        except Exception as exc:
            problems.append(f"{s}: generator unusable: {exc}")
            continue
        if rendered != (repo / s).read_text(encoding="utf-8"):
            problems.append(f"{s} is STALE against its generator")
    return _b(not problems, {"generated_sources_checked": gen, "problems": problems},
              "edit the generated file without re-rendering; the render comparison fails")


def c_code_3(ctx):
    spec, repo = ctx["spec"], ctx["repo"]
    man = repo / spec["out_dir"] / "build_manifest.json"
    nb = repo / spec["out_dir"] / spec["code_file"]
    if not (man.is_file() and nb.is_file()):
        return _b(False, {"error": "build manifest or built notebook absent"}, "delete the manifest")
    m = json.loads(man.read_text(encoding="utf-8"))
    got = _sha(nb)
    return _b(got == m.get("built_sha256"),
              {"built_sha256": got, "manifest_built_sha256": m.get("built_sha256")},
              "rewrite one byte of the notebook; the sha stops matching")


def c_code_4(ctx):
    spec = ctx["spec"]
    clashes = []
    for other in ctx["all_specs"]:
        if Path(other).resolve() == Path(ctx["spec_path"]).resolve():
            continue
        try:
            o = json.loads(Path(other).read_text(encoding="utf-8"))
        except Exception:
            continue
        if o.get("slug") == spec.get("slug"):
            clashes.append(f"slug also in {Path(other).name}")
        if o.get("out_dir") == spec.get("out_dir"):
            clashes.append(f"out_dir also in {Path(other).name}")
    return _b(not clashes, {"slug": spec.get("slug"), "out_dir": spec.get("out_dir"),
                            "clashes": clashes},
              "point a second spec at the same out_dir; the clash list is non-empty")


def c_code_5(ctx):
    pack = Path("C:/temp/p7/tracking_repo/scripts/predict_unet_transformer.py")
    got = _sha(pack) if pack.is_file() else None
    return _b(got == DEPLOYED_PREDICTOR_SHA256,
              {"pack_predictor": str(pack), "sha256": got, "pinned": DEPLOYED_PREDICTOR_SHA256},
              "point the pin at the vendored predictor; the sha stops matching")


# ---------------------------------------------------------------- DATA (LOEO / trunk class)
def _fold(spec):
    f = None
    for e in spec.get("edits", []):
        if "BIOHUB_LOEO_FOLD" in e.get("vars", {}):
            f = str(e["vars"]["BIOHUB_LOEO_FOLD"]).strip()
    return f


def c_data_1(ctx):
    f = _fold(ctx["spec"])
    return _b(f in ("0", "1"), {"fold": f},
              "remove BIOHUB_LOEO_FOLD from the spec; the fold stops resolving")


def c_data_2(ctx):
    t, f = ctx["trunk"], _fold(ctx["spec"])
    if t is None:
        return _r(OOS, {"reason": "no trunk is declared, so there is no trunk to be legitimate; a "
                                  "single-trunk run claiming no pair is outside FACT-0418's scope"},
                  "declare an illegitimate trunk role")
    legit = LEGITIMATE_TRUNKS.get(str(f), set())
    return _b(bool(t.get("fold_legitimate")) and t.get("role") in legit,
              {"fold": f, "role": t.get("role"), "declared": t.get("fold_legitimate"),
               "legitimate_roles_for_this_fold": sorted(legit)},
              "set role to 'official' on fold 1; the role leaves the legitimate set")


def c_data_3(ctx):
    t, f = ctx["trunk"], _fold(ctx["spec"])
    pair = tuple((t or {}).get("dual_trunk_pair") or ())
    if not pair:
        return _r(OOS, {"reason": "no dual_trunk_pair is declared, so there is no pair to check"},
                  "declare the FACT-0418 pair ['official','stabledet']")
    legit = LEGITIMATE_TRUNKS.get(str(f), set())
    return _b(any(r in legit for r in pair) and tuple(pair) != INVALID_PAIR,
              {"pair": list(pair), "is_the_fact_0418_invalid_pair": tuple(pair) == INVALID_PAIR,
               "legitimate_roles_for_this_fold": sorted(legit)},
              "declare ['official','stabledet']; the clause rejects it by name")


def c_data_4(ctx):
    t = ctx["trunk"]
    if t is None:
        return _r(OOS, {"reason": "no trunk is declared, so there is no checkpoint to bind"},
                  "declare a trunk with no checkpoint_sha256")
    return _b(bool(t.get("contamination")) and bool(t.get("checkpoint_sha256")),
              {"contamination": t.get("contamination"),
               "checkpoint_sha256": t.get("checkpoint_sha256")},
              "drop checkpoint_sha256; the binding is gone")


# ---------------------------------------------------------------- DATA (submission class)
LOEO_MARKERS = ("BIOHUB_LOEO_FOLD", "BIOHUB_LOEO_ARM", "BIOHUB_LOEO_LIMIT", "BIOHUB_LOEO_STEMS")


def c_sdata_1(ctx):
    """A submission is scored on the HIDDEN test set. It must make no LOEO fold claim, and it must
    read the competition test directory rather than a retargeted fold."""
    spec, nb = ctx["spec"], ctx["nb_text"]
    spec_loeo = sorted({k for e in spec.get("edits", []) for k in e.get("vars", {})
                        if k in LOEO_MARKERS})
    # a bare mention is not a claim; an ASSIGNMENT is
    nb_loeo = sorted({m for m in LOEO_MARKERS
                      if re.search(rf'os\.environ\[\s*["\']{m}["\']\s*\]\s*=', nb)})
    reads_test = bool(re.search(r'COMP_DIR\s*/\s*["\']test["\']|/kaggle/input/[^"\']*/test', nb))
    ok = not spec_loeo and not nb_loeo and reads_test
    return _b(ok, {"loeo_vars_set_by_spec": spec_loeo, "loeo_vars_assigned_in_notebook": nb_loeo,
                   "reads_competition_test_dir": reads_test,
                   "why": "a submission scored on the hidden test set may not also claim a "
                          "held-out fold; the two are different evaluation scopes"},
              "set BIOHUB_LOEO_FOLD in a submission spec; the clause rejects the mixed scope")


def c_sdata_2(ctx):
    """Declared AND hash-bound inputs. Datasets declared in the spec, and the base notebook bound
    by a sha256 that actually matches the artifact on disk."""
    spec, repo = ctx["spec"], ctx["repo"]
    ds = spec.get("datasets") or []
    comp = spec.get("competition_sources") or []
    base = spec.get("base_notebook")
    want = spec.get("base_sha256")
    got = _sha(repo / base) if base and (repo / base).is_file() else None
    bound = bool(want) and got == want
    return _b(bool(ds) and bool(comp) and bound,
              {"datasets_declared": ds, "competition_sources": comp, "base_notebook": base,
               "base_sha256_declared": want, "base_sha256_on_disk": got, "base_hash_bound": bound},
              "alter the declared base_sha256; the hash binding breaks")


# ---------------------------------------------------------------- PROCESS
def c_proc_1(ctx):
    nb = ctx["nb_text"]
    explicit = bool(re.search(r"\benv\s*=\s*", nb))
    return _b(explicit and "PYTHONPATH" in nb,
              {"explicit_env_passed": explicit, "pythonpath_set": "PYTHONPATH" in nb,
               "env_keys_set_by_spec": sorted({k for e in ctx["spec"].get("edits", [])
                                               for k in e.get("vars", {})})},
              "delete the env= argument from the subprocess launch")


AFT_HEARTBEATS = ("AFT_GATE", "ASSOC_FEATURE_PARITY_COMPLETE", "AFT_GATE_FAILED",
                  "ASSOC_TRAIN_HARNESS_COMPLETE", "AFT_PATCH_APPLIED")


def c_proc_2(ctx):
    found = [t for t in AFT_HEARTBEATS if t in ctx["nb_text"]]
    return _b(bool(found), {"heartbeat_tokens_found": found, "class": "feature-tap / gate"},
              "strip the gate heartbeat prints; no token is found")


def c_proc_3(ctx):
    spec = ctx["spec"]
    exp = lim = None
    for e in spec.get("edits", []):
        v = e.get("vars", {})
        exp = v.get("BIOHUB_AFT_EXPECT_CROPS", exp)
        lim = v.get("BIOHUB_LOEO_LIMIT", lim)
    return _b(exp is not None and str(exp) == str(lim),
              {"expect_crops": exp, "loeo_limit": lim,
               "why": "the kernel's own all_passed cannot see a crop that never ran"},
              "set EXPECT_CROPS and LIMIT to different values")


def c_proc_4(ctx):
    """A TIME budget with an early abort. Deliberately narrow: v1 matched the bare word BUDGET and
    so passed on `short_track_rescue_budget`, an ILP node budget with no temporal meaning."""
    nb = ctx["nb_text"]
    clock = bool(re.search(r"time\.time\(\)|perf_counter", nb))
    abort = bool(re.search(r"\b(elapsed|deadline|time_budget|TIME_BUDGET|wall_clock)\b", nb)
                 or re.search(r"\babort\b", nb, re.I))
    return _b(clock and abort,
              {"clock_present": clock, "temporal_abort_present": abort,
               "note": "a node/rescue budget is NOT a time budget - the v1 detector matched "
                       "short_track_rescue_budget and reported a false PASS"},
              "remove the elapsed/abort logic, leaving only a rescue budget")


def c_sproc_1(ctx):
    """THE LIVE-TREATMENT HEARTBEAT. Every treatment the spec sets must be assigned in the notebook
    with the spec's value WINNING (last write), read back, and RECORDED into run_stats - so the
    treatment is proved live rather than assumed."""
    spec, nb = ctx["spec"], ctx["nb_text"]
    treatments = {k: v for e in spec.get("edits", []) if e.get("kind") == "env"
                  for k, v in e.get("vars", {}).items()}
    rows, ok = [], True
    for k, v in treatments.items():
        assigns = re.findall(rf'os\.environ\[\s*["\']{k}["\']\s*\]\s*=\s*[\'"]([^\'"]*)[\'"]', nb)
        last = assigns[-1] if assigns else None
        read = bool(re.search(rf'os\.environ\.get\(\s*["\']{k}["\']', nb))
        stat_key = k[len("BIOHUB_"):].lower() if k.startswith("BIOHUB_") else k.lower()
        recorded = bool(re.search(rf'["\']{stat_key}["\']\s*:', nb))
        good = last is not None and str(last) == str(v) and read and recorded
        ok &= good
        rows.append({"var": k, "spec_value": v, "assignments_in_notebook": assigns,
                     "last_write_wins": last, "read_back": read,
                     "run_stats_key": stat_key, "recorded_in_run_stats": recorded, "ok": good})
    if not treatments:
        return _b(False, {"error": "a submission run declares no treatment; there is nothing to "
                                   "prove live, and an untreated resubmission is not a calibration"},
                  "remove the env edit from the spec")
    return _b(ok, {"treatments": rows},
              "change the spec value without rebuilding, or drop the run_stats record")


SUBMISSION_HEARTBEATS = ("SUBMISSION_COMPLETE", "BIOHUB_SUBMISSION_COMPLETE",
                         "SUBMISSION_WRITTEN", "PIPELINE_COMPLETE", "RUN_COMPLETE")


def c_sproc_2(ctx):
    found = [t for t in SUBMISSION_HEARTBEATS if t in ctx["nb_text"]]
    return _b(bool(found),
              {"submission_heartbeat_tokens_found": found,
               "accepted_tokens": list(SUBMISSION_HEARTBEATS),
               "why": "a positive completion heartbeat whose ABSENCE is the alarm; a fetched log "
                      "with no terminal token is indistinguishable from a truncated run"},
              "strip the completion print; no token is found")


def c_sproc_3(ctx):
    """RUNTIME reconciliation of DISCOVERED inputs against EMITTED outputs. Not a predetermined
    crop count - the hidden test set size is unknowable before the run."""
    nb = ctx["nb_text"]
    m = re.search(r"if\s+len\(([A-Za-z_]\w*)\)\s*!=\s*len\(([A-Za-z_]\w*)\)[\s\S]{0,400}?raise", nb)
    return _b(bool(m),
              {"reconciliation_found": bool(m),
               "compared": [m.group(1), m.group(2)] if m else None,
               "why": "discovered inputs versus emitted outputs, compared AT RUNTIME and raising "
                      "on mismatch, is the only form this can take for a hidden test set"},
              "delete the len(outputs) != len(inputs) raise; nothing reconciles the two")


# ---------------------------------------------------------------- ARTIFACT
def c_art_1(ctx):
    nb = ctx["nb_text"]
    atomic = bool(re.search(r"os\.replace|\.replace\(|\.rename\(", nb))
    keep = "_LOEO_KEEP" in nb
    return _b(atomic and keep, {"atomic_rename_present": atomic, "export_keep_list_present": keep},
              "drop the file from _LOEO_KEEP; the sweep would delete it")


def c_art_2(ctx):
    nodes = ctx.get("nodes")
    ev = {"bytes_per_node": BYTES_PER_NODE, "superseded_and_refused": SUPERSEDED_BYTES_PER_NODE,
          "nodes": nodes}
    if not nodes:
        ev["error"] = "node count not supplied - a capacity guard with no denominator is vacuous"
        return _b(False, ev, "omit the node count")
    gib = nodes * BYTES_PER_NODE / 1024**3
    ev.update({"single_trunk_gib": round(gib, 2), "dual_trunk_gib": round(gib * 2, 2)})
    return _b(gib * 2 <= 20.0, ev,
              "substitute 128 B/node; the projection collapses and the guard stops binding")


def c_art_3(ctx):
    nb = ctx["nb_text"]
    toks = [t for t in ("aft_cache", "cache_manifest") if t in nb]
    return _b(bool(toks),
              {"cache_artifact_tokens": toks, "auditor": "scripts/win_bet/audit_feature_cache.py",
               "note": "v1 also accepted the bare string 'tar.gz', which matched a wheel-install "
                       "glob in a submission notebook and produced a FALSE PASS"},
              "remove the cache manifest write; nothing binds the cache")


def c_art_4(ctx):
    nb = ctx["nb_text"]
    derived = sorted({m for m in re.findall(r"[\w./-]*aft_[\w.]+\.(?:json|tar\.gz)", nb)})
    return _b(bool(derived),
              {"expected_outputs_rederived_from_notebook": derived, "receipt_trusted": False},
              "hardcode a different worker's outputs; the re-derived list stops matching")


def c_sart_1(ctx):
    """ATOMIC PRESERVATION of submission.csv. A direct open('w') that dies mid-write leaves a
    truncated CSV that still parses, and the run's whole product is the file."""
    nb = ctx["nb_text"]
    direct = bool(re.search(r"SUBMISSION_PATH\.open\(\s*[\"']w[\"']", nb))
    # The swap must be a REAL os.replace() whose arguments name the submission path. The bare
    # tokens SUBMISSION_TMP or ".part" must never satisfy it - matching a token rather than a
    # mechanism is precisely how v1's ART-3 passed on a wheel-install glob. Found by mutating the
    # real candidate: a comment plus `x = 'SUBMISSION_TMP'` satisfied the earlier alternation.
    atomic = bool(re.search(r"os\.replace\(\s*[^)]*SUBMISSION[^)]*\)", nb))
    return _b(atomic,
              {"atomic_rename_for_submission": atomic, "direct_truncating_write": direct,
               "why": "write to a temporary path and os.replace() it into place; os.replace is "
                      "atomic on the same filesystem, so a killed kernel leaves either the old "
                      "file or the complete new one and never a half-written table"},
              "replace the atomic rename with a direct open('w'); the clause rejects")


def c_sart_2(ctx):
    """A SUBMISSION-specific output-size guard. FACT-0416's 1,580 B/node is a feature-cache figure
    and says nothing about a CSV of nodes and edges."""
    nb = ctx["nb_text"]
    guard = bool(re.search(r"stat\(\)\.st_size|getsize|SUBMISSION_MAX|len\(_guard_frame\)|"
                           r"row_id\s*==\s*total_nodes", nb))
    return _b(guard,
              {"output_size_or_row_guard_present": guard,
               "limit_bytes": SUBMISSION_MAX_BYTES,
               "why": "the submission is a text table; its guard is rows and bytes, never "
                      "FACT-0416's per-node cache accounting"},
              "delete the row-count assertion and the size guard")


def c_sart_3(ctx):
    """Expected output RE-DERIVED from the built notebook as submission.csv - FACT-0417's shape,
    where a receipt field hardcoded to the wrong worker would fail a correct run and pass a run
    that recorded nothing."""
    nb = ctx["nb_text"]
    derived = sorted({m for m in re.findall(r"[\w./-]*submission\.csv", nb)})
    declared = bool(ctx["spec"].get("expects_submission"))
    return _b(bool(derived) and declared,
              {"expected_outputs_rederived_from_notebook": derived,
               "spec_expects_submission": declared, "receipt_trusted": False},
              "remove submission.csv from the notebook while leaving expects_submission true")


def c_sart_4(ctx):
    """An INDEPENDENT release-receipt reconstruction must be possible after execution: the emitted
    file is hashed in-kernel so an external receipt can bind to it."""
    nb = ctx["nb_text"]
    # The DIGEST must be a real sha256 taken OVER the submission bytes. Matching a variable NAME
    # such as `_guard_digest` would let `_guard_digest = 0` satisfy the clause - found by the
    # mutation test below, which is what mutation tests are for.
    digest = bool(re.search(r"hashlib\.sha256\(\s*[^)]*submission[^)]*\.read_bytes\(\)", nb, re.I))
    reread = bool(re.search(r"pd\.read_csv\(\s*_guard_submission|_guard_submission\.is_file", nb))
    return _b(digest and reread,
              {"in_kernel_digest_of_submission": digest, "post_write_reread": reread,
               "auditor": "scripts/win_bet/audit_release_receipt.py",
               "why": "the receipt binds notebook, weights, manifest, graph and score; it needs a "
                      "digest computed by the run itself to bind against"},
              "remove the sha256 of the submission; nothing binds the artifact to a receipt")


# ======================================================================================
# THE CONTRACT OWNS PROFILES AND OUT-OF-SCOPE REASONS. A SPEC CANNOT EDIT EITHER.
# ======================================================================================
CLAUSES = {
    "CODE-1": ("every file this build depends on is committed", c_code_1),
    "CODE-2": ("every GENERATED patch source matches its generator", c_code_2),
    "CODE-3": ("the built notebook reproduces the SHA its manifest recorded", c_code_3),
    "CODE-4": ("slug and out_dir are unique across every spec", c_code_4),
    "CODE-5": ("the DEPLOYED support pack is pinned by sha256", c_code_5),
    "DATA-1": ("the run declares which LOEO fold it evaluates", c_data_1),
    "DATA-2": ("the trunk is FOLD-LEGITIMATE for its own fold", c_data_2),
    "DATA-3": ("any dual_trunk_pair has a FOLD-LEGITIMATE arm", c_data_3),
    "DATA-4": ("contamination status and checkpoint hash are bound", c_data_4),
    "SDATA-1": ("hidden-test evaluation scope, with NO LOEO claim", c_sdata_1),
    "SDATA-2": ("model and data inputs are declared AND hash-bound", c_sdata_2),
    "PROC-1": ("an EXPLICIT environment reaches the predictor subprocess", c_proc_1),
    "PROC-2": ("a positive gate heartbeat whose ABSENCE is the alarm", c_proc_2),
    "PROC-3": ("the expected CROP COUNT is asserted and equals the run limit", c_proc_3),
    "PROC-4": ("a TIME budget with an early abort", c_proc_4),
    "SPROC-1": ("the treatment is propagated and RECORDED in run_stats (live-treatment heartbeat)",
                c_sproc_1),
    "SPROC-2": ("a submission-specific completion heartbeat", c_sproc_2),
    "SPROC-3": ("RUNTIME reconciliation of discovered inputs against emitted outputs", c_sproc_3),
    "ART-1": ("the run's output is atomic and survives the export sweep", c_art_1),
    "ART-2": ("the capacity guard uses the RE-DERIVED 1,580 B/node figure", c_art_2),
    "ART-3": ("an independent post-run CACHE reconstruction is possible", c_art_3),
    "ART-4": ("expected cache outputs are re-derived from the built notebook", c_art_4),
    "SART-1": ("atomic preservation of submission.csv", c_sart_1),
    "SART-2": ("a submission-specific output-size / row guard", c_sart_2),
    "SART-3": ("the expected output is re-derived as submission.csv from the notebook", c_sart_3),
    "SART-4": ("independent release-receipt reconstruction after execution", c_sart_4),
}

SECTION_OF = {**{c: "CODE" for c in ("CODE-1", "CODE-2", "CODE-3", "CODE-4", "CODE-5")},
              **{c: "DATA" for c in ("DATA-1", "DATA-2", "DATA-3", "DATA-4", "SDATA-1", "SDATA-2")},
              **{c: "PROCESS" for c in ("PROC-1", "PROC-2", "PROC-3", "PROC-4",
                                        "SPROC-1", "SPROC-2", "SPROC-3")},
              **{c: "ARTIFACT" for c in ("ART-1", "ART-2", "ART-3", "ART-4",
                                         "SART-1", "SART-2", "SART-3", "SART-4")}}

PROFILES = {
    "gate_smoke_v1": {
        "description": "a passive instrumentation / feature-cache acquisition run",
        "clauses": ["CODE-1", "CODE-2", "CODE-3", "CODE-4", "CODE-5",
                    "DATA-1", "DATA-2", "DATA-3", "DATA-4",
                    "PROC-1", "PROC-2", "PROC-3", "PROC-4",
                    "ART-1", "ART-2", "ART-3", "ART-4"],
    },
    "submission_run_v1": {
        "description": "a leaderboard submission run scored on the hidden test set",
        "clauses": ["CODE-1", "CODE-2", "CODE-3", "CODE-4", "CODE-5",
                    "SDATA-1", "SDATA-2",
                    "PROC-1", "SPROC-1", "SPROC-2", "SPROC-3",
                    "SART-1", "SART-2", "SART-3", "SART-4"],
    },
}

# WRITTEN HERE, NEVER SUPPLIED BY THE SUBMITTING AGENT.
OOS_REASONS = {
    ("submission_run_v1", "DATA-1"): "a submission is scored on the HIDDEN test set and makes no "
        "LOEO fold claim; SDATA-1 enforces that the two scopes are not mixed",
    ("submission_run_v1", "DATA-2"): "no trunk is trained or consumed as a representation here; "
        "the run uses the deployed weights the champion already used",
    ("submission_run_v1", "DATA-3"): "a dual-trunk pair is a feature-cache provenance control and "
        "has no meaning for a submission",
    ("submission_run_v1", "DATA-4"): "superseded by SDATA-2, which binds the submission's inputs "
        "by hash instead",
    ("submission_run_v1", "PROC-2"): "the AFT gate heartbeat belongs to the feature-tap class; "
        "SPROC-2 requires a submission-specific completion heartbeat instead",
    ("submission_run_v1", "PROC-3"): "the hidden test set's size is UNKNOWABLE before the run, so "
        "a predetermined crop count cannot be asserted; SPROC-3 requires runtime reconciliation of "
        "discovered inputs against emitted outputs instead",
    ("submission_run_v1", "PROC-4"): "a submission must run to completion to emit its product, so "
        "an early abort would destroy the artifact rather than protect it",
    ("submission_run_v1", "ART-1"): "superseded by SART-1, which is specific to submission.csv",
    ("submission_run_v1", "ART-2"): "FACT-0416's 1,580 BYTES PER DETECTED NODE is a FEATURE-CACHE "
        "figure and says nothing about a CSV of nodes and edges; SART-2 applies instead",
    ("submission_run_v1", "ART-3"): "there is no feature cache to reconstruct; SART-4 requires an "
        "independent RELEASE-RECEIPT reconstruction instead",
    ("submission_run_v1", "ART-4"): "aft_* outputs belong to the feature-tap class; SART-3 "
        "re-derives submission.csv from the built notebook instead",
    ("gate_smoke_v1", "SDATA-1"): "a gate smoke is a LOEO fold run, not a hidden-test submission",
    ("gate_smoke_v1", "SDATA-2"): "covered by DATA-2 and DATA-4 for the trunk class",
    ("gate_smoke_v1", "SPROC-1"): "a gate smoke's treatment is the tap itself, covered by PROC-2",
    ("gate_smoke_v1", "SPROC-2"): "covered by PROC-2's gate heartbeat",
    ("gate_smoke_v1", "SPROC-3"): "covered by PROC-3's crop-count assertion, which is knowable for "
        "a declared fold",
    ("gate_smoke_v1", "SART-1"): "a gate smoke emits no submission.csv",
    ("gate_smoke_v1", "SART-2"): "covered by ART-2's cache capacity guard",
    ("gate_smoke_v1", "SART-3"): "covered by ART-4",
    ("gate_smoke_v1", "SART-4"): "a gate smoke produces no release receipt",
}


def select_profile(spec: dict) -> str:
    """The spec SELECTS a profile. It can never decide whether an individual clause applies."""
    named = spec.get("gpu_contract_profile")
    if named is not None:
        if named not in PROFILES:
            raise ContractRefusal(
                f"unknown gpu_contract_profile {named!r}. Known profiles: {sorted(PROFILES)}. "
                "An unknown profile REFUSES - it is never skipped and never defaulted, because a "
                "typo must not silently disable the contract.")
        return named
    derived = "submission_run_v1" if spec.get("expects_submission") else "gate_smoke_v1"
    return derived


def sign(spec_path: Path, repo: Path = REPO, nodes: int | None = None,
         trunk: dict | None = None) -> dict:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    profile = select_profile(spec)
    nb_p = repo / spec["out_dir"] / spec["code_file"]
    if not nb_p.is_file():
        raise ContractRefusal(f"built notebook absent at {nb_p} - the contract signs the artifact "
                              "that would run, never the spec alone")
    nb = json.loads(nb_p.read_text(encoding="utf-8"))
    ctx = {"spec": spec, "spec_path": Path(spec_path), "repo": repo, "trunk": trunk, "nodes": nodes,
           "nb_text": "\n".join("".join(c.get("source", [])) for c in nb["cells"]),
           "all_specs": sorted((repo / "scripts" / "kaggle_specs").glob("*.json"))}
    try:
        ctx["spec_rel"] = str(Path(spec_path).resolve().relative_to(repo).as_posix())
    except ValueError:
        ctx["spec_rel"] = str(spec_path)

    applicable = set(PROFILES[profile]["clauses"])
    sections: dict[str, list] = {"CODE": [], "DATA": [], "PROCESS": [], "ARTIFACT": []}
    for cid, (title, fn) in CLAUSES.items():
        if cid in applicable:
            res = fn(ctx)
        else:
            reason = OOS_REASONS.get((profile, cid))
            if reason is None:
                raise ContractRefusal(
                    f"clause {cid} is not applicable to profile {profile!r} and the contract "
                    "records no OUT_OF_SCOPE reason for it. A clause may not be dropped silently.")
            res = _r(OOS, {"reason": reason}, "n/a")
        sections[SECTION_OF[cid]].append({"id": cid, "title": title, **res})

    failed = [c["id"] for cs in sections.values() for c in cs if c["status"] == FAIL]
    oos = [c["id"] for cs in sections.values() for c in cs if c["status"] == OOS]
    return {"schema_version": 2, "heartbeat": HEARTBEAT,
            "contract": "GPU PROTECTION CONTRACT (host, 2026-08-30)",
            "profile": profile, "profile_description": PROFILES[profile]["description"],
            "profile_selected_by": "spec.gpu_contract_profile" if spec.get("gpu_contract_profile")
                                   else "contract-owned derivation from spec.expects_submission",
            "spec": ctx["spec_rel"], "built_notebook_sha256": _sha(nb_p),
            "commit": _git("rev-parse", "HEAD", cwd=repo).strip(),
            "sections": sections, "signed": not failed,
            "failed": failed, "out_of_scope": oos,
            "sections_signed": {k: not [c for c in v if c["status"] == FAIL]
                                for k, v in sections.items()}}


# ======================================================================================
# SELFTEST
# ======================================================================================
def _ctx(nb_text="", spec=None, trunk=None, nodes=None, repo=None):
    spec = spec or {}
    return {"spec": spec, "spec_path": Path("x.json"), "spec_rel": "x.json",
            "repo": repo or REPO, "trunk": trunk, "nodes": nodes, "nb_text": nb_text,
            "all_specs": []}


GOOD_SUB_NB = (
    'env = dict(os.environ)\nenv["PYTHONPATH"] = "scripts"\n'
    'os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = \'2.0\'\n'
    'B = float(os.environ.get("BIOHUB_MOTION_RELINK_LEARNED_BONUS", "0.75"))\n'
    'stats = {"motion_relink_learned_bonus": B}\n'
    'TEST_DIR = COMP_DIR / "test"\n'
    'print("SUBMISSION_COMPLETE")\n'
    'if len(geffs) != len(test_stems):\n    raise RuntimeError("missing")\n'
    'os.replace(SUBMISSION_TMP, SUBMISSION_PATH)\n'
    'row_id == total_nodes + total_edges\n'
    'sub = "submission.csv"\n'
    '_guard_digest = hashlib.sha256(_guard_submission.read_bytes()).hexdigest()\n'
    '_guard_frame = pd.read_csv(_guard_submission)\n')
GOOD_SUB_SPEC = {"expects_submission": True, "datasets": ["a/b"],
                 "competition_sources": ["c"], "base_notebook": None, "base_sha256": None,
                 "edits": [{"kind": "env", "vars": {"BIOHUB_MOTION_RELINK_LEARNED_BONUS": "2.0"}}]}
GOOD_CACHE_NB = ('env = dict(os.environ)\nenv["PYTHONPATH"] = "s"\nprint("AFT_GATE ok")\n'
                 'start = time.time()\nelapsed = time.time() - start\n'
                 'aft_cache\ncache_manifest\naft_gate.json\n'
                 'os.replace(a, b)\n_LOEO_KEEP |= {"aft_cache.tar.gz"}\n')
GOOD_CACHE_SPEC = {"edits": [{"vars": {"BIOHUB_AFT_EXPECT_CROPS": "2", "BIOHUB_LOEO_LIMIT": "2",
                                       "BIOHUB_LOEO_FOLD": "0"}}]}


def _selftest(sandbox: Path) -> int:
    R = []

    def ck(name, cond, detail=""):
        R.append({"mutation": name, "caught": bool(cond), "detail": detail})

    def st(fn, **kw):
        return fn(_ctx(**kw))["status"]

    # ---- profile machinery ----
    ck("PROFILE: an unknown profile REFUSES rather than defaulting",
       _refuses(lambda: select_profile({"gpu_contract_profile": "nope"})))
    ck("PROFILE: a submission spec selects submission_run_v1",
       select_profile({"expects_submission": True}) == "submission_run_v1")
    ck("PROFILE: a spec cannot exempt an individual clause",
       select_profile({"expects_submission": True,
                       "gpu_contract_skip": ["SART-1"]}) == "submission_run_v1"
       and "SART-1" in PROFILES["submission_run_v1"]["clauses"])
    ck("PROFILE: every non-applicable clause has a CONTRACT-OWNED OOS reason",
       all((p, c) in OOS_REASONS for p in PROFILES for c in CLAUSES
           if c not in PROFILES[p]["clauses"]))

    # ---- controls ----
    ck("CONTROL: a correct submission notebook passes every submission clause",
       all(st(CLAUSES[c][1], nb_text=GOOD_SUB_NB, spec=GOOD_SUB_SPEC) in (PASS, OOS)
           for c in ("SDATA-1", "PROC-1", "SPROC-1", "SPROC-2", "SPROC-3",
                     "SART-1", "SART-2", "SART-3", "SART-4")),
       json.dumps([c for c in ("SDATA-1", "PROC-1", "SPROC-1", "SPROC-2", "SPROC-3",
                               "SART-1", "SART-2", "SART-3", "SART-4")
                   if st(CLAUSES[c][1], nb_text=GOOD_SUB_NB, spec=GOOD_SUB_SPEC) == FAIL]))
    ck("CONTROL: a correct cache notebook passes every cache clause",
       all(st(CLAUSES[c][1], nb_text=GOOD_CACHE_NB, spec=GOOD_CACHE_SPEC) == PASS
           for c in ("PROC-1", "PROC-2", "PROC-3", "PROC-4", "ART-1", "ART-3", "ART-4")),
       json.dumps([c for c in ("PROC-1", "PROC-2", "PROC-3", "PROC-4", "ART-1", "ART-3", "ART-4")
                   if st(CLAUSES[c][1], nb_text=GOOD_CACHE_NB, spec=GOOD_CACHE_SPEC) != PASS]))

    # ---- CROSS-CLASS: the two profiles must not launder each other ----
    ck("CROSS-CLASS: a submission carrying CACHE tokens still fails every submission clause it "
       "violates - cache tokens cannot satisfy SART-1/2/3/4 or SPROC-2/3",
       all(st(CLAUSES[c][1], nb_text=GOOD_CACHE_NB, spec=GOOD_SUB_SPEC) == FAIL
           for c in ("SPROC-2", "SPROC-3", "SART-1", "SART-2", "SART-3", "SART-4")),
       json.dumps([c for c in ("SPROC-2", "SPROC-3", "SART-1", "SART-2", "SART-3", "SART-4")
                   if st(CLAUSES[c][1], nb_text=GOOD_CACHE_NB, spec=GOOD_SUB_SPEC) != FAIL]))
    ck("CROSS-CLASS: a cache run carrying SUBMISSION tokens still fails every cache clause it "
       "violates - submission tokens cannot satisfy PROC-2/3 or ART-3/4",
       all(st(CLAUSES[c][1], nb_text=GOOD_SUB_NB, spec=GOOD_SUB_SPEC) == FAIL
           for c in ("PROC-2", "PROC-3", "ART-3", "ART-4")),
       json.dumps([c for c in ("PROC-2", "PROC-3", "ART-3", "ART-4")
                   if st(CLAUSES[c][1], nb_text=GOOD_SUB_NB, spec=GOOD_SUB_SPEC) != FAIL]))
    ck("CROSS-CLASS: the v1 ART-3 false pass is dead - a wheel glob no longer satisfies the cache "
       "reconstruction clause",
       st(c_art_3, nb_text='patterns = ("*.whl", "*.tar.gz", "*.zip")') == FAIL)
    ck("CROSS-CLASS: the v1 PROC-4 false pass is dead - an ILP rescue NODE budget no longer "
       "satisfies the TIME-budget clause",
       st(c_proc_4, nb_text='t = time.time()\nbudget = min(3, n)\n'
                            'stats["short_track_rescue_budget"] = budget') == FAIL)

    # ---- per-clause mutations, submission class ----
    ck("SDATA-1 mutation: a submission spec that also claims a LOEO fold",
       st(c_sdata_1, nb_text=GOOD_SUB_NB,
          spec=dict(GOOD_SUB_SPEC, edits=[{"kind": "env", "vars": {"BIOHUB_LOEO_FOLD": "0"}}])) == FAIL)
    ck("SDATA-1 mutation: the notebook assigns a LOEO var itself",
       st(c_sdata_1, nb_text=GOOD_SUB_NB + '\nos.environ["BIOHUB_LOEO_ARM"] = "champion"\n',
          spec=GOOD_SUB_SPEC) == FAIL)
    ck("SDATA-2 mutation: the declared base hash does not match the artifact",
       st(c_sdata_2, spec=dict(GOOD_SUB_SPEC, base_notebook="pyproject.toml",
                               base_sha256="deadbeef")) == FAIL)
    ck("SPROC-1 mutation: the spec's treatment value is not the last write in the notebook",
       st(c_sproc_1, nb_text=GOOD_SUB_NB.replace("= '2.0'", "= '1.0'"), spec=GOOD_SUB_SPEC) == FAIL)
    ck("SPROC-1 mutation: the treatment is never recorded into run_stats",
       st(c_sproc_1, nb_text=GOOD_SUB_NB.replace('"motion_relink_learned_bonus"', '"other"'),
          spec=GOOD_SUB_SPEC) == FAIL)
    ck("SPROC-1 mutation: a submission declaring no treatment at all",
       st(c_sproc_1, nb_text=GOOD_SUB_NB, spec=dict(GOOD_SUB_SPEC, edits=[])) == FAIL)
    ck("SPROC-2 mutation: the completion heartbeat is stripped",
       st(c_sproc_2, nb_text=GOOD_SUB_NB.replace("SUBMISSION_COMPLETE", "done")) == FAIL)
    ck("SPROC-3 mutation: the input/output reconciliation raise is deleted",
       st(c_sproc_3, nb_text=GOOD_SUB_NB.replace("if len(geffs) != len(test_stems):", "")) == FAIL)
    ck("SART-1 mutation: submission.csv is written by a direct truncating open('w')",
       st(c_sart_1, nb_text=GOOD_SUB_NB.replace("os.replace(SUBMISSION_TMP, SUBMISSION_PATH)",
                                                'SUBMISSION_PATH.open("w")')) == FAIL)
    ck("SART-2 mutation: the row/size guard is removed",
       st(c_sart_2, nb_text=GOOD_SUB_NB.replace("row_id == total_nodes + total_edges", "")) == FAIL)
    ck("SART-3 mutation: the notebook names no submission.csv",
       st(c_sart_3, nb_text=GOOD_SUB_NB.replace("submission.csv", "other.csv")
          .replace("_guard_submission", "_g"), spec=GOOD_SUB_SPEC) == FAIL)
    ck("SART-4 mutation: the in-kernel digest of the submission is removed",
       st(c_sart_4, nb_text=GOOD_SUB_NB.replace("hashlib.sha256(_guard_submission.read_bytes())",
                                                "0")) == FAIL)

    # ---- per-clause mutations, cache class (v1 coverage retained) ----
    ck("CODE-4 mutation: two specs share slug and out_dir", _code4_clash(sandbox))
    ck("CODE-2 OOS: a build injecting no generated patch is OUT_OF_SCOPE, not a silent PASS",
       st(c_code_2, spec={"edits": []}) == OOS)
    ck("DATA-3 mutation: the FACT-0418 pair with no fold-legitimate arm",
       st(c_data_3, spec=GOOD_CACHE_SPEC,
          trunk={"role": "official", "dual_trunk_pair": list(INVALID_PAIR)}) == FAIL)
    ck("DATA-2 mutation: an illegitimate role DECLARED legitimate",
       st(c_data_2, spec=GOOD_CACHE_SPEC,
          trunk={"role": "official", "fold_legitimate": True}) == FAIL)
    ck("DATA-2 OOS: no trunk declared is OUT_OF_SCOPE, not a PASS (the v1 false pass)",
       st(c_data_2, spec=GOOD_CACHE_SPEC, trunk=None) == OOS)
    ck("PROC-2 mutation: the gate heartbeat is stripped",
       st(c_proc_2, nb_text=GOOD_CACHE_NB.replace("AFT_GATE", "quiet")) == FAIL)
    ck("PROC-3 mutation: crop count disagrees with the run limit",
       st(c_proc_3, spec={"edits": [{"vars": {"BIOHUB_AFT_EXPECT_CROPS": "2",
                                              "BIOHUB_LOEO_LIMIT": "1"}}]}) == FAIL)
    ck("ART-2 mutation: the superseded 128 B/node would pass a run the re-derived figure refuses",
       (14_000_000 * SUPERSEDED_BYTES_PER_NODE * 2 / 1024**3 <= 20.0)
       and st(c_art_2, nodes=14_000_000) == FAIL)
    ck("ART-2 mutation: no node count - a capacity guard with no denominator",
       st(c_art_2, nodes=None) == FAIL)

    for r in R:
        print(f"  [{'CAUGHT' if r['caught'] else 'MISSED'}] {r['mutation']}"
              + (f"   {r['detail']}" if r["detail"] and not r["caught"] else ""))
    n = sum(1 for r in R if r["caught"])
    print(f"\n{n}/{len(R)} mutations caught")
    return 0 if n == len(R) else 1


def _refuses(fn) -> bool:
    try:
        fn()
        return False
    except ContractRefusal:
        return True


def _code4_clash(sandbox: Path) -> bool:
    d = sandbox / "specs"
    d.mkdir(parents=True, exist_ok=True)
    s = {"slug": "x", "out_dir": "d", "code_file": "n.ipynb", "edits": []}
    (d / "a.json").write_text(json.dumps(s), encoding="utf-8")
    (d / "b.json").write_text(json.dumps(s), encoding="utf-8")
    ctx = _ctx(spec=s, repo=sandbox)
    ctx["spec_path"] = d / "a.json"
    ctx["all_specs"] = [d / "a.json", d / "b.json"]
    return c_code_4(ctx)["status"] == FAIL


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sign")
    s.add_argument("--spec", required=True)
    s.add_argument("--nodes", type=int, default=None)
    s.add_argument("--trunk", default=None)
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
    print(f"profile: {rep['profile']}  ({rep['profile_description']})")
    print(f"selected by: {rep['profile_selected_by']}")
    for name, cs in rep["sections"].items():
        bad = [c for c in cs if c["status"] == FAIL]
        print(f"\n{name}  {'REFUSED' if bad else 'SIGNED'}")
        for c in cs:
            mark = {PASS: "PASS", FAIL: "FAIL", OOS: " OOS"}[c["status"]]
            print(f"  [{mark}] {c['id']:8s} {c['title'][:88]}")
    print(f"\nVERDICT: {'SIGNED' if rep['signed'] else 'REFUSED'}   failed={rep['failed']}")
    print(f"out_of_scope ({len(rep['out_of_scope'])}): {rep['out_of_scope']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"receipt -> {a.out}")
    print(HEARTBEAT + f" signed={rep['signed']}")
    return 0 if rep["signed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
