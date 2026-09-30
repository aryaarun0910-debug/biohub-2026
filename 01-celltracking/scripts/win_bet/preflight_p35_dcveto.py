r"""CPU preflight for PKT-0039 / LEVER-0042 - the ONE capped exploitation of the 0.931 base.

WHY THIS EXISTS
---------------
The standing GPU rule (head of `research/00-system/registry/levers.yaml`) forbids using GPU to
discover mounts, filenames, subprocess boundaries or empty inputs after two consecutive sessions
were spent that way (FACT-0387, FACT-0397). Everything answerable on CPU is answered here, against
the EXACT built notebook, the EXACT declared datasets and the REAL subprocess boundary.

WHAT IT PROVES, AND WHAT IT CANNOT
----------------------------------
It proves the artifact is well formed and that the single declared component actually takes effect
on the deployed code path. It CANNOT predict the score: the 0.931 configuration cannot be validated
offline at all, because its secondary saw all 199 movies (FACT-0378, FACT-0393 rule 4). No check
here licenses any part of the preregistered band, which is a prior.

CALIBRATION, NOT ASSERTION
--------------------------
The checkpoint-resolution check does not assert what the notebook "should" do. It executes the
notebook's OWN resolver, rebased onto a simulated mount holding the REAL dataset bytes, and first
requires that resolver to REPRODUCE the two fetched kernel logs of runs that actually scored -
P32 at 0.931 and P24 at 0.928. A model that cannot reproduce two known outcomes is not allowed to
predict a third (AGENTS.md: calibrate any derived quantity against an independently known value).

FAIL CLOSED
-----------
Any check that cannot be EVALUATED is a FAIL, never a skip. A silent no-op is worse than a crash.
Exit status is 0 only when every check passed.

Usage:
    .\.venv\Scripts\python.exe scripts\win_bet\preflight_p35_dcveto.py \
        --spec scripts/kaggle_specs/p35_dcveto_on_931.json \
        --dc-pack C:/temp/biohub_deepcenter_p10_audit \
        --out C:/temp/exploit931/preflight_p35.json
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import kaggle_mounts as KM  # noqa: E402  the ONE central mount resolver (Agent 1, PKT-0038)


# --------------------------------------------------------------------------- result plumbing
class Preflight:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def record(self, name: str, passed: bool, **detail) -> bool:
        self.checks.append({"check": name, "passed": bool(passed), **detail})
        return bool(passed)

    def fail(self, name: str, why: str, **detail) -> bool:
        return self.record(name, False, error=why, **detail)

    @property
    def verdict(self) -> str:
        return "PASS" if all(c["passed"] for c in self.checks) else "FAIL"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cell_sources(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def cell_types(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return [c["cell_type"] for c in nb["cells"]]


def extract_def(source: str, name: str) -> str:
    """Lift one top-level `def` out of a cell, byte-for-byte."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = source.splitlines(keepends=True)
            return "".join(lines[node.lineno - 1:node.end_lineno])
    raise KeyError(name)


# --------------------------------------------------------------------------- checks
def check_build_binding(pf: Preflight, spec: dict, spec_path: Path) -> dict:
    """Spec, base and built notebook agree, and the FACT-0396 E1 trap is not re-armed."""
    out_dir = ROOT / spec["out_dir"]
    built = out_dir / spec["code_file"]
    base = ROOT / spec["base_notebook"]
    man_path = out_dir / "build_manifest.json"

    for label, p in (("base", base), ("built", built), ("manifest", man_path)):
        if not p.is_file():
            pf.fail("build_binding", f"{label} missing: {p}")
            return {}
    man = json.loads(man_path.read_text(encoding="utf-8"))

    base_sha, built_sha = sha256_file(base), sha256_file(built)
    same_path = base.resolve() == built.resolve()
    ok = (
        not same_path
        and spec.get("base_sha256") == base_sha
        and man.get("base_sha256") == base_sha
        and man.get("built_sha256") == built_sha
        and man.get("spec") == str(spec_path)
    )
    pf.record(
        "build_binding", ok,
        base_notebook=spec["base_notebook"], base_sha256=base_sha,
        built_notebook=str(built.relative_to(ROOT)), built_sha256=built_sha,
        base_equals_built=same_path,
        note=("FACT-0396 exception E1: the P32 spec set base_notebook == built_notebook and "
              "consumed its own base. This spec must NOT repeat that."),
    )
    return {"base": base, "built": built, "base_sha256": base_sha, "built_sha256": built_sha}


def check_base_is_the_scored_artifact(pf: Preflight, base_sha: str) -> None:
    """The base must be the notebook the P32 release receipt binds to the 0.931 submission."""
    receipt = ROOT / "_evidence" / "audit" / "receipts" / "release_p32.json"
    if not receipt.is_file():
        pf.fail("base_is_the_scored_artifact", f"receipt missing: {receipt}")
        return
    r = json.loads(receipt.read_text(encoding="utf-8"))
    checks = r.get("checks", {})
    recorded = (checks.get("notebook_matches_manifest") or {}).get("notebook_sha256")
    # this receipt's schema records a boolean `passed`; accept only an explicit True
    if "passed" not in r:
        pf.fail("base_is_the_scored_artifact", "receipt has no `passed` field to read",
                receipt_keys=sorted(k for k in r if k != "checks"))
        return
    pf.record(
        "base_is_the_scored_artifact",
        recorded == base_sha and r.get("passed") is True
        and r.get("name") == "p32_public931_exact" and r.get("experiment") == "EXP-0038",
        receipt=str(receipt.relative_to(ROOT)), receipt_passed=r.get("passed"),
        receipt_name=r.get("name"), receipt_experiment=r.get("experiment"),
        receipt_notebook_sha256=recorded, base_sha256=base_sha,
        chain="EXP-0038 is the experiment FACT-0393 binds to the 0.931 submission",
    )


def check_blast_radius(pf: Preflight, base: Path, built: Path, spec: dict) -> None:
    """Exactly one cell differs, and the difference is exactly the three declared env lines."""
    a, b = cell_sources(base), cell_sources(built)
    if len(a) != len(b):
        pf.fail("blast_radius_one_cell", f"cell count changed {len(a)} -> {len(b)}")
        return
    changed = [i for i in range(len(a)) if a[i] != b[i]]
    added = [
        ln[1:] for ln in difflib.ndiff(a[changed[0]].splitlines(), b[changed[0]].splitlines())
        if ln.startswith("+ ")
    ] if len(changed) == 1 else []
    removed = [
        ln[1:] for ln in difflib.ndiff(a[changed[0]].splitlines(), b[changed[0]].splitlines())
        if ln.startswith("- ")
    ] if len(changed) == 1 else []
    meaningful = [ln.strip() for ln in added if ln.strip() and not ln.strip().startswith("#")]

    declared = spec["edits"][0]["vars"]
    expected = {f'os.environ["{k}"] = {v!r}' for k, v in declared.items()}
    pf.record(
        "blast_radius_one_cell",
        len(changed) == 1 and not removed and set(meaningful) == expected,
        cells_changed=changed, lines_removed=removed, lines_added=meaningful,
        declared_vars=sorted(declared),
    )


def check_component_is_the_p24_bundle(pf: Preflight, spec: dict) -> None:
    """The three variables must BE the P24 spec's single env edit, up to the mount root only."""
    p24 = json.loads((ROOT / "scripts" / "kaggle_specs"
                      / "p24_deepcenter_best_veto.json").read_text(encoding="utf-8"))
    p24_edits = p24.get("edits", [])
    mine = spec["edits"][0]["vars"]
    if len(p24_edits) != 1 or len(spec["edits"]) != 1:
        pf.fail("component_is_the_p24_bundle",
                f"edit counts p24={len(p24_edits)} treatment={len(spec['edits'])}; expected 1 and 1")
        return
    theirs = p24_edits[0]["vars"]
    same_keys = set(mine) == set(theirs)
    # evaluate over the INTERSECTION: an extra or missing key is already caught by same_keys,
    # and must FAIL this condition rather than raise out of the whole preflight.
    shared = (set(mine) & set(theirs)) - {"BIOHUB_DEEPCENTER_CHECKPOINT"}
    same_values = all(mine[k] == theirs[k] for k in shared)
    same_file = (
        "BIOHUB_DEEPCENTER_CHECKPOINT" in mine and "BIOHUB_DEEPCENTER_CHECKPOINT" in theirs
        and Path(mine["BIOHUB_DEEPCENTER_CHECKPOINT"]).name
        == Path(theirs["BIOHUB_DEEPCENTER_CHECKPOINT"]).name == "best.pt"
    )
    pf.record(
        "component_is_the_p24_bundle", same_keys and same_values and same_file,
        treatment_vars=mine, p24_vars=theirs,
        only_difference="the mount-root prefix of the checkpoint path; the FILE is identical",
    )


def check_cells_compile(pf: Preflight, built: Path) -> None:
    src, kinds = cell_sources(built), cell_types(built)
    bad = []
    for i, (text, kind) in enumerate(zip(src, kinds)):
        if kind != "code":
            continue
        try:
            compile(text, f"<cell {i}>", "exec")
        except SyntaxError as exc:
            bad.append({"cell": i, "error": str(exc)})
    pf.record("every_cell_compiles", not bad, code_cells=sum(k == "code" for k in kinds), failures=bad)


def check_kernel_metadata(pf: Preflight, spec: dict) -> None:
    """Datasets, accelerator and internet must be byte-identical to the base kernel's."""
    mine = json.loads((ROOT / spec["out_dir"] / "kernel-metadata.json").read_text(encoding="utf-8"))
    base_meta_path = ROOT / Path(spec["base_notebook"]).parent / "kernel-metadata.json"
    if not base_meta_path.is_file():
        pf.fail("kernel_metadata_matches_base", f"base kernel-metadata missing: {base_meta_path}")
        return
    theirs = json.loads(base_meta_path.read_text(encoding="utf-8"))
    keys = ("dataset_sources", "competition_sources", "kernel_sources", "model_sources",
            "enable_gpu", "enable_tpu", "enable_internet", "machine_shape", "docker_image")
    diffs = {k: {"base": theirs.get(k), "treatment": mine.get(k)}
             for k in keys if theirs.get(k) != mine.get(k)}
    pf.record(
        "kernel_metadata_matches_base", not diffs,
        differing=diffs, dataset_sources=mine.get("dataset_sources"),
        slug_declared=spec["slug"], id_in_metadata=mine.get("id"),
        limit="dataset_sources carry no version pin (FACT-0396 exception E4); the binding is by "
              "NAME, so Kaggle resolves whatever version is current at run time.",
    )


def check_env_across_real_subprocess(pf: Preflight, base: Path, built: Path, spec: dict) -> None:
    """Run the config cells in a FRESH interpreter - the real process boundary, not an import."""
    declared = spec["edits"][0]["vars"]

    def run_config_cells(nb: Path) -> dict:
        cells = cell_sources(nb)
        probe = (
            "\n\nimport json as _pf_json, os as _pf_os, sys as _pf_sys\n"
            "_pf_sys.stdout.write('###PREFLIGHT###' + _pf_json.dumps("
            "{k: v for k, v in _pf_os.environ.items() if k.startswith('BIOHUB_')}) + '\\n')\n"
        )
        script = cells[0] + "\n\n" + cells[1] + probe
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "config_cells.py"
            f.write_text(script, encoding="utf-8")
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            # strip any inherited BIOHUB_* so the cells are the only source of truth
            for k in [k for k in env if k.startswith("BIOHUB_")]:
                env.pop(k)
            r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, env=env)
        out = r.stdout or ""
        marker = "###PREFLIGHT###"
        payload = json.loads(out.split(marker, 1)[1].splitlines()[0]) if marker in out else None
        return {"returncode": r.returncode, "env": payload,
                "guard_pass": "Configuration guard: PASS" in out,
                "stderr_tail": (r.stderr or "").strip().splitlines()[-5:]}

    base_run, treat_run = run_config_cells(base), run_config_cells(built)
    if base_run["env"] is None or treat_run["env"] is None:
        pf.fail("env_takes_effect_across_a_real_subprocess",
                "config cells did not run to completion in a fresh interpreter",
                base=base_run, treatment=treat_run)
        return

    applied = {k: treat_run["env"].get(k) for k in declared}
    intended = applied == declared
    others = {k for k in set(base_run["env"]) | set(treat_run["env"])
              if base_run["env"].get(k) != treat_run["env"].get(k)} - set(declared)
    pf.record(
        "env_takes_effect_across_a_real_subprocess",
        intended and not others and treat_run["guard_pass"] and treat_run["returncode"] == 0,
        boundary=f"fresh {Path(sys.executable).name} subprocess, inherited BIOHUB_* stripped",
        applied=applied, intended=declared,
        base_configuration_guard=base_run["guard_pass"],
        treatment_configuration_guard=treat_run["guard_pass"],
        other_BIOHUB_vars_that_differ=sorted(others),
        returncode=treat_run["returncode"], stderr_tail=treat_run["stderr_tail"],
    )


def check_subprocess_env_passthrough(pf: Preflight, built: Path) -> None:
    """Does the state cross the prediction process boundary? (FACT-0060's failure class.)"""
    src = "\n".join(cell_sources(built))
    passthrough = re.findall(r"env\s*=\s*\{\*\*os\.environ", src)
    shard = re.findall(r"shard_env\s*=\s*\{\*\*os\.environ", src)
    gate_after_predict = src.index("Prediction completed in") < src.index(
        "def load_deepcenter_veto_detector") if (
        "Prediction completed in" in src and "def load_deepcenter_veto_detector" in src) else False
    pf.record(
        "subprocess_env_passthrough", bool(passthrough) and bool(shard),
        env_star_star_os_environ_sites=len(passthrough), shard_env_sites=len(shard),
        deepcenter_gate_defined_after_prediction=gate_after_predict,
        note="The three variables are consumed in the PARENT after the prediction shards merge, "
             "and the shard launcher forwards os.environ wholesale, so neither reading depends "
             "on a variable failing to cross the boundary.",
    )


def _resolver_namespace(built: Path, sandbox_input: Path, env_overrides: dict) -> dict:
    """Execute the BUILT notebook's own DeepCenter candidate ladder, rebased onto a sandbox."""
    src = "\n".join(cell_sources(built))
    manifest_default = env_overrides.get(
        "BIOHUB_DEEPCENTER_MANIFEST_DEFAULT",
        "/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/"
        "ARTIFACT_MANIFEST.json")
    fake_env = {
        "BIOHUB_DEEPCENTER_CHECKPOINT": env_overrides["BIOHUB_DEEPCENTER_CHECKPOINT"],
        "BIOHUB_DEEPCENTER_MANIFEST": "",
    }
    # module-level constants the ladder closes over; read from the BUILT notebook, not assumed
    rel = re.search(r'DEEPCENTER_RELATIVE\s*=\s*os\.environ\.get\(\s*"BIOHUB_DEEPCENTER_RELATIVE"'
                    r'\s*,\s*"([^"]+)"\s*\)', src)
    if not rel:
        raise KeyError("DEEPCENTER_RELATIVE default not found in the built notebook")
    ns_extra = {
        "os": type("_OsShim", (), {"environ": type("_E", (), {
            "get": staticmethod(lambda k, d="": fake_env.get(k, d))})()})(),
        "json": json,
        "DEEPCENTER_CHECKPOINT_DEFAULT": env_overrides["BIOHUB_DEEPCENTER_CHECKPOINT"],
        "DEEPCENTER_MANIFEST_DEFAULT": manifest_default,
        "DEEPCENTER_RELATIVE": rel.group(1),
    }
    body = extract_def(src, "_dc_manifest_weight_paths") + "\n\n" + \
        extract_def(src, "_dc_checkpoint_candidates")
    fn = KM.rebase_and_exec(body, "_dc_checkpoint_candidates", sandbox_input, extra=ns_extra)
    return {"fn": fn, "source": body}


def check_deepcenter_resolution(pf: Preflight, built: Path, base: Path, spec: dict,
                                dc_pack: Path) -> None:
    """Calibrate the resolver against two scored runs, THEN predict the treatment's resolution."""
    need = ["best.pt", "checkpoint_last.pt", "ARTIFACT_MANIFEST.json"]
    missing = [n for n in need if not (dc_pack / n).is_file()]
    if missing:
        pf.fail("deepcenter_resolution", f"DeepCenter pack incomplete at {dc_pack}: {missing}")
        return

    try:
        import torch
    except Exception as exc:                                    # fail closed, never degrade
        pf.fail("deepcenter_resolution", f"torch unavailable, epoch cannot be READ: {exc!r}")
        return

    facts = {}
    for name in ("best.pt", "checkpoint_last.pt"):
        ck = torch.load(dc_pack / name, map_location="cpu", weights_only=False)
        facts[name] = {"sha256": sha256_file(dc_pack / name), "epoch": int(ck.get("epoch", -1))}

    slug = "biohub-deepcenter-unet3d-center-prior-v1"
    payload = {f"weights/full_frame_center/{n}": (dc_pack / n).read_bytes()
               for n in ("best.pt", "checkpoint_last.pt")}
    payload["ARTIFACT_MANIFEST.json"] = (dc_pack / "ARTIFACT_MANIFEST.json").read_bytes()

    integrity_expected = re.search(r'_deepcenter_expected_sha256\s*=\s*"([0-9a-f]{64})"',
                                   "\n".join(cell_sources(built)))
    if not integrity_expected:
        pf.fail("deepcenter_resolution", "base integrity block declares no expected DeepCenter sha")
        return
    integrity_expected = integrity_expected.group(1)

    def resolve_for(env_checkpoint: str, expected_epoch: int, convention: str) -> dict:
        with tempfile.TemporaryDirectory() as td:
            sandbox = Path(td) / "kaggle" / "input"
            sandbox.mkdir(parents=True)
            KM.simulate_mount_tree(sandbox, slug, payload,
                                   convention=convention, owner="pilkwang")
            # (1) the base notebook's integrity block: first EXISTING candidate must hash right
            integ_order = [env_checkpoint,
                           f"/kaggle/input/{slug}/weights/full_frame_center/best.pt",
                           f"/kaggle/input/datasets/pilkwang/{slug}/weights/"
                           f"full_frame_center/best.pt"]
            sroot = str(sandbox).replace("\\", "/").rstrip("/")
            materialized = next(
                (p for p in (Path(c.replace("/kaggle/input", sroot)) for c in integ_order)
                 if p.is_file()), None)
            if materialized is None:
                return {"integrity": "FileNotFoundError", "loaded": None}
            got = sha256_file(materialized)
            if got != integrity_expected:
                return {"integrity": f"RuntimeError sha {got[:8]} != {integrity_expected[:8]}",
                        "materialized": materialized.name, "loaded": None}
            # the block REWRITES the env var to the materialized path before the loader runs
            fn = _resolver_namespace(
                built, sandbox,
                {"BIOHUB_DEEPCENTER_CHECKPOINT": str(materialized).replace("\\", "/")})["fn"]
            tried = []
            for cand in fn():
                p = Path(cand)
                if not p.exists():
                    continue
                epoch = facts.get(p.name, {}).get("epoch", -1)
                tried.append({"file": p.name, "epoch": epoch})
                if expected_epoch > 0 and epoch != expected_epoch:
                    continue                                    # notebook: ValueError -> skip
                return {"integrity": "PASS", "materialized": materialized.name,
                        "loaded": p.name, "loaded_epoch": epoch, "tried": tried}
            return {"integrity": "PASS", "materialized": materialized.name,
                    "loaded": None, "tried": tried,
                    "consequence": "REQUIRE_DEEPCENTER_VETO=1 -> FileNotFoundError, kernel dies"}

    flat_slug_path = f"/kaggle/input/{slug}/weights/full_frame_center"
    ds_path = (f"/kaggle/input/datasets/pilkwang/{slug}/weights/full_frame_center")

    # ---- calibration: reproduce two runs that actually scored -----------------------------
    cal = {
        "p32_base_0931": {
            "observed": {"loaded": "checkpoint_last.pt", "loaded_epoch": 500},
            "predicted": resolve_for(f"{flat_slug_path}/checkpoint_last.pt", 500, "datasets"),
            "evidence": "C:/temp/exploit931/p32log/kernel.log lines 300-304, 533-537",
        },
        "p24_champion_0928": {
            "observed": {"loaded": "best.pt", "loaded_epoch": 2},
            "predicted": resolve_for(f"{flat_slug_path}/best.pt", 2, "datasets"),
            "evidence": "C:/temp/exploit931/p24log/kernel.log lines 295-299",
        },
    }
    calibrated = all(
        c["predicted"].get("loaded") == c["observed"]["loaded"]
        and c["predicted"].get("loaded_epoch") == c["observed"]["loaded_epoch"]
        for c in cal.values()
    )

    declared = spec["edits"][0]["vars"]
    treatment = {
        conv: resolve_for(declared["BIOHUB_DEEPCENTER_CHECKPOINT"],
                          int(declared["BIOHUB_DEEPCENTER_EXPECTED_EPOCH"]), conv)
        for conv in ("datasets", "flat")
    }
    treatment_ok = all(
        r.get("integrity") == "PASS" and r.get("loaded") == "best.pt" and r.get("loaded_epoch") == 2
        for r in treatment.values()
    )
    pf.record(
        "deepcenter_resolution", calibrated and treatment_ok,
        calibrated_against_two_scored_runs=calibrated, calibration=cal,
        treatment_under_both_mount_conventions=treatment,
        real_checkpoint_facts=facts,
        integrity_block_expects_sha256=integrity_expected,
        declared_checkpoint=declared["BIOHUB_DEEPCENTER_CHECKPOINT"],
        method="the notebook's OWN _dc_checkpoint_candidates, lifted by ast and rebased onto a "
               "simulated mount holding the REAL dataset bytes (kaggle_mounts.simulate_mount_tree)",
        ladder_sha256=KM.LADDER_SHA256,
        unrelated_note=("the base resolver contains an unbounded '**' glob under /kaggle/input "
                        "(pre-existing in the public source; it completed in both scored runs). "
                        "Changing it would be a SECOND component change and is out of scope."),
    )


def check_veto_is_currently_inert(pf: Preflight, log_path: Path) -> None:
    """The base loads a DeepCenter model that rejects nothing on the division path."""
    if not log_path.is_file():
        pf.fail("base_veto_is_inert", f"fetched base log missing: {log_path}")
        return
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rejected = [int(m) for m in re.findall(r"deepcenter_rejected=(\d+)", text)]
    geometric = [int(m) for m in re.findall(r"geometric_candidates=(\d+)", text)]
    manifest_off = "safe-div veto:       False" in text
    pf.record(
        "base_veto_is_inert", bool(rejected) and set(rejected) == {0} and manifest_off,
        movies_reporting=len(rejected), deepcenter_rejected_values=sorted(set(rejected)),
        geometric_candidates_total=sum(geometric),
        base_pipeline_manifest_says_safe_div_veto_off=manifest_off,
        meaning="the treatment's safe-division veto can only act on this candidate population; "
                "it is a PRUNER and its sign on this base is not knowable offline (FACT-0378).",
    )


# --------------------------------------------------------------------------- self-test
def selftest(spec: dict, spec_path: Path, dc_pack: Path, base_log: Path) -> int:
    """Plant a defect per condition and require THAT condition - not a generic error - to catch it.

    A preflight that has only ever returned PASS is a rubber stamp. FACT-0396's receipt harness
    earned its 26 conditions exactly this way.
    """
    out_dir = ROOT / spec["out_dir"]
    base, built = ROOT / spec["base_notebook"], out_dir / spec["code_file"]
    results = []

    def plant(label: str, condition: str, run) -> None:
        pf = Preflight()
        try:
            run(pf)
            caught = any(c["check"] == condition and not c["passed"] for c in pf.checks)
            other = [c["check"] for c in pf.checks if not c["passed"] and c["check"] != condition]
        except Exception as exc:                                # a crash is NOT a clean catch
            caught, other = False, [f"raised {type(exc).__name__}: {exc}"]
        results.append({"defect": label, "condition": condition,
                        "caught_by_its_own_condition": caught, "also_failed": other})

    def mutated(**vars_) -> dict:
        s = json.loads(json.dumps(spec))
        s["edits"][0]["vars"].update(vars_)
        return s

    # 1 wrong checkpoint file: the base integrity block must reject it on sha
    plant("checkpoint points at checkpoint_last.pt", "deepcenter_resolution",
          lambda pf: check_deepcenter_resolution(
              pf, built, base,
              mutated(BIOHUB_DEEPCENTER_CHECKPOINT=spec["edits"][0]["vars"]
                      ["BIOHUB_DEEPCENTER_CHECKPOINT"].replace("best.pt", "checkpoint_last.pt")),
              dc_pack))
    # 2 checkpoint kept but the epoch left at the base's 500: the loader falls back to the
    #   collapsed prior and the treatment silently becomes the control
    plant("expected epoch left at 500", "deepcenter_resolution",
          lambda pf: check_deepcenter_resolution(
              pf, built, base, mutated(BIOHUB_DEEPCENTER_EXPECTED_EPOCH="500"), dc_pack))
    # 3 a fourth variable smuggled in: no longer one component
    plant("a fourth variable smuggled into the edit", "blast_radius_one_cell",
          lambda pf: check_blast_radius(
              pf, base, built, mutated(BIOHUB_DET_THRESHOLD="0.96")))
    # 4 the same, seen by the P24-equivalence condition
    plant("edit no longer the P24 bundle", "component_is_the_p24_bundle",
          lambda pf: check_component_is_the_p24_bundle(
              pf, mutated(BIOHUB_DET_THRESHOLD="0.96")))
    # 5 base drifted from the sha the spec declares
    plant("declared base_sha256 drifted", "build_binding",
          lambda pf: check_build_binding(pf, {**spec, "base_sha256": "0" * 64}, spec_path))
    # 6 base is not the artifact the release receipt binds to 0.931
    plant("base is not the receipt's notebook", "base_is_the_scored_artifact",
          lambda pf: check_base_is_the_scored_artifact(pf, "0" * 64))
    # 7 fail closed: the DeepCenter pack is absent rather than wrong
    plant("DeepCenter pack absent", "deepcenter_resolution",
          lambda pf: check_deepcenter_resolution(pf, built, base, spec, ROOT / "no_such_pack"))
    # 8 fail closed: the base log the inertness claim rests on is absent
    plant("base kernel log absent", "base_veto_is_inert",
          lambda pf: check_veto_is_currently_inert(pf, ROOT / "no_such.log"))

    ok = all(r["caught_by_its_own_condition"] for r in results)
    print(json.dumps({"selftest": "PASS" if ok else "FAIL",
                      "defects": len(results), "results": results}, indent=2))
    return 0 if ok else 1


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true",
                    help="plant one defect per condition and require each to be caught")
    ap.add_argument("--spec", default="scripts/kaggle_specs/p35_dcveto_on_931.json")
    ap.add_argument("--dc-pack", default="C:/temp/biohub_deepcenter_p10_audit")
    ap.add_argument("--base-log", default="C:/temp/exploit931/p32log/kernel.log")
    ap.add_argument("--out", default="C:/temp/exploit931/preflight_p35.json")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    spec_path = spec_path if spec_path.is_absolute() else ROOT / spec_path
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    if args.selftest:
        return selftest(spec, spec_path, Path(args.dc_pack), Path(args.base_log))

    pf = Preflight()
    built_info = check_build_binding(pf, spec, spec_path)
    if not built_info:
        print(json.dumps({"verdict": "FAIL", "checks": pf.checks}, indent=2))
        return 1

    base, built = built_info["base"], built_info["built"]
    check_base_is_the_scored_artifact(pf, built_info["base_sha256"])
    check_blast_radius(pf, base, built, spec)
    check_component_is_the_p24_bundle(pf, spec)
    check_cells_compile(pf, built)
    check_kernel_metadata(pf, spec)
    check_env_across_real_subprocess(pf, base, built, spec)
    check_subprocess_env_passthrough(pf, built)
    check_deepcenter_resolution(pf, built, base, spec, Path(args.dc_pack))
    check_veto_is_currently_inert(pf, Path(args.base_log))

    report = {
        "packet": "PKT-0039", "lever": "LEVER-0042",
        "spec": str(spec_path), "spec_sha256": sha256_file(spec_path),
        "built_notebook_sha256": built_info["built_sha256"],
        "verdict": pf.verdict,
        "passed": sum(c["passed"] for c in pf.checks), "of": len(pf.checks),
        "licenses": "that the artifact is well formed and the single declared component takes "
                    "effect on the deployed code path, under BOTH observed mount conventions",
        "does_not_license": "any statement about the score. This configuration cannot be "
                            "validated offline at all (FACT-0378, FACT-0393 rule 4); the "
                            "preregistered band is a prior, not a measurement.",
        "checks": pf.checks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    for c in pf.checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}"
              + ("" if c["passed"] else f"  <- {c.get('error', 'see report')}"))
    print(f"\nfull report -> {out}")
    return 0 if pf.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
