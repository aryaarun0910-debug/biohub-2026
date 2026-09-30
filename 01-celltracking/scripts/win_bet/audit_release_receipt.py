r"""Fail-closed RELEASE RECEIPT: bind notebook, weights, manifest, graph and score together.

WHY THIS EXISTS
---------------
A champion transition is a claim that one artifact replaces another as the campaign's baseline.
Until now that claim has been carried by a single number read off a leaderboard page. FACT-0341
records the outgoing champion's audit receipt as living at
`notebooks/kaggle_p24_deepcenter_best_veto/_out/` - a directory that does not exist, because
`notebooks/**/_out/` is gitignored and nothing was ever fetched there. So the baseline of record
was bound to its artifact by assertion alone.

This instrument refuses to let that happen again. It joins, in ONE receipt:

    spec  ->  built notebook (sha256, and the commit it is identical to)
          ->  kernel-metadata (slug, datasets, accelerator, internet)
          ->  build manifest (edits, defect gate, pushed slug)
          ->  the fetched GRAPH (submission.csv sha256 + every structural check)
          ->  the registry's preregistered band and the score that landed
          ->  optionally, the UPSTREAM SOURCE the artifact claims to reproduce

WHAT IT CATCHES THAT A READING OF THE LEADERBOARD DOES NOT
----------------------------------------------------------
* a graph that was never fetched, so no submitted artifact exists to audit (P24's case);
* FLOAT coordinates in the emitted graph - FACT-0344 measured a float export losing 0.014 on this
  leaderboard, so `fractional == 0` is a release condition, not a nicety;
* a metric-illegal graph (in-degree > 1, out-degree > 2, cross-movie or non-consecutive edges);
* slug divergence between spec, manifest, pushed kernel and kernel-metadata;
* a defect gate that PASSED HAVING EXERCISED NOTHING - `"passed": []` is reported honestly as
  VACUOUS rather than as evidence, the same empty-band failure the P33 audit exists to catch;
* a score outside its own preregistered band, or a fired falsifier, or a registry whose recorded
  lb disagrees with the fact that cites it;
* an artifact that claims to reproduce an upstream source but whose cell sources differ from it.

WHAT IT CANNOT DO, STATED PLAINLY
----------------------------------
1. It cannot prove which KERNEL VERSION Kaggle executed, nor re-read the leaderboard. It binds the
   version recorded in `_out/audit_receipt.json` (written by `kaggle_factory audit`, which IS
   version-bound) and the score recorded in the registry. If neither exists the condition FAILS.
2. `dataset_sources` carry NO VERSION PIN. Kaggle resolves `owner/slug` to whatever version is
   current at run time, so the weights binding is by name, not by content. This is reported as a
   standing limit on every receipt.
3. The fetched `submission.csv` is the VISIBLE-movie output of the same code path, not the hidden
   test graph that was scored. It proves the emitted graph's shape and legality, not the score.

FAIL CLOSED. A missing artifact is a FAIL, never a skip. Heartbeat `RELEASE_RECEIPT_COMPLETE`;
its absence is the alarm.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HEARTBEAT = "RELEASE_RECEIPT_COMPLETE"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
REGISTRY = ROOT / "research" / "00-system" / "registry"

WEIGHTS_PIN_LIMIT = (
    "kernel-metadata dataset_sources are owner/slug with no version pin; Kaggle resolves them to "
    "the current version at run time, so the weights binding is by NAME, not by content."
)
GRAPH_SCOPE_LIMIT = (
    "the fetched submission.csv is the VISIBLE-movie output of the same code path, not the hidden "
    "graph that was scored; it binds the emitted graph's shape and legality, not the score."
)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def lf_sha256_file(path: Path) -> str:
    """Line-ending-independent identity. core.autocrlf=true makes the worktree sha differ from the
    committed blob's, so a raw sha256 is not portable across clones."""
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def normalized_spec_sha256(spec: dict) -> str:
    """Reimplements kaggle_factory.normalized_spec_sha256 - the receipt records the NORMALISED
    spec hash (portable keys, sorted, compact), not the raw file hash. Comparing a raw file hash
    against it produces a false FAIL; this was caught by reading the factory at source."""
    portable = {k: v for k, v in spec.items() if not str(k).startswith("_")}
    payload = json.dumps(portable, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return sha256_bytes(payload)


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True)
    return (r.stdout or "").strip()


def _git_ok(*args: str) -> bool:
    r = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True)
    return r.returncode == 0


def _cells(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"]]


def _load_registry_yaml(name: str) -> dict:
    import yaml
    return yaml.safe_load((REGISTRY / name).read_text(encoding="utf-8"))


def _applicable_defect_rules(ledger: Path, spec_name: str) -> list[str]:
    """How many ledger rules were IN SCOPE for this spec. `passed: []` means nothing was checked -
    which is a vacuous gate, not a clean one."""
    try:
        data = json.loads(ledger.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    hits = []
    for d in data.get("defects", []):
        globs = (d.get("scope") or {}).get("spec_name_globs") or []
        if any(fnmatch.fnmatch(spec_name, g) for g in globs):
            hits.append(d.get("id"))
    return hits


def _parse_fractional(structural: list) -> int | None:
    for row in structural:
        for c in row.get("checks", []):
            if "numeric" in str(c.get("check", "")):
                m = re.search(r"fractional=(\d+)", str(c.get("detail", "")))
                if m:
                    return int(m.group(1))
    return None


def audit_release(spec_path: Path, experiment_id: str, reference_nb: Path | None,
                  at_commit: str | None, fetched_dir: Path | None = None) -> dict:
    checks: dict[str, dict] = {}

    def check(name: str, passed: bool, **detail) -> None:
        checks[name] = {"passed": bool(passed), **detail}

    out: dict = {"spec": str(spec_path), "experiment": experiment_id, "checks": checks,
                 "limits": [WEIGHTS_PIN_LIMIT, GRAPH_SCOPE_LIMIT]}

    if not spec_path.is_file():
        check("spec_present", False, path=str(spec_path))
        out["passed"] = False
        return out
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    out["name"] = spec.get("name")
    check("spec_present", True, file_sha256=sha256_file(spec_path),
          normalized_sha256=normalized_spec_sha256(spec))

    nb_dir = ROOT / spec["out_dir"]
    nb = nb_dir / spec["code_file"]
    manifest_p = nb_dir / "build_manifest.json"
    meta_p = nb_dir / "kernel-metadata.json"
    # WHERE THE FETCHED ARTIFACT ACTUALLY LIVES - a false FAIL this instrument produced
    # on its first run. `kaggle_factory fetch` writes to <out_dir>/_out/, but
    # `kaggle_queue.py` writes to C:/temp/queue/<spec name>/ instead. Auditing only the
    # factory path reported the P24 CHAMPION as having no submission artifact when a
    # complete, passing one was on disk the whole time. Both known locations are searched,
    # an explicit --fetched-dir overrides, and the location used is RECORDED so a future
    # reader can see which convention produced the evidence.
    candidates = [fetched_dir] if fetched_dir else [
        nb_dir / "_out",
        Path("C:/temp/queue") / str(spec.get("name", "")),
    ]
    out_dir = next(
        (d for d in candidates if d and (d / "audit_receipt.json").is_file()),
        candidates[0],
    )
    receipt_p = out_dir / "audit_receipt.json"
    csv_p = out_dir / "submission.csv"
    structural_p = out_dir / "structural_audit.json"

    for label, path in (("notebook", nb), ("manifest", manifest_p), ("kernel_metadata", meta_p)):
        if not path.is_file():
            check(f"{label}_present", False, path=str(path))
            out["passed"] = False
            return out
    check("notebook_present", True)
    check("manifest_present", True)
    check("kernel_metadata_present", True)

    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    meta = json.loads(meta_p.read_text(encoding="utf-8"))

    # ---- C1 notebook identity -------------------------------------------------------------
    nb_sha = sha256_file(nb)
    check("notebook_matches_manifest", nb_sha == manifest.get("built_sha256"),
          notebook_sha256=nb_sha, manifest_built_sha256=manifest.get("built_sha256"),
          lf_normalised_sha256=lf_sha256_file(nb),
          note="lf_normalised_sha256 is the portable identity; core.autocrlf makes the raw sha "
               "clone-dependent")

    rel = str(nb.relative_to(ROOT)).replace("\\", "/")
    if at_commit:
        present = _git_ok("cat-file", "-e", f"{at_commit}:{rel}")
        check("notebook_identical_to_commit",
              present and _git_ok("diff", "--quiet", at_commit, "--", rel),
              commit=at_commit, path_present_at_commit=present,
              method="git diff --quiet (autocrlf-safe)")
    else:
        check("notebook_identical_to_commit", False, reason="no --at-commit given; a release "
              "receipt must name the commit its notebook came from")

    check("notebook_clean_in_git", not _git("status", "--porcelain", "--", rel),
          last_commit=_git("log", "-1", "--format=%h %ad", "--date=short", "--", rel))

    # ---- C2 slug binding ------------------------------------------------------------------
    meta_slug = str(meta.get("id", "")).split("/")[-1]
    slugs = {"spec": spec.get("slug"), "declared": manifest.get("declared_slug"),
             "pushed": manifest.get("pushed_slug"), "kernel_metadata": meta_slug}
    check("slug_binding", len(set(v for v in slugs.values() if v)) == 1 and all(slugs.values()),
          **slugs)

    # ---- C3 weights / datasets ------------------------------------------------------------
    spec_ds = sorted(spec.get("datasets") or [])
    meta_ds = sorted(meta.get("dataset_sources") or [])
    check("weights_binding", spec_ds == meta_ds and bool(meta_ds),
          spec_datasets=spec_ds, kernel_metadata_datasets=meta_ds,
          version_pinned=False, limit=WEIGHTS_PIN_LIMIT)
    check("accelerator_and_internet",
          bool(meta.get("enable_gpu")) == bool(spec.get("enable_gpu"))
          and bool(meta.get("enable_internet")) == bool(spec.get("enable_internet")),
          enable_gpu=meta.get("enable_gpu"), enable_internet=meta.get("enable_internet"),
          machine_shape=meta.get("machine_shape"))

    # ---- C4 defect gate: exercised or vacuous ---------------------------------------------
    applicable = _applicable_defect_rules(ROOT / "scripts" / "core" / "experiment_defects.json",
                                          spec.get("name", ""))
    gate_passed = (manifest.get("defect_gate") or {}).get("passed") or []
    check("defect_gate_not_silently_vacuous", sorted(applicable) == sorted(gate_passed),
          applicable_rules=applicable, recorded_passed=gate_passed,
          vacuous=(not applicable),
          note=("the defect ledger scopes NO rule to this spec, so `passed: []` is honest but "
                "carries no evidence" if not applicable else
                "rules were in scope and every one is recorded as passed"))

    # ---- C5 the graph ---------------------------------------------------------------------
    if not (csv_p.is_file() and structural_p.is_file() and receipt_p.is_file()):
        check("graph_fetched", False, searched=[str(d) for d in candidates if d],
              expected=[str(csv_p), str(structural_p), str(receipt_p)],
              reason="no fetched submission artifact on disk: there is no graph to audit and no "
                     "version-bound audit receipt. notebooks/**/_out/ is gitignored, so absence "
                     "here means the artifact was never fetched or was deleted.")
        out["passed"] = all(c["passed"] for c in checks.values())
        return out
    check("graph_fetched", True, fetched_from=str(out_dir),
          searched=[str(d) for d in candidates if d])

    receipt = json.loads(receipt_p.read_text(encoding="utf-8"))
    structural = json.loads(structural_p.read_text(encoding="utf-8"))
    csv_sha = sha256_file(csv_p)

    check("graph_matches_audit_receipt",
          csv_sha == (receipt.get("submission") or {}).get("sha256")
          and receipt.get("verdict") == "PASS",
          submission_sha256=csv_sha,
          receipt_submission_sha256=(receipt.get("submission") or {}).get("sha256"),
          receipt_verdict=receipt.get("verdict"))
    spec_norm = normalized_spec_sha256(spec)
    check("audit_receipt_binds_this_notebook",
          (receipt.get("build") or {}).get("notebook_sha256") == nb_sha
          and (receipt.get("spec") or {}).get("sha256") == spec_norm,
          receipt_notebook_sha256=(receipt.get("build") or {}).get("notebook_sha256"),
          receipt_spec_sha256=(receipt.get("spec") or {}).get("sha256"),
          current_normalized_spec_sha256=spec_norm,
          note="the factory records the NORMALISED spec hash, not the raw file hash")
    kernel = receipt.get("kernel") or {}
    check("audit_receipt_names_kernel_version",
          isinstance(kernel.get("version"), int) and kernel.get("slug") == meta_slug,
          kernel=kernel)

    rows = structural if isinstance(structural, list) else [structural]
    all_checks = [c for r in rows for c in r.get("checks", [])]
    failed = [c for c in all_checks if not c.get("pass")]
    check("graph_structurally_legal",
          bool(all_checks) and not failed and all(r.get("verdict") == "PASS" for r in rows),
          n_checks=len(all_checks), failed=[c.get("check") for c in failed],
          verdicts=[r.get("verdict") for r in rows])

    frac = _parse_fractional(rows)
    check("graph_coordinates_integer", frac == 0, fractional=frac,
          why="FACT-0344: a float export lost 0.014 on this leaderboard; integer coordinates are "
              "a release condition on this competition, not a formatting preference")

    head = rows[0] if rows else {}
    check("graph_non_empty_and_degree_legal",
          (head.get("nodes") or 0) > 0 and (head.get("edges") or 0) > 0
          and head.get("max_indegree") == 1 and head.get("max_outdegree") in (1, 2),
          nodes=head.get("nodes"), edges=head.get("edges"), divisions=head.get("divisions"),
          max_indegree=head.get("max_indegree"), max_outdegree=head.get("max_outdegree"),
          datasets=head.get("datasets"))

    # ---- C6 the score, against its own preregistration ------------------------------------
    exps = {e["id"]: e for e in _load_registry_yaml("experiments.yaml")["experiments"]}
    facts = _load_registry_yaml("facts.yaml")["facts"]
    exp = exps.get(experiment_id)
    if exp is None:
        check("experiment_in_registry", False, experiment=experiment_id)
        out["passed"] = all(c["passed"] for c in checks.values())
        return out
    check("experiment_in_registry", True, status=exp.get("status"), lb=exp.get("lb"),
          submission=exp.get("submission"), kernel=exp.get("kernel"),
          kernel_version=exp.get("kernel_version"))
    check("experiment_binds_this_spec",
          Path(str(exp.get("spec"))).name == spec_path.name
          and str(exp.get("kernel", "")).split("/")[-1] == meta_slug
          and exp.get("status") == "scored",
          registry_spec=exp.get("spec"), registry_kernel=exp.get("kernel"),
          registry_status=exp.get("status"))

    # TWO PREREGISTRATION SHAPES EXIST IN facts.yaml AND BOTH ARE REAL PREREGISTRATIONS.
    # FACT-0393 nests them under scope.preregistered / scope.outcome; FACT-0341 (the outgoing
    # champion) uses the older flat scope.band_prereg / central_prereg / falsifier / fired. Reading
    # only the new shape reported the champion as never having preregistered a band, which is false
    # - it declared 0.921-0.929 before the run. Normalise, do not FAIL on a schema difference.
    def _prereg(f: dict) -> tuple[dict, dict] | None:
        sc = f.get("scope") or {}
        if isinstance(sc.get("preregistered"), dict):
            return sc["preregistered"], (sc.get("outcome") or {})
        if isinstance(sc.get("band_prereg"), list):
            return ({"band": sc["band_prereg"], "central": sc.get("central_prereg"),
                     "falsifier": sc.get("falsifier"), "_shape": "legacy_flat"},
                    {"scored": f.get("value"), "falsifier_fired": sc.get("fired")})
        return None

    score_facts = [f for f in facts
                   if f.get("experiment") == experiment_id and _prereg(f) is not None]
    if not score_facts:
        check("score_fact_present", False, experiment=experiment_id,
              reason="no fact for this experiment carries a preregistered band")
        out["passed"] = all(c["passed"] for c in checks.values())
        return out
    sf = score_facts[-1]
    pre, outcome = _prereg(sf)
    lo, hi = (pre.get("band") or [None, None])[:2]
    scored = outcome.get("scored", sf.get("value"))
    check("score_fact_present", True, fact=sf["id"], value=sf.get("value"),
          provenance=sf.get("provenance"), validity=sf.get("validity"),
          preregistration_shape=pre.get("_shape", "scope.preregistered"))
    check("score_matches_registry_experiment", scored == exp.get("lb"),
          fact_score=scored, experiment_lb=exp.get("lb"))
    check("score_inside_preregistered_band",
          lo is not None and hi is not None and lo <= scored <= hi,
          band=[lo, hi], central=pre.get("central"), scored=scored)
    check("falsifier_did_not_fire", outcome.get("falsifier_fired") is False,
          falsifier=pre.get("falsifier"), fired=outcome.get("falsifier_fired"))
    check("submission_id_consistent",
          str((sf.get("scope") or {}).get("submission")) == str(exp.get("submission")),
          fact_submission=(sf.get("scope") or {}).get("submission"),
          experiment_submission=exp.get("submission"))

    # ---- C7 upstream provenance (optional but named) ---------------------------------------
    if reference_nb is not None:
        if not reference_nb.is_file():
            check("reproduces_upstream_source", False, reference=str(reference_nb),
                  reason="reference notebook not found")
        else:
            ours = _cells(json.loads(nb.read_text(encoding="utf-8")))
            theirs = _cells(json.loads(reference_nb.read_text(encoding="utf-8")))
            joiner = "\n<<CELL>>\n"
            check("reproduces_upstream_source", ours == theirs,
                  reference=str(reference_nb), n_cells_ours=len(ours), n_cells_reference=len(theirs),
                  cell_source_sha256_ours=sha256_bytes(joiner.join(ours).encode()),
                  cell_source_sha256_reference=sha256_bytes(joiner.join(theirs).encode()),
                  note="cell SOURCES are compared, not file bytes: kaggle_factory re-serialises "
                       "the JSON at build time, which changes the file hash without changing a "
                       "single character of executed code")

    out["passed"] = all(c["passed"] for c in checks.values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--experiment", required=True, help="EXP-#### this artifact was run as")
    ap.add_argument("--at-commit", default=None,
                    help="commit the built notebook must be identical to")
    ap.add_argument("--fetched-dir", type=Path, default=None,
                    help="where the fetched submission artifact lives, if neither "
                         "<out_dir>/_out/ nor C:/temp/queue/<name>/")
    ap.add_argument("--reference-notebook", type=Path, default=None,
                    help="upstream source this artifact claims to reproduce, for cell-source equality")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = args.spec if args.spec.is_absolute() else ROOT / args.spec
    res = audit_release(spec, args.experiment, args.reference_notebook, args.at_commit,
                        args.fetched_dir)
    res["schema_version"] = 1
    res["head_commit"] = _git("rev-parse", "HEAD")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    for name, c in res["checks"].items():
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  {mark}  {name}"
              + ("" if c["passed"] else f"   {json.dumps({k: v for k, v in c.items() if k != 'passed'})[:400]}"))
    print(f"{HEARTBEAT} spec={res.get('name')} experiment={res['experiment']} "
          f"verdict={'PASS' if res['passed'] else 'FAIL'} -> {args.out}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
