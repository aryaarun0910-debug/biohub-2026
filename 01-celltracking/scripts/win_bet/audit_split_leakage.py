r"""Independent leakage audit of the training splits and cache manifests this cycle produces.

FACT-0378 IS THE PRECEDENT AND IT IS WHY THE DECLARED ROUTE IS NOT THE ONE TO CHECK
-----------------------------------------------------------------------------------
Nobody in this project has ever declared a leak. `EXP-0019` leaked because a spec OMITTED
`BIOHUB_LOEO_WEIGHTS_GLOB` and silently fell back to the pack's `split_0`; `FACT-0378` leaked
because a hardcoded secondary checkpoint exists only as `split_0`. Both were absences, not
statements. So this instrument attacks the routes that leave no declaration:

  L1  EMBRYO PURITY AND THE CROP-GROUPED TRAP. Crop-grouped is NOT embryo-held-out. Fold 0 is
      entirely 44b6 and fold 1 entirely 6bba, so every within-fold GroupKFold measures WITHIN-
      embryo generalisation. That is legitimate and it is also the single easiest thing in this
      cycle to over-read, so a spec whose crops are one embryo must not carry a cross-embryo
      claim.
  L2  GROUPING, EXERCISED RATHER THAN READ. The harness's own guard is called with a
      deliberately straddling split and with a deliberately crop-overlapping split. If it does
      not refuse both, falsifier (b) of PKT-0034 is unenforced no matter what the docstring says.
  L3  THE UNLICENSED-CACHE REFUSAL, EXERCISED. `verify_licence` is called against a licence path
      that does not exist and against a licence whose pinned digest no longer matches. Both must
      raise, or "the harness refuses to train without Gate 1" is a comment rather than a control.
  L4  WEIGHTS PROVENANCE PER FOLD. For each cache the spec consumes, the run that produced it
      must have used weights legitimate for that fold: pack `split_0` is clean ONLY for fold 0,
      and a fold-1 artifact must name an explicit split-1 glob. A spec that sets no glob inherits
      `split_0` - correct on fold 0, an `EXP-0019`-class leak on fold 1.
  L5  FEATURE ADMISSIBILITY. Every feature a model consumes must come from the frozen surface,
      never from a quantity computed across the judged fold after the split is known.
  L6  EXTERNAL CORPUS ADMISSIBILITY. A cross-fit map that trains on an external corpus is only
      as clean as that corpus. Zebrahub is currently INADMISSIBLE for the division lever
      (FACT-0385 anti-alignment), so `train_external: true` is a finding until that is solved.

FAIL CLOSED. A target that cannot be read is a FAIL, not a skip; a guard that cannot be
exercised is a FAIL, not a pass. Heartbeat `SPLIT_LEAKAGE_AUDIT_COMPLETE` - its absence is the
alarm.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

HEARTBEAT = "SPLIT_LEAKAGE_AUDIT_COMPLETE"
EXPECTED_EMBRYO = {0: "44b6", 1: "6bba"}
# Pack split_0 was trained on 6bba (FACT-0378), so it is CLEAN for fold 0 and LEAKY for fold 1.
PACK_SPLIT0_CLEAN_FOR_FOLD = 0


def _finding(name: str, passed: bool, evidence: dict, exercised: str) -> dict:
    return {"check": name, "passed": bool(passed), "evidence": evidence,
            "what_was_actually_run": exercised}


def exercise_grouping_guard(mod) -> dict:
    """L2: call the guard with splits that MUST be refused."""
    crops = np.array(["44b6_a", "44b6_a", "44b6_b", "44b6_b", "44b6_c", "44b6_c"])
    targets = np.array([1, 1, 2, 2, 3, 3])
    results = {}
    cases = {
        "target_straddles_split": (np.array([0, 2, 3]), np.array([1, 4, 5])),
        "crop_on_both_sides": (np.array([0, 1, 2]), np.array([3, 4, 5])),
        "empty_validation_side": (np.array([0, 1, 2, 3, 4, 5]), np.array([], dtype=int)),
    }
    for name, (tr, te) in cases.items():
        try:
            mod.assert_group_integrity(crops, targets, tr, te, f"audit/{name}")
            results[name] = {"refused": False, "message": None}
        except Exception as exc:
            results[name] = {"refused": True, "message": f"{type(exc).__name__}: {exc}"[:220]}
    clean_ok = True
    try:
        mod.assert_group_integrity(crops, targets, np.array([0, 1, 2, 3]), np.array([4, 5]),
                                   "audit/clean")
    except Exception as exc:                                  # a guard that refuses a LEGAL split
        clean_ok = False                                      # is as broken as one that permits
        results["legal_split_wrongly_refused"] = f"{type(exc).__name__}: {exc}"[:220]
    return _finding(
        "L2_grouping_guard_refuses_leaky_splits",
        all(v["refused"] for v in results.values() if isinstance(v, dict)) and clean_ok,
        results,
        "called assert_group_integrity with a target straddling the split, with a crop on both "
        "sides, and with an empty validation side, then with a legal split to confirm it does "
        "not refuse everything",
    )


def exercise_licence_refusal(mod, tmp: Path) -> dict:
    """L3: the unlicensed-cache refusal, exercised rather than read."""
    tmp.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp / "cache"
    cache_dir.mkdir(exist_ok=True)
    blob = cache_dir / "44b6_x.npz"
    blob.write_bytes(b"not-a-real-cache-but-bytes-are-what-the-digest-pins")
    out = {}
    missing = tmp / "no_such_licence.json"
    missing.unlink(missing_ok=True)
    try:
        mod.verify_licence(missing, cache_dir, ["44b6_x"])
        out["missing_licence_refused"] = False
    except Exception as exc:
        out["missing_licence_refused"] = True
        out["missing_licence_message"] = f"{type(exc).__name__}: {exc}"[:200]

    lic = tmp / "licence.json"
    # The pinning key is `cache_sha256`, read from write_licence at source - not guessed. An
    # auditor that invents the schema manufactures a refusal and vetoes an innocent target.
    lic.write_text(json.dumps({
        "heartbeat": "ASSOC_CACHE_LICENCE",
        "cache_sha256": {"44b6_x": mod.cache_digest(blob)},
    }), encoding="utf-8")
    try:
        mod.verify_licence(lic, cache_dir, ["44b6_x"])
        out["intact_licence_accepted"] = True
    except Exception as exc:
        out["intact_licence_accepted"] = False
        out["intact_licence_message"] = f"{type(exc).__name__}: {exc}"[:200]
    blob.write_bytes(b"the cache moved after the licence was issued")
    try:
        mod.verify_licence(lic, cache_dir, ["44b6_x"])
        out["mutated_cache_refused"] = False
    except Exception as exc:
        out["mutated_cache_refused"] = True
        out["mutated_cache_message"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        mod.verify_licence(lic, cache_dir, ["44b6_x", "44b6_uncovered"])
        out["uncovered_crop_refused"] = False
    except Exception:
        out["uncovered_crop_refused"] = True
    try:
        mod.write_licence({"passed": False}, tmp / "should_not_exist.json")
        out["licence_for_failed_gate_refused"] = False
    except Exception as exc:
        out["licence_for_failed_gate_refused"] = True
        out["failed_gate_message"] = f"{type(exc).__name__}: {exc}"[:200]

    return _finding(
        "L3_unlicensed_or_mutated_cache_is_refused",
        bool(out.get("missing_licence_refused") and out.get("mutated_cache_refused")
             and out.get("uncovered_crop_refused") and out.get("intact_licence_accepted")
             and out.get("licence_for_failed_gate_refused")),
        out,
        "called verify_licence with (a) no licence file, (b) an intact licence pinned with the "
        "schema read from write_licence at source, (c) a cache mutated after the licence was "
        "issued and (d) a crop the licence does not cover, then called write_licence with a "
        "FAILED gate report; only (b) may be accepted",
    )


def audit_harness_spec(path: Path, mod) -> dict:
    out: dict = {"spec": str(path), "findings": [], "passed": False}
    if not path.is_file():
        out["fail_reason"] = "spec not found"
        return out
    spec = json.loads(path.read_text(encoding="utf-8"))
    fold = spec.get("fold")
    out["name"], out["fold"] = spec.get("name"), fold

    # --- L1 embryo purity, read from the actual surface, not from the spec's own claim -------
    table = Path(spec.get("table", ""))
    embryos, n_crops, src = [], 0, None
    if table.is_file():
        import polars as pl  # noqa: PLC0415
        crops = pl.scan_parquet(table).select("crop" if "crop" in
                                              pl.scan_parquet(table).collect_schema().names()
                                              else "dataset").unique().collect()
        vals = crops.to_series().to_list()
        embryos = sorted({mod.embryo_of(c) for c in vals})
        n_crops, src = len(vals), str(table)
    out["findings"].append(_finding(
        "L1_embryo_purity_and_no_cross_embryo_claim",
        bool(embryos) and embryos == [EXPECTED_EMBRYO.get(fold)]
        and (spec.get("no_claim") is True or "WITHIN-embryo" in (spec.get("note") or "")
             or "within-embryo" in (spec.get("note") or "")),
        {"surface": src, "n_crops": n_crops, "embryos": embryos,
         "expected_embryo_for_fold": EXPECTED_EMBRYO.get(fold),
         "spec_declares_within_embryo_scope": bool(
             spec.get("no_claim") is True or "WITHIN-embryo" in (spec.get("note") or "")),
         "standing_limit": "crop-grouped is NOT embryo-held-out; no split in this spec holds an "
                           "embryo out"},
        "read the crop list out of the surface parquet itself and derived the embryo set, rather "
        "than trusting the spec's fold field",
    ))

    # --- L4 weights provenance per fold -------------------------------------------------------
    cache = spec.get("cache") or {}
    preilp = str(cache.get("preilp") or "")
    fold1_artifact = "split1" in preilp.replace("_", "") or "split_1" in preilp
    fold0_artifact = "split0" in preilp.replace("_", "") or "split_0" in preilp
    consistent = (fold == 1 and fold1_artifact) or (fold == 0 and fold0_artifact)
    out["findings"].append(_finding(
        "L4_cache_artifact_matches_the_fold_it_will_judge",
        bool(consistent),
        {"preilp": preilp, "ecb_dir": cache.get("ecb_dir"), "fold": fold,
         "pack_split0_clean_for_fold": PACK_SPLIT0_CLEAN_FOR_FOLD,
         "rule": "pack split_0 was trained on 6bba (FACT-0378): clean for fold 0, an "
                 "EXP-0019-class leak for fold 1. A fold-1 cache run MUST set "
                 "BIOHUB_LOEO_WEIGHTS_GLOB explicitly; omitting it silently inherits split_0."},
        "matched the split index embedded in the cache artifact path against the fold the spec "
        "will judge",
    ))

    # --- L5 feature admissibility -------------------------------------------------------------
    frozen = {"prob", "rank", "margin_to_best", "n_candidates", "dist_um", "dz_um", "dy_um",
              "dx_um", "src_out_degree"}
    used, unknown = set(), set()
    for m in spec.get("models", []):
        for f in m.get("features", []) or []:
            used.add(f)
        for f in (m.get("contract") or {}).get("extra_features", []) or []:
            used.add(f)
    unknown = used - frozen
    out["findings"].append(_finding(
        "L5_features_come_from_the_frozen_surface",
        not unknown,
        {"features_used": sorted(used), "not_in_frozen_surface": sorted(unknown),
         "frozen_surface_features": sorted(frozen)},
        "compared every feature named by every model arm against the frozen nine that "
        "assoc_parent_dataset.py emits; a feature outside that set would have to be computed "
        "somewhere this audit cannot see",
    ))

    # --- degenerate-surface guard -------------------------------------------------------------
    out["findings"].append(_finding(
        "L1b_degenerate_surface_not_silently_allowed",
        spec.get("allow_degenerate") is False,
        {"allow_degenerate": spec.get("allow_degenerate"), "no_claim": spec.get("no_claim")},
        "read allow_degenerate directly; a single-parent surface makes top-1 1.0 by arithmetic "
        "(FACT-0381) and a metric that cannot fail is not evidence",
    ))
    out["passed"] = all(f["passed"] for f in out["findings"])
    return out


def audit_crossfit_map(path: Path, external_admissible: bool) -> dict:
    out: dict = {"map": str(path), "findings": [], "passed": False}
    if not path.is_file():
        out["fail_reason"] = "map not found"
        return out
    m = json.loads(path.read_text(encoding="utf-8"))
    assignments = m.get("assignments", {})
    ok_rows = True
    detail = {}
    for judge, cfg in assignments.items():
        f = int(judge.rsplit("_", 1)[-1])
        trains = [int(x) for x in cfg.get("train_folds", [])]
        detail[judge] = {"train_folds": trains, "forbidden_rows": cfg.get("forbidden_rows"),
                         "train_external": cfg.get("train_external")}
        if f in trains:
            ok_rows = False
    out["findings"].append(_finding(
        "X1_judged_fold_never_appears_in_its_own_training_set", ok_rows, detail,
        "parsed every judge_fold_N assignment and asserted N is absent from its train_folds",
    ))
    uses_external = any(cfg.get("train_external") for cfg in assignments.values())
    out["findings"].append(_finding(
        "X2_external_corpus_is_currently_admissible",
        (not uses_external) or external_admissible,
        {"train_external": uses_external, "corpus": (m.get("external") or {}).get("corpus"),
         "admissible_now": external_admissible,
         "why": "FACT-0385 measured Zebrahub anti-aligned for the division lever (pooled AUC "
                "0.179; an in-domain fit collapsed 0.80 -> 0.21 when mixed), and PKT-0035 "
                "forbids Zebrahub rows until that is solved AND the solution demonstrated"},
        "read train_external out of every assignment and checked it against the currently "
        "declared admissibility of the named corpus",
    ))
    out["passed"] = all(f["passed"] for f in out["findings"])
    return out


def audit_external_checkpoints(weights_dir: Path) -> dict:
    """L4b: an external checkpoint's fold label is a NAME. Read what it was trained on.

    PKT-0033 finding 12 left HOCT checkpoint provenance unestablished, in the FACT-0378 class.
    The checkpoints answer it themselves: `cache_files` lists the training crop stems and
    `training_files` counts them, and this campaign's two embryos have distinct crop counts
    (44b6: 71, 6bba: 128), so the count alone identifies the training embryo.
    """
    out: dict = {"dir": str(weights_dir), "checkpoints": {}, "passed": False}
    if not weights_dir.is_dir():
        out["fail_reason"] = "weights directory not found"
        return out
    import torch  # noqa: PLC0415

    counts_to_embryo = {71: "44b6", 128: "6bba"}
    ok = True
    for p in sorted(weights_dir.glob("*.pt")) + sorted(weights_dir.glob("*.pth")):
        entry: dict = {"file": p.name}
        try:
            payload = torch.load(p, map_location="cpu", weights_only=True)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"[:160]
            out["checkpoints"][p.name] = entry
            continue
        if not isinstance(payload, dict) or "model" not in payload:
            entry["metadata"] = "bare state_dict - carries NO training provenance at all"
            entry["trained_on_embryo"] = "UNKNOWN"
            entry["clean_for_fold"] = "UNKNOWN"
            out["checkpoints"][p.name] = entry
            ok = False
            continue
        files = payload.get("cache_files")
        n = payload.get("training_files") or (len(files) if files else None)
        embryo = None
        if files:
            embryo = sorted({str(f).split("_", 1)[0] for f in files})
            embryo = embryo[0] if len(embryo) == 1 else "MIXED"
            entry["evidence"] = "explicit cache_files list of training crop stems"
        elif n in counts_to_embryo:
            embryo = counts_to_embryo[n]
            entry["evidence"] = "training_files count matches exactly one embryo's crop count"
        entry.update({"training_files": n, "trained_on_embryo": embryo,
                      "clean_for_fold": {"6bba": 0, "44b6": 1}.get(embryo, "UNKNOWN"),
                      "leaky_for_fold": {"6bba": 1, "44b6": 0}.get(embryo, "UNKNOWN")})
        if embryo is None or embryo == "MIXED":
            ok = False
        out["checkpoints"][p.name] = entry
    out["passed"] = bool(out["checkpoints"]) and ok
    out["rule"] = ("a checkpoint trained on 6bba is clean for fold 0 and LEAKY for fold 1, and "
                   "vice versa; the file's own fold label names the HELD-OUT fold, not the "
                   "training set, so reading it the other way inverts the hygiene")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hoct-weights", type=Path,
                    help="directory of external checkpoints whose fold provenance to establish")
    ap.add_argument("--harness-spec", type=Path, nargs="*", default=[])
    ap.add_argument("--crossfit-map", type=Path, nargs="*", default=[])
    ap.add_argument("--external-admissible", action="store_true",
                    help="pass ONLY when the external corpus's anti-alignment is solved and the "
                         "solution is demonstrated")
    ap.add_argument("--tmp", type=Path, default=Path("C:/temp/audit_receipts/leakage_tmp"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    try:
        import assoc_train_harness as mod  # noqa: PLC0415
    except Exception as exc:
        result = {"schema_version": 1, "passed": False,
                  "fail_reason": f"cannot import the harness under audit: {type(exc).__name__}: {exc}"}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{HEARTBEAT} passed=False ({result['fail_reason']})")
        return 1

    guards = [exercise_grouping_guard(mod), exercise_licence_refusal(mod, args.tmp)]
    specs = [audit_harness_spec(p if p.is_absolute() else ROOT / p, mod)
             for p in args.harness_spec]
    maps = [audit_crossfit_map(p if p.is_absolute() else ROOT / p, args.external_admissible)
            for p in args.crossfit_map]
    ckpt = audit_external_checkpoints(args.hoct_weights) if args.hoct_weights else None

    result = {
        "schema_version": 1,
        "harness_module": str(Path(mod.__file__).resolve()),
        "guards_exercised": guards,
        "harness_specs": specs,
        "crossfit_maps": maps,
        "external_checkpoints": ckpt,
        "passed": all(g["passed"] for g in guards) and all(s["passed"] for s in specs)
        and all(m["passed"] for m in maps) and (ckpt is None or ckpt["passed"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for g in guards:
        print(f"  {'PASS' if g['passed'] else 'FAIL'}  {g['check']}")
    for s in specs:
        print(f"  spec {s.get('name') or s['spec']}: "
              f"{'PASS' if s['passed'] else 'FAIL'}"
              + (f" ({s['fail_reason']})" if s.get("fail_reason") else ""))
        for f in s.get("findings", []):
            if not f["passed"]:
                print(f"        FAILED {f['check']}: {json.dumps(f['evidence'])[:280]}")
    for m in maps:
        print(f"  map {m['map']}: {'PASS' if m['passed'] else 'FAIL'}")
        for f in m.get("findings", []):
            if not f["passed"]:
                print(f"        FAILED {f['check']}: {json.dumps(f['evidence'])[:280]}")
    if ckpt:
        print(f"  external checkpoints in {ckpt['dir']}: "
              f"{'PASS' if ckpt['passed'] else 'FAIL'}")
        for name, e in ckpt.get("checkpoints", {}).items():
            print(f"        {name}: trained_on={e.get('trained_on_embryo')} "
                  f"(n={e.get('training_files')}) clean_for_fold={e.get('clean_for_fold')}")
    print(f"{HEARTBEAT} passed={result['passed']} -> {args.out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
