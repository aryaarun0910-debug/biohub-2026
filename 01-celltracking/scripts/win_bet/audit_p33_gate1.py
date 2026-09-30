r"""Fail-closed receipt for PKT-0029 Gate 1 (EXP-0041), against all SIX preregistered conditions.

WHY THIS EXISTS AS A SEPARATE INSTRUMENT FROM THE KERNEL'S OWN `all_passed`
---------------------------------------------------------------------------
`all_passed` is computed inside the run, and it is NOT the gate. It requires a non-empty crop
list with every entry passing - but it does NOT assert how many crops there are. A run that
silently processed ONE crop reports `all_passed: true`. Condition (1) is therefore EXTERNAL BY
CONSTRUCTION and is enforced here: a crop count other than the preregistered one is a FAIL
regardless of what the kernel said. The kernel was pushed unchanged as directed, so this is the
only place that check can live.

TWO MORE PLACES A KERNEL PASS IS NOT A GATE PASS, both enforced here and NOT in-kernel
---------------------------------------------------------------------------------------
  * BAND A CAN COMPARE NOTHING. `passed` in the kernel requires `b_checked > 0` for band B but
    imposes no equivalent floor on band A: if the frame cap leaves no recorded pre-ILP edge in
    scope, `a_missing` and `a_extra` are both empty and `a_delta` is 0.0, so band A passes having
    compared zero pairs. Condition (4) would then be vacuous - and so would condition (3), which
    is proven only TRANSITIVELY through band A. This audit requires a strictly positive number of
    SHARED band-A pairs per crop.
  * THE CACHE THE VERIFIER READ IS NOT PROVEN TO BE THE CACHE THE WRITER WROTE by the report
    alone. The two phases are separate processes, which is what makes "from the cache alone" real,
    but a stale cache directory from an earlier attempt would satisfy every in-report check. The
    per-crop node count printed by the CACHE phase is compared here against the node count printed
    by the VERIFY phase, which is the only cross-phase tie the retained artifacts carry.

ON CONDITION (3) AND WHY THE TRANSITIVE ARGUMENT IS ACCEPTED ONLY WITH A FLOOR AND A CAVEAT
-------------------------------------------------------------------------------------------
The preregistered text says node ordering and identity are proven transitively: a reordering
would change which positional index each recorded edge names, so band A matching exactly implies
the reconstruction addressed the same nodes. That reasoning holds - but only over nodes that
appear in at least one COMPARED pair. A permutation confined to nodes that appear in no band-A
and no band-B pair is invisible to it, and so is any permutation that happens to be an
automorphism of the compared edge set. The receipt records that as a stated limit on the licence,
alongside the softmax blind spot, rather than reporting condition (3) as unconditional.

FAIL CLOSED. A missing report, unreadable JSON, an empty crop list, a missing log or a report
carrying an `error` key all produce a FAIL receipt - never an exception that could be mistaken
for "nothing to check". The positive heartbeat is the last line: its ABSENCE is the alarm.

SELF-TEST. `--selftest` mutates a synthetic PASSING report six ways - one per condition, each a
mutation the kernel's own `all_passed` would either accept or never see - and asserts this harness
rejects every one. It is the evidence that a PASS from this instrument was earned rather than
defaulted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

TOL = 1e-4
HEARTBEAT = "P33_GATE1_RECEIPT_COMPLETE"

# The verify phase prints one of these per crop; the cache phase prints AFP_CACHE.
RE_VERIFY = re.compile(r"AFP crop=(\S+) frames=(\d+) nodes=(\d+) ")
RE_CACHE = re.compile(r"AFP_CACHE crop=(\S+) frames=(\d+) nodes=(\d+)")
RE_PHASE = re.compile(r"AFP_PHASE (cache|verify):")


def _cond(name: str, passed: bool, evidence: dict, exercised: str) -> dict:
    """Every condition carries what was RUN that would have caught a failure."""
    return {
        "condition": name,
        "passed": bool(passed),
        "evidence": evidence,
        "what_would_have_caught_a_failure": exercised,
    }


def check_report(report: dict, log: str, expected_crops: int, tol: float,
                 patch_src: str | None) -> list[dict]:
    crops = report.get("crops") or []
    conds: list[dict] = []

    # ---- (1) both crop heartbeats appear -- EXTERNAL BY CONSTRUCTION -------------------
    log_verify = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in RE_VERIFY.finditer(log)}
    log_cache = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in RE_CACHE.finditer(log)}
    conds.append(_cond(
        "1_both_crop_heartbeats",
        len(crops) == expected_crops
        and len(log_verify) == expected_crops
        and len(log_cache) == expected_crops,
        {
            "report_crops": len(crops),
            "expected_crops": expected_crops,
            "log_verify_crops": sorted(log_verify),
            "log_cache_crops": sorted(log_cache),
            "kernel_all_passed": report.get("all_passed"),
        },
        "counted crops in the report AND distinct AFP/AFP_CACHE crop heartbeats in the log; "
        "the kernel's all_passed does not assert the count, so a one-crop run would pass it",
    ))

    # ---- (2) the fresh reader used only persisted cache artifacts ---------------------
    phases = [m.group(1) for m in RE_PHASE.finditer(log)]
    node_ties = {
        c: {"cache_nodes": log_cache[c][1], "verify_nodes": log_verify.get(c, (0, -1))[1]}
        for c in log_cache
    }
    ties_agree = bool(node_ties) and all(
        v["cache_nodes"] == v["verify_nodes"] for v in node_ties.values()
    )
    verify_body = ""
    if patch_src:
        i = patch_src.find("def phase_verify(")
        j = patch_src.find("\ndef main(", i) if i >= 0 else -1
        verify_body = patch_src[i:j] if i >= 0 and j > i else ""
    reader_is_clean = bool(verify_body) and "open_dataset" not in verify_body
    conds.append(_cond(
        "2_fresh_reader_used_only_persisted_cache",
        phases[:2] == ["cache", "verify"] and ties_agree and reader_is_clean,
        {
            "phase_launch_order": phases,
            "per_crop_node_count_tie": node_ties,
            "verify_body_opens_image_dataset": (not reader_is_clean) if verify_body else "UNKNOWN",
            "verify_body_chars": len(verify_body),
        },
        "matched the CACHE-phase node count against the VERIFY-phase node count per crop, which "
        "a stale or partial cache directory would break, and re-read the committed worker source "
        "to confirm phase_verify never opens the image dataset",
    ))

    # ---- (3) node ordering and identities reconstruct exactly (TRANSITIVE) -------------
    mism = {c.get("crop"): c.get("node_count_mismatches") or [] for c in crops}
    shared = {}
    for c in crops:
        a = c.get("band_a") or {}
        shared[c.get("crop")] = int(a.get("recorded", 0)) - int(a.get("missing", 0))
    conds.append(_cond(
        "3_node_ordering_and_identity_transitive",
        bool(crops)
        and all(not v for v in mism.values())
        and all(v > 0 for v in shared.values()),
        {
            "node_count_mismatches_per_crop": mism,
            "band_a_shared_pairs_per_crop": shared,
            "transitive_premise": "band A matches exactly on a NON-EMPTY shared set",
            "limit": "a permutation confined to nodes appearing in no compared pair, or one that "
                     "is an automorphism of the compared edge set, is not visible to this check",
        },
        "required a strictly positive band-A shared count per crop -- the kernel imposes that "
        "floor only on band B, so a frame cap leaving no in-scope recorded edge would make both "
        "condition (3) and condition (4) vacuous while all_passed stayed true",
    ))

    # ---- (4) deployed candidates match exactly ----------------------------------------
    a_rows = {c.get("crop"): c.get("band_a") or {} for c in crops}
    a_ok = bool(crops) and all(
        int(a.get("recorded", 0)) > 0
        and int(a.get("missing", 1)) == 0
        and int(a.get("extra", 1)) == 0
        and float(a.get("max_abs_prob_delta", 1.0)) <= tol
        for a in a_rows.values()
    )
    conds.append(_cond("4_deployed_candidates_exact", a_ok,
                       {"band_a_per_crop": a_rows, "tolerance": tol},
                       "asserted zero missing, zero extra and max |dp| <= tol on a NON-EMPTY "
                       "recorded set for every crop"))

    # ---- (5) the richer probability band matches within the frozen tolerance -----------
    b_rows = {c.get("crop"): c.get("band_b") or {} for c in crops}
    b_ok = bool(crops) and all(
        int(b.get("checked", 0)) > 0
        and int(b.get("missing", 1)) == 0
        and float(b.get("max_abs_prob_delta", 1.0)) <= tol
        for b in b_rows.values()
    )
    conds.append(_cond("5_sub_threshold_band_within_tolerance", b_ok,
                       {"band_b_per_crop": b_rows, "tolerance": tol,
                        "extras_beyond_sidecar_topk": "expected, not a failure"},
                       "asserted a non-zero checked count, zero missing and max |dp| <= tol per "
                       "crop -- this is the band FACT-0382 puts all 691 contested errors in, and "
                       "the band attempt 1 never compared"))

    # ---- (6) missing or empty cache output fails closed --------------------------------
    structural = None
    if patch_src:
        structural = {
            "raises_on_missing_cache": "raise FileNotFoundError(" in patch_src,
            "requires_nonzero_band_b": "b_checked > 0" in patch_src,
            "band_b_absent_is_fatal": "band-A-only pass is not Gate 1" in patch_src,
        }
    conds.append(_cond(
        "6_missing_or_empty_cache_fails_closed",
        "error" not in report and "traceback" not in report
        and "AFP_FAILED" not in log
        and "ASSOC_FEATURE_PARITY_COMPLETE all_passed= True" in log
        and bool(structural) and all(structural.values()),
        {
            "report_error_key": report.get("error"),
            "log_has_AFP_FAILED": "AFP_FAILED" in log,
            "completion_heartbeat": "ASSOC_FEATURE_PARITY_COMPLETE all_passed= True" in log,
            "worker_source_structure": structural,
        },
        "re-read the committed worker source for the FileNotFoundError raise, the b_checked>0 "
        "term in `passed` and the fatal missing-sidecar branch, and required the completion "
        "heartbeat with the absence of AFP_FAILED; the runtime behaviour of both is exercised "
        "by tests/test_assoc_feature_parity.py, which runs the worker as real subprocesses",
    ))
    return conds


def build_receipt(report_path: Path, log_path: Path, expected_crops: int, tol: float,
                  patch_path: Path | None, commit: str, kernel: str,
                  kernel_version: int | None) -> dict:
    receipt = {
        "schema_version": 1,
        "artifact": "P33 v2 association feature-parity smoke (EXP-0041, PKT-0029 Gate 1)",
        "kernel": kernel,
        "kernel_version": kernel_version,
        "commit": commit,
        "report_path": str(report_path),
        "log_path": str(log_path),
        "conditions": [],
        "verdict": "FAIL",
        "licences": [],
        "blind_spot": (
            "The gate compares probabilities AFTER a softmax over the SOURCE axis. Softmax is "
            "invariant to a constant offset along the axis it normalises, so a cache error that "
            "shifts the LOGIT of every source for a given target by the same amount is "
            "mathematically invisible here. A PASS therefore licenses only the absence of "
            "NON-UNIFORM cache error - a single node's features, an index remap, a dtype change "
            "or a stale frame all show up. It does not license belief that the cache is perfect. "
            "Closing it would need pre-softmax logits, which neither the pre-ILP export nor the "
            "ECB sidecars record, i.e. a new acquisition rather than a change to this gate."
        ),
    }
    if not report_path.is_file():
        receipt["fail_reason"] = f"report not found: {report_path}"
        return receipt
    if not log_path.is_file():
        receipt["fail_reason"] = f"kernel log not found: {log_path}"
        return receipt
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        receipt["fail_reason"] = f"unreadable report: {exc}"
        return receipt
    log = log_path.read_text(encoding="utf-8", errors="replace")
    patch_src = patch_path.read_text(encoding="utf-8") if patch_path and patch_path.is_file() else None
    if patch_src is None:
        receipt["fail_reason"] = f"worker source not readable: {patch_path}"
        return receipt

    receipt["kernel_all_passed"] = report.get("all_passed")
    receipt["conditions"] = check_report(report, log, expected_crops, tol, patch_src)
    ok = all(c["passed"] for c in receipt["conditions"])
    receipt["verdict"] = "PASS" if ok else "FAIL"
    receipt["failed_conditions"] = [c["condition"] for c in receipt["conditions"] if not c["passed"]]
    receipt["licences"] = (
        ["the two fold feature-cache GPU sessions for LEVER-0034 may proceed"] if ok else []
    )
    if not ok:
        receipt["veto"] = (
            "VETOED: no feature cache may be built and no head may be trained on a cache this "
            "gate did not pass. Failed: " + ", ".join(receipt["failed_conditions"])
        )
    return receipt


# --------------------------------------------------------------------------- self-test
_SYNTH_CROP = {
    "crop": "44b6_0113de3b", "frames": 8, "nodes": 1000, "node_count_mismatches": [],
    "band_a": {"recorded": 900, "reproduced": 900, "missing": 0, "extra": 0,
               "max_abs_prob_delta": 1e-7},
    "band_b": {"recorded_sub_threshold": 4000, "checked": 4000, "missing": 0,
               "max_abs_prob_delta": 2e-7},
    "passed": True,
}


def _synth(n_crops: int = 2) -> tuple[dict, str]:
    crops = []
    for i in range(n_crops):
        c = json.loads(json.dumps(_SYNTH_CROP))
        c["crop"] = f"44b6_crop{i}"
        crops.append(c)
    report = {"gate": "feature_parity", "attempt": 2, "crops": crops, "all_passed": True}
    lines = ["AFP_PHASE cache: /usr/bin/python3 afp_gate1.py --phase cache ..."]
    lines += [f"AFP_CACHE crop={c['crop']} frames=8 nodes={c['nodes']}" for c in crops]
    lines += ["AFP_PHASE verify: /usr/bin/python3 afp_gate1.py --phase verify ..."]
    lines += [
        f"AFP crop={c['crop']} frames=8 nodes={c['nodes']} A[rec=900 miss=0 extra=0 d=1.0e-07] "
        f"B[rec=4000 checked=4000 miss=0 d=2.0e-07] ncount_mismatch=0 passed=True"
        for c in crops
    ]
    lines.append("ASSOC_FEATURE_PARITY_COMPLETE all_passed= True")
    return report, "\n".join(lines) + "\n"


def selftest(patch_path: Path) -> int:
    """Each mutation is one the kernel's own all_passed would accept or never see."""
    src = patch_path.read_text(encoding="utf-8")
    base_report, base_log = _synth(2)
    base = check_report(base_report, base_log, 2, TOL, src)
    failures = []
    if not all(c["passed"] for c in base):
        failures.append(("BASELINE", [c["condition"] for c in base if not c["passed"]]))

    mutations = {
        # (1) one crop silently processed -- all_passed stays TRUE in the kernel
        "1_one_crop_only": (lambda r, l: (_synth(1)[0], _synth(1)[1]), "1_both_crop_heartbeats"),
        # (2) the verifier read a cache with a different node count -- stale cache dir
        "2_stale_cache_node_count": (
            lambda r, l: (r, l.replace("AFP_CACHE crop=44b6_crop0 frames=8 nodes=1000",
                                       "AFP_CACHE crop=44b6_crop0 frames=8 nodes=997")),
            "2_fresh_reader_used_only_persisted_cache"),
        # (3)+(4) band A compared NOTHING -- vacuous pass, kernel imposes no floor
        "3_band_a_empty": (
            lambda r, l: (_mut(r, 0, "band_a", {"recorded": 0, "reproduced": 0, "missing": 0,
                                                "extra": 0, "max_abs_prob_delta": 0.0}), l),
            "3_node_ordering_and_identity_transitive"),
        # (4) a probability drifts past the frozen tolerance
        "4_band_a_delta": (
            lambda r, l: (_mut(r, 1, "band_a", {**_SYNTH_CROP["band_a"],
                                                "max_abs_prob_delta": 1e-3}), l),
            "4_deployed_candidates_exact"),
        # (5) band B compared nothing on ONE crop
        "5_band_b_zero_checked": (
            lambda r, l: (_mut(r, 1, "band_b", {"recorded_sub_threshold": 0, "checked": 0,
                                                "missing": 0, "max_abs_prob_delta": 0.0}), l),
            "5_sub_threshold_band_within_tolerance"),
        # (6) the run failed but left a well-formed report behind
        "6_afp_failed_in_log": (
            lambda r, l: (r, l + "AFP_FAILED RuntimeError: gate-1 verify phase exited 1\n"),
            "6_missing_or_empty_cache_fails_closed"),
        # cross-cutting: a node-count mismatch, which shrinks the comparison denominator
        "3b_node_count_mismatch": (
            lambda r, l: (_mut(r, 0, "node_count_mismatches",
                               [{"t": 3, "cached": 120, "recorded": 121}]), l),
            "3_node_ordering_and_identity_transitive"),
    }
    for name, (mutate, expect) in mutations.items():
        r, l = mutate(json.loads(json.dumps(base_report)), base_log)
        conds = check_report(r, l, 2, TOL, src)
        failed = [c["condition"] for c in conds if not c["passed"]]
        if expect not in failed:
            failures.append((name, failed))
        print(f"  selftest {name:28s} -> caught by {failed or 'NOTHING'}")

    if failures:
        for name, got in failures:
            print(f"SELFTEST FAILED {name}: expected rejection, got failing={got}")
        return 1
    print(f"{HEARTBEAT} selftest=PASS mutations={len(mutations)}")
    return 0


def _mut(report: dict, crop_index: int, key: str, value) -> dict:
    report["crops"][crop_index][key] = value
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", type=Path)
    ap.add_argument("--log", type=Path)
    ap.add_argument("--patch", type=Path,
                    default=Path("scripts/kaggle_edits/assoc_feature_parity.py"))
    ap.add_argument("--expected-crops", type=int, default=2)
    ap.add_argument("--tolerance", type=float, default=TOL)
    ap.add_argument("--commit", default="")
    ap.add_argument("--kernel", default="aryaarun07/biohub-p33-assoc-feature-parity-smoke")
    ap.add_argument("--kernel-version", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.patch)
    if not args.report or not args.log or not args.out:
        raise SystemExit("--report, --log and --out are required unless --selftest")

    receipt = build_receipt(args.report, args.log, args.expected_crops, args.tolerance,
                            args.patch, args.commit, args.kernel, args.kernel_version)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    for c in receipt["conditions"]:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['condition']}")
    if receipt.get("fail_reason"):
        print(f"  FAIL  {receipt['fail_reason']}")
    print(f"{HEARTBEAT} verdict={receipt['verdict']} "
          f"kernel_all_passed={receipt.get('kernel_all_passed')} -> {args.out}")
    return 0 if receipt["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
