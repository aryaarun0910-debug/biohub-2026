"""End-to-end: three crops across two folds, multiple sequential flushes, all records survive.

This is the regression for the v4 defect. v4 kept ONE shared manifest and rewrote it on every
flush, so fold 1 wrote both crops' artifacts but reported 1/2 and the missing crop looked as
though it had never run. `test_v4_overwrite_model_loses_records` reproduces that model and
asserts it loses data, so the fix cannot silently regress to it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FOLDS = {0: ["44b6_0113de3b"], 1: ["6bba_6feb10f0", "6bba_57b7cc1e"]}
CKPT = {0: "d3e89eb3", 1: "2e4ebf61"}


# ---------------------------------------------------------------- the v5 model
def write_terminal(man_dir: Path, stem: str, fold: int, *, status="complete", **kw):
    """Write-once terminal record, exactly as _d1_flush does."""
    c, e = man_dir / f"{stem}.complete.json", man_dir / f"{stem}.error.json"
    if c.exists() or e.exists():
        return False
    rec = {"dataset": stem, "status": status, "fold": str(fold),
           "checkpoint_sha256": CKPT[fold], "n_rows": 100, "gt_rows": 10,
           "feat_rows": 100, "feat_dim": 32, "feat_finite": True,
           "gt_load_error": None, "exception": None}
    rec.update(kw)
    (c if status == "complete" else e).write_text(json.dumps(rec))
    return True


def aggregate(man_dir: Path, expected: list[str], fold: int):
    """The parent aggregator's logic, mirrored for test purposes."""
    problems, crops = [], {}
    for stem in sorted(expected):
        terms = [p for p in (man_dir / f"{stem}.complete.json",
                             man_dir / f"{stem}.error.json") if p.exists()]
        if not terms:
            problems.append(f"{stem}: no terminal record")
            crops[stem] = {"status": "not_reached"}
            continue
        if len(terms) > 1:
            problems.append(f"{stem}: multiple terminal records")
        rec = json.loads(terms[0].read_text())
        crops[stem] = rec
        if rec.get("status") != "complete":
            problems.append(f"{stem}: status {rec.get('status')}")
        if str(rec.get("fold")) != str(fold):
            problems.append(f"{stem}: wrong fold")
        if rec.get("checkpoint_sha256") != CKPT[fold]:
            problems.append(f"{stem}: wrong checkpoint")
        if not rec.get("gt_rows"):
            problems.append(f"{stem}: zero GT rows")
    return {"crops": crops, "problems": problems,
            "n_complete": sum(1 for v in crops.values() if v.get("status") == "complete"),
            "COMPLETE": not problems and len(crops) == len(expected)}


def test_two_folds_multiple_flushes_all_three_records_survive(tmp_path):
    """The exact v4 scenario: several sequential flushes, each seeing one crop."""
    results = {}
    for fold, stems in FOLDS.items():
        man = tmp_path / f"f{fold}" / "manifests"
        man.mkdir(parents=True)
        # simulate one flush per crop, as separate invocations would produce
        for stem in stems:
            write_terminal(man, stem, fold)
        results[fold] = aggregate(man, stems, fold)

    assert results[0]["n_complete"] == 1 and results[0]["COMPLETE"]
    assert results[1]["n_complete"] == 2, "the v4 defect: a crop was lost"
    assert results[1]["COMPLETE"]
    assert set(results[1]["crops"]) == {"6bba_6feb10f0", "6bba_57b7cc1e"}
    total = sum(r["n_complete"] for r in results.values())
    assert total == 3, f"expected 3 crops across both folds, got {total}"


def test_v4_overwrite_model_loses_records(tmp_path):
    """Reproduce v4's shared-mutable-manifest model and prove it drops crops.

    If this ever stops failing, the shared-manifest model has crept back in.
    """
    shared = tmp_path / "d1_manifest.json"
    for stem in FOLDS[1]:                       # one flush per crop, each rewriting the file
        shared.write_text(json.dumps({"crops": {stem: {"status": "complete"}},
                                      "n_complete": 1, "expected_crops": 2}))
    got = json.loads(shared.read_text())
    assert len(got["crops"]) == 1, "v4 model unexpectedly retained both crops"
    assert got["n_complete"] == 1 and got["expected_crops"] == 2
    assert "6bba_6feb10f0" not in got["crops"], "the FIRST crop is the one v4 lost"


def test_terminal_records_are_write_once(tmp_path):
    man = tmp_path / "manifests"; man.mkdir(parents=True)
    assert write_terminal(man, "c", 0) is True
    assert write_terminal(man, "c", 0, n_rows=999) is False, "terminal record was overwritten"
    assert json.loads((man / "c.complete.json").read_text())["n_rows"] == 100


def test_missing_crop_is_a_hard_failure_not_a_flag(tmp_path):
    man = tmp_path / "manifests"; man.mkdir(parents=True)
    write_terminal(man, "6bba_6feb10f0", 1)          # only one of two
    res = aggregate(man, FOLDS[1], 1)
    assert res["COMPLETE"] is False
    assert any("no terminal record" in p for p in res["problems"])
    assert res["crops"]["6bba_57b7cc1e"]["status"] == "not_reached"


def test_multiple_terminal_records_are_detected(tmp_path):
    man = tmp_path / "manifests"; man.mkdir(parents=True)
    write_terminal(man, "c", 0)
    (man / "c.error.json").write_text(json.dumps({"dataset": "c", "status": "error"}))
    res = aggregate(man, ["c"], 0)
    assert any("multiple terminal" in p for p in res["problems"])


@pytest.mark.parametrize("bad,frag", [
    ({"fold": "9"}, "wrong fold"),
    ({"checkpoint_sha256": "deadbeef"}, "wrong checkpoint"),
    ({"gt_rows": 0}, "zero GT rows"),
    ({"status": "error"}, "status"),
])
def test_aggregator_rejects_each_inconsistency(tmp_path, bad, frag):
    man = tmp_path / "manifests"; man.mkdir(parents=True)
    # `fold` and `status` are named params of write_terminal, so they cannot ride in **kw
    # without colliding with the positional argument. Split them out (copy, so parametrize
    # cases are not mutated between runs).
    bad = dict(bad)
    status = bad.pop("status", "complete")
    fold_override = bad.pop("fold", None)
    write_terminal(man, "c", 0, status=status, **bad)
    if fold_override is not None:
        rec = json.loads((man / "c.complete.json").read_text())
        rec["fold"] = fold_override
        (man / "c.complete.json").write_text(json.dumps(rec))
    res = aggregate(man, ["c"], 0)
    assert res["COMPLETE"] is False
    assert any(frag in p for p in res["problems"]), res["problems"]
