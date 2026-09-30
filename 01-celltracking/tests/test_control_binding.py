"""THE CONTROL MUST BE THE DEPLOYED CONTROL, PROVED BY CONSTRUCTION RATHER THAN BY ASSERTION.

``FACT-0432`` defect 1 was not found by reading ``assoc_report``; it was found by CALLING it. A
report was built whose every channel was favourable and whose ONLY defect was the control's
identity, and ``verdict()`` returned ``promotable: true`` with ``blockers: []``. So every test
here builds the world it is testing - a real control file, a real graph, a real receipt, a real
manifest - and then asks the guard. None of them asserts that a check exists.

Five worlds, one per requirement of the binding:

  1. the DEPLOYED control binds;
  2. a BYTE-IDENTICAL COPY at a different path binds - the binding is to CONTENT, not to a path;
  3. a WIDENED control whose other metrics are FAVOURABLE is refused - the live FACT-0382
     violation, and the exact case that reached ``promotable: true``;
  4. a TAMPERED control is refused, including one that LIES about its own digest;
  5. a MISSING receipt, or missing identity, FAILS CLOSED.

These are software contract tests. They decide nothing scientific.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import assoc_report as ar          # noqa: E402
import control_binding as cb       # noqa: E402

EXPERIMENT = "EXP-TEST-CONTROL"


# ---------------------------------------------------------------------------------------------
# the world: per-crop metric rows in the layout `per_sample_metrics` returns
# ---------------------------------------------------------------------------------------------
def crop(edge_tp, edge_fp, edge_fn, n_pred, n_est, div=(0, 0, 0), node_recall=0.99):
    denom = edge_tp + edge_fp + edge_fn
    ratio = (n_pred - n_est) / n_est
    jac = edge_tp / denom if denom else float("nan")
    return {
        "edge_tp": edge_tp, "edge_fp": edge_fp, "edge_fn": edge_fn,
        "division_tp": div[0], "division_fp": div[1], "division_fn": div[2],
        "num_pred_nodes": n_pred, "node_recall": node_recall,
        "total_node_ratio": ratio, "edge_jaccard": jac,
        "adj_edge_jaccard": max(0.0, jac * (1 - ar.ADJUSTMENT_ALPHA * ratio)),
    }


def fake_summarise(rows):
    tp = sum(r["edge_tp"] for r in rows)
    fp = sum(r["edge_fp"] for r in rows)
    fn = sum(r["edge_fn"] for r in rows)
    dtp = sum(r["division_tp"] for r in rows)
    dfp = sum(r["division_fp"] for r in rows)
    dfn = sum(r["division_fn"] for r in rows)
    weights = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rows]
    adj = sum(w * r["adj_edge_jaccard"] for w, r in zip(weights, rows)) / sum(weights)
    div_denom = dtp + dfp + dfn
    div_j = dtp / div_denom if div_denom else 0.0
    return {
        "edge_jaccard": tp / (tp + fp + fn), "adj_edge_jaccard": adj,
        "division_jaccard": div_j, "division_tp": dtp, "division_fp": dfp, "division_fn": dfn,
        "node_recall": sum(r["node_recall"] for r in rows) / len(rows),
        "score": adj + ar.SCORE_DIVISION_WEIGHT * div_j,
    }


DEPLOYED_ROWS = [crop(900, 50, 50, 1000, 1000, div=(5, 60, 20)),
                 crop(880, 60, 60, 998, 1000, div=(4, 55, 18))]
# A candidate that is favourable in EVERY channel the verdict reads. This is deliberate: the whole
# point of FACT-0432 is that an unbound control is invisible when nothing else is wrong.
FAVOURABLE_CANDIDATE = [crop(940, 30, 30, 1000, 1000, div=(6, 58, 19)),
                        crop(920, 40, 40, 998, 1000, div=(5, 53, 17))]
# The WIDENED control: the same crops scored on the widened acquisition surface, which associates
# WORSE (FACT-0376 measured -0.00906 under the unchanged argmax). Substituting it as the base
# hands the candidate that deficit as a gain - and every channel still reads favourable.
WIDENED_ROWS = [crop(870, 80, 80, 1000, 1000, div=(5, 60, 20)),
                crop(850, 90, 90, 998, 1000, div=(4, 55, 18))]


@pytest.fixture
def world(tmp_path):
    """A registered deployed control: rows, the graph they were scored from, and its receipt."""
    graph = tmp_path / "control_graph.geff"
    graph.mkdir()
    (graph / "nodes.bin").write_bytes(b"NODES-OF-THE-DEPLOYED-GRAPH" * 32)
    (graph / "edges.bin").write_bytes(b"EDGES-OF-THE-DEPLOYED-GRAPH" * 32)
    receipt = tmp_path / "control_receipt.json"
    receipt.write_text(json.dumps({"scorer": "tracking_cellmot", "crops": 2}), encoding="utf-8")

    control = tmp_path / "deployed" / "control_rows.json"
    control.parent.mkdir()
    control.write_text(json.dumps(DEPLOYED_ROWS), encoding="utf-8")

    manifest = tmp_path / "deployed_controls.json"
    manifest.write_text(json.dumps(cb.empty_manifest()), encoding="utf-8")
    cb.register(manifest_path=manifest, experiment=EXPERIMENT, fold="0", control_path=control,
                graph=graph, receipt=receipt, n_crops=len(DEPLOYED_ROWS),
                provenance="constructed by tests/test_control_binding.py", date="2026-08-30")
    return {"root": tmp_path, "graph": graph, "receipt": receipt, "control": control,
            "manifest": manifest}


def identity(world, **over):
    out = {"experiment": EXPERIMENT, "fold": "0", "graph": str(world["graph"]),
           "receipt": str(world["receipt"])}
    out.update(over)
    return out


def bind(world, control=None, ident=None, n_rows=len(DEPLOYED_ROWS)):
    return cb.bind_control(control_path=control or world["control"],
                           identity=ident if ident is not None else identity(world),
                           manifest=world["manifest"], n_rows=n_rows)


def report_with(binding, control=DEPLOYED_ROWS, candidate=FAVOURABLE_CANDIDATE, fold=0):
    return ar.build_report(
        model="chain-arm", fold=fold, control=control, candidate=candidate,
        summarise=fake_summarise, control_binding=binding,
        conversions=ar.parent_conversions({("c", i): 0 for i in range(10)},
                                          {("c", i): 1 for i in range(10)}),
        draws=200,
    )


# =============================================================================================
# 1. THE DEPLOYED CONTROL PASSES  (the accept control - a guard that refuses everything is not
#    a guard, and without this one the other four prove nothing)
# =============================================================================================
def test_1_the_deployed_control_binds_and_the_report_is_readable(world):
    binding = bind(world)
    assert binding["bound"] is True, binding["refusals"]
    assert binding["heartbeat"] == ar.CONTROL_BINDING_HEARTBEAT
    assert binding["artifact_sha256"] == cb.sha256_path(world["control"])
    assert binding["graph_sha256"] == cb.sha256_path(world["graph"])
    assert binding["receipt_sha256"] == cb.sha256_path(world["receipt"])

    report = report_with(binding)
    assert ar.control_binding_blockers(report) == []
    assert report["verdict"]["promotable"] is True, report["verdict"]["blockers"]


def test_1b_the_digests_are_recomputed_not_read_from_a_field(world):
    """The identity may CLAIM digests. They are ignored, and a claim that lies is a refusal."""
    lying = identity(world, artifact_sha256="0" * 64, graph_sha256="1" * 64)
    binding = bind(world, ident=lying)
    assert binding["bound"] is False
    assert any("declared_artifact_sha256_is_false" in r for r in binding["refusals"])
    assert any("declared_graph_sha256_is_false" in r for r in binding["refusals"])
    # and the block still carries the TRUE digests, recomputed from the bytes
    assert binding["artifact_sha256"] == cb.sha256_path(world["control"])


# =============================================================================================
# 2. A BYTE-IDENTICAL COPY PASSES - the binding is to CONTENT, not to a path
# =============================================================================================
def test_2_a_byte_identical_copy_at_another_path_binds(world):
    copy = world["root"] / "somewhere" / "else" / "renamed_control.json"
    copy.parent.mkdir(parents=True)
    shutil.copyfile(world["control"], copy)
    assert copy.read_bytes() == world["control"].read_bytes()
    assert str(copy) != str(world["manifest"].read_text(encoding="utf-8"))

    binding = bind(world, control=copy)
    assert binding["bound"] is True, binding["refusals"]
    assert binding["artifact_sha256"] == cb.sha256_path(world["control"])
    assert binding["control_path"] == str(copy), "the path is recorded, but it is not the identity"
    assert report_with(binding)["verdict"]["promotable"] is True


def test_2b_the_registered_path_can_move_and_the_copy_still_binds(world):
    """Deleting the registered file does not unbind a byte-identical copy - content is identity."""
    copy = world["root"] / "moved_control.json"
    shutil.copyfile(world["control"], copy)
    world["control"].unlink()
    assert bind(world, control=copy)["bound"] is True


# =============================================================================================
# 3. A WIDENED CONTROL WITH FAVOURABLE METRICS FAILS  (the live FACT-0382 violation, and the
#    exact case the auditor drove to promotable: true)
# =============================================================================================
def test_3_a_widened_control_with_favourable_metrics_is_refused(world):
    """Every channel reads favourable. The ONLY defect is that the base is the widened surface."""
    widened = world["root"] / "widened" / "control_rows.json"
    widened.parent.mkdir()
    widened.write_text(json.dumps(WIDENED_ROWS), encoding="utf-8")
    widened_graph = world["root"] / "widened_graph.geff"
    widened_graph.mkdir()
    (widened_graph / "nodes.bin").write_bytes(b"NODES-OF-THE-WIDENED-GRAPH" * 32)
    (widened_graph / "edges.bin").write_bytes(b"EDGES-OF-THE-WIDENED-GRAPH" * 32)

    # Register it for what it is, so the refusal names it rather than merely failing to find it.
    cb.register(manifest_path=world["manifest"], experiment=EXPERIMENT, fold="0",
                control_path=widened, graph=widened_graph, receipt=world["receipt"],
                surface="widened", candidate_floor=0.1, n_crops=len(WIDENED_ROWS),
                provenance="the widened P30/P34 acquisition surface", date="2026-08-30")

    ident = identity(world, graph=str(widened_graph))
    binding = bind(world, control=widened, ident=ident, n_rows=len(WIDENED_ROWS))
    assert binding["bound"] is False
    assert any("control_is_a_registered_refused_control" in r for r in binding["refusals"]), \
        binding["refusals"]
    assert any("FACT-0382" in r for r in binding["refusals"])

    # THE FACT-0432 SHAPE, END TO END: favourable in every channel, and NOT promotable.
    report = report_with(binding, control=WIDENED_ROWS)
    assert report["channels"]["edge_jaccard_raw"]["delta"] > 0
    assert report["channels"]["final_graph_edges"]["delta_tp"] > 0
    assert report["channels"]["division_counts"]["delta_tp"] >= 0
    assert report["paired_bootstrap"]["favourable"] is True
    assert report["channels"]["count_adjustment"]["count_adjustment_artifact"] is False
    assert report["verdict"]["promotable"] is False
    assert any("deployed fold control" in b for b in report["verdict"]["blockers"])


def test_3b_a_widened_control_is_refused_by_its_own_declaration_too(world):
    """Even unregistered, an arm that declares a widened surface or a sub-deployed floor fails.

    Two independent paths to the same refusal on purpose: refusal by registered digest needs
    someone to have registered the widened artifact, and this one does not.
    """
    unknown = world["root"] / "unregistered_widened.json"
    unknown.write_text(json.dumps(WIDENED_ROWS), encoding="utf-8")
    binding = bind(world, control=unknown,
                   ident=identity(world, surface="widened", candidate_floor=0.1),
                   n_rows=len(WIDENED_ROWS))
    assert binding["bound"] is False
    reasons = " | ".join(binding["refusals"])
    assert "widened_control_substitution" in reasons
    assert f"below the deployed floor {cb.DEPLOYED_CANDIDATE_FLOOR}" in reasons
    assert "control_not_registry_bound" in reasons


def test_3c_an_unregistered_control_is_refused_even_when_it_claims_nothing(world):
    """The default answer to an unknown file is NO. Silence is not a clean bill of health."""
    stranger = world["root"] / "some_file_called_control.json"
    stranger.write_text(json.dumps(DEPLOYED_ROWS[:1]), encoding="utf-8")
    binding = bind(world, control=stranger, n_rows=1)
    assert binding["bound"] is False
    assert any("control_not_registry_bound" in r for r in binding["refusals"])


# =============================================================================================
# 4. A TAMPERED CONTROL FAILS
# =============================================================================================
def test_4_a_tampered_control_is_refused(world):
    """One metric edited in the control's favour. The digest moves; the binding does not hold."""
    rows = json.loads(world["control"].read_text(encoding="utf-8"))
    rows[0]["edge_tp"] -= 40          # a weaker control makes the candidate look better
    rows[0]["edge_fn"] += 40
    tampered = world["root"] / "tampered_control.json"
    tampered.write_text(json.dumps(rows), encoding="utf-8")
    assert tampered.read_bytes() != world["control"].read_bytes()

    binding = bind(world, control=tampered)
    assert binding["bound"] is False
    assert any("control_not_registry_bound" in r for r in binding["refusals"])
    assert report_with(binding, control=rows)["verdict"]["promotable"] is False


def test_4b_a_tampered_control_that_claims_the_original_digest_still_fails(world):
    """The strongest form: the tamper also forges the identity. Recomputation is authoritative."""
    original = cb.sha256_path(world["control"])
    rows = json.loads(world["control"].read_text(encoding="utf-8"))
    rows[1]["edge_fp"] += 25
    tampered = world["root"] / "forged_control.json"
    tampered.write_text(json.dumps(rows), encoding="utf-8")

    binding = bind(world, control=tampered, ident=identity(world, artifact_sha256=original))
    assert binding["bound"] is False
    assert any("declared_artifact_sha256_is_false" in r for r in binding["refusals"])
    assert binding["artifact_sha256"] != original


def test_4c_a_tampered_graph_under_an_untouched_control_fails(world):
    """The rows are the registered control's, but they were not scored from the bound graph."""
    (world["graph"] / "edges.bin").write_bytes(b"A-DIFFERENT-GRAPH" * 32)
    binding = bind(world)
    assert binding["bound"] is False
    assert any("graph_mismatch" in r for r in binding["refusals"]), binding["refusals"]


# =============================================================================================
# 5. A MISSING RECEIPT OR MISSING IDENTITY FAILS CLOSED
# =============================================================================================
def test_5_a_missing_receipt_fails_closed(world):
    world["receipt"].unlink()
    binding = bind(world)
    assert binding["bound"] is False
    assert any("receipt_unreadable" in r for r in binding["refusals"]), binding["refusals"]
    assert binding["receipt_sha256"] is None


def test_5b_missing_identity_fields_fail_closed_one_by_one(world):
    for key in cb.REQUIRED_IDENTITY:
        ident = identity(world)
        ident.pop(key)
        binding = bind(world, ident=ident)
        assert binding["bound"] is False, f"{key} was allowed to be absent"
        assert any(f"identity_missing_{key}" in r for r in binding["refusals"])


def test_5c_no_identity_at_all_fails_closed(world):
    # bind_control is called directly: the `bind` helper substitutes a good identity for None,
    # and the world being tested here is an arm that supplied NO identity at all.
    binding = cb.bind_control(control_path=world["control"], identity=None,
                              manifest=world["manifest"])
    assert binding["bound"] is False
    assert any("identity_missing:" in r for r in binding["refusals"])
    for key in cb.REQUIRED_IDENTITY:
        assert any(f"identity_missing_{key}" in r for r in binding["refusals"])


def test_5d_a_report_built_with_no_binding_is_blocked(world):
    """`build_report` without a binding must write a REFUSING block, not an absent one."""
    report = ar.build_report(
        model="unbound-arm", fold=0, control=DEPLOYED_ROWS, candidate=FAVOURABLE_CANDIDATE,
        summarise=fake_summarise,
        conversions=ar.parent_conversions({("c", i): 0 for i in range(10)},
                                          {("c", i): 1 for i in range(10)}),
        draws=200,
    )
    assert report["control_binding"]["bound"] is False
    assert report["verdict"]["promotable"] is False
    assert any("deployed fold control" in b for b in report["verdict"]["blockers"])


def test_5e_verdict_called_directly_on_an_all_favourable_report_fails_closed():
    """THE FACT-0432 EXERCISE, REPRODUCED. This report is the one that returned promotable: true.

    It is built by hand, with no `control_binding` key at all, exactly as the auditor built it -
    which is why the check lives in `verdict()` and not only in `build_report()`.
    """
    report = {
        "fold": 0,
        "channels": {
            "count_adjustment": {"count_adjustment_artifact": False},
            "edge_jaccard_raw": {"delta": +0.01},
            "division_counts": {"delta_tp": +3},
            "final_graph_edges": {"delta_tp": +40},
            "parent_conversions": {"net": 12},
            "score": {"identity_check": 0.0},
        },
        "paired_bootstrap": {"favourable": True},
    }
    v = ar.verdict(report)
    assert v["promotable"] is False, "this is the exact FACT-0432 report; it must not pass"
    assert any("control identity not bound" in b for b in v["blockers"])


def test_5f_a_binding_for_the_wrong_fold_is_blocked(world):
    """An f0 control read against an f1 result is a different experiment with the right filename."""
    binding = bind(world)
    assert binding["bound"] is True
    report = report_with(binding, fold=1)
    assert report["verdict"]["promotable"] is False
    assert any("control binding is for fold" in b for b in report["verdict"]["blockers"])


def test_5g_a_fabricated_binding_block_without_digests_is_blocked():
    """`{"bound": True}` is not a binding. The block must carry digests it was made to compute."""
    report = {
        "fold": 0,
        "control_binding": {"heartbeat": ar.CONTROL_BINDING_HEARTBEAT, "bound": True,
                            "refusals": [], "fold": "0"},
        "channels": {
            "count_adjustment": {"count_adjustment_artifact": False},
            "edge_jaccard_raw": {"delta": +0.01},
            "division_counts": {"delta_tp": +3},
            "final_graph_edges": {"delta_tp": +40},
            "parent_conversions": {"net": 12},
            "score": {"identity_check": 0.0},
        },
        "paired_bootstrap": {"favourable": True},
    }
    v = ar.verdict(report)
    assert v["promotable"] is False
    assert any("carries no artifact_sha256" in b for b in v["blockers"])


# =============================================================================================
# the shipped manifest, and the registration path PKT-0042 uses
# =============================================================================================
def test_an_unregistered_control_is_refused_against_the_shipped_manifest(tmp_path):
    """A control the SHIPPED manifest does not register must REFUSE, not wave through.

    REWRITTEN 2026-08-31. The original asserted ``deployed_controls == []`` - a fact about
    CAMPAIGN STATE, not about the guard - and it broke the moment PKT-0042 legitimately
    registered EXP-0030 and EXP-0031. A guard test that fails because the campaign made
    progress is testing the wrong thing, and worse, the obvious "fix" is to delete the
    registrations. What must hold for the life of the project is the BEHAVIOUR: an identity
    absent from the manifest is refused however many identities are present. The shipped
    manifest is still read - so this exercises the real file and not a fixture - but only its
    well-formedness is asserted, and every entry it does hold must itself be well formed.
    """
    man = cb.load_manifest(cb.MANIFEST_PATH)
    assert man["kind"] == "deployed_control_manifest"
    assert isinstance(man["deployed_controls"], list)
    for entry in man["deployed_controls"]:
        for field in ("experiment", "fold"):
            assert entry.get(field), f"shipped manifest entry missing {field!r}: {entry}"
    assert not any(e.get("experiment") == "EXP-0044" for e in man["deployed_controls"]), (
        "EXP-0044 must remain unregistered for this test to exercise a refusal"
    )
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps(DEPLOYED_ROWS), encoding="utf-8")
    graph = tmp_path / "g.bin"; graph.write_bytes(b"g")
    receipt = tmp_path / "r.json"; receipt.write_text("{}", encoding="utf-8")
    binding = cb.bind_control(
        control_path=rows,
        identity={"experiment": "EXP-0044", "fold": "0", "graph": str(graph),
                  "receipt": str(receipt)},
        manifest=cb.MANIFEST_PATH)
    assert binding["bound"] is False
    assert any("control_not_registry_bound" in r for r in binding["refusals"])


def test_a_control_assembled_from_many_crop_payloads_binds_as_one_identity(tmp_path):
    """The shape a full-chain panel actually has: one payload per crop, not one control file.

    ``finaledge_gates`` builds its control rows from ``f{fold}_{crop}.json`` payloads, so the
    binding must take the ORDERED SET as the identity - and must move if any member moves, or a
    single tampered crop would slip through under an unchanged headline digest.
    """
    man = tmp_path / "m.json"
    man.write_text(json.dumps(cb.empty_manifest()), encoding="utf-8")
    graph = tmp_path / "preilp.parquet"; graph.write_bytes(b"PREILP" * 64)
    receipt = tmp_path / "run_stats.csv"; receipt.write_text("dataset,raw_nodes\n", encoding="utf-8")
    payloads = []
    for i in range(3):
        p = tmp_path / f"f0_crop{i}.json"
        p.write_text(json.dumps({"crop": i, "arms": {"control": DEPLOYED_ROWS[0]}}),
                     encoding="utf-8")
        payloads.append(p)

    cb.register(manifest_path=man, experiment=EXPERIMENT, fold="0", control_path=payloads,
                graph=graph, receipt=receipt, n_crops=3, provenance="per-crop control payloads",
                date="2026-08-30")
    ident = {"experiment": EXPERIMENT, "fold": "0", "graph": str(graph), "receipt": str(receipt)}
    assert cb.bind_control(control_path=payloads, identity=ident, manifest=man,
                           n_rows=3)["bound"] is True

    # ONE crop tampered, everything else untouched
    payloads[1].write_text(json.dumps({"crop": 1, "arms": {"control": DEPLOYED_ROWS[1]}}),
                           encoding="utf-8")
    tampered = cb.bind_control(control_path=payloads, identity=ident, manifest=man, n_rows=3)
    assert tampered["bound"] is False
    assert any("control_not_registry_bound" in r for r in tampered["refusals"])


def test_registration_refuses_a_placeholder_provenance(tmp_path):
    man = tmp_path / "m.json"
    man.write_text(json.dumps(cb.empty_manifest()), encoding="utf-8")
    rows = tmp_path / "rows.json"; rows.write_text("[]", encoding="utf-8")
    graph = tmp_path / "g.bin"; graph.write_bytes(b"g")
    receipt = tmp_path / "r.json"; receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(cb.BindingRefusal, match="placeholder"):
        cb.register(manifest_path=man, experiment="EXP-0044", fold="0", control_path=rows,
                    graph=graph, receipt=receipt, provenance="tbd")


def test_a_missing_manifest_file_refuses_rather_than_defaulting_open(tmp_path):
    rows = tmp_path / "rows.json"; rows.write_text("[]", encoding="utf-8")
    graph = tmp_path / "g.bin"; graph.write_bytes(b"g")
    receipt = tmp_path / "r.json"; receipt.write_text("{}", encoding="utf-8")
    binding = cb.bind_control(
        control_path=rows,
        identity={"experiment": "E", "fold": "0", "graph": str(graph), "receipt": str(receipt)},
        manifest=tmp_path / "not_here.json")
    assert binding["bound"] is False
    assert any("manifest_missing" in r for r in binding["refusals"])
