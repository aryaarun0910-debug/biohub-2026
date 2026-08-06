"""Mechanical assertions on the D1 2x2 checkpoint x family factorial manifests (correction C1).

These tests exist because the v5 export confounds family with checkpoint: 44b6 was only ever
encoded by split_0 and 6bba only by split_1, and the two 32-D bases are independent, so no
head fitted in one could be applied in the other.  Everything below is a guard against that
mistake being re-introduced silently.

The manifests are treated as data under test.  They are checked for coverage, disjointness,
role correctness, fold honesty and byte-level reproducibility.  Nothing here touches the
dataset: ``build`` is a pure function of the committed census, so regeneration runs anywhere.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "build_d1_factorial_manifests.py"
FACT_DIR = REPO / "data" / "d1_factorial"
CENSUS = FACT_DIR / "crop_census.json"

sys.path.insert(0, str(REPO / "scripts"))

import build_d1_factorial_manifests as bdf  # noqa: E402

TIERS = ("smoke", "pilot", "full")


def lf(raw: bytes) -> bytes:
    """Normalise the checkout filter away.

    The repo runs with ``core.autocrlf=true`` and ships no ``.gitattributes``, so a checked-out
    manifest arrives with CRLF while the generator always writes LF.  Determinism here means
    "identical modulo the checkout filter"; the generator's own output is checked for LF
    separately in ``test_generator_always_writes_lf``.
    """
    return raw.replace(b"\r\n", b"\n")

pytestmark = pytest.mark.skipif(
    not CENSUS.exists(),
    reason="data/d1_factorial/crop_census.json absent (run the `census` subcommand)",
)


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def census() -> dict:
    return json.loads(CENSUS.read_text())


@pytest.fixture(scope="module")
def manifests() -> dict:
    out = {}
    for tier in TIERS:
        p = FACT_DIR / f"manifest_{tier}.json"
        assert p.exists(), f"missing manifest for tier {tier}: {p}"
        out[tier] = json.loads(p.read_text())
    return out


# --------------------------------------------------------------------------------------
# 1. the canonical crop set -- one source of truth, never a glob
# --------------------------------------------------------------------------------------


def test_census_is_the_full_199_crop_roster(census):
    assert census["kind"] == "d1_factorial_crop_census"
    ids = [c["crop_id"] for c in census["crops"]]
    assert len(ids) == 199
    assert len(set(ids)) == 199, "census contains duplicate crops"
    assert ids == sorted(ids), "census crops must be sorted for determinism"
    fam = {f: sum(1 for i in ids if i.startswith(f)) for f in bdf.FAMILIES}
    assert fam == {"44b6": 71, "6bba": 128}


def test_census_gt_totals_match_the_independently_recorded_corpus(census):
    """20,197 (44b6) + 113,121 (6bba) is quoted in the decision package; re-derive it."""
    tot = {f: 0 for f in bdf.FAMILIES}
    for c in census["crops"]:
        tot[c["family"]] += c["gt_nodes"]
    assert tot == {"44b6": 20197, "6bba": 113121}
    assert sum(tot.values()) == 133318


def test_census_roster_is_pinned_by_hash(census):
    assert len(census["provenance"]["roster_sha256"]) == 64
    assert census["provenance"]["match_um"] == 7.0, "must be the scorer's MAX_DISTANCE"
    # the label must be machine-independent so the census is reproducible off-box
    for key in ("roster_source", "gt_dir", "oof_pred_dir"):
        val = census["provenance"][key]
        assert ":" not in val and not val.startswith("/"), f"{key} embeds an absolute path"


def test_crop_size_has_no_variance_so_the_size_axis_must_be_load(census):
    """Justifies replacing 'crop size' by the occupied load in the PILOT stratification."""
    assert census["vol_shapes_observed"] == [[100, 64, 256, 256]]
    sizes = {c["vol_voxels"] for c in census["crops"]}
    assert len(sizes) == 1, "voxel crop size unexpectedly varies; revisit the size axis"
    loads = {c[bdf.PILOT_SIZE_KEY] for c in census["crops"]}
    assert len(loads) > 100, "the chosen size axis must actually vary across crops"


# --------------------------------------------------------------------------------------
# 2. the 2x2 design: roles are correct and explicit
# --------------------------------------------------------------------------------------


def test_loeo_semantics_are_the_ones_the_design_rests_on():
    assert bdf.FOLD_TRAIN_FAMILY == {0: "6bba", 1: "44b6"}
    assert bdf.FOLD_HELDOUT_FAMILY == {0: "44b6", 1: "6bba"}
    # split 0: encodes 44b6 as held-out TARGET, 6bba as training SOURCE
    assert bdf.role_of(0, "44b6") == bdf.ROLE_TARGET
    assert bdf.role_of(0, "6bba") == bdf.ROLE_SOURCE
    # split 1: mirror image
    assert bdf.role_of(1, "44b6") == bdf.ROLE_SOURCE
    assert bdf.role_of(1, "6bba") == bdf.ROLE_TARGET


@pytest.mark.parametrize("tier", TIERS)
def test_every_cell_of_the_2x2_declares_its_role_explicitly(manifests, tier):
    cells = manifests[tier]["cells"]
    assert set(cells) == {
        "split_0__44b6",
        "split_0__6bba",
        "split_1__44b6",
        "split_1__6bba",
    }
    expect = {
        "split_0__44b6": (0, "44b6", bdf.ROLE_TARGET, False),
        "split_0__6bba": (0, "6bba", bdf.ROLE_SOURCE, True),
        "split_1__44b6": (1, "44b6", bdf.ROLE_SOURCE, True),
        "split_1__6bba": (1, "6bba", bdf.ROLE_TARGET, False),
    }
    for name, (fold, fam, role, seen) in expect.items():
        c = cells[name]
        assert (c["fold"], c["family"], c["role"]) == (fold, fam, role)
        assert c["encoder_saw_this_family_in_training"] is seen
        assert c["fit_and_select_here"] is (role == bdf.ROLE_SOURCE)
        assert c["open_once_after_freeze"] is (role == bdf.ROLE_TARGET)


@pytest.mark.parametrize("tier", TIERS)
def test_the_two_cells_of_a_checkpoint_are_different_families(manifests, tier):
    """The whole point: one checkpoint, one basis, both families."""
    cells = manifests[tier]["cells"]
    for fold in (0, 1):
        fams = {c["family"] for c in cells.values() if c["fold"] == fold}
        roles = {c["role"] for c in cells.values() if c["fold"] == fold}
        assert fams == set(bdf.FAMILIES)
        assert roles == {bdf.ROLE_SOURCE, bdf.ROLE_TARGET}


# --------------------------------------------------------------------------------------
# 3. fold honesty
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_target_role_never_uses_a_checkpoint_trained_on_its_own_family(manifests, tier):
    """The measurement side must be out-of-sample for the encoder.

    This is the guarantee that makes the transfer number honest: whenever a crop is opened as
    a TARGET, the checkpoint encoding it never saw that crop's embryo in training.
    """
    for row in manifests[tier]["rows"]:
        if row["role"] == bdf.ROLE_TARGET:
            assert row["family"] != bdf.FOLD_TRAIN_FAMILY[row["fold"]], row["row_id"]
            assert row["family"] == bdf.FOLD_HELDOUT_FAMILY[row["fold"]], row["row_id"]


@pytest.mark.parametrize("tier", TIERS)
def test_source_role_is_exactly_the_checkpoints_training_family(manifests, tier):
    """The confound is made explicit rather than hidden.

    NOTE on wording.  A SOURCE crop *is* encoded by the checkpoint trained on its own family --
    that is unavoidable with two embryos and two LOEO checkpoints, and it is what "source"
    means here.  The honest invariant is the pair of assertions in this file: TARGET rows are
    always out-of-sample (test above), and no TARGET-family crop ever enters the fitting set
    (test below).  Any manifest that inverted these roles would fail here.
    """
    for row in manifests[tier]["rows"]:
        if row["role"] == bdf.ROLE_SOURCE:
            assert row["family"] == bdf.FOLD_TRAIN_FAMILY[row["fold"]], row["row_id"]


@pytest.mark.parametrize("tier", TIERS)
def test_no_target_family_crop_can_leak_into_the_fitting_set(manifests, tier):
    """Fit and select EXCLUSIVELY on the source family within the same checkpoint basis."""
    m = manifests[tier]
    for fold in (0, 1):
        fit_rows = [r for r in m["rows"] if r["fold"] == fold and r["role"] == bdf.ROLE_SOURCE]
        eval_rows = [r for r in m["rows"] if r["fold"] == fold and r["role"] == bdf.ROLE_TARGET]
        fit_fams = {r["family"] for r in fit_rows}
        eval_fams = {r["family"] for r in eval_rows}
        assert fit_fams and eval_fams
        assert fit_fams.isdisjoint(eval_fams), f"fold {fold}: fit/eval families overlap"
        assert {r["crop_id"] for r in fit_rows}.isdisjoint({r["crop_id"] for r in eval_rows})


@pytest.mark.parametrize("tier", TIERS)
def test_a_source_shard_is_never_scheduled_after_its_target_shard(manifests, tier):
    """Freeze before opening: within a fold, SOURCE stages strictly before TARGET."""
    for fold in (0, 1):
        src = [s["stage"] for s in manifests[tier]["shards"]
               if s["fold"] == fold and s["role"] == bdf.ROLE_SOURCE]
        tgt = [s["stage"] for s in manifests[tier]["shards"]
               if s["fold"] == fold and s["role"] == bdf.ROLE_TARGET]
        assert src and tgt
        assert max(src) < min(tgt), f"fold {fold}: a target shard precedes a source shard"


@pytest.mark.parametrize("tier", TIERS)
def test_split_1_stages_before_split_0(manifests, tier):
    """Held-out 6bba is ~85% of the objective, so split 1 is staged first at full scale."""
    stages = {s["shard_id"]: (s["stage"], s["fold"]) for s in manifests[tier]["shards"]}
    f1 = [st for st, fo in stages.values() if fo == 1]
    f0 = [st for st, fo in stages.values() if fo == 0]
    assert f1 and f0
    assert max(f1) < min(f0), "split 0 must not start before split 1 is complete"
    assert manifests[tier]["design"]["stage_order"][0] == {
        "stage": 1,
        "fold": 1,
        "role": bdf.ROLE_SOURCE,
    }


# --------------------------------------------------------------------------------------
# 4. coverage: every crop under BOTH checkpoints, no gap, no duplicate
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_every_crop_appears_under_both_checkpoints(manifests, tier):
    m = manifests[tier]
    seen: dict[str, set[int]] = {}
    for row in m["rows"]:
        seen.setdefault(row["crop_id"], set()).add(row["fold"])
    assert set(seen) == set(m["crops"])
    for crop, folds in seen.items():
        assert folds == {0, 1}, f"{crop} is not encoded under both checkpoints: {folds}"
    assert len(m["rows"]) == 2 * len(m["crops"])
    assert m["n_rows"] == len(m["rows"])
    assert m["n_crops"] == len(m["crops"])


@pytest.mark.parametrize("tier", TIERS)
def test_row_ids_are_unique_and_derived_from_fold_plus_crop(manifests, tier):
    rows = manifests[tier]["rows"]
    ids = [r["row_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate row_id"
    assert ids == sorted(ids), "rows must be sorted by row_id for determinism"
    for r in rows:
        assert r["row_id"] == f"split_{r['fold']}::{r['crop_id']}"
        assert r["cell"] == f"split_{r['fold']}__{r['family']}"


def test_tier_crop_sets_are_exactly_their_intended_sets(manifests, census):
    all_ids = sorted(c["crop_id"] for c in census["crops"])
    assert manifests["smoke"]["crops"] == sorted(bdf.SMOKE_CROPS)
    assert manifests["full"]["crops"] == all_ids
    assert manifests["full"]["n_rows"] == 398
    assert manifests["smoke"]["n_rows"] == 6
    pilot = manifests["pilot"]["crops"]
    assert 16 <= len(pilot) <= 24, f"PILOT must hold 16-24 crops, got {len(pilot)}"
    assert len(set(pilot)) == len(pilot)


def test_tiers_are_nested_so_paid_encoder_passes_stay_reusable(manifests):
    smoke = set(manifests["smoke"]["crops"])
    pilot = set(manifests["pilot"]["crops"])
    full = set(manifests["full"]["crops"])
    assert smoke < pilot < full


@pytest.mark.parametrize("tier", TIERS)
def test_tier_crops_all_exist_in_the_census(manifests, census, tier):
    known = {c["crop_id"] for c in census["crops"]}
    assert set(manifests[tier]["crops"]) <= known


# --------------------------------------------------------------------------------------
# 5. shards
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_shards_partition_the_tier_with_empty_intersections(manifests, tier):
    m = manifests[tier]
    union: list[str] = []
    for s in m["shards"]:
        union.extend(f"split_{s['fold']}::{stem}" for stem in s["stems"])
    assert len(union) == len(set(union)), "shard stems intersect"
    assert set(union) == {r["row_id"] for r in m["rows"]}, "shards do not cover the tier"

    # pairwise disjointness stated directly, not only via the multiset count
    for i, a in enumerate(m["shards"]):
        sa = {f"split_{a['fold']}::{x}" for x in a["stems"]}
        for b in m["shards"][i + 1 :]:
            sb = {f"split_{b['fold']}::{x}" for x in b["stems"]}
            assert sa.isdisjoint(sb), f"{a['shard_id']} intersects {b['shard_id']}"


@pytest.mark.parametrize("tier", TIERS)
def test_shards_are_fold_pure_role_pure_and_family_pure(manifests, tier):
    """The LOEO retarget binds one fold at module level and rebinds TEST_DIR globally."""
    for s in manifests[tier]["shards"]:
        assert s["expected_crops"] == len(s["stems"]) > 0
        assert s["stems"] == sorted(s["stems"])
        assert len(set(s["stems"])) == len(s["stems"])
        for stem in s["stems"]:
            assert bdf.family_of(stem) == s["family"]
            assert bdf.role_of(s["fold"], s["family"]) == s["role"]
        assert s["stems_are_in_checkpoint_training_set"] is (s["role"] == bdf.ROLE_SOURCE)
        assert s["kernel_edit_change_required"] is False


@pytest.mark.parametrize("tier", TIERS)
def test_shard_env_is_self_consistent(manifests, tier):
    for s in manifests[tier]["shards"]:
        env = s["env"]
        assert env["BIOHUB_LOEO_FOLD"] == env["BIOHUB_D1_FOLD"] == str(s["fold"])
        assert env["BIOHUB_D1_CKPT_SHA"] == s["checkpoint_sha256"]
        assert env["BIOHUB_D1_EXPECTED_CROPS"] == str(len(s["stems"]))
        assert json.loads(env["BIOHUB_LOEO_STEMS"]) == s["stems"]
        assert f"split_{s['fold']}" in env["BIOHUB_LOEO_WEIGHTS_GLOB"]
        assert f"split_{s['fold']}" in env["BIOHUB_LOEO_CONFIG_GLOB"]


@pytest.mark.parametrize("tier", TIERS)
def test_no_shard_is_planned_past_a_session(manifests, tier):
    limit = bdf.SESSION_SECONDS * bdf.SESSION_USABLE_FRACTION / 3600.0
    for s in manifests[tier]["shards"]:
        assert s["est_wall_hours"] <= limit + 1e-9, (
            f"{s['shard_id']} plans {s['est_wall_hours']:.2f} h against a {limit:.2f} h budget"
        )


# --------------------------------------------------------------------------------------
# 6. checkpoint pinning
# --------------------------------------------------------------------------------------


def test_checkpoint_hashes_are_the_pinned_loeo_ones():
    assert bdf.CHECKPOINTS[0]["sha256"].startswith("d3e89eb361eeadef")
    assert bdf.CHECKPOINTS[1]["sha256"].startswith("2e4ebf616b3d4fb5")
    assert (
        bdf.CHECKPOINTS[0]["sha256"]
        == "d3e89eb361eeadef06d18159d834594776174a9f8cabd81f65271abbb42a492f"
    )
    assert (
        bdf.CHECKPOINTS[1]["sha256"]
        == "2e4ebf616b3d4fb53881eadf42c7972e04978c28f74f229c5cb30bee835063de"
    )
    assert bdf.CHECKPOINTS[0]["sha256"] != bdf.CHECKPOINTS[1]["sha256"]


def test_census_verified_the_pinned_hashes_against_the_real_files(census):
    for fold, spec in bdf.CHECKPOINTS.items():
        rec = census["checkpoints"][str(fold)]
        assert rec["sha256"] == spec["sha256"]
        assert rec["weights_file"] == spec["weights_file"]
        assert int(rec["size_bytes"]) == 8_357_783


@pytest.mark.parametrize("tier", TIERS)
def test_every_row_and_cell_pins_the_right_checkpoint(manifests, tier):
    m = manifests[tier]
    for row in m["rows"]:
        assert row["checkpoint_sha256"] == bdf.CHECKPOINTS[row["fold"]]["sha256"]
    for name, cell in m["cells"].items():
        assert cell["checkpoint_sha256"] == bdf.CHECKPOINTS[cell["fold"]]["sha256"]
    for s in m["shards"]:
        assert s["checkpoint_sha256"] == bdf.CHECKPOINTS[s["fold"]]["sha256"]


# --------------------------------------------------------------------------------------
# 7. PILOT stratification
# --------------------------------------------------------------------------------------


def test_pilot_covers_every_non_empty_stratum_exactly_once(manifests, census):
    strat = manifests["pilot"]["stratification"]
    cells = [s for s in strat["strata"] if not s.get("co_tenant")]
    keys = [(s["family"], s["gt_mass_tertile"], s["miss_burden_tertile"]) for s in cells]
    assert len(keys) == len(set(keys)), "a stratum is listed twice"
    assert len(keys) == len(bdf.FAMILIES) * bdf.PILOT_MASS_BINS * bdf.PILOT_MISS_BINS
    for s in cells:
        assert s["n_available"] > 0
        assert len(s["selected"]) == bdf.PILOT_PER_CELL
    selected = [c for s in strat["strata"] for c in s["selected"]]
    assert sorted(set(selected)) == manifests["pilot"]["crops"]


def test_pilot_is_balanced_across_families_and_tertiles(manifests):
    strat = manifests["pilot"]["stratification"]
    per_fam: dict[str, int] = {}
    for s in strat["strata"]:
        per_fam[s["family"]] = per_fam.get(s["family"], 0) + len(s["selected"])
    assert set(per_fam) == set(bdf.FAMILIES)
    # both families must carry enough crops for their own TARGET cell to be measurable
    for fam, n in per_fam.items():
        assert n >= bdf.PILOT_MASS_BINS * bdf.PILOT_MISS_BINS, (fam, n)


def test_pilot_balance_report_is_present_and_quantitative(manifests):
    bal = manifests["pilot"]["stratification"]["achieved_balance"]
    assert set(bal) >= {"all", "44b6", "6bba"}
    for scope in ("44b6", "6bba"):
        blk = bal[scope]
        assert 0.0 < blk["crop_share"] < 1.0
        assert 0.0 < blk["gt_mass_share_of_full"] < 1.0
        for key in ("gt_nodes", "miss_rate", "pred_nodes"):
            p, f = blk[key]["pilot"], blk[key]["full"]
            assert p["n"] > 0 and f["n"] > 0
            # a stratified pilot must at least span the population's central range
            assert p["min"] <= f["median"] <= p["max"], (scope, key)


def test_pilot_medians_track_the_population(manifests):
    """Equal-count tertiles deliberately over-weight the tails, but the centre must hold."""
    bal = manifests["pilot"]["stratification"]["achieved_balance"]
    for scope in ("44b6", "6bba"):
        for key in ("gt_nodes", "pred_nodes"):
            p = bal[scope][key]["pilot"]["median"]
            f = bal[scope][key]["full"]["median"]
            assert 0.5 <= p / f <= 2.0, (scope, key, p, f)


# --------------------------------------------------------------------------------------
# 8. cost model
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_budget_accounting_adds_up(manifests, tier):
    m = manifests[tier]
    tot = m["budget"]["total"]
    assert tot["crop_inferences"] == m["n_rows"]
    assert tot["bytes"] == sum(r["est_bytes"] for r in m["rows"])
    assert sum(v["crop_inferences"] for v in m["budget"]["by_cell"].values()) == m["n_rows"]
    assert sum(v["crop_inferences"] for v in m["budget"]["by_stage"].values()) == m["n_rows"]
    assert sum(s["est_bytes"] for s in m["shards"]) == tot["bytes"]
    assert sum(s["expected_crops"] for s in m["shards"]) == m["n_rows"]


def test_storage_model_reproduces_the_three_measured_v5_smoke_crops():
    """feat_gt + feat_max + rows.parquet, exact bytes from agent4/v5_pull."""
    measured = {
        52: 1_235_584 + 1_235_584 + 179_139,
        1659: 1_441_280 + 1_441_280 + 376_774,
        1368: 1_404_032 + 1_404_032 + 342_132,
    }
    for gt_nodes, actual in measured.items():
        modelled = bdf.crop_bytes({"gt_nodes": gt_nodes})
        assert abs(modelled - actual) / actual < 0.005, (gt_nodes, modelled, actual)


def test_full_tier_is_flagged_as_over_a_single_session(manifests):
    """The factorial doubles crop-inferences; the FULL tier cannot be one sitting."""
    full = manifests["full"]["budget"]
    session_h = bdf.SESSION_SECONDS / 3600.0
    assert full["total"]["crop_inferences"] == 398
    assert full["total"]["predict_hours"] > 2 * session_h
    # the single largest cell must still fit one session, or the plan is unschedulable
    for cell, blk in full["by_cell"].items():
        assert blk["session_fraction_per_kernel"] < 1.0, cell


@pytest.mark.parametrize("tier", TIERS)
def test_feasibility_block_is_present_and_internally_consistent(manifests, tier):
    f = manifests[tier]["feasibility"]
    assert f["required_t4_hours"] == pytest.approx(
        manifests[tier]["budget"]["total"]["wall_hours_incl_fixed"], abs=5e-4
    )
    assert f["min_kernel_runs"] == len(manifests[tier]["shards"])
    assert f["every_shard_fits_one_session"] is True
    assert f["fits_allocated_full_export_budget"] == (
        f["required_t4_hours"] <= bdf.ROADMAP_GPU_HOURS_FULL_EXPORT
    )
    w = f["what_has_to_give"]
    assert w["reduced_plan_t4_hours"] <= f["required_t4_hours"]
    assert w["split_1_only_first_window_t4_hours"] <= w["reduced_plan_t4_hours"]


def test_full_tier_does_not_fit_and_says_so(manifests):
    """The load-bearing finding: the factorial doubles crop-inferences and blows the budget."""
    f = manifests["full"]["feasibility"]
    assert f["fits_allocated_full_export_budget"] is False
    assert f["fits_total_remaining_budget"] is False
    assert f["required_t4_hours"] > bdf.ROADMAP_GPU_HOURS_TOTAL
    w = f["what_has_to_give"]
    # the reduced plan keeps BOTH target cells whole
    assert w["reduced_plan_crop_inferences"] == 199 + len(manifests["pilot"]["crops"])
    assert w["reduced_plan_saving_t4_hours"] > 0.0


def test_fold_1_target_cell_reproduces_the_decision_package_session_fraction(manifests):
    """0.766 of a 9 h session for the 128 held-out 6bba crops is the planning anchor."""
    blk = manifests["full"]["budget"]["by_cell"]["split_1__6bba"]
    assert blk["crop_inferences"] == 128
    assert 0.74 <= blk["session_fraction_per_kernel"] <= 0.83


# --------------------------------------------------------------------------------------
# 9. determinism -- regeneration must byte-reproduce the committed manifests
# --------------------------------------------------------------------------------------


def test_regeneration_byte_reproduces_every_manifest(tmp_path):
    res = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "build",
            "--census",
            str(CENSUS),
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    for name in [f"manifest_{t}.json" for t in TIERS] + ["manifest_index.json"]:
        got = lf((tmp_path / name).read_bytes())
        want = lf((FACT_DIR / name).read_bytes())
        assert got == want, f"{name} is not byte-reproducible from the census"


def test_index_hashes_match_the_manifests_on_disk():
    idx = json.loads((FACT_DIR / "manifest_index.json").read_text())
    import hashlib

    assert idx["census_sha256"] == hashlib.sha256(lf(CENSUS.read_bytes())).hexdigest()
    for tier in TIERS:
        raw = lf((FACT_DIR / f"manifest_{tier}.json").read_bytes())
        assert idx["tiers"][tier]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert idx["tiers"][tier]["file"] == f"manifest_{tier}.json"


def test_build_is_a_pure_function_of_the_census(tmp_path):
    """Two runs from the same census, different output dirs, identical bytes."""
    outs = []
    for i in range(2):
        d = tmp_path / f"run{i}"
        res = subprocess.run(
            [sys.executable, str(SCRIPT), "build", "--census", str(CENSUS), "--out-dir", str(d)],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, res.stderr
        outs.append({p.name: p.read_bytes() for p in sorted(d.glob("*.json"))})
    assert outs[0] == outs[1]


def test_generator_always_writes_lf(tmp_path):
    """The generator's own bytes must be LF-only, whatever the checkout filter does later."""
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "build", "--census", str(CENSUS), "--out-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    for p in sorted(tmp_path.glob("*.json")):
        raw = p.read_bytes()
        assert b"\r\n" not in raw, f"{p.name} was written with CRLF"
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n"), p.name


def test_committed_manifests_end_with_exactly_one_newline():
    for name in [f"manifest_{t}.json" for t in TIERS] + ["manifest_index.json", "crop_census.json"]:
        raw = lf((FACT_DIR / name).read_bytes())
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n"), name
