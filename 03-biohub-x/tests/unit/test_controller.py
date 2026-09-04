"""The rule table that spends GPU and unlocks stages, exercised row by row.

A wrong branch here either spends compute on a condition that already failed or
withholds it from an arm that earned it, so every rule in Arya Arun's table has
a test and the order of precedence is asserted rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from biohubx.research.controller import (
    ArmState,
    CampaignState,
    next_action,
    state_from_registries,
)


def _state(**overrides: object) -> CampaignState:
    base: dict[str, object] = {
        "propensity": "absent",
        "propensity_replicates": False,
        "ceiling_degraded": False,
        "propensity_temporal": False,
        "arms": (
            ArmState("A3", smoke_passed=None, folds_improved=None),
            ArmState("A4", smoke_passed=None, folds_improved=None),
        ),
        "proposal_ceiling_44b6": 1.0195,
        "proposal_ceiling_6bba": 0.9283,
        "official_metric_passes": None,
    }
    base.update(overrides)
    return CampaignState(**base)  # type: ignore[arg-type]


def test_a_signal_on_one_embryo_is_weak_evidence_and_does_not_stop_the_campaign() -> None:
    """Arya Arun's amendment: present on one embryo is embryo-specific, not a kill."""
    decision = next_action(_state(propensity="present", propensity_temporal=True))
    assert not decision.rule.startswith("propensity falsified")
    assert decision.arms["A4"].startswith("hold")
    assert decision.arms["A3"] == "run the representative smoke"
    assert "Stage 2B under the current objective" not in " ".join(decision.blocked)


def test_a_signal_replicated_across_both_embryos_stops_and_redesigns() -> None:
    decision = next_action(_state(propensity="present", propensity_replicates=True, propensity_temporal=True))
    assert decision.rule.startswith("propensity falsified objective (replicated")
    assert decision.arms["A4"].startswith("hold; neighbouring")
    assert "Stage 2B under the current objective" in " ".join(decision.blocked)
    assert decision.unlocked == ()


def test_a_degraded_trained_ceiling_stops_and_redesigns() -> None:
    decision = next_action(_state(propensity="present", ceiling_degraded=True))
    assert decision.rule.startswith("propensity falsified objective (trained ceiling degraded")


def test_nothing_is_decided_before_the_probe_has_run() -> None:
    decision = next_action(_state(propensity=None))
    assert "E07-PROPENSITY-01" in decision.next_action
    assert "representative Stage 2B" in decision.blocked


def test_a_failed_smoke_kills_or_holds_the_arm() -> None:
    decision = next_action(
        _state(
            arms=(
                ArmState("A3", smoke_passed=False, folds_improved=None),
                ArmState("A4", smoke_passed=False, folds_improved=None),
            )
        )
    )
    assert decision.rule == "smoke failed -> kill/hold arm"
    assert all(v.startswith("kill or hold") for v in decision.arms.values())


def test_one_improved_direction_is_a_challenger_and_nothing_more() -> None:
    decision = next_action(
        _state(
            arms=(
                ArmState("A3", smoke_passed=True, folds_improved=1),
                ArmState("A4", smoke_passed=True, folds_improved=0),
            )
        )
    )
    assert decision.rule == "one fold improved -> challenger only"
    assert decision.arms["A3"].startswith("challenger only")
    assert decision.arms["A4"].startswith("kill")
    assert decision.unlocked == ()


def test_both_directions_improved_qualifies_the_full_fold_stage() -> None:
    decision = next_action(
        _state(
            arms=(
                ArmState("A3", smoke_passed=True, folds_improved=2),
                ArmState("A4", smoke_passed=True, folds_improved=1),
            )
        )
    )
    assert decision.rule == "both folds improved -> qualify full-fold stage"
    assert "full-fold stage for A3" in decision.unlocked
    assert "full-fold stage for A4" not in decision.unlocked


def test_association_and_submission_stay_blocked_below_the_ceiling_target() -> None:
    decision = next_action(_state())
    assert "association (E06 resume)" in decision.blocked
    assert "submission" in decision.blocked

    clear = next_action(_state(proposal_ceiling_44b6=0.96, proposal_ceiling_6bba=0.951))
    assert "submission" not in clear.blocked


def test_a_passing_official_metric_requests_approval_rather_than_taking_it() -> None:
    decision = next_action(
        _state(
            proposal_ceiling_44b6=0.97,
            proposal_ceiling_6bba=0.96,
            official_metric_passes=True,
            arms=(ArmState("A3", smoke_passed=True, folds_improved=2),),
        )
    )
    assert decision.rule.startswith("full official metric passes")
    assert "Arya Arun" in decision.next_action
    assert "integration or submission request" in decision.unlocked


def test_the_probe_outranks_every_other_rule() -> None:
    """A present signal must stop the campaign even when an arm looks qualified."""
    decision = next_action(
        _state(
            propensity="present",
            propensity_replicates=True,
            arms=(ArmState("A3", smoke_passed=True, folds_improved=2),),
            official_metric_passes=True,
        )
    )
    assert decision.rule.startswith("propensity falsified")


def _registry(tmp_path: Path, a3: dict[str, object]) -> Path:
    (tmp_path / "registry").mkdir()
    body = {
        "experiments": [
            {"id": "E07", "baseline": {"ceilings": {"44b6": 1.0, "6bba": 0.9}}, "stage_results": {"A3": a3}}
        ]
    }
    (tmp_path / "registry/experiments.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    return tmp_path


def test_a_stage2b_attempt_without_a_measurement_leaves_the_smoke_unmeasured(tmp_path: Path) -> None:
    """Wave-1 attempts 1 to 3 were integration failures. They say nothing about the
    arm, so the controller must not read them as a failed smoke, and must not fall
    back to the superseded Stage 2 reading either."""
    root = _registry(
        tmp_path,
        {
            "stage2": {"stage3_condition_met": False},
            "stage2b_fold_44b6_attempt1": {"attempt": 1, "outcome": "no summary manifest retrieved"},
        },
    )
    state = state_from_registries(root)
    assert next(a for a in state.arms if a.arm == "A3").smoke_passed is None


def test_a_measured_stage2b_in_both_directions_decides_the_smoke(tmp_path: Path) -> None:
    root = _registry(
        tmp_path,
        {
            "stage2b_fold_44b6_attempt3": {
                "advancement_condition_met_per_seed": {"0": True, "1": True},
                "ceiling_degraded_any_seed": False,
            },
            "stage2b_fold_6bba_attempt4": {
                "advancement_condition_met_per_seed": {"0": True, "1": False},
                "ceiling_degraded_any_seed": True,
            },
        },
    )
    state = state_from_registries(root)
    a3 = next(a for a in state.arms if a.arm == "A3")
    assert a3.smoke_passed is False
    assert state.ceiling_degraded is True


def test_one_measured_direction_that_degraded_is_enough_to_flag_degradation(tmp_path: Path) -> None:
    """R-0031: seed 0 of the fold_6bba direction fell from A0's 0.904 to 0.654 while
    the other seed died and the other direction never ran. Degradation is D-0046's
    second kill branch and does not wait for replication; the smoke does."""
    root = _registry(
        tmp_path,
        {
            "stage2b_fold_6bba_attempt4": {
                "advancement_condition_met_per_seed": {"0": False},
                "ceiling_degraded_any_seed": True,
            }
        },
    )
    state = state_from_registries(root)
    assert state.ceiling_degraded is True
    assert next(a for a in state.arms if a.arm == "A3").smoke_passed is None
