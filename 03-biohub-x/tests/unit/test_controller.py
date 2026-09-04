"""The rule table that spends GPU and unlocks stages, exercised row by row.

A wrong branch here either spends compute on a condition that already failed or
withholds it from an arm that earned it, so every rule in Arya Arun's table has
a test and the order of precedence is asserted rather than assumed.
"""

from __future__ import annotations

from biohubx.research.controller import ArmState, CampaignState, next_action


def _state(**overrides: object) -> CampaignState:
    base: dict[str, object] = {
        "propensity": "absent",
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


def test_a_present_propensity_signal_stops_everything_and_redesigns() -> None:
    decision = next_action(_state(propensity="present", propensity_temporal=True))
    assert decision.rule.startswith("propensity falsified")
    assert decision.arms["A4"].startswith("hold; neighbouring")
    assert "Stage 2B under the current objective" in " ".join(decision.blocked)
    assert decision.unlocked == ()


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
            arms=(ArmState("A3", smoke_passed=True, folds_improved=2),),
            official_metric_passes=True,
        )
    )
    assert decision.rule.startswith("propensity falsified")
