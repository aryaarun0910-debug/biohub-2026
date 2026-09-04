"""The controller's decision: what the campaign does next, from recorded state and nothing else.

Arya Arun's rule table of 2026-09-04, applied without discretion:

    propensity falsified objective -> stop and redesign
    smoke failed                   -> kill or hold the arm
    one fold improved              -> challenger only
    both folds improved            -> qualify the full-fold stage
    proposal ceiling below 0.950   -> association and submission stay blocked
    full official metric passes    -> request integration or submission approval

Agents may propose hypotheses. Only these rules spend GPU, promote a component
or unlock a downstream stage, and they read the registries rather than a
conversation. The state is assembled from ``registry/experiments.yaml`` and the
propensity report; a fact that is not recorded is ``None`` and the decision
says so instead of assuming.

Consumer: ``biohubx research collect``, which writes the decision into the
campaign's collection and dossier.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

PROPOSAL_CEILING_TARGET = 0.950
"""The declared target on each embryo (E07 promotion metric); not tuned here."""


@dataclass(frozen=True)
class ArmState:
    arm: str
    smoke_passed: bool | None
    folds_improved: int | None
    """Held-out directions on which the arm beat A0's ceiling, out of two; None until measured."""


@dataclass(frozen=True)
class CampaignState:
    propensity: str | None
    """``present``, ``absent`` or None when the probe has not run."""

    propensity_temporal: bool | None
    arms: tuple[ArmState, ...]
    proposal_ceiling_44b6: float | None
    proposal_ceiling_6bba: float | None
    official_metric_passes: bool | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["arms"] = [asdict(arm) for arm in self.arms]
        return payload


@dataclass(frozen=True)
class Decision:
    next_action: str
    rule: str
    arms: dict[str, str]
    blocked: tuple[str, ...]
    unlocked: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_action": self.next_action,
            "rule": self.rule,
            "arms": dict(self.arms),
            "blocked": list(self.blocked),
            "unlocked": list(self.unlocked),
        }


def next_action(state: CampaignState) -> Decision:
    """The rule table, top to bottom; the first rule that applies decides."""
    blocked: list[str] = []
    ceilings = (state.proposal_ceiling_44b6, state.proposal_ceiling_6bba)
    ceiling_known = all(value is not None for value in ceilings)
    ceiling_met = ceiling_known and all(
        float(value) >= PROPOSAL_CEILING_TARGET for value in ceilings if value
    )
    if not ceiling_met:
        blocked.extend(["association (E06 resume)", "submission"])

    if state.propensity == "present":
        held: dict[str, str] = {}
        for arm in state.arms:
            if arm.arm == "A4" and state.propensity_temporal:
                held[arm.arm] = "hold; neighbouring frames can learn tracing propensity"
            else:
                held[arm.arm] = "hold under the current objective; eligible only under a corrected one"
        return Decision(
            next_action="stop the current nnPU training and design a bias-robust objective",
            rule="propensity falsified objective -> stop and redesign",
            arms=held,
            blocked=(*blocked, "representative Stage 2B under the current objective"),
            unlocked=(),
        )

    if state.propensity is None:
        return Decision(
            next_action="run E07-PROPENSITY-01 before any further GPU",
            rule="no rule applies before the probe has answered",
            arms={arm.arm: "wait" for arm in state.arms},
            blocked=(*blocked, "representative Stage 2B"),
            unlocked=(),
        )

    arms: dict[str, str] = {}
    unlocked: list[str] = []
    for arm in state.arms:
        if arm.smoke_passed is False:
            arms[arm.arm] = "kill or hold: the smoke failed"
        elif arm.smoke_passed is None:
            arms[arm.arm] = "run the representative smoke"
        elif arm.folds_improved is None:
            arms[arm.arm] = "run the mini-fold in both directions"
        elif arm.folds_improved >= 2:
            arms[arm.arm] = "qualified for the full-fold stage"
            unlocked.append(f"full-fold stage for {arm.arm}")
        elif arm.folds_improved == 1:
            arms[arm.arm] = "challenger only; one direction improved"
        else:
            arms[arm.arm] = "kill: neither direction improved"

    if state.official_metric_passes:
        return Decision(
            next_action="request integration or submission approval from Arya Arun",
            rule="full official metric passes -> request approval",
            arms=arms,
            blocked=tuple(blocked),
            unlocked=(*unlocked, "integration or submission request"),
        )
    if unlocked:
        return Decision(
            next_action="qualify the full-fold stage for the arms that improved both directions",
            rule="both folds improved -> qualify full-fold stage",
            arms=arms,
            blocked=tuple(blocked),
            unlocked=tuple(unlocked),
        )
    if any(arm.folds_improved == 1 for arm in state.arms):
        return Decision(
            next_action="record the arm as challenger; no promotion, no full fold",
            rule="one fold improved -> challenger only",
            arms=arms,
            blocked=tuple(blocked),
            unlocked=(),
        )
    if any(arm.smoke_passed is False for arm in state.arms) and not any(
        arm.smoke_passed is None for arm in state.arms
    ):
        return Decision(
            next_action="kill or hold the arms whose smoke failed",
            rule="smoke failed -> kill/hold arm",
            arms=arms,
            blocked=tuple(blocked),
            unlocked=(),
        )
    return Decision(
        next_action="run the next stage the arms are waiting on",
        rule="no terminal rule applies; the funnel continues",
        arms=arms,
        blocked=tuple(blocked),
        unlocked=(),
    )


def state_from_registries(root: Path) -> CampaignState:
    """Assemble the state from what is recorded; anything unrecorded stays None."""
    registry = root / "registry/experiments.yaml"
    experiments: dict[str, Any] = (
        yaml.safe_load(registry.read_text(encoding="utf-8")) or {} if registry.is_file() else {}
    )
    e07: dict[str, Any] = next((e for e in experiments.get("experiments", []) if e.get("id") == "E07"), {})
    stage_results = dict(e07.get("stage_results", {}))

    propensity: str | None = None
    temporal: bool | None = None
    report = root / "artifacts/e07-propensity.json"
    if report.is_file():
        body = json.loads(report.read_text(encoding="utf-8"))
        verdicts = {name: block["decision"]["verdict"] for name, block in body.get("per_embryo", {}).items()}
        if verdicts:
            propensity = (
                "present" if any(v == "propensity_signal_present" for v in verdicts.values()) else "absent"
            )
            temporal = any(
                bool(block["null"]["temporal"]["signal"]) for block in body.get("per_embryo", {}).values()
            )

    arms = []
    for name in ("A3", "A4"):
        block = stage_results.get(name, {})
        smoke = block.get("stage2b", block.get("stage2"))
        smoke_passed: bool | None = None
        if isinstance(smoke, dict) and "stage3_condition_met" in smoke:
            smoke_passed = bool(smoke["stage3_condition_met"])
        folds = block.get("stage3")
        improved: int | None = None
        if isinstance(folds, dict) and "directions_improved" in folds:
            improved = int(folds["directions_improved"])
        arms.append(ArmState(arm=name, smoke_passed=smoke_passed, folds_improved=improved))

    baseline = dict(e07.get("baseline", {}).get("ceilings", {}))
    promoted = dict(e07.get("promoted_ceilings", {}))
    ceiling_44b6 = promoted.get("44b6", baseline.get("44b6"))
    ceiling_6bba = promoted.get("6bba", baseline.get("6bba"))
    official = e07.get("official_metric_passes")
    return CampaignState(
        propensity=propensity,
        propensity_temporal=temporal,
        arms=tuple(arms),
        proposal_ceiling_44b6=float(ceiling_44b6) if ceiling_44b6 is not None else None,
        proposal_ceiling_6bba=float(ceiling_6bba) if ceiling_6bba is not None else None,
        official_metric_passes=bool(official) if official is not None else None,
    )
