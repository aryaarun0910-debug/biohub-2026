"""Contract tests for the relink division mechanism.

These test SOFTWARE INVARIANTS only. Whether admitting a second child improves the score is
a scientific question and belongs to a scored experiment, not to a unit test.

The mechanism: `motion_relink_edges -> assign_pass` builds an (n_src x n_tgt) cost matrix
and calls `linear_sum_assignment`, a bijection that cannot express out-degree 2 and so
collapsed 14 of the 24 divisions present pre-ILP (FACT-0090). Duplicating each source row
and pricing the copy lets the solver SELECT which sources earn a second child.

What must hold:
  * penalty < 0 reproduces the champion EXACTLY (byte-identical output)
  * in-degree <= 1 always (each target gets at most one parent)
  * out-degree <= 2 always (each source has exactly two rows)
  * the penalty is monotone: raising it never yields more second children
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
BUILT = REPO / "notebooks" / "kaggle_p18_relink_division" / "biohub-p18-relink-division.ipynb"


def champion_assign(cost: np.ndarray, big: float) -> list[tuple[int, int]]:
    """The deployed bijection, cell 6:487."""
    row_ind, col_ind = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if cost[r, c] < big]


def division_assign(cost: np.ndarray, big: float, penalty: float) -> list[tuple[int, int]]:
    """The patched form: a second, priced row per source."""
    n_src = cost.shape[0]
    cost_aug = np.vstack([cost, cost + penalty]) if penalty >= 0.0 else cost
    row_ind, col_ind = linear_sum_assignment(cost_aug)
    return [(int(r) % n_src, int(c))
            for r, c in zip(row_ind, col_ind) if cost_aug[r, c] < big]


def random_cost(rng: np.random.Generator, n_src: int, n_tgt: int, big: float) -> np.ndarray:
    cost = rng.uniform(0.5, 6.0, size=(n_src, n_tgt))
    gated = rng.random(size=cost.shape) < 0.4          # most pairs are out of gate
    cost[gated] = big
    return cost


# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(12))
def test_disabled_penalty_reproduces_champion_exactly(seed):
    """The default MUST be a no-op. A silent behaviour change in the deployed path is the
    failure mode this whole guard exists to prevent."""
    rng = np.random.default_rng(seed)
    big = 10_001.0
    cost = random_cost(rng, rng.integers(3, 25), rng.integers(3, 25), big)
    assert division_assign(cost, big, -1.0) == champion_assign(cost, big)


@pytest.mark.parametrize("seed", range(12))
def test_in_degree_never_exceeds_one(seed):
    rng = np.random.default_rng(100 + seed)
    big = 10_001.0
    cost = random_cost(rng, rng.integers(3, 25), rng.integers(3, 25), big)
    targets = [c for _, c in division_assign(cost, big, 1.5)]
    assert len(targets) == len(set(targets)), "a target was given two parents"


@pytest.mark.parametrize("seed", range(12))
def test_out_degree_never_exceeds_two(seed):
    rng = np.random.default_rng(200 + seed)
    big = 10_001.0
    cost = random_cost(rng, rng.integers(3, 25), rng.integers(3, 25), big)
    sources = [s for s, _ in division_assign(cost, big, 0.0)]
    counts = {s: sources.count(s) for s in set(sources)}
    assert max(counts.values(), default=0) <= 2, f"out-degree exceeded 2: {counts}"


def test_penalty_is_monotone_in_second_children():
    """Raising the price must never buy MORE divisions, or the knob is not a selector."""
    rng = np.random.default_rng(7)
    big = 10_001.0
    totals = []
    for penalty in (0.0, 1.0, 2.0, 4.0, 8.0):
        n = 0
        for _ in range(25):
            cost = random_cost(rng, 12, 12, big)
            sources = [s for s, _ in division_assign(cost, big, penalty)]
            n += sum(1 for s in set(sources) if sources.count(s) == 2)
        totals.append(n)
    assert totals == sorted(totals, reverse=True), f"not monotone: {totals}"
    assert totals[0] > totals[-1], "the penalty had no effect at all"


# --------------------------------------------------------------------------------------

@pytest.mark.skipif(not BUILT.is_file(), reason="p18 notebook not built")
def test_built_notebook_carries_the_mechanism_and_defaults_off():
    src = "\n".join("".join(c["source"]) for c in json.load(BUILT.open(encoding="utf-8"))["cells"])
    assert 'BIOHUB_RELINK_DIVISION_PENALTY", "-1"' in src, "default is not the disabled sentinel"
    assert "cost_aug = np.vstack([cost, cost + RELINK_DIVISION_PENALTY])" in src
    assert src.count("linear_sum_assignment(cost_aug)") == 1
    # close_single_frame_gaps must be untouched - it is a different bijection (cell 6:697)
    assert src.count("linear_sum_assignment(cost)") == 1, (
        "the gap-close bijection should not have been modified by this arm"
    )


# --------------------------------------------------------------------------------------
# The penalty SWEEP (EXP-0019) — buys the whole curve from one GPU session.
# --------------------------------------------------------------------------------------

SWEEP = REPO / "notebooks" / "kaggle_p19_relink_sweep_f1" / "biohub-p19-relink-sweep-f1.ipynb"


@pytest.mark.skipif(not SWEEP.is_file(), reason="p19 sweep notebook not built")
class TestSweepNotebook:
    @staticmethod
    def cells() -> list[str]:
        return ["".join(c["source"]) for c in json.load(SWEEP.open(encoding="utf-8"))["cells"]]

    def test_sweep_runs_before_the_cleanup_that_deletes_the_geffs(self):
        """The LOEO cleanup removes every /kaggle/working entry not in _LOEO_KEEP, and the
        geffs live under /kaggle/working/tracking_repo. A sweep ordered after it would find
        nothing to post-process and would burn a full GPU session producing empty files."""
        for cell in self.cells():
            if "for _sw_pen in _sw_pens:" in cell and "_LOEO_KEEP = {" in cell:
                assert cell.index("for _sw_pen in _sw_pens:") < cell.index("_LOEO_KEEP = {")
                return
        pytest.fail("sweep and cleanup are not in the same cell; ordering is unverified")

    def test_sweep_outputs_survive_the_cleanup(self):
        src = "\n".join(self.cells())
        assert '_LOEO_KEEP |= set(globals().get("_SWEEP_KEEP", []))' in src

    def test_sweep_includes_the_disabled_control(self):
        """Penalty -1 reproduces the champion, so it is the in-run control. Without it the
        sweep has no way to show it is measuring what it claims to."""
        src = "\n".join(self.cells())
        assert 'BIOHUB_RELINK_DIVISION_SWEEP' in src
        assert '"-1,0,1,2,4,8"' in src or "'-1,0,1,2,4,8'" in src

    def test_sweep_emits_no_submission(self):
        """This is a diagnostic export. It must not consume a submission slot."""
        spec = json.loads((REPO / "scripts/kaggle_specs/p19_relink_sweep_f1.json").read_text())
        assert spec.get("expects_submission") is False

    def test_sweep_is_on_the_champion_substrate(self):
        spec = json.loads((REPO / "scripts/kaggle_specs/p19_relink_sweep_f1.json").read_text())
        assert "p3_harmonic" in spec["base_notebook"], (
            "the sweep must measure the substrate we deploy, not the older p0b LOEO base"
        )
