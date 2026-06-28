"""Tests for normalized regret against the per-task oracle."""

from __future__ import annotations

import pytest

from membench.analysis.tables import ScoredRow
from membench.metrics.regret import (
    ArmRegret,
    normalized_regret,
    regret_by_arm,
    regret_per_task,
)


def _row(arm: str, compliance: float, *, depth: int = 1) -> ScoredRow:
    """Build a ScoredRow varying only arm/compliance/depth (other fields fixed)."""
    return ScoredRow(
        dataset="synthetic",
        arm=arm,
        model="claude",
        depth=depth,
        spec_variant="stripped",
        compliance_rate=compliance,
        chain_recovered=compliance >= 1.0,
        recall=compliance,
        precision=compliance,
        correct=compliance >= 1.0,
        total_tokens=100,
        dollars=0.01,
    )


class TestScalar:
    def test_golden_midpoint(self) -> None:
        # Hand-computed: (best - achieved)/(best - worst) = (1.0 - 0.5)/(1.0 - 0.0) = 0.5
        assert normalized_regret(0.5, 1.0, 0.0) == pytest.approx(0.5)

    def test_oracle_has_zero_regret(self) -> None:
        # achieved == best -> numerator 0 -> regret 0
        assert normalized_regret(1.0, 1.0, 0.0) == pytest.approx(0.0)

    def test_floor_has_unit_regret(self) -> None:
        # achieved == worst -> numerator == denominator -> regret 1
        assert normalized_regret(0.0, 1.0, 0.0) == pytest.approx(1.0)

    def test_clipped_below_floor(self) -> None:
        # An arm below the floor still clips to 1, never above.
        assert normalized_regret(-0.5, 1.0, 0.0) == pytest.approx(1.0)

    def test_clipped_above_oracle(self) -> None:
        # An arm above the oracle clips to 0, never negative.
        assert normalized_regret(1.5, 1.0, 0.0) == pytest.approx(0.0)

    def test_degenerate_no_spread(self) -> None:
        # best == worst -> no spread to normalise -> regret 0
        assert normalized_regret(0.4, 0.7, 0.7) == pytest.approx(0.0)

    def test_arbitrary_scale(self) -> None:
        # (8 - 5)/(8 - 2) = 3/6 = 0.5; normalisation is scale-free.
        assert normalized_regret(5.0, 8.0, 2.0) == pytest.approx(0.5)


class TestPerTask:
    def test_baseline_floor(self) -> None:
        scores = {"oracle": 1.0, "mid": 0.5, "none": 0.0}
        out = regret_per_task(scores, baseline_arm="none")
        assert out == pytest.approx({"oracle": 0.0, "mid": 0.5, "none": 1.0})

    def test_min_floor_when_baseline_absent(self) -> None:
        # No "none" arm here -> floor is the min score (0.2).
        # mid: (1.0 - 0.6)/(1.0 - 0.2) = 0.4/0.8 = 0.5
        out = regret_per_task({"oracle": 1.0, "mid": 0.6, "low": 0.2})
        assert out == pytest.approx({"oracle": 0.0, "mid": 0.5, "low": 1.0})

    def test_empty(self) -> None:
        assert regret_per_task({}) == {}


class TestByArm:
    def test_golden_two_task_set(self) -> None:
        # Two tasks (distinct depths -> distinct task keys), three arms each.
        # Per task: best = 1.0 (oracle), floor = 0.0 (none).
        #   oracle: (1.0 - 1.0)/1.0 = 0.0
        #   mid:    (1.0 - 0.5)/1.0 = 0.5  <-- the asserted golden
        #   none:   (1.0 - 0.0)/1.0 = 1.0
        # Each arm's reported regret = mean over the two identical tasks.
        rows = [
            _row("oracle", 1.0, depth=1),
            _row("mid", 0.5, depth=1),
            _row("none", 0.0, depth=1),
            _row("oracle", 1.0, depth=2),
            _row("mid", 0.5, depth=2),
            _row("none", 0.0, depth=2),
        ]
        result = regret_by_arm(rows)
        assert result["mid"].regret == pytest.approx(0.5)
        assert result["oracle"].regret == pytest.approx(0.0)
        assert result["none"].regret == pytest.approx(1.0)

        # Invariants: oracle regret 0, floor regret 1, all regrets in [0, 1].
        assert all(0.0 <= ar.regret <= 1.0 for ar in result.values())
        assert all(ar.n_tasks == 2 for ar in result.values())
        assert isinstance(result["mid"], ArmRegret)

    def test_duplicate_arm_in_task_rejected(self) -> None:
        rows = [_row("mid", 0.5, depth=1), _row("mid", 0.6, depth=1)]
        with pytest.raises(ValueError, match="ambiguous oracle"):
            regret_by_arm(rows)

    def test_empty_rows(self) -> None:
        assert regret_by_arm([]) == {}
