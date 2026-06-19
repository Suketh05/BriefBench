"""Tests for the config-driven sweep (run_from_config) and its run manifest.

The sweep is the reproducible end-to-end entry point: a validated config in, scored
rows plus a manifest out, with unavailable arms skipped rather than fatal. It delegates
the per-task work to run_benchmark, so these tests also guard that one execution path.
"""

from __future__ import annotations

from typing import Any

from membench.config.schema import BenchmarkConfig
from membench.harness import SweepResult, run_from_config

_CREATED = "2026-01-01T00:00:00Z"


def _config(**overrides: Any) -> BenchmarkConfig:
    base: dict[str, Any] = {
        "datasets": ["synthetic"],
        "arms": ["none", "brief_graph_3hop"],
        "models": ["claude"],
        "budget_tokens_by_dataset": {"synthetic": 150, "dcbench": 150, "swebench": 150},
        "per_depth": 3,
        "seed": 0,
    }
    base.update(overrides)
    return BenchmarkConfig(**base)


class TestRunFromConfig:
    def test_produces_rows_and_manifest(self) -> None:
        result = run_from_config(_config(), created_at=_CREATED)
        assert isinstance(result, SweepResult)
        assert result.rows  # non-empty
        assert result.manifest.created_at == _CREATED
        assert result.manifest.config["seed"] == 0
        assert not result.skipped  # both arms run offline

    def test_fresh_arm_per_task_no_state_leak(self) -> None:
        # brief_graph_3hop accumulates nodes; reusing one arm across tasks raised
        # "duplicate node id". Delegating to run_benchmark (a fresh arm per task) avoids it.
        result = run_from_config(_config(arms=["brief_graph_3hop"], per_depth=5))
        assert result.rows
        assert "brief_graph_3hop" not in result.skipped

    def test_unavailable_arm_is_skipped_not_fatal(self) -> None:
        # mem0 needs its own package/env; offline it raises -> recorded skip, sweep survives.
        result = run_from_config(_config(arms=["none", "mem0"]))
        assert "mem0" in result.skipped
        assert "mem0" in result.skipped["mem0"]
        assert result.rows  # 'none' still ran

    def test_runs_every_configured_arm(self) -> None:
        # the sweep executes each configured arm over the dataset (the fairness lock
        # itself -- one budget per dataset for all arms -- is validated in test_config).
        result = run_from_config(_config(arms=["none", "brief_graph_3hop"]))
        assert {r.arm for r in result.rows} == {"none", "brief_graph_3hop"}

    def test_is_deterministic(self) -> None:
        a = run_from_config(_config(), created_at=_CREATED)
        b = run_from_config(_config(), created_at=_CREATED)
        assert [r.compliance_rate for r in a.rows] == [r.compliance_rate for r in b.rows]
