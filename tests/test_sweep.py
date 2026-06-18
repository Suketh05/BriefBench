"""Tests for the config-driven sweep (the full end-to-end with a manifest)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from membench.cli import app
from membench.config.schema import BenchmarkConfig
from membench.harness import run_from_config

runner = CliRunner()


def _config(**over: object) -> BenchmarkConfig:
    base: dict[str, object] = {
        "datasets": ["synthetic"],
        "arms": ["none", "dense", "brief_graph_3hop"],
        "models": ["claude"],
        "budget_tokens_by_dataset": {"synthetic": 150},
        "per_depth": 6,
        "seed": 0,
        "offline": True,
    }
    base.update(over)
    return BenchmarkConfig.model_validate(base)


class TestRunFromConfig:
    def test_sweep_produces_rows_and_manifest(self) -> None:
        result = run_from_config(_config())
        assert result.rows
        assert {r.arm for r in result.rows} == {"none", "dense", "brief_graph_3hop"}
        assert result.manifest.membench_version == "0.1.0"
        assert result.manifest.config["datasets"] == ["synthetic"]
        assert not result.skipped

    def test_unavailable_competitor_arm_is_skipped_not_fatal(self) -> None:
        # mem0 needs its package (absent here) -> recorded as skipped, sweep still runs
        result = run_from_config(_config(arms=["none", "mem0", "brief_graph_3hop"]))
        assert "mem0" in result.skipped
        assert {r.arm for r in result.rows} == {"none", "brief_graph_3hop"}

    def test_uses_per_dataset_budget(self) -> None:
        # tiny budget -> none arm still floor; just assert it runs across the budget
        result = run_from_config(_config(budget_tokens_by_dataset={"synthetic": 80}))
        assert result.rows


class TestSweepCLI:
    def test_sweep_writes_rows_and_manifest(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "datasets: [synthetic]\n"
            "arms: [none, brief_graph_3hop]\n"
            "budget_tokens_by_dataset: {synthetic: 150}\n"
            "per_depth: 4\n"
        )
        out = tmp_path / "out"
        res = runner.invoke(app, ["sweep", "--config", str(cfg_path), "--out-dir", str(out)])
        assert res.exit_code == 0, res.stdout
        assert (out / "rows.jsonl").exists()
        assert (out / "manifest.json").exists()
        assert "device:" in res.stdout
