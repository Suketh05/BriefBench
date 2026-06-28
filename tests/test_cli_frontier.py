"""Tests for the ``frontier`` CLI subcommand (Pareto frontier + AUDC).

The golden values are derived by hand below, never by running the command under
test, so the assertions are an independent check rather than a circular one.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from membench.analysis.tables import ScoredRow
from membench.cli import app
from membench.harness import save_rows

runner = CliRunner()


def _row(arm: str, depth: int, *, correct: bool, tokens: int, recovered: bool) -> ScoredRow:
    """Build a minimal ScoredRow; only the frontier-relevant fields vary."""
    return ScoredRow(
        dataset="synthetic",
        arm=arm,
        model="claude",
        depth=depth,
        spec_variant="stripped",
        compliance_rate=1.0 if correct else 0.0,
        chain_recovered=recovered,
        recall=1.0 if recovered else 0.0,
        precision=1.0 if recovered else 0.0,
        correct=correct,
        total_tokens=tokens,
        dollars=0.0,
    )


def _fixture_rows() -> list[ScoredRow]:
    """Three arms over depths 1-3 with hand-chosen accuracy/cost/recovery.

    Per-arm aggregates (accuracy = mean correct, cost = mean total_tokens):

    * ``brief_graph_3hop`` -- correct=[T,T,T] -> acc 1.000; tokens 200 -> cost 200.0;
      recovery y=[1,1,1].
    * ``none``             -- correct=[F,F,F] -> acc 0.000; tokens 50  -> cost 50.0;
      recovery y=[1,0,0].
    * ``mid_arm``          -- correct=[T,F,F] -> acc 0.333; tokens 300 -> cost 300.0;
      recovery y=[1,0,0].
    """
    rows: list[ScoredRow] = []
    for depth in (1, 2, 3):
        rows.append(_row("brief_graph_3hop", depth, correct=True, tokens=200, recovered=True))
        rows.append(_row("none", depth, correct=False, tokens=50, recovered=(depth == 1)))
        rows.append(
            _row("mid_arm", depth, correct=(depth == 1), tokens=300, recovered=(depth == 1))
        )
    return rows


def test_frontier_reports_pareto_and_audc(tmp_path: Path) -> None:
    """`frontier` prints the non-dominated arms and the hand-computed AUDC."""
    rows_path = tmp_path / "rows.jsonl"
    save_rows(_fixture_rows(), rows_path)

    result = runner.invoke(app, ["frontier", "--src", str(rows_path)])
    assert result.exit_code == 0

    # Structural markers.
    assert "Pareto frontier" in result.stdout
    assert "AUDC" in result.stdout

    frontier_part, _, audc_part = result.stdout.partition("area under depth-recovery curve")
    assert audc_part, "AUDC section header missing"

    # GOLDEN -- Pareto frontier (minimise cost, maximise accuracy).
    # Points (cost, acc): none=(50,0.0), brief_graph_3hop=(200,1.0), mid_arm=(300,0.333).
    # brief_graph_3hop dominates mid_arm (200<=300 and 1.0>=0.333, strict on both) ->
    # mid_arm is off the frontier. none and brief_graph_3hop are mutually non-dominated
    # (cheaper vs. more accurate), so the frontier is exactly {none, brief_graph_3hop}.
    assert "none" in frontier_part
    assert "brief_graph_3hop" in frontier_part
    assert "mid_arm" not in frontier_part
    assert "mid_arm" in audc_part

    # GOLDEN -- AUDC via the composite trapezoidal rule over depths [1, 2, 3]:
    #   brief_graph_3hop: y=[1,1,1] -> 0.5*(1+1)*1 + 0.5*(1+1)*1 = 2.000
    #   none:             y=[1,0,0] -> 0.5*(1+0)*1 + 0.5*(0+0)*1 = 0.500
    assert "audc=2.000" in audc_part
    assert "audc=0.500" in audc_part
