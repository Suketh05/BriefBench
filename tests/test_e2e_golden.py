"""End-to-end golden test: the whole pipeline must reproduce the headline result.

Runs the benchmark from datasets through arms, runner, scoring, tables, and report,
and asserts the depth-crossover signature holds: similarity decays with depth while the
structured graph arm stays flat, and the structured arm wins at depth by a margin. If
the core result ever breaks, this fails -- a guard on the claim the whole repo exists to
make. Offline and deterministic (seeded), so it is a stable regression check.
"""

from __future__ import annotations

import pytest

from membench.analysis.report import generate_report
from membench.analysis.tables import (
    ablation_table,
    depth_crossover_table,
    headline_table,
)
from membench.harness import DEFAULT_ARMS, run_benchmark
from membench.theory.crossover import find_crossover_depth, overhead_ratio
from membench.theory.decay import SimilarityDecayModel
from membench.theory.recovery import RecoveryModel

# Include the negative control so the 4-arm ablation populates (brief_graph_3hop is
# already in DEFAULT_ARMS; it is the variant that reaches a depth-3 chain).
_ARMS = [*DEFAULT_ARMS, "random_context"]
_ABLATION_CELLS = {
    ("none", 1): "full_spec/none",
    ("none", 3): "stripped/none",
    ("brief_graph_3hop", 3): "stripped/brief",
    ("random_context", 3): "stripped/random",
}


@pytest.fixture(scope="module")
def rows() -> list:  # type: ignore[type-arg]
    return run_benchmark("synthetic", arms=_ARMS, per_depth=10, budget=150, seed=0)


class TestGoldenCrossover:
    def test_structured_arm_stays_flat_high(self, rows: list) -> None:  # type: ignore[type-arg]
        table = depth_crossover_table(rows)
        for d in (1, 2, 3):
            assert table[("brief_graph_3hop", d)] >= 0.9

    def test_similarity_decays_with_depth(self, rows: list) -> None:  # type: ignore[type-arg]
        table = depth_crossover_table(rows)
        dense = [table[("dense", d)] for d in (1, 2, 3)]
        assert dense[0] >= 0.8  # fine when shallow
        assert dense[2] < dense[0]  # decays
        assert dense[2] < 0.8  # collapsed by depth 3

    def test_structured_beats_similarity_at_depth(self, rows: list) -> None:  # type: ignore[type-arg]
        table = depth_crossover_table(rows)
        sims = [table[(a, 3)] for a in ("bm25", "tfidf", "dense", "hybrid_rrf", "rerank_ce")]
        assert table[("brief_graph_3hop", 3)] > max(sims)  # wins outright at d=3

    def test_none_arm_is_floor(self, rows: list) -> None:  # type: ignore[type-arg]
        assert depth_crossover_table(rows)[("none", 3)] == 0.0


class TestGoldenTablesAndReport:
    def test_headline_and_ablation_present(self, rows: list) -> None:  # type: ignore[type-arg]
        assert ("synthetic", "brief_graph_3hop") in headline_table(rows)
        ablation = ablation_table(rows, _ABLATION_CELLS)
        # stripped/brief recovers toward the ceiling; stripped/none and stripped/random
        # stay at the floor -- the load-bearing four-arm ablation.
        assert ablation["stripped/brief"] >= 0.9
        assert ablation["stripped/none"] == 0.0
        assert ablation["stripped/random"] < ablation["stripped/brief"]

    def test_report_renders_with_exec_summary(self, rows: list) -> None:  # type: ignore[type-arg]
        md = generate_report(rows)
        assert "Executive summary" in md and "Return on Tokens" in md


class TestGoldenTheory:
    def test_theory_predicts_a_finite_crossover(self) -> None:
        model = RecoveryModel(decay=SimilarityDecayModel(s0=0.9, rho=0.5), q=0.97)
        result = find_crossover_depth(model, overhead_ratio(0.2, 1.0))
        assert result.exists and result.d_star is not None and result.d_star >= 1
