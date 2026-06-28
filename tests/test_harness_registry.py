"""Tests for ad-hoc (unregistered) arm specs accepted by ``run_benchmark``.

An arm may be supplied as a registry name (the legacy path), a ready
``MemorySystem`` instance, or a ``(name, factory)`` pair. These tests pin the two
new paths and a hand-computed retrieval golden, leaving the registry-name path
(covered by ``test_harness_cli``) untouched.
"""

from __future__ import annotations

from collections.abc import Iterable

from membench.analysis.tables import ScoredRow
from membench.harness import run_benchmark
from membench.retrieval.base import MemorySystem
from membench.types import MemoryItem, RetrievedContext


class _SilentMemory(MemorySystem):
    """A trivial unregistered arm that ingests its corpus but retrieves nothing.

    Retrieving an empty context is the cleanest hand-checkable case: for every
    synthetic task the gold set is exactly one governing decision (non-empty), so
    the standard information-retrieval definitions give an exact golden,

        recall@k    = |R ∩ G| / |G| = |∅ ∩ G| / 1 = 0.0
        precision@k = 0.0   (nothing retrieved while gold is non-empty)
        chain_recovered = (G ⊆ R) = (G ⊆ ∅) = False

    independent of the stub model's response text. ``written`` lets a test prove the
    factory path builds a *fresh* instance per task (no state leak).
    """

    def __init__(self) -> None:
        self.written = 0

    def write(self, items: Iterable[MemoryItem]) -> None:
        self.written += sum(1 for _ in items)

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        del query, budget_tokens
        return RetrievedContext.empty()


def _assert_well_formed(rows: list[ScoredRow], expected_arm: str) -> None:
    assert rows, "expected at least one scored row"
    for row in rows:
        assert isinstance(row, ScoredRow)
        assert row.arm == expected_arm
        assert row.dataset == "synthetic"
        assert row.depth >= 1
        assert 0.0 <= row.compliance_rate <= 1.0


def test_run_benchmark_accepts_arm_instance() -> None:
    """An ad-hoc ``MemorySystem`` instance runs and yields well-formed rows."""
    arm = _SilentMemory()
    rows = run_benchmark("synthetic", arms=[arm], per_depth=3, budget=150, seed=0)

    # Name is derived from the instance's class.
    _assert_well_formed(rows, expected_arm="_SilentMemory")

    # Hand-computed golden: empty retrieval over non-empty gold.
    assert all(row.recall == 0.0 for row in rows)
    assert all(row.precision == 0.0 for row in rows)
    assert all(row.chain_recovered is False for row in rows)

    # Deep-copied fresh per task: the caller's original is never written to.
    assert arm.written == 0


def test_run_benchmark_accepts_name_factory_pair() -> None:
    """A ``(name, factory)`` pair labels rows with the given name and runs fresh."""
    instances: list[_SilentMemory] = []

    def factory() -> _SilentMemory:
        made = _SilentMemory()
        instances.append(made)
        return made

    rows = run_benchmark(
        "synthetic", arms=[("custom_silent", factory)], per_depth=3, budget=150, seed=0
    )

    _assert_well_formed(rows, expected_arm="custom_silent")
    assert all(row.recall == 0.0 for row in rows)
    assert all(row.chain_recovered is False for row in rows)

    # One fresh instance built per task (state must not leak between tasks).
    n_tasks = len({(row.depth, row.spec_variant) for row in rows})
    assert n_tasks >= 1
    assert len(instances) == len(rows)
    assert all(made.written > 0 for made in instances)


def test_string_and_instance_paths_agree() -> None:
    """The derived-name instance path matches the registry ``none`` control.

    ``none`` is the registered empty-retrieval control, so a hand-rolled silent arm
    must reproduce its retrieval scores exactly -- evidence the new path is additive
    and behaviour-preserving.
    """
    reg_rows = run_benchmark("synthetic", arms=["none"], per_depth=3, budget=150, seed=0)
    adhoc_rows = run_benchmark("synthetic", arms=[_SilentMemory()], per_depth=3, budget=150, seed=0)

    assert len(reg_rows) == len(adhoc_rows)
    for reg, adhoc in zip(reg_rows, adhoc_rows, strict=True):
        assert reg.recall == adhoc.recall == 0.0
        assert reg.precision == adhoc.precision == 0.0
        assert reg.chain_recovered == adhoc.chain_recovered is False
