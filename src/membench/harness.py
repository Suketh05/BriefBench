"""End-to-end orchestration: run datasets x arms x model, score, persist.

Ties the pieces together into one call: load a dataset's tasks, run each memory arm
against the (offline-stub by default) model under the fixed budget, score every
attempt, and return the canonical scored rows. Helpers persist those rows to JSONL
so the tables/report/figures commands can consume a finished run.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from membench.agents.combos import Combo, build_system
from membench.agents.runner import RunConfig, run_task
from membench.analysis.tables import ABLATION_DATASETS, ScoredRow, score_records
from membench.config.schema import BenchmarkConfig
from membench.datasets.dcbench import load_dcbench_tasks
from membench.datasets.longmemeval import load_longmemeval_tasks
from membench.datasets.swebench import load_swebench_tasks
from membench.datasets.synthetic import load_synthetic_tasks
from membench.manifest import RunManifest, capture_manifest
from membench.retrieval import builtin  # noqa: F401  (register all arms)
from membench.types import SpecVariant, Task

__all__ = [
    "ABLATION_ARMS",
    "DEFAULT_ARMS",
    "SweepResult",
    "load_rows",
    "load_tasks",
    "run_benchmark",
    "run_from_config",
    "run_sweep",
    "save_rows",
]

# Representative, offline-runnable arms (competitor adapters need their own envs).
DEFAULT_ARMS: tuple[str, ...] = (
    "none",
    "bm25",
    "tfidf",
    "dense",
    "hybrid_rrf",
    "rerank_ce",
    "raptor",
    "brief_graph_3hop",
)

# The dedicated four-arm ablation set, run on ABLATION_DATASETS (dcbench + swebench)
# only and kept separate from the headline DEFAULT_ARMS sweep. random_context is the
# budget-matched negative control (same retrieval budget as the structured arm, random
# nodes) -- it is intentionally NOT in DEFAULT_ARMS, so the sweep must add it here for
# the four ablation cells to render. none and brief_graph_3hop overlap with the headline
# sweep and are de-duplicated by (dataset, arm) so no arm is executed twice.
ABLATION_ARMS: tuple[str, ...] = ("none", "brief_graph_3hop", "random_context")


def load_tasks(dataset: str, *, per_depth: int = 10, seed: int = 0) -> list[Task]:
    """Load a dataset's tasks across its depth range."""
    if dataset == "synthetic":
        return load_synthetic_tasks(depths=(1, 2, 3), per_depth=per_depth, seed=seed)
    if dataset == "dcbench":
        return (
            load_dcbench_tasks(SpecVariant.FULL, 1)
            + load_dcbench_tasks(SpecVariant.STRIPPED, 2)
            + load_dcbench_tasks(SpecVariant.STRIPPED, 3)
        )
    if dataset == "swebench":
        return (
            load_swebench_tasks(SpecVariant.FULL, 1)
            + load_swebench_tasks(SpecVariant.STRIPPED, 2)
            + load_swebench_tasks(SpecVariant.STRIPPED, 3)
        )
    if dataset == "longmemeval":
        return load_longmemeval_tasks()
    raise ValueError(f"unknown dataset {dataset!r}")


def run_benchmark(
    dataset: str = "synthetic",
    arms: Sequence[str] = DEFAULT_ARMS,
    budget: int = 150,
    *,
    per_depth: int = 10,
    seed: int = 0,
    offline: bool = True,
    model: str = "claude",
    retry_policy: str = "single_shot",
) -> list[ScoredRow]:
    """Run every arm over a dataset's tasks and return the scored rows.

    The single place arms are built and scored: a fresh arm is constructed per task
    (memory arms accumulate state, so one task's corpus must never leak into another's),
    every arm runs at the same ``budget`` and ``model`` (the fairness lock), and the
    attempts are scored into canonical rows. :func:`run_sweep` and :func:`run_from_config`
    both delegate here so there is one execution path.
    """
    tasks = load_tasks(dataset, per_depth=per_depth, seed=seed)
    tasks_by_id = {t.task_id: t for t in tasks}
    config = RunConfig(budget_tokens=budget, retry_policy=retry_policy)
    records = []
    for arm in arms:
        for task in tasks:
            llm, memory = build_system(Combo(arm, model, arm), offline=offline)
            records.extend(run_task(task, memory, llm, config, arm_name=arm))
    return score_records(records, tasks_by_id)


def run_sweep(
    datasets: Sequence[str] = ("synthetic", "dcbench", "swebench"),
    budget: int = 150,
    *,
    per_depth: int = 10,
    seed: int = 0,
    offline: bool = True,
) -> list[ScoredRow]:
    """Run the headline sweep plus the dedicated four-arm ablation, returning all rows.

    Two passes feed one results file. The headline pass runs ``DEFAULT_ARMS`` over every
    requested dataset (headline / depth-crossover / robustness / competitor tables). The
    ablation pass is a separate, dcbench+swebench-only run of :data:`ABLATION_ARMS` so the
    budget-matched ``random_context`` control -- deliberately kept out of the headline
    sweep -- is present and all four ablation cells render. Rows are de-duplicated by
    ``(dataset, arm)`` so an arm shared by both passes is executed only once.
    """
    rows: list[ScoredRow] = []
    done: set[tuple[str, str]] = set()

    def _run(dataset: str, arms: Sequence[str]) -> None:
        pending = tuple(a for a in arms if (dataset, a) not in done)
        if not pending:
            return
        rows.extend(
            run_benchmark(dataset, pending, budget, per_depth=per_depth, seed=seed, offline=offline)
        )
        done.update((dataset, a) for a in pending)

    for dataset in datasets:
        _run(dataset, DEFAULT_ARMS)
    for dataset in ABLATION_DATASETS:
        _run(dataset, ABLATION_ARMS)
    return rows


@dataclass(frozen=True, slots=True)
class SweepResult:
    """The outcome of a config-driven sweep: scored rows, manifest, and any skips."""

    rows: list[ScoredRow]
    manifest: RunManifest
    skipped: dict[str, str]  # arm -> reason it was skipped (e.g. missing competitor package)


def run_from_config(config: BenchmarkConfig, *, created_at: str | None = None) -> SweepResult:
    """Run a sweep described by a validated config, across its datasets and models.

    Delegates the per-task work to :func:`run_benchmark` (the single place arms are built
    and scored), one arm at a time, each at its dataset's budget -- the fairness lock. An
    arm that cannot be built or run in this environment (e.g. an uninstalled competitor,
    or a live backend offline) is skipped with a recorded reason rather than aborting the
    whole sweep, so a partial environment still yields the arms it can run. Captures a
    reproducibility manifest of the run.
    """
    rows: list[ScoredRow] = []
    skipped: dict[str, str] = {}
    for dataset in config.datasets:
        budget = config.budget_for(dataset)
        for arm in config.arms:
            for model in config.models:
                try:
                    rows.extend(
                        run_benchmark(
                            dataset,
                            [arm],
                            budget,
                            per_depth=config.per_depth,
                            seed=config.seed,
                            offline=config.offline,
                            model=model,
                            retry_policy=config.retry_policy,
                        )
                    )
                except (RuntimeError, NotImplementedError) as exc:
                    skipped[arm] = str(exc)
    manifest = capture_manifest(config.model_dump(), created_at=created_at)
    return SweepResult(rows=rows, manifest=manifest, skipped=skipped)


def save_rows(rows: Sequence[ScoredRow], path: str | Path) -> Path:
    """Write scored rows to a JSONL file (one row per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dataclasses.asdict(row)) + "\n")
    return out


def load_rows(path: str | Path) -> list[ScoredRow]:
    """Read scored rows back from a JSONL file."""
    rows: list[ScoredRow] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(ScoredRow(**json.loads(line)))
    return rows
