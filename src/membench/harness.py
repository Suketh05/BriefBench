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
from membench.analysis.tables import ScoredRow, score_records
from membench.config.schema import BenchmarkConfig
from membench.datasets.dcbench import load_dcbench_tasks
from membench.datasets.longmemeval import load_longmemeval_tasks
from membench.datasets.swebench import load_swebench_tasks
from membench.datasets.synthetic import load_synthetic_tasks
from membench.manifest import RunManifest, capture_manifest
from membench.retrieval import builtin  # noqa: F401  (register all arms)
from membench.types import SpecVariant, Task

__all__ = [
    "DEFAULT_ARMS",
    "SweepResult",
    "load_rows",
    "load_tasks",
    "run_benchmark",
    "run_from_config",
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
) -> list[ScoredRow]:
    """Run every arm over a dataset's tasks and return the scored rows."""
    tasks = load_tasks(dataset, per_depth=per_depth, seed=seed)
    tasks_by_id = {t.task_id: t for t in tasks}
    config = RunConfig(budget_tokens=budget)
    records = []
    for arm in arms:
        for task in tasks:
            llm, memory = build_system(Combo(arm, "claude", arm), offline=offline)
            records.extend(run_task(task, memory, llm, config, arm_name=arm))
    return score_records(records, tasks_by_id)


@dataclass(frozen=True, slots=True)
class SweepResult:
    """The outcome of a full config-driven sweep: scored rows, manifest, and any skips."""

    rows: list[ScoredRow]
    manifest: RunManifest
    skipped: dict[str, str]  # arm -> reason it was skipped (e.g. missing competitor package)


def run_from_config(config: BenchmarkConfig, *, created_at: str | None = None) -> SweepResult:
    """Run a full sweep described by a validated config, across datasets and models.

    Uses the per-dataset budget (the fairness lock) for every arm and model. An arm
    that cannot be built or run (e.g. a competitor whose package is not installed in
    this environment) is skipped with a recorded reason rather than aborting the sweep,
    so a partial environment still yields the arms it can run. Captures a
    reproducibility manifest of the run.
    """
    tasks_by_id: dict[str, Task] = {}
    records = []
    skipped: dict[str, str] = {}
    for dataset in config.datasets:
        budget = config.budget_for(dataset)
        tasks = load_tasks(dataset, per_depth=config.per_depth, seed=config.seed)
        for task in tasks:
            tasks_by_id[task.task_id] = task
        run_cfg = RunConfig(budget_tokens=budget, retry_policy=config.retry_policy)
        for arm in config.arms:
            for model in config.models:
                arm_records = []
                reason: str | None = None
                for task in tasks:
                    # A fresh arm per task: memory arms accumulate state, so one task's
                    # corpus must never leak into another's retrieval.
                    try:
                        llm, memory = build_system(Combo(arm, model, arm), offline=config.offline)
                        arm_records.extend(run_task(task, memory, llm, run_cfg, arm_name=arm))
                    except (RuntimeError, NotImplementedError) as exc:
                        # Arm unavailable in this environment (e.g. uninstalled
                        # competitor, or a live backend offline) -- skip it, don't abort.
                        reason = str(exc)
                        break
                if reason is not None:
                    skipped[arm] = reason
                else:
                    records.extend(arm_records)
    manifest = capture_manifest(config.model_dump(), created_at=created_at)
    return SweepResult(rows=score_records(records, tasks_by_id), manifest=manifest, skipped=skipped)


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
