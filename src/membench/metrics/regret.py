r"""Normalized regret of each arm against the per-task oracle.

Regret measures how far an arm falls short of the best decision available in
hindsight (the *oracle*, the best fixed action per task), the canonical quantity in
online learning -- see Cesa-Bianchi & Lugosi, *Prediction, Learning, and Games*
(Cambridge University Press, 2006), ch. 2-3. Raw regret is unbounded and not
comparable across tasks of different difficulty, so we min-max normalise it onto
``[0, 1]`` using a fixed floor baseline (e.g. the ``none`` control). This is the same
range-normalisation used for benchmark-normalised scores in reinforcement learning,
e.g. the human-normalised score of Mnih et al., "Human-level control through deep
reinforcement learning", *Nature* 518 (2015), where a raw score is rescaled by
``(score - random) / (human - random)``.

For one task, with per-arm achieved score ``a``, oracle ``best`` (the max over arms)
and floor ``worst`` (a chosen baseline arm, or the min over arms), the normalized
regret of that arm is

.. math::

    R = \operatorname{clip}\!\left(
        \frac{\text{best} - a}{\text{best} - \text{worst}},\; 0,\; 1
    \right).

By construction the oracle arm has regret ``0``, the floor baseline has regret ``1``,
and every value lies in ``[0, 1]``. When the task is degenerate (``best == worst``,
i.e. no arm beats the floor) there is no spread to normalise and the regret is ``0``.
Per-arm regret is the mean of the per-task regrets, so a lower number is better.

Inputs follow the metrics-package conventions: the scalar core operates on plain
floats, while :func:`regret_by_arm` operates on a sequence of
:class:`~membench.analysis.tables.ScoredRow` (the canonical scored row), grouping
rows into tasks by an explicit key.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

from membench.analysis.tables import ScoredRow

__all__ = [
    "ArmRegret",
    "default_score",
    "default_task_key",
    "normalized_regret",
    "regret_by_arm",
    "regret_per_task",
]


def normalized_regret(achieved: float, best_achievable: float, worst_baseline: float) -> float:
    r"""Min-max normalise one arm's regret against the oracle onto ``[0, 1]``.

    Parameters
    ----------
    achieved : float
        The score the arm actually obtained on the task.
    best_achievable : float
        The per-task oracle: the best score obtained by any arm (in hindsight).
    worst_baseline : float
        The floor score that anchors regret ``1`` (e.g. the ``none`` control).

    Returns
    -------
    float
        ``(best_achievable - achieved) / (best_achievable - worst_baseline)``,
        clipped to ``[0, 1]``. Returns ``0.0`` when ``best_achievable`` does not
        exceed ``worst_baseline`` (a degenerate task with no spread to normalise).
    """
    spread = best_achievable - worst_baseline
    if spread <= 0.0:
        return 0.0
    regret = (best_achievable - achieved) / spread
    return min(1.0, max(0.0, regret))


def regret_per_task(
    scores_by_arm: Mapping[str, float],
    *,
    baseline_arm: str | None = None,
) -> dict[str, float]:
    """Compute each arm's normalized regret for a single task.

    Parameters
    ----------
    scores_by_arm : Mapping[str, float]
        The achieved score of every arm on this one task.
    baseline_arm : str or None, optional
        Arm whose score is the floor (regret ``1``). When ``None`` or absent from
        ``scores_by_arm``, the minimum score over the arms is used as the floor.

    Returns
    -------
    dict[str, float]
        Mapping from arm name to its normalized regret in ``[0, 1]``.
    """
    if not scores_by_arm:
        return {}
    values = scores_by_arm.values()
    best = max(values)
    if baseline_arm is not None and baseline_arm in scores_by_arm:
        worst = scores_by_arm[baseline_arm]
    else:
        worst = min(values)
    return {arm: normalized_regret(score, best, worst) for arm, score in scores_by_arm.items()}


@dataclass(frozen=True, slots=True)
class ArmRegret:
    """An arm's mean normalized regret over the tasks it was scored on."""

    arm: str
    regret: float
    n_tasks: int


def default_score(row: ScoredRow) -> float:
    """Return the score a row is judged on for regret (the compliance rate)."""
    return row.compliance_rate


def default_task_key(row: ScoredRow) -> Hashable:
    """Return the key that groups rows into one comparable task across arms.

    Two rows belong to the same task (so their arms compete for the oracle) when they
    share dataset, model, depth and spec variant -- everything that identifies a task
    instance except the arm under test.
    """
    return (row.dataset, row.model, row.depth, row.spec_variant)


def regret_by_arm(
    rows: Sequence[ScoredRow],
    *,
    score: Callable[[ScoredRow], float] = default_score,
    baseline_arm: str | None = "none",
    task_key: Callable[[ScoredRow], Hashable] = default_task_key,
) -> dict[str, ArmRegret]:
    """Aggregate per-task normalized regret into one number per arm.

    Rows are grouped into tasks by ``task_key``; within each task the oracle is the
    max score over arms and the floor is ``baseline_arm`` (or the min over arms). An
    arm's reported regret is the mean of its per-task regrets.

    Parameters
    ----------
    rows : Sequence[ScoredRow]
        The scored rows to aggregate (typically a benchmark sweep).
    score : Callable[[ScoredRow], float], optional
        Extracts the scalar an arm is judged on (default: ``compliance_rate``).
    baseline_arm : str or None, optional
        Arm used as the per-task floor (default ``"none"``); falls back to the min
        score over arms on tasks where it is absent.
    task_key : Callable[[ScoredRow], Hashable], optional
        Groups rows into comparable tasks (default: dataset, model, depth, spec
        variant).

    Returns
    -------
    dict[str, ArmRegret]
        Mapping from arm name to its mean normalized regret and the task count it
        was averaged over.

    Raises
    ------
    ValueError
        If two rows for the same arm fall in the same task (an ambiguous oracle).
    """
    grouped: dict[Hashable, dict[str, float]] = {}
    for row in rows:
        task = grouped.setdefault(task_key(row), {})
        if row.arm in task:
            raise ValueError(
                f"duplicate arm {row.arm!r} in task {task_key(row)!r}: ambiguous oracle"
            )
        task[row.arm] = score(row)

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for scores_by_arm in grouped.values():
        for arm, regret in regret_per_task(scores_by_arm, baseline_arm=baseline_arm).items():
            totals[arm] = totals.get(arm, 0.0) + regret
            counts[arm] = counts.get(arm, 0) + 1

    return {
        arm: ArmRegret(arm=arm, regret=totals[arm] / counts[arm], n_tasks=counts[arm])
        for arm in totals
    }
