"""Multi-turn sessions with a per-turn token ledger (paper Results VII, ``sec:tokecon``).

The paper's token-economics section evaluates *sessions*, not single completions:
a backend coding model is billed on every turn it runs, the context layer is
injected once on turn 1, and the entire cost difference between configurations is
how many tokens they burn before the task resolves — dominated by *turn count*
(paper ``sec:tokecon``; per-turn ledger table ``tab:tok_agent_context_gpt55``).
This module supplies the session primitives that section's tables are built from:

* :class:`TurnRecord` / :class:`Session` — one turn's prompt/completion token
  spend and the whole multi-turn trajectory, with the convergence turn (the turn
  at which the task resolved) recorded explicitly.
* :func:`run_session` — a deterministic multi-turn driver layered over the
  existing single-turn offline runner (:func:`membench.agents.runner.run_task`):
  turn 1 writes the corpus, retrieves under the fairness-locked budget, and
  prompts; later turns continue the conversation without re-injecting context.
* Ledger helpers — per-turn rows with cumulative token/dollar columns, and the
  paper's presentation collapse to Turn 1 / Turn 2 / Turn 3 / Turn 4+ columns
  (``tab:tok_agent_context_gpt55``).
* Matchup helpers — the session-token win-rate rule of
  ``tab:tok_context_winrate`` (winner = strictly fewer session tokens) and the
  competitor-saving margin of ``tab:tok_context_brief_losses``.

Everything here is offline and deterministic under the stub backend; no paper
number is produced by this module — the published values are asserted as goldens
in ``tests/test_agents_session.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "TurnRecord",
]


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One turn of a multi-turn session: token spend, cost, and resolution flag.

    Mirrors one cell row of the paper's per-turn ledger
    (``tab:tok_agent_context_gpt55``): each turn is a separate billed call to the
    *same* backend model, so the session cost is the sum of these rows.

    Parameters
    ----------
    turn
        1-based turn index within the session.
    prompt_tokens, completion_tokens
        Token counts for this turn as reported by the backend (per-call tokens
        are a reported control, never an optimisation target — same convention
        as :class:`membench.agents.runner.AttemptRecord`).
    dollars
        Cost of this turn under the pricing table.
    response_text
        The model output for this turn (kept so a resolution predicate and the
        compliance scorer can be applied after the fact).
    resolved
        Whether the resolution predicate accepted this turn's output. A session
        stops at the first resolved turn, so this may be ``True`` only on the
        final turn of a session (enforced by :class:`Session`).
    """

    turn: int
    prompt_tokens: int
    completion_tokens: int
    dollars: float
    response_text: str
    resolved: bool = False

    def __post_init__(self) -> None:
        if self.turn < 1:
            raise ValueError(f"turn must be >= 1, got {self.turn}")
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.dollars < 0:
            raise ValueError("dollars must be non-negative")

    @property
    def turn_tokens(self) -> int:
        """Total tokens billed for this turn (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens
