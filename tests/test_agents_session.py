"""Tests for multi-turn sessions and the per-turn token ledger (paper sec:tokecon).

Golden values are transcribed VERBATIM from the paper's token-economics tables
(tab:tok_agent_context_gpt55, tab:tok_context_winrate,
tab:tok_context_by_competitor, tab:tok_agent_economics, tab:tok_pareto,
tab:tok_context_brief_losses) — never produced by running this repo's code.
Where the paper prints a rounded decimal, the assertion tolerance is half an
ulp of the printed precision (0.51 * 10^-dp), so a reader can reproduce every
check with a hand calculator.
"""

from __future__ import annotations

import pytest

from membench.agents.llm.pricing import blended_session_dollars
from membench.agents.session import PaperLedgerRow

# ---------------------------------------------------------------------------
# tab:tok_agent_context_gpt55 — "Multi-turn collapse, one fixed backend model
# (GPT-5.5)". Each row: configuration, Turn 1, Turn 2, Turn 3, Turn 4+,
# session tokens, printed USD, printed USD decimal places, outcome turn count,
# resolved?  All numbers verbatim from the paper table.
# ---------------------------------------------------------------------------
GPT55_LEDGER_ROWS: list[tuple[str, int, int, int, int | None, int, float, int, int, bool]] = [
    ("alone", 1403, 698, 601, 1461, 4163, 0.05, 2, 6, False),
    ("alone (best case)", 1420, 715, 612, 939, 3686, 0.0442, 4, 5, True),
    ("Brief context", 1598, 203, 34, None, 1835, 0.022, 3, 3, True),
    ("Brief context (slow task)", 1598, 412, 118, 44, 2172, 0.0261, 4, 4, True),
    ("Mem0 context", 1712, 684, 598, 1743, 4737, 0.0568, 4, 7, True),
    ("Mem0 context (worst)", 1708, 691, 605, 2166, 5170, 0.062, 3, 8, False),
    ("ContextQ context", 1648, 612, 541, 926, 3727, 0.0447, 4, 5, False),
    ("Zep context", 1576, 448, 382, 318, 2724, 0.0327, 4, 4, True),
]


class TestGpt55PaperLedgerGoldens:
    """Every published GPT-5.5 ledger row satisfies the ledger arithmetic."""

    @pytest.mark.parametrize(
        ("config", "t1", "t2", "t3", "t4p", "session"),
        [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in GPT55_LEDGER_ROWS],
    )
    def test_columns_sum_to_session_tokens(
        self, config: str, t1: int, t2: int, t3: int, t4p: int | None, session: int
    ) -> None:
        # Constructing PaperLedgerRow *is* the assertion: its validator raises
        # unless T1 + T2 + T3 + T4+ == session tokens. E.g. the first "alone"
        # row: 1403 + 698 + 601 + 1461 = 4163 (paper tab:tok_agent_context_gpt55).
        row = PaperLedgerRow(t1, t2, t3, t4p, session)
        assert row.session_tokens == session

    @pytest.mark.parametrize(
        ("config", "session", "usd", "dp"),
        [(r[0], r[5], r[6], r[7]) for r in GPT55_LEDGER_ROWS],
    )
    def test_usd_column_is_flat_12_dollars_per_mtok(
        self, config: str, session: int, usd: float, dp: int
    ) -> None:
        # The caption fixes the backend model and "its per-token price ...
        # identical across rows"; the constant that reproduces every printed
        # USD cell is $12.00 per million session tokens. Worked example, first
        # row: 4163 tok x 12e-6 = $0.049956 -> printed $0.05.
        computed = blended_session_dollars("gpt-5.5", session)
        assert abs(computed - usd) <= 0.51 * 10.0**-dp

    def test_turn4_plus_column_is_present_iff_session_ran_past_turn_3(self) -> None:
        # The 3-turn "Brief context" row prints "---" for Turn 4+; every row
        # whose outcome column reports >= 4 turns has a Turn 4+ entry.
        for _config, _t1, _t2, _t3, t4p, _sess, _usd, _dp, turns, _res in GPT55_LEDGER_ROWS:
            assert (t4p is None) == (turns <= 3)

    def test_brief_collapse_headline(self) -> None:
        # sec:tokecon: with Brief's context the session "collapses" — it is the
        # strictly cheapest published row (1835 session tokens), resolves, and
        # its convergence turn is 3 ("3 turns, resolved").
        brief = GPT55_LEDGER_ROWS[2]
        assert brief[5] == min(row[5] for row in GPT55_LEDGER_ROWS)
        assert all(row[5] > brief[5] for row in GPT55_LEDGER_ROWS if row[0] != brief[0])
        assert brief[8] == 3 and brief[9] is True
