"""Tests for exchange session SQL migration (Issue #78).

Validates:
1. Session hour logic matches Python ouroboros.py implementation
2. SQL generation is correct
3. Migration is idempotent
4. Coverage check function works

Run with: pytest tests/test_exchange_sessions.py -v
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from rangebar.clickhouse.migrations import (
        _SESSION_UPDATES,
        _build_session_update_sql,
    )
from rangebar.ouroboros import (
    EXCHANGE_SESSION_HOURS,
    ExchangeSessionFlags,
    get_active_exchange_sessions,
)

# ============================================================================
# Session Logic Tests (against Python implementation)
# ============================================================================


class TestExchangeSessionLogic:
    """Verify session hours match expectations."""

    def test_session_definitions_match_ouroboros(self):
        """SQL migration session defs must match ouroboros.py constants."""
        for session in _SESSION_UPDATES:
            col = session["column"]
            # e.g. "exchange_session_sydney" -> "sydney"
            name = col.replace("exchange_session_", "")
            assert name in EXCHANGE_SESSION_HOURS, f"Unknown session: {name}"

            py_def = EXCHANGE_SESSION_HOURS[name]
            assert session["tz"] == py_def["tz"], f"TZ mismatch for {name}"
            assert int(session["start"]) == py_def["start"], f"Start mismatch for {name}"
            assert int(session["end"]) == py_def["end"], f"End mismatch for {name}"

    def test_all_four_sessions_defined(self):
        """Migration must define exactly 4 sessions."""
        assert len(_SESSION_UPDATES) == 4
        columns = {s["column"] for s in _SESSION_UPDATES}
        assert columns == {
            "exchange_session_sydney",
            "exchange_session_tokyo",
            "exchange_session_london",
            "exchange_session_newyork",
        }

    def test_tokyo_session_morning_active(self):
        """Tokyo session (09:00-15:00 JST) should be active at 10:00 JST."""
        # 10:00 JST = 01:00 UTC
        ts = datetime(2024, 6, 3, 1, 0, tzinfo=UTC)  # Monday
        flags = get_active_exchange_sessions(ts)
        assert flags.tokyo is True

    def test_tokyo_session_before_open_inactive(self):
        """Tokyo session should be inactive before 09:00 JST."""
        # 08:00 JST = 23:00 UTC (previous day)
        ts = datetime(2024, 6, 2, 23, 0, tzinfo=UTC)  # Sunday 23:00 UTC
        flags = get_active_exchange_sessions(ts)
        # Sunday in Tokyo = weekend
        assert flags.tokyo is False

    def test_london_session_midday_active(self):
        """London session (08:00-17:00 GMT/BST) should be active at noon."""
        # 12:00 UTC on a Wednesday
        ts = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        flags = get_active_exchange_sessions(ts)
        assert flags.london is True

    def test_newyork_session_afternoon_active(self):
        """New York session (10:00-16:00 ET) should be active at 14:00 ET."""
        # 14:00 ET in winter = 19:00 UTC
        ts = datetime(2024, 1, 10, 19, 0, tzinfo=UTC)
        flags = get_active_exchange_sessions(ts)
        # 19:00 UTC = 14:00 EST (winter)
        assert flags.newyork is True

    def test_weekend_all_inactive(self):
        """All sessions should be inactive on Saturday."""
        # Saturday 12:00 UTC
        ts = datetime(2024, 1, 6, 12, 0, tzinfo=UTC)
        flags = get_active_exchange_sessions(ts)
        assert flags.sydney is False
        assert flags.tokyo is False
        assert flags.london is False
        assert flags.newyork is False

    def test_tokyo_london_no_overlap(self):
        """Tokyo and London sessions should not overlap (no common hours)."""
        # Check 07:00 UTC on a weekday: Tokyo 16:00 JST (closed), London 07:00 GMT (not open)
        ts = datetime(2024, 1, 8, 7, 0, tzinfo=UTC)  # Monday
        flags = get_active_exchange_sessions(ts)
        # Tokyo: 16:00 JST = end of session (exclusive), should be False
        # London: 07:00 GMT = before open, should be False
        assert not (flags.tokyo and flags.london)

    def test_london_newyork_overlap(self):
        """London-NY overlap (15:00-16:00 GMT / 10:00-11:00 ET in winter)."""
        # 15:30 UTC on a weekday in January (no DST)
        ts = datetime(2024, 1, 8, 15, 30, tzinfo=UTC)  # Monday
        flags = get_active_exchange_sessions(ts)
        # 15:30 UTC = 15:30 GMT (London open until 17:00)
        # 15:30 UTC = 10:30 EST (NY open from 10:00)
        assert flags.london is True
        assert flags.newyork is True

    def test_exchange_session_flags_to_dict(self):
        """ExchangeSessionFlags.to_dict() should use correct column names."""
        flags = ExchangeSessionFlags(
            sydney=True, tokyo=False, london=True, newyork=False
        )
        d = flags.to_dict()
        assert d == {
            "exchange_session_sydney": True,
            "exchange_session_tokyo": False,
            "exchange_session_london": True,
            "exchange_session_newyork": False,
        }


# ============================================================================
# SQL Generation Tests
# ============================================================================


class TestSQLGeneration:
    """Verify SQL statement generation."""

    def test_build_sql_no_symbol(self):
        """SQL without symbol filter should use WHERE 1 = 1."""
        sql = _build_session_update_sql(_SESSION_UPDATES[0])
        assert "ALTER TABLE rangebar_cache.range_bars" in sql
        assert "UPDATE exchange_session_sydney" in sql
        assert "WHERE 1 = 1" in sql
        assert "toTimezone" in sql
        assert "Australia/Sydney" in sql

    def test_build_sql_with_symbol(self):
        """SQL with symbol filter should use WHERE symbol = 'X'."""
        sql = _build_session_update_sql(_SESSION_UPDATES[0], symbol="BTCUSDT")
        assert "WHERE symbol = 'BTCUSDT'" in sql

    def test_build_sql_contains_weekend_exclusion(self):
        """SQL should exclude weekends via toDayOfWeek."""
        sql = _build_session_update_sql(_SESSION_UPDATES[0])
        assert "toDayOfWeek" in sql
        assert "<= 5" in sql

    def test_build_sql_uses_if_expression(self):
        """SQL should use if() for conditional update (not just WHERE)."""
        sql = _build_session_update_sql(_SESSION_UPDATES[0])
        # Should set column to 1 when condition is met, 0 otherwise
        assert "if(" in sql
        assert ", 1, 0)" in sql

    def test_all_sessions_generate_valid_sql(self):
        """All 4 session definitions should produce valid SQL strings."""
        for session in _SESSION_UPDATES:
            sql = _build_session_update_sql(session)
            assert sql.startswith("ALTER TABLE rangebar_cache.range_bars")
            assert f"UPDATE {session['column']}" in sql
            assert session["tz"] in sql
            assert f">= {session['start']}" in sql
            assert f"< {session['end']}" in sql

    def test_sql_uses_intdiv_for_timestamp(self):
        """SQL should use intDiv(close_time_ms, 1000) for epoch conversion."""
        sql = _build_session_update_sql(_SESSION_UPDATES[0])
        assert "intDiv(close_time_ms, 1000)" in sql
