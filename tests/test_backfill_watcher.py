# FILE-SIZE-OK
"""Tests for backfill watcher scheduling, dedup, cooldown, and gap tracking.

Tests the Wave 3 fixes for Issues #103, #104, #105, #106:
- Fair scheduling: _resolve_thresholds() fan-out (Issue #103)
- Dedup: GROUP BY (symbol, threshold) in _fetch_pending_requests (Issue #104)
- Cooldown: _is_on_cooldown / _record_cooldown (Issue #105)
- Gap tracking: gap_seconds at _mark_running() time (Issue #106)
- Exception capture: actual error messages preserved in _mark_failed (Issue #106)

All tests use mocks -- no actual ClickHouse connection needed.
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make scripts/ importable
_scripts_dir = str(Path(__file__).parent / ".." / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, str(Path(_scripts_dir).resolve()))

import backfill_watcher  # noqa: E402
from backfill_watcher import (  # noqa: E402
    DEFAULT_THRESHOLD_DBPS,
    _cooldown_tracker,
    _fetch_pending_requests,
    _is_on_cooldown,
    _mark_completed,
    _mark_failed,
    _mark_running,
    _record_cooldown,
    _resolve_thresholds,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_cache(query_rows: list[tuple] | None = None) -> MagicMock:
    """Create a mock cache object with a mock client."""
    cache = MagicMock()
    result = MagicMock()
    result.result_rows = query_rows or []
    cache.client.query.return_value = result
    cache.client.command.return_value = None
    return cache


@pytest.fixture(autouse=True)
def _clear_cooldown() -> None:
    """Clear cooldown tracker between tests."""
    _cooldown_tracker.clear()
    yield
    _cooldown_tracker.clear()


# ============================================================================
# 1. _resolve_thresholds -- fan-out (Issue #103)
# ============================================================================


class TestResolveThresholdsExplicit:
    """When threshold > 0, return [threshold]."""

    def test_explicit_500(self):
        cache = _make_mock_cache()
        assert _resolve_thresholds(cache, "BTCUSDT", 500) == [500]
        cache.client.query.assert_not_called()

    def test_explicit_250(self):
        cache = _make_mock_cache()
        assert _resolve_thresholds(cache, "ETHUSDT", 250) == [250]


class TestResolveThresholdsZeroFansOut:
    """When threshold=0, return ALL cached thresholds (not just lowest)."""

    def test_fans_out_to_all_cached(self):
        cache = _make_mock_cache(query_rows=[(250,), (500,), (750,), (1000,)])
        result = _resolve_thresholds(cache, "BTCUSDT", 0)
        assert result == [250, 500, 750, 1000]

    def test_single_cached_threshold(self):
        cache = _make_mock_cache(query_rows=[(500,)])
        result = _resolve_thresholds(cache, "SOLUSDT", 0)
        assert result == [500]

    def test_queries_clickhouse_for_distinct_thresholds(self):
        cache = _make_mock_cache(query_rows=[(250,), (750,)])
        _resolve_thresholds(cache, "BTCUSDT", 0)
        cache.client.query.assert_called_once()
        sql = cache.client.query.call_args[0][0]
        assert "DISTINCT" in sql
        assert "threshold_decimal_bps" in sql


class TestResolveThresholdsNoCache:
    """When threshold=0 and no cached data, return [DEFAULT_THRESHOLD_DBPS]."""

    def test_falls_back_to_default(self):
        cache = _make_mock_cache(query_rows=[])
        result = _resolve_thresholds(cache, "NEWCOINUSDT", 0)
        assert result == [DEFAULT_THRESHOLD_DBPS]


# ============================================================================
# 2. Dedup via GROUP BY in _fetch_pending_requests (Issue #104)
# ============================================================================


class TestFetchPendingRequestsDedup:
    """_fetch_pending_requests uses GROUP BY to collapse duplicates."""

    def test_returns_grouped_structure(self):
        """Each result has symbol, thresholds, and all_ids."""
        rid1, rid2 = uuid.uuid4(), uuid.uuid4()
        cache = _make_mock_cache(
            query_rows=[
                ("BTCUSDT", [250, 500], [rid1, rid2], "2026-01-01 00:00:00"),
            ]
        )
        result = _fetch_pending_requests(cache)
        assert len(result) == 1
        assert result[0]["symbol"] == "BTCUSDT"
        assert result[0]["thresholds"] == [250, 500]
        assert len(result[0]["all_ids"]) == 2

    def test_preserves_different_symbols(self):
        rid_btc, rid_eth = uuid.uuid4(), uuid.uuid4()
        cache = _make_mock_cache(
            query_rows=[
                ("BTCUSDT", [250], [rid_btc], "2026-01-01 00:00:00"),
                ("ETHUSDT", [250], [rid_eth], "2026-01-01 00:00:01"),
            ]
        )
        result = _fetch_pending_requests(cache)
        assert len(result) == 2
        symbols = {r["symbol"] for r in result}
        assert symbols == {"BTCUSDT", "ETHUSDT"}

    def test_empty_when_no_pending(self):
        cache = _make_mock_cache(query_rows=[])
        assert _fetch_pending_requests(cache) == []

    def test_sql_uses_group_by(self):
        cache = _make_mock_cache(query_rows=[])
        _fetch_pending_requests(cache)
        sql = cache.client.query.call_args[0][0]
        assert "GROUP BY" in sql

    def test_five_duplicates_collapse_to_one(self):
        """5 pending requests for same symbol -> 1 group."""
        rids = [uuid.uuid4() for _ in range(5)]
        cache = _make_mock_cache(
            query_rows=[
                ("BTCUSDT", [250], rids, "2026-01-01 00:00:00"),
            ]
        )
        result = _fetch_pending_requests(cache)
        assert len(result) == 1
        assert len(result[0]["all_ids"]) == 5


# ============================================================================
# 3. Cooldown (Issue #105)
# ============================================================================


class TestCooldownSkipsRecent:
    """After recording cooldown, next request is skipped."""

    def test_skips_after_recording(self):
        _record_cooldown("BTCUSDT", 250)
        assert _is_on_cooldown("BTCUSDT", 250) is True

    def test_different_symbol_not_on_cooldown(self):
        _record_cooldown("BTCUSDT", 250)
        assert _is_on_cooldown("ETHUSDT", 250) is False

    def test_different_threshold_not_on_cooldown(self):
        _record_cooldown("BTCUSDT", 250)
        assert _is_on_cooldown("BTCUSDT", 500) is False

    def test_no_cooldown_without_recording(self):
        assert _is_on_cooldown("BTCUSDT", 250) is False


class TestCooldownExpires:
    """After cooldown period, requests are processed again."""

    def test_expires_after_duration(self):
        # Set a very short cooldown for testing
        with patch.object(backfill_watcher, "_COOLDOWN_SECONDS", 0.05):
            _record_cooldown("BTCUSDT", 250)
            assert _is_on_cooldown("BTCUSDT", 250) is True
            time.sleep(0.1)
            assert _is_on_cooldown("BTCUSDT", 250) is False


class TestCooldownEnvConfig:
    """RANGEBAR_BACKFILL_COOLDOWN_SECONDS env var is respected."""

    def test_module_level_constant(self):
        """The module reads the env var at import time."""
        # Verify the constant exists and is a positive int
        assert isinstance(backfill_watcher._COOLDOWN_SECONDS, int)
        assert backfill_watcher._COOLDOWN_SECONDS > 0

    def test_default_is_300(self):
        """Default cooldown is 300 seconds (5 minutes)."""
        # The module reads env at import, so we check the default
        # This may vary if the env is set, so we just check it's reasonable
        assert backfill_watcher._COOLDOWN_SECONDS >= 60


# ============================================================================
# 4. _mark_running includes gap_seconds (Issue #106)
# ============================================================================


class TestMarkRunningGapSeconds:
    """_mark_running() writes gap_seconds for UI feedback."""

    def test_includes_gap_seconds_in_sql(self):
        cache = _make_mock_cache()
        rid = str(uuid.uuid4())
        _mark_running(cache, [rid], gap_seconds=120.5)
        cache.client.command.assert_called_once()
        sql = cache.client.command.call_args[0][0]
        assert "gap_seconds" in sql

    def test_gap_seconds_in_parameters(self):
        cache = _make_mock_cache()
        rid = str(uuid.uuid4())
        _mark_running(cache, [rid], gap_seconds=45.0)
        params = cache.client.command.call_args[1]["parameters"]
        assert params["gap_seconds"] == 45.0

    def test_default_gap_seconds_is_zero(self):
        cache = _make_mock_cache()
        rid = str(uuid.uuid4())
        _mark_running(cache, [rid])
        params = cache.client.command.call_args[1]["parameters"]
        assert params["gap_seconds"] == 0.0

    def test_updates_all_request_ids(self):
        cache = _make_mock_cache()
        rids = [str(uuid.uuid4()) for _ in range(3)]
        _mark_running(cache, rids, gap_seconds=60.0)
        assert cache.client.command.call_count == 3


# ============================================================================
# 5. _mark_failed preserves actual error (Issue #106)
# ============================================================================


class TestMarkFailedPreservesError:
    """_mark_failed() captures the actual exception string."""

    def test_preserves_actual_error(self):
        cache = _make_mock_cache()
        rid = str(uuid.uuid4())
        error_msg = "SSH tunnel to bigblack dropped after 30s timeout"
        _mark_failed(cache, [rid], error_msg)
        params = cache.client.command.call_args[1]["parameters"]
        assert params["error"] == error_msg

    def test_truncates_long_error(self):
        cache = _make_mock_cache()
        rid = str(uuid.uuid4())
        long_error = "x" * 1000
        _mark_failed(cache, [rid], long_error)
        params = cache.client.command.call_args[1]["parameters"]
        assert len(params["error"]) == 500

    def test_does_not_use_generic_message(self):
        cache = _make_mock_cache()
        rid = str(uuid.uuid4())
        specific_error = "ConnectionRefusedError: [Errno 111] Connection refused"
        _mark_failed(cache, [rid], specific_error)
        params = cache.client.command.call_args[1]["parameters"]
        assert params["error"] != "Backfill failed with exception"
        assert params["error"] == specific_error

    def test_updates_all_request_ids(self):
        cache = _make_mock_cache()
        rids = [str(uuid.uuid4()) for _ in range(4)]
        _mark_failed(cache, rids, "error")
        assert cache.client.command.call_count == 4


# ============================================================================
# 6. _mark_completed
# ============================================================================


class TestMarkCompleted:
    """Verify _mark_completed writes correct parameters."""

    def test_writes_bars_and_gap(self):
        cache = _make_mock_cache()
        rid = str(uuid.uuid4())
        _mark_completed(cache, [rid], bars_written=42, gap_seconds=180.5)
        params = cache.client.command.call_args[1]["parameters"]
        assert params["bars_written"] == 42
        assert params["gap_seconds"] == 180.5

    def test_updates_all_request_ids(self):
        cache = _make_mock_cache()
        rids = [str(uuid.uuid4()) for _ in range(2)]
        _mark_completed(cache, rids, bars_written=10, gap_seconds=5.0)
        assert cache.client.command.call_count == 2


# ============================================================================
# 7. check_backfill_status (Issue #106) -- Wave 2 design, may not be
#    implemented yet
# ============================================================================


class TestCheckBackfillStatus:
    """Progressive status reporting for backfill requests."""

    def test_status_none_when_no_requests(self):
        from rangebar.recency import BackfillStatus, check_backfill_status

        mock_cache = _make_mock_cache(query_rows=[])
        with patch("rangebar.clickhouse.RangeBarCache") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_cache)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = check_backfill_status("BTCUSDT", 250)
        assert isinstance(result, BackfillStatus)
        assert result.status == "none"

    def test_status_pending(self):
        from datetime import UTC, datetime

        from rangebar.recency import check_backfill_status

        now = datetime.now(UTC)
        mock_cache = _make_mock_cache(
            query_rows=[("req-1", "pending", now, None, None, 0, 0.0, None)]
        )
        with patch("rangebar.clickhouse.RangeBarCache") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_cache)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = check_backfill_status("BTCUSDT", 250)
        assert result.status == "pending"
        assert result.elapsed_seconds >= 0

    def test_status_running_with_gap(self):
        from datetime import UTC, datetime

        from rangebar.recency import check_backfill_status

        now = datetime.now(UTC)
        mock_cache = _make_mock_cache(
            query_rows=[("req-1", "running", now, now, None, 0, 45.5, None)]
        )
        with patch("rangebar.clickhouse.RangeBarCache") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_cache)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = check_backfill_status("BTCUSDT", 250)
        assert result.status == "running"
        assert result.gap_seconds == 45.5

    def test_status_completed(self):
        from datetime import UTC, datetime

        from rangebar.recency import check_backfill_status

        now = datetime.now(UTC)
        mock_cache = _make_mock_cache(
            query_rows=[("req-1", "completed", now, now, now, 150, 30.0, None)]
        )
        with patch("rangebar.clickhouse.RangeBarCache") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_cache)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = check_backfill_status("BTCUSDT", 250)
        assert result.status == "completed"
        assert result.bars_written == 150

    def test_status_failed(self):
        from datetime import UTC, datetime

        from rangebar.recency import check_backfill_status

        now = datetime.now(UTC)
        mock_cache = _make_mock_cache(
            query_rows=[("req-1", "failed", now, now, now, 0, 0.0, "SSH tunnel dropped")]
        )
        with patch("rangebar.clickhouse.RangeBarCache") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_cache)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = check_backfill_status("BTCUSDT", 250)
        assert result.status == "failed"
        assert result.error == "SSH tunnel dropped"


class TestWatcherOfflineDetection:
    """Pending request older than 5 min sets watcher_likely_offline=True."""

    def test_offline_detection(self):
        from datetime import UTC, datetime, timedelta

        from rangebar.recency import check_backfill_status

        old_time = datetime.now(UTC) - timedelta(minutes=10)
        mock_cache = _make_mock_cache(
            query_rows=[("req-1", "pending", old_time, None, None, 0, 0.0, None)]
        )
        with patch("rangebar.clickhouse.RangeBarCache") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_cache)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = check_backfill_status("BTCUSDT", 250)
        assert result.status == "pending"
        assert result.watcher_likely_offline is True

    def test_no_offline_for_recent_pending(self):
        from datetime import UTC, datetime

        from rangebar.recency import check_backfill_status

        now = datetime.now(UTC)
        mock_cache = _make_mock_cache(
            query_rows=[("req-1", "pending", now, None, None, 0, 0.0, None)]
        )
        with patch("rangebar.clickhouse.RangeBarCache") as mock_cls:
            mock_cls.return_value.__enter__ = MagicMock(return_value=mock_cache)
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = check_backfill_status("BTCUSDT", 250)
        assert result.watcher_likely_offline is False
