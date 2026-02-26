"""ClickHouse isolation and mode guard tests for monthly ouroboros. Issue #97.

Tests cover:
- Checkpoint isolation by ouroboros mode
- get_n_bars ouroboros filter
- get_bars_by_timestamp_range isolation (existing filter verification)
- Mode guard rejects mixed modes
- Mode guard allows same mode
- Mode guard allows after force_refresh

All tests require ClickHouse (@pytest.mark.clickhouse).
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.clickhouse


@pytest.fixture
def cache():
    """Get a RangeBarCache instance, skip if ClickHouse unavailable."""
    try:
        from rangebar.clickhouse import RangeBarCache

        c = RangeBarCache()
        c.__enter__()
        yield c
        c.__exit__(None, None, None)
    except (ImportError, ConnectionError, OSError, RuntimeError) as e:
        pytest.skip(f"ClickHouse not available: {e}")


@pytest.fixture
def test_symbol():
    """Use a test symbol that won't conflict with production data."""
    return "TESTUSDT_OUROBOROS"


@pytest.fixture
def test_threshold():
    return 9999  # Unusual threshold to avoid conflicts


def _make_bars_df(n: int, start_ts_ms: int = 1704067200000) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame for testing."""
    data = {
        "timestamp_ms": [start_ts_ms + i * 60000 for i in range(n)],
        "Open": [100.0 + i * 0.1 for i in range(n)],
        "High": [100.5 + i * 0.1 for i in range(n)],
        "Low": [99.5 + i * 0.1 for i in range(n)],
        "Close": [100.2 + i * 0.1 for i in range(n)],
        "Volume": [10.0] * n,
    }
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df.drop(columns=["timestamp_ms"])
    return df


def _cleanup(cache, symbol: str, threshold: int) -> None:
    """Delete test data from ClickHouse."""
    import contextlib

    with contextlib.suppress(OSError, RuntimeError):
        cache.client.command(
            """
            ALTER TABLE rangebar_cache.range_bars
            DELETE WHERE symbol = {symbol:String}
              AND threshold_decimal_bps = {threshold:UInt32}
            """,
            parameters={"symbol": symbol, "threshold": threshold},
        )


class TestCheckpointIsolationByMode:
    """Test that checkpoint load/save respects ouroboros_mode filter."""

    def test_save_year_load_month_returns_none(self, cache, test_symbol, test_threshold):
        """Save a year checkpoint, load with month → returns None."""
        try:
            # Save year checkpoint
            cache.save_checkpoint(
                symbol=test_symbol,
                threshold_decimal_bps=test_threshold,
                start_date="2024-01-01",
                end_date="2024-12-31",
                last_completed_date="2024-06-15",
                last_trade_timestamp_ms=1718409600000,
                processor_checkpoint="{}",
                bars_written=1000,
                ouroboros_mode="year",
            )

            # Load with month mode → should return None
            result = cache.load_checkpoint(
                test_symbol, test_threshold, "2024-01-01", "2024-12-31",
                ouroboros_mode="month",
            )
            assert result is None, "Month-mode load should not find year-mode checkpoint"

            # Load with year mode → should find it
            result = cache.load_checkpoint(
                test_symbol, test_threshold, "2024-01-01", "2024-12-31",
                ouroboros_mode="year",
            )
            assert result is not None, "Year-mode load should find year-mode checkpoint"
            assert result["bars_written"] == 1000

        finally:
            cache.delete_checkpoint(
                test_symbol, test_threshold, "2024-01-01", "2024-12-31",
                ouroboros_mode="year",
            )


class TestGetNBarsOuroborosFilter:
    """Test that get_n_bars respects ouroboros_mode filter."""

    def test_returns_only_matching_mode(self, cache, test_symbol, test_threshold):
        """Insert bars with both modes; query returns only matching mode."""
        try:
            # Write year bars
            year_bars = _make_bars_df(5, start_ts_ms=1704067200000)
            cache.store_bars_bulk(
                test_symbol, test_threshold, year_bars,
                ouroboros_mode="year", skip_dedup=True,
            )

            # Query with year mode
            df, count = cache.get_n_bars(
                test_symbol, test_threshold, 100,
                ouroboros_mode="year",
            )
            assert df is not None
            assert len(df) == 5

            # Query with month mode (no month bars written)
            df_month, count_month = cache.get_n_bars(
                test_symbol, test_threshold, 100,
                ouroboros_mode="month",
            )
            assert df_month is None or len(df_month) == 0
            assert count_month == 0

        finally:
            _cleanup(cache, test_symbol, test_threshold)


class TestBarsTimestampRangeIsolation:
    """Verify existing ouroboros_mode filter in get_bars_by_timestamp_range."""

    def test_filters_by_mode(self, cache, test_symbol, test_threshold):
        """Existing filter in get_bars_by_timestamp_range works correctly."""
        try:
            year_bars = _make_bars_df(5, start_ts_ms=1704067200000)
            cache.store_bars_bulk(
                test_symbol, test_threshold, year_bars,
                ouroboros_mode="year", skip_dedup=True,
            )

            # Query with matching mode
            df = cache.get_bars_by_timestamp_range(
                test_symbol, test_threshold,
                start_ts=1704067100000,
                end_ts=1704067500000,
                ouroboros_mode="year",
            )
            assert df is not None

            # Query with non-matching mode
            df_month = cache.get_bars_by_timestamp_range(
                test_symbol, test_threshold,
                start_ts=1704067100000,
                end_ts=1704067500000,
                ouroboros_mode="month",
            )
            assert df_month is None

        finally:
            _cleanup(cache, test_symbol, test_threshold)


class TestModeGuardRejectsMixed:
    """Test that store_bars_bulk/batch rejects writes with a different mode."""

    def test_rejects_month_after_year(self, cache, test_symbol, test_threshold):
        """Write year bars, then attempt month bars → CacheWriteError."""
        from rangebar.exceptions import CacheWriteError

        try:
            # Write year bars first
            year_bars = _make_bars_df(3, start_ts_ms=1704067200000)
            cache.store_bars_bulk(
                test_symbol, test_threshold, year_bars,
                ouroboros_mode="year", skip_dedup=True,
            )

            # Attempt month bars → should raise
            month_bars = _make_bars_df(3, start_ts_ms=1704167200000)
            with pytest.raises(CacheWriteError, match="existing data uses year"):
                cache.store_bars_bulk(
                    test_symbol, test_threshold, month_bars,
                    ouroboros_mode="month", skip_dedup=True,
                )
        finally:
            _cleanup(cache, test_symbol, test_threshold)


class TestModeGuardAllowsSameMode:
    """Test that writing same mode bars multiple times works."""

    def test_allows_additional_month_bars(self, cache, test_symbol, test_threshold):
        """Write month bars, then write more month bars → no error."""
        try:
            bars1 = _make_bars_df(3, start_ts_ms=1704067200000)
            cache.store_bars_bulk(
                test_symbol, test_threshold, bars1,
                ouroboros_mode="month", skip_dedup=True,
            )

            bars2 = _make_bars_df(3, start_ts_ms=1704167200000)
            # Should not raise
            cache.store_bars_bulk(
                test_symbol, test_threshold, bars2,
                ouroboros_mode="month", skip_dedup=True,
            )

        finally:
            _cleanup(cache, test_symbol, test_threshold)


class TestModeGuardAllowsAfterForceRefresh:
    """Test that after clearing data, a different mode can be written."""

    def test_allows_month_after_clear(self, cache, test_symbol, test_threshold):
        """Write year bars, clear, write month bars → no error."""
        try:
            # Write year bars
            year_bars = _make_bars_df(3, start_ts_ms=1704067200000)
            cache.store_bars_bulk(
                test_symbol, test_threshold, year_bars,
                ouroboros_mode="year", skip_dedup=True,
            )

            # Clear (simulating force_refresh)
            _cleanup(cache, test_symbol, test_threshold)

            # Wait for async delete to propagate
            import time
            time.sleep(1)

            # Write month bars → should work now
            month_bars = _make_bars_df(3, start_ts_ms=1704167200000)
            cache.store_bars_bulk(
                test_symbol, test_threshold, month_bars,
                ouroboros_mode="month", skip_dedup=True,
            )

        finally:
            _cleanup(cache, test_symbol, test_threshold)
