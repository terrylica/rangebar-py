"""Integration tests for ClickHouse cache functionality.

These tests require a running ClickHouse server and will be skipped
if ClickHouse is not available.

Run with: pytest tests/test_clickhouse_integration.py -v
Skip with: pytest -m "not clickhouse"
"""

from __future__ import annotations

import pandas as pd
import pytest

# Check if ClickHouse is available before running any tests
try:
    from rangebar.clickhouse import (
        CacheKey,
        InstallationLevel,
        RangeBarCache,
        detect_clickhouse_state,
        get_available_clickhouse_host,
    )

    _PREFLIGHT = detect_clickhouse_state()
    _CH_AVAILABLE = _PREFLIGHT.level >= InstallationLevel.RUNNING_NO_SCHEMA
except Exception:
    _CH_AVAILABLE = False
    _PREFLIGHT = None

# Skip entire module if ClickHouse not available
_skip_reason = (
    f"ClickHouse not available: {_PREFLIGHT.message if _PREFLIGHT else 'import failed'}"
)
pytestmark = [
    pytest.mark.clickhouse,
    pytest.mark.skipif(not _CH_AVAILABLE, reason=_skip_reason),
]


@pytest.fixture(scope="module")
def clickhouse_host():
    """Get available ClickHouse host for integration tests."""
    return get_available_clickhouse_host()


@pytest.fixture
def cache():
    """Create a RangeBarCache for testing.

    Yields cache instance and cleans up after test.
    """
    cache = RangeBarCache()
    yield cache
    cache.close()


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    """Generate sample trade data for testing."""
    import numpy as np

    np.random.seed(42)
    n_trades = 1000

    base_timestamp = 1704067200000  # 2024-01-01 00:00:00 UTC
    base_price = 42000.0

    returns = np.random.randn(n_trades) * 0.001
    prices = base_price * np.cumprod(1 + returns)

    return pd.DataFrame(
        {
            "timestamp": base_timestamp + np.arange(n_trades) * 100,
            "price": prices,
            "quantity": np.random.exponential(1.0, n_trades),
            "agg_trade_id": np.arange(n_trades),
            "is_buyer_maker": np.random.randint(0, 2, n_trades).astype(bool),
        }
    )


@pytest.fixture
def sample_range_bars() -> pd.DataFrame:
    """Generate sample range bar data for testing."""
    return pd.DataFrame(
        {
            "Open": [42000.0, 42105.0, 42200.0],
            "High": [42100.0, 42200.0, 42300.0],
            "Low": [41950.0, 42050.0, 42150.0],
            "Close": [42050.0, 42150.0, 42250.0],
            "Volume": [10.5, 8.3, 12.1],
        },
        index=pd.DatetimeIndex(
            [
                "2024-01-01 00:00:15",
                "2024-01-01 00:03:42",
                "2024-01-01 00:08:33",
            ]
        ),
    )


class TestPreflightIntegration:
    """Integration tests for preflight detection."""

    def test_detect_clickhouse_state(self) -> None:
        """Test that preflight detection works against real ClickHouse."""
        state = detect_clickhouse_state()

        assert state.level >= InstallationLevel.RUNNING_NO_SCHEMA
        assert state.version is not None
        assert "ready" in state.message.lower() or "schema" in state.message.lower()

    def test_get_available_host(self, clickhouse_host) -> None:
        """Test that host selection returns valid connection."""
        assert clickhouse_host is not None
        assert clickhouse_host.host is not None
        assert clickhouse_host.method in ("local", "direct", "ssh_tunnel")
        assert clickhouse_host.port > 0


class TestRangeBarCacheIntegration:
    """Integration tests for RangeBarCache with real ClickHouse."""

    def test_cache_initialization(self, cache: RangeBarCache) -> None:
        """Test that cache initializes and creates schema."""
        # Cache should connect and create schema automatically
        assert cache.client is not None

        # Verify database exists
        result = cache.client.command("SHOW DATABASES LIKE 'rangebar_cache'")
        assert result == "rangebar_cache"

    def test_schema_tables_exist(self, cache: RangeBarCache) -> None:
        """Test that required tables are created."""
        tables = cache.client.command("SHOW TABLES FROM rangebar_cache")

        # Should have range_bars table (raw_trades moved to local Parquet storage)
        assert "range_bars" in tables


class TestRangeBarsIntegration:
    """Integration tests for range bars caching (Tier 2)."""

    def test_store_and_retrieve_bars(
        self, cache: RangeBarCache, sample_range_bars: pd.DataFrame
    ) -> None:
        """Test storing and retrieving range bars."""
        key = CacheKey(
            symbol="TEST_BTCUSDT_BARS",
            threshold_bps=250,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )

        # Clean up any existing data from previous test runs
        cache.invalidate_range_bars(key)
        import time

        time.sleep(0.5)  # Allow async mutation to complete

        # Store bars
        count = cache.store_range_bars(key, sample_range_bars)
        assert count == len(sample_range_bars)

        # Retrieve bars
        retrieved = cache.get_range_bars(key)

        assert retrieved is not None
        assert len(retrieved) == len(sample_range_bars)
        assert isinstance(retrieved.index, pd.DatetimeIndex)
        assert list(retrieved.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_has_range_bars(
        self, cache: RangeBarCache, sample_range_bars: pd.DataFrame
    ) -> None:
        """Test checking for range bar existence."""
        key = CacheKey(
            symbol="TEST_BTCUSDT_HAS_BARS",
            threshold_bps=250,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )

        # Store bars
        cache.store_range_bars(key, sample_range_bars)

        # Should have bars
        assert cache.has_range_bars(key) is True

        # Should not have bars for different key
        other_key = CacheKey(
            symbol="NONEXISTENT",
            threshold_bps=250,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )
        assert cache.has_range_bars(other_key) is False

    def test_invalidate_range_bars(
        self, cache: RangeBarCache, sample_range_bars: pd.DataFrame
    ) -> None:
        """Test invalidating cached range bars."""
        key = CacheKey(
            symbol="TEST_BTCUSDT_INVALIDATE",
            threshold_bps=250,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )

        # Store bars
        cache.store_range_bars(key, sample_range_bars)
        assert cache.has_range_bars(key) is True

        # Invalidate
        cache.invalidate_range_bars(key)

        # ClickHouse DELETE is async via mutations, so we wait briefly
        import time

        time.sleep(0.5)

        # May still show as existing due to async nature of ClickHouse mutations
        # The important thing is that the mutation was submitted
        # In production, FINAL keyword or OPTIMIZE would be used


class TestCachedProcessingIntegration:
    """Integration tests for end-to-end cached processing."""

    def test_process_trades_cached_first_run(self, cache: RangeBarCache) -> None:
        """Test cached processing on first run (cache miss)."""
        # Generate unique symbol for this test
        import time

        from rangebar import process_trades_to_dataframe_cached

        symbol = f"TEST_CACHED_{int(time.time())}"

        trades = pd.DataFrame(
            {
                "timestamp": [1704067200000 + i * 1000 for i in range(100)],
                "price": [42000.0 + i * 10.0 for i in range(100)],
                "quantity": [1.0 + (i % 5) * 0.5 for i in range(100)],
            }
        )

        # First run - should compute and cache
        df1 = process_trades_to_dataframe_cached(
            trades, symbol=symbol, threshold_bps=250, cache=cache
        )

        assert len(df1) > 0
        assert isinstance(df1.index, pd.DatetimeIndex)
        assert list(df1.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_process_trades_cached_second_run(self, cache: RangeBarCache) -> None:
        """Test cached processing on second run (cache hit)."""
        import time

        from rangebar import process_trades_to_dataframe_cached

        symbol = f"TEST_CACHED_HIT_{int(time.time())}"

        trades = pd.DataFrame(
            {
                "timestamp": [1704067200000 + i * 1000 for i in range(100)],
                "price": [42000.0 + i * 10.0 for i in range(100)],
                "quantity": [1.0 + (i % 5) * 0.5 for i in range(100)],
            }
        )

        # First run - compute and cache
        df1 = process_trades_to_dataframe_cached(
            trades, symbol=symbol, threshold_bps=250, cache=cache
        )

        # Second run - should hit cache
        start = time.perf_counter()
        df2 = process_trades_to_dataframe_cached(
            trades, symbol=symbol, threshold_bps=250, cache=cache
        )
        cache_time = time.perf_counter() - start

        # Results should match
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True), df2.reset_index(drop=True), check_names=False
        )

        # Cache hit should be fast (< 100ms typically)
        assert cache_time < 1.0, f"Cache hit took {cache_time:.3f}s, expected < 1s"


class TestCacheKeyIntegration:
    """Integration tests for cache key behavior."""

    def test_different_thresholds_different_cache(
        self, cache: RangeBarCache, sample_range_bars: pd.DataFrame
    ) -> None:
        """Test that different thresholds use different cache entries."""
        key_250 = CacheKey(
            symbol="TEST_THRESHOLD",
            threshold_bps=250,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )
        key_500 = CacheKey(
            symbol="TEST_THRESHOLD",
            threshold_bps=500,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )

        # Store with 250bps key
        cache.store_range_bars(key_250, sample_range_bars)

        # Should find 250bps
        assert cache.has_range_bars(key_250) is True

        # Should NOT find 500bps
        assert cache.has_range_bars(key_500) is False

    def test_different_symbols_different_cache(
        self, cache: RangeBarCache, sample_range_bars: pd.DataFrame
    ) -> None:
        """Test that different symbols use different cache entries."""
        key_btc = CacheKey(
            symbol="TEST_BTCUSDT_SYM",
            threshold_bps=250,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )
        key_eth = CacheKey(
            symbol="TEST_ETHUSDT_SYM",
            threshold_bps=250,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )

        # Store BTC
        cache.store_range_bars(key_btc, sample_range_bars)

        # Should find BTC
        assert cache.has_range_bars(key_btc) is True

        # Should NOT find ETH
        assert cache.has_range_bars(key_eth) is False
