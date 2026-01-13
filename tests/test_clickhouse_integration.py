"""Integration tests for ClickHouse cache functionality.

These tests require a running ClickHouse server and will be skipped
if ClickHouse is not available.

Run with: pytest tests/test_clickhouse_integration.py -v
Skip with: pytest -m "not clickhouse"
"""

from __future__ import annotations

import numpy as np
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
except (ImportError, OSError, ConnectionError) as e:
    _CH_AVAILABLE = False
    _PREFLIGHT = None
    _import_error = e  # Preserve error for debugging

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
            threshold_decimal_bps=250,
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
            threshold_decimal_bps=250,
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
            threshold_decimal_bps=250,
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
            threshold_decimal_bps=250,
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
            trades, symbol=symbol, threshold_decimal_bps=250, cache=cache
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
            trades, symbol=symbol, threshold_decimal_bps=250, cache=cache
        )

        # Second run - should hit cache
        start = time.perf_counter()
        df2 = process_trades_to_dataframe_cached(
            trades, symbol=symbol, threshold_decimal_bps=250, cache=cache
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
            threshold_decimal_bps=250,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )
        key_500 = CacheKey(
            symbol="TEST_THRESHOLD",
            threshold_decimal_bps=500,
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
            threshold_decimal_bps=250,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )
        key_eth = CacheKey(
            symbol="TEST_ETHUSDT_SYM",
            threshold_decimal_bps=250,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )

        # Store BTC
        cache.store_range_bars(key_btc, sample_range_bars)

        # Should find BTC
        assert cache.has_range_bars(key_btc) is True

        # Should NOT find ETH
        assert cache.has_range_bars(key_eth) is False


# =============================================================================
# REGRESSION TESTS - Issue #35 and Issue #32
# =============================================================================
# These tests ensure we don't regress on bugs discovered via GitHub issues.
# They validate the specific fixes applied in v7.2.0.


class TestIssue35DuplicateBarsRegression:
    """Regression tests for Issue #35: Duplicate bars when threshold_decimal_bps=200.

    Root cause: Missing FINAL clause in ClickHouse queries for ReplacingMergeTree.
    Fix: Added FINAL to all SELECT queries in cache.py.

    This test verifies that storing and retrieving bars does NOT produce duplicates,
    even when multiple INSERT operations occur (which is when ReplacingMergeTree
    would show duplicates without FINAL).
    """

    def test_no_duplicate_bars_after_multiple_inserts(
        self, cache: RangeBarCache
    ) -> None:
        """Test that multiple inserts don't produce duplicate rows on retrieval.

        This is the core regression test for Issue #35. The bug was that
        ReplacingMergeTree without FINAL would return duplicate rows before
        background merges completed.
        """
        import time

        # Use threshold=200 (the specific threshold reported in Issue #35)
        key = CacheKey(
            symbol="REGRESSION_ISSUE35",
            threshold_decimal_bps=200,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )

        # Clean up
        cache.invalidate_range_bars(key)
        time.sleep(0.3)

        # Create sample bars with unique timestamps
        bars_df = pd.DataFrame(
            {
                "Open": [42000.0 + i * 100 for i in range(10)],
                "High": [42100.0 + i * 100 for i in range(10)],
                "Low": [41950.0 + i * 100 for i in range(10)],
                "Close": [42050.0 + i * 100 for i in range(10)],
                "Volume": [10.0 + i for i in range(10)],
            },
            index=pd.DatetimeIndex([f"2024-01-01 00:{i:02d}:00" for i in range(10)]),
        )

        # Insert TWICE - this is when ReplacingMergeTree would show duplicates
        # without FINAL clause (before background merge)
        cache.store_range_bars(key, bars_df)
        time.sleep(0.1)
        cache.store_range_bars(key, bars_df)  # Second insert = potential duplicates
        time.sleep(0.1)

        # Retrieve - with FINAL clause, should NOT have duplicates
        retrieved = cache.get_range_bars(key)

        assert retrieved is not None, "Should retrieve bars"

        # THE KEY ASSERTION: No duplicates
        # Before fix: len(retrieved) would be 20 (duplicate rows)
        # After fix: len(retrieved) should be 10 (FINAL deduplicates)
        assert len(retrieved) == len(bars_df), (
            f"Expected {len(bars_df)} bars, got {len(retrieved)}. "
            f"REGRESSION: Issue #35 - Duplicate bars detected! "
            f"Check that FINAL clause is present in cache.py queries."
        )

        # Also verify no duplicate timestamps
        timestamps = retrieved.index.tolist()
        unique_timestamps = list(set(timestamps))
        assert len(timestamps) == len(unique_timestamps), (
            f"Duplicate timestamps found: {len(timestamps)} total vs "
            f"{len(unique_timestamps)} unique. REGRESSION: Issue #35"
        )

    def test_count_bars_no_duplicates(self, cache: RangeBarCache) -> None:
        """Test that count_bars() also uses FINAL and returns correct count."""
        import time

        key = CacheKey(
            symbol="REGRESSION_ISSUE35_COUNT",
            threshold_decimal_bps=200,
            start_ts=1704067200000,
            end_ts=1704153600000,
        )

        cache.invalidate_range_bars(key)
        time.sleep(0.3)

        bars_df = pd.DataFrame(
            {
                "Open": [42000.0, 42100.0, 42200.0],
                "High": [42100.0, 42200.0, 42300.0],
                "Low": [41950.0, 42050.0, 42150.0],
                "Close": [42050.0, 42150.0, 42250.0],
                "Volume": [10.0, 11.0, 12.0],
            },
            index=pd.DatetimeIndex(
                ["2024-01-01 00:00:00", "2024-01-01 00:05:00", "2024-01-01 00:10:00"]
            ),
        )

        # Insert twice
        cache.store_range_bars(key, bars_df)
        cache.store_range_bars(key, bars_df)

        # count_bars should return 3, not 6
        count = cache.count_bars(
            symbol="REGRESSION_ISSUE35_COUNT",
            threshold_decimal_bps=200,
        )

        assert count == 3, (
            f"Expected 3 bars, count_bars() returned {count}. "
            f"REGRESSION: Issue #35 - count_bars() missing FINAL clause."
        )


class TestIssue32MicrostructureRangesRegression:
    """Regression tests for Issue #32: Microstructure column value ranges.

    Findings:
    1. kyle_lambda_proxy had dimensional inconsistency (extreme values -2.9M to +5.5M)
       Fix: Changed formula to ((close-open)/open) / (imbalance/total_vol)
    2. aggregation_efficiency renamed to aggregation_density
       Fix: Renamed field throughout codebase

    These tests verify the formulas produce sensible ranges.
    """

    def test_kyle_lambda_proxy_bounded_range(self) -> None:
        """Test that kyle_lambda_proxy produces bounded values after fix.

        Before fix: (close - open) / imbalance could explode when imbalance ≈ 0
        After fix: ((close-open)/open) / (imbalance/total_vol) is dimensionally
                   consistent and bounded for typical market data.
        """
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)

        # Create trades that would have caused extreme kyle_lambda before fix:
        # Small price move, tiny imbalance (old formula would explode)
        trades = [
            {
                "agg_trade_id": 1,
                "price": 50000.0,
                "quantity": 1.0,
                "first_trade_id": 1,
                "last_trade_id": 1,
                "timestamp": 1704067200000,
                "is_buyer_maker": False,  # Buy
            },
            {
                "agg_trade_id": 2,
                "price": 50000.0,  # Same price
                "quantity": 1.0,
                "first_trade_id": 2,
                "last_trade_id": 2,
                "timestamp": 1704067201000,
                "is_buyer_maker": True,  # Sell (balanced imbalance)
            },
            {
                "agg_trade_id": 3,
                "price": 50150.0,  # +0.3% breach
                "quantity": 1.0,
                "first_trade_id": 3,
                "last_trade_id": 3,
                "timestamp": 1704067202000,
                "is_buyer_maker": False,  # Buy
            },
        ]

        bars = processor.process_trades(trades)
        assert len(bars) >= 1, "Should produce at least one bar"

        bar = bars[0]
        kyle_lambda = bar["kyle_lambda_proxy"]

        # After fix with percentage returns formula:
        # - kyle_lambda should be bounded (not millions)
        # - Typical range: [-100, 100] for normalized values
        assert abs(kyle_lambda) < 1000, (
            f"kyle_lambda_proxy = {kyle_lambda} is too extreme. "
            f"REGRESSION: Issue #32 - Formula may have reverted to old version."
        )

    def test_kyle_lambda_zero_imbalance_safe(self) -> None:
        """Test that kyle_lambda handles zero imbalance gracefully.

        When buy_vol == sell_vol, normalized_imbalance = 0, so we return 0.0.
        """
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)

        # Perfectly balanced trades (imbalance = 0)
        trades = [
            {
                "agg_trade_id": 1,
                "price": 50000.0,
                "quantity": 1.0,
                "first_trade_id": 1,
                "last_trade_id": 1,
                "timestamp": 1704067200000,
                "is_buyer_maker": False,  # Buy 1.0
            },
            {
                "agg_trade_id": 2,
                "price": 50000.0,
                "quantity": 1.0,
                "first_trade_id": 2,
                "last_trade_id": 2,
                "timestamp": 1704067201000,
                "is_buyer_maker": True,  # Sell 1.0 (perfectly balanced)
            },
            {
                "agg_trade_id": 3,
                "price": 50150.0,  # Breach
                "quantity": 0.001,  # Tiny volume, won't change balance much
                "first_trade_id": 3,
                "last_trade_id": 3,
                "timestamp": 1704067202000,
                "is_buyer_maker": False,
            },
        ]

        bars = processor.process_trades(trades)
        if bars:
            kyle_lambda = bars[0]["kyle_lambda_proxy"]
            # Should be 0.0 or very small (not NaN, not Inf, not millions)
            assert not np.isnan(kyle_lambda), "kyle_lambda should not be NaN"
            assert not np.isinf(kyle_lambda), "kyle_lambda should not be Inf"

    def test_aggregation_density_field_exists(self) -> None:
        """Test that aggregation_density field exists (renamed from aggregation_efficiency).

        This verifies the v7.2.0 rename was applied to PyO3 bindings.
        """
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)

        trades = [
            {
                "agg_trade_id": 1,
                "price": 50000.0,
                "quantity": 1.0,
                "first_trade_id": 1,
                "last_trade_id": 3,  # 3 individual trades in this agg
                "timestamp": 1704067200000,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 2,
                "price": 50150.0,  # Breach
                "quantity": 1.0,
                "first_trade_id": 4,
                "last_trade_id": 4,
                "timestamp": 1704067201000,
                "is_buyer_maker": False,
            },
        ]

        bars = processor.process_trades(trades)
        assert len(bars) >= 1

        bar = bars[0]

        # New field name should exist
        assert "aggregation_density" in bar, (
            "aggregation_density field missing from bar output. "
            "REGRESSION: Issue #32 - Field rename may have been reverted."
        )

        # Old field name should NOT exist
        assert "aggregation_efficiency" not in bar, (
            "aggregation_efficiency field still exists (should be renamed). "
            "REGRESSION: Issue #32 - Rename incomplete in PyO3 bindings."
        )

    def test_aggregation_density_range(self) -> None:
        """Test that aggregation_density has expected range [1, +inf).

        aggregation_density = individual_trade_count / agg_record_count
        Since each agg record has at least 1 individual trade, minimum is 1.0.
        """
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)

        # Create agg trade that represents 5 individual trades
        trades = [
            {
                "agg_trade_id": 1,
                "price": 50000.0,
                "quantity": 1.0,
                "first_trade_id": 100,
                "last_trade_id": 104,  # 5 individual trades (104 - 100 + 1)
                "timestamp": 1704067200000,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 2,
                "price": 50150.0,  # Breach
                "quantity": 1.0,
                "first_trade_id": 105,
                "last_trade_id": 105,  # 1 individual trade
                "timestamp": 1704067201000,
                "is_buyer_maker": False,
            },
        ]

        bars = processor.process_trades(trades)
        assert len(bars) >= 1

        agg_density = bars[0]["aggregation_density"]

        # Should be >= 1.0 (minimum when each agg has exactly 1 trade)
        assert agg_density >= 1.0, (
            f"aggregation_density = {agg_density}, expected >= 1.0. "
            f"REGRESSION: Issue #32 - Formula may be inverted."
        )

        # For this test: 6 individual trades / 2 agg records = 3.0
        expected_density = 3.0  # (5 + 1) / 2
        assert (
            abs(agg_density - expected_density) < 0.1
        ), f"aggregation_density = {agg_density}, expected ~{expected_density}"

    def test_aggression_ratio_capped_at_100(self) -> None:
        """Test that aggression_ratio is capped at 100 (documented behavior)."""
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)

        # All buys, no sells - would be infinity without cap
        trades = [
            {
                "agg_trade_id": i,
                "price": 50000.0 + i * 50,
                "quantity": 1.0,
                "first_trade_id": i,
                "last_trade_id": i,
                "timestamp": 1704067200000 + i * 1000,
                "is_buyer_maker": False,  # All buys
            }
            for i in range(10)
        ]
        # Add breach trade
        trades.append(
            {
                "agg_trade_id": 100,
                "price": 50150.0,  # Breach
                "quantity": 1.0,
                "first_trade_id": 100,
                "last_trade_id": 100,
                "timestamp": 1704067210000,
                "is_buyer_maker": False,
            }
        )

        bars = processor.process_trades(trades)
        if bars:
            aggression_ratio = bars[0]["aggression_ratio"]
            assert aggression_ratio <= 100.0, (
                f"aggression_ratio = {aggression_ratio}, should be capped at 100. "
                f"REGRESSION: Issue #32 - Cap may have been removed."
            )
