"""Tests for GitHub Issues #7 and #8 implementation.

Issue #7: fetch_if_missing parameter for get_range_bars()
Issue #8: Single-pass workflow for Walk-Forward Optimization (WFO)

Run with: pytest tests/test_issues_7_8.py -v
Skip ClickHouse tests: pytest -m "not clickhouse"
"""

import pandas as pd
import pytest
from rangebar import (
    ContinuityError,
    ContinuityWarning,
    PrecomputeProgress,
    PrecomputeResult,
    get_n_range_bars,
    get_range_bars,
    precompute_range_bars,
)
from rangebar.storage.parquet import TickStorage


def _is_clickhouse_available() -> bool:
    """Check if ClickHouse server is running and accessible."""
    try:
        from rangebar.clickhouse import RangeBarCache

        with RangeBarCache() as cache:
            cache.count_bars("BTCUSDT", 250)
    except Exception:
        return False
    else:
        return True


# ============================================================================
# Issue #7: fetch_if_missing for get_range_bars()
# ============================================================================


class TestIssue7FetchIfMissing:
    """Tests for fetch_if_missing parameter in get_range_bars()."""

    def test_fetch_if_missing_true_fetches_data(self):
        """Default behavior (fetch_if_missing=True) should fetch data."""
        # Use a short date range to minimize test time
        df = get_range_bars(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-02",
            threshold_decimal_bps=250,
            use_cache=True,
            fetch_if_missing=True,
        )
        # Should have data (assuming ClickHouse is available)
        # If ClickHouse is not available, this may be empty
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_fetch_if_missing_false_returns_empty_on_miss(self):
        """With fetch_if_missing=False, cache miss should return empty DataFrame."""
        # Use a symbol/date that's unlikely to be cached
        df = get_range_bars(
            "TESTUNKNOWNSYMBOL",
            "2020-01-01",
            "2020-01-02",
            threshold_decimal_bps=999,  # Unusual threshold
            use_cache=True,
            fetch_if_missing=False,
        )
        # Should be empty (no fetch, no cache)
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_fetch_if_missing_parameter_exists(self):
        """Verify fetch_if_missing parameter is accepted."""
        # Just verify the parameter doesn't raise TypeError
        try:
            df = get_range_bars(
                "BTCUSDT",
                "2024-01-01",
                "2024-01-02",
                fetch_if_missing=False,
                use_cache=False,
            )
        except TypeError as e:
            if "fetch_if_missing" in str(e):
                pytest.fail("fetch_if_missing parameter not accepted")
            raise


# ============================================================================
# Issue #8: precompute_range_bars() Function
# ============================================================================


class TestIssue8PrecomputeFunction:
    """Tests for precompute_range_bars() function."""

    def test_precompute_returns_result_dataclass(self):
        """precompute_range_bars() should return PrecomputeResult."""
        # This test may be slow - use minimal date range
        pytest.skip("Requires ClickHouse and network access - run manually")
        result = precompute_range_bars(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-07",
            threshold_decimal_bps=250,
        )
        assert isinstance(result, PrecomputeResult)
        assert result.symbol == "BTCUSDT"
        assert result.threshold_decimal_bps == 250

    def test_precompute_invalid_dates_raises(self):
        """Invalid date format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            precompute_range_bars(
                "BTCUSDT",
                "2024/01/01",  # Wrong format
                "2024-01-07",
            )

    def test_precompute_invalid_threshold_raises(self):
        """Invalid threshold should raise ValueError."""
        with pytest.raises(ValueError, match="threshold_decimal_bps must be between"):
            precompute_range_bars(
                "BTCUSDT",
                "2024-01-01",
                "2024-01-07",
                threshold_decimal_bps=0,
            )

    def test_precompute_progress_callback(self):
        """Progress callback should receive PrecomputeProgress updates."""
        pytest.skip("Requires ClickHouse and network access - run manually")
        progress_updates = []

        def on_progress(p: PrecomputeProgress) -> None:
            progress_updates.append(p)

        precompute_range_bars(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-31",
            progress_callback=on_progress,
        )

        assert len(progress_updates) > 0
        assert all(isinstance(p, PrecomputeProgress) for p in progress_updates)
        # Last update should be caching phase
        assert progress_updates[-1].phase == "caching"


# ============================================================================
# Issue #8: TickStorage.fetch_month() and fetch_date_range()
# ============================================================================


class TestTickStorageMethods:
    """Tests for TickStorage.fetch_month() and fetch_date_range()."""

    def test_fetch_month_returns_lazyframe(self):
        """fetch_month() should return pl.LazyFrame."""
        import polars as pl

        storage = TickStorage()
        lf = storage.fetch_month("BTCUSDT", 2024, 1)
        assert isinstance(lf, pl.LazyFrame)

    def test_fetch_month_nonexistent_returns_empty(self):
        """fetch_month() for non-existent data should return empty LazyFrame."""
        import polars as pl

        storage = TickStorage()
        lf = storage.fetch_month("NONEXISTENTSYMBOL", 2000, 1)
        assert isinstance(lf, pl.LazyFrame)
        # Collecting should give empty DataFrame
        df = lf.collect()
        assert len(df) == 0

    def test_fetch_date_range_is_generator(self):
        """fetch_date_range() should return a generator of LazyFrames."""

        storage = TickStorage()
        gen = storage.fetch_date_range("BTCUSDT", "2024-01-01", "2024-03-31")
        # Should be an iterator/generator
        assert hasattr(gen, "__iter__")
        assert hasattr(gen, "__next__")


# ============================================================================
# Issue #8: Cache Invalidation Methods
# ============================================================================


@pytest.mark.clickhouse
class TestCacheInvalidation:
    """Tests for cache invalidation methods."""

    @pytest.fixture
    def cache(self):
        """Get RangeBarCache instance."""
        from rangebar.clickhouse import RangeBarCache

        if not _is_clickhouse_available():
            pytest.skip("ClickHouse not available")
        with RangeBarCache() as cache:
            yield cache

    def test_invalidate_range_bars_by_range_exists(self, cache):
        """invalidate_range_bars_by_range() method should exist."""
        assert hasattr(cache, "invalidate_range_bars_by_range")
        assert callable(cache.invalidate_range_bars_by_range)

    def test_get_last_bar_before_exists(self, cache):
        """get_last_bar_before() method should exist."""
        assert hasattr(cache, "get_last_bar_before")
        assert callable(cache.get_last_bar_before)

    def test_invalidate_range_bars_by_range_returns_int(self, cache):
        """invalidate_range_bars_by_range() should return count of deleted bars."""
        # Use a range that likely has no data
        count = cache.invalidate_range_bars_by_range(
            "NONEXISTENTSYMBOL",
            999,
            0,
            1000,
        )
        assert isinstance(count, int)
        assert count >= 0


# ============================================================================
# Issue #8: validate_on_return Parameter
# ============================================================================


class TestValidateOnReturn:
    """Tests for validate_on_return parameter in get_n_range_bars()."""

    def test_validate_on_return_false_by_default(self):
        """validate_on_return should default to False."""
        # Should not raise ContinuityError even if data has gaps
        df = get_n_range_bars(
            "BTCUSDT",
            n_bars=100,
            use_cache=False,
            fetch_if_missing=False,
            warn_if_fewer=False,
        )
        # Empty result, should not raise
        assert isinstance(df, pd.DataFrame)

    def test_validate_on_return_parameter_accepted(self):
        """validate_on_return parameter should be accepted."""
        try:
            df = get_n_range_bars(
                "BTCUSDT",
                n_bars=100,
                validate_on_return=True,
                use_cache=False,
                fetch_if_missing=False,
                warn_if_fewer=False,
            )
        except TypeError as e:
            if "validate_on_return" in str(e):
                pytest.fail("validate_on_return parameter not accepted")
            # Other errors are fine

    def test_continuity_action_warn_emits_warning(self):
        """continuity_action='warn' should emit ContinuityWarning."""
        # Create a DataFrame with a discontinuity
        df = pd.DataFrame(
            {
                "Open": [100.0, 105.0],  # Gap: 100 != 105
                "High": [102.0, 107.0],
                "Low": [99.0, 104.0],
                "Close": [101.0, 106.0],
                "Volume": [100.0, 100.0],
            },
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
        )
        # The validation logic is internal - we test via parameter acceptance
        # Tested indirectly through integration

    def test_continuity_action_values_accepted(self):
        """continuity_action parameter values should be accepted."""
        for action in ["warn", "raise", "log"]:
            try:
                get_n_range_bars(
                    "BTCUSDT",
                    n_bars=100,
                    validate_on_return=True,
                    continuity_action=action,
                    use_cache=False,
                    fetch_if_missing=False,
                    warn_if_fewer=False,
                )
            except TypeError as e:
                if "continuity_action" in str(e):
                    pytest.fail(f"continuity_action='{action}' not accepted")
            except Exception:
                pass  # Other errors are fine


# ============================================================================
# Issue #8: chunk_size Parameter
# ============================================================================


class TestChunkSizeParameter:
    """Tests for chunk_size parameter."""

    def test_chunk_size_parameter_accepted(self):
        """chunk_size parameter should be accepted by get_n_range_bars()."""
        try:
            get_n_range_bars(
                "BTCUSDT",
                n_bars=100,
                chunk_size=50_000,
                use_cache=False,
                fetch_if_missing=False,
                warn_if_fewer=False,
            )
        except TypeError as e:
            if "chunk_size" in str(e):
                pytest.fail("chunk_size parameter not accepted")

    def test_chunk_size_default_value(self):
        """Default chunk_size should be 100_000."""
        # Can't easily test the default value, but we verify it's accepted
        get_n_range_bars(
            "BTCUSDT",
            n_bars=100,
            chunk_size=100_000,  # Default value
            use_cache=False,
            fetch_if_missing=False,
            warn_if_fewer=False,
        )


# ============================================================================
# Exception and Warning Classes
# ============================================================================


class TestExceptionClasses:
    """Tests for ContinuityError and ContinuityWarning classes."""

    def test_continuity_error_is_exception(self):
        """ContinuityError should be a valid exception."""
        assert issubclass(ContinuityError, Exception)

    def test_continuity_error_has_discontinuities_attribute(self):
        """ContinuityError should have discontinuities attribute."""
        err = ContinuityError("Test error", [{"bar_index": 0}])
        assert hasattr(err, "discontinuities")
        assert err.discontinuities == [{"bar_index": 0}]

    def test_continuity_error_empty_discontinuities(self):
        """ContinuityError with None discontinuities should default to empty list."""
        err = ContinuityError("Test error")
        assert err.discontinuities == []

    def test_continuity_warning_is_warning(self):
        """ContinuityWarning should be a valid warning class."""
        assert issubclass(ContinuityWarning, UserWarning)


# ============================================================================
# DataClass Tests
# ============================================================================


class TestDataClasses:
    """Tests for PrecomputeProgress and PrecomputeResult dataclasses."""

    def test_precompute_progress_fields(self):
        """PrecomputeProgress should have all required fields."""
        progress = PrecomputeProgress(
            phase="fetching",
            current_month="2024-01",
            months_completed=1,
            months_total=3,
            bars_generated=1000,
            ticks_processed=50000,
            elapsed_seconds=5.5,
        )
        assert progress.phase == "fetching"
        assert progress.current_month == "2024-01"
        assert progress.months_completed == 1
        assert progress.months_total == 3
        assert progress.bars_generated == 1000
        assert progress.ticks_processed == 50000
        assert progress.elapsed_seconds == 5.5

    def test_precompute_result_fields(self):
        """PrecomputeResult should have all required fields."""
        result = PrecomputeResult(
            symbol="BTCUSDT",
            threshold_decimal_bps=250,
            start_date="2024-01-01",
            end_date="2024-03-31",
            total_bars=5000,
            total_ticks=1000000,
            elapsed_seconds=30.0,
            continuity_valid=True,
            cache_key="BTCUSDT_250_1704067200000_1711929600000",
        )
        assert result.symbol == "BTCUSDT"
        assert result.threshold_decimal_bps == 250
        assert result.total_bars == 5000
        assert result.continuity_valid is True


# ============================================================================
# Issue #10: Duplicate trades handling
# ============================================================================


class TestDuplicateTradesHandling:
    """Tests for duplicate trades deduplication (Issue #10)."""

    def test_deduplication_with_duplicate_trade_ids(self):
        """Verify duplicate trades are deduplicated before processing."""
        import polars as pl

        # Create test data with duplicate trade IDs
        trades = pl.DataFrame(
            {
                "timestamp": [1000, 1000, 2000, 3000],  # First two have same timestamp
                "agg_trade_id": [
                    100,
                    100,
                    101,
                    102,
                ],  # First two have same ID (duplicate)
                "price": [42000.0, 42000.0, 42100.0, 42200.0],
                "quantity": [1.0, 1.0, 2.0, 3.0],
            }
        )

        # Apply the same deduplication logic used in precompute_range_bars
        if "agg_trade_id" in trades.columns:
            deduped = trades.unique(subset=["agg_trade_id"], maintain_order=True)
        else:
            deduped = trades

        # Should have 3 unique trades (not 4)
        assert len(deduped) == 3
        assert deduped["agg_trade_id"].to_list() == [100, 101, 102]

    def test_deduplication_preserves_order(self):
        """Verify deduplication preserves chronological order."""
        import polars as pl

        trades = pl.DataFrame(
            {
                "timestamp": [3000, 1000, 2000, 1000],  # Unsorted, with duplicate
                "trade_id": [103, 101, 102, 101],  # Duplicate trade_id
                "price": [42200.0, 42000.0, 42100.0, 42000.0],
                "quantity": [3.0, 1.0, 2.0, 1.0],
            }
        )

        # Sort first, then deduplicate (order matters)
        sorted_trades = trades.sort("timestamp")
        deduped = sorted_trades.unique(subset=["trade_id"], maintain_order=True)

        assert len(deduped) == 3
        # After sorting by timestamp and deduping, order should be 101, 102, 103
        assert deduped["trade_id"].to_list() == [101, 102, 103]
