"""End-to-end tests for get_n_range_bars() with real Binance data.

These tests validate bar-count-based retrieval with ClickHouse caching.
Designed to run on both local development and GPU workstations.

Run with: pytest tests/test_get_n_range_bars.py -v
Skip ClickHouse tests: pytest -m "not clickhouse"
"""

import pandas as pd
import pytest
from rangebar import THRESHOLD_PRESETS, get_n_range_bars

# Test constants
TEST_SYMBOL = "BTCUSDT"


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


class TestBarCountBasics:
    """Basic bar-count API tests - no ClickHouse required."""

    def test_n_bars_zero_raises(self):
        """n_bars=0 should raise ValueError."""
        with pytest.raises(ValueError, match="n_bars must be > 0"):
            get_n_range_bars(TEST_SYMBOL, n_bars=0, use_cache=False)

    def test_n_bars_negative_raises(self):
        """Negative n_bars should raise ValueError."""
        with pytest.raises(ValueError, match="n_bars must be > 0"):
            get_n_range_bars(TEST_SYMBOL, n_bars=-10, use_cache=False)

    def test_invalid_threshold_preset_raises(self):
        """Invalid threshold preset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown threshold preset"):
            get_n_range_bars(
                TEST_SYMBOL,
                n_bars=100,
                threshold_decimal_bps="invalid",
                use_cache=False,
            )

    def test_invalid_source_raises(self):
        """Invalid source should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown source"):
            get_n_range_bars(TEST_SYMBOL, n_bars=100, source="invalid", use_cache=False)

    def test_invalid_market_raises(self):
        """Invalid market should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown market"):
            get_n_range_bars(TEST_SYMBOL, n_bars=100, market="invalid", use_cache=False)

    def test_invalid_date_format_raises(self):
        """Invalid end_date format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            get_n_range_bars(
                TEST_SYMBOL, n_bars=100, end_date="2024/01/01", use_cache=False
            )

    def test_threshold_out_of_range_raises(self):
        """Threshold outside valid range should raise ValueError."""
        with pytest.raises(ValueError, match="threshold_decimal_bps must be between"):
            get_n_range_bars(
                TEST_SYMBOL, n_bars=100, threshold_decimal_bps=0, use_cache=False
            )

        with pytest.raises(ValueError, match="threshold_decimal_bps must be between"):
            get_n_range_bars(
                TEST_SYMBOL, n_bars=100, threshold_decimal_bps=200000, use_cache=False
            )


class TestOutputFormat:
    """Tests for output DataFrame format."""

    def test_empty_result_has_correct_columns(self):
        """Empty result should have correct OHLCV columns."""
        # Force empty result with impossible params
        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=1000000,
            max_lookback_days=0,
            use_cache=False,
            fetch_if_missing=False,
            warn_if_fewer=False,
        )
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_empty_result_returns_empty_dataframe(self):
        """Empty result should return empty DataFrame."""
        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=100,
            use_cache=False,
            fetch_if_missing=False,
            warn_if_fewer=False,
        )
        assert len(df) == 0


class TestThresholdPresets:
    """Tests for threshold preset strings."""

    @pytest.mark.parametrize("preset", THRESHOLD_PRESETS.keys())
    def test_preset_strings_are_valid(self, preset):
        """All preset strings should be valid (no ValueError)."""
        # Just validate that presets don't raise - no actual fetching
        import contextlib

        with contextlib.suppress(RuntimeError):
            # RuntimeError expected when no data/cache available
            get_n_range_bars(
                TEST_SYMBOL,
                n_bars=50,
                threshold_decimal_bps=preset,
                use_cache=False,
                fetch_if_missing=False,
                warn_if_fewer=False,
            )


class TestWarnings:
    """Tests for warning behavior."""

    def test_warn_if_fewer_emits_warning(self):
        """warn_if_fewer=True should emit warning if returning less."""
        with pytest.warns(UserWarning, match="bars instead of requested"):
            get_n_range_bars(
                TEST_SYMBOL,
                n_bars=1000000,
                max_lookback_days=0,
                use_cache=False,
                fetch_if_missing=False,
                warn_if_fewer=True,
            )

    def test_warn_if_fewer_false_no_warning(self):
        """warn_if_fewer=False should not emit warning."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not raise any warning
            get_n_range_bars(
                TEST_SYMBOL,
                n_bars=1000000,
                max_lookback_days=0,
                use_cache=False,
                fetch_if_missing=False,
                warn_if_fewer=False,
            )


@pytest.mark.slow
class TestDataFetching:
    """Tests that require actual data fetching (marked slow)."""

    def test_returns_bars_with_fetch(self):
        """Should return bars when fetch_if_missing=True."""
        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=10,
            threshold_decimal_bps=250,
            use_cache=False,
            fetch_if_missing=True,
            max_lookback_days=7,
            warn_if_fewer=False,
        )
        # May return fewer if data not available, but should have valid format
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
            assert isinstance(df.index, pd.DatetimeIndex)

    def test_output_format_ohlcv(self):
        """Output format: OHLCV columns, DatetimeIndex."""
        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=50,
            use_cache=False,
            max_lookback_days=14,
            warn_if_fewer=False,
        )
        if len(df) > 0:
            assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
            assert isinstance(df.index, pd.DatetimeIndex)

    def test_chronological_order(self):
        """Output is sorted oldest-first (chronological)."""
        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=100,
            use_cache=False,
            max_lookback_days=14,
            warn_if_fewer=False,
        )
        if len(df) > 0:
            assert df.index.is_monotonic_increasing

    def test_ohlc_invariants(self):
        """OHLC invariants: High >= max(O,C), Low <= min(O,C)."""
        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=100,
            use_cache=False,
            max_lookback_days=14,
            warn_if_fewer=False,
        )
        if len(df) > 0:
            assert (df["High"] >= df["Open"]).all()
            assert (df["High"] >= df["Close"]).all()
            assert (df["Low"] <= df["Open"]).all()
            assert (df["Low"] <= df["Close"]).all()

    def test_no_nan_values(self):
        """Output should have no NaN values."""
        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=50,
            use_cache=False,
            max_lookback_days=14,
            warn_if_fewer=False,
        )
        if len(df) > 0:
            assert not df.isna().any().any()


@pytest.mark.slow
class TestEndDateBehavior:
    """Tests for end_date parameter behavior."""

    def test_end_date_boundary(self):
        """Bars should not extend beyond end_date."""
        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=100,
            end_date="2024-06-01",
            threshold_decimal_bps=250,
            use_cache=False,
            max_lookback_days=30,
            warn_if_fewer=False,
        )
        if len(df) > 0:
            # All timestamps should be before 2024-06-02 00:00:00
            # Handle both tz-naive and tz-aware indices
            max_ts = df.index.max()
            boundary = pd.Timestamp("2024-06-02", tz=max_ts.tz if max_ts.tz else None)
            assert max_ts < boundary


@pytest.mark.slow
class TestMicrostructure:
    """Tests for microstructure data."""

    def test_microstructure_columns(self):
        """include_microstructure=True adds extra columns."""
        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=50,
            include_microstructure=True,
            use_cache=False,
            max_lookback_days=14,
            warn_if_fewer=False,
        )
        if len(df) > 0:
            # Should have microstructure columns
            assert "vwap" in df.columns or "buy_volume" in df.columns


@pytest.mark.clickhouse
class TestClickHouseCache:
    """Tests requiring ClickHouse (marked for optional skip)."""

    def test_cache_returns_bars(self):
        """Cache should return bars if available."""
        if not _is_clickhouse_available():
            pytest.skip("ClickHouse server not available")

        # First call - may or may not populate cache
        df1 = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=100,
            use_cache=True,
            max_lookback_days=30,
            warn_if_fewer=False,
        )

        if len(df1) >= 100:
            # Second call - should hit cache
            df2 = get_n_range_bars(TEST_SYMBOL, n_bars=100, use_cache=True)
            assert len(df2) == 100

    def test_cache_hit_fast_path(self):
        """Second call should hit cache (faster)."""
        if not _is_clickhouse_available():
            pytest.skip("ClickHouse server not available")

        import time

        # First call - populates cache
        df1 = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=100,
            use_cache=True,
            max_lookback_days=30,
            warn_if_fewer=False,
        )

        if len(df1) >= 100:
            # Second call - should hit cache
            start = time.perf_counter()
            df2 = get_n_range_bars(TEST_SYMBOL, n_bars=100, use_cache=True)
            cache_time = time.perf_counter() - start

            assert len(df2) == 100
            # Cache hit should be fast (< 1 second)
            assert cache_time < 1.0, f"Cache hit took {cache_time:.2f}s, expected <1s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
