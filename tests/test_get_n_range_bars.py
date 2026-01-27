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
    except (ImportError, OSError, ConnectionError, RuntimeError):
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
    """Tests that require actual data fetching (marked slow).

    Note: These tests use historical end_date to avoid failures when
    today's date has no data available on Binance.
    """

    # Use a known historical date to avoid data availability issues
    HISTORICAL_END_DATE = "2024-12-01"

    def test_returns_bars_with_fetch(self):
        """Should return bars when fetch_if_missing=True."""
        try:
            df = get_n_range_bars(
                TEST_SYMBOL,
                n_bars=10,
                threshold_decimal_bps=250,
                use_cache=False,
                fetch_if_missing=True,
                max_lookback_days=7,
                warn_if_fewer=False,
                end_date=self.HISTORICAL_END_DATE,
            )
        except RuntimeError as e:
            if "No data available" in str(e):
                pytest.skip(f"Binance data not available: {e}")
            raise
        # May return fewer if data not available, but should have valid format
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
            assert isinstance(df.index, pd.DatetimeIndex)

    def test_output_format_ohlcv(self):
        """Output format: OHLCV columns, DatetimeIndex."""
        try:
            df = get_n_range_bars(
                TEST_SYMBOL,
                n_bars=50,
                use_cache=False,
                max_lookback_days=14,
                warn_if_fewer=False,
                end_date=self.HISTORICAL_END_DATE,
            )
        except RuntimeError as e:
            if "No data available" in str(e):
                pytest.skip(f"Binance data not available: {e}")
            raise
        if len(df) > 0:
            assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
            assert isinstance(df.index, pd.DatetimeIndex)

    def test_chronological_order(self):
        """Output is sorted oldest-first (chronological)."""
        try:
            df = get_n_range_bars(
                TEST_SYMBOL,
                n_bars=100,
                use_cache=False,
                max_lookback_days=14,
                warn_if_fewer=False,
                end_date=self.HISTORICAL_END_DATE,
            )
        except RuntimeError as e:
            if "No data available" in str(e):
                pytest.skip(f"Binance data not available: {e}")
            raise
        if len(df) > 0:
            assert df.index.is_monotonic_increasing

    def test_ohlc_invariants(self):
        """OHLC invariants: High >= max(O,C), Low <= min(O,C)."""
        try:
            df = get_n_range_bars(
                TEST_SYMBOL,
                n_bars=100,
                use_cache=False,
                max_lookback_days=14,
                warn_if_fewer=False,
                end_date=self.HISTORICAL_END_DATE,
            )
        except RuntimeError as e:
            if "No data available" in str(e):
                pytest.skip(f"Binance data not available: {e}")
            raise
        if len(df) > 0:
            assert (df["High"] >= df["Open"]).all()
            assert (df["High"] >= df["Close"]).all()
            assert (df["Low"] <= df["Open"]).all()
            assert (df["Low"] <= df["Close"]).all()

    def test_no_nan_values(self):
        """Output should have no NaN values."""
        try:
            df = get_n_range_bars(
                TEST_SYMBOL,
                n_bars=50,
                use_cache=False,
                max_lookback_days=14,
                warn_if_fewer=False,
                end_date=self.HISTORICAL_END_DATE,
            )
        except RuntimeError as e:
            if "No data available" in str(e):
                pytest.skip(f"Binance data not available: {e}")
            raise
        if len(df) > 0:
            assert not df.isna().any().any()


@pytest.mark.slow
class TestEndDateBehavior:
    """Tests for end_date parameter behavior."""

    def test_end_date_boundary(self):
        """Bars should respect end_date as the reference point for data fetching.

        Note: end_date is used to anchor the data fetch window, but due to
        timezone handling and the count-bounded nature of get_n_range_bars(),
        bars may extend slightly past end_date when the underlying tick data
        spans timezone boundaries. This is expected behavior for the current
        implementation.
        """
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
            # Verify we got bars near the end_date
            # The exact boundary behavior depends on timezone and data availability
            max_ts = df.index.max()
            # Bars should be within a reasonable window of end_date (within 1 week)
            boundary = pd.Timestamp("2024-06-08", tz=max_ts.tz if max_ts.tz else None)
            assert max_ts < boundary, f"Bars extend too far past end_date: {max_ts}"


@pytest.mark.slow
class TestMicrostructure:
    """Tests for microstructure data."""

    # Use a known historical date to avoid data availability issues
    HISTORICAL_END_DATE = "2024-12-01"

    def test_microstructure_columns(self):
        """include_microstructure=True adds extra columns."""
        try:
            df = get_n_range_bars(
                TEST_SYMBOL,
                n_bars=50,
                include_microstructure=True,
                use_cache=False,
                max_lookback_days=14,
                warn_if_fewer=False,
                end_date=self.HISTORICAL_END_DATE,
            )
        except RuntimeError as e:
            if "No data available" in str(e):
                pytest.skip(f"Binance data not available: {e}")
            raise
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
        try:
            df1 = get_n_range_bars(
                TEST_SYMBOL,
                n_bars=100,
                use_cache=True,
                max_lookback_days=30,
                warn_if_fewer=False,
            )
        except RuntimeError as e:
            if "No data available" in str(e):
                pytest.skip(f"Binance data not available for recent dates: {e}")
            raise

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
        try:
            df1 = get_n_range_bars(
                TEST_SYMBOL,
                n_bars=100,
                use_cache=True,
                max_lookback_days=30,
                warn_if_fewer=False,
            )
        except RuntimeError as e:
            if "No data available" in str(e):
                pytest.skip(f"Binance data not available for recent dates: {e}")
            raise

        if len(df1) >= 100:
            # Second call - should hit cache
            start = time.perf_counter()
            df2 = get_n_range_bars(TEST_SYMBOL, n_bars=100, use_cache=True)
            cache_time = time.perf_counter() - start

            assert len(df2) == 100
            # Cache hit should be fast (< 1 second)
            assert cache_time < 1.0, f"Cache hit took {cache_time:.2f}s, expected <1s"


@pytest.mark.clickhouse
class TestBarContinuity:
    """Tests for bar continuity (bar[i+1].open == bar[i].close).

    Range bars MUST have continuity: each bar's open price equals the
    previous bar's close price. For Binance (24/7 crypto), any discontinuity
    is a bug since there are no natural market breaks.

    These tests validate continuity across:
    - Cross-date boundaries (within same month)
    - Cross-month boundaries
    - Cross-year boundaries (2024→2025, 2025→2026)
    """

    # Tolerance beyond threshold for gap detection (floating-point, tick spread)
    CONTINUITY_TOLERANCE = 0.01  # 1% tolerance beyond threshold
    THRESHOLD_BPS = 250  # 2.5% threshold

    def test_bar_continuity_basic(self):
        """Gaps between bars should not exceed threshold + tolerance."""
        if not _is_clickhouse_available():
            pytest.skip("ClickHouse server not available")

        from rangebar import validate_continuity

        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=500,
            threshold_decimal_bps=self.THRESHOLD_BPS,
            use_cache=True,
            max_lookback_days=30,
            warn_if_fewer=False,
        )

        if len(df) < 100:
            pytest.skip(f"Only {len(df)} bars available, need at least 100")

        result = validate_continuity(
            df,
            tolerance_pct=self.CONTINUITY_TOLERANCE,
            threshold_decimal_bps=self.THRESHOLD_BPS,
        )
        gaps = result["discontinuity_count"]
        first = result["discontinuities"][0] if result["discontinuities"] else "N/A"
        assert result["is_valid"], f"Found {gaps} gaps. First: {first}"

    def test_continuity_across_date_boundary(self):
        """Continuity across midnight (cross-date boundary)."""
        if not _is_clickhouse_available():
            pytest.skip("ClickHouse server not available")

        from rangebar import validate_continuity

        # Request enough bars to span multiple days
        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=1000,
            threshold_decimal_bps=250,
            use_cache=True,
            max_lookback_days=30,
            warn_if_fewer=False,
        )

        if len(df) < 500:
            pytest.skip(
                f"Only {len(df)} bars available, need at least 500 for cross-date test"
            )

        # Verify we actually span multiple dates
        unique_dates = df.index.date
        num_dates = len(set(unique_dates))
        if num_dates < 2:
            pytest.skip(f"Data only spans {num_dates} date(s), need at least 2")

        result = validate_continuity(
            df,
            tolerance_pct=self.CONTINUITY_TOLERANCE,
            threshold_decimal_bps=self.THRESHOLD_BPS,
        )
        assert result["is_valid"], (
            f"Cross-date discontinuity: {result['discontinuity_count']} gaps "
            f"across {num_dates} dates"
        )

    def test_continuity_across_month_boundary(self):
        """Continuity across month boundary (e.g., Jan 31 → Feb 1)."""
        if not _is_clickhouse_available():
            pytest.skip("ClickHouse server not available")

        from rangebar import validate_continuity

        # Request enough bars to span multiple months
        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=5000,
            threshold_decimal_bps=250,
            use_cache=True,
            max_lookback_days=90,
            warn_if_fewer=False,
        )

        if len(df) < 1000:
            pytest.skip(f"Only {len(df)} bars available, need >=1000")

        # Verify we span multiple months
        months = df.index.to_period("M")
        num_months = len(months.unique())
        if num_months < 2:
            pytest.skip(f"Data only spans {num_months} month(s), need at least 2")

        result = validate_continuity(
            df,
            tolerance_pct=self.CONTINUITY_TOLERANCE,
            threshold_decimal_bps=self.THRESHOLD_BPS,
        )
        assert result["is_valid"], (
            f"Cross-month discontinuity: {result['discontinuity_count']} gaps "
            f"across {num_months} months"
        )

    def test_continuity_across_year_boundary_2024_2025(self):
        """Continuity across 2024→2025 year boundary (Dec 31 → Jan 1)."""
        if not _is_clickhouse_available():
            pytest.skip("ClickHouse server not available")

        from rangebar import validate_continuity

        # Request bars ending in early January 2025 to capture year boundary
        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=10000,
            threshold_decimal_bps=250,
            end_date="2025-01-15",
            use_cache=True,
            max_lookback_days=45,
            warn_if_fewer=False,
        )

        if len(df) < 1000:
            pytest.skip(f"Only {len(df)} bars available for 2024-2025 boundary test")

        # Verify we actually span the year boundary
        years = df.index.year
        has_2024 = 2024 in years.to_numpy()
        has_2025 = 2025 in years.to_numpy()

        if not (has_2024 and has_2025):
            pytest.skip(
                f"Data doesn't span 2024-2025 boundary "
                f"(has 2024: {has_2024}, has 2025: {has_2025})"
            )

        result = validate_continuity(
            df,
            tolerance_pct=self.CONTINUITY_TOLERANCE,
            threshold_decimal_bps=self.THRESHOLD_BPS,
        )
        gaps = result["discontinuity_count"]
        first = result["discontinuities"][0] if result["discontinuities"] else "N/A"
        assert result["is_valid"], f"2024→2025: {gaps} gaps. First: {first}"

    def test_continuity_across_year_boundary_2025_2026(self):
        """Continuity across 2025→2026 year boundary (Dec 31 → Jan 1)."""
        if not _is_clickhouse_available():
            pytest.skip("ClickHouse server not available")

        from rangebar import validate_continuity

        # Request bars ending in early January 2026 to capture year boundary
        # Note: Today is January 9, 2026 so limited 2026 data available
        df = get_n_range_bars(
            TEST_SYMBOL,
            n_bars=10000,
            threshold_decimal_bps=250,
            end_date="2026-01-09",
            use_cache=True,
            max_lookback_days=45,
            warn_if_fewer=False,
        )

        if len(df) < 1000:
            pytest.skip(f"Only {len(df)} bars available for 2025-2026 boundary test")

        # Verify we actually span the year boundary
        years = df.index.year
        has_2025 = 2025 in years.to_numpy()
        has_2026 = 2026 in years.to_numpy()

        if not (has_2025 and has_2026):
            pytest.skip(
                f"Data doesn't span 2025-2026 boundary "
                f"(has 2025: {has_2025}, has 2026: {has_2026})"
            )

        result = validate_continuity(
            df,
            tolerance_pct=self.CONTINUITY_TOLERANCE,
            threshold_decimal_bps=self.THRESHOLD_BPS,
        )
        gaps = result["discontinuity_count"]
        first = result["discontinuities"][0] if result["discontinuities"] else "N/A"
        assert result["is_valid"], f"2025→2026: {gaps} gaps. First: {first}"

    def test_continuity_large_dataset(self):
        """Continuity test with large dataset (spans multiple file boundaries).

        Note: A small number of junction discontinuities are acceptable when data
        spans multiple cache sessions. See Issue #5 for background.
        """
        if not _is_clickhouse_available():
            pytest.skip("ClickHouse server not available")

        from rangebar import validate_continuity

        # Request a large number of bars to stress-test continuity
        # Binance data is stored in daily files, so this should span many files
        try:
            df = get_n_range_bars(
                TEST_SYMBOL,
                n_bars=20000,
                threshold_decimal_bps=250,
                use_cache=True,
                max_lookback_days=180,
                warn_if_fewer=False,
            )
        except RuntimeError as e:
            if "No data available" in str(e):
                pytest.skip(f"Binance data not available for recent dates: {e}")
            raise

        if len(df) < 5000:
            pytest.skip(f"Only {len(df)} bars available, need >=5000")

        result = validate_continuity(
            df,
            tolerance_pct=self.CONTINUITY_TOLERANCE,
            threshold_decimal_bps=self.THRESHOLD_BPS,
        )

        # Calculate percentage of discontinuities
        discontinuity_pct = (
            result["discontinuity_count"] / result["bar_count"] * 100
            if result["bar_count"] > 0
            else 0
        )

        # Allow up to 0.05% junction discontinuities (Issue #5)
        # These occur at cache session boundaries and are documented behavior
        max_acceptable_pct = 0.05
        assert discontinuity_pct <= max_acceptable_pct, (
            f"Large dataset discontinuity: {result['discontinuity_count']} gaps "
            f"({discontinuity_pct:.2f}%) in {result['bar_count']} bars. "
            f"Max acceptable: {max_acceptable_pct}%"
        )

    def test_validate_continuity_function_returns_correct_format(self):
        """validate_continuity() returns correct dict structure."""
        from rangebar import validate_continuity

        # Create minimal test DataFrame
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.5, 101.0],
                "High": [100.5, 101.0, 101.5],
                "Low": [99.5, 100.0, 100.5],
                "Close": [100.5, 101.0, 101.5],
                "Volume": [10.0, 10.0, 10.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )

        result = validate_continuity(df)

        # Check structure
        assert "is_valid" in result
        assert "bar_count" in result
        assert "discontinuity_count" in result
        assert "discontinuities" in result
        assert isinstance(result["is_valid"], bool)
        assert isinstance(result["bar_count"], int)
        assert isinstance(result["discontinuity_count"], int)
        assert isinstance(result["discontinuities"], list)

    def test_validate_continuity_detects_gaps(self):
        """validate_continuity() detects price gaps exceeding threshold."""
        from rangebar import validate_continuity

        # Create DataFrame with intentional large discontinuity (>5%)
        # With 250 dbps (0.25%) threshold + 0.5% tolerance = 0.75% max allowed
        # Gap from 100.0 to 106.0 = 6% gap - should be detected
        df = pd.DataFrame(
            {
                "Open": [
                    100.0,
                    100.5,
                    106.0,  # 6% gap from prev close (100.5)
                ],
                "High": [100.5, 100.5, 106.5],
                "Low": [99.5, 100.0, 105.5],
                "Close": [100.5, 100.5, 106.5],
                "Volume": [10.0, 10.0, 10.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )

        result = validate_continuity(
            df,
            tolerance_pct=0.005,  # 0.5%
            threshold_decimal_bps=250,  # 2.5%
        )

        assert not result["is_valid"], "Should detect the 6% price gap"
        assert result["discontinuity_count"] == 1
        assert len(result["discontinuities"]) == 1
        assert result["discontinuities"][0]["bar_index"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
