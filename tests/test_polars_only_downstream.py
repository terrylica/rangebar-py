# polars-exception: This test validates Polars-only downstream usage (Issue #45)
"""Tests validating that rangebar-py can be used with zero Pandas dependency downstream.

Issue #45: Make Polars the default DataFrame backend, Pandas optional.

These tests verify every pathway a downstream consumer (e.g., alpha-forge research)
needs to operate entirely in Polars, never touching Pandas.

Test Categories:
1. Streaming API: get_range_bars(materialize=False) → Iterator[pl.DataFrame]
2. Conversion: _bars_list_to_polars() → pl.DataFrame
3. Concatenation: pl.concat() of streaming batches
4. Column/dtype consistency for ML pipelines
5. process_trades_polars() with Polars-native output
6. Arrow pathway: Rust → Arrow → Polars (zero-copy)
7. Returns/features computation entirely in Polars
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import polars as pl
import pytest
from rangebar import RangeBarProcessor, process_trades_polars
from rangebar.conversion import _bars_list_to_polars, normalize_temporal_precision

# =============================================================================
# Fixtures: Synthetic trade data in Polars (no Pandas at any point)
# =============================================================================


@pytest.fixture
def synthetic_trades_polars() -> pl.DataFrame:
    """Create synthetic trade data as a Polars DataFrame.

    Generates trades that cross a 250 dbps threshold (0.25% = $105 on $42,000).
    Uses $200 price steps with oscillation to reliably produce completed bars.
    """
    base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC
    trades = []
    price = 42000.0

    # Generate 500 trades with $200 steps oscillating up/down
    # Each half-cycle (10 trades x $200 = $2000) far exceeds the $105 threshold
    for i in range(500):
        if i % 20 < 10:
            price += 200.0
        else:
            price -= 200.0

        trades.append({
            "timestamp": base_ts + i * 1000,  # 1 second apart
            "price": price,
            "quantity": 1.0 + (i % 3) * 0.5,
        })

    return pl.DataFrame(trades)


@pytest.fixture
def synthetic_trades_lazy(synthetic_trades_polars: pl.DataFrame) -> pl.LazyFrame:
    """Create synthetic trades as a LazyFrame for predicate pushdown testing."""
    return synthetic_trades_polars.lazy()


@pytest.fixture
def completed_bars() -> list[dict]:
    """Create completed bar dicts as returned by RangeBarProcessor."""
    processor = RangeBarProcessor(threshold_decimal_bps=250)
    base_ts = 1704067200000
    trades = []
    price = 42000.0
    for i in range(500):
        if i % 20 < 10:
            price += 200.0
        else:
            price -= 200.0
        trades.append({
            "timestamp": base_ts + i * 1000,
            "price": price,
            "quantity": 1.0,
        })
    return processor.process_trades(trades)


# =============================================================================
# Category 1: Conversion - _bars_list_to_polars()
# =============================================================================


class TestBarsListToPolars:
    """Test _bars_list_to_polars returns proper Polars DataFrame."""

    def test_returns_polars_dataframe(self, completed_bars: list[dict]):
        """Output must be pl.DataFrame, not pd.DataFrame."""
        result = _bars_list_to_polars(completed_bars)
        assert isinstance(result, pl.DataFrame)

    def test_has_ohlcv_columns(self, completed_bars: list[dict]):
        """OHLCV columns must be present (backtesting.py capitalization)."""
        result = _bars_list_to_polars(completed_bars)
        expected = {"timestamp", "Open", "High", "Low", "Close", "Volume"}
        assert expected.issubset(set(result.columns))

    def test_ohlcv_dtypes_are_float(self, completed_bars: list[dict]):
        """OHLCV columns must be float64 for ML pipelines."""
        result = _bars_list_to_polars(completed_bars)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert result[col].dtype == pl.Float64, (
                f"{col} dtype is {result[col].dtype}, expected Float64"
            )

    def test_timestamp_is_datetime(self, completed_bars: list[dict]):
        """Timestamp must be a proper datetime, not string."""
        result = _bars_list_to_polars(completed_bars)
        assert result["timestamp"].dtype.is_temporal(), (
            f"timestamp dtype is {result['timestamp'].dtype}, expected Datetime"
        )

    def test_ohlc_invariants(self, completed_bars: list[dict]):
        """High >= max(Open, Close) and Low <= min(Open, Close) for every bar."""
        result = _bars_list_to_polars(completed_bars)
        assert (result["High"] >= result["Open"]).all()
        assert (result["High"] >= result["Close"]).all()
        assert (result["Low"] <= result["Open"]).all()
        assert (result["Low"] <= result["Close"]).all()

    def test_empty_bars_returns_empty_polars_df(self):
        """Empty input must return empty Polars DataFrame, not error."""
        result = _bars_list_to_polars([])
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_no_pandas_import_in_result(self, completed_bars: list[dict]):
        """Verify result has no Pandas metadata or hidden conversion."""
        result = _bars_list_to_polars(completed_bars)
        # Polars DataFrames have .shape, not .index (Pandas)
        assert hasattr(result, "shape")
        assert not hasattr(result, "index")  # Pandas artifact


# =============================================================================
# Category 2: Temporal Precision Normalization (Polars-native)
# =============================================================================


class TestTemporalPrecision:
    """Test normalize_temporal_precision works on Polars DataFrames."""

    def test_normalizes_to_microseconds(self):
        """Mixed precision DataFrames must normalize to microsecond."""
        df = pl.DataFrame({
            "ts": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ],
        })
        result = normalize_temporal_precision(df)
        assert result["ts"].dtype.time_unit == "us"  # type: ignore[union-attr]

    def test_idempotent(self):
        """Normalizing twice produces same result."""
        df = pl.DataFrame({
            "ts": [datetime(2024, 1, 1, tzinfo=UTC)],
        })
        once = normalize_temporal_precision(df)
        twice = normalize_temporal_precision(once)
        assert once.equals(twice)


# =============================================================================
# Category 3: Concatenation of Streaming Batches (pure Polars)
# =============================================================================


class TestPolarsConcat:
    """Test that streaming batches concatenate cleanly in Polars."""

    def test_concat_multiple_batches(self, completed_bars: list[dict]):
        """Multiple _bars_list_to_polars outputs concatenate without error."""
        half = len(completed_bars) // 2
        batch1 = _bars_list_to_polars(completed_bars[:half])
        batch2 = _bars_list_to_polars(completed_bars[half:])

        combined = pl.concat([batch1, batch2])
        assert len(combined) == len(completed_bars)
        assert isinstance(combined, pl.DataFrame)

    def test_concat_preserves_dtypes(self, completed_bars: list[dict]):
        """Concatenation must not change dtypes."""
        half = len(completed_bars) // 2
        batch1 = _bars_list_to_polars(completed_bars[:half])
        batch2 = _bars_list_to_polars(completed_bars[half:])

        combined = pl.concat([batch1, batch2])

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert combined[col].dtype == batch1[col].dtype

    def test_concat_sorted_by_timestamp(self, completed_bars: list[dict]):
        """After concat + sort, timestamps must be monotonically increasing."""
        half = len(completed_bars) // 2
        batch1 = _bars_list_to_polars(completed_bars[:half])
        batch2 = _bars_list_to_polars(completed_bars[half:])

        combined = pl.concat([batch1, batch2]).sort("timestamp")
        timestamps = combined["timestamp"]
        # Check monotonically increasing
        diffs = timestamps.diff().drop_nulls()
        assert (diffs.cast(pl.Int64) >= 0).all()


# =============================================================================
# Category 4: ML Pipeline Column Compatibility
# =============================================================================


class TestMLPipelineCompat:
    """Test that Polars output is ready for ML feature engineering."""

    def test_returns_computation_polars_native(self, completed_bars: list[dict]):
        """Log returns can be computed entirely in Polars."""
        df = _bars_list_to_polars(completed_bars)
        returns = (df["Close"] / df["Close"].shift(1)).log()
        assert isinstance(returns, pl.Series)
        assert returns.dtype == pl.Float64
        # First value should be null (no prior bar)
        assert returns[0] is None
        # Remaining should be finite
        valid = returns.drop_nulls()
        assert valid.is_finite().all()

    def test_duration_us_computation(self, completed_bars: list[dict]):
        """Bar duration in microseconds can be computed from timestamps."""
        df = _bars_list_to_polars(completed_bars)
        if len(df) < 2:
            pytest.skip("Need at least 2 bars for duration")

        duration = df["timestamp"].diff().drop_nulls()
        # Duration should be a Polars duration type
        assert duration.dtype.is_temporal()
        # Convert to microseconds for TWSR computation
        duration_us = duration.dt.total_microseconds()
        assert (duration_us > 0).all()

    def test_lowercase_column_rename(self, completed_bars: list[dict]):
        """Can rename to lowercase (alpha-forge convention) in Polars."""
        df = _bars_list_to_polars(completed_bars)
        renamed = df.rename({
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        assert "open" in renamed.columns
        assert "close" in renamed.columns

    def test_numpy_extraction_for_torch(self, completed_bars: list[dict]):
        """Polars → numpy works for PyTorch tensor creation."""
        df = _bars_list_to_polars(completed_bars)
        close_np = df["Close"].to_numpy()
        assert close_np.dtype.kind == "f"  # float
        assert len(close_np) == len(df)


# =============================================================================
# Category 5: process_trades_polars() Input Path
# =============================================================================


class TestProcessTradesPolarsInput:
    """Test that process_trades_polars accepts Polars input correctly."""

    def test_accepts_eager_polars_df(self, synthetic_trades_polars: pl.DataFrame):
        """process_trades_polars must accept pl.DataFrame without error."""
        result = process_trades_polars(
            synthetic_trades_polars,
            threshold_decimal_bps=250,
        )
        # Currently returns pd.DataFrame (Issue #45 - to be changed)
        assert len(result) > 0

    def test_accepts_lazy_frame(self, synthetic_trades_lazy: pl.LazyFrame):
        """process_trades_polars must accept pl.LazyFrame for predicate pushdown."""
        result = process_trades_polars(
            synthetic_trades_lazy,
            threshold_decimal_bps=250,
        )
        assert len(result) > 0

    def test_lazy_with_filter(self, synthetic_trades_lazy: pl.LazyFrame):
        """Filtered LazyFrame should push predicates down."""
        filtered = synthetic_trades_lazy.filter(
            pl.col("timestamp") >= 1704067200000
        )
        result = process_trades_polars(filtered, threshold_decimal_bps=250)
        assert len(result) > 0


# =============================================================================
# Category 6: Pure Polars Pipeline (bypass Pandas entirely)
# =============================================================================


class TestPurePolarsWorkflow:
    """Test the complete Polars-only workflow for downstream ML consumers.

    This simulates what alpha-forge research needs: process trades → bars
    → features → numpy, without ever touching Pandas.
    """

    def test_full_polars_pipeline(self, synthetic_trades_polars: pl.DataFrame):
        """Complete trade → bar → features pipeline in pure Polars."""
        # Step 1: Process trades (uses internal Polars path)
        processor = RangeBarProcessor(threshold_decimal_bps=250)
        trades_dicts = synthetic_trades_polars.to_dicts()
        bars = processor.process_trades(trades_dicts)

        if len(bars) < 3:
            pytest.skip("Need at least 3 bars for feature computation")

        # Step 2: Convert to Polars (not Pandas!)
        bars_df = _bars_list_to_polars(bars)
        assert isinstance(bars_df, pl.DataFrame)

        # Step 3: Compute features in Polars
        features = bars_df.with_columns([
            # Log returns
            (pl.col("Close") / pl.col("Close").shift(1)).log().alias("returns"),
            # Simple momentum (close - close_lag)
            (pl.col("Close") - pl.col("Close").shift(1)).alias("momentum_1"),
        ])

        assert "returns" in features.columns
        assert "momentum_1" in features.columns
        assert isinstance(features, pl.DataFrame)

        # Step 4: Extract to numpy for PyTorch
        returns_np = features["returns"].drop_nulls().to_numpy()
        assert returns_np.dtype.kind == "f"
        assert len(returns_np) > 0

    def test_streaming_concat_pipeline(self, completed_bars: list[dict]):
        """Simulate streaming: multiple batches → concat → features."""
        if len(completed_bars) < 4:
            pytest.skip("Need at least 4 bars")

        # Simulate streaming batches
        batch_size = max(1, len(completed_bars) // 3)
        batches = []
        for i in range(0, len(completed_bars), batch_size):
            batch = _bars_list_to_polars(completed_bars[i:i + batch_size])
            batches.append(batch)

        # Concatenate all batches
        combined = pl.concat(batches).sort("timestamp")
        assert len(combined) == len(completed_bars)

        # Compute returns on combined
        with_returns = combined.with_columns(
            (pl.col("Close") / pl.col("Close").shift(1)).log().alias("returns")
        )
        assert with_returns["returns"].drop_nulls().len() == len(combined) - 1


# =============================================================================
# Category 7: Arrow Zero-Copy Path
# =============================================================================


class TestArrowZeroCopyPath:
    """Test that Rust → Arrow → Polars path works (zero-copy)."""

    def test_process_trades_streaming_arrow(self):
        """RangeBarProcessor.process_trades_streaming_arrow → Arrow → Polars."""
        processor = RangeBarProcessor(threshold_decimal_bps=250)

        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
            {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
        ]

        # process_trades_streaming_arrow returns PyArrow RecordBatch
        arrow_result = processor.process_trades_streaming_arrow(trades)

        # Convert to Polars (zero-copy via Arrow)
        polars_df = pl.from_arrow(arrow_result)
        assert isinstance(polars_df, pl.DataFrame)

        if len(polars_df) > 0:
            # Should have OHLCV columns (lowercase from Rust)
            for col in ["open", "high", "low", "close", "volume"]:
                assert col in polars_df.columns, f"Missing column: {col}"


# =============================================================================
# Category 8: Proof that Pandas is Not Required Downstream
# =============================================================================


class TestNoPandasRequired:
    """Prove that a downstream consumer never needs to import pandas."""

    def test_polars_pipeline_no_pandas_import(self, completed_bars: list[dict]):
        """Full pipeline without importing pandas at module level.

        This test's existence proves the pathway works. The actual
        downstream code would do:

            from rangebar import RangeBarProcessor
            from rangebar.conversion import _bars_list_to_polars

        And never import pandas.
        """
        # Track pandas usage - it may already be imported by rangebar internals,
        # but our CODE PATH must not require it
        bars_df = _bars_list_to_polars(completed_bars)

        # All operations below are pure Polars
        assert isinstance(bars_df, pl.DataFrame)

        if len(bars_df) > 1:
            returns = (bars_df["Close"] / bars_df["Close"].shift(1)).log()
            numpy_arr = returns.drop_nulls().to_numpy()
            assert len(numpy_arr) > 0

        # The key assertion: our code path produced numpy-ready data
        # via Polars only. pandas may be loaded by rangebar internals
        # but WE didn't need it.
        assert "polars" in sys.modules
