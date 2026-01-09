"""E2E tests for Python pipeline optimization (real data, no fakes)."""

from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from rangebar import (
    process_trades_chunked,
    process_trades_polars,
    process_trades_to_dataframe,
)
from rangebar.storage import TickStorage


class TestE2ELazyParquetLoading:
    """E2E tests for Phase 1: Lazy Parquet loading."""

    def test_read_ticks_uses_lazy_scan(self, tmp_path: Path) -> None:
        """Verify lazy loading with predicate pushdown."""
        storage = TickStorage(cache_dir=tmp_path)

        # Create real tick data
        ticks = pl.DataFrame(
            {
                "timestamp": [1704067200000 + i * 1000 for i in range(1000)],
                "price": [42000.0 + i * 0.1 for i in range(1000)],
                "quantity": [1.0] * 1000,
            }
        )

        # Write to storage
        count = storage.write_ticks("BTCUSDT", ticks)
        assert count == 1000

        # Read with filter (should use predicate pushdown)
        start_ts = 1704067200000 + 100 * 1000  # Skip first 100
        end_ts = 1704067200000 + 200 * 1000  # Take 100

        filtered = storage.read_ticks("BTCUSDT", start_ts, end_ts)

        # Verify filtered read
        assert len(filtered) == 101  # inclusive range
        assert filtered["timestamp"].min() >= start_ts
        assert filtered["timestamp"].max() <= end_ts


class TestE2EProcessTradesChunked:
    """E2E tests for Phase 2: Chunked processing."""

    def test_chunked_processing_produces_bars(self) -> None:
        """Verify chunked processing produces bars from real data."""
        # Create real trade data with significant price movement
        # 25bps of $42000 = $105, so we need moves > $105 to breach threshold
        trades = []
        base_price = 42000.0
        for i in range(500):
            # Alternate price to breach threshold every few trades
            if i % 10 < 5:
                price = base_price + (i // 10) * 150  # Move up $150 per cycle
            else:
                price = base_price + (i // 10) * 150 + 50
            trades.append(
                {"timestamp": 1704067200000 + i * 100, "price": price, "quantity": 1.0}
            )

        # Process chunked (small chunks to test)
        bars_chunks = list(
            process_trades_chunked(
                iter(trades), threshold_decimal_bps=250, chunk_size=100
            )
        )

        # Should produce bars from chunked processing
        assert len(bars_chunks) >= 1, "Should produce at least one chunk with bars"
        for chunk_df in bars_chunks:
            assert list(chunk_df.columns) == ["Open", "High", "Low", "Close", "Volume"]
            assert isinstance(chunk_df.index, pd.DatetimeIndex)


class TestE2EProcessTradesPolars:
    """E2E tests for Phase 4: process_trades_polars()."""

    def test_polars_dataframe_processing(self) -> None:
        """Test processing Polars DataFrame directly."""
        # Create real data as Polars DataFrame with price breach
        trades = pl.DataFrame(
            {
                "timestamp": [
                    1704067200000,
                    1704067210000,
                    1704067220000,
                    1704067230000,
                ],
                "price": [
                    42000.0,
                    42050.0,
                    42150.0,
                    42200.0,
                ],  # $150 move = breach at 25bps
                "quantity": [1.5, 2.0, 1.0, 3.0],
            }
        )

        # Process using optimized function
        df = process_trades_polars(trades, threshold_decimal_bps=250)

        # Verify output format
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df["Open"].dtype == "float64"

    def test_polars_lazyframe_processing(self) -> None:
        """Test processing Polars LazyFrame (lazy eval)."""
        # Create as LazyFrame (simulates scan_parquet) with price breach
        trades_lazy = pl.DataFrame(
            {
                "timestamp": [
                    1704067200000,
                    1704067210000,
                    1704067220000,
                    1704067230000,
                ],
                "price": [42000.0, 42050.0, 42150.0, 42200.0],  # Breach threshold
                "quantity": [1.5, 2.0, 1.0, 3.0],
            }
        ).lazy()

        # Process using optimized function
        df = process_trades_polars(trades_lazy, threshold_decimal_bps=250)

        # Verify output format
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_polars_with_volume_column(self) -> None:
        """Test Polars processing with 'volume' instead of 'quantity'."""
        trades = pl.DataFrame(
            {
                "timestamp": [1704067200000, 1704067210000, 1704067220000],
                "price": [42000.0, 42110.0, 42200.0],  # Breach threshold
                "volume": [1.5, 2.0, 1.0],  # Note: 'volume' not 'quantity'
            }
        )

        # Should handle 'volume' column
        df = process_trades_polars(trades, threshold_decimal_bps=250)
        assert "Volume" in df.columns


class TestE2EDictOptimization:
    """E2E tests for Phase 3: Dict conversion optimization."""

    def test_minimal_column_extraction(self) -> None:
        """Verify only required columns are extracted."""
        # DataFrame with extra columns
        trades = pl.DataFrame(
            {
                "timestamp": [1704067200000, 1704067210000, 1704067220000],
                "price": [42000.0, 42110.0, 42200.0],  # Breach threshold
                "quantity": [1.5, 2.0, 1.0],
                "extra_col_1": ["a", "b", "c"],
                "extra_col_2": [1, 2, 3],
                "extra_col_3": [True, False, True],
            }
        )

        # Process - should only use timestamp, price, quantity
        df = process_trades_polars(trades, threshold_decimal_bps=250)

        # Verify processing succeeded despite extra columns
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


class TestE2EAPICompatibility:
    """E2E tests for API backward compatibility."""

    def test_process_trades_to_dataframe_unchanged(self) -> None:
        """Verify original API still works."""
        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
            {"timestamp": 1704067210000, "price": 42110.0, "quantity": 2.0},  # Breach
        ]

        df = process_trades_to_dataframe(trades, threshold_decimal_bps=250)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
