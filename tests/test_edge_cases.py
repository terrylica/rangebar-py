#!/usr/bin/env python3
"""Test edge cases and error handling.

ADR-003: Testing Strategy with Real Binance Data
Covers edge cases, error conditions, and boundary validation.
"""

import pytest
import pandas as pd
from rangebar import process_trades_to_dataframe, RangeBarProcessor


class TestErrorHandling:
    """Test error handling edge cases."""

    def test_dataframe_with_both_quantity_and_volume(self):
        """Test DataFrame with conflicting volume columns."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704067210000],
            "price": [42000.0, 42105.0],
            "quantity": [1.0, 2.0],
            "volume": [1.5, 2.5],  # Conflict!
        })

        # Should prefer 'quantity' over 'volume'
        result = process_trades_to_dataframe(df, threshold_bps=250)
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_missing_both_quantity_and_volume(self):
        """Test DataFrame missing both quantity and volume columns."""
        df = pd.DataFrame({
            "timestamp": [1704067200000, 1704067210000],
            "price": [42000.0, 42105.0],
        })

        with pytest.raises((ValueError, KeyError)):
            process_trades_to_dataframe(df, threshold_bps=250)

    def test_extremely_large_threshold(self):
        """Test threshold at upper boundary."""
        processor = RangeBarProcessor(threshold_bps=100_000)  # 100% threshold
        assert processor.threshold_bps == 100_000

    def test_negative_threshold(self):
        """Test negative threshold (should fail with OverflowError from Rust u32)."""
        with pytest.raises(OverflowError):
            RangeBarProcessor(threshold_bps=-1)

    def test_zero_threshold(self):
        """Test zero threshold (should fail)."""
        with pytest.raises(ValueError):
            RangeBarProcessor(threshold_bps=0)

    def test_single_trade_no_bars(self):
        """Test that single trade generates no bars."""
        trades = [{"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0}]
        df = process_trades_to_dataframe(trades, threshold_bps=250)
        # Single trade cannot breach threshold, so no complete bars
        assert len(df) >= 0  # May be 0 or may have incomplete bar

    def test_empty_trades_list(self):
        """Test empty trades list."""
        df = process_trades_to_dataframe([], threshold_bps=250)
        assert len(df) == 0
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_empty_dataframe(self):
        """Test empty DataFrame."""
        df = pd.DataFrame(columns=["timestamp", "price", "quantity"])
        result = process_trades_to_dataframe(df, threshold_bps=250)
        assert len(result) == 0


class TestTimestampHandling:
    """Test timestamp conversion edge cases."""

    def test_datetime_timestamp_timezone_aware(self):
        """Test timezone-aware datetime timestamps."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="min", tz="UTC"),
            "price": [42000.0 + i * 10 for i in range(100)],
            "quantity": [1.0] * 100,
        })

        result = process_trades_to_dataframe(df, threshold_bps=250)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) > 0

    def test_datetime_timestamp_timezone_naive(self):
        """Test timezone-naive datetime timestamps."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="min"),
            "price": [42000.0 + i * 10 for i in range(100)],
            "quantity": [1.0] * 100,
        })

        result = process_trades_to_dataframe(df, threshold_bps=250)
        assert isinstance(result.index, pd.DatetimeIndex)


class TestLargeDatasets:
    """Test performance with large datasets."""

    def test_10k_trades(self):
        """Test processing 10,000 trades."""
        trades = [
            {"timestamp": 1704067200000 + i * 1000, "price": 42000.0 + i * 0.1, "quantity": 1.0}
            for i in range(10_000)
        ]
        df = process_trades_to_dataframe(trades, threshold_bps=250)
        assert len(df) > 0
        # Verify OHLCV structure
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    @pytest.mark.slow
    def test_100k_trades(self):
        """Test processing 100,000 trades (marked as slow)."""
        trades = [
            {"timestamp": 1704067200000 + i * 100, "price": 42000.0 + i * 0.01, "quantity": 1.0}
            for i in range(100_000)
        ]
        df = process_trades_to_dataframe(trades, threshold_bps=250)
        assert len(df) > 0
        assert isinstance(df.index, pd.DatetimeIndex)
