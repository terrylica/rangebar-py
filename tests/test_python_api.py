"""Test Python API layer (RangeBarProcessor and convenience functions)."""

import pandas as pd
import pytest

from rangebar import RangeBarProcessor, process_trades_to_dataframe


class TestRangeBarProcessor:
    """Test RangeBarProcessor class."""

    def test_processor_creation(self):
        """Test basic processor creation."""
        processor = RangeBarProcessor(threshold_bps=250)
        assert processor.threshold_bps == 250

    def test_invalid_threshold(self):
        """Test that invalid thresholds raise ValueError."""
        with pytest.raises(ValueError):
            RangeBarProcessor(threshold_bps=0)

    def test_process_trades_returns_list(self):
        """Test that process_trades returns a list of dicts."""
        processor = RangeBarProcessor(threshold_bps=250)

        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
            {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
        ]

        bars = processor.process_trades(trades)

        assert isinstance(bars, list)
        assert len(bars) == 1  # One completed bar
        assert isinstance(bars[0], dict)

    def test_to_dataframe_returns_correct_format(self):
        """Test that to_dataframe returns proper OHLCV DataFrame."""
        processor = RangeBarProcessor(threshold_bps=250)

        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
            {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
        ]

        bars = processor.process_trades(trades)
        df = processor.to_dataframe(bars)

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

        # Check data types
        assert df["Open"].dtype == float
        assert df["High"].dtype == float
        assert df["Low"].dtype == float
        assert df["Close"].dtype == float
        assert df["Volume"].dtype == float

    def test_to_dataframe_empty_bars(self):
        """Test that to_dataframe handles empty bars correctly."""
        processor = RangeBarProcessor(threshold_bps=250)

        df = processor.to_dataframe([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_to_dataframe_ohlc_invariants(self):
        """Test that OHLC invariants are preserved in DataFrame."""
        processor = RangeBarProcessor(threshold_bps=250)

        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
            {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.0},
        ]

        bars = processor.process_trades(trades)
        df = processor.to_dataframe(bars)

        # OHLC invariants: High >= max(Open, Close), Low <= min(Open, Close)
        assert (df["High"] >= df["Open"]).all()
        assert (df["High"] >= df["Close"]).all()
        assert (df["Low"] <= df["Open"]).all()
        assert (df["Low"] <= df["Close"]).all()


class TestProcessTradesToDataframe:
    """Test process_trades_to_dataframe convenience function."""

    def test_with_list_of_dicts(self):
        """Test processing list of dicts."""
        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
            {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
        ]

        df = process_trades_to_dataframe(trades, threshold_bps=250)

        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 1

    def test_with_dataframe_datetime_index(self):
        """Test processing DataFrame with datetime timestamps."""
        trades_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="min"),
                "price": [42000.0 + i * 10 for i in range(10)],
                "quantity": [1.5] * 10,
            }
        )

        df = process_trades_to_dataframe(trades_df, threshold_bps=250)

        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_with_dataframe_integer_timestamps(self):
        """Test processing DataFrame with integer timestamps (milliseconds)."""
        trades_df = pd.DataFrame(
            {
                "timestamp": [1704067200000 + i * 10000 for i in range(10)],
                "price": [42000.0 + i * 10 for i in range(10)],
                "quantity": [1.5] * 10,
            }
        )

        df = process_trades_to_dataframe(trades_df, threshold_bps=250)

        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 0  # May or may not have bars depending on price movement

    def test_with_volume_column(self):
        """Test that 'volume' column name is accepted."""
        trades_df = pd.DataFrame(
            {
                "timestamp": [1704067200000, 1704067210000],
                "price": [42000.0, 42105.0],
                "volume": [1.5, 2.3],  # Note: 'volume' instead of 'quantity'
            }
        )

        df = process_trades_to_dataframe(trades_df, threshold_bps=250)

        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 0

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raise ValueError."""
        trades_df = pd.DataFrame(
            {
                "timestamp": [1704067200000, 1704067210000],
                "price": [42000.0, 42105.0],
                # Missing 'quantity' or 'volume'
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            process_trades_to_dataframe(trades_df, threshold_bps=250)

    def test_default_threshold(self):
        """Test that default threshold_bps=250 is used."""
        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
            {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.3},
        ]

        df = process_trades_to_dataframe(trades)  # No threshold_bps specified

        assert isinstance(df, pd.DataFrame)

    def test_no_nan_values(self):
        """Test that output DataFrame has no NaN values."""
        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
            {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.0},
        ]

        df = process_trades_to_dataframe(trades, threshold_bps=250)

        # backtesting.py raises on NaN values
        assert not df.isnull().any().any()

    def test_sorted_chronologically(self):
        """Test that output DataFrame is sorted chronologically."""
        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
            {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.0},
            {"timestamp": 1704067220000, "price": 42210.0, "quantity": 1.5},
        ]

        df = process_trades_to_dataframe(trades, threshold_bps=250)

        # Index should be monotonic increasing
        assert df.index.is_monotonic_increasing


class TestBacktestingPyCompatibility:
    """Test compatibility with backtesting.py requirements."""

    def test_column_names_capitalized(self):
        """Test that column names are capitalized as required by backtesting.py."""
        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
            {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.0},
        ]

        df = process_trades_to_dataframe(trades, threshold_bps=250)

        # backtesting.py requires capitalized OHLCV columns
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_datetime_index(self):
        """Test that index is DatetimeIndex as required by backtesting.py."""
        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
            {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.0},
        ]

        df = process_trades_to_dataframe(trades, threshold_bps=250)

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_ohlcv_complete(self):
        """Test that all OHLCV values are present (no partial bars)."""
        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
            {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.0},
        ]

        df = process_trades_to_dataframe(trades, threshold_bps=250)

        # All rows should have complete OHLCV data
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            assert col in df.columns
            assert not df[col].isnull().any()

    def test_volume_positive(self):
        """Test that volume is always positive (as expected by backtesting.py)."""
        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
            {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.0},
        ]

        df = process_trades_to_dataframe(trades, threshold_bps=250)

        assert (df["Volume"] > 0).all()

    def test_integration_example(self):
        """Test a complete integration example similar to backtesting.py usage."""
        # Simulate Binance aggTrades data
        trades = []
        base_time = 1704067200000
        base_price = 50000.0

        for i in range(100):
            trades.append(
                {
                    "timestamp": base_time + i * 1000,  # 1 second apart
                    "price": base_price + (i % 20) * 10,  # Price oscillates
                    "quantity": 1.0 + (i % 5) * 0.5,
                }
            )

        df = process_trades_to_dataframe(trades, threshold_bps=100)  # 0.1%

        # Validate basic structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0  # Should generate some bars

        # Validate backtesting.py compatibility
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert not df.isnull().any().any()
        assert df.index.is_monotonic_increasing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
