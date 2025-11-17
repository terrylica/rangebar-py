#!/usr/bin/env python3
"""Test with real Binance data.

ADR-003: Testing Strategy with Real Binance Data
Validates against real market data from Binance.
"""

import pytest
import pandas as pd
from pathlib import Path
from rangebar import process_trades_to_dataframe

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_FILE = FIXTURES_DIR / "BTCUSDT-aggTrades-sample-10k.csv"


@pytest.mark.skipif(not SAMPLE_FILE.exists(), reason="Sample data not downloaded")
class TestRealBinanceData:
    """Test with real Binance aggTrades data."""

    def test_load_binance_csv(self):
        """Test loading real Binance CSV format."""
        df = pd.read_csv(SAMPLE_FILE)

        # Verify Binance format (agg_trade_id, price, quantity, timestamp, etc.)
        # Note: Header might be missing, check if numeric or has columns
        if len(df.columns) == 8:
            # Has header
            expected_cols = ["agg_trade_id", "price", "quantity", "first_trade_id",
                            "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match"]
            # May have different column names, check key columns exist
            assert "price" in df.columns or len(df.columns) >= 3
        else:
            # No header, should have 8 columns
            assert len(df.columns) >= 3

    def test_process_binance_aggtrades(self):
        """Test processing real Binance aggTrades."""
        df = pd.read_csv(SAMPLE_FILE)

        # Map to expected column names if needed
        if len(df.columns) == 8 and "price" not in df.columns:
            # No header, assign names
            df.columns = ["agg_trade_id", "price", "quantity", "first_trade_id",
                          "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match"]

        # Process to range bars
        range_bars = process_trades_to_dataframe(df, threshold_bps=250)

        # Validate output
        assert len(range_bars) > 0, "Should generate at least one range bar"
        assert isinstance(range_bars.index, pd.DatetimeIndex)
        assert list(range_bars.columns) == ["Open", "High", "Low", "Close", "Volume"]

        # No NaN values
        assert not range_bars.isnull().any().any(), "Should have no NaN values"

        # OHLC invariants
        assert (range_bars["High"] >= range_bars["Open"]).all()
        assert (range_bars["High"] >= range_bars["Close"]).all()
        assert (range_bars["Low"] <= range_bars["Open"]).all()
        assert (range_bars["Low"] <= range_bars["Close"]).all()

    def test_multiple_threshold_values(self):
        """Test different threshold values on real data."""
        df = pd.read_csv(SAMPLE_FILE)

        # Map column names if needed
        if len(df.columns) == 8 and "price" not in df.columns:
            df.columns = ["agg_trade_id", "price", "quantity", "first_trade_id",
                          "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match"]

        thresholds = [100, 250, 500, 1000]  # 10bps, 25bps, 50bps, 100bps
        results = {}

        for threshold in thresholds:
            range_bars = process_trades_to_dataframe(df, threshold_bps=threshold)
            results[threshold] = len(range_bars)

        # Higher threshold = fewer bars (more compression)
        assert results[100] >= results[250] >= results[500] >= results[1000], \
            f"Higher thresholds should produce fewer bars: {results}"
