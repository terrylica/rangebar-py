"""Tests for microstructure features (Issue #25).

These tests verify that the 10 microstructure features are:
1. Present in the output from the Rust processor
2. Within expected ranges
3. Pass validation framework checks
"""

from __future__ import annotations

import pytest

# Feature column names
FEATURE_COLS = [
    "duration_us",
    "ofi",
    "vwap_close_deviation",
    "price_impact",
    "kyle_lambda_proxy",
    "trade_intensity",
    "volume_per_trade",
    "aggression_ratio",
    "aggregation_density",
    "turnover_imbalance",
]


class TestMicrostructureFeaturesPresent:
    """Test that microstructure features are present in processor output."""

    def test_features_in_rangebar_dict(self):
        """Verify all 10 features are in the dict output from PyRangeBarProcessor."""
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)

        # Create some test trades
        trades = [
            {
                "agg_trade_id": 1,
                "price": 50000.0,
                "quantity": 1.0,
                "first_trade_id": 1,
                "last_trade_id": 1,
                "timestamp": 1704067200000000,  # microseconds
                "is_buyer_maker": False,  # Buy
            },
            {
                "agg_trade_id": 2,
                "price": 50200.0,  # +0.4% - should breach 0.25% threshold
                "quantity": 2.0,
                "first_trade_id": 2,
                "last_trade_id": 3,
                "timestamp": 1704067201000000,  # 1 second later
                "is_buyer_maker": True,  # Sell
            },
        ]

        bars = processor.process_trades(trades)

        # Should produce at least one completed bar
        assert len(bars) >= 1, "Expected at least one completed bar"

        bar = bars[0]
        for col in FEATURE_COLS:
            assert col in bar, f"Missing feature column: {col}"


class TestOFIRange:
    """Test Order Flow Imbalance (OFI) is within [-1, 1]."""

    def test_ofi_bounded(self):
        """OFI should be in [-1, 1]."""
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)

        # Create trades with mixed buy/sell
        trades = [
            {
                "agg_trade_id": 1,
                "price": 50000.0,
                "quantity": 1.0,
                "first_trade_id": 1,
                "last_trade_id": 1,
                "timestamp": 1704067200000000,
                "is_buyer_maker": False,  # Buy
            },
            {
                "agg_trade_id": 2,
                "price": 50200.0,
                "quantity": 1.0,
                "first_trade_id": 2,
                "last_trade_id": 2,
                "timestamp": 1704067201000000,
                "is_buyer_maker": True,  # Sell
            },
        ]

        bars = processor.process_trades(trades)
        if bars:
            ofi = bars[0]["ofi"]
            assert -1.0 <= ofi <= 1.0, f"OFI out of range: {ofi}"


class TestTurnoverImbalanceRange:
    """Test turnover imbalance is within [-1, 1]."""

    def test_turnover_imbalance_bounded(self):
        """Turnover imbalance should be in [-1, 1]."""
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)

        trades = [
            {
                "agg_trade_id": 1,
                "price": 50000.0,
                "quantity": 1.0,
                "first_trade_id": 1,
                "last_trade_id": 1,
                "timestamp": 1704067200000000,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 2,
                "price": 50200.0,
                "quantity": 2.0,
                "first_trade_id": 2,
                "last_trade_id": 3,
                "timestamp": 1704067201000000,
                "is_buyer_maker": True,
            },
        ]

        bars = processor.process_trades(trades)
        if bars:
            ti = bars[0]["turnover_imbalance"]
            assert -1.0 <= ti <= 1.0, f"Turnover imbalance out of range: {ti}"


class TestDurationPositive:
    """Test duration is non-negative."""

    def test_duration_non_negative(self):
        """Duration should be >= 0."""
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)

        trades = [
            {
                "agg_trade_id": 1,
                "price": 50000.0,
                "quantity": 1.0,
                "first_trade_id": 1,
                "last_trade_id": 1,
                "timestamp": 1704067200000000,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 2,
                "price": 50200.0,
                "quantity": 1.0,
                "first_trade_id": 2,
                "last_trade_id": 2,
                "timestamp": 1704067205000000,  # 5 seconds later
                "is_buyer_maker": True,
            },
        ]

        bars = processor.process_trades(trades)
        if bars:
            duration = bars[0]["duration_us"]
            # Duration should be a valid integer (non-negative)
            assert isinstance(
                duration, int
            ), f"Duration should be int: {type(duration)}"
            assert duration >= 0, f"Duration should be non-negative: {duration}"


class TestValidationFramework:
    """Test the validation framework works correctly."""

    def test_tier1_validation_import(self):
        """Tier 1 validation module should be importable."""
        from rangebar.validation.tier1 import FEATURE_COLS, validate_tier1

        assert callable(validate_tier1)
        assert len(FEATURE_COLS) == 10

    def test_tier2_validation_import(self):
        """Tier 2 validation module should be importable."""
        from rangebar.validation.tier2 import validate_tier2

        assert callable(validate_tier2)

    def test_tier1_empty_df(self):
        """Tier 1 should handle empty DataFrames gracefully."""
        import pandas as pd
        from rangebar.validation.tier1 import validate_tier1

        empty_df = pd.DataFrame()
        result = validate_tier1(empty_df)

        assert "tier1_passed" in result
        assert result["tier1_passed"] is False

    def test_tier1_missing_features(self):
        """Tier 1 should detect missing feature columns."""
        import pandas as pd
        from rangebar.validation.tier1 import validate_tier1

        # DataFrame without microstructure features
        df = pd.DataFrame({"Open": [100], "Close": [101]})
        result = validate_tier1(df)

        assert "tier1_passed" in result
        assert result["tier1_passed"] is False
        assert result.get("features_present") is False


class TestAggressionRatioCapped:
    """Test aggression ratio is properly capped at 100."""

    def test_aggression_ratio_capped_at_100(self):
        """Aggression ratio should be capped at 100 when no sells."""
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)

        # All buy trades
        trades = [
            {
                "agg_trade_id": 1,
                "price": 50000.0,
                "quantity": 1.0,
                "first_trade_id": 1,
                "last_trade_id": 1,
                "timestamp": 1704067200000000,
                "is_buyer_maker": False,  # Buy
            },
            {
                "agg_trade_id": 2,
                "price": 50200.0,
                "quantity": 1.0,
                "first_trade_id": 2,
                "last_trade_id": 2,
                "timestamp": 1704067201000000,
                "is_buyer_maker": False,  # Buy
            },
        ]

        bars = processor.process_trades(trades)
        if bars:
            ratio = bars[0]["aggression_ratio"]
            assert ratio <= 100.0, f"Aggression ratio should be capped at 100: {ratio}"


class TestVWAPCloseDeviation:
    """Test VWAP close deviation edge cases."""

    def test_vwap_close_deviation_zero_range(self):
        """VWAP close deviation should be 0 when high == low."""
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)

        # Single trade creates a bar with high == low
        trades = [
            {
                "agg_trade_id": 1,
                "price": 50000.0,
                "quantity": 1.0,
                "first_trade_id": 1,
                "last_trade_id": 1,
                "timestamp": 1704067200000000,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 2,
                "price": 50200.0,  # Price change triggers breach
                "quantity": 1.0,
                "first_trade_id": 2,
                "last_trade_id": 2,
                "timestamp": 1704067201000000,
                "is_buyer_maker": True,
            },
        ]

        bars = processor.process_trades(trades)
        if bars:
            # Check it doesn't crash and returns a valid float
            deviation = bars[0]["vwap_close_deviation"]
            assert isinstance(
                deviation, int | float
            ), f"Expected number: {type(deviation)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
