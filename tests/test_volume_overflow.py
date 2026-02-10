"""Tests for volume overflow fix (Issue #88).

FixedPoint(i64) with SCALE=100_000_000 overflows when cumulative bar volume
exceeds 92.23 billion tokens. SHIBUSDT (price ~$0.00003) regularly hits this
at larger thresholds. The fix widens volume/buy_volume/sell_volume to i128.

These tests verify:
1. Volume stays positive with SHIB-scale quantities (50B+ tokens)
2. OFI remains bounded [-1, 1]
3. buy_volume + sell_volume == volume (conservation)
4. VWAP is between open and close prices
5. Tier 1 validation catches negative volumes
"""

from __future__ import annotations

import pytest


def _make_shib_trades(n_buys: int = 3, n_sells: int = 2) -> list[dict]:
    """Create synthetic SHIB-like trades with extreme quantities.

    Price ~0.00003, quantity 50 billion tokens per trade.
    Total volume will exceed i64::MAX / SCALE = 92.23B tokens.
    """
    trades = []
    base_price = 0.00003
    base_ts = 1704067200000000  # microseconds

    # Buy trades
    for i in range(n_buys):
        trades.append({
            "agg_trade_id": i + 1,
            "price": base_price + (i * 0.0000001),  # tiny price increment
            "quantity": 50_000_000_000.0,  # 50B tokens per trade
            "first_trade_id": i + 1,
            "last_trade_id": i + 1,
            "timestamp": base_ts + (i * 1_000_000),  # 1s apart
            "is_buyer_maker": False,  # Buy
        })

    # Sell trades (breach threshold to close bar)
    for i in range(n_sells):
        idx = n_buys + i
        trades.append({
            "agg_trade_id": idx + 1,
            "price": base_price * 1.005,  # +0.5% to breach 0.25% threshold
            "quantity": 50_000_000_000.0,  # 50B tokens
            "first_trade_id": idx + 1,
            "last_trade_id": idx + 1,
            "timestamp": base_ts + (idx * 1_000_000),
            "is_buyer_maker": True,  # Sell
        })

    return trades


class TestVolumeNoOverflow:
    """Volume must stay positive even with SHIB-scale quantities."""

    def test_volume_positive_with_large_trades(self):
        """Volume > 0 with 250B+ total tokens (exceeds i64::MAX / SCALE)."""
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)
        bars = processor.process_trades(_make_shib_trades())

        assert len(bars) >= 1, "Expected at least one bar"
        bar = bars[0]

        assert bar["volume"] > 0, f"Volume should be positive: {bar['volume']}"
        assert bar["buy_volume"] >= 0, f"buy_volume negative: {bar['buy_volume']}"
        assert bar["sell_volume"] >= 0, f"sell_volume negative: {bar['sell_volume']}"

    def test_volume_conservation(self):
        """buy_volume + sell_volume should equal total volume."""
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)
        bars = processor.process_trades(_make_shib_trades())

        assert len(bars) >= 1
        bar = bars[0]

        total = bar["buy_volume"] + bar["sell_volume"]
        assert abs(total - bar["volume"]) < 1e-6, (
            f"Volume mismatch: buy={bar['buy_volume']} + sell={bar['sell_volume']} "
            f"= {total}, but volume={bar['volume']}"
        )


class TestOFIBoundedWithLargeVolume:
    """OFI must remain in [-1, 1] even with i128-scale volumes."""

    def test_ofi_bounded(self):
        """OFI in [-1, 1] with extreme quantities."""
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)
        bars = processor.process_trades(_make_shib_trades())

        assert len(bars) >= 1
        ofi = bars[0]["ofi"]
        assert -1.0 <= ofi <= 1.0, f"OFI out of range: {ofi}"


class TestVWAPWithLargeVolume:
    """VWAP should be between open and close with large volumes."""

    def test_vwap_between_open_close(self):
        """VWAP is a valid price (between low and high of bar)."""
        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)
        bars = processor.process_trades(_make_shib_trades())

        assert len(bars) >= 1
        bar = bars[0]

        vwap = bar["vwap"]
        low = bar["low"]
        high = bar["high"]
        assert low <= vwap <= high, f"VWAP {vwap} not in [{low}, {high}]"

    def test_vwap_not_nan(self):
        """VWAP should not be NaN (division by zero was possible with old code)."""
        import math

        from rangebar._core import PyRangeBarProcessor

        processor = PyRangeBarProcessor(threshold_decimal_bps=250)
        bars = processor.process_trades(_make_shib_trades())

        assert len(bars) >= 1
        assert not math.isnan(bars[0]["vwap"]), "VWAP is NaN"


class TestTier1VolumeOverflowDetection:
    """Tier 1 validation should catch negative volumes (overflow indicator)."""

    def test_tier1_rejects_negative_volume(self):
        """Tier 1 validation fails when volume < 0."""
        import pandas as pd
        from rangebar.validation.tier1 import validate_tier1

        # Simulate corrupted data with negative volume (old i64 overflow)
        df = pd.DataFrame({
            "open": [0.00003],
            "high": [0.000031],
            "low": [0.000029],
            "close": [0.000031],
            "volume": [-1.23e15],  # Negative = overflow!
            "buy_volume": [1.0e15],
            "sell_volume": [1.0e15],
            "duration_us": [1000000],
            "ofi": [0.0],
            "vwap_close_deviation": [0.1],
            "price_impact": [0.001],
            "kyle_lambda_proxy": [0.5],
            "trade_intensity": [100.0],
            "volume_per_trade": [1000.0],
            "aggression_ratio": [50.0],
            "aggregation_density": [2.0],
            "turnover_imbalance": [0.1],
        })

        result = validate_tier1(df)
        assert result["tier1_passed"] is False, (
            "Tier 1 should FAIL with negative volume (overflow indicator)"
        )

    def test_tier1_passes_positive_volume(self):
        """Tier 1 validation passes when volumes are all positive."""
        import pandas as pd
        from rangebar.validation.tier1 import validate_tier1

        df = pd.DataFrame({
            "open": [0.00003],
            "high": [0.000031],
            "low": [0.000029],
            "close": [0.000031],
            "volume": [1.0e15],
            "buy_volume": [0.5e15],
            "sell_volume": [0.5e15],
            "duration_us": [1000000],
            "ofi": [0.0],
            "vwap_close_deviation": [0.1],
            "price_impact": [0.001],
            "kyle_lambda_proxy": [0.5],
            "trade_intensity": [100.0],
            "volume_per_trade": [1000.0],
            "aggression_ratio": [50.0],
            "aggregation_density": [2.0],
            "turnover_imbalance": [0.1],
        })

        result = validate_tier1(df)
        assert result["tier1_passed"] is True, (
            f"Tier 1 should pass with positive volumes: {result}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
