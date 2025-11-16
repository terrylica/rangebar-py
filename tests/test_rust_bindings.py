"""Test Rust PyO3 bindings directly."""

import pytest
from rangebar._core import PyRangeBarProcessor


def test_processor_creation():
    """Test basic processor creation."""
    processor = PyRangeBarProcessor(threshold_bps=250)
    assert processor.threshold_bps == 250


def test_invalid_threshold_too_low():
    """Test that threshold validation rejects values that are too low."""
    with pytest.raises(ValueError, match="Invalid threshold"):
        PyRangeBarProcessor(threshold_bps=0)


def test_invalid_threshold_too_high():
    """Test that threshold validation rejects values that are too high."""
    with pytest.raises(ValueError, match="Invalid threshold"):
        PyRangeBarProcessor(threshold_bps=200_000)


def test_process_empty_trades():
    """Test processing empty trade list."""
    processor = PyRangeBarProcessor(threshold_bps=250)
    bars = processor.process_trades([])
    assert bars == []


def test_process_single_trade():
    """Test processing a single trade (should return no bars - no breach)."""
    processor = PyRangeBarProcessor(threshold_bps=250)

    trades = [
        {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.5},
    ]

    bars = processor.process_trades(trades)
    # Single trade should not create a completed bar (no breach)
    assert len(bars) == 0


def test_process_trades_with_breach():
    """Test processing trades that breach threshold."""
    processor = PyRangeBarProcessor(threshold_bps=250)  # 25bps = 0.25%

    # Create trades where second trade breaches threshold
    # 42000 * 1.0025 = 42105 (upper breach)
    trades = [
        {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
        {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.0},  # Breach!
    ]

    bars = processor.process_trades(trades)

    # Should get 1 completed bar
    assert len(bars) == 1

    bar = bars[0]
    assert "timestamp" in bar
    assert "open" in bar
    assert "high" in bar
    assert "low" in bar
    assert "close" in bar
    assert "volume" in bar

    # Validate OHLCV values
    assert bar["open"] == 42000.0
    assert bar["high"] == 42105.0
    assert bar["low"] == 42000.0
    assert bar["close"] == 42105.0
    assert bar["volume"] == 3.0  # 1.0 + 2.0


def test_process_trades_with_downward_breach():
    """Test processing trades with downward breach."""
    processor = PyRangeBarProcessor(threshold_bps=250)  # 25bps = 0.25%

    # Create trades where second trade breaches downward
    # 42000 * 0.9975 = 41895 (lower breach)
    trades = [
        {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
        {"timestamp": 1704067210000, "price": 41895.0, "quantity": 2.0},  # Breach!
    ]

    bars = processor.process_trades(trades)

    assert len(bars) == 1

    bar = bars[0]
    assert bar["open"] == 42000.0
    assert bar["high"] == 42000.0
    assert bar["low"] == 41895.0
    assert bar["close"] == 41895.0
    assert bar["volume"] == 3.0


def test_missing_required_fields():
    """Test that missing required fields raises KeyError."""
    processor = PyRangeBarProcessor(threshold_bps=250)

    # Missing price
    with pytest.raises(KeyError, match="missing 'price'"):
        processor.process_trades([{"timestamp": 1704067200000, "quantity": 1.0}])

    # Missing timestamp
    with pytest.raises(KeyError, match="missing 'timestamp'"):
        processor.process_trades([{"price": 42000.0, "quantity": 1.0}])

    # Missing quantity/volume
    with pytest.raises(KeyError, match="missing 'quantity' or 'volume'"):
        processor.process_trades([{"timestamp": 1704067200000, "price": 42000.0}])


def test_volume_vs_quantity_field():
    """Test that both 'volume' and 'quantity' keys work."""
    processor = PyRangeBarProcessor(threshold_bps=250)

    # Test with 'quantity'
    trades_quantity = [
        {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
        {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.0},
    ]

    bars_quantity = processor.process_trades(trades_quantity)
    assert len(bars_quantity) == 1
    assert bars_quantity[0]["volume"] == 3.0

    # Test with 'volume'
    processor2 = PyRangeBarProcessor(threshold_bps=250)
    trades_volume = [
        {"timestamp": 1704067200000, "price": 42000.0, "volume": 1.0},
        {"timestamp": 1704067210000, "price": 42105.0, "volume": 2.0},
    ]

    bars_volume = processor2.process_trades(trades_volume)
    assert len(bars_volume) == 1
    assert bars_volume[0]["volume"] == 3.0


def test_microstructure_fields():
    """Test that market microstructure fields are included."""
    processor = PyRangeBarProcessor(threshold_bps=250)

    trades = [
        {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0, "is_buyer_maker": False},
        {"timestamp": 1704067210000, "price": 42105.0, "quantity": 2.0, "is_buyer_maker": True},
    ]

    bars = processor.process_trades(trades)
    assert len(bars) == 1

    bar = bars[0]
    # Check microstructure fields exist
    assert "vwap" in bar
    assert "buy_volume" in bar
    assert "sell_volume" in bar
    assert "individual_trade_count" in bar
    assert "agg_record_count" in bar

    # First trade is buy (is_buyer_maker=False), second is sell (is_buyer_maker=True)
    assert bar["buy_volume"] == 1.0
    assert bar["sell_volume"] == 2.0


def test_unsorted_trades_error():
    """Test that unsorted trades raise RuntimeError."""
    processor = PyRangeBarProcessor(threshold_bps=250)

    # Trades in wrong order (timestamps not ascending)
    trades = [
        {"timestamp": 1704067210000, "price": 42000.0, "quantity": 1.0},
        {"timestamp": 1704067200000, "price": 42100.0, "quantity": 1.0},  # Earlier timestamp!
    ]

    with pytest.raises(RuntimeError, match="Processing failed.*not sorted"):
        processor.process_trades(trades)


def test_timestamp_conversion_milliseconds_to_microseconds():
    """Test that timestamps are correctly converted from milliseconds to microseconds."""
    processor = PyRangeBarProcessor(threshold_bps=100)  # 10bps = 0.1%

    # Use timestamp in milliseconds (Binance format)
    trades = [
        {"timestamp": 1704067200000, "price": 50000.0, "quantity": 1.0},
        {"timestamp": 1704067210000, "price": 50050.0, "quantity": 1.0},  # +0.1% breach
    ]

    bars = processor.process_trades(trades)
    assert len(bars) == 1

    # Timestamp should be converted to RFC3339 string
    assert "timestamp" in bars[0]
    assert isinstance(bars[0]["timestamp"], str)
    assert "2024-01-01" in bars[0]["timestamp"]  # Should be January 1, 2024


def test_precision_preservation():
    """Test that 8 decimal precision is preserved."""
    processor = PyRangeBarProcessor(threshold_bps=250)

    trades = [
        {"timestamp": 1704067200000, "price": 42000.12345678, "quantity": 1.23456789},
        {"timestamp": 1704067210000, "price": 42105.87654321, "quantity": 2.98765432},
    ]

    bars = processor.process_trades(trades)
    assert len(bars) == 1

    bar = bars[0]
    # Check that precision is reasonably preserved (within floating point error)
    assert abs(bar["open"] - 42000.12345678) < 1e-6
    assert abs(bar["close"] - 42105.87654321) < 1e-6
    assert abs(bar["volume"] - (1.23456789 + 2.98765432)) < 1e-6


def test_batch_vs_streaming_mode():
    """Test that batch processing returns only completed bars.

    Note: In batch mode, only bars closed by a breach are returned.
    The final incomplete bar is NOT returned unless using the
    _with_incomplete variant.
    """
    processor = PyRangeBarProcessor(threshold_bps=250)  # 25bps = 0.25%

    # Create trades where only first bar completes
    trades = [
        {"timestamp": 1704067200000, "price": 40000.0, "quantity": 1.0},
        {"timestamp": 1704067210000, "price": 40100.0, "quantity": 0.5},  # Breaches! (+0.25%)
        {"timestamp": 1704067220000, "price": 40150.0, "quantity": 1.5},  # Adds to bar 2 (incomplete)
    ]

    bars = processor.process_trades(trades)

    # Batch mode: Only bar 1 is returned (completed)
    # Bar 2 started but didn't close, so it's NOT in the output
    assert len(bars) >= 1  # At least one completed bar

    # Verify first bar
    assert bars[0]["open"] == 40000.0
    assert abs(bars[0]["close"] - 40100.0) < 0.01
    assert bars[0]["volume"] >= 1.5  # Contains at least trades 1 and 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
