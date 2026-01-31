# polars-exception: process_trades_to_dataframe returns Pandas for backtesting.py compatibility
# ADR: docs/adr/2026-01-31-realtime-streaming-api.md
"""Tests for real-time streaming API.

These tests validate the streaming range bar construction functionality:
1. StreamingRangeBarProcessor - single-trade processing
2. Batch vs streaming consistency - identical results
3. Edge cases - gaps, high-frequency bursts
4. Performance benchmarks

Run with: pytest tests/test_streaming.py -v
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
from rangebar import (
    StreamingRangeBarProcessor,
    process_trades_to_dataframe,
)
from rangebar.streaming import AsyncStreamingProcessor, ReconnectionConfig


class TestStreamingRangeBarProcessor:
    """Tests for StreamingRangeBarProcessor."""

    def test_creation(self) -> None:
        """Test processor creation with valid threshold."""
        processor = StreamingRangeBarProcessor(250)
        assert processor.threshold_decimal_bps == 250
        assert processor.trades_processed == 0
        assert processor.bars_generated == 0

    def test_invalid_threshold_zero(self) -> None:
        """Test that zero threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            StreamingRangeBarProcessor(0)

    def test_invalid_threshold_too_large(self) -> None:
        """Test that threshold > 100000 raises ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            StreamingRangeBarProcessor(100_001)

    def test_process_single_trade(self) -> None:
        """Test processing a single trade."""
        processor = StreamingRangeBarProcessor(250)

        trade = {
            "timestamp": 1704067200000,
            "price": 42000.0,
            "quantity": 1.5,
            "is_buyer_maker": False,
        }

        bars = processor.process_trade(trade)
        assert processor.trades_processed == 1
        assert isinstance(bars, list)
        # First trade opens a bar but doesn't complete it
        assert len(bars) == 0

    def test_process_trades_batch(self) -> None:
        """Test processing multiple trades at once."""
        processor = StreamingRangeBarProcessor(250)

        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
            {"timestamp": 1704067201000, "price": 42050.0, "quantity": 1.0},
            {"timestamp": 1704067202000, "price": 42100.0, "quantity": 1.0},
        ]

        bars = processor.process_trades(trades)
        assert processor.trades_processed == 3
        assert isinstance(bars, list)

    def test_bar_completion_on_threshold_breach(self) -> None:
        """Test that bar is completed when threshold is breached."""
        # Use 100 dbps = 0.1% threshold (decimal basis points)
        # 1 dbps = 0.001%, so 100 dbps = 0.1%
        processor = StreamingRangeBarProcessor(100)
        base_price = 10000.0

        # First trade opens the bar
        processor.process_trade(
            {"timestamp": 1704067200000, "price": base_price, "quantity": 1.0}
        )

        # Second trade breaches threshold (0.5% move > 0.1% threshold)
        # This should complete the first bar
        bars = processor.process_trade(
            {"timestamp": 1704067201000, "price": base_price * 1.005, "quantity": 1.0}
        )

        # Should have completed a bar on the second trade
        assert len(bars) == 1
        assert processor.bars_generated == 1

        # Verify bar structure
        bar = bars[0]
        assert "open" in bar
        assert "high" in bar
        assert "low" in bar
        assert "close" in bar
        assert "volume" in bar
        assert "timestamp" in bar
        assert bar["open"] == base_price
        assert bar["volume"] == 2.0  # Both trades included in the bar

    def test_metrics(self) -> None:
        """Test streaming metrics tracking."""
        processor = StreamingRangeBarProcessor(250)

        # Process some trades
        for i in range(10):
            processor.process_trade(
                {
                    "timestamp": 1704067200000 + i * 1000,
                    "price": 42000.0 + i * 10,
                    "quantity": 1.0,
                }
            )

        metrics = processor.get_metrics()
        assert metrics.trades_processed == 10
        assert metrics.bars_generated >= 0
        assert metrics.error_rate() == 0.0

    def test_incomplete_bar(self) -> None:
        """Test getting incomplete bar state."""
        processor = StreamingRangeBarProcessor(250)

        # No bar yet
        assert processor.get_incomplete_bar() is None

        # Process a trade to start a bar
        processor.process_trade(
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0}
        )

        # Should have incomplete bar
        incomplete = processor.get_incomplete_bar()
        assert incomplete is not None
        assert incomplete["open"] == 42000.0

    def test_repr(self) -> None:
        """Test string representation."""
        processor = StreamingRangeBarProcessor(250)
        repr_str = repr(processor)
        assert "StreamingRangeBarProcessor" in repr_str
        assert "250" in repr_str


class TestBatchVsStreamingConsistency:
    """Tests verifying batch and streaming produce identical results."""

    @pytest.fixture
    def sample_trades(self) -> pd.DataFrame:
        """Generate sample trade data."""
        np.random.seed(42)
        n_trades = 1000

        base_timestamp = 1704067200000
        base_price = 42000.0

        returns = np.random.randn(n_trades) * 0.001
        prices = base_price * np.cumprod(1 + returns)

        return pd.DataFrame(
            {
                "timestamp": base_timestamp + np.arange(n_trades) * 100,
                "price": prices,
                "quantity": np.random.exponential(1.0, n_trades),
                "agg_trade_id": np.arange(n_trades),
                "is_buyer_maker": np.random.randint(0, 2, n_trades).astype(bool),
            }
        )

    def test_streaming_matches_batch(self, sample_trades: pd.DataFrame) -> None:
        """Test that streaming produces same bars as batch processing."""
        threshold = 250

        # Batch processing
        batch_df = process_trades_to_dataframe(
            sample_trades, threshold_decimal_bps=threshold
        )
        batch_bars = len(batch_df)

        # Streaming processing
        processor = StreamingRangeBarProcessor(threshold)
        streaming_bars: list[dict[str, Any]] = []

        for _, row in sample_trades.iterrows():
            trade = {
                "timestamp": int(row["timestamp"]),
                "price": float(row["price"]),
                "quantity": float(row["quantity"]),
                "is_buyer_maker": bool(row["is_buyer_maker"]),
            }
            bars = processor.process_trade(trade)
            streaming_bars.extend(bars)

        # Compare counts
        assert len(streaming_bars) == batch_bars, (
            f"Bar count mismatch: streaming={len(streaming_bars)}, batch={batch_bars}"
        )

        # Compare OHLCV values
        if batch_bars > 0:
            for i, (stream_bar, (_, batch_row)) in enumerate(
                zip(streaming_bars, batch_df.iterrows(), strict=True)
            ):
                assert abs(stream_bar["open"] - batch_row["Open"]) < 1e-6, (
                    f"Bar {i} open mismatch"
                )
                assert abs(stream_bar["high"] - batch_row["High"]) < 1e-6, (
                    f"Bar {i} high mismatch"
                )
                assert abs(stream_bar["low"] - batch_row["Low"]) < 1e-6, (
                    f"Bar {i} low mismatch"
                )
                assert abs(stream_bar["close"] - batch_row["Close"]) < 1e-6, (
                    f"Bar {i} close mismatch"
                )

    def test_streaming_with_different_thresholds(
        self, sample_trades: pd.DataFrame
    ) -> None:
        """Test consistency across multiple thresholds."""
        for threshold in [100, 250, 500, 1000]:
            batch_df = process_trades_to_dataframe(
                sample_trades, threshold_decimal_bps=threshold
            )

            processor = StreamingRangeBarProcessor(threshold)
            streaming_count = 0

            for _, row in sample_trades.iterrows():
                trade = {
                    "timestamp": int(row["timestamp"]),
                    "price": float(row["price"]),
                    "quantity": float(row["quantity"]),
                }
                bars = processor.process_trade(trade)
                streaming_count += len(bars)

            assert streaming_count == len(batch_df), (
                f"Threshold {threshold}: streaming={streaming_count}, batch={len(batch_df)}"
            )


class TestStreamingEdgeCases:
    """Tests for edge cases in streaming processing."""

    def test_high_frequency_burst(self) -> None:
        """Test handling of high-frequency trade bursts."""
        processor = StreamingRangeBarProcessor(100)

        # Simulate 1000 trades in 1 second (same timestamp)
        bars_generated = 0
        for i in range(1000):
            trade = {
                "timestamp": 1704067200000,  # Same timestamp
                "price": 42000.0 + i * 0.1,  # Small price increments
                "quantity": 0.001,
            }
            bars = processor.process_trade(trade)
            bars_generated += len(bars)

        assert processor.trades_processed == 1000
        # Should generate some bars even with same timestamp
        # (price movement still triggers threshold breach)

    def test_large_price_gap(self) -> None:
        """Test handling of large price gaps (circuit breaker scenarios)."""
        processor = StreamingRangeBarProcessor(250)

        # Normal price
        processor.process_trade(
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0}
        )

        # Massive gap (10% move)
        bars = processor.process_trade(
            {"timestamp": 1704067201000, "price": 46200.0, "quantity": 1.0}
        )

        # Should complete the bar
        assert len(bars) == 1
        bar = bars[0]
        assert bar["high"] >= bar["low"]
        assert bar["high"] >= max(bar["open"], bar["close"])
        assert bar["low"] <= min(bar["open"], bar["close"])

    def test_zero_volume_trade(self) -> None:
        """Test handling of zero-volume trades."""
        processor = StreamingRangeBarProcessor(250)

        trade = {
            "timestamp": 1704067200000,
            "price": 42000.0,
            "quantity": 0.0,  # Zero volume
        }

        # Should handle gracefully
        bars = processor.process_trade(trade)
        assert isinstance(bars, list)

    def test_negative_price(self) -> None:
        """Test that negative prices are handled (futures can go negative)."""
        processor = StreamingRangeBarProcessor(250)

        # Oil futures went negative in 2020
        trade = {
            "timestamp": 1704067200000,
            "price": -5.0,
            "quantity": 1.0,
        }

        bars = processor.process_trade(trade)
        assert isinstance(bars, list)


class TestStreamingPerformance:
    """Performance benchmarks for streaming processor."""

    @pytest.mark.benchmark
    def test_throughput_1m_trades(self) -> None:
        """Benchmark: 1M trades throughput."""
        processor = StreamingRangeBarProcessor(250)

        np.random.seed(42)
        n_trades = 1_000_000
        base_price = 42000.0

        # Pre-generate trade data
        prices = base_price + np.cumsum(np.random.randn(n_trades) * 10)
        timestamps = 1704067200000 + np.arange(n_trades) * 100

        start = time.perf_counter()

        for i in range(n_trades):
            processor.process_trade(
                {
                    "timestamp": int(timestamps[i]),
                    "price": float(prices[i]),
                    "quantity": 1.0,
                }
            )

        elapsed = time.perf_counter() - start

        trades_per_sec = n_trades / elapsed
        print(f"\nThroughput: {trades_per_sec:,.0f} trades/sec")
        print(f"Elapsed: {elapsed:.2f}s for {n_trades:,} trades")
        print(f"Bars generated: {processor.bars_generated:,}")

        # Should process at least 100k trades/sec
        assert trades_per_sec > 100_000, (
            f"Throughput too low: {trades_per_sec:,.0f} trades/sec"
        )

    @pytest.mark.benchmark
    def test_latency_per_trade(self) -> None:
        """Benchmark: Per-trade latency."""
        processor = StreamingRangeBarProcessor(250)

        # Warm up
        for i in range(1000):
            processor.process_trade(
                {"timestamp": 1704067200000 + i, "price": 42000.0, "quantity": 1.0}
            )

        # Measure latency
        latencies = []
        for i in range(10000):
            start = time.perf_counter()
            processor.process_trade(
                {
                    "timestamp": 1704067300000 + i,
                    "price": 42000.0 + i * 0.1,
                    "quantity": 1.0,
                }
            )
            latencies.append((time.perf_counter() - start) * 1_000_000)  # microseconds

        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)

        print(f"\nLatency: avg={avg_latency:.1f}µs, p99={p99_latency:.1f}µs")

        # Average latency should be < 10 microseconds
        assert avg_latency < 10, f"Average latency too high: {avg_latency:.1f}µs"


try:
    import pytest_asyncio  # noqa: F401

    HAS_PYTEST_ASYNCIO = True
except ImportError:
    HAS_PYTEST_ASYNCIO = False


@pytest.mark.skipif(not HAS_PYTEST_ASYNCIO, reason="pytest-asyncio not installed")
class TestAsyncStreamingProcessor:
    """Tests for AsyncStreamingProcessor."""

    @pytest.mark.asyncio
    async def test_async_process_trade(self) -> None:
        """Test async trade processing."""
        processor = AsyncStreamingProcessor(250)

        trade = {
            "timestamp": 1704067200000,
            "price": 42000.0,
            "quantity": 1.0,
        }

        bars = await processor.process_trade(trade)
        assert isinstance(bars, list)
        assert processor.trades_processed == 1

    @pytest.mark.asyncio
    async def test_async_process_trades_batch(self) -> None:
        """Test async batch processing."""
        processor = AsyncStreamingProcessor(250)

        trades = [
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},
            {"timestamp": 1704067201000, "price": 42100.0, "quantity": 1.0},
            {"timestamp": 1704067202000, "price": 42200.0, "quantity": 1.0},
        ]

        bars = await processor.process_trades(trades)
        assert isinstance(bars, list)
        assert processor.trades_processed == 3

    @pytest.mark.asyncio
    async def test_async_get_incomplete_bar(self) -> None:
        """Test async incomplete bar retrieval."""
        processor = AsyncStreamingProcessor(250)

        # Process a trade
        await processor.process_trade(
            {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0}
        )

        # Get incomplete bar
        incomplete = await processor.get_incomplete_bar()
        assert incomplete is not None
        assert incomplete["open"] == 42000.0


class TestReconnectionConfig:
    """Tests for ReconnectionConfig."""

    def test_default_values(self) -> None:
        """Test default reconnection config."""
        config = ReconnectionConfig()
        assert config.max_retries == 0  # Infinite
        assert config.initial_delay_s == 1.0
        assert config.max_delay_s == 60.0
        assert config.backoff_factor == 2.0

    def test_custom_values(self) -> None:
        """Test custom reconnection config."""
        config = ReconnectionConfig(
            max_retries=5,
            initial_delay_s=0.5,
            max_delay_s=30.0,
            backoff_factor=1.5,
        )
        assert config.max_retries == 5
        assert config.initial_delay_s == 0.5
        assert config.max_delay_s == 30.0
        assert config.backoff_factor == 1.5
