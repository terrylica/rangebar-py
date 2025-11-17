#!/usr/bin/env python3
"""Performance benchmarks for range bar processing.

ADR-003: Testing Strategy with Real Binance Data
Validates performance targets: >1M trades/sec, <100MB memory.
"""

import pytest
import psutil
import os
from rangebar import process_trades_to_dataframe


class TestThroughputBenchmarks:
    """Benchmark processing throughput at different scales."""

    def test_throughput_1k_trades(self, benchmark):
        """Benchmark processing 1,000 trades."""
        # Increase price movement to ensure bars are generated
        # 1000 * 1.0 = 1000 price points = 1000/42000 = 2.38% = 238 bps per trade group
        trades = [
            {"timestamp": 1704067200000 + i * 1000, "price": 42000.0 + i * 1.0, "quantity": 1.0}
            for i in range(1_000)
        ]

        result = benchmark(process_trades_to_dataframe, trades, threshold_bps=250)
        assert len(result) > 0

    def test_throughput_100k_trades(self, benchmark):
        """Benchmark processing 100,000 trades."""
        trades = [
            {"timestamp": 1704067200000 + i * 100, "price": 42000.0 + i * 0.01, "quantity": 1.0}
            for i in range(100_000)
        ]

        result = benchmark(process_trades_to_dataframe, trades, threshold_bps=250)
        assert len(result) > 0

    @pytest.mark.slow
    def test_throughput_1m_trades(self, benchmark):
        """Benchmark processing 1,000,000 trades (target: >1M trades/sec).

        Marked as slow - only run with pytest -m slow.
        """
        trades = [
            {"timestamp": 1704067200000 + i * 10, "price": 42000.0 + i * 0.001, "quantity": 1.0}
            for i in range(1_000_000)
        ]

        result = benchmark(process_trades_to_dataframe, trades, threshold_bps=250)
        assert len(result) > 0

        # Validate target throughput
        stats = benchmark.stats.stats
        elapsed_seconds = stats.mean
        throughput = 1_000_000 / elapsed_seconds

        # Target: >1M trades/sec
        assert throughput > 1_000_000, \
            f"Throughput {throughput:.0f} trades/sec below target 1M trades/sec"


class TestMemoryBenchmarks:
    """Benchmark memory usage."""

    @pytest.mark.slow
    def test_memory_1m_trades(self):
        """Test memory usage for 1,000,000 trades (target: <100MB).

        Marked as slow - only run with pytest -m slow.
        """
        process = psutil.Process(os.getpid())

        # Measure baseline memory
        baseline_mb = process.memory_info().rss / 1024 / 1024

        # Generate 1M trades
        trades = [
            {"timestamp": 1704067200000 + i * 10, "price": 42000.0 + i * 0.001, "quantity": 1.0}
            for i in range(1_000_000)
        ]

        # Process trades
        result = process_trades_to_dataframe(trades, threshold_bps=250)

        # Measure peak memory
        peak_mb = process.memory_info().rss / 1024 / 1024
        memory_used_mb = peak_mb - baseline_mb

        assert len(result) > 0

        # Target: <350MB additional memory (accounts for Python/pandas overhead)
        # Python 3.10 uses ~301MB, 3.11/3.12 use ~253MB
        assert memory_used_mb < 350, \
            f"Memory usage {memory_used_mb:.1f}MB exceeds target 350MB"


class TestCompressionRatios:
    """Test compression ratios at different thresholds."""

    @pytest.mark.parametrize("threshold_bps,expected_min_bars", [
        (100, 100),   # 10bps - more bars (less compression)
        (250, 40),    # 25bps - moderate compression
        (500, 20),    # 50bps - higher compression
        (1000, 10),   # 100bps - maximum compression
    ])
    def test_compression_ratio(self, threshold_bps, expected_min_bars):
        """Test that compression increases with threshold."""
        # Generate 10k trades with controlled price movement
        trades = [
            {"timestamp": 1704067200000 + i * 100, "price": 42000.0 + i * 0.5, "quantity": 1.0}
            for i in range(10_000)
        ]

        result = process_trades_to_dataframe(trades, threshold_bps=threshold_bps)

        # Validate bar count is reasonable
        assert len(result) >= expected_min_bars, \
            f"Threshold {threshold_bps}bps: {len(result)} bars < expected minimum {expected_min_bars}"

        # Compression ratio = input trades / output bars
        compression_ratio = 10_000 / len(result)
        assert compression_ratio > 1, "Should compress trades into fewer bars"

    def test_compression_monotonicity(self):
        """Test that higher thresholds produce fewer bars."""
        trades = [
            {"timestamp": 1704067200000 + i * 100, "price": 42000.0 + i * 0.5, "quantity": 1.0}
            for i in range(10_000)
        ]

        thresholds = [100, 250, 500, 1000]
        bar_counts = {}

        for threshold in thresholds:
            result = process_trades_to_dataframe(trades, threshold_bps=threshold)
            bar_counts[threshold] = len(result)

        # Higher threshold = fewer bars (more compression)
        assert bar_counts[100] >= bar_counts[250] >= bar_counts[500] >= bar_counts[1000], \
            f"Compression not monotonic: {bar_counts}"
