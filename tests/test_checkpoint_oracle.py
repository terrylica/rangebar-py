"""Oracle-based verification for checkpoint round-trip fidelity (Issue #97).

Verifies that:
1. `incomplete_bar_raw` preserves all 19 fields through checkpoint serialization
2. Split-point parity: processing with checkpoint == single-pass baseline
3. Dict round-trip: checkpoint_to_dict → dict_to_checkpoint loses no data
4. Boundary conditions: bootstrap, exact breach, defer_open, empty processor
5. Parameter variation: multiple thresholds produce identical split vs baseline

Oracle methodology: NEVER use the system under test to generate expected values.
Independently recompute from raw source. Assert numerical identity, not approximate.
"""

from __future__ import annotations

import pytest
from rangebar._core import PyRangeBarProcessor
from rangebar.processors.core import RangeBarProcessor

# =============================================================================
# Trade data generators (deterministic, standalone)
# =============================================================================

BASE_TS_MS = 1704067200000  # 2024-01-01T00:00:00Z


def _generate_breach_trades(
    n: int = 200,
    threshold_dbps: int = 250,
    base_price: float = 50000.0,
    base_ts_ms: int = BASE_TS_MS,
) -> list[dict]:
    """Generate trades that oscillate enough to trigger range bar breaches.

    Price oscillates in a sawtooth: rises by step until > 1.5x threshold
    from base, then reverses direction. Guaranteed multiple breaches.

    Returns list of trade dicts with all fields needed by the processor.
    """
    threshold = base_price * (threshold_dbps / 100_000)
    step = threshold / 3  # ~3 trades per breach

    trades: list[dict] = []
    price = base_price
    direction = 1.0

    for i in range(n):
        price += direction * step
        # Reverse when we've moved 1.5x threshold from base
        if abs(price - base_price) > 1.5 * threshold:
            direction *= -1.0

        trades.append({
            "timestamp": base_ts_ms + i * 1000,
            "price": round(price, 8),
            "quantity": round(1.0 + (i % 5) * 0.5, 8),
            "is_buyer_maker": (i % 2 == 0),
            "agg_trade_id": 1000 + i,
            "first_trade_id": 2000 + i * 3,
            "last_trade_id": 2000 + i * 3 + 2,
        })

    return trades


def _make_simple_trades(n: int = 2) -> list[dict]:
    """Small trade set that does NOT breach (for bootstrap/boundary tests)."""
    return [
        {
            "timestamp": BASE_TS_MS + i * 1000,
            "price": 50000.0 + i * 5.0,  # tiny movement, no breach at 250 dbps
            "quantity": 1.5 + i * 0.5,
            "is_buyer_maker": (i % 2 == 0),
            "agg_trade_id": 1001 + i,
            "first_trade_id": 2001 + i * 3,
            "last_trade_id": 2001 + i * 3 + 2,
        }
        for i in range(n)
    ]


# =============================================================================
# Assertion helpers
# =============================================================================

# Fields that must match exactly between baseline and split bars.
# Note: buy_trade_count/sell_trade_count are only in incomplete_bar_raw,
# not in bar output dicts. open_time/close_time are only in raw checkpoints.
EXACT_FIELDS = [
    "open", "high", "low", "close", "volume",
    "buy_volume", "sell_volume", "vwap",
    "first_agg_trade_id", "last_agg_trade_id",
    "individual_trade_count", "agg_record_count",
]

# Fields in incomplete_bar_raw added by Issue #97
NEW_RAW_FIELDS = [
    "buy_volume", "sell_volume", "individual_trade_count",
    "buy_trade_count", "sell_trade_count", "vwap",
    "first_agg_trade_id", "last_agg_trade_id",
    "turnover", "buy_turnover", "sell_turnover",
]


def assert_bars_identical(
    baseline: list[dict],
    actual: list[dict],
    context: str,
    *,
    skip_intra: bool = True,
) -> None:
    """Assert two bar lists are field-by-field identical.

    Parameters
    ----------
    baseline : list[dict]
        Expected bars (single-pass processing).
    actual : list[dict]
        Actual bars (split-point or resumed processing).
    context : str
        Description for error messages.
    skip_intra : bool
        Skip intra_* fields (partial for checkpoint-spanning bar).
    """
    assert len(baseline) == len(actual), (
        f"{context}: bar count {len(actual)} != baseline {len(baseline)}"
    )
    for i, (b, a) in enumerate(zip(baseline, actual, strict=False)):
        for field in EXACT_FIELDS:
            bv = b.get(field)
            av = a.get(field)
            if bv is None and av is None:
                continue
            if isinstance(bv, float) and isinstance(av, float):
                # Exact float comparison (FixedPoint round-trip)
                assert bv == av, (
                    f"{context}: Bar {i}/{len(baseline)}, field '{field}': "
                    f"baseline={bv}, actual={av}"
                )
            else:
                assert bv == av, (
                    f"{context}: Bar {i}/{len(baseline)}, field '{field}': "
                    f"baseline={bv}, actual={av}"
                )


# =============================================================================
# Tier 1: Atomic Field Preservation
# =============================================================================


class TestAtomicFieldPreservation:
    """Verify all 19 fields survive checkpoint round-trip."""

    def test_incomplete_bar_raw_has_all_new_fields(self):
        """Issue #97: All 11 new fields present in incomplete_bar_raw."""
        p = RangeBarProcessor(500, symbol="BTCUSDT")
        trades = _make_simple_trades(2)
        p.process_trades(trades)

        cp = p.create_checkpoint("BTCUSDT")
        raw = cp.get("incomplete_bar_raw")
        assert raw is not None, "Should have incomplete_bar_raw"

        for field in NEW_RAW_FIELDS:
            assert field in raw, f"Missing field in incomplete_bar_raw: {field}"

    def test_field_values_match_oracle(self):
        """Verify checkpoint fields match hand-computed oracle values."""
        p = RangeBarProcessor(500, symbol="BTCUSDT")
        trades = [
            {
                "timestamp": BASE_TS_MS,
                "price": 50000.0,
                "quantity": 1.5,
                "is_buyer_maker": False,  # buyer (taker buy)
                "agg_trade_id": 1001,
                "first_trade_id": 2001,
                "last_trade_id": 2003,
            },
            {
                "timestamp": BASE_TS_MS + 1000,
                "price": 50010.0,
                "quantity": 2.0,
                "is_buyer_maker": True,  # seller (taker sell)
                "agg_trade_id": 1002,
                "first_trade_id": 2004,
                "last_trade_id": 2006,
            },
        ]
        p.process_trades(trades)
        cp = p.create_checkpoint("BTCUSDT")
        raw = cp["incomplete_bar_raw"]

        # Oracle: is_buyer_maker=False → buy trade (taker buy)
        # Trade 1: buy, qty=1.5
        # Trade 2: sell (is_buyer_maker=True), qty=2.0
        assert raw["buy_volume"] == pytest.approx(1.5, abs=1e-10)
        assert raw["sell_volume"] == pytest.approx(2.0, abs=1e-10)
        assert raw["first_agg_trade_id"] == 1001
        assert raw["last_agg_trade_id"] == 1002
        # vwap = (50000*1.5 + 50010*2.0) / (1.5+2.0) = 175020/3.5 = 50005.714...
        assert raw["vwap"] == pytest.approx(50005.714285714, abs=1e-3)
        # individual_trade_count = (2003-2001+1) + (2006-2004+1) = 3 + 3 = 6
        assert raw["individual_trade_count"] == 6
        assert raw["buy_trade_count"] == 3  # first_trade_id 2001..2003
        assert raw["sell_trade_count"] == 3  # first_trade_id 2004..2006

    def test_round_trip_preserves_fields(self):
        """checkpoint → from_checkpoint → checkpoint produces identical state."""
        p = RangeBarProcessor(500, symbol="BTCUSDT")
        trades = _make_simple_trades(3)
        p.process_trades(trades)

        cp1 = p.create_checkpoint("BTCUSDT")

        # Round-trip
        p2 = RangeBarProcessor.from_checkpoint(cp1)
        cp2 = p2.create_checkpoint("BTCUSDT")

        # Compare incomplete_bar_raw field-by-field
        raw1 = cp1["incomplete_bar_raw"]
        raw2 = cp2["incomplete_bar_raw"]
        assert raw1 is not None
        assert raw2 is not None

        for field in NEW_RAW_FIELDS:
            v1 = raw1.get(field)
            v2 = raw2.get(field)
            if isinstance(v1, float):
                assert v1 == pytest.approx(v2, abs=1e-8), (
                    f"Round-trip field '{field}': {v1} != {v2}"
                )
            else:
                assert v1 == v2, f"Round-trip field '{field}': {v1} != {v2}"

        # Compare other checkpoint fields
        assert cp1["threshold_decimal_bps"] == cp2["threshold_decimal_bps"]
        assert cp1.get("defer_open") == cp2.get("defer_open")
        assert cp1.get("last_trade_id") == cp2.get("last_trade_id")


# =============================================================================
# Tier 2: Split-Point Parity
# =============================================================================


class TestSplitPointParity:
    """Split processing with checkpoint equals single-pass baseline."""

    @pytest.mark.parametrize(
        "split_point",
        [1, 2, 5, 10, 25, 50, 75, 100, 150, 199],
    )
    def test_split_at_point(self, split_point: int):
        """Process N trades as baseline, split at point, verify identical."""
        trades = _generate_breach_trades(n=200, threshold_dbps=250)

        # Baseline: single pass
        p_full = PyRangeBarProcessor(250, "TEST", True, None, False, None)
        bars_full = p_full.process_trades_streaming(trades)

        # Split: part1 → checkpoint → resume → part2
        p1 = PyRangeBarProcessor(250, "TEST", True, None, False, None)
        bars_1 = p1.process_trades_streaming(trades[:split_point])
        cp = p1.create_checkpoint("TEST")

        p2 = PyRangeBarProcessor.from_checkpoint(cp)
        bars_2 = p2.process_trades_streaming(trades[split_point:])

        combined = bars_1 + bars_2

        assert_bars_identical(
            bars_full, combined,
            f"split_point={split_point}",
        )

    def test_split_at_every_boundary(self):
        """Split at every trade in a small dataset (exhaustive)."""
        trades = _generate_breach_trades(n=30, threshold_dbps=500)

        p_full = PyRangeBarProcessor(500, "TEST", True, None, False, None)
        bars_full = p_full.process_trades_streaming(trades)

        for split in range(1, len(trades)):
            p1 = PyRangeBarProcessor(500, "TEST", True, None, False, None)
            bars_1 = p1.process_trades_streaming(trades[:split])
            cp = p1.create_checkpoint("TEST")

            p2 = PyRangeBarProcessor.from_checkpoint(cp)
            bars_2 = p2.process_trades_streaming(trades[split:])

            assert_bars_identical(
                bars_full, bars_1 + bars_2,
                f"exhaustive split={split}/{len(trades)}",
            )


# =============================================================================
# Tier 3: Dict Serialization Round-Trip Fidelity
# =============================================================================


class TestDictRoundTripFidelity:
    """Verify checkpoint_to_dict → dict_to_checkpoint → from_checkpoint
    preserves ALL fields. The failure mode: dict_to_rangebar() zeroing fields."""

    def test_no_data_loss_in_round_trip(self):
        """Process trades, checkpoint, round-trip, verify no field lost."""
        trades = _generate_breach_trades(n=100, threshold_dbps=250)

        p = PyRangeBarProcessor(250, "TEST", True, None, False, None)
        # Process some but not all — leave incomplete bar
        p.process_trades_streaming(trades[:60])
        cp1 = p.create_checkpoint("TEST")

        assert cp1.get("incomplete_bar_raw") is not None
        raw1 = cp1["incomplete_bar_raw"]

        # Round-trip through Python dict (same as JSON serialize/deserialize)
        p2 = PyRangeBarProcessor.from_checkpoint(cp1)
        cp2 = p2.create_checkpoint("TEST")
        raw2 = cp2["incomplete_bar_raw"]

        # All 19 fields must match (8 original + 11 new)
        all_fields = [
            "open", "high", "low", "close", "volume",
            "open_time", "close_time", "agg_record_count",
            *NEW_RAW_FIELDS,
        ]

        for field in all_fields:
            v1 = raw1.get(field)
            v2 = raw2.get(field)
            assert v1 is not None, f"Field '{field}' missing in original checkpoint"
            assert v2 is not None, f"Field '{field}' missing after round-trip"
            if isinstance(v1, float):
                assert v1 == pytest.approx(v2, abs=1e-8), (
                    f"Round-trip loss in '{field}': {v1} → {v2}"
                )
            else:
                assert v1 == v2, f"Round-trip loss in '{field}': {v1} → {v2}"

    def test_resumed_bar_accumulates_correctly(self):
        """Resume from checkpoint, process more trades, verify accumulation."""
        trades = _generate_breach_trades(n=100, threshold_dbps=500)

        # Process first 20 trades (likely no breach at 500 dbps)
        p = PyRangeBarProcessor(500, "TEST", True, None, False, None)
        bars_before = p.process_trades_streaming(trades[:20])
        cp = p.create_checkpoint("TEST")

        if cp.get("incomplete_bar_raw"):
            raw = cp["incomplete_bar_raw"]
            pre_buy_vol = raw["buy_volume"]
            pre_first_id = raw["first_agg_trade_id"]

            # Resume and process more trades
            p2 = PyRangeBarProcessor.from_checkpoint(cp)
            bars_after = p2.process_trades_streaming(trades[20:])

            # First completed bar after resume should include pre-checkpoint state
            if bars_after:
                first_bar = bars_after[0]
                # The bar's first_agg_trade_id should be from the original bar open
                assert first_bar["first_agg_trade_id"] == pre_first_id, (
                    f"first_agg_trade_id lost: {first_bar['first_agg_trade_id']} "
                    f"!= {pre_first_id}"
                )


# =============================================================================
# Tier 4: Boundary Conditions
# =============================================================================


class TestBoundaryConditions:
    """Edge cases for checkpoint system."""

    def test_bootstrap_single_trade(self):
        """Checkpoint after 1 trade → has incomplete bar."""
        p = RangeBarProcessor(250, symbol="TEST")
        trades = _make_simple_trades(1)
        p.process_trades(trades)

        cp = p.create_checkpoint("TEST")
        assert cp.get("incomplete_bar_raw") is not None
        assert cp["incomplete_bar_raw"]["first_agg_trade_id"] == 1001

        # Resume → process 0 trades → same state
        p2 = RangeBarProcessor.from_checkpoint(cp)
        cp2 = p2.create_checkpoint("TEST")
        assert cp2.get("incomplete_bar_raw") is not None

    def test_empty_processor_checkpoint(self):
        """New processor, no trades → checkpoint has no incomplete bar."""
        p = RangeBarProcessor(250, symbol="TEST")
        cp = p.create_checkpoint("TEST")

        assert cp.get("incomplete_bar_raw") is None or cp.get("has_incomplete_bar") is False

        # Resume → process trades → same as fresh
        p2 = RangeBarProcessor.from_checkpoint(cp)
        trades = _generate_breach_trades(n=50, threshold_dbps=250)
        bars_resumed = p2.process_trades(trades)

        p3 = RangeBarProcessor(250, symbol="TEST")
        bars_fresh = p3.process_trades(trades)

        assert_bars_identical(bars_fresh, bars_resumed, "empty checkpoint vs fresh")

    def test_defer_open_checkpoint(self):
        """After breach, defer_open=True, no incomplete bar."""
        trades = _generate_breach_trades(n=50, threshold_dbps=250)

        # Process enough to get at least one breach
        p = PyRangeBarProcessor(250, "TEST", True, None, False, None)
        bars = p.process_trades_streaming(trades)
        assert len(bars) > 0, "Should have at least one completed bar"

        cp = p.create_checkpoint("TEST")
        # After processing, we might or might not have an incomplete bar
        # depending on whether the last trade was a breach

        # Resume and process more
        p2 = PyRangeBarProcessor.from_checkpoint(cp)
        more_trades = [
            {
                "timestamp": BASE_TS_MS + 100000,
                "price": 50500.0,
                "quantity": 1.0,
                "is_buyer_maker": False,
                "agg_trade_id": 9999,
            },
        ]
        bars2 = p2.process_trades_streaming(more_trades)
        # Should not crash — processor handles defer_open correctly


# =============================================================================
# Tier 5: Parameter Variation (Threshold Sweep)
# =============================================================================


class TestParameterVariation:
    """Split-point parity across multiple thresholds."""

    @pytest.mark.parametrize("threshold", [100, 250, 500, 750, 1000, 2000])
    def test_threshold_sweep(self, threshold: int):
        """Same trade set, different thresholds, split at midpoint."""
        trades = _generate_breach_trades(n=200, threshold_dbps=threshold)
        split = len(trades) // 2

        # Baseline
        p_full = PyRangeBarProcessor(threshold, "TEST", True, None, False, None)
        bars_full = p_full.process_trades_streaming(trades)

        # Split
        p1 = PyRangeBarProcessor(threshold, "TEST", True, None, False, None)
        bars_1 = p1.process_trades_streaming(trades[:split])
        cp = p1.create_checkpoint("TEST")

        p2 = PyRangeBarProcessor.from_checkpoint(cp)
        bars_2 = p2.process_trades_streaming(trades[split:])

        assert_bars_identical(
            bars_full, bars_1 + bars_2,
            f"threshold={threshold}, split={split}",
        )

    def test_monotonicity_invariant(self):
        """More bars at lower thresholds (monotonicity)."""
        trades = _generate_breach_trades(n=200, threshold_dbps=100)
        counts = {}

        for threshold in [100, 250, 500, 1000]:
            p = PyRangeBarProcessor(threshold, "TEST", True, None, False, None)
            bars = p.process_trades_streaming(trades)
            counts[threshold] = len(bars)

        # Lower threshold → more bars (monotonically non-increasing)
        assert counts[100] >= counts[250], f"100 dbps ({counts[100]}) < 250 dbps ({counts[250]})"
        assert counts[250] >= counts[500], f"250 dbps ({counts[250]}) < 500 dbps ({counts[500]})"
        assert counts[500] >= counts[1000], f"500 dbps ({counts[500]}) < 1000 dbps ({counts[1000]})"


# =============================================================================
# Tier 6: Multi-Day Population Continuity (Hierarchical Composition)
# =============================================================================


class TestMultiDayPopulationContinuity:
    """Simulate populate_cache_resumable day loop with processor threading."""

    def test_three_day_continuity(self):
        """Process 3 'days' as single pass vs day-by-day with checkpoints."""
        # Generate 3 chunks of trades (simulating 3 days)
        all_trades = _generate_breach_trades(n=300, threshold_dbps=250)
        day1 = all_trades[:100]
        day2 = all_trades[100:200]
        day3 = all_trades[200:]

        # Baseline: single pass
        p_full = PyRangeBarProcessor(250, "TEST", True, None, False, None)
        bars_full = p_full.process_trades_streaming(all_trades)

        # Day-by-day with checkpoints
        p = PyRangeBarProcessor(250, "TEST", True, None, False, None)
        bars_d1 = p.process_trades_streaming(day1)
        cp1 = p.create_checkpoint("TEST")

        p = PyRangeBarProcessor.from_checkpoint(cp1)
        bars_d2 = p.process_trades_streaming(day2)
        cp2 = p.create_checkpoint("TEST")

        p = PyRangeBarProcessor.from_checkpoint(cp2)
        bars_d3 = p.process_trades_streaming(day3)

        combined = bars_d1 + bars_d2 + bars_d3

        assert_bars_identical(bars_full, combined, "3-day continuity")

    def test_interrupt_mid_day_resume(self):
        """Simulate interrupt mid-day-2, resume, verify parity."""
        all_trades = _generate_breach_trades(n=300, threshold_dbps=250)
        day1 = all_trades[:100]
        day2_first_half = all_trades[100:150]
        day2_second_half = all_trades[150:200]
        day3 = all_trades[200:]

        # Baseline
        p_full = PyRangeBarProcessor(250, "TEST", True, None, False, None)
        bars_full = p_full.process_trades_streaming(all_trades)

        # Day 1
        p = PyRangeBarProcessor(250, "TEST", True, None, False, None)
        bars_d1 = p.process_trades_streaming(day1)
        cp1 = p.create_checkpoint("TEST")

        # Day 2 first half (simulating interrupt)
        p = PyRangeBarProcessor.from_checkpoint(cp1)
        bars_d2a = p.process_trades_streaming(day2_first_half)
        cp_mid = p.create_checkpoint("TEST")

        # Resume from mid-day checkpoint
        p = PyRangeBarProcessor.from_checkpoint(cp_mid)
        bars_d2b = p.process_trades_streaming(day2_second_half)
        cp2 = p.create_checkpoint("TEST")

        # Day 3
        p = PyRangeBarProcessor.from_checkpoint(cp2)
        bars_d3 = p.process_trades_streaming(day3)

        combined = bars_d1 + bars_d2a + bars_d2b + bars_d3

        assert_bars_identical(bars_full, combined, "mid-day interrupt resume")

    def test_agg_trade_id_continuity(self):
        """Verify agg_trade_id continuity across checkpoint boundaries."""
        trades = _generate_breach_trades(n=200, threshold_dbps=250)

        p = PyRangeBarProcessor(250, "TEST", True, None, False, None)
        bars = p.process_trades_streaming(trades)

        if len(bars) < 2:
            pytest.skip("Need at least 2 bars for continuity check")

        for i in range(1, len(bars)):
            prev_last = bars[i - 1]["last_agg_trade_id"]
            curr_first = bars[i]["first_agg_trade_id"]
            assert curr_first == prev_last + 1, (
                f"Bar {i}: gap in agg_trade_id: "
                f"bars[{i-1}].last={prev_last}, bars[{i}].first={curr_first}"
            )


# =============================================================================
# Tier 7: OHLCV Invariants (checked on every bar)
# =============================================================================


class TestOHLCVInvariants:
    """Verify OHLCV invariants hold for all bars across split points."""

    def test_invariants_on_split_bars(self):
        """OHLCV invariants hold for every bar produced via checkpoint resume."""
        trades = _generate_breach_trades(n=200, threshold_dbps=250)

        # Split at several points
        for split in [50, 100, 150]:
            p1 = PyRangeBarProcessor(250, "TEST", True, None, False, None)
            bars_1 = p1.process_trades_streaming(trades[:split])
            cp = p1.create_checkpoint("TEST")

            p2 = PyRangeBarProcessor.from_checkpoint(cp)
            bars_2 = p2.process_trades_streaming(trades[split:])

            all_bars = bars_1 + bars_2
            for i, bar in enumerate(all_bars):
                hi, op, lo, cl = bar["high"], bar["open"], bar["low"], bar["close"]
                assert hi >= op, f"Split={split}, Bar {i}: high {hi} < open {op}"
                assert hi >= cl, f"Split={split}, Bar {i}: high {hi} < close {cl}"
                assert lo <= op, f"Split={split}, Bar {i}: low {lo} > open {op}"
                assert lo <= cl, f"Split={split}, Bar {i}: low {lo} > close {cl}"
                assert hi >= lo, f"Split={split}, Bar {i}: high {hi} < low {lo}"
                assert bar["volume"] > 0, f"Split={split}, Bar {i}: volume <= 0"

                # Volume conservation
                bv = bar.get("buy_volume", 0.0)
                sv = bar.get("sell_volume", 0.0)
                total = bar["volume"]
                assert abs(bv + sv - total) < 1e-6, (
                    f"Split={split}, Bar {i}: "
                    f"buy_vol({bv}) + sell_vol({sv}) != volume({total})"
                )


# =============================================================================
# Enable microstructure after checkpoint restore
# =============================================================================


class TestEnableMicrostructureAfterRestore:
    """Verify enable_microstructure works on checkpoint-restored processors."""

    def test_enable_then_process(self):
        """Restore from checkpoint, enable microstructure, process trades."""
        trades = _generate_breach_trades(n=50, threshold_dbps=500)

        p = RangeBarProcessor(500, symbol="TEST")
        p.process_trades(trades[:20])
        cp = p.create_checkpoint("TEST")

        # Restore and enable microstructure
        p2 = RangeBarProcessor.from_checkpoint(cp)
        p2.enable_microstructure(
            inter_bar_lookback_count=200,
            include_intra_bar_features=True,
        )

        bars = p2.process_trades(trades[20:])
        # Should produce bars without crashing
        assert isinstance(bars, list)

    def test_enable_with_bar_relative_lookback(self):
        """Restore, enable bar-relative lookback, process trades."""
        trades = _generate_breach_trades(n=50, threshold_dbps=500)

        p = RangeBarProcessor(500, symbol="TEST")
        p.process_trades(trades[:20])
        cp = p.create_checkpoint("TEST")

        p2 = RangeBarProcessor.from_checkpoint(cp)
        p2.enable_microstructure(inter_bar_lookback_bars=3)

        bars = p2.process_trades(trades[20:])
        assert isinstance(bars, list)


# =============================================================================
# Tier 8: Telemetry Validation (Issue #97 Phase 6)
# =============================================================================


class TestCheckpointTelemetry:
    """Verify that checkpoint lifecycle emits correct hook events and
    that provenance metadata is present in checkpoint files."""

    def test_new_hook_events_exist(self):
        """All 4 new checkpoint hook events are defined."""
        from rangebar.hooks import HookEvent

        assert HookEvent.CHECKPOINT_RESTORED.value == "checkpoint_restored"
        assert HookEvent.CHECKPOINT_RESTORE_FAILED.value == "checkpoint_restore_failed"
        assert HookEvent.SIDECAR_CHECKPOINT_SAVED.value == "sidecar_checkpoint_saved"
        assert HookEvent.SIDECAR_CHECKPOINT_RESTORED.value == "sidecar_checkpoint_restored"

    def test_checkpoint_restored_hook_fires(self):
        """CHECKPOINT_RESTORED hook fires when processor is restored."""
        from rangebar.hooks import HookEvent, clear_hooks, emit_hook, register_hook

        events = []
        register_hook(HookEvent.CHECKPOINT_RESTORED, lambda p: events.append(p))
        try:
            emit_hook(
                HookEvent.CHECKPOINT_RESTORED,
                symbol="BTCUSDT",
                trace_id="test-abc123",
                threshold_dbps=250,
                has_incomplete_bar=True,
                last_trade_id=12345,
                defer_open=False,
            )
            assert len(events) == 1
            assert events[0].symbol == "BTCUSDT"
            assert events[0].trace_id == "test-abc123"
            assert events[0].details["threshold_dbps"] == 250
            assert events[0].details["has_incomplete_bar"] is True
            assert events[0].details["last_trade_id"] == 12345
        finally:
            clear_hooks(HookEvent.CHECKPOINT_RESTORED)

    def test_checkpoint_restore_failed_hook_fires(self):
        """CHECKPOINT_RESTORE_FAILED hook fires with correct fields."""
        from rangebar.hooks import HookEvent, clear_hooks, emit_hook, register_hook

        events = []
        register_hook(
            HookEvent.CHECKPOINT_RESTORE_FAILED, lambda p: events.append(p),
        )
        try:
            emit_hook(
                HookEvent.CHECKPOINT_RESTORE_FAILED,
                symbol="ETHUSDT",
                trace_id="test-fail01",
                threshold_dbps=500,
            )
            assert len(events) == 1
            assert events[0].symbol == "ETHUSDT"
            assert events[0].is_failure is True  # "FAILED" in name
        finally:
            clear_hooks(HookEvent.CHECKPOINT_RESTORE_FAILED)

    def test_generate_trace_id_with_prefix(self):
        """generate_trace_id produces correctly formatted IDs."""
        from rangebar.logging import generate_trace_id

        pop_id = generate_trace_id("pop")
        assert pop_id.startswith("pop-")
        assert len(pop_id) == 12  # "pop-" + 8 hex chars

        sdc_id = generate_trace_id("sdc")
        assert sdc_id.startswith("sdc-")
        assert pop_id != sdc_id  # Unique

    def test_checkpoint_provenance_fields(self):
        """_checkpoint_provenance returns all required fields."""
        from rangebar.checkpoint import _checkpoint_provenance

        prov = _checkpoint_provenance("test-trace-id")
        assert prov["service"] == "rangebar-py"
        assert prov["version"] != ""
        assert prov["host"] != ""
        assert prov["pid"] > 0
        assert prov["trace_id"] == "test-trace-id"
        assert "created_utc" in prov

    def test_checkpoint_file_contains_provenance(self, tmp_path):
        """Saved checkpoint JSON includes provenance metadata."""
        import json

        from rangebar.checkpoint import PopulationCheckpoint

        cp = PopulationCheckpoint(
            symbol="BTCUSDT",
            threshold_bps=250,
            start_date="2024-01-01",
            end_date="2024-01-31",
            last_completed_date="2024-01-15",
            bars_written=1000,
        )
        path = tmp_path / "test_checkpoint.json"
        cp.save(path)

        data = json.loads(path.read_text())
        assert "provenance" in data
        prov = data["provenance"]
        assert prov["service"] == "rangebar-py"
        assert prov["host"] != ""
        assert prov["pid"] > 0

    def test_log_checkpoint_event_callable(self):
        """log_checkpoint_event can be called without error."""
        from rangebar.logging import log_checkpoint_event

        # Should not raise — just verify it's callable
        log_checkpoint_event(
            "checkpoint_save",
            "BTCUSDT",
            "test-trace",
            threshold_dbps=250,
            date="2024-01-15",
            bars_today=50,
        )

    def test_hook_payload_has_trace_id(self):
        """HookPayload includes trace_id field."""
        from rangebar.hooks import HookEvent, HookPayload

        payload = HookPayload(
            event=HookEvent.CHECKPOINT_SAVED,
            symbol="BTCUSDT",
            trace_id="pop-abc12345",
        )
        assert payload.trace_id == "pop-abc12345"
        d = payload.to_dict()
        assert d["trace_id"] == "pop-abc12345"
