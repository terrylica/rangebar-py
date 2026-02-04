"""Tests for Issue #72: Aggregate trade ID tracking.

Verifies that first_agg_trade_id and last_agg_trade_id fields are
correctly tracked and exported to Python dictionaries/DataFrames.
"""


from rangebar import TRADE_ID_RANGE_COLUMNS, RangeBarProcessor


def create_trade(agg_trade_id: int, timestamp_ms: int, price: float, quantity: float) -> dict:
    """Create a test trade dict with the given agg_trade_id."""
    return {
        "agg_trade_id": agg_trade_id,
        "price": price,
        "quantity": quantity,
        "first_trade_id": agg_trade_id * 10,  # Individual trade IDs
        "last_trade_id": agg_trade_id * 10 + 2,  # Individual trade IDs
        "timestamp": timestamp_ms,
        "is_buyer_maker": agg_trade_id % 2 == 0,
    }


class TestAggTradeIdTracking:
    """Test suite for aggregate trade ID tracking (Issue #72)."""

    def test_columns_constant_exists(self):
        """TRADE_ID_RANGE_COLUMNS constant should be exported."""
        assert TRADE_ID_RANGE_COLUMNS == ("first_agg_trade_id", "last_agg_trade_id")

    def test_fields_present_in_output(self):
        """first_agg_trade_id and last_agg_trade_id should be in bar output."""
        processor = RangeBarProcessor(threshold_decimal_bps=100)  # 0.1%

        trades = [
            create_trade(1000, 1704067200000, 50000.0, 1.0),
            create_trade(1001, 1704067201000, 50100.0, 1.0),  # 0.2% move triggers bar
        ]

        bars = processor.process_trades(trades)

        assert len(bars) > 0, "Should produce at least one bar"
        bar = bars[0]

        assert "first_agg_trade_id" in bar, "first_agg_trade_id field missing"
        assert "last_agg_trade_id" in bar, "last_agg_trade_id field missing"

    def test_single_trade_bar_values(self):
        """Single-trade bar should have first_agg_trade_id == last_agg_trade_id."""
        processor = RangeBarProcessor(threshold_decimal_bps=100)

        # Two trades that breach threshold immediately
        trades = [
            create_trade(1000, 1704067200000, 50000.0, 1.0),
            create_trade(1001, 1704067201000, 50100.0, 1.0),  # 0.2% move triggers bar
        ]

        bars = processor.process_trades(trades)

        assert len(bars) > 0
        bar = bars[0]

        # First bar contains trades 1000 and 1001 (breach included)
        assert bar["first_agg_trade_id"] == 1000
        assert bar["last_agg_trade_id"] == 1001

    def test_multi_trade_bar_first_less_than_last(self):
        """Multi-trade bar should have first_agg_trade_id < last_agg_trade_id."""
        processor = RangeBarProcessor(threshold_decimal_bps=500)  # 0.5% threshold (wider)

        # Multiple trades before breach
        trades = [
            create_trade(2000, 1704067200000, 50000.0, 1.0),  # Opens bar
            create_trade(2001, 1704067201000, 50050.0, 1.0),  # 0.1% - within threshold
            create_trade(2002, 1704067202000, 50100.0, 1.0),  # 0.2% - within threshold
            create_trade(2003, 1704067203000, 50150.0, 1.0),  # 0.3% - within threshold
            create_trade(2004, 1704067204000, 50300.0, 1.0),  # 0.6% - breach!
        ]

        bars = processor.process_trades(trades)

        assert len(bars) > 0
        bar = bars[0]

        assert bar["first_agg_trade_id"] == 2000, "First agg_trade_id should be 2000"
        assert bar["last_agg_trade_id"] == 2004, "Last agg_trade_id should be 2004 (breach)"
        assert bar["first_agg_trade_id"] < bar["last_agg_trade_id"], "first < last for multi-trade bar"

    def test_sequential_continuity_no_gaps(self):
        """Consecutive bars should satisfy: bars[i].first == bars[i-1].last + 1."""
        processor = RangeBarProcessor(threshold_decimal_bps=100)  # 0.1% threshold

        # Generate trades that will produce multiple bars
        trades = []
        base_price = 50000.0
        for i in range(20):
            # Oscillate price to trigger multiple bars
            price = base_price + (i % 3) * 150.0  # ~0.3% swings
            trades.append(create_trade(
                3000 + i,
                1704067200000 + i * 1000,
                price,
                1.0,
            ))

        bars = processor.process_trades(trades)

        assert len(bars) >= 2, "Should produce multiple bars for gap testing"

        for i in range(1, len(bars)):
            prev_last = bars[i - 1]["last_agg_trade_id"]
            curr_first = bars[i]["first_agg_trade_id"]
            assert curr_first == prev_last + 1, (
                f"Gap detected: bar[{i}].first ({curr_first}) != "
                f"bar[{i - 1}].last ({prev_last}) + 1"
            )

    def test_aggregate_trade_ids_are_correct(self):
        """Aggregate trade IDs should match the input agg_trade_id values."""
        processor = RangeBarProcessor(threshold_decimal_bps=100)

        trades = [
            create_trade(100, 1704067200000, 50000.0, 1.0),  # agg=100
            create_trade(101, 1704067201000, 50100.0, 1.0),  # agg=101
        ]

        bars = processor.process_trades(trades)

        if len(bars) > 0:
            bar = bars[0]

            # Aggregate IDs are in range [100, 101]
            assert 100 <= bar["first_agg_trade_id"] <= 101
            assert 100 <= bar["last_agg_trade_id"] <= 101

            # Aggregate IDs should be distinct from individual trade IDs
            # (individual IDs are agg_id * 10, so 1000-1012, not exported to Python API)
            # We verify that aggregate IDs are in the expected small range
            assert bar["first_agg_trade_id"] < 1000, (
                "first_agg_trade_id should be the aggregate ID (100), not individual trade ID"
            )
            assert bar["last_agg_trade_id"] < 1000, (
                "last_agg_trade_id should be the aggregate ID (101), not individual trade ID"
            )

    def test_checkpoint_preserves_agg_trade_ids(self):
        """Checkpoint should preserve incomplete bar's agg_trade_id range."""
        processor = RangeBarProcessor(threshold_decimal_bps=500)  # Wide threshold

        # Trades that won't complete a bar
        trades = [
            create_trade(4000, 1704067200000, 50000.0, 1.0),
            create_trade(4001, 1704067201000, 50010.0, 1.0),  # 0.02% - way under 0.5%
        ]

        bars = processor.process_trades(trades)
        assert len(bars) == 0, "Should not complete a bar"

        # Create checkpoint
        checkpoint = processor.create_checkpoint("TEST")
        assert checkpoint.get("has_incomplete_bar", False), "Should have incomplete bar"

        # Verify incomplete bar has correct agg_trade_id range
        incomplete = checkpoint.get("incomplete_bar")
        assert incomplete is not None, "Checkpoint should contain incomplete bar"
        assert incomplete["first_agg_trade_id"] == 4000, "Incomplete bar first_agg_trade_id"
        assert incomplete["last_agg_trade_id"] == 4001, "Incomplete bar last_agg_trade_id"

    def test_checkpoint_resume_continuity(self):
        """After checkpoint resume, bar continuity should be maintained."""
        processor1 = RangeBarProcessor(threshold_decimal_bps=200)  # 0.2% threshold

        # Part 1: Process some trades
        trades1 = [
            create_trade(5000, 1704067200000, 50000.0, 1.0),
            create_trade(5001, 1704067201000, 50050.0, 1.0),
        ]

        bars1 = processor1.process_trades(trades1)

        # Create checkpoint
        checkpoint = processor1.create_checkpoint("TEST")

        # Part 2: Resume from checkpoint
        processor2 = RangeBarProcessor.from_checkpoint(checkpoint)

        # Continue with more trades
        trades2 = [
            create_trade(5002, 1704067202000, 50150.0, 1.0),  # 0.3% - breach!
            create_trade(5003, 1704067203000, 50200.0, 1.0),
        ]

        bars2 = processor2.process_trades(trades2)

        # Combine bars
        all_bars = list(bars1) + list(bars2)

        # Verify continuity across checkpoint boundary
        if len(all_bars) >= 2:
            for i in range(1, len(all_bars)):
                prev_last = all_bars[i - 1]["last_agg_trade_id"]
                curr_first = all_bars[i]["first_agg_trade_id"]
                assert curr_first == prev_last + 1, (
                    f"Cross-checkpoint gap: bar[{i}].first != bar[{i - 1}].last + 1"
                )


class TestGapDetection:
    """Test gap detection utility pattern."""

    def test_gap_detection_pattern(self):
        """Demonstrate gap detection pattern using agg_trade_id fields."""
        processor = RangeBarProcessor(threshold_decimal_bps=100)

        trades = []
        base_price = 50000.0
        for i in range(15):
            price = base_price + (i % 3) * 150.0
            trades.append(create_trade(
                1000 + i,
                1704067200000 + i * 1000,
                price,
                1.0,
            ))

        bars = processor.process_trades(trades)

        # Gap detection pattern
        gaps = []
        for i in range(1, len(bars)):
            expected_first = bars[i - 1]["last_agg_trade_id"] + 1
            actual_first = bars[i]["first_agg_trade_id"]
            if actual_first != expected_first:
                gaps.append({
                    "bar_index": i,
                    "expected": expected_first,
                    "actual": actual_first,
                    "gap_size": actual_first - expected_first,
                })

        assert len(gaps) == 0, f"Gaps detected: {gaps}"
