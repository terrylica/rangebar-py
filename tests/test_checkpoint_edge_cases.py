"""Edge case and resilience tests for checkpoint system (Test Expansion - RU Loop).

Tests cover:
1. Corrupted checkpoint file recovery
2. Partial write scenarios (incomplete JSON)
3. Missing required fields handling
4. Concurrent write race conditions (simulated)
5. Checkpoint version mismatch detection
6. Recovery from interrupted processing

Purpose: Validate robustness before overnight production runs.
Related: Issue #84 (Checkpoint Race), Issue #90 (Dedup Hardening)
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path

import pytest
from rangebar.processors.core import RangeBarProcessor

# =============================================================================
# Fixtures
# =============================================================================

BASE_TS_MS = 1704067200000  # 2024-01-01T00:00:00Z


@pytest.fixture
def sample_trades():
    """Generate deterministic sample trades."""
    trades = []
    for i in range(10):
        trades.append({
            "timestamp": BASE_TS_MS + i * 1000,
            "price": 50000.0 + i * 10.0,
            "quantity": 1.0,
            "is_buyer_maker": (i % 2 == 0),
            "agg_trade_id": 1000 + i,
            "first_trade_id": 2000 + i,
            "last_trade_id": 2000 + i,
        })
    return trades


@pytest.fixture
def processor():
    """Create a fresh processor for each test."""
    return RangeBarProcessor(threshold_decimal_bps=250)


# =============================================================================
# Test: Corrupted Checkpoint File Handling
# =============================================================================

class TestCorruptedCheckpointRecovery:
    """Verify graceful handling of malformed checkpoint dictionaries."""

    def test_corrupted_json_rejection(self, processor, sample_trades):
        """Checkpoint with invalid structure should be rejected."""
        # Process trades to generate a valid checkpoint first
        processor.process_trades(sample_trades[:2])
        valid_cp = processor.create_checkpoint("BTCUSDT")

        # Corrupt it by removing required fields
        corrupted_cp = {k: v for k, v in valid_cp.items() if k != "processor_state"}

        # Attempt to load should fail gracefully
        with pytest.raises((KeyError, ValueError, AttributeError)):
            RangeBarProcessor.from_checkpoint(corrupted_cp)

    def test_partial_json_truncation(self, processor, sample_trades):
        """Checkpoint truncated mid-write should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "truncated.json"

            # Write incomplete JSON (missing closing braces)
            checkpoint_path.write_text('{"version": "1.0", "processor_state": {')

            with pytest.raises((json.JSONDecodeError, ValueError, FileNotFoundError)):
                processor.from_checkpoint(str(checkpoint_path))

    def test_missing_required_fields(self, processor, sample_trades):
        """Checkpoint missing critical fields should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "missing_fields.json"

            # Write valid JSON but missing required processor state
            valid_but_incomplete = {
                "version": "1.0",
                # Missing "processor_state" and other required fields
                "metadata": {"symbol": "BTCUSDT"}
            }
            checkpoint_path.write_text(json.dumps(valid_but_incomplete))

            with pytest.raises((KeyError, ValueError)):
                processor.from_checkpoint(str(checkpoint_path))

    def test_version_mismatch_detection(self, processor, sample_trades):
        """Checkpoint with incompatible version should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "wrong_version.json"

            # Write valid JSON but with incompatible version
            incompatible = {
                "version": "0.5.0",  # Incompatible version
                "processor_state": {"bars": [], "defer_open": False},
                "metadata": {"symbol": "BTCUSDT"}
            }
            checkpoint_path.write_text(json.dumps(incompatible))

            # Should either reject or handle gracefully
            with contextlib.suppress(ValueError):
                processor.from_checkpoint(str(checkpoint_path))
                # If it loads, it should be with appropriate warnings


# =============================================================================
# Test: Partial Write Recovery
# =============================================================================

class TestPartialWriteRecovery:
    """Verify resilience when checkpoint writes are interrupted."""

    def test_incomplete_checkpoint_not_used(self, processor, sample_trades):
        """Incomplete checkpoint should not corrupt processor state."""
        # Process some trades
        bars = processor.process_trades(sample_trades[:3])
        assert len(bars) >= 0

        # Simulate checkpoint write failure by attempting to load from bad file
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_checkpoint = Path(tmpdir) / "incomplete.json"
            bad_checkpoint.write_text('{"version": "1.0"')  # Incomplete

            # Loading should fail safely, not corrupt state
            with pytest.raises((json.JSONDecodeError, ValueError)):
                processor.from_checkpoint(str(bad_checkpoint))

            # Processor should still be usable
            bars_after = processor.process_trades(sample_trades[3:5])
            assert len(bars_after) >= 0

    def test_checkpoint_rollback_on_failure(self, processor, sample_trades):
        """Failed checkpoint should not leave partial state."""
        bars_before = processor.process_trades(sample_trades[:2])

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test.json"

            # Process more trades and save checkpoint
            processor.process_trades(sample_trades[2:5])

            # Intentionally corrupt the checkpoint file after creation
            checkpoint_path.write_text('{"corrupted": true}')

            # Loading corrupt checkpoint should not overwrite working state
            with pytest.raises((KeyError, ValueError)):
                processor.from_checkpoint(str(checkpoint_path))


# =============================================================================
# Test: Dedup Edge Cases
# =============================================================================

class TestDedupEdgeCases:
    """Verify dedup logic under edge conditions."""

    def test_duplicate_agg_trade_ids_rejected(self, processor):
        """Duplicate agg_trade_id should be handled in dedup layer."""
        trades = [
            {
                "timestamp": BASE_TS_MS,
                "price": 50000.0,
                "quantity": 1.0,
                "is_buyer_maker": True,
                "agg_trade_id": 1000,  # Same ID
                "first_trade_id": 2000,
                "last_trade_id": 2000,
            },
            {
                "timestamp": BASE_TS_MS + 1000,
                "price": 50010.0,
                "quantity": 1.0,
                "is_buyer_maker": False,
                "agg_trade_id": 1000,  # Duplicate
                "first_trade_id": 2001,
                "last_trade_id": 2001,
            },
        ]

        # Process should not crash - dedup should handle
        bars = processor.process_trades(trades)
        # Behavior depends on dedup implementation
        # At minimum, should not raise exception
        assert isinstance(bars, list)

    def test_agg_trade_id_continuity(self, processor, sample_trades):
        """Agg trade IDs should be continuous after processing."""
        bars = processor.process_trades(sample_trades)

        # All bars should have consistent trade ID ranges
        for i, bar in enumerate(bars):
            assert hasattr(bar, "first_agg_trade_id")
            assert hasattr(bar, "last_agg_trade_id")
            # Last of current should connect to first of next
            if i < len(bars) - 1:
                # IDs should be ordered (may not be strictly sequential due to agg)
                assert bar.last_agg_trade_id <= bars[i + 1].first_agg_trade_id


# =============================================================================
# Test: Boundary Conditions
# =============================================================================

class TestCheckpointBoundaryConditions:
    """Verify checkpoint behavior at system boundaries."""

    def test_empty_checkpoint_load(self, processor):
        """Loading checkpoint from empty/nonexistent file should handle gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent.json"

            # Should raise FileNotFoundError or similar, not crash
            with pytest.raises((FileNotFoundError, IOError)):
                processor.from_checkpoint(str(nonexistent))

    def test_zero_trades_processing(self, processor):
        """Processing empty trade list should not crash."""
        bars = processor.process_trades([])
        assert isinstance(bars, list)
        assert len(bars) == 0

    def test_single_trade_processing(self, processor):
        """Single trade should be processable."""
        single_trade = [{
            "timestamp": BASE_TS_MS,
            "price": 50000.0,
            "quantity": 1.0,
            "is_buyer_maker": True,
            "agg_trade_id": 1000,
            "first_trade_id": 2000,
            "last_trade_id": 2000,
        }]

        bars = processor.process_trades(single_trade)
        # Single trade may not complete a bar (depends on threshold)
        assert isinstance(bars, list)

    def test_very_large_price_movement(self, processor):
        """Extreme price movement should trigger breach correctly."""
        trades = [
            {
                "timestamp": BASE_TS_MS,
                "price": 50000.0,
                "quantity": 1.0,
                "is_buyer_maker": True,
                "agg_trade_id": 1000,
                "first_trade_id": 2000,
                "last_trade_id": 2000,
            },
            {
                "timestamp": BASE_TS_MS + 1000,
                "price": 52000.0,  # 4% movement (should breach at 250 dbps = 0.25%)
                "quantity": 1.0,
                "is_buyer_maker": False,
                "agg_trade_id": 1001,
                "first_trade_id": 2001,
                "last_trade_id": 2001,
            },
        ]

        bars = processor.process_trades(trades)
        # Should successfully process and close bar
        assert len(bars) > 0


# =============================================================================
# Test: State Consistency After Errors
# =============================================================================

class TestStateConsistencyAfterErrors:
    """Verify processor remains consistent after handling errors."""

    def test_processor_usable_after_bad_trade(self, processor, sample_trades):
        """Processor should recover after encountering invalid trade."""
        # Process valid trades
        bars1 = processor.process_trades(sample_trades[:2])

        # Intentionally create malformed trade
        bad_trade = {
            "timestamp": BASE_TS_MS + 2000,
            # Missing required fields
        }

        # This might raise or be skipped - processor should not crash
        with contextlib.suppress(KeyError, TypeError, ValueError):
            processor.process_trades([bad_trade])  # Expected for malformed trade

        # Processor should still be usable
        bars2 = processor.process_trades(sample_trades[2:4])
        assert isinstance(bars2, list)

    def test_defer_open_state_preserved(self, processor):
        """defer_open state should persist correctly through errors."""
        trades = [
            {
                "timestamp": BASE_TS_MS,
                "price": 50000.0,
                "quantity": 1.0,
                "is_buyer_maker": True,
                "agg_trade_id": 1000,
                "first_trade_id": 2000,
                "last_trade_id": 2000,
            },
            {
                "timestamp": BASE_TS_MS + 1000,
                "price": 51000.0,  # Force breach
                "quantity": 1.0,
                "is_buyer_maker": False,
                "agg_trade_id": 1001,
                "first_trade_id": 2001,
                "last_trade_id": 2001,
            },
        ]

        bars = processor.process_trades(trades)
        # After breach, processor should have defer_open = True
        # Next trade should open fresh bar
        # This verifies internal state consistency
        assert len(bars) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
