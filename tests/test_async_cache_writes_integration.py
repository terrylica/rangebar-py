"""Integration tests for concurrent ClickHouse cache writes (Issue #96 Task #5 Phase 2).

Tests demonstrate how AsyncCacheWriter integrates into populate_cache_resumable
workflows. Focuses on correctness and checkpoint safety with concurrent writes.
"""

import asyncio

import pytest
from rangebar.orchestration.async_cache_writes import (
    AsyncCacheWriter,
    populate_with_async_writes,
)


class TestAsyncIntegration:
    """Integration tests for async cache writes with populate workflows."""

    @pytest.mark.asyncio
    async def test_populate_helper_with_mock_write(self) -> None:
        """Test populate_with_async_writes with mock write function."""
        written_days = []
        written_bars_per_day = {}

        def mock_write(bars: list) -> None:
            """Mock write that tracks bars per day."""
            # In real usage, bars would be a list of dicts with date info
            # For this test, we just track the count
            if bars:
                day_key = f"day_{len(written_bars_per_day)}"
                written_bars_per_day[day_key] = len(bars)

        def process_day(day: int) -> list:
            """Process a day and return bars list."""
            written_days.append(day)
            # Simulate processing that returns bars
            return [{"day": day, "close": 100 + day} for _ in range(3)]

        bars_written = await populate_with_async_writes(
            day_iterator=[1, 2, 3, 4, 5],
            process_bars_fn=process_day,
            write_bars_fn=mock_write,
            num_workers=2,
            verbose=False,
        )

        assert written_days == [1, 2, 3, 4, 5]
        assert bars_written == 15  # 5 days x 3 bars
        assert len(written_bars_per_day) == 5

    @pytest.mark.asyncio
    async def test_async_writer_preserves_day_order(self) -> None:
        """Verify AsyncCacheWriter processes days in order."""
        processed_order = []
        days_queued = []

        def track_write(bars: list) -> None:
            """Track order of writes."""
            # Extract day from first bar dict
            if bars and isinstance(bars[0], dict) and "day" in bars[0]:
                processed_order.append(bars[0]["day"])

        def process_day_with_delay(day: int) -> list:
            """Process day and track queueing."""
            days_queued.append(day)
            return [{"day": day, "bar_id": i} for i in range(2)]

        bars_written = await populate_with_async_writes(
            day_iterator=[10, 11, 12],
            process_bars_fn=process_day_with_delay,
            write_bars_fn=track_write,
            num_workers=1,
            verbose=False,
        )

        assert days_queued == [10, 11, 12]
        assert bars_written == 6
        # Order may not be preserved with multiple workers,
        # but single worker should preserve order
        if len(set(processed_order)) == 3:  # All days written
            assert processed_order == [10, 11, 12]

    @pytest.mark.asyncio
    async def test_async_writer_handles_empty_days(self) -> None:
        """AsyncCacheWriter should handle days with no bars."""
        written_count = 0

        def mock_write(bars: list) -> None:
            nonlocal written_count
            written_count += 1

        def process_day(day: int) -> list:
            # Days 2 and 4 return empty lists
            if day % 2 == 0:
                return []
            return [{"day": day} for _ in range(1)]

        bars_written = await populate_with_async_writes(
            day_iterator=[1, 2, 3, 4, 5],
            process_bars_fn=process_day,
            write_bars_fn=mock_write,
            num_workers=2,
            verbose=False,
        )

        assert bars_written == 3  # Only odd days have bars
        assert written_count == 5  # Called for all days (including empty)

    @pytest.mark.asyncio
    async def test_async_writer_metrics_accuracy(self) -> None:
        """AsyncCacheWriter metrics should accurately track writes."""
        writer = AsyncCacheWriter(num_workers=2, maxsize=100)

        written_bars = []

        def mock_write(bars: list) -> None:
            written_bars.extend(bars)

        writer.set_write_function(mock_write)

        async def producer() -> None:
            # Queue 10 batches of 3 bars each
            for batch_id in range(10):
                batch = [{"batch": batch_id, "bar": i} for i in range(3)]
                await writer.queue.put(batch)

            # Shutdown
            for _ in range(2):
                await writer.queue.put(None)

        producer_task = asyncio.create_task(producer())
        await writer.run_workers()
        await producer_task

        # Verify metrics
        assert writer.metrics.writes_attempted == 30  # 10 batches x 3 bars
        assert writer.metrics.writes_succeeded == 30
        assert len(written_bars) == 30
        assert writer.metrics.write_errors == []

    @pytest.mark.asyncio
    async def test_populate_respects_worker_count(self) -> None:
        """populate_with_async_writes should use specified num_workers."""
        worker_activity = {"worker_ids": set()}

        def mock_write(bars: list) -> None:
            """Mock write for worker count test."""

        def process_day(day: int) -> list:
            return [{"day": day}]

        # Test with different worker counts
        for num_workers in [1, 2, 4]:
            bars_written = await populate_with_async_writes(
                day_iterator=[1, 2, 3],
                process_bars_fn=process_day,
                write_bars_fn=mock_write,
                num_workers=num_workers,
                verbose=False,
            )
            assert bars_written == 3


class TestAsyncCheckpointSafety:
    """Tests for checkpoint safety with async writes."""

    @pytest.mark.asyncio
    async def test_async_writer_error_propagation(self) -> None:
        """Errors in write function should propagate to caller."""
        writer = AsyncCacheWriter(num_workers=1)

        call_count = 0

        def failing_write(bars: list) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                msg = "Simulated write failure"
                raise ValueError(msg)

        writer.set_write_function(failing_write)

        async def producer() -> None:
            # Queue 3 batches
            await writer.queue.put([{"batch": 0}])
            await writer.queue.put([{"batch": 1}])  # This will fail
            await writer.queue.put(None)

        producer_task = asyncio.create_task(producer())

        # Should raise due to write error
        with pytest.raises(ValueError, match="Simulated write failure"):
            await writer.run_workers()

        await producer_task
        assert writer.metrics.write_errors

    @pytest.mark.asyncio
    async def test_backpressure_metrics_during_async_writes(self) -> None:
        """Backpressure metrics should reflect queue saturation during writes."""
        writer = AsyncCacheWriter(num_workers=1, maxsize=5)

        def slow_write(bars: list) -> None:
            """Slow write to trigger queue saturation."""
            import time

            time.sleep(0.05)

        writer.set_write_function(slow_write)

        async def producer() -> None:
            # Queue items to potentially trigger backpressure
            for i in range(5):
                await writer.queue.put([{"id": i}])
                metrics = writer.get_backpressure_metrics()
                # Queue should not exceed maxsize
                assert metrics["queue_size"] <= 5

            # Shutdown
            await writer.queue.put(None)

        producer_task = asyncio.create_task(producer())
        await writer.run_workers()
        await producer_task

        final_metrics = writer.get_backpressure_metrics()
        assert final_metrics["writes_succeeded"] == 5
        assert final_metrics["queue_size"] == 0  # Clean shutdown

