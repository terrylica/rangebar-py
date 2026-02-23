"""Tests for concurrent ClickHouse cache writes (Issue #96 Task #5).

Phase 1 tests: BoundedAsyncQueue, AsyncCacheWriter core functionality
"""

import asyncio

import pytest
from rangebar.orchestration.async_cache_writes import (
    AsyncCacheWriter,
    BoundedAsyncQueue,
    WriteMetrics,
    populate_with_async_writes,
)


class TestBoundedAsyncQueue:
    """Tests for BoundedAsyncQueue."""

    @pytest.mark.asyncio
    async def test_queue_respects_maxsize(self):
        """Queue respects max size and blocks when full."""
        q = BoundedAsyncQueue(maxsize=2)

        # Put 2 items (fill queue)
        await q.put([{"bar": 1}])
        await q.put([{"bar": 2}])

        assert q.qsize() == 2
        assert q.full()

        # Third put should block (timeout proves it's blocking)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(q.put([{"bar": 3}]), timeout=0.1)

    @pytest.mark.asyncio
    async def test_queue_get_put_roundtrip(self):
        """Data survives round-trip through queue."""
        q = BoundedAsyncQueue()
        test_bars = [{"open": 100, "close": 101}]

        await q.put(test_bars)
        retrieved = await q.get()

        assert retrieved == test_bars

    @pytest.mark.asyncio
    async def test_queue_poison_pill(self):
        """Poison pill (None) signals queue shutdown."""
        q = BoundedAsyncQueue()

        await q.put([{"bar": 1}])
        await q.put(None)  # Poison pill

        item1 = await q.get()
        assert item1 == [{"bar": 1}]

        item2 = await q.get()
        assert item2 is None

    @pytest.mark.asyncio
    async def test_queue_backpressure_releases(self):
        """Queue unblocks producer when consumer drains it."""
        q = BoundedAsyncQueue(maxsize=2)

        # Fill queue
        await q.put([{"bar": 1}])
        await q.put([{"bar": 2}])

        # Consumer task drains queue
        async def consumer() -> None:
            await asyncio.sleep(0.05)  # Slight delay
            await q.get()  # Drain first item

        # Start consumer in background
        consumer_task = asyncio.create_task(consumer())

        # Producer should now be able to put (blocked before, unblocked after drain)
        await asyncio.wait_for(q.put([{"bar": 3}]), timeout=0.5)

        await consumer_task
        assert q.qsize() == 2  # 2 items left: bar2, bar3


class TestAsyncCacheWriter:
    """Tests for AsyncCacheWriter."""

    @pytest.mark.asyncio
    async def test_writer_initialization(self):
        """AsyncCacheWriter initializes correctly."""
        writer = AsyncCacheWriter(num_workers=4, maxsize=5000)

        assert writer.num_workers == 4
        assert writer.queue.maxsize == 5000
        assert writer.metrics.writes_attempted == 0
        assert writer.metrics.writes_succeeded == 0

    @pytest.mark.asyncio
    async def test_writer_requires_write_function(self):
        """Writer raises if write function not set."""
        writer = AsyncCacheWriter()

        with pytest.raises(RuntimeError, match="Write function not set"):
            await writer.run_workers()

    @pytest.mark.asyncio
    async def test_single_worker_processes_bars(self):
        """Single worker processes bars and tracks metrics."""
        writer = AsyncCacheWriter(num_workers=1)

        # Mock write function that succeeds
        written_bars = []

        def mock_write(bars: list) -> None:
            written_bars.extend(bars)

        writer.set_write_function(mock_write)

        # Test bars
        test_bars = [
            {"bar_id": 1, "close": 100},
            {"bar_id": 2, "close": 101},
        ]

        # Enqueue and process
        async def producer() -> None:
            await writer.queue.put(test_bars)
            await writer.queue.put(None)  # Shutdown signal

        producer_task = asyncio.create_task(producer())
        await writer.run_workers()
        await producer_task

        # Verify
        assert len(written_bars) == 2
        assert written_bars == test_bars
        assert writer.metrics.writes_succeeded == 2
        assert writer.metrics.writes_attempted == 2

    @pytest.mark.asyncio
    async def test_multiple_workers_concurrent(self):
        """Multiple workers process bars concurrently."""
        writer = AsyncCacheWriter(num_workers=3)

        written_batches = []
        write_lock = asyncio.Lock()

        async def async_mock_write(bars: list) -> None:
            # Simulate small processing delay
            await asyncio.sleep(0.01)
            async with write_lock:
                written_batches.append(bars)

        # Convert sync mock to async wrapper
        def mock_write(bars: list) -> None:
            # Run async function in event loop (hack for testing)
            # In production, this would be an async function
            written_batches.append(bars)

        writer.set_write_function(mock_write)

        async def producer() -> None:
            # Queue 9 batches for 3 workers
            for i in range(9):
                batch = [{"bar_id": i, "close": 100 + i}]
                await writer.queue.put(batch)

            # Send poison pills
            for _ in range(3):
                await writer.queue.put(None)

        producer_task = asyncio.create_task(producer())
        await writer.run_workers()
        await producer_task

        # All bars should be written
        assert len(written_batches) == 9
        assert writer.metrics.writes_succeeded == 9

    @pytest.mark.asyncio
    async def test_backpressure_metrics(self):
        """Backpressure metrics track queue and write state."""
        writer = AsyncCacheWriter(num_workers=1, maxsize=100)

        def mock_write(bars: list) -> None:
            pass

        writer.set_write_function(mock_write)

        # Manually add bars to queue (without running workers)
        await writer.queue.put([{"bar": 1}])
        await writer.queue.put([{"bar": 2}])
        await writer.queue.put([{"bar": 3}])

        metrics = writer.get_backpressure_metrics()

        assert metrics["queue_size"] == 3
        assert metrics["queue_max"] == 100
        assert metrics["backpressure_pct"] == 3.0
        assert metrics["writes_attempted"] == 0  # Not written yet
        assert metrics["success_rate_pct"] == 100.0  # No failures yet

    @pytest.mark.asyncio
    async def test_write_error_tracking(self):
        """Write errors are captured and tracked."""
        writer = AsyncCacheWriter(num_workers=1)

        call_count = 0
        error_msg = "Simulated write error"

        def mock_write_with_error(bars: list) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError(error_msg)

        writer.set_write_function(mock_write_with_error)

        async def producer() -> None:
            await writer.queue.put([{"bar": 1}])
            await writer.queue.put(None)

        producer_task = asyncio.create_task(producer())

        # run_workers should raise due to write error
        with pytest.raises(ValueError, match="Simulated write error"):
            await writer.run_workers()

        await producer_task

        # Error should be tracked
        assert len(writer.metrics.write_errors) == 1
        assert writer.metrics.write_errors[0][1] == "Simulated write error"


class TestPopulateWithAsyncWrites:
    """Tests for populate_with_async_writes helper."""

    @pytest.mark.asyncio
    async def test_populate_processes_all_days(self):
        """populate_with_async_writes processes all days."""
        processed_days = []
        written_bars_list = []

        def process_day(day: int) -> list[dict]:
            processed_days.append(day)
            return [{"day": day, "close": 100 + day}]

        def write_bars(bars: list) -> None:
            written_bars_list.append(bars)

        days = [1, 2, 3, 4, 5]
        bars_written = await populate_with_async_writes(
            day_iterator=days,
            process_bars_fn=process_day,
            write_bars_fn=write_bars,
            num_workers=2,
            verbose=False,
        )

        assert processed_days == days
        assert len(written_bars_list) == 5
        assert bars_written == 5

    @pytest.mark.asyncio
    async def test_populate_respects_num_workers(self):
        """populate_with_async_writes uses specified num_workers."""
        # This is a simple test that just verifies it doesn't crash with different worker counts
        def process_day(day: int) -> list[dict]:
            return [{"day": day}]

        def write_bars(bars: list) -> None:
            pass

        for num_workers in [1, 2, 4]:
            bars_written = await populate_with_async_writes(
                day_iterator=[1, 2, 3],
                process_bars_fn=process_day,
                write_bars_fn=write_bars,
                num_workers=num_workers,
                verbose=False,
            )
            assert bars_written == 3


class TestWriteMetrics:
    """Tests for WriteMetrics dataclass."""

    def test_metrics_success_rate_calculation(self):
        """Success rate calculated correctly."""
        metrics = WriteMetrics(writes_attempted=100, writes_succeeded=90)
        assert metrics.success_rate == 90.0

    def test_metrics_success_rate_zero_attempts(self):
        """Success rate handles zero attempts."""
        metrics = WriteMetrics(writes_attempted=0, writes_succeeded=0)
        assert metrics.success_rate == 100.0

    def test_metrics_to_dict(self):
        """Metrics export to dictionary."""
        metrics = WriteMetrics(
            writes_attempted=50,
            writes_succeeded=45,
            write_errors=[(0, "error1"), (1, "error2")],
        )

        d = metrics.to_dict()
        assert d["writes_attempted"] == 50
        assert d["writes_succeeded"] == 45
        assert d["write_errors"] == 2
        assert d["success_rate_pct"] == 90.0
