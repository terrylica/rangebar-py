"""Concurrent ClickHouse cache writes with backpressure (Issue #96 Task #5).

Phase 1-2: Async infrastructure for populate_cache_resumable()
Provides bounded queue and worker pool for parallel cache writes.

Performance target: 2-3x faster cache population vs sequential writes
Architecture: Producer (process bars) + Bounded Queue + Worker Pool (write to ClickHouse)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class WriteMetrics:
    """Metrics for concurrent write operations."""

    writes_attempted: int = 0
    writes_succeeded: int = 0
    write_errors: list[tuple[int, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.writes_attempted == 0:
            return 100.0
        return (self.writes_succeeded / self.writes_attempted) * 100

    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        return {
            "writes_attempted": self.writes_attempted,
            "writes_succeeded": self.writes_succeeded,
            "write_errors": len(self.write_errors),
            "success_rate_pct": round(self.success_rate, 2),
        }


class BoundedAsyncQueue:
    """Bounded queue with backpressure.

    Blocks producer when queue reaches maxsize, preventing unbounded growth.
    Used for coordinating between bar processing and ClickHouse writes.
    """

    def __init__(self, maxsize: int = 10_000) -> None:
        """Initialize queue with maximum size.

        Parameters
        ----------
        maxsize : int
            Maximum number of items in queue before blocking producer.
            Default: 10,000 bars (~100MB of bars).
        """
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.maxsize = maxsize

    async def put(self, bars: list | None) -> None:
        """Put bars into queue, blocking if full.

        Parameters
        ----------
        bars : list or None
            List of range bars to write, or None as poison pill for shutdown.
        """
        await self.queue.put(bars)

    async def get(self) -> list | None:
        """Get next batch of bars from queue.

        Returns
        -------
        list or None
            Next batch of bars to write, or None if poison pill received.
        """
        return await self.queue.get()

    def qsize(self) -> int:
        """Current queue size."""
        return self.queue.qsize()

    def full(self) -> bool:
        """Whether queue is at capacity."""
        return self.queue.full()


class AsyncCacheWriter:
    """Manages concurrent ClickHouse writes with bounded queue.

    Coordinates bar processing with async writes to ClickHouse.
    Implements backpressure: producer blocks when queue reaches maxsize.

    Typical usage:
        writer = AsyncCacheWriter(num_workers=4)
        # Start workers
        worker_task = asyncio.create_task(writer.run_workers())
        # Enqueue bars
        for bars in bar_batches:
            await writer.queue.put(bars)
        # Shutdown
        for _ in range(4):
            await writer.queue.put(None)
        await worker_task
    """

    def __init__(self, num_workers: int = 4, maxsize: int = 10_000) -> None:
        """Initialize concurrent writer.

        Parameters
        ----------
        num_workers : int
            Number of concurrent write workers (default: 4).
        maxsize : int
            Maximum pending bars in queue (default: 10,000).
        """
        self.queue = BoundedAsyncQueue(maxsize=maxsize)
        self.num_workers = num_workers
        self.metrics = WriteMetrics()
        self._write_fn = None  # Will be set by caller

    def set_write_function(self, write_fn: callable) -> None:
        """Set the synchronous write function.

        Parameters
        ----------
        write_fn : callable
            Function to write bars to ClickHouse (e.g., _fatal_cache_write).
            Signature: write_fn(bars: list) -> None
        """
        self._write_fn = write_fn

    async def write_worker(self, worker_id: int) -> None:
        """Worker task: dequeue and write bars to ClickHouse.

        Parameters
        ----------
        worker_id : int
            Unique identifier for this worker (for logging/debugging).
        """
        if self._write_fn is None:
            msg = "Write function not set. Call set_write_function() first."
            raise RuntimeError(msg)

        while True:
            try:
                bars = await self.queue.get()
                if bars is None:  # Poison pill for graceful shutdown
                    logger.debug("Worker %d received shutdown signal", worker_id)
                    break

                self.metrics.writes_attempted += len(bars)
                self._write_fn(bars)  # Synchronous write to ClickHouse
                self.metrics.writes_succeeded += len(bars)

            except Exception as e:
                logger.exception("Worker %d error", worker_id)
                self.metrics.write_errors.append((worker_id, str(e)))
                raise

    async def run_workers(self) -> None:
        """Start all worker tasks concurrently.

        Raises
        ------
        RuntimeError
            If any worker fails or write function not set.
        """
        if self._write_fn is None:
            msg = "Write function not set. Call set_write_function() first."
            raise RuntimeError(msg)

        worker_tasks = [self.write_worker(i) for i in range(self.num_workers)]
        await asyncio.gather(*worker_tasks)

    def get_backpressure_metrics(self) -> dict:
        """Get current backpressure and write metrics.

        Returns
        -------
        dict
            Metrics including queue size, write counts, success rate.
        """
        queue_size = self.queue.qsize()
        backpressure_pct = (queue_size / self.queue.maxsize) * 100

        return {
            "queue_size": queue_size,
            "queue_max": self.queue.maxsize,
            "backpressure_pct": round(backpressure_pct, 2),
            **self.metrics.to_dict(),
        }


async def populate_with_async_writes(
    day_iterator: list,
    process_bars_fn: callable,
    write_bars_fn: callable,
    num_workers: int = 4,
    verbose: bool = True,
) -> int:
    """Populate cache using concurrent writes.

    Coordinates fetching/processing of daily bars with async ClickHouse writes.
    Returns immediately when all bars queued; workers continue in background.

    Parameters
    ----------
    day_iterator : iterable
        Iterator of days to process.
    process_bars_fn : callable
        Function to process trades for a day: process_bars_fn(day) -> list[bars].
    write_bars_fn : callable
        Function to write bars to ClickHouse: write_bars_fn(bars) -> None.
    num_workers : int
        Number of concurrent write workers (default: 4).
    verbose : bool
        Whether to log progress (default: True).

    Returns
    -------
    int
        Total number of bars written.

    Example
    -------
    >>> bars_written = await populate_with_async_writes(
    ...     day_iterator=date_range,
    ...     process_bars_fn=process_day_bars,
    ...     write_bars_fn=_fatal_cache_write,
    ...     num_workers=4,
    ... )
    """
    writer = AsyncCacheWriter(num_workers=num_workers)
    writer.set_write_function(write_bars_fn)

    # Start worker tasks
    worker_task = asyncio.create_task(writer.run_workers())

    try:
        # Main loop: process days and queue writes
        for idx, day in enumerate(day_iterator, start=1):
            bars = process_bars_fn(day)
            await writer.queue.put(bars)

            if verbose and idx % 10 == 0:
                metrics = writer.get_backpressure_metrics()
                logger.info(
                    "Processed %d days; queue: %d/%d (%.1f%%)",
                    idx,
                    metrics["queue_size"],
                    metrics["queue_max"],
                    metrics["backpressure_pct"],
                )

        # Shutdown: send poison pills to workers
        for _ in range(num_workers):
            await writer.queue.put(None)

        # Wait for workers to finish
        await worker_task

        final_metrics = writer.get_backpressure_metrics()
        if verbose:
            logger.info(
                "Cache population complete: %d bars written (success rate: %.1f%%)",
                final_metrics["writes_succeeded"],
                final_metrics["success_rate_pct"],
            )

        return writer.metrics.writes_succeeded

    except Exception:
        logger.exception("Populate failed")
        raise
