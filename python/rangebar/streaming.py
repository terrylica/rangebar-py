# ADR: docs/adr/2026-01-31-realtime-streaming-api.md
"""Real-time streaming API for range bar construction.

This module provides async Python APIs for constructing range bars from live
data sources (Binance WebSocket, Exness tick feeds).

Architecture:
- Low-level: Callback-based Rust bindings (PyBinanceLiveStream)
- High-level: Python async generators built on top

Examples
--------
Async generator (recommended for most use cases):

>>> import asyncio
>>> from rangebar.streaming import stream_binance_live
>>>
>>> async def main():
...     async for bar in stream_binance_live("BTCUSDT", threshold_bps=250):
...         print(f"New bar: {bar['close']}")
...
>>> asyncio.run(main())

Low-level callback interface:

>>> from rangebar.streaming import BinanceLiveStream
>>>
>>> stream = BinanceLiveStream("BTCUSDT", threshold_decimal_bps=250)
>>> stream.connect()
>>> while stream.is_connected:
...     bar = stream.next_bar(timeout_ms=5000)
...     if bar:
...         print(f"New bar: {bar['close']}")

Custom data source with StreamingRangeBarProcessor:

>>> from rangebar.streaming import StreamingRangeBarProcessor
>>>
>>> processor = StreamingRangeBarProcessor(250)
>>> for trade in my_trade_source():
...     bars = processor.process_trade(trade)
...     for bar in bars:
...         print(f"Completed bar: {bar['close']}")
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._core import (
    BinanceLiveStream,
    StreamingConfig,
    StreamingMetrics,
    StreamingRangeBarProcessor,
)

if TYPE_CHECKING:
    from typing import Any

__all__ = [
    "BinanceLiveStream",
    "ReconnectionConfig",
    "StreamingConfig",
    "StreamingError",
    "StreamingMetrics",
    "StreamingRangeBarProcessor",
    "stream_binance_live",
]


class StreamingError(Exception):
    """Error during streaming operation."""


@dataclass
class ReconnectionConfig:
    """Configuration for automatic reconnection.

    Attributes:
        max_retries: Maximum reconnection attempts (0 = infinite)
        initial_delay_s: Initial delay before first retry
        max_delay_s: Maximum delay between retries
        backoff_factor: Multiplier for exponential backoff
    """

    max_retries: int = 0  # 0 = infinite
    initial_delay_s: float = 1.0
    max_delay_s: float = 60.0
    backoff_factor: float = 2.0


async def stream_binance_live(
    symbol: str,
    threshold_bps: int = 250,
    *,
    reconnect: bool = True,
    reconnect_config: ReconnectionConfig | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream range bars from Binance WebSocket in real-time.

    This is an async generator that yields completed range bars as they
    are constructed from live trade data.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        threshold_bps: Range bar threshold in decimal basis points (250 = 0.25%)
        reconnect: Whether to automatically reconnect on disconnect
        reconnect_config: Custom reconnection settings

    Yields:
        Range bar dicts with OHLCV + microstructure features

    Raises:
        StreamingError: If connection fails and reconnection is disabled

    Example:
        >>> async for bar in stream_binance_live("BTCUSDT", threshold_bps=250):
        ...     print(f"New bar: {bar['close']}, OFI: {bar['ofi']}")
    """
    if reconnect_config is None:
        reconnect_config = ReconnectionConfig()

    retry_count = 0
    current_delay = reconnect_config.initial_delay_s

    while True:
        try:
            # Create stream and connect
            stream = BinanceLiveStream(symbol, threshold_bps)

            # Run connect in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, stream.connect)

            # Reset retry state on successful connection
            retry_count = 0
            current_delay = reconnect_config.initial_delay_s

            # Yield bars as they arrive
            while stream.is_connected:
                # Poll for bar with timeout (non-blocking via thread pool)
                # Bind stream as default arg to avoid B023 closure issue
                bar = await loop.run_in_executor(
                    None, lambda s=stream: s.next_bar(timeout_ms=1000)
                )

                if bar is not None:
                    yield bar

                # Allow other coroutines to run
                await asyncio.sleep(0)

            # Stream disconnected
            if not reconnect:
                break

        except Exception as e:
            if not reconnect:
                msg = f"Stream connection failed: {e}"
                raise StreamingError(msg) from e

            retry_count += 1

            # Check max retries
            if (
                reconnect_config.max_retries > 0
                and retry_count > reconnect_config.max_retries
            ):
                msg = f"Max retries ({reconnect_config.max_retries}) exceeded"
                raise StreamingError(msg) from e

            # Log and wait before retry
            print(
                f"Stream disconnected, retrying in {current_delay:.1f}s "
                f"(attempt {retry_count})"
            )
            await asyncio.sleep(current_delay)

            # Exponential backoff
            current_delay = min(
                current_delay * reconnect_config.backoff_factor,
                reconnect_config.max_delay_s,
            )


class AsyncStreamingProcessor:
    """Async wrapper for StreamingRangeBarProcessor.

    This class provides an async interface for processing trades from
    any data source into range bars.

    Example:
        >>> processor = AsyncStreamingProcessor(250)
        >>> async for trade in my_async_trade_source():
        ...     bars = await processor.process_trade(trade)
        ...     for bar in bars:
        ...         await handle_new_bar(bar)
    """

    def __init__(self, threshold_decimal_bps: int) -> None:
        """Create async streaming processor.

        Args:
            threshold_decimal_bps: Range bar threshold (250 = 0.25%)
        """
        self._processor = StreamingRangeBarProcessor(threshold_decimal_bps)
        self._lock = asyncio.Lock()

    async def process_trade(self, trade: dict[str, Any]) -> list[dict[str, Any]]:
        """Process a single trade asynchronously.

        Args:
            trade: Trade dict with timestamp, price, quantity/volume

        Returns:
            List of completed bar dicts (usually 0 or 1)
        """
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._processor.process_trade, trade
            )

    async def process_trades(
        self, trades: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process multiple trades asynchronously.

        Args:
            trades: List of trade dicts

        Returns:
            List of completed bar dicts
        """
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._processor.process_trades, trades
            )

    async def get_incomplete_bar(self) -> dict[str, Any] | None:
        """Get current incomplete bar asynchronously."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._processor.get_incomplete_bar
            )

    @property
    def trades_processed(self) -> int:
        """Number of trades processed."""
        return self._processor.trades_processed

    @property
    def bars_generated(self) -> int:
        """Number of bars generated."""
        return self._processor.bars_generated

    def get_metrics(self) -> StreamingMetrics:
        """Get streaming metrics."""
        return self._processor.get_metrics()
