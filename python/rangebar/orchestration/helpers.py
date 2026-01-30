# polars-exception: backtesting.py requires Pandas DataFrames with DatetimeIndex
# Issue #46: Modularization M4 - Extract shared helpers from __init__.py
"""Shared helper functions for orchestration modules.

These internal functions handle data fetching, trade processing,
and streaming for both Binance and Exness data sources.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pandas as pd

from rangebar.processors.core import RangeBarProcessor

if TYPE_CHECKING:
    import polars as pl


def _stream_range_bars_binance(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: int,
    market: str,
    batch_size: int = 10_000,
    include_microstructure: bool = False,
    include_incomplete: bool = False,
    prevent_same_timestamp_close: bool = True,
    verify_checksum: bool = True,
) -> Iterator[pl.DataFrame]:
    """Stream range bars in batches using memory-efficient chunked processing.

    This is the internal generator for Phase 4 streaming API. It:
    1. Streams trades in 6-hour chunks from Binance via stream_binance_trades()
    2. Processes each chunk to bars via process_trades_streaming_arrow()
    3. Yields batches of bars as Polars DataFrames

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    start_date : str
        Start date "YYYY-MM-DD"
    end_date : str
        End date "YYYY-MM-DD"
    threshold_decimal_bps : int
        Range bar threshold (250 = 0.25%)
    market : str
        Normalized market type: "spot", "um", or "cm"
    batch_size : int, default=10_000
        Number of bars per yielded DataFrame (~500 KB each)
    include_microstructure : bool, default=False
        Include microstructure columns in output
    include_incomplete : bool, default=False
        Include the final incomplete bar
    prevent_same_timestamp_close : bool, default=True
        Timestamp gating for flash crash prevention
    verify_checksum : bool, default=True
        Verify SHA-256 checksum of downloaded data

    Yields
    ------
    pl.DataFrame
        Batches of range bars (OHLCV format, backtesting.py compatible)

    Memory Usage
    ------------
    Peak: ~50 MB (6-hour trade chunk + bar buffer)
    Per yield: ~500 KB (10,000 bars)
    """
    import polars as pl

    from rangebar.conversion import _bars_list_to_polars

    try:
        from rangebar._core import MarketType, stream_binance_trades
    except ImportError as e:
        msg = (
            "Streaming requires the 'data-providers' feature. "
            "Rebuild with: maturin develop --features data-providers"
        )
        raise RuntimeError(msg) from e

    # Map market string to enum
    market_enum = {
        "spot": MarketType.Spot,
        "um": MarketType.FuturesUM,
        "cm": MarketType.FuturesCM,
    }[market]

    # Create processor with symbol for checkpoint support
    processor = RangeBarProcessor(
        threshold_decimal_bps,
        symbol=symbol,
        prevent_same_timestamp_close=prevent_same_timestamp_close,
    )
    bar_buffer: list[dict] = []

    # Stream trades in 6-hour chunks
    for trade_batch in stream_binance_trades(
        symbol,
        start_date,
        end_date,
        chunk_hours=6,
        market_type=market_enum,
        verify_checksum=verify_checksum,
    ):
        # Process to bars via Arrow (zero-copy to Polars)
        arrow_batch = processor.process_trades_streaming_arrow(trade_batch)
        bars_df = pl.from_arrow(arrow_batch)

        if not bars_df.is_empty():
            # Add to buffer
            bar_buffer.extend(bars_df.to_dicts())

        # Yield when buffer reaches batch_size
        while len(bar_buffer) >= batch_size:
            batch = bar_buffer[:batch_size]
            bar_buffer = bar_buffer[batch_size:]
            yield _bars_list_to_polars(batch, include_microstructure)

    # Handle incomplete bar at end
    if include_incomplete:
        incomplete = processor.get_incomplete_bar()
        if incomplete:
            bar_buffer.append(incomplete)

    # Yield remaining bars
    if bar_buffer:
        yield _bars_list_to_polars(bar_buffer, include_microstructure)


def _fetch_binance(
    symbol: str,
    start_date: str,
    end_date: str,
    market: str,
) -> pl.DataFrame:
    """Fetch Binance aggTrades data (internal).

    DEPRECATED: Use stream_binance_trades() for memory-efficient streaming.
    This function loads all trades into memory at once.
    """
    import warnings
    from datetime import datetime

    import polars as pl

    # MEM-007: Guard deprecated batch path with date range limit (Issue #49)
    # This function loads ALL trades into a single DataFrame. For high-volume
    # symbols (BTCUSDT), a single month can be ~6GB. Limit to 30 days.
    max_days = 30
    days = (
        datetime.strptime(end_date, "%Y-%m-%d")
        - datetime.strptime(start_date, "%Y-%m-%d")
    ).days
    if days > max_days:
        msg = (
            f"_fetch_binance() cannot safely load {days} days of data. "
            f"This deprecated path loads all trades into memory at once "
            f"(limit: {max_days} days). Use precompute_range_bars() or "
            f"get_range_bars() with per-segment loading instead."
        )
        raise MemoryError(msg)

    warnings.warn(
        "_fetch_binance() is deprecated. Use stream_binance_trades() for "
        "memory-efficient streaming. This function will be removed in v9.0.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        from rangebar._core import MarketType, fetch_binance_aggtrades

        market_enum = {
            "spot": MarketType.Spot,
            "um": MarketType.FuturesUM,
            "cm": MarketType.FuturesCM,
        }[market]

        trades_list = fetch_binance_aggtrades(symbol, start_date, end_date, market_enum)
        return pl.DataFrame(trades_list)

    except ImportError as e:
        msg = (
            "Binance data fetching requires the 'data-providers' feature. "
            "Rebuild with: maturin develop --features data-providers"
        )
        raise RuntimeError(msg) from e


def _fetch_exness(
    symbol: str,
    start_date: str,
    end_date: str,
    validation: str,
) -> pl.DataFrame:
    """Fetch Exness tick data (internal)."""
    try:
        from rangebar.exness import fetch_exness_ticks

        return fetch_exness_ticks(symbol, start_date, end_date)

    except ImportError as e:
        msg = (
            "Exness data fetching requires the 'exness' feature. "
            "Rebuild with: maturin develop --features data-providers"
        )
        raise RuntimeError(msg) from e


def _process_binance_trades(
    trades: pl.DataFrame,
    threshold_decimal_bps: int,
    include_incomplete: bool,
    include_microstructure: bool,
    *,
    processor: RangeBarProcessor | None = None,
    symbol: str | None = None,
    prevent_same_timestamp_close: bool = True,
) -> tuple[pd.DataFrame, RangeBarProcessor]:
    """Process Binance trades to range bars (internal).

    Parameters
    ----------
    trades : pl.DataFrame
        Polars DataFrame with tick data
    threshold_decimal_bps : int
        Threshold in decimal basis points
    include_incomplete : bool
        Include incomplete bar (not yet implemented)
    include_microstructure : bool
        Include microstructure columns
    processor : RangeBarProcessor, optional
        Existing processor with state (for cross-file continuity).
        If None, creates a new processor.
    symbol : str, optional
        Symbol for checkpoint creation
    prevent_same_timestamp_close : bool, default=True
        Timestamp gating for flash crash prevention

    Returns
    -------
    tuple[pd.DataFrame, RangeBarProcessor]
        (bars DataFrame, processor with updated state)
        The processor can be used to create a checkpoint for the next file.
    """
    import polars as pl

    # MEM-003: Apply column selection BEFORE collecting LazyFrame
    # This enables predicate pushdown and avoids materializing unused columns
    # Memory impact: 10-100x reduction depending on filter selectivity

    # Determine volume column name (works for both DataFrame and LazyFrame)
    if isinstance(trades, pl.LazyFrame):
        available_cols = trades.collect_schema().names()
    else:
        available_cols = trades.columns

    volume_col = "quantity" if "quantity" in available_cols else "volume"

    # Build column list - include is_buyer_maker for microstructure features (Issue #30)
    columns = [
        pl.col("timestamp"),
        pl.col("price"),
        pl.col(volume_col).alias("quantity"),
    ]
    if "is_buyer_maker" in available_cols:
        columns.append(pl.col("is_buyer_maker"))

    # Apply selection (predicates pushed down for LazyFrame)
    trades_selected = trades.select(columns)

    # Collect AFTER selection (for LazyFrame)
    if isinstance(trades_selected, pl.LazyFrame):
        trades_minimal = trades_selected.collect()
    else:
        trades_minimal = trades_selected

    # Use provided processor or create new one
    if processor is None:
        processor = RangeBarProcessor(
            threshold_decimal_bps,
            symbol=symbol,
            prevent_same_timestamp_close=prevent_same_timestamp_close,
        )

    # MEM-002: Process in chunks to bound memory (2.5 GB → ~50 MB per chunk)
    # Chunked .to_dicts() avoids materializing 1M+ trade dicts at once
    chunk_size = 100_000
    all_bars: list[dict] = []

    n_rows = len(trades_minimal)
    for start in range(0, n_rows, chunk_size):
        chunk = trades_minimal.slice(start, chunk_size).to_dicts()
        bars = processor.process_trades_streaming(chunk)
        all_bars.extend(bars)

    bars = all_bars

    if not bars:
        empty_df = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        ).set_index(pd.DatetimeIndex([]))
        return empty_df, processor

    # Build DataFrame with all fields
    result = pd.DataFrame(bars)
    result["timestamp"] = pd.to_datetime(result["timestamp"], format="ISO8601")
    result = result.set_index("timestamp")

    # Rename OHLCV columns to backtesting.py format
    result = result.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    if include_microstructure:
        # Return all columns including microstructure
        return result, processor

    # Return only OHLCV columns (backtesting.py compatible)
    return result[["Open", "High", "Low", "Close", "Volume"]], processor


def _process_exness_ticks(
    ticks: pl.DataFrame,
    symbol: str,
    threshold_decimal_bps: int,
    validation: str,
    include_incomplete: bool,
    include_microstructure: bool,
) -> pd.DataFrame:
    """Process Exness ticks to range bars (internal)."""
    try:
        # Map validation string to enum
        from rangebar._core import ValidationStrictness
        from rangebar.exness import process_exness_ticks_to_dataframe

        validation_enum = {
            "permissive": ValidationStrictness.Permissive,
            "strict": ValidationStrictness.Strict,
            "paranoid": ValidationStrictness.Paranoid,
        }[validation]

        # Get instrument enum
        from rangebar._core import ExnessInstrument

        instrument_map = {
            "EURUSD": ExnessInstrument.EURUSD,
            "GBPUSD": ExnessInstrument.GBPUSD,
            "USDJPY": ExnessInstrument.USDJPY,
            "AUDUSD": ExnessInstrument.AUDUSD,
            "USDCAD": ExnessInstrument.USDCAD,
            "NZDUSD": ExnessInstrument.NZDUSD,
            "EURGBP": ExnessInstrument.EURGBP,
            "EURJPY": ExnessInstrument.EURJPY,
            "GBPJPY": ExnessInstrument.GBPJPY,
            "XAUUSD": ExnessInstrument.XAUUSD,
        }

        if symbol.upper() not in instrument_map:
            msg = (
                f"Unknown Exness instrument: {symbol}. "
                f"Valid instruments: {list(instrument_map.keys())}"
            )
            raise ValueError(msg)

        instrument = instrument_map[symbol.upper()]

        df = process_exness_ticks_to_dataframe(
            ticks.to_pandas(),
            instrument,
            threshold_decimal_bps,
            validation_enum,
        )

        if not include_microstructure:
            return df[["Open", "High", "Low", "Close", "Volume"]]

        return df

    except ImportError as e:
        msg = (
            "Exness processing requires the 'exness' feature. "
            "Rebuild with: maturin develop --features data-providers"
        )
        raise RuntimeError(msg) from e
