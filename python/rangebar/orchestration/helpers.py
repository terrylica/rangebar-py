# polars-exception: backtesting.py requires Pandas DataFrames with DatetimeIndex
# Issue #46: Modularization M4 - Extract shared helpers from __init__.py
"""Shared helper functions for orchestration modules.

These internal functions handle data fetching, trade processing,
and streaming for both Binance and Exness data sources.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from rangebar.processors.core import RangeBarProcessor

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Shared Constants & Utilities (Issue #79 follow-up: code clone cleanup)
# =============================================================================

MARKET_TYPE_MAP: dict[str, str] = {
    "spot": "spot",
    "futures-um": "um",
    "futures-cm": "cm",
    "um": "um",
    "cm": "cm",
}

_END_OF_DAY_OFFSET_SECONDS = 86399


def _validate_and_normalize_source_market(
    source: str,
    market: str,
) -> tuple[str, str]:
    """Validate source/market and return (source_lower, market_normalized).

    Raises ValueError for unsupported source or invalid Binance market.
    """
    source = source.lower()
    if source not in ("binance", "exness"):
        msg = f"Unknown source: {source!r}. Must be 'binance' or 'exness'"
        raise ValueError(msg)
    market = market.lower()
    if source == "binance" and market not in MARKET_TYPE_MAP:
        msg = (
            f"Unknown market: {market!r}. "
            "Must be 'spot', 'futures-um'/'um', or 'futures-cm'/'cm'"
        )
        raise ValueError(msg)
    return source, MARKET_TYPE_MAP.get(market, market)


def _parse_and_validate_dates(
    start_date: str,
    end_date: str,
) -> tuple[datetime, datetime]:
    """Parse YYYY-MM-DD strings and validate start <= end.

    Returns (start_dt, end_dt) as datetime objects.
    Raises ValueError for bad format or start > end.
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        msg = f"Invalid date format. Use YYYY-MM-DD: {e}"
        raise ValueError(msg) from e
    if start_dt > end_dt:
        msg = f"start_date must be <= end_date: {start_date} > {end_date}"
        raise ValueError(msg)
    return start_dt, end_dt


def _datetime_to_start_ms(dt: datetime) -> int:
    """Convert datetime to start-of-day millisecond timestamp."""
    return int(dt.timestamp() * 1000)


def _datetime_to_end_ms(dt: datetime) -> int:
    """Convert datetime to end-of-day millisecond timestamp (23:59:59)."""
    return int((dt.timestamp() + _END_OF_DAY_OFFSET_SECONDS) * 1000)


def _parse_microstructure_env_vars(
    include_microstructure: bool,
    inter_bar_lookback_count: int | None,
    inter_bar_lookback_bars: int | None = None,
) -> tuple[int | None, int | None, bool]:
    """Parse lookback env vars and RANGEBAR_INCLUDE_INTRA_BAR_FEATURES.

    Returns (effective_lookback_count, effective_lookback_bars, enable_intra_bar_features).

    Priority: inter_bar_lookback_bars > inter_bar_lookback_count > env vars.
    Env var RANGEBAR_INTER_BAR_LOOKBACK_BARS takes precedence over
    RANGEBAR_INTER_BAR_LOOKBACK_COUNT.
    """
    effective_count = inter_bar_lookback_count
    effective_bars = inter_bar_lookback_bars

    if include_microstructure and effective_count is None and effective_bars is None:
        # Check env vars: RANGEBAR_INTER_BAR_LOOKBACK_BARS takes precedence
        bars_env = os.environ.get("RANGEBAR_INTER_BAR_LOOKBACK_BARS")
        if bars_env is not None:
            effective_bars = int(bars_env)
        else:
            effective_count = int(
                os.environ.get("RANGEBAR_INTER_BAR_LOOKBACK_COUNT", "200")
            )

    enable_intra = False
    if include_microstructure:
        intra_env = os.environ.get("RANGEBAR_INCLUDE_INTRA_BAR_FEATURES", "true")
        enable_intra = intra_env.lower() in ("true", "1", "yes")

    return effective_count, effective_bars, enable_intra


def _create_processor(
    threshold_decimal_bps: int,
    include_microstructure: bool,
    *,
    symbol: str = "",
    prevent_same_timestamp_close: bool = True,
    inter_bar_lookback_count: int | None = None,
    inter_bar_lookback_bars: int | None = None,
    include_intra_bar_features: bool = False,
    processor: RangeBarProcessor | None = None,
) -> RangeBarProcessor:
    """Create or reuse a RangeBarProcessor with env var defaults.

    If processor is provided, returns it as-is (streaming continuation).
    Otherwise creates a new one with parsed env vars.
    """
    if processor is not None:
        return processor

    effective_count, effective_bars, enable_intra = _parse_microstructure_env_vars(
        include_microstructure, inter_bar_lookback_count, inter_bar_lookback_bars,
    )

    # Honour explicit include_intra_bar_features=True even when env says false
    enable_intra = enable_intra or include_intra_bar_features

    return RangeBarProcessor(
        threshold_decimal_bps,
        symbol=symbol,
        prevent_same_timestamp_close=prevent_same_timestamp_close,
        inter_bar_lookback_count=effective_count,
        include_intra_bar_features=enable_intra,
        inter_bar_lookback_bars=effective_bars,
    )


def _select_trade_columns(
    trades: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:
    """Select minimal trade columns for range bar processing (MEM-003).

    Handles: timestamp, price, quantity/volume alias, is_buyer_maker,
    trade ID columns (Issue #75).
    Works with both LazyFrame (predicate pushdown) and DataFrame.
    """
    import polars as pl

    # Determine volume column name (works for both DataFrame and LazyFrame)
    if isinstance(trades, pl.LazyFrame):
        available_cols = trades.collect_schema().names()
    else:
        available_cols = trades.columns

    volume_col = "quantity" if "quantity" in available_cols else "volume"

    columns = [
        pl.col("timestamp"),
        pl.col("price"),
        pl.col(volume_col).alias("quantity"),
    ]
    if "is_buyer_maker" in available_cols:
        columns.append(pl.col("is_buyer_maker"))

    # Issue #75: Include trade ID columns for data integrity tracking (Issue #72)
    for trade_id_col in ("agg_trade_id", "first_trade_id", "last_trade_id"):
        if trade_id_col in available_cols:
            columns.append(pl.col(trade_id_col))

    return trades.select(columns)


def _empty_ohlcv_dataframe() -> pd.DataFrame:
    """Return an empty OHLCV DataFrame with standard columns."""
    return pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"]
    ).set_index(pd.DatetimeIndex([]))


def _no_data_error_message(
    symbol: str,
    start_date: str,
    end_date: str,
    market: str = "spot",
) -> str:
    """Build informative error message when no data is available."""
    earliest = detect_earliest_available_date(symbol, market)
    if earliest and earliest > start_date:
        return (
            f"No data available for {symbol} from {start_date} to {end_date}. "
            f"Symbol listing date detected: {earliest}. "
            f"Try: get_range_bars('{symbol}', '{earliest}', '{end_date}')"
        )
    return f"No data available for {symbol} from {start_date} to {end_date}"


# =============================================================================
# Streaming & Processing
# =============================================================================


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
    inter_bar_lookback_count: int | None = None,
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
    inter_bar_lookback_count : int, optional
        Lookback trade count for inter-bar features (Issue #59)

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

    # Create processor with env var defaults for microstructure features
    processor = _create_processor(
        threshold_decimal_bps,
        include_microstructure,
        symbol=symbol,
        prevent_same_timestamp_close=prevent_same_timestamp_close,
        inter_bar_lookback_count=inter_bar_lookback_count,
    )
    # Issue #88: Buffer Polars DataFrames directly (no bars_df.to_dicts() roundtrip)
    bar_frames: list[pl.DataFrame] = []
    total_buffered = 0

    # Stream trades in 6-hour chunks
    # NOTE: Trade input stays dict-based here (stream_binance_trades yields dicts — Phase 5 scope)
    for trade_batch in stream_binance_trades(
        symbol,
        start_date,
        end_date,
        chunk_hours=6,
        market_type=market_enum,
        verify_checksum=verify_checksum,
    ):
        # Process to bars via Arrow output (trade input still dict-based)
        arrow_batch = processor.process_trades_streaming_arrow(trade_batch)
        bars_df = pl.from_arrow(arrow_batch)

        if not bars_df.is_empty():
            bar_frames.append(bars_df)
            total_buffered += len(bars_df)

        # Yield when buffer reaches batch_size
        while total_buffered >= batch_size:
            combined = pl.concat(bar_frames)
            yield combined.head(batch_size)
            remainder = combined.slice(batch_size)
            bar_frames = [remainder] if not remainder.is_empty() else []
            total_buffered = len(remainder) if not remainder.is_empty() else 0

    # Handle incomplete bar at end
    if include_incomplete:
        incomplete = processor.get_incomplete_bar()
        if incomplete:
            incomplete_df = pl.DataFrame([incomplete], infer_schema_length=None)
            bar_frames.append(incomplete_df)
            total_buffered += 1

    # Yield remaining bars
    if bar_frames:
        yield pl.concat(bar_frames)


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

    except RuntimeError as e:
        # Issue #76: Enhance "No data available" errors with listing date detection
        error_msg = str(e)
        if "No data available" in error_msg:
            enhanced = _no_data_error_message(symbol, start_date, end_date, market)
            if enhanced != error_msg:
                raise RuntimeError(enhanced) from e
        raise


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


def _stream_bars_from_trades(
    trades: pl.DataFrame,
    threshold_decimal_bps: int,
    include_microstructure: bool,
    processor: RangeBarProcessor,
    bar_batch_size: int = 10_000,
) -> Iterator[pl.DataFrame]:
    """Stream range bars in batches from trades (MEM-012).

    Issue #88: Uses Arrow-native input path — eliminates .to_dicts() bottleneck.
    Yields Polars DataFrames instead of list[dict] for zero-copy downstream.

    Parameters
    ----------
    trades : pl.DataFrame
        Polars DataFrame with tick data (already column-selected)
    threshold_decimal_bps : int
        Threshold in decimal basis points
    include_microstructure : bool
        Affects trade chunk size (50K vs 100K)
    processor : RangeBarProcessor
        Processor with state for cross-file continuity
    bar_batch_size : int, default=10_000
        Number of bars per yielded batch (~60 MB with microstructure)

    Yields
    ------
    pl.DataFrame
        Batches of bar DataFrames (bounded memory)

    Memory Usage
    ------------
    Peak: ~60 MB per batch (10K bars x 58 columns)
    vs. Previous: 2.9 GB for 50K bars (unbounded accumulation)
    """
    import polars as pl

    # MEM-002 + MEM-011: Adaptive chunk size based on output features
    # Base: 100K trades (~15 MB dicts, ~50 MB with OHLCV bars)
    # Microstructure: 50K trades (62 columns = 12x more memory per bar)
    trade_chunk_size = 50_000 if include_microstructure else 100_000

    # Issue #88: Buffer Polars DataFrames instead of dicts
    bar_frames: list[pl.DataFrame] = []
    total_buffered = 0

    n_rows = len(trades)
    for start in range(0, n_rows, trade_chunk_size):
        # Issue #88: Arrow-native input — .to_arrow() is zero-copy (0.45ms vs 600ms for .to_dicts())
        chunk_arrow = trades.slice(start, trade_chunk_size).to_arrow()
        bars_arrow = processor.process_trades_arrow(chunk_arrow)
        bars_df = pl.from_arrow(bars_arrow)

        if not bars_df.is_empty():
            bar_frames.append(bars_df)
            total_buffered += len(bars_df)

        # MEM-012: Yield when buffer exceeds batch_size to bound memory
        while total_buffered >= bar_batch_size:
            combined = pl.concat(bar_frames)
            yield combined.head(bar_batch_size)
            remainder = combined.slice(bar_batch_size)
            bar_frames = [remainder] if not remainder.is_empty() else []
            total_buffered = len(remainder) if not remainder.is_empty() else 0

    # Yield remaining bars
    if bar_frames:
        yield pl.concat(bar_frames)


def _process_binance_trades(
    trades: pl.DataFrame,
    threshold_decimal_bps: int,
    include_incomplete: bool,
    include_microstructure: bool,
    *,
    processor: RangeBarProcessor | None = None,
    symbol: str | None = None,
    prevent_same_timestamp_close: bool = True,
    inter_bar_lookback_count: int | None = None,
    include_intra_bar_features: bool = False,
    inter_bar_lookback_bars: int | None = None,
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
    inter_bar_lookback_count : int, optional
        Lookback trade count for inter-bar features (Issue #59)
    include_intra_bar_features : bool, default=False
        Enable intra-bar features (Issue #59)

    Returns
    -------
    tuple[pd.DataFrame, RangeBarProcessor]
        (bars DataFrame, processor with updated state)
        The processor can be used to create a checkpoint for the next file.
    """
    import polars as pl

    # MEM-003: Apply column selection BEFORE collecting LazyFrame
    trades_selected = _select_trade_columns(trades)

    # Collect AFTER selection (for LazyFrame)
    if isinstance(trades_selected, pl.LazyFrame):
        trades_minimal = trades_selected.collect()
    else:
        trades_minimal = trades_selected

    # Use provided processor or create new one (with env var defaults)
    processor = _create_processor(
        threshold_decimal_bps,
        include_microstructure,
        symbol=symbol or "",
        prevent_same_timestamp_close=prevent_same_timestamp_close,
        inter_bar_lookback_count=inter_bar_lookback_count,
        include_intra_bar_features=include_intra_bar_features,
        inter_bar_lookback_bars=inter_bar_lookback_bars,
        processor=processor,
    )

    # MEM-012: Stream bars in batches instead of accumulating all in memory
    # This prevents OOM on large date ranges (2.5+ years) with microstructure
    # Previous: all_bars.extend() caused 2.9GB allocation for 50K bars
    # Now: bounded batches of 10K bars (~60 MB each)
    # Issue #88: _stream_bars_from_trades now yields pl.DataFrame directly (Arrow path)
    bar_batches: list[pl.DataFrame] = []

    for bar_batch_df in _stream_bars_from_trades(
        trades_minimal,
        threshold_decimal_bps,
        include_microstructure,
        processor,
        bar_batch_size=10_000,
    ):
        if not bar_batch_df.is_empty():
            bar_batches.append(bar_batch_df)

    if not bar_batches:
        return _empty_ohlcv_dataframe(), processor

    # MEM-006: Single Polars concat (efficient, no 2x memory spike like pandas)
    from rangebar.processors.api import _arrow_bars_to_pandas

    result_pl = pl.concat(bar_batches)
    result = _arrow_bars_to_pandas(result_pl, include_microstructure=True)

    # Issue #75: Always return all available columns including trade IDs
    # The caller (get_range_bars) handles filtering for user-facing output AFTER cache write.
    # This ensures cache always has trade IDs for data integrity verification.
    return result, processor


def _process_exness_ticks(
    ticks: pl.DataFrame,
    symbol: str,
    threshold_decimal_bps: int,
    validation: str,
    include_incomplete: bool,
    include_microstructure: bool,
    *,
    inter_bar_lookback_count: int | None = None,
) -> pd.DataFrame:
    """Process Exness ticks to range bars (internal).

    Note: inter_bar_lookback_count is accepted but not yet implemented for Exness.
    TODO(Issue #59): Add inter-bar feature support for Exness source.
    """
    _ = inter_bar_lookback_count  # Unused for now, Exness uses separate processing path
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


# Issue #76: Auto-detect earliest available date for Binance symbols
# REMOVED (Issue #79): KNOWN_LISTING_DATES dict migrated to symbols.toml
# SSoT is now python/rangebar/data/symbols.toml (Unified Symbol Registry)


def detect_earliest_available_date(
    symbol: str,
    market: str = "spot",
    *,
    max_probes: int = 10,
) -> str | None:
    """Detect the earliest date with available data for a Binance symbol.

    Uses binary search on Binance Vision API to find the first available date.
    Returns known listing date if available to avoid network overhead.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    market : str
        Market type: "spot", "um" (USDT-M futures), "cm" (COIN-M futures)
    max_probes : int
        Maximum number of HTTP probes for binary search (default: 10)

    Returns
    -------
    str | None
        Earliest available date as "YYYY-MM-DD", or None if detection fails

    Example
    -------
    >>> detect_earliest_available_date("SOLUSDT")
    '2020-08-11'
    """
    import urllib.error
    import urllib.request
    from datetime import UTC, datetime, timedelta

    http_ok = 200

    # Check symbol registry first (no network needed) -- Issue #79
    from rangebar.symbol_registry import get_effective_start_date

    effective = get_effective_start_date(symbol)
    if effective is not None:
        return effective

    # Map market to Vision API path
    market_path = {
        "spot": "spot",
        "um": "futures/um",
        "cm": "futures/cm",
    }.get(market, "spot")

    def check_date(date_str: str) -> bool:
        """Check if data exists for a specific date via HTTP HEAD."""
        url = (
            f"https://data.binance.vision/data/{market_path}/daily/"
            f"aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
        )
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == http_ok
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            # HTTPError: 404/403/etc, URLError: DNS/connection, TimeoutError, OSError: network
            return False

    # Binary search between 2017-01-01 and today
    left = datetime(2017, 1, 1, tzinfo=UTC)
    right = datetime.now(UTC)
    earliest_found: str | None = None

    for _ in range(max_probes):
        if left >= right:
            break

        mid = left + (right - left) // 2
        mid_str = mid.strftime("%Y-%m-%d")

        if check_date(mid_str):
            earliest_found = mid_str
            right = mid - timedelta(days=1)
        else:
            left = mid + timedelta(days=1)

    # Verify the found date
    if earliest_found:
        return earliest_found

    # Fallback: linear probe from recent history
    probe_date = datetime.now(UTC) - timedelta(days=365)
    for _ in range(30):
        date_str = probe_date.strftime("%Y-%m-%d")
        if check_date(date_str):
            return date_str
        probe_date -= timedelta(days=30)

    return None
