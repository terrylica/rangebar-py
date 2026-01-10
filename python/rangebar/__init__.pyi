"""Type stubs for rangebar package.

Public API
----------
get_range_bars : Get range bars with automatic data fetching and caching (date-bounded).
get_n_range_bars : Get exactly N range bars (count-bounded, deterministic).
precompute_range_bars : Pre-compute continuous range bars for a date range (single-pass).
ContinuityError : Exception raised when range bar continuity is violated.
ContinuityWarning : Warning issued when range bar discontinuities are detected.
PrecomputeProgress : Progress update for precomputation.
PrecomputeResult : Result of precomputation.
TIER1_SYMBOLS : High-liquidity symbols available on all Binance markets.
THRESHOLD_PRESETS : Named threshold presets (micro, tight, standard, etc.).
THRESHOLD_DECIMAL_MIN : Minimum valid threshold (1 = 0.1bps).
THRESHOLD_DECIMAL_MAX : Maximum valid threshold (100,000 = 10,000bps).
__version__ : Package version string.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pandas as pd

# ============================================================================
# Exceptions and Warnings
# ============================================================================

class ContinuityError(Exception):
    """Raised when range bar continuity is violated.

    The bar[i+1].open == bar[i].close invariant is broken, indicating
    discontinuities in the range bar sequence.
    """

    discontinuities: list[dict]
    """List of discontinuity details (bar_index, prev_close, next_open, gap_pct)."""

    def __init__(
        self, message: str, discontinuities: list[dict] | None = None
    ) -> None: ...

class ContinuityWarning(UserWarning):
    """Warning issued when range bar discontinuities are detected but not fatal."""

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PrecomputeProgress:
    """Progress update for precomputation.

    Passed to progress_callback during precompute_range_bars() execution.
    """

    phase: Literal["fetching", "processing", "caching"]
    """Current processing phase."""
    current_month: str
    """Current month being processed (YYYY-MM format)."""
    months_completed: int
    """Number of months already processed."""
    months_total: int
    """Total number of months to process."""
    bars_generated: int
    """Cumulative bars generated so far."""
    ticks_processed: int
    """Cumulative ticks processed so far."""
    elapsed_seconds: float
    """Seconds elapsed since precomputation started."""

@dataclass
class PrecomputeResult:
    """Result of precomputation.

    Returned by precompute_range_bars() after successful execution.
    """

    symbol: str
    """Trading symbol that was precomputed."""
    threshold_decimal_bps: int
    """Threshold used for bar construction."""
    start_date: str
    """Start date of precomputed range (YYYY-MM-DD)."""
    end_date: str
    """End date of precomputed range (YYYY-MM-DD)."""
    total_bars: int
    """Total number of bars generated."""
    total_ticks: int
    """Total number of ticks processed."""
    elapsed_seconds: float
    """Total time taken for precomputation."""
    continuity_valid: bool
    """Whether all bars pass continuity validation."""
    cache_key: str
    """Cache key for the stored bars."""

__version__: str

# ============================================================================
# Configuration Constants
# ============================================================================

TIER1_SYMBOLS: tuple[str, ...]
"""18 high-liquidity symbols available on ALL Binance markets.

AAVE, ADA, AVAX, BCH, BNB, BTC, DOGE, ETH, FIL,
LINK, LTC, NEAR, SOL, SUI, UNI, WIF, WLD, XRP
"""

THRESHOLD_DECIMAL_MIN: int
"""Minimum valid threshold: 1 (0.1bps = 0.001%)"""

THRESHOLD_DECIMAL_MAX: int
"""Maximum valid threshold: 100,000 (10,000bps = 100%)"""

THRESHOLD_PRESETS: dict[str, int]
"""Named threshold presets (in 0.1bps units).

- "micro": 10 (1bps = 0.01%) - scalping
- "tight": 50 (5bps = 0.05%) - day trading
- "standard": 100 (10bps = 0.1%) - swing trading
- "medium": 250 (25bps = 0.25%) - default
- "wide": 500 (50bps = 0.5%) - position trading
- "macro": 1000 (100bps = 1%) - long-term
"""

# ============================================================================
# Main API
# ============================================================================

def get_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: (
        int | Literal["micro", "tight", "standard", "medium", "wide", "macro"]
    ) = 250,
    *,
    # Data source configuration
    source: Literal["binance", "exness"] = "binance",
    market: Literal["spot", "futures-um", "futures-cm", "um", "cm"] = "spot",
    # Exness-specific options
    validation: Literal["permissive", "strict", "paranoid"] = "strict",
    # Processing options
    include_incomplete: bool = False,
    include_microstructure: bool = False,
    # Caching options
    use_cache: bool = True,
    fetch_if_missing: bool = True,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Get range bars for a symbol with automatic data fetching and caching.

    This is the single entry point for all range bar generation. It supports
    multiple data sources (Binance crypto, Exness forex), all market types,
    and exposes the full configurability of the underlying Rust engine.

    Parameters
    ----------
    symbol : str
        Trading symbol (uppercase).
        - Binance: "BTCUSDT", "ETHUSDT", etc.
        - Exness: "EURUSD", "GBPUSD", "XAUUSD", etc.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    threshold_decimal_bps : int or str, default=250
        Threshold in 0.1bps units. Can be:
        - Integer: Direct value (250 = 25bps = 0.25%)
        - String preset: "micro" (1bps), "tight" (5bps), "standard" (10bps),
          "medium" (25bps), "wide" (50bps), "macro" (100bps)
        Valid range: 1-100,000 (0.001% to 100%)

    source : str, default="binance"
        Data source: "binance" or "exness"
    market : str, default="spot"
        Market type (Binance only):
        - "spot": Spot market
        - "futures-um" or "um": USD-M perpetual futures
        - "futures-cm" or "cm": COIN-M perpetual futures
    validation : str, default="strict"
        Validation strictness (Exness only):
        - "permissive": Basic checks (bid > 0, ask > 0, bid < ask)
        - "strict": + Spread < 10% (catches obvious errors)
        - "paranoid": + Spread < 1% (flags suspicious data)
    include_incomplete : bool, default=False
        Include the final incomplete bar (useful for analysis).
        If False (default), only completed bars are returned.
    include_microstructure : bool, default=False
        Include market microstructure columns:
        - buy_volume, sell_volume: Volume by aggressor side
        - vwap: Volume-weighted average price
        - trade_count: Number of trades in bar
        - (Exness) spread_min, spread_max, spread_avg: Spread statistics
    use_cache : bool, default=True
        Cache tick data locally in Parquet format.
    cache_dir : str or None, default=None
        Custom cache directory. If None, uses platform default:
        - macOS: ~/Library/Caches/rangebar/
        - Linux: ~/.cache/rangebar/
        - Windows: %LOCALAPPDATA%/terrylica/rangebar/Cache/

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py, with:
        - DatetimeIndex (timestamp)
        - Columns: Open, High, Low, Close, Volume
        - (if include_microstructure) Additional columns

    Raises
    ------
    ValueError
        - Invalid threshold (outside 1-100,000 range)
        - Invalid dates or date format
        - Unknown source, market, or validation level
        - Unknown threshold preset name
    RuntimeError
        - Data fetching failed
        - No data available for date range
        - Feature not enabled (e.g., Exness without exness feature)

    Examples
    --------
    Basic usage - Binance spot:

    >>> from rangebar import get_range_bars
    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

    Using threshold presets:

    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", threshold_decimal_bps="tight")

    Binance USD-M Futures:

    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", market="futures-um")

    Exness forex with spread monitoring:

    >>> df = get_range_bars(
    ...     "EURUSD", "2024-01-01", "2024-01-31",
    ...     source="exness",
    ...     threshold_decimal_bps="standard",
    ...     include_microstructure=True,
    ... )

    Use with backtesting.py:

    >>> from backtesting import Backtest, Strategy
    >>> df = get_range_bars("BTCUSDT", "2024-01-01", "2024-12-31")
    >>> bt = Backtest(df, MyStrategy, cash=10000, commission=0.0002)
    >>> stats = bt.run()

    Notes
    -----
    Threshold units (0.1bps):
        The threshold is specified in tenths of basis points for precision.
        Common conversions:
        - 10 = 1bps = 0.01%
        - 100 = 10bps = 0.1%
        - 250 = 25bps = 0.25%
        - 1000 = 100bps = 1%

    Tier-1 symbols:
        18 high-liquidity symbols available on ALL Binance markets:
        AAVE, ADA, AVAX, BCH, BNB, BTC, DOGE, ETH, FIL,
        LINK, LTC, NEAR, SOL, SUI, UNI, WIF, WLD, XRP

    Non-lookahead guarantee:
        - Threshold computed from bar OPEN price only
        - Breaching trade included in closing bar
        - No future information used in bar construction
    """

def get_n_range_bars(
    symbol: str,
    n_bars: int,
    threshold_decimal_bps: (
        int | Literal["micro", "tight", "standard", "medium", "wide", "macro"]
    ) = 250,
    *,
    end_date: str | None = None,
    source: Literal["binance", "exness"] = "binance",
    market: Literal["spot", "futures-um", "futures-cm", "um", "cm"] = "spot",
    include_microstructure: bool = False,
    use_cache: bool = True,
    fetch_if_missing: bool = True,
    max_lookback_days: int = 90,
    warn_if_fewer: bool = True,
    validate_on_return: bool = False,
    continuity_action: Literal["warn", "raise", "log"] = "warn",
    chunk_size: int = 100_000,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Get exactly N range bars ending at or before a given date.

    Unlike `get_range_bars()` which uses date bounds (producing variable bar counts),
    this function returns a deterministic number of bars. This is useful for:
    - ML training (exactly 10,000 samples)
    - Walk-forward optimization (fixed window sizes)
    - Consistent backtest comparisons

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    n_bars : int
        Number of bars to retrieve. Must be > 0.
    threshold_decimal_bps : int or str, default=250
        Threshold in decimal basis points. Can be:
        - Integer: Direct value (250 = 25bps = 0.25%)
        - String preset: "micro", "tight", "standard", "medium", "wide", "macro"
    end_date : str or None, default=None
        End date in YYYY-MM-DD format. If None, uses most recent available data.
    source : str, default="binance"
        Data source: "binance" or "exness"
    market : str, default="spot"
        Market type (Binance only): "spot", "futures-um", or "futures-cm"
    include_microstructure : bool, default=False
        Include microstructure columns (vwap, buy_volume, sell_volume)
    use_cache : bool, default=True
        Use ClickHouse cache for bar retrieval/storage
    fetch_if_missing : bool, default=True
        Fetch and process new data if cache doesn't have enough bars
    max_lookback_days : int, default=90
        Safety limit: maximum days to look back when fetching missing data.
        Prevents runaway fetches on empty caches.
    warn_if_fewer : bool, default=True
        Emit UserWarning if returning fewer bars than requested.
    validate_on_return : bool, default=False
        If True, validate bar continuity before returning.
        Uses continuity_action to determine behavior on failure.
    continuity_action : str, default="warn"
        Action when discontinuity found during validation:
        - "warn": Log warning but return data
        - "raise": Raise ContinuityError
        - "log": Silent logging only
    chunk_size : int, default=100_000
        Number of ticks per processing chunk for memory efficiency.
        Larger values = faster processing, more memory.
        Default 100K = ~15MB memory overhead.
    cache_dir : str or None, default=None
        Custom cache directory for tick data (Tier 1).

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with exactly n_bars rows (or fewer if not enough data),
        sorted chronologically (oldest first). Columns:
        - Open, High, Low, Close, Volume
        - (if include_microstructure) vwap, buy_volume, sell_volume

    Raises
    ------
    ValueError
        - n_bars <= 0
        - Invalid threshold
        - Invalid date format
    RuntimeError
        - ClickHouse not available when use_cache=True
        - Data fetching failed

    Examples
    --------
    Get last 10,000 bars for ML training:

    >>> from rangebar import get_n_range_bars
    >>> df = get_n_range_bars("BTCUSDT", n_bars=10000)
    >>> assert len(df) == 10000

    Get 5,000 bars ending at specific date for walk-forward:

    >>> df = get_n_range_bars("BTCUSDT", n_bars=5000, end_date="2024-06-01")

    With safety limit (won't fetch more than 30 days of data):

    >>> df = get_n_range_bars("BTCUSDT", n_bars=1000, max_lookback_days=30)

    Notes
    -----
    Cache behavior:
        - Fast path: If cache has >= n_bars, returns immediately (~50ms)
        - Slow path: If cache has < n_bars and fetch_if_missing=True,
          fetches additional data, computes bars, stores in cache, returns

    Gap-filling algorithm:
        Uses adaptive exponential backoff to estimate how many ticks to fetch.
        Learns compression ratio (ticks/bar) for each (symbol, threshold) pair.

    See Also
    --------
    get_range_bars : Date-bounded bar retrieval (variable bar count)
    precompute_range_bars : Pre-compute continuous bars for WFO workflows
    THRESHOLD_PRESETS : Named threshold values
    """

def precompute_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: (
        int | Literal["micro", "tight", "standard", "medium", "wide", "macro"]
    ) = 250,
    *,
    source: Literal["binance", "exness"] = "binance",
    market: Literal["spot", "futures-um", "futures-cm", "um", "cm"] = "spot",
    chunk_size: int = 100_000,
    invalidate_existing: Literal["overlap", "full", "none", "smart"] = "smart",
    progress_callback: Callable[[PrecomputeProgress], None] | None = None,
    include_microstructure: bool = False,
    validate_on_complete: bool = True,
) -> PrecomputeResult:
    """Precompute continuous range bars for a date range (single-pass, guaranteed continuity).

    Designed for ML workflows requiring continuous bar sequences for training/validation.
    Uses single-pass processing to guarantee the bar[i+1].open == bar[i].close invariant.

    Parameters
    ----------
    symbol : str
        Trading pair (e.g., "BTCUSDT")
    start_date : str
        Start date (inclusive) "YYYY-MM-DD"
    end_date : str
        End date (inclusive) "YYYY-MM-DD"
    threshold_decimal_bps : int or str, default=250
        Range bar threshold. Can be integer (250 = 0.25%) or preset name.
    source : str, default="binance"
        Data source: "binance" or "exness"
    market : str, default="spot"
        Market type for Binance: "spot", "futures-um"/"um", or "futures-cm"/"cm"
    chunk_size : int, default=100_000
        Ticks per processing chunk (~15MB memory per 100K ticks)
    invalidate_existing : str, default="smart"
        Cache invalidation strategy:
        - "overlap": Invalidate only bars in date range
        - "full": Invalidate ALL bars for symbol/threshold
        - "none": Skip if any cached bars exist in range
        - "smart": Invalidate overlapping + validate junction continuity
    progress_callback : callable, optional
        Callback for progress updates. Receives PrecomputeProgress dataclass.
    include_microstructure : bool, default=False
        Include order flow metrics (vwap, buy_volume, sell_volume)
    validate_on_complete : bool, default=True
        Validate continuity after precomputation. Raises ContinuityError if failed.

    Returns
    -------
    PrecomputeResult
        Dataclass with statistics: total_bars, total_ticks, elapsed_seconds,
        continuity_valid, cache_key

    Raises
    ------
    ValueError
        Invalid parameters (dates, threshold, symbol)
    RuntimeError
        Fetch or processing failure
    ContinuityError
        If validate_on_complete=True and discontinuities found

    Examples
    --------
    Basic precomputation:

    >>> from rangebar import precompute_range_bars
    >>> result = precompute_range_bars("BTCUSDT", "2024-01-01", "2024-03-31")
    >>> print(f"Generated {result.total_bars} bars in {result.elapsed_seconds:.1f}s")

    With progress callback:

    >>> def on_progress(p):
    ...     print(f"{p.phase}: {p.months_completed}/{p.months_total} months")
    >>> precompute_range_bars("BTCUSDT", "2024-01-01", "2024-06-30",
    ...                       progress_callback=on_progress)

    See Also
    --------
    get_n_range_bars : Count-bounded bar retrieval (uses precomputed cache)
    get_range_bars : Date-bounded bar retrieval
    """
