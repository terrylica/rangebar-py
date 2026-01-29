"""Type stubs for rangebar package.

Public API
----------
get_range_bars : Get range bars with automatic data fetching and caching (date-bounded).
get_n_range_bars : Get exactly N range bars (count-bounded, deterministic).
precompute_range_bars : Pre-compute continuous range bars for a date range (single-pass).
validate_continuity_tiered : Validate range bar continuity with tiered gap classification.
ContinuityError : Exception raised when range bar continuity is violated.
ContinuityWarning : Warning issued when range bar discontinuities are detected.
PrecomputeProgress : Progress update for precomputation.
PrecomputeResult : Result of precomputation.
GapTier : Gap severity classification enum.
AssetClass : Asset class enum for tolerance calibration.
TierThresholds : Configurable boundaries between gap tiers.
ValidationPreset : Immutable validation configuration preset.
GapInfo : Details of a single gap between consecutive bars.
TierSummary : Per-tier statistics for gap analysis.
TieredValidationResult : Comprehensive validation result with tier breakdown.
TIER1_SYMBOLS : High-liquidity symbols available on all Binance markets.
THRESHOLD_PRESETS : Named threshold presets (micro, tight, standard, etc.).
VALIDATION_PRESETS : Named validation presets (research, strict, crypto, etc.).
THRESHOLD_DECIMAL_MIN : Minimum valid threshold (1 = 0.1bps).
THRESHOLD_DECIMAL_MAX : Maximum valid threshold (100,000 = 10,000bps).
__version__ : Package version string.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Literal, overload

import pandas as pd
import polars as pl

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
    continuity_valid: bool | None
    """Whether all bars pass continuity validation. None if validation was skipped."""
    cache_key: str
    """Cache key for the stored bars."""

__version__: str

# ============================================================================
# Ouroboros: Cyclical Reset Boundaries for Reproducibility
# Plan: /Users/terryli/.claude/plans/sparkling-coalescing-dijkstra.md
# ============================================================================

from datetime import date, datetime

class OuroborosMode(str, Enum):
    """Ouroboros granularity modes for reset boundaries.

    Ouroboros (Greek: οὐροβόρος) represents cyclical reset boundaries
    that enable reproducible range bar construction.
    """

    YEAR = "year"
    """Reset at January 1 00:00:00 UTC each year."""
    MONTH = "month"
    """Reset at 1st of each month 00:00:00 UTC."""
    WEEK = "week"
    """Reset at Sunday 00:00:00 UTC each week (crypto) or first tick after market open (forex)."""

@dataclass(frozen=True)
class OuroborosBoundary:
    """A single ouroboros reset boundary.

    Represents a specific timestamp where the range bar processor should reset
    its state to enable reproducible bar construction across segments.
    """

    timestamp: datetime
    """UTC datetime of the boundary."""
    mode: OuroborosMode
    """Which granularity created this boundary."""
    reason: str
    """Human-readable reason (e.g., 'year_boundary', 'month_boundary')."""

    @property
    def timestamp_ms(self) -> int:
        """Timestamp in milliseconds (for comparison with trade data)."""

    @property
    def timestamp_us(self) -> int:
        """Timestamp in microseconds."""

@dataclass
class OrphanedBarMetadata:
    """Metadata for orphaned bars at ouroboros boundaries.

    Orphaned bars are incomplete bars that existed when the processor
    was reset at an ouroboros boundary. They can be included or excluded
    from results based on the `include_orphaned_bars` parameter.
    """

    is_orphan: bool = True
    """Always True for orphaned bars."""
    ouroboros_boundary: datetime | None = None
    """Which boundary caused the orphan."""
    reason: str | None = None
    """Reason string: 'year_boundary', 'month_boundary', 'week_boundary'."""
    expected_duration_us: int | None = None
    """Expected duration if bar had completed normally."""

def get_ouroboros_boundaries(
    start: date,
    end: date,
    mode: Literal["year", "month", "week"],
) -> list[OuroborosBoundary]:
    """Return all ouroboros reset points within the date range.

    Parameters
    ----------
    start : date
        Start date (inclusive)
    end : date
        End date (inclusive)
    mode : {"year", "month", "week"}
        Ouroboros granularity

    Returns
    -------
    list[OuroborosBoundary]
        Sorted list of boundaries within the date range

    Examples
    --------
    >>> from datetime import date
    >>> from rangebar import get_ouroboros_boundaries
    >>> boundaries = get_ouroboros_boundaries(date(2024, 1, 1), date(2024, 3, 31), "month")
    >>> len(boundaries)
    3
    >>> boundaries[0].reason
    'month_boundary'
    """

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
# Tiered Validation System (Issue #19 - v6.2.0+)
# ============================================================================

class GapTier(IntEnum):
    """Gap severity classification for range bar continuity validation.

    Tiers are based on empirical analysis of 30-month BTC data which
    identified 49 legitimate market microstructure events.
    """

    PRECISION = 1
    """< 0.001% - Floating-point artifacts (always ignored)"""
    NOISE = 2
    """0.001% - 0.01% - Tick-level noise (logged, not flagged)"""
    MARKET_MOVE = 3
    """0.01% - 0.1% - Normal market movement (configurable)"""
    MICROSTRUCTURE = 4
    """> 0.1% - Flash crashes, liquidations (warning/error)"""
    SESSION_BOUNDARY = 5
    """> threshold*2 - Definite session break (always error)"""

class AssetClass(Enum):
    """Asset class for tolerance calibration.

    Different asset classes have different typical gap magnitudes.
    """

    CRYPTO = "crypto"
    """24/7 markets, flash crashes possible"""
    FOREX = "forex"
    """Session-based, weekend gaps"""
    EQUITIES = "equities"
    """Overnight gaps, circuit breakers"""
    UNKNOWN = "unknown"
    """Fallback to crypto defaults"""

ASSET_CLASS_MULTIPLIERS: dict[AssetClass, float]
"""Tolerance multipliers by asset class (relative to baseline)."""

def detect_asset_class(symbol: str) -> AssetClass:
    """Auto-detect asset class from symbol pattern.

    Detection Rules:
    - Crypto: Contains common crypto bases (BTC, ETH, etc.) or ends with USDT/BUSD
    - Forex: Standard 6-char pairs (EURUSD) or commodities (XAU, XAG)
    - Unknown: Fallback for unrecognized patterns

    Parameters
    ----------
    symbol : str
        Trading symbol (case-insensitive)

    Returns
    -------
    AssetClass
        Detected asset class

    Examples
    --------
    >>> detect_asset_class("BTCUSDT")
    <AssetClass.CRYPTO: 'crypto'>
    >>> detect_asset_class("EURUSD")
    <AssetClass.FOREX: 'forex'>
    """

@dataclass(frozen=True)
class TierThresholds:
    """Configurable boundaries between gap tiers (in percentage).

    These thresholds define the boundaries for classifying gaps into tiers.
    Values are percentages (e.g., 0.00001 = 0.001%).
    """

    precision: float = ...
    """Tier 1/2 boundary (default: 0.00001 = 0.001%)"""
    noise: float = ...
    """Tier 2/3 boundary (default: 0.0001 = 0.01%)"""
    market_move: float = ...
    """Tier 3/4 boundary (default: 0.001 = 0.1%)"""
    session_factor: float = ...
    """Tier 5 multiplier (default: 2.0)"""

@dataclass(frozen=True)
class ValidationPreset:
    """Immutable validation configuration preset.

    Presets bundle tolerance, behavior mode, and tier thresholds into
    named configurations for common use cases.
    """

    tolerance_pct: float
    """Maximum gap percentage before flagging (e.g., 0.01 = 1%)"""
    mode: Literal["error", "warn", "skip"]
    """Behavior on validation failure"""
    tier_thresholds: TierThresholds = ...
    """Boundaries for gap tier classification"""
    asset_class: AssetClass | None = ...
    """Override auto-detection if set"""
    description: str = ...
    """Human-readable description of the preset"""

VALIDATION_PRESETS: dict[str, ValidationPreset]
"""Named validation presets for common scenarios.

General-purpose:
- "permissive": 5% tolerance, warn mode
- "research": 2% tolerance, warn mode (exploratory analysis)
- "standard": 1% tolerance, warn mode (production backtesting)
- "strict": 0.5% tolerance, error mode (ML training data)
- "paranoid": 0.1% tolerance, error mode (original v6.1.0 behavior)

Asset-class specific:
- "crypto": 2% tolerance, crypto asset class
- "forex": 1% tolerance, forex asset class
- "equities": 3% tolerance, equities asset class

Special:
- "skip": Disable validation entirely
- "audit": 0.2% tolerance, error mode (data quality audit)
"""

@dataclass
class GapInfo:
    """Details of a single gap between consecutive bars."""

    bar_index: int
    """Index of the bar with the gap (0-based)"""
    prev_close: float
    """Close price of the previous bar"""
    curr_open: float
    """Open price of the current bar"""
    gap_pct: float
    """Gap magnitude as percentage (e.g., 0.01 = 1%)"""
    tier: GapTier
    """Severity classification of this gap"""
    timestamp: pd.Timestamp | None = ...
    """Timestamp of the bar (if available from DataFrame index)"""

@dataclass
class TierSummary:
    """Per-tier statistics for gap analysis."""

    count: int = ...
    """Number of gaps in this tier"""
    max_gap_pct: float = ...
    """Maximum gap percentage in this tier"""
    avg_gap_pct: float = ...
    """Average gap percentage in this tier (0 if count == 0)"""

@dataclass
class TieredValidationResult:
    """Comprehensive validation result with tier breakdown.

    This result provides detailed gap analysis categorized by severity tier,
    enabling nuanced handling of different gap magnitudes.
    """

    is_valid: bool
    """True if no SESSION_BOUNDARY gaps (tier 5) detected"""
    bar_count: int
    """Total number of bars validated"""
    gaps_by_tier: dict[GapTier, TierSummary]
    """Per-tier statistics"""
    all_gaps: list[GapInfo]
    """All gaps above PRECISION tier (detailed list)"""
    threshold_used_pct: float
    """Range bar threshold used for validation (as percentage)"""
    asset_class_detected: AssetClass
    """Auto-detected or overridden asset class"""
    preset_used: str | None
    """Name of preset used, or None for custom config"""

    @property
    def has_session_breaks(self) -> bool:
        """True if any SESSION_BOUNDARY gaps detected."""

    @property
    def has_microstructure_events(self) -> bool:
        """True if any MICROSTRUCTURE gaps detected."""

    def summary_dict(self) -> dict[str, int]:
        """Return gap counts by tier name for logging.

        Returns
        -------
        dict[str, int]
            Mapping of tier name to gap count
        """

def validate_continuity_tiered(
    df: pd.DataFrame,
    threshold_decimal_bps: int = 250,
    *,
    validation: str | dict | ValidationPreset = "standard",
    symbol: str | None = None,
) -> TieredValidationResult:
    """Validate range bar continuity with tiered gap classification.

    This function categorizes gaps by severity tier, enabling nuanced
    handling of different gap magnitudes. It's the opt-in v6.2.0 API
    that will become the default in v7.0.

    Parameters
    ----------
    df : pd.DataFrame
        Range bar DataFrame with OHLCV columns
    threshold_decimal_bps : int, default=250
        Range bar threshold (250 = 0.25% = 25 basis points)
    validation : str, dict, or ValidationPreset, default="standard"
        Validation configuration:
        - "auto": Auto-detect asset class from symbol
        - str: Preset name ("research", "strict", "crypto", etc.)
        - dict: Custom config {"tolerance_pct": 0.01, "mode": "warn"}
        - ValidationPreset: Direct preset instance
    symbol : str, optional
        Symbol for asset class auto-detection

    Returns
    -------
    TieredValidationResult
        Comprehensive result with per-tier statistics

    Raises
    ------
    ContinuityError
        If validation mode is "error" and tolerance exceeded
    ContinuityWarning
        If validation mode is "warn" and tolerance exceeded (via warnings module)

    Examples
    --------
    >>> result = validate_continuity_tiered(df, validation="research")
    >>> print(f"Valid: {result.is_valid}")
    Valid: True
    """

# ============================================================================
# Main API
# ============================================================================

@overload
def get_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: (
        int | Literal["micro", "tight", "standard", "medium", "wide", "macro"]
    ) = 250,
    *,
    ouroboros: Literal["year", "month", "week"] = ...,
    include_orphaned_bars: bool = ...,
    materialize: Literal[True] = ...,
    batch_size: int = ...,
    source: Literal["binance", "exness"] = ...,
    market: Literal["spot", "futures-um", "futures-cm", "um", "cm"] = ...,
    validation: Literal["permissive", "strict", "paranoid"] = ...,
    include_incomplete: bool = ...,
    include_microstructure: bool = ...,
    include_exchange_sessions: bool = ...,  # Issue #8
    prevent_same_timestamp_close: bool = ...,
    verify_checksum: bool = ...,
    use_cache: bool = ...,
    fetch_if_missing: bool = ...,
    cache_dir: str | None = ...,
) -> pd.DataFrame: ...
@overload
def get_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: (
        int | Literal["micro", "tight", "standard", "medium", "wide", "macro"]
    ) = 250,
    *,
    ouroboros: Literal["year", "month", "week"] = ...,
    include_orphaned_bars: bool = ...,
    materialize: Literal[False],
    batch_size: int = ...,
    source: Literal["binance", "exness"] = ...,
    market: Literal["spot", "futures-um", "futures-cm", "um", "cm"] = ...,
    validation: Literal["permissive", "strict", "paranoid"] = ...,
    include_incomplete: bool = ...,
    include_microstructure: bool = ...,
    include_exchange_sessions: bool = ...,  # Issue #8
    prevent_same_timestamp_close: bool = ...,
    verify_checksum: bool = ...,
    use_cache: bool = ...,
    fetch_if_missing: bool = ...,
    cache_dir: str | None = ...,
) -> Iterator[pl.DataFrame]: ...
def get_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: (
        int | Literal["micro", "tight", "standard", "medium", "wide", "macro"]
    ) = 250,
    *,
    # Ouroboros: Cyclical reset boundaries (v11.0+)
    ouroboros: Literal["year", "month", "week"] = "year",
    include_orphaned_bars: bool = False,
    # Streaming options (v8.0+)
    materialize: bool = True,
    batch_size: int = 10_000,
    # Data source configuration
    source: Literal["binance", "exness"] = "binance",
    market: Literal["spot", "futures-um", "futures-cm", "um", "cm"] = "spot",
    # Exness-specific options
    validation: Literal["permissive", "strict", "paranoid"] = "strict",
    # Processing options
    include_incomplete: bool = False,
    include_microstructure: bool = False,
    include_exchange_sessions: bool = False,  # Issue #8: Exchange session flags
    prevent_same_timestamp_close: bool = True,
    # Data integrity (Issue #43)
    verify_checksum: bool = True,
    # Caching options
    use_cache: bool = True,
    fetch_if_missing: bool = True,
    cache_dir: str | None = None,
) -> pd.DataFrame | Iterator[pl.DataFrame]:
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
    materialize : bool, default=True
        If True, return a single pd.DataFrame (legacy behavior).
        If False, return an Iterator[pl.DataFrame] that yields batches
        of bars for memory-efficient streaming (v8.0+).
    batch_size : int, default=10_000
        Number of bars per batch when materialize=False.
        Each batch is ~500 KB. Only used in streaming mode.

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
        - (Issue #25) duration_us: Bar duration in microseconds
        - (Issue #25) ofi: Order Flow Imbalance [-1, 1]
        - (Issue #25) vwap_close_deviation: (close - vwap) / (high - low)
        - (Issue #25) price_impact: Amihud-style illiquidity
        - (Issue #25) kyle_lambda_proxy: Market depth proxy
        - (Issue #25) trade_intensity: Trades per second
        - (Issue #25) volume_per_trade: Average trade size
        - (Issue #25) aggression_ratio: Buy/sell trade count ratio
        - (Issue #25) aggregation_density: Trade fragmentation proxy
        - (Issue #25) turnover_imbalance: Dollar-weighted OFI [-1, 1]
    prevent_same_timestamp_close : bool, default=True
        Prevent consecutive bars from having identical timestamps.
    verify_checksum : bool, default=True
        Verify SHA-256 checksum of downloaded data (Issue #43).
        Enabled by default for data integrity. Set to False for
        faster downloads when data integrity is verified elsewhere.
    use_cache : bool, default=True
        Cache tick data locally in Parquet format.
    cache_dir : str or None, default=None
        Custom cache directory. If None, uses platform default:
        - macOS: ~/Library/Caches/rangebar/
        - Linux: ~/.cache/rangebar/
        - Windows: %LOCALAPPDATA%/terrylica/rangebar/Cache/

    Returns
    -------
    pd.DataFrame or Iterator[pl.DataFrame]
        If materialize=True (default): Single pd.DataFrame ready for
        backtesting.py, with DatetimeIndex and OHLCV columns.

        If materialize=False: Iterator yielding pl.DataFrame batches
        (batch_size bars each) for memory-efficient streaming.

        Columns: Open, High, Low, Close, Volume
        (if include_microstructure) Additional columns

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

def get_range_bars_pandas(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: (
        int | Literal["micro", "tight", "standard", "medium", "wide", "macro"]
    ) = 250,
    **kwargs: Any,  # noqa: ANN401
) -> pd.DataFrame:
    """Get range bars as pandas DataFrame (deprecated compatibility shim).

    .. deprecated:: 8.0
        Use ``get_range_bars(materialize=True)`` directly instead.
        This function will be removed in v9.0.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    threshold_decimal_bps : int or str, default=250
        Threshold in decimal basis points
    **kwargs
        Additional arguments passed to ``get_range_bars()``

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py
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
        Include microstructure columns (vwap, buy_volume, sell_volume,
        plus Issue #25 features: ofi, duration_us, price_impact, etc.)
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
    validate_on_complete: Literal["error", "warn", "skip"] = "error",
    continuity_tolerance_pct: float = 0.001,
    cache_dir: str | None = None,
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
    validate_on_complete : str, default="error"
        Continuity validation mode after precomputation:
        - "error": Raise ContinuityError if discontinuities found
        - "warn": Log warning but continue (sets continuity_valid=False)
        - "skip": Skip validation entirely (continuity_valid=None)
    continuity_tolerance_pct : float, default=0.001
        Maximum allowed price gap percentage for continuity validation.
        Default 0.1% (0.001) accommodates market microstructure events.
        The total allowed gap is threshold_pct + continuity_tolerance_pct.
    cache_dir : str or None, optional
        Custom cache directory for tick data.

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

def process_trades_polars(
    trades: pl.DataFrame | pl.LazyFrame,
    threshold_decimal_bps: int = 250,
) -> pd.DataFrame:
    """Process trades from Polars DataFrame (optimized pipeline).

    This is the recommended API for Polars users. Uses lazy evaluation
    and minimal dict conversion for best performance.

    Parameters
    ----------
    trades : polars.DataFrame or polars.LazyFrame
        Trade data with columns:
        - timestamp: int64 (milliseconds since epoch)
        - price: float
        - quantity (or volume): float
    threshold_decimal_bps : int, default=250
        Threshold in decimal basis points (250 = 25bps = 0.25%)

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py, with:
        - DatetimeIndex (timestamp)
        - Capitalized columns: Open, High, Low, Close, Volume

    Examples
    --------
    With LazyFrame (predicate pushdown):

    >>> import polars as pl
    >>> from rangebar import process_trades_polars
    >>> lazy_df = pl.scan_parquet("trades.parquet")
    >>> lazy_filtered = lazy_df.filter(pl.col("timestamp") >= 1704067200000)
    >>> df = process_trades_polars(lazy_filtered, threshold_decimal_bps=250)

    With DataFrame:

    >>> df = pl.read_parquet("trades.parquet")
    >>> bars = process_trades_polars(df)

    Notes
    -----
    Performance optimization:
    - Only required columns are extracted (timestamp, price, quantity)
    - Lazy evaluation: predicates pushed to I/O layer
    - 2-3x faster than process_trades_to_dataframe() for Polars inputs

    See Also
    --------
    process_trades_to_dataframe : Process trades from pandas DataFrame or dict list
    get_range_bars : Full pipeline with data fetching and caching
    """

def process_trades_to_dataframe(
    trades: list[dict] | pd.DataFrame,
    threshold_decimal_bps: int = 250,
    include_microstructure: bool = False,
) -> pd.DataFrame:
    """Process trades into range bars from pandas DataFrame or dict list.

    Parameters
    ----------
    trades : list[dict] or pd.DataFrame
        Trade data. If list[dict], each dict needs:
        - timestamp: int (milliseconds since epoch)
        - price: float
        - quantity: float
    threshold_decimal_bps : int, default=250
        Threshold in decimal basis points (250 = 25bps = 0.25%)
    include_microstructure : bool, default=False
        Include microstructure columns (vwap, buy_volume, sell_volume)

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting.py

    See Also
    --------
    process_trades_polars : Faster alternative for Polars inputs
    get_range_bars : Full pipeline with data fetching and caching
    """
