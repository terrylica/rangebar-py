"""Type stubs for rangebar package.

Public API
----------
get_range_bars : Get range bars with automatic data fetching and caching.
TIER1_SYMBOLS : High-liquidity symbols available on all Binance markets.
THRESHOLD_PRESETS : Named threshold presets (micro, tight, standard, etc.).
THRESHOLD_DECIMAL_MIN : Minimum valid threshold (1 = 0.1bps).
THRESHOLD_DECIMAL_MAX : Maximum valid threshold (100,000 = 10,000bps).
__version__ : Package version string.
"""

from typing import Literal

import pandas as pd

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
