"""Constants and presets for rangebar-py.

This module centralizes all constants to eliminate duplication across the codebase.
Import from here instead of defining locally.

SSoT (Single Source of Truth) for:
- MICROSTRUCTURE_COLUMNS: Optional microstructure feature columns
- TIER1_SYMBOLS: High-liquidity crypto symbols
- THRESHOLD_PRESETS: Named threshold values in decimal basis points
- THRESHOLD_DECIMAL_MIN/MAX: Valid threshold range
- _CRYPTO_BASES: Known crypto base symbols for asset class detection
- _FOREX_CURRENCIES: Known forex currencies for asset class detection
"""

from __future__ import annotations

# =============================================================================
# Schema Version Constants (Cache Evolution)
# =============================================================================
# Used for cache validation and schema evolution tracking.
# Increment when schema changes require cache invalidation.
#
# Version history:
# - 6.0.0: OHLCV only (legacy, pre-microstructure)
# - 7.0.0: Added 15 microstructure columns (Issue #25)
# - 10.0.0: Added ouroboros_mode column
# - 11.0.0: Current version with modular architecture

SCHEMA_VERSION_OHLCV_ONLY: str = "6.0.0"  # Pre-microstructure (legacy)
SCHEMA_VERSION_MICROSTRUCTURE: str = "7.0.0"  # Added 15 microstructure columns
SCHEMA_VERSION_OUROBOROS: str = "10.0.0"  # Added ouroboros_mode column

# Minimum versions required for features
MIN_VERSION_FOR_MICROSTRUCTURE: str = SCHEMA_VERSION_MICROSTRUCTURE
MIN_VERSION_FOR_OUROBOROS: str = SCHEMA_VERSION_OUROBOROS

# =============================================================================
# Microstructure Columns (Issue #25, v7.0+)
# =============================================================================
# These columns are optional and only present when include_microstructure=True
# or when bars are generated with microstructure features enabled.
#
# IMPORTANT: Keep this list in sync with:
# - crates/rangebar-core/src/bar.rs (Rust struct fields)
# - python/rangebar/clickhouse/schema.sql (ClickHouse columns)

MICROSTRUCTURE_COLUMNS: tuple[str, ...] = (
    # Basic extended columns
    "vwap",
    "buy_volume",
    "sell_volume",
    "individual_trade_count",
    "agg_record_count",
    # Microstructure features (Issue #25)
    "duration_us",
    "ofi",
    "vwap_close_deviation",
    "price_impact",
    "kyle_lambda_proxy",
    "trade_intensity",
    "volume_per_trade",
    "aggression_ratio",
    "aggregation_density",
    "turnover_imbalance",
)

# =============================================================================
# Tier-1 Symbols (high-liquidity, available on all Binance markets)
# =============================================================================

TIER1_SYMBOLS: tuple[str, ...] = (
    "AAVE",
    "ADA",
    "AVAX",
    "BCH",
    "BNB",
    "BTC",
    "DOGE",
    "ETH",
    "FIL",
    "LINK",
    "LTC",
    "NEAR",
    "SOL",
    "SUI",
    "UNI",
    "WIF",
    "WLD",
    "XRP",
)

# =============================================================================
# Threshold Range (from rangebar-core)
# =============================================================================

THRESHOLD_DECIMAL_MIN: int = 1  # 1 dbps = 0.001%
THRESHOLD_DECIMAL_MAX: int = 100_000  # 100,000 dbps = 100%

# =============================================================================
# Threshold Presets (in decimal basis points)
# =============================================================================
# 1 dbps = 0.001% = 0.00001 (one-tenth of a basis point)
# Example: 250 dbps = 0.25%

THRESHOLD_PRESETS: dict[str, int] = {
    "micro": 10,  # 10 dbps = 0.01% (scalping)
    "tight": 50,  # 50 dbps = 0.05% (day trading)
    "standard": 100,  # 100 dbps = 0.1% (swing trading)
    "medium": 250,  # 250 dbps = 0.25% (default)
    "wide": 500,  # 500 dbps = 0.5% (position trading)
    "macro": 1000,  # 100bps = 1% (long-term)
}

# =============================================================================
# Asset Class Detection Helpers
# =============================================================================

# Common crypto base symbols for detection
_CRYPTO_BASES: frozenset[str] = frozenset(
    {
        "BTC",
        "ETH",
        "BNB",
        "SOL",
        "XRP",
        "ADA",
        "DOGE",
        "DOT",
        "MATIC",
        "AVAX",
        "LINK",
        "UNI",
        "ATOM",
        "LTC",
        "ETC",
        "XLM",
        "ALGO",
        "NEAR",
        "FIL",
        "APT",
    }
)

# Common forex base/quote currencies
_FOREX_CURRENCIES: frozenset[str] = frozenset(
    {
        "EUR",
        "USD",
        "GBP",
        "JPY",
        "CHF",
        "AUD",
        "NZD",
        "CAD",
        "SEK",
        "NOK",
    }
)

# =============================================================================
# Exchange Session Column Names (Ouroboros feature)
# =============================================================================

EXCHANGE_SESSION_COLUMNS: tuple[str, ...] = (
    "exchange_session_sydney",
    "exchange_session_tokyo",
    "exchange_session_london",
    "exchange_session_newyork",
)

# =============================================================================
# All Optional Columns (for cache operations)
# =============================================================================
# Union of microstructure + exchange session columns

ALL_OPTIONAL_COLUMNS: tuple[str, ...] = (
    *MICROSTRUCTURE_COLUMNS,
    *EXCHANGE_SESSION_COLUMNS,
)
