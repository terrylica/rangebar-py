# polars-exception: backtesting.py requires Pandas DataFrames for OHLCV data
"""rangebar: Python bindings for range bar construction.

This package provides high-performance range bar construction for cryptocurrency
trading backtesting, with non-lookahead bias guarantees and temporal integrity.

Examples
--------
Basic usage:

>>> from rangebar import process_trades_to_dataframe
>>> import pandas as pd
>>>
>>> # Load Binance aggTrades data
>>> trades = pd.read_csv("BTCUSDT-aggTrades-2024-01.csv")
>>>
>>> # Convert to range bars (25 basis points = 0.25%)
>>> df = process_trades_to_dataframe(trades, threshold_decimal_bps=250)
>>>
>>> # Use with backtesting.py
>>> from backtesting import Backtest, Strategy
>>> bt = Backtest(df, MyStrategy, cash=10000, commission=0.0002)
>>> stats = bt.run()
"""

from __future__ import annotations

from ._core import PositionVerification, __version__

__all__ = [
    # Constants (from rangebar.constants - SSoT)
    "ALL_OPTIONAL_COLUMNS",
    "ASSET_CLASS_MULTIPLIERS",
    "EXCHANGE_SESSION_COLUMNS",
    "MICROSTRUCTURE_COLUMNS",
    "MIN_VERSION_FOR_MICROSTRUCTURE",
    "MIN_VERSION_FOR_OUROBOROS",
    "SCHEMA_VERSION_MICROSTRUCTURE",
    "SCHEMA_VERSION_OHLCV_ONLY",
    "SCHEMA_VERSION_OUROBOROS",
    "THRESHOLD_DECIMAL_MAX",
    "THRESHOLD_DECIMAL_MIN",
    "THRESHOLD_PRESETS",
    "TIER1_SYMBOLS",
    "VALIDATION_PRESETS",
    # Core classes
    "AssetClass",
    "ContinuityError",
    "ContinuityWarning",
    "GapInfo",
    "GapTier",
    "OrphanedBarMetadata",
    "OuroborosBoundary",
    "OuroborosMode",
    "PositionVerification",
    "PrecomputeProgress",
    "PrecomputeResult",
    "RangeBarProcessor",
    "StalenessResult",
    "TierSummary",
    "TierThresholds",
    "TieredValidationResult",
    "ValidationPreset",
    "__version__",
    # Functions
    "detect_asset_class",
    "detect_staleness",
    "get_n_range_bars",
    "get_ouroboros_boundaries",
    "get_range_bars",
    "get_range_bars_pandas",
    # Conversion utilities (from rangebar.conversion - SSoT)
    "normalize_arrow_dtypes",
    "normalize_temporal_precision",
    "populate_cache_resumable",
    "precompute_range_bars",
    "process_trades_polars",
    "process_trades_to_dataframe",
    "validate_continuity",
    "validate_continuity_tiered",
]

# Re-export checkpoint API per plan (#40)
from .checkpoint import populate_cache_resumable

# Import constants from centralized module (SSoT)
from .constants import (
    ALL_OPTIONAL_COLUMNS,
    EXCHANGE_SESSION_COLUMNS,
    MICROSTRUCTURE_COLUMNS,
    MIN_VERSION_FOR_MICROSTRUCTURE,
    MIN_VERSION_FOR_OUROBOROS,
    SCHEMA_VERSION_MICROSTRUCTURE,
    SCHEMA_VERSION_OHLCV_ONLY,
    SCHEMA_VERSION_OUROBOROS,
    THRESHOLD_DECIMAL_MAX,
    THRESHOLD_DECIMAL_MIN,
    THRESHOLD_PRESETS,
    TIER1_SYMBOLS,
)

# Import conversion utilities from centralized module (SSoT)
from .conversion import normalize_arrow_dtypes, normalize_temporal_precision

# Import orchestration functions from extracted module (M4 modularization)
from .orchestration.count_bounded import get_n_range_bars
from .orchestration.models import PrecomputeProgress, PrecomputeResult
from .orchestration.precompute import precompute_range_bars
from .orchestration.range_bars import get_range_bars, get_range_bars_pandas

# Re-export ouroboros API (cyclical reset boundaries for reproducibility)
from .ouroboros import (
    OrphanedBarMetadata,
    OuroborosBoundary,
    OuroborosMode,
    get_ouroboros_boundaries,
)

# Import RangeBarProcessor from extracted module (M2 modularization)
# Import process_trades_* functions from extracted module (M3 modularization)
from .processors.api import (
    process_trades_chunked,
    process_trades_polars,
    process_trades_to_dataframe,
    process_trades_to_dataframe_cached,
)
from .processors.core import RangeBarProcessor

# Import staleness detection for cache validation (Issue #39: Schema Evolution)
from .validation.cache_staleness import StalenessResult, detect_staleness

# Import continuity validation from extracted module (M1 modularization)
from .validation.continuity import (
    ASSET_CLASS_MULTIPLIERS,
    VALIDATION_PRESETS,
    AssetClass,
    ContinuityError,
    ContinuityWarning,
    GapInfo,
    GapTier,
    TieredValidationResult,
    TierSummary,
    TierThresholds,
    ValidationPreset,
    detect_asset_class,
    validate_continuity,
    validate_continuity_tiered,
    validate_junction_continuity,
)

# Re-export ClickHouse components for convenience
# NOTE: TIER1_SYMBOLS, THRESHOLD_PRESETS, THRESHOLD_DECIMAL_MIN/MAX
# are imported from constants.py (SSoT) at the top of this file


# =============================================================================
# Tiered Validation System (Issue #19 - v6.2.0+)
# =============================================================================
# MOVED to rangebar.validation.continuity (M1 modularization)
# All types, presets, and functions are imported at the top of this file.


# The following block was removed during M1 modularization.
# detect_asset_class, GapTier, AssetClass, TierThresholds, ValidationPreset,
# VALIDATION_PRESETS, ASSET_CLASS_MULTIPLIERS, GapInfo, TierSummary,
# TieredValidationResult, _resolve_validation, _classify_gap,
# validate_continuity_tiered are now imported from rangebar.validation.continuity.

# The following blocks were removed during M4 modularization:
# PrecomputeProgress, PrecomputeResult → rangebar.orchestration.models
# get_range_bars, get_range_bars_pandas → rangebar.orchestration.range_bars
# precompute_range_bars → rangebar.orchestration.precompute
# get_n_range_bars, _fill_gap_and_cache, _fetch_and_compute_bars → rangebar.orchestration.count_bounded
# _stream_range_bars_binance, _fetch_binance, _fetch_exness → rangebar.orchestration.helpers
# _process_binance_trades, _process_exness_ticks → rangebar.orchestration.helpers



def __getattr__(name: str) -> object:
    """Lazy attribute access for ClickHouse and Exness components."""
    if name in {
        "RangeBarCache",
        "CacheKey",
        "get_available_clickhouse_host",
        "detect_clickhouse_state",
        "ClickHouseNotConfiguredError",
    }:
        from . import clickhouse

        return getattr(clickhouse, name)

    if name == "is_exness_available":
        from .exness import is_exness_available

        return is_exness_available

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
