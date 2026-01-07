"""ClickHouse cache layer for computed range bars.

This module provides caching for computed range bars (Tier 2) using ClickHouse.
Raw tick data (Tier 1) is stored locally via `rangebar.storage.TickStorage`.

Configuration
-------------
Set environment variables via mise (recommended) or directly:

    # Connection mode (RANGEBAR_MODE)
    export RANGEBAR_MODE=local  # Force localhost:8123 only
    export RANGEBAR_MODE=cloud  # Require CLICKHOUSE_HOST env var
    export RANGEBAR_MODE=auto   # Auto-detect (default)

    # Host configuration (for AUTO/CLOUD modes)
    export RANGEBAR_CH_HOSTS="host1,host2"  # SSH aliases from ~/.ssh/config
    export RANGEBAR_CH_PRIMARY="host1"       # Default host
    export CLICKHOUSE_HOST="localhost"       # Direct host (CLOUD mode)

If no env vars set, falls back to localhost:8123 in AUTO mode.

Example
-------
>>> from rangebar.clickhouse import RangeBarCache, get_available_clickhouse_host
>>> from rangebar import process_trades_to_dataframe_cached
>>>
>>> # Check ClickHouse availability
>>> host = get_available_clickhouse_host()
>>> print(f"Using ClickHouse at {host.host} via {host.method}")
>>>
>>> # Use cached processing
>>> df = process_trades_to_dataframe_cached(trades, symbol="BTCUSDT")

See Also
--------
rangebar.storage.TickStorage : Local Parquet storage for raw tick data
"""

from __future__ import annotations

import os
import warnings

from .cache import CacheKey, RangeBarCache
from .client import (
    ClickHouseQueryError,
    ClickHouseUnavailableError,
    get_client,
)
from .config import ClickHouseConfig, ConnectionMode, get_connection_mode
from .mixin import ClickHouseClientMixin
from .preflight import (
    ClickHouseNotConfiguredError,
    HostConnection,
    InstallationLevel,
    PreflightResult,
    detect_clickhouse_state,
    get_available_clickhouse_host,
)
from .tunnel import SSHTunnel

__all__ = [
    # Sorted for ruff RUF022
    "CacheKey",
    "ClickHouseClientMixin",
    "ClickHouseConfig",
    "ClickHouseNotConfiguredError",
    "ClickHouseQueryError",
    "ClickHouseUnavailableError",
    "ConnectionMode",
    "HostConnection",
    "InstallationLevel",
    "PreflightResult",
    "RangeBarCache",
    "SSHTunnel",
    "detect_clickhouse_state",
    "get_available_clickhouse_host",
    "get_client",
    "get_connection_mode",
]


def _emit_import_warning() -> None:
    """Emit warning at import time if ClickHouse not ready."""
    try:
        state = detect_clickhouse_state()
        if state.level < InstallationLevel.RUNNING_NO_SCHEMA:
            warnings.warn(
                f"ClickHouse cache not available: {state.message}. "
                f"Cached functions will fail. {state.action_required or ''}",
                UserWarning,
                stacklevel=3,
            )
    except Exception:
        pass  # Don't fail import on preflight errors


# Optional: emit warning at import time (can be disabled via env var)
if not os.getenv("RANGEBAR_SKIP_IMPORT_CHECK"):
    _emit_import_warning()
