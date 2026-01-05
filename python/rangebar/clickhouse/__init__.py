"""ClickHouse cache layer for rangebar.

This module provides two-tier caching (raw trades + computed range bars)
using a local ClickHouse database for high-performance repeated processing.

Configuration
-------------
Set environment variables via mise (recommended) or directly:

    # mise.toml or ~/.config/mise/config.toml
    [env]
    RANGEBAR_CH_HOSTS = "host1,host2"  # SSH aliases from ~/.ssh/config
    RANGEBAR_CH_PRIMARY = "host1"       # Default host

If no env vars set, falls back to localhost:8123.

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
from .config import ClickHouseConfig
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
    "HostConnection",
    "InstallationLevel",
    "PreflightResult",
    "RangeBarCache",
    "SSHTunnel",
    "detect_clickhouse_state",
    "get_available_clickhouse_host",
    "get_client",
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
