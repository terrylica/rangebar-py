"""Local Parquet storage for tick data (Tier 1 cache).

This module provides cross-platform tick data storage using Parquet files
with ZSTD compression (level 3) for optimal balance of size and speed.

Issue #73: Added atomic write and corruption validation helpers.

Storage Location (via platformdirs):
- macOS:   ~/Library/Caches/rangebar/ticks/
- Linux:   ~/.cache/rangebar/ticks/ (respects XDG_CACHE_HOME)
- Windows: %USERPROFILE%\\AppData\\Local\\terrylica\\rangebar\\Cache\\ticks\\

Examples
--------
>>> from rangebar.storage import TickStorage
>>>
>>> storage = TickStorage()
>>> storage.write_ticks("BTCUSDT", trades_df)
>>> df = storage.read_ticks("BTCUSDT", start_ts, end_ts)
"""

from .parquet import (
    TickStorage,
    _atomic_write_parquet,
    _is_valid_parquet,
    _validate_and_recover_parquet,
    get_cache_dir,
)

__all__ = [
    "TickStorage",
    "_atomic_write_parquet",
    "_is_valid_parquet",
    "_validate_and_recover_parquet",
    "get_cache_dir",
]
