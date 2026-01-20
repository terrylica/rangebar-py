"""Exception hierarchy for rangebar-py.

This module defines a structured exception hierarchy for consistent error
handling across the rangebar library. All rangebar-specific exceptions
inherit from RangeBarError.

Exception Hierarchy
-------------------
RangeBarError (base)
├── CacheError (base for cache operations)
│   ├── CacheConnectionError (connection failures)
│   ├── CacheReadError (read operation failures)
│   ├── CacheWriteError (write operation failures)
│   └── CacheSchemaError (schema mismatches)
├── ValidationError (data validation failures)
└── ProcessingError (range bar computation failures)

Usage
-----
>>> from rangebar.exceptions import CacheReadError, CacheWriteError
>>>
>>> try:
...     bars = cache.get_range_bars(key)
... except CacheReadError as e:
...     logger.error(f"Cache read failed: {e}")
...     # Fall back to computation
"""

from __future__ import annotations


class RangeBarError(Exception):
    """Base exception for all rangebar-py errors.

    All exceptions raised by the rangebar library inherit from this class,
    making it easy to catch any rangebar-related error.

    Examples
    --------
    >>> try:
    ...     df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31")
    ... except RangeBarError as e:
    ...     print(f"Rangebar error: {e}")
    """


class CacheError(RangeBarError):
    """Base exception for cache-related errors.

    All ClickHouse cache operations that fail raise a subclass of this
    exception. Use this to catch any cache-related error.

    Attributes
    ----------
    symbol : str | None
        Trading symbol involved in the operation, if applicable.
    operation : str | None
        The cache operation that failed (e.g., "read", "write").
    """

    def __init__(
        self,
        message: str,
        *,
        symbol: str | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.symbol = symbol
        self.operation = operation


class CacheConnectionError(CacheError):
    """Raised when ClickHouse connection fails.

    This exception is raised when:
    - ClickHouse server is unreachable
    - SSH tunnel cannot be established
    - Authentication fails

    Examples
    --------
    >>> try:
    ...     cache = RangeBarCache()
    ... except CacheConnectionError as e:
    ...     print(f"Cannot connect to ClickHouse: {e}")
    """


class CacheReadError(CacheError):
    """Raised when reading from cache fails.

    This exception is raised when a cache read operation fails after
    a successful connection. Possible causes include:
    - Query syntax errors
    - Schema mismatches
    - Query timeout

    Examples
    --------
    >>> try:
    ...     bars = cache.get_range_bars(key)
    ... except CacheReadError as e:
    ...     logger.warning(f"Cache miss: {e}")
    ...     # Fall back to computation
    """


class CacheWriteError(CacheError):
    """Raised when writing to cache fails.

    This exception is raised when a cache write operation fails.
    Possible causes include:
    - Disk space exhaustion
    - Network interruption during write
    - Schema mismatch on insert

    Note: Cache write failures are typically non-fatal. The computation
    result is still valid; it just won't be cached for next time.

    Examples
    --------
    >>> try:
    ...     cache.store_range_bars(key, bars)
    ... except CacheWriteError as e:
    ...     logger.warning(f"Cache write failed (non-fatal): {e}")
    ...     # Continue - computation succeeded
    """


class CacheSchemaError(CacheError):
    """Raised when there's a schema mismatch between code and database.

    This exception indicates the ClickHouse table schema doesn't match
    what the code expects. This typically happens when:
    - New columns added to code but not migrated in DB
    - Database migrated but code not updated
    - Different rangebar versions writing to same cache

    Resolution: Run schema migration or clear cache for affected symbol.

    Examples
    --------
    >>> try:
    ...     cache.store_range_bars(key, bars)
    ... except CacheSchemaError as e:
    ...     print(f"Schema mismatch: {e}")
    ...     print("Run: ALTER TABLE rangebar_cache.range_bars ADD COLUMN ...")
    """


class ValidationError(RangeBarError):
    """Raised when data validation fails.

    This exception is raised when:
    - Post-storage validation detects data corruption
    - Microstructure feature values are out of expected bounds
    - Input data fails sanity checks

    Attributes
    ----------
    validation_tier : int | None
        The validation tier that failed (1 or 2).
    failed_checks : dict | None
        Dictionary of failed validation checks.
    """

    def __init__(
        self,
        message: str,
        *,
        validation_tier: int | None = None,
        failed_checks: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.validation_tier = validation_tier
        self.failed_checks = failed_checks


class ProcessingError(RangeBarError):
    """Raised when range bar computation fails.

    This exception is raised when the Rust backend fails to process
    trades into range bars. Possible causes include:
    - Invalid trade data (missing fields, wrong types)
    - Trades not sorted chronologically
    - Internal processing error

    Examples
    --------
    >>> try:
    ...     bars = processor.process_trades(trades)
    ... except ProcessingError as e:
    ...     print(f"Processing failed: {e}")
    """


__all__ = [
    "CacheConnectionError",
    "CacheError",
    "CacheReadError",
    "CacheSchemaError",
    "CacheWriteError",
    "ProcessingError",
    "RangeBarError",
    "ValidationError",
]
