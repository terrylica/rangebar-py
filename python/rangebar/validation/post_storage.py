"""Tier 0: Post-storage validation for cache integrity (<1 sec).

Run after every cache write to verify data was stored correctly.
This is the fastest validation tier - critical for detecting cache corruption.

Issue #39: Post-storage validation to verify cached data matches computed data.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of post-storage validation.

    Attributes
    ----------
    passed : bool
        True if all validation checks passed.
    checks : dict[str, bool]
        Individual check results.
    details : dict[str, Any]
        Additional details about validation (counts, timestamps, etc.).
    timestamp : datetime
        When validation was performed.
    duration_ms : float
        How long validation took in milliseconds.
    """

    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "passed": self.passed,
            "checks": self.checks,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }


def compute_dataframe_checksum(df: pd.DataFrame) -> str:
    """Compute a checksum for a DataFrame's key columns.

    Uses MD5 hash of (timestamp, OHLCV) tuples for fast comparison.
    xxHash64 would be faster but MD5 is available without deps.

    Parameters
    ----------
    df : pd.DataFrame
        Range bar DataFrame with DatetimeIndex and OHLCV columns.

    Returns
    -------
    str
        Hex digest of the checksum.
    """
    import pandas as pd

    # Use key columns for checksum
    key_cols = ["Open", "High", "Low", "Close", "Volume"]
    present_cols = [c for c in key_cols if c in df.columns]

    if not present_cols or df.empty:
        return "empty"

    # Create string representation of key data
    # Include index (timestamps) and OHLCV values
    hasher = hashlib.md5()

    # Hash index (timestamps)
    if isinstance(df.index, pd.DatetimeIndex):
        index_str = df.index.astype(int).astype(str).str.cat(sep=",")
    else:
        index_str = ",".join(str(x) for x in df.index)
    hasher.update(index_str.encode())

    # Hash each column
    for col in present_cols:
        col_str = df[col].astype(str).str.cat(sep=",")
        hasher.update(col_str.encode())

    return hasher.hexdigest()


def validate_post_storage(
    expected: pd.DataFrame,
    retrieved: pd.DataFrame | None,
    *,
    symbol: str = "",
    threshold_bps: int = 0,
) -> ValidationResult:
    """Validate that retrieved data matches expected data after cache operation.

    This is a fast (<1 sec) validation that should run after every cache write
    to verify data integrity. It checks:
    1. Row count matches
    2. First timestamp matches
    3. Last timestamp matches
    4. Checksum matches (OHLCV data)

    Parameters
    ----------
    expected : pd.DataFrame
        The DataFrame that was written to cache.
    retrieved : pd.DataFrame | None
        The DataFrame read back from cache (None if read failed).
    symbol : str, optional
        Symbol for logging context.
    threshold_bps : int, optional
        Threshold for logging context.

    Returns
    -------
    ValidationResult
        Validation result with pass/fail and details.

    Examples
    --------
    >>> from rangebar.validation.post_storage import validate_post_storage
    >>> result = validate_post_storage(computed_df, cached_df, symbol="BTCUSDT")
    >>> if not result.passed:
    ...     logger.error("Post-storage validation FAILED: %s", result.checks)
    """
    import time

    start_time = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {
        "symbol": symbol,
        "threshold_bps": threshold_bps,
    }

    # Check 1: Retrieved data exists
    if retrieved is None:
        checks["data_retrieved"] = False
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ValidationResult(
            passed=False,
            checks=checks,
            details={**details, "error": "No data retrieved from cache"},
            duration_ms=duration_ms,
        )
    checks["data_retrieved"] = True

    # Check 2: Row count matches
    expected_count = len(expected)
    retrieved_count = len(retrieved)
    checks["row_count_match"] = expected_count == retrieved_count
    details["expected_count"] = expected_count
    details["retrieved_count"] = retrieved_count

    if not checks["row_count_match"]:
        logger.warning(
            "Row count mismatch for %s: expected %d, got %d",
            symbol,
            expected_count,
            retrieved_count,
        )

    # Check 3: First timestamp matches
    if not expected.empty and not retrieved.empty:
        expected_first = expected.index[0]
        retrieved_first = retrieved.index[0]
        checks["first_timestamp_match"] = expected_first == retrieved_first
        details["expected_first_ts"] = str(expected_first)
        details["retrieved_first_ts"] = str(retrieved_first)

        # Check 4: Last timestamp matches
        expected_last = expected.index[-1]
        retrieved_last = retrieved.index[-1]
        checks["last_timestamp_match"] = expected_last == retrieved_last
        details["expected_last_ts"] = str(expected_last)
        details["retrieved_last_ts"] = str(retrieved_last)
    else:
        checks["first_timestamp_match"] = expected.empty == retrieved.empty
        checks["last_timestamp_match"] = expected.empty == retrieved.empty

    # Check 5: Checksum matches (OHLCV data integrity)
    expected_checksum = compute_dataframe_checksum(expected)
    retrieved_checksum = compute_dataframe_checksum(retrieved)
    checks["checksum_match"] = expected_checksum == retrieved_checksum
    details["expected_checksum"] = expected_checksum[:16]  # Truncate for logging
    details["retrieved_checksum"] = retrieved_checksum[:16]

    if not checks["checksum_match"]:
        logger.warning(
            "Checksum mismatch for %s: data corruption detected",
            symbol,
        )

    # Overall pass/fail
    passed = all(checks.values())
    duration_ms = (time.perf_counter() - start_time) * 1000

    result = ValidationResult(
        passed=passed,
        checks=checks,
        details=details,
        duration_ms=duration_ms,
    )

    if passed:
        logger.debug(
            "Post-storage validation PASSED for %s (%d bars, %.1fms)",
            symbol,
            expected_count,
            duration_ms,
        )
    else:
        logger.warning(
            "Post-storage validation FAILED for %s: %s",
            symbol,
            {k: v for k, v in checks.items() if not v},
        )

    return result


def validate_ohlc_invariants(df: pd.DataFrame) -> ValidationResult:
    """Validate OHLC price invariants.

    Checks that for all bars:
    - High >= max(Open, Close)
    - Low <= min(Open, Close)

    These invariants should always hold for valid OHLC data.

    Parameters
    ----------
    df : pd.DataFrame
        Range bar DataFrame with Open, High, Low, Close columns.

    Returns
    -------
    ValidationResult
        Validation result.
    """
    import time

    start_time = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {"bar_count": len(df)}

    if df.empty:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ValidationResult(
            passed=True,
            checks={"empty_dataframe": True},
            details=details,
            duration_ms=duration_ms,
        )

    # Check required columns
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ValidationResult(
            passed=False,
            checks={"columns_present": False},
            details={**details, "missing_columns": missing},
            duration_ms=duration_ms,
        )
    checks["columns_present"] = True

    # High >= max(Open, Close)
    high_valid = (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
    checks["high_ge_open_close"] = bool(high_valid)

    if not high_valid:
        invalid_rows = df[df["High"] < df[["Open", "Close"]].max(axis=1)]
        details["high_invalid_count"] = len(invalid_rows)
        details["high_invalid_first_ts"] = str(invalid_rows.index[0])

    # Low <= min(Open, Close)
    low_valid = (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()
    checks["low_le_open_close"] = bool(low_valid)

    if not low_valid:
        invalid_rows = df[df["Low"] > df[["Open", "Close"]].min(axis=1)]
        details["low_invalid_count"] = len(invalid_rows)
        details["low_invalid_first_ts"] = str(invalid_rows.index[0])

    passed = all(checks.values())
    duration_ms = (time.perf_counter() - start_time) * 1000

    return ValidationResult(
        passed=passed,
        checks=checks,
        details=details,
        duration_ms=duration_ms,
    )


__all__ = [
    "ValidationResult",
    "compute_dataframe_checksum",
    "validate_ohlc_invariants",
    "validate_post_storage",
]
