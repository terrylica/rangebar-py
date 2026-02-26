"""Threshold validation with hierarchical SSoT configuration.

Configuration Hierarchy (highest priority first):
1. Per-symbol: RANGEBAR_MIN_THRESHOLD_<SYMBOL>
2. Asset-class: RANGEBAR_<ASSET_CLASS>_MIN_THRESHOLD
3. Fallback: hardcoded defaults (with deprecation warning)

SSoT: All values should be set via mise.toml [env] section.

Issue #62: Crypto Minimum Threshold Enforcement
Research: docs/research/2026-02-03-cfm-optimal-threshold-gemini-3-pro.md
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING

from rangebar.constants import (
    THRESHOLD_DECIMAL_MAX,
    THRESHOLD_DECIMAL_MIN,
    THRESHOLD_PRESETS,
)
from rangebar.validation.gap_classification import AssetClass, detect_asset_class

if TYPE_CHECKING:
    pass

__all__ = [
    "CryptoThresholdError",
    "ThresholdError",
    "clear_threshold_cache",
    "get_min_threshold",
    "get_min_threshold_for_symbol",
    "resolve_and_validate_threshold",
    "validate_checkpoint_threshold",
]

# =============================================================================
# Fallback Defaults (used only if env vars not set)
# =============================================================================
# WARNING: These should be set in mise.toml, not relied upon.
# Values are in decimal basis points (dbps): 1000 dbps = 1%

_FALLBACK_DEFAULTS: dict[AssetClass, int] = {
    AssetClass.CRYPTO: 250,  # 0.25% - minimum viable crypto threshold
    AssetClass.FOREX: 50,  # 0.05% - tighter spreads allow lower
    AssetClass.EQUITIES: 100,  # 0.1% - standard equities minimum
    AssetClass.UNKNOWN: 1,  # No enforcement for unknown assets
}

# =============================================================================
# Environment Variable Patterns
# =============================================================================

_ASSET_CLASS_ENV_PATTERN = "RANGEBAR_{}_MIN_THRESHOLD"
_SYMBOL_ENV_PATTERN = "RANGEBAR_MIN_THRESHOLD_{}"


# =============================================================================
# Exception Classes
# =============================================================================


class ThresholdError(ValueError):
    """Raised when threshold is below configured minimum for asset class.

    This exception indicates that the requested threshold is too low for
    the asset class, which would result in unprofitable trading due to
    transaction costs exceeding potential gains.

    Attributes
    ----------
    threshold : int
        The requested threshold in decimal basis points
    min_threshold : int
        The minimum required threshold for the asset class
    symbol : str | None
        The symbol that triggered the error
    asset_class : AssetClass
        The detected asset class

    Examples
    --------
    >>> from rangebar import get_range_bars
    >>> get_range_bars("BTCUSDT", "2024-01-01", "2024-01-02", threshold_decimal_bps=100)
    ThresholdError: Threshold 100 dbps below minimum 1000 dbps for crypto symbol
    'BTCUSDT'. Configure via: RANGEBAR_MIN_THRESHOLD_BTCUSDT or
    RANGEBAR_CRYPTO_MIN_THRESHOLD
    """

    def __init__(
        self,
        message: str,
        *,
        threshold: int | None = None,
        min_threshold: int | None = None,
        symbol: str | None = None,
        asset_class: AssetClass | None = None,
    ) -> None:
        super().__init__(message)
        self.threshold = threshold
        self.min_threshold = min_threshold
        self.symbol = symbol
        self.asset_class = asset_class


# Backward compatibility alias
CryptoThresholdError = ThresholdError


# =============================================================================
# Threshold Lookup Functions
# =============================================================================


@lru_cache(maxsize=128)
def get_min_threshold_for_symbol(symbol: str) -> int:
    """Get minimum threshold for specific symbol.

    Resolution order:
    1. Per-symbol env var: RANGEBAR_MIN_THRESHOLD_<SYMBOL>
    2. Asset-class env var: RANGEBAR_<ASSET_CLASS>_MIN_THRESHOLD
    3. Fallback default (with warning)

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT", "EURUSD")

    Returns
    -------
    int
        Minimum threshold in decimal basis points

    Examples
    --------
    >>> # With env var set: RANGEBAR_CRYPTO_MIN_THRESHOLD=1000
    >>> get_min_threshold_for_symbol("BTCUSDT")
    1000

    >>> # With per-symbol override: RANGEBAR_MIN_THRESHOLD_BTCUSDT=1500
    >>> get_min_threshold_for_symbol("BTCUSDT")
    1500
    """
    symbol_upper = symbol.upper()

    # 1. Check per-symbol env var override (highest priority — operational escape hatch)
    symbol_env_var = _SYMBOL_ENV_PATTERN.format(symbol_upper)
    symbol_value = os.environ.get(symbol_env_var)
    if symbol_value is not None:
        return int(symbol_value)

    # 2. Check symbol registry (SSoT for per-symbol min_threshold)
    try:
        from rangebar.symbol_registry import get_symbol_entries

        entries = get_symbol_entries()
        entry = entries.get(symbol_upper)
        if entry is not None and entry.min_threshold is not None:
            return entry.min_threshold
    except (ImportError, OSError):
        pass  # Registry unavailable — fall through to asset-class

    # 3. Check asset-class default
    asset_class = detect_asset_class(symbol)
    class_env_var = _ASSET_CLASS_ENV_PATTERN.format(asset_class.name)
    class_value = os.environ.get(class_env_var)
    if class_value is not None:
        return int(class_value)

    # 4. Fallback with warning (mise.toml should define these)
    fallback = _FALLBACK_DEFAULTS.get(asset_class, 1)
    warnings.warn(
        f"Using fallback threshold {fallback} dbps for {asset_class.value}. "
        f"Set {class_env_var} in mise.toml for SSoT.",
        UserWarning,
        stacklevel=2,
    )
    return fallback


def get_min_threshold(asset_class: AssetClass) -> int:
    """Get default minimum threshold for asset class.

    Parameters
    ----------
    asset_class : AssetClass
        Asset classification

    Returns
    -------
    int
        Minimum threshold from env var or fallback

    Examples
    --------
    >>> from rangebar import AssetClass
    >>> # With env var: RANGEBAR_CRYPTO_MIN_THRESHOLD=1000
    >>> get_min_threshold(AssetClass.CRYPTO)
    1000
    """
    env_var = _ASSET_CLASS_ENV_PATTERN.format(asset_class.name)
    value = os.environ.get(env_var)
    if value is not None:
        return int(value)

    fallback = _FALLBACK_DEFAULTS.get(asset_class, 1)
    warnings.warn(
        f"Using fallback threshold {fallback} dbps for {asset_class.value}. "
        f"Set {env_var} in mise.toml for SSoT.",
        UserWarning,
        stacklevel=2,
    )
    return fallback


# =============================================================================
# Validation Functions
# =============================================================================


def resolve_and_validate_threshold(
    symbol: str | None,
    threshold_decimal_bps: int | str,
) -> int:
    """Resolve preset names and validate threshold for symbol.

    This is the primary validation function used by all API entry points.
    It handles both preset name resolution and asset-class minimum validation.

    Parameters
    ----------
    symbol : str | None
        Trading symbol. If None, skip asset-class validation.
    threshold_decimal_bps : int | str
        Threshold as integer or preset name (e.g., "standard", "tight")

    Returns
    -------
    int
        Validated threshold in decimal basis points

    Raises
    ------
    ThresholdError
        If threshold is below configured minimum for symbol/asset-class
    ValueError
        If threshold is out of global range or preset unknown

    Examples
    --------
    >>> resolve_and_validate_threshold("BTCUSDT", 1000)
    1000

    >>> resolve_and_validate_threshold("BTCUSDT", "macro")  # preset = 1000 dbps
    1000

    >>> resolve_and_validate_threshold("BTCUSDT", 100)
    ThresholdError: Threshold 100 dbps below minimum 1000 dbps for crypto...
    """
    # Step 1: Resolve preset name to integer
    if isinstance(threshold_decimal_bps, str):
        if threshold_decimal_bps not in THRESHOLD_PRESETS:
            msg = (
                f"Unknown threshold preset: {threshold_decimal_bps!r}. "
                f"Valid presets: {list(THRESHOLD_PRESETS.keys())}"
            )
            raise ValueError(msg)
        threshold = THRESHOLD_PRESETS[threshold_decimal_bps]
    else:
        threshold = threshold_decimal_bps

    # Step 2: Global range validation
    if not THRESHOLD_DECIMAL_MIN <= threshold <= THRESHOLD_DECIMAL_MAX:
        msg = (
            f"threshold_decimal_bps must be between {THRESHOLD_DECIMAL_MIN} "
            f"and {THRESHOLD_DECIMAL_MAX}, got {threshold}"
        )
        raise ValueError(msg)

    # Step 3: Symbol-specific minimum validation (skip if no symbol)
    if symbol is not None:
        # Issue #96 Task #79: Symbol already uppercase from get_range_bars() entry
        # Eliminate redundant .upper() call in downstream functions
        min_threshold = get_min_threshold_for_symbol(symbol)
        asset_class = detect_asset_class(symbol)

        if threshold < min_threshold:
            # Determine which env var to reference in error message
            # symbol is already uppercase, no clone needed
            symbol_env_var = _SYMBOL_ENV_PATTERN.format(symbol)
            class_env_var = _ASSET_CLASS_ENV_PATTERN.format(asset_class.name)

            msg = (
                f"Threshold {threshold} dbps below minimum {min_threshold} dbps "
                f"for {asset_class.value} symbol '{symbol}'. "
                f"Configure via: {symbol_env_var} or {class_env_var}"
            )

            # Log NDJSON event and send Pushover alert
            _emit_threshold_violation_telemetry(
                symbol=symbol,
                threshold=threshold,
                min_threshold=min_threshold,
                asset_class=asset_class,
            )

            raise ThresholdError(
                msg,
                threshold=threshold,
                min_threshold=min_threshold,
                symbol=symbol,
                asset_class=asset_class,
            )

    return threshold


def validate_checkpoint_threshold(checkpoint: dict) -> None:
    """Validate checkpoint threshold for symbol.

    Called during checkpoint restoration to ensure the checkpoint's
    threshold meets current minimum requirements.

    Parameters
    ----------
    checkpoint : dict
        Checkpoint dictionary with 'symbol' and 'threshold_decimal_bps' keys

    Raises
    ------
    ThresholdError
        If checkpoint has threshold below minimum for its symbol

    Examples
    --------
    >>> checkpoint = {"symbol": "BTCUSDT", "threshold_decimal_bps": 100}
    >>> validate_checkpoint_threshold(checkpoint)
    ThresholdError: Threshold 100 dbps below minimum 1000 dbps for crypto...
    """
    symbol = checkpoint.get("symbol")
    threshold = checkpoint.get("threshold_decimal_bps")

    if symbol and threshold:
        resolve_and_validate_threshold(symbol, threshold)


def clear_threshold_cache() -> None:
    """Clear the LRU cache for threshold lookups.

    Call after modifying environment variables at runtime to ensure
    new values are picked up.

    Examples
    --------
    >>> import os
    >>> os.environ["RANGEBAR_CRYPTO_MIN_THRESHOLD"] = "750"
    >>> clear_threshold_cache()  # Required for new value to take effect
    """
    get_min_threshold_for_symbol.cache_clear()


# =============================================================================
# Telemetry (NDJSON logging + Pushover alerts)
# =============================================================================


def _emit_threshold_violation_telemetry(
    symbol: str,
    threshold: int,
    min_threshold: int,
    asset_class: AssetClass,
) -> None:
    """Emit telemetry for threshold violations.

    Logs NDJSON event and sends Pushover alert for CRITICAL violations.
    This function never raises - telemetry failures are suppressed because
    the primary ThresholdError exception should always be raised.
    """
    import contextlib

    # Suppress telemetry failures - the ThresholdError is the important signal
    with contextlib.suppress(ImportError, OSError, ValueError):
        _log_threshold_violation_ndjson(
            symbol=symbol,
            threshold=threshold,
            min_threshold=min_threshold,
            asset_class=asset_class,
        )

    with contextlib.suppress(ImportError, OSError, ValueError):
        _send_threshold_pushover_alert(
            symbol=symbol,
            threshold=threshold,
            min_threshold=min_threshold,
            asset_class=asset_class,
        )


def _log_threshold_violation_ndjson(
    symbol: str,
    threshold: int,
    min_threshold: int,
    asset_class: AssetClass,
) -> None:
    """Log threshold violation as NDJSON event."""
    try:
        from loguru import logger

        logger.bind(
            component="threshold_validation",
            event_type="threshold_violation",
            symbol=symbol,
            threshold_requested=threshold,
            threshold_minimum=min_threshold,
            asset_class=asset_class.value,
        ).error(
            f"Threshold violation: {threshold} < {min_threshold} dbps for {symbol}"
        )
    except ImportError:
        pass  # loguru not installed


def _send_threshold_pushover_alert(
    symbol: str,
    threshold: int,
    min_threshold: int,
    asset_class: AssetClass,
) -> None:
    """Send Pushover alert for threshold violations.

    Raises
    ------
    ImportError
        If httpx is not installed
    OSError
        If network request fails
    ValueError
        If Pushover credentials are invalid
    """
    # Only send alerts if Pushover is configured
    pushover_token = os.environ.get("PUSHOVER_API_TOKEN")
    pushover_user = os.environ.get("PUSHOVER_USER_KEY")

    if not pushover_token or not pushover_user:
        return

    import httpx

    httpx.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": pushover_token,
            "user": pushover_user,
            "title": f"Threshold Violation: {symbol}",
            "message": (
                f"Attempted threshold {threshold} dbps < minimum {min_threshold} dbps\n"
                f"Asset class: {asset_class.value}"
            ),
            "priority": 1,  # High priority
        },
        timeout=5.0,
    )
