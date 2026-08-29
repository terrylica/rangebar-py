"""Pushover critical alerts for rangebar-py.

Implements GitHub Issue #43: Loud alerting for checksum verification failures.

Pushover alerts are used for CRITICAL data integrity issues that require
immediate attention, such as:
- SHA-256 checksum mismatches (data corruption detected)
- Tier 1 cache integrity failures

Configuration
-------------
Set environment variables (via Doppler, mise ``[env]``, or .env):

    PUSHOVER_CHECKSUM_APP_TOKEN - App token for the "RB Checksum Fail" app
    PUSHOVER_USER_KEY - Pushover account user key

If either is unset, alerts are skipped (logged, never raised) -- alerting is
secondary and must never break a data pipeline.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import requests

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"


@lru_cache(maxsize=1)
def get_pushover_config() -> dict[str, str | None]:
    """Load Pushover configuration from environment.

    Returns
    -------
    dict
        Configuration with 'token' and 'user_key' keys.
        Values may be None if not configured.
    """
    return {
        "token": os.environ.get("PUSHOVER_CHECKSUM_APP_TOKEN"),
        "user_key": os.environ.get("PUSHOVER_USER_KEY"),
    }


def is_configured() -> bool:
    """Check if Pushover alerting is configured.

    Returns
    -------
    bool
        True if both app token and user key are available.
    """
    config = get_pushover_config()
    return bool(config.get("token") and config.get("user_key"))


def send_critical_alert(
    title: str,
    message: str,
    url: str | None = None,
    url_title: str | None = None,
) -> bool:
    """Send LOUD critical alert via Pushover.

    Uses:
    - Priority 2 (emergency) - requires acknowledgment
    - Dune custom sound for maximum attention
    - Retry every 60s for 1 hour until acknowledged

    Args:
        title: Alert title (e.g., "🚨 CHECKSUM FAIL: BTCUSDT")
        message: Alert body with details
        url: Optional URL for more information
        url_title: Display title for the URL

    Returns:
        True if alert was sent successfully, False otherwise
    """
    config = get_pushover_config()
    if not is_configured():
        _log_alert_failure(
            "Pushover not configured: set PUSHOVER_CHECKSUM_APP_TOKEN "
            "and PUSHOVER_USER_KEY"
        )
        return False

    payload = {
        "token": config["token"],
        "user": config["user_key"],
        "title": title,
        "message": message,
        "priority": 2,  # Emergency - requires acknowledgment
        "retry": 60,  # Retry every 60 seconds
        "expire": 3600,  # Stop retrying after 1 hour
        "sound": "dune",  # Dune custom sound for maximum attention
    }

    if url:
        payload["url"] = url
        payload["url_title"] = url_title or "View Details"

    try:
        response = requests.post(PUSHOVER_API_URL, data=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        # Log failure but don't crash - alerting is secondary
        _log_alert_failure(str(e))
        return False


def _log_alert_failure(error: str) -> None:
    """Log Pushover alert failure without crashing."""
    try:
        from ..logging import get_logger

        logger = get_logger()
        logger.bind(component="pushover").error(f"Pushover alert failed: {error}")
    except ImportError:
        # Fallback - print to stderr if logging module not available
        print(f"Pushover alert failed: {error}", file=sys.stderr)


def alert_checksum_failure(
    symbol: str,
    date: str,
    expected_hash: str,
    actual_hash: str,
    data_source: str = "binance",
) -> None:
    """Alert on checksum mismatch - CRITICAL data integrity issue.

    This function sends an emergency Pushover alert when a downloaded file's
    SHA-256 hash does not match the expected value from Binance.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        date: Date of the corrupted data (YYYY-MM-DD)
        expected_hash: Expected SHA-256 hash from Binance
        actual_hash: Actual computed hash of downloaded data
        data_source: Data source identifier (default: "binance")
    """
    title = f"🚨 CHECKSUM FAIL: {symbol}"
    message = f"""Data corruption detected!

Symbol: {symbol}
Date: {date}
Source: {data_source}
Expected: {expected_hash[:16]}...
Actual: {actual_hash[:16]}...

ACTION REQUIRED: Investigate immediately.
Data may be corrupted or tampered with."""

    send_critical_alert(title, message)


def alert_tier1_cache_unverified(
    symbol: str,
    date_range: str,
    unverified_count: int,
    total_count: int,
) -> None:
    """Alert when Tier 1 cache contains unverified files.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        date_range: Date range being audited (e.g., "2024-01-01 to 2024-01-07")
        unverified_count: Number of unverified dates
        total_count: Total number of dates in range
    """
    title = f"⚠️ CACHE AUDIT: {symbol}"
    message = f"""Tier 1 cache audit found unverified files.

Symbol: {symbol}
Date Range: {date_range}
Unverified: {unverified_count}/{total_count} dates

Consider re-downloading with verify_checksum=True
to ensure data integrity."""

    config = get_pushover_config()
    if not is_configured():
        _log_alert_failure(
            "Pushover not configured: set PUSHOVER_CHECKSUM_APP_TOKEN "
            "and PUSHOVER_USER_KEY"
        )
        return

    # Use lower priority (1) for audit warnings vs checksum failures (2)
    payload = {
        "token": config["token"],
        "user": config["user_key"],
        "title": title,
        "message": message,
        "priority": 1,  # High priority but not emergency
        "sound": "siren",
    }

    try:
        response = requests.post(PUSHOVER_API_URL, data=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        _log_alert_failure(str(e))
