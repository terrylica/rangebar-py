"""Pushover critical alerts for rangebar-py.

Implements GitHub Issue #43: Loud alerting for checksum verification failures.

Pushover alerts are used for CRITICAL data integrity issues that require
immediate attention, such as:
- SHA-256 checksum mismatches (data corruption detected)
- Tier 1 cache integrity failures
"""

from __future__ import annotations

import sys

import requests

# Pushover API configuration
# These credentials are for the "RB Checksum Fail" app
PUSHOVER_APP_TOKEN = "asxuepwiaqkwc5e749xj1qx2eg1e3b"
PUSHOVER_USER_KEY = "ury88s1def6v16seeueoefqn1zbua1"
PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"


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
    payload = {
        "token": PUSHOVER_APP_TOKEN,
        "user": PUSHOVER_USER_KEY,
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

    # Use lower priority (1) for audit warnings vs checksum failures (2)
    payload = {
        "token": PUSHOVER_APP_TOKEN,
        "user": PUSHOVER_USER_KEY,
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
