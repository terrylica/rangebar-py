"""Telegram notification integration for rangebar-py.

This module provides Telegram notifications for hook events, enabling
"loud" alerts for failures and successes during cache operations.

Configuration
-------------
Set environment variables (via Doppler or .env):

    RANGEBAR_TELEGRAM_TOKEN - Bot token from @BotFather
    RANGEBAR_TELEGRAM_CHAT_ID - Your chat ID (send /start to bot to get it)

Or use Doppler:
    doppler secrets set RANGEBAR_TELEGRAM_TOKEN --project rangebar --config prd
    doppler secrets set RANGEBAR_TELEGRAM_CHAT_ID --project rangebar --config prd

Usage
-----
>>> from rangebar.notify.telegram import enable_telegram_notifications
>>> enable_telegram_notifications()
>>> # All hook events will now be sent to Telegram

Or selective registration for failures only:
>>> from rangebar.notify.telegram import telegram_notify
>>> from rangebar.hooks import register_for_failures
>>> register_for_failures(telegram_notify)
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..hooks import HookPayload

logger = logging.getLogger(__name__)

# Default chat ID (Terry Li @EonLabsOperations) - can be overridden via env
# SSoT-OK: This is a Telegram chat ID, not a version number
_DEFAULT_CHAT_ID = "90417581"


@lru_cache(maxsize=1)
def get_telegram_config() -> dict[str, str | None]:
    """Load Telegram configuration from environment.

    Returns
    -------
    dict
        Configuration with 'token' and 'chat_id' keys.
        Values may be None if not configured.

    Notes
    -----
    Configuration sources (in priority order):
    1. Environment variables (RANGEBAR_TELEGRAM_TOKEN, RANGEBAR_TELEGRAM_CHAT_ID)
    2. Default chat ID (for internal use)
    """
    return {
        "token": os.environ.get("RANGEBAR_TELEGRAM_TOKEN"),
        "chat_id": os.environ.get("RANGEBAR_TELEGRAM_CHAT_ID", _DEFAULT_CHAT_ID),
    }


def is_configured() -> bool:
    """Check if Telegram notifications are configured.

    Returns
    -------
    bool
        True if both token and chat_id are available.
    """
    config = get_telegram_config()
    return bool(config.get("token") and config.get("chat_id"))


def send_telegram(
    message: str,
    *,
    parse_mode: str = "HTML",
    disable_notification: bool = False,
) -> bool:
    """Send a message via Telegram bot.

    Parameters
    ----------
    message : str
        Message text (supports HTML formatting).
    parse_mode : str
        Telegram parse mode ("HTML" or "Markdown").
    disable_notification : bool
        If True, send silently without notification sound.

    Returns
    -------
    bool
        True if message was sent successfully, False otherwise.

    Examples
    --------
    >>> send_telegram("<b>Alert:</b> Cache write failed for BTCUSDT")
    True
    """
    config = get_telegram_config()
    token = config.get("token")
    chat_id = config.get("chat_id")

    if not token:
        logger.warning("Telegram not configured: RANGEBAR_TELEGRAM_TOKEN not set")
        return False

    if not chat_id:
        logger.warning("Telegram not configured: RANGEBAR_TELEGRAM_CHAT_ID not set")
        return False

    # Import requests only when needed (optional dependency)
    try:
        import requests
    except ImportError:
        logger.warning("Telegram notifications require 'requests' package")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_notification": disable_notification,
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.debug("Telegram message sent successfully")
        return True
    except requests.exceptions.Timeout:
        logger.warning("Telegram notification timed out")
        return False
    except requests.exceptions.RequestException as e:
        logger.warning("Telegram notification failed: %s", e)
        return False


def telegram_notify(payload: HookPayload) -> None:
    """Hook callback that sends notifications to Telegram.

    This function is designed to be registered as a hook callback.
    It formats the payload into a human-readable message and sends
    it to Telegram.

    Parameters
    ----------
    payload : HookPayload
        Event payload from the hooks system.

    Examples
    --------
    >>> from rangebar.hooks import register_hook, HookEvent
    >>> register_hook(HookEvent.CACHE_WRITE_FAILED, telegram_notify)
    """
    emoji = "\u274c" if payload.is_failure else "\u2705"  # ❌ or ✅
    status = "FAILED" if payload.is_failure else "SUCCESS"

    # Format details as readable text
    details_text = ""
    if payload.details:
        details_lines = []
        for key, value in payload.details.items():
            # Handle nested dicts/lists
            if isinstance(value, dict | list):
                display_value = json.dumps(value, indent=2)
            else:
                display_value = value
            details_lines.append(f"  {key}: {display_value}")
        details_text = "\n".join(details_lines)

    message = f"""{emoji} <b>rangebar-py {status}</b>

<b>Event:</b> {payload.event.value}
<b>Symbol:</b> {payload.symbol}
<b>Time:</b> {payload.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}"""

    if details_text:
        message += f"""

<b>Details:</b>
<pre>{details_text}</pre>"""

    # Send silently for success, loudly for failures
    send_telegram(
        message,
        disable_notification=not payload.is_failure,
    )


def enable_telegram_notifications() -> bool:
    """Enable Telegram notifications for all hook events.

    Registers telegram_notify as a callback for all HookEvent types.

    Returns
    -------
    bool
        True if Telegram is configured and hooks were registered,
        False if Telegram is not configured.

    Examples
    --------
    >>> if enable_telegram_notifications():
    ...     print("Telegram notifications enabled")
    ... else:
    ...     print("Telegram not configured - set RANGEBAR_TELEGRAM_TOKEN")
    """
    if not is_configured():
        logger.warning(
            "Telegram not configured. Set RANGEBAR_TELEGRAM_TOKEN environment variable."
        )
        return False

    from ..hooks import HookEvent, register_hook

    for event in HookEvent:
        register_hook(event, telegram_notify)

    logger.info("Telegram notifications enabled for all hook events")
    return True


def enable_failure_notifications() -> bool:
    """Enable Telegram notifications for failure events only.

    Registers telegram_notify as a callback for all *_FAILED events.
    This is useful when you only want to be alerted about problems.

    Returns
    -------
    bool
        True if Telegram is configured and hooks were registered,
        False if Telegram is not configured.

    Examples
    --------
    >>> enable_failure_notifications()
    True
    """
    if not is_configured():
        logger.warning(
            "Telegram not configured. Set RANGEBAR_TELEGRAM_TOKEN environment variable."
        )
        return False

    from ..hooks import register_for_failures

    register_for_failures(telegram_notify)

    logger.info("Telegram notifications enabled for failure events")
    return True


__all__ = [
    "enable_failure_notifications",
    "enable_telegram_notifications",
    "get_telegram_config",
    "is_configured",
    "send_telegram",
    "telegram_notify",
]
