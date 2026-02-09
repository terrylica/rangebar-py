"""Binance Vision end-date availability probe (Issue #88).

Binance Vision publishes daily aggTrade archives with a variable lag (typically
1-2 days, occasionally longer). Population jobs that assume "yesterday" is always
available will crash at 99.96% completion when the final day's archive hasn't been
published yet (see: LINKUSDT@750 failure).

This module probes Binance Vision via HTTP HEAD requests to find the most recent
available archive date, walking backwards from yesterday until a 200 response is
received.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

__all__ = [
    "BINANCE_VISION_AGGTRADES_URL",
    "probe_latest_available_date",
]

logger = logging.getLogger(__name__)

BINANCE_VISION_AGGTRADES_URL: str = (
    "https://data.binance.vision/data/futures/um/daily/aggTrades"
    "/{symbol}/{symbol}-aggTrades-{date}.zip"
)
"""URL pattern for Binance Vision aggTrade archives.

Placeholders:
    {symbol} - Trading symbol (e.g., "BTCUSDT")
    {date}   - Date in YYYY-MM-DD format
"""

_DEFAULT_PROBE_SYMBOL = "BTCUSDT"
_DEFAULT_MAX_LOOKBACK = 10
_PROBE_TIMEOUT_SECONDS = 10


def probe_latest_available_date(
    symbol: str = _DEFAULT_PROBE_SYMBOL,
    max_lookback: int = _DEFAULT_MAX_LOOKBACK,
    *,
    timeout: int = _PROBE_TIMEOUT_SECONDS,
) -> str:
    """Probe Binance Vision to find the latest available aggTrade archive date.

    Sends HTTP HEAD requests starting from yesterday, walking backwards
    until a 200 response confirms the archive exists.

    Parameters
    ----------
    symbol : str, default="BTCUSDT"
        Trading symbol to probe. BTCUSDT is recommended as it is always
        published first due to highest liquidity.
    max_lookback : int, default=10
        Maximum number of days to walk back from yesterday.
    timeout : int, default=10
        HTTP request timeout in seconds.

    Returns
    -------
    str
        The latest available date in YYYY-MM-DD format.

    Examples
    --------
    >>> from rangebar.binance_vision import probe_latest_available_date
    >>> end_date = probe_latest_available_date()
    >>> print(end_date)  # e.g., "2026-02-07"

    Use as end_date for population jobs:

    >>> from rangebar import populate_cache_resumable
    >>> end_date = probe_latest_available_date("BTCUSDT")
    >>> populate_cache_resumable("BTCUSDT", "2024-01-01", end_date)
    """
    today = datetime.now(tz=UTC).date()

    for days_ago in range(1, max_lookback + 1):
        probe_date = today - timedelta(days=days_ago)
        date_str = probe_date.isoformat()
        url = BINANCE_VISION_AGGTRADES_URL.format(symbol=symbol, date=date_str)

        try:
            req = Request(url, method="HEAD")
            with urlopen(req, timeout=timeout) as resp:
                if resp.status == HTTPStatus.OK:
                    logger.info(
                        "Binance Vision probe: %s available at %s",
                        symbol,
                        date_str,
                    )
                    return date_str
        except (HTTPError, URLError, TimeoutError, OSError):
            logger.debug(
                "Binance Vision probe: %s not available at %s",
                symbol,
                date_str,
            )
            continue

    # Fallback: return max_lookback days ago
    fallback_date = today - timedelta(days=max_lookback)
    fallback_str = fallback_date.isoformat()
    logger.warning(
        "Binance Vision probe: no archive found for %s in last %d days, "
        "falling back to %s",
        symbol,
        max_lookback,
        fallback_str,
    )
    return fallback_str
