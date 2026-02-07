"""SQL migrations for ClickHouse range bar cache.

Issue #78: Exchange session column population via SQL UPDATE.

This module provides idempotent SQL migrations that populate derived columns
from existing data, avoiding full reprocessing from raw ticks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import clickhouse_connect

logger = logging.getLogger(__name__)

# ============================================================================
# Exchange Session Definitions
# ============================================================================
# These match python/rangebar/ouroboros.py:EXCHANGE_SESSION_HOURS exactly.
#
# ClickHouse's toTimezone() handles DST automatically when given IANA tz names,
# matching Python's zoneinfo behavior used in get_active_exchange_sessions().
#
# Session hours are in LOCAL time:
#   Sydney (ASX):     10:00-16:00 Australia/Sydney
#   Tokyo (TSE):      09:00-15:00 Asia/Tokyo
#   London (LSE):     08:00-17:00 Europe/London
#   New York (NYSE):  10:00-16:00 America/New_York
#
# Weekend exclusion: toDayOfWeek() returns 6=Saturday, 7=Sunday

_SESSION_UPDATES: list[dict[str, str]] = [
    {
        "column": "exchange_session_sydney",
        "tz": "Australia/Sydney",
        "start": "10",
        "end": "16",
    },
    {
        "column": "exchange_session_tokyo",
        "tz": "Asia/Tokyo",
        "start": "9",
        "end": "15",
    },
    {
        "column": "exchange_session_london",
        "tz": "Europe/London",
        "start": "8",
        "end": "17",
    },
    {
        "column": "exchange_session_newyork",
        "tz": "America/New_York",
        "start": "10",
        "end": "16",
    },
]


def _build_session_update_sql(
    session: dict[str, str],
    *,
    symbol: str | None = None,
) -> str:
    """Build ALTER TABLE UPDATE SQL for one exchange session column.

    Parameters
    ----------
    session : dict
        Session definition with column, tz, start, end keys.
    symbol : str or None
        If provided, restrict update to this symbol only.

    Returns
    -------
    str
        ClickHouse ALTER TABLE UPDATE statement.
    """
    col = session["column"]
    tz = session["tz"]
    start = session["start"]
    end = session["end"]

    # toTimezone converts to local time with DST; toHour extracts hour.
    # toDayOfWeek mode=0: 1=Monday..7=Sunday. Weekdays are 1-5.
    ts_local = (
        f"toTimezone(toDateTime(intDiv(timestamp_ms, 1000)), '{tz}')"
    )
    condition = (
        f"toHour({ts_local}) >= {start} "
        f"AND toHour({ts_local}) < {end} "
        f"AND toDayOfWeek({ts_local}) <= 5"
    )

    where = "1 = 1"
    if symbol:
        where = f"symbol = '{symbol}'"

    return (
        f"ALTER TABLE rangebar_cache.range_bars "
        f"UPDATE {col} = if({condition}, 1, 0) "
        f"WHERE {where}"
    )


def migrate_exchange_sessions(
    client: clickhouse_connect.driver.Client,
    *,
    symbol: str | None = None,
    dry_run: bool = False,
) -> int:
    """Populate exchange_session_* columns from timestamp_ms.

    Computes session flags using timezone-aware hour extraction matching
    the Python implementation in ouroboros.py. Handles DST automatically
    via ClickHouse's toTimezone().

    Idempotent — safe to run multiple times. Each run overwrites all session
    columns with freshly computed values.

    Parameters
    ----------
    client : clickhouse_connect.driver.Client
        Active ClickHouse client connection.
    symbol : str or None
        If provided, only update rows for this symbol.
        If None, update all rows.
    dry_run : bool
        If True, log SQL statements without executing them.

    Returns
    -------
    int
        Number of ALTER TABLE UPDATE statements executed (always 4).

    Raises
    ------
    clickhouse_connect.driver.exceptions.DatabaseError
        If ClickHouse query fails.

    Examples
    --------
    >>> from rangebar.clickhouse import RangeBarCache
    >>> with RangeBarCache() as cache:
    ...     n = migrate_exchange_sessions(cache.client, symbol="SOLUSDT")
    ...     print(f"Executed {n} UPDATE statements")
    """
    from ..hooks import HookEvent, emit_hook

    executed = 0
    target = symbol or "*"

    for session in _SESSION_UPDATES:
        sql = _build_session_update_sql(session, symbol=symbol)

        if dry_run:
            logger.info("[DRY RUN] %s", sql)
        else:
            logger.info("Executing: %s", sql)
            client.command(sql)
            emit_hook(
                HookEvent.MIGRATION_PROGRESS,
                symbol=target,
                column=session["column"],
                step=executed + 1,
                total=len(_SESSION_UPDATES),
            )

        executed += 1

    scope = f"symbol={symbol}" if symbol else "all symbols"
    logger.info(
        "Exchange session migration complete: %d statements (%s)", executed, scope
    )

    return executed


def check_exchange_session_coverage(
    client: clickhouse_connect.driver.Client,
    *,
    symbol: str | None = None,
) -> dict[str, dict[str, int]]:
    """Check how many rows have non-zero exchange session flags.

    Useful for verifying migration ran successfully.

    Parameters
    ----------
    client : clickhouse_connect.driver.Client
        Active ClickHouse client connection.
    symbol : str or None
        If provided, only check this symbol.

    Returns
    -------
    dict[str, dict[str, int]]
        Per-session counts: {"exchange_session_sydney": {"active": N, "total": M}, ...}
    """
    where = f"WHERE symbol = '{symbol}'" if symbol else ""

    results = {}
    for col in (
        "exchange_session_sydney",
        "exchange_session_tokyo",
        "exchange_session_london",
        "exchange_session_newyork",
    ):
        row = client.query(
            f"SELECT countIf({col} = 1) AS active, count() AS total "
            f"FROM rangebar_cache.range_bars {where}"
        ).first_row

        results[col] = {"active": row[0], "total": row[1]}

    return results
