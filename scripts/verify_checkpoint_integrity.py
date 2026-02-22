#!/usr/bin/env python3
"""Cross-validate checkpoint files against ClickHouse actual bar counts.

Reads all local checkpoint files and queries ClickHouse to verify that
bars_written in the checkpoint matches the actual number of bars stored.

Reports discrepancies where checkpoint claims more bars than ClickHouse has.

Usage:
    # Run from rangebar-py project root (requires ClickHouse connection)
    python scripts/verify_checkpoint_integrity.py

    # Check specific checkpoint directory
    python scripts/verify_checkpoint_integrity.py --checkpoint-dir /path/to/checkpoints

    # Verbose output with per-day breakdown
    python scripts/verify_checkpoint_integrity.py --per-day
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = Path(user_cache_dir("rangebar", "terrylica")) / "checkpoints"


def load_all_checkpoints(checkpoint_dir: Path) -> list[dict]:
    """Load all checkpoint JSON files from the given directory.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing checkpoint JSON files.

    Returns
    -------
    list[dict]
        List of parsed checkpoint dicts with their file paths.
    """
    if not checkpoint_dir.exists():
        logger.warning("Checkpoint directory does not exist: %s", checkpoint_dir)
        return []

    checkpoints = []
    for path in sorted(checkpoint_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            data["_file_path"] = str(path)
            # Skip streaming checkpoints (no start_date/end_date) — Issue #102
            if path.name.startswith("streaming_"):
                logger.debug("Skipping streaming checkpoint: %s", path.name)
                continue
            checkpoints.append(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load %s: %s", path, e)

    return checkpoints


def query_clickhouse_bar_count(
    client,
    symbol: str,
    threshold_decimal_bps: int,
    start_date: str,
    end_date: str,
) -> int:
    """Query ClickHouse for actual bar count in a date range.

    Parameters
    ----------
    client : clickhouse_connect.driver.Client
        ClickHouse client.
    symbol : str
        Trading symbol (e.g., "BTCUSDT").
    threshold_decimal_bps : int
        Threshold in decimal basis points.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).

    Returns
    -------
    int
        Actual bar count in ClickHouse.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    # Convert to milliseconds (start of start_date to end of end_date)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int((end_dt + timedelta(days=1)).timestamp() * 1000) - 1

    query = """
        SELECT count()
        FROM rangebar_cache.range_bars FINAL
        WHERE symbol = {symbol:String}
          AND threshold_decimal_bps = {threshold:UInt32}
          AND timestamp_ms >= {start_ms:Int64}
          AND timestamp_ms <= {end_ms:Int64}
    """
    result = client.command(
        query,
        parameters={
            "symbol": symbol,
            "threshold": threshold_decimal_bps,
            "start_ms": start_ms,
            "end_ms": end_ms,
        },
        settings={"do_not_merge_across_partitions_select_final": 1},
    )
    return int(result) if result else 0


def query_clickhouse_per_day(
    client,
    symbol: str,
    threshold_decimal_bps: int,
    start_date: str,
    end_date: str,
) -> dict[str, int]:
    """Query ClickHouse for bar count per day in a date range.

    Parameters
    ----------
    client : clickhouse_connect.driver.Client
        ClickHouse client.
    symbol : str
        Trading symbol.
    threshold_decimal_bps : int
        Threshold in decimal basis points.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).

    Returns
    -------
    dict[str, int]
        Mapping of date string (YYYY-MM-DD) to bar count for that day.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int((end_dt + timedelta(days=1)).timestamp() * 1000) - 1

    query = """
        SELECT
            toDate(toDateTime(timestamp_ms / 1000)) AS bar_date,
            count() AS bar_count
        FROM rangebar_cache.range_bars FINAL
        WHERE symbol = {symbol:String}
          AND threshold_decimal_bps = {threshold:UInt32}
          AND timestamp_ms >= {start_ms:Int64}
          AND timestamp_ms <= {end_ms:Int64}
        GROUP BY bar_date
        ORDER BY bar_date
    """
    result = client.query(
        query,
        parameters={
            "symbol": symbol,
            "threshold": threshold_decimal_bps,
            "start_ms": start_ms,
            "end_ms": end_ms,
        },
        settings={"do_not_merge_across_partitions_select_final": 1},
    )

    per_day = {}
    for row in result.result_rows:
        date_val = row[0]
        count_val = row[1]
        # date_val may be a datetime.date or string depending on driver
        if hasattr(date_val, "strftime"):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)
        per_day[date_str] = int(count_val)

    return per_day


def find_gap_dates(
    per_day_counts: dict[str, int],
    start_date: str,
    last_completed_date: str,
) -> list[str]:
    """Find dates that should have bars (within checkpoint range) but have zero.

    Parameters
    ----------
    per_day_counts : dict[str, int]
        Mapping of date -> bar count from ClickHouse.
    start_date : str
        Start of the checkpoint range.
    last_completed_date : str
        Last date the checkpoint claims was completed.

    Returns
    -------
    list[str]
        Dates with zero bars within the expected range.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(last_completed_date, "%Y-%m-%d").replace(tzinfo=UTC)

    gap_dates = []
    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        if per_day_counts.get(date_str, 0) == 0:
            gap_dates.append(date_str)
        current += timedelta(days=1)

    return gap_dates


def verify_checkpoints(
    checkpoint_dir: Path,
    per_day: bool = False,
    min_discrepancy_pct: float = 0.0,
) -> list[dict]:
    """Cross-validate all checkpoints against ClickHouse.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing checkpoint JSON files.
    per_day : bool
        If True, also check per-day coverage and report gap dates.
    min_discrepancy_pct : float
        Minimum missing percentage to count as a discrepancy. Deltas below
        this threshold are logged as DEDUP (expected from OPTIMIZE FINAL)
        but excluded from the failure list. Default 0.0 (all deltas fail).

    Returns
    -------
    list[dict]
        List of discrepancy records.
    """
    from rangebar.clickhouse import RangeBarCache

    checkpoints = load_all_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []

    print(f"Found {len(checkpoints)} checkpoint(s) in {checkpoint_dir}\n")

    discrepancies = []

    with RangeBarCache() as cache:
        client = cache.client

        for cp in checkpoints:
            symbol = cp.get("symbol", "???")
            threshold = cp.get("threshold_bps", 0)
            start_date = cp.get("start_date", "")
            end_date = cp.get("end_date", "")
            last_completed = cp.get("last_completed_date", "")
            claimed_bars = cp.get("bars_written", 0)
            file_path = cp.get("_file_path", "")

            print(f"--- {symbol} @ {threshold} dbps ({start_date} to {end_date}) ---")
            print(f"  Checkpoint file: {file_path}")
            print(f"  last_completed_date: {last_completed}")
            print(f"  bars_written (claimed): {claimed_bars}")

            # Query ClickHouse for actual count
            actual_bars = query_clickhouse_bar_count(
                client, symbol, threshold, start_date, last_completed,
            )
            print(f"  bars in ClickHouse (actual): {actual_bars}")

            delta = claimed_bars - actual_bars
            pct = (delta / claimed_bars * 100) if claimed_bars > 0 else 0

            if delta > 0 and pct < min_discrepancy_pct:
                # Issue #106: Small deltas from OPTIMIZE FINAL dedup are expected
                print(f"  DEDUP: {delta} fewer bars ({pct:.1f}%) — below {min_discrepancy_pct}% threshold")
            elif delta > 0:
                print(f"  DISCREPANCY: checkpoint claims {delta} more bars ({pct:.1f}%)")
                record = {
                    "symbol": symbol,
                    "threshold_decimal_bps": threshold,
                    "start_date": start_date,
                    "end_date": end_date,
                    "last_completed_date": last_completed,
                    "claimed_bars": claimed_bars,
                    "actual_bars": actual_bars,
                    "missing_bars": delta,
                    "missing_pct": round(pct, 1),
                    "file_path": file_path,
                }

                if per_day:
                    per_day_counts = query_clickhouse_per_day(
                        client, symbol, threshold, start_date, last_completed,
                    )
                    gap_dates = find_gap_dates(
                        per_day_counts, start_date, last_completed,
                    )
                    record["gap_dates"] = gap_dates
                    record["gap_date_count"] = len(gap_dates)

                    if gap_dates:
                        print(f"  Gap dates ({len(gap_dates)} days with 0 bars):")
                        # Show first 10 and last 5
                        show_dates = gap_dates[:10]
                        if len(gap_dates) > 15:
                            show_dates.append("...")
                            show_dates.extend(gap_dates[-5:])
                        elif len(gap_dates) > 10:
                            show_dates.extend(gap_dates[10:])
                        for d in show_dates:
                            print(f"    {d}")

                discrepancies.append(record)
            elif delta == 0:
                print("  OK: counts match")
            else:
                print(f"  NOTE: ClickHouse has {-delta} MORE bars than checkpoint claims")

            print()

    return discrepancies


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-validate checkpoint files against ClickHouse bar counts.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--per-day",
        action="store_true",
        help="Check per-day coverage and report gap dates",
    )
    parser.add_argument(
        "--min-discrepancy-pct",
        type=float,
        default=0.0,
        help="Minimum missing %% to count as failure (default: 0). "
        "Deltas below this are logged as DEDUP (expected from OPTIMIZE FINAL).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    discrepancies = verify_checkpoints(
        args.checkpoint_dir,
        per_day=args.per_day,
        min_discrepancy_pct=args.min_discrepancy_pct,
    )

    if discrepancies:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(discrepancies)} checkpoint(s) with discrepancies")
        print(f"{'='*60}")
        for d in discrepancies:
            print(
                f"  {d['symbol']} @ {d['threshold_decimal_bps']} dbps: "
                f"claimed {d['claimed_bars']}, actual {d['actual_bars']}, "
                f"missing {d['missing_bars']} ({d['missing_pct']}%)"
            )
        return 1
    print("\nAll checkpoints verified OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
