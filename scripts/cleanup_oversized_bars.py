#!/usr/bin/env python3
"""Issue #112: Identify and clean up oversized bars in ClickHouse.

Scans range_bars for bars violating the threshold invariant:
  (high - low) / open > multiplier * threshold_decimal_bps / 100_000

Usage:
    python scripts/cleanup_oversized_bars.py --dry-run
    python scripts/cleanup_oversized_bars.py --symbol BTCUSDT --threshold 250
    python scripts/cleanup_oversized_bars.py --delete
"""

from __future__ import annotations

import argparse
import sys

import clickhouse_connect


def find_oversized_bars(
    client: clickhouse_connect.driver.Client,
    symbol: str | None = None,
    threshold_decimal_bps: int | None = None,
    multiplier: int = 3,
) -> list[dict]:
    """Find bars violating the range invariant."""
    where_clauses = []
    params: dict = {"multiplier": multiplier}

    if symbol:
        where_clauses.append("symbol = {symbol:String}")
        params["symbol"] = symbol
    if threshold_decimal_bps:
        where_clauses.append("threshold_decimal_bps = {threshold:UInt32}")
        params["threshold"] = threshold_decimal_bps

    where_sql = f"AND {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
    SELECT
        symbol,
        threshold_decimal_bps,
        timestamp_ms,
        open,
        high,
        low,
        close,
        (high - low) / open AS range_ratio,
        threshold_decimal_bps / 100000.0 AS threshold_ratio,
        (high - low) / open / (threshold_decimal_bps / 100000.0) AS range_multiplier
    FROM rangebar_cache.range_bars FINAL
    WHERE (high - low) / open > {{multiplier:UInt32}} * threshold_decimal_bps / 100000.0
    {where_sql}
    ORDER BY range_multiplier DESC
    LIMIT 1000
    """

    result = client.query(query, parameters=params)
    columns = result.column_names
    return [dict(zip(columns, row, strict=False)) for row in result.result_rows]


def delete_oversized_bars(
    client: clickhouse_connect.driver.Client,
    symbol: str | None = None,
    threshold_decimal_bps: int | None = None,
    multiplier: int = 3,
) -> int:
    """Delete bars violating the range invariant. Returns count deleted."""
    where_clauses = [
        "(high - low) / open > {multiplier:UInt32} * threshold_decimal_bps / 100000.0"
    ]
    params: dict = {"multiplier": multiplier}

    if symbol:
        where_clauses.append("symbol = {symbol:String}")
        params["symbol"] = symbol
    if threshold_decimal_bps:
        where_clauses.append("threshold_decimal_bps = {threshold:UInt32}")
        params["threshold"] = threshold_decimal_bps

    where_sql = " AND ".join(where_clauses)

    # Count first
    count_query = f"""
    SELECT count() FROM rangebar_cache.range_bars FINAL
    WHERE {where_sql}
    """
    count_result = client.query(count_query, parameters=params)
    count = count_result.result_rows[0][0]

    if count > 0:
        delete_query = f"""
        ALTER TABLE rangebar_cache.range_bars
        DELETE WHERE {where_sql}
        """
        client.command(delete_query, parameters=params)

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #112: Clean up oversized bars")
    parser.add_argument("--symbol", type=str, help="Filter by symbol")
    parser.add_argument("--threshold", type=int, help="Filter by threshold_decimal_bps")
    parser.add_argument("--multiplier", type=int, default=3, help="Range multiplier (default: 3)")
    parser.add_argument("--delete", action="store_true", help="Actually delete (default: dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Just report, don't delete")
    parser.add_argument("--host", type=str, default="localhost", help="ClickHouse host")
    parser.add_argument("--port", type=int, default=18123, help="ClickHouse HTTP port")
    args = parser.parse_args()

    client = clickhouse_connect.get_client(host=args.host, port=args.port)

    print(f"Scanning for oversized bars (>{args.multiplier}x threshold)...")
    bars = find_oversized_bars(client, args.symbol, args.threshold, args.multiplier)

    if not bars:
        print("No oversized bars found.")
        sys.exit(0)

    print(f"\nFound {len(bars)} oversized bar(s):")
    print("-" * 100)
    for bar in bars[:20]:
        print(
            f"  {bar['symbol']} @ {bar['threshold_decimal_bps']}dbps | "
            f"ts={bar['timestamp_ms']} | "
            f"O={bar['open']:.2f} H={bar['high']:.2f} L={bar['low']:.2f} | "
            f"range={bar['range_ratio']:.6f} ({bar['range_multiplier']:.1f}x threshold)"
        )
    if len(bars) > 20:
        print(f"  ... and {len(bars) - 20} more")

    if args.delete and not args.dry_run:
        print(f"\nDeleting {len(bars)} oversized bars...")
        deleted = delete_oversized_bars(client, args.symbol, args.threshold, args.multiplier)
        print(f"Deleted {deleted} bars.")
        print("Run 'OPTIMIZE TABLE rangebar_cache.range_bars FINAL' to reclaim space.")
    else:
        print("\nDry run. Use --delete to actually remove these bars.")


if __name__ == "__main__":
    main()
