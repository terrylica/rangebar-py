#!/usr/bin/env python3
"""Issue #111: Validate Ariadne zero-gap invariant on ClickHouse bars.

Queries all bars for a (symbol, threshold) pair and verifies that
consecutive bars have contiguous agg_trade_id ranges:

    bars[i].first_agg_trade_id == bars[i-1].last_agg_trade_id + 1

Usage:
    python scripts/validate_ariadne.py --symbol BTCUSDT --threshold 250
    python scripts/validate_ariadne.py --symbol BTCUSDT --threshold 250 --json
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #111: Validate Ariadne zero-gap invariant",
    )
    parser.add_argument("--symbol", required=True, help="Trading symbol")
    parser.add_argument("--threshold", type=int, required=True, help="Threshold (dbps)")
    parser.add_argument("--json", action="store_true", help="Output NDJSON report")
    parser.add_argument("--limit", type=int, default=0, help="Limit bars to check (0=all)")
    args = parser.parse_args()

    from rangebar.clickhouse import RangeBarCache

    with RangeBarCache() as cache:
        limit_clause = f"LIMIT {args.limit}" if args.limit > 0 else ""
        query = f"""
            SELECT
                close_time_ms,
                first_agg_trade_id,
                last_agg_trade_id
            FROM rangebar_cache.range_bars FINAL
            WHERE symbol = {{symbol:String}}
              AND threshold_decimal_bps = {{threshold:UInt32}}
              AND first_agg_trade_id > 0
              AND last_agg_trade_id > 0
            ORDER BY close_time_ms ASC
            {limit_clause}
        """
        result = cache.client.query(
            query,
            parameters={"symbol": args.symbol, "threshold": args.threshold},
        )

    rows = result.result_rows
    if not rows:
        print(f"No bars with trade IDs found for {args.symbol}@{args.threshold}")
        return 1

    gaps = []
    total = len(rows)

    for i in range(1, total):
        prev_last = rows[i - 1][2]  # last_agg_trade_id
        curr_first = rows[i][1]     # first_agg_trade_id
        expected = prev_last + 1

        if curr_first != expected:
            gap = {
                "index": i,
                "close_time_ms": rows[i][0],
                "prev_last_agg_trade_id": prev_last,
                "curr_first_agg_trade_id": curr_first,
                "expected": expected,
                "gap": curr_first - expected,
            }
            gaps.append(gap)

            if args.json:
                print(json.dumps({"gap": True, **gap}))

    if args.json:
        summary = {
            "symbol": args.symbol,
            "threshold": args.threshold,
            "total_bars": total,
            "gaps_found": len(gaps),
            "zero_gap": len(gaps) == 0,
        }
        print(json.dumps(summary))
    else:
        print(f"{args.symbol}@{args.threshold}: {total} bars checked")
        if gaps:
            print(f"  FAIL: {len(gaps)} gap(s) found")
            for g in gaps[:10]:
                print(
                    f"    Bar {g['index']}: expected first_id={g['expected']}, "
                    f"got {g['curr_first_agg_trade_id']} (gap={g['gap']})"
                )
            if len(gaps) > 10:
                print(f"    ... and {len(gaps) - 10} more")
            return 1
        print("  PASS: Zero-gap invariant holds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
