#!/usr/bin/env python3
"""Detect volume overflow (negative volumes) in ClickHouse cache. Issue #88."""
from __future__ import annotations

import sys

from rangebar.clickhouse import RangeBarCache


def main():
    with RangeBarCache() as cache:
        result = cache.client.query("""
            SELECT symbol, threshold_decimal_bps,
              countIf(volume < 0) AS neg_vol,
              countIf(buy_volume < 0) AS neg_buy,
              countIf(sell_volume < 0) AS neg_sell,
              count() AS total
            FROM rangebar_cache.range_bars
            GROUP BY symbol, threshold_decimal_bps
            HAVING neg_vol > 0 OR neg_buy > 0 OR neg_sell > 0
            ORDER BY symbol, threshold_decimal_bps
        """)

        if not result.result_rows:
            print("No volume overflow detected. All clean.")
            sys.exit(0)

        print(f"{'Symbol':<12} {'Thresh':<8} {'Neg Vol':>10} {'Neg Buy':>10} {'Neg Sell':>10} {'Total':>12} {'Corrupt%':>10}")
        print("-" * 80)
        for row in result.result_rows:
            sym, thresh, neg_vol, neg_buy, neg_sell, total = row
            pct = (neg_vol + neg_buy + neg_sell) / (total * 3) * 100
            print(f"{sym:<12} {thresh:<8} {neg_vol:>10,} {neg_buy:>10,} {neg_sell:>10,} {total:>12,} {pct:>9.3f}%")

        sys.exit(1)

if __name__ == "__main__":
    main()
