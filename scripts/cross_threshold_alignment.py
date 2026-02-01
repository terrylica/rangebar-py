#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Cross-Threshold Signal Alignment Analysis via ClickHouse.

Tests if range bars at different thresholds (50, 100, 200 dbps) agreeing on
direction provides ODD robust predictive signals.

Hypothesis: When bars at multiple granularities all close in the same direction,
this alignment may be a stronger signal than single-threshold direction.

Methodology:
1. Join bars from 50, 100, 200 dbps by timestamp proximity
2. Classify alignment: all_up, all_down, mixed
3. Compute forward returns for aligned states
4. Test ODD robustness (same sign + |t| >= 3.0 across all quarters)

Issue #52, #56: Post-Audit Multi-Factor Research
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from math import sqrt


def log_info(message: str, **kwargs: object) -> None:
    """Log info message in NDJSON format."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry))


def run_clickhouse_query(query: str) -> list[dict]:
    """Run ClickHouse query and return results as list of dicts."""
    cmd = [
        "clickhouse-client",
        "--query",
        query,
        "--format",
        "JSONEachRow",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    rows = []
    for line in result.stdout.strip().split("\n"):
        if line:
            rows.append(json.loads(line))
    return rows


def analyze_cross_threshold_alignment(
    symbol: str,
    min_samples: int = 100,
    min_t_stat: float = 3.0,
) -> list[dict]:
    """Analyze cross-threshold alignment for a symbol.

    Strategy: Bucket bars by hour, compute majority direction for each threshold
    in that hour, then check alignment across thresholds.

    This avoids complex JOINs and works with ClickHouse limitations.
    """
    # Step 1: Bucket by hour, get direction counts per threshold
    query = f"""
    WITH
        -- Hour-bucketed bars with direction
        hourly_bars AS (
            SELECT
                toStartOfHour(toDateTime(timestamp_ms / 1000)) AS hour_ts,
                threshold_decimal_bps AS thresh,
                CASE WHEN close > open THEN 1 ELSE 0 END AS is_up,
                close
            FROM rangebar_cache.range_bars
            WHERE symbol = '{symbol}'
              AND threshold_decimal_bps IN (50, 100, 200)
              AND ouroboros_mode = 'year'
        ),
        -- Aggregate direction per hour per threshold
        hourly_agg AS (
            SELECT
                hour_ts,
                thresh,
                sum(is_up) AS up_count,
                count() - sum(is_up) AS down_count,
                argMax(close, hour_ts) AS last_close
            FROM hourly_bars
            GROUP BY hour_ts, thresh
        ),
        -- Pivot to get all thresholds in one row per hour
        pivoted AS (
            SELECT
                hour_ts,
                sumIf(up_count, thresh = 50) AS up_50,
                sumIf(down_count, thresh = 50) AS down_50,
                sumIf(up_count, thresh = 100) AS up_100,
                sumIf(down_count, thresh = 100) AS down_100,
                sumIf(up_count, thresh = 200) AS up_200,
                sumIf(down_count, thresh = 200) AS down_200,
                argMaxIf(last_close, thresh = 100, thresh = 100) AS close_100
            FROM hourly_agg
            GROUP BY hour_ts
            HAVING up_50 + down_50 > 0 AND up_100 + down_100 > 0 AND up_200 + down_200 > 0
        ),
        -- Classify alignment based on majority direction
        classified AS (
            SELECT
                hour_ts,
                close_100,
                leadInFrame(close_100, 1) OVER (ORDER BY hour_ts ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING) AS next_close,
                CASE WHEN up_50 >= down_50 THEN 'U' ELSE 'D' END AS dir_50,
                CASE WHEN up_100 >= down_100 THEN 'U' ELSE 'D' END AS dir_100,
                CASE WHEN up_200 >= down_200 THEN 'U' ELSE 'D' END AS dir_200
            FROM pivoted
        ),
        -- Add alignment classification
        with_alignment AS (
            SELECT
                hour_ts,
                close_100,
                next_close,
                dir_50,
                dir_100,
                dir_200,
                multiIf(
                    dir_50 = 'U' AND dir_100 = 'U' AND dir_200 = 'U', 'all_up',
                    dir_50 = 'D' AND dir_100 = 'D' AND dir_200 = 'D', 'all_down',
                    dir_50 = dir_100 AND dir_100 != dir_200, 'aligned_50_100',
                    dir_100 = dir_200 AND dir_50 != dir_100, 'aligned_100_200',
                    'mixed'
                ) AS alignment
            FROM classified
            WHERE next_close IS NOT NULL AND next_close > 0
        )
    SELECT
        alignment,
        concat(
            toString(toYear(hour_ts)),
            '-Q',
            toString(toQuarter(hour_ts))
        ) AS period,
        count() AS n,
        avg((next_close / close_100 - 1)) * 10000 AS mean_bps,
        stddevSamp((next_close / close_100 - 1)) AS std_ret
    FROM with_alignment
    GROUP BY alignment, period
    HAVING n >= {min_samples}
    ORDER BY alignment, period
    """

    try:
        rows = run_clickhouse_query(query)
    except subprocess.CalledProcessError as e:
        log_info("Query failed", symbol=symbol, error=str(e.stderr)[:500])
        return []

    if not rows:
        log_info("No results", symbol=symbol)
        return []

    # Group by alignment and compute t-stats
    alignment_data: dict[str, list[dict]] = {}
    for row in rows:
        alignment = row["alignment"]
        if alignment not in alignment_data:
            alignment_data[alignment] = []

        n = row["n"]
        mean_bps = row["mean_bps"]
        std_ret = row["std_ret"]

        if std_ret and std_ret > 0 and n > 1:
            se = std_ret / sqrt(n)
            t_stat = (mean_bps / 10000) / se
        else:
            t_stat = 0

        alignment_data[alignment].append({
            "period": row["period"],
            "n": n,
            "mean_bps": mean_bps,
            "t_stat": t_stat,
        })

    results = []
    for alignment, periods in alignment_data.items():
        if len(periods) < 4:
            continue

        t_stats = [p["t_stat"] for p in periods]
        signs = [1 if t > 0 else -1 for t in t_stats]

        all_same_sign = len(set(signs)) == 1
        all_significant = all(abs(t) >= min_t_stat for t in t_stats)
        is_odd_robust = all_same_sign and all_significant

        mean_t = sum(t_stats) / len(t_stats)
        mean_bps_val = sum(p["mean_bps"] for p in periods) / len(periods)

        results.append({
            "alignment": alignment,
            "n_periods": len(periods),
            "total_n": sum(p["n"] for p in periods),
            "mean_t_stat": round(mean_t, 2),
            "min_abs_t": round(min(abs(t) for t in t_stats), 2),
            "same_sign": all_same_sign,
            "is_odd_robust": is_odd_robust,
            "direction": "+" if mean_t > 0 else "-",
            "mean_bps": round(mean_bps_val, 2),
        })

    return results


def main() -> None:
    """Run cross-threshold alignment analysis."""
    log_info("Starting cross-threshold alignment analysis")

    # Check ClickHouse connection
    try:
        run_clickhouse_query("SELECT 1")
        log_info("ClickHouse connected")
    except FileNotFoundError:
        log_info("ERROR: clickhouse-client not found. Run on bigblack.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        log_info("ERROR: ClickHouse query failed", error=str(e))
        sys.exit(1)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    all_results = []

    for symbol in symbols:
        log_info("Processing symbol", symbol=symbol)

        results = analyze_cross_threshold_alignment(symbol)

        for r in results:
            r["symbol"] = symbol
            all_results.append(r)

        symbol_odd = sum(
            1 for r in all_results
            if r["symbol"] == symbol and r["is_odd_robust"]
        )
        log_info("Symbol complete", symbol=symbol, odd_robust=symbol_odd)

    # Print results
    print("\n" + "=" * 130)
    print("CROSS-THRESHOLD ALIGNMENT ANALYSIS (50/100/200 dbps)")
    print("=" * 130)

    print("\n" + "-" * 130)
    print("RESULTS (|t| >= 3.0)")
    print("-" * 130)
    print(
        f"{'Alignment':<20} {'Symbol':<10} {'Periods':>8} {'Total N':>12} "
        f"{'Mean t':>10} {'Min |t|':>10} {'Dir':>4} {'Mean bps':>10} {'ODD':>5}"
    )
    print("-" * 130)

    for r in sorted(
        all_results,
        key=lambda x: (x["alignment"], x["symbol"]),
    ):
        print(
            f"{r['alignment']:<20} {r['symbol']:<10} "
            f"{r['n_periods']:>8} {r['total_n']:>12} {r['mean_t_stat']:>10.2f} "
            f"{r['min_abs_t']:>10.2f} {r['direction']:>4} "
            f"{r['mean_bps']:>10.2f} {'YES' if r['is_odd_robust'] else 'no':>5}"
        )

    # Summary
    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)

    total = len(all_results)
    odd_robust = sum(1 for r in all_results if r["is_odd_robust"])
    same_sign = sum(1 for r in all_results if r["same_sign"])

    print(f"\nTotal alignment-symbol combinations: {total}")
    print(f"Same sign across periods: {same_sign} ({100*same_sign/max(total, 1):.1f}%)")
    print(f"ODD robust (|t| >= 3.0): {odd_robust} ({100*odd_robust/max(total, 1):.1f}%)")

    # Cross-symbol patterns
    if odd_robust > 0:
        print("\n" + "-" * 130)
        print("CROSS-SYMBOL PATTERNS")
        print("-" * 130)

        pattern_by_symbol: dict[str, list[str]] = {}
        for r in all_results:
            if r["is_odd_robust"]:
                key = r["alignment"]
                if key not in pattern_by_symbol:
                    pattern_by_symbol[key] = []
                pattern_by_symbol[key].append(r["symbol"])

        universal = [
            alignment for alignment, syms in pattern_by_symbol.items() if len(syms) >= 4
        ]

        print(f"\nUniversal patterns (4 symbols): {len(universal)}")
        for alignment in universal:
            print(f"  - {alignment}")

    log_info(
        "Analysis complete",
        total=total,
        same_sign=same_sign,
        odd_robust=odd_robust,
    )


if __name__ == "__main__":
    main()
