#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Coarse-to-Fine Threshold Cascade Analysis via ClickHouse.

Tests if patterns in coarser bars (200 dbps) predict returns in finer bars (100 dbps).

Hypothesis: When a 200 dbps bar completes in a direction (U/D), subsequent 100 dbps
bars may show momentum continuation or reversal patterns.

Key difference from cross-threshold alignment:
- Uses BAR COMPLETION as signal (not hourly aggregation)
- Cross-threshold cascade (not same-period alignment)
- Avoids argMax bug by using bar timestamps directly

Methodology:
1. For each 200 dbps bar completion, get its direction (U/D)
2. Find the next 5 bars at 100 dbps threshold (within 1 hour of 200 dbps close)
3. Compute cumulative return over those 5 bars
4. Group by 200 dbps direction and test ODD robustness

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


def analyze_coarse_to_fine_cascade(
    symbol: str,
    coarse_threshold: int = 200,
    fine_threshold: int = 100,
    n_fine_bars: int = 5,
    min_samples: int = 100,
    min_t_stat: float = 3.0,
) -> list[dict]:
    """Analyze coarse-to-fine cascade patterns.

    For each 200 dbps bar completion:
    1. Get its direction (U/D)
    2. Find next N bars at 100 dbps
    3. Compute cumulative return
    4. Test ODD robustness
    """
    # Strategy: Use ASOF JOIN to match coarse bar completions to fine bars
    # that occur AFTER the coarse bar closes
    query = f"""
    WITH
        -- Coarse bars with direction
        coarse_bars AS (
            SELECT
                timestamp_ms AS coarse_close_ms,
                close AS coarse_close,
                CASE WHEN close > open THEN 'U' ELSE 'D' END AS coarse_direction,
                toYear(toDateTime(timestamp_ms / 1000)) AS year,
                toQuarter(toDateTime(timestamp_ms / 1000)) AS quarter
            FROM rangebar_cache.range_bars
            WHERE symbol = '{symbol}'
              AND threshold_decimal_bps = {coarse_threshold}
              AND ouroboros_mode = 'year'
        ),
        -- Fine bars with close price
        fine_bars AS (
            SELECT
                timestamp_ms AS fine_close_ms,
                close AS fine_close,
                open AS fine_open,
                row_number() OVER (ORDER BY timestamp_ms) AS bar_idx
            FROM rangebar_cache.range_bars
            WHERE symbol = '{symbol}'
              AND threshold_decimal_bps = {fine_threshold}
              AND ouroboros_mode = 'year'
        ),
        -- For each coarse bar, find the first fine bar that closes AFTER it
        -- and compute return over next N fine bars
        matched AS (
            SELECT
                c.coarse_close_ms,
                c.coarse_direction,
                c.year,
                c.quarter,
                -- Find first fine bar after coarse bar closes
                (
                    SELECT min(bar_idx)
                    FROM fine_bars f
                    WHERE f.fine_close_ms > c.coarse_close_ms
                ) AS first_fine_idx
            FROM coarse_bars c
        ),
        -- Get the close prices of the first and Nth fine bar after each coarse bar
        with_returns AS (
            SELECT
                m.coarse_direction,
                m.year,
                m.quarter,
                f_first.fine_close AS first_fine_close,
                f_nth.fine_close AS nth_fine_close
            FROM matched m
            INNER JOIN fine_bars f_first ON f_first.bar_idx = m.first_fine_idx
            INNER JOIN fine_bars f_nth ON f_nth.bar_idx = m.first_fine_idx + {n_fine_bars} - 1
            WHERE m.first_fine_idx IS NOT NULL
        )
    SELECT
        coarse_direction,
        concat(toString(year), '-Q', toString(quarter)) AS period,
        count() AS n,
        avg((nth_fine_close / first_fine_close - 1)) * 10000 AS mean_bps,
        stddevSamp((nth_fine_close / first_fine_close - 1)) AS std_ret
    FROM with_returns
    GROUP BY coarse_direction, period
    HAVING n >= {min_samples}
    ORDER BY coarse_direction, period
    """

    try:
        rows = run_clickhouse_query(query)
    except subprocess.CalledProcessError as e:
        log_info("Query failed", symbol=symbol, error=str(e.stderr)[:500])
        return []

    if not rows:
        log_info("No results", symbol=symbol)
        return []

    # Group by direction and compute t-stats
    direction_data: dict[str, list[dict]] = {}
    for row in rows:
        direction = row["coarse_direction"]
        if direction not in direction_data:
            direction_data[direction] = []

        n = row["n"]
        mean_bps = row["mean_bps"]
        std_ret = row["std_ret"]

        if std_ret and std_ret > 0 and n > 1:
            se = std_ret / sqrt(n)
            t_stat = (mean_bps / 10000) / se
        else:
            t_stat = 0

        direction_data[direction].append({
            "period": row["period"],
            "n": n,
            "mean_bps": mean_bps,
            "t_stat": t_stat,
        })

    results = []
    for direction, periods in direction_data.items():
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
            "direction": direction,
            "n_periods": len(periods),
            "total_n": sum(p["n"] for p in periods),
            "mean_t_stat": round(mean_t, 2),
            "min_abs_t": round(min(abs(t) for t in t_stats), 2),
            "same_sign": all_same_sign,
            "is_odd_robust": is_odd_robust,
            "signal": "+" if mean_t > 0 else "-",
            "mean_bps": round(mean_bps_val, 2),
        })

    return results


def main() -> None:
    """Run coarse-to-fine cascade analysis."""
    log_info("Starting coarse-to-fine cascade analysis")

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

        results = analyze_coarse_to_fine_cascade(symbol)

        for r in results:
            r["symbol"] = symbol
            all_results.append(r)

        symbol_odd = sum(
            1 for r in all_results
            if r["symbol"] == symbol and r["is_odd_robust"]
        )
        log_info("Symbol complete", symbol=symbol, odd_robust=symbol_odd)

    # Print results
    print("\n" + "=" * 120)
    print("COARSE-TO-FINE CASCADE ANALYSIS (200 dbps → 100 dbps)")
    print("=" * 120)

    print("\n" + "-" * 120)
    print("RESULTS (|t| >= 3.0)")
    print("-" * 120)
    print(
        f"{'200 dbps Dir':<15} {'Symbol':<10} {'Periods':>8} {'Total N':>12} "
        f"{'Mean t':>10} {'Min t':>10} {'Signal':>6} {'Mean bps':>10} {'ODD':>5}"
    )
    print("-" * 120)

    for r in sorted(
        all_results,
        key=lambda x: (x["direction"], x["symbol"]),
    ):
        print(
            f"{r['direction']:<15} {r['symbol']:<10} "
            f"{r['n_periods']:>8} {r['total_n']:>12} {r['mean_t_stat']:>10.2f} "
            f"{r['min_abs_t']:>10.2f} {r['signal']:>6} "
            f"{r['mean_bps']:>10.2f} {'YES' if r['is_odd_robust'] else 'no':>5}"
        )

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    total = len(all_results)
    odd_robust = sum(1 for r in all_results if r["is_odd_robust"])
    same_sign = sum(1 for r in all_results if r["same_sign"])

    print(f"\nTotal direction-symbol combinations: {total}")
    print(f"Same sign across periods: {same_sign} ({100*same_sign/max(total, 1):.1f}%)")
    print(f"ODD robust (|t| >= 3.0): {odd_robust} ({100*odd_robust/max(total, 1):.1f}%)")

    # Cross-symbol patterns
    if odd_robust > 0:
        print("\n" + "-" * 120)
        print("CROSS-SYMBOL PATTERNS")
        print("-" * 120)

        pattern_by_symbol: dict[str, list[str]] = {}
        for r in all_results:
            if r["is_odd_robust"]:
                key = r["direction"]
                if key not in pattern_by_symbol:
                    pattern_by_symbol[key] = []
                pattern_by_symbol[key].append(r["symbol"])

        universal = [
            direction for direction, syms in pattern_by_symbol.items() if len(syms) >= 4
        ]

        print(f"\nUniversal patterns (4 symbols): {len(universal)}")
        for direction in universal:
            print(f"  - 200 dbps {direction} → 100 dbps cascade")

    log_info(
        "Analysis complete",
        total=total,
        same_sign=same_sign,
        odd_robust=odd_robust,
    )


if __name__ == "__main__":
    main()
