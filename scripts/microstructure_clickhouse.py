#!/usr/bin/env python3
# allow-simple-sharpe: pattern research uses forward returns without duration tracking
"""Microstructure Pattern Analysis via ClickHouse - Memory Efficient.

Queries ClickHouse directly to avoid pandas memory overhead.
Run this on bigblack where ClickHouse is running.

Issue #52, #56: Post-Audit Microstructure Research
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


def analyze_microstructure_feature(
    symbol: str,
    feature: str,
    threshold: int = 100,
    min_samples: int = 100,
    min_t_stat: float = 3.0,
) -> list[dict]:
    """Analyze predictive power of a microstructure feature using ClickHouse."""
    # ClickHouse-compatible query:
    # 1. First compute forward returns using window function in subquery
    # 2. Then compute quantiles and assign terciles
    # 3. Finally aggregate by tercile and period
    query = f"""
    WITH
        -- Step 1: Add forward returns using window function
        bars_with_fwd AS (
            SELECT
                {feature},
                close,
                close_time_ms,
                leadInFrame(close, 1) OVER (
                    ORDER BY close_time_ms
                    ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING
                ) AS next_close
            FROM rangebar_cache.range_bars
            WHERE symbol = '{symbol}'
              AND threshold_decimal_bps = {threshold}
              AND ouroboros_mode = 'year'
              AND {feature} IS NOT NULL
        ),
        -- Step 2: Compute quantiles
        quantile_bounds AS (
            SELECT
                quantile(0.33)({feature}) AS q33,
                quantile(0.67)({feature}) AS q67
            FROM bars_with_fwd
        )
    SELECT
        multiIf(
            b.{feature} <= qb.q33, 'LOW',
            b.{feature} <= qb.q67, 'MID',
            'HIGH'
        ) AS tercile,
        concat(
            toString(toYear(toDateTime(b.close_time_ms / 1000))),
            '-Q',
            toString(toQuarter(toDateTime(b.close_time_ms / 1000)))
        ) AS period,
        count() AS n,
        avg((b.next_close / b.close - 1)) * 10000 AS mean_bps,
        stddevSamp((b.next_close / b.close - 1)) AS std_ret
    FROM bars_with_fwd b
    CROSS JOIN quantile_bounds qb
    WHERE b.next_close IS NOT NULL
      AND b.next_close > 0
    GROUP BY tercile, period
    HAVING n >= {min_samples}
    ORDER BY tercile, period
    """

    try:
        rows = run_clickhouse_query(query)
    except subprocess.CalledProcessError as e:
        log_info("Query failed", symbol=symbol, feature=feature, error=str(e))
        return []

    # Group by tercile and compute t-stats
    tercile_data = {}
    for row in rows:
        tercile = row["tercile"]
        if tercile not in tercile_data:
            tercile_data[tercile] = []

        n = row["n"]
        mean_bps = row["mean_bps"]
        std_ret = row["std_ret"]

        if std_ret > 0 and n > 1:
            se = std_ret / sqrt(n)
            t_stat = (mean_bps / 10000) / se
        else:
            t_stat = 0

        tercile_data[tercile].append({
            "period": row["period"],
            "n": n,
            "mean_bps": mean_bps,
            "t_stat": t_stat,
        })

    results = []
    for tercile, periods in tercile_data.items():
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
            "feature": feature,
            "tercile": tercile,
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
    """Run microstructure analysis via ClickHouse."""
    log_info("Starting ClickHouse microstructure analysis")

    # Check if we can connect to ClickHouse
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
    features = [
        "ofi",
        "vwap_close_deviation",
        "trade_intensity",
        "aggression_ratio",
        "turnover_imbalance",
    ]

    all_results = []

    for symbol in symbols:
        log_info("Processing symbol", symbol=symbol)

        for feat in features:
            results = analyze_microstructure_feature(symbol, feat)

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
    print("MICROSTRUCTURE PATTERN ANALYSIS (ClickHouse)")
    print("=" * 120)

    print("\n" + "-" * 120)
    print("RESULTS (|t| >= 3.0)")
    print("-" * 120)
    print(
        f"{'Feature':<25} {'Tercile':<8} {'Symbol':<10} {'Periods':>8} "
        f"{'Mean t':>10} {'Min |t|':>10} {'Dir':>4} {'Mean bps':>10} {'ODD':>5}"
    )
    print("-" * 120)

    for r in sorted(
        all_results,
        key=lambda x: (x["feature"], x["tercile"], x["symbol"]),
    ):
        print(
            f"{r['feature']:<25} {r['tercile']:<8} {r['symbol']:<10} "
            f"{r['n_periods']:>8} {r['mean_t_stat']:>10.2f} "
            f"{r['min_abs_t']:>10.2f} {r['direction']:>4} "
            f"{r['mean_bps']:>10.2f} {'YES' if r['is_odd_robust'] else 'no':>5}"
        )

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    total = len(all_results)
    odd_robust = sum(1 for r in all_results if r["is_odd_robust"])
    same_sign = sum(1 for r in all_results if r["same_sign"])

    print(f"\nTotal feature-tercile combinations: {total}")
    print(f"Same sign across periods: {same_sign} ({100*same_sign/max(total, 1):.1f}%)")
    print(f"ODD robust (|t| >= 3.0): {odd_robust} ({100*odd_robust/max(total, 1):.1f}%)")

    log_info(
        "Analysis complete",
        total=total,
        same_sign=same_sign,
        odd_robust=odd_robust,
    )


if __name__ == "__main__":
    main()
