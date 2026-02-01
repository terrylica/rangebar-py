#!/usr/bin/env python3
# allow-simple-sharpe: volatility research uses forward duration without return tracking
"""Duration Autocorrelation Analysis via ClickHouse.

Tests if range bar duration shows autocorrelation (volatility clustering).

Hypothesis: If bar N has short duration (high volatility), bar N+1 is more likely
to also have short duration. This is the simplest form of volatility clustering.

Methodology:
1. Compute duration_us for each bar
2. Classify into terciles (SHORT/MID/LONG)
3. Test if current tercile predicts next bar's tercile
4. Validate ODD robustness across quarterly periods

This is a simpler test than forward RV - just testing if duration[N] → duration[N+1].

Issue #52, #56: Post-Audit Volatility Research
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
        "ssh",
        "bigblack",
        f'clickhouse-client --query "{query}" --format JSONEachRow',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    rows = []
    for line in result.stdout.strip().split("\n"):
        if line:
            rows.append(json.loads(line))
    return rows


def analyze_duration_autocorrelation(
    symbol: str,
    threshold: int = 100,
    min_samples: int = 100,
    min_t_stat: float = 3.0,
) -> list[dict]:
    """Analyze if bar duration predicts next bar's duration.

    Strategy:
    1. Compute duration_us for each bar
    2. Classify duration into terciles (SHORT/MID/LONG)
    3. Compute next bar's duration tercile
    4. Test if current tercile predicts next tercile
    """
    query = f"""
    WITH
        -- Bars with duration
        bars_with_duration AS (
            SELECT
                timestamp_ms,
                duration_us,
                toYear(toDateTime(timestamp_ms / 1000)) AS year,
                toQuarter(toDateTime(timestamp_ms / 1000)) AS quarter,
                -- Get next bar's duration
                leadInFrame(duration_us, 1) OVER (
                    ORDER BY timestamp_ms
                    ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING
                ) AS next_duration_us
            FROM rangebar_cache.range_bars
            WHERE symbol = '{symbol}'
              AND threshold_decimal_bps = {threshold}
              AND ouroboros_mode = 'year'
        ),
        -- Compute tercile boundaries
        quantiles AS (
            SELECT
                quantile(0.33)(duration_us) AS q33,
                quantile(0.67)(duration_us) AS q67
            FROM bars_with_duration
            WHERE next_duration_us IS NOT NULL AND next_duration_us > 0
        )
    SELECT
        multiIf(
            b.duration_us <= q.q33, 'SHORT',
            b.duration_us <= q.q67, 'MID',
            'LONG'
        ) AS current_tercile,
        multiIf(
            b.next_duration_us <= q.q33, 'SHORT',
            b.next_duration_us <= q.q67, 'MID',
            'LONG'
        ) AS next_tercile,
        concat(toString(b.year), '-Q', toString(b.quarter)) AS period,
        count() AS n
    FROM bars_with_duration b
    CROSS JOIN quantiles q
    WHERE b.next_duration_us IS NOT NULL AND b.next_duration_us > 0
    GROUP BY current_tercile, next_tercile, period
    HAVING n >= {min_samples}
    ORDER BY current_tercile, next_tercile, period
    """

    try:
        rows = run_clickhouse_query(query)
    except subprocess.CalledProcessError as e:
        log_info("Query failed", symbol=symbol, error=str(e.stderr)[:500])
        return []

    if not rows:
        log_info("No results", symbol=symbol)
        return []

    # Compute P(same_tercile | current_tercile) for each period
    # Group by (current_tercile, period) to get total counts
    period_totals: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["current_tercile"], row["period"])
        period_totals[key] = period_totals.get(key, 0) + row["n"]

    # Compute P(next=current | current) for each period - persistence probability
    results_by_tercile: dict[str, list[dict]] = {}

    for row in rows:
        current = row["current_tercile"]
        next_t = row["next_tercile"]
        period = row["period"]
        n = row["n"]

        if current not in results_by_tercile:
            results_by_tercile[current] = []

        total = period_totals.get((current, period), 0)
        # Only count when next = current (persistence)
        if total > 0 and current == next_t:
            prob = n / total
            results_by_tercile[current].append({
                "period": period,
                "n": total,
                "prob_persist": prob,
            })

    # Compute ODD robustness for each tercile
    results = []
    for tercile, periods in results_by_tercile.items():
        if len(periods) < 4:
            continue

        probs = [p["prob_persist"] for p in periods]

        # Test if P(persist) > 0.33 consistently
        mean_prob = sum(probs) / len(probs)

        # Compute t-stat for deviation from null (0.33)
        null_prob = 1/3
        std_prob = sqrt(sum((p - mean_prob)**2 for p in probs) / (len(probs) - 1)) if len(probs) > 1 else 0

        if std_prob > 0:
            se = std_prob / sqrt(len(probs))
            t_stat = (mean_prob - null_prob) / se
        else:
            t_stat = 0

        # Check sign consistency
        signs = [1 if p > null_prob else -1 for p in probs]
        all_same_sign = len(set(signs)) == 1
        is_odd_robust = all_same_sign and abs(t_stat) >= min_t_stat

        results.append({
            "tercile": tercile,
            "n_periods": len(periods),
            "total_n": sum(p["n"] for p in periods),
            "mean_prob_persist": round(mean_prob, 4),
            "min_prob": round(min(probs), 4),
            "max_prob": round(max(probs), 4),
            "t_stat_vs_null": round(t_stat, 2),
            "same_sign": all_same_sign,
            "is_odd_robust": is_odd_robust,
            "direction": "+" if mean_prob > null_prob else "-",
        })

    return results


def main() -> None:
    """Run duration autocorrelation analysis."""
    log_info("Starting duration autocorrelation analysis")

    # Check ClickHouse connection via SSH
    try:
        run_clickhouse_query("SELECT 1")
        log_info("ClickHouse connected via SSH")
    except FileNotFoundError:
        log_info("ERROR: SSH failed")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        log_info("ERROR: ClickHouse query failed", error=str(e))
        sys.exit(1)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    all_results = []

    for symbol in symbols:
        log_info("Processing symbol", symbol=symbol)

        results = analyze_duration_autocorrelation(symbol)

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
    print("DURATION AUTOCORRELATION (Volatility Clustering)")
    print("Hypothesis: P(duration[N+1] = X | duration[N] = X) > 0.33 (persistence)")
    print("=" * 130)

    print("\n" + "-" * 130)
    print("P(next = current | current) - Null hypothesis: P = 0.33 (no persistence)")
    print("-" * 130)
    print(
        f"{'Tercile':<10} {'Symbol':<10} {'Periods':>8} {'Total N':>12} "
        f"{'P(persist)':>12} {'Min P':>8} {'Max P':>8} {'t-stat':>10} {'Sign':>5} {'ODD':>5}"
    )
    print("-" * 130)

    for r in sorted(
        all_results,
        key=lambda x: (x["tercile"], x["symbol"]),
    ):
        print(
            f"{r['tercile']:<10} {r['symbol']:<10} "
            f"{r['n_periods']:>8} {r['total_n']:>12} {r['mean_prob_persist']:>12.4f} "
            f"{r['min_prob']:>8.4f} {r['max_prob']:>8.4f} "
            f"{r['t_stat_vs_null']:>10.2f} {r['direction']:>5} "
            f"{'YES' if r['is_odd_robust'] else 'no':>5}"
        )

    # Summary
    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)

    total = len(all_results)
    odd_robust = sum(1 for r in all_results if r["is_odd_robust"])
    same_sign = sum(1 for r in all_results if r["same_sign"])

    print(f"\nTotal tercile-symbol combinations: {total}")
    print(f"Same sign across periods: {same_sign} ({100*same_sign/max(total, 1):.1f}%)")
    print(f"ODD robust (|t| >= 3.0, +): {odd_robust} ({100*odd_robust/max(total, 1):.1f}%)")

    # Check for universal patterns
    if odd_robust > 0:
        print("\n" + "-" * 130)
        print("VOLATILITY CLUSTERING CHECK")
        print("-" * 130)

        for tercile in ["SHORT", "MID", "LONG"]:
            tercile_results = [r for r in all_results if r["tercile"] == tercile and r["is_odd_robust"]]
            if tercile_results:
                avg_prob = sum(r["mean_prob_persist"] for r in tercile_results) / len(tercile_results)
                symbols_str = ", ".join(r["symbol"] for r in tercile_results)
                print(f"  {tercile}: {len(tercile_results)}/4 symbols ODD robust, avg P(persist) = {avg_prob:.4f}")
                print(f"    Symbols: {symbols_str}")

        # Universal = all 4 symbols ODD robust
        for tercile in ["SHORT", "MID", "LONG"]:
            universal = [r for r in all_results if r["tercile"] == tercile and r["is_odd_robust"] and r["direction"] == "+"]
            if len(universal) >= 4:
                avg_prob = sum(r["mean_prob_persist"] for r in universal) / len(universal)
                print(f"\n  ✓ UNIVERSAL PATTERN: {tercile} duration persistence is ODD robust")
                print(f"    Average P(persist) = {avg_prob:.4f} vs null 0.333")
    else:
        print("\n  ✗ No ODD robust volatility clustering detected")

    log_info(
        "Analysis complete",
        total=total,
        same_sign=same_sign,
        odd_robust=odd_robust,
    )


if __name__ == "__main__":
    main()
