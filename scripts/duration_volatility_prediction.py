#!/usr/bin/env python3
# allow-simple-sharpe: volatility research uses forward volatility without duration tracking
"""Duration-Based Volatility Prediction via ClickHouse.

Tests if range bar duration predicts future realized volatility.

Hypothesis: Short-duration bars indicate high market activity/volatility.
If bar N has short duration, subsequent bars may also have short duration
(volatility clustering). This is NOT directional prediction - it's volatility
forecasting.

Methodology:
1. Compute duration_us for each bar (close_time - open_time)
2. Classify into terciles (SHORT/MID/LONG)
3. Compute forward realized volatility (std of next N bar returns)
4. Test if duration tercile predicts RV tercile
5. Validate ODD robustness across quarterly periods

Use cases:
- Position sizing (reduce size in volatile regimes)
- Stop placement (wider stops in volatile periods)
- Hedging decisions (increase hedges when volatility expected)

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


def analyze_duration_volatility_prediction(
    symbol: str,
    threshold: int = 100,
    forward_window: int = 10,
    min_samples: int = 100,
    min_t_stat: float = 3.0,
) -> list[dict]:
    """Analyze if bar duration predicts future realized volatility.

    Strategy:
    1. Compute duration_us for each bar
    2. Classify duration into terciles (SHORT/MID/LONG)
    3. Compute forward realized volatility (std of next N returns)
    4. Classify forward RV into terciles
    5. Test if duration tercile predicts RV tercile
    """
    # Query to compute duration and forward volatility
    query = f"""
    WITH
        -- Bars with duration
        bars_with_duration AS (
            SELECT
                timestamp_ms,
                duration_us,
                close,
                toYear(toDateTime(timestamp_ms / 1000)) AS year,
                toQuarter(toDateTime(timestamp_ms / 1000)) AS quarter,
                -- Compute log return
                log(close / lagInFrame(close, 1) OVER (ORDER BY timestamp_ms)) AS log_ret
            FROM rangebar_cache.range_bars
            WHERE symbol = '{symbol}'
              AND threshold_decimal_bps = {threshold}
              AND ouroboros_mode = 'year'
        ),
        -- Compute rolling forward volatility (std of next N returns)
        with_forward_vol AS (
            SELECT
                timestamp_ms,
                duration_us,
                year,
                quarter,
                -- Forward RV: std of next {forward_window} returns
                stddevPopStable(log_ret) OVER (
                    ORDER BY timestamp_ms
                    ROWS BETWEEN 1 FOLLOWING AND {forward_window} FOLLOWING
                ) AS forward_rv
            FROM bars_with_duration
            WHERE log_ret IS NOT NULL
        ),
        -- Compute tercile boundaries
        duration_quantiles AS (
            SELECT
                quantile(0.33)(duration_us) AS dur_q33,
                quantile(0.67)(duration_us) AS dur_q67
            FROM with_forward_vol
            WHERE forward_rv IS NOT NULL AND forward_rv > 0
        ),
        rv_quantiles AS (
            SELECT
                quantile(0.33)(forward_rv) AS rv_q33,
                quantile(0.67)(forward_rv) AS rv_q67
            FROM with_forward_vol
            WHERE forward_rv IS NOT NULL AND forward_rv > 0
        )
    SELECT
        multiIf(
            b.duration_us <= dq.dur_q33, 'SHORT',
            b.duration_us <= dq.dur_q67, 'MID',
            'LONG'
        ) AS duration_tercile,
        multiIf(
            b.forward_rv <= rq.rv_q33, 'LOW_RV',
            b.forward_rv <= rq.rv_q67, 'MID_RV',
            'HIGH_RV'
        ) AS forward_rv_tercile,
        concat(toString(b.year), '-Q', toString(b.quarter)) AS period,
        count() AS n
    FROM with_forward_vol b
    CROSS JOIN duration_quantiles dq
    CROSS JOIN rv_quantiles rq
    WHERE b.forward_rv IS NOT NULL AND b.forward_rv > 0
    GROUP BY duration_tercile, forward_rv_tercile, period
    HAVING n >= {min_samples}
    ORDER BY duration_tercile, forward_rv_tercile, period
    """

    try:
        rows = run_clickhouse_query(query)
    except subprocess.CalledProcessError as e:
        log_info("Query failed", symbol=symbol, error=str(e.stderr)[:500])
        return []

    if not rows:
        log_info("No results", symbol=symbol)
        return []

    # Compute conditional probabilities: P(HIGH_RV | duration_tercile)
    # Group by (duration_tercile, period) to get total counts
    period_totals: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["duration_tercile"], row["period"])
        period_totals[key] = period_totals.get(key, 0) + row["n"]

    # Compute P(HIGH_RV | duration_tercile) for each period
    results_by_duration: dict[str, list[dict]] = {}

    for row in rows:
        dur = row["duration_tercile"]
        rv = row["forward_rv_tercile"]
        period = row["period"]
        n = row["n"]

        if dur not in results_by_duration:
            results_by_duration[dur] = []

        total = period_totals.get((dur, period), 0)
        if total > 0 and rv == "HIGH_RV":
            prob = n / total
            results_by_duration[dur].append({
                "period": period,
                "n": total,
                "prob_high_rv": prob,
            })

    # Compute ODD robustness for each duration tercile
    results = []
    for dur, periods in results_by_duration.items():
        if len(periods) < 4:
            continue

        probs = [p["prob_high_rv"] for p in periods]

        # Test if P(HIGH_RV | SHORT) > 0.33 consistently
        # or if P(HIGH_RV | LONG) < 0.33 consistently
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
            "duration_tercile": dur,
            "n_periods": len(periods),
            "total_n": sum(p["n"] for p in periods),
            "mean_prob_high_rv": round(mean_prob, 3),
            "min_prob": round(min(probs), 3),
            "max_prob": round(max(probs), 3),
            "t_stat_vs_null": round(t_stat, 2),
            "same_sign": all_same_sign,
            "is_odd_robust": is_odd_robust,
            "direction": "+" if mean_prob > null_prob else "-",
        })

    return results


def main() -> None:
    """Run duration-based volatility prediction analysis."""
    log_info("Starting duration-volatility prediction analysis")

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

        results = analyze_duration_volatility_prediction(symbol)

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
    print("DURATION-BASED VOLATILITY PREDICTION")
    print("Hypothesis: SHORT duration bars → HIGH forward realized volatility")
    print("=" * 130)

    print("\n" + "-" * 130)
    print("P(HIGH_RV | duration_tercile) - Null hypothesis: P = 0.33")
    print("-" * 130)
    print(
        f"{'Duration':<10} {'Symbol':<10} {'Periods':>8} {'Total N':>12} "
        f"{'P(HIGH_RV)':>12} {'Min P':>8} {'Max P':>8} {'t-stat':>10} {'Sign':>5} {'ODD':>5}"
    )
    print("-" * 130)

    for r in sorted(
        all_results,
        key=lambda x: (x["duration_tercile"], x["symbol"]),
    ):
        print(
            f"{r['duration_tercile']:<10} {r['symbol']:<10} "
            f"{r['n_periods']:>8} {r['total_n']:>12} {r['mean_prob_high_rv']:>12.3f} "
            f"{r['min_prob']:>8.3f} {r['max_prob']:>8.3f} "
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

    print(f"\nTotal duration-symbol combinations: {total}")
    print(f"Same sign across periods: {same_sign} ({100*same_sign/max(total, 1):.1f}%)")
    print(f"ODD robust (|t| >= 3.0): {odd_robust} ({100*odd_robust/max(total, 1):.1f}%)")

    # Check if SHORT duration → HIGH RV hypothesis holds
    if odd_robust > 0:
        print("\n" + "-" * 130)
        print("HYPOTHESIS CHECK")
        print("-" * 130)

        for dur in ["SHORT", "MID", "LONG"]:
            dur_results = [r for r in all_results if r["duration_tercile"] == dur and r["is_odd_robust"]]
            if dur_results:
                avg_prob = sum(r["mean_prob_high_rv"] for r in dur_results) / len(dur_results)
                print(f"  {dur}: {len(dur_results)} ODD robust, avg P(HIGH_RV) = {avg_prob:.3f}")

        short_high_rv = [r for r in all_results if r["duration_tercile"] == "SHORT" and r["is_odd_robust"] and r["direction"] == "+"]
        long_low_rv = [r for r in all_results if r["duration_tercile"] == "LONG" and r["is_odd_robust"] and r["direction"] == "-"]

        print(f"\n  SHORT → HIGH_RV (ODD robust, +): {len(short_high_rv)} symbols")
        print(f"  LONG → LOW_RV (ODD robust, -): {len(long_low_rv)} symbols")

        if len(short_high_rv) >= 4 or len(long_low_rv) >= 4:
            print("\n  ✓ UNIVERSAL PATTERN FOUND: Volatility clustering is ODD robust")
        else:
            print("\n  ✗ No universal pattern: Volatility clustering not consistently ODD robust")

    log_info(
        "Analysis complete",
        total=total,
        same_sign=same_sign,
        odd_robust=odd_robust,
    )


if __name__ == "__main__":
    main()
