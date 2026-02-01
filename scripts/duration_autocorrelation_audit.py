#!/usr/bin/env python3
# allow-simple-sharpe: volatility research audit
"""Duration Autocorrelation Forensic Audit.

Adversarial audit of duration persistence findings to check for:
1. Mechanical artifacts from adjacent bars sharing ticks
2. Temporal overlap between consecutive bars
3. Tercile boundary sensitivity (different quantile thresholds)
4. Cross-bar gap analysis

Issue #52, #56: Post-Audit Volatility Research
"""

import json
import subprocess
import sys
from datetime import datetime, timezone


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


def audit_temporal_overlap(symbol: str = "BTCUSDT", threshold: int = 100) -> dict:
    """Check if consecutive bars overlap temporally.

    Returns percentage of bar pairs where bar N+1 starts before bar N ends.
    """
    query = f"""
    WITH
        bars AS (
            SELECT
                timestamp_ms AS close_ms,
                timestamp_ms - duration_us / 1000 AS open_ms,
                leadInFrame(timestamp_ms - duration_us / 1000, 1) OVER (
                    ORDER BY timestamp_ms
                ) AS next_open_ms
            FROM rangebar_cache.range_bars
            WHERE symbol = '{symbol}'
              AND threshold_decimal_bps = {threshold}
              AND ouroboros_mode = 'year'
        )
    SELECT
        count() AS total_pairs,
        countIf(next_open_ms < close_ms) AS overlapping_pairs,
        round(100.0 * countIf(next_open_ms < close_ms) / count(), 2) AS overlap_pct,
        avg(close_ms - next_open_ms) / 1000 AS avg_overlap_sec
    FROM bars
    WHERE next_open_ms IS NOT NULL
    """

    rows = run_clickhouse_query(query)
    if rows:
        return {
            "total_pairs": rows[0]["total_pairs"],
            "overlapping_pairs": rows[0]["overlapping_pairs"],
            "overlap_pct": rows[0]["overlap_pct"],
            "avg_overlap_sec": round(rows[0]["avg_overlap_sec"], 2) if rows[0]["avg_overlap_sec"] else 0,
        }
    return {}


def audit_gap_distribution(symbol: str = "BTCUSDT", threshold: int = 100) -> dict:
    """Analyze time gaps between consecutive bars.

    If gaps are very small/zero, persistence is mechanical.
    """
    query = f"""
    WITH
        bars AS (
            SELECT
                timestamp_ms AS close_ms,
                leadInFrame(timestamp_ms - duration_us / 1000, 1) OVER (
                    ORDER BY timestamp_ms
                ) AS next_open_ms
            FROM rangebar_cache.range_bars
            WHERE symbol = '{symbol}'
              AND threshold_decimal_bps = {threshold}
              AND ouroboros_mode = 'year'
        ),
        gaps AS (
            SELECT (next_open_ms - close_ms) / 1000.0 AS gap_sec
            FROM bars
            WHERE next_open_ms IS NOT NULL
        )
    SELECT
        count() AS n_pairs,
        avg(gap_sec) AS avg_gap_sec,
        median(gap_sec) AS median_gap_sec,
        quantile(0.05)(gap_sec) AS p5_gap_sec,
        quantile(0.95)(gap_sec) AS p95_gap_sec,
        min(gap_sec) AS min_gap_sec,
        max(gap_sec) AS max_gap_sec,
        countIf(gap_sec <= 0) AS zero_or_negative_gaps
    FROM gaps
    """

    rows = run_clickhouse_query(query)
    if rows:
        r = rows[0]
        return {
            "n_pairs": r["n_pairs"],
            "avg_gap_sec": round(r["avg_gap_sec"], 2),
            "median_gap_sec": round(r["median_gap_sec"], 2),
            "p5_gap_sec": round(r["p5_gap_sec"], 2),
            "p95_gap_sec": round(r["p95_gap_sec"], 2),
            "min_gap_sec": round(r["min_gap_sec"], 2),
            "max_gap_sec": round(r["max_gap_sec"], 2),
            "zero_or_negative_gap_pct": round(100.0 * r["zero_or_negative_gaps"] / r["n_pairs"], 2),
        }
    return {}


def audit_quantile_sensitivity(symbol: str = "BTCUSDT", threshold: int = 100) -> list[dict]:
    """Test if persistence holds with different tercile boundaries.

    Tests: (25/75), (33/67), (40/60) quantile splits.
    """
    results = []

    for q_low, q_high in [(0.25, 0.75), (0.33, 0.67), (0.40, 0.60)]:
        query = f"""
        WITH
            bars_with_duration AS (
                SELECT
                    timestamp_ms,
                    duration_us,
                    leadInFrame(duration_us, 1) OVER (
                        ORDER BY timestamp_ms
                    ) AS next_duration_us
                FROM rangebar_cache.range_bars
                WHERE symbol = '{symbol}'
                  AND threshold_decimal_bps = {threshold}
                  AND ouroboros_mode = 'year'
            ),
            quantiles AS (
                SELECT
                    quantile({q_low})(duration_us) AS q_low,
                    quantile({q_high})(duration_us) AS q_high
                FROM bars_with_duration
                WHERE next_duration_us IS NOT NULL
            ),
            classified AS (
                SELECT
                    multiIf(
                        b.duration_us <= q.q_low, 'SHORT',
                        b.duration_us <= q.q_high, 'MID',
                        'LONG'
                    ) AS current_tercile,
                    multiIf(
                        b.next_duration_us <= q.q_low, 'SHORT',
                        b.next_duration_us <= q.q_high, 'MID',
                        'LONG'
                    ) AS next_tercile
                FROM bars_with_duration b
                CROSS JOIN quantiles q
                WHERE b.next_duration_us IS NOT NULL
            )
        SELECT
            current_tercile,
            count() AS n,
            countIf(next_tercile = current_tercile) AS persist_count,
            round(100.0 * countIf(next_tercile = current_tercile) / count(), 2) AS persist_pct
        FROM classified
        GROUP BY current_tercile
        ORDER BY current_tercile
        """

        rows = run_clickhouse_query(query)
        for row in rows:
            results.append({
                "quantile_split": f"{int(q_low*100)}/{int(q_high*100)}",
                "tercile": row["current_tercile"],
                "n": row["n"],
                "persist_pct": row["persist_pct"],
            })

    return results


def audit_skip_bar_persistence(symbol: str = "BTCUSDT", threshold: int = 100) -> list[dict]:
    """Test persistence with lag=2 (skip one bar) to check for mechanical artifacts.

    If persistence only exists at lag=1 but not lag=2, it's more likely mechanical.
    """
    results = []

    for lag in [1, 2, 3, 5]:
        query = f"""
        WITH
            bars_with_duration AS (
                SELECT
                    timestamp_ms,
                    duration_us,
                    leadInFrame(duration_us, {lag}) OVER (
                        ORDER BY timestamp_ms
                        ROWS BETWEEN CURRENT ROW AND {lag} FOLLOWING
                    ) AS future_duration_us
                FROM rangebar_cache.range_bars
                WHERE symbol = '{symbol}'
                  AND threshold_decimal_bps = {threshold}
                  AND ouroboros_mode = 'year'
            ),
            quantiles AS (
                SELECT
                    quantile(0.33)(duration_us) AS q33,
                    quantile(0.67)(duration_us) AS q67
                FROM bars_with_duration
                WHERE future_duration_us IS NOT NULL
            ),
            classified AS (
                SELECT
                    multiIf(
                        b.duration_us <= q.q33, 'SHORT',
                        b.duration_us <= q.q67, 'MID',
                        'LONG'
                    ) AS current_tercile,
                    multiIf(
                        b.future_duration_us <= q.q33, 'SHORT',
                        b.future_duration_us <= q.q67, 'MID',
                        'LONG'
                    ) AS future_tercile
                FROM bars_with_duration b
                CROSS JOIN quantiles q
                WHERE b.future_duration_us IS NOT NULL
            )
        SELECT
            current_tercile,
            count() AS n,
            round(100.0 * countIf(future_tercile = current_tercile) / count(), 2) AS persist_pct
        FROM classified
        GROUP BY current_tercile
        ORDER BY current_tercile
        """

        rows = run_clickhouse_query(query)
        for row in rows:
            results.append({
                "lag": lag,
                "tercile": row["current_tercile"],
                "n": row["n"],
                "persist_pct": row["persist_pct"],
            })

    return results


def main() -> None:
    """Run duration autocorrelation audit."""
    log_info("Starting duration autocorrelation audit")

    # Check connection
    try:
        run_clickhouse_query("SELECT 1")
        log_info("ClickHouse connected via SSH")
    except subprocess.CalledProcessError as e:
        log_info("ERROR: ClickHouse query failed", error=str(e))
        sys.exit(1)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

    print("\n" + "=" * 100)
    print("DURATION AUTOCORRELATION FORENSIC AUDIT")
    print("=" * 100)

    # Audit 1: Temporal Overlap
    print("\n" + "-" * 100)
    print("AUDIT 1: Temporal Overlap Check")
    print("Do consecutive bars overlap in time?")
    print("-" * 100)
    print(f"{'Symbol':<10} {'Total Pairs':>15} {'Overlapping':>15} {'Overlap %':>12} {'Avg Overlap (sec)':>18}")
    print("-" * 100)

    for symbol in symbols:
        result = audit_temporal_overlap(symbol)
        print(
            f"{symbol:<10} {result['total_pairs']:>15,} {result['overlapping_pairs']:>15,} "
            f"{result['overlap_pct']:>12.2f}% {result['avg_overlap_sec']:>18.2f}"
        )

    # Audit 2: Gap Distribution
    print("\n" + "-" * 100)
    print("AUDIT 2: Inter-Bar Gap Distribution")
    print("How much time elapses between bar N close and bar N+1 open?")
    print("-" * 100)
    print(f"{'Symbol':<10} {'Avg Gap':>12} {'Median':>12} {'P5':>12} {'P95':>12} {'Zero/Neg %':>12}")
    print("-" * 100)

    for symbol in symbols:
        result = audit_gap_distribution(symbol)
        print(
            f"{symbol:<10} {result['avg_gap_sec']:>11.2f}s {result['median_gap_sec']:>11.2f}s "
            f"{result['p5_gap_sec']:>11.2f}s {result['p95_gap_sec']:>11.2f}s "
            f"{result['zero_or_negative_gap_pct']:>11.2f}%"
        )

    # Audit 3: Quantile Sensitivity (BTCUSDT only as representative)
    print("\n" + "-" * 100)
    print("AUDIT 3: Quantile Boundary Sensitivity (BTCUSDT)")
    print("Does persistence hold with different tercile definitions?")
    print("-" * 100)
    print(f"{'Quantile Split':<15} {'Tercile':<10} {'N':>15} {'Persist %':>12}")
    print("-" * 100)

    results = audit_quantile_sensitivity("BTCUSDT")
    for r in results:
        print(f"{r['quantile_split']:<15} {r['tercile']:<10} {r['n']:>15,} {r['persist_pct']:>12.2f}%")

    # Audit 4: Lag Sensitivity (BTCUSDT)
    print("\n" + "-" * 100)
    print("AUDIT 4: Lag Sensitivity (BTCUSDT)")
    print("Does persistence decay with lag? (Mechanical = drops sharply)")
    print("-" * 100)
    print(f"{'Lag':<8} {'Tercile':<10} {'N':>15} {'Persist %':>12}")
    print("-" * 100)

    results = audit_skip_bar_persistence("BTCUSDT")
    for r in results:
        print(f"{r['lag']:<8} {r['tercile']:<10} {r['n']:>15,} {r['persist_pct']:>12.2f}%")

    # Summary
    print("\n" + "=" * 100)
    print("AUDIT SUMMARY")
    print("=" * 100)

    print("""
Key Questions:
1. TEMPORAL OVERLAP: If bars overlap significantly, persistence is mechanical
2. GAP DISTRIBUTION: Zero/negative gaps indicate deferred-open semantics
3. QUANTILE SENSITIVITY: If persistence varies wildly, it's boundary-dependent
4. LAG DECAY: If persistence drops sharply at lag=2, it's primarily mechanical

Verdict will be based on these audit findings.
    """)

    log_info("Audit complete")


if __name__ == "__main__":
    main()
