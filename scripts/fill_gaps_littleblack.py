"""Fill data gaps in littleblack ClickHouse cache.

Runs LOCALLY (macOS ARM64) where tick data is cached, but writes
to littleblack ClickHouse via SSH tunnel (auto-detected by preflight.py).

Uses NDJSON (.jsonl) logging format for all output.

Requires: RANGEBAR_CH_HOSTS=littleblack environment variable.
"""

import json
import os
import sys
import time
from datetime import UTC, datetime


def log(level: str, message: str, **kwargs: object) -> None:
    """Log in NDJSON format."""
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry), flush=True)


def main() -> None:
    log("INFO", "Starting gap fill process", script="fill_gaps_littleblack.py")

    # Verify environment
    ch_hosts = os.environ.get("RANGEBAR_CH_HOSTS", "")
    if ch_hosts != "littleblack":
        log("ERROR", "RANGEBAR_CH_HOSTS must be set to 'littleblack'",
            current_value=ch_hosts)
        sys.exit(1)

    log("INFO", "Environment verified", RANGEBAR_CH_HOSTS=ch_hosts)

    # Fail-fast: Check rangebar import
    try:
        import rangebar
        from rangebar import get_range_bars
        log("INFO", "Rangebar imported successfully", version=rangebar.__version__)
    except ImportError as e:
        log("ERROR", "Failed to import rangebar", error=str(e))
        sys.exit(1)

    # Fail-fast: Check ClickHouse connectivity
    try:
        from rangebar.clickhouse import RangeBarCache
        with RangeBarCache() as cache:
            count = cache.count_bars("BTCUSDT", 100)
            log("INFO", "ClickHouse connection OK", current_bars=count)
    except (ConnectionError, OSError, RuntimeError) as e:
        log("ERROR", "ClickHouse connection failed", error=str(e))
        sys.exit(1)

    # Define gaps to fill (BTCUSDT @ 100 dbps)
    # Issue #52: Using monthly segments for market regime research
    symbol = "BTCUSDT"
    threshold = 100

    # Gap 1: 2022-07 to 2023-05 (10 months)
    # Gap 2: 2024-03 to 2024-11 (8 months)
    # Process month by month for memory safety
    segments = [
        # 2022 gap months
        ("2022-07-01", "2022-07-31"),
        ("2022-08-01", "2022-08-31"),
        ("2022-09-01", "2022-09-30"),
        ("2022-10-01", "2022-10-31"),
        ("2022-11-01", "2022-11-30"),
        ("2022-12-01", "2022-12-31"),
        # 2023 gap months
        ("2023-01-01", "2023-01-31"),
        ("2023-02-01", "2023-02-28"),
        ("2023-03-01", "2023-03-31"),
        ("2023-04-01", "2023-04-30"),
        ("2023-05-01", "2023-05-31"),
        # 2024 gap months
        ("2024-03-01", "2024-03-31"),
        ("2024-04-01", "2024-04-30"),
        ("2024-05-01", "2024-05-31"),
        ("2024-06-01", "2024-06-30"),
        ("2024-07-01", "2024-07-31"),
        ("2024-08-01", "2024-08-31"),
        ("2024-09-01", "2024-09-30"),
        ("2024-10-01", "2024-10-31"),
        ("2024-11-01", "2024-11-30"),
        ("2024-12-01", "2024-12-31"),
        # 2025 full year
        ("2025-01-01", "2025-01-31"),
        ("2025-02-01", "2025-02-28"),
        ("2025-03-01", "2025-03-31"),
        ("2025-04-01", "2025-04-30"),
        ("2025-05-01", "2025-05-31"),
        ("2025-06-01", "2025-06-30"),
        ("2025-07-01", "2025-07-31"),
        ("2025-08-01", "2025-08-31"),
        ("2025-09-01", "2025-09-30"),
        ("2025-10-01", "2025-10-31"),
        ("2025-11-01", "2025-11-30"),
        ("2025-12-01", "2025-12-31"),
        # 2026 January
        ("2026-01-01", "2026-01-31"),
    ]

    log("INFO", "Starting gap fill",
        total_segments=len(segments),
        symbol=symbol,
        threshold=threshold)

    total_bars = 0
    for i, (start, end) in enumerate(segments, 1):
        log("INFO", "Processing segment", segment_number=i, start=start, end=end)

        start_time = time.time()
        try:
            df = get_range_bars(
                symbol,
                start_date=start,
                end_date=end,
                threshold_decimal_bps=threshold,
                ouroboros="month",  # Memory-safe year boundaries
                use_cache=True,
                fetch_if_missing=True,
                include_microstructure=True,
                max_memory_mb=0,  # Disable memory guard - we have 32GB
            )
            elapsed = time.time() - start_time
            bars = len(df)
            total_bars += bars
            log("INFO", "Segment filled successfully",
                segment_number=i, start=start, end=end,
                bars=bars, elapsed_seconds=round(elapsed, 1))
        except (ValueError, RuntimeError, MemoryError) as e:
            log("ERROR", "Segment fill failed",
                segment_number=i, start=start, end=end, error=str(e))
            # Continue to next segment instead of failing

    log("INFO", "Gap fill complete", total_bars=total_bars, total_segments=len(segments))


if __name__ == "__main__":
    main()
