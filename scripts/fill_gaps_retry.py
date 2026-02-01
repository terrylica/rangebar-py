"""Retry failed segments from the gap fill (Oct-Nov 2022).

Issue #52: These segments failed due to memory guard being too conservative.
Now using max_memory_mb=0 to disable the guard.

Runs LOCALLY (macOS ARM64) where tick data is cached, but writes
to littleblack ClickHouse via SSH tunnel.
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
    log("INFO", "Retrying failed segments", script="fill_gaps_retry.py")

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

    # Issue #52: Retry failed segments with memory guard disabled
    symbol = "BTCUSDT"
    threshold = 100

    # These failed due to memory guard
    segments = [
        ("2022-10-01", "2022-10-31"),
        ("2022-11-01", "2022-11-30"),
    ]

    log("INFO", "Starting retry",
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
                ouroboros="year",
                use_cache=True,
                fetch_if_missing=True,
                include_microstructure=True,
                max_memory_mb=0,  # Issue #52: Disable memory guard
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

    log("INFO", "Retry complete", total_bars=total_bars, total_segments=len(segments))


if __name__ == "__main__":
    main()
