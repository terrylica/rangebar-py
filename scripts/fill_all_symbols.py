"""Fill data for ALL symbols at ALL granularities (50, 100, 200 dbps).

Issue #52: Multi-instrument ODD robustness validation requires:
- BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT
- Each at 50, 100, 200 dbps
- Continuous data from 2022-01-01 to 2026-01-31

Runs LOCALLY (macOS ARM64) where tick data is cached, but writes
to littleblack ClickHouse via SSH tunnel.

Uses NDJSON (.jsonl) logging format for all output.
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
        **kwargs
    }
    print(json.dumps(entry), flush=True)


def generate_monthly_segments(start_year: int, start_month: int, end_year: int, end_month: int):
    """Generate monthly segment tuples."""
    segments = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        # Determine last day of month
        if month in (1, 3, 5, 7, 8, 10, 12):
            last_day = 31
        elif month in (4, 6, 9, 11):
            last_day = 30
        else:  # February
            last_day = 29 if year % 4 == 0 else 28

        start_date = f"{year:04d}-{month:02d}-01"
        end_date = f"{year:04d}-{month:02d}-{last_day:02d}"
        segments.append((start_date, end_date))

        # Next month
        month += 1
        if month > 12:
            month = 1
            year += 1

    return segments


def main():
    log("INFO", "Starting multi-symbol gap fill", script="fill_all_symbols.py")

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

    # Issue #52: Multi-instrument configuration
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    thresholds = [50, 100, 200]

    # Symbol-specific start dates (data availability varies)
    symbol_start = {
        "BTCUSDT": (2022, 1),
        "ETHUSDT": (2022, 1),
        "SOLUSDT": (2023, 6),  # SOL data starts later
        "BNBUSDT": (2022, 1),
    }

    # End date: January 2026
    end_year, end_month = 2026, 1

    # Generate all combinations
    combinations = []
    for symbol in symbols:
        start_year, start_month = symbol_start[symbol]
        segments = generate_monthly_segments(start_year, start_month, end_year, end_month)
        for threshold in thresholds:
            combinations.append((symbol, threshold, segments))

    total_combinations = len(combinations)
    log("INFO", "Starting multi-symbol fill",
        total_combinations=total_combinations,
        symbols=symbols,
        thresholds=thresholds)

    grand_total_bars = 0
    for combo_idx, (symbol, threshold, segments) in enumerate(combinations, 1):
        log("INFO", "Starting symbol/threshold combination",
            combination=combo_idx,
            total_combinations=total_combinations,
            symbol=symbol,
            threshold=threshold,
            segments_count=len(segments))

        combo_bars = 0
        for seg_idx, (start, end) in enumerate(segments, 1):
            log("INFO", "Processing segment",
                combination=combo_idx,
                segment=seg_idx,
                total_segments=len(segments),
                symbol=symbol,
                threshold=threshold,
                start=start,
                end=end)

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
                combo_bars += bars
                log("INFO", "Segment filled successfully",
                    combination=combo_idx,
                    segment=seg_idx,
                    symbol=symbol,
                    threshold=threshold,
                    start=start,
                    end=end,
                    bars=bars,
                    elapsed_seconds=round(elapsed, 1))
            except (ValueError, RuntimeError, MemoryError) as e:
                log("ERROR", "Segment fill failed",
                    combination=combo_idx,
                    segment=seg_idx,
                    symbol=symbol,
                    threshold=threshold,
                    start=start,
                    end=end,
                    error=str(e))

        grand_total_bars += combo_bars
        log("INFO", "Combination complete",
            combination=combo_idx,
            symbol=symbol,
            threshold=threshold,
            bars=combo_bars)

    log("INFO", "Multi-symbol fill complete",
        grand_total_bars=grand_total_bars,
        total_combinations=total_combinations)


if __name__ == "__main__":
    main()
