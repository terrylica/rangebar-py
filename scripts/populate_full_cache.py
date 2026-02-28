#!/usr/bin/env python3
"""Full cache population script for BigBlack (resource-aware).

Issue #84: Full ClickHouse repopulation with all 38 microstructure features.

SAFE EXECUTION STRATEGY:
========================
The 75 jobs are organized into phases to avoid blowing up server resources.

Phase 1 (Quick wins):     15 jobs @ 1000 dbps  - ~8 hrs  - Run 4 parallel
Phase 2 (Day trading):    15 jobs @ 250 dbps   - ~24 hrs - Run 2 parallel max
Phase 3 (Mid thresholds): 30 jobs @ 500,750    - ~16 hrs - Run 2-3 parallel
Phase 4 (Scalping):       15 jobs @ 100 dbps   - ~80 hrs - Run ONE at a time

MEMORY USAGE:
  - Each job: ~2 GB peak (processes day-by-day, memory safe)
  - 2 parallel: ~4 GB
  - 3 parallel: ~6 GB
  - 4 parallel: ~8 GB (risky on 32GB system with GPU workloads)

USAGE:
======
# Safe sequential (recommended first run)
uv run python scripts/populate_full_cache.py --phase 1

# Full repopulation with all features (Issue #84)
uv run python scripts/populate_full_cache.py --symbol BTCUSDT --threshold 250 \
    --force-refresh --include-microstructure

# Check progress
uv run python scripts/populate_full_cache.py --status

# Run specific symbol/threshold
uv run python scripts/populate_full_cache.py --symbol BTCUSDT --threshold 250

# Parallel execution (only for phases 2-3, with caution)
uv run python scripts/populate_full_cache.py --phase 2 --parallel 2

RESUME:
=======
Jobs automatically resume from checkpoint if interrupted.
Just run the same command again. Use --force-refresh to wipe and restart.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime

from rangebar.binance_vision import probe_latest_available_date

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Symbol list derived from the unified symbol registry (Issue #79).
# symbols.toml is the SSoT — no hardcoded fallback.
# If the registry fails to load, crash loudly.

def _get_symbols() -> dict[str, str]:
    """Get symbol -> start_date mapping from registry SSoT.

    Raises
    ------
    RuntimeError
        If the symbol registry cannot be loaded. Never falls back silently.
    """
    from rangebar.symbol_registry import (
        get_effective_start_date,
        get_registered_symbols,
    )

    symbols = {}
    registered = get_registered_symbols(asset_class="crypto")
    if not registered:
        msg = (
            "Symbol registry returned 0 crypto symbols. "
            "Is symbols.toml present and maturin develop run?"
        )
        raise RuntimeError(msg)
    skipped = []
    for symbol in registered:
        start = get_effective_start_date(symbol)
        if start:
            symbols[symbol] = start
        else:
            skipped.append(symbol)
    if skipped:
        logger.warning(
            "Symbols without effective_start (skipped): %s", ", ".join(skipped)
        )
    logger.info(
        "Registry loaded: %d symbols (skipped %d without effective_start)",
        len(symbols), len(skipped),
    )
    return symbols


SYMBOLS = _get_symbols()

# Script-level T-1 guard via probe_latest_available_date().
# Defense-in-depth: populate_cache_resumable() also clamps end_date >= today
# to T-1 (yesterday UTC) at the library level, so even if this probe fails
# or a caller passes --end-date with a future date, the library catches it.
END_DATE = probe_latest_available_date()

# Phases organized by resource requirements
PHASES = {
    1: {
        "name": "Quick wins (1000 dbps)",
        "thresholds": [1000],
        "max_parallel": 4,
        "estimated_hours": 8,
    },
    2: {
        "name": "Day trading (250 dbps)",
        "thresholds": [250],
        "max_parallel": 2,
        "estimated_hours": 24,
    },
    3: {
        "name": "Mid thresholds (500, 750 dbps)",
        "thresholds": [500, 750],
        "max_parallel": 3,
        "estimated_hours": 16,
    },
    4: {
        "name": "Scalping (100 dbps) - RESOURCE INTENSIVE",
        "thresholds": [100],
        "max_parallel": 1,  # ONE at a time!
        "estimated_hours": 80,
    },
}


def populate_job(
    symbol: str,
    threshold: int,
    start_date: str,
    *,
    end_date: str | None = None,
    force_refresh: bool = False,
    include_microstructure: bool = True,
    ouroboros: str = "year",
) -> int:
    """Run a single population job. Returns bar count."""
    from rangebar import populate_cache_resumable

    job_end = end_date or END_DATE
    logger.info(
        "Starting: %s @ %d dbps from %s to %s (force_refresh=%s, microstructure=%s)",
        symbol, threshold, start_date, job_end, force_refresh, include_microstructure,
    )
    start_time = datetime.now(tz=UTC)

    try:
        bars = populate_cache_resumable(
            symbol,
            start_date,
            job_end,
            threshold_decimal_bps=threshold,
            include_microstructure=include_microstructure,
            force_refresh=force_refresh,
            ouroboros=ouroboros,
            notify=True,
        )
        elapsed = (datetime.now(tz=UTC) - start_time).total_seconds() / 60
        logger.info(
            "Completed: %s @ %d - %d bars in %.1f min",
            symbol, threshold, bars, elapsed,
        )
        return bars
    except Exception:
        logger.exception("Failed: %s @ %d", symbol, threshold)
        raise


def run_phase(
    phase_num: int,
    parallel: int = 1,
    *,
    force_refresh: bool = False,
    include_microstructure: bool = True,
    ouroboros: str = "year",
) -> None:
    """Run a specific phase with optional parallelization."""
    if phase_num not in PHASES:
        logger.error("Invalid phase: %d. Use 1-4.", phase_num)
        sys.exit(1)

    phase = PHASES[phase_num]
    max_parallel = phase["max_parallel"]

    if parallel > max_parallel:
        logger.warning(
            "Phase %d max parallel is %d, reducing from %d",
            phase_num,
            max_parallel,
            parallel,
        )
        parallel = max_parallel

    logger.info("=" * 70)
    logger.info("PHASE %d: %s", phase_num, phase["name"])
    logger.info("Estimated time: ~%d hours (sequential)", phase["estimated_hours"])
    logger.info("Running with parallelism: %d", parallel)
    logger.info(
        "force_refresh=%s, include_microstructure=%s",
        force_refresh, include_microstructure,
    )
    logger.info("=" * 70)

    jobs = [
        (symbol, thresh, SYMBOLS[symbol])
        for symbol in SYMBOLS
        for thresh in phase["thresholds"]
    ]

    if parallel == 1:
        # Sequential execution (safest, recommended)
        for symbol, threshold, start_date in jobs:
            populate_job(
                symbol, threshold, start_date,
                force_refresh=force_refresh,
                include_microstructure=include_microstructure,
                ouroboros=ouroboros,
            )
    else:
        # Bounded parallel execution using subprocess isolation
        # This avoids ProcessPoolExecutor memory leak issues by running
        # each job in a completely isolated subprocess
        import subprocess
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def run_job_subprocess(job_tuple: tuple) -> tuple:
            """Run a single job in isolated subprocess to prevent memory leaks."""
            sym, thresh, _start = job_tuple
            cmd = [
                sys.executable,
                __file__,
                "--symbol",
                sym,
                "--threshold",
                str(thresh),
            ]
            if force_refresh:
                cmd.append("--force-refresh")
            if not include_microstructure:
                cmd.append("--no-microstructure")
            if ouroboros != "year":
                cmd.extend(["--ouroboros", ouroboros])
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return (sym, thresh, result.returncode, result.stderr)

        # Use ThreadPoolExecutor to manage subprocess spawning (not CPU work)
        # Each subprocess is fully isolated - no shared memory
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(run_job_subprocess, job): job[:2] for job in jobs}

            for future in as_completed(futures):
                symbol, threshold = futures[future]
                try:
                    _sym, _thresh, returncode, stderr = future.result()
                    if returncode == 0:
                        logger.info("Job done: %s @ %d", symbol, threshold)
                    else:
                        logger.error(
                            "Job failed: %s @ %d - %s", symbol, threshold, stderr[:500]
                        )
                except (ValueError, RuntimeError, OSError, ConnectionError):
                    logger.exception("Job failed: %s @ %d", symbol, threshold)


def show_status() -> None:
    """Show current cache status and remaining jobs."""
    try:
        from rangebar.clickhouse import RangeBarCache

        with RangeBarCache() as cache:
            result = cache.client.query("""
                SELECT symbol, threshold_decimal_bps, count(*) as bars,
                       formatDateTime(fromUnixTimestamp64Milli(min(close_time_ms)), '%Y-%m-%d') as earliest,
                       formatDateTime(fromUnixTimestamp64Milli(max(close_time_ms)), '%Y-%m-%d') as latest
                FROM rangebar_cache.range_bars FINAL
                GROUP BY symbol, threshold_decimal_bps
                ORDER BY symbol, threshold_decimal_bps
            """)

            cached = {(row[0], row[1]): (row[2], row[3], row[4]) for row in result.result_rows}

        print("\n" + "=" * 90)
        print("CACHE STATUS")
        print("=" * 90)
        print(f"\n{'Symbol':<12} {'Thresh':<8} {'Status':<12} {'Bars':>12} {'Coverage':<25}")
        print("-" * 90)

        all_thresholds = [100, 250, 500, 750, 1000]
        completed = 0
        remaining = 0

        for symbol in SYMBOLS:
            for thresh in all_thresholds:
                if (symbol, thresh) in cached:
                    bars, earliest, latest = cached[(symbol, thresh)]
                    status = "Done"
                    coverage = f"{earliest} to {latest}"
                    completed += 1
                else:
                    bars = 0
                    status = "Pending"
                    coverage = "-"
                    remaining += 1

                print(f"{symbol:<12} {thresh:<8} {status:<12} {bars:>12,} {coverage:<25}")

        total = completed + remaining
        print("-" * 90)
        print(f"Completed: {completed}/{total} jobs | Remaining: {remaining}/{total} jobs")
        print()

    except (ImportError, ConnectionError, OSError, RuntimeError):
        logger.exception("Could not connect to ClickHouse")


def show_plan() -> None:
    """Show the full execution plan."""
    total_jobs = len(SYMBOLS) * sum(
        len(p["thresholds"]) for p in PHASES.values()
    )
    print("\n" + "=" * 90)
    print(f"EXECUTION PLAN ({total_jobs} jobs total)")
    print("=" * 90)

    total_hours = 0
    for phase_num, phase in PHASES.items():
        n_jobs = len(SYMBOLS) * len(phase["thresholds"])
        print(f"""
Phase {phase_num}: {phase['name']}
  Jobs: {n_jobs}
  Thresholds: {phase['thresholds']}
  Max parallel: {phase['max_parallel']}
  Estimated time: ~{phase['estimated_hours']} hours
  Command: uv run python scripts/populate_full_cache.py --phase {phase_num}
""")
        total_hours += phase["estimated_hours"]

    print("=" * 90)
    print(f"Total estimated time (sequential): ~{total_hours} hours")
    print("With parallelization: ~50-60 hours")
    print()
    print("RECOMMENDED APPROACH:")
    print("  1. Run Phase 1 first (quick wins, low resource)")
    print("  2. Run Phase 2 overnight (most useful data)")
    print("  3. Run Phase 3 when not using GPU")
    print("  4. Run Phase 4 over weekend (resource intensive)")
    print()


def main() -> None:
    """Main entry point."""
    from rangebar.logging import setup_service_logging

    setup_service_logging("populate-cache")

    parser = argparse.ArgumentParser(
        description="Populate rangebar cache (resource-aware)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --plan                    # Show full execution plan
  %(prog)s --status                  # Show current cache status
  %(prog)s --phase 1                 # Run Phase 1 (1000 dbps, quick)
  %(prog)s --phase 2 --parallel 2    # Run Phase 2 with 2 parallel jobs
  %(prog)s --symbol BTCUSDT --threshold 250  # Run single job
        """,
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], help="Run phase 1-4")
    parser.add_argument(
        "--parallel", type=int, default=1, help="Number of parallel jobs (default: 1)"
    )
    parser.add_argument("--symbol", type=str, help="Single symbol (requires --threshold)")
    parser.add_argument("--threshold", type=int, help="Single threshold (requires --symbol)")
    parser.add_argument(
        "--start-date", type=str,
        help="Override start date (YYYY-MM-DD). For per-year parallelization.",
    )
    parser.add_argument(
        "--end-date", type=str,
        help="Override end date (YYYY-MM-DD). For per-year parallelization.",
    )
    parser.add_argument("--status", action="store_true", help="Show cache status")
    parser.add_argument("--plan", action="store_true", help="Show execution plan")
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Wipe existing cache and checkpoint, start fresh",
    )
    parser.add_argument(
        "--include-microstructure", action="store_true", default=True,
        help="Include all 38 microstructure features (default: True)",
    )
    parser.add_argument(
        "--no-microstructure", action="store_true",
        help="Disable microstructure features",
    )
    parser.add_argument(
        "--ouroboros", type=str, default=None,
        choices=["year", "month", "week"],
        help="Ouroboros reset period (default: from RANGEBAR_OUROBOROS_MODE).",  # Issue #126
    )

    args = parser.parse_args()

    # Resolve microstructure flag (--no-microstructure overrides default)
    include_micro = args.include_microstructure and not args.no_microstructure

    if args.plan:
        show_plan()
    elif args.status:
        show_status()
    elif args.symbol and args.threshold:
        if args.symbol not in SYMBOLS:
            logger.error("Unknown symbol: %s", args.symbol)
            sys.exit(1)
        job_start = args.start_date or SYMBOLS[args.symbol]
        populate_job(
            args.symbol, args.threshold, job_start,
            end_date=args.end_date,
            force_refresh=args.force_refresh,
            include_microstructure=include_micro,
            ouroboros=args.ouroboros,
        )
    elif args.phase:
        run_phase(
            args.phase, args.parallel,
            force_refresh=args.force_refresh,
            include_microstructure=include_micro,
            ouroboros=args.ouroboros,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
