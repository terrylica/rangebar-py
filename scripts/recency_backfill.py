#!/usr/bin/env python3
"""Layer 2 recency backfill CLI (Issue #92).

Bridges the gap between latest Binance Vision archive and now via REST API.

USAGE:
======
# Single-shot: backfill one symbol
uv run python scripts/recency_backfill.py --symbol BTCUSDT --threshold 250

# Single-shot: backfill all cached pairs
uv run python scripts/recency_backfill.py --all

# Adaptive loop (long-running sidecar)
uv run python scripts/recency_backfill.py --loop

# Adaptive loop with verbose logging
uv run python scripts/recency_backfill.py --loop --verbose
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Layer 2 recency backfill for range bar cache",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--symbol",
        help="Backfill a single symbol (requires --threshold)",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Backfill all cached symbol x threshold pairs",
    )
    mode.add_argument(
        "--loop",
        action="store_true",
        help="Run adaptive backfill loop (Ctrl+C to stop)",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=250,
        help="Threshold in decimal basis points (default: 250)",
    )
    parser.add_argument(
        "--no-microstructure",
        action="store_true",
        help="Skip microstructure features (faster)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    from rangebar.logging import setup_service_logging

    logger = setup_service_logging("recency-backfill", verbose=args.verbose)

    include_micro = not args.no_microstructure

    from rangebar.recency import (
        backfill_all_recent,
        backfill_recent,
        run_adaptive_loop,
    )

    if args.symbol:
        result = backfill_recent(
            args.symbol,
            args.threshold,
            include_microstructure=include_micro,
            verbose=args.verbose,
        )
        if result.error:
            logger.bind(error=str(result.error)).error("Backfill error")
            return 1
        logger.bind(
            bars_written=result.bars_written,
            gap_minutes=round(result.gap_seconds / 60, 1),
        ).info("Backfill complete")
        return 0

    if args.all:
        results = backfill_all_recent(
            include_microstructure=include_micro,
            verbose=args.verbose,
        )
        total = sum(r.bars_written for r in results)
        errors = sum(1 for r in results if r.error)
        logger.bind(
            pairs=len(results), bars_written=total, errors=errors,
        ).info("Backfill all complete")
        return 1 if errors > 0 else 0

    if args.loop:
        run_adaptive_loop(
            include_microstructure=include_micro,
            verbose=args.verbose,
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
