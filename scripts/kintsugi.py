#!/usr/bin/env python3
# Issue #115: Kintsugi self-healing gap reconciliation
"""Kintsugi CLI: Self-healing gap reconciliation for ClickHouse cache.

Usage:
    python scripts/kintsugi.py                    # Single pass
    python scripts/kintsugi.py --daemon           # Long-running
    python scripts/kintsugi.py --dry-run          # Discover + classify only
    python scripts/kintsugi.py --symbol BTCUSDT   # Filter by symbol
    python scripts/kintsugi.py --max-p2-jobs 2    # Limit heavy fills

Exit codes:
    0 - All shards healed (or no shards found)
    1 - Some repairs failed
    2 - Error (connection, configuration)
"""

from __future__ import annotations

import argparse
import logging
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kintsugi: self-healing gap reconciliation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Run as long-lived daemon with adaptive polling",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover and classify shards without repairing",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Filter to a specific symbol (e.g., BTCUSDT)",
    )
    parser.add_argument(
        "--max-p2-jobs", type=int, default=1,
        help="Max P2 (historical) repairs per pass (default: 1)",
    )
    parser.add_argument(
        "--interval-clean", type=int, default=1800,
        help="Daemon sleep when clean, seconds (default: 1800)",
    )
    parser.add_argument(
        "--interval-active", type=int, default=900,
        help="Daemon sleep when active, seconds (default: 900)",
    )
    parser.add_argument(
        "--replay-dead-letters", action="store_true",
        help="Replay dead-letter files from sidecar flush failures",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from rangebar.kintsugi import (
        kintsugi_daemon,
        kintsugi_pass,
        replay_dead_letters,
    )

    # Replay dead-letters first (if requested or in daemon mode)
    if args.replay_dead_letters or args.daemon:
        replayed = replay_dead_letters()
        if replayed > 0:
            logging.getLogger(__name__).info(
                "replayed %d dead-letter bars", replayed,
            )

    if args.daemon:
        kintsugi_daemon(
            interval_clean_s=args.interval_clean,
            interval_active_s=args.interval_active,
            max_p2_jobs=args.max_p2_jobs,
            symbol=args.symbol,
        )
        return 0

    # Single pass
    results = kintsugi_pass(
        dry_run=args.dry_run,
        symbol=args.symbol,
        max_p2_jobs=args.max_p2_jobs,
        verbose=args.verbose,
    )

    if args.dry_run:
        return 0

    failed = sum(1 for r in results if not r.healed)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
