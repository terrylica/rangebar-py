#!/usr/bin/env python3
"""Layer 3 streaming sidecar CLI (Issue #91).

Streams live trades from Binance WebSocket, constructs range bars in real-time
with full 58-column microstructure features, and writes them to ClickHouse.

USAGE:
======
# Start sidecar for one symbol
uv run python scripts/streaming_sidecar.py --symbol BTCUSDT --threshold 250

# Start sidecar for multiple symbols (all thresholds)
uv run python scripts/streaming_sidecar.py --symbol BTCUSDT --symbol ETHUSDT

# Start from env vars (RANGEBAR_STREAMING_*)
uv run python scripts/streaming_sidecar.py --from-env

# Verbose logging
uv run python scripts/streaming_sidecar.py --symbol BTCUSDT --verbose

# Skip gap-fill on startup
uv run python scripts/streaming_sidecar.py --symbol BTCUSDT --no-gap-fill
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Layer 3 streaming sidecar for live range bar construction",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help="Symbol to stream (can be repeated: --symbol BTCUSDT --symbol ETHUSDT)",
    )
    mode.add_argument(
        "--from-env",
        action="store_true",
        help="Load config from RANGEBAR_STREAMING_* env vars",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        action="append",
        dest="thresholds",
        help="Threshold in dbps (can be repeated; default: 250,500,750,1000)",
    )
    parser.add_argument(
        "--no-microstructure",
        action="store_true",
        help="Skip microstructure features (10 columns instead of 58)",
    )
    parser.add_argument(
        "--no-gap-fill",
        action="store_true",
        help="Skip Layer 2 gap-fill on startup",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    from rangebar.logging import setup_service_logging

    logger = setup_service_logging("sidecar", verbose=args.verbose)

    from rangebar.sidecar import SidecarConfig, run_sidecar

    if args.from_env:
        config = SidecarConfig.from_env()
    else:
        config = SidecarConfig(
            symbols=[s.upper() for s in args.symbols],
            thresholds=args.thresholds or [250, 500, 750, 1000],
            include_microstructure=not args.no_microstructure,
            gap_fill_on_startup=not args.no_gap_fill,
            verbose=args.verbose,
        )

    if not config.symbols:
        logger.error("No symbols configured. Use --symbol or RANGEBAR_STREAMING_SYMBOLS.")
        return 1

    logger.bind(
        symbols=config.symbols,
        thresholds=config.thresholds,
        microstructure=config.include_microstructure,
    ).info("Starting sidecar")

    run_sidecar(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
