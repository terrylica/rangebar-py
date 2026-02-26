#!/usr/bin/env python3
"""Issue #109: Health check CLI for rangebar-py.

Runs all health probes and outputs JSON. Exit code 0 on all-pass, 1 on any failure.

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --host bigblack --port 8123
    python scripts/health_check.py | jq .
"""

from __future__ import annotations

import argparse
import json
import sys

from rangebar.health_checks import run_all_checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #109: rangebar-py health check")
    parser.add_argument("--host", type=str, default="localhost", help="ClickHouse host")
    parser.add_argument("--port", type=int, default=18123, help="ClickHouse HTTP port")
    parser.add_argument("--memory-threshold", type=int, default=2048, help="Memory threshold (MB)")
    parser.add_argument("--min-free-gb", type=float, default=10.0, help="Minimum free disk (GB)")
    args = parser.parse_args()

    results = run_all_checks(
        ch_host=args.host,
        ch_port=args.port,
        memory_threshold_mb=args.memory_threshold,
        min_free_gb=args.min_free_gb,
    )

    # Compute overall status (circuit_breaker is a string, not bool)
    bool_checks = {k: v for k, v in results.items() if isinstance(v, bool)}
    all_pass = all(bool_checks.values())

    output = {
        "status": "healthy" if all_pass else "unhealthy",
        "checks": results,
    }

    json.dump(output, sys.stdout, indent=2)
    print()  # trailing newline
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
