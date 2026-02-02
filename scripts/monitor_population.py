#!/usr/bin/env python3
"""Monitor cache population progress on littleblack.

Designed for Claude Code background task monitoring.
Outputs structured status updates that can be checked via TaskOutput.

Usage:
    python scripts/monitor_population.py [--interval SECONDS] [--host HOST]

GitHub Issue: https://github.com/terrylica/rangebar-py/issues/58
"""

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PopulationStatus:
    """Current population status."""

    is_running: bool
    current_symbol: str | None
    current_threshold: int | None
    progress: str  # e.g., "3/18"
    last_month: str | None
    last_bars: int | None
    last_time: float | None
    total_bars_session: int
    errors: list[str]
    phase: str  # "250 dbps", "1000 dbps", "complete", "failed"


def run_ssh_command(host: str, cmd: str) -> tuple[int, str]:
    """Run SSH command and return exit code + output."""
    result = subprocess.run(
        ["ssh", host, cmd],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,  # We handle non-zero exits via return code
    )
    return result.returncode, result.stdout + result.stderr


def check_tmux_session(host: str) -> bool:
    """Check if populate-all tmux session is running."""
    code, output = run_ssh_command(host, "tmux list-sessions 2>/dev/null | grep populate-all")
    return code == 0 and "populate-all" in output


def check_python_process(host: str) -> bool:
    """Check if population Python process is running."""
    code, output = run_ssh_command(host, "pgrep -f 'populate' 2>/dev/null")
    return code == 0 and output.strip() != ""


def get_log_tail(host: str, lines: int = 100) -> str:
    """Get last N lines of log file."""
    code, output = run_ssh_command(host, f"tail -{lines} /tmp/populate_all_mise.log 2>/dev/null")
    return output if code == 0 else ""


def parse_status(log_content: str) -> PopulationStatus:
    """Parse log content to extract status."""
    lines = log_content.strip().split("\n")

    status = PopulationStatus(
        is_running=True,
        current_symbol=None,
        current_threshold=None,
        progress="?/?",
        last_month=None,
        last_bars=None,
        last_time=None,
        total_bars_session=0,
        errors=[],
        phase="unknown",
    )

    # Find current phase
    if "1000 dbps" in log_content and "=== SEQUENTIAL CACHE POPULATION (1000 dbps) ===" in log_content:
        status.phase = "1000 dbps"
    elif "250 dbps" in log_content:
        status.phase = "250 dbps"

    if "ALL POPULATION COMPLETE" in log_content:
        status.phase = "complete"
        status.is_running = False

    # Find current symbol and progress
    progress_pattern = re.compile(r"\[(\d+)/(\d+)\] (\w+) @ (\d+) dbps")
    for line in reversed(lines):
        match = progress_pattern.search(line)
        if match:
            current, total, symbol, threshold = match.groups()
            status.progress = f"{current}/{total}"
            status.current_symbol = symbol
            status.current_threshold = int(threshold)
            break

    # Find last successful month
    month_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}):\s*([\d,]+)\s*bars\s*\(\s*([\d.]+)s\)")
    for line in reversed(lines):
        match = month_pattern.search(line)
        if match:
            status.last_month = match.group(1)
            status.last_bars = int(match.group(2).replace(",", ""))
            status.last_time = float(match.group(3))
            break

    # Count total bars in session
    for line in lines:
        match = month_pattern.search(line)
        if match:
            bars = int(match.group(2).replace(",", ""))
            status.total_bars_session += bars

    # Find errors
    for line in lines:
        if "ERROR" in line or "error" in line.lower():
            status.errors.append(line.strip())

    return status


def format_status(status: PopulationStatus, host: str) -> str:
    """Format status for display."""
    from datetime import UTC
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"=== Population Monitor ({now}) ===",
        f"Host: {host}",
        f"Running: {'YES' if status.is_running else 'NO'}",
        f"Phase: {status.phase}",
        f"Progress: {status.progress}",
    ]

    if status.current_symbol:
        lines.append(f"Current: {status.current_symbol} @ {status.current_threshold} dbps")

    if status.last_month:
        lines.append(f"Last month: {status.last_month} ({status.last_bars:,} bars in {status.last_time:.1f}s)")

    lines.append(f"Session total: {status.total_bars_session:,} bars")

    if status.errors:
        lines.append(f"Errors: {len(status.errors)}")
        for err in status.errors[-3:]:  # Last 3 errors
            lines.append(f"  - {err[:80]}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Monitor cache population")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--host", default="littleblack", help="SSH host")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    print(f"Starting population monitor for {args.host}", flush=True)
    print(f"Check interval: {args.interval}s", flush=True)
    print("=" * 50, flush=True)

    consecutive_failures = 0
    max_failures = 3

    while True:
        try:
            # Check if process is running
            tmux_running = check_tmux_session(args.host)
            process_running = check_python_process(args.host)

            if not tmux_running and not process_running:
                print("\n[ALERT] No population process detected!", flush=True)
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            # Get and parse log
            log_content = get_log_tail(args.host)
            status = parse_status(log_content)

            # Update running status based on process check
            if not process_running:
                status.is_running = False

            # Print status
            print(f"\n{format_status(status, args.host)}", flush=True)

            # Check for completion or failure
            if status.phase == "complete":
                print("\n[SUCCESS] Population complete!", flush=True)
                break

            if consecutive_failures >= max_failures:
                print(f"\n[FAILURE] Process not running for {consecutive_failures} consecutive checks", flush=True)
                print("Last log content:", flush=True)
                print(log_content[-2000:] if log_content else "(empty)", flush=True)
                sys.exit(1)

            if status.errors:
                print(f"\n[WARNING] {len(status.errors)} errors detected", flush=True)

            if args.once:
                break

            time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\nMonitor stopped by user", flush=True)
            break
        except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            print(f"\n[ERROR] Monitor error: {e}", flush=True)
            if args.once:
                sys.exit(1)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
