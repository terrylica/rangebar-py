#!/usr/bin/env python3
# PROCESS-STORM-OK: Migration script intentionally uses subprocess for systemctl/git
"""Migrate ClickHouse range_bars: timestamp_ms → close_time_ms + add open_time_ms.

# Issue #121, #122, #123 — Full Timestamp & Column Naming Revamp (Phase 14)

Full Timestamp & Column Naming Revamp — table recreation via CREATE + INSERT SELECT
+ atomic RENAME. Handles the complete deploy lifecycle when --deploy is passed.

Usage:
    # Migrate only (code already deployed):
    python scripts/migrate_timestamp_column.py

    # Full deploy + migrate (stops services, deploys code, migrates, restarts):
    python scripts/migrate_timestamp_column.py --deploy

    # Dry-run (show what would happen):
    python scripts/migrate_timestamp_column.py --dry-run

    # Rollback (swap back to old table):
    python scripts/migrate_timestamp_column.py --rollback
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], *, check: bool = True, capture: bool = False) -> str:
    """Run a command as a list of args, return stdout."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=capture, text=True)
    return result.stdout.strip() if capture else ""


def _ch_command(client, sql: str) -> object:
    """Execute a ClickHouse command."""
    return client.command(sql)


def _stop_services() -> None:
    """Stop rangebar services (sidecar, kintsugi, recency-backfill)."""
    print("\n=== Stopping services ===")
    services = [
        "rangebar-sidecar",
        "rangebar-kintsugi",
        "rangebar-recency-backfill",
    ]
    for svc in services:
        _run(["systemctl", "--user", "stop", svc], check=False)
    time.sleep(2)
    for svc in services:
        status = _run(
            ["systemctl", "--user", "is-active", svc], check=False, capture=True,
        )
        print(f"  {svc}: {status or 'stopped'}")


def _start_services() -> None:
    """Start rangebar services."""
    print("\n=== Starting services ===")
    services = [
        "rangebar-sidecar",
        "rangebar-kintsugi",
        "rangebar-recency-backfill",
    ]
    for svc in services:
        _run(["systemctl", "--user", "start", svc], check=False)
    time.sleep(2)
    for svc in services:
        status = _run(
            ["systemctl", "--user", "is-active", svc], check=False, capture=True,
        )
        print(f"  {svc}: {status}")


def _deploy_code() -> None:
    """Deploy new code: git pull + rebuild editable install.

    On bigblack, the venv has no pip — use uv pip install instead.
    The deploy:bigblack mise task uses PyPI wheels, but for migration
    we need the local editable install (code just pulled via git).
    """
    print("\n=== Deploying new code ===")
    _run(["git", "fetch", "origin", "main"])
    _run(["git", "reset", "--hard", "origin/main"])
    # Use uv (available at ~/.local/bin/uv on bigblack)
    import shutil
    uv_path = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    venv_python = str(Path.cwd() / ".venv" / "bin" / "python3")
    _run([uv_path, "pip", "install", "--python", venv_python, "--no-deps", "-e", "."])


def _check_legacy_schema(client) -> bool:
    """Check if the table still has the legacy timestamp_ms column."""
    result = client.query(
        "SELECT name FROM system.columns "
        "WHERE database='rangebar_cache' AND table='range_bars' "
        "AND name='timestamp_ms'"
    )
    return len(result.result_rows) > 0


def _check_new_schema(client) -> bool:
    """Check if the table already has close_time_ms column."""
    result = client.query(
        "SELECT name FROM system.columns "
        "WHERE database='rangebar_cache' AND table='range_bars' "
        "AND name='close_time_ms'"
    )
    return len(result.result_rows) > 0


def _get_all_columns(client, table: str) -> list[str]:
    """Get all column names for a table."""
    result = client.query(
        "SELECT name FROM system.columns "
        f"WHERE database='rangebar_cache' AND table='{table}' "
        "ORDER BY position"
    )
    return [row[0] for row in result.result_rows]


def migrate(client, *, dry_run: bool = False) -> bool:
    """Perform the table migration."""
    print("\n=== Checking current schema ===")

    if not _check_legacy_schema(client):
        if _check_new_schema(client):
            print("  Already migrated (close_time_ms exists, no timestamp_ms)")
            return True
        print("  ERROR: Neither timestamp_ms nor close_time_ms found!")
        return False

    # Get old table row count
    old_count = client.query(
        "SELECT count() FROM rangebar_cache.range_bars"
    ).result_rows[0][0]
    print(f"  Old table: {old_count:,} rows")

    if dry_run:
        print("\n  [DRY RUN] Would create range_bars_v2, INSERT SELECT, RENAME")
        print(f"  [DRY RUN] {old_count:,} rows to migrate")
        return True

    # Step 1: Drop v2 if exists from a previous failed attempt
    print("\n=== Step 1: Clean up any previous v2 table ===")
    _ch_command(client, "DROP TABLE IF EXISTS rangebar_cache.range_bars_v2")

    # Step 2: Read the new schema and create v2 table
    print("\n=== Step 2: Create range_bars_v2 with new schema ===")
    schema_path = (
        Path(__file__).parent.parent
        / "python" / "rangebar" / "clickhouse" / "schema.sql"
    )
    schema_sql = schema_path.read_text()

    # Extract the CREATE TABLE statement for range_bars
    match = re.search(
        r"(CREATE TABLE IF NOT EXISTS rangebar_cache\.range_bars\s*\(.*?\)\s*"
        r"ENGINE\s*=.*?;)",
        schema_sql,
        re.DOTALL,
    )
    if not match:
        print("  ERROR: Could not extract CREATE TABLE from schema.sql")
        return False

    create_sql = match.group(1).replace(
        "rangebar_cache.range_bars",
        "rangebar_cache.range_bars_v2",
    )
    _ch_command(client, create_sql)
    print("  Created range_bars_v2")

    # Step 3: INSERT SELECT with column mapping
    print("\n=== Step 3: INSERT SELECT (migration) ===")
    v2_columns = _get_all_columns(client, "range_bars_v2")
    old_columns = _get_all_columns(client, "range_bars")

    # Build SELECT clause: map old columns to new names
    select_parts = []
    insert_cols = []
    for col in v2_columns:
        if col == "close_time_ms":
            select_parts.append("timestamp_ms AS close_time_ms")
            insert_cols.append("close_time_ms")
        elif col == "open_time_ms":
            select_parts.append(
                "timestamp_ms - intDiv(duration_us, 1000) AS open_time_ms"
            )
            insert_cols.append("open_time_ms")
        elif col in old_columns:
            select_parts.append(col)
            insert_cols.append(col)
        # else: column has DEFAULT in v2 schema, ClickHouse fills it

    insert_sql = (
        f"INSERT INTO rangebar_cache.range_bars_v2 "
        f"({', '.join(insert_cols)}) "
        f"SELECT {', '.join(select_parts)} "
        f"FROM rangebar_cache.range_bars"
    )

    start = time.monotonic()
    _ch_command(client, insert_sql)
    elapsed = time.monotonic() - start
    print(f"  INSERT SELECT completed in {elapsed:.1f}s")

    # Step 4: Verify row counts
    print("\n=== Step 4: Verify row counts ===")
    v2_count = client.query(
        "SELECT count() FROM rangebar_cache.range_bars_v2"
    ).result_rows[0][0]
    print(f"  Old table: {old_count:,} rows")
    print(f"  New table: {v2_count:,} rows")

    if v2_count != old_count:
        print(f"  ERROR: Row count mismatch! ({v2_count:,} != {old_count:,})")
        print("  Aborting migration. range_bars_v2 left for inspection.")
        return False
    print("  PASS: Row counts match")

    # Step 5: Verify open_time_ms invariant
    print("\n=== Step 5: Verify open_time_ms invariant ===")
    bad_rows = client.query(
        "SELECT count() FROM rangebar_cache.range_bars_v2 FINAL "
        "WHERE open_time_ms > close_time_ms"
    ).result_rows[0][0]
    if bad_rows > 0:
        print(f"  ERROR: {bad_rows:,} rows have open_time_ms > close_time_ms!")
        return False
    print("  PASS: open_time_ms <= close_time_ms for all rows")

    # Step 6: Atomic RENAME
    print("\n=== Step 6: Atomic RENAME ===")
    _ch_command(
        client,
        "RENAME TABLE "
        "rangebar_cache.range_bars TO rangebar_cache.range_bars_old, "
        "rangebar_cache.range_bars_v2 TO rangebar_cache.range_bars"
    )
    print("  DONE: range_bars → range_bars_old, range_bars_v2 → range_bars")

    # Step 7: Post-migration verification
    print("\n=== Step 7: Post-migration verification ===")
    new_count = client.query(
        "SELECT count() FROM rangebar_cache.range_bars"
    ).result_rows[0][0]
    print(f"  New range_bars: {new_count:,} rows")

    has_close_time = _check_new_schema(client)
    print(f"  close_time_ms column: {'YES' if has_close_time else 'NO'}")

    has_legacy = client.query(
        "SELECT name FROM system.columns "
        "WHERE database='rangebar_cache' AND table='range_bars' "
        "AND name='timestamp_ms'"
    )
    has_legacy_col = len(has_legacy.result_rows) > 0
    print(
        f"  Legacy timestamp_ms: "
        f"{'STILL PRESENT (BAD)' if has_legacy_col else 'GONE (GOOD)'}"
    )

    print("\n=== Migration complete ===")
    print("  Keep range_bars_old for 7 days, then DROP TABLE.")
    return True


def rollback(client) -> bool:
    """Rollback: swap range_bars_old back to range_bars."""
    print("\n=== Rollback ===")

    result = client.query(
        "SELECT name FROM system.tables "
        "WHERE database='rangebar_cache' AND name='range_bars_old'"
    )
    if not result.result_rows:
        print("  ERROR: range_bars_old does not exist — nothing to rollback")
        return False

    _ch_command(
        client,
        "RENAME TABLE "
        "rangebar_cache.range_bars TO rangebar_cache.range_bars_v2_broken, "
        "rangebar_cache.range_bars_old TO rangebar_cache.range_bars"
    )
    print("  DONE: Rolled back to range_bars_old")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate range_bars: timestamp_ms → close_time_ms + open_time_ms"
    )
    parser.add_argument(
        "--deploy", action="store_true",
        help="Full lifecycle: stop → deploy → migrate → start",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without making changes",
    )
    parser.add_argument(
        "--rollback", action="store_true",
        help="Rollback: swap range_bars_old back to range_bars",
    )
    parser.add_argument(
        "--host", default="localhost",
        help="ClickHouse host (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=8123,
        help="ClickHouse HTTP port (default: 8123)",
    )
    args = parser.parse_args()

    import clickhouse_connect
    client = clickhouse_connect.get_client(host=args.host, port=args.port)

    if args.rollback:
        return 0 if rollback(client) else 1

    if args.deploy and not args.dry_run:
        _stop_services()
        _deploy_code()

    success = migrate(client, dry_run=args.dry_run)

    if args.deploy and not args.dry_run:
        _start_services()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
