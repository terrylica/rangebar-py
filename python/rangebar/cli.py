"""Command-line interface for rangebar-py cache management.

This module provides CLI commands for managing the ClickHouse cache,
including status checks, population, and clearing operations.

Usage
-----
After installation, the CLI is available as `rangebar`:

    $ rangebar status BTCUSDT
    $ rangebar populate BTCUSDT --start 2024-01-01 --end 2024-06-30
    $ rangebar clear BTCUSDT --confirm

Or run as a module:

    $ python -m rangebar.cli status BTCUSDT
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime

import click

# Configure logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.version_option(package_name="rangebar")
def cli(verbose: bool) -> None:
    """rangebar-py cache management CLI.

    Manage ClickHouse cache for computed range bars.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument("symbol")
@click.option(
    "--threshold",
    "-t",
    type=int,
    default=250,
    help="Threshold in dbps (default: 250)",
)
def status(symbol: str, threshold: int) -> None:
    """Show cache status for a symbol.

    Example: rangebar status BTCUSDT
    """
    try:
        from .clickhouse import RangeBarCache
    except ImportError as e:
        click.echo(f"Error: ClickHouse support not available: {e}", err=True)
        sys.exit(1)

    try:
        with RangeBarCache() as cache:
            # Get bar count
            count = cache.count_bars(symbol, threshold)

            # Get timestamp range
            oldest = cache.get_oldest_bar_timestamp(symbol, threshold)
            newest = cache.get_newest_bar_timestamp(symbol, threshold)

            click.echo(f"Symbol: {symbol}")
            click.echo(f"Threshold: {threshold} dbps")
            click.echo(f"Cached bars: {count:,}")

            if oldest and newest:
                oldest_dt = datetime.fromtimestamp(oldest / 1000)  # noqa: DTZ006
                newest_dt = datetime.fromtimestamp(newest / 1000)  # noqa: DTZ006
                click.echo(f"Date range: {oldest_dt.date()} to {newest_dt.date()}")
            else:
                click.echo("Date range: N/A (no cached bars)")

    except ConnectionError as e:
        click.echo(f"Error: Cannot connect to ClickHouse: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("symbol")
@click.option("--start", "-s", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", "-e", required=True, help="End date (YYYY-MM-DD)")
@click.option(
    "--threshold",
    "-t",
    type=int,
    default=250,
    help="Threshold in dbps (default: 250)",
)
@click.option(
    "--notify/--no-notify",
    default=True,
    help="Send Telegram notifications (default: on)",
)
def populate(
    symbol: str,
    start: str,
    end: str,
    threshold: int,
    notify: bool,
) -> None:
    """Populate cache for a date range.

    Fetches tick data and computes range bars, storing results in ClickHouse.

    Example: rangebar populate BTCUSDT --start 2024-01-01 --end 2024-06-30
    """
    # Validate dates
    try:
        datetime.strptime(start, "%Y-%m-%d")  # noqa: DTZ007
        datetime.strptime(end, "%Y-%m-%d")  # noqa: DTZ007
    except ValueError:
        click.echo("Error: Invalid date format. Use YYYY-MM-DD", err=True)
        sys.exit(1)

    # Enable Telegram notifications if requested
    if notify:
        try:
            from .notify.telegram import enable_telegram_notifications

            enable_telegram_notifications()
        except ImportError:
            click.echo("Warning: Telegram notifications not available", err=True)

    click.echo(f"Populating cache for {symbol} from {start} to {end}...")
    click.echo(f"Threshold: {threshold} dbps")

    try:
        from . import get_range_bars

        df = get_range_bars(
            symbol,
            start,
            end,
            threshold_decimal_bps=threshold,
            use_cache=True,
            fetch_if_missing=True,
        )

        click.echo(f"Computed {len(df):,} bars")
        click.echo("Bars should now be cached in ClickHouse")

    except (ValueError, RuntimeError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("symbol")
@click.option(
    "--threshold",
    "-t",
    type=int,
    default=None,
    help="Threshold in dbps (default: all thresholds)",
)
@click.option("--confirm", is_flag=True, help="Confirm deletion (required)")
def clear(symbol: str, threshold: int | None, confirm: bool) -> None:
    """Clear cache for a symbol.

    Requires --confirm flag to prevent accidental deletion.

    Example: rangebar clear BTCUSDT --confirm
    """
    if not confirm:
        click.echo("Error: Add --confirm flag to delete cache data", err=True)
        click.echo(f"  rangebar clear {symbol} --confirm")
        sys.exit(1)

    try:
        from .clickhouse import RangeBarCache
    except ImportError as e:
        click.echo(f"Error: ClickHouse support not available: {e}", err=True)
        sys.exit(1)

    try:
        with RangeBarCache() as cache:
            if threshold:
                # Clear specific threshold
                click.echo(f"Clearing cache for {symbol} @ {threshold} dbps...")

                # Get count before clearing
                count = cache.count_bars(symbol, threshold)
                if count == 0:
                    click.echo("No cached bars found")
                    return

                # Delete using timestamp range (all time)
                cache.invalidate_range_bars_by_range(
                    symbol=symbol,
                    threshold_decimal_bps=threshold,
                    start_timestamp_ms=0,
                    end_timestamp_ms=int(datetime.now(tz=UTC).timestamp() * 1000),
                )
                click.echo(f"Cleared {count:,} bars (deletion is async)")
            else:
                click.echo(f"Clearing all cache for {symbol}...")
                # Would need to query all thresholds and clear each
                # For now, just show a message
                click.echo(
                    "Note: Clearing all thresholds not yet implemented. "
                    "Specify --threshold to clear a specific threshold."
                )

    except ConnectionError as e:
        click.echo(f"Error: Cannot connect to ClickHouse: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_symbols() -> None:
    """List all cached symbols and their thresholds."""
    try:
        from .clickhouse import RangeBarCache
    except ImportError as e:
        click.echo(f"Error: ClickHouse support not available: {e}", err=True)
        sys.exit(1)

    try:
        with RangeBarCache() as cache:
            # Query distinct symbol/threshold combinations
            query = """
                SELECT symbol, threshold_decimal_bps, count(*) as bar_count
                FROM rangebar_cache.range_bars FINAL
                GROUP BY symbol, threshold_decimal_bps
                ORDER BY symbol, threshold_decimal_bps
            """
            result = cache.client.query(query)

            if not result.result_rows:
                click.echo("No cached data found")
                return

            click.echo("Cached symbols:")
            click.echo("-" * 50)
            click.echo(f"{'Symbol':<12} {'Threshold':<12} {'Bars':>12}")
            click.echo("-" * 50)

            for row in result.result_rows:
                symbol, threshold, count = row
                click.echo(f"{symbol:<12} {threshold:<12} {count:>12,}")

    except ConnectionError as e:
        click.echo(f"Error: Cannot connect to ClickHouse: {e}", err=True)
        sys.exit(1)


@cli.command()
def test_telegram() -> None:
    """Send a test Telegram notification."""
    try:
        from .notify.telegram import is_configured, send_telegram
    except ImportError:
        click.echo("Error: Telegram module not available", err=True)
        sys.exit(1)

    if not is_configured():
        click.echo(
            "Error: Telegram not configured. "
            "Set RANGEBAR_TELEGRAM_TOKEN environment variable.",
            err=True,
        )
        sys.exit(1)

    success = send_telegram(
        "<b>rangebar-py CLI Test</b>\n\n"
        "This is a test notification from the rangebar CLI.\n"
        "If you see this, Telegram notifications are working correctly."
    )

    if success:
        click.echo("Test notification sent successfully")
    else:
        click.echo("Failed to send test notification", err=True)
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
