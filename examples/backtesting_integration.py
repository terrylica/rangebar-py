#!/usr/bin/env python3
"""backtesting.py integration example using real Binance BTCUSDT data.

This example demonstrates complete integration with backtesting.py:
1. Fetch real BTCUSDT tick data from Binance (via rangebar cache)
2. Convert to range bars at 250 dbps (0.25%) threshold
3. Define a moving average crossover strategy
4. Run backtest on the quietest month (Sep 2025: ~829 bars)
5. Analyze results with full provenance metadata

Requires: pip install backtesting.py

Data source: Binance Spot BTCUSDT aggregate trades
Period: September 2025 (empirically the quietest month for BTCUSDT @ 250 dbps)
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Check if backtesting.py is installed
try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    from backtesting.test import SMA
except ImportError:
    print("Error: backtesting.py is not installed")
    print("\nInstall with:")
    print("  pip install backtesting.py")
    print("\nOr:")
    print("  uv pip install backtesting.py")
    sys.exit(1)

import rangebar
from rangebar import get_range_bars
from rangebar.telemetry import BacktestTelemetry

# ── Configuration ──────────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
THRESHOLD_DBPS = 250  # 0.25%
START_DATE = "2025-09-01"
END_DATE = "2025-10-01"
FAST_MA = 10
SLOW_MA = 30
INITIAL_CASH = 1_000_000
COMMISSION = 0.002  # 0.2% (Binance spot taker fee)
# ──────────────────────────────────────────────────────────────────────


class RangeBarMACrossover(Strategy):
    """Moving Average Crossover on Range Bars.

    Buy when fast SMA crosses above slow SMA (golden cross).
    Sell when fast SMA crosses below slow SMA (death cross).
    """

    fast = FAST_MA
    slow = SLOW_MA

    def init(self) -> None:
        self.sma_fast = self.I(SMA, self.data.Close, self.fast)
        self.sma_slow = self.I(SMA, self.data.Close, self.slow)

    def next(self) -> None:
        if crossover(self.sma_fast, self.sma_slow):
            if not self.position:
                self.buy()
        elif crossover(self.sma_slow, self.sma_fast) and self.position:
            self.position.close()


def _build_provenance(bars: pd.DataFrame) -> str:
    """Build a provenance string for the HTML output filename and title."""
    n_bars = len(bars)
    price_start = bars["Close"].iloc[0]
    price_end = bars["Close"].iloc[-1]
    return (
        f"{SYMBOL} @ {THRESHOLD_DBPS}dbps | "
        f"{START_DATE} to {END_DATE} | "
        f"SMA({FAST_MA},{SLOW_MA}) | "
        f"{n_bars:,} bars | "
        f"${price_start:,.0f}-${price_end:,.0f} | "
        f"rangebar v{rangebar.__version__}"
    )


def main() -> None:
    """Run backtesting.py integration with real BTCUSDT data."""
    tel = BacktestTelemetry(symbol=SYMBOL, threshold_dbps=THRESHOLD_DBPS)

    print("=" * 70)
    print("rangebar-py: backtesting.py Integration (Real Data)")
    print("=" * 70)

    # ── Step 1: Fetch real range bars ──────────────────────────────────
    print(f"\nFetching {SYMBOL} range bars...")
    print(f"  Threshold:  {THRESHOLD_DBPS} dbps ({THRESHOLD_DBPS / 100:.2f}%)")
    print(f"  Period:     {START_DATE} to {END_DATE}")
    print("  Data source: Binance Spot aggregate trades")

    bars = get_range_bars(
        SYMBOL,
        START_DATE,
        END_DATE,
        threshold_decimal_bps=THRESHOLD_DBPS,
    )

    print(f"  Generated:  {len(bars):,} range bars")
    print(f"  Price range: ${bars['Close'].min():,.2f} - ${bars['Close'].max():,.2f}")

    # ── Telemetry: bar delivery ───────────────────────────────────────
    tel.log_bar_delivery(
        n_bars=len(bars),
        price_min=float(bars["Close"].min()),
        price_max=float(bars["Close"].max()),
        ts_min=str(bars.index[0]),
        ts_max=str(bars.index[-1]),
        columns=list(bars.columns),
    )

    # ── Step 2: Validate OHLCV format ─────────────────────────────────
    print("\nValidating OHLCV format for backtesting.py...")
    assert isinstance(bars.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    assert list(bars.columns) == expected_cols, f"Columns must be {expected_cols}"
    assert not bars.isna().any().any(), "DataFrame must have no NaN values"
    assert (bars["High"] >= bars["Low"]).all(), "High must be >= Low"
    print("  Validation passed")

    # ── Step 3: Run backtest ──────────────────────────────────────────
    provenance = _build_provenance(bars)
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Running Backtest: {provenance}")
    print(sep)

    bt = Backtest(
        bars,
        RangeBarMACrossover,
        cash=INITIAL_CASH,
        commission=COMMISSION,
        exclusive_orders=True,
    )

    stats = bt.run()

    # ── Telemetry: strategy + trades + backtest ───────────────────────
    tel.log_strategy_init(
        "RangeBarMACrossover",
        params={"fast": FAST_MA, "slow": SLOW_MA},
        n_bars=len(bars),
    )

    # Log individual trades from backtesting.py internal DataFrame
    trades_df = stats["_trades"]
    for _, trade in trades_df.iterrows():
        tel.log_trade(
            entry_time=str(trade["EntryTime"]),
            exit_time=str(trade["ExitTime"]),
            entry_price=float(trade["EntryPrice"]),
            exit_price=float(trade["ExitPrice"]),
            pnl=float(trade["PnL"]),
            pnl_pct=float(trade["ReturnPct"]),
            duration=str(trade["Duration"]),
            is_long=float(trade["Size"]) > 0,
            size=float(trade["Size"]),
        )

    tel.log_backtest_complete(
        total_return_pct=float(stats["Return [%]"]),
        sharpe=float(stats["Sharpe Ratio"]),
        max_drawdown_pct=float(stats["Max. Drawdown [%]"]),
        n_trades=int(stats["# Trades"]),
        win_rate_pct=float(stats["Win Rate [%]"]),
        buy_hold_return_pct=float(stats["Buy & Hold Return [%]"]),
        equity_final=float(stats["Equity Final [$]"]),
        provenance=provenance,
    )

    # ── Step 4: Display results ───────────────────────────────────────
    print(f"\n{sep}")
    print("Key Performance Metrics")
    print(sep)
    print(f"  Symbol:              {SYMBOL}")
    print(f"  Threshold:           {THRESHOLD_DBPS} dbps ({THRESHOLD_DBPS / 100:.2f}%)")
    print(f"  Period:              {START_DATE} to {END_DATE}")
    print(f"  Strategy:            SMA Crossover ({FAST_MA}/{SLOW_MA})")
    print(f"  Range Bars:          {len(bars):,}")
    print(f"  Initial Capital:     ${INITIAL_CASH:,.2f}")
    print(f"  Final Equity:        ${float(stats['Equity Final [$]']):,.2f}")
    print(f"  Total Return:        {float(stats['Return [%]']):.2f}%")
    print(f"  Buy & Hold Return:   {float(stats['Buy & Hold Return [%]']):.2f}%")
    print(f"  Sharpe Ratio:        {float(stats['Sharpe Ratio']):.4f}")
    print(f"  Max Drawdown:        {float(stats['Max. Drawdown [%]']):.2f}%")
    print(f"  Number of Trades:    {stats['# Trades']}")
    print(f"  Win Rate:            {float(stats['Win Rate [%]']):.2f}%")
    print(f"  rangebar version:    {rangebar.__version__}")
    print("  Data source:         Binance Spot")

    # ── Step 5: Save plot with provenance ─────────────────────────────
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{SYMBOL}_{THRESHOLD_DBPS}dbps_"
        f"SMA{FAST_MA}_{SLOW_MA}_"
        f"{START_DATE}_{END_DATE}_"
        f"{ts}.html"
    )
    output_path = output_dir / filename

    print(f"\n{sep}")
    print("Visualization")
    print(sep)

    try:
        print(f"  Saving plot to: {output_path}")
        bt.plot(
            filename=str(output_path),
            resample=False,  # Don't resample - show range bars as-is
            superimpose=False,  # Range bars have irregular intervals
            plot_drawdown=True,
            open_browser=True,
        )
        print(f"  Plot saved: {output_path.name}")
    except (OSError, ValueError, RuntimeError) as e:
        print(f"  Could not save plot: {e}")
        print("  (This is normal in headless environments)")

    # Save stats to CSV with provenance
    stats_path = output_dir / filename.replace(".html", "_stats.csv")
    stats_series = pd.Series(
        {
            "symbol": SYMBOL,
            "threshold_dbps": THRESHOLD_DBPS,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "fast_ma": FAST_MA,
            "slow_ma": SLOW_MA,
            "n_bars": len(bars),
            "rangebar_version": rangebar.__version__,
            "data_source": "Binance Spot",
            **{k: v for k, v in stats.items() if not isinstance(v, pd.Series)},
        }
    )
    stats_series.to_csv(stats_path)
    print(f"  Stats saved: {stats_path.name}")

    print(f"\n{sep}")
    print("backtesting.py integration completed successfully!")
    print(sep)


if __name__ == "__main__":
    main()
