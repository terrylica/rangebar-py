"""Forensics-grade telemetry for the backtesting pipeline.

Emits structured NDJSON events to logs/events.jsonl via the existing
loguru infrastructure in logging.py. Each function logs one event with
a specific component tag for filtering.

Usage
-----
>>> from rangebar.telemetry import BacktestTelemetry
>>>
>>> tel = BacktestTelemetry(symbol="BTCUSDT", threshold_dbps=250)
>>> tel.log_cache_query(hit=True, bar_count=871, query_ms=12.3)
>>> tel.log_bar_delivery(n_bars=871, price_min=92000, price_max=107000)
>>> tel.log_strategy_init("SMA Crossover", params={"fast": 10, "slow": 30})
>>> tel.log_backtest_complete(total_return_pct=-1.97, sharpe=-0.12, ...)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from rangebar.hooks import HookEvent, emit_hook
from rangebar.logging import generate_trace_id, get_logger

if TYPE_CHECKING:
    from loguru import Logger


class BacktestTelemetry:
    """Correlation-aware telemetry for a single backtesting pipeline run.

    All events share the same ``trace_id`` and ``strategy_run_id`` so they
    can be correlated across pipeline stages when querying events.jsonl.
    """

    def __init__(self, symbol: str, threshold_dbps: int) -> None:
        self.symbol = symbol
        self.threshold_dbps = threshold_dbps
        self.trace_id = generate_trace_id("bt")
        self.strategy_run_id = generate_trace_id("sr")
        self._logger = get_logger()

    def _bind(self, component: str, **kwargs: object) -> Logger:
        """Bind common fields to the logger."""
        return self._logger.bind(
            component=component,
            trace_id=self.trace_id,
            strategy_run_id=self.strategy_run_id,
            symbol=self.symbol,
            threshold_dbps=self.threshold_dbps,
            **kwargs,
        )

    def log_cache_query(
        self,
        *,
        hit: bool,
        bar_count: int,
        query_ms: float,
        ch_host: str | None = None,
    ) -> None:
        """Log a ClickHouse cache query result."""
        host = ch_host or os.environ.get("RANGEBAR_CH_HOSTS", "unknown")
        self._bind(
            "cache_query",
            hit=hit,
            bar_count=bar_count,
            query_ms=round(query_ms, 2),
            ch_host=host,
        ).info(
            f"cache_query: {self.symbol}@{self.threshold_dbps} "
            f"{'HIT' if hit else 'MISS'} ({bar_count} bars, {query_ms:.1f}ms)"
        )

    def log_bar_delivery(
        self,
        *,
        n_bars: int,
        price_min: float,
        price_max: float,
        ts_min: str | None = None,
        ts_max: str | None = None,
        columns: list[str] | None = None,
    ) -> None:
        """Log bar DataFrame delivery metadata."""
        self._bind(
            "bar_delivery",
            n_bars=n_bars,
            price_min=round(price_min, 2),
            price_max=round(price_max, 2),
            ts_min=ts_min,
            ts_max=ts_max,
            columns=columns,
        ).info(f"bar_delivery: {n_bars} bars, ${price_min:,.0f}-${price_max:,.0f}")

    def log_strategy_init(
        self,
        strategy_name: str,
        *,
        params: dict[str, Any] | None = None,
        n_bars: int | None = None,
    ) -> None:
        """Log strategy initialization."""
        self._bind(
            "strategy",
            event_type="strategy_init",
            strategy_name=strategy_name,
            strategy_params=params or {},
            n_bars=n_bars,
        ).info(f"strategy_init: {strategy_name} {params}")
        emit_hook(
            HookEvent.STRATEGY_INIT,
            symbol=self.symbol,
            trace_id=self.trace_id,
            strategy_name=strategy_name,
            params=params,
        )

    def log_signal(
        self,
        signal_type: str,
        *,
        bar_index: int,
        price: float,
        indicator_values: dict[str, float] | None = None,
    ) -> None:
        """Log a strategy signal (buy/sell/close)."""
        self._bind(
            "strategy",
            event_type="strategy_signal",
            signal_type=signal_type,
            bar_index=bar_index,
            price=round(price, 2),
            indicator_values=indicator_values or {},
        ).info(f"signal: {signal_type} at bar {bar_index}, price=${price:,.2f}")
        emit_hook(
            HookEvent.STRATEGY_SIGNAL,
            symbol=self.symbol,
            trace_id=self.trace_id,
            signal_type=signal_type,
            bar_index=bar_index,
            price=price,
        )

    def log_trade(
        self,
        *,
        entry_time: str,
        exit_time: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        commission: float | None = None,
        duration: str | None = None,
        is_long: bool = True,
        size: float | None = None,
        tag: str | None = None,
    ) -> None:
        """Log an individual completed trade."""
        self._bind(
            "trade",
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            commission=round(commission, 2) if commission is not None else None,
            duration=duration,
            is_long=is_long,
            size=size,
            tag=tag,
        ).info(
            f"trade: {'LONG' if is_long else 'SHORT'} "
            f"${entry_price:,.2f}→${exit_price:,.2f} "
            f"PnL=${pnl:,.2f} ({pnl_pct:.2%})"
        )
        emit_hook(
            HookEvent.TRADE_COMPLETE,
            symbol=self.symbol,
            trace_id=self.trace_id,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )

    def log_backtest_complete(
        self,
        *,
        total_return_pct: float,
        sharpe: float,
        max_drawdown_pct: float,
        n_trades: int,
        win_rate_pct: float,
        buy_hold_return_pct: float,
        equity_final: float,
        provenance: str | None = None,
    ) -> None:
        """Log aggregate backtest results."""
        self._bind(
            "backtest",
            total_return_pct=round(total_return_pct, 4),
            sharpe=round(sharpe, 4),
            max_drawdown_pct=round(max_drawdown_pct, 4),
            n_trades=n_trades,
            win_rate_pct=round(win_rate_pct, 2),
            buy_hold_return_pct=round(buy_hold_return_pct, 4),
            equity_final=round(equity_final, 2),
            provenance=provenance,
        ).info(
            f"backtest_complete: {n_trades} trades, "
            f"return={total_return_pct:.2f}%, sharpe={sharpe:.4f}, "
            f"max_dd={max_drawdown_pct:.2f}%"
        )
        emit_hook(
            HookEvent.BACKTEST_COMPLETE,
            symbol=self.symbol,
            trace_id=self.trace_id,
            total_return_pct=total_return_pct,
            n_trades=n_trades,
            sharpe=sharpe,
        )

    def log_profiling_start(
        self,
        *,
        profiler: str,
        output_path: str,
        pid: int | None = None,
    ) -> None:
        """Log profiling session start."""
        self._bind(
            "profiling",
            event_type="profiling_start",
            profiler=profiler,
            output_path=output_path,
            target_pid=pid or os.getpid(),
        ).info(f"profiling_start: {profiler} → {output_path}")

    def log_profiling_complete(
        self,
        *,
        output_path: str,
        duration_sec: float,
        flamegraph_path: str | None = None,
    ) -> None:
        """Log profiling session completion."""
        self._bind(
            "profiling",
            event_type="profiling_complete",
            output_path=output_path,
            duration_sec=round(duration_sec, 2),
            flamegraph_path=flamegraph_path,
        ).info(f"profiling_complete: {duration_sec:.1f}s → {output_path}")


__all__ = ["BacktestTelemetry"]
