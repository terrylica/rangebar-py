from collections.abc import Iterator
from typing import Any, Literal, overload

import pandas as pd
import polars as pl

@overload
def get_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: (
        int | Literal["micro", "tight", "standard", "medium", "wide", "macro"]
    ) = 250,
    *,
    ouroboros: Literal["year", "month", "week"] = ...,
    include_orphaned_bars: bool = ...,
    materialize: Literal[True] = ...,
    batch_size: int = ...,
    source: Literal["binance", "exness"] = ...,
    market: Literal["spot", "futures-um", "futures-cm", "um", "cm"] = ...,
    validation: Literal["permissive", "strict", "paranoid"] = ...,
    include_incomplete: bool = ...,
    include_microstructure: bool = ...,
    include_exchange_sessions: bool = ...,
    prevent_same_timestamp_close: bool = ...,
    verify_checksum: bool = ...,
    use_cache: bool = ...,
    fetch_if_missing: bool = ...,
    cache_dir: str | None = ...,
    max_memory_mb: int | None = ...,
    inter_bar_lookback_count: int | None = ...,
    inter_bar_lookback_bars: int | None = ...,
) -> pd.DataFrame: ...
@overload
def get_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: (
        int | Literal["micro", "tight", "standard", "medium", "wide", "macro"]
    ) = 250,
    *,
    ouroboros: Literal["year", "month", "week"] = ...,
    include_orphaned_bars: bool = ...,
    materialize: Literal[False],
    batch_size: int = ...,
    source: Literal["binance", "exness"] = ...,
    market: Literal["spot", "futures-um", "futures-cm", "um", "cm"] = ...,
    validation: Literal["permissive", "strict", "paranoid"] = ...,
    include_incomplete: bool = ...,
    include_microstructure: bool = ...,
    include_exchange_sessions: bool = ...,
    prevent_same_timestamp_close: bool = ...,
    verify_checksum: bool = ...,
    use_cache: bool = ...,
    fetch_if_missing: bool = ...,
    cache_dir: str | None = ...,
    max_memory_mb: int | None = ...,
    inter_bar_lookback_count: int | None = ...,
    inter_bar_lookback_bars: int | None = ...,
) -> Iterator[pl.DataFrame]: ...
def get_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: (
        int | Literal["micro", "tight", "standard", "medium", "wide", "macro"]
    ) = 250,
    *,
    ouroboros: Literal["year", "month", "week"] = "year",
    include_orphaned_bars: bool = False,
    materialize: bool = True,
    batch_size: int = 10_000,
    source: Literal["binance", "exness"] = "binance",
    market: Literal["spot", "futures-um", "futures-cm", "um", "cm"] = "spot",
    validation: Literal["permissive", "strict", "paranoid"] = "strict",
    include_incomplete: bool = False,
    include_microstructure: bool = False,
    include_exchange_sessions: bool = False,
    prevent_same_timestamp_close: bool = True,
    verify_checksum: bool = True,
    use_cache: bool = True,
    fetch_if_missing: bool = True,
    cache_dir: str | None = None,
    max_memory_mb: int | None = None,
    inter_bar_lookback_count: int | None = None,
    inter_bar_lookback_bars: int | None = None,
) -> pd.DataFrame | Iterator[pl.DataFrame]:
    ...

def get_range_bars_pandas(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: (
        int | Literal["micro", "tight", "standard", "medium", "wide", "macro"]
    ) = 250,
    **kwargs: Any,  # noqa: ANN401
) -> pd.DataFrame:
    ...
