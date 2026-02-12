from typing import Literal

import pandas as pd

def get_n_range_bars(
    symbol: str,
    n_bars: int,
    threshold_decimal_bps: (
        int | Literal["micro", "tight", "standard", "medium", "wide", "macro"]
    ) = 250,
    *,
    end_date: str | None = None,
    source: Literal["binance", "exness"] = "binance",
    market: Literal["spot", "futures-um", "futures-cm", "um", "cm"] = "spot",
    include_microstructure: bool = False,
    use_cache: bool = True,
    fetch_if_missing: bool = True,
    max_lookback_days: int = 90,
    warn_if_fewer: bool = True,
    validate_on_return: bool = False,
    continuity_action: Literal["warn", "raise", "log"] = "warn",
    continuity_tolerance_pct: float | None = None,
    chunk_size: int = 100_000,
    cache_dir: str | None = None,
    inter_bar_lookback_bars: int | None = None,
) -> pd.DataFrame:
    ...
