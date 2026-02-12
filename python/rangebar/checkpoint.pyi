from typing import Literal

def populate_cache_resumable(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    threshold_decimal_bps: int = 250,
    force_refresh: bool = False,
    include_microstructure: bool = False,
    ouroboros: Literal["year", "month", "week"] = "year",
    checkpoint_dir: str | None = None,
    notify: bool = True,
    verbose: bool = True,
    inter_bar_lookback_bars: int | None = None,
) -> int:
    ...
