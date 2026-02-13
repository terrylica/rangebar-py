import pandas as pd

def to_panel_format(
    df: pd.DataFrame,
    symbol: str,
    *,
    feature_prefix: str = "feature",
) -> pd.DataFrame: ...

def get_range_bars_panel(
    symbols: list[str] | str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: int | str = 250,
    *,
    include_microstructure: bool = False,
    ouroboros: str = "year",
    use_cache: bool = True,
    feature_prefix: str = "feature",
) -> pd.DataFrame: ...
