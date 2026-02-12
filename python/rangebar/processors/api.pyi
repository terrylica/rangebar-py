import pandas as pd
import polars as pl

def process_trades_polars(
    trades: pl.DataFrame | pl.LazyFrame,
    threshold_decimal_bps: int = 250,
    *,
    symbol: str | None = None,
    include_microstructure: bool = False,
) -> pd.DataFrame:
    ...

def process_trades_to_dataframe(
    trades: list[dict] | pd.DataFrame,
    threshold_decimal_bps: int = 250,
    include_microstructure: bool = False,
) -> pd.DataFrame:
    ...
