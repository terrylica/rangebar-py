BINANCE_VISION_AGGTRADES_URL: str

def probe_latest_available_date(
    symbol: str = "BTCUSDT",
    max_lookback: int = 10,
    *,
    timeout: int = 10,
) -> str: ...
