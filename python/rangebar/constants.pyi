TIER1_SYMBOLS: tuple[str, ...]
"""18 high-liquidity symbols available on ALL Binance markets.

AAVE, ADA, AVAX, BCH, BNB, BTC, DOGE, ETH, FIL,
LINK, LTC, NEAR, SOL, SUI, UNI, WIF, WLD, XRP
"""

THRESHOLD_DECIMAL_MIN: int
"""Minimum valid threshold: 1 (0.1bps = 0.001%)"""

THRESHOLD_DECIMAL_MAX: int
"""Maximum valid threshold: 100,000 (10,000bps = 100%)"""

THRESHOLD_PRESETS: dict[str, int]
"""Named threshold presets (in 0.1bps units).

- "micro": 10 (1bps = 0.01%) - scalping
- "tight": 50 (5bps = 0.05%) - day trading
- "standard": 100 (10bps = 0.1%) - swing trading
- "medium": 250 (25bps = 0.25%) - default
- "wide": 500 (50bps = 0.5%) - position trading
- "macro": 1000 (100bps = 1%) - long-term
"""

# Issue #59: Inter-bar microstructure features
INTER_BAR_FEATURE_COLUMNS: tuple[str, ...]
"""16 inter-bar microstructure feature column names (Issue #59).

Tier 1 - Core (7 features):
- lookback_trade_count: Trade count in lookback window
- lookback_ofi: Order flow imbalance [-1, 1]
- lookback_duration_us: Lookback window duration (microseconds)
- lookback_intensity: Trade intensity (trades/second)
- lookback_vwap_raw: Volume-weighted average price (raw i64)
- lookback_vwap_position: VWAP position in price range [0, 1]
- lookback_count_imbalance: Trade count imbalance [-1, 1]

Tier 2 - Statistical (5 features):
- lookback_kyle_lambda: Kyle's lambda (price impact)
- lookback_burstiness: Goh-Barabasi burstiness [-1, 1]
- lookback_volume_skew: Volume distribution skewness
- lookback_volume_kurt: Volume distribution kurtosis
- lookback_price_range: Price range / first price [0, +inf)

Tier 3 - Advanced (4 features):
- lookback_kaufman_er: Kaufman efficiency ratio [0, 1]
- lookback_garman_klass_vol: Garman-Klass volatility [0, 1)
- lookback_hurst: Hurst exponent [0, 1]
- lookback_permutation_entropy: Permutation entropy [0, 1]

All inter-bar features are Optional - None when no lookback data available.
"""

# Issue #72: Trade ID range for data integrity verification
TRADE_ID_RANGE_COLUMNS: tuple[str, ...]
"""2 trade ID range columns for data integrity verification (Issue #72).

- first_agg_trade_id: First aggregate trade ID in the bar
- last_agg_trade_id: Last aggregate trade ID in the bar

Enables gap detection: bars[i].first_agg_trade_id == bars[i-1].last_agg_trade_id + 1
"""

# Issue #69: Long range threshold for MEM-013 guard
LONG_RANGE_DAYS: int
"""Maximum days for direct get_range_bars() processing (30).

Date ranges exceeding this limit require populate_cache_resumable() first.
This is MEM-013 guard to prevent OOM on long backfills.
"""
