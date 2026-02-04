# Primary API Reference

**Navigation**: [INDEX.md](./INDEX.md) | [Cache API](./cache-api.md) | [Processing API](./processing-api.md)

---

## get_range_bars()

**The recommended single entry point for all range bar generation.**

```python
def get_range_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold_decimal_bps: int | str = 250,
    *,
    ouroboros: Literal["year", "month", "week"] = "week",
    source: str = "binance",
    market: str = "spot",
    include_microstructure: bool = False,
    include_exchange_sessions: bool = False,
    include_orphaned_bars: bool = False,
    verify_checksum: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame
```

### Parameters

- **symbol**: Trading symbol (e.g., "BTCUSDT", "ETHUSDT")
- **start_date**: Start date in YYYY-MM-DD format
- **end_date**: End date in YYYY-MM-DD format
- **threshold_decimal_bps**: Threshold in decimal basis points or preset name (default: 250)
  - Integer: `250` = 25bps = 0.25%
  - Preset: `"micro"`, `"tight"`, `"standard"`, `"medium"`, `"wide"`, `"macro"`
- **ouroboros**: Reset mode for reproducible bar construction (default: `"week"`)
  - `"week"`: Reset at Sunday 00:00:00 UTC (smallest granularity, default)
  - `"month"`: Reset at 1st of each month 00:00:00 UTC
  - `"year"`: Reset at January 1st 00:00:00 UTC
- **source**: Data source - `"binance"` or `"exness"` (default: `"binance"`)
- **market**: Market type - `"spot"`, `"futures-um"`, `"futures-cm"` (default: `"spot"`)
- **include_microstructure**: Include vwap, buy_volume, sell_volume (default: False)
- **include_exchange_sessions**: Include exchange session flags (default: False). See [Exchange Sessions](#exchange-sessions) below.
- **include_orphaned_bars**: Include incomplete bars at ouroboros boundaries (default: False)
- **verify_checksum**: Verify SHA-256 checksum of downloaded data (default: True). Enabled by default for data integrity. Set to False for faster downloads when integrity is verified elsewhere. (Issue #43)
- **use_cache**: Cache tick data locally (default: True)

### Returns

- **pd.DataFrame**: OHLCV DataFrame with DatetimeIndex and columns `Open`, `High`, `Low`, `Close`, `Volume`

### Example

```python
from rangebar import get_range_bars

# Basic usage
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")

# Using preset threshold
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", threshold_decimal_bps="tight")

# Binance USD-M Futures
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-03-31", market="futures-um")

# With microstructure data
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", include_microstructure=True)

# With year-based ouroboros (reproducible across researchers)
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-12-31", ouroboros="year")

# With exchange session flags
df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", include_exchange_sessions=True)
# Adds: exchange_session_sydney, exchange_session_tokyo, exchange_session_london, exchange_session_newyork
```

### Long Date Ranges (>30 days)

Date ranges > 30 days require the cache workflow to prevent OOM:

```python
from rangebar import populate_cache_resumable, get_range_bars

# Step 1: Populate cache (resumable, memory-safe)
populate_cache_resumable("BTCUSDT", "2019-01-01", "2025-12-31")

# Step 2: Read from cache
df = get_range_bars("BTCUSDT", "2019-01-01", "2025-12-31")
```

See [cache-api.md](./cache-api.md) for details.

---

## get_n_range_bars()

**Get exactly N range bars - useful for ML training, walk-forward optimization, and consistent backtest comparisons.**

Unlike `get_range_bars()` which uses date bounds (producing variable bar counts), this function returns a deterministic number of bars.

```python
def get_n_range_bars(
    symbol: str,
    n_bars: int,
    threshold_decimal_bps: int | str = 250,
    *,
    end_date: str | None = None,
    source: str = "binance",
    market: str = "spot",
    include_microstructure: bool = False,
    use_cache: bool = True,
    fetch_if_missing: bool = True,
    max_lookback_days: int = 90,
    warn_if_fewer: bool = True,
    cache_dir: str | None = None,
) -> pd.DataFrame
```

### Parameters

- **symbol**: Trading symbol (e.g., "BTCUSDT")
- **n_bars**: Number of bars to retrieve (must be > 0)
- **threshold_decimal_bps**: Threshold in decimal basis points or preset name (default: 250)
- **end_date**: End date in YYYY-MM-DD format, or None for most recent data
- **source**: Data source - `"binance"` or `"exness"` (default: `"binance"`)
- **market**: Market type - `"spot"`, `"futures-um"`, `"futures-cm"` (default: `"spot"`)
- **include_microstructure**: Include vwap, buy_volume, sell_volume (default: False)
- **use_cache**: Use ClickHouse cache for bar retrieval/storage (default: True)
- **fetch_if_missing**: Fetch and process new data if cache doesn't have enough bars (default: True)
- **max_lookback_days**: Safety limit - maximum days to look back when fetching (default: 90)
- **warn_if_fewer**: Emit UserWarning if returning fewer bars than requested (default: True)
- **cache_dir**: Custom cache directory for tick data (default: platform-specific)

### Returns

- **pd.DataFrame**: OHLCV DataFrame with exactly `n_bars` rows (or fewer if insufficient data), sorted chronologically (oldest first)

### Cache Behavior

- **Fast path**: If ClickHouse cache has >= n_bars, returns immediately (~50ms)
- **Slow path**: If cache has < n_bars and `fetch_if_missing=True`, fetches additional data, computes bars, stores in cache, returns

### Example

```python
from rangebar import get_n_range_bars

# Get last 10,000 bars for ML training
df = get_n_range_bars("BTCUSDT", n_bars=10000)
assert len(df) == 10000

# Get 5,000 bars ending at specific date for walk-forward
df = get_n_range_bars("BTCUSDT", n_bars=5000, end_date="2024-06-01")

# With safety limit (won't fetch more than 30 days of data)
df = get_n_range_bars("BTCUSDT", n_bars=1000, max_lookback_days=30)

# Using preset threshold
df = get_n_range_bars("BTCUSDT", n_bars=500, threshold_decimal_bps="tight")
```

### Raises

- **ValueError**: If `n_bars <= 0`, invalid threshold, or invalid date format
- **RuntimeError**: If ClickHouse not available when `use_cache=True`, or data fetching failed

---

## Exchange Sessions

When `include_exchange_sessions=True`, the output includes boolean columns indicating which traditional exchange market sessions were active at each bar's close time:

| Column                     | Exchange | Local Hours           | Notes                          |
| -------------------------- | -------- | --------------------- | ------------------------------ |
| `exchange_session_sydney`  | ASX      | 10:00-16:00 AEDT/AEST | Australian Securities Exchange |
| `exchange_session_tokyo`   | TSE      | 09:00-15:00 JST       | Tokyo Stock Exchange           |
| `exchange_session_london`  | LSE      | 08:00-17:00 GMT/BST   | London Stock Exchange          |
| `exchange_session_newyork` | NYSE     | 10:00-16:00 EST/EDT   | New York Stock Exchange        |

**Use Cases**:

- Analyze crypto volatility during traditional market hours
- Identify session overlaps (e.g., London/NY overlap)
- Feature engineering for ML models

**Example**:

```python
df = get_range_bars("BTCUSDT", "2024-01-15", "2024-01-16", include_exchange_sessions=True)

# Filter bars during London session only
london_bars = df[df["exchange_session_london"]]

# Find bars during London/NY overlap
overlap_bars = df[df["exchange_session_london"] & df["exchange_session_newyork"]]
```

**Note**: Session detection uses `zoneinfo` for DST-aware timezone conversion. Session boundaries are hour-granularity approximations of actual exchange trading hours.

---

## Ouroboros: Reproducible Range Bar Construction

The **Ouroboros** feature (named after the serpent eating its tail, representing cyclical boundaries) enables reproducible range bar construction by resetting processor state at deterministic boundaries.

### Why Ouroboros?

| Problem                    | How Ouroboros Helps                                    |
| -------------------------- | ------------------------------------------------------ |
| **Reproducibility**        | Two researchers starting from Jan 1 get identical bars |
| **Cache granularity**      | Entire years/months/weeks can be cached independently  |
| **Cross-study comparison** | Standardized starting points for academic/research use |

### Ouroboros Modes

| Mode      | Boundary                  | Use Case                                         |
| --------- | ------------------------- | ------------------------------------------------ |
| `"week"`  | Sunday 00:00:00 UTC       | Default, smallest granularity                    |
| `"month"` | 1st of month 00:00:00 UTC | Monthly analysis                                 |
| `"year"`  | January 1 00:00:00 UTC    | Annual reports, cross-researcher reproducibility |

### Example: Year-Based Reproducibility

```python
from rangebar import get_range_bars

# Two researchers with the same parameters get identical results
df1 = get_range_bars("BTCUSDT", "2024-01-01", "2024-12-31", ouroboros="year")
df2 = get_range_bars("BTCUSDT", "2024-01-01", "2024-12-31", ouroboros="year")
assert df1.equals(df2)  # Guaranteed reproducibility!
```

### Orphaned Bars

At ouroboros boundaries, incomplete bars are "orphaned" (reset). You can include these for analysis:

```python
df = get_range_bars(
    "BTCUSDT",
    "2023-12-01",
    "2024-01-31",
    ouroboros="year",
    include_orphaned_bars=True,
)

# Orphaned bars have metadata
orphans = df[df.get("is_orphan", False)]
print(f"Orphaned bars at year boundary: {len(orphans)}")
```

### Programmatic Boundary Access

```python
from rangebar import get_ouroboros_boundaries
from datetime import date

# Get all week boundaries in a date range
boundaries = get_ouroboros_boundaries(
    start=date(2024, 1, 1),
    end=date(2024, 1, 31),
    mode="week",
)
# [datetime(2024, 1, 7), datetime(2024, 1, 14), datetime(2024, 1, 21), datetime(2024, 1, 28)]
```

---

**Last Updated**: 2026-02-03
