# Validation & Data Formats API Reference

**Navigation**: [INDEX.md](./INDEX.md) | [Primary API](./primary-api.md) | [Processing API](./processing-api.md)

---

## Data Formats

### Input: Trade Data

**List of Dicts Format**:

```python
[
    {
        "timestamp": 1704067200000,  # Unix ms (required)
        "price": 42000.0,            # Trade price (required)
        "quantity": 1.5,             # Trade volume (required, or use "volume")
    },
    # ... more trades
]
```

**DataFrame Format**:

```python
pd.DataFrame({
    "timestamp": [1704067200000, 1704067210000, ...],  # Unix ms or datetime
    "price": [42000.0, 42105.0, ...],
    "quantity": [1.5, 2.3, ...],  # or "volume" column
})
```

**Field Requirements**:

- `timestamp`:
  - Must be Unix timestamp in milliseconds (int)
  - Or pandas datetime (auto-converted)
  - Must be monotonically increasing (sorted)
- `price`:
  - Must be positive float
  - Precision preserved to 8 decimal places
- `quantity` (or `volume`):
  - Must be positive float
  - Represents trade volume

---

### Output: Range Bar DataFrame

**Structure**:

```python
                          Open      High       Low     Close  Volume
timestamp
2024-01-01 00:00:10  42000.0  42105.0  42000.0  42105.0    3.8
2024-01-01 00:03:25  42105.0  42210.0  42105.0  42210.0    7.2
...
```

**Properties**:

- **Index**: `pd.DatetimeIndex`
  - Timezone-naive UTC timestamps
  - Monotonically increasing
  - No duplicates
  - Bar closing timestamp
- **Columns**: `["Open", "High", "Low", "Close", "Volume"]`
  - Capitalized (backtesting.py convention)
  - All float64 dtype
  - No NaN values
- **OHLC Invariants**:
  - `High >= max(Open, Close)`
  - `Low <= min(Open, Close)`
  - `Volume > 0`

---

## Error Handling

### ValueError

**Causes**:

- Empty trades list
- Invalid threshold_decimal_bps (<=0)
- Missing required columns/keys
- Negative price/quantity
- Long date range (>30 days) without cache

**Example**:

```python
# Empty trades
process_trades_to_dataframe([])
# ValueError: Trades list is empty

# Invalid threshold
RangeBarProcessor(threshold_decimal_bps=0)
# ValueError: threshold_decimal_bps must be positive

# Missing fields
process_trades_to_dataframe([{"timestamp": 123, "price": 42000}])
# ValueError: Trade 0 missing: {'quantity'}

# Long range without cache
get_range_bars("BTCUSDT", "2024-01-01", "2024-12-31", use_cache=False)
# ValueError: Date range of 365 days requires use_cache=True...
```

---

### RuntimeError

**Causes**:

- Trades not sorted chronologically
- Internal processing error
- Rust-level failure

**Example**:

```python
# Unsorted trades
trades = [
    {"timestamp": 1704067210000, "price": 42105.0, "quantity": 1.0},
    {"timestamp": 1704067200000, "price": 42000.0, "quantity": 1.0},  # Earlier!
]
process_trades_to_dataframe(trades)
# RuntimeError: Processing failed: trades not sorted by timestamp
```

**Fix**: Sort trades before processing:

```python
df = df.sort_values("timestamp")
trades = sorted(trades, key=lambda t: t["timestamp"])
```

---

### KeyError

**Causes**:

- Trade dict missing required keys (`timestamp`, `price`, `quantity`)

**Example**:

```python
trades = [{"timestamp": 123, "price": 42000}]  # Missing 'quantity'
process_trades_to_dataframe(trades)
# KeyError: Trade 0: missing 'quantity'
```

---

## Continuity Validation

For validating bar continuity and detecting gaps:

```python
from rangebar.validation.continuity import validate_continuity, validate_continuity_tiered

# Basic validation
result = validate_continuity(df)
if not result.is_valid:
    print(f"Found {len(result.gaps)} gaps")

# Tiered validation with asset-class-specific thresholds
from rangebar import validate_continuity_tiered, AssetClass

result = validate_continuity_tiered(
    df,
    asset_class=AssetClass.CRYPTO,
    preset="strict",
)
print(f"Tier 1 (critical): {result.tier1_summary.count} gaps")
print(f"Tier 2 (warning): {result.tier2_summary.count} gaps")
```

### Related Types

```python
from rangebar import (
    ContinuityError,       # Exception for continuity violations
    ContinuityWarning,     # Warning for non-critical gaps
    GapTier,               # Enum: TIER1, TIER2, TIER3
    AssetClass,            # Enum: CRYPTO, FOREX, EQUITY
    GapInfo,               # Gap details dataclass
    TieredValidationResult,  # Full validation result
)
```

---

## Microstructure Feature Validation

For v7.0+ microstructure features:

### Tier 1: Smoke Test

```python
from rangebar.validation.tier1 import validate_tier1

df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", include_microstructure=True)
result = validate_tier1(df)

if not result["tier1_passed"]:
    print("Validation failed:", result)
```

### Tier 2: Statistical Validation

```python
from rangebar.validation.tier2 import validate_tier2

# Add forward return for predictive power analysis
df["forward_return"] = df["Close"].shift(-1) / df["Close"] - 1
result = validate_tier2(df)

if result["tier2_passed"]:
    print("Ready for ML training")
```

### Expected Feature Ranges

| Feature                | Expected Range | Interpretation                                        |
| ---------------------- | -------------- | ----------------------------------------------------- |
| `duration_us`          | [0, +inf)      | Microseconds; 0 = instantaneous bar (rapid trades)    |
| `ofi`                  | [-1, 1]        | Order Flow Imbalance; -1 = all sells, +1 = all buys   |
| `vwap_close_deviation` | ~[-1, 1]       | Can exceed +/-1 during gaps; near 0 = close near VWAP |
| `price_impact`         | [0, +inf)      | Price change per unit volume; higher = more impact    |
| `kyle_lambda_proxy`    | (-inf, +inf)   | Market impact coefficient; bounded for typical data   |
| `trade_intensity`      | [0, +inf)      | Trades per second; higher = more active               |
| `volume_per_trade`     | [0, +inf)      | Average trade size; institutional vs retail           |
| `aggression_ratio`     | [0, 100]       | Buy count / sell count (capped at 100)                |
| `aggregation_density`  | [1, +inf)      | Individual trades per agg record; 1 = no aggregation  |
| `turnover_imbalance`   | [-1, 1]        | Buy-sell turnover difference normalized               |

---

**Last Updated**: 2026-02-03
