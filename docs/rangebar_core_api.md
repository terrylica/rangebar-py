# rangebar-core v5.0.0 API Reference

**Research Date**: 2025-11-15
**Purpose**: Document rangebar-core API for PyO3 bindings implementation

## Overview

The rangebar-core crate provides non-lookahead bias range bar construction. This document describes the exact API that our Python bindings must interface with.

## Key Types

### AggTrade (Aggregate Trade)

```rust
pub struct AggTrade {
    pub agg_trade_id: i64,           // Unique aggregate trade ID
    pub price: FixedPoint,           // Price (8 decimal places)
    pub volume: FixedPoint,          // Volume/quantity (8 decimal places)
    pub first_trade_id: i64,         // First individual trade ID
    pub last_trade_id: i64,          // Last individual trade ID
    pub timestamp: i64,              // ⚠️ MICROSECONDS (not milliseconds!)
    pub is_buyer_maker: bool,        // True = sell pressure, False = buy pressure
    pub is_best_match: Option<bool>, // Spot only (None for futures)
}
```

**Critical Details**:

- `timestamp` is in **microseconds** (1/1,000,000 second), NOT milliseconds
- `price` and `volume` use `FixedPoint` type (8 decimal precision)
- Binance aggTrades from CSV are typically in milliseconds → must convert to microseconds
- `is_buyer_maker` indicates order flow direction (microstructure analysis)

### RangeBar (Output)

```rust
pub struct RangeBar {
    // Timestamps (microseconds)
    pub open_time: i64,
    pub close_time: i64,

    // OHLC (FixedPoint = 8 decimals)
    pub open: FixedPoint,
    pub high: FixedPoint,
    pub low: FixedPoint,
    pub close: FixedPoint,

    // Volume and turnover
    pub volume: FixedPoint,
    pub turnover: i128,

    // Trade counting
    pub individual_trade_count: u32,  // Total individual trades
    pub agg_record_count: u32,        // Number of AggTrade records
    pub first_trade_id: i64,
    pub last_trade_id: i64,

    // Data source
    pub data_source: DataSource,

    // Market microstructure (order flow analysis)
    pub buy_volume: FixedPoint,       // Aggressive buy volume
    pub sell_volume: FixedPoint,      // Aggressive sell volume
    pub buy_trade_count: u32,
    pub sell_trade_count: u32,
    pub vwap: FixedPoint,             // Volume-weighted average price
    pub buy_turnover: i128,
    pub sell_turnover: i128,
}
```

**For backtesting.py compatibility**, we need:

- Convert `open_time`/`close_time` from microseconds → Python datetime
- Convert `open/high/low/close/volume` from FixedPoint → f64
- Capitalize column names: `Open`, `High`, `Low`, `Close`, `Volume`

### RangeBarProcessor (Main API)

```rust
pub struct RangeBarProcessor {
    threshold_bps: u32,
    current_bar_state: Option<RangeBarState>,
}

impl RangeBarProcessor {
    /// Create processor
    /// threshold_bps: In 0.1 basis point units (v3.0.0+)
    /// Example: 250 = 25bps = 0.25%
    pub fn new(threshold_bps: u32) -> Result<Self, ProcessingError>

    /// Process batch of trades
    pub fn process_agg_trade_records(
        &mut self,
        agg_trade_records: &[AggTrade]
    ) -> Result<Vec<RangeBar>, ProcessingError>

    /// Process with incomplete bars (analysis mode)
    pub fn process_agg_trade_records_with_incomplete(
        &mut self,
        agg_trade_records: &[AggTrade]
    ) -> Result<Vec<RangeBar>, ProcessingError>
}
```

**Threshold Units**:

- **v3.0.0+ uses 0.1 basis point units**
- `250` = 25 bps = 0.25% price movement
- `100` = 10 bps = 0.10% price movement
- `1` = 0.1 bps = 0.001% (minimum)
- `100,000` = 10,000 bps = 100% (maximum)

### FixedPoint (Precision Type)

```rust
pub struct FixedPoint(pub i64);

impl FixedPoint {
    // 8 decimal places precision
    // Example: 50000.12345678 → FixedPoint(5000012345678)

    pub fn to_f64(&self) -> f64  // Convert to float
    // Also has From<&str> and Display trait
}
```

**Conversion**:

- Internal: `i64` with 8 decimal places (multiply by 100,000,000)
- Example: `42000.5` → `FixedPoint(4200050000000)`
- Python → Rust: Use `FixedPoint::from_str()` or construct from f64
- Rust → Python: Use `.to_f64()` method

### ProcessingError

```rust
pub enum ProcessingError {
    UnsortedTrades { ... },    // Trades not chronologically sorted
    EmptyData,                  // Empty input slice
    InvalidThreshold { threshold_bps: u32 },  // Out of range [1, 100_000]
}
```

**Python Mapping**:

- All errors → `PyValueError` or `PyRuntimeError`
- Include detailed error messages with context

## PyO3 Binding Strategy

### Input: Python → Rust AggTrade

Python input format:

```python
{
    "timestamp": 1704067200000,      # Milliseconds (Binance format)
    "price": 42000.12345678,
    "quantity": 1.5,                 # Note: Python uses "quantity"
    # Optional fields:
    "agg_trade_id": 12345,
    "first_trade_id": 100,
    "last_trade_id": 102,
    "is_buyer_maker": False
}
```

Rust conversion:

```rust
AggTrade {
    agg_trade_id: trade_dict.get("agg_trade_id").unwrap_or(0),
    price: FixedPoint::from_f64(price_f64),
    volume: FixedPoint::from_f64(quantity_f64),
    first_trade_id: trade_dict.get("first_trade_id").unwrap_or(agg_trade_id),
    last_trade_id: trade_dict.get("last_trade_id").unwrap_or(agg_trade_id),
    timestamp: timestamp_ms * 1000,  // ⚠️ CONVERT MS → MICROSECONDS!
    is_buyer_maker: trade_dict.get("is_buyer_maker").unwrap_or(false),
    is_best_match: None,  // Not needed for futures
}
```

### Output: Rust RangeBar → Python dict

Rust output → Python conversion:

```rust
{
    "timestamp": RFC3339_string,     // Convert microseconds → datetime → ISO8601
    "open": bar.open.to_f64(),
    "high": bar.high.to_f64(),
    "low": bar.low.to_f64(),
    "close": bar.close.to_f64(),
    "volume": bar.volume.to_f64(),
    // Optional: Include microstructure data
    "vwap": bar.vwap.to_f64(),
    "buy_volume": bar.buy_volume.to_f64(),
    "sell_volume": bar.sell_volume.to_f64(),
}
```

## Critical Implementation Notes

### 1. Timestamp Conversion

**Binance CSV → rangebar-core:**

- Binance aggTrades CSV: **milliseconds** (13 digits, e.g., `1704067200000`)
- rangebar-core expects: **microseconds** (16 digits, e.g., `1704067200000000`)
- **Conversion**: `timestamp_us = timestamp_ms * 1000`

**rangebar-core → Python:**

- Output: microseconds i64
- Convert to Python datetime: `datetime.fromtimestamp(timestamp_us / 1_000_000)`
- Or RFC3339 string for pandas: `chrono::DateTime::from_timestamp_micros()`

### 2. FixedPoint Precision

- ⚠️ **DO NOT ROUND** - Preserve full 8 decimal precision
- Use `.to_f64()` for Python output
- Use `FixedPoint::from_f64(value)` for Python input
- Never truncate or approximate

### 3. Error Propagation

- rangebar-core returns `Result<Vec<RangeBar>, ProcessingError>`
- **Must propagate errors** - no silent failures
- Map `ProcessingError` → appropriate Python exception
- Include full context in error messages

### 4. Validation Requirements

- Trades **MUST** be sorted by `(timestamp, agg_trade_id)` ascending
- Processor validates automatically and returns `UnsortedTrades` error
- Do NOT sort in Python - raise error and instruct user to pre-sort

## Minimal PyO3 Implementation

```rust
use pyo3::prelude::*;
use rangebar_core::{RangeBarProcessor, AggTrade, RangeBar, FixedPoint, ProcessingError};

#[pyclass]
struct PyRangeBarProcessor {
    processor: RangeBarProcessor,
}

#[pymethods]
impl PyRangeBarProcessor {
    #[new]
    fn new(threshold_bps: u32) -> PyResult<Self> {
        let processor = RangeBarProcessor::new(threshold_bps)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to create processor: {}", e)
            ))?;
        Ok(Self { processor })
    }

    fn process_trades(
        &mut self,
        py: Python,
        trades: Vec<&PyDict>
    ) -> PyResult<Vec<PyObject>> {
        // Convert Python dicts → AggTrade
        let agg_trades = convert_trades(trades)?;

        // Process through rangebar-core
        let bars = self.processor.process_agg_trade_records(&agg_trades)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Processing failed: {}", e)
            ))?;

        // Convert RangeBar → Python dicts
        bars.iter()
            .map(|bar| bar_to_dict(py, bar))
            .collect()
    }
}
```

## Testing Strategy

### Unit Tests (Rust side)

```rust
#[test]
fn test_process_simple_trades() {
    let mut processor = PyRangeBarProcessor::new(250).unwrap();

    let trades = vec![
        create_trade_dict(1704067200000, 42000.0, 1.5),
        create_trade_dict(1704067210000, 42105.0, 2.3),  // +0.25% = breach
    ];

    let bars = processor.process_trades(trades).unwrap();
    assert_eq!(bars.len(), 1);  // One completed bar
}
```

### Integration Tests (Python side)

```python
def test_backtesting_py_format():
    from rangebar import process_trades_to_dataframe

    trades = pd.read_csv("BTCUSDT-aggTrades-2024-01.csv")
    df = process_trades_to_dataframe(trades, threshold_bps=250)

    # Validate backtesting.py compatibility
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert not df.isnull().any().any()
```

## References

- **Crate location**: `/Users/terryli/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/rangebar-core-5.0.0`
- **Upstream repo**: https://github.com/terrylica/rangebar
- **Crate version**: v5.0.0 (crates.io)
- **Key modules**:
  - `src/types.rs` - AggTrade, RangeBar definitions
  - `src/processor.rs` - RangeBarProcessor implementation
  - `src/fixed_point.rs` - FixedPoint arithmetic

## Architectural Decision

**ADR-001: Use rangebar-core as-is, no modifications**

- **Decision**: Import rangebar-core v5.0 from crates.io without forking
- **Rationale**: Zero maintainer burden - upstream crate owner does ZERO work
- **Consequence**: Must adapt to rangebar-core API (microseconds, FixedPoint, threshold units)
- **Trade-off**: Some conversion overhead (ms→μs, FixedPoint→f64) but gains upstream maintenance

**ADR-002: Preserve full precision in conversions**

- **Decision**: Use FixedPoint.to_f64() for all price/volume conversions
- **Rationale**: Backtesting accuracy depends on price precision
- **Consequence**: No rounding, no truncation, no approximation
- **Validation**: Compare roundtrip conversions (Python → Rust → Python)
