# Crates Workspace

**Parent**: [/CLAUDE.md](/CLAUDE.md) | **Architecture**: [/docs/ARCHITECTURE.md](/docs/ARCHITECTURE.md)

8-crate modular Rust workspace. All crates have `publish = false` (PyPI-only distribution).

---

## Quick Reference

| Crate                | Purpose         | Key Types                                     |
| -------------------- | --------------- | --------------------------------------------- |
| `rangebar-core`      | Core algorithm  | `RangeBarProcessor`, `RangeBar`, `FixedPoint` |
| `rangebar-providers` | Data sources    | `HistoricalDataLoader`, `ExnessFetcher`       |
| `rangebar-io`        | I/O operations  | Polars integration                            |
| `rangebar-streaming` | Real-time       | `StreamingProcessor`                          |
| `rangebar-batch`     | Batch analytics | `BatchAnalysisEngine`                         |
| `rangebar-config`    | Configuration   | `Settings`                                    |
| `rangebar-cli`       | CLI tools       | (disabled for PyPI)                           |
| `rangebar`           | Meta-crate      | v4.0 compatibility                            |

---

## Dependency Graph

```
                    rangebar (meta-crate)
                           │
     ┌─────────────────────┼─────────────────────┐
     │                     │                     │
rangebar-cli         rangebar-batch      rangebar-streaming
     │                     │                     │
     └──────────┬──────────┴──────────┬──────────┘
                │                     │
          rangebar-io          rangebar-providers
                │                     │
                └──────────┬──────────┘
                           │
                     rangebar-core
                           │
                     rangebar-config
```

---

## Microstructure Features (v7.0)

**Location**: `rangebar-core/src/processor.rs`

10 features computed in Rust during bar construction:

| #   | Feature                | Formula                                     | Range        |
| --- | ---------------------- | ------------------------------------------- | ------------ |
| 1   | `duration_us`          | (close_time - open_time) \* 1000            | [0, +inf)    |
| 2   | `ofi`                  | (buy_vol - sell_vol) / total                | [-1, 1]      |
| 3   | `vwap_close_deviation` | (close - vwap) / (high - low)               | ~[-1, 1]     |
| 4   | `price_impact`         | abs(close - open) / volume                  | [0, +inf)    |
| 5   | `kyle_lambda_proxy`    | ((close-open)/open) / (imbalance/total_vol) | (-inf, +inf) |
| 6   | `trade_intensity`      | trade_count / duration_sec                  | [0, +inf)    |
| 7   | `volume_per_trade`     | volume / trade_count                        | [0, +inf)    |
| 8   | `aggression_ratio`     | buy_count / sell_count                      | [0, 100]     |
| 9   | `aggregation_density`  | individual_count / agg_count                | [1, +inf)    |
| 10  | `turnover_imbalance`   | (buy_turn - sell_turn) / volume             | [-1, 1]      |

**Edge cases**: Division by zero returns 0.0 (no information).

**Academic backing**: Kyle (1985), Amihud (2002), Easley et al. (2012).

---

## Crate Details

### Layer 0: Foundation

#### rangebar-config

- Dependencies: `config`, `serde`
- Public API: `Settings::load()`, `Settings::default()`

#### rangebar-core

Core algorithm with minimal dependencies.

- Dependencies: `chrono`, `serde`, `serde_json`, `thiserror`, `ahash`
- Features:
  - `test-utils` - CSV loading for tests
  - `python` - PyO3 type exports
  - `api` - utoipa OpenAPI schemas

**Key types**:

- `AggTrade` - Input trade data
- `RangeBar` - Output bar with OHLCV + microstructure
- `FixedPoint` - 8-decimal fixed-point arithmetic
- `RangeBarProcessor` - Stateful processor

**Key properties**:

- Non-lookahead breach detection
- Temporal integrity (breach tick in closing bar)
- Streaming-friendly (maintains state between calls)

#### Range Bar Construction Algorithm

How a range bar is built, step by step:

1. **First trade arrives** → Opens a new bar. `open = high = low = close = trade.price`. Thresholds computed: `upper = open * (1 + threshold)`, `lower = open * (1 - threshold)`.

2. **Subsequent trades arrive** → `update_with_trade()`: updates `close`, extends `high`/`low`, accumulates `volume`, `turnover`, `buy_volume`/`sell_volume`, `trade_count`, `vwap`.

3. **Breach detected** → Trade price exceeds `upper` or `lower` threshold AND timestamp differs from `open_time` (Issue #36 timestamp gate). The breaching trade is **included in the closing bar** (its price/volume updates the bar), then the bar is finalized with `compute_microstructure_features()`.

4. **After breach** → `defer_open = true`. The processor holds **no bar state**. The breaching trade does NOT open the next bar (Issue #46).

5. **Next trade after breach** → Because `defer_open == true`, this trade opens a fresh bar. `defer_open` resets to `false`.

**Key invariants**:

| Rule                                   | Detail                                                                              |
| -------------------------------------- | ----------------------------------------------------------------------------------- |
| Breaching trade belongs to closing bar | Its OHLCV contribution is in the bar that closes                                    |
| Next trade opens the new bar           | NOT the breaching trade (Issue #46, `defer_open`)                                   |
| `bar.timestamp` = `close_time`         | Downstream consumers (pandas DatetimeIndex) use close time                          |
| `bar.open_time` and `bar.close_time`   | Milliseconds UTC; close_time >= open_time always                                    |
| Timestamp gate (Issue #36)             | Bar cannot close on same timestamp as it opened                                     |
| Batch/streaming parity                 | `process_agg_trade_records()` and `process_single_trade()` produce identical output |
| Checkpoint continuity                  | `defer_open` persists in checkpoints for cross-file processing                      |
| Ouroboros reset                        | `reset_at_ouroboros()` clears bar state and resets `defer_open`                     |

### Layer 1: Data Access

#### rangebar-providers

- Depends on: `rangebar-core`
- Features:
  - `binance` (default) - Spot, UM futures, CM futures
  - `exness` - 10 forex pairs
  - `all-providers` - Enable all

**Binance types**: `HistoricalDataLoader`, `BinanceMarket`
**Exness types**: `ExnessFetcher`, `ExnessInstrument`

#### rangebar-io

- Depends on: `rangebar-core`
- Features:
  - `parquet` - Enables Polars
  - `all`

Export formats: CSV, Parquet, Arrow IPC

### Layer 2: Engines

#### rangebar-streaming

- Depends on: `rangebar-core`, `rangebar-providers` (optional)
- Features: `binance-integration`, `stats`, `indicators`, `all`
- Bounded memory, circuit breaker pattern

**Types**: `StreamingProcessor`, `StreamingConfig`

#### rangebar-batch

- Depends on: `rangebar-core`, `rangebar-io`
- Rayon parallel processing

**Types**: `BatchAnalysisEngine`, `BatchConfig`, `AnalysisReport`

### Layer 3: Applications

#### rangebar-cli (disabled)

6 CLI binaries preserved as source, not built for PyPI:

- `tier1-symbol-discovery`
- `rangebar-analyze`
- `data-structure-validator`
- `spot-tier1-processor`
- `polars-benchmark`
- `temporal-integrity-validator`

#### rangebar (meta-crate)

Re-exports for v4.0 backward compatibility.

---

## Development Commands

```bash
# Test specific crate
cargo nextest run -p rangebar-core

# Test with features
cargo nextest run -p rangebar-core --features test-utils

# Test all (excludes PyO3 crate)
cargo nextest run --workspace --exclude rangebar-py

# Build with features
cargo build -p rangebar-providers --features exness
cargo build -p rangebar-core --features test-utils
```

---

## Common Patterns

### Adding a New Feature to rangebar-core

1. Add field to `RangeBar` struct in `types.rs`
2. Update `compute_microstructure_features()` in `processor.rs`
3. Add tests in `processor.rs` or dedicated test file
4. Update PyO3 bindings in `/src/lib.rs`
5. Update Python types in `/python/rangebar/__init__.pyi`
6. Update ClickHouse schema if caching

### Feature Flags

```toml
# Enable in Cargo.toml
[dependencies]
rangebar-core = { path = "../rangebar-core", features = ["test-utils"] }

# Or via CLI
cargo build -p rangebar-core --features test-utils,python
```

---

## Testing

### Unit Tests

```bash
cargo nextest run -p rangebar-core
```

### Property-Based Tests

Uses `proptest` for invariant checking:

```rust
proptest! {
    #[test]
    fn ofi_always_bounded(buy in 0.0f64..1e12, sell in 0.0f64..1e12) {
        let total = buy + sell;
        if total > f64::EPSILON {
            let ofi = (buy - sell) / total;
            prop_assert!(ofi >= -1.0 && ofi <= 1.0);
        }
    }
}
```

### Integration Tests

```bash
# Tests in crates/rangebar/tests/
cargo nextest run -p rangebar --features full
```

---

## Critical Implementation Details

### Fixed-Point Arithmetic

The rangebar crate uses **8-decimal fixed-point arithmetic** to avoid floating-point errors:

```rust
// Convert rangebar FixedPoint to f64 for Python
let price_f64 = bar.open.to_f64();
```

**Warning**: Do NOT round or truncate - preserve full precision.

### Timestamp Handling

**Rust Side** (`src/lib.rs`):

```rust
// rangebar uses i64 milliseconds
let timestamp_ms: i64 = bar.timestamp_ms;

// Convert to RFC3339 string for Python
let timestamp = chrono::DateTime::from_timestamp_millis(timestamp_ms)
    .unwrap()
    .to_rfc3339();
```

**Python Side** (`python/rangebar/__init__.py`):

```python
# Parse timestamp string to DatetimeIndex
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### Error Handling (PyO3)

Map Rust errors to Python exceptions:

```rust
let processor = RangeBarProcessor::new(threshold_decimal_bps)
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
        format!("Failed to create processor: {}", e)
    ))?;
```

**Python exception types**:

- `ValueError`: Invalid input (negative threshold, missing fields)
- `RuntimeError`: Processing errors (unsorted trades, internal failures)
- `KeyError`: Missing required dictionary keys

---

## Related

- [/CLAUDE.md](/CLAUDE.md) - Project hub
- [/docs/ARCHITECTURE.md](/docs/ARCHITECTURE.md) - System design
- [/src/lib.rs](/src/lib.rs) - PyO3 bindings
- [/python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md) - Python layer
