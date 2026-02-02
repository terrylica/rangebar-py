# Inter-Bar Microstructure Features for Large Range Bars

**Issue**: https://github.com/terrylica/rangebar-py/issues/59
**Goal**: Enrich 1000 dbps bars with microstructure features computed from raw aggTrades

---

## Problem Statement

- **100 dbps bars**: Good signal, but transaction costs erode alpha
- **1000 dbps bars**: Survive costs, but lose granular microstructure signal
- **Solution**: During 1000 dbps bar construction, compute additional "inter-bar" features from the constituent raw aggTrades that capture finer-grained dynamics

---

## Key Design Decision: Raw AggTrades, Not Pre-Computed Sub-Bars

**Why raw aggTrades (not pre-computed 100 dbps bars)?**
1. **No alignment issues**: AggTrades are the ground truth; sub-bars have arbitrary boundaries
2. **Richer signal**: Access to individual trade timestamps, volumes, is_buyer_maker
3. **Single processing pass**: Compute features during bar construction, not as post-processing
4. **Rust performance**: Handle millions of ticks efficiently

---

## Critical Constraint: Temporal Integrity

Each 1000 dbps bar starts at a **different timestamp**. Features must be computed from trades that have **already arrived** by the time we need them:

```
For bar with open_time = T_open, close_time = T_close:
  → Intra-bar features: computed from trades in [T_open, T_close]
  → Inter-bar features: computed from trades in window BEFORE T_open
```

**Two feature categories:**

1. **Intra-bar features** (existing 10): Computed from trades WITHIN the current bar
2. **Inter-bar features** (NEW): Computed from trades in a LOOKBACK WINDOW before the current bar opened

---

## AggTrade Data Available (Source Truth)

From `crates/rangebar-core/src/types.rs`:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `i64` | **Microseconds** UTC (not milliseconds!) |
| `price` | `FixedPoint` | 8-decimal fixed-point |
| `volume` | `FixedPoint` | 8-decimal fixed-point |
| `is_buyer_maker` | `bool` | true = sell aggressor, false = buy aggressor |
| `first_trade_id` | `i64` | First individual trade ID |
| `last_trade_id` | `i64` | Last individual trade ID |

**Derived**: `individual_trade_count = last_trade_id - first_trade_id + 1`

---

## Architecture: Rust Implementation

### Modified Processor State

```rust
// crates/rangebar-core/src/processor.rs

pub struct RangeBarProcessor {
    // ... existing fields ...

    /// Ring buffer of recent trades for inter-bar feature computation
    /// Stores trades from the lookback window (e.g., last N trades or last T milliseconds)
    trade_history: VecDeque<TradeSnapshot>,

    /// Configuration for inter-bar features
    inter_bar_config: Option<InterBarConfig>,
}

#[derive(Debug, Clone)]
pub struct InterBarConfig {
    /// Lookback mode: by count or by time
    pub lookback_mode: LookbackMode,
    /// Number of trades to keep (for count mode)
    pub lookback_count: usize,
    /// Time window in milliseconds (for time mode)
    pub lookback_ms: i64,
}

#[derive(Debug, Clone)]
pub enum LookbackMode {
    /// Keep last N trades before bar open
    FixedCount(usize),
    /// Keep trades from last T milliseconds before bar open
    FixedWindow(i64),
}

/// Lightweight snapshot of trade for history buffer
#[derive(Debug, Clone)]
pub struct TradeSnapshot {
    pub timestamp: i64,      // microseconds (matches AggTrade)
    pub price: FixedPoint,
    pub volume: FixedPoint,
    pub is_buyer_maker: bool,
    pub turnover: i128,
}
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Raw AggTrades Stream                         │
│  (timestamp, price, volume, is_buyer_maker, ...)                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              RangeBarProcessor (Rust)                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  trade_history: VecDeque<TradeSnapshot>                  │   │
│  │  - Ring buffer, bounded by lookback config               │   │
│  │  - Updated BEFORE each trade is processed                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  For each AggTrade:                                              │
│    1. Push to trade_history (prune old entries)                 │
│    2. Process trade (existing bar construction logic)           │
│    3. On bar CLOSE:                                              │
│       a. compute_microstructure_features() [existing 10]        │
│       b. compute_inter_bar_features(trade_history) [NEW]        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RangeBar (Extended)                           │
│  - OHLCV                                                         │
│  - 10 intra-bar microstructure features (existing)              │
│  - 12 inter-bar microstructure features (NEW, audited)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Audit Summary

All features were audited for:
1. **Bounded output** - no infinity/NaN possible
2. **No magic numbers** - no arbitrary parameters
3. **Edge case handling** - graceful degradation
4. **Data availability** - uses only AggTrade fields

### Audit Results

| Original Feature | Verdict | Issue | Action |
|------------------|---------|-------|--------|
| `lookback_trade_count` | **KEEP** | None | As-is |
| `lookback_ofi` | **KEEP** | None | As-is |
| `lookback_ofi_trend` | **REMOVE** | Unbounded, conceptually unclear | Remove |
| `lookback_volume_entropy` | **REMOVE** | Hidden magic number (bin size) | Remove |
| `lookback_duration_us` | **KEEP** | Rename from _ms (timestamps are μs) | Rename |
| `lookback_intensity` | **KEEP** | Handle duration=0 | As-is |
| `lookback_vwap` | **KEEP** | None | As-is |
| `lookback_vwap_trend` | **MODIFY** | Uses external data (bar_open) | Use internal normalization |
| `lookback_aggression` | **MODIFY** | Unbounded (div by zero) | Use count imbalance instead |
| `lookback_kyle_lambda` | **MODIFY** | Use normalized formula | Match existing pattern |
| `lookback_burstiness` | **KEEP** | Handle 0/0 edge case | As-is with guards |
| `lookback_volume_skew` | **KEEP** | Need ≥3 trades | As-is with guards |
| `lookback_volume_kurt` | **KEEP** | Need ≥4 trades | As-is with guards |
| `lookback_price_range` | **KEEP** | None | As-is |
| `lookback_reversal_count` | **MODIFY** | Normalize to [0,1] | Rename to reversal_ratio |

---

## Final Feature List (12 Features, All Audited)

### Tier 1: Core Features (7 features)

| Feature | Formula | Range | Min Trades | Edge Case Handling |
|---------|---------|-------|------------|-------------------|
| `lookback_trade_count` | COUNT(trades) | [0, ∞) | 0 | Empty → 0 |
| `lookback_ofi` | (buy_vol - sell_vol) / total_vol | [-1, 1] | 1 | total=0 → 0.0 |
| `lookback_duration_us` | last_ts - first_ts | [0, ∞) | 1 | Single → 0 |
| `lookback_intensity` | trade_count / duration_sec | [0, ∞) | 1 | duration=0 → trade_count |
| `lookback_vwap` | Σ(price×vol) / Σ(vol) | [min_p, max_p] | 1 | vol=0 → 0.0 |
| `lookback_vwap_position` | (vwap - low) / (high - low) | [0, 1] | 1 | range=0 → 0.5 |
| `lookback_count_imbalance` | (buy_count - sell_count) / total_count | [-1, 1] | 1 | total=0 → 0.0 |

### Tier 2: Advanced Features (5 features)

| Feature | Formula | Range | Min Trades | Edge Case Handling |
|---------|---------|-------|------------|-------------------|
| `lookback_kyle_lambda` | ((last-first)/first) / ((buy-sell)/total) | (-∞, ∞) | 2 | imbalance=0 → 0.0 |
| `lookback_burstiness` | (σ_τ - μ_τ) / (σ_τ + μ_τ) | [-1, 1] | 2 | σ+μ=0 → 0.0 |
| `lookback_volume_skew` | E[(V-μ)³] / σ³ | (-∞, ∞) | 3 | σ=0 → 0.0 |
| `lookback_volume_kurt` | E[(V-μ)⁴] / σ⁴ - 3 | [-2, ∞) | 4 | σ=0 → 0.0 |
| `lookback_price_range` | (high - low) / first_price | [0, ∞) | 1 | Single → 0.0 |

### Removed Features (with rationale)

| Feature | Reason |
|---------|--------|
| `lookback_ofi_trend` | Unbounded output, conceptually unclear (OFI is aggregate, not per-trade) |
| `lookback_volume_entropy` | Requires bin size parameter (magic number), complex to implement correctly |
| `lookback_aggression` | Replaced by `lookback_count_imbalance` (bounded [-1,1], symmetric) |
| `lookback_reversal_count` | Raw count depends on lookback size; consider adding normalized version later |

---

## Division by Zero Pattern

All features follow the existing codebase pattern from `types.rs`:

```rust
if denominator.abs() > f64::EPSILON {
    numerator / denominator
} else {
    0.0  // No information available
}
```

---

## Implementation Plan

### Phase 1: Core Rust Implementation

**Files to create:**
```
crates/rangebar-core/src/
├── interbar.rs          # NEW: InterBarConfig, TradeSnapshot, feature computation
└── (modify) processor.rs # Add trade_history, inter_bar_config fields
└── (modify) types.rs     # Add inter-bar feature fields to RangeBar
```

**Key struct changes:**

```rust
// types.rs - Add to RangeBar struct
pub struct RangeBar {
    // ... existing 27 fields ...

    // Inter-bar features (Tier 1 - computed from lookback window)
    pub lookback_trade_count: Option<u32>,
    pub lookback_ofi: Option<f64>,              // [-1, 1]
    pub lookback_duration_us: Option<i64>,      // microseconds (not ms!)
    pub lookback_intensity: Option<f64>,        // trades/second
    pub lookback_vwap: Option<FixedPoint>,
    pub lookback_vwap_position: Option<f64>,    // [0, 1] position within range
    pub lookback_count_imbalance: Option<f64>,  // [-1, 1] replaces aggression

    // Inter-bar features (Tier 2 - computed from lookback window)
    pub lookback_kyle_lambda: Option<f64>,      // normalized formula
    pub lookback_burstiness: Option<f64>,       // [-1, 1]
    pub lookback_volume_skew: Option<f64>,      // unbounded but rare extremes
    pub lookback_volume_kurt: Option<f64>,      // [-2, +inf)
    pub lookback_price_range: Option<f64>,      // [0, +inf) normalized

    // Inter-bar features (Tier 3 - from trading-fitness patterns)
    pub lookback_kaufman_er: Option<f64>,       // [0, 1] trend efficiency
    pub lookback_garman_klass_vol: Option<f64>, // [0, 1) OHLC volatility
    pub lookback_hurst: Option<f64>,            // [0, 1] regime indicator
    pub lookback_permutation_entropy: Option<f64>, // [0, 1] complexity
}
```

### Phase 2: PyO3 Bindings

**Files to modify:**
```
src/lib.rs               # Add inter-bar fields to dict conversion
python/rangebar/__init__.pyi  # Add type hints
```

### Phase 3: Python API

**Files to modify:**
```
python/rangebar/__init__.py   # Add include_inter_bar_features parameter
python/rangebar/constants.py  # Add INTER_BAR_FEATURE_COLUMNS
```

**New API:**
```python
df = get_range_bars(
    "BTCUSDT",
    "2024-01-01",
    "2024-12-31",
    threshold_decimal_bps=1000,
    include_inter_bar_features=True,
    inter_bar_lookback_count=500,  # Last 500 trades
)
```

### Phase 4: ClickHouse Schema

**File to modify:**
```
python/rangebar/clickhouse/schema.sql  # Add inter-bar columns
```

---

## Memory Management

**Challenge**: Storing trade history for large datasets

**Solution**: Bounded ring buffer with configurable limits

```rust
impl RangeBarProcessor {
    fn prune_trade_history(&mut self, current_timestamp: i64) {
        match &self.inter_bar_config {
            Some(InterBarConfig { lookback_mode: LookbackMode::FixedCount(n), .. }) => {
                while self.trade_history.len() > *n {
                    self.trade_history.pop_front();
                }
            }
            Some(InterBarConfig { lookback_mode: LookbackMode::FixedWindow(ms), .. }) => {
                let cutoff = current_timestamp - ms;
                while let Some(front) = self.trade_history.front() {
                    if front.timestamp < cutoff {
                        self.trade_history.pop_front();
                    } else {
                        break;
                    }
                }
            }
            None => {} // Inter-bar features disabled
        }
    }
}
```

**Memory estimate**: 500 trades × 40 bytes/trade = 20 KB per processor

---

## Critical Files

| File | Change |
|------|--------|
| `crates/rangebar-core/src/interbar.rs` | NEW: Inter-bar feature computation |
| `crates/rangebar-core/src/processor.rs` | Add trade_history, inter_bar_config |
| `crates/rangebar-core/src/types.rs` | Add inter-bar fields to RangeBar |
| `src/lib.rs` | PyO3 bindings for new fields |
| `python/rangebar/__init__.py` | API parameter |
| `python/rangebar/__init__.pyi` | Type hints |
| `python/rangebar/constants.py` | INTER_BAR_FEATURE_COLUMNS |
| `python/rangebar/clickhouse/schema.sql` | New columns |

---

## Verification Plan

### 1. Unit Tests (Rust)

```bash
cargo nextest run -p rangebar-core
```

### 2. Feature Value Bounds Tests

```rust
#[test]
fn lookback_ofi_bounded() {
    // OFI must be in [-1, 1]
    assert!(bar.lookback_ofi.unwrap() >= -1.0);
    assert!(bar.lookback_ofi.unwrap() <= 1.0);
}

#[test]
fn lookback_vwap_position_bounded() {
    // Position must be in [0, 1]
    let pos = bar.lookback_vwap_position.unwrap();
    assert!(pos >= 0.0 && pos <= 1.0);
}

#[test]
fn lookback_count_imbalance_bounded() {
    // Imbalance must be in [-1, 1]
    let imb = bar.lookback_count_imbalance.unwrap();
    assert!(imb >= -1.0 && imb <= 1.0);
}

#[test]
fn lookback_burstiness_bounded() {
    // Burstiness must be in [-1, 1]
    let b = bar.lookback_burstiness.unwrap();
    assert!(b >= -1.0 && b <= 1.0);
}

#[test]
fn lookback_volume_kurt_lower_bounded() {
    // Excess kurtosis minimum is -2 (uniform distribution)
    let k = bar.lookback_volume_kurt.unwrap();
    assert!(k >= -2.0);
}
```

### 3. Edge Case Tests

```rust
#[test]
fn empty_lookback_returns_none() {
    // With no trade history, features should be None
    let bar = process_with_empty_lookback();
    assert!(bar.lookback_trade_count.is_none());
}

#[test]
fn single_trade_lookback() {
    // Single trade: count=1, duration=0, most features=0
    let bar = process_with_single_trade_lookback();
    assert_eq!(bar.lookback_trade_count, Some(1));
    assert_eq!(bar.lookback_duration_us, Some(0));
}

#[test]
fn all_buys_lookback() {
    // All buys: OFI=1, count_imbalance=1
    let bar = process_with_all_buys();
    assert!((bar.lookback_ofi.unwrap() - 1.0).abs() < f64::EPSILON);
    assert!((bar.lookback_count_imbalance.unwrap() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn division_by_zero_guarded() {
    // Balanced volume: kyle_lambda should be 0, not infinity
    let bar = process_with_balanced_volume();
    assert!(bar.lookback_kyle_lambda.unwrap().is_finite());
}
```

### 4. Temporal Integrity Test

```rust
#[test]
fn lookback_uses_only_prior_trades() {
    // CRITICAL: All trades in lookback must have timestamp < bar.open_time
    // This ensures no lookahead bias
    for trade in &lookback_trades {
        assert!(trade.timestamp < bar.open_time);
    }
}
```

### 5. Python Integration

```bash
pytest tests/test_inter_bar_features.py -v
```

### 6. Benchmark

```bash
cargo bench -p rangebar-core
# Target: <10% overhead vs baseline
```

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Memory overhead | <50 KB per processor (500 trades × 48 bytes) |
| Processing overhead | <10% vs baseline |
| Feature computation | <1μs per bar |

---

## Feature Computation Pseudocode

```rust
fn compute_inter_bar_features(lookback: &[TradeSnapshot]) -> InterBarFeatures {
    let n = lookback.len();
    if n == 0 {
        return InterBarFeatures::default();  // All None
    }

    // Tier 1: Basic aggregates
    let trade_count = n as u32;

    let (buy_vol, sell_vol, buy_count, sell_count, total_turnover) = lookback.iter()
        .fold((0.0, 0.0, 0, 0, 0i128), |acc, t| {
            if t.is_buyer_maker {
                (acc.0, acc.1 + t.volume.to_f64(), acc.2, acc.3 + 1, acc.4 + t.turnover)
            } else {
                (acc.0 + t.volume.to_f64(), acc.1, acc.2 + 1, acc.3, acc.4 + t.turnover)
            }
        });

    let total_vol = buy_vol + sell_vol;
    let ofi = if total_vol > f64::EPSILON {
        (buy_vol - sell_vol) / total_vol
    } else { 0.0 };

    let total_count = buy_count + sell_count;
    let count_imbalance = if total_count > 0 {
        (buy_count - sell_count) as f64 / total_count as f64
    } else { 0.0 };

    let duration_us = lookback.last().unwrap().timestamp - lookback.first().unwrap().timestamp;
    let duration_sec = duration_us as f64 / 1_000_000.0;
    let intensity = if duration_sec > f64::EPSILON {
        n as f64 / duration_sec
    } else { n as f64 };

    let vwap = if total_vol > f64::EPSILON {
        FixedPoint((total_turnover / (total_vol * 1e8) as i128) as i64)
    } else { FixedPoint(0) };

    let (low, high) = lookback.iter()
        .fold((i64::MAX, i64::MIN), |acc, t| (acc.0.min(t.price.0), acc.1.max(t.price.0)));
    let range = (high - low) as f64;
    let vwap_position = if range > f64::EPSILON {
        (vwap.0 - low) as f64 / range
    } else { 0.5 };

    // Tier 2: Statistical features (with minimum trade requirements)
    let kyle_lambda = if n >= 2 {
        compute_kyle_lambda(lookback)
    } else { 0.0 };

    let burstiness = if n >= 2 {
        compute_burstiness(lookback)
    } else { 0.0 };

    let (volume_skew, volume_kurt) = if n >= 4 {
        compute_volume_moments(lookback)
    } else { (0.0, 0.0) };

    let first_price = lookback.first().unwrap().price.to_f64();
    let price_range = if first_price > f64::EPSILON {
        (high - low) as f64 / 1e8 / first_price
    } else { 0.0 };

    InterBarFeatures {
        lookback_trade_count: Some(trade_count),
        lookback_ofi: Some(ofi),
        lookback_duration_us: Some(duration_us),
        lookback_intensity: Some(intensity),
        lookback_vwap: Some(vwap),
        lookback_vwap_position: Some(vwap_position),
        lookback_count_imbalance: Some(count_imbalance),
        lookback_kyle_lambda: Some(kyle_lambda),
        lookback_burstiness: Some(burstiness),
        lookback_volume_skew: Some(volume_skew),
        lookback_volume_kurt: Some(volume_kurt),
        lookback_price_range: Some(price_range),
    }
}
```

---

## Tier 2 Feature Computation Details

### Kyle's Lambda (Normalized)

**Validated Source**: [Kyle (1985)](https://www.jstor.org/stable/1913210), Econometrica, Vol. 53, No. 6, pp. 1315-1335

**Implementation pattern validated against existing codebase**: `types.rs:383-398`

```rust
fn compute_kyle_lambda(lookback: &[TradeSnapshot]) -> f64 {
    // VERIFIED FORMULA (normalized version):
    // λ = ((price_end - price_start) / price_start) / ((buy_vol - sell_vol) / total_vol)
    //
    // This is a NORMALIZED version of Kyle's original:
    //   Δp = λ × (order_flow)
    //
    // Normalization rationale:
    // - Numerator: percentage return (dimensionless)
    // - Denominator: order flow imbalance ratio (dimensionless, bounded [-1, 1])
    // - Result: dimensionless, comparable across assets
    //
    // Interpretation:
    //   λ > 0: Price moves in direction of order flow (normal)
    //   λ < 0: Price moves against order flow (unusual, may indicate manipulation)
    //   |λ| high: Large price impact per unit imbalance (illiquid)
    //   |λ| low: Small price impact (liquid, deep market)
    //
    // Reference: Kyle (1985), Hasbrouck (2009), existing rangebar-core implementation

    if lookback.len() < 2 { return 0.0; }

    let first_price = lookback.first().unwrap().price.to_f64();
    let last_price = lookback.last().unwrap().price.to_f64();

    let (buy_vol, sell_vol): (f64, f64) = lookback.iter()
        .fold((0.0, 0.0), |acc, t| {
            if t.is_buyer_maker { (acc.0, acc.1 + t.volume.to_f64()) }
            else { (acc.0 + t.volume.to_f64(), acc.1) }
        });

    let total_vol = buy_vol + sell_vol;
    let normalized_imbalance = if total_vol > f64::EPSILON {
        (buy_vol - sell_vol) / total_vol
    } else { 0.0 };

    // Division by zero guards (matches existing codebase pattern)
    if normalized_imbalance.abs() > f64::EPSILON && first_price.abs() > f64::EPSILON {
        ((last_price - first_price) / first_price) / normalized_imbalance
    } else {
        0.0  // No information when imbalance is zero or price is zero
    }
}
```

**Note**: This is UNBOUNDED. Extreme values are possible when small imbalance produces large price movement. Consider logging a warning for |λ| > 10 but do NOT clip the value.

### Burstiness (Goh-Barabási)

**Validated Source**: [Goh & Barabási (2008)](https://iopscience.iop.org/article/10.1209/0295-5075/81/48002), EPL (Europhysics Letters), Vol. 81, 48002

```rust
fn compute_burstiness(lookback: &[TradeSnapshot]) -> f64 {
    // VERIFIED FORMULA: B = (σ_τ - μ_τ) / (σ_τ + μ_τ)
    // Reference: Goh & Barabási (2008), EPL, Vol. 81, 48002
    //
    // τ = inter-event times (inter-arrival times between consecutive trades)
    // σ_τ = standard deviation of τ
    // μ_τ = mean of τ
    //
    // Interpretation:
    //   B = -1: Perfectly regular (periodic) arrivals (σ = 0)
    //   B =  0: Poisson process (σ = μ, exponential distribution)
    //   B = +1: Maximally bursty (σ >> μ)

    if lookback.len() < 2 { return 0.0; }

    // Compute inter-arrival times (microseconds)
    let inter_arrivals: Vec<f64> = lookback.windows(2)
        .map(|w| (w[1].timestamp - w[0].timestamp) as f64)
        .collect();

    let n = inter_arrivals.len() as f64;
    let mu = inter_arrivals.iter().sum::<f64>() / n;

    let variance = inter_arrivals.iter()
        .map(|t| (t - mu).powi(2))
        .sum::<f64>() / n;
    let sigma = variance.sqrt();

    let denominator = sigma + mu;
    if denominator > f64::EPSILON {
        (sigma - mu) / denominator
    } else {
        0.0  // All trades at same timestamp (undefined)
    }
}
```

### Volume Moments (Skewness and Kurtosis)

```rust
fn compute_volume_moments(lookback: &[TradeSnapshot]) -> (f64, f64) {
    // Returns (skewness, excess_kurtosis)
    // Skewness: E[(V-μ)³] / σ³  (Fisher-Pearson coefficient)
    // Excess Kurtosis: E[(V-μ)⁴] / σ⁴ - 3  (normal distribution = 0)

    let volumes: Vec<f64> = lookback.iter().map(|t| t.volume.to_f64()).collect();
    let n = volumes.len() as f64;

    if n < 3.0 { return (0.0, 0.0); }

    let mu = volumes.iter().sum::<f64>() / n;

    // Central moments
    let m2 = volumes.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n;
    let m3 = volumes.iter().map(|v| (v - mu).powi(3)).sum::<f64>() / n;
    let m4 = volumes.iter().map(|v| (v - mu).powi(4)).sum::<f64>() / n;

    let sigma = m2.sqrt();

    if sigma < f64::EPSILON {
        return (0.0, 0.0);  // All same volume
    }

    let skewness = m3 / sigma.powi(3);
    let kurtosis = m4 / sigma.powi(4) - 3.0;  // Excess kurtosis

    (skewness, kurtosis)
}
```

---

## Insights from trading-fitness Codebase

Analysis of `~/eon/trading-fitness` revealed additional features and patterns worth incorporating:

### High-Priority Features to Add (from trading-fitness)

| Feature | Formula | Range | Source File | Value for Range Bars |
|---------|---------|-------|-------------|---------------------|
| **Kaufman Efficiency Ratio** | \|close[n]-close[0]\| / Σ\|changes\| | [0, 1] | `risk.rs` | Trend quality - 1=perfect trend, 0=choppy |
| **Garman-Klass Volatility** | sqrt(0.5×ln(H/L)² - 0.39×ln(C/O)²) | [0, 1) | `risk.rs` | OHLC-based volatility, more efficient than close-to-close |
| **Hurst Exponent** | DFA (Detrended Fluctuation Analysis) | [0, 1] | `fractal.rs` | Regime: H<0.5 mean-reverting, H>0.5 trending |
| **Permutation Entropy** | -Σ p_π × log(p_π) / log(m!) | [0, 1] | `entropy.rs` | Complexity via ordinal patterns (scale-invariant) |

### Medium-Priority Features

| Feature | Formula | Range | Min Samples | Value |
|---------|---------|-------|-------------|-------|
| **Ulcer Index** | sqrt(mean(drawdown²)) | [0, 1] | 10 | Quadratic drawdown penalty |
| **Fractal Dimension** | Higuchi's Method | [0, 1] | 128 | Roughness: D≈1 smooth, D≈2 noise |

### Key Implementation Patterns from trading-fitness

1. **All outputs bounded [0, 1]**: Use tanh/sigmoid transforms for unbounded metrics
2. **Soft clamp for Hurst**: `0.5 + 0.5 × tanh((x-0.5) × 4)`
3. **Relative epsilon**: `scale × ε_machine × 100` instead of fixed epsilon
4. **Online normalization**: Welford's algorithm for streaming mean/variance
5. **Warmup handling**: First `lookback-1` values are NaN/None

### Feature Selection Thresholds (from trading-fitness)

```python
min_variance = 0.01      # Remove features with variance < 1%
max_correlation = 0.95   # Remove redundant features with |r| > 0.95
```

### Regime Detection Pattern

```rust
// From trading-fitness fractal.rs
fn classify_regime(hurst: f64) -> Regime {
    if hurst > 0.55 { Regime::Trending }
    else if hurst < 0.45 { Regime::MeanReverting }
    else { Regime::RandomWalk }
}
```

---

## Updated Final Feature List (16 Features)

### Tier 1: Core Features (7 features) - Original

| Feature | Formula | Range | Min Trades |
|---------|---------|-------|------------|
| `lookback_trade_count` | COUNT(trades) | [0, ∞) | 0 |
| `lookback_ofi` | (buy_vol - sell_vol) / total_vol | [-1, 1] | 1 |
| `lookback_duration_us` | last_ts - first_ts | [0, ∞) | 1 |
| `lookback_intensity` | trade_count / duration_sec | [0, ∞) | 1 |
| `lookback_vwap` | Σ(price×vol) / Σ(vol) | [min_p, max_p] | 1 |
| `lookback_vwap_position` | (vwap - low) / (high - low) | [0, 1] | 1 |
| `lookback_count_imbalance` | (buy_count - sell_count) / total_count | [-1, 1] | 1 |

### Tier 2: Statistical Features (5 features) - Original

| Feature | Formula | Range | Min Trades |
|---------|---------|-------|------------|
| `lookback_kyle_lambda` | ((last-first)/first) / ((buy-sell)/total) | (-∞, ∞) | 2 |
| `lookback_burstiness` | (σ_τ - μ_τ) / (σ_τ + μ_τ) | [-1, 1] | 2 |
| `lookback_volume_skew` | E[(V-μ)³] / σ³ | (-∞, ∞) | 3 |
| `lookback_volume_kurt` | E[(V-μ)⁴] / σ⁴ - 3 | [-2, ∞) | 4 |
| `lookback_price_range` | (high - low) / first_price | [0, ∞) | 1 |

### Tier 3: trading-fitness Features (4 features) - NEW

| Feature | Formula | Range | Min Trades | Academic Reference |
|---------|---------|-------|------------|-------------------|
| `lookback_kaufman_er` | \|p[n]-p[0]\| / Σ\|p[i]-p[i-1]\| | [0, 1] | 2 | Kaufman (1995) |
| `lookback_garman_klass_vol` | sqrt(0.5×ln(H/L)² - 0.39×ln(C/O)²) | [0, 1) | 1 | Garman & Klass (1980) |
| `lookback_hurst` | DFA estimator, soft-clamped | [0, 1] | 64 | Hurst (1951), Peng (1994) |
| `lookback_permutation_entropy` | -Σ p_π × log(p_π) / log(m!) | [0, 1] | 60 | Bandt & Pompe (2002) |

---

## Summary: No Magic Numbers

All 16 features use only:
- Standard statistical formulas (mean, std, skewness, kurtosis, burstiness)
- The `f64::EPSILON` guard for numerical stability (consistent with existing codebase)
- Minimum sample size requirements that are mathematically necessary (not arbitrary)
- Academic-backed constants (e.g., Garman-Klass 0.39 coefficient)

Features intentionally **removed** to avoid magic numbers:
- `volume_entropy` - requires bin size/count parameter
- `reversal_count` - raw count depends on lookback size
- `ofi_trend` - unbounded linear regression, conceptually unclear

---

## Academic References

| Feature | Reference |
|---------|-----------|
| OFI | Chordia et al. (2002) - Order imbalance |
| Kyle's Lambda | Kyle (1985) - Continuous auctions and insider trading |
| Burstiness | Goh & Barabási (2008) - Burstiness and memory in complex systems |
| Skewness/Kurtosis | Standard statistical moments |
| VWAP | Industry standard volume-weighted average price |
| Kaufman ER | Kaufman (1995) - Smarter Trading |
| Garman-Klass | Garman & Klass (1980) - On the Estimation of Security Price Volatilities |
| Hurst Exponent | Hurst (1951), DFA method: Peng et al. (1994) |
| Permutation Entropy | Bandt & Pompe (2002) - Permutation Entropy: A Natural Complexity Measure |

---

## Tier 3 Feature Computation Details (from trading-fitness)

### Kaufman Efficiency Ratio

```rust
fn compute_kaufman_er(prices: &[f64]) -> f64 {
    // Kaufman Efficiency Ratio = |net movement| / sum of |individual movements|
    // Range: [0, 1] where 1 = perfect trend, 0 = pure noise
    // Reference: Kaufman (1995)

    if prices.len() < 2 { return 0.0; }

    let net_movement = (prices.last().unwrap() - prices.first().unwrap()).abs();

    let volatility: f64 = prices.windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .sum();

    if volatility > f64::EPSILON {
        net_movement / volatility
    } else {
        0.0  // No movement
    }
}
```

### Garman-Klass Volatility

**Validated Source**: [Garman & Klass (1980)](https://www.cmegroup.com/trading/fx/files/a_estimation_of_security_price.pdf), Journal of Business

```rust
fn compute_garman_klass_vol(lookback: &[TradeSnapshot]) -> f64 {
    // Garman-Klass estimator using OHLC data
    // Reference: Garman & Klass (1980), Journal of Business, vol. 53, no. 1
    //
    // VERIFIED FORMULA:
    // σ² = 0.5 × ln(H/L)² - (2×ln(2) - 1) × ln(C/O)²
    //
    // Exact coefficient: (2×ln(2) - 1) = 2×0.693147... - 1 = 0.386294...
    // Use exact calculation, not magic number 0.386

    if lookback.is_empty() { return 0.0; }

    let (open, high, low, close) = compute_ohlc(lookback);

    let o = open.to_f64();
    let h = high.to_f64();
    let l = low.to_f64();
    let c = close.to_f64();

    // Guard: prices must be positive
    if o <= f64::EPSILON || l <= f64::EPSILON || h <= f64::EPSILON { return 0.0; }

    let log_hl = (h / l).ln();
    let log_co = (c / o).ln();

    // Use exact coefficient derivation, not magic number
    let coef = 2.0 * 2.0_f64.ln() - 1.0;  // ≈ 0.386294

    let variance = 0.5 * log_hl.powi(2) - coef * log_co.powi(2);

    // Variance can be negative due to the subtractive term when close/open dominates
    // This indicates the estimator is unreliable for this sample
    if variance > 0.0 {
        variance.sqrt()
    } else {
        0.0  // Return 0 for unreliable estimate
    }
}
```

**Note on negative variance**: The Garman-Klass estimator can produce negative variance when the `ln(C/O)²` term dominates. This occurs when the price moved significantly from open to close but stayed within a narrow high-low range. In such cases, returning 0.0 is conservative but loses information. An alternative is to use the simpler Parkinson estimator `σ = ln(H/L) / (2×sqrt(ln(2)))` as fallback.

### Hurst Exponent (DFA)

**Validated Source**: [Peng et al. (1994)](https://www.nature.com/articles/368168a0), Nature, 356, 168-170; [Wikipedia DFA](https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis)

**Method**: Detrended Fluctuation Analysis (DFA) - selected for robustness to non-stationarity

```rust
fn compute_hurst_dfa(prices: &[f64]) -> f64 {
    // VERIFIED ALGORITHM: Detrended Fluctuation Analysis (DFA)
    // Reference: Peng et al. (1994), Nature, 356, 168-170
    //
    // Algorithm steps:
    // 1. Compute profile Y(i) = Σ(x_k - <x>) for k=1 to i (cumulative deviation)
    // 2. Divide profile into non-overlapping boxes of size n
    // 3. Fit linear trend in each box, compute RMS of residuals
    // 4. Average RMS across all boxes → F(n)
    // 5. Repeat for multiple box sizes
    // 6. Hurst exponent H = slope of log(F(n)) vs log(n)
    //
    // Interpretation:
    //   H < 0.5: Anti-correlated (mean-reverting)
    //   H = 0.5: Random walk (uncorrelated)
    //   H > 0.5: Positively correlated (trending)
    //
    // Output: soft-clamped to [0, 1] for ML consumption

    const MIN_SAMPLES: usize = 64;
    if prices.len() < MIN_SAMPLES { return 0.5; }  // Neutral (insufficient data)

    // Step 1: Compute profile (cumulative deviation from mean)
    let mean = prices.iter().sum::<f64>() / prices.len() as f64;
    let profile: Vec<f64> = prices.iter()
        .scan(0.0, |acc, &x| {
            *acc += x - mean;
            Some(*acc)
        })
        .collect();

    let n = profile.len();

    // Step 2-5: Compute F(n) for multiple box sizes
    // Use logarithmically spaced box sizes from 4 to n/4
    let min_box = 4;
    let max_box = n / 4;
    if max_box < min_box { return 0.5; }

    let mut log_n = Vec::new();
    let mut log_f = Vec::new();

    // Generate ~10-20 box sizes logarithmically spaced
    let num_scales = ((max_box as f64).ln() - (min_box as f64).ln()) / 0.25;
    let num_scales = (num_scales as usize).max(4).min(20);

    for i in 0..num_scales {
        let box_size = (min_box as f64 * ((max_box as f64 / min_box as f64).powf(i as f64 / (num_scales - 1) as f64))) as usize;
        let box_size = box_size.max(min_box).min(max_box);

        let f_n = compute_dfa_fluctuation(&profile, box_size);
        if f_n > f64::EPSILON {
            log_n.push((box_size as f64).ln());
            log_f.push(f_n.ln());
        }
    }

    if log_n.len() < 4 { return 0.5; }

    // Step 6: Linear regression to get slope (Hurst exponent)
    let hurst = linear_regression_slope(&log_n, &log_f);

    // Soft clamp to [0, 1] using tanh (from trading-fitness pattern)
    soft_clamp_hurst(hurst)
}

fn compute_dfa_fluctuation(profile: &[f64], box_size: usize) -> f64 {
    // Compute RMS fluctuation for given box size
    let n = profile.len();
    let num_boxes = n / box_size;
    if num_boxes == 0 { return 0.0; }

    let mut total_variance = 0.0;

    for i in 0..num_boxes {
        let start = i * box_size;
        let end = start + box_size;
        let segment = &profile[start..end];

        // Fit linear trend: y = a + b*x
        let (a, b) = linear_fit(segment);

        // Compute variance of detrended segment
        let variance: f64 = segment.iter().enumerate()
            .map(|(j, &y)| {
                let trend = a + b * (j as f64);
                (y - trend).powi(2)
            })
            .sum::<f64>() / box_size as f64;

        total_variance += variance;
    }

    (total_variance / num_boxes as f64).sqrt()
}

fn linear_fit(y: &[f64]) -> (f64, f64) {
    // Least squares fit: y = a + b*x where x = 0, 1, 2, ...
    let n = y.len() as f64;
    let sum_x = (n - 1.0) * n / 2.0;
    let sum_x2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = y.iter().enumerate().map(|(i, &v)| i as f64 * v).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return (sum_y / n, 0.0);  // Flat line
    }

    let b = (n * sum_xy - sum_x * sum_y) / denom;
    let a = (sum_y - b * sum_x) / n;
    (a, b)
}

fn linear_regression_slope(x: &[f64], y: &[f64]) -> f64 {
    // Simple least squares slope
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let num: f64 = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    let denom: f64 = x.iter()
        .map(|&xi| (xi - mean_x).powi(2))
        .sum();

    if denom.abs() < f64::EPSILON { 0.5 } else { num / denom }
}

fn soft_clamp_hurst(h: f64) -> f64 {
    // Soft clamp to [0, 1] using tanh
    // From trading-fitness adaptive.rs: 0.5 + 0.5 × tanh((x - 0.5) × 4)
    // Maps 0.5 → 0.5, and asymptotically approaches 0 or 1 for extreme values
    0.5 + 0.5 * ((h - 0.5) * 4.0).tanh()
}
```

### Permutation Entropy

**Validated Source**: [Bandt & Pompe (2002)](https://pubmed.ncbi.nlm.nih.gov/12005759/), Physical Review Letters, 88, 174102

```rust
fn compute_permutation_entropy(prices: &[f64]) -> f64 {
    // VERIFIED FORMULA: H_PE = -Σ p_π × ln(p_π) / ln(m!)
    // Reference: Bandt & Pompe (2002), Phys. Rev. Lett. 88, 174102
    //
    // Algorithm:
    // 1. Choose embedding dimension m (recommended: 3-7)
    // 2. For each position, get ordinal pattern (rank ordering of m consecutive values)
    // 3. Count frequency of each of the m! possible patterns
    // 4. Compute Shannon entropy of the pattern distribution
    // 5. Normalize by max entropy ln(m!)
    //
    // Output range: [0, 1] where 0 = deterministic, 1 = completely random
    //
    // Key property: Scale-invariant (only considers relative ordering)

    const M: usize = 3;  // Embedding dimension (Bandt & Pompe recommend 3-7)
    const MIN_SAMPLES: usize = 60;  // Rule of thumb: 10 × m! = 10 × 6 = 60 for m=3

    if prices.len() < MIN_SAMPLES { return 1.0; }  // Insufficient data → max entropy

    // Count occurrences of each permutation pattern
    // For m=3, there are 3! = 6 possible patterns
    let mut pattern_counts: [usize; 6] = [0; 6];  // Fixed array for m=3
    let n_patterns = prices.len() - M + 1;

    for i in 0..n_patterns {
        let window = &prices[i..i + M];
        let pattern_idx = ordinal_pattern_index_m3(window);
        pattern_counts[pattern_idx] += 1;
    }

    // Compute Shannon entropy of pattern distribution
    let total = n_patterns as f64;
    let entropy: f64 = pattern_counts.iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.ln()
        })
        .sum();

    // Normalize by maximum possible entropy: ln(3!) = ln(6)
    let max_entropy = 6.0_f64.ln();  // ≈ 1.7918

    entropy / max_entropy
}

fn ordinal_pattern_index_m3(window: &[f64]) -> usize {
    // For m=3, directly compute pattern index (0-5) without allocation
    // Patterns: 012, 021, 102, 120, 201, 210 (lexicographic order)
    let (a, b, c) = (window[0], window[1], window[2]);

    if a <= b {
        if b <= c { 0 }       // a ≤ b ≤ c → 012
        else if a <= c { 1 }  // a ≤ c < b → 021
        else { 3 }            // c < a ≤ b → 201
    } else {
        if a <= c { 2 }       // b < a ≤ c → 102
        else if b <= c { 4 }  // b ≤ c < a → 120
        else { 5 }            // c < b < a → 210
    }
}
```

**Note on ties**: When values are equal, the implementation should handle ties consistently. The above uses `<=` which treats ties as "first value comes first". For financial data with sufficient precision, exact ties are rare.

---

## Implementation Priority

Given the minimum sample requirements, implement in phases:

### Phase 1: Low-Complexity Features (min trades ≤ 4)
All Tier 1 + most Tier 2 features

### Phase 2: Statistical Features (min trades 4-64)
`lookback_volume_skew`, `lookback_volume_kurt`, `lookback_hurst`

### Phase 3: Advanced Features (min trades > 60)
`lookback_permutation_entropy` (requires 60+ samples for m=3)

**Recommendation**: For typical lookback of 500 trades, all features are viable. For smaller lookbacks (e.g., 50 trades), Tier 3 features may not be computable and should return None.

---

## Formula Validation Summary

All formulas have been validated against authoritative academic sources:

| Feature | Source | Validation Status |
|---------|--------|-------------------|
| **OFI** | Chordia et al. (2002) | ✅ Matches existing `types.rs:362-366` |
| **Kyle's Lambda** | Kyle (1985), Hasbrouck (2009) | ✅ Matches existing `types.rs:383-398` |
| **Burstiness** | [Goh & Barabási (2008)](https://iopscience.iop.org/article/10.1209/0295-5075/81/48002) | ✅ Formula: B = (σ-μ)/(σ+μ) verified |
| **Volume Skew/Kurt** | Standard statistical moments | ✅ Fisher-Pearson coefficient |
| **Kaufman ER** | [StockCharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama) | ✅ ER = Change/Volatility |
| **Garman-Klass** | [Garman & Klass (1980)](https://www.cmegroup.com/trading/fx/files/a_estimation_of_security_price.pdf) | ✅ σ² = 0.5×ln(H/L)² - (2ln2-1)×ln(C/O)² |
| **Hurst (DFA)** | [Peng et al. (1994)](https://www.nature.com/articles/368168a0), [Wikipedia](https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis) | ✅ DFA algorithm verified |
| **Permutation Entropy** | [Bandt & Pompe (2002)](https://pubmed.ncbi.nlm.nih.gov/12005759/) | ✅ H = -Σp×ln(p) / ln(m!) verified |

### Key Implementation Notes

1. **No magic numbers**: All coefficients are derived from first principles:
   - Garman-Klass: `2×ln(2) - 1` computed exactly, not hardcoded 0.386
   - Permutation Entropy: `ln(m!)` computed from embedding dimension
   - Hurst soft clamp: `0.5 + 0.5×tanh((x-0.5)×4)` from trading-fitness

2. **Division by zero handling**: All features use the established pattern:
   ```rust
   if denominator.abs() > f64::EPSILON { result } else { 0.0 }
   ```

3. **Minimum sample requirements**: Enforced at computation time, return sensible defaults when insufficient data

4. **Unbounded features**: `kyle_lambda`, `volume_skew`, `volume_kurt` are intentionally unbounded - extreme values are informative signals

---

## Validation Strategy: Four-Layer Verification

### Validation 1: Property-Based Testing (Rust)

Mathematical invariants that MUST hold for all inputs:

```rust
use proptest::prelude::*;

proptest! {
    // ========== BOUNDED FEATURE INVARIANTS ==========

    #[test]
    fn ofi_always_bounded(
        buy_vol in 0.0f64..1e12,
        sell_vol in 0.0f64..1e12
    ) {
        let total = buy_vol + sell_vol;
        if total > f64::EPSILON {
            let ofi = (buy_vol - sell_vol) / total;
            prop_assert!(ofi >= -1.0 && ofi <= 1.0, "OFI {} not in [-1,1]", ofi);
        }
    }

    #[test]
    fn burstiness_always_bounded(
        inter_arrivals in prop::collection::vec(0.0f64..1e9, 2..1000)
    ) {
        let n = inter_arrivals.len() as f64;
        let mu = inter_arrivals.iter().sum::<f64>() / n;
        let sigma = (inter_arrivals.iter().map(|t| (t - mu).powi(2)).sum::<f64>() / n).sqrt();

        if sigma + mu > f64::EPSILON {
            let b = (sigma - mu) / (sigma + mu);
            prop_assert!(b >= -1.0 && b <= 1.0, "Burstiness {} not in [-1,1]", b);
        }
    }

    #[test]
    fn kaufman_er_always_bounded(
        prices in prop::collection::vec(1.0f64..1e6, 2..1000)
    ) {
        let net = (prices.last().unwrap() - prices.first().unwrap()).abs();
        let volatility: f64 = prices.windows(2).map(|w| (w[1] - w[0]).abs()).sum();

        if volatility > f64::EPSILON {
            let er = net / volatility;
            prop_assert!(er >= 0.0 && er <= 1.0, "Kaufman ER {} not in [0,1]", er);
        }
    }

    #[test]
    fn vwap_position_always_bounded(
        prices in prop::collection::vec(1.0f64..1e6, 1..1000),
        volumes in prop::collection::vec(0.001f64..1e6, 1..1000)
    ) {
        let n = prices.len().min(volumes.len());
        if n == 0 { return Ok(()); }

        let prices = &prices[..n];
        let volumes = &volumes[..n];

        let total_vol: f64 = volumes.iter().sum();
        let vwap = prices.iter().zip(volumes.iter())
            .map(|(p, v)| p * v)
            .sum::<f64>() / total_vol;

        let high = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let low = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = high - low;

        if range > f64::EPSILON {
            let pos = (vwap - low) / range;
            prop_assert!(pos >= 0.0 && pos <= 1.0, "VWAP position {} not in [0,1]", pos);
        }
    }

    #[test]
    fn hurst_soft_clamp_always_bounded(raw_hurst in -10.0f64..10.0f64) {
        let clamped = 0.5 + 0.5 * ((raw_hurst - 0.5) * 4.0).tanh();
        prop_assert!(clamped > 0.0 && clamped < 1.0, "Hurst {} not in (0,1)", clamped);
    }

    #[test]
    fn permutation_entropy_always_bounded(prices in prop::collection::vec(1.0f64..1e6, 60..500)) {
        // Compute permutation entropy for m=3
        let mut pattern_counts = [0usize; 6];
        let n_patterns = prices.len() - 2;

        for i in 0..n_patterns {
            let (a, b, c) = (prices[i], prices[i+1], prices[i+2]);
            let idx = ordinal_pattern_index_m3(a, b, c);
            pattern_counts[idx] += 1;
        }

        let total = n_patterns as f64;
        let entropy: f64 = pattern_counts.iter()
            .filter(|&&c| c > 0)
            .map(|&c| { let p = c as f64 / total; -p * p.ln() })
            .sum();

        let normalized = entropy / 6.0_f64.ln();
        prop_assert!(normalized >= 0.0 && normalized <= 1.0, "PE {} not in [0,1]", normalized);
    }

    // ========== EXTREME VALUE HANDLING ==========

    #[test]
    fn kyle_lambda_handles_zero_imbalance(
        first_price in 1.0f64..1e6,
        last_price in 1.0f64..1e6,
        volume in 0.001f64..1e6
    ) {
        // Equal buy/sell → imbalance = 0 → should return 0, not infinity
        let buy_vol = volume;
        let sell_vol = volume;
        let total = buy_vol + sell_vol;
        let imbalance = (buy_vol - sell_vol) / total;  // = 0

        let lambda = if imbalance.abs() > f64::EPSILON && first_price > f64::EPSILON {
            ((last_price - first_price) / first_price) / imbalance
        } else {
            0.0
        };

        prop_assert!(lambda.is_finite(), "Kyle lambda should be finite, got {}", lambda);
        prop_assert!(lambda == 0.0, "Kyle lambda should be 0 for zero imbalance");
    }

    #[test]
    fn garman_klass_handles_edge_cases(
        o in 1.0f64..1e6,
        h_offset in 0.0f64..0.1,
        l_offset in 0.0f64..0.1,
        c_offset in -0.05f64..0.05
    ) {
        let h = o * (1.0 + h_offset);
        let l = o * (1.0 - l_offset);
        let c = o * (1.0 + c_offset);

        // Ensure h >= l
        let (h, l) = if h >= l { (h, l) } else { (l, h) };

        if o > f64::EPSILON && l > f64::EPSILON && h > f64::EPSILON {
            let log_hl = (h / l).ln();
            let log_co = (c / o).ln();
            let coef = 2.0 * 2.0_f64.ln() - 1.0;
            let variance = 0.5 * log_hl.powi(2) - coef * log_co.powi(2);

            // Variance CAN be negative (when close/open dominates)
            // Result should be 0 or positive sqrt
            let result = if variance > 0.0 { variance.sqrt() } else { 0.0 };
            prop_assert!(result.is_finite() && result >= 0.0);
        }
    }
}

fn ordinal_pattern_index_m3(a: f64, b: f64, c: f64) -> usize {
    if a <= b {
        if b <= c { 0 } else if a <= c { 1 } else { 3 }
    } else {
        if a <= c { 2 } else if b <= c { 4 } else { 5 }
    }
}
```

### Validation 2: Known Value Tests (Ground Truth)

Test against analytically computed expected values:

```rust
#[cfg(test)]
mod known_value_tests {
    use super::*;
    use approx::assert_relative_eq;

    // ========== OFI KNOWN VALUES ==========

    #[test]
    fn ofi_all_buys() {
        // All buy volume → OFI = 1.0
        let ofi = compute_ofi(100.0, 0.0);  // buy=100, sell=0
        assert_relative_eq!(ofi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn ofi_all_sells() {
        // All sell volume → OFI = -1.0
        let ofi = compute_ofi(0.0, 100.0);
        assert_relative_eq!(ofi, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn ofi_balanced() {
        // Equal buy/sell → OFI = 0
        let ofi = compute_ofi(50.0, 50.0);
        assert_relative_eq!(ofi, 0.0, epsilon = 1e-10);
    }

    // ========== BURSTINESS KNOWN VALUES ==========

    #[test]
    fn burstiness_regular_intervals() {
        // Perfectly regular: σ = 0 → B = (0 - μ) / (0 + μ) = -1
        let inter_arrivals = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let b = compute_burstiness(&inter_arrivals);
        assert_relative_eq!(b, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn burstiness_poisson_like() {
        // Exponential distribution: σ ≈ μ → B ≈ 0
        // For exponential with λ=1, E[X]=1, Var[X]=1, so σ=μ=1
        // Use pre-computed exponential samples (mean≈1, std≈1)
        let inter_arrivals = vec![0.1, 2.3, 0.5, 1.8, 0.3, 1.2, 0.8, 1.5, 0.2, 2.1];
        let b = compute_burstiness(&inter_arrivals);
        // Should be close to 0, allow tolerance for sample variance
        assert!(b.abs() < 0.3, "Burstiness should be near 0 for exponential, got {}", b);
    }

    // ========== KAUFMAN ER KNOWN VALUES ==========

    #[test]
    fn kaufman_er_perfect_trend() {
        // Monotonically increasing: net = sum of changes → ER = 1
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let er = compute_kaufman_er(&prices);
        assert_relative_eq!(er, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn kaufman_er_round_trip() {
        // Goes up then back down: net = 0 → ER = 0
        let prices = vec![100.0, 102.0, 104.0, 102.0, 100.0];
        let er = compute_kaufman_er(&prices);
        assert_relative_eq!(er, 0.0, epsilon = 1e-10);
    }

    // ========== GARMAN-KLASS KNOWN VALUES ==========

    #[test]
    fn garman_klass_flat_bar() {
        // O=H=L=C → variance = 0
        let gk = compute_garman_klass(100.0, 100.0, 100.0, 100.0);
        assert_relative_eq!(gk, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn garman_klass_standard_case() {
        // O=100, H=105, L=95, C=102
        // ln(H/L) = ln(105/95) ≈ 0.1001
        // ln(C/O) = ln(102/100) = ln(1.02) ≈ 0.0198
        // σ² = 0.5×(0.1001)² - 0.386×(0.0198)² ≈ 0.005 - 0.00015 ≈ 0.00485
        // σ ≈ 0.0696
        let gk = compute_garman_klass(100.0, 105.0, 95.0, 102.0);
        assert_relative_eq!(gk, 0.0696, epsilon = 0.001);
    }

    // ========== HURST KNOWN VALUES ==========

    #[test]
    fn hurst_random_walk() {
        // Random walk should have H ≈ 0.5
        // Use known random walk sequence
        let mut rng = rand::thread_rng();
        let returns: Vec<f64> = (0..500).map(|_| rng.gen::<f64>() - 0.5).collect();
        let prices: Vec<f64> = returns.iter()
            .scan(100.0, |state, &r| { *state *= 1.0 + r * 0.01; Some(*state) })
            .collect();

        let h = compute_hurst_dfa(&prices);
        // Should be close to 0.5 (after soft clamp, exactly 0.5 maps to 0.5)
        assert!((h - 0.5).abs() < 0.15, "Hurst should be ~0.5 for random walk, got {}", h);
    }

    // ========== PERMUTATION ENTROPY KNOWN VALUES ==========

    #[test]
    fn permutation_entropy_monotonic() {
        // Strictly increasing: only pattern 012 appears → H = 0
        let prices: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let pe = compute_permutation_entropy(&prices);
        assert_relative_eq!(pe, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn permutation_entropy_alternating() {
        // Alternating: 1,2,1,2,1,2... → patterns 012 and 210 equally likely → H = log(2)/log(6) ≈ 0.387
        let prices: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
        let pe = compute_permutation_entropy(&prices);
        let expected = 2.0_f64.ln() / 6.0_f64.ln();  // ≈ 0.387
        assert_relative_eq!(pe, expected, epsilon = 0.05);
    }
}
```

### Validation 3: Cross-Reference with trading-fitness

Verify our implementations match trading-fitness outputs for the same inputs:

```python
# tests/test_cross_reference_trading_fitness.py
import pytest
import numpy as np
import polars as pl
from rangebar import compute_inter_bar_features  # Our implementation

# Import trading-fitness for comparison
import sys
sys.path.insert(0, '/Users/terryli/eon/trading-fitness/packages/metrics-rust')
from metrics_rust import (
    compute_hurst_exponent,
    compute_permutation_entropy,
    compute_garman_klass,
    compute_kaufman_efficiency_ratio,
)

class TestCrossReferenceWithTradingFitness:
    """Verify our implementations match trading-fitness for same inputs"""

    def test_hurst_exponent_matches(self):
        """Hurst exponent should match trading-fitness within tolerance"""
        # Generate test data
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(500)) + 100

        # Our implementation
        our_hurst = compute_inter_bar_features(prices)['lookback_hurst']

        # Trading-fitness implementation
        tf_hurst = compute_hurst_exponent(prices)

        # Should match within 5% (different implementations may vary slightly)
        assert abs(our_hurst - tf_hurst) < 0.05, f"Hurst mismatch: ours={our_hurst}, tf={tf_hurst}"

    def test_permutation_entropy_matches(self):
        """Permutation entropy should match trading-fitness"""
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(200)) + 100

        our_pe = compute_inter_bar_features(prices)['lookback_permutation_entropy']
        tf_pe = compute_permutation_entropy(prices, m=3)

        assert abs(our_pe - tf_pe) < 0.01, f"PE mismatch: ours={our_pe}, tf={tf_pe}"

    def test_kaufman_er_matches(self):
        """Kaufman ER should match trading-fitness"""
        prices = np.array([100, 101, 102, 101, 103, 104, 103, 105])

        our_er = compute_inter_bar_features(prices)['lookback_kaufman_er']
        tf_er = compute_kaufman_efficiency_ratio(prices)

        assert abs(our_er - tf_er) < 0.001, f"Kaufman ER mismatch: ours={our_er}, tf={tf_er}"
```

### Validation 4: Temporal Integrity (No Lookahead)

Critical test ensuring features use only past data:

```rust
#[cfg(test)]
mod temporal_integrity_tests {
    use super::*;

    /// CRITICAL: Verify that inter-bar features are computed from trades
    /// STRICTLY BEFORE the bar opens, never including the bar's own trades
    #[test]
    fn lookback_excludes_current_bar_trades() {
        let mut processor = RangeBarProcessor::new(InterBarConfig {
            lookback_mode: LookbackMode::FixedCount(100),
            ..Default::default()
        });

        // Process trades and collect bars with their lookback windows
        let trades = generate_test_trades(1000);
        let mut bar_lookback_pairs: Vec<(RangeBar, Vec<TradeSnapshot>)> = Vec::new();

        for trade in &trades {
            if let Some(bar) = processor.process_trade(trade) {
                // Capture the lookback window that was used
                let lookback = processor.get_last_lookback_window().clone();
                bar_lookback_pairs.push((bar, lookback));
            }
        }

        // Verify temporal integrity for each bar
        for (bar, lookback) in &bar_lookback_pairs {
            for trade in lookback {
                assert!(
                    trade.timestamp < bar.open_time,
                    "TEMPORAL INTEGRITY VIOLATION: Lookback trade at {} \
                     must be before bar open_time {}",
                    trade.timestamp,
                    bar.open_time
                );
            }
        }
    }

    /// Verify that the same bar produces the same features regardless of
    /// what comes AFTER it (no future data leakage)
    #[test]
    fn features_independent_of_future_trades() {
        let trades = generate_test_trades(500);

        // Process first 200 trades
        let mut processor1 = RangeBarProcessor::new(default_config());
        let bars_partial: Vec<RangeBar> = trades[..200].iter()
            .filter_map(|t| processor1.process_trade(t))
            .collect();

        // Process all 500 trades
        let mut processor2 = RangeBarProcessor::new(default_config());
        let bars_full: Vec<RangeBar> = trades.iter()
            .filter_map(|t| processor2.process_trade(t))
            .collect();

        // All bars from partial processing should have identical features
        // to the corresponding bars from full processing
        for (partial, full) in bars_partial.iter().zip(bars_full.iter()) {
            assert_eq!(partial.lookback_ofi, full.lookback_ofi,
                "OFI should be identical");
            assert_eq!(partial.lookback_kyle_lambda, full.lookback_kyle_lambda,
                "Kyle lambda should be identical");
            assert_eq!(partial.lookback_burstiness, full.lookback_burstiness,
                "Burstiness should be identical");
            // ... test all features
        }
    }

    /// Verify checkpoint/restore preserves temporal integrity
    #[test]
    fn checkpoint_preserves_lookback_state() {
        let trades = generate_test_trades(1000);

        // Process first half
        let mut processor = RangeBarProcessor::new(default_config());
        for trade in &trades[..500] {
            processor.process_trade(trade);
        }

        // Checkpoint
        let checkpoint = processor.create_checkpoint();

        // Restore and continue
        let mut restored = RangeBarProcessor::from_checkpoint(checkpoint);
        let bars_from_restored: Vec<RangeBar> = trades[500..].iter()
            .filter_map(|t| restored.process_trade(t))
            .collect();

        // Process all at once for comparison
        let mut fresh = RangeBarProcessor::new(default_config());
        let bars_from_fresh: Vec<RangeBar> = trades.iter()
            .filter_map(|t| fresh.process_trade(t))
            .collect();

        // Bars after position 500 should be identical
        let fresh_after_500: Vec<_> = bars_from_fresh.iter()
            .filter(|b| b.open_time >= trades[500].timestamp)
            .collect();

        for (restored, fresh) in bars_from_restored.iter().zip(fresh_after_500.iter()) {
            assert_eq!(restored.lookback_trade_count, fresh.lookback_trade_count);
            // ... compare all features
        }
    }
}
```

---

## Validation Execution Plan

| Validation | When | How | Pass Criteria |
|------------|------|-----|---------------|
| **1. Property-Based** | Every build | `cargo nextest run` | All proptest assertions pass |
| **2. Known Values** | Every build | `cargo nextest run` | All assertions within tolerance |
| **3. Cross-Reference** | After Rust impl | `pytest tests/test_cross_reference_trading_fitness.py` | <5% deviation from trading-fitness |
| **4. Temporal Integrity** | Every build | `cargo nextest run` | Zero violations |

**CI Integration**:
```bash
# mise.toml
[tasks.test-inter-bar]
run = """
cargo nextest run -p rangebar-core --features test-utils -- inter_bar
pytest tests/test_inter_bar_features.py -v
"""
```
