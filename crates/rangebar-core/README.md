# rangebar-core

Core algorithm for non-lookahead range bar construction from tick data, with 10 intra-bar and 16 inter-bar microstructure features computed during bar construction.

[![crates.io](https://img.shields.io/crates/v/rangebar-core.svg)](https://crates.io/crates/rangebar-core)
[![docs.rs](https://docs.rs/rangebar-core/badge.svg)](https://docs.rs/rangebar-core)

## Installation

### From crates.io

```toml
[dependencies]
rangebar-core = "12"
```

### From git (latest)

```toml
[dependencies]
rangebar-core = { git = "https://github.com/terrylica/rangebar-py", path = "crates/rangebar-core" }
```

## Quick Start

```rust
use rangebar_core::{RangeBarProcessor, AggTrade, FixedPoint};

// Create processor: 250 dbps = 0.25% threshold
let mut processor = RangeBarProcessor::new(250)?;

// Process a batch of trades
let bars = processor.process_agg_trade_records(&trades)?;

for bar in &bars {
    println!(
        "O={} H={} L={} C={} V={} trades={}",
        bar.open.to_f64(), bar.high.to_f64(),
        bar.low.to_f64(), bar.close.to_f64(),
        bar.volume.to_f64(), bar.trade_count,
    );
}
```

### Streaming (trade-by-trade)

```rust
use rangebar_core::RangeBarProcessor;

let mut processor = RangeBarProcessor::new(250)?;

for trade in trades {
    if let Some(completed_bar) = processor.process_single_trade(trade) {
        // Bar completed — write to storage, emit to downstream
        store_bar(&completed_bar);
    }
}

// Check for incomplete bar at end of session
if let Some(incomplete) = processor.get_incomplete_bar() {
    println!("Incomplete bar: open={}", incomplete.open.to_f64());
}
```

### Checkpoint (crash recovery)

```rust
use rangebar_core::RangeBarProcessor;
use rangebar_core::checkpoint::Checkpoint;

let mut processor = RangeBarProcessor::new(250)?;

// ... process some trades ...

// Save state
let checkpoint = processor.create_checkpoint();
let json = serde_json::to_string(&checkpoint)?;

// Later: restore from checkpoint
let restored_checkpoint: Checkpoint = serde_json::from_str(&json)?;
let mut restored = RangeBarProcessor::from_checkpoint(restored_checkpoint)?;

// Continue processing from where we left off
let bars = restored.process_agg_trade_records(&more_trades)?;
```

## Threshold Units

All thresholds use **decimal basis points (dbps)**: 1 dbps = 0.001% = 0.00001.

| dbps | Percentage | Use Case         |
| ---- | ---------- | ---------------- |
| 100  | 0.10%      | Tight (scalping) |
| 250  | 0.25%      | Standard         |
| 500  | 0.50%      | Wide (swing)     |
| 1000 | 1.00%      | Very wide        |

## Features

| Feature      | Default | Description                            |
| ------------ | ------- | -------------------------------------- |
| `test-utils` | No      | CSV loading for tests (adds `csv` dep) |
| `python`     | No      | PyO3 type exports                      |
| `api`        | No      | utoipa OpenAPI schemas                 |
| `arrow`      | No      | Arrow RecordBatch export               |

## Microstructure Features

10 intra-bar features computed during bar construction (zero additional passes):

`ofi`, `vwap_close_deviation`, `kyle_lambda_proxy`, `trade_intensity`, `volume_per_trade`, `aggression_ratio`, `aggregation_density`, `turnover_imbalance`, `duration_us`, `price_impact`

16 inter-bar lookback features from trades before each bar opens:

`lookback_ofi`, `lookback_intensity`, `lookback_kyle_lambda`, `lookback_burstiness`, `lookback_hurst`, `lookback_permutation_entropy`, and 10 more.

## MSRV

Minimum supported Rust version: **1.90**

## License

MIT
