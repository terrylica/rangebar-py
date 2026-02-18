# rangebar-streaming

Real-time streaming engine for range bar construction over Binance WebSocket feeds, with bounded memory, circuit breaker pattern, and checkpoint recovery.

[![crates.io](https://img.shields.io/crates/v/rangebar-streaming.svg)](https://crates.io/crates/rangebar-streaming)
[![docs.rs](https://docs.rs/rangebar-streaming/badge.svg)](https://docs.rs/rangebar-streaming)

## Installation

### From crates.io

```toml
[dependencies]
rangebar-streaming = "12"
```

### From git (latest)

```toml
[dependencies]
rangebar-streaming = { git = "https://github.com/terrylica/rangebar-py", path = "crates/rangebar-streaming" }
```

### With Binance WebSocket integration

```toml
[dependencies]
rangebar-streaming = { version = "12", features = ["binance-integration"] }
```

## Quick Start

### Stream Processing

```rust
use rangebar_streaming::StreamingProcessor;
use futures::stream::StreamExt;

// Create processor: 250 dbps = 0.25% threshold
let mut processor = StreamingProcessor::new(250)?;

// Process an async trade stream
let metrics = processor.process_stream(agg_trade_stream).await?;

println!("Processed {} trades → {} bars", metrics.trades_processed, metrics.bars_generated);
```

### Live Bar Engine (multiplexed WebSocket)

```rust
use rangebar_streaming::LiveBarEngine;

// Process multiple symbols × thresholds concurrently
let engine = LiveBarEngine::new(
    vec!["BTCUSDT".into(), "ETHUSDT".into()],
    vec![250, 500],
    true, // include microstructure features
);

engine.start();

loop {
    if let Some(bar) = engine.next_bar(timeout_ms: 5000) {
        let symbol = bar.get("_symbol");
        let threshold = bar.get("_threshold");
        println!("{symbol}@{threshold}: close={}", bar.get("close"));
    }
}
```

### Custom Configuration

```rust
use rangebar_streaming::{StreamingProcessor, StreamingProcessorConfig};

let config = StreamingProcessorConfig {
    max_buffer_size: 50_000,
    circuit_breaker_threshold: 100,
    metrics_interval_secs: 60,
    ..Default::default()
};

let mut processor = StreamingProcessor::with_config(250, config)?;
```

## Features

| Feature               | Default | Description                                  |
| --------------------- | ------- | -------------------------------------------- |
| `binance-integration` | No      | Binance WebSocket stream support             |
| `stats`               | No      | Rolling statistics (rolling-stats, tdigests) |
| `indicators`          | No      | Technical indicators                         |
| `all`                 | No      | Enable all features                          |

## Architecture

```
Binance WS → LiveBarEngine (tokio, multiplexed)
                    │
                    ├─ RangeBarProcessor (per symbol × threshold)
                    │  └─ process_single_trade() → Option<CompletedBar>
                    │
                    └─ next_bar() → completed bars to consumer
```

- **Bounded memory**: Configurable buffer size prevents exhaustion
- **Circuit breaker**: Fault tolerance with automatic error threshold detection
- **Graceful shutdown**: `engine.stop()` + `engine.collect_checkpoints()` for recovery

## MSRV

Minimum supported Rust version: **1.90**

## License

MIT
