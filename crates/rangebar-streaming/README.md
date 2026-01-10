# rangebar-streaming

Real-time streaming processor for range bars with bounded memory and circuit breaker pattern.

## Overview

`rangebar-streaming` provides a streaming implementation of the range bar algorithm optimized for real-time tick processing with bounded memory usage. Suitable for live trading systems, WebSocket data ingestion, and continuous market analysis.

## Features

- **Bounded Memory**: Configurable buffer size prevents memory exhaustion
- **Circuit Breaker Pattern**: Fault tolerance with automatic error threshold detection
- **Real-Time Metrics**: Processing statistics for monitoring
- **Async Processing**: Tokio-based async stream processing
- **Graceful Degradation**: Handles errors without cascade failures

## Usage

### Basic Streaming

```rust
use rangebar_streaming::StreamingProcessor;
use futures::stream::StreamExt;

// Create processor with 25 BPS (0.25%) threshold
let mut processor = StreamingProcessor::new(250)?; // v3.0.0: 250 = 25 BPS

// Process async stream
let metrics = processor.process_stream(agg_trade_stream).await?;

println!("Processed {} trades", metrics.trades_processed);
println!("Generated {} bars", metrics.bars_generated);
```

### Custom Configuration

```rust
use rangebar_streaming::{StreamingProcessor, StreamingProcessorConfig};

let config = StreamingProcessorConfig {
    max_buffer_size: 50_000,           // Max 50k trades in buffer
    circuit_breaker_threshold: 100,     // Open circuit after 100 errors
    metrics_interval_secs: 60,          // Report metrics every 60s
    ..Default::default()
};

let mut processor = StreamingProcessor::with_config(250, config)?;
```

### Replay Buffer

For testing and development, use the replay buffer for time-aware playback:

```rust
use rangebar_streaming::replay::ReplayBuffer;

let mut buffer = ReplayBuffer::new(10_000); // 10k trade buffer

// Push trades
for trade in trades {
    buffer.push(trade);
}

// Get trades from specific timestamp
let recent_trades = buffer.get_trades_from(start_timestamp);
```

## Metrics

```rust
pub struct StreamingMetrics {
    pub trades_processed: u64,
    pub bars_generated: u64,
    pub processing_duration: Duration,
    pub throughput_trades_per_sec: f64,
    pub errors_encountered: u64,
    pub circuit_breaker_active: bool,
}
```

## Circuit Breaker

The circuit breaker pattern prevents cascade failures:

1. **Closed State**: Normal operation, all trades processed
2. **Open State**: Error threshold exceeded, processing halted
3. **Half-Open State**: Testing if errors cleared

```rust
// Circuit breaker activates when error count exceeds threshold
if metrics.circuit_breaker_active {
    eprintln!("Circuit breaker active - too many errors");
}
```

## Performance Characteristics

- **Memory Usage**: O(buffer_size) - bounded and configurable
- **Throughput**: Moderate (single-threaded async processing)
- **Latency**: Low (< 1ms per trade in normal conditions)
- **Parallelism**: Single-threaded (use rangebar-batch for multi-threaded)

## Comparison: Streaming vs Batch

| Feature         | Streaming       | Batch                  |
| --------------- | --------------- | ---------------------- |
| Memory Usage    | Bounded         | Unbounded              |
| Throughput      | Moderate        | High                   |
| Parallelism     | Single-threaded | Multi-threaded (Rayon) |
| Real-time       | Yes             | No                     |
| Use Case        | Live trading    | Historical analysis    |
| Circuit Breaker | Yes             | No                     |

## Dependencies

- **rangebar-core** - Core algorithm and types
- **rangebar-providers** - Data fetching
- **tokio** - Async runtime
- **futures** - Stream processing

## Version

Current version: **6.1.0** (modular crate architecture with checkpoint system)

## Documentation

- Architecture: `../../docs/ARCHITECTURE.md`
- Examples: `../../examples/interactive/`

## License

See LICENSE file in the repository root.
