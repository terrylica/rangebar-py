# ADR: Real-Time Streaming API Architecture

**Date**: 2026-01-31
**Status**: Proposed
**Deciders**: Terry Li

## Context

The rangebar-py library has production-ready streaming infrastructure in Rust:

- `BinanceWebSocketStream` in `rangebar-providers::binance::websocket`
- `StreamingProcessor` in `rangebar-streaming::processor`

However, this infrastructure is **not exposed to Python**. Users requesting real-time range bar construction from live data sources (Binance aggTrades, Exness forex ticks) cannot access these capabilities.

### Current State

| Component                | Rust Status         | Python Status     |
| ------------------------ | ------------------- | ----------------- |
| `BinanceWebSocketStream` | ✅ Production-ready | ❌ Not exposed    |
| `StreamingProcessor`     | ✅ Production-ready | ❌ Not exposed    |
| Backpressure handling    | ✅ Bounded channels | ❌ Not exposed    |
| Circuit breaker          | ✅ Implemented      | ❌ Not exposed    |
| Exness tick streaming    | ❌ Not implemented  | ❌ Not applicable |

### Requirements

1. **Binance Live Streaming**: Connect to `wss://stream.binance.com:9443/ws/{symbol}@aggTrade` and emit range bars in real-time
2. **Exness Live Streaming**: Connect to Exness tick data feed and emit range bars for forex pairs
3. **Python Async Support**: Expose as `async def` functions or async generators for modern Python async/await patterns
4. **Fault Tolerance**: Automatic reconnection with exponential backoff
5. **Checkpoint Integration**: Resume from checkpoint after disconnect/restart
6. **Memory Bounded**: Maintain bounded memory even with infinite streams

## Decision

### Architecture: Callback-Based with Optional Async Wrapper

We will use a **callback-based architecture** exposed via PyO3, with an optional Python async wrapper built on top.

```
┌─────────────────────────────────────────────────────────────┐
│  Python Layer (python/rangebar/streaming/)                  │
├─────────────────────────────────────────────────────────────┤
│  stream_binance_live()     → AsyncGenerator[RangeBar]       │
│  stream_exness_live()      → AsyncGenerator[RangeBar]       │
│  RangeBarStreamHandler     → Callback interface             │
└────────────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│  PyO3 Bindings (src/lib.rs)                                 │
├─────────────────────────────────────────────────────────────┤
│  PyBinanceStream           → WebSocket + processor          │
│  PyExnessStream            → Tick feed + processor          │
│  PyStreamingProcessor      → Wraps StreamingProcessor       │
└────────────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Rust Layer                                                 │
├─────────────────────────────────────────────────────────────┤
│  BinanceWebSocketStream    → rangebar-providers             │
│  StreamingProcessor        → rangebar-streaming             │
│  ExnessTickStream          → NEW in rangebar-providers      │
└─────────────────────────────────────────────────────────────┘
```

### Rationale

**Why callback-based over pyo3-asyncio?**

1. **Stability**: pyo3-asyncio is experimental and version-sensitive. Pure callback pattern is stable across PyO3 versions.
2. **Flexibility**: Callbacks work with any Python async framework (asyncio, trio, anyio).
3. **Simplicity**: Rust side uses blocking recv, Python side handles async conversion.
4. **Testability**: Callback interface is easier to mock and test.

**Alternative Considered**: pyo3-asyncio

- Would require `pyo3-asyncio = "0.22"` dependency
- Locks to specific async runtime (usually asyncio)
- Version coupling with PyO3 major releases
- Rejected: Too experimental for production use

### API Design

#### Low-Level (Callback-Based)

```python
from rangebar import PyBinanceStream, PyStreamCallback

class MyHandler(PyStreamCallback):
    def on_bar(self, bar: dict) -> None:
        print(f"New bar: {bar['close']}")

    def on_error(self, error: str) -> None:
        print(f"Error: {error}")

    def on_disconnect(self) -> None:
        print("Disconnected")

stream = PyBinanceStream("BTCUSDT", threshold_decimal_bps=250)
stream.set_callback(MyHandler())
stream.connect()  # Blocking until disconnect
```

#### High-Level (Async Generator)

```python
from rangebar import stream_binance_live

async def main():
    async for bar in stream_binance_live("BTCUSDT", threshold_bps=250):
        print(f"New bar: {bar['close']}")

asyncio.run(main())
```

The async generator is implemented in Python using `asyncio.Queue` and a background thread running the callback-based stream.

### Reconnection Strategy

```
┌──────────────────────────────────────────────────────────────┐
│  Reconnection with Exponential Backoff                       │
├──────────────────────────────────────────────────────────────┤
│  1. Disconnect detected                                      │
│  2. Save checkpoint (incomplete bar + state)                 │
│  3. Wait: min(2^attempt × 1s, 60s)                          │
│  4. Reconnect to WebSocket                                   │
│  5. Restore from checkpoint                                  │
│  6. Resume processing                                        │
│  7. Reset backoff counter on successful processing           │
└──────────────────────────────────────────────────────────────┘
```

### Memory Management

- **Trade buffer**: 5,000 trades (existing `StreamingProcessorConfig`)
- **Bar buffer**: 100 bars (consumer backpressure)
- **Checkpoint interval**: Every completed bar
- **Memory threshold**: 100MB (existing config)

### Exness Implementation

Exness requires custom handling for forex tick data:

1. **Data Source**: Identify Exness WebSocket or REST API for live ticks
2. **Tick Format**: Bid/ask prices, not trade prices
3. **Mid-Price Conversion**: Use `(bid + ask) / 2` as trade price
4. **Volume Estimation**: Tick count or spread-based estimation
5. **Session Handling**: Forex market hours (Sydney → Tokyo → London → NY)

## Consequences

### Positive

- Production-ready real-time range bar construction
- Leverages existing Rust infrastructure (no new algorithms)
- Memory-bounded for infinite streams
- Works with any Python async framework
- Checkpoint-based recovery for reliability

### Negative

- Callback-based API is less Pythonic than native async
- Requires Python wrapper for async generator pattern
- Two layers of abstraction (Rust → callback → async)
- Exness requires additional research for live data API

### Risks

1. **WebSocket Rate Limits**: Binance may rate-limit connections
   - Mitigation: Document limits, implement connection pooling if needed
2. **Exness API Availability**: Unknown if live tick API exists
   - Mitigation: Research API docs, may need broker-specific integration
3. **GIL Contention**: Callback from Rust thread may block Python GIL
   - Mitigation: Use `pyo3::Python::allow_threads` for Rust processing

## Implementation Plan

1. **Phase 1**: Expose existing Rust infrastructure via PyO3
   - `PyBinanceStream` (WebSocket + processor)
   - `PyStreamingProcessor` (standalone processor)
   - Callback interface (`PyStreamCallback` trait)

2. **Phase 2**: Python async wrapper
   - `stream_binance_live()` async generator
   - Reconnection logic in Python
   - Integration with existing checkpoint API

3. **Phase 3**: Exness integration
   - Research Exness live data API
   - Implement `ExnessTickStream` in Rust
   - PyO3 bindings and Python wrapper

## References

- [rangebar-streaming processor.rs](../crates/rangebar-streaming/src/processor.rs)
- [rangebar-providers websocket.rs](../crates/rangebar-providers/src/binance/websocket.rs)
- [PyO3 Guide - Python Classes](https://pyo3.rs/v0.22.0/class.html)
- [Binance WebSocket Streams](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams)
