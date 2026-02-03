---
title: Memory-Efficient Streaming Processing for Large-Scale Time Series Data
source_type: ai-research
scraped_at: 2026-02-03T16:09:57Z
model_name: claude-3-5-sonnet, gemini-3-pro
tools:
  - claude-artifacts
  - gemini-deep-research
claude_code_uuid: 0c08e926-80d5-4d69-9f96-9ea307441aa5
claude_code_project_path: /Users/terryli/eon/rangebar-py
github_issue_url: "https://github.com/terrylica/rangebar-py/issues/66"
related_issue: "#65"
tags:
  - memory-optimization
  - streaming
  - arrow
  - pyo3
  - adaptive-chunking
---

<!-- SSoT-OK: Version numbers below are external library references, not this project's version -->

# Memory-Efficient Streaming for Rust+Python Time Series Processing

**Processing 1M+ financial trades into range bars with 62 microstructure features causes OOM failures due to fixed chunk sizes, DataFrame accumulation, and pandas concat spikes.** Four complementary solutions eliminate these bottlenecks: adaptive chunking via fixed-point iteration, Arrow-based zero-copy accumulation, pyo3-arrow for Rust→Python transfer, and sysinfo-based memory guards.

---

## Executive Summary

| Solution                 | Impact                     | Implementation Status    |
| ------------------------ | -------------------------- | ------------------------ |
| **Adaptive Chunking**    | Prevents per-chunk OOM     | MEM-011 implemented      |
| **Arrow Accumulation**   | Eliminates 2x concat spike | Infrastructure exists    |
| **pyo3-arrow Zero-Copy** | 3x memory reduction        | Already integrated       |
| **Memory Guards**        | Parameter-free auto-tuning | resource_guard.py exists |

---

## Part 1: Claude Artifact Research (Implementation-Focused)

### Adaptive Chunking Eliminates Hardcoded Batch Sizes

The hardcoded 100K trades per chunk ignores both memory constraints and actual processing characteristics. Research from Spark Streaming's creators at UC Berkeley provides the most robust parameter-free solution: **Fixed-Point Iteration** converges to optimal batch size by measuring processing time and iteratively adjusting.

The algorithm finds the intersection where processing time equals a fraction of the batch interval using the update formula `x_{n+1} = processing_time(x_n) / ρ`, where ρ (typically **0.7-0.8**) provides stability margin.

**Polars' streaming engine** offers a simpler formula-based approach:

```
chunk_size = max(50_000 / n_columns * thread_factor, 1000)
```

where `thread_factor = max(12 / n_threads, 1)`. For 62-column financial data on 8 threads, this yields approximately **1,200 rows per chunk**.

The recommended hybrid implementation combines memory-based initialization with processing-time feedback:

```rust
fn adaptive_chunk_size(record_bytes: usize, available_memory: u64,
                       last_chunk: usize, last_proc_ms: f64, rho: f64) -> usize {
    let memory_max = (available_memory as f64 * 0.5 / record_bytes as f64) as usize;
    let throughput_target = (last_chunk as f64 * rho / last_proc_ms * 1000.0) as usize;
    memory_max.min(throughput_target).max(1000)
}
```

### Zero-Copy Arrow Accumulation Prevents the Concat Memory Spike

The **2x memory spike during `pd.concat()`** occurs because pandas allocates a new contiguous memory block, copies all data, then frees originals. Arrow's architecture fundamentally solves this through **ChunkedArray**.

**PyArrow's `concat_tables` with `promote_options="none"`** performs zero-copy concatenation by linking existing memory locations rather than copying.

Memory characteristics by approach:

| Approach             | During Accumulation       | At Finalization     |
| -------------------- | ------------------------- | ------------------- |
| pandas concat        | O(total)                  | **2x O(total)**     |
| Arrow concat_tables  | O(pointers)               | O(total) once       |
| Polars vstack        | O(pointers)               | O(total) at rechunk |
| Parquet StreamWriter | **O(row_group)** constant | Disk only           |

### pyo3-arrow Enables True Zero-Copy Rust→Python Transfer

The current dict-based transfer through PyO3 creates **3+ copies**: Rust data → Python dicts → pandas DataFrame construction. The **pyo3-arrow crate** eliminates this entirely using the Arrow PyCapsule Interface.

```rust
use pyo3_arrow::{PyRecordBatch, error::PyArrowResult};

#[pyfunction]
pub fn process_bars(py: Python) -> PyArrowResult<PyObject> {
    let batch = build_record_batch_with_62_columns();
    Ok(PyRecordBatch::new(batch).to_arro3(py)?)  // Zero-copy export
}
```

**All major Python data libraries now support PyCapsule Interface**: PyArrow ≥14, Polars ≥1.3, DuckDB, pandas ≥2.2 (export only).

**Expected improvement**: ~3x memory reduction and **10-20x faster** serialization for 50K rows × 62 columns.

### Memory Guards Adapt Automatically Through System Detection

The `sysinfo` crate provides cross-platform memory detection without magic numbers:

```rust
use sysinfo::System;
let mut sys = System::new_all();
sys.refresh_memory();
let available = sys.available_memory();
let pressure = sys.used_memory() as f64 / sys.total_memory() as f64;
```

**DuckDB's production pattern** sets memory limit to **80% of physical RAM** auto-detected at startup, with automatic spill-to-disk when exceeded.

---

## Part 2: Gemini 3 Pro Deep Research (Theoretical Foundations)

### The "Concatenation Cliff" and Virtual Memory

The observed 2.9GB spike during `pd.concat` is a deterministic consequence of how dynamic array concatenation functions in memory-managed languages. When a list of DataFrames is passed to a concatenation routine:

1. Calculate the total size S_total = Σ size(D_i)
2. Request a contiguous block of virtual memory of size S_total
3. Copy the data from each D_i into the new block

During this operation, the resident set size (RSS) effectively doubles to **2×S_total+δ**. This is the "Concatenation Cliff."

### Control Theory: TCP Vegas and Congestion

The problem of determining the optimal batch size without user intervention is isomorphic to determining the optimal Window Size in network transmission. **TCP Vegas** uses _latency_ as a precursor signal, making it ideal for preventing OOMs rather than recovering from them.

In our context:

- **Packet RTT** ≈ **Batch Processing Time**
- **Congestion Window (CWND)** ≈ **Batch Size (Number of Trades)**
- **BaseRTT** ≈ **Minimum Unit Processing Time (per trade)**

### The AIMD Algorithm

For a robust, parameter-free adaptation strategy, **Additive Increase, Multiplicative Decrease (AIMD)** provides a mathematically proven convergence property:

- **Additive Increase:** w(t+1) = w(t) + α
- **Multiplicative Decrease:** w(t+1) = w(t) × β (where β < 1)

This asymmetry is crucial. The linear increase prevents sudden shocks to the memory allocator, while the exponential decrease ensures immediate relief when the memory guard signals danger.

### Dynamic Block and Batch Sizing (DyBBS)

The **DyBBS** algorithm from IBM/Wayne State (IEEE ICAC 2016) uses isotonic regression on historical statistics rather than just the last two measurements. It finds the smallest batch interval where `IsotonicRegression(x) + c < x`, achieving **35-67% latency reduction** over fixed-point iteration alone.

### Handling Heteroscedasticity in Financial Data

Financial data is unique because the "size" of a chunk is not just the number of trades (input) but the number of range bars generated (output). A 100k trade chunk in a flat market might yield 10 bars. In a volatile market, it might yield 10,000 bars.

The adaptive algorithm must monitor **Output Size** in addition to Input Size:

- **Metric:** `Expansion Ratio` = Output Rows / Input Trades
- **Formula:** `S_next = Target Output Rows / Expansion Ratio`

### System Introspection & Memory Guards

**Pressure Stall Information (PSI)** at `/proc/pressure/memory`:

- **Metric:** `some` (avg10, avg60)
- **Meaning:** Percentage of time some task was stalled waiting for memory
- **Signal:** If `some avg10 > 0.00`, the system is under pressure

**The Guard Logic:** If PSI > 0, the Adaptive Batcher should immediately switch to Multiplicative Decrease (AIMD), halving the batch size.

---

## Implementation Roadmap

### Phase 1 (Immediate - Issue #65)

- **MEM-011**: Adaptive chunk size (100K → 50K with microstructure)

### Phase 2 (Short-term)

- Switch from dict accumulation to Arrow RecordBatch accumulation
- Use existing `process_trades_streaming_arrow()` → `pl.from_arrow()` → `pl.concat()`

### Phase 3 (Medium-term)

- Change `_process_binance_trades()` to use Arrow path exclusively
- Document migration in CLAUDE.md

### Phase 4 (Long-term)

- Implement fully adaptive batching with TCP Vegas/AIMD
- Add Expansion Ratio monitoring for heteroscedastic financial data

---

## Key References

### Academic Papers

- [Berkeley EECS-2014-133](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2014/EECS-2014-133.pdf) - "Adaptive Stream Processing using Dynamic Batch Sizing" (Das, Zhong, Stoica, Shenker)
- [IEEE ICAC 2016](https://ieeexplore.ieee.org/document/7573114/) - "Adaptive Block and Batch Sizing for Batched Stream Processing System" (Zhang, Song, Routray, Shi)

### Specifications

- [Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
- [Polars Streaming](https://docs.pola.rs/api/python/dev/reference/api/polars.LazyFrame.sink_parquet.html)

### Crates

- [pyo3-arrow](https://crates.io/crates/pyo3-arrow) - Zero-copy Rust↔Python via Arrow PyCapsule
- [sysinfo](https://crates.io/crates/sysinfo) - Cross-platform system information
- [tikv-jemalloc-ctl](https://crates.io/crates/tikv-jemalloc-ctl) - jemalloc introspection APIs

### Blog Posts

- [Rho Signal: Sinking larger-than-memory Parquet files](https://www.rhosignal.com/posts/sink-parquet-files/)
- [Rho Signal: Streaming large datasets in Polars](https://www.rhosignal.com/posts/streaming-in-polars/)
- [DuckDB Memory Management](https://duckdb.org/2024/07/09/memory-management)

---

## Conclusion

The memory bottleneck in financial tick processing stems from three independent issues—each requiring a different solution class:

1. **Fixed-point iteration** with memory bounds provides truly adaptive chunking without per-workload tuning
2. **Arrow's ChunkedArray architecture** eliminates the concat spike through zero-copy linking
3. **pyo3-arrow's PyCapsule Interface** removes the multi-copy dict serialization overhead
4. **sysinfo-based watermarks** enable automatic backpressure without hardcoded thresholds

Together, these approaches transform a system with 2.9GB spikes into one with predictable, bounded memory usage—all without introducing external distributed systems or sacrificing API compatibility.
