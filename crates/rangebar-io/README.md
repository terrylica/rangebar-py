# rangebar-io

I/O operations and export formats with Polars DataFrame integration.

## Overview

`rangebar-io` provides export functionality for range bars in multiple formats (CSV, Parquet, Arrow IPC) with Polars integration for high-performance DataFrame operations and Python interoperability.

## Features

- **Multiple Export Formats**: CSV, Parquet, Arrow IPC
- **Polars Integration**: DataFrame operations with zero-copy Python transfer
- **Streaming CSV Export**: Memory-bounded export for large datasets
- **Compression**: Parquet compression achieves 70%+ file size reduction

## Exporters

### PolarsExporter

General-purpose exporter supporting multiple formats:

```rust
use rangebar_io::PolarsExporter;

let exporter = PolarsExporter::new();

// Export to Parquet (70%+ compression)
exporter.export_parquet(&bars, "output.parquet")?;

// Export to Arrow IPC (zero-copy Python)
exporter.export_arrow_ipc(&bars, "output.arrow")?;

// Export to CSV
exporter.export_streaming_csv(&bars, "output.csv")?;
```

### ParquetExporter

Dedicated Parquet exporter with optimized compression:

```rust
use rangebar_io::ParquetExporter;

let exporter = ParquetExporter::new();
exporter.export(&bars, "output.parquet")?;
```

### ArrowExporter

Arrow IPC exporter for zero-copy Python integration:

```rust
use rangebar_io::ArrowExporter;

let exporter = ArrowExporter::new();
exporter.export(&bars, "output.arrow")?;
```

### StreamingCsvExporter

Memory-bounded streaming CSV export:

```rust
use rangebar_io::StreamingCsvExporter;

let mut exporter = StreamingCsvExporter::new("output.csv".into())?;

for bar in bars {
    exporter.write_bar(&bar)?;
}
```

## DataFrame Conversion

Convert `Vec<RangeBar>` to Polars DataFrame:

```rust
use rangebar_io::formats::DataFrameConverter;

let df = bars.to_vec().to_polars_dataframe()?;

// DataFrame operations
let sorted_df = df.sort(["open_time"], Default::default())?;
let filtered_df = df.filter(&df.column("volume")?.gt(1000)?)?;
```

## Python Interoperability

### Arrow IPC (Zero-Copy)

Export to Arrow IPC for zero-copy Python loading:

```rust
// Rust: Export to Arrow
let exporter = ArrowExporter::new();
exporter.export(&bars, "bars.arrow")?;
```

```python
# Python: Zero-copy load with PyArrow
import pyarrow as pa

with pa.ipc.open_file("bars.arrow") as f:
    table = f.read_all()
    df = table.to_pandas()
```

### Parquet (Compressed)

Export to Parquet for compressed storage:

```rust
// Rust: Export to Parquet
let exporter = ParquetExporter::new();
exporter.export(&bars, "bars.parquet")?;
```

```python
# Python: Load with Polars or Pandas
import polars as pl

df = pl.read_parquet("bars.parquet")
```

## Performance Benchmarks

Use `polars-benchmark` binary to validate performance claims:

```bash
cargo run --bin polars-benchmark --features polars-io -- \
  --input ./data/BTCUSDT_bars.csv \
  --output-dir ./benchmark_output
```

**Expected Performance**:

- 70%+ file size reduction (Parquet vs CSV)
- 10x-20x faster Python loading (Arrow zero-copy)
- 2x-5x faster export operations (streaming)

## Dependencies

- **rangebar-core** - Core types
- **polars** `^0.51.0` - DataFrame operations (feature-gated)
- **arrow** - Arrow format support
- **parquet** - Parquet format support

## Feature Flags

Enable Polars integration with the `polars-io` feature:

```toml
[dependencies]
rangebar-io = { version = "6.1.0", features = ["polars-io"] }
```

## Version

Current version: **6.1.0** (modular crate architecture with checkpoint system)

## Documentation

- Architecture: `../../docs/ARCHITECTURE.md`
- Benchmarks: Run `polars-benchmark` binary

## License

See LICENSE file in the repository root.
