# Profile-Guided Optimization (PGO) Workflow

**Status**: Phase 1b - Compilation Optimization
**Expected Speedup**: Additional 10-20% on top of LTO (cumulative 20-35% vs baseline)
**Issue**: #96 Task #20

---

## Overview

Profile-Guided Optimization uses runtime profiling data to guide compiler optimizations. This achieves better branch prediction, inlining decisions, and register allocation than static analysis alone.

### Why PGO for rangebar?

- **Hot paths**: Permutation entropy, Hurst exponent computation called millions of times per bar
- **Branch prediction**: Complex conditional logic benefits from profile-directed optimization
- **Real-world data**: Profiling on representative Binance trade data makes optimization decisions realistic

---

## Prerequisites

- Rust 1.90+ (already pinned in `rust-version` in Cargo.toml)
- LLVM 12+ (bundled with Rust toolchain)
- ~2 GB free disk space (for instrumented binary + profiling data)

---

## Quick Start (Manual Workflow)

### Step 1: Collect PGO Profile

```bash
# Set environment for PGO data collection
export LLVM_PROFILE_FILE="pgo-data/rangebar-%m.profraw"

# Compile instrumented binary (Rust 1.90+ native support)
RUSTFLAGS="-Cprofile-generate=./pgo-data -Cllvm-args=-pgo-warn-missing-function" \
  cargo build -p rangebar-py --release

# Run representative workload to collect profiles
# This should exercise the hot paths in your application
uv run python3 -c "
from rangebar import get_range_bars
import pandas as pd

# Process 100K trades (1-hour BTCUSDT data at ~500 bps threshold)
df = get_range_bars('BTCUSDT', '2024-06-15', '2024-06-15')
print(f'Processed {len(df)} bars')
"

# Merge profiling data
llvm-tools-preview package provides llvm-tools with llvm-tools
# Usually at: ~/.rustup/toolchains/1.90.0-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/bin/llvm-profdata

LLVM_PROFDATA_PATH=$(rustc --print sysroot)/lib/rustlib/$(rustc -Vv | grep host | awk '{print $NF}')/bin/llvm-profdata

$LLVM_PROFDATA_PATH merge -o pgo-data/merged.profdata pgo-data/*.profraw
```

### Step 2: Optimize with PGO Data

```bash
# Compile optimized binary using profiling data
RUSTFLAGS="-Cprofile-use=./pgo-data/merged.profdata -Cllvm-args=-pgo-warn-missing-function" \
  cargo build -p rangebar-py --release

# Install optimized wheel
maturin develop --release
```

### Step 3: Verify Speedup

```bash
# Run benchmarks
cargo bench -p rangebar-core --bench profiling_analysis

# Expected results: 10-20% faster than LTO-only baseline
```

---

## Automated Workflow (via mise tasks)

```bash
# Full PGO cycle (collect, merge, optimize)
mise run pgo:profile-full

# Collect profiles only
mise run pgo:collect

# Merge profiling data
mise run pgo:merge

# Rebuild with PGO data
mise run pgo:optimize

# Benchmarks
mise run bench:pgo
```

---

## Configuration Files

### `.cargo/config.local` (Example)

```toml
[build]
# PGO collection configuration
rustflags = [
    "-Cprofile-generate=./pgo-data",
    "-Cllvm-args=-pgo-warn-missing-function",
]

# Or for PGO optimization:
# rustflags = [
#     "-Cprofile-use=./pgo-data/merged.profdata",
#     "-Cllvm-args=-pgo-warn-missing-function",
# ]
```

**DO NOT commit `.cargo/config.local`** — it's environment-specific.
Use `.cargo/config.local.example` as reference.

---

## Profiling Recommendations

### What to Profile?

Profile on **representative, realistic workloads**:

- **Tier 1** (essential): 100K BTCUSDT trades (1 symbol, 1 day)
- **Tier 2** (comprehensive): 500K trades across 5 symbols (BTC, ETH, BNB, SOL, XRP)
- **Tier 3** (exhaustive): 1M trades + edge cases (gaps, circuit breakers, liquidations)

### Duration

- **Minimum**: 30 seconds profiling time (hot path stabilization)
- **Recommended**: 5-10 minutes (captures diurnal trade patterns)
- **Too long**: >30 minutes (diminishing returns, excessive I/O)

### Workload Scaling

```python
# Tier 1: Fast profiling (1 day BTCUSDT)
df = get_range_bars('BTCUSDT', '2024-06-15', '2024-06-15')  # ~1M trades

# Tier 2: Comprehensive (1 week, 5 symbols)
for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLSOL', 'XRPUSDT']:
    df = get_range_bars(symbol, '2024-06-10', '2024-06-16')

# Tier 3: Exhaustive (1 month coverage)
df = get_range_bars('BTCUSDT', '2024-06-01', '2024-06-30')
```

---

## Performance Impact & Metrics

| Phase        | Speedup    | Cumulative | Build Time | Notes                           |
| ------------ | ---------- | ---------- | ---------- | ------------------------------- |
| Baseline     | 1.0x       | 1.0x       | ~10s       | No optimizations                |
| + LTO (thin) | 1.10-1.15x | 1.10-1.15x | +30s       | ~40s total                      |
| + PGO        | 1.10-1.20x | 1.20-1.35x | +5-10m     | ~5.5m total (profiling + build) |

**Cumulative Impact**: 20-35% end-to-end speedup for typical workloads

---

## Troubleshooting

### PGO Data Not Merging

```bash
# Check that profraw files were created
ls -lah pgo-data/

# Manually merge with verbose output
llvm-profdata merge -o pgo-data/merged.profdata pgo-data/*.profraw -v
```

### Binary Crashes During Profiling

- Profile data collection adds ~10-20% overhead
- If binary OOMs, reduce workload size
- Ensure `profile-generate` flag is correctly set

### No Speedup Observed

- Profiling data may not match production workload
- Hot paths may be I/O bound (not CPU bound)
- LTO may have already optimized most gain areas

---

## Maintenance

### Re-profile After Changes

Re-profile if you modify:

- Hot-path algorithms (permutation entropy, Hurst, Kyle lambda)
- Feature computation loops
- Tier 2/3 feature implementations

**Do NOT re-profile for**:

- Python API changes
- CLI tool updates
- Test-only changes

### Cache Strategy

```bash
# Keep merged.profdata in version control? NO (it's generated)
# Keep pgo-data/ in .gitignore? YES

# For CI/CD: Re-run profiling on every release build
# For local dev: Cache profdata across rebuilds (--profile pgo-use)
```

---

## Advanced: Custom Profiling Workloads

For domain-specific optimization, create custom profiling scripts:

```python
# scripts/pgo_profile_workload.py
import sys
from rangebar import get_range_bars

def profile_typical_usage():
    """Simulate real-world trading system workload"""

    # Morning market open (high volatility)
    df = get_range_bars('BTCUSDT', '2024-06-15', '2024-06-15',
                        include_microstructure=True)

    # Multi-symbol correlation analysis
    for sym in ['ETHUSDT', 'BNBUSDT']:
        df = get_range_bars(sym, '2024-06-15', '2024-06-15')

    # Streaming edge case (rapid bar updates)
    for i in range(100):
        df = get_range_bars('BTCUSDT', '2024-06-15', '2024-06-15')

if __name__ == '__main__':
    profile_typical_usage()
    print("PGO profiling complete")
```

Then use in workflow:

```bash
# Collect profiles with custom workload
LLVM_PROFILE_FILE="pgo-data/rangebar-%m.profraw" \
  python scripts/pgo_profile_workload.py

# Merge and optimize as usual
llvm-profdata merge ...
RUSTFLAGS="-Cprofile-use=..." cargo build ...
```

---

## References

- [Rust PGO Guide](https://doc.rust-lang.org/nightly/rustc/profile-guided-optimization.html)
- [LLVM PGO Documentation](https://llvm.org/docs/GettingStarted.html#generating-api-documentation)
- [Rust Compiler Performance Book](https://nnethercote.github.io/perf-book/build-configuration.html)

---

## Next Steps

After PGO stabilizes, consider:

1. **Phase 2**: SIMD vectorization for entropy/Hurst (2-8x speedup, medium effort)
2. **Phase 3**: GPU acceleration (conditional on bottleneck analysis)
3. **Phase 4**: Streaming stats caching (1.5-3x for Tier 2+ features)
