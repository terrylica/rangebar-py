# Cache Population API Reference

**Navigation**: [INDEX.md](./INDEX.md) | [Primary API](./primary-api.md) | [Processing API](./processing-api.md)

---

## populate_cache_resumable()

**Required for date ranges > 30 days. Memory-safe incremental caching with bar-level resumability.**

```python
def populate_cache_resumable(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    threshold_decimal_bps: int = 250,
    force_refresh: bool = False,
    include_microstructure: bool = False,
    ouroboros: Literal["year", "month", "week"] = "year",
    checkpoint_dir: str | None = None,
    notify: bool = True,
    verbose: bool = True,
) -> int
```

### Parameters

- **symbol**: Trading symbol (e.g., "BTCUSDT")
- **start_date**: Start date in YYYY-MM-DD format
- **end_date**: End date in YYYY-MM-DD format
- **threshold_decimal_bps**: Threshold in decimal basis points or preset name (default: 250)
- **force_refresh**: Wipe existing cache and checkpoint, start fresh (default: False)
- **include_microstructure**: Include microstructure features (default: False)
- **ouroboros**: Reset mode for reproducibility (default: "year")
- **checkpoint_dir**: Custom checkpoint directory (default: platform-specific)
- **notify**: Send progress notifications (default: True)
- **verbose**: Show tqdm progress bar and structured logging (default: True)

### Returns

- **int**: Total number of bars written

### Resumability

Checkpoints are saved after each day to both:

- **Local**: `~/.cache/rangebar/checkpoints/` (fast access)
- **ClickHouse**: `population_checkpoints` table (cross-machine resume)

The checkpoint includes processor state, preserving incomplete bars across interrupts.

### Progress Logging

When `verbose=True` (default), the function shows:

- **tqdm progress bar**: Real-time ETA estimation with day-level progress
- **Structured NDJSON logging**: Events logged to `logs/events.jsonl`

```
BTCUSDT:  45%|████████▌          | 274/608 [02:15<02:45, 2.02days/s]
```

Progress bar shows:

- Symbol name as description
- Percentage complete
- Days processed / total days
- Elapsed time and ETA
- Processing rate (days per second)

Disable for batch/CI environments:

```python
populate_cache_resumable(
    "BTCUSDT",
    "2019-01-01",
    "2025-12-31",
    verbose=False,  # Silent mode for scripts
)
```

### Example

```python
from rangebar import populate_cache_resumable, get_range_bars

# Multi-year backfill (can be interrupted and resumed)
bars = populate_cache_resumable(
    "BTCUSDT",
    "2019-01-01",
    "2025-12-31",
    threshold_decimal_bps=250,
)
print(f"Populated {bars:,} bars")

# Now read from cache (fast, memory-safe)
df = get_range_bars("BTCUSDT", "2019-01-01", "2025-12-31")

# Force restart (wipe and repopulate)
populate_cache_resumable(
    "BTCUSDT",
    "2019-01-01",
    "2025-12-31",
    force_refresh=True,
)
```

### Raises

- **ValueError**: Invalid parameters (dates, threshold, symbol)
- **RuntimeError**: Fetch or processing failure
- **ConnectionError**: ClickHouse unavailable

---

## Why Cache Workflow for Long Ranges?

Date ranges > 30 days require the cache workflow (MEM-013 guard) to prevent OOM:

```python
# This fails with ValueError for >30 day ranges:
df = get_range_bars("BTCUSDT", "2019-01-01", "2025-12-31")  # OOM protection!

# Correct approach:
from rangebar import populate_cache_resumable, get_range_bars

# Step 1: Populate cache (resumable, memory-safe)
populate_cache_resumable("BTCUSDT", "2019-01-01", "2025-12-31")

# Step 2: Read from cache
df = get_range_bars("BTCUSDT", "2019-01-01", "2025-12-31")
```

This prevents OOM by forcing incremental day-by-day processing with checkpoints. The `populate_cache_resumable()` function:

1. Processes data day-by-day (memory-safe)
2. Saves checkpoints after each day (resumable)
3. Stores bars incrementally to ClickHouse (cross-machine)
4. Preserves incomplete bar state across interrupts

---

## Cache Configuration

### LONG_RANGE_DAYS Constant

```python
from rangebar import LONG_RANGE_DAYS

# Threshold for MEM-013 guard
print(LONG_RANGE_DAYS)  # 30
```

### Checkpoint Storage

| Storage    | Location                         | Purpose              |
| ---------- | -------------------------------- | -------------------- |
| Local      | `~/.cache/rangebar/checkpoints/` | Fast local resume    |
| ClickHouse | `population_checkpoints` table   | Cross-machine resume |

### Related ClickHouse Tables

| Table                    | Purpose                            |
| ------------------------ | ---------------------------------- |
| `range_bars`             | Cached range bar data              |
| `population_checkpoints` | Resume state for long-running jobs |

---

## Workflow Patterns

### Pattern 1: Initial Population

```python
# First-time population of a new symbol/threshold
bars = populate_cache_resumable(
    "BTCUSDT",
    "2020-01-01",
    "2025-12-31",
    threshold_decimal_bps=250,
)
```

### Pattern 2: Incremental Update

```python
# Extend existing cache with new data
# (automatically resumes from last checkpoint)
bars = populate_cache_resumable(
    "BTCUSDT",
    "2020-01-01",  # Original start
    "2026-06-30",  # Extended end
    threshold_decimal_bps=250,
)
```

### Pattern 3: Full Refresh

```python
# Wipe and regenerate all data
bars = populate_cache_resumable(
    "BTCUSDT",
    "2020-01-01",
    "2025-12-31",
    threshold_decimal_bps=250,
    force_refresh=True,  # Clears cache and checkpoint
)
```

### Pattern 4: With Microstructure

```python
# Include v7.0+ microstructure features
bars = populate_cache_resumable(
    "BTCUSDT",
    "2024-01-01",
    "2024-12-31",
    include_microstructure=True,
)
```

---

## Monitoring Progress

Progress is logged and can be monitored via hooks:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Now populate_cache_resumable() will log progress
bars = populate_cache_resumable("BTCUSDT", "2024-01-01", "2024-06-30")
# INFO: Processing BTCUSDT [1/182]: 2024-01-01
# INFO: Processing BTCUSDT [2/182]: 2024-01-02
# ...
```

---

**Last Updated**: 2026-02-03
