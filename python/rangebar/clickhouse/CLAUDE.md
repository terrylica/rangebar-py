# ClickHouse Cache Layer

**Parent**: [/python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md) | **Schema**: [schema.sql](./schema.sql)

---

## Cache Population

### Native Method (Recommended)

Use `get_range_bars()` with `fetch_if_missing=True` to populate the cache:

```python
from rangebar import get_range_bars

# Populate threshold 100 data for BTCUSDT
df = get_range_bars(
    "BTCUSDT",
    start_date="2023-06-01",
    end_date="2025-12-01",
    threshold_decimal_bps=100,
    use_cache=True,
    fetch_if_missing=True,
    include_microstructure=False,
)
print(f"Fetched {len(df):,} bars")
```

**How it works**:

1. Checks ClickHouse cache for existing data
2. If missing, fetches raw tick data from Binance (day-by-day to prevent OOM)
3. Computes range bars via Rust backend
4. Stores results in `rangebar_cache.range_bars` table
5. Returns pandas DataFrame

### Populating Remote Hosts

Run on the target machine:

```bash
# SSH to host and run in tmux (long-running)
ssh <host> "tmux new-session -d -s rangebar-fetch 'cd ~/alpha-forge-research && source .venv/bin/activate && python3 << \"PYEOF\"
from rangebar import get_range_bars
import time

print(\"Fetching BTCUSDT threshold 100 data...\")
start = time.time()
df = get_range_bars(
    \"BTCUSDT\",
    start_date=\"2023-06-01\",
    end_date=\"2025-12-01\",
    threshold_decimal_bps=100,
    use_cache=True,
    fetch_if_missing=True,
)
elapsed = time.time() - start
print(f\"Fetched {len(df):,} bars in {elapsed:.1f}s\")
PYEOF
 2>&1 | tee ~/fetch_rangebar.log'"

# Monitor progress
ssh <host> "tail -f ~/fetch_rangebar.log"
```

---

## Checking Cache Status

### Using cache_status.py script

```bash
python scripts/cache_status.py
```

### Direct ClickHouse query

```python
import clickhouse_connect
client = clickhouse_connect.get_client(host='localhost')
result = client.query('''
    SELECT symbol, threshold_decimal_bps, count(*) as bars,
           min(timestamp_ms) as earliest, max(timestamp_ms) as latest
    FROM rangebar_cache.range_bars
    GROUP BY symbol, threshold_decimal_bps
''')
for row in result.result_rows:
    print(f"{row[0]} @ {row[1]} dbps: {row[2]:,} bars")
```

---

## Threshold Presets

| Preset       | Value | Use Case         |
| ------------ | ----- | ---------------- |
| `"micro"`    | 10    | Scalping         |
| `"tight"`    | 50    | Day trading      |
| `"standard"` | 100   | Swing trading    |
| `"medium"`   | 250   | Default          |
| `"wide"`     | 500   | Position trading |
| `"macro"`    | 1000  | Long-term        |

---

## Host-Specific Cache Status

| Host        | Thresholds Cached | Notes            |
| ----------- | ----------------- | ---------------- |
| bigblack    | 100, 700          | Primary GPU host |
| littleblack | 700               | Secondary host   |
| local       | varies            | Development      |

To add a threshold to a host, run the population script above on that host.

---

## Files

| File                  | Purpose                                                        |
| --------------------- | -------------------------------------------------------------- |
| `cache.py`            | RangeBarCache class, core cache operations                     |
| `bulk_operations.py`  | BulkStoreMixin (store_bars_bulk, store_bars_batch)             |
| `query_operations.py` | QueryOperationsMixin (get_n_bars, get_bars_by_timestamp_range) |
| `schema.sql`          | ClickHouse table schema (v7.0: 10 microstructure cols)         |
| `config.py`           | Connection configuration                                       |
| `preflight.py`        | Installation checks                                            |
| `tunnel.py`           | SSH tunnel support                                             |

---

## Related

- [/CLAUDE.md](/CLAUDE.md) - Project hub
- [/python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md) - Python API
- [/scripts/cache_status.py](/scripts/cache_status.py) - Status script
