# ClickHouse Cache Layer

**Parent**: [/python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md) | **Schema**: [schema.sql](./schema.sql)

---

## Cache Population

### Resumable Population (Recommended for Long Ranges)

For date ranges > 30 days, use `populate_cache_resumable()`:

```python
from rangebar import populate_cache_resumable

# Incremental, resumable, memory-safe
populate_cache_resumable(
    "BTCUSDT",
    start_date="2019-01-01",
    end_date="2025-12-31",
    threshold_decimal_bps=100,
)

# Force restart (wipe cache and checkpoint)
populate_cache_resumable(
    "BTCUSDT",
    start_date="2019-01-01",
    end_date="2025-12-31",
    force_refresh=True,
)
```

**How it works**:

1. Resumes from last checkpoint (local or ClickHouse)
2. Fetches raw tick data day-by-day
3. Processes with stateful `RangeBarProcessor` (preserves incomplete bars)
4. Stores bars incrementally to `rangebar_cache.range_bars`
5. Saves checkpoint after each day (both local and ClickHouse)

### Native Method (Short Ranges ≤ 30 days)

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

#### Prerequisites: ClickHouse Setup

**Docker-based setup** (recommended for hosts without sudo):

```bash
# SSH to host
ssh <host>

# Create data directory
mkdir -p ~/clickhouse-data

# Start ClickHouse container with passwordless HTTP access
# CRITICAL: -e CLICKHOUSE_PASSWORD= (empty) enables passwordless default user
docker run -d \
  --name clickhouse-server \
  -p 8123:8123 -p 9000:9000 \
  -v ~/clickhouse-data:/var/lib/clickhouse \
  -e CLICKHOUSE_PASSWORD= \
  --restart unless-stopped \
  clickhouse/clickhouse-server:latest

# Verify connection
curl "http://localhost:8123/?query=SELECT%201"
```

**Create schema** (run once after container starts):

```bash
# Download and execute schema
curl -s "http://localhost:8123/" --data-binary @- << 'EOF'
CREATE DATABASE IF NOT EXISTS rangebar_cache
EOF

# Copy schema.sql content or use setup script
# See: scripts/setup-littleblack-clickhouse.sh for full schema
```

#### Install rangebar

**CRITICAL**: Use `uv` instead of `pip` (Python 3.13 externally-managed-environment).

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install rangebar (latest version)
~/.local/bin/uv pip install rangebar
```

#### Environment Variables

**CRITICAL**: mise SSoT pattern - check `mise.toml` for hardcoded hosts.

```bash
# Required for remote hosts not in default RANGEBAR_CH_HOSTS
export RANGEBAR_CH_HOSTS=localhost

# Required for low thresholds (below default minimum)
export RANGEBAR_CRYPTO_MIN_THRESHOLD=1
```

**Gotcha**: If `mise.toml` sets `RANGEBAR_CH_HOSTS=bigblack`, remote hosts will fail silently. Override explicitly.

#### Run Population

```bash
# Start tmux session (long-running, survives SSH disconnect)
tmux new-session -d -s rangebar-populate

# Attach and run
tmux attach -t rangebar-populate

# Inside tmux:
export RANGEBAR_CH_HOSTS=localhost
export RANGEBAR_CRYPTO_MIN_THRESHOLD=1

~/.local/bin/uv run python3 << 'PYEOF'
from rangebar import populate_cache_resumable

populate_cache_resumable(
    "BTCUSDT",
    start_date="2023-06-01",
    end_date="2025-12-31",
    threshold_decimal_bps=100,
)
PYEOF

# Detach: Ctrl+B, D
# Reattach later: tmux attach -t rangebar-populate
```

#### Troubleshooting Remote Setup

| Error                            | Cause                                       | Fix                                         |
| -------------------------------- | ------------------------------------------- | ------------------------------------------- |
| "Authentication failed"          | Container missing `-e CLICKHOUSE_PASSWORD=` | Recreate container with empty password flag |
| "externally-managed-environment" | pip blocked on Python 3.13                  | Use `uv pip install` instead                |
| ThresholdError "below minimum"   | Default minimum threshold validation        | Set `RANGEBAR_CRYPTO_MIN_THRESHOLD=1`       |
| Connection refused               | Wrong host in environment                   | Set `RANGEBAR_CH_HOSTS=localhost`           |
| "command not found: mise"        | mise not installed on remote                | Use explicit uv/python paths                |

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

| Host        | Symbols                                     | Thresholds Cached     | ClickHouse Setup | Notes             |
| ----------- | ------------------------------------------- | --------------------- | ---------------- | ----------------- |
| bigblack    | BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT (crypto) | 25, 50, 100, 200 dbps | Native           | Primary GPU host  |
| bigblack    | EURUSD (forex)                              | 50, 100, 200 dbps     | Native           | Exness Raw_Spread |
| littleblack | BTCUSDT                                     | 100 dbps              | Docker           | Secondary host    |
| local       | varies                                      | varies                | Native/Docker    | Development       |

**Total cached**: 260M+ bars (crypto) + 130K bars (forex)

To add a threshold to a host, run the population script above on that host.

**Docker vs Native**: Use Docker when sudo unavailable. Native preferred for production (lower overhead).

---

## Files

| File                  | Purpose                                                              |
| --------------------- | -------------------------------------------------------------------- |
| `cache.py`            | RangeBarCache class, core cache operations                           |
| `bulk_operations.py`  | BulkStoreMixin (store_bars_bulk, store_bars_batch)                   |
| `query_operations.py` | QueryOperationsMixin (get_n_bars, get_bars_by_timestamp_range)       |
| `schema.sql`          | ClickHouse table schema (range_bars + population_checkpoints tables) |
| `config.py`           | Connection configuration                                             |
| `preflight.py`        | Installation checks                                                  |
| `tunnel.py`           | SSH tunnel support                                                   |

---

## Related

- [/CLAUDE.md](/CLAUDE.md) - Project hub
- [/python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md) - Python API
- [/scripts/cache_status.py](/scripts/cache_status.py) - Status script
