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

## Connection Mode

`RANGEBAR_MODE` env var (set in `.mise.toml`) controls host resolution in `preflight.py:get_available_clickhouse_host()`:

| Mode     | Behavior                                                  | Use Case                      |
| -------- | --------------------------------------------------------- | ----------------------------- |
| `auto`   | Try localhost first, then `RANGEBAR_CH_HOSTS` via SSH     | Default (auto-detect)         |
| `local`  | Force `localhost:8123` only                               | Local ClickHouse only         |
| `remote` | Skip localhost, use `RANGEBAR_CH_HOSTS` via direct or SSH | **Current** (always bigblack) |
| `cloud`  | Require `CLICKHOUSE_HOST` env var                         | Managed ClickHouse Cloud      |

**Current config** (`.mise.toml`): `RANGEBAR_MODE = "remote"` — all queries route through bigblack via SSH tunnel. No local ClickHouse.

**Preflight gate**: `mise run db:ensure` verifies bigblack reachable. All test tasks (`test-py`, `test:e2e`, `test:clickhouse`, `test:all`) depend on `db:ensure`.

---

## Host-Specific Cache Status

| Host        | Symbols                  | Thresholds Cached (dbps) | ClickHouse Setup | Notes             |
| ----------- | ------------------------ | ------------------------ | ---------------- | ----------------- |
| bigblack    | 15 Tier 1 crypto symbols | 250, 500, 750, 1000      | Native           | Primary GPU host  |
| bigblack    | EURUSD (forex)           | 50, 100, 200             | Native           | Exness Raw_Spread |
| littleblack | BTCUSDT                  | 100                      | Docker           | Secondary host    |

**Note**: Local ClickHouse has been removed. All data served from bigblack via SSH tunnel (`localhost:18123 → bigblack:8123`). The flowsurface app auto-starts the tunnel via `mise run preflight`.

**Minimum crypto threshold**: 250 dbps (enforced by `RANGEBAR_CRYPTO_MIN_THRESHOLD` in `.mise.toml`)

**Total cached**: 260M+ bars (crypto) + 130K bars (forex)

To add a threshold to a host, run the population script above on that host.

**Docker vs Native**: Use Docker when sudo unavailable. Native preferred for production (lower overhead).

---

## Dedup Hardening (Issue #90)

Five-layer protection against duplicate rows from OPTIMIZE timeout crashes and retry storms. Validated on bigblack (260M+ rows, ALL LAYERS PASS).

### Five-Layer Architecture

| Layer | Mechanism                                        | Where                      | Scope        |
| ----- | ------------------------------------------------ | -------------------------- | ------------ |
| 1     | `non_replicated_deduplication_window = 1000`     | `_ensure_schema()` ALTER   | Engine-level |
| 2     | INSERT dedup token (MD5 of `cache_key`)          | `_build_insert_settings()` | Per-INSERT   |
| 3     | Fire-and-forget `OPTIMIZE` + merge polling       | `deduplicate_bars()`       | Post-write   |
| 4     | `SELECT ... FINAL` with partition parallelism    | `_FINAL_READ_SETTINGS`     | Query-time   |
| 5     | Schema migration on every `RangeBarCache()` init | `_ensure_schema()`         | Bootstrap    |

### Write Entry Point Map

All ClickHouse write paths emit INSERT dedup tokens:

| Caller                                 | Store Method         | Dedup Token | `skip_dedup`            |
| -------------------------------------- | -------------------- | ----------- | ----------------------- |
| `get_range_bars()`                     | `store_range_bars()` | Yes         | No (always `False`)     |
| `populate_cache_resumable()`           | `store_bars_batch()` | Yes         | Param (default `False`) |
| `get_n_range_bars()`                   | `store_bars_bulk()`  | Yes         | Param (default `False`) |
| `precompute_range_bars()`              | `store_bars_bulk()`  | Yes         | Param (default `False`) |
| `process_trades_to_dataframe_cached()` | `store_range_bars()` | Yes         | No (always `False`)     |

### Key Functions and Constants

| Symbol                     | File                 | Purpose                                                       |
| -------------------------- | -------------------- | ------------------------------------------------------------- |
| `_build_insert_settings()` | `bulk_operations.py` | Build `{insert_deduplicate, insert_deduplication_token}` dict |
| `_FINAL_READ_SETTINGS`     | `cache.py`           | `{"do_not_merge_across_partitions_select_final": 1}`          |
| `deduplicate_bars()`       | `cache.py`           | Fire-and-forget OPTIMIZE per partition (60s timeout)          |
| `_wait_for_merges()`       | `cache.py`           | Poll `system.merges` until OPTIMIZE completes                 |
| `_ensure_schema()`         | `cache.py`           | ALTER TABLE MODIFY SETTING (idempotent migration)             |

### `skip_dedup` Parameter

`store_bars_bulk()` and `store_bars_batch()` accept `skip_dedup: bool = False`:

- **`False` (default)**: Generates dedup token from `cache_key` column. Retrying the same INSERT is silently dropped by ClickHouse.
- **`True`**: Skips token. Used for `force_refresh=True` paths where cache was already deleted.

`store_range_bars()` always passes `False` (no `force_refresh` pathway).

### Validation

```bash
mise run cache:validate-dedup    # Run on bigblack (requires ClickHouse)
```

Validates all 4 runtime layers against live data. See `scripts/validate_dedup_hardening.py`.

---

## Files

| File                  | Purpose                                                              |
| --------------------- | -------------------------------------------------------------------- |
| `cache.py`            | RangeBarCache class, core cache operations, dedup Layer 3-5          |
| `bulk_operations.py`  | BulkStoreMixin (store_bars_bulk, store_bars_batch), dedup Layer 2    |
| `query_operations.py` | QueryOperationsMixin (get_n_bars, get_bars_by_timestamp_range)       |
| `schema.sql`          | ClickHouse table schema (range_bars + population_checkpoints tables) |
| `config.py`           | Connection configuration                                             |
| `preflight.py`        | Installation checks                                                  |
| `tunnel.py`           | SSH tunnel support                                                   |

---

## Related

- [/CLAUDE.md](/CLAUDE.md) - Project hub
- [/python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md) - Python API
- [/scripts/CLAUDE.md](/scripts/CLAUDE.md) - Pueue jobs, per-year parallelization, autoscaler
- [/scripts/cache_status.py](/scripts/cache_status.py) - Status script
