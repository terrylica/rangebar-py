# Plugin System (Issue #98)

**Parent**: [/python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md)

FeatureProvider plugin system for external feature enrichment of range bars via entry-point discovery.

---

## File Map

| File           | Purpose                                                            |
| -------------- | ------------------------------------------------------------------ |
| `protocol.py`  | `FeatureProvider` runtime-checkable Protocol                       |
| `loader.py`    | Entry-point discovery, caching, `enrich_bars()`                    |
| `migration.py` | ClickHouse `ALTER TABLE ADD COLUMN IF NOT EXISTS`                  |
| `__init__.py`  | Re-exports: `FeatureProvider`, `discover_providers`, `enrich_bars` |

---

## Writing a FeatureProvider Plugin

### 1. Implement the protocol

```python
# my_plugin/rangebar_plugin.py
class MyFeatureProvider:
    name = "myplugin"
    version = "<version>"  # SSoT: plugin's own pyproject.toml
    columns = ("myplugin_signal", "myplugin_regime")
    min_bars = 20

    def enrich(self, bars, symbol, threshold_decimal_bps):
        bars["myplugin_signal"] = compute_signal(bars["Close"].values)
        bars["myplugin_regime"] = classify(bars["myplugin_signal"].values)
        # NaN for warmup
        bars.loc[bars.index[:self.min_bars], self.columns] = float("nan")
        return bars  # MUST return same object
```

### 2. Register entry point

```toml
# In plugin's pyproject.toml:
[project.entry-points."rangebar.feature_providers"]
myplugin = "my_plugin.rangebar_plugin:MyFeatureProvider"
```

### 3. Optional: add to rangebar's optional deps

```toml
# In rangebar's pyproject.toml [project.optional-dependencies]:
myplugin = ["my-plugin-package>=1.0"]
```

---

## Protocol Constraints

| Constraint         | Rule                                                           |
| ------------------ | -------------------------------------------------------------- |
| In-place mutation  | Add columns to input DataFrame, return same object             |
| Column prefix      | All columns prefixed with provider `name` (e.g., `laguerre_*`) |
| Warmup NaN         | Rows `0..min_bars-1` must be `NaN`, not `0.0`                  |
| Idempotent         | Calling `enrich()` twice produces same result                  |
| Non-anticipative   | Feature at `bar[i]` uses only `bars[0..i]`                     |
| Row-preserving     | No reordering, filtering, or adding rows                       |
| Read-only existing | Must NOT modify columns already present                        |

---

## Pipeline Integration

```
Rust processor → bars_df → enrich_bars() → bulk INSERT → ClickHouse
                           ^^^^^^^^^^^^
                    python/rangebar/orchestration/range_bars.py:665-669
```

`enrich_bars()` is called post-Rust, pre-cache. Zero overhead when no plugins installed (lazy discovery, cached after first call).

---

## Data Flow

1. `_get_providers()` → lazy `discover_providers()` (scans `rangebar.feature_providers` entry-point group)
2. `register_plugin_columns()` → adds column names to `constants._PLUGIN_FEATURE_COLUMNS`
3. `enrich_bars()` → calls each provider's `enrich()`, emits hook events
4. `bulk_operations.py` → includes `_PLUGIN_FEATURE_COLUMNS` in INSERT column list
5. `query_operations.py` → includes plugin columns in SELECT when `include_plugin_features=True`
6. `migration.py` → `ALTER TABLE ADD COLUMN IF NOT EXISTS` for each plugin column

---

## Hook Events

| Event                    | When                           | Payload                                             |
| ------------------------ | ------------------------------ | --------------------------------------------------- |
| `PLUGIN_ENRICH_COMPLETE` | All providers finished         | `symbol`, `provider_count`, `threshold_decimal_bps` |
| `PLUGIN_ENRICH_FAILED`   | A provider raised an exception | `symbol`, `provider_name`, `threshold_decimal_bps`  |

Provider failures are non-fatal (logged, hook emitted, pipeline continues).

---

## ClickHouse Schema

Plugin columns are `Nullable(Float64)`, added via `migrate_plugin_columns()`. Idempotent — safe to call on every startup.

Existing columns: [/python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md)
