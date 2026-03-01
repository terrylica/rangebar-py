# Configuration Subpackage

**Parent**: [/python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md)

Centralized configuration via pydantic-settings BaseSettings. Automatic priority:
**CLI > env vars > rangebar.toml > defaults**.

---

## File Map

| File            | Purpose                                               | Env Prefix            |
| --------------- | ----------------------------------------------------- | --------------------- |
| `settings.py`   | `Settings` singleton (`get()`, `reload()`)            | —                     |
| `population.py` | Cache population knobs + feature toggles (Issue #128) | `RANGEBAR_`           |
| `monitoring.py` | Telegram, recency thresholds, environment             | `RANGEBAR_`           |
| `clickhouse.py` | ClickHouse host, port, mode, tunnel                   | `RANGEBAR_CH_`        |
| `sidecar.py`    | Streaming sidecar watchdog, backpressure              | `RANGEBAR_STREAMING_` |
| `algorithm.py`  | Rust AlgorithmConfig subset (batch size, memory opt)  | `RANGEBAR_ALGORITHM_` |
| `streaming.py`  | Rust StreamingConfig (channels, circuit breaker)      | `RANGEBAR_STREAMING_` |
| `cli_models.py` | Pydantic CliApp models (Populate subcommands)         | `RANGEBAR_`           |
| `__init__.py`   | Re-exports: Settings, PopulationConfig, ...           | —                     |

---

## Configuration Priority (Automatic)

    1. CLI flags           rangebar populate range --hurst --threshold 500
    2. Environment vars    RANGEBAR_COMPUTE_HURST=true  (from .mise.toml)
    3. TOML config file    rangebar.toml [population] compute_hurst = true
    4. Model defaults      compute_hurst: bool = False

No manual wiring — pydantic-settings handles the cascade.

---

## Settings Singleton

    from rangebar.config import Settings

    s = Settings.get()              # Thread-safe, cached
    s.population.compute_tier2      # True
    s.clickhouse.host               # "localhost"
    s.sidecar.watchdog_timeout_s    # 300
    s.algorithm.processing_batch_size  # 100000
    s.streaming.circuit_breaker_threshold  # 0.5

    Settings.reload()               # Re-read from env
    Settings.reload_and_clear()     # + clear threshold cache (Issue #126)

---

## Related

- [/CLAUDE.md](/CLAUDE.md) - Project hub
- [/python/rangebar/CLAUDE.md](/python/rangebar/CLAUDE.md) - Python API layer
- [/python/rangebar/clickhouse/CLAUDE.md](/python/rangebar/clickhouse/CLAUDE.md) - ClickHouse cache
- [/scripts/CLAUDE.md](/scripts/CLAUDE.md) - Pueue ops, per-year parallelization
