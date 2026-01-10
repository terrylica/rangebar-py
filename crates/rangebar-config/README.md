# rangebar-config

Configuration management for rangebar workspace using environment-aware settings.

## Overview

`rangebar-config` provides centralized configuration management using the `config` crate. Supports environment-specific settings, default values, and structured configuration loading.

## Features

- Environment-aware configuration (development, production, testing)
- Structured settings with type safety
- Default value fallbacks
- Integration with rangebar-core types

## Usage

### Load Configuration

```rust
use rangebar_config::Settings;

// Load from config files and environment
let settings = Settings::load()?;

// Use default settings
let settings = Settings::default();
```

### Configuration Structure

```rust
pub struct Settings {
    pub app: AppSettings,
    pub binance: BinanceSettings,
    // ... other provider settings
}

pub struct AppSettings {
    pub name: String,
    pub environment: String,
    pub log_level: String,
}

pub struct BinanceSettings {
    pub data_dir: String,
    pub default_market: String,
    pub cache_enabled: bool,
}
```

## Configuration Files

Place configuration files in the workspace root or config directory:

- `config/default.toml` - Default settings
- `config/development.toml` - Development overrides
- `config/production.toml` - Production overrides

### Example: `config/default.toml`

```toml
[app]
name = "rangebar"
environment = "development"
log_level = "info"

[binance]
data_dir = "./data/binance"
default_market = "spot"
cache_enabled = true
```

## Environment Variables

Override settings via environment variables with prefix `RANGEBAR_`:

```bash
export RANGEBAR_APP__LOG_LEVEL=debug
export RANGEBAR_BINANCE__DEFAULT_MARKET=um
```

## Dependencies

- **rangebar-core** - Core types
- **config** - Configuration management framework
- **serde** - Serialization support

## Version

Current version: **6.1.0** (modular crate architecture with checkpoint system)

## Documentation

- Architecture: `../../docs/ARCHITECTURE.md`

## License

See LICENSE file in the repository root.
