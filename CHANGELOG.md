## [5.1.1](https://github.com/terrylica/rangebar-py/compare/v5.1.0...v5.1.1) (2026-01-09)


### Bug Fixes

* **build:** sync Cargo.toml version in semantic-release config ([e0b7a94](https://github.com/terrylica/rangebar-py/commit/e0b7a9478a33f3c16017d9477cfa223f871fd365)), closes [#2](https://github.com/terrylica/rangebar-py/issues/2)

# [5.1.0](https://github.com/terrylica/rangebar-py/compare/v5.0.0...v5.1.0) (2026-01-09)


### Features

* **api:** add get_n_range_bars() for count-bounded bar retrieval ([0e31d1f](https://github.com/terrylica/rangebar-py/commit/0e31d1f5fea08014084074fb982dfb45d2fe4b02))

# [5.0.0](https://github.com/terrylica/rangebar-py/compare/v4.0.1...v5.0.0) (2026-01-09)


* feat(api)!: rename threshold_bps to threshold_decimal_bps ([2ef2d96](https://github.com/terrylica/rangebar-py/commit/2ef2d960f80617466c05218415ca9e36d1b4f001))


### BREAKING CHANGES

* Aligns with upstream rangebar-core v6.0.0 API changes.

The parameter name now accurately reflects that values are in
decimal basis points (0.1bps = 0.001%).

Affected APIs:
- RangeBarProcessor(threshold_decimal_bps=...)
- get_range_bars(..., threshold_decimal_bps=...)
- process_trades_to_dataframe(..., threshold_decimal_bps=...)
- process_trades_polars(..., threshold_decimal_bps=...)
- process_trades_chunked(..., threshold_decimal_bps=...)

Constants renamed:
- THRESHOLD_MIN → THRESHOLD_DECIMAL_MIN
- THRESHOLD_MAX → THRESHOLD_DECIMAL_MAX

ClickHouse cache schema column renamed (migration script provided in schema.sql).

Dependencies updated:
- rangebar-core: 5.2 → 6.0
- rangebar-providers: 5.2 → 6.0

# CHANGELOG

All notable changes to this project will be documented in this file.

## [4.0.1](https://github.com/terrylica/rangebar-py/compare/v4.0.0...v4.0.1) (2026-01-08)

### Bug Fixes

* **build:** enable data-providers by default for PyPI wheels ([ae40068](https://github.com/terrylica/rangebar-py/commit/ae40068a8932bb7d590fc012f4e32afe9c6f50b4))

## [4.0.0](https://github.com/terrylica/rangebar-py/compare/v3.0.0...v4.0.0) (2026-01-08)

### Breaking Changes

* **api:** Simplified `__all__` exports to public API only
  - `get_range_bars()` is now the recommended primary entry point
  - Legacy APIs (`process_trades_to_dataframe`, `RangeBarProcessor`) still work but not exported in `__all__`

### Documentation

* Aligned README, examples, and API documentation with new primary API

## [3.0.0](https://github.com/terrylica/rangebar-py/compare/v2.1.0...v3.0.0) (2026-01-07)

### Features

- **api**: Add `get_range_bars()` as single entry point for downstream users
  - Automatic data fetching from Binance (spot, futures-um, futures-cm)
  - Exness forex support (10 instruments)
  - Threshold presets: micro, tight, standard, medium, wide, macro
  - Microstructure data: vwap, buy_volume, sell_volume
  - Local Parquet caching with configurable cache directory

- **constants**: Export configuration constants for discoverability
  - `TIER1_SYMBOLS`: 18 high-liquidity symbols on all Binance markets
  - `THRESHOLD_PRESETS`: Named threshold presets
  - `THRESHOLD_MIN`/`THRESHOLD_MAX`: Valid threshold range (1-100,000)

- **polars**: Add `process_trades_polars()` with lazy evaluation
  - Optimized for Polars users: 2-3x faster than `process_trades_to_dataframe()`
  - Full LazyFrame support with predicate pushdown

- **streaming**: Add `process_trades_chunked()` for memory-safe processing
  - Iterator-based API avoids OOM on datasets >10M trades

### Breaking Changes

- **api**: Simplified `__all__` exports to public API only
  - Removed internal APIs from public exports
  - `get_range_bars()` is now the recommended primary entry point
  - Legacy APIs (`process_trades_to_dataframe`, `RangeBarProcessor`) still work but not exported in `__all__`


## [2.1.0](https://github.com/terrylica/rangebar-py/compare/v2.0.2...v2.1.0) (2026-01-05)

### Features

- **clickhouse**: Add two-tier ClickHouse cache layer
  ([54a58de](https://github.com/terrylica/rangebar-py/commit/54a58def45d9f67b2665c6bf926ba9ecf7b262e2))


## [2.0.2](https://github.com/terrylica/rangebar-py/compare/v2.0.1...v2.0.2) (2025-12-29)

### Bug Fixes

- Resolve validation gaps from comprehensive audit
  ([c6b2d08](https://github.com/terrylica/rangebar-py/commit/c6b2d087286cda3e9904712a4e2868bf3df5958d))


## [2.0.1](https://github.com/terrylica/rangebar-py/compare/v2.0.0...v2.0.1) (2025-12-29)

### Bug Fixes

- **docs**: Convert relative links to repo-relative paths in api.md
  ([4d5c84d](https://github.com/terrylica/rangebar-py/commit/4d5c84d0a223f67e38e726bc04cb7336730c9655))


## [2.0.0](https://github.com/terrylica/rangebar-py/compare/v1.0.3...v2.0.0) (2025-11-27)

### Bug Fixes

- **readme**: Publish AOD/IOI/DRY-aligned documentation to PyPI
  ([422b50c](https://github.com/terrylica/rangebar-py/commit/422b50c40782f9662355d44c806f0c14516f1007))


## [1.0.3](https://github.com/terrylica/rangebar-py/compare/v1.0.2...v1.0.3) (2025-11-23)

### Bug Fixes

- **validation**: Update pattern to match data.js instead of .json
  ([d71b5a0](https://github.com/terrylica/rangebar-py/commit/d71b5a0e7cee86d1f1dcf5d426f6d789eba981d8))


## [1.0.2](https://github.com/terrylica/rangebar-py/compare/v1.0.1...v1.0.2) (2025-11-23)

### Bug Fixes

- **ci**: Add psutil dependency to workflow test environment
  ([59f6972](https://github.com/terrylica/rangebar-py/commit/59f6972903b7e7e4647efcec2252348b53a422ea))


## [1.0.1](https://github.com/terrylica/rangebar-py/compare/v1.0.0...v1.0.1) (2025-11-23)

### Bug Fixes

- **ci**: Create virtualenv before maturin develop in workflows
  ([30c8f34](https://github.com/terrylica/rangebar-py/commit/30c8f340b568717ed51b75199d98a6732fddf681))


## [1.0.0](https://github.com/terrylica/rangebar-py/compare/v0.3.0...v1.0.0) (2025-11-23)

### Features

- Add daily performance monitoring with github-action-benchmark
  ([8140cff](https://github.com/terrylica/rangebar-py/commit/8140cff02b6e00e8fd3e37175209bd36071389ee))

- **perf-monitoring**: Switch to GitHub Actions deployment source
  ([c14403f](https://github.com/terrylica/rangebar-py/commit/c14403f25058c435db9260ecfa364da232bdf9ed))


## [0.3.0](https://github.com/terrylica/rangebar-py/compare/v0.2.0...v0.3.0) (2025-11-17)

### Features

- Release v0.3.0 with complete CI/CD infrastructure
  ([5f6a19f](https://github.com/terrylica/rangebar-py/commit/5f6a19f6b80c48afc278020c6689451b06250148))


## [0.2.0](https://github.com/terrylica/rangebar-py/compare/v0.1.1...v0.2.0) (2025-11-17)

### Features

- Complete release automation infrastructure
  ([4886cca](https://github.com/terrylica/rangebar-py/commit/4886cca5e2a8020d066534a130cac4b40874b3b6))


## [0.1.1](https://github.com/terrylica/rangebar-py/compare/v0.1.0...v0.1.1) (2025-11-17)

### Bug Fixes

- **ci**: Add pypi environment to publish job
  ([4236b69](https://github.com/terrylica/rangebar-py/commit/4236b69386e7be8732b3dd76a74ae6cc31b76f53))


## [0.1.0](https://github.com/terrylica/rangebar-py/releases/tag/v0.1.0) (2025-11-17)

- Initial Release
