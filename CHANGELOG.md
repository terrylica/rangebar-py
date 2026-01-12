## [7.1.3](https://github.com/terrylica/rangebar-py/compare/v7.1.2...v7.1.3) (2026-01-12)


### Bug Fixes

* compute microstructure features in batch processing path (Issue [#34](https://github.com/terrylica/rangebar-py/issues/34)) ([531c9b1](https://github.com/terrylica/rangebar-py/commit/531c9b16fc29b270665574adabaf23577fb966ce))

## [7.1.1](https://github.com/terrylica/rangebar-py/compare/v7.1.0...v7.1.1) (2026-01-12)


### Bug Fixes

* pass is_buyer_maker to Rust for microstructure features (Issue [#30](https://github.com/terrylica/rangebar-py/issues/30)) ([6c32dbb](https://github.com/terrylica/rangebar-py/commit/6c32dbb2d0f6860d10d0e3316b877e5371f34804))

# [7.1.0](https://github.com/terrylica/rangebar-py/compare/v7.0.1...v7.1.0) (2026-01-12)


### Features

* add incremental ClickHouse caching with post-cache validation (Issue [#27](https://github.com/terrylica/rangebar-py/issues/27)) ([7318da4](https://github.com/terrylica/rangebar-py/commit/7318da4002c681c54caf898db9510db811d20410)), closes [#5](https://github.com/terrylica/rangebar-py/issues/5)

## [7.0.1](https://github.com/terrylica/rangebar-py/compare/v7.0.0...v7.0.1) (2026-01-11)


### Bug Fixes

* update crate readme paths to local READMEs for sdist build ([9559ff9](https://github.com/terrylica/rangebar-py/commit/9559ff979faf701b5b049ca3d194ca247501dc20))
* update internal crate version references from 6.0 to 7.0 ([41d4198](https://github.com/terrylica/rangebar-py/commit/41d4198aa223855e95d09ffbef85224831ae58bf))

# [7.0.0](https://github.com/terrylica/rangebar-py/compare/v6.2.1...v7.0.0) (2026-01-11)


* feat!: add 10 market microstructure features computed in Rust (Issue [#25](https://github.com/terrylica/rangebar-py/issues/25)) ([be4be75](https://github.com/terrylica/rangebar-py/commit/be4be75c691d39863c25ccb297004f956e66fde7))


### BREAKING CHANGES

* RangeBar struct extended with 10 new fields. Cached bars
must be re-precomputed to populate new columns.

Microstructure features added (all computed at bar finalization):
- duration_us: Bar duration in microseconds
- ofi: Order Flow Imbalance [-1, 1]
- vwap_close_deviation: (close - vwap) / (high - low)
- price_impact: Amihud-style illiquidity ratio
- kyle_lambda_proxy: Market depth proxy
- trade_intensity: Trades per second
- volume_per_trade: Average trade size
- aggression_ratio: Buy/sell trade count ratio [0, 100]
- aggregation_efficiency_f64: Trade fragmentation proxy
- turnover_imbalance: Dollar-weighted OFI [-1, 1]

Implementation:
- Extended RangeBar struct with 10 new fields (#[serde(default)])
- Added compute_microstructure_features() method in types.rs
- Updated processor.rs to compute features at bar finalization
- Updated PyO3 bindings to expose all fields to Python
- Updated ClickHouse schema with 10 new columns (DEFAULT values)
- Updated cache.py with new column handling
- Created tiered validation framework (tier1.py, tier2.py)
- Added 11 Rust unit tests, 10 Python integration tests
- Created GPU workstation validation script

Academic backing: Kyle (1985), Cont et al. (2014), Amihud (2002),
Easley et al. (2012), Welford (1962).

## [6.2.1](https://github.com/terrylica/rangebar-py/compare/v6.2.0...v6.2.1) (2026-01-11)


### Bug Fixes

* add ClickHouse bar cache lookup to get_range_bars() and fix TZ-aware timestamps (Issues [#20](https://github.com/terrylica/rangebar-py/issues/20), [#21](https://github.com/terrylica/rangebar-py/issues/21)) ([b9fa078](https://github.com/terrylica/rangebar-py/commit/b9fa078e6bfc8f87da1753412693e74b5dd17d5d))

# [6.2.0](https://github.com/terrylica/rangebar-py/compare/v6.1.0...v6.2.0) (2026-01-11)


### Features

* add tiered validation system for market microstructure events (Issue [#19](https://github.com/terrylica/rangebar-py/issues/19)) ([685bb49](https://github.com/terrylica/rangebar-py/commit/685bb498654d87506ba9cd686e700b089d78e491))

# [6.1.0](https://github.com/terrylica/rangebar-py/compare/v6.0.0...v6.1.0) (2026-01-11)


### Bug Fixes

* add process_trades_streaming() for cross-batch state persistence (Issue [#16](https://github.com/terrylica/rangebar-py/issues/16)) ([5951fce](https://github.com/terrylica/rangebar-py/commit/5951fce665662997916b4602e67b44211b572e3d))
* align internal crate version requirements with 6.0.0 ([73409eb](https://github.com/terrylica/rangebar-py/commit/73409eb48fed56a75866f6b7b080cc171b23a369))


### Features

* add granular mise release tasks with 4-phase workflow ([6b6482a](https://github.com/terrylica/rangebar-py/commit/6b6482a602770b6758c425a1c312c513d2ab7e4a)), closes [#17](https://github.com/terrylica/rangebar-py/issues/17)

# [6.0.0](https://github.com/terrylica/rangebar-py/compare/v5.4.0...v6.0.0) (2026-01-10)


### Features

* consolidate 8-crate Rust workspace from upstream rangebar ([f16f50b](https://github.com/terrylica/rangebar-py/commit/f16f50b12853dbaf4213c97ca45c2f94940d6eb9))


### BREAKING CHANGES

* Architecture now includes full 8-crate workspace

- Layer 0: rangebar-core (algorithm), rangebar-config (settings)
- Layer 1: rangebar-providers (Binance/Exness), rangebar-io (Polars)
- Layer 2: rangebar-streaming (real-time), rangebar-batch (analytics)
- Layer 3: rangebar-cli (disabled), rangebar (meta-crate)

Build optimizations preserved:
- Thin LTO for cross-platform compatibility
- codegen-units=1 for maximum optimization
- Custom [profile.wheel] for wheel builds
- build.rs compile-time optimization guard

New infrastructure:
- Criterion benchmarks (1M ticks < 100ms target)
- cargo-deny security/license checks
- mise tasks for full dev workflow
- Test data fixtures (~900KB)

All 175 Rust tests passing.

# [5.4.0](https://github.com/terrylica/rangebar-py/compare/v5.3.6...v5.4.0) (2026-01-10)


### Features

* add memory efficiency validation script ([558b6b7](https://github.com/terrylica/rangebar-py/commit/558b6b7520c857e6b5d411029bd44f052d7e6aff)), closes [#12](https://github.com/terrylica/rangebar-py/issues/12) [hi#volume](https://github.com/hi/issues/volume) [#15](https://github.com/terrylica/rangebar-py/issues/15)

## [5.3.6](https://github.com/terrylica/rangebar-py/compare/v5.3.5...v5.3.6) (2026-01-10)


### Bug Fixes

* fetch data day-by-day to prevent OOM during fetch phase ([7090ded](https://github.com/terrylica/rangebar-py/commit/7090dedabcc49b744da078c010f6f90643534099)), closes [hi#volume](https://github.com/hi/issues/volume) [#14](https://github.com/terrylica/rangebar-py/issues/14)

## [5.3.5](https://github.com/terrylica/rangebar-py/compare/v5.3.4...v5.3.5) (2026-01-10)


### Bug Fixes

* sort trades by (timestamp, trade_id) to prevent Rust sorting error ([a6d15c6](https://github.com/terrylica/rangebar-py/commit/a6d15c6ac57c42c67b10f241e2455af92dba839e)), closes [#13](https://github.com/terrylica/rangebar-py/issues/13)

## [5.3.4](https://github.com/terrylica/rangebar-py/compare/v5.3.3...v5.3.4) (2026-01-10)


### Bug Fixes

* correct range bar continuity validation semantics ([67e5e32](https://github.com/terrylica/rangebar-py/commit/67e5e325b8f3317eae3d9d09b6d20c0ab10e49a9)), closes [#12](https://github.com/terrylica/rangebar-py/issues/12)

## [5.3.3](https://github.com/terrylica/rangebar-py/compare/v5.3.2...v5.3.3) (2026-01-10)


### Bug Fixes

* streaming read for large months to prevent OOM (Issue [#12](https://github.com/terrylica/rangebar-py/issues/12)) ([f3584e7](https://github.com/terrylica/rangebar-py/commit/f3584e713e152a7847144644eedd37072c9cef49)), closes [hi#volume](https://github.com/hi/issues/volume)

## [5.3.2](https://github.com/terrylica/rangebar-py/compare/v5.3.1...v5.3.2) (2026-01-10)


### Bug Fixes

* reduce memory usage in precompute_range_bars() to prevent OOM ([7adf579](https://github.com/terrylica/rangebar-py/commit/7adf5791186aa874962e67dc70574b3582326c4e)), closes [hi#volume](https://github.com/hi/issues/volume) [#11](https://github.com/terrylica/rangebar-py/issues/11)

## [5.3.1](https://github.com/terrylica/rangebar-py/compare/v5.3.0...v5.3.1) (2026-01-10)


### Bug Fixes

* deduplicate trades before processing to handle Binance duplicates ([fde2a19](https://github.com/terrylica/rangebar-py/commit/fde2a19d80c4c596d17f0eda026d0533c786e368)), closes [#10](https://github.com/terrylica/rangebar-py/issues/10)

# [5.3.0](https://github.com/terrylica/rangebar-py/compare/v5.2.1...v5.3.0) (2026-01-10)


### Features

* implement Issues [#7](https://github.com/terrylica/rangebar-py/issues/7) and [#8](https://github.com/terrylica/rangebar-py/issues/8) - API consistency and WFO workflow ([c5ae438](https://github.com/terrylica/rangebar-py/commit/c5ae438c9d0ac2fc996f03d9316c0a701a89f572))

# [5.2.0](https://github.com/terrylica/rangebar-py/compare/v5.1.1...v5.2.0) (2026-01-10)


### Features

* **continuity:** add Checkpoint API and validate_continuity() for cross-file range bar continuity ([07fc472](https://github.com/terrylica/rangebar-py/commit/07fc47252f1e04dcd76011790ead458cc90ff8c9)), closes [#3](https://github.com/terrylica/rangebar-py/issues/3)

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
