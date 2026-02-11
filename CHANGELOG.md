# [12.16.0](https://github.com/terrylica/rangebar-py/compare/v12.15.1...v12.16.0) (2026-02-11)


### Bug Fixes

* **core:** widen volume accumulators from i64 to i128 to prevent overflow ([#88](https://github.com/terrylica/rangebar-py/issues/88)) ([3edb388](https://github.com/terrylica/rangebar-py/commit/3edb388b86360ae9b2a5948c9cf88700345ea650))


### Features

* **pueue:** wire optimize + detect-overflow as --after dependency jobs ([#88](https://github.com/terrylica/rangebar-py/issues/88)) ([e0cbe8c](https://github.com/terrylica/rangebar-py/commit/e0cbe8cc1437db558a6627f3ddd921b16702f46f))
* **scripts:** add --start-date/--end-date to populate_full_cache.py ([c671034](https://github.com/terrylica/rangebar-py/commit/c671034cd066a3a0ea90a05ccb1e9a95da234350))
* **scripts:** add pipeline monitor for pueue job groups ([#88](https://github.com/terrylica/rangebar-py/issues/88)) ([d5f54b8](https://github.com/terrylica/rangebar-py/commit/d5f54b83dd5e141e0f16b62ba67ae5a47680cb43))
* **threshold:** per-symbol min threshold for SHIB/DOGE via mise SSoT ([#89](https://github.com/terrylica/rangebar-py/issues/89)) ([c5cd702](https://github.com/terrylica/rangebar-py/commit/c5cd702bff2149fee72252f42d46377750802183))

## [12.15.1](https://github.com/terrylica/rangebar-py/compare/v12.15.0...v12.15.1) (2026-02-10)


### Bug Fixes

* **arrow:** wire include_microstructure through process_trades_polars() ([617d7f9](https://github.com/terrylica/rangebar-py/commit/617d7f9a59c56f2cc019672ab9a3deb06b96bcad))

# [12.15.0](https://github.com/terrylica/rangebar-py/compare/v12.14.0...v12.15.0) (2026-02-10)


### Features

* **telemetry:** add forensics-grade NDJSON telemetry for backtesting pipeline ([9588caf](https://github.com/terrylica/rangebar-py/commit/9588cafb93965bdc6a0bd859c3bd4ec4fc4005d3))

# [12.14.0](https://github.com/terrylica/rangebar-py/compare/v12.13.0...v12.14.0) (2026-02-10)


### Features

* **arrow:** eliminate stream round-trip and structural optimizations ([#88](https://github.com/terrylica/rangebar-py/issues/88)) ([30dd0d2](https://github.com/terrylica/rangebar-py/commit/30dd0d2f0c67c88fb693b234b670bf67e6297f65))

# [12.13.0](https://github.com/terrylica/rangebar-py/compare/v12.12.0...v12.13.0) (2026-02-10)


### Features

* **arrow:** add Arrow-native input path for 3x pipeline speedup ([#88](https://github.com/terrylica/rangebar-py/issues/88)) ([0aecc6b](https://github.com/terrylica/rangebar-py/commit/0aecc6b42e9e767a931c11a46d8bc62e9622789c))

# [12.12.0](https://github.com/terrylica/rangebar-py/compare/v12.11.1...v12.12.0) (2026-02-09)


### Bug Fixes

* **clickhouse:** add timestamp_ms scale guard and oracle verification report ([#87](https://github.com/terrylica/rangebar-py/issues/87)) ([b79b39d](https://github.com/terrylica/rangebar-py/commit/b79b39dfe1fa51b3c02cd64718fa0575c14cd9bc)), closes [#85](https://github.com/terrylica/rangebar-py/issues/85)


### Features

* **binance-vision:** add end-date availability probe and lower crypto threshold to 250 dbps ([#84](https://github.com/terrylica/rangebar-py/issues/84)) ([faf576a](https://github.com/terrylica/rangebar-py/commit/faf576a22e90a8a15aaf7afc93051e1bfd4f5b69))
* **cache:** add systemd-run resource guards and autoscaler ([#84](https://github.com/terrylica/rangebar-py/issues/84)) ([9e945de](https://github.com/terrylica/rangebar-py/commit/9e945de57ca60ebd5e713563ccfa9b7e3b0749f6))

## [12.11.1](https://github.com/terrylica/rangebar-py/compare/v12.11.0...v12.11.1) (2026-02-08)


### Bug Fixes

* **checkpoint:** include threshold in filename to prevent concurrent job collisions ([#84](https://github.com/terrylica/rangebar-py/issues/84)) ([128b33a](https://github.com/terrylica/rangebar-py/commit/128b33a705e15f7c07d0f0ae9e819dc85b740b51))

# [12.11.0](https://github.com/terrylica/rangebar-py/compare/v12.10.5...v12.11.0) (2026-02-08)


### Features

* **cache:** wire pueue population into mise task DAG ([#84](https://github.com/terrylica/rangebar-py/issues/84)) ([9b88a70](https://github.com/terrylica/rangebar-py/commit/9b88a707180d5551f58a3698fb78f5c1fc11e208))

## [12.10.5](https://github.com/terrylica/rangebar-py/compare/v12.10.4...v12.10.5) (2026-02-08)


### Bug Fixes

* **registry:** oracle-verified ghost trade anomalies, schema v2 with data quality fields ([#79](https://github.com/terrylica/rangebar-py/issues/79)) ([3acdcd8](https://github.com/terrylica/rangebar-py/commit/3acdcd816f3481fb26db0c072f077ae70ebd2f1b))

## [12.10.4](https://github.com/terrylica/rangebar-py/compare/v12.10.3...v12.10.4) (2026-02-08)


### Bug Fixes

* **helpers:** resolve Polars schema inference crash on AVAXUSDT and use yesterday as END_DATE ([ce3fd18](https://github.com/terrylica/rangebar-py/commit/ce3fd186517b4b00d3eca9f29fc7526d96a4f038)), closes [#84](https://github.com/terrylica/rangebar-py/issues/84)

## [12.10.3](https://github.com/terrylica/rangebar-py/compare/v12.10.2...v12.10.3) (2026-02-08)


### Bug Fixes

* **registry:** correct effective_start dates for corrupted Binance aggTrade CSVs ([c1950ff](https://github.com/terrylica/rangebar-py/commit/c1950ffafcbae98213b2a96366fcc45ccb56a7a7))

## [12.10.2](https://github.com/terrylica/rangebar-py/compare/v12.10.1...v12.10.2) (2026-02-08)


### Bug Fixes

* **wheel:** include symbol registry data package in PyPI wheels ([#86](https://github.com/terrylica/rangebar-py/issues/86)) ([a762022](https://github.com/terrylica/rangebar-py/commit/a7620228804739f20299631c543ec45cf77f965f))

## [12.10.1](https://github.com/terrylica/rangebar-py/compare/v12.10.0...v12.10.1) (2026-02-08)


### Bug Fixes

* **clickhouse:** use resolution-agnostic timestamp conversion for pandas 3.0 ([#85](https://github.com/terrylica/rangebar-py/issues/85)) ([2b6e509](https://github.com/terrylica/rangebar-py/commit/2b6e50964b8e9e867111acbb847f549a72502ce8))

# [12.10.0](https://github.com/terrylica/rangebar-py/compare/v12.9.0...v12.10.0) (2026-02-08)


### Bug Fixes

* **clickhouse:** numeric version comparison for microstructure queries ([#82](https://github.com/terrylica/rangebar-py/issues/82)) ([19b4f1c](https://github.com/terrylica/rangebar-py/commit/19b4f1c1515cec5021047f0b1744db7befcec9d4))
* **scripts:** fix pueue job invocation quoting issue ([9364e8d](https://github.com/terrylica/rangebar-py/commit/9364e8d26c7d11a765cd9c8f1367b7c4832d5b78))


### Features

* **interbar:** add BarRelative lookback mode for inter-bar features ([#81](https://github.com/terrylica/rangebar-py/issues/81)) ([9ae8a57](https://github.com/terrylica/rangebar-py/commit/9ae8a5744b6465deeab835625015e4ff68fa7151))
* **scripts:** add force-refresh + microstructure params for full repopulation ([#84](https://github.com/terrylica/rangebar-py/issues/84)) ([73e998e](https://github.com/terrylica/rangebar-py/commit/73e998ee8982092c5ae73de1d687daa1da445188)), closes [#79](https://github.com/terrylica/rangebar-py/issues/79)

# [12.9.0](https://github.com/terrylica/rangebar-py/compare/v12.8.1...v12.9.0) (2026-02-07)


### Features

* implement 5-issue batch — exchange sessions, continuity tolerance, memory guards, JSONL tracing, backfill infra ([#78](https://github.com/terrylica/rangebar-py/issues/78), [#18](https://github.com/terrylica/rangebar-py/issues/18), [#53](https://github.com/terrylica/rangebar-py/issues/53), [#48](https://github.com/terrylica/rangebar-py/issues/48), [#80](https://github.com/terrylica/rangebar-py/issues/80)) ([b27540e](https://github.com/terrylica/rangebar-py/commit/b27540ee638166e8d423c41d57410546a797d486))

## [12.8.1](https://github.com/terrylica/rangebar-py/compare/v12.8.0...v12.8.1) (2026-02-07)


### Bug Fixes

* **clickhouse:** add intra-bar schema columns and resolve test suite issues ([6ee8bb3](https://github.com/terrylica/rangebar-py/commit/6ee8bb3e5fe34d1fb234bf9be74684f3b985b1de)), closes [#78](https://github.com/terrylica/rangebar-py/issues/78)

# [12.8.0](https://github.com/terrylica/rangebar-py/compare/v12.7.2...v12.8.0) (2026-02-06)


### Features

* **registry:** add unified symbol registry with mandatory gating ([#79](https://github.com/terrylica/rangebar-py/issues/79)) ([6c05890](https://github.com/terrylica/rangebar-py/commit/6c0589085257075139003a4d474b1146cec25cad))

## [12.7.2](https://github.com/terrylica/rangebar-py/compare/v12.7.1...v12.7.2) (2026-02-05)


### Bug Fixes

* **parquet:** deduplicate ticks on append to prevent accumulation ([#78](https://github.com/terrylica/rangebar-py/issues/78)) ([27275cb](https://github.com/terrylica/rangebar-py/commit/27275cb737fa593d28595a672c4867f456d1c008))

## [12.7.1](https://github.com/terrylica/rangebar-py/compare/v12.7.0...v12.7.1) (2026-02-05)


### Bug Fixes

* **clickhouse:** wire inter-bar and intra-bar feature columns ([#78](https://github.com/terrylica/rangebar-py/issues/78)) ([1ceacef](https://github.com/terrylica/rangebar-py/commit/1ceacef2dd0e8782be95f81ac2802aa639c5c6a4))

# [12.7.0](https://github.com/terrylica/rangebar-py/compare/v12.6.3...v12.7.0) (2026-02-05)


### Features

* **checkpoint:** auto-deduplicate after population completes ([#77](https://github.com/terrylica/rangebar-py/issues/77)) ([e0de832](https://github.com/terrylica/rangebar-py/commit/e0de832203b2512a3d0cf6f799078aee5c8ffb12))

## [12.6.3](https://github.com/terrylica/rangebar-py/compare/v12.6.2...v12.6.3) (2026-02-05)


### Bug Fixes

* **clickhouse:** add duplicate detection and deduplication methods ([#77](https://github.com/terrylica/rangebar-py/issues/77)) ([b62e99a](https://github.com/terrylica/rangebar-py/commit/b62e99a8bb3ceaacade913cb1c2b8b146f3095c8))

## [12.6.2](https://github.com/terrylica/rangebar-py/compare/v12.6.1...v12.6.2) (2026-02-05)


### Bug Fixes

* **memory:** add explicit cleanup in populate_cache_resumable and get_range_bars ([06828ab](https://github.com/terrylica/rangebar-py/commit/06828ab5408fb5f6a5e5d6d6f10061a847f445ae)), closes [#76](https://github.com/terrylica/rangebar-py/issues/76)

## [12.6.1](https://github.com/terrylica/rangebar-py/compare/v12.6.0...v12.6.1) (2026-02-05)


### Bug Fixes

* **build:** add cargo-zigbuild for Linux manylinux_2_17 compliance ([07e6097](https://github.com/terrylica/rangebar-py/commit/07e6097a3e22081bc2c994217beb2b60b46adbbb))

# [12.6.0](https://github.com/terrylica/rangebar-py/compare/v12.5.2...v12.6.0) (2026-02-05)


### Features

* **helpers:** auto-detect symbol listing dates ([#76](https://github.com/terrylica/rangebar-py/issues/76)) ([ae21d51](https://github.com/terrylica/rangebar-py/commit/ae21d51fc58edd65b388f5419ce8009e738cf8e3))

## [12.5.2](https://github.com/terrylica/rangebar-py/compare/v12.5.1...v12.5.2) (2026-02-04)


### Bug Fixes

* **orchestration:** preserve trade ID columns through cache write ([#75](https://github.com/terrylica/rangebar-py/issues/75)) ([c103c50](https://github.com/terrylica/rangebar-py/commit/c103c5068f917b6663095444329d50ae611aa4f5))

## [12.5.1](https://github.com/terrylica/rangebar-py/compare/v12.5.0...v12.5.1) (2026-02-04)


### Bug Fixes

* **helpers:** include trade ID columns in column selection ([#75](https://github.com/terrylica/rangebar-py/issues/75)) ([7cf0eb9](https://github.com/terrylica/rangebar-py/commit/7cf0eb983010c73c2bdaff01ca6e3b921aedf98c))

# [12.5.0](https://github.com/terrylica/rangebar-py/compare/v12.4.0...v12.5.0) (2026-02-04)


### Features

* **storage:** atomic Parquet writes with corruption detection ([#73](https://github.com/terrylica/rangebar-py/issues/73)) ([58d05be](https://github.com/terrylica/rangebar-py/commit/58d05be5aaa93a815a63abd16cc654aa290f68a5))

# [12.4.0](https://github.com/terrylica/rangebar-py/compare/v12.3.0...v12.4.0) (2026-02-04)


### Features

* **tracking:** add first/last agg_trade_id to Range Bars ([#72](https://github.com/terrylica/rangebar-py/issues/72)) ([9b6e379](https://github.com/terrylica/rangebar-py/commit/9b6e379d3773e6825a41c3700a0563a44eacef1e))

# [12.3.0](https://github.com/terrylica/rangebar-py/compare/v12.2.0...v12.3.0) (2026-02-04)


### Bug Fixes

* **publish:** add 1Password vault parameter for PyPI token ([559c0cf](https://github.com/terrylica/rangebar-py/commit/559c0cf6554353a4763dfd14d8c9ef0aefaae30b))


### Features

* **progress:** add tqdm progress bar for populate_cache_resumable ([#70](https://github.com/terrylica/rangebar-py/issues/70)) ([cd5a662](https://github.com/terrylica/rangebar-py/commit/cd5a6623fddb93986e1dae562ede680e65206e02))

# [12.2.0](https://github.com/terrylica/rangebar-py/compare/v12.1.2...v12.2.0) (2026-02-04)


### Features

* **mem-013:** long-range date protection and cache workflow ([#69](https://github.com/terrylica/rangebar-py/issues/69)) ([8a07b50](https://github.com/terrylica/rangebar-py/commit/8a07b5076344d837459cadfab9f3c99270420f73))

## [12.1.2](https://github.com/terrylica/rangebar-py/compare/v12.1.1...v12.1.2) (2026-02-04)


### Bug Fixes

* **interbar:** preserve lookback trades across bar boundaries ([#68](https://github.com/terrylica/rangebar-py/issues/68)) ([d8e3fc9](https://github.com/terrylica/rangebar-py/commit/d8e3fc94bc160c20314801e5e3be684969972888))

## [12.1.1](https://github.com/terrylica/rangebar-py/compare/v12.1.0...v12.1.1) (2026-02-03)


### Bug Fixes

* **microstructure:** auto-enable v12 features when include_microstructure=True ([9288e2f](https://github.com/terrylica/rangebar-py/commit/9288e2f5bc2f412d81d1954d308cea938d027b48)), closes [#68](https://github.com/terrylica/rangebar-py/issues/68)

# [12.1.0](https://github.com/terrylica/rangebar-py/compare/v12.0.2...v12.1.0) (2026-02-03)


### Bug Fixes

* **mem:** MEM-012 streaming bar accumulation prevents OOM ([#67](https://github.com/terrylica/rangebar-py/issues/67)) ([6cf640c](https://github.com/terrylica/rangebar-py/commit/6cf640c2264ffdcf75efdb38af935e83b903c6c4))
* **mise:** update tracking ref after SSH fallback push ([1b4f037](https://github.com/terrylica/rangebar-py/commit/1b4f037c145a0e79f91e4b1121ee9e0586cd6d44))


### Features

* **mise:** SSH fallback, PyPI pre-check, diagnostics ([c0533d8](https://github.com/terrylica/rangebar-py/commit/c0533d869a03bfcd11d35eb400682767d93e38cf))

## [12.0.2](https://github.com/terrylica/rangebar-py/compare/v12.0.1...v12.0.2) (2026-02-03)


### Bug Fixes

* **memory:** adaptive chunk size for microstructure mode (MEM-011) ([214d3bc](https://github.com/terrylica/rangebar-py/commit/214d3bc6d965a15c869441265e393bd6e5678dd3)), closes [#65](https://github.com/terrylica/rangebar-py/issues/65)

# [12.0.0](https://github.com/terrylica/rangebar-py/compare/v11.7.0...v12.0.0) (2026-02-03)


### Features

* **threshold:** enforce configurable minimum threshold for crypto (Issue [#64](https://github.com/terrylica/rangebar-py/issues/64)) ([4030c7d](https://github.com/terrylica/rangebar-py/commit/4030c7dc9e6789a4c9ae11f21c03261dc56e6c95))


### BREAKING CHANGES

* **threshold:** ThresholdError raised for crypto symbols with threshold < 1000 dbps

- Add threshold.py module with hierarchical SSoT configuration
- Per-symbol override: RANGEBAR_MIN_THRESHOLD_<SYMBOL>
- Asset-class default: RANGEBAR_<ASSET_CLASS>_MIN_THRESHOLD
- Default crypto minimum: 1000 dbps (1%) per CFM research
- Add validation to all Python entry points (21 locations)
- Add validation to Rust checkpoint restoration (lib.rs, processor.rs)
- Add NDJSON logging and Pushover alerts for violations
- Add cache purge script for invalidating low-threshold data
- Add 32 comprehensive tests for threshold enforcement

SRED-Type: applied-research
SRED-Claim: THRESHOLD

# [11.7.0](https://github.com/terrylica/rangebar-py/compare/v11.6.1...v11.7.0) (2026-02-02)


### Features

* **intrabar:** add 22 intra-bar microstructure features (Issue [#59](https://github.com/terrylica/rangebar-py/issues/59)) ([4200631](https://github.com/terrylica/rangebar-py/commit/4200631576269ad90b957f6c250e07866d30c3e4))

## [11.6.1](https://github.com/terrylica/rangebar-py/compare/v11.6.0...v11.6.1) (2026-02-02)


### Bug Fixes

* **ci:** correct rust-toolchain action name ([fa252ce](https://github.com/terrylica/rangebar-py/commit/fa252ceeb1b2f3d4216fceb3e0eaf3fbdc4bed6a))
* **ci:** simplify workflow to match working configuration ([5128fd6](https://github.com/terrylica/rangebar-py/commit/5128fd6a5ade367dbde371c186de0512095c0116))
* **resource:** make auto_memory_guard() multiprocessing-safe (Issue [#61](https://github.com/terrylica/rangebar-py/issues/61)) ([b55057a](https://github.com/terrylica/rangebar-py/commit/b55057afb3a10f530ec8d1ac1520d81169ad5fcc))

# [11.6.0](https://github.com/terrylica/rangebar-py/compare/v11.5.0...v11.6.0) (2026-02-02)


### Features

* **core:** add inter-bar microstructure features (Issue [#59](https://github.com/terrylica/rangebar-py/issues/59) Phase 1) ([dd27556](https://github.com/terrylica/rangebar-py/commit/dd275565fbd093b6d5647e51c72548ef4be1e715))
* **inter-bar:** add inter-bar feature computation (Issue [#59](https://github.com/terrylica/rangebar-py/issues/59)) ([94dbf64](https://github.com/terrylica/rangebar-py/commit/94dbf647f3ba524fb818b4927080afc2fc9b7ea5))

# [11.5.0](https://github.com/terrylica/rangebar-py/compare/v11.4.0...v11.5.0) (2026-02-02)


### Bug Fixes

* **mise:** include all symbols in sequential population task ([18e0c4d](https://github.com/terrylica/rangebar-py/commit/18e0c4dc44c0404a0dec483a6e8e50fe5f7256b5)), closes [#58](https://github.com/terrylica/rangebar-py/issues/58)
* **mise:** use uv run for venv and sequential execution ([957354e](https://github.com/terrylica/rangebar-py/commit/957354e46c1a0aa310560e44ecdfeff81b4c42ef)), closes [#58](https://github.com/terrylica/rangebar-py/issues/58)
* **research:** correct datetime timezone import in regime analysis ([d1da329](https://github.com/terrylica/rangebar-py/commit/d1da329239a6ef3278a2231e28f988948cd38bd2)), closes [#76](https://github.com/terrylica/rangebar-py/issues/76) [#52](https://github.com/terrylica/rangebar-py/issues/52)
* **research:** correct transaction costs to 15 dbps for high VIP tier ([3c129cf](https://github.com/terrylica/rangebar-py/commit/3c129cf970e04ae49e5d893981399bc359a7943a)), closes [#90](https://github.com/terrylica/rangebar-py/issues/90)
* **scripts:** disable RLIMIT_AS for population on Linux ([aa3b541](https://github.com/terrylica/rangebar-py/commit/aa3b541062a88cd8c8e088a24776b7de45506068))


### Features

* **forex:** add EURUSD range bar pipeline scripts ([5e1628c](https://github.com/terrylica/rangebar-py/commit/5e1628c31e027824de208bd24acfdb1d5e789e7c)), closes [#143](https://github.com/terrylica/rangebar-py/issues/143)
* **memory:** auto-enable memory guard on import (MEM-011) ([5e57767](https://github.com/terrylica/rangebar-py/commit/5e5776798104189fdd7c3d5b9a371a5b5c0908e2)), closes [#49](https://github.com/terrylica/rangebar-py/issues/49)
* **mise:** add 1000 dbps cache population and unified task ([9ad487e](https://github.com/terrylica/rangebar-py/commit/9ad487e11c725c069e0c6cf9d3b2f487ef5038a7)), closes [#58](https://github.com/terrylica/rangebar-py/issues/58)
* **mise:** add population status and monitor tasks ([f4a6197](https://github.com/terrylica/rangebar-py/commit/f4a619793a4144958dcdb18c0211cfba2a0946e8)), closes [#58](https://github.com/terrylica/rangebar-py/issues/58)
* **mise:** add sequential cache population tasks for Issue [#58](https://github.com/terrylica/rangebar-py/issues/58) ([f5010d0](https://github.com/terrylica/rangebar-py/commit/f5010d02c01a97f6009e49ac74eb53d2a9833595))
* **research:** add 3-bar + alignment combined analysis ([ff9a538](https://github.com/terrylica/rangebar-py/commit/ff9a5384c0ad6387cfa2de34e2adafe1bc3d5247)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54) [#55](https://github.com/terrylica/rangebar-py/issues/55)
* **research:** add 3-bar pattern adversarial audit ([f2c8ba9](https://github.com/terrylica/rangebar-py/commit/f2c8ba9dd75f93496a5def5eb7395f660fc8b424)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54) [#55](https://github.com/terrylica/rangebar-py/issues/55)
* **research:** add 3-bar pattern analysis script ([4f984ad](https://github.com/terrylica/rangebar-py/commit/4f984adf78635066ed9d297612646228cbe77721)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54) [#55](https://github.com/terrylica/rangebar-py/issues/55)
* **research:** add ADWIN regime detection analysis ([de1f721](https://github.com/terrylica/rangebar-py/commit/de1f721401d3224df2067eae1b0fc14cfcfeb1ef)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54)
* **research:** add Benjamini-Hochberg FDR correction for pattern testing ([7c159f6](https://github.com/terrylica/rangebar-py/commit/7c159f651f6373e3a3b8bbc7aac1d1cb163c424f)), closes [#52](https://github.com/terrylica/rangebar-py/issues/52) [#56](https://github.com/terrylica/rangebar-py/issues/56)
* **research:** add combined SMA/RSI x RV regime analysis ([789b5b0](https://github.com/terrylica/rangebar-py/commit/789b5b0301567aba3c04afc0d96037df7b2338d1)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54)
* **research:** add cross-asset correlation analysis (Issue [#145](https://github.com/terrylica/rangebar-py/issues/145)) ([ee5260c](https://github.com/terrylica/rangebar-py/commit/ee5260c8abc234792b06b4d6f2c898a314ee28a9))
* **research:** add extended symbol set for 250 dbps cross-board examination ([2cbc3f7](https://github.com/terrylica/rangebar-py/commit/2cbc3f72a58c23967619443e8acfa8f183c26c12)), closes [#58](https://github.com/terrylica/rangebar-py/issues/58)
* **research:** add GPU-accelerated TDA structural break detection ([afc4201](https://github.com/terrylica/rangebar-py/commit/afc42017a1452aba00bc7414c77707f2ca58e4e5))
* **research:** add Hurst exponent analysis for long memory effects ([cc66023](https://github.com/terrylica/rangebar-py/commit/cc660235b1059be7c6730cd989f2ca6dadb64314)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54)
* **research:** add Hurst exponent analysis per TDA regime ([5a731fc](https://github.com/terrylica/rangebar-py/commit/5a731fc8836b24822fc8e0ff919feff565425746)), closes [#56](https://github.com/terrylica/rangebar-py/issues/56)
* **research:** add Hurst-adjusted Kelly fraction analysis ([149b754](https://github.com/terrylica/rangebar-py/commit/149b754d7276874ec0fe088357d9e221938834ec)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54)
* **research:** add Hurst-adjusted PSR/MinTRL validation ([c2709da](https://github.com/terrylica/rangebar-py/commit/c2709da5c5d54b7715e7fcfab2d7b5d4d40c7d3a)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54)
* **research:** add Kelly criterion position sizing analysis ([25dab7b](https://github.com/terrylica/rangebar-py/commit/25dab7b1963224dffeb574e78986556459aad6cb)), closes [#92](https://github.com/terrylica/rangebar-py/issues/92)
* **research:** add market regime analysis script ([4eea8f9](https://github.com/terrylica/rangebar-py/commit/4eea8f99dc7777440536c1221f6df0088df7b84f)), closes [#74](https://github.com/terrylica/rangebar-py/issues/74) [#52](https://github.com/terrylica/rangebar-py/issues/52)
* **research:** add memory-safe population script ([e7e871e](https://github.com/terrylica/rangebar-py/commit/e7e871e11632701470d0e14a271a1e67b0e1465b)), closes [hi#volume](https://github.com/hi/issues/volume)
* **research:** add microstructure pattern analysis scripts ([bb0a8d2](https://github.com/terrylica/rangebar-py/commit/bb0a8d25dbce4a3fc1a0c9af50d51fa7adb826f6)), closes [#52](https://github.com/terrylica/rangebar-py/issues/52) [#56](https://github.com/terrylica/rangebar-py/issues/56)
* **research:** add multi-bar horizon analysis - confirms genuine alpha ([bd2d326](https://github.com/terrylica/rangebar-py/commit/bd2d326e3e9ba9727d6bea640719a16b9952bf5b)), closes [#80](https://github.com/terrylica/rangebar-py/issues/80)
* **research:** add multi-factor multi-granularity pattern analysis ([61847d4](https://github.com/terrylica/rangebar-py/commit/61847d44511500ef533cdc9d2bc425f6357cfde3)), closes [#52](https://github.com/terrylica/rangebar-py/issues/52)
* **research:** add multi-threshold combination analysis ([74a658d](https://github.com/terrylica/rangebar-py/commit/74a658d59d1d708f9c72e57d003ec3da37d5900c)), closes [#86](https://github.com/terrylica/rangebar-py/issues/86) [#52](https://github.com/terrylica/rangebar-py/issues/52) [#86](https://github.com/terrylica/rangebar-py/issues/86)
* **research:** add parameter sensitivity analysis for regime patterns ([3d7b969](https://github.com/terrylica/rangebar-py/commit/3d7b9698fba12ed8ee9c5d075ca6d7df06f357cd)), closes [#89](https://github.com/terrylica/rangebar-py/issues/89)
* **research:** add pattern correlation analysis for portfolio diversification ([5b2a536](https://github.com/terrylica/rangebar-py/commit/5b2a5369e1d125c5213ab7e9c232d5ae7b1a18df))
* **research:** add population monitoring script ([5e461aa](https://github.com/terrylica/rangebar-py/commit/5e461aa3c0d6f60897e0a8af501b8835afc2a9e1)), closes [#58](https://github.com/terrylica/rangebar-py/issues/58)
* **research:** add PSR/MinTRL stationarity gap analysis ([0557b27](https://github.com/terrylica/rangebar-py/commit/0557b274c6cdd950e8bd3ae9a6c972a544f055e2)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54) [#55](https://github.com/terrylica/rangebar-py/issues/55)
* **research:** add regime + multi-threshold combination analysis ([f996275](https://github.com/terrylica/rangebar-py/commit/f996275b295196aa2495457edae34836f79fa190)), closes [#52](https://github.com/terrylica/rangebar-py/issues/52) [#87](https://github.com/terrylica/rangebar-py/issues/87)
* **research:** add Ripser++ GPU TDA script ([72d13c7](https://github.com/terrylica/rangebar-py/commit/72d13c7f34c4a13208b2a6539d81ab90681c3df8))
* **research:** add rolling TDA threshold (no data leakage) ([38f67f5](https://github.com/terrylica/rangebar-py/commit/38f67f587f21595d0beeef16b644c15eaee75262)), closes [#52](https://github.com/terrylica/rangebar-py/issues/52) [#56](https://github.com/terrylica/rangebar-py/issues/56)
* **research:** add TDA break event alignment analysis ([3960eb6](https://github.com/terrylica/rangebar-py/commit/3960eb6cf3c30eef528cb6701319973f86893040)), closes [#56](https://github.com/terrylica/rangebar-py/issues/56)
* **research:** add TDA regime Hurst exponent analysis ([9626d07](https://github.com/terrylica/rangebar-py/commit/9626d0742aa98e0e8387111ad76fb6e3219df3ed)), closes [#56](https://github.com/terrylica/rangebar-py/issues/56)
* **research:** add TDA regime pattern analysis ([7055673](https://github.com/terrylica/rangebar-py/commit/70556738af8a6f5d2e267ba9af751f2c8647dcd9)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54) [#56](https://github.com/terrylica/rangebar-py/issues/56)
* **research:** add TDA regime-conditioned pattern ODD robustness testing ([89a0b19](https://github.com/terrylica/rangebar-py/commit/89a0b193013942b4777cba36b831506c854e5f7d)), closes [#56](https://github.com/terrylica/rangebar-py/issues/56)
* **research:** add TDA structural break detection analysis ([32b20a1](https://github.com/terrylica/rangebar-py/commit/32b20a1a678ea9e9c9ba6a7e34df1c57e866b5d5)), closes [#114](https://github.com/terrylica/rangebar-py/issues/114)
* **research:** add TDA structural break detection script ([1480ee9](https://github.com/terrylica/rangebar-py/commit/1480ee94bd4220a17acb6066ff982c7d62caebea)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54)
* **research:** add temporal-safe pattern validation ([bcf1c0d](https://github.com/terrylica/rangebar-py/commit/bcf1c0d25e94ee2537775764aa88eef818bbfeff)), closes [#52](https://github.com/terrylica/rangebar-py/issues/52) [#56](https://github.com/terrylica/rangebar-py/issues/56)
* **research:** add three-factor Hurst exponent analysis ([dd15576](https://github.com/terrylica/rangebar-py/commit/dd15576280742ed0adf168c175521de53ae0d27b)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54)
* **research:** add three-factor pattern analysis ([26659a8](https://github.com/terrylica/rangebar-py/commit/26659a8d0b633d13e275313cd731b288a7c8ab1d)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54) [#55](https://github.com/terrylica/rangebar-py/issues/55)
* **research:** add transaction cost analysis for regime patterns ([8c555e0](https://github.com/terrylica/rangebar-py/commit/8c555e0460e0b18710ceff34ee5831798b1c8e02)), closes [#90](https://github.com/terrylica/rangebar-py/issues/90)
* **research:** add volatility regime pattern analysis (Issue [#54](https://github.com/terrylica/rangebar-py/issues/54)) ([64feab1](https://github.com/terrylica/rangebar-py/commit/64feab154dc838b67b9fe41858f0fd536637f843))
* **research:** adversarial audit of combined RV+alignment patterns ([c5178b6](https://github.com/terrylica/rangebar-py/commit/c5178b6efe72e7706e362aeff8cc65a2a8c78732)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54) [#55](https://github.com/terrylica/rangebar-py/issues/55)
* **research:** adversarial audit of volatility regime patterns ([7c30a46](https://github.com/terrylica/rangebar-py/commit/7c30a46d4d9b874aa916cc550ce3667d5b1f3edd)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54)
* **research:** analyze 50 dbps patterns - fewer robust than 100 dbps ([da7af16](https://github.com/terrylica/rangebar-py/commit/da7af1623b987e283ddbd7ae332262a4840c7aab)), closes [#91](https://github.com/terrylica/rangebar-py/issues/91)
* **research:** combined RV regime + multi-threshold alignment analysis ([7934be9](https://github.com/terrylica/rangebar-py/commit/7934be923127e240d5b0380a1785fcf64c107921)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54) [#55](https://github.com/terrylica/rangebar-py/issues/55)
* **research:** cross-regime correlation analysis for portfolio diversification ([bd5a13c](https://github.com/terrylica/rangebar-py/commit/bd5a13c5e175b753035d1639f392334cafd901f8)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54)
* **research:** cross-threshold signal alignment analysis ([1f43e1e](https://github.com/terrylica/rangebar-py/commit/1f43e1e783ebabd1c8bfd3919a4d0e63f9c32bfe)), closes [#52](https://github.com/terrylica/rangebar-py/issues/52) [#56](https://github.com/terrylica/rangebar-py/issues/56)
* **research:** microstructure feature analysis via ClickHouse ([45f1627](https://github.com/terrylica/rangebar-py/commit/45f16276bf46af8fb5c9040346bf67e59beed490)), closes [#52](https://github.com/terrylica/rangebar-py/issues/52) [#56](https://github.com/terrylica/rangebar-py/issues/56)
* **research:** multi-threshold pattern confirmation signals ([b4fc08a](https://github.com/terrylica/rangebar-py/commit/b4fc08a3dd232788d783a5d53a7ed63e74645845)), closes [#55](https://github.com/terrylica/rangebar-py/issues/55)
* **research:** return profile analysis for RV regime patterns ([6b626e7](https://github.com/terrylica/rangebar-py/commit/6b626e76c74e5ba2619f3244087262a2d1538e16)), closes [#54](https://github.com/terrylica/rangebar-py/issues/54)

# [11.4.0](https://github.com/terrylica/rangebar-py/compare/v11.3.0...v11.4.0) (2026-01-31)


### Bug Fixes

* **release:** enforce wheel build dependency before PyPI publish ([39e8fe4](https://github.com/terrylica/rangebar-py/commit/39e8fe40581156c7dc131920cd819281078039da))
* **research:** correct triple barrier chart to show profit target hit first ([4ca393e](https://github.com/terrylica/rangebar-py/commit/4ca393eef4a7b2fbe1aa4a304c5b0582bd11cd2d))


### Features

* **streaming:** add real-time streaming API for Binance WebSocket ([1b2dbd3](https://github.com/terrylica/rangebar-py/commit/1b2dbd31e80ee31ad05b70d12a81911d6bde6134))
* **streaming:** add reconnection jitter and max duration to prevent thundering herd ([5acc4ba](https://github.com/terrylica/rangebar-py/commit/5acc4ba3373d9ba8c5ea894349f1bfe396758f64))

# [11.3.0](https://github.com/terrylica/rangebar-py/compare/v11.2.2...v11.3.0) (2026-01-30)


### Bug Fixes

* **clickhouse:** cast exchange session bools for ClickHouse insert (Issue [#50](https://github.com/terrylica/rangebar-py/issues/50)) ([24c5676](https://github.com/terrylica/rangebar-py/commit/24c56765e514aced311b335b3203fcc39c93ab1a))
* **memory:** MEM-004 add size guard to read_ticks (Issue [#49](https://github.com/terrylica/rangebar-py/issues/49) T1.2) ([4aa54e5](https://github.com/terrylica/rangebar-py/commit/4aa54e50dc37f9ca0896a88cb896532473ae5f3d))
* **memory:** MEM-007 guard deprecated _fetch_binance path (Issue [#49](https://github.com/terrylica/rangebar-py/issues/49) T1.3) ([fefd3ab](https://github.com/terrylica/rangebar-py/commit/fefd3ab19dfc4bb31c280ed120037b132e2a2817)), closes [hi#volume](https://github.com/hi/issues/volume)
* **range-bars:** load ticks per-segment to prevent OOM (Issue [#51](https://github.com/terrylica/rangebar-py/issues/51)) ([6134fd8](https://github.com/terrylica/rangebar-py/commit/6134fd80e96d3230d54259b68542a45c1a7edc6f))


### Features

* **hooks:** add memory snapshot to hook payloads (Issue [#49](https://github.com/terrylica/rangebar-py/issues/49) T2.3) ([2ed58aa](https://github.com/terrylica/rangebar-py/commit/2ed58aad0ca40c6eba8aad24eb1e1a007f7e3a25))
* **memory:** add memory cap to precompute_range_bars (Issue [#49](https://github.com/terrylica/rangebar-py/issues/49) T2.2) ([6199ff1](https://github.com/terrylica/rangebar-py/commit/6199ff1b50059f8e23a79564f8a89c3c5251ef91))
* **memory:** integrate pre-flight estimation into get_range_bars (Issue [#49](https://github.com/terrylica/rangebar-py/issues/49) T2.1) ([aa33210](https://github.com/terrylica/rangebar-py/commit/aa3321090922c360db99e728f64cb7a55d2f95d0))
* **memory:** MEM-009/010 resource guard module (Issue [#49](https://github.com/terrylica/rangebar-py/issues/49) T1.4+T1.5) ([c974f05](https://github.com/terrylica/rangebar-py/commit/c974f052e2b8175a374396306f793b1f7f013055))

## [11.2.2](https://github.com/terrylica/rangebar-py/compare/v11.2.1...v11.2.2) (2026-01-29)


### Bug Fixes

* **scripts:** add OOM-safe cache regeneration script (Issue [#47](https://github.com/terrylica/rangebar-py/issues/47)) ([d182ed6](https://github.com/terrylica/rangebar-py/commit/d182ed6893bbf5585652bc53bd2aed020d2ae2e4))

## [11.2.1](https://github.com/terrylica/rangebar-py/compare/v11.2.0...v11.2.1) (2026-01-29)


### Bug Fixes

* **release:** enforce sequential task dependencies in release:full ([7ef30ea](https://github.com/terrylica/rangebar-py/commit/7ef30ea8b88517156eedecddde563b40a067d3a5))
* **streaming:** defer new bar open after threshold breach (Issue [#46](https://github.com/terrylica/rangebar-py/issues/46)) ([938658c](https://github.com/terrylica/rangebar-py/commit/938658c6aad4dedc064b377fac10a29f9483c7a9))

# [11.2.0](https://github.com/terrylica/rangebar-py/compare/v11.1.0...v11.2.0) (2026-01-29)


### Bug Fixes

* **cache:** add exchange session columns to store_range_bars write path ([65a7ecd](https://github.com/terrylica/rangebar-py/commit/65a7ecd9ee83c3fe485b410d80cae93e3e65cf2d))


### Features

* **api:** add exchange sessions and fix ClickHouse connectivity ([d76eb82](https://github.com/terrylica/rangebar-py/commit/d76eb82a5726043d2456df0de91707c37133d698)), closes [#8](https://github.com/terrylica/rangebar-py/issues/8)

# [11.1.0](https://github.com/terrylica/rangebar-py/compare/v11.0.0...v11.1.0) (2026-01-29)


### Features

* **api:** expose process_trades_polars and enhance Polars documentation (Issue [#45](https://github.com/terrylica/rangebar-py/issues/45)) ([c821bb8](https://github.com/terrylica/rangebar-py/commit/c821bb8542ce2e05efaa552871f0b410b3b9bb7c))
* **cache:** add schema evolution with staleness detection (Issue [#39](https://github.com/terrylica/rangebar-py/issues/39)) ([2a13f33](https://github.com/terrylica/rangebar-py/commit/2a13f33b45bf0a335621895db91ecac96cac8d5c))

# [11.0.0](https://github.com/terrylica/rangebar-py/compare/v10.1.0...v11.0.0) (2026-01-27)


### Bug Fixes

* **polars:** normalize datetime precision before concat ([#44](https://github.com/terrylica/rangebar-py/issues/44)) ([716af12](https://github.com/terrylica/rangebar-py/commit/716af12387500c58e9b7370109818162a270ca39))
* **pushover:** use Dune custom sound for critical alerts ([13485b8](https://github.com/terrylica/rangebar-py/commit/13485b8ab4c847fb46181aa10e75e230c9187958)), closes [#43](https://github.com/terrylica/rangebar-py/issues/43)


### Features

* **ouroboros:** add reset_at_ouroboros method for reproducible bar construction ([8ba9113](https://github.com/terrylica/rangebar-py/commit/8ba9113f36a0dd5eb37fcfed241656248fb399da))
* **ouroboros:** implement mandatory cyclical reset boundaries for reproducible range bars ([1337bed](https://github.com/terrylica/rangebar-py/commit/1337bed2eac4e592a5699824d314dae6814cf183))


### BREAKING CHANGES

* **ouroboros:** get_range_bars() now requires ouroboros parameter.
Default is "year", which resets processor state at January 1 boundaries.
Different ouroboros modes cache separately to prevent mixing.

Resolves reproducibility issue: identical parameters now produce
identical bar sequences across different execution times.

SRED-Type: experimental-development
SRED-Claim: OUROBOROS

# [10.1.0](https://github.com/terrylica/rangebar-py/compare/v10.0.0...v10.1.0) (2026-01-27)


### Features

* **checksum:** add SHA-256 verification for Binance downloads ([#43](https://github.com/terrylica/rangebar-py/issues/43)) ([21da3b6](https://github.com/terrylica/rangebar-py/commit/21da3b6d6bf2bd3453640bd6deaf0021fd09fea8))

# [10.0.0](https://github.com/terrylica/rangebar-py/compare/v9.1.0...v10.0.0) (2026-01-23)


### Build System

* **python:** drop Python 3.10-3.12, require Python 3.13 only ([a648fba](https://github.com/terrylica/rangebar-py/commit/a648fba87743a92beb99e697877a30ba07117ef3))


### BREAKING CHANGES

* **python:** Minimum Python version raised from 3.10 to 3.13.

Updates across all configuration and documentation:
- pyproject.toml: requires-python, classifiers, mypy/black/ruff targets
- .mise.toml: python tool version, build task descriptions, Docker commands
- scripts/build-release.sh: PYTHON_VERSIONS array
- README.md: pre-built wheels, runtime requirements
- docs/api.md: Python versions compatibility
- docs/migration-v8.md: wheel build note

SRED-Type: support-work
SRED-Claim: RANGEBAR-BUILD

# [9.1.0](https://github.com/terrylica/rangebar-py/compare/v9.0.2...v9.1.0) (2026-01-20)


### Features

* **cache:** implement GitHub issues [#35](https://github.com/terrylica/rangebar-py/issues/35)-42 cache reliability and UX improvements ([1143634](https://github.com/terrylica/rangebar-py/commit/11436340225466b343d593cebdb09df644e306ca)), closes [#35-42](https://github.com/terrylica/rangebar-py/issues/35-42)

## [9.0.2](https://github.com/terrylica/rangebar-py/compare/v9.0.1...v9.0.2) (2026-01-20)


### Bug Fixes

* prevent process storms with read_file() for GitHub tokens ([d9cd3c1](https://github.com/terrylica/rangebar-py/commit/d9cd3c1d73a51ad4841cfd3127ffddb26daa381a))
* remove conflicting --release flag with --profile wheel ([f13b002](https://github.com/terrylica/rangebar-py/commit/f13b002c6b3d450ce20b22e8b05bd2550a9b19c4))
* update publish script for dynamic version SSoT ([7c190c2](https://github.com/terrylica/rangebar-py/commit/7c190c2b259a847eb9faef979ffe99891151ba33))

## [9.0.1](https://github.com/terrylica/rangebar-py/compare/v9.0.0...v9.0.1) (2026-01-14)

### Bug Fixes

- establish Cargo.toml as single source of truth for version ([3c68614](https://github.com/terrylica/rangebar-py/commit/3c68614dc4a277d8e6ee6154b5f4487f2bb2bb69))

# [9.0.0](https://github.com/terrylica/rangebar-py/compare/v8.0.1...v9.0.0) (2026-01-14)

- feat!: add timestamp-gated breach detection to eliminate duplicate timestamps ([54a5340](https://github.com/terrylica/rangebar-py/commit/54a53405cbafd10c6902aa4c49f9a3858d74297b)), closes [#36](https://github.com/terrylica/rangebar-py/issues/36)

### BREAKING CHANGES

- Default behavior now prevents bars from closing at the
  same timestamp they opened. Set prevent_same_timestamp_close=False for
  previous behavior.

See: docs/analysis/2025-10-10-flash-crash.md

## [8.0.1](https://github.com/terrylica/rangebar-py/compare/v8.0.0...v8.0.1) (2026-01-13)

### Bug Fixes

- rename aggregation_efficiency to aggregation_density ([c8bc1d8](https://github.com/terrylica/rangebar-py/commit/c8bc1d864d27e5094f210a67c0a647ef9c10f5ed))
- rename aggregation_efficiency to aggregation_density in schema ([9d33b10](https://github.com/terrylica/rangebar-py/commit/9d33b106ed87a9f1875e580d103c4199f58bc419))

# [8.0.0](https://github.com/terrylica/rangebar-py/compare/v7.1.3...v8.0.0) (2026-01-13)

### Bug Fixes

- apply column selection before .collect() for predicate pushdown (MEM-003) ([03f1c47](https://github.com/terrylica/rangebar-py/commit/03f1c471c8dfc88f6d96cec0d51709622e2f8277))
- **MEM-001:** vectorize \_timestamp_to_year_month to eliminate per-tick Python calls ([39245c5](https://github.com/terrylica/rangebar-py/commit/39245c53ce3b19ba092ebd735cb65890bd638ece))
- **MEM-002:** chunk .to_dicts() calls to bound memory usage ([d190d83](https://github.com/terrylica/rangebar-py/commit/d190d832d4621b31986153bc3133169db0aef4aa))
- **MEM-005:** add test suite memory isolation with GC fixture ([3b0c144](https://github.com/terrylica/rangebar-py/commit/3b0c1442fd2828de0a1d7bf3416c8afb8909b48e))
- **MEM-006:** replace pd.concat with Polars for memory efficiency ([e6fb056](https://github.com/terrylica/rangebar-py/commit/e6fb056a8c05ddaaebf4bb85168656908dab5cfa))

### Features

- add get_range_bars_pandas backward compatibility shim (Phase 5.1) ([b582d72](https://github.com/terrylica/rangebar-py/commit/b582d72dd47f53b20bdb8d9a9320224b73abbc4b))
- add store_bars_batch for Arrow-based streaming cache writes (Phase 4.3) ([c7d293e](https://github.com/terrylica/rangebar-py/commit/c7d293e3ed7b70d964954c53d53e2b6297fba352))
- add streaming-first API with materialize parameter (Phase 4) ([2d6fa25](https://github.com/terrylica/rangebar-py/commit/2d6fa250f2acc23af192bcd7a8121ed57c05450f))
- **python:** expose process_trades_streaming_arrow in RangeBarProcessor ([ee16d47](https://github.com/terrylica/rangebar-py/commit/ee16d479bef977e8fc121478f6fe50da79e58287))
- **streaming:** add memory-efficient streaming architecture (Phases 1-3) ([4ec670d](https://github.com/terrylica/rangebar-py/commit/4ec670d95b01af0f05478220e817a144eb4fc478)), closes [#32](https://github.com/terrylica/rangebar-py/issues/32)

### BREAKING CHANGES

- get_range_bars() return type is now
  pd.DataFrame | Iterator[pl.DataFrame]. Use materialize=True
  (default) for backward compatibility.

Memory impact: ~50 MB per 6-hour chunk vs 5.6 GB for full day

## [7.1.3](https://github.com/terrylica/rangebar-py/compare/v7.1.2...v7.1.3) (2026-01-12)

### Bug Fixes

- compute microstructure features in batch processing path (Issue [#34](https://github.com/terrylica/rangebar-py/issues/34)) ([531c9b1](https://github.com/terrylica/rangebar-py/commit/531c9b16fc29b270665574adabaf23577fb966ce))

## [7.1.1](https://github.com/terrylica/rangebar-py/compare/v7.1.0...v7.1.1) (2026-01-12)

### Bug Fixes

- pass is_buyer_maker to Rust for microstructure features (Issue [#30](https://github.com/terrylica/rangebar-py/issues/30)) ([6c32dbb](https://github.com/terrylica/rangebar-py/commit/6c32dbb2d0f6860d10d0e3316b877e5371f34804))

# [7.1.0](https://github.com/terrylica/rangebar-py/compare/v7.0.1...v7.1.0) (2026-01-12)

### Features

- add incremental ClickHouse caching with post-cache validation (Issue [#27](https://github.com/terrylica/rangebar-py/issues/27)) ([7318da4](https://github.com/terrylica/rangebar-py/commit/7318da4002c681c54caf898db9510db811d20410)), closes [#5](https://github.com/terrylica/rangebar-py/issues/5)

## [7.0.1](https://github.com/terrylica/rangebar-py/compare/v7.0.0...v7.0.1) (2026-01-11)

### Bug Fixes

- update crate readme paths to local READMEs for sdist build ([9559ff9](https://github.com/terrylica/rangebar-py/commit/9559ff979faf701b5b049ca3d194ca247501dc20))
- update internal crate version references from 6.0 to 7.0 ([41d4198](https://github.com/terrylica/rangebar-py/commit/41d4198aa223855e95d09ffbef85224831ae58bf))

# [7.0.0](https://github.com/terrylica/rangebar-py/compare/v6.2.1...v7.0.0) (2026-01-11)

- feat!: add 10 market microstructure features computed in Rust (Issue [#25](https://github.com/terrylica/rangebar-py/issues/25)) ([be4be75](https://github.com/terrylica/rangebar-py/commit/be4be75c691d39863c25ccb297004f956e66fde7))

### BREAKING CHANGES

- RangeBar struct extended with 10 new fields. Cached bars
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

- add ClickHouse bar cache lookup to get_range_bars() and fix TZ-aware timestamps (Issues [#20](https://github.com/terrylica/rangebar-py/issues/20), [#21](https://github.com/terrylica/rangebar-py/issues/21)) ([b9fa078](https://github.com/terrylica/rangebar-py/commit/b9fa078e6bfc8f87da1753412693e74b5dd17d5d))

# [6.2.0](https://github.com/terrylica/rangebar-py/compare/v6.1.0...v6.2.0) (2026-01-11)

### Features

- add tiered validation system for market microstructure events (Issue [#19](https://github.com/terrylica/rangebar-py/issues/19)) ([685bb49](https://github.com/terrylica/rangebar-py/commit/685bb498654d87506ba9cd686e700b089d78e491))

# [6.1.0](https://github.com/terrylica/rangebar-py/compare/v6.0.0...v6.1.0) (2026-01-11)

### Bug Fixes

- add process_trades_streaming() for cross-batch state persistence (Issue [#16](https://github.com/terrylica/rangebar-py/issues/16)) ([5951fce](https://github.com/terrylica/rangebar-py/commit/5951fce665662997916b4602e67b44211b572e3d))
- align internal crate version requirements with 6.0.0 ([73409eb](https://github.com/terrylica/rangebar-py/commit/73409eb48fed56a75866f6b7b080cc171b23a369))

### Features

- add granular mise release tasks with 4-phase workflow ([6b6482a](https://github.com/terrylica/rangebar-py/commit/6b6482a602770b6758c425a1c312c513d2ab7e4a)), closes [#17](https://github.com/terrylica/rangebar-py/issues/17)

# [6.0.0](https://github.com/terrylica/rangebar-py/compare/v5.4.0...v6.0.0) (2026-01-10)

### Features

- consolidate 8-crate Rust workspace from upstream rangebar ([f16f50b](https://github.com/terrylica/rangebar-py/commit/f16f50b12853dbaf4213c97ca45c2f94940d6eb9))

### BREAKING CHANGES

- Architecture now includes full 8-crate workspace

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

- add memory efficiency validation script ([558b6b7](https://github.com/terrylica/rangebar-py/commit/558b6b7520c857e6b5d411029bd44f052d7e6aff)), closes [#12](https://github.com/terrylica/rangebar-py/issues/12) high-volume [#15](https://github.com/terrylica/rangebar-py/issues/15)

## [5.3.6](https://github.com/terrylica/rangebar-py/compare/v5.3.5...v5.3.6) (2026-01-10)

### Bug Fixes

- fetch data day-by-day to prevent OOM during fetch phase ([7090ded](https://github.com/terrylica/rangebar-py/commit/7090dedabcc49b744da078c010f6f90643534099)), closes high-volume [#14](https://github.com/terrylica/rangebar-py/issues/14)

## [5.3.5](https://github.com/terrylica/rangebar-py/compare/v5.3.4...v5.3.5) (2026-01-10)

### Bug Fixes

- sort trades by (timestamp, trade_id) to prevent Rust sorting error ([a6d15c6](https://github.com/terrylica/rangebar-py/commit/a6d15c6ac57c42c67b10f241e2455af92dba839e)), closes [#13](https://github.com/terrylica/rangebar-py/issues/13)

## [5.3.4](https://github.com/terrylica/rangebar-py/compare/v5.3.3...v5.3.4) (2026-01-10)

### Bug Fixes

- correct range bar continuity validation semantics ([67e5e32](https://github.com/terrylica/rangebar-py/commit/67e5e325b8f3317eae3d9d09b6d20c0ab10e49a9)), closes [#12](https://github.com/terrylica/rangebar-py/issues/12)

## [5.3.3](https://github.com/terrylica/rangebar-py/compare/v5.3.2...v5.3.3) (2026-01-10)

### Bug Fixes

- streaming read for large months to prevent OOM (Issue [#12](https://github.com/terrylica/rangebar-py/issues/12)) ([f3584e7](https://github.com/terrylica/rangebar-py/commit/f3584e713e152a7847144644eedd37072c9cef49)), closes high-volume

## [5.3.2](https://github.com/terrylica/rangebar-py/compare/v5.3.1...v5.3.2) (2026-01-10)

### Bug Fixes

- reduce memory usage in precompute_range_bars() to prevent OOM ([7adf579](https://github.com/terrylica/rangebar-py/commit/7adf5791186aa874962e67dc70574b3582326c4e)), closes high-volume [#11](https://github.com/terrylica/rangebar-py/issues/11)

## [5.3.1](https://github.com/terrylica/rangebar-py/compare/v5.3.0...v5.3.1) (2026-01-10)

### Bug Fixes

- deduplicate trades before processing to handle Binance duplicates ([fde2a19](https://github.com/terrylica/rangebar-py/commit/fde2a19d80c4c596d17f0eda026d0533c786e368)), closes [#10](https://github.com/terrylica/rangebar-py/issues/10)

# [5.3.0](https://github.com/terrylica/rangebar-py/compare/v5.2.1...v5.3.0) (2026-01-10)

### Features

- implement Issues [#7](https://github.com/terrylica/rangebar-py/issues/7) and [#8](https://github.com/terrylica/rangebar-py/issues/8) - API consistency and WFO workflow ([c5ae438](https://github.com/terrylica/rangebar-py/commit/c5ae438c9d0ac2fc996f03d9316c0a701a89f572))

# [5.2.0](https://github.com/terrylica/rangebar-py/compare/v5.1.1...v5.2.0) (2026-01-10)

### Features

- **continuity:** add Checkpoint API and validate_continuity() for cross-file range bar continuity ([07fc472](https://github.com/terrylica/rangebar-py/commit/07fc47252f1e04dcd76011790ead458cc90ff8c9)), closes [#3](https://github.com/terrylica/rangebar-py/issues/3)

## [5.1.1](https://github.com/terrylica/rangebar-py/compare/v5.1.0...v5.1.1) (2026-01-09)

### Bug Fixes

- **build:** sync Cargo.toml version in semantic-release config ([e0b7a94](https://github.com/terrylica/rangebar-py/commit/e0b7a9478a33f3c16017d9477cfa223f871fd365)), closes [#2](https://github.com/terrylica/rangebar-py/issues/2)

# [5.1.0](https://github.com/terrylica/rangebar-py/compare/v5.0.0...v5.1.0) (2026-01-09)

### Features

- **api:** add get_n_range_bars() for count-bounded bar retrieval ([0e31d1f](https://github.com/terrylica/rangebar-py/commit/0e31d1f5fea08014084074fb982dfb45d2fe4b02))

# [5.0.0](https://github.com/terrylica/rangebar-py/compare/v4.0.1...v5.0.0) (2026-01-09)

- feat(api)!: rename threshold_bps to threshold_decimal_bps ([2ef2d96](https://github.com/terrylica/rangebar-py/commit/2ef2d960f80617466c05218415ca9e36d1b4f001))

### BREAKING CHANGES

- Aligns with upstream rangebar-core v6.0.0 API changes.

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

- **build:** enable data-providers by default for PyPI wheels ([ae40068](https://github.com/terrylica/rangebar-py/commit/ae40068a8932bb7d590fc012f4e32afe9c6f50b4))

## [4.0.0](https://github.com/terrylica/rangebar-py/compare/v3.0.0...v4.0.0) (2026-01-08)

### Breaking Changes

- **api:** Simplified `__all__` exports to public API only
  - `get_range_bars()` is now the recommended primary entry point
  - Legacy APIs (`process_trades_to_dataframe`, `RangeBarProcessor`) still work but not exported in `__all__`

### Documentation

- Aligned README, examples, and API documentation with new primary API

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
