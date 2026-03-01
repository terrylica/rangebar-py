# [12.37.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.36.0...v12.37.0) (2026-03-01)


### Features

* implement per-feature computation toggles (Issue [#128](https://github.com-terrylica/terrylica/rangebar-py/issues/128) Phase 1-4) ([73943cb](https://github.com-terrylica/terrylica/rangebar-py/commit/73943cb240a1a51f0f60bcc6b1b55c1c1c29a24a))

# [12.36.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.35.2...v12.36.0) (2026-03-01)


### Features

* add LaguerreFeatureProvider plugin integration tests + optimize test suite ([094a239](https://github.com-terrylica/terrylica/rangebar-py/commit/094a239ef8c5bdd66d50fe6d3fef5a13f65f2a40))

## [12.35.2](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.35.1...v12.35.2) (2026-02-28)


### Bug Fixes

* **deploy:** add --refresh-package to bypass uv index cache + wheel fallback ([27d647e](https://github.com-terrylica/terrylica/rangebar-py/commit/27d647ed7b5709138f02e3332fa30b24c6bddbd9))

## [12.35.1](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.35.0...v12.35.1) (2026-02-28)


### Bug Fixes

* **#126:** rename ouroboros → ouroboros_mode in populate_full_cache.py ([c5f84c5](https://github.com-terrylica/terrylica/rangebar-py/commit/c5f84c58f359e73cb1e7d38a3f0dfe6d32046142)), closes [#126](https://github.com-terrylica/terrylica/rangebar-py/issues/126)
* **#126:** respect RANGEBAR_OUROBOROS_GUARD for mode mismatch, not just connection failures ([dbf7078](https://github.com-terrylica/terrylica/rangebar-py/commit/dbf7078cac81001968dcda390bad583e63a9f6a5)), closes [#126](https://github.com-terrylica/terrylica/rangebar-py/issues/126)
* **deploy:** use mise-managed Python 3.13 for bigblack venv creation ([14b8744](https://github.com-terrylica/terrylica/rangebar-py/commit/14b87440251000db33df1c0b411d2ccac9c6fed6))

# [12.35.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.34.0...v12.35.0) (2026-02-28)


### Features

* **#126:** Monthly ouroboros migration - Phase 1-4 implementation ([3abbc20](https://github.com-terrylica/terrylica/rangebar-py/commit/3abbc205fd1a8e62fef58ad3d8c81e962f342c7d)), closes [#126](https://github.com-terrylica/terrylica/rangebar-py/issues/126)

# [12.34.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.33.0...v12.34.0) (2026-02-28)


### Bug Fixes

* resolve LOG_DIR correctly when installed as package ([c5c99b5](https://github.com-terrylica/terrylica/rangebar-py/commit/c5c99b5dddfb80a1c3147e584c17e0df618640c7))


### Features

* add per-service NDJSON telemetry logging with auto-rotation ([8eff5ab](https://github.com-terrylica/terrylica/rangebar-py/commit/8eff5ab4c4636799f1ba35defa839af9ecb40d61))

# [12.33.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.32.2...v12.33.0) (2026-02-28)


### Bug Fixes

* **#121:** Harden migration script with 3 production fixes from bigblack deploy ([925fc62](https://github.com-terrylica/terrylica/rangebar-py/commit/925fc62d3166544f5c377fa46cb5c90ab7bf0993))
* **#121:** Idempotent PyPI uploads + local wheel fallback for deploy ([d5db71b](https://github.com-terrylica/terrylica/rangebar-py/commit/d5db71b1527b28ea8bc0cdb7429fbd76bfa72907)), closes [#121](https://github.com-terrylica/terrylica/rangebar-py/issues/121)
* **#121:** Remove User=tca from systemd user services ([c79ba87](https://github.com-terrylica/terrylica/rangebar-py/commit/c79ba87027ca1b07123db8bf73b7dca36a050d9a)), closes [#121](https://github.com-terrylica/terrylica/rangebar-py/issues/121)
* **#121:** Run sidecar directly without healthdog (not installed on bigblack) ([78a5296](https://github.com-terrylica/terrylica/rangebar-py/commit/78a529619288ce9a9e8eb8fc1c282615f8b51aa5)), closes [#121](https://github.com-terrylica/terrylica/rangebar-py/issues/121)
* harden test suite against side effects (sidecar Telegram, Pushover, slow tests) ([ce90522](https://github.com-terrylica/terrylica/rangebar-py/commit/ce905221f330c2fa4ea85387d7f479fa682e4d80))
* migration script uses uv instead of pip on bigblack ([8eec428](https://github.com-terrylica/terrylica/rangebar-py/commit/8eec4285339031befc6d10bac806ddc5de3ceefc)), closes [#121](https://github.com-terrylica/terrylica/rangebar-py/issues/121) [#122](https://github.com-terrylica/terrylica/rangebar-py/issues/122) [#123](https://github.com-terrylica/terrylica/rangebar-py/issues/123)
* update ClickHouse roundtrip tests for timestamp revamp column names ([4cf2b3a](https://github.com-terrylica/terrylica/rangebar-py/commit/4cf2b3a971e3b499a79783726d1acb5382ad186e))


### Features

* **#122:** Add daily exhaustive gap detection systemd timer for bigblack ([7151317](https://github.com-terrylica/terrylica/rangebar-py/commit/715131783f5b27cc8aed3b6f98965456ed577e3e)), closes [#122](https://github.com-terrylica/terrylica/rangebar-py/issues/122)
* add ClickHouse table migration script (timestamp_ms → close_time_ms) ([a0353d3](https://github.com-terrylica/terrylica/rangebar-py/commit/a0353d3a9df396ded4cb0c23abc252d9e16f4519))
* enhance gap detection with duration anomalies, trade ID continuity, and backfill queue health ([2eebbc5](https://github.com-terrylica/terrylica/rangebar-py/commit/2eebbc533234356dd23b5dd6702d9a45decffe60)), closes [#122](https://github.com-terrylica/terrylica/rangebar-py/issues/122) [#123](https://github.com-terrylica/terrylica/rangebar-py/issues/123)

## [12.32.2](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.32.1...v12.32.2) (2026-02-27)


### Bug Fixes

* **#121:** Universal gap prevention system for bigblack ([848985d](https://github.com-terrylica/terrylica/rangebar-py/commit/848985d1a1921fb28f84f70ca2f6fcf4ba6f618c)), closes [#121](https://github.com-terrylica/terrylica/rangebar-py/issues/121)

## [12.32.1](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.32.0...v12.32.1) (2026-02-27)


### Bug Fixes

* **#120:** Multi-threshold sidecar coverage + deploy hardening ([4e8b486](https://github.com-terrylica/terrylica/rangebar-py/commit/4e8b486313d87e13a1db508a56f0ad138c349e7b)), closes [#120](https://github.com-terrylica/terrylica/rangebar-py/issues/120) [#120](https://github.com-terrylica/terrylica/rangebar-py/issues/120) [#120](https://github.com-terrylica/terrylica/rangebar-py/issues/120) [#120](https://github.com-terrylica/terrylica/rangebar-py/issues/120)

# [12.32.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.31.0...v12.32.0) (2026-02-27)


### Features

* **#117-119:** End-to-end sidecar reliability + monitoring ([697d993](https://github.com-terrylica/terrylica/rangebar-py/commit/697d9933e64d4fba9ad3346d70de7f3aa307d9ad)), closes [#117-119](https://github.com-terrylica/terrylica/rangebar-py/issues/117-119) [#117](https://github.com-terrylica/terrylica/rangebar-py/issues/117) [#118](https://github.com-terrylica/terrylica/rangebar-py/issues/118) [#119](https://github.com-terrylica/terrylica/rangebar-py/issues/119) [#117-119](https://github.com-terrylica/terrylica/rangebar-py/issues/117-119) [#117-119](https://github.com-terrylica/terrylica/rangebar-py/issues/117-119) [#117-119](https://github.com-terrylica/terrylica/rangebar-py/issues/117-119)

# [12.31.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.30.0...v12.31.0) (2026-02-26)


### Bug Fixes

* **#113,#114:** add rangebar-hurst to crates.io publish chain + fix cleanup port ([d211421](https://github.com-terrylica/terrylica/rangebar-py/commit/d21142179017b486e55a34bc3b3c68e22aa4f57a))
* health_check.py default port 18123 -> 8123 ([dae744d](https://github.com-terrylica/terrylica/rangebar-py/commit/dae744d188abaddfdf49a362dd29d9df27c44185))
* **release:** wire missing dependencies in release:full pipeline ([9d285b0](https://github.com-terrylica/terrylica/rangebar-py/commit/9d285b00628651fd950e47296e97f0bb78a8bfae))


### Features

* **#113:** replace bespoke topo sort with native cargo publish --workspace ([43822ac](https://github.com-terrylica/terrylica/rangebar-py/commit/43822ac598800359041040585bbf96c0a94183cc)), closes [#113](https://github.com-terrylica/terrylica/rangebar-py/issues/113)
* **#115:** add min_threshold to symbol registry for Kintsugi validation ([5e31beb](https://github.com-terrylica/terrylica/rangebar-py/commit/5e31beb7f3c01697b01366061cc912d6167c18f2)), closes [#115](https://github.com-terrylica/terrylica/rangebar-py/issues/115)
* **#115:** Kintsugi self-healing gap reconciliation ([7cb0bf5](https://github.com-terrylica/terrylica/rangebar-py/commit/7cb0bf58ad54f543b6e8ab1bd68d998efe580963)), closes [#115](https://github.com-terrylica/terrylica/rangebar-py/issues/115)
* **#115:** set min_threshold 500 dbps for DOGE and SOL in symbols.toml ([e5fc9f4](https://github.com-terrylica/terrylica/rangebar-py/commit/e5fc9f4b76879d4cffa254a31059f396da821a99)), closes [#115](https://github.com-terrylica/terrylica/rangebar-py/issues/115)
* **#115:** set min_threshold 500 dbps for XRP and ADA in symbols.toml ([fa1ea71](https://github.com-terrylica/terrylica/rangebar-py/commit/fa1ea718bc418731cf7ac28fbd34e65afa01a000)), closes [#115](https://github.com-terrylica/terrylica/rangebar-py/issues/115) [#89](https://github.com-terrylica/terrylica/rangebar-py/issues/89)
* **#115:** set min_threshold for remaining tier-1 symbols in symbols.toml ([7a45dba](https://github.com-terrylica/terrylica/rangebar-py/commit/7a45dba46dce41441dc218aa68bbe2ce267571d6)), closes [#115](https://github.com-terrylica/terrylica/rangebar-py/issues/115) [#89](https://github.com-terrylica/terrylica/rangebar-py/issues/89)
* **#115:** set min_threshold for tier-2 symbols (WIF, TRX, ZEC, FTM, PEPE, TON) ([845c942](https://github.com-terrylica/terrylica/rangebar-py/commit/845c94202d1dc937fcdb2f5ed61094cfef147fc7)), closes [#115](https://github.com-terrylica/terrylica/rangebar-py/issues/115) [#89](https://github.com-terrylica/terrylica/rangebar-py/issues/89)
* **release:** auto-discover publishable crates from cargo metadata ([d6d9203](https://github.com-terrylica/terrylica/rangebar-py/commit/d6d9203b3f1009a100778be783dd2c04d96486b6)), closes [#114](https://github.com-terrylica/terrylica/rangebar-py/issues/114)

# [12.30.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.29.0...v12.30.0) (2026-02-26)


### Bug Fixes

* **#112:** gap-aware checkpoint recovery and bar range validation ([24c4fc1](https://github.com-terrylica/terrylica/rangebar-py/commit/24c4fc17bef2ab17a41b27dd32eddaea71a7936c)), closes [#112](https://github.com-terrylica/terrylica/rangebar-py/issues/112)
* **#96:** update inter-bar tests for max_safe_capacity and real market data ([a42453c](https://github.com-terrylica/terrylica/rangebar-py/commit/a42453ca1e7a3e349e3da316efffff2373de41ab)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **deploy:bigblack:** bulletproof git sync and venv installation ([e3b2f25](https://github.com-terrylica/terrylica/rangebar-py/commit/e3b2f25b45fc19373c6db2ea08999d2da22894c3))


### Features

* **#104:** backfill progress reporting and status query ([3b7a75b](https://github.com-terrylica/terrylica/rangebar-py/commit/3b7a75bb3c6e16710a52059a96e75776ecac680c)), closes [#104](https://github.com-terrylica/terrylica/rangebar-py/issues/104)
* **#108:** live streaming range bar chart with direct WS + processor ([0fa2a47](https://github.com-terrylica/terrylica/rangebar-py/commit/0fa2a47513c1446e0e2e1580ab4cd619851556e0)), closes [#108](https://github.com-terrylica/terrylica/rangebar-py/issues/108)
* **#108:** wire circuit breaker into fatal_cache_write ([c54f80d](https://github.com-terrylica/terrylica/rangebar-py/commit/c54f80dd1fbe3653aa01d4d720155e305ed0bce0)), closes [#108](https://github.com-terrylica/terrylica/rangebar-py/issues/108)
* **#109:** health check endpoint with CLI and HTTP sidecar ([ab35915](https://github.com-terrylica/terrylica/rangebar-py/commit/ab35915562fc77d80e9ecb27244779361a61bf22)), closes [#109](https://github.com-terrylica/terrylica/rangebar-py/issues/109)
* **#110:** unified Settings singleton for configuration ([9ebe3b2](https://github.com-terrylica/terrylica/rangebar-py/commit/9ebe3b2120e9f3594b8a55c907163f90b57b14e8)), closes [#110](https://github.com-terrylica/terrylica/rangebar-py/issues/110)
* **#111:** Ariadne at every boundary and every touchpoint ([8686bc3](https://github.com-terrylica/terrylica/rangebar-py/commit/8686bc31cd4f0608f2cf093151221bdd1f7117ee)), closes [#111](https://github.com-terrylica/terrylica/rangebar-py/issues/111)
* **#111:** Ariadne trade-ID resume via fromId pagination ([00d9ac8](https://github.com-terrylica/terrylica/rangebar-py/commit/00d9ac8843aa2a499e6840d5a4ee7546a047e424)), closes [#111](https://github.com-terrylica/terrylica/rangebar-py/issues/111)
* **#97:** add health checks, resilience module, and ouroboros tests ([ce7490a](https://github.com-terrylica/terrylica/rangebar-py/commit/ce7490ad79beed44bc6f14af855f6300cc158c80)), closes [#97](https://github.com-terrylica/terrylica/rangebar-py/issues/97)
* **#97:** add ouroboros_mode filter to ClickHouse queries and checkpoints ([0e763e7](https://github.com-terrylica/terrylica/rangebar-py/commit/0e763e79f1b0c1996f901a7292f64caf0fdddaf1)), closes [#97](https://github.com-terrylica/terrylica/rangebar-py/issues/97)

# [12.29.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.28.0...v12.29.0) (2026-02-26)


### Bug Fixes

* **#96:** Fix PE u8 pattern count overflow for large lookback windows ([2932ed6](https://github.com-terrylica/terrylica/rangebar-py/commit/2932ed6292ec0f807c25a4f18aadbbdcd008d30a))
* **#96:** Fix permutation entropy early-exit bug + powi/reciprocal optimizations ([e959c28](https://github.com-terrylica/terrylica/rangebar-py/commit/e959c28fcb430b103aa20b3fb40069bb4502789b))
* **#96:** Resolve clippy errors across rangebar-core and rangebar-hurst ([6a22f50](https://github.com-terrylica/terrylica/rangebar-py/commit/6a22f50ebad792eab48dcb6ead5b9b6f2950958b))
* **deploy:** Recreate venv before pip install to avoid .pth conflicts ([96a8a50](https://github.com-terrylica/terrylica/rangebar-py/commit/96a8a50a644655a7dcebd1427fbabd1c4d71aeba))
* Full pipeline remediation for recurring stale alerts ([39601e1](https://github.com-terrylica/terrylica/rangebar-py/commit/39601e1bcd002ff6976b3f21acf5b716a7d6e2bd))
* Move polars import outside conditional branch in populate_cache_resumable ([57eff5f](https://github.com-terrylica/terrylica/rangebar-py/commit/57eff5f5b169aa6c8c004576ea416997dc11a4fd))
* Update sidecar env to 18 TIER1 symbols and resolve compiler warnings ([06b3e09](https://github.com-terrylica/terrylica/rangebar-py/commit/06b3e096d7e5f54b6d95d2ef73a077911a800f73))


### Features

* **#92:** Add fromId PyO3 bindings (Phases 2+3) ([54db56a](https://github.com-terrylica/terrylica/rangebar-py/commit/54db56a4f766abfaeed671e9fd9212a63b30d007)), closes [#92](https://github.com-terrylica/terrylica/rangebar-py/issues/92)
* **#92:** Add fromId Python wrappers (Phase 4) ([4a9149d](https://github.com-terrylica/terrylica/rangebar-py/commit/4a9149d9acb4eeeb3bd2dab551f3bce03139b29c)), closes [#92](https://github.com-terrylica/terrylica/rangebar-py/issues/92)
* **#92:** Wire fromId + 429 backoff into rangebar-providers ([c5ec13f](https://github.com-terrylica/terrylica/rangebar-py/commit/c5ec13fa74b6b7fbd5a181d11ff9de1c7d991016)), closes [#92](https://github.com-terrylica/terrylica/rangebar-py/issues/92)


### Performance Improvements

* **#96:** Add #[cold] hints to error/uncommon paths in hot loop ([7d42375](https://github.com-terrylica/terrylica/rangebar-py/commit/7d42375e1b565cf24794fc6de9a731d5d22ce942)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add #[inline] + duration reciprocal in microstructure features ([aa04283](https://github.com-terrylica/terrylica/rangebar-py/commit/aa04283f2d643a4414db6b71fef9bdcf2f18f9c1)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add #[inline] to 5 hot-path dispatcher/computation functions ([d7cf452](https://github.com-terrylica/terrylica/rangebar-py/commit/d7cf45233cd105a2bf7d4b3abe96b20e4f6ab906)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add #[inline] to 6 hot-path compute functions ([4911856](https://github.com-terrylica/terrylica/rangebar-py/commit/49118562983e85ea10090bc3f010d3f4e9643291)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add #[inline] to AggTrade hot-path methods ([8b9e021](https://github.com-terrylica/terrylica/rangebar-py/commit/8b9e0211f0ab57821c1dd49cc9dfe28a80e82c9e)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add #[inline] to drawdown functions and reciprocal optimizations ([8481d13](https://github.com-terrylica/terrylica/rangebar-py/commit/8481d13c610f2a5453b93912267a01e419e7a46f)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add #[inline] to hot-path + direct indexing + error tests ([9cd9f4e](https://github.com-terrylica/terrylica/rangebar-py/commit/9cd9f4eaf329196dd82e1f0790ee21f2a30a798b)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add #[inline] to hot-path FixedPoint::to_f64() and is_breach() ([94fc9ef](https://github.com-terrylica/terrylica/rangebar-py/commit/94fc9eff92467bdd6928a1fd8b95018244f323d7)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add #[inline] to per-bar setter and merge functions ([48e2861](https://github.com-terrylica/terrylica/rangebar-py/commit/48e286196a141586ddcb8ac60f02854f3c4065a2)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add #[inline] to RangeBar::new() and update_with_trade() ([3d0aa91](https://github.com-terrylica/terrylica/rangebar-py/commit/3d0aa91f4e1d648beb97236d0bb106c7fe35925f)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add #[inline] to RangeBarState constructors + trade_count reciprocal ([8c04273](https://github.com-terrylica/terrylica/rangebar-py/commit/8c04273bc9fbf5fd6338527d34c851c00e2c10e3)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add agg_count/total_turn/n_points reciprocal caches ([d921796](https://github.com-terrylica/terrylica/rangebar-py/commit/d921796cb145212ae37a5ea8ed42184c9ff652d1)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add all_prices_finite flag to LookbackCache ([484adde](https://github.com-terrylica/terrylica/rangebar-py/commit/484addeaeca41b9bd5e7e7c73cbda674480fefa5)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add all_volumes_finite flag + proptest for finite invariants ([dd711ab](https://github.com-terrylica/terrylica/rangebar-py/commit/dd711ab6339e2122a8ca00b16711712e5ae6f953)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add Copy derive to InterBarFeatures, eliminate .clone() ([bec363e](https://github.com-terrylica/terrylica/rangebar-py/commit/bec363e093687cabca9e509afbda16238170190c)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add n_inv reciprocal to ApEn adaptive entropy paths ([7a47dd8](https://github.com-terrylica/terrylica/rangebar-py/commit/7a47dd8d9b0dafc8e3bb6ad893cf1391193f9350)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Add n_inv reciprocal to volume moment computations ([396c859](https://github.com-terrylica/terrylica/rangebar-py/commit/396c8590d44060de700752cb03e79edd778bfe4f)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Adjust Vec capacity heuristic from /100 to /50 ([55284f7](https://github.com-terrylica/terrylica/rangebar-py/commit/55284f711753c01b2aeb731c01069d5573d4aa14)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Apply reciprocal pattern to M=3 PE entropy computation ([3224473](https://github.com-terrylica/terrylica/rangebar-py/commit/3224473dbf8abefb2a426b572144b080b26feffa)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Branchless buy/sell accumulation in Tier 1 single-pass fold ([6b62fa1](https://github.com-terrylica/terrylica/rangebar-py/commit/6b62fa114e7a876ea33b98f90e5c2a96120d0013)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Cache .len() and .abs() calls in hot-path functions ([7a6cc03](https://github.com-terrylica/terrylica/rangebar-py/commit/7a6cc0348a900c873c7d19492a8f96f56c3437de)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Cache num_cpus::get() with OnceLock in compute_features ([334dbbe](https://github.com-terrylica/terrylica/rangebar-py/commit/334dbbe489227f8d48b1d33efe1fdd8dc4e65407)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#18](https://github.com-terrylica/terrylica/rangebar-py/issues/18)
* **#96:** Cache volume_i128 in RangeBar::new() constructor ([8a60768](https://github.com-terrylica/terrylica/rangebar-py/commit/8a60768df9fc7252577be5688dc41f443394dc7b)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Cache volume_i128 in update_with_trade hot path ([8fa8504](https://github.com-terrylica/terrylica/rangebar-py/commit/8fa85047b5bff2c720f9f741cc30e0a2151cfa40)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Deduplicate InterBarCacheKey computation in compute_features ([440ace2](https://github.com-terrylica/terrylica/rangebar-py/commit/440ace29b839a86a5fcab6d67a7433d3683ad89c)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#15](https://github.com-terrylica/terrylica/rangebar-py/issues/15)
* **#96:** Eliminate double binary search in compute_features hot path ([11ea492](https://github.com-terrylica/terrylica/rangebar-py/commit/11ea49203b284ad30fd63bc3f42d115740480fc6)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Eliminate per-bar Vec allocation in binary search predicted region ([c297dde](https://github.com-terrylica/terrylica/rangebar-py/commit/c297dde19f561ef4b72260ac9447fa93ea1b1c63)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#167](https://github.com-terrylica/terrylica/rangebar-py/issues/167)
* **#96:** Eliminate String allocation in FixedPoint::from_str ([e726bca](https://github.com-terrylica/terrylica/rangebar-py/commit/e726bca3f81b834806e5a702d636962b1ba626a7)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Fuse Hurst DFA two-pass segment loop into single-pass ([e5fd070](https://github.com-terrylica/terrylica/rangebar-py/commit/e5fd070a2faed1fdf4a35d4772d3a8b579f1d020)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Inline accumulate_trade + add 8 update_with_trade invariant tests ([e9cc8dd](https://github.com-terrylica/terrylica/rangebar-py/commit/e9cc8dd1591a5bea265f0da5b7b48701ddb5574e)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Inline intra-bar hot path + interbar_cache edge case tests ([4e56bf1](https://github.com-terrylica/terrylica/rangebar-py/commit/4e56bf1c9c113ffbd76c0f8b977a1c840ac1b839)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Inline sort3 decision tree for PE ordinal patterns ([7fab124](https://github.com-terrylica/terrylica/rangebar-py/commit/7fab124e80e752d1675693db85eea4952dad8d2a)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#13](https://github.com-terrylica/terrylica/rangebar-py/issues/13)
* **#96:** Optimize rssimple from 5-pass to 2-pass with zero allocations ([d8f7b35](https://github.com-terrylica/terrylica/rangebar-py/commit/d8f7b3564c88d16e4da67507ceb496470776b356)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Pass pre-computed mean to volume moments, skip O(n) sum ([25e4183](https://github.com-terrylica/terrylica/rangebar-py/commit/25e4183924e52d64d05312a43d9753b057d0cad5)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Pre-compute DFA xx_sum analytically in Hurst computation ([2313a93](https://github.com-terrylica/terrylica/rangebar-py/commit/2313a932eccbaa9fc039ca260bb7de131baee81b)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Pre-compute PE reciprocal before entropy accumulation loops ([4737644](https://github.com-terrylica/terrylica/rangebar-py/commit/4737644255ae7f754e30df2bda9686ed1f5a5786)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Pre-compute period reciprocals in streaming indicators ([fbe8a47](https://github.com-terrylica/terrylica/rangebar-py/commit/fbe8a47bff080724ecf6f8c5947242c7718e7af8)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Pre-compute total_vol reciprocal and cache .abs() in microstructure ([aa15959](https://github.com-terrylica/terrylica/rangebar-py/commit/aa1595914b8247fc501d8d3eca356509868cbbd0)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Remove Arc wrapper from binary search cache and lookahead buffer ([7e40b00](https://github.com-terrylica/terrylica/rangebar-py/commit/7e40b0086eb23cfd94a1c97d2fe7863a2ff9b9c5)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Replace .collect() with manual loop in cache-hit lookback path ([c881ca2](https://github.com-terrylica/terrylica/rangebar-py/commit/c881ca22bda1068e4aa2680f7b120129f5fa194d)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#185](https://github.com-terrylica/terrylica/rangebar-py/issues/185)
* **#96:** Replace ahash with foldhash for 20-40% faster numeric hashing ([71745e2](https://github.com-terrylica/terrylica/rangebar-py/commit/71745e2194faaaf64a45207c2effc0acc7641009)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Replace ApEn C(n,2) division with reciprocal multiplication ([5c78af4](https://github.com-terrylica/terrylica/rangebar-py/commit/5c78af4fa5bde2bb4e025c8667c1bf04e26e679b)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Replace binary_search_by with partition_point for lookback ([4770d75](https://github.com-terrylica/terrylica/rangebar-py/commit/4770d7547823f37c2c7ddde93dfd00712b5fadff)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Replace DefaultHasher with AHasher in interbar_cache ([41dcd9d](https://github.com-terrylica/terrylica/rangebar-py/commit/41dcd9d90627b4272fbf56f2c0a030e5ca1594c8)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Replace export_processor f64 turnover with integer path ([20a22a5](https://github.com-terrylica/terrylica/rangebar-py/commit/20a22a56d7ccb38349ad2abaad01347b8dc189e6)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Replace heap Vec with SmallVec<[f64; 64]> in SIMD burstiness ([c75ffc3](https://github.com-terrylica/terrylica/rangebar-py/commit/c75ffc310e0244e1228857e460ed70fc89a48552)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#17](https://github.com-terrylica/terrylica/rangebar-py/issues/17)
* **#96:** Replace inter-arrivals Vec with SmallVec<[f64; 256]> ([833ba1a](https://github.com-terrylica/terrylica/rangebar-py/commit/833ba1aaf28b201b1e91abe14012db6178cae56a)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Replace moka with quick_cache for all LRU caches ([d611351](https://github.com-terrylica/terrylica/rangebar-py/commit/d61135133d041a15311d079268ee0b37d3e54e0c)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Replace normalize_cv() exp() with precomputed LUT ([ecb66fc](https://github.com-terrylica/terrylica/rangebar-py/commit/ecb66fce694b5e181a12a79e0e3db4151e1440c3)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Replace normalize_cv() exp() with precomputed LUT ([1300c0c](https://github.com-terrylica/terrylica/rangebar-py/commit/1300c0c37b12e3bba9314cb314ee0fdc55d27963)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#197](https://github.com-terrylica/terrylica/rangebar-py/issues/197) [#10](https://github.com-terrylica/terrylica/rangebar-py/issues/10)
* **#96:** Replace SmallVec with VecDeque for lookahead buffer ([2538bda](https://github.com-terrylica/terrylica/rangebar-py/commit/2538bda938be2be762fbed74a33de60a7910b126)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Replace Vec with SmallVec for DFA log vectors ([7d067db](https://github.com-terrylica/terrylica/rangebar-py/commit/7d067db9a829a55e7692a4c932e850d14fa8fe51)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Replace Vec with SmallVec for scratch price/volume buffers ([d6569c3](https://github.com-terrylica/terrylica/rangebar-py/commit/d6569c3b13a0eae09d5f577e060e70e320f5ffc9)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Share inv_n reciprocal between SIMD mean and variance ([b0918b9](https://github.com-terrylica/terrylica/rangebar-py/commit/b0918b95abd2ca8783d1980ef8af3fe09e8bba40)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Single-pass epoch interval CV computation in ITH ([2f96b7b](https://github.com-terrylica/terrylica/rangebar-py/commit/2f96b7b8ececa6f9d52c7fe006e18e6017716c0f)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#16](https://github.com-terrylica/terrylica/rangebar-py/issues/16)
* **#96:** Upgrade wide crate 0.7 → 1.1 for improved SIMD codegen ([0776ab1](https://github.com-terrylica/terrylica/rangebar-py/commit/0776ab1df4e1017c247e3f1d345546c1a8bf5083)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Use branchless f64::max/min for OHLC tracking in cache extraction ([ede519c](https://github.com-terrylica/terrylica/rangebar-py/commit/ede519cc23e308af09181630279ab8c1ef1b28fa)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Wire up Hurst LUT + pre-compute PE max_entropy constant ([3060542](https://github.com-terrylica/terrylica/rangebar-py/commit/306054275c241c3e806158aa735cbd6694a25cab)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#8](https://github.com-terrylica/terrylica/rangebar-py/issues/8) [#9](https://github.com-terrylica/terrylica/rangebar-py/issues/9)
* **build:** upgrade release profile to fat LTO for native builds ([8c879e9](https://github.com-terrylica/terrylica/rangebar-py/commit/8c879e9a524598ee0630a9cd7188153bdd75e7e8))

# [12.28.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.27.0...v12.28.0) (2026-02-24)


### Bug Fixes

* **#107:** Add watchdog error recovery and TCP keepalive detection ([0172702](https://github.com-terrylica/terrylica/rangebar-py/commit/01727021e7519594bfb76987427a059a61b7e0de)), closes [#107](https://github.com-terrylica/terrylica/rangebar-py/issues/107) [#107](https://github.com-terrylica/terrylica/rangebar-py/issues/107)
* **#85 Phase 4:** Fix test assertion for RangeBar (no symbol field) ([9970b5f](https://github.com-terrylica/terrylica/rangebar-py/commit/9970b5ffbdf3ba1d5dcc0dbf811c39598c164dba))
* **#96 Task #122:** Complete performance benchmark suite with optimization validation ([af4329f](https://github.com-terrylica/terrylica/rangebar-py/commit/af4329f7e2f9c21362bb129dc85f67e121ba917b)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#122](https://github.com-terrylica/terrylica/rangebar-py/issues/122) [#115](https://github.com-terrylica/terrylica/rangebar-py/issues/115) [#116](https://github.com-terrylica/terrylica/rangebar-py/issues/116) [#118](https://github.com-terrylica/terrylica/rangebar-py/issues/118) [#119](https://github.com-terrylica/terrylica/rangebar-py/issues/119) [#121](https://github.com-terrylica/terrylica/rangebar-py/issues/121) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#122](https://github.com-terrylica/terrylica/rangebar-py/issues/122)
* **#96 Task #150:** Resolve GPL-3.0 license conflict via internal Hurst fork ([140cdb3](https://github.com-terrylica/terrylica/rangebar-py/commit/140cdb3c7d23baa453642f8ef263052ca360256e))
* **#96 Task #193:** Correct early-exit optimization to preserve ITH features ([c769ea8](https://github.com-terrylica/terrylica/rangebar-py/commit/c769ea8eee9a5eeed0003dc466285f4a5c91c9a8)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#193](https://github.com-terrylica/terrylica/rangebar-py/issues/193)
* **deploy:** Use uv pip instead of venv pip for bigblack deployment ([87a52e4](https://github.com-terrylica/terrylica/rangebar-py/commit/87a52e423a0a639b97b1fc27d8517de7f29b52ba))
* refactor checkpoint edge case tests to use dictionary API (Issue [#84](https://github.com-terrylica/terrylica/rangebar-py/issues/84)) ([177bbcc](https://github.com-terrylica/terrylica/rangebar-py/commit/177bbcccc38a1eb27d7430b5ceea034588d0071f))


### Features

* **#103/#105 Phase 2:** Backfill queue fairness - group by symbol only ([2c6fe35](https://github.com-terrylica/terrylica/rangebar-py/commit/2c6fe35c0afe46ea1e3b99147a1d336eb642dbbb)), closes [103/#105](https://github.com-terrylica/terrylica/rangebar-py/issues/105)
* **#107 Phase 1b:** TCP keepalive infrastructure for half-open WebSocket detection ([f937c16](https://github.com-terrylica/terrylica/rangebar-py/commit/f937c16be8921f58c1e974af878dd183ce0cd852)), closes [#107](https://github.com-terrylica/terrylica/rangebar-py/issues/107) [#107](https://github.com-terrylica/terrylica/rangebar-py/issues/107) [#107](https://github.com-terrylica/terrylica/rangebar-py/issues/107)
* **#108:** Optimize permutation entropy with branchless pattern index + 8x unroll ([647e92f](https://github.com-terrylica/terrylica/rangebar-py/commit/647e92fbefe6f4eab3ed226bb8b9f22015adf617)), closes [#108](https://github.com-terrylica/terrylica/rangebar-py/issues/108) [#108](https://github.com-terrylica/terrylica/rangebar-py/issues/108) [#107](https://github.com-terrylica/terrylica/rangebar-py/issues/107)
* **#110:** Eliminate redundant lookback cache clone in Tier 3 features ([47321b7](https://github.com-terrylica/terrylica/rangebar-py/commit/47321b741acb6192c03546064a636eea434077b1)), closes [#110](https://github.com-terrylica/terrylica/rangebar-py/issues/110) [#110](https://github.com-terrylica/terrylica/rangebar-py/issues/110)
* **#111:** Optimize trade history pruning with adaptive batch sizes ([cdaf9e5](https://github.com-terrylica/terrylica/rangebar-py/commit/cdaf9e507bbf803787c61a1f768155809d41e336)), closes [#111](https://github.com-terrylica/terrylica/rangebar-py/issues/111) [#111](https://github.com-terrylica/terrylica/rangebar-py/issues/111)
* **#112:** Optimize Arrow columnar value extraction with iterator-based batch processing ([27ab976](https://github.com-terrylica/terrylica/rangebar-py/commit/27ab976812f380d4260c5297211b7db1932d6330)), closes [#112](https://github.com-terrylica/terrylica/rangebar-py/issues/112) [#112](https://github.com-terrylica/terrylica/rangebar-py/issues/112)
* **#115:** Parallelize Tier 2/3 inter-bar feature computation with rayon ([58e6a96](https://github.com-terrylica/terrylica/rangebar-py/commit/58e6a96078193db69b693511b4a1f301350993b6)), closes [#115](https://github.com-terrylica/terrylica/rangebar-py/issues/115) [#115](https://github.com-terrylica/terrylica/rangebar-py/issues/115)
* **#116:** Optimize permutation entropy and ApEn with libm::log ([c1de346](https://github.com-terrylica/terrylica/rangebar-py/commit/c1de3465716b190811ff7477a46c13eec3a08116)), closes [#116](https://github.com-terrylica/terrylica/rangebar-py/issues/116) [#14](https://github.com-terrylica/terrylica/rangebar-py/issues/14) [#116](https://github.com-terrylica/terrylica/rangebar-py/issues/116)
* **#118:** Optimize VecDeque capacity sizing for trade history ([6e0b383](https://github.com-terrylica/terrylica/rangebar-py/commit/6e0b3836acb42e913ed07a11da99084ab576e240)), closes [#118](https://github.com-terrylica/terrylica/rangebar-py/issues/118) [#118](https://github.com-terrylica/terrylica/rangebar-py/issues/118)
* **#119:** Optimize trade accumulation with SmallVec inline buffer ([bd7dd41](https://github.com-terrylica/terrylica/rangebar-py/commit/bd7dd4146631378740355783c5ebe383f0483d3f)), closes [#119](https://github.com-terrylica/terrylica/rangebar-py/issues/119) [hi#frequency](https://github.com-terrylica/hi/issues/frequency) [#41](https://github.com-terrylica/terrylica/rangebar-py/issues/41) [#119](https://github.com-terrylica/terrylica/rangebar-py/issues/119)
* **#145 Phase 1:** Global entropy cache infrastructure for multi-symbol sharing ([ac11144](https://github.com-terrylica/terrylica/rangebar-py/commit/ac111445414922361576b1be1b576b197da7b7e1)), closes [#145](https://github.com-terrylica/terrylica/rangebar-py/issues/145) [#145](https://github.com-terrylica/terrylica/rangebar-py/issues/145)
* **#145 Phase 2:** TradeHistory integration with optional external entropy cache ([7a67147](https://github.com-terrylica/terrylica/rangebar-py/commit/7a671471facf1ee5196a3efe5a0da09021e7106d)), closes [#145](https://github.com-terrylica/terrylica/rangebar-py/issues/145) [#145](https://github.com-terrylica/terrylica/rangebar-py/issues/145)
* **#145 Phase 3:** RangeBarProcessor integration with optional entropy cache ([b1a3c3e](https://github.com-terrylica/terrylica/rangebar-py/commit/b1a3c3e2baf2d9a9483c01e34eeaafdebeecbbac)), closes [#145](https://github.com-terrylica/terrylica/rangebar-py/issues/145) [#145](https://github.com-terrylica/terrylica/rangebar-py/issues/145)
* **#85 Phase 2:** Add checkpoint schema versioning for v1→v2 migration ([a7bf00d](https://github.com-terrylica/terrylica/rangebar-py/commit/a7bf00d0899e44d1e97ca79755250458ed147ace)), closes [#85](https://github.com-terrylica/terrylica/rangebar-py/issues/85) [#85](https://github.com-terrylica/terrylica/rangebar-py/issues/85)
* **#85 Phase 3:** Reorganize dict conversion functions to match tier-based field ordering ([bfeba52](https://github.com-terrylica/terrylica/rangebar-py/commit/bfeba52da7cbfa7e3471916a16be1acf0e32300b)), closes [#85](https://github.com-terrylica/terrylica/rangebar-py/issues/85)
* **#96 Task #101:** Python trade batch coalescing for FFI speedup ([e32e53e](https://github.com-terrylica/terrylica/rangebar-py/commit/e32e53e9e5019c72624fd10f96d866dbaf014eba)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#101](https://github.com-terrylica/terrylica/rangebar-py/issues/101) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #10:** Optimize TradeHistory allocations with SmallVec (1.5-2x speedup) ([bb73366](https://github.com-terrylica/terrylica/rangebar-py/commit/bb73366c198b1de840cfe5dee62350382ed516ab)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#10](https://github.com-terrylica/terrylica/rangebar-py/issues/10) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#10](https://github.com-terrylica/terrylica/rangebar-py/issues/10)
* **#96 Task #117:** Cache permutation entropy results for deterministic price sequences ([3a2151a](https://github.com-terrylica/terrylica/rangebar-py/commit/3a2151abc8e6d453b606c6e86115e4466ee6b268)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#117](https://github.com-terrylica/terrylica/rangebar-py/issues/117) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#117](https://github.com-terrylica/terrylica/rangebar-py/issues/117)
* **#96 Task #125:** Add production-grade LRU caching with moka crate ([4956f1e](https://github.com-terrylica/terrylica/rangebar-py/commit/4956f1e4a81ed2699fd2e730fff6d0f20ca393c5)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#125](https://github.com-terrylica/terrylica/rangebar-py/issues/125) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#125](https://github.com-terrylica/terrylica/rangebar-py/issues/125)
* **#96 Task #126:** Add Tier 2 CPU contribution profiling benchmark (Phase 5A) ([caa1ac0](https://github.com-terrylica/terrylica/rangebar-py/commit/caa1ac08b7eb5988b5236a6a4007d4b436385a3e)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#126](https://github.com-terrylica/terrylica/rangebar-py/issues/126) [#126](https://github.com-terrylica/terrylica/rangebar-py/issues/126)
* **#96 Task #127:** Implement true SIMD burstiness with wide crate (Phase 5B) ([7e6a2df](https://github.com-terrylica/terrylica/rangebar-py/commit/7e6a2dfda3441cd961b165932685ba589e7cb55e)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#127](https://github.com-terrylica/terrylica/rangebar-py/issues/127) [#127](https://github.com-terrylica/terrylica/rangebar-py/issues/127)
* **#96 Task #128:** Add Tier 3 feature profiling and bottleneck analysis ([357575b](https://github.com-terrylica/terrylica/rangebar-py/commit/357575bd37e0c281a69ccc0bcf7f93895e180240)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#128](https://github.com-terrylica/terrylica/rangebar-py/issues/128)
* **#96 Task #129:** Add entropy SIMD baseline profiling benchmark ([a967de6](https://github.com-terrylica/terrylica/rangebar-py/commit/a967de6d9155fd3ff02f5d7bbcdee955b423826a)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#129](https://github.com-terrylica/terrylica/rangebar-py/issues/129) [#129](https://github.com-terrylica/terrylica/rangebar-py/issues/129)
* **#96 Task #129:** Design entropy SIMD vectorization scaffold (profiling + architecture) ([45d1282](https://github.com-terrylica/terrylica/rangebar-py/commit/45d12823cc8d67639ce3db6d744ced3d2f0bef27)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#129](https://github.com-terrylica/terrylica/rangebar-py/issues/129) [#130](https://github.com-terrylica/terrylica/rangebar-py/issues/130) [#129](https://github.com-terrylica/terrylica/rangebar-py/issues/129) [#130](https://github.com-terrylica/terrylica/rangebar-py/issues/130) [#130](https://github.com-terrylica/terrylica/rangebar-py/issues/130)
* **#96 Task #12:** Add comprehensive performance validation benchmarks ([f0b22ab](https://github.com-terrylica/terrylica/rangebar-py/commit/f0b22abf097b6866403ad968d846e172e1e44abc)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#12](https://github.com-terrylica/terrylica/rangebar-py/issues/12) [#9](https://github.com-terrylica/terrylica/rangebar-py/issues/9) [#10](https://github.com-terrylica/terrylica/rangebar-py/issues/10) [#9](https://github.com-terrylica/terrylica/rangebar-py/issues/9) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #12:** Ring buffer metrics integration for LiveBarEngine ([05fbea8](https://github.com-terrylica/terrylica/rangebar-py/commit/05fbea8b2a8624c03dbbd442e91a69c6d4cb761c)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#12](https://github.com-terrylica/terrylica/rangebar-py/issues/12)
* **#96 Task #13 Phase A-B:** Implement Arrow-native cache pipeline ([4f9ce49](https://github.com-terrylica/terrylica/rangebar-py/commit/4f9ce495d9282f7ef93bc09da1cd2ba2dd635c9b)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#13](https://github.com-terrylica/terrylica/rangebar-py/issues/13) [#13](https://github.com-terrylica/terrylica/rangebar-py/issues/13)
* **#96 Task #13 Phase D:** Integrate Arrow optimization into cache pipeline ([45d266f](https://github.com-terrylica/terrylica/rangebar-py/commit/45d266ffb84f1e50eb344f3186fe3577f63d9dc9)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#13](https://github.com-terrylica/terrylica/rangebar-py/issues/13) [#13](https://github.com-terrylica/terrylica/rangebar-py/issues/13) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #135:** Add entropy cache metrics tracking ([0a67062](https://github.com-terrylica/terrylica/rangebar-py/commit/0a67062c0e5799eed6033d63eb0dc9c8fe7cc37d)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#135](https://github.com-terrylica/terrylica/rangebar-py/issues/135)
* **#96 Task #136:** Optimize SmallVec buffer from 512 to 64 slots ([7912125](https://github.com-terrylica/terrylica/rangebar-py/commit/79121257647cfb0b9cdb687f70ed9f884d75cfb7)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#136](https://github.com-terrylica/terrylica/rangebar-py/issues/136) [#119](https://github.com-terrylica/terrylica/rangebar-py/issues/119) [#118](https://github.com-terrylica/terrylica/rangebar-py/issues/118)
* **#96 Task #137:** Add comprehensive cumulative performance validation ([c346b33](https://github.com-terrylica/terrylica/rangebar-py/commit/c346b3307ffbc907a6984dba97d57debf7d64928)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#137](https://github.com-terrylica/terrylica/rangebar-py/issues/137)
* **#96 Task #138:** Add production validation benchmark with real Binance data ([a31dc6b](https://github.com-terrylica/terrylica/rangebar-py/commit/a31dc6b52edb633e2712e7d0be419878cfc9d31b)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#138](https://github.com-terrylica/terrylica/rangebar-py/issues/138) [#139](https://github.com-terrylica/terrylica/rangebar-py/issues/139)
* **#96 Task #140:** Add Tier-by-Tier performance breakdown on real data ([e9d12b3](https://github.com-terrylica/terrylica/rangebar-py/commit/e9d12b3baa95a73bd39264b22793bc3ef438731a)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#140](https://github.com-terrylica/terrylica/rangebar-py/issues/140)
* **#96 Task #143 Phase 1:** Implement Arrow zero-copy path for DataFrames ([800d84c](https://github.com-terrylica/terrylica/rangebar-py/commit/800d84c81c7f0c2bea4b7febb49760b5b983ce37)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#143](https://github.com-terrylica/terrylica/rangebar-py/issues/143) [#88](https://github.com-terrylica/terrylica/rangebar-py/issues/88) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#143](https://github.com-terrylica/terrylica/rangebar-py/issues/143)
* **#96 Task #144 Phase 1:** Create streaming latency profiling benchmark ([c9db94e](https://github.com-terrylica/terrylica/rangebar-py/commit/c9db94eca178d6ed58f147048b8ad90c1accb1de)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#144](https://github.com-terrylica/terrylica/rangebar-py/issues/144)
* **#96 Task #144 Phase 2:** Add feature attribution profiler benchmark ([1e28984](https://github.com-terrylica/terrylica/rangebar-py/commit/1e28984c9bcf0e51f927f8de79bc70da7a339772)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#144](https://github.com-terrylica/terrylica/rangebar-py/issues/144)
* **#96 Task #144 Phase 3:** Implement batch streaming writes for latency optimization ([7f7a66b](https://github.com-terrylica/terrylica/rangebar-py/commit/7f7a66b3d3361a86d019b72d2847e163eea799f1)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#144](https://github.com-terrylica/terrylica/rangebar-py/issues/144)
* **#96 Task #144 Phase 4:** Complete inter-bar feature result caching integration ([9d617ec](https://github.com-terrylica/terrylica/rangebar-py/commit/9d617ecde878cc3fd0ed18b2fb7cda382a58a573)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#144](https://github.com-terrylica/terrylica/rangebar-py/issues/144) [#144](https://github.com-terrylica/terrylica/rangebar-py/issues/144)
* **#96 Task #148 Phase 2:** Implement Kyle Lambda SIMD acceleration ([4db44b5](https://github.com-terrylica/terrylica/rangebar-py/commit/4db44b540fd43c4adadbf1c8e374555af9077161)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#148](https://github.com-terrylica/terrylica/rangebar-py/issues/148) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#148](https://github.com-terrylica/terrylica/rangebar-py/issues/148)
* **#96 Task #155:** Implement adaptive VecDeque pruning batch sizing ([1817cf6](https://github.com-terrylica/terrylica/rangebar-py/commit/1817cf69f0d114bbf9c7cdc04801813509173932)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#155](https://github.com-terrylica/terrylica/rangebar-py/issues/155)
* **#96 Task #156:** Implement entropy cache try-lock fast-path optimization ([f31688c](https://github.com-terrylica/terrylica/rangebar-py/commit/f31688c6d9fe00f4343bdbc3ba00a2fb7591b7e4)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#156](https://github.com-terrylica/terrylica/rangebar-py/issues/156)
* **#96 Task #158:** Allocator tuning analysis and benchmarking ([db3f1de](https://github.com-terrylica/terrylica/rangebar-py/commit/db3f1de360432c555737610509df65d917ed26fa)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#158](https://github.com-terrylica/terrylica/rangebar-py/issues/158)
* **#96 Task #15:** Expand SIMD burstiness test coverage to 13 comprehensive cases ([7c7a417](https://github.com-terrylica/terrylica/rangebar-py/commit/7c7a417662947053956c2c515a8acbc6c9cf121f)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#15](https://github.com-terrylica/terrylica/rangebar-py/issues/15) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#15](https://github.com-terrylica/terrylica/rangebar-py/issues/15)
* **#96 Task #160:** Hurst early-exit via entropy threshold optimization ([f2a9f38](https://github.com-terrylica/terrylica/rangebar-py/commit/f2a9f38a83f3a563a76f77254852f98aeda5940b)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#160](https://github.com-terrylica/terrylica/rangebar-py/issues/160)
* **#96 Task #161 Phase 2:** SIMD vectorization for Approximate Entropy pattern distance checks ([b699c52](https://github.com-terrylica/terrylica/rangebar-py/commit/b699c52fcff8a16da45d2acdf4f0d7c708111c11)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#161](https://github.com-terrylica/terrylica/rangebar-py/issues/161)
* **#96 Task #161 Phase 3:** Adaptive pattern sampling for ApEn algorithm optimization ([62f5060](https://github.com-terrylica/terrylica/rangebar-py/commit/62f5060afffcc2baff870f94bf59e36136f0a3d6)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#161](https://github.com-terrylica/terrylica/rangebar-py/issues/161)
* **#96 Task #161:** ApEn scalar optimization - Phase 1 (1.27x speedup) ([cd744c1](https://github.com-terrylica/terrylica/rangebar-py/commit/cd744c18eab10cf862398a94ba6fffa2cc2f7e9f)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#161](https://github.com-terrylica/terrylica/rangebar-py/issues/161)
* **#96 Task #162:** Single-pass optimization for trade window hashing ([79ee411](https://github.com-terrylica/terrylica/rangebar-py/commit/79ee411764b24de24a0888a3f212111a08067ad6)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#162](https://github.com-terrylica/terrylica/rangebar-py/issues/162)
* **#96 Task #163:** Cache binary search results in get_lookback_trades ([e3d70e6](https://github.com-terrylica/terrylica/rangebar-py/commit/e3d70e68c72aa35df9a36c688fb7363b7a8fc2f1)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#163](https://github.com-terrylica/terrylica/rangebar-py/issues/163)
* **#96 Task #164:** Adaptive Tier computation parallelization for small windows ([ad6e5e8](https://github.com-terrylica/terrylica/rangebar-py/commit/ad6e5e8dee7a0669d345f083592f22411931453d)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#164](https://github.com-terrylica/terrylica/rangebar-py/issues/164)
* **#96 Task #166:** Optimize intra-bar feature computation with dual-pass elimination ([80a5d4d](https://github.com-terrylica/terrylica/rangebar-py/commit/80a5d4dcbdf86f2b69519863060b77ef5dad75cf)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#166](https://github.com-terrylica/terrylica/rangebar-py/issues/166)
* **#96 Task #167 Phase 1:** Establish lookahead buffer infrastructure for binary search optimization ([b441439](https://github.com-terrylica/terrylica/rangebar-py/commit/b441439115f4d92e9ace17396dbb8554b4d600a0)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#167](https://github.com-terrylica/terrylica/rangebar-py/issues/167) [#163](https://github.com-terrylica/terrylica/rangebar-py/issues/163) [#167](https://github.com-terrylica/terrylica/rangebar-py/issues/167)
* **#96 Task #167:** Implement micro-batch lookahead binary search Phase 2 ([acb6064](https://github.com-terrylica/terrylica/rangebar-py/commit/acb60647e905bde31b127e746e8ba76185b155fa)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#167](https://github.com-terrylica/terrylica/rangebar-py/issues/167)
* **#96 Task #177:** Implement epsilon branch prediction optimization ([2598e58](https://github.com-terrylica/terrylica/rangebar-py/commit/2598e584031b8b4e467612cd5a9f886bdbd85ff2)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#177](https://github.com-terrylica/terrylica/rangebar-py/issues/177)
* **#96 Task #178:** Optimize SmallVec empty case for zero-trade lookback windows ([34e56d2](https://github.com-terrylica/terrylica/rangebar-py/commit/34e56d20149bdf9e1e591b3e989705aa1e7c4409)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#178](https://github.com-terrylica/terrylica/rangebar-py/issues/178)
* **#96 Task #181:** Memoize power exponentiation in intra-bar volume moments ([85bb856](https://github.com-terrylica/terrylica/rangebar-py/commit/85bb8562a9af617e0736ef76530c01fb5a890215)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#181](https://github.com-terrylica/terrylica/rangebar-py/issues/181) [#170](https://github.com-terrylica/terrylica/rangebar-py/issues/170)
* **#96 Task #182:** Optimize saturating arithmetic in permutation entropy histograms ([a37fc17](https://github.com-terrylica/terrylica/rangebar-py/commit/a37fc17f0a18cbabd20791cdb4a5f718000181bd)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#182](https://github.com-terrylica/terrylica/rangebar-py/issues/182) [#170](https://github.com-terrylica/terrylica/rangebar-py/issues/170) [#182](https://github.com-terrylica/terrylica/rangebar-py/issues/182)
* **#96 Task #183:** Reduce cache lock contention with try-lock pattern ([0c114b1](https://github.com-terrylica/terrylica/rangebar-py/commit/0c114b185f905b122ccd78e9f982d0f18e8b6002)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#183](https://github.com-terrylica/terrylica/rangebar-py/issues/183) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [hi#contention](https://github.com-terrylica/hi/issues/contention)
* **#96 Task #184:** Optimize Kyle Lambda with branchless volume accumulation ([8b3e5d8](https://github.com-terrylica/terrylica/rangebar-py/commit/8b3e5d8b4fa686c6fff20a2c110d1b53d1495ff8)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#184](https://github.com-terrylica/terrylica/rangebar-py/issues/184) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #185:** Optimize SmallVec collection with manual loop ([9a9b586](https://github.com-terrylica/terrylica/rangebar-py/commit/9a9b586de770518d27d509f1269d419ba640bb3b)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#185](https://github.com-terrylica/terrylica/rangebar-py/issues/185) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #186:** Optimize hash computation with u64 and branchless buy_count ([d09fd7f](https://github.com-terrylica/terrylica/rangebar-py/commit/d09fd7ff353ad322f3ee1c9aa1deaf0b08fc537a)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#186](https://github.com-terrylica/terrylica/rangebar-py/issues/186) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [177/#184](https://github.com-terrylica/terrylica/rangebar-py/issues/184)
* **#96 Task #187:** Eliminate redundant LookbackCache clones in tier computation ([ea1d079](https://github.com-terrylica/terrylica/rangebar-py/commit/ea1d0792b440fc5ad57dc7529c80f4f8b7b76f1e)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#187](https://github.com-terrylica/terrylica/rangebar-py/issues/187)
* **#96 Task #188:** Implement conversion caching in statistical feature computation ([d32cd09](https://github.com-terrylica/terrylica/rangebar-py/commit/d32cd09f8d99470045384cca230be2ad8f4ca12e)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#188](https://github.com-terrylica/terrylica/rangebar-py/issues/188)
* **#96 Task #189:** Implement dynamic parallelization dispatch for inter-bar features ([ed27f25](https://github.com-terrylica/terrylica/rangebar-py/commit/ed27f250abe6a4cc9801eca151af6a1916c0cdad)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#189](https://github.com-terrylica/terrylica/rangebar-py/issues/189)
* **#96 Task #18:** Eliminate unnecessary reset_index() copy in cache writes ([2269cc4](https://github.com-terrylica/terrylica/rangebar-py/commit/2269cc479437445835c9705343f4e48829dc0610)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#18](https://github.com-terrylica/terrylica/rangebar-py/issues/18) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#18](https://github.com-terrylica/terrylica/rangebar-py/issues/18)
* **#96 Task #190:** Optimize TradeSnapshot memory layout for cache efficiency ([289958c](https://github.com-terrylica/terrylica/rangebar-py/commit/289958cc4b936f116055832affc63b23eed13961)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#190](https://github.com-terrylica/terrylica/rangebar-py/issues/190)
* **#96 Task #191:** Implement entropy cache warm-up optimization ([872dedd](https://github.com-terrylica/terrylica/rangebar-py/commit/872dedddfc8b56c0ce7bed478680b0a9cd8007d1)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#191](https://github.com-terrylica/terrylica/rangebar-py/issues/191)
* **#96 Task #192:** Optimize Hurst DFA memoization ([6754df6](https://github.com-terrylica/terrylica/rangebar-py/commit/6754df6cb3271fb89a624996c5535698f4a5802d)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#192](https://github.com-terrylica/terrylica/rangebar-py/issues/192) [#192](https://github.com-terrylica/terrylica/rangebar-py/issues/192)
* **#96 Task #193:** Add benchmark for intra-bar early-exit optimization ([82e8d7e](https://github.com-terrylica/terrylica/rangebar-py/commit/82e8d7ee69f1b277fd2297a6317da17065f6dc56)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#193](https://github.com-terrylica/terrylica/rangebar-py/issues/193)
* **#96 Task #194:** Add branchless ILP optimization for OFI computation ([7344c7a](https://github.com-terrylica/terrylica/rangebar-py/commit/7344c7a834c2d31fbf0454df8f4c2bd7f5c59320)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#194](https://github.com-terrylica/terrylica/rangebar-py/issues/194)
* **#96 Task #197:** Implement normalization LUT for sigmoid and tanh functions ([2735b5f](https://github.com-terrylica/terrylica/rangebar-py/commit/2735b5f564951f3c59654515b86ae92914fb73c5)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#197](https://github.com-terrylica/terrylica/rangebar-py/issues/197) [#197](https://github.com-terrylica/terrylica/rangebar-py/issues/197) [hi#precision](https://github.com-terrylica/hi/issues/precision) [hi#precision](https://github.com-terrylica/hi/issues/precision)
* **#96 Task #198:** Extend normalization LUT to Hurst soft-clamping ([72d11f9](https://github.com-terrylica/terrylica/rangebar-py/commit/72d11f9d215ac455c6a009d488a1c085aa069124)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#198](https://github.com-terrylica/terrylica/rangebar-py/issues/198) [#197](https://github.com-terrylica/terrylica/rangebar-py/issues/197) [#197](https://github.com-terrylica/terrylica/rangebar-py/issues/197) [#197](https://github.com-terrylica/terrylica/rangebar-py/issues/197) [#197](https://github.com-terrylica/terrylica/rangebar-py/issues/197)
* **#96 Task #199:** Eliminate redundant .abs() calls in Kyle Lambda ([08d612f](https://github.com-terrylica/terrylica/rangebar-py/commit/08d612fb1321916ff989fc6c3247dfa6a39634fb)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#199](https://github.com-terrylica/terrylica/rangebar-py/issues/199) [197/#198](https://github.com-terrylica/terrylica/rangebar-py/issues/198)
* **#96 Task #200:** Cache reciprocal in OFI computation ([9317b46](https://github.com-terrylica/terrylica/rangebar-py/commit/9317b46278b4c1a0bc4c9caea747577b92c77d6f)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#200](https://github.com-terrylica/terrylica/rangebar-py/issues/200) [#197-199](https://github.com-terrylica/terrylica/rangebar-py/issues/197-199)
* **#96 Task #204:** Optimize permutation entropy with early-exit for sorted sequences ([8d017db](https://github.com-terrylica/terrylica/rangebar-py/commit/8d017db56dec94aa32c80542bfb15fe99bf6ade8)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#204](https://github.com-terrylica/terrylica/rangebar-py/issues/204)
* **#96 Task #206:** Add early validity checks for Tier 3 feature computation ([0c06c9d](https://github.com-terrylica/terrylica/rangebar-py/commit/0c06c9d7009c238e62003b47ed071bd7b00c872e)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#206](https://github.com-terrylica/terrylica/rangebar-py/issues/206)
* **#96 Task #208:** Optimize Kyle Lambda with early-exit for zero imbalance ([bc3419d](https://github.com-terrylica/terrylica/rangebar-py/commit/bc3419d4a251d5fff895560c5d2cd8af748d3497)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#208](https://github.com-terrylica/terrylica/rangebar-py/issues/208)
* **#96 Task #209:** Optimize Hurst DFA with powi elimination in variance calculation ([b5fd769](https://github.com-terrylica/terrylica/rangebar-py/commit/b5fd769c83391ba5f93634b51b58abb8def5581c)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#209](https://github.com-terrylica/terrylica/rangebar-py/issues/209) [#202](https://github.com-terrylica/terrylica/rangebar-py/issues/202)
* **#96 Task #210:** Memoize .unwrap() calls on first/last lookback elements ([21af153](https://github.com-terrylica/terrylica/rangebar-py/commit/21af1533024f768f66b65795ad72bfc35f2b3e1d)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#210](https://github.com-terrylica/terrylica/rangebar-py/issues/210)
* **#96 Task #211:** Eliminate redundant .any() checks in permutation entropy histogram ([4b8b0e7](https://github.com-terrylica/terrylica/rangebar-py/commit/4b8b0e7c5623ce823ee2aac95f0e3a7947c8e580)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#211](https://github.com-terrylica/terrylica/rangebar-py/issues/211)
* **#96 Task #212:** Cache reciprocal division in entropy normalization ([2fb3c3a](https://github.com-terrylica/terrylica/rangebar-py/commit/2fb3c3ab68776768f705e080980efd87d98cd7f0)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#212](https://github.com-terrylica/terrylica/rangebar-py/issues/212)
* **#96 Task #213:** Branchless epsilon check in burstiness computation ([d6e08f3](https://github.com-terrylica/terrylica/rangebar-py/commit/d6e08f3b8fb166ddf157645b25208b2125f4e449)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#213](https://github.com-terrylica/terrylica/rangebar-py/issues/213) [#203](https://github.com-terrylica/terrylica/rangebar-py/issues/203)
* **#96 Task #214:** Eliminate filter iterator overhead in entropy calculation ([f0f2439](https://github.com-terrylica/terrylica/rangebar-py/commit/f0f24398cbc5b6c4b59064ab9b84bc171b661fbb)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#214](https://github.com-terrylica/terrylica/rangebar-py/issues/214)
* **#96 Task #4:** Implement SIMD acceleration for burstiness computation ([347c59d](https://github.com-terrylica/terrylica/rangebar-py/commit/347c59d5b1084cc391a9cae7f9e2e62ec0c6c233)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#4](https://github.com-terrylica/terrylica/rangebar-py/issues/4)
* **#96 Task #5 Phase 2:** Add integration tests and pattern documentation for async cache writes ([6499299](https://github.com-terrylica/terrylica/rangebar-py/commit/64992994ba5d3e05263d840e95cb913301050132)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#5](https://github.com-terrylica/terrylica/rangebar-py/issues/5)
* **#96 Task #5 Phase 3:** Add ClickHouse connection pooling ([577cca9](https://github.com-terrylica/terrylica/rangebar-py/commit/577cca9e4d38143a0cee3e37ce2cca0dc33728ab)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#5](https://github.com-terrylica/terrylica/rangebar-py/issues/5)
* **#96 Task #5 Phase 4:** Integrate async writes into populate_cache_resumable ([40e1c4d](https://github.com-terrylica/terrylica/rangebar-py/commit/40e1c4d73ea9780bb5525ba92c4d8e056180ddb8)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#5](https://github.com-terrylica/terrylica/rangebar-py/issues/5)
* **#96 Task #5:** Add concurrent ClickHouse cache writes with backpressure (Phase 1) ([51de7b2](https://github.com-terrylica/terrylica/rangebar-py/commit/51de7b2892dd163a354840b3d1f2fdad64b1695e)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#5](https://github.com-terrylica/terrylica/rangebar-py/issues/5)
* **#96 Task #6 Phase 2:** Add RANGEBAR_MAX_PENDING_BARS env var support ([860d351](https://github.com-terrylica/terrylica/rangebar-py/commit/860d3514b0fcd0f83a90112156b247cb30ce3da2)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#6](https://github.com-terrylica/terrylica/rangebar-py/issues/6) [#6](https://github.com-terrylica/terrylica/rangebar-py/issues/6)
* **#96 Task #6 Phase 3:** Add backpressure metrics infrastructure ([0ee109c](https://github.com-terrylica/terrylica/rangebar-py/commit/0ee109c2f01a2c1c96010a7d16a58529056628fe)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#6](https://github.com-terrylica/terrylica/rangebar-py/issues/6) [#6](https://github.com-terrylica/terrylica/rangebar-py/issues/6)
* **#96 Task #7 Phase 2:** Implement Approximate Entropy (ApEn) for Strategy B ([cf5a4b4](https://github.com-terrylica/terrylica/rangebar-py/commit/cf5a4b46fa25533bf34df2d42852217d0498db9d)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#7](https://github.com-terrylica/terrylica/rangebar-py/issues/7)
* **#96 Task #7 Phase 2:** Implement batch OHLC extraction for Tier 3 features ([a389508](https://github.com-terrylica/terrylica/rangebar-py/commit/a389508c3e8448cea84bcdd0372add65861dcedc)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#7](https://github.com-terrylica/terrylica/rangebar-py/issues/7)
* **#96 Task #7 Phase 3:** Integrate adaptive entropy switching (ApEn for large windows) ([f8f5ce9](https://github.com-terrylica/terrylica/rangebar-py/commit/f8f5ce993c667c3cc2b091d0d7042d08878c4e33)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#7](https://github.com-terrylica/terrylica/rangebar-py/issues/7) [#7](https://github.com-terrylica/terrylica/rangebar-py/issues/7) [hi#volume](https://github.com-terrylica/hi/issues/volume) [#7](https://github.com-terrylica/terrylica/rangebar-py/issues/7) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#7](https://github.com-terrylica/terrylica/rangebar-py/issues/7)
* **#96 Task #76:** Extract Arrow helper functions to module level for 1.5-2.5x batch speedup ([4c59882](https://github.com-terrylica/terrylica/rangebar-py/commit/4c598821f16613378f77b598ba52cce9db2c6bbe)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#76](https://github.com-terrylica/terrylica/rangebar-py/issues/76)
* **#96 Task #78:** Eliminate trade clones in streaming fan-out loops (1.2-1.8x speedup) ([811999c](https://github.com-terrylica/terrylica/rangebar-py/commit/811999c656ebc8fc5739519e23e259fe1cecb4a3)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#78](https://github.com-terrylica/terrylica/rangebar-py/issues/78)
* **#96 Task #79:** Normalize symbol once at entry point (partial) ([b6c36c6](https://github.com-terrylica/terrylica/rangebar-py/commit/b6c36c64a89574db9032e070d79593cb3a268834)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#79](https://github.com-terrylica/terrylica/rangebar-py/issues/79) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #79:** Remove redundant symbol.upper() downstream ([d81820e](https://github.com-terrylica/terrylica/rangebar-py/commit/d81820ef43edb2ff3a5f51b23cbf3aeb512b1e26)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#79](https://github.com-terrylica/terrylica/rangebar-py/issues/79) [#79](https://github.com-terrylica/terrylica/rangebar-py/issues/79) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#79](https://github.com-terrylica/terrylica/rangebar-py/issues/79)
* **#96 Task #80:** Pre-compute volume scale constant ([1225a53](https://github.com-terrylica/terrylica/rangebar-py/commit/1225a532bdf67d979a8a545e8bb1efb8bec71dd5)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#80](https://github.com-terrylica/terrylica/rangebar-py/issues/80)
* **#96 Task #82:** Consolidate numeric scale constants in helpers.rs ([8926b23](https://github.com-terrylica/terrylica/rangebar-py/commit/8926b23c114714f58072221bc60f634ad7f4ef3d)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#82](https://github.com-terrylica/terrylica/rangebar-py/issues/82)
* **#96 Task #83:** Consolidate error message strings in binance_bindings.rs ([6e5010e](https://github.com-terrylica/terrylica/rangebar-py/commit/6e5010e4c31f36f1121ef665311b6542e7914671)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#83](https://github.com-terrylica/terrylica/rangebar-py/issues/83)
* **#96 Task #84:** Pre-allocate Vec capacity in process_trades hot paths ([5c730e1](https://github.com-terrylica/terrylica/rangebar-py/commit/5c730e1f161698b2e0757ffea71c80e1d0b5ee6d)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#84](https://github.com-terrylica/terrylica/rangebar-py/issues/84)
* **#96 Task #86:** Expose Arrow export parameter in process_trades() ([9ec9276](https://github.com-terrylica/terrylica/rangebar-py/commit/9ec9276f9188078cd4e48c58ed8c02c592125cfb)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#86](https://github.com-terrylica/terrylica/rangebar-py/issues/86)
* **#96 Task #97:** Automate PGO workflow in mise tasks + GitHub Actions ([d0fa884](https://github.com-terrylica/terrylica/rangebar-py/commit/d0fa884aa5316e71691b2dd024770596ca78b22c)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#97](https://github.com-terrylica/terrylica/rangebar-py/issues/97) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#97](https://github.com-terrylica/terrylica/rangebar-py/issues/97)
* **#96 Task #9:** Implement fixed-size ring buffer for streaming (phase 1) ([9d175c7](https://github.com-terrylica/terrylica/rangebar-py/commit/9d175c7eecca98a1f406fcda272c93f80fcdacbc)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#9](https://github.com-terrylica/terrylica/rangebar-py/issues/9) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #9:** Integrate ring buffer into LiveBarEngine (phase 2) ([2a3657d](https://github.com-terrylica/terrylica/rangebar-py/commit/2a3657d6fed4b9eafdc1b046cb44cf37c850a1e9)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#9](https://github.com-terrylica/terrylica/rangebar-py/issues/9) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #9:** Promote Arrow export to primary format in process_trades() ([b0797e0](https://github.com-terrylica/terrylica/rangebar-py/commit/b0797e0ea99f248a4bad31ef6a53934c26e68644)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#9](https://github.com-terrylica/terrylica/rangebar-py/issues/9) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#9](https://github.com-terrylica/terrylica/rangebar-py/issues/9)
* **#96:** Add profiling analysis benchmark and performance report ([8a044df](https://github.com-terrylica/terrylica/rangebar-py/commit/8a044dfd80863048a696c435230fe7b68a161a3e)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#16](https://github.com-terrylica/terrylica/rangebar-py/issues/16)


### Performance Improvements

* **#103 Task #12:** Optimize permutation entropy for small windows ([a5afaad](https://github.com-terrylica/terrylica/rangebar-py/commit/a5afaad502e2032480c60c54aeff1638ecc883c8)), closes [#103](https://github.com-terrylica/terrylica/rangebar-py/issues/103) [#12](https://github.com-terrylica/terrylica/rangebar-py/issues/12)
* **#104:** Optimize trade history ring buffer pruning strategy ([79a7e14](https://github.com-terrylica/terrylica/rangebar-py/commit/79a7e14b51ca5d41005a187d204ece1e33f4c0fd)), closes [#104](https://github.com-terrylica/terrylica/rangebar-py/issues/104) [hi#trade-rate](https://github.com-terrylica/hi/issues/trade-rate)
* **#105:** Optimize RangeBar dict conversion efficiency ([0bad913](https://github.com-terrylica/terrylica/rangebar-py/commit/0bad9139fb89f687b1fa1014f72edad3a8657e55)), closes [#105](https://github.com-terrylica/terrylica/rangebar-py/issues/105)
* **#106:** Optimize dict-to-AggTrade FFI extraction efficiency ([564c10a](https://github.com-terrylica/terrylica/rangebar-py/commit/564c10a5935b55dc127ce7bfe6f3986149d9c099)), closes [#106](https://github.com-terrylica/terrylica/rangebar-py/issues/106)
* **#147 Phase 9:** Integrate Mimalloc for high-performance memory allocation ([1662185](https://github.com-terrylica/terrylica/rangebar-py/commit/1662185f2a41fc3498a72f847df30c7c8f6387e1)), closes [#147](https://github.com-terrylica/terrylica/rangebar-py/issues/147) [hi#performance](https://github.com-terrylica/hi/issues/performance)
* **#96 Phase 3b:** integrate hurst::rssimple for 75-80% speedup ([9d88a75](https://github.com-terrylica/terrylica/rangebar-py/commit/9d88a75733737514152096ea3f568307f3116cd0)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Phase 4a:** optimize Garman-Klass with pre-computed coefficient ([a376e66](https://github.com-terrylica/terrylica/rangebar-py/commit/a376e66836073e8d7400161a53f8f9efd5a4e298)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #102:** Enable SIMD burstiness vectorization by default ([1c46746](https://github.com-terrylica/terrylica/rangebar-py/commit/1c46746b3b47bb0d44892eadd5dbd0309d36cc08)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#102](https://github.com-terrylica/terrylica/rangebar-py/issues/102) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#102](https://github.com-terrylica/terrylica/rangebar-py/issues/102)
* **#96 Task #124:** Replace RwLock with parking_lot for lower-latency entropy cache ([ab2eb2d](https://github.com-terrylica/terrylica/rangebar-py/commit/ab2eb2dd703d1c23a33e61c095556d127f124732)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#124](https://github.com-terrylica/terrylica/rangebar-py/issues/124) [hi#contention](https://github.com-terrylica/hi/issues/contention) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#124](https://github.com-terrylica/terrylica/rangebar-py/issues/124)
* **#96 Task #148:** Add Kyle Lambda profiling benchmark for SIMD decision ([cc72077](https://github.com-terrylica/terrylica/rangebar-py/commit/cc72077df7c4f4a23be0f0296cf41a416cd2713d)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#148](https://github.com-terrylica/terrylica/rangebar-py/issues/148) [#148](https://github.com-terrylica/terrylica/rangebar-py/issues/148)
* **#96 Task #14:** Optimize Garman-Klass volatility with libm::ln() ([f70620a](https://github.com-terrylica/terrylica/rangebar-py/commit/f70620aec08c3940a9425e58091758bccc585f49)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#14](https://github.com-terrylica/terrylica/rangebar-py/issues/14) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #168:** Eliminate .to_vec() clone in Hurst DFA computation (1-2% speedup) ([f0b0249](https://github.com-terrylica/terrylica/rangebar-py/commit/f0b02496ce9625a071f84a5ca1ba1484ad84ac98)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#168](https://github.com-terrylica/terrylica/rangebar-py/issues/168)
* **#96 Task #168:** Optimize powi(2) to direct multiplication (0.5-1.2% speedup) ([b891b3d](https://github.com-terrylica/terrylica/rangebar-py/commit/b891b3d0626a207ceabe98b4eb73b15d99c2adbe)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#168](https://github.com-terrylica/terrylica/rangebar-py/issues/168)
* **#96 Task #168:** Switch to ahash for price sequence hashing (0.8-1.5% speedup) ([1f35073](https://github.com-terrylica/terrylica/rangebar-py/commit/1f3507346d53770871c649b6985014a2a325434a)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#168](https://github.com-terrylica/terrylica/rangebar-py/issues/168)
* **#96 Task #169:** Vectorize Kaufman ER with SIMD f64x4 ([e7d323c](https://github.com-terrylica/terrylica/rangebar-py/commit/e7d323c75b79b5069dfd5232fc8b56ca60f72ea6)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#169](https://github.com-terrylica/terrylica/rangebar-py/issues/169) [#168](https://github.com-terrylica/terrylica/rangebar-py/issues/168) [#2](https://github.com-terrylica/terrylica/rangebar-py/issues/2)
* **#96 Task #170:** Memoize powi() in volume moments computation ([25dc856](https://github.com-terrylica/terrylica/rangebar-py/commit/25dc856ff03030c0189e9644b8286f228641f175)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#170](https://github.com-terrylica/terrylica/rangebar-py/issues/170)
* **#96 Task #171:** Early-exit zero-volume lookback in cache key computation ([fe70fff](https://github.com-terrylica/terrylica/rangebar-py/commit/fe70fffc438432fded0e678c4145770a387f8340)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#171](https://github.com-terrylica/terrylica/rangebar-py/issues/171)
* **#96 Task #173:** Add reusable scratch buffers for intra-bar feature computation ([7b93d22](https://github.com-terrylica/terrylica/rangebar-py/commit/7b93d22b311598d1d7f426131c5522ab1c657924)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#173](https://github.com-terrylica/terrylica/rangebar-py/issues/173) [hi#frequency](https://github.com-terrylica/hi/issues/frequency)
* **#96 Task #175:** Apply instruction-level parallelism to Kyle Lambda scalar fold ([a8bc733](https://github.com-terrylica/terrylica/rangebar-py/commit/a8bc7337047665d19001af95599e588df7b911aa)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#175](https://github.com-terrylica/terrylica/rangebar-py/issues/175)
* **#96 Task #176:** Optimize EntropyCache price hashing with direct bitcast ([ea805ec](https://github.com-terrylica/terrylica/rangebar-py/commit/ea805ecef76aeda5b3e14478fe7b6267d2b56b00)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#176](https://github.com-terrylica/terrylica/rangebar-py/issues/176)
* **#96 Task #19:** Eliminate redundant .all() call in cache staleness detection ([693877c](https://github.com-terrylica/terrylica/rangebar-py/commit/693877cf7d56a9d0f63df42d90a1ace9cbc3ffc9)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#19](https://github.com-terrylica/terrylica/rangebar-py/issues/19) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #20:** Use set intersection for validation column checking ([58d432c](https://github.com-terrylica/terrylica/rangebar-py/commit/58d432c932d888f40e8d8669daafd8347812d1af)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#20](https://github.com-terrylica/terrylica/rangebar-py/issues/20) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #21:** implement adaptive permutation entropy window strategy ([c46ee53](https://github.com-terrylica/terrylica/rangebar-py/commit/c46ee53b8bf3119ecbb33364f0aaec74b327af58)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#21](https://github.com-terrylica/terrylica/rangebar-py/issues/21)
* **#96 Task #21:** Reuse _hit variable to eliminate double condition check ([02f8b3f](https://github.com-terrylica/terrylica/rangebar-py/commit/02f8b3f044e47f5ad8c77e535f2fcb35996c196b)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#21](https://github.com-terrylica/terrylica/rangebar-py/issues/21) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #22:** Cache .is_empty() result in streaming functions ([0c9a44f](https://github.com-terrylica/terrylica/rangebar-py/commit/0c9a44f6fee526e6a252e847c867d7ae101a0760)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#22](https://github.com-terrylica/terrylica/rangebar-py/issues/22) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #23:** Skip normalization when no temporal columns present ([a08f28f](https://github.com-terrylica/terrylica/rangebar-py/commit/a08f28f7a7ec5eb2945ea625ca574576330bfca4)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#23](https://github.com-terrylica/terrylica/rangebar-py/issues/23) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96 Task #34:** Add fast-path skip for column rename in bulk_operations ([3bb3e4e](https://github.com-terrylica/terrylica/rangebar-py/commit/3bb3e4e392c028139a222e7ec84ca756eee2a171)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#34](https://github.com-terrylica/terrylica/rangebar-py/issues/34)
* **#96 Task #35:** Optimize symbol set membership with frozenset caching ([eb42ee1](https://github.com-terrylica/terrylica/rangebar-py/commit/eb42ee13f4586fb5e630e4862d59aa972a40f842)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#35](https://github.com-terrylica/terrylica/rangebar-py/issues/35)
* **#96 Task #36:** Consolidate exchange session map/apply chains (15-25%) ([6202c2e](https://github.com-terrylica/terrylica/rangebar-py/commit/6202c2e274149ee44cc99d59954f15f001e749d7)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#36](https://github.com-terrylica/terrylica/rangebar-py/issues/36)
* **#96 Task #37:** Cache available column set for O(1) membership testing ([89ef76c](https://github.com-terrylica/terrylica/rangebar-py/commit/89ef76c00710ca5d11d6fd66f9f4c0694c1d4683)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#37](https://github.com-terrylica/terrylica/rangebar-py/issues/37)
* **#96 Task #38:** Eliminate redundant reset_index in query result reversal ([419b840](https://github.com-terrylica/terrylica/rangebar-py/commit/419b840e5420109242193ac703b423c83ec195bf)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#38](https://github.com-terrylica/terrylica/rangebar-py/issues/38)
* **#96 Task #39:** Add early-exit for tiny Parquet files in validation ([1e11068](https://github.com-terrylica/terrylica/rangebar-py/commit/1e1106877f9122d165151004f12f5fb6d169fe38)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#39](https://github.com-terrylica/terrylica/rangebar-py/issues/39)
* **#96 Task #40:** Vectorize dtype conversion in normalize_arrow_dtypes ([c66945f](https://github.com-terrylica/terrylica/rangebar-py/commit/c66945f8eac7d842a53cabb377120399c88802bf)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#40](https://github.com-terrylica/terrylica/rangebar-py/issues/40)
* **#96 Task #41:** optimize lookback filter with binary search ([4f2af87](https://github.com-terrylica/terrylica/rangebar-py/commit/4f2af87e0dbb6654c7ad3a9f0adaa338a0730d70)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#41](https://github.com-terrylica/terrylica/rangebar-py/issues/41)
* **#96 Task #42:** consolidate volume moment calculations ([3a784cb](https://github.com-terrylica/terrylica/rangebar-py/commit/3a784cb6109e256ca3b02c1b1f4ba6cd1577e9e8)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#42](https://github.com-terrylica/terrylica/rangebar-py/issues/42)
* **#96 Task #44:** reduce trade cloning overhead in intra-bar computation ([bd6dc0c](https://github.com-terrylica/terrylica/rangebar-py/commit/bd6dc0c751d6633bc6290d70252421418c5d7fff)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#44](https://github.com-terrylica/terrylica/rangebar-py/issues/44)
* **#96 Task #45:** reduce RangeBar clone size in checkpoint path ([ce05a99](https://github.com-terrylica/terrylica/rangebar-py/commit/ce05a9902f18344f1717975769aa59b976f46418)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#45](https://github.com-terrylica/terrylica/rangebar-py/issues/45) [#45](https://github.com-terrylica/terrylica/rangebar-py/issues/45)
* **#96 Task #46:** merge Tier 1 inter-bar feature fold passes ([e13d1a0](https://github.com-terrylica/terrylica/rangebar-py/commit/e13d1a0a2d17bc4f86ba6887742c6ebc05c6fb57)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#46](https://github.com-terrylica/terrylica/rangebar-py/issues/46) [#46](https://github.com-terrylica/terrylica/rangebar-py/issues/46)
* **#96 Task #47:** replace .windows(2) iterators with direct indexing ([73a98db](https://github.com-terrylica/terrylica/rangebar-py/commit/73a98dbcb2dc017ac7e48937e0da36bb7539c87f)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#47](https://github.com-terrylica/terrylica/rangebar-py/issues/47)
* **#96 Task #48:** use SmallVec for inter-arrival times in burstiness ([ec8c137](https://github.com-terrylica/terrylica/rangebar-py/commit/ec8c1375c9bb7c223100c1f46788ca9de2a3d972)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#48](https://github.com-terrylica/terrylica/rangebar-py/issues/48)
* **#96 Task #49:** implement batch caching for permutation entropy computation ([7073519](https://github.com-terrylica/terrylica/rangebar-py/commit/70735190d6eb96e7861fe7b1c0189f0b9575b232)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#49](https://github.com-terrylica/terrylica/rangebar-py/issues/49) [#49](https://github.com-terrylica/terrylica/rangebar-py/issues/49)
* **#96 Task #50:** fuse mean/variance computation in burstiness using Welford's algorithm ([6acb451](https://github.com-terrylica/terrylica/rangebar-py/commit/6acb45114c0bd8e723da2b34c2867b0b7134294c)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#50](https://github.com-terrylica/terrylica/rangebar-py/issues/50)
* **#96 Task #51:** single-pass high/low computation in VWAP position ([114ada4](https://github.com-terrylica/terrylica/rangebar-py/commit/114ada4c0b3b5906b45c416ccfa6b5051ee6d1bb)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#51](https://github.com-terrylica/terrylica/rangebar-py/issues/51) [#51](https://github.com-terrylica/terrylica/rangebar-py/issues/51)
* **#96 Task #52:** add adaptive Kyle Lambda computation for small/large windows ([241b233](https://github.com-terrylica/terrylica/rangebar-py/commit/241b23375520be5b80eda94a8e2a2970aa8d7753)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#52](https://github.com-terrylica/terrylica/rangebar-py/issues/52) [#52](https://github.com-terrylica/terrylica/rangebar-py/issues/52)
* **#96 Task #53:** replace HashMap pattern encoding with bounded array in intra-bar entropy ([0287b81](https://github.com-terrylica/terrylica/rangebar-py/commit/0287b81b48b61e1d335f42cf9c7d51ca78115bb1)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#53](https://github.com-terrylica/terrylica/rangebar-py/issues/53) [#53](https://github.com-terrylica/terrylica/rangebar-py/issues/53)
* **#96 Task #54:** hoist SmallVec allocation and add early-exit in entropy computation ([237baa9](https://github.com-terrylica/terrylica/rangebar-py/commit/237baa9de5ea7c2f69a96ed67261326f74170b8d)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#54](https://github.com-terrylica/terrylica/rangebar-py/issues/54) [#53](https://github.com-terrylica/terrylica/rangebar-py/issues/53) [#54](https://github.com-terrylica/terrylica/rangebar-py/issues/54)
* **#96 Task #55:** consolidate volume skewness and kurtosis in single pass ([4f59300](https://github.com-terrylica/terrylica/rangebar-py/commit/4f593009eae22708caabcc708aa712b0e5773ec9)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#55](https://github.com-terrylica/terrylica/rangebar-py/issues/55) [#55](https://github.com-terrylica/terrylica/rangebar-py/issues/55)
* **#96 Task #56:** extract volume moments computation into reusable helper ([43003d9](https://github.com-terrylica/terrylica/rangebar-py/commit/43003d9ec61380f3e1d5d462a6bf7a427155b643)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#56](https://github.com-terrylica/terrylica/rangebar-py/issues/56) [#55](https://github.com-terrylica/terrylica/rangebar-py/issues/55) [#56](https://github.com-terrylica/terrylica/rangebar-py/issues/56)
* **#96 Task #57:** optimize Hurst DFA allocations with SmallVec and pre-sizing ([f74e4dd](https://github.com-terrylica/terrylica/rangebar-py/commit/f74e4ddcd7f48e3b5c67cb16cefc76050932b7b3)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#57](https://github.com-terrylica/terrylica/rangebar-py/issues/57) [#57](https://github.com-terrylica/terrylica/rangebar-py/issues/57)
* **#96 Task #58:** specialize and unroll ordinal pattern index computation ([e1c4c19](https://github.com-terrylica/terrylica/rangebar-py/commit/e1c4c194d8eb6f8ae1dcabbed1330ce9b2eb711c)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#58](https://github.com-terrylica/terrylica/rangebar-py/issues/58) [#58](https://github.com-terrylica/terrylica/rangebar-py/issues/58)
* **#96 Task #59:** replace .windows(2) iterator with direct indexing in Kaufman ER ([d40b4d8](https://github.com-terrylica/terrylica/rangebar-py/commit/d40b4d895b49083ab448b765636acfe628e220c6)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#59](https://github.com-terrylica/terrylica/rangebar-py/issues/59) [#59](https://github.com-terrylica/terrylica/rangebar-py/issues/59)
* **#96 Task #61:** optimize burstiness with SmallVec and early-exit conditions ([91c1040](https://github.com-terrylica/terrylica/rangebar-py/commit/91c10406687db451daae2e65532463297034aa6c)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#61](https://github.com-terrylica/terrylica/rangebar-py/issues/61) [#59](https://github.com-terrylica/terrylica/rangebar-py/issues/59) [#61](https://github.com-terrylica/terrylica/rangebar-py/issues/61)
* **#96 Task #62:** add early-exit to trade ordering validation ([1261f00](https://github.com-terrylica/terrylica/rangebar-py/commit/1261f00f38616d4445b434ab3922a5f6fe8a5061)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#62](https://github.com-terrylica/terrylica/rangebar-py/issues/62)
* **#96 Task #63:** cache high/low extremes during VWAP computation with single fold ([9dd62a2](https://github.com-terrylica/terrylica/rangebar-py/commit/9dd62a27a314f6128d1141178b475620c48696b1)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#63](https://github.com-terrylica/terrylica/rangebar-py/issues/63)
* **#96 Task #64:** eliminate redundant price and volume vector allocations with pre-allocation ([596e037](https://github.com-terrylica/terrylica/rangebar-py/commit/596e037397d3b983bbc07e27aed15473fccbb844)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#64](https://github.com-terrylica/terrylica/rangebar-py/issues/64)
* **#96 Task #65:** add coarse bounds check to Kyle Lambda for extreme imbalance early-exit ([d0a67f6](https://github.com-terrylica/terrylica/rangebar-py/commit/d0a67f6194489777969c86b4e6b8a39d3e9af816)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#65](https://github.com-terrylica/terrylica/rangebar-py/issues/65)
* **#96 Task #66:** merge max drawdown and runup computation into single pass ([db66968](https://github.com-terrylica/terrylica/rangebar-py/commit/db669682142add658068223c8f955b85dac4ef65)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#66](https://github.com-terrylica/terrylica/rangebar-py/issues/66)
* **#96 Task #67:** add early-exit for monotonic sequences in permutation entropy ([2dde384](https://github.com-terrylica/terrylica/rangebar-py/commit/2dde384f20405668cd954887b1bd337595bff1e0)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#67](https://github.com-terrylica/terrylica/rangebar-py/issues/67)
* **#96 Task #68:** eliminate SmallVec allocation overhead on tiny lookbacks (0-2 trades) ([228a218](https://github.com-terrylica/terrylica/rangebar-py/commit/228a218542733b776756a370703bbc5944e58aa1)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#68](https://github.com-terrylica/terrylica/rangebar-py/issues/68)
* **#96 Task #69:** fuse volume moments computation into main trades loop ([d9067ab](https://github.com-terrylica/terrylica/rangebar-py/commit/d9067abae7b75f8e266df0af3689ff8eaf13f849)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#69](https://github.com-terrylica/terrylica/rangebar-py/issues/69)
* **#96 Task #70:** batch PyO3 optional field setting by skipping None values ([ef25743](https://github.com-terrylica/terrylica/rangebar-py/commit/ef257432e2167b5d279bf1e8da9c1115cd5ff84d)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#70](https://github.com-terrylica/terrylica/rangebar-py/issues/70)
* **#96 Task #71:** implement vec reuse pool for ExportProcessor streaming hot path ([404e30b](https://github.com-terrylica/terrylica/rangebar-py/commit/404e30b06b9a2f14b619b24cc458f8780653f3d4)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#71](https://github.com-terrylica/terrylica/rangebar-py/issues/71)
* **#96 Task #81:** batch dict set_item calls to reduce FFI crossings ([a23c84a](https://github.com-terrylica/terrylica/rangebar-py/commit/a23c84a55bd0df6cc7a1880b49c78a9633818640)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#81](https://github.com-terrylica/terrylica/rangebar-py/issues/81)
* **#96 Task #85 Phase 1:** Reorder RangeBar fields for cache locality ([467bd2a](https://github.com-terrylica/terrylica/rangebar-py/commit/467bd2ac9bf0084f1e1e0a74806eff65a38a816b)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#85](https://github.com-terrylica/terrylica/rangebar-py/issues/85)
* **#96 Task #95:** Optimize checkpoint state cloning with std::mem::take() ([ebed3bc](https://github.com-terrylica/terrylica/rangebar-py/commit/ebed3bc60624929ba32d911cfce1dda04212904e)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#95](https://github.com-terrylica/terrylica/rangebar-py/issues/95)
* **#96 Task #98:** Cache threshold coefficient in RangeBarProcessor ([e0b316a](https://github.com-terrylica/terrylica/rangebar-py/commit/e0b316a48e64fcdfb15def04bde8b25260a17a43)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#98](https://github.com-terrylica/terrylica/rangebar-py/issues/98) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#98](https://github.com-terrylica/terrylica/rangebar-py/issues/98)
* **#96 Task #99:** Memoize float conversions in inter-bar lookback processing ([7ba1502](https://github.com-terrylica/terrylica/rangebar-py/commit/7ba1502de5be43ace5a93543084df068bb346db8)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#99](https://github.com-terrylica/terrylica/rangebar-py/issues/99) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#99](https://github.com-terrylica/terrylica/rangebar-py/issues/99)
* **#96 Tasks #195-197:** Implement Tier 1 power/ILP/constant optimizations ([b9de9cd](https://github.com-terrylica/terrylica/rangebar-py/commit/b9de9cd98bcb4399f18fe6a669c02df2c9678f14)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#195-197](https://github.com-terrylica/terrylica/rangebar-py/issues/195-197)
* **#96 Tasks #89-92:** FFI and memory optimization blitz ([6ad1a0c](https://github.com-terrylica/terrylica/rangebar-py/commit/6ad1a0c6b7b380daeee7861563b2b67040b77452)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#89-92](https://github.com-terrylica/terrylica/rangebar-py/issues/89-92) [#91-97](https://github.com-terrylica/terrylica/rangebar-py/issues/91-97) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **#96:** Pre-allocate Vecs in inter-bar test data creation for 8-15% speedup ([82d4234](https://github.com-terrylica/terrylica/rangebar-py/commit/82d423457fd544283c430b83f4da5bfa0f660f32)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96) [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **phase-1:** Add Hurst DFA profiling benchmark, validate O(n²) complexity ([b0b7df0](https://github.com-terrylica/terrylica/rangebar-py/commit/b0b7df0d28f4c6746e6f55e948b2c28beb781fc4)), closes [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)
* **phase-2:** Optimize Hurst DFA computation with inline directives and faster variance ([6ff6d07](https://github.com-terrylica/terrylica/rangebar-py/commit/6ff6d07315e3ba09f201a664aaac4dbf39410692))
* **phase-3a:** Add DFA performance research benchmark template ([534f0a0](https://github.com-terrylica/terrylica/rangebar-py/commit/534f0a0d89c6c562eef364900c5cccdea5e351fa))
* **phase-3:** Explore hurst crate integration, benchmark optimization analysis ([2017743](https://github.com-terrylica/terrylica/rangebar-py/commit/2017743af738de6253e339f07578b4b7bec0ca37))

# [12.27.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.26.0...v12.27.0) (2026-02-23)


### Bug Fixes

* **#106:** improve heartbeat message clarity and checkpoint integrity tolerance ([70096f5](https://github.com-terrylica/terrylica/rangebar-py/commit/70096f5b7b2afd78693847241a857b7aba855480)), closes [#106](https://github.com-terrylica/terrylica/rangebar-py/issues/106)


### Features

* **#103-106:** backfill watcher v2, freshness detection, deploy task, heartbeat fixes ([9bcb95b](https://github.com-terrylica/terrylica/rangebar-py/commit/9bcb95b6b7f80553a3280e7ced80e1236cb808e9)), closes [#103-106](https://github.com-terrylica/terrylica/rangebar-py/issues/103-106)
* **#107:** implement sidecar watchdog for half-open TCP detection ([54d0b46](https://github.com-terrylica/terrylica/rangebar-py/commit/54d0b46ba3341ce2a4a4506cd113001ae031a00c)), closes [#107](https://github.com-terrylica/terrylica/rangebar-py/issues/107)

# [12.26.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.25.1...v12.26.0) (2026-02-22)


### Bug Fixes

* **data-quality:** Issue [#99](https://github.com-terrylica/terrylica/rangebar-py/issues/99) — laguerre bars_in_regime already correct, add verification ([48f1b44](https://github.com-terrylica/terrylica/rangebar-py/commit/48f1b447b853329d98196919bcd3cfcc31451af9)), closes [terrylica/atr-adaptive-laguerre#2](https://github.com-terrylica/terrylica/atr-adaptive-laguerre/issues/2)
* **deps:** bump laguerre dep to >=2.4.1 now that PyPI is updated ([bf61ff0](https://github.com-terrylica/terrylica/rangebar-py/commit/bf61ff0c4206bd8b11af5303f6f5cc6718f42262)), closes [#98](https://github.com-terrylica/terrylica/rangebar-py/issues/98) [#99](https://github.com-terrylica/terrylica/rangebar-py/issues/99)
* **plugins:** auto-migrate plugin columns in _ensure_schema + add E2E validation ([60c9b13](https://github.com-terrylica/terrylica/rangebar-py/commit/60c9b13e3c3bd9670f62589032b45fc90fa8b617))
* **plugins:** bump laguerre dep to require bars_in_regime fix, add re-population script ([2dbba0d](https://github.com-terrylica/terrylica/rangebar-py/commit/2dbba0d5b1ba6d88f882725ce1e2bc0299074446))
* **release:** migrate PyPI publish credentials to Claude Automation vault ([1511f87](https://github.com-terrylica/terrylica/rangebar-py/commit/1511f871940e3e7b3082029ea0d9a1df6db9cbb3))
* **release:** wire release:crates into release:full depends chain ([40f5d70](https://github.com-terrylica/terrylica/rangebar-py/commit/40f5d70ab0c4d81c67040e75fe80f33599c4e99f))
* **schema:** correct is_liquidation_cascade threshold formula ([29bdd7a](https://github.com-terrylica/terrylica/rangebar-py/commit/29bdd7a70f82bf93693f1d5b6ae64eb9ebe05009)), closes [#101](https://github.com-terrylica/terrylica/rangebar-py/issues/101)
* **symbols:** raise SHIBUSDT minimum threshold to 1000 dbps ([e8351f3](https://github.com-terrylica/terrylica/rangebar-py/commit/e8351f34debbdebdcc62e4513f3d72154ccbfd49))
* **thresholds:** add WIFUSDT 1000 dbps hard floor + document meme coin Tier A policy ([6579b33](https://github.com-terrylica/terrylica/rangebar-py/commit/6579b33263ea0601d4b2a7d864f7bb9a29840d8c))


### Features

* **#101:** add is_liquidation_cascade flag for Binance atomic batch bars ([2f8a7d8](https://github.com-terrylica/terrylica/rangebar-py/commit/2f8a7d854cc2ea1f5e1cca815dbb6a232db01f2e)), closes [#101](https://github.com-terrylica/terrylica/rangebar-py/issues/101) [hi#low](https://github.com-terrylica/hi/issues/low) [#101](https://github.com-terrylica/terrylica/rangebar-py/issues/101) [#101](https://github.com-terrylica/terrylica/rangebar-py/issues/101)
* **#102:** implement backfill_watcher.py for on-demand flowsurface backfill ([39c0df7](https://github.com-terrylica/terrylica/rangebar-py/commit/39c0df76897a0e442fa19e4f2bfbbc075dd4846b)), closes [#102](https://github.com-terrylica/terrylica/rangebar-py/issues/102) [#102](https://github.com-terrylica/terrylica/rangebar-py/issues/102)
* **registry:** add TRXUSDT, ZECUSDT, HBARUSDT, PAXGUSDT, PEPEUSDT, TONUSDT ([5737097](https://github.com-terrylica/terrylica/rangebar-py/commit/57370972536175b8a858aa80a801274ab63d1a5d))

## [12.25.1](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.25.0...v12.25.1) (2026-02-19)

# [12.25.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.24.0...v12.25.0) (2026-02-19)


### Bug Fixes

* **release:** add file-based fallback for crates.io token in release:crates task ([c9093db](https://github.com-terrylica/terrylica/rangebar-py/commit/c9093db7c8a3176abcd6b9b96d08472479e94c12))


### Features

* **plugins:** FeatureProvider plugin system for external feature enrichment (Issue [#98](https://github.com-terrylica/terrylica/rangebar-py/issues/98)) ([c38fe52](https://github.com-terrylica/terrylica/rangebar-py/commit/c38fe52d9c70963da6e8e2d915b3a273b6bfb8e8))

# [12.24.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.23.1...v12.24.0) (2026-02-18)


### Features

* **rust:** enable crates.io publishing for core, providers, streaming (Issue [#97](https://github.com-terrylica/terrylica/rangebar-py/issues/97)) ([50ba54a](https://github.com-terrylica/terrylica/rangebar-py/commit/50ba54af21a2005fe9e88164cb0df7a1760e99aa))

## [12.23.1](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.23.0...v12.23.1) (2026-02-18)


### Bug Fixes

* **sidecar:** skip stale checkpoint injection after gap-fill (Issue [#96](https://github.com-terrylica/terrylica/rangebar-py/issues/96)) ([cd9242a](https://github.com-terrylica/terrylica/rangebar-py/commit/cd9242aa06207be47b6ecd48966ba83d505e97f8))

# [12.23.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.22.0...v12.23.0) (2026-02-18)


### Bug Fixes

* **test:** filter duplicate bars in month-boundary continuity test ([fdc927d](https://github.com-terrylica/terrylica/rangebar-py/commit/fdc927d459433d3fb9b29e811e96a0097fa96db6))


### Features

* **checkpoint:** wire Rust checkpoint system through all Python layers (Issue [#97](https://github.com-terrylica/terrylica/rangebar-py/issues/97)) ([743796f](https://github.com-terrylica/terrylica/rangebar-py/commit/743796f53cc2046e91c83544e2e2491c48534736))
* **mise:** add recency backfill tasks (Issue [#92](https://github.com-terrylica/terrylica/rangebar-py/issues/92)) ([7f54566](https://github.com-terrylica/terrylica/rangebar-py/commit/7f545664e0e6d4f144f65daa7d3e0c450ad9401f))

# [12.22.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.21.1...v12.22.0) (2026-02-17)


### Features

* **perf:** memory efficiency optimizations for bar processing (Issue [#95](https://github.com-terrylica/terrylica/rangebar-py/issues/95)) ([6d84cb8](https://github.com-terrylica/terrylica/rangebar-py/commit/6d84cb8b0ad2c82d04780a81befd938f9e927f47))

## [12.21.1](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.21.0...v12.21.1) (2026-02-13)

# [12.21.0](https://github.com-terrylica/terrylica/rangebar-py/compare/v12.20.0...v12.21.0) (2026-02-13)


### Bug Fixes

* **test:** respect per-symbol min threshold in tier1 symbol test ([bf450d3](https://github.com-terrylica/terrylica/rangebar-py/commit/bf450d3f361d6f1ecf146055d0875282aae27586)), closes [#89](https://github.com-terrylica/terrylica/rangebar-py/issues/89)


### Features

* **compat:** add alpha-forge integration layer with TOML-embedded feature manifest (Issue [#95](https://github.com-terrylica/terrylica/rangebar-py/issues/95)) ([42e392a](https://github.com-terrylica/terrylica/rangebar-py/commit/42e392a852439de6dac73abb7310a1be0a1e8efb))

# [12.20.0](https://github.com/terrylica/rangebar-py/compare/v12.19.0...v12.20.0) (2026-02-12)


### Features

* **streaming:** add Layer 3 live streaming sidecar (Issues [#91](https://github.com/terrylica/rangebar-py/issues/91), [#92](https://github.com/terrylica/rangebar-py/issues/92), [#93](https://github.com/terrylica/rangebar-py/issues/93)) ([b3e8e2e](https://github.com/terrylica/rangebar-py/commit/b3e8e2e93d69431427ac8a13ce992cddf8b82299))

# [12.19.0](https://github.com/terrylica/rangebar-py/compare/v12.18.4...v12.19.0) (2026-02-12)


### Features

* **recency:** add Layer 2 recency backfill with adaptive polling (Issue [#92](https://github.com/terrylica/rangebar-py/issues/92)) ([cecae88](https://github.com/terrylica/rangebar-py/commit/cecae8886b429bc78a135b8dca3a77bd0695e94c))

## [12.18.4](https://github.com/terrylica/rangebar-py/compare/v12.18.3...v12.18.4) (2026-02-12)

## [12.18.3](https://github.com/terrylica/rangebar-py/compare/v12.18.2...v12.18.3) (2026-02-12)


### Bug Fixes

* **release:** add releaseRules so docs/style/refactor/perf trigger patch bump ([941a7e4](https://github.com/terrylica/rangebar-py/commit/941a7e4a995f1a1ef6e1a880ff9ae66add9a02fb))

## [12.18.2](https://github.com/terrylica/rangebar-py/compare/v12.18.1...v12.18.2) (2026-02-11)


### Bug Fixes

* **release:** tolerate semantic-release success step failures ([1b6fe2c](https://github.com/terrylica/rangebar-py/commit/1b6fe2cc2c3d20218b06ed241ebc1ffe446cb7e9))

## [12.18.1](https://github.com/terrylica/rangebar-py/compare/v12.18.0...v12.18.1) (2026-02-11)


### Bug Fixes

* **scripts:** derive symbol list from registry SSoT, remove hardcoded fallback ([282df67](https://github.com/terrylica/rangebar-py/commit/282df6769d2138558300e015f7828db66f16a6b1)), closes [#91](https://github.com/terrylica/rangebar-py/issues/91)

# [12.18.0](https://github.com/terrylica/rangebar-py/compare/v12.17.0...v12.18.0) (2026-02-11)


### Bug Fixes

* **scripts:** check per-table settings via SHOW CREATE TABLE ([#90](https://github.com/terrylica/rangebar-py/issues/90)) ([131e478](https://github.com/terrylica/rangebar-py/commit/131e4780f0139ed8086ecb42c6cfa31cdd2a3ab6))


### Features

* **clickhouse:** add REMOTE connection mode, always route through bigblack ([#90](https://github.com/terrylica/rangebar-py/issues/90)) ([3618fed](https://github.com/terrylica/rangebar-py/commit/3618fed318b367e8ad499a6e983e57e8f10a986d)), closes [#89](https://github.com/terrylica/rangebar-py/issues/89)

# [12.17.0](https://github.com/terrylica/rangebar-py/compare/v12.16.0...v12.17.0) (2026-02-11)


### Bug Fixes

* **clickhouse:** five-layer dedup hardening against OPTIMIZE timeout crashes ([#90](https://github.com/terrylica/rangebar-py/issues/90)) ([ebf4e08](https://github.com/terrylica/rangebar-py/commit/ebf4e08fc4504429b38fc9fd37eb9e35e1dfb28e))


### Features

* **threshold:** provisional 500 dbps floor for 11 mid-tier symbols ([#89](https://github.com/terrylica/rangebar-py/issues/89)) ([9adf8f4](https://github.com/terrylica/rangebar-py/commit/9adf8f4f3110aab96933a2bd483111ff090b4beb))

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
