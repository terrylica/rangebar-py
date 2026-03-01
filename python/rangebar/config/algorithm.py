"""Algorithm configuration for Rust tunables.

Issue #110 Phase 7: Expose user-tunable subset of rangebar-config AlgorithmConfig
via pydantic-settings. Safety invariants (validate_*, fixed_point_decimals) stay
Rust-only — they're correctness guarantees, not user knobs.

Environment variables use RANGEBAR_ALGORITHM_ prefix.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class AlgorithmConfig(BaseSettings):
    """User-tunable subset of Rust AlgorithmConfig.

    Environment Variables
    ---------------------
    RANGEBAR_ALGORITHM_DEFAULT_THRESHOLD_DECIMAL_BPS : int
        Default threshold in dbps (default: 250)
    RANGEBAR_ALGORITHM_PROCESSING_BATCH_SIZE : int
        Processing batch size for chunked operations (default: 100000)
    RANGEBAR_ALGORITHM_ENABLE_MEMORY_OPTIMIZATION : bool
        Enable memory optimization in Rust processor (default: True)
    RANGEBAR_ALGORITHM_COLLECT_PERFORMANCE_METRICS : bool
        Collect performance metrics during processing (default: False)
    """

    model_config = SettingsConfigDict(
        env_prefix="RANGEBAR_ALGORITHM_",
        case_sensitive=False,
    )

    default_threshold_decimal_bps: int = 250
    processing_batch_size: int = 100_000
    enable_memory_optimization: bool = True
    collect_performance_metrics: bool = False
