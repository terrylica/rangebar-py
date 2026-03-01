"""Streaming configuration for Rust tunables.

Issue #110 Phase 7: Expose user-tunable subset of rangebar-streaming
StreamingProcessorConfig via pydantic-settings. Maps 1:1 with the existing
PyStreamingConfig PyO3 class in src/streaming_bindings.rs.

Environment variables use RANGEBAR_STREAMING_ prefix (same as SidecarConfig
but different field names — no collision).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class StreamingConfig(BaseSettings):
    """User-tunable Rust streaming processor parameters.

    These map to the PyStreamingConfig PyO3 class (src/streaming_bindings.rs).
    When constructing a LiveBarEngine or StreamingProcessor, these values
    can be used as defaults that env vars override.

    Environment Variables
    ---------------------
    RANGEBAR_STREAMING_TRADE_CHANNEL_CAPACITY : int
        Trade channel capacity (default: 5000)
    RANGEBAR_STREAMING_BAR_CHANNEL_CAPACITY : int
        Bar channel capacity (default: 10000)
    RANGEBAR_STREAMING_MEMORY_THRESHOLD_BYTES : int
        Memory threshold before backpressure (default: 100000000 = 100MB)
    RANGEBAR_STREAMING_BACKPRESSURE_TIMEOUT_MS : int
        Backpressure timeout in ms (default: 100)
    RANGEBAR_STREAMING_CIRCUIT_BREAKER_THRESHOLD : float
        Circuit breaker error rate threshold (default: 0.5)
    RANGEBAR_STREAMING_CIRCUIT_BREAKER_TIMEOUT_SECS : int
        Circuit breaker recovery timeout in seconds (default: 30)
    """

    model_config = SettingsConfigDict(
        env_prefix="RANGEBAR_STREAMING_",
        case_sensitive=False,
    )

    trade_channel_capacity: int = 5_000
    bar_channel_capacity: int = 10_000
    memory_threshold_bytes: int = 100_000_000
    backpressure_timeout_ms: int = 100
    circuit_breaker_threshold: float = 0.5
    circuit_breaker_timeout_secs: int = 30
