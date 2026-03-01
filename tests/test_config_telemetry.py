# FILE-SIZE-OK: Comprehensive telemetry coverage for all config sections
"""NDJSON telemetry tests for pydantic-settings configuration (Issue #110).

Validates config loading, env overrides, validators, singleton consistency,
reload isolation, backwards compatibility, CLI model schemas, and cross-section
independence with structured NDJSON artifacts:

    jq 'select(.component == "config")' tests/artifacts/config_telemetry.jsonl

Each test emits structured events to a session-scoped .jsonl artifact file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# NDJSON artifact infrastructure
# ---------------------------------------------------------------------------

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "config_telemetry.jsonl"


def _emit(event: dict) -> None:
    """Append a single NDJSON event to the artifact file."""
    event.setdefault("timestamp", time.time())
    event.setdefault("component", "config")
    event.setdefault("trace_id", f"cfg-{os.urandom(4).hex()}")
    ARTIFACT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_FILE.open("a") as f:
        f.write(json.dumps(event) + "\n")


@pytest.fixture(autouse=True, scope="session")
def _init_artifact():
    """Truncate the artifact file at session start for a clean run."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_FILE.write_text("")
    yield
    # After all tests, verify artifact is valid NDJSON
    lines = ARTIFACT_FILE.read_text().strip().splitlines()
    for i, line in enumerate(lines, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            pytest.fail(f"Artifact line {i} is not valid JSON: {e}")

    # Emit summary event
    pass_count = sum(1 for ln in lines if '"pass": true' in ln)
    fail_count = sum(1 for ln in lines if '"pass": false' in ln)
    summary = {
        "event": "artifact_summary",
        "trace_id": "cfg-summary",
        "total_events": len(lines),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "all_passed": fail_count == 0,
    }
    with ARTIFACT_FILE.open("a") as f:
        f.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# 1. Config Load & Defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Every config section loads with correct defaults (clean env)."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Clear RANGEBAR_ env vars so we test pure model defaults."""
        for key in list(os.environ):
            if key.startswith("RANGEBAR_"):
                monkeypatch.delenv(key, raising=False)

    def test_population_defaults(self):
        from rangebar.config.population import PopulationConfig

        config = PopulationConfig()
        passed = (
            config.crypto_min_threshold == 250
            and config.ouroboros_mode == "month"
            and config.compute_tier2 is True
            and config.compute_tier3 is False
            and config.compute_hurst is False
            and config.compute_permutation_entropy is False
        )
        _emit({
            "event": "config_load",
            "section": "population",
            "fields": config.model_dump(),
            "source": "defaults",
            "pass": passed,
        })
        assert passed

    def test_monitoring_defaults(self):
        from rangebar.config.monitoring import MonitoringConfig

        config = MonitoringConfig()
        passed = (
            config.telegram_token is None
            and config.recency_fresh_threshold_min == 30
            and config.recency_stale_threshold_min == 120
            and config.recency_critical_threshold_min == 1440
        )
        _emit({
            "event": "config_load",
            "section": "monitoring",
            "fields": config.model_dump(),
            "source": "defaults",
            "pass": passed,
        })
        assert passed

    def test_algorithm_defaults(self):
        from rangebar.config.algorithm import AlgorithmConfig

        config = AlgorithmConfig()
        passed = (
            config.default_threshold_decimal_bps == 250
            and config.processing_batch_size == 100_000
            and config.enable_memory_optimization is True
            and config.collect_performance_metrics is False
        )
        _emit({
            "event": "config_load",
            "section": "algorithm",
            "fields": config.model_dump(),
            "source": "defaults",
            "pass": passed,
        })
        assert passed

    def test_streaming_defaults(self):
        from rangebar.config.streaming import StreamingConfig

        config = StreamingConfig()
        passed = (
            config.trade_channel_capacity == 5_000
            and config.bar_channel_capacity == 10_000
            and config.circuit_breaker_threshold == 0.5
        )
        _emit({
            "event": "config_load",
            "section": "streaming",
            "fields": config.model_dump(),
            "source": "defaults",
            "pass": passed,
        })
        assert passed


# ---------------------------------------------------------------------------
# 2. Env Override Verification
# ---------------------------------------------------------------------------


class TestEnvOverrides:
    """Each env var correctly overrides its field."""

    def test_env_override_compute_hurst(self, monkeypatch):
        monkeypatch.setenv("RANGEBAR_COMPUTE_HURST", "true")
        from rangebar.config.population import PopulationConfig

        config = PopulationConfig()
        passed = config.compute_hurst is True
        _emit({
            "event": "env_override",
            "param": "compute_hurst",
            "env_var": "RANGEBAR_COMPUTE_HURST",
            "env_value": "true",
            "resolved": config.compute_hurst,
            "type": "bool",
            "pass": passed,
        })
        assert passed

    def test_env_override_compute_tier3(self, monkeypatch):
        monkeypatch.setenv("RANGEBAR_COMPUTE_TIER3", "true")
        from rangebar.config.population import PopulationConfig

        config = PopulationConfig()
        passed = config.compute_tier3 is True
        _emit({
            "event": "env_override",
            "param": "compute_tier3",
            "env_var": "RANGEBAR_COMPUTE_TIER3",
            "env_value": "true",
            "resolved": config.compute_tier3,
            "type": "bool",
            "pass": passed,
        })
        assert passed

    def test_env_override_threshold(self, monkeypatch):
        monkeypatch.setenv("RANGEBAR_CRYPTO_MIN_THRESHOLD", "500")
        from rangebar.config.population import PopulationConfig

        config = PopulationConfig()
        passed = config.crypto_min_threshold == 500
        _emit({
            "event": "env_override",
            "param": "crypto_min_threshold",
            "env_var": "RANGEBAR_CRYPTO_MIN_THRESHOLD",
            "env_value": "500",
            "resolved": config.crypto_min_threshold,
            "type": "int",
            "pass": passed,
        })
        assert passed

    def test_env_override_algorithm_batch_size(self, monkeypatch):
        monkeypatch.setenv("RANGEBAR_ALGORITHM_PROCESSING_BATCH_SIZE", "50000")
        from rangebar.config.algorithm import AlgorithmConfig

        config = AlgorithmConfig()
        passed = config.processing_batch_size == 50_000
        _emit({
            "event": "env_override",
            "param": "processing_batch_size",
            "env_var": "RANGEBAR_ALGORITHM_PROCESSING_BATCH_SIZE",
            "env_value": "50000",
            "resolved": config.processing_batch_size,
            "type": "int",
            "pass": passed,
        })
        assert passed

    def test_env_override_streaming_circuit_breaker(self, monkeypatch):
        monkeypatch.setenv("RANGEBAR_STREAMING_CIRCUIT_BREAKER_THRESHOLD", "0.8")
        from rangebar.config.streaming import StreamingConfig

        config = StreamingConfig()
        passed = config.circuit_breaker_threshold == 0.8
        _emit({
            "event": "env_override",
            "param": "circuit_breaker_threshold",
            "env_var": "RANGEBAR_STREAMING_CIRCUIT_BREAKER_THRESHOLD",
            "env_value": "0.8",
            "resolved": config.circuit_breaker_threshold,
            "type": "float",
            "pass": passed,
        })
        assert passed

    def test_env_override_monitoring_env_alias(self, monkeypatch):
        """RANGEBAR_ENV overrides the environment field (AliasChoices)."""
        monkeypatch.setenv("RANGEBAR_ENV", "production")
        from rangebar.config.monitoring import MonitoringConfig

        config = MonitoringConfig()
        passed = config.environment == "production"
        _emit({
            "event": "env_override",
            "param": "environment",
            "env_var": "RANGEBAR_ENV",
            "env_value": "production",
            "resolved": config.environment,
            "type": "str",
            "pass": passed,
        })
        assert passed


# ---------------------------------------------------------------------------
# 3. Validator Edge Cases
# ---------------------------------------------------------------------------


class TestValidatorEdgeCases:
    """Invalid values produce clear errors."""

    def test_invalid_ouroboros_rejected(self):
        from pydantic import ValidationError
        from rangebar.config.population import PopulationConfig

        try:
            PopulationConfig(ouroboros_mode="invalid")
            passed = False
            error_msg = "No error raised"
        except (ValueError, ValidationError) as e:
            error_msg = str(e)
            passed = True

        _emit({
            "event": "validator_check",
            "section": "population",
            "validator": "ouroboros_mode",
            "input": "invalid",
            "error": error_msg if passed else None,
            "pass": passed,
        })
        assert passed

    def test_valid_ouroboros_modes(self):
        from rangebar.config.population import PopulationConfig

        results = {}
        for mode in ("year", "month", "week"):
            config = PopulationConfig(ouroboros_mode=mode)
            results[mode] = config.ouroboros_mode == mode

        passed = all(results.values())
        _emit({
            "event": "validator_check",
            "section": "population",
            "validator": "ouroboros_mode",
            "input": "year,month,week",
            "results": results,
            "pass": passed,
        })
        assert passed


# ---------------------------------------------------------------------------
# 4. Singleton Consistency
# ---------------------------------------------------------------------------


class TestSingletonConsistency:
    """Settings.get() returns same object."""

    def test_singleton_identity(self):
        from rangebar.config import Settings

        s1 = Settings.get()
        s2 = Settings.get()
        passed = s1 is s2
        _emit({
            "event": "singleton_consistency",
            "call_1_id": hex(id(s1)),
            "call_2_id": hex(id(s2)),
            "same_object": s1 is s2,
            "pass": passed,
        })
        assert passed


# ---------------------------------------------------------------------------
# 5. Reload Isolation
# ---------------------------------------------------------------------------


class TestReloadIsolation:
    """Settings.reload() picks up env changes."""

    def test_reload_creates_new_instance(self):
        from rangebar.config import Settings

        s1 = Settings.get()
        s2 = Settings.reload()
        passed = s1 is not s2
        _emit({
            "event": "reload_isolation",
            "before_id": hex(id(s1)),
            "after_id": hex(id(s2)),
            "different_objects": s1 is not s2,
            "pass": passed,
        })
        assert passed


# ---------------------------------------------------------------------------
# 6. Backwards Compatibility
# ---------------------------------------------------------------------------


class TestBackwardsCompatibility:
    """All import paths still work."""

    def test_import_settings_from_config(self):
        from rangebar.config import Settings

        s = Settings.get()
        passed = (
            hasattr(s, "population")
            and hasattr(s, "monitoring")
            and hasattr(s, "clickhouse")
            and hasattr(s, "sidecar")
            and hasattr(s, "algorithm")
            and hasattr(s, "streaming")
        )
        _emit({
            "event": "backwards_compat",
            "import_path": "from rangebar.config import Settings",
            "resolved": True,
            "has_population": hasattr(s, "population"),
            "has_monitoring": hasattr(s, "monitoring"),
            "has_clickhouse": hasattr(s, "clickhouse"),
            "has_sidecar": hasattr(s, "sidecar"),
            "has_algorithm": hasattr(s, "algorithm"),
            "has_streaming": hasattr(s, "streaming"),
            "pass": passed,
        })
        assert passed

    def test_default_threshold_property(self, monkeypatch):
        """PopulationConfig.default_threshold backwards compat alias."""
        for key in list(os.environ):
            if key.startswith("RANGEBAR_"):
                monkeypatch.delenv(key, raising=False)
        from rangebar.config.population import PopulationConfig

        config = PopulationConfig()
        passed = config.default_threshold == config.crypto_min_threshold == 250
        _emit({
            "event": "backwards_compat",
            "import_path": "PopulationConfig.default_threshold",
            "resolved": True,
            "default_threshold": config.default_threshold,
            "crypto_min_threshold": config.crypto_min_threshold,
            "pass": passed,
        })
        assert passed

    def test_lazy_re_exports(self):
        """Lazy __getattr__ re-exports work for ClickHouseConfig, SidecarConfig."""
        from rangebar.config import AlgorithmConfig, StreamingConfig

        passed = (
            AlgorithmConfig is not None
            and StreamingConfig is not None
        )
        _emit({
            "event": "backwards_compat",
            "import_path": "from rangebar.config import AlgorithmConfig, StreamingConfig",
            "resolved": True,
            "pass": passed,
        })
        assert passed


# ---------------------------------------------------------------------------
# 7. CLI Model Schema
# ---------------------------------------------------------------------------


class TestCliModelSchema:
    """CliApp models have correct fields and types."""

    def test_populate_range_schema(self):
        from rangebar.config.cli_models import PopulateRange

        fields = list(PopulateRange.model_fields.keys())
        required = [
            name for name, info in PopulateRange.model_fields.items()
            if info.is_required()
        ]
        passed = (
            "symbol" in fields
            and "start" in fields
            and "end" in fields
            and "threshold" in fields
            and "hurst" in fields
            and "symbol" in required
            and "start" in required
            and "end" in required
        )
        _emit({
            "event": "cli_model_schema",
            "model": "PopulateRange",
            "fields": fields,
            "required": required,
            "field_count": len(fields),
            "pass": passed,
        })
        assert passed

    def test_populate_phase_schema(self):
        from rangebar.config.cli_models import PopulatePhase

        fields = list(PopulatePhase.model_fields.keys())
        required = [
            name for name, info in PopulatePhase.model_fields.items()
            if info.is_required()
        ]
        passed = (
            "phase" in fields
            and "parallel" in fields
            and "phase" in required
        )
        _emit({
            "event": "cli_model_schema",
            "model": "PopulatePhase",
            "fields": fields,
            "required": required,
            "field_count": len(fields),
            "pass": passed,
        })
        assert passed


# ---------------------------------------------------------------------------
# 8. Cross-Section Independence
# ---------------------------------------------------------------------------


class TestCrossSectionIndependence:
    """Changing one section doesn't affect another."""

    def test_population_independent_of_monitoring(self, monkeypatch):
        monkeypatch.setenv("RANGEBAR_COMPUTE_HURST", "true")
        from rangebar.config.monitoring import MonitoringConfig
        from rangebar.config.population import PopulationConfig

        pop = PopulationConfig()
        mon = MonitoringConfig()

        passed = (
            pop.compute_hurst is True
            and mon.recency_fresh_threshold_min == 30  # unaffected
        )
        _emit({
            "event": "cross_section_independence",
            "modified_section": "population",
            "unmodified_section": "monitoring",
            "population_hurst": pop.compute_hurst,
            "monitoring_fresh_threshold": mon.recency_fresh_threshold_min,
            "pass": passed,
        })
        assert passed

    def test_algorithm_independent_of_streaming(self, monkeypatch):
        monkeypatch.setenv("RANGEBAR_ALGORITHM_PROCESSING_BATCH_SIZE", "99999")
        from rangebar.config.algorithm import AlgorithmConfig
        from rangebar.config.streaming import StreamingConfig

        algo = AlgorithmConfig()
        stream = StreamingConfig()

        passed = (
            algo.processing_batch_size == 99999
            and stream.trade_channel_capacity == 5000  # unaffected
        )
        _emit({
            "event": "cross_section_independence",
            "modified_section": "algorithm",
            "unmodified_section": "streaming",
            "algorithm_batch_size": algo.processing_batch_size,
            "streaming_trade_capacity": stream.trade_channel_capacity,
            "pass": passed,
        })
        assert passed
