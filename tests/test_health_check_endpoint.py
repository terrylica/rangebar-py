"""Tests for Issue #109: Health check CLI and endpoint.

Validates health check output format and circuit breaker integration.
"""

from __future__ import annotations

import json
import subprocess
import sys


def test_health_check_cli_json_output():
    """health_check.py outputs valid JSON with required keys."""
    result = subprocess.run(
        [sys.executable, "scripts/health_check.py"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    # Parse JSON output (may be exit code 1 if ClickHouse not running)
    output = json.loads(result.stdout)
    assert "status" in output
    assert output["status"] in ("healthy", "unhealthy")
    assert "checks" in output

    checks = output["checks"]
    assert "memory" in checks
    assert "disk_space" in checks
    assert "circuit_breaker" in checks


def test_health_check_includes_circuit_state():
    """Health check includes circuit breaker state."""
    from rangebar.health_checks import run_all_checks

    results = run_all_checks()
    assert "circuit_breaker" in results
    assert results["circuit_breaker"] in ("closed", "open", "half_open", "unknown")


def test_run_all_checks_returns_dict():
    """run_all_checks() returns a dict with all expected keys."""
    from rangebar.health_checks import run_all_checks

    results = run_all_checks()
    assert isinstance(results, dict)
    expected_keys = {"clickhouse", "memory", "disk_space", "websocket", "circuit_breaker"}
    assert expected_keys == set(results.keys())


def test_memory_check_passes():
    """Memory check should pass in test environment."""
    from rangebar.health_checks import check_memory

    assert check_memory(threshold_mb=8192) is True


def test_disk_space_check_passes():
    """Disk space check should pass with generous threshold."""
    from rangebar.health_checks import check_disk_space

    assert check_disk_space(min_free_gb=0.1) is True
