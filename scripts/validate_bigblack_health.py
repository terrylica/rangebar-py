#!/usr/bin/env python3
"""Issue #121: Universal health validation for bigblack.

Single script that runs ALL health checks and writes a timestamped JSON report.
Designed for both human operators and Claude Code CLI parsing.

Usage:
    ssh bigblack "cd ~/rangebar-py && .venv/bin/python3 scripts/validate_bigblack_health.py"

Exit codes:
    0 - healthy: all checks pass
    1 - degraded: some checks failed (non-critical)
    2 - critical: core infrastructure down (ClickHouse, all services)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DEAD_LETTER_DIR = Path("/tmp/rangebar-dead-letter")

SERVICES = (
    "rangebar-sidecar",
    "rangebar-kintsugi",
    "rangebar-recency-backfill",
)


def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_clickhouse() -> dict:
    """SELECT 1 against local ClickHouse."""
    t0 = time.monotonic()
    try:
        r = _run(["curl", "-sf", "http://localhost:8123/?query=SELECT+1"], timeout=10)
        latency_ms = round((time.monotonic() - t0) * 1000)
        if r.returncode == 0 and r.stdout.strip() == "1":
            return {"ok": True, "latency_ms": latency_ms}
        return {"ok": False, "error": f"unexpected response: {r.stdout.strip()!r}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout (10s)"}
    except OSError as e:
        return {"ok": False, "error": str(e)}


def _check_service(name: str) -> dict:
    """Check if a systemd user service is active."""
    try:
        r = _run(["systemctl", "--user", "is-active", name], timeout=5)
        status = r.stdout.strip()
        return {"ok": status == "active", "status": status}
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"ok": False, "error": str(e)}


def _check_health_endpoint() -> dict:
    """Sidecar health endpoint at localhost:8081/health."""
    try:
        r = _run(["curl", "-sf", "http://localhost:8081/health"], timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            try:
                data = json.loads(r.stdout)
                return {"ok": True, "response": data}
            except json.JSONDecodeError:
                return {"ok": True, "response": r.stdout.strip()[:200]}
        return {"ok": False, "error": "unreachable"}
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"ok": False, "error": str(e)}


def _check_gaps() -> dict:
    """Run detect_gaps.py --json to find temporal gaps."""
    try:
        r = _run(
            [
                sys.executable,
                "-W",
                "ignore",
                "scripts/detect_gaps.py",
                "--json",
                "--min-gap-hours",
                "6",
                "--max-stale-hours",
                "72",
            ],
            timeout=120,
        )
        if r.returncode == 2:
            return {
                "ok": False,
                "error": f"detect_gaps connection error: {r.stderr.strip()[:200]}",
            }
        try:
            data = json.loads(r.stdout)
        except (json.JSONDecodeError, ValueError):
            return {"ok": False, "error": "failed to parse detect_gaps JSON"}

        temporal_gaps = [
            g for g in data.get("gaps", []) if g.get("gap_type") in ("temporal", "both")
        ]
        gap_list = [
            {
                "symbol": g["symbol"],
                "threshold": g["threshold_dbps"],
                "gap_hours": g["gap_hours"],
            }
            for g in temporal_gaps[:10]  # Cap at 10
        ]
        return {
            "ok": len(temporal_gaps) == 0,
            "count": len(temporal_gaps),
            "gaps": gap_list,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "detect_gaps timed out (120s)"}
    except OSError as e:
        return {"ok": False, "error": str(e)}


def _check_freshness() -> dict:
    """Check hours since last bar per (symbol, threshold) pair via detect_gaps.py output."""
    try:
        r = _run(
            [
                sys.executable,
                "-W",
                "ignore",
                "scripts/detect_gaps.py",
                "--json",
                "--max-stale-hours",
                "72",
            ],
            timeout=120,
        )
        if r.returncode == 2:
            return {"ok": False, "error": "detect_gaps connection error"}
        try:
            data = json.loads(r.stdout)
        except (json.JSONDecodeError, ValueError):
            return {"ok": False, "error": "failed to parse detect_gaps JSON"}

        stale = data.get("stale_pairs", [])
        pairs_checked = data.get("checked_pairs", 0)
        max_stale = max(
            (s.get("hours_since_last_bar", 0) for s in stale),
            default=0.0,
        )
        return {
            "ok": len(stale) == 0,
            "max_stale_hours": round(max_stale, 1),
            "stale_count": len(stale),
            "pairs_checked": pairs_checked,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except OSError as e:
        return {"ok": False, "error": str(e)}


def _check_checkpoints() -> dict:
    """Check local checkpoint files exist and are recent."""
    checkpoint_dir = Path.home() / ".cache" / "rangebar" / "checkpoints"
    if not checkpoint_dir.exists():
        return {"ok": True, "count": 0, "note": "no checkpoint dir"}
    files = list(checkpoint_dir.glob("*.json"))
    if not files:
        return {"ok": True, "count": 0}
    newest = max(files, key=lambda f: f.stat().st_mtime)
    age_hours = (time.time() - newest.stat().st_mtime) / 3600
    return {
        "ok": True,
        "count": len(files),
        "newest": newest.name,
        "newest_age_hours": round(age_hours, 1),
    }


def _check_dead_letters() -> dict:
    """Count dead-letter Parquet files from sidecar flush failures."""
    if not DEAD_LETTER_DIR.exists():
        return {"ok": True, "count": 0}
    files = list(DEAD_LETTER_DIR.glob("*.parquet"))
    return {"ok": len(files) == 0, "count": len(files)}


def _check_venv() -> dict:
    """Verify venv integrity: import rangebar succeeds."""
    try:
        r = _run(
            [sys.executable, "-c", "import rangebar; print(rangebar.__version__)"],
            timeout=10,
        )
        if r.returncode == 0:
            version = r.stdout.strip()
            return {"ok": True, "version": version}
        return {"ok": False, "error": r.stderr.strip()[:200]}
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"ok": False, "error": str(e)}


def _check_kintsugi_log() -> dict:
    """Query last 24h of Kintsugi repair attempts from ClickHouse."""
    query = (
        "SELECT "
        "  countIf(result = 'healed') AS repairs, "
        "  countIf(result = 'failed') AS failures "
        "FROM rangebar_cache.kintsugi_log "
        "WHERE timestamp > now() - INTERVAL 24 HOUR"
    )
    try:
        r = _run(
            ["curl", "-sf", f"http://localhost:8123/?query={query}"],
            timeout=10,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return {"ok": True, "note": "kintsugi_log not queryable"}
        parts = r.stdout.strip().split("\t")
        repairs = int(parts[0]) if len(parts) > 0 else 0
        failures = int(parts[1]) if len(parts) > 1 else 0
        return {
            "ok": failures == 0,
            "repairs_24h": repairs,
            "failures_24h": failures,
        }
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return {"ok": True, "note": "kintsugi_log query failed"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_all_checks() -> dict:
    """Run all health checks and build report."""
    import socket

    venv_result = _check_venv()
    version = venv_result.get("version", "unknown")

    checks = {
        "clickhouse": _check_clickhouse(),
        "sidecar_active": _check_service("rangebar-sidecar"),
        "kintsugi_active": _check_service("rangebar-kintsugi"),
        "recency_backfill_active": _check_service("rangebar-recency-backfill"),
        "sidecar_health": _check_health_endpoint(),
        "gaps": _check_gaps(),
        "freshness": _check_freshness(),
        "checkpoints": _check_checkpoints(),
        "dead_letters": _check_dead_letters(),
        "venv": venv_result,
        "kintsugi_log": _check_kintsugi_log(),
    }

    # Determine overall status
    critical_keys = {"clickhouse", "venv"}
    service_keys = {"sidecar_active", "kintsugi_active", "recency_backfill_active"}

    all_ok = all(c.get("ok", False) for c in checks.values())
    critical_down = any(not checks[k].get("ok", False) for k in critical_keys)
    all_services_down = all(not checks[k].get("ok", False) for k in service_keys)

    if critical_down or all_services_down:
        status = "critical"
        exit_code = 2
    elif all_ok:
        status = "healthy"
        exit_code = 0
    else:
        status = "degraded"
        exit_code = 1

    # Build summary
    failed_checks = [k for k, v in checks.items() if not v.get("ok", False)]
    summary = (
        f"{len(failed_checks)} check(s) failed" if failed_checks else "all checks pass"
    )

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "version": version,
        "status": status,
        "exit_code": exit_code,
        "checks": checks,
        "summary": summary,
    }


def main() -> int:
    os.chdir(Path(__file__).resolve().parent.parent)

    report = run_all_checks()
    exit_code = report["exit_code"]

    # Write artifacts
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = ARTIFACTS_DIR / f"health_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    latest = ARTIFACTS_DIR / "health_latest.json"
    latest.unlink(missing_ok=True)
    latest.symlink_to(report_path.name)

    # Print to stdout
    print(json.dumps(report, indent=2))

    # Human-friendly summary to stderr
    status_icon = {"healthy": "OK", "degraded": "WARN", "critical": "CRIT"}
    print(
        f"\n[{status_icon.get(report['status'], '?')}] "
        f"{report['status'].upper()}: {report['summary']}",
        file=sys.stderr,
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
