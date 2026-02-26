# FILE-SIZE-OK: health check probes are cohesive
"""Health check implementations for rangebar-py.

GitHub Issue: https://github.com/terrylica/rangebar-py/issues/109
"""

from __future__ import annotations

import logging
import subprocess

import clickhouse_connect
import psutil
from clickhouse_connect.common import ConnectError
from clickhouse_connect.driver.exceptions import DatabaseError

logger = logging.getLogger(__name__)


def check_clickhouse(host: str = "localhost", port: int = 8123) -> bool:
    """Check ClickHouse connectivity.

    Parameters
    ----------
    host : str
        ClickHouse host address.
    port : int
        ClickHouse HTTP port.

    Returns
    -------
    bool
        True if ClickHouse is reachable and responds.
    """
    try:
        client = clickhouse_connect.get_client(host=host, port=port)
        result = client.query("SELECT 1")
    except (ConnectError, DatabaseError, OSError):
        logger.warning("ClickHouse health check failed", exc_info=True)
        return False
    else:
        return result.result_rows[0][0] == 1


def check_memory(threshold_mb: int = 2048) -> bool:
    """Check memory usage is below threshold.

    Parameters
    ----------
    threshold_mb : int
        Maximum allowed RSS in MB.

    Returns
    -------
    bool
        True if memory usage is below threshold.
    """
    try:
        process = psutil.Process()
        rss_mb = process.memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        logger.warning("Memory health check failed", exc_info=True)
        return False
    else:
        return rss_mb < threshold_mb


def check_disk_space(min_free_gb: float = 10.0) -> bool:
    """Check minimum free disk space.

    Parameters
    ----------
    min_free_gb : float
        Minimum required free space in GB.

    Returns
    -------
    bool
        True if sufficient free space exists.
    """
    try:
        usage = psutil.disk_usage("/")
        free_gb = usage.free / (1024**3)
    except OSError:
        logger.warning("Disk space health check failed", exc_info=True)
        return False
    else:
        return free_gb > min_free_gb


def check_websocket(service_name: str = "rangebar-sidecar") -> bool:
    """Check if the rangebar-sidecar service is running.

    Parameters
    ----------
    service_name : str
        Name of the systemd user service to check.

    Returns
    -------
    bool
        True if the service is active.
    """
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", service_name],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError):
        logger.warning("systemctl not available, checking process")
        return check_process_running(service_name)
    except subprocess.TimeoutExpired:
        logger.warning("WebSocket health check timeout", exc_info=True)
        return False
    else:
        is_active = (
            result.returncode == 0
            and result.stdout.strip() == "active"
        )
        if not is_active:
            logger.warning(
                "WebSocket service '%s' is not active: %s",
                service_name,
                result.stdout.strip(),
            )
        return is_active


def check_process_running(process_name: str) -> bool:
    """Check if a process is running by name.

    Parameters
    ----------
    process_name : str
        Name of the process to check.

    Returns
    -------
    bool
        True if process is found running.
    """
    try:
        found = any(
            process_name in proc.info["name"]
            for proc in psutil.process_iter(["name"])
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        logger.warning("Process check failed", exc_info=True)
        return False
    else:
        return found


def run_all_checks(
    ch_host: str = "localhost",
    ch_port: int = 8123,
    memory_threshold_mb: int = 2048,
    min_free_gb: float = 10.0,
    service_name: str = "rangebar-sidecar",
) -> dict[str, bool]:
    """Run all health checks and return results.

    Parameters
    ----------
    ch_host : str
        ClickHouse host.
    ch_port : int
        ClickHouse port.
    memory_threshold_mb : int
        Memory threshold in MB.
    min_free_gb : float
        Minimum free disk space in GB.
    service_name : str
        WebSocket service name.

    Returns
    -------
    dict[str, bool]
        Dictionary mapping check names to their results.
    """
    return {
        "clickhouse": check_clickhouse(ch_host, ch_port),
        "memory": check_memory(memory_threshold_mb),
        "disk_space": check_disk_space(min_free_gb),
        "websocket": check_websocket(service_name),
    }
