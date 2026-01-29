"""Preflight detection for ClickHouse availability.

This module provides comprehensive detection of ClickHouse installation state
and host selection logic for multi-GPU workstation environments.

The preflight system detects:
- Level -1: ENV not configured (no RANGEBAR_CH_HOSTS and localhost unavailable)
- Level 0: ClickHouse not installed
- Level 1: ClickHouse installed but not running
- Level 2: ClickHouse running but schema missing
- Level 3: Fully configured and ready

Host selection priority:
1. localhost:8123 (prefer local GPU workstation)
2. RANGEBAR_CH_HOSTS via direct network (if exposed)
3. RANGEBAR_CH_HOSTS via SSH tunnel (if firewalled)
4. FAIL LOUDLY with Claude Code CLI guidance
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import urllib.request
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Literal

from .config import ConnectionMode, get_connection_mode

# Loud warning message for Claude Code CLI users
ENV_NOT_CONFIGURED_WARNING = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  RANGEBAR CLICKHOUSE CACHE - CONFIGURATION REQUIRED                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  No ClickHouse host configured and localhost:8123 unavailable.               ║
║                                                                              ║
║  FOR CLAUDE CODE CLI USERS:                                                  ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  1. Check your ~/.ssh/config for GPU workstations with ClickHouse            ║
║     (Look for hosts used for ML training that have ClickHouse installed)     ║
║                                                                              ║
║  2. Configure mise environment variables:                                    ║
║                                                                              ║
║     # Add to ~/.config/mise/config.toml or project mise.toml:                ║
║     [env]                                                                    ║
║     RANGEBAR_CH_HOSTS = "host1,host2,host3"  # SSH aliases from your config  ║
║     RANGEBAR_CH_PRIMARY = "host1"            # Default host                  ║
║                                                                              ║
║  3. Reload mise: mise trust && mise install                                  ║
║                                                                              ║
║  4. Re-run your command                                                      ║
║                                                                              ║
║  ALTERNATIVE - Local ClickHouse:                                             ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  macOS:  brew install --cask clickhouse                                      ║
║  Linux:  curl https://clickhouse.com/ | sh && ./clickhouse install           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


class ClickHouseNotConfiguredError(RuntimeError):
    """Raised when no ClickHouse hosts configured and localhost unavailable.

    This error includes detailed guidance for Claude Code CLI users
    on how to configure their environment.
    """

    def __init__(self, message: str | None = None) -> None:
        """Initialize with optional custom message."""
        super().__init__(message or ENV_NOT_CONFIGURED_WARNING)


class InstallationLevel(IntEnum):
    """ClickHouse installation state levels."""

    ENV_NOT_CONFIGURED = -1
    NOT_INSTALLED = 0
    INSTALLED_NOT_RUNNING = 1
    RUNNING_NO_SCHEMA = 2
    FULLY_CONFIGURED = 3


@dataclass
class PreflightResult:
    """Result of preflight detection.

    Attributes
    ----------
    level : InstallationLevel
        Current installation state
    version : str | None
        ClickHouse version if running
    binary_path : str | None
        Path to ClickHouse binary if found
    message : str
        Human-readable status message
    action_required : str | None
        Guidance for fixing issues
    """

    level: InstallationLevel
    version: str | None = None
    binary_path: str | None = None
    message: str = ""
    action_required: str | None = None


@dataclass
class HostConnection:
    """Connection details for a ClickHouse host.

    Attributes
    ----------
    host : str
        Host identifier (SSH alias or 'localhost')
    method : Literal["local", "direct", "ssh_tunnel"]
        Connection method
    port : int
        Port to connect to
    """

    host: str
    method: Literal["local", "direct", "ssh_tunnel"]
    port: int = 8123


def detect_clickhouse_state() -> PreflightResult:
    """Detect ClickHouse installation state on localhost.

    Returns
    -------
    PreflightResult
        Detection result with level, version, and guidance

    Examples
    --------
    >>> result = detect_clickhouse_state()
    >>> if result.level >= InstallationLevel.RUNNING_NO_SCHEMA:
    ...     print("ClickHouse is running")
    """
    # Level 0: Check for binary
    binary = _find_clickhouse_binary()
    if binary is None:
        return PreflightResult(
            level=InstallationLevel.NOT_INSTALLED,
            message="ClickHouse not installed",
            action_required=(
                "Install ClickHouse:\n"
                "  macOS: brew install --cask clickhouse\n"
                "  Linux: curl https://clickhouse.com/ | sh"
            ),
        )

    # Level 1: Check if server is running
    if not _is_port_open("localhost", 8123):
        return PreflightResult(
            level=InstallationLevel.INSTALLED_NOT_RUNNING,
            binary_path=binary,
            message=f"ClickHouse binary at {binary}, but server not running",
            action_required=(
                f"Start ClickHouse server:\n"
                f"  {binary} server --daemon\n"
                f"  OR: systemctl start clickhouse-server"
            ),
        )

    # Level 2: Check for schema
    version = _get_clickhouse_version()
    if not _has_rangebar_schema():
        return PreflightResult(
            level=InstallationLevel.RUNNING_NO_SCHEMA,
            binary_path=binary,
            version=version,
            message=f"ClickHouse {version} running, but rangebar_cache schema missing",
            action_required="Schema will be auto-created on first use",
        )

    # Level 3: Fully configured
    return PreflightResult(
        level=InstallationLevel.FULLY_CONFIGURED,
        binary_path=binary,
        version=version,
        message=f"ClickHouse {version} ready with rangebar_cache schema",
    )


def get_available_clickhouse_host() -> HostConnection:
    """Find best available ClickHouse host based on connection mode.

    Connection Mode (via RANGEBAR_MODE env var):
    - LOCAL: Force localhost:8123 only
    - CLOUD: Require CLICKHOUSE_HOST env var (remote only)
    - AUTO: Try localhost first, then SSH aliases (default)

    Priority Order (AUTO mode):
    1. LOCAL: localhost:8123 (if running on a GPU workstation with ClickHouse)
    2. DIRECT: Remote host:8123 over network (if exposed)
    3. SSH_TUNNEL: SSH to remote host, access localhost:8123 there

    This prefers local execution - if you're running on a GPU workstation
    that has ClickHouse, use it directly without network overhead.

    Returns
    -------
    HostConnection
        Connection details for available host

    Raises
    ------
    ClickHouseNotConfiguredError
        If no hosts available (FAIL LOUDLY with guidance)

    Examples
    --------
    >>> # Default AUTO mode
    >>> host = get_available_clickhouse_host()
    >>> print(f"Using {host.host} via {host.method}")

    >>> # Force local mode
    >>> import os
    >>> os.environ["RANGEBAR_MODE"] = "local"
    >>> host = get_available_clickhouse_host()  # Only checks localhost
    """
    mode = get_connection_mode()

    # LOCAL mode: Force localhost only
    if mode == ConnectionMode.LOCAL:
        if _is_port_open("localhost", 8123):
            return HostConnection(host="localhost", method="local")
        msg = (
            "RANGEBAR_MODE=local but ClickHouse not running on localhost:8123.\n"
            "Start ClickHouse locally or change mode to 'auto' or 'cloud'."
        )
        raise ClickHouseNotConfiguredError(msg)

    # CLOUD mode: Require remote host only
    if mode == ConnectionMode.CLOUD:
        cloud_host = os.getenv("CLICKHOUSE_HOST")
        if not cloud_host:
            msg = (
                "RANGEBAR_MODE=cloud but CLICKHOUSE_HOST not set.\n"
                "Set CLICKHOUSE_HOST to your ClickHouse server address."
            )
            raise ClickHouseNotConfiguredError(msg)
        cloud_port = int(os.getenv("CLICKHOUSE_PORT", "8123"))
        if _is_port_open(cloud_host, cloud_port):
            return HostConnection(host=cloud_host, method="direct", port=cloud_port)
        msg = (
            f"RANGEBAR_MODE=cloud but cannot connect to {cloud_host}:{cloud_port}.\n"
            "Verify CLICKHOUSE_HOST and CLICKHOUSE_PORT are correct."
        )
        raise ClickHouseNotConfiguredError(msg)

    # AUTO mode: Try localhost first, then remote hosts
    # PRIORITY 1: Always check localhost first (prefer local execution)
    # NOTE: Use _verify_clickhouse() not _is_port_open() to verify
    # it's actually ClickHouse, not another service (e.g., Chrome extension)
    if _verify_clickhouse("localhost", 8123):
        return HostConnection(host="localhost", method="local")

    # PRIORITY 2-3: Try configured remote hosts
    hosts_csv = os.getenv("RANGEBAR_CH_HOSTS", "")
    primary = os.getenv("RANGEBAR_CH_PRIMARY", "")

    # Build host list with primary first
    hosts: list[str] = []
    if primary:
        hosts.append(primary)
    for host in hosts_csv.split(","):
        host = host.strip()
        if host and host not in hosts:
            hosts.append(host)

    for host in hosts:
        # Try direct connection first (faster if ClickHouse is exposed)
        if _is_direct_available(host):
            return HostConnection(host=host, method="direct")

        # Fall back to SSH tunnel (works with firewalled hosts)
        if _is_ssh_available(host):
            return HostConnection(host=host, method="ssh_tunnel")

    # FAIL LOUDLY - no hosts available
    raise ClickHouseNotConfiguredError()


def _find_clickhouse_binary() -> str | None:
    """Search for ClickHouse binary in common locations.

    Returns
    -------
    str | None
        Path to binary, or None if not found
    """
    # Check PATH
    if binary := shutil.which("clickhouse-server"):
        return binary
    if binary := shutil.which("clickhouse"):
        return binary

    # Check common locations
    common_paths = [
        "/usr/bin/clickhouse",
        "/usr/local/bin/clickhouse",
        Path.home() / ".local/bin/clickhouse",
        "/opt/homebrew/bin/clickhouse",
    ]

    for path in common_paths:
        path = Path(path)
        if path.exists():
            return str(path)

    return None


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if TCP port is open.

    Parameters
    ----------
    host : str
        Host to check
    port : int
        Port to check
    timeout : float
        Connection timeout in seconds

    Returns
    -------
    bool
        True if port is open
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


def _is_direct_available(host: str, port: int = 8123) -> bool:
    """Check if ClickHouse is directly accessible over network.

    Parameters
    ----------
    host : str
        SSH alias to resolve
    port : int
        Port to check

    Returns
    -------
    bool
        True if ClickHouse is accessible
    """
    try:
        # Resolve SSH alias to IP if needed
        ip = _resolve_ssh_alias_to_ip(host)
        if ip and _is_port_open(ip, port, timeout=2.0):
            # Verify it's actually ClickHouse responding
            return _verify_clickhouse(ip, port)
    except (OSError, TimeoutError, urllib.error.URLError):
        # Network/connection errors are expected during preflight probing
        pass
    return False


def _is_ssh_available(ssh_alias: str) -> bool:
    """Check if ClickHouse is available via SSH tunnel.

    Parameters
    ----------
    ssh_alias : str
        SSH alias from ~/.ssh/config

    Returns
    -------
    bool
        True if ClickHouse is accessible via SSH
    """
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=3",
                "-o",
                "BatchMode=yes",
                ssh_alias,
                "curl -s http://localhost:8123/ -d 'SELECT 1'",
            ],
            capture_output=True,
            timeout=8,
            check=False,
        )
        return result.returncode == 0 and b"1" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _resolve_ssh_alias_to_ip(ssh_alias: str) -> str | None:
    """Resolve SSH alias to IP address from ~/.ssh/config.

    Parameters
    ----------
    ssh_alias : str
        SSH alias to resolve

    Returns
    -------
    str | None
        Resolved hostname/IP, or None if not found
    """
    try:
        result = subprocess.run(
            ["ssh", "-G", ssh_alias],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        for line in result.stdout.splitlines():
            if line.startswith("hostname "):
                return line.split()[1]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _verify_clickhouse(host: str, port: int) -> bool:
    """Verify ClickHouse is responding at host:port.

    Parameters
    ----------
    host : str
        Host to verify
    port : int
        Port to verify

    Returns
    -------
    bool
        True if ClickHouse responds correctly
    """
    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/",
            data=b"SELECT 1",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.read().strip() == b"1"
    except (OSError, TimeoutError, urllib.error.URLError):
        # Network/connection errors expected during preflight probing
        return False


def _get_clickhouse_version() -> str | None:
    """Get ClickHouse version from running server.

    Returns
    -------
    str | None
        Version string, or None if unavailable
    """
    try:
        req = urllib.request.Request(
            "http://localhost:8123/",
            data=b"SELECT version()",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.read().decode().strip()
    except (OSError, TimeoutError, urllib.error.URLError):
        # Network/connection errors expected during preflight probing
        return None


def _has_rangebar_schema() -> bool:
    """Check if rangebar_cache database exists.

    Returns
    -------
    bool
        True if database exists
    """
    try:
        req = urllib.request.Request(
            "http://localhost:8123/",
            data=b"SHOW DATABASES LIKE 'rangebar_cache'",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return bool(resp.read().strip())
    except (OSError, TimeoutError, urllib.error.URLError):
        # Network/connection errors expected during preflight probing
        return False
