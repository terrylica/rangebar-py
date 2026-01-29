"""SSH tunnel manager for remote ClickHouse hosts.

This module provides SSH tunnel management for connecting to ClickHouse
on remote GPU workstations when direct network access is not available.
"""

from __future__ import annotations

import socket
import subprocess
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType


def _find_free_port() -> int:
    """Find a free local port.

    Returns
    -------
    int
        Available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check if a port is open.

    Parameters
    ----------
    host : str
        Host to check
    port : int
        Port to check
    timeout : float
        Connection timeout

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


class SSHTunnel:
    """Manages SSH tunnel to remote ClickHouse host.

    Creates an SSH tunnel from a local port to the remote host's
    ClickHouse port (default 8123). Use as a context manager.

    Parameters
    ----------
    ssh_alias : str
        SSH alias from ~/.ssh/config
    remote_port : int
        Remote ClickHouse port (default: 8123)
    local_port : int | None
        Local port to use (default: auto-assign)

    Examples
    --------
    >>> with SSHTunnel("my-gpu-host") as local_port:
    ...     client = get_client("localhost", local_port)
    ...     # Use client...
    """

    def __init__(
        self,
        ssh_alias: str,
        remote_port: int = 8123,
        local_port: int | None = None,
    ) -> None:
        """Initialize tunnel configuration."""
        self.ssh_alias = ssh_alias
        self.remote_port = remote_port
        self._local_port = local_port
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def local_port(self) -> int | None:
        """Get the local port (assigned when tunnel starts)."""
        return self._local_port

    @property
    def is_active(self) -> bool:
        """Check if tunnel is active."""
        return (
            self._process is not None
            and self._process.poll() is None
            and self._local_port is not None
            and _is_port_open("localhost", self._local_port)
        )

    def start(self, timeout: float = 5.0) -> int:
        """Start the SSH tunnel.

        Parameters
        ----------
        timeout : float
            Maximum time to wait for tunnel to be ready

        Returns
        -------
        int
            Local port the tunnel is listening on

        Raises
        ------
        RuntimeError
            If tunnel fails to start
        """
        if self._process is not None:
            msg = "Tunnel already started"
            raise RuntimeError(msg)

        # Assign local port if not specified
        if self._local_port is None:
            self._local_port = _find_free_port()

        # Start SSH tunnel process
        self._process = subprocess.Popen(
            [
                "ssh",
                "-N",  # Don't execute remote command
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                "-L",
                f"{self._local_port}:localhost:{self.remote_port}",
                self.ssh_alias,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Wait for tunnel to be ready
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Check if port is open first - handles SSH ControlMaster case
            # where the ssh process exits immediately after handing off to master
            if _is_port_open("localhost", self._local_port):
                return self._local_port

            # Check if process died with error (non-zero exit)
            exit_code = self._process.poll()
            if exit_code is not None:
                # Exit code 0 with ControlMaster means forwarding was handed off
                # to master connection - give it a moment to activate
                if exit_code == 0:
                    time.sleep(0.2)
                    if _is_port_open("localhost", self._local_port):
                        # Forwarding via ControlMaster succeeded, but we don't
                        # own the process anymore - set to None so stop() is a no-op
                        self._process = None
                        return self._local_port
                # Non-zero exit or port still not open after ControlMaster handoff
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read().decode()
                msg = (
                    f"SSH tunnel to {self.ssh_alias} failed "
                    f"(exit={exit_code}): {stderr}"
                )
                self._process = None
                raise RuntimeError(msg)

            time.sleep(0.1)

        # Timeout - kill process
        self._cleanup()
        msg = f"SSH tunnel to {self.ssh_alias} timed out after {timeout}s"
        raise RuntimeError(msg)

    def stop(self) -> None:
        """Stop the SSH tunnel."""
        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up tunnel resources."""
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self._process.kill()
                    self._process.wait(timeout=1)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            finally:
                self._process = None

    def __enter__(self) -> int:
        """Start tunnel and return local port."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop tunnel on exit."""
        self.stop()

    def __del__(self) -> None:
        """Clean up on garbage collection."""
        self._cleanup()
