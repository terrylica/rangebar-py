"""Checksum registry for Tier 1 Parquet cache.

Implements GitHub Issue #43: Track which cached Parquet files have verified checksums.

This module maintains a JSONL registry of checksum verifications, allowing:
- Audit of which cached files have been checksum-verified
- Detection of unverified dates that should be re-downloaded
- Correlation between raw downloads and cached data
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from datetime import date as dt_date
from pathlib import Path


@dataclass
class ChecksumRecord:
    """Record of checksum verification for a cached file."""

    symbol: str
    date: str
    file_path: str
    expected_hash: str
    actual_hash: str
    verified_at: str  # ISO8601 UTC
    data_source: str  # "binance", "exness", etc.

    @classmethod
    def create(
        cls,
        symbol: str,
        date: str,
        file_path: str,
        expected_hash: str,
        actual_hash: str,
        data_source: str = "binance",
    ) -> ChecksumRecord:
        """Create a new checksum record with current timestamp."""
        return cls(
            symbol=symbol,
            date=date,
            file_path=file_path,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
            verified_at=datetime.now(UTC).isoformat(),
            data_source=data_source,
        )


class ChecksumRegistry:
    """Registry tracking which Tier 1 cache files have verified checksums.

    The registry is stored as NDJSON (one JSON object per line) for:
    - Append-only writes (crash-safe)
    - Easy parsing with standard tools (jq, grep)
    - Streaming reads for large registries
    """

    def __init__(self, registry_path: Path | None = None) -> None:
        """Initialize the registry.

        Args:
            registry_path: Path to the registry file. If None, uses the default
                location in the project logs directory.
        """
        if registry_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            registry_path = project_root / "logs" / "checksum_registry.jsonl"
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def record_verification(self, record: ChecksumRecord) -> None:
        """Append verification record to registry.

        Args:
            record: ChecksumRecord to append
        """
        with self.registry_path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def is_verified(self, symbol: str, date: str) -> bool:
        """Check if a specific date's data has been checksum-verified.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            date: Date string (YYYY-MM-DD)

        Returns:
            True if the date has a verification record, False otherwise
        """
        if not self.registry_path.exists():
            return False

        with self.registry_path.open() as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record["symbol"] == symbol and record["date"] == date:
                        return True
                except json.JSONDecodeError:
                    continue
        return False

    def get_verification(self, symbol: str, date: str) -> ChecksumRecord | None:
        """Get the verification record for a specific date.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            date: Date string (YYYY-MM-DD)

        Returns:
            ChecksumRecord if found, None otherwise
        """
        if not self.registry_path.exists():
            return None

        with self.registry_path.open() as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data["symbol"] == symbol and data["date"] == date:
                        return ChecksumRecord(**data)
                except json.JSONDecodeError:
                    continue
        return None

    def get_unverified_dates(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> list[str]:
        """Find dates in range that lack checksum verification.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            start_date: Start date (YYYY-MM-DD, inclusive)
            end_date: End date (YYYY-MM-DD, inclusive)

        Returns:
            List of unverified date strings in YYYY-MM-DD format
        """
        verified = set()
        if self.registry_path.exists():
            with self.registry_path.open() as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record["symbol"] == symbol:
                            verified.add(record["date"])
                    except json.JSONDecodeError:
                        continue

        # Generate all dates in range
        start = dt_date.fromisoformat(start_date)
        end = dt_date.fromisoformat(end_date)
        all_dates = []
        current = start
        while current <= end:
            all_dates.append(current.isoformat())
            current += timedelta(days=1)

        return [d for d in all_dates if d not in verified]

    def get_verified_count(self, symbol: str) -> int:
        """Get the count of verified dates for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")

        Returns:
            Number of verified dates
        """
        if not self.registry_path.exists():
            return 0

        count = 0
        with self.registry_path.open() as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record["symbol"] == symbol:
                        count += 1
                except json.JSONDecodeError:
                    continue
        return count

    def audit_and_alert(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> None:
        """Audit cache and send Pushover alert if unverified files found.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        unverified = self.get_unverified_dates(symbol, start_date, end_date)
        if unverified:
            from ..notify.pushover import alert_tier1_cache_unverified

            start = dt_date.fromisoformat(start_date)
            end = dt_date.fromisoformat(end_date)
            total_count = (end - start).days + 1

            alert_tier1_cache_unverified(
                symbol=symbol,
                date_range=f"{start_date} to {end_date}",
                unverified_count=len(unverified),
                total_count=total_count,
            )
