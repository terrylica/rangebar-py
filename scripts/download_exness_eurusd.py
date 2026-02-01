"""Download Exness EURUSD Raw_Spread tick data for pattern research.

Downloads monthly tick data from ticks.ex2archive.com.
Period: 2022-01 to 2026-01 (continuous multi-year)

# Issue #143: Download Exness EURUSD Raw_Spread tick data (2022-2026)

Usage:
    RANGEBAR_CH_HOSTS=bigblack uv run python scripts/download_exness_eurusd.py
"""

import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path


def log(level: str, message: str, **kwargs: object) -> None:
    """Log in NDJSON format."""
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry), flush=True)


def download_month(year: int, month: int, cache_dir: Path) -> tuple[bool, int]:
    """Download and extract a single month of tick data.

    Returns (success, file_size_bytes).
    """
    url = (
        f"https://ticks.ex2archive.com/ticks/EURUSD_Raw_Spread/{year:04d}/{month:02d}/"
        f"Exness_EURUSD_Raw_Spread_{year:04d}_{month:02d}.zip"
    )
    zip_path = cache_dir / f"Exness_EURUSD_Raw_Spread_{year:04d}_{month:02d}.zip"
    csv_path = cache_dir / f"Exness_EURUSD_Raw_Spread_{year:04d}_{month:02d}.csv"

    # Skip if already extracted
    if csv_path.exists():
        return True, csv_path.stat().st_size

    # Download zip
    if not zip_path.exists():
        result = subprocess.run(
            ["curl", "-sL", "-o", str(zip_path), url],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            log("ERROR", "Download failed", year=year, month=month, url=url)
            return False, 0

    # Extract
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)
        zip_path.unlink()  # Remove zip after extraction
        if csv_path.exists():
            return True, csv_path.stat().st_size
        log("ERROR", "Extraction failed - CSV not found", year=year, month=month)
        return False, 0
    except zipfile.BadZipFile:
        log("ERROR", "Bad zip file", year=year, month=month)
        zip_path.unlink()
        return False, 0


def main() -> None:
    """Download EURUSD Raw_Spread tick data 2022-2026."""
    cache_dir = Path.home() / "Library/Caches/rangebar/forex/EURUSD_Raw_Spread"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log("INFO", "Starting EURUSD Raw_Spread download", cache_dir=str(cache_dir))

    # Date range: 2022-01 to 2026-01
    months = []
    for year in range(2022, 2027):
        end_month = 1 if year == 2026 else 12
        for month in range(1, end_month + 1):
            months.append((year, month))

    log("INFO", "Download plan", total_months=len(months), start="2022-01", end="2026-01")

    success_count = 0
    total_size = 0
    for i, (year, month) in enumerate(months, 1):
        log("INFO", "Downloading", progress=f"{i}/{len(months)}", year=year, month=month)
        success, size = download_month(year, month, cache_dir)
        if success:
            success_count += 1
            total_size += size
            log("INFO", "Downloaded", year=year, month=month, size_mb=round(size / 1024 / 1024, 1))
        else:
            log("WARN", "Failed to download", year=year, month=month)

    log("INFO", "Download complete",
        success=success_count,
        failed=len(months) - success_count,
        total_size_gb=round(total_size / 1024 / 1024 / 1024, 2))


if __name__ == "__main__":
    main()
