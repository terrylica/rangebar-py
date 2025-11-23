#!/usr/bin/env python3
"""Non-interactive dashboard content validation.

Validates the deployed performance dashboard at terrylica.github.io/rangebar-py/
without requiring a browser. Checks HTML structure, benchmark data validity,
and ensures all expected metrics are present.

Exit codes:
    0: All validations passed
    1: Validation failure (details in stderr)
"""

import json
import re
import sys
import urllib.request
from datetime import datetime
from typing import Any, Dict, List


class DashboardValidator:
    """Validates performance dashboard deployment."""

    BASE_URL = "https://terrylica.github.io/rangebar-py"

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.checks_passed = 0
        self.checks_failed = 0

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("=" * 80)
        print("Performance Dashboard Validation Suite")
        print("=" * 80)
        print()

        self.validate_main_dashboard()
        self.validate_benchmark_data()
        self.validate_github_action_dashboard()

        self.print_summary()
        return len(self.errors) == 0

    def fetch_url(self, url: str) -> str:
        """Fetch URL content with error handling."""
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                content = response.read().decode('utf-8')
                status = response.getcode()

                if status != 200:
                    self.fail(f"HTTP {status} for {url}")
                    return ""

                self.check(f"✓ Fetched {url}")
                return content
        except Exception as e:
            self.fail(f"Failed to fetch {url}: {e}")
            return ""

    def validate_main_dashboard(self):
        """Validate custom Chart.js dashboard."""
        print("Validating Main Dashboard")
        print("-" * 80)

        url = f"{self.BASE_URL}/"
        html = self.fetch_url(url)

        if not html:
            return

        # Check HTML structure
        required_elements = [
            (r"<title>rangebar-py Performance Dashboard</title>", "Page title"),
            (r"chart\.js", "Chart.js library reference"),
            (r"rangebar-py Performance Dashboard", "Dashboard heading"),
            (r"Throughput Trends", "Throughput chart section"),
            (r"Memory Usage Trends", "Memory chart section"),
            (r"dev/bench/data\.js", "Benchmark data reference"),
        ]

        for pattern, description in required_elements:
            if re.search(pattern, html):
                self.check(f"✓ Found {description}")
            else:
                self.fail(f"Missing {description}")

        print()

    def validate_benchmark_data(self):
        """Validate benchmark data structure and content."""
        print("Validating Benchmark Data")
        print("-" * 80)

        url = f"{self.BASE_URL}/dev/bench/data.js"
        js_content = self.fetch_url(url)

        if not js_content:
            return

        # Extract JSON from JavaScript wrapper
        # Remove the JavaScript assignment wrapper
        json_str = re.sub(r'^\s*window\.BENCHMARK_DATA\s*=\s*', '', js_content)
        json_str = re.sub(r'\s*;?\s*$', '', json_str)

        try:
            data = json.loads(json_str)
            self.check("✓ Parsed benchmark data JSON")
        except json.JSONDecodeError as e:
            self.fail(f"Invalid JSON in data.js: {e}")
            return

        # Validate structure
        self.validate_data_structure(data)

        print()

    def validate_data_structure(self, data: Dict[str, Any]):
        """Validate benchmark data structure."""
        # Check top-level fields
        required_fields = ["lastUpdate", "repoUrl", "entries"]
        for field in required_fields:
            if field in data:
                self.check(f"✓ Field present: {field}")
            else:
                self.fail(f"Missing field: {field}")
                return

        # Check timestamp
        try:
            timestamp = data["lastUpdate"]
            dt = datetime.fromtimestamp(timestamp / 1000)
            self.check(f"✓ Last update: {dt.isoformat()}")
        except Exception as e:
            self.fail(f"Invalid timestamp: {e}")

        # Check repo URL
        if data["repoUrl"] == "https://github.com/terrylica/rangebar-py":
            self.check("✓ Repository URL correct")
        else:
            self.fail(f"Unexpected repo URL: {data['repoUrl']}")

        # Check entries
        entries = data.get("entries", {})
        if "Python API Benchmarks" not in entries:
            self.fail("Missing 'Python API Benchmarks' entry")
            return

        self.check("✓ Found Python API Benchmarks entry")

        # Validate benchmark entries
        benchmarks = entries["Python API Benchmarks"]
        if not benchmarks:
            self.fail("No benchmark data points")
            return

        self.check(f"✓ Found {len(benchmarks)} benchmark runs")

        # Validate latest benchmark
        latest = benchmarks[-1]
        self.validate_benchmark_entry(latest)

    def validate_benchmark_entry(self, entry: Dict[str, Any]):
        """Validate a single benchmark entry."""
        # Check commit metadata
        if "commit" not in entry:
            self.fail("Missing commit metadata")
            return

        commit = entry["commit"]
        required_commit_fields = ["id", "message", "timestamp", "url"]
        for field in required_commit_fields:
            if field in commit:
                self.check(f"✓ Commit {field} present")
            else:
                self.fail(f"Missing commit {field}")

        # Check benchmarks
        if "benches" not in entry:
            self.fail("Missing benches array")
            return

        benches = entry["benches"]
        self.check(f"✓ Found {len(benches)} benchmark tests")

        # Expected benchmark tests
        expected_tests = [
            "test_throughput_1k_trades",
            "test_throughput_100k_trades",
            "test_throughput_1m_trades",
        ]

        found_tests = {b["name"].split("::")[-1] for b in benches}

        for test in expected_tests:
            if test in found_tests:
                self.check(f"✓ Found benchmark: {test}")
            else:
                self.warn(f"Missing benchmark: {test}")

        # Validate benchmark metrics
        for bench in benches:
            self.validate_benchmark_metrics(bench)

    def validate_benchmark_metrics(self, bench: Dict[str, Any]):
        """Validate individual benchmark metrics."""
        name = bench.get("name", "unknown").split("::")[-1]

        required_fields = ["name", "value", "unit"]
        for field in required_fields:
            if field not in bench:
                self.fail(f"Benchmark {name} missing {field}")
                return

        # Check value is numeric and reasonable
        value = bench["value"]
        if not isinstance(value, (int, float)):
            self.fail(f"Benchmark {name} has non-numeric value: {value}")
            return

        if value <= 0:
            self.fail(f"Benchmark {name} has non-positive value: {value}")
            return

        # Check unit
        unit = bench["unit"]
        if unit not in ["iter/sec", "ops/sec", "ns/iter", "us/iter", "ms/iter", "s/iter"]:
            self.warn(f"Benchmark {name} has unexpected unit: {unit}")

    def validate_github_action_dashboard(self):
        """Validate github-action-benchmark dashboard."""
        print("Validating GitHub Action Benchmark Dashboard")
        print("-" * 80)

        url = f"{self.BASE_URL}/dev/bench/index.html"
        html = self.fetch_url(url)

        if not html:
            return

        # Check for github-action-benchmark elements
        required_elements = [
            (r"<title>Benchmarks</title>", "Page title"),
            (r"github-action-benchmark", "github-action-benchmark reference"),
            (r"window\.BENCHMARK_DATA", "Benchmark data reference"),
        ]

        for pattern, description in required_elements:
            if re.search(pattern, html):
                self.check(f"✓ Found {description}")
            else:
                self.fail(f"Missing {description}")

        print()

    def check(self, message: str):
        """Record a passed check."""
        print(f"  {message}")
        self.checks_passed += 1

    def fail(self, message: str):
        """Record a failed check."""
        print(f"  ✗ {message}", file=sys.stderr)
        self.errors.append(message)
        self.checks_failed += 1

    def warn(self, message: str):
        """Record a warning."""
        print(f"  ⚠ {message}")
        self.warnings.append(message)

    def print_summary(self):
        """Print validation summary."""
        print("=" * 80)
        print("Validation Summary")
        print("=" * 80)
        print(f"Checks passed: {self.checks_passed}")
        print(f"Checks failed: {self.checks_failed}")
        print(f"Warnings: {len(self.warnings)}")
        print()

        if self.errors:
            print("FAILED - Errors encountered:")
            for error in self.errors:
                print(f"  - {error}")
            print()
            sys.exit(1)

        if self.warnings:
            print("PASSED (with warnings):")
            for warning in self.warnings:
                print(f"  - {warning}")
            print()
        else:
            print("✅ ALL VALIDATIONS PASSED")
            print()


if __name__ == "__main__":
    validator = DashboardValidator()
    success = validator.validate_all()
    sys.exit(0 if success else 1)
