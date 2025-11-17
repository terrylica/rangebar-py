# Makefile for rangebar-py quality tooling
# ADR-003: Testing Strategy with Real Binance Data

.PHONY: help test coverage lint format check benchmark clean install dev

# Default target
help:
	@echo "rangebar-py quality tooling commands:"
	@echo ""
	@echo "  make install      Install package in editable mode"
	@echo "  make dev          Install development dependencies"
	@echo "  make test         Run all tests (excluding slow tests)"
	@echo "  make test-all     Run all tests (including slow tests)"
	@echo "  make coverage     Run tests with coverage report (target: 95% Python, 90% Rust)"
	@echo "  make lint         Run all linters (ruff, mypy, black check, clippy)"
	@echo "  make format       Auto-format code (black, ruff --fix)"
	@echo "  make check        Combined check (lint + test)"
	@echo "  make benchmark    Run performance benchmarks"
	@echo "  make clean        Remove build artifacts and caches"
	@echo ""

# Install package in editable mode
install:
	maturin develop

# Install development dependencies
dev:
	uv pip install --python .venv -e ".[dev]"
	uv pip install --python .venv pytest-cov psutil

# Run tests (exclude slow tests by default)
test:
	pytest tests/ -v -m "not slow"

# Run all tests including slow ones
test-all:
	pytest tests/ -v

# Run Rust tests
test-rust:
	cargo test

# Run tests with coverage
coverage:
	@echo "=== Python Coverage ==="
	pytest tests/ -v --cov=python/rangebar --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report: htmlcov/index.html"
	@echo ""
	@echo "=== Rust Coverage ==="
	cargo llvm-cov --html --open
	@echo ""
	@echo "Rust coverage report: target/llvm-cov/html/index.html"

# Run all linters
lint: lint-python lint-rust

# Python linting
lint-python:
	@echo "=== Running ruff ==="
	ruff check python/ tests/
	@echo ""
	@echo "=== Running mypy ==="
	mypy python/rangebar/
	@echo ""
	@echo "=== Running black (check only) ==="
	black python/ tests/ --check

# Rust linting
lint-rust:
	@echo "=== Running clippy ==="
	cargo clippy -- -D warnings

# Auto-format code
format: format-python format-rust

# Format Python code
format-python:
	@echo "=== Formatting with black ==="
	black python/ tests/
	@echo ""
	@echo "=== Auto-fixing with ruff ==="
	ruff check python/ tests/ --fix

# Format Rust code
format-rust:
	@echo "=== Formatting with rustfmt ==="
	cargo fmt

# Combined check (lint + test)
check: lint test

# Run performance benchmarks
benchmark:
	@echo "=== Running performance benchmarks ==="
	pytest tests/test_performance.py -v --benchmark-only
	@echo ""
	@echo "Targets: >1M trades/sec throughput, <100MB memory for 1M trades"

# Clean build artifacts
clean:
	rm -rf target/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf python/rangebar/__pycache__/
	rm -rf tests/__pycache__/
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
