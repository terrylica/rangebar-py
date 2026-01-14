#!/usr/bin/env bash
# Show version from all sources
echo "=== VERSION SOURCES ==="
echo "Cargo.toml [workspace.package]: $(grep -A5 '\[workspace.package\]' Cargo.toml | grep '^version' | head -1 | sed 's/.*= "\(.*\)"/\1/')"
echo "Cargo.toml [package]:           $(grep -A10 '\[package\]' Cargo.toml | grep -E '^version' | head -1 | sed 's/.*= "\(.*\)"/\1/')"
echo "Python rangebar.__version__:    $(python -c 'import rangebar; print(rangebar.__version__)' 2>/dev/null || echo 'not installed')"
