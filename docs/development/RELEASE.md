# Release Workflow

**Parent**: [/CLAUDE.md](/CLAUDE.md)

This document describes the complete release workflow for rangebar-py.

---

## Overview

rangebar-py uses a **local-first release workflow**:

1. **Versioning**: `semantic-release` (Node.js) analyzes commits, bumps version
2. **Build**: `mise` tasks build wheels for macOS ARM64 + Linux x86_64
3. **Publish**: `uv publish` to PyPI using 1Password credentials

**Key principle**: Local releases are faster (30 sec) than GitHub Actions (3-5 min).

---

## Quick Commands

```bash
# Full release (recommended)
mise run release:full

# Individual phases
mise run release:preflight   # Verify clean state
mise run release:sync        # Pull/push main
mise run release:version     # semantic-release
mise run release:build-all   # Build all wheels
mise run publish             # Upload to PyPI
```

---

## Prerequisites

### One-Time Setup

```bash
# Install tools via mise (includes zig for cross-compilation)
mise install

# Authenticate GitHub (for semantic-release)
gh auth login

# Verify 1Password access (for PyPI token)
op item get djevteztvbcqgcm3yl4njkawjq --fields credential --reveal | head -c 10
```

### Environment Variables

Configured in `.mise.toml`:

```toml
[env]
GH_TOKEN = "{{ read_file(path=env.HOME ~ '/.claude/.secrets/gh-token-terrylica') }}"
GITHUB_TOKEN = "{{ read_file(path=env.HOME ~ '/.claude/.secrets/gh-token-terrylica') }}"
LINUX_BUILD_STRATEGY = "zig"  # "zig" or "remote"
LINUX_BUILD_HOST = "bigblack"
LINUX_BUILD_USER = "tca"
BUILD_LOCK_DIR = "/tmp/rangebar-py-locks"
```

---

## 4-Phase Workflow

### Phase 1: Preflight

```bash
mise run release:preflight
```

Validates:

- Working directory is clean
- On `main` branch
- Git remote is accessible

### Phase 2: Sync

```bash
mise run release:sync
```

Actions:

- `git pull --rebase origin main`
- `git push origin main`

### Phase 3: Version

```bash
mise run release:version
```

Actions:

- Runs `semantic-release --no-ci`
- Analyzes commits for version bump
- Updates `Cargo.toml` version (SSoT)
- Generates `CHANGELOG.md`
- Creates git tag
- Creates GitHub release

**Commit types**:

- `feat:` → MINOR bump
- `fix:` → PATCH bump
- `feat!:` or `BREAKING CHANGE:` → MAJOR bump

### Phase 4: Build

```bash
mise run release:build-all
```

Builds:

- macOS ARM64: `mise run release:macos-arm64` (~10 sec)
- Linux x86_64: `mise run release:linux` (~55 sec via zig)

Output in `dist/`:

```
rangebar-X.Y.Z-cp313-cp313-macosx_11_0_arm64.whl
rangebar-X.Y.Z-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
rangebar-X.Y.Z.tar.gz
```

### Phase 5: Publish

```bash
mise run publish
```

Uses `scripts/publish-wheels.sh`:

- Fetches `PYPI_TOKEN` from 1Password
- Uploads all wheels via `uv publish`
- Verifies on PyPI

---

## Linux Build: Zig Cross-Compilation

**Primary strategy**: Build Linux wheels directly from macOS using zig.

### Why Zig?

| Approach     | Time    | Dependencies      | Reliability |
| ------------ | ------- | ----------------- | ----------- |
| SSH + Docker | ~5 min  | Remote host, SSH  | Medium      |
| **Zig**      | ~55 sec | Local zig install | High        |

### Prerequisites

1. **rustls-tls**: Cargo.toml uses `rustls-tls` instead of `native-tls` (no OpenSSL)
2. **Cross-compile rustflags**: `.cargo/config.toml` removes `target-cpu=native`
3. **Linux target**: `rustup target add x86_64-unknown-linux-gnu`
4. **Zig**: Managed by mise (`zig = "0.13"` in `.mise.toml`)

### How It Works

```bash
mise run release:linux
```

1. Acquires build lock (prevents concurrent builds)
2. Clears `RUSTFLAGS` (avoids `target-cpu=native` pollution)
3. Runs: `maturin build --release --target x86_64-unknown-linux-gnu --zig --compatibility manylinux_2_17`
4. Copies wheel to `dist/`

### Manual Build

```bash
# Ensure clean RUSTFLAGS for cross-compile
RUSTFLAGS="" RUSTC_WRAPPER="" mise exec -- maturin build \
    --release \
    --target x86_64-unknown-linux-gnu \
    --zig \
    --compatibility manylinux_2_17 \
    -i python3.13
```

### Fallback: Remote Docker Build

If zig fails, use SSH + Docker:

```bash
LINUX_BUILD_STRATEGY=remote mise run release:linux
```

Process:

1. rsync project to remote host
2. Run Docker manylinux container
3. Build wheel inside container
4. scp wheel back to local `dist/`

---

## Build Locking

The `release:linux` task uses atomic locking to prevent stale processes:

- **Lock file**: `/tmp/rangebar-py-locks/release-linux.lock`
- **Lock dir**: `/tmp/rangebar-py-locks/release-linux.lock.d` (atomic mkdir)
- **Stale detection**: Locks older than 10 minutes are auto-removed

### Force Unlock

```bash
rm -rf /tmp/rangebar-py-locks/release-linux.lock*
```

---

## Key Configuration Files

| File                 | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| `.mise.toml`         | Tools (zig, rust, python) + all release tasks    |
| `.cargo/config.toml` | Cross-compile rustflags (no `target-cpu=native`) |
| `Cargo.toml`         | `rustls-tls` features, version SSoT              |
| `pyproject.toml`     | Maturin config, `dynamic = ["version"]`          |

### Critical Cargo.toml Settings

```toml
# HTTP/Network (rustls for cross-compilation - no OpenSSL dependency)
reqwest = { version = "0.12", default-features = false, features = ["stream", "rustls-tls"] }
tokio-tungstenite = { version = "0.23", default-features = false, features = ["connect", "rustls-tls-native-roots"] }
```

### Critical .cargo/config.toml Settings

```toml
[build]
# No target-cpu=native here (breaks cross-compile)
rustflags = ["-C", "opt-level=3", "-C", "strip=symbols"]

[target.x86_64-unknown-linux-gnu]
# Linux cross-compile: generic x86_64
rustflags = ["-C", "opt-level=3", "-C", "strip=symbols"]
```

---

## PyPI Credentials

Stored in 1Password:

| Item ID                      | Purpose        |
| ---------------------------- | -------------- |
| `djevteztvbcqgcm3yl4njkawjq` | PyPI API token |

### Manual Publish

```bash
PYPI_TOKEN=$(op item get djevteztvbcqgcm3yl4njkawjq --fields credential --reveal) && \
uv publish dist/rangebar-*.whl --token "$PYPI_TOKEN"
```

### Rotate Token

```bash
# Create new token at https://pypi.org/manage/account/token/
# Update in 1Password
```

---

## Troubleshooting

### "No releasable commits"

semantic-release found no `feat:` or `fix:` commits since last tag.

**Fix**: Use conventional commit format.

### "Another Linux build is running"

Build lock held by previous invocation.

**Fix**: `rm -rf /tmp/rangebar-py-locks/release-linux.lock*`

### "target-cpu=native" LLVM error

RUSTFLAGS from `~/.cargo/config.toml` polluting cross-compile.

**Fix**: Ensure `.cargo/config.toml` in project root overrides global config, or use `RUSTFLAGS=""`.

### "OpenSSL not found" during cross-compile

Using `native-tls` instead of `rustls-tls`.

**Fix**: Verify `Cargo.toml` uses `rustls-tls` features for `reqwest` and `tokio-tungstenite`.

### "Linux build failed" (remote fallback)

SSH or Docker issue on remote host.

**Debug**:

```bash
mise run release:linux-preflight
ssh bigblack "docker ps"
```

### "PyPI 403 Forbidden"

Token expired or invalid.

**Fix**: Verify token in 1Password, rotate if needed.

---

## Related

- [/CLAUDE.md](/CLAUDE.md) - Project hub
- [/.mise.toml](/.mise.toml) - All release tasks
- [/.cargo/config.toml](/.cargo/config.toml) - Cross-compile rustflags
- [/scripts/publish-wheels.sh](/scripts/publish-wheels.sh) - PyPI publisher
- Skill: `itp:semantic-release` - Versioning details
