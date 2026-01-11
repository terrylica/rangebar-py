# Release Workflow

**Parent**: [/CLAUDE.md](/CLAUDE.md)

This document describes the complete release workflow for rangebar-py.

---

## Overview

rangebar-py uses a **local-first release workflow**:

1. **Versioning**: `semantic-release` (Node.js) analyzes commits, bumps version
2. **Build**: `mise` tasks build wheels for macOS ARM64 + Linux x86_64
3. **Publish**: `uv publish` to PyPI using Doppler credentials

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
# Install tools via mise
mise install

# Authenticate GitHub (for semantic-release)
gh auth login

# Authenticate Doppler (for PyPI token)
doppler login
```

### Environment Variables

Configured in `.mise.toml`:

```toml
[env]
GH_TOKEN = "{{ read_file(path=env.HOME ~ '/.claude/.secrets/gh-token-terrylica') }}"
GITHUB_TOKEN = "{{ read_file(path=env.HOME ~ '/.claude/.secrets/gh-token-terrylica') }}"
LINUX_BUILD_HOST = "bigblack"
LINUX_BUILD_USER = "tca"
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
- Updates `pyproject.toml` version
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
- macOS ARM64: `mise run release:macos-arm64`
- Linux x86_64: `mise run release:linux` (via SSH to build host)

Output in `dist/`:
```
rangebar-X.Y.Z-cp39-abi3-macosx_11_0_arm64.whl
rangebar-X.Y.Z-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
rangebar-X.Y.Z.tar.gz
```

### Phase 5: Publish

```bash
mise run publish
```

Uses `scripts/publish-wheels.sh`:
- Fetches `PYPI_TOKEN` from Doppler
- Uploads all wheels via `uv publish`
- Verifies on PyPI

---

## Linux Build Setup

The Linux wheel is built on a remote host via SSH + Docker manylinux.

### Requirements

- SSH access to `LINUX_BUILD_HOST`
- Docker installed on remote
- Rust toolchain on remote

### Verify Connectivity

```bash
mise run release:linux-preflight
```

### Manual Build

```bash
mise run release:linux
```

Process:
1. rsync project to remote
2. Run Docker manylinux container
3. Build wheel inside container
4. scp wheel back to local `dist/`

---

## PyPI Credentials

Stored in Doppler (`claude-config` project, `prd` config):

| Secret | Purpose |
|--------|---------|
| `PYPI_TOKEN` | Production PyPI upload |
| `TESTPYPI_TOKEN` | Test PyPI (optional) |

### Rotate Token

```bash
# Create new token at https://pypi.org/manage/account/token/
doppler secrets set PYPI_TOKEN='pypi-...' --project claude-config --config prd
```

---

## Troubleshooting

### "No releasable commits"

semantic-release found no `feat:` or `fix:` commits since last tag.

**Fix**: Use conventional commit format.

### "Linux build failed"

SSH or Docker issue on remote host.

**Debug**:
```bash
mise run release:linux-preflight
ssh bigblack "docker ps"
```

### "PyPI 403 Forbidden"

Token expired or invalid.

**Fix**: Rotate token in Doppler.

### "Version not updated"

Forgot to pull after semantic-release.

**Fix**: `git pull origin main` before publish.

---

## Related

- [/CLAUDE.md](/CLAUDE.md) - Project hub
- [/.mise.toml](/.mise.toml) - All release tasks
- [/scripts/publish-wheels.sh](/scripts/publish-wheels.sh) - PyPI publisher
- Skill: `itp:semantic-release` - Versioning details
- Skill: `itp:pypi-doppler` - PyPI credential management
