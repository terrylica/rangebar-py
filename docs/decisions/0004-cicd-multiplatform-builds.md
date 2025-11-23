# ADR-004: CI/CD Multi-Platform Wheel Building

**Status**: Accepted

**Date**: 2024-11-16

**Decision Makers**: Engineering Team

---

## Context

rangebar-py is a PyO3/maturin Python package wrapping Rust code. Users expect `pip install rangebar` to work without requiring Rust toolchain installation. This requires pre-built binary wheels for common platforms.

### Requirements

- **Availability**: CI must run within 10 minutes of push (p95)
- **Platform coverage**: Support 85%+ of Python developers
- **Automation**: Zero manual intervention for releases
- **Correctness**: Version synchronization between Python (`pyproject.toml`) and Rust (`Cargo.toml`)

### Constraints

- GitHub Actions free tier: 2,000 minutes/month
- Maturin limitations: Requires platform-specific builds (no cross-compilation)
- PyPI requirements: Wheels must follow naming conventions (PEP 427)

---

## Decision

Use GitHub Actions with PyO3/maturin-action for automated multi-platform wheel builds targeting minimum viable platform set.

### Platform Matrix

**Required platforms** (minimum):

- Linux x86_64 (manylinux2014) - `ubuntu-22.04` runner
- macOS ARM64 (M1/M2/M3) - `macos-14` runner
- Source distribution (sdist) - fallback for unsupported platforms

**Coverage**: 85% of Python developers (based on 2025 usage data)

**Rationale**: Two-platform minimum balances coverage vs CI cost. Linux servers + Apple Silicon development environments are most common.

### CI Workflow Architecture

**Workflow 1**: `ci-test-build.yml`

- **Trigger**: Every push to main, all pull requests
- **Jobs**:
  - Test matrix: Python 3.9-3.12 on ubuntu-latest
  - Rust tests: cargo test on ubuntu-latest
  - Build wheels: Linux x86_64, macOS ARM64 (no publish)
- **Runtime**: ~8 minutes (parallel execution)

**Workflow 2**: `release.yml`

- **Trigger**: Push to main branch only
- **Jobs**:
  - Semantic-release: version bump, tag, CHANGELOG
  - Build wheels: If released, build for all platforms
  - Publish PyPI: If released, upload wheels
  - Create GitHub Release: Upload wheels, generated notes
- **Runtime**: ~12 minutes (if release triggered)

### Version Synchronization

**Challenge**: Maturin projects have two version sources

**Solution**: python-semantic-release updates both files

```toml
[tool.semantic_release]
version_toml = [
    "pyproject.toml:project.version",
    "Cargo.toml:package.version"
]
```

**Validation**: CI checks version match before build

### Conventional Commits

**Enforcement**: Pre-commit hooks validate commit message format

**Mapping to versions**:

- `feat:` → minor bump (0.1.0 → 0.2.0)
- `fix:` → patch bump (0.1.0 → 0.1.1)
- `BREAKING CHANGE:` → major bump (0.1.0 → 1.0.0)\*

\*Note: `major_on_zero = true` means breaking changes bump minor for 0.x versions

### PyPI Publishing

**Method**: Trusted Publisher (OpenID Connect)

**Rationale**:

- No long-lived API tokens (security)
- Automatic credential rotation
- Scoped to specific GitHub workflow
- Zero maintenance after setup

**Fallback**: API token as `PYPI_API_TOKEN` secret if Trusted Publisher unavailable

---

## Consequences

### Positive

**Availability**: Users install without Rust toolchain (85% platform coverage)

**Correctness**: Automated version sync prevents drift between Python and Rust versions

**Observability**: GitHub Actions logs provide full build/release audit trail

**Maintainability**: Zero manual release process (semantic-release automates everything)

### Negative

**CI cost**: Multi-platform builds consume ~20 minutes of GitHub Actions time per release

**Limited platforms**: 15% of users (Windows, Linux ARM, macOS Intel) must build from source

**Workflow complexity**: Two workflows with conditional logic harder to debug than single workflow

### Mitigations

- CI cost acceptable (~$1-2/month overage)
- sdist fallback enables source builds for unsupported platforms
- Comprehensive workflow testing in `/tmp` before deployment

---

## Alternatives Considered

### Alternative 1: All Platforms (10+ matrices)

**Rejected**: 5× CI time, marginal coverage gain (85% → 95%)

### Alternative 2: Manual Releases

**Rejected**: Violates automation requirement, error-prone version management

### Alternative 3: setuptools-rust Instead of Maturin

**Rejected**: Maturin simpler for pure Rust extensions, better ecosystem support

### Alternative 4: Cross-Compilation

**Rejected**: PyO3 cross-compilation immature, platform-specific issues

---

## Compliance

**SLO**:

- **Availability**: CI completes within 10 minutes (p95)
- **Correctness**: Version sync validated before release
- **Observability**: GitHub Actions UI + logs provide visibility
- **Maintainability**: Workflows use maintained actions (maturin-action, semantic-release)

**Error Handling**: Workflows fail loudly on errors (no partial publishes)

**OSS Dependencies**:

- PyO3/maturin-action (not custom build scripts)
- python-semantic-release (not custom versioning)
- pypa/gh-action-pypi-publish (not manual upload)

---

## References

- maturin-action: https://github.com/PyO3/maturin-action
- python-semantic-release: https://python-semantic-release.readthedocs.io/
- PyPI Trusted Publishers: https://docs.pypi.org/trusted-publishers/

---

**Related**:

- ADR-003: Testing Strategy
- ADR-005: Automated Release Management
- Plan: `docs/plan/0004-cicd-architecture/plan.yaml`
