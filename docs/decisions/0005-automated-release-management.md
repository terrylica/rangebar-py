# ADR-005: Automated Release Management

**Status**: Accepted

**Date**: 2024-11-16

**Decision Makers**: Engineering Team

---

## Context

rangebar-py follows semantic versioning and conventional commits. Manual release process requires:

1. Manually increment version in `pyproject.toml` and `Cargo.toml`
2. Manually write CHANGELOG.md entries
3. Manually create git tag
4. Manually build wheels
5. Manually upload to PyPI
6. Manually create GitHub Release

This is error-prone and doesn't scale as project matures.

### Requirements

- **Automation**: Releases triggered by conventional commits, not manual steps
- **Correctness**: Version synchronization across Python and Rust files
- **Observability**: Auto-generated CHANGELOG from commit history
- **Maintainability**: Sustainable process for ongoing releases (v0.1.1, v0.2.0, etc.)

### User Constraint

User specified: "Use semantic-release with GH token (conv-commits→tag→GH release→changelog)" and "Set up semantic-release NOW, use it for v0.1.0" (not manual first release then automate)

---

## Decision

Implement fully automated release management using python-semantic-release from initial v0.1.0 release.

### Release Workflow

**Trigger**: Push to `main` branch with conventional commits

**Automation steps**:

1. **Analyze commits**: semantic-release determines version bump type (major/minor/patch)
2. **Update versions**: Writes new version to `pyproject.toml` AND `Cargo.toml`
3. **Generate CHANGELOG**: Extracts features/fixes from commits, updates `CHANGELOG.md`
4. **Create git tag**: Tags release (e.g., `v0.1.0`)
5. **Push changes**: Commits version bump + CHANGELOG, pushes tag
6. **Build wheels**: maturin builds platform-specific wheels
7. **Publish PyPI**: Uploads wheels to PyPI
8. **Create GitHub Release**: Publishes release with CHANGELOG excerpt

**Concurrency**: Single release at a time (no race conditions)

### Conventional Commit Enforcement

**Pre-commit hook**: commitizen validates commit message format before allowing commit

**Valid formats**:

- `feat(scope): description` - New feature
- `fix(scope): description` - Bug fix
- `perf(scope): description` - Performance improvement
- `docs(scope): description` - Documentation (no release)
- `chore(scope): description` - Maintenance (no release)
- `BREAKING CHANGE: description` in footer - Breaking change

**Rejected commits**: Invalid format blocks commit immediately (fail-fast)

### CHANGELOG Generation

**Auto-excluded commit types** (not in CHANGELOG):

- `chore:` - Build/dependency updates
- `ci:` - CI configuration
- `test:` - Test changes
- `docs:` - Documentation-only changes
- `build(deps):` - Dependency bumps

**Grouped by type**:

```markdown
## v0.1.0 (2024-11-16)

### Features

- **core**: Add RangeBarProcessor Python bindings ([abc123])
- **backtesting**: Add process_trades_to_dataframe helper ([def456])

### Bug Fixes

- **types**: Fix timestamp conversion precision loss ([ghi789])
```

**Intent over implementation**: Commit messages describe what/why, not how

### Version Bump Logic

**python-semantic-release configuration**:

```toml
[tool.semantic_release]
version_toml = [
    "pyproject.toml:project.version",
    "Cargo.toml:package.version"
]
commit_parser = "conventional"
major_on_zero = true  # 0.x treats breaking changes as minor
```

**Bump rules** (pre-1.0):

- `feat:` → 0.1.0 → 0.2.0 (minor)
- `fix:` → 0.1.0 → 0.1.1 (patch)
- `BREAKING CHANGE:` → 0.1.0 → 0.2.0 (minor, not major)

### GitHub Token Setup

**Primary**: `secrets.GITHUB_TOKEN` (automatic)

**Fallback**: Personal Access Token (`secrets.GH_TOKEN`) if branch protection requires

**Permissions required**:

- `contents: write` - Push commits, tags
- `id-token: write` - Trusted Publisher for PyPI

---

## Consequences

### Positive

**Correctness**: Automated version sync eliminates manual drift between `pyproject.toml` and `Cargo.toml`

**Observability**: CHANGELOG automatically documents all changes, searchable by version

**Maintainability**: Release process scales to 100+ releases without manual overhead

**Availability**: Releases complete within 15 minutes of merge to main (automated)

### Negative

**Learning curve**: Contributors must learn conventional commit format

**Git history pollution**: Version bump commits add noise to history

**Rollback complexity**: Undoing release requires manual intervention (delete tag, yank from PyPI)

### Mitigations

- Pre-commit hooks enforce format (fail fast)
- Version bump commits use consistent format (searchable/filterable)
- Document rollback procedure in `docs/RELEASING.md`

---

## Alternatives Considered

### Alternative 1: Manual v0.1.0, Automate Later

**Rejected**: User requirement specified "semantic-release NOW"

### Alternative 2: GitHub Release-Please

**Rejected**: Less Python ecosystem integration than python-semantic-release

### Alternative 3: Commitizen Bump (No semantic-release)

**Rejected**: Less automation (no PyPI publish, no GitHub Release creation)

### Alternative 4: Keep `Cargo.toml` and `pyproject.toml` Separate

**Rejected**: Maturin requires matching versions; drift causes build failures

---

## Compliance

**SLO**:

- **Correctness**: Version sync validated before publish (fail if mismatch)
- **Observability**: CHANGELOG + GitHub Release notes provide change visibility
- **Maintainability**: Zero manual version management
- **Availability**: Releases complete within 15 minutes (p95)

**Error Handling**:

- Release fails loudly if version sync breaks (no partial release)
- PyPI upload failure prevents GitHub Release creation (atomic)
- No silent fallbacks (conventional commit validation strict)

**OSS Dependencies**:

- python-semantic-release (not custom versioning)
- commitizen (not custom commit validation)
- pre-commit (not custom git hooks)

---

## Validation Plan

**Pre-v0.1.0 validation**:

1. Dry-run: `semantic-release --noop version` (verify version detection)
2. Test in `/tmp` mock project (verify CHANGELOG generation)
3. Verify version sync: `grep version pyproject.toml Cargo.toml`
4. Test pre-commit hook: Invalid commit rejected

**Post-v0.1.0 validation**:

1. Verify PyPI package: `pip install rangebar==0.1.0`
2. Verify GitHub Release created with CHANGELOG excerpt
3. Verify `CHANGELOG.md` committed to repository
4. Verify git tag: `git tag | grep v0.1.0`

---

## References

- python-semantic-release: https://python-semantic-release.readthedocs.io/
- Conventional Commits: https://www.conventionalcommits.org/
- Commitizen: https://commitizen-tools.github.io/commitizen/

---

**Related**:

- ADR-003: Testing Strategy
- ADR-004: CI/CD Multi-Platform Builds
- Plan: `docs/plan/0005-release-management/plan.yaml`
