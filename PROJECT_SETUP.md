# Project Setup Summary

**Project**: rangebar-py
**Location**: `~/eon/rangebar-py`
**Status**: Ready for implementation
**Created**: 2025-10-06

---

## What This Project Does

Python bindings (PyO3/maturin) for the [rangebar](https://github.com/terrylica/rangebar) Rust crate, enabling backtesting.py users to leverage high-performance range bar construction without time-based artifacts.

**Key Innovation**: Zero burden on upstream maintainer - we import their crate as a dependency.

---

## Files Created

### 📋 Documentation & Planning

| File                         | Purpose                                                                  |
| ---------------------------- | ------------------------------------------------------------------------ |
| **`CLAUDE.md`**              | Complete project memory - read this first when working in this directory |
| **`IMPLEMENTATION_PLAN.md`** | Step-by-step roadmap (9 phases, ~19 hours)                               |
| **`TODO.md`**                | Quick checklist of immediate next steps                                  |
| **`README.md`**              | User-facing documentation (expand during implementation)                 |
| **`PROJECT_SETUP.md`**       | This file - overview of project structure                                |

### ⚙️ Configuration

| File             | Purpose                                 |
| ---------------- | --------------------------------------- |
| **`.gitignore`** | Rust + Python ignore patterns           |
| **`LICENSE`**    | MIT license (matches upstream rangebar) |

---

## Directory Structure (To Be Created)

When you start Phase 1 implementation, create:

```
rangebar-py/
├── src/                      # Rust code (PyO3 bindings)
│   └── lib.rs
├── python/                   # Python code
│   └── rangebar/
│       ├── __init__.py       # Main API
│       ├── __init__.pyi      # Type stubs
│       └── backtesting.py    # backtesting.py integration
├── tests/                    # Tests
│   ├── test_core.py
│   ├── test_backtesting.py
│   └── test_performance.py
├── examples/                 # Examples
│   ├── basic_usage.py
│   └── backtesting_integration.py
└── docs/                     # Additional documentation
    └── api.md
```

---

## What to Read First

### If you want to START IMPLEMENTING:

1. **`CLAUDE.md`** - Understand the complete context
2. **`IMPLEMENTATION_PLAN.md`** - Follow Phase 1 step-by-step
3. **`TODO.md`** - Quick checklist for immediate actions

### If you want to UNDERSTAND THE CONTEXT:

1. **`README.md`** - High-level overview
2. **`CLAUDE.md`** - Complete project background
3. **Upstream rangebar**: https://github.com/terrylica/rangebar

### If you want to SEE THE ROADMAP:

1. **`IMPLEMENTATION_PLAN.md`** - Detailed phase breakdown
2. **`TODO.md`** - Current status and next steps

---

## Quick Start Commands

### When Ready to Implement

```bash
# Enter project directory
cd ~/eon/rangebar-py

# Read project memory
cat CLAUDE.md

# Start Phase 1: Create structure
mkdir -p src python/rangebar tests examples docs

# Initialize Git
git init
git add .
git commit -m "chore: initialize rangebar-py project"

# Install dependencies
pip install maturin pytest pandas

# Continue with IMPLEMENTATION_PLAN.md Phase 2
```

---

## Key Dependencies

### Rust Crates (in Cargo.toml)

- **rangebar-core**: v5.0 (upstream crate we're wrapping)
- **pyo3**: v0.22 (Python bindings)
- **chrono**: v0.4 (timestamp handling)

### Python Packages (in pyproject.toml)

- **pandas**: ≥2.0 (DataFrame operations)
- **numpy**: ≥1.24 (numerical operations)
- **backtesting.py**: ≥0.3 (optional, target integration)

### Build Tools

- **maturin**: ≥1.7 (Rust → Python package builder)
- **pytest**: ≥7.0 (testing)
- **mypy**: ≥1.0 (type checking)

---

## Integration with backtesting.py Project

This project is a **companion tool** for the backtesting.py fork:

- **Location**: `~/eon/backtesting.py`
- **Branch**: `research/compression-breakout`
- **Status**: Research terminated after 17 failed strategies on time-based bars
- **Motivation**: Range bars may reveal market structure that time bars obscure

**Research Question**: Do strategies perform better on range bars than time bars?

---

## Success Criteria

### MVP (Minimum Viable Product)

- [ ] `pip install rangebar` works
- [ ] Convert Binance CSV → range bars → backtesting.py
- [ ] Performance: >1M trades/sec
- [ ] Test coverage: ≥95%
- [ ] Documentation: README with examples

### Production Release

- [ ] Published to PyPI
- [ ] Wheels for Linux, macOS, Windows
- [ ] GitHub release with v0.1.0 tag
- [ ] CI/CD with GitHub Actions

---

## Timeline Estimate

| Milestone                       | Time      | Date (if started today) |
| ------------------------------- | --------- | ----------------------- |
| **MVP** (Phases 1-5)            | ~11 hours | Day 2 afternoon         |
| **Docs & Testing** (Phases 6-7) | +5 hours  | Day 2 evening           |
| **Distribution** (Phase 8)      | +2 hours  | Day 3 morning           |
| **Release** (Phase 9)           | +1 hour   | Day 3 afternoon         |
| **Total**                       | ~19 hours | 2.5 days                |

---

## Current Status

✅ **Planning Complete**

- [x] Project structure designed
- [x] Documentation written
- [x] Implementation roadmap created
- [x] Dependencies identified

⏸️ **Ready for Phase 1**

- [ ] Directory structure creation
- [ ] Git initialization
- [ ] Build configuration

---

## How to Use This Setup

### Scenario 1: Starting Fresh

You're about to start implementing rangebar-py from scratch.

**Action**:

1. `cd ~/eon/rangebar-py`
2. Read `CLAUDE.md` (5 minutes)
3. Skim `IMPLEMENTATION_PLAN.md` (5 minutes)
4. Start Phase 1 in `TODO.md`

### Scenario 2: Returning After Break

You've started implementation but took a break.

**Action**:

1. `cd ~/eon/rangebar-py`
2. Check `TODO.md` for current status
3. Review `CLAUDE.md` for context refresh
4. Continue where you left off

### Scenario 3: Someone Else Joining

Another developer wants to contribute.

**Action**:

1. Direct them to `README.md` for overview
2. Then `CLAUDE.md` for complete context
3. Then `IMPLEMENTATION_PLAN.md` for current phase
4. Check `TODO.md` for what's done vs. pending

---

## Important Notes

### Do NOT Modify Upstream

- ❌ **Never** submit PRs to rangebar Rust crate for Python support
- ✅ **Always** import rangebar-core as a Cargo dependency
- ✅ **Focus** on Python API only

### Separation of Concerns

- **rangebar crate**: Algorithm implementation (maintained by terrylica)
- **rangebar-py**: Python bindings (this project)
- **backtesting.py**: Target framework (user's fork)

### Zero Maintainer Burden

The rangebar maintainer should be able to:

- Continue developing the Rust crate
- Publish new versions to crates.io
- **Never** think about Python

We handle all Python integration ourselves.

---

## Support & Resources

### Documentation

- **This project**: See files in this directory
- **Upstream rangebar**: https://github.com/terrylica/rangebar
- **PyO3 Guide**: https://pyo3.rs/
- **Maturin Docs**: https://www.maturin.rs/

### Reference Projects

- **backtesting.py**: https://kernc.github.io/backtesting.py/
- **PyO3 Examples**: https://github.com/PyO3/pyo3/tree/main/examples

---

## Questions & Troubleshooting

### Q: Where do I start?

**A**: Read `CLAUDE.md`, then follow `IMPLEMENTATION_PLAN.md` Phase 1.

### Q: What if rangebar-core API changes?

**A**: Update `Cargo.toml` version, adjust bindings in `src/lib.rs`, release new version.

### Q: How do I test without backtesting.py?

**A**: Use synthetic data in tests (see `tests/test_core.py` examples in plan).

### Q: Can I skip the Rust code and use CLI?

**A**: No - this defeats the purpose. Python bindings provide native API.

---

## Next Steps

**When ready to begin**:

```bash
cd ~/eon/rangebar-py
# You're here! Read CLAUDE.md next.
```

**First implementation command**:

```bash
mkdir -p src python/rangebar tests examples docs
```

Happy coding! 🚀
