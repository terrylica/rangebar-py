# Test Data

Small fixtures for unit and integration tests. **Committed to git.**

## Purpose

Provide **minimal, representative** test data for:

- Unit tests (algorithm correctness)
- Integration tests (end-to-end pipeline)
- Edge case validation (year boundaries, threshold breaches)
- CI/CD pipelines (fast, deterministic tests)

## Constraints

### Size Limits

- **Max ticks per file**: 1,000 ticks
- **Max file size**: 100 KB
- **Total test_data size**: < 5 MB

Rationale: Keep git clone fast, CI/CD responsive

### Data Quality

- **Real data samples** from Binance (not synthetic)
- **Edge cases prioritized**:
    - Year boundary crossings (2024-12-31 → 2025-01-01)
    - Threshold breach sequences
    - High/low volatility periods
    - Crossed market detection (bid > ask)
    - Zero volume ticks
- **Anonymized if needed** (though public market data is typically OK)

## Structure

```
test_data/
├── BTCUSDT/
│   ├── 2024-07-01_sample.csv       # Normal trading day
│   ├── year_boundary_2024.csv      # Edge case: year transition
│   └── high_volatility_spike.csv   # Edge case: rapid price moves
├── ETHUSDT/
│   ├── 2024-07-01_sample.csv
│   └── low_volatility_calm.csv     # Edge case: minimal movement
└── README.md
```

## Naming Convention

**Format**: `[descriptive-name].csv`

**Examples**:

- `2024-07-01_sample.csv` - Representative day
- `year_boundary_2024.csv` - Dec 31, 2024 → Jan 1, 2025
- `high_volatility_spike.csv` - Flash crash or pump
- `threshold_breach_sequence.csv` - Multiple rapid breaches

## Creation Guidelines

### From Real Data

```bash
# Extract first 1000 ticks from production file
head -1001 binance/spot/BTCUSDT/2024-07-01.csv > test_data/BTCUSDT/2024-07-01_sample.csv
```

### Validation

```bash
# Ensure file meets size constraints
du -h test_data/BTCUSDT/2024-07-01_sample.csv  # Should be < 100KB

# Count ticks
wc -l test_data/BTCUSDT/2024-07-01_sample.csv  # Should be ≤ 1001 (header + 1000 ticks)
```

## Usage in Tests

```rust
#[test]
fn test_normal_trading_day() {
    let path = "test_data/BTCUSDT/2024-07-01_sample.csv";
    let trades = load_test_data(path);
    let processor = RangeBarProcessor::new(25);
    let bars = processor.process_agg_trade_records(&trades).unwrap();
    assert!(bars.len() > 0);
}
```

## Git Policy

- **ALWAYS** commit to git (small, essential for tests)
- Include in every CI/CD run
- Do not delete unless replacing with better fixture

## Maintenance

- Review annually (ensure still representative of current market structure)
- Add new edge cases as discovered
- Remove outdated fixtures if market structure changes
