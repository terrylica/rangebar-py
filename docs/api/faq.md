# FAQ & Troubleshooting

**Navigation**: [INDEX.md](./INDEX.md) | [Primary API](./primary-api.md) | [Validation API](./validation-api.md)

---

## Common Questions

### Q: Why did I get a ValueError for a long date range?

**A**: Date ranges > 30 days require the cache workflow (MEM-013 guard):

```python
# This fails with ValueError for >30 day ranges:
df = get_range_bars("BTCUSDT", "2019-01-01", "2025-12-31")  # OOM protection!

# Correct approach:
from rangebar import populate_cache_resumable, get_range_bars

# Step 1: Populate cache (resumable, memory-safe)
populate_cache_resumable("BTCUSDT", "2019-01-01", "2025-12-31")

# Step 2: Read from cache
df = get_range_bars("BTCUSDT", "2019-01-01", "2025-12-31")
```

This prevents OOM by forcing incremental day-by-day processing with checkpoints. See [cache-api.md](./cache-api.md) for details.

---

### Q: What threshold should I use?

**A**: Start with `threshold_decimal_bps=250` (0.25%). This generates ~200 bars/day for BTC/USDT, similar to 1-hour time bars. Adjust based on:

- Higher volatility -> higher threshold (avoid too many bars)
- Lower volatility -> lower threshold (avoid too few bars)
- Shorter strategies -> lower threshold (more bars)
- Longer strategies -> higher threshold (fewer bars)

---

### Q: Why are no range bars generated?

**A**: Price movement is below threshold. Either:

1. Lower threshold (e.g., 250 -> 100)
2. Use more volatile data
3. Check data timespan (need sufficient trades)

---

### Q: Can I use this with live trading?

**A**: rangebar-py is designed for backtesting, not live trading. For live trading, you'd need:

- Streaming API (process trades incrementally)
- Real-time bar completion detection
- State persistence (recover from failures)

These features are planned for future releases.

---

### Q: How do I handle multiple symbols?

**A**: Create separate processors for each symbol:

```python
processors = {
    "BTCUSDT": RangeBarProcessor(threshold_decimal_bps=250),
    "ETHUSDT": RangeBarProcessor(threshold_decimal_bps=250),
}

for symbol, processor in processors.items():
    trades = load_trades(symbol)
    bars = processor.process_trades(trades)
    save_bars(symbol, processor.to_dataframe(bars))
```

---

### Q: Can I process tick data?

**A**: Yes, as long as you have timestamp, price, and quantity. Convert to trade format:

```python
ticks = pd.read_csv("ticks.csv")  # time, bid, ask, bid_size, ask_size

# Convert to trades (midpoint)
trades = pd.DataFrame({
    "timestamp": ticks["time"],
    "price": (ticks["bid"] + ticks["ask"]) / 2,
    "quantity": (ticks["bid_size"] + ticks["ask_size"]) / 2,
})

df = process_trades_to_dataframe(trades, threshold_decimal_bps=250)
```

---

### Q: Why did I get 97K bars during a flash crash? (Issue #36)

**A**: This is expected behavior for range bars during extreme volatility events.

During a flash crash (e.g., BTC dropping 15% in minutes), the price rapidly crosses the threshold many times. Each threshold breach creates a new bar. For example:

- **Normal day**: BTC moves 2% total -> ~80 bars with 250bps threshold
- **Flash crash**: BTC moves 15% in 10 minutes -> hundreds of threshold breaches -> thousands of bars

This is actually a **feature, not a bug**:

1. **Information-preserving**: Each bar represents equal price movement (threshold size)
2. **No lookahead bias**: Bars close at threshold breach, not at arbitrary times
3. **Volatility-adaptive**: More bars during volatile periods = more granular data for backtesting

**Practical implications for backtesting**:

```python
# Flash crash day will have many more bars than quiet day
quiet_day = get_range_bars("BTCUSDT", "2024-01-15", "2024-01-15")  # ~200 bars
crash_day = get_range_bars("BTCUSDT", "2024-03-05", "2024-03-05")  # ~5,000 bars (hypothetical crash)
```

**References**:

- Mandelbrot & Hudson (2004): "The (Mis)behavior of Markets" - fat tails and volatility clustering
- Easley, Lopez de Prado & O'Hara (2012): "The Volume Clock" - information-based sampling

**If you want fewer bars during volatile periods**: Increase threshold or use time-based bars (but lose the benefits of information-based sampling).

---

### Q: What are the expected ranges for microstructure features? (Issue #32)

**A**: See [validation-api.md](./validation-api.md#expected-feature-ranges) for the full table.

**Out-of-range values indicate data issues**:

```python
from rangebar.validation.tier1 import validate_tier1

df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-31", include_microstructure=True)
result = validate_tier1(df)

if not result["tier1_passed"]:
    print("Validation failed:", result)
    # Check specific bounds violations
    if not result["ofi_bounded"]:
        print("OFI values outside [-1, 1] - possible data corruption")
```

**Academic references for feature formulas**:

- **OFI**: Cont, Kukanov & Stoikov (2014) - "The Price Impact of Order Book Events"
- **Kyle Lambda**: Kyle (1985) - "Continuous Auctions and Insider Trading"
- **Trade Intensity**: Hasbrouck (1991) - "Measuring the Information Content of Stock Trades"
- **Aggression Ratio**: Biais, Hillion & Spatt (1995) - "An Empirical Analysis of the Limit Order Book"

---

## Comparison with Alternatives

### vs Time-Based Bars

**Range Bars**:

- Market-adaptive (bars form based on volatility)
- No lookahead bias
- Equal information per bar (fixed price movement)
- Variable time intervals (harder to align with external events)

**Time Bars**:

- Fixed time intervals (easy to align with news, etc.)
- Variable information per bar (volatile periods compressed)
- Potential lookahead bias (if not careful)
- Inverse timeframe effect (shorter timeframes != better)

### vs Tick Bars

**Range Bars**:

- Filters noise (only price movements matter)
- Fewer bars (more efficient)
- Better for trend-following strategies

**Tick Bars**:

- Fixed number of trades per bar
- Ignores price movement (1 cent move = 10% move)
- More bars (slower backtests)

### vs Volume Bars

**Range Bars**:

- Focus on price movement (more intuitive for trading)
- Works well with MA crossover strategies

**Volume Bars**:

- Fixed volume per bar
- Ignores price movement (large volume with small price change)

---

## Common Errors

| Error                                          | Cause                    | Fix                                    |
| ---------------------------------------------- | ------------------------ | -------------------------------------- |
| `RangeBarProcessor has no attribute X`         | Outdated binding         | `maturin develop`                      |
| `Invalid threshold_decimal_bps`                | Wrong units              | Use 250 for 0.25%                      |
| `High < Low` assertion                         | Bad input data           | Check sorting                          |
| `target-cpu=native` cross-compile error        | RUSTFLAGS pollution      | Use `RUSTFLAGS=""`                     |
| OOM with `include_microstructure=True`         | Large date range         | Fixed by MEM-011                       |
| `Date range of N days requires use_cache=True` | Long range without cache | Use `populate_cache_resumable()` first |

---

## See Also

- [README.md](/README.md) - User guide and quick start
- [examples/README.md](/examples/README.md) - Usage examples
- [rangebar_core_api.md](/docs/rangebar_core_api.md) - Rust API documentation
- [backtesting.py docs](https://kernc.github.io/backtesting.py/) - Target framework

---

**Last Updated**: 2026-02-03
