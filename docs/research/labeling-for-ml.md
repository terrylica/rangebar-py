<!-- SSoT-OK: version references below are historical context, not declarations -->

# Range Bar Labeling for Machine Learning

**Context**: rangebar-py | **Framework**: López de Prado, AFML (2018)
**Purpose**: Educational reference for range bar label construction and ML evaluation

---

## 1. Is "Labeling" the Right Term?

**Yes — with precision.** In López de Prado's _Advances in Financial Machine Learning_ (AFML, Chapter 3), **labeling** refers specifically to constructing the **target variable** (`y`) for supervised learning. It is the process of assigning a ground-truth outcome to each observation (bar) based on future price behavior.

The key distinction:

```
┌──────────────┬──────────────────────────────────────────────────┐
│              │              AFML Terminology                    │
├──────────────┼──────────────────────────────────────────────────┤
│  Features    │  Observable metrics from current/past data       │
│  (X)         │  e.g., ofi, vwap_close_deviation, duration_us    │
│              │  → Already in rangebar-py (v7.0 microstructure)  │
├──────────────┼──────────────────────────────────────────────────┤
│  Labels      │  Future outcome relative to prediction point     │
│  (y)         │  e.g., "next bar return", triple barrier outcome │
│              │  → NOT yet in rangebar-py (this document)        │
├──────────────┼──────────────────────────────────────────────────┤
│  Returns     │  Raw price change metrics (continuous)           │
│  (r)         │  e.g., close-to-close, open-to-close             │
│              │  → Become labels when discretized or used as y   │
└──────────────┴──────────────────────────────────────────────────┘
```

What you described — pre-calculating close-to-close and open-to-close returns for
each bar — is **label construction**. These are forward-looking return metrics
that become supervised learning targets. López de Prado would call them labels
once discretized (e.g., +1 / 0 / -1) or used directly as regression targets.

---

## 2. The Four Combinations Per Bar

Each bar produces **4 label combinations** from 2 return types × 2 prediction
directions:

```
                       Return Metric
                ┌────────────────┬────────────────┐
                │ Close-to-Close │ Open-to-Close  │
                │     (c2c)      │     (o2c)      │
┌───────────────┼────────────────┼────────────────┤
│  Predict UP   │  c2c_up_pnl    │   o2c_up_pnl   │
│  (long)       │                │                │
├───────────────┼────────────────┼────────────────┤
│  Predict DN   │  c2c_dn_pnl    │   o2c_dn_pnl   │
│  (short)      │                │                │
└───────────────┴────────────────┴────────────────┘

4 combinations per bar
```

### 2.1 Return Definitions

**Close-to-Close (c2c)**: Measures return from previous bar's close to current
bar's close. This captures the **full inter-bar movement** and is the standard
for evaluating prediction quality across bars.

```
Bar N-1                        Bar N
┌────────────────────┐        ┌────────────────────┐
│       ...   Close ─┼────────┼─ Open   ...        │
│             $100   │  gap?  │  $100.50    Close  │
│                    │        │             $101   │
└────────────────────┘        └────────────────────┘
                      ◄────────────────────────────►
                       c2c = ($101 - $100) / $100
                           = +1.00%
```

**Open-to-Close (o2c)**: Measures return within the current bar only. This
captures the **intra-bar movement** and represents what you'd capture if you
entered at bar open and exited at bar close.

```
Bar N
┌──────────────────────────────────────┐
│ Open                          Close  │
│ $100.50 ─────────────────── $101.00  │
└──────────────────────────────────────┘
  o2c = ($101 - $100.50) / $100.50
      = +0.498%
```

### 2.2 The Four PnL Labels

For each bar, assuming no transaction costs and no slippage:

| Label        | Formula                                | Interpretation                                                 |
| ------------ | -------------------------------------- | -------------------------------------------------------------- |
| `c2c_up_pnl` | `(close[i] - close[i-1]) / close[i-1]` | Return if you predicted UP and went long from prev close       |
| `c2c_dn_pnl` | `(close[i-1] - close[i]) / close[i-1]` | Return if you predicted DOWN and went short from prev close    |
| `o2c_up_pnl` | `(close[i] - open[i]) / open[i]`       | Return if you predicted UP and went long at this bar's open    |
| `o2c_dn_pnl` | `(open[i] - close[i]) / open[i]`       | Return if you predicted DOWN and went short at this bar's open |

**Key identity**: `c2c_dn_pnl = -c2c_up_pnl` and `o2c_dn_pnl = -o2c_up_pnl`

This means **only 2 independent values** exist per bar:

- `c2c_return = (close[i] - close[i-1]) / close[i-1]`
- `o2c_return = (close[i] - open[i]) / open[i]`

The "predict UP" PnL equals the raw return; the "predict DOWN" PnL is its negation.
We store all 4 for readability and direct lookup, but only 2 are independent.

---

## 3. Intentional Lookahead: How Labels Work in the DataFrame

This is the single most important nuance in labeling. A label on row `i`
**intentionally contains future information** — information that was unknowable
at the time you would have made a prediction. This is not a bug. This is by
design. But you must understand the mechanics precisely to avoid lookahead bias
during training.

### 3.1 The Same-Row Paradox

When you call `add_labels(df)`, the output looks like this:

```
         Features (known)              Labels (future)
         ──────────────────            ──────────────────
Row i:   ofi, vwap_dev, ...    │     c2c_return, o2c_return
                               │
                               │
    The features describe      │     The labels describe
    what happened DURING       │     what happened AFTER
    bar i (or before it)       │     the prediction point
```

Both sit on the **same row** of the DataFrame. This is standard in supervised
learning — the training pair `(X_i, y_i)` is always stored together. But you
must understand what each column's **temporal reference point** is:

```
Timeline for row i:

    ──────────────────────────────────────────────────────►  time
         │                │                    │
    close[i-1]        open[i]             close[i]
         │                │                    │
         │   ◄── gap ──►  │   ◄── bar i ────►  │
         │                │                    │
    ─────┼────────────────┼────────────────────┼──────────
         │                │                    │
    Features use     Features may        Labels use
    close[i-1]       use open[i]         close[i]
    and earlier      (known at open)     (UNKNOWN at
                                         prediction time)
```

### 3.2 Why This Is Intentional (Not Lookahead Bias)

In supervised learning, you always need ground truth to train against. The label
is the **answer key** — what actually happened. The entire point of a predictive
model is to learn patterns in features (`X`) that correlate with future outcomes
(`y`).

```
┌───────────────────────────────────────────────────────────────────┐
│                   Intentional vs Accidental Lookahead             │
├────────────────────────────────┬──────────────────────────────────┤
│ INTENTIONAL (correct)          │ ACCIDENTAL (bias — a bug)        │
├────────────────────────────────┼──────────────────────────────────┤
│ Label y_i uses close[i]        │ Feature X_i uses close[i+1]      │
│ (future outcome as target)     │ (future data as input)           │
├────────────────────────────────┼──────────────────────────────────┤
│ Model never sees y during      │ Model sees future data during    │
│ inference — only during        │ inference — produces unrealistic │
│ training as ground truth       │ accuracy that vanishes in live   │
├────────────────────────────────┼──────────────────────────────────┤
│ Example: c2c_return on row i   │ Example: using c2c_return as a   │
│ as target variable             │ feature for same-row prediction  │
└────────────────────────────────┴──────────────────────────────────┘
```

The label is **always** lookahead by definition. The danger is when **features**
contain lookahead — when the model's inputs include information from the future.

### 3.3 Feature-Label Time Boundaries

For each return type, the prediction point is different, and the features
available at that prediction point are different:

```
c2c prediction (at bar i-1's close):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ◄──── Features: bars [0..i-1] ────►│◄── Label: c2c on row i ─► │
│                                     │                           │
│  Everything up to and including     │  close[i] - close[i-1]    │
│  bar i-1 is known                   │  UNKNOWN at prediction    │
│                                     │  time                     │
│  Prediction point: close[i-1]       │                           │
└─────────────────────────────────────┴───────────────────────────┘

o2c prediction (at bar i's open):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ◄── Features: bars [0..i-1] + open[i] ──► │◄─ Label: o2c ────► │
│                                            │                    │
│  Everything up to and including bar i's    │  close[i] - open[i]│
│  open price is known                       │  UNKNOWN at        │
│                                            │  prediction time   │
│  Prediction point: open[i]                 │                    │
└────────────────────────────────────────────┴────────────────────┘
```

**Critical rule**: During training, the model receives `(X_i, y_i)` pairs.
During inference (live trading), the model receives only `X_i` and must
produce a prediction **without** seeing `y_i`. The label column exists in
the training data but is absent in production.

### 3.4 Practical Safeguards Against Accidental Lookahead

| Safeguard                                       | What It Prevents                                    |
| ----------------------------------------------- | --------------------------------------------------- |
| Never use `c2c_return` as a feature             | The label itself leaking into inputs                |
| Never use `close[i]` in features for c2c models | Future close leaking into same-row features         |
| Never use `close[i]` in features for o2c models | Same — close is unknowable at bar open              |
| Use `open[i]` in features only for o2c models   | Open is known at o2c prediction time                |
| Use `shift(1)` for all cross-bar features       | Ensures features come from completed bars           |
| Walk-forward validation, never random split     | Prevents future bars from appearing in training set |

**Example of correct feature construction**:

```python
# CORRECT: features from bar i-1, label from bar i
df["feature_prev_ofi"]  = df["ofi"].shift(1)          # bar i-1's OFI
df["feature_prev_vwap"] = df["vwap_close_deviation"].shift(1)
df["label_c2c"]         = df["c2c_return"]             # bar i's return

# WRONG: feature from bar i, label from bar i (lookahead!)
df["feature_ofi"]       = df["ofi"]                    # bar i's OFI
df["label_c2c"]         = df["c2c_return"]             # bar i's return
# BUG: ofi is computed at bar i's CLOSE, same time as the label
```

### 3.5 The o2c Subtlety

`o2c_return` uses both `open[i]` and `close[i]`. At bar open time, only
`open[i]` is known — `close[i]` is unknowable. So o2c is also a lookahead
label. The difference from c2c is only **how far ahead** it looks:

```
c2c looks ahead: from close[i-1] to close[i]  (full inter-bar span)
o2c looks ahead: from open[i]   to close[i]   (intra-bar span only)
```

Both are labels. Both contain future information. Both are intentional.
The model's job is to predict them from features that do NOT contain
future information.

---

## 4. Why Both Return Types Matter

### 3.1 Close-to-Close: The Prediction Benchmark

```
Time ─────────────────────────────────────────────────►

Bar N-1            Bar N              Bar N+1
┌──────────┐      ┌──────────┐      ┌──────────┐
│   C=$100 ├─────►│O  C=$103 ├─────►│O  C=$101 │
└──────────┘      └──────────┘      └──────────┘
            ◄────────────────►
               c2c = +3.0%
                              ◄────────────────►
                                 c2c = -1.94%
```

**Use case**: Evaluating a model that predicts _before bar N forms_ whether the
next bar will go up or down. The model makes its prediction at bar N-1's close,
and the realized PnL is measured close-to-close.

**Post-defer_open significance**: After the `defer_open` fix (Issue #46),
`close[N-1] ≠ open[N]` in general. The breaching trade belongs to bar N-1's
close, and the _next_ trade opens bar N. This means c2c captures any **gap**
between bars — a gap that o2c misses entirely.

### 3.2 Open-to-Close: The Execution Benchmark

```
Bar N
┌──────────────────────────────────────────┐
│ O=$100.50                      C=$103    │
│  │                               │       │
│  ├────────── o2c = +2.49% ───────┤       │
└──────────────────────────────────────────┘
```

**Use case**: Evaluating a model that predicts _at bar open_ which direction
the bar will move. More realistic for execution: you observe the bar opening
and decide to go long or short within it.

### 3.3 The Gap Between c2c and o2c

```
                   ┌──── c2c = +3.00% ────┐
                   │                      │
Bar N-1 Close ─────┤                      ├── Bar N Close
    $100           │  ┌── o2c = +2.49% ───┤      $103
                   │  │                   │
                   │  Bar N Open          │
                   │  $100.50             │
                   │  │                   │
                   └──┘                   │
                   gap = +0.50%           │
                                          │
                   c2c = gap + o2c        │
                   3.00% ≈ 0.50% + 2.49%  │
```

The difference between c2c and o2c reveals the **overnight/inter-bar gap**. In
range bars, this gap exists because:

1. The breaching trade closes bar N-1 at price P
2. The _next_ trade (at price Q ≠ P typically) opens bar N
3. `gap = (Q - P) / P`

This gap is **not predictable** from bar N's features alone and represents
market movement between bars. A model evaluated on c2c must implicitly predict
this gap; a model evaluated on o2c does not.

---

## 4. Range Bar Properties Relevant to Labeling

Range bars have unique properties that affect label construction:

```
┌──────────────────────────────────────────────────────┐
│             Range Bar Labeling Properties            │
├──────────────────────┬───────────────────────────────┤
│ Property             │ Implication                   │
├──────────────────────┼───────────────────────────────┤
│ Fixed price range    │ o2c bounded by ±threshold     │
│ (threshold)          │ (e.g., ±0.25% for 250 dbps)   │
├──────────────────────┼───────────────────────────────┤
│ Variable duration    │ Labels span unequal time      │
│                      │ periods — not calendar-based  │
├──────────────────────┼───────────────────────────────┤
│ Activity-dependent   │ More labels during volatile   │
│ sampling             │ periods (when they matter)    │
├──────────────────────┼───────────────────────────────┤
│ No time seasonality  │ Labels free from time-of-day  │
│                      │ artifacts in returns          │
├──────────────────────┼───────────────────────────────┤
│ defer_open           │ close[N-1] ≠ open[N] in       │
│ (Issue #46)          │ general — gaps exist          │
└──────────────────────┴───────────────────────────────┘
```

### 4.1 The Bounded o2c Property

For a range bar with threshold `T` decimal basis points:

```
Maximum |o2c| ≈ T / 100,000

Example: 250 dbps → max |o2c| ≈ 0.25%
```

This is because the bar closes when price breaches the threshold from the open.
The o2c return is therefore **capped** by construction. This doesn't limit its
usefulness — it means o2c labels have a known, bounded distribution, which is
actually advantageous for ML normalization.

The c2c return is **not bounded** because inter-bar gaps can be arbitrarily large
(though practically bounded by market microstructure).

---

## 5. López de Prado's Labeling Methods

### 5.1 Fixed-Horizon Labeling (AFML Ch. 3.2)

The simplest method. Look ahead `h` bars and label based on return:

```
At bar i, look at bar i+h:
  r = close[i+h] / close[i] - 1

  y = +1 if r > τ      (buy)
  y =  0 if |r| ≤ τ    (hold)
  y = -1 if r < -τ     (sell)
```

**Limitation**: Uses fixed threshold `τ` regardless of volatility.
During volatile periods, `τ` is too tight; during calm periods, too loose.

**For range bars**: Less problematic because range bars already normalize for
volatility (more bars form during volatile periods). But still ignores the
price _path_ between bars i and i+h.

### 5.2 Triple Barrier Method (AFML Ch. 3.3–3.6)

Three simultaneous barriers:

```
Price
  │
  │  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌  Upper barrier (+pt × σ)
  │                              ╱
  │                   ╱╲    ╱╲  ╱
  │             ╱╲ ╱    ╲  ╱  ╲╱
  │  ───────────╱──╳──────╲╱────────  Entry price
  │           ╱
  │  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌  Lower barrier (-sl × σ)
  │
  │           │                     │
  │           Entry                 │  Vertical barrier (h bars)
  └───────────┴─────────────────────┴──── Time/Bars
```

| Barrier  | Hit First → Label | Meaning               |
| -------- | ----------------- | --------------------- |
| Upper    | +1                | Profit target reached |
| Lower    | -1                | Stop-loss triggered   |
| Vertical | sign(return) or 0 | Time expired          |

**Key innovation**: Barriers scale with volatility (`σ`), making labels
comparable across different market regimes.

**For range bars**: The vertical barrier should be **N bars** (not N minutes),
since range bars already normalize time by activity.

### 5.3 Meta-Labeling (AFML Ch. 3.6)

A two-stage architecture:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Primary Model   │────►│   Meta Model     │────►│  Position        │
│  (Direction)     │     │   (Confidence)   │     │  Sizing          │
│                  │     │                  │     │                  │
│  Predicts: ±1    │     │  Predicts: P(win)│     │  Size ∝ P        │
│  Optimized for:  │     │  Optimized for:  │     │                  │
│  HIGH RECALL     │     │  HIGH PRECISION  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

The meta-model doesn't predict direction — it predicts whether the primary
model's prediction will be **correct**. This separation of concerns lets each
model focus on one task.

---

## 6. Mapping to rangebar-py

### 6.1 Current State

```
rangebar-py output (today):

┌──────────┬───────────────────────────────────────────────┐
│  Bar i   │                                               │
├──────────┼───────────────────────────────────────────────┤
│ OHLCV    │ Open, High, Low, Close, Volume                │
├──────────┼───────────────────────────────────────────────┤
│ Features │ ofi, vwap_close_deviation, price_impact,      │
│ (X)      │ kyle_lambda_proxy, trade_intensity,           │
│          │ volume_per_trade, aggression_ratio,           │
│          │ aggregation_density, turnover_imbalance,      │
│          │ duration_us, vwap, buy_volume, sell_volume    │
├──────────┼───────────────────────────────────────────────┤
│ Labels   │ *** NOT YET COMPUTED ***                      │
│ (y)      │                                               │
└──────────┴───────────────────────────────────────────────┘
```

### 6.2 Proposed Label Columns

```
rangebar-py output (with labels):

┌──────────┬──────────────────────────────────────────────┐
│ Returns  │ c2c_return   = (C[i] - C[i-1]) / C[i-1]      │
│ (raw)    │ o2c_return   = (C[i] - O[i])   / O[i]        │
├──────────┼──────────────────────────────────────────────┤
│ Directnl │ c2c_up_pnl   = +c2c_return                   │
│ Labels   │ c2c_dn_pnl   = -c2c_return                   │
│ (4)      │ o2c_up_pnl   = +o2c_return                   │
│          │ o2c_dn_pnl   = -o2c_return                   │
├──────────┼──────────────────────────────────────────────┤
│ Binary   │ c2c_direction = +1 if c2c_return > 0         │
│ Labels   │                 -1 if c2c_return < 0         │
│ (opt.)   │ o2c_direction = +1 if o2c_return > 0         │
│          │                 -1 if o2c_return < 0         │
├──────────┼──────────────────────────────────────────────┤
│ Gap      │ gap_return   = (O[i] - C[i-1]) / C[i-1]      │
│ (opt.)   │ (inter-bar gap from defer_open)              │
└──────────┴──────────────────────────────────────────────┘
```

### 6.3 Label Computation (Pure Python — No Rust Needed)

Labels are computed from the output DataFrame, not during bar construction.
This keeps the Rust core focused on OHLCV + microstructure:

```python
import pandas as pd

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add ML labels to range bar DataFrame.

    Adds close-to-close and open-to-close return labels
    for both long (up) and short (down) predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Output from get_range_bars() with at least Open, Close columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with label columns appended.
    """
    # Raw returns
    df["c2c_return"] = df["Close"].pct_change()       # close-to-close
    df["o2c_return"] = (df["Close"] - df["Open"]) / df["Open"]  # open-to-close

    # Directional PnL (4 combinations)
    df["c2c_up_pnl"] = df["c2c_return"]               # long from prev close
    df["c2c_dn_pnl"] = -df["c2c_return"]              # short from prev close
    df["o2c_up_pnl"] = df["o2c_return"]               # long at bar open
    df["o2c_dn_pnl"] = -df["o2c_return"]              # short at bar open

    # Inter-bar gap (post-defer_open)
    df["gap_return"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # Binary direction labels
    df["c2c_direction"] = (df["c2c_return"] > 0).astype(int) * 2 - 1  # +1 / -1
    df["o2c_direction"] = (df["o2c_return"] > 0).astype(int) * 2 - 1  # +1 / -1

    return df
```

---

## 7. Practical ML Evaluation Scenarios

### Scenario A: "Predict Next Bar Direction"

```
Model trains on: features of bar i (microstructure)
Model predicts:  direction of bar i+1 (up or down)
Evaluation:      c2c_return of bar i+1

  Bar i (known)                Bar i+1 (predicted)
  ┌─────────────────┐         ┌─────────────────┐
  │ Features: X_i   │         │ y = c2c_dir     │
  │ ofi, vwap_dev   │ ──────► │ PnL = c2c_pnl   │
  │ kyle_lambda     │ predict │                 │
  └────────┬────────┘         └────────┬────────┘
           │                           │
      Close = $100               Close = $103
                                 c2c = +3%

  If model predicted UP:  PnL = +3%
  If model predicted DN:  PnL = -3%
```

**Why c2c**: The model decides at bar i's close. It is "in" the position from
close[i] to close[i+1]. The c2c return is the realized PnL.

### Scenario B: "Predict This Bar's Direction at Open"

```
Model trains on: features of bars [1..i-1] + open[i]
Model predicts:  direction of bar i (up or down from open)
Evaluation:      o2c_return of bar i

  Bar i
  ┌──────────────────────────────────────┐
  │ O=$100.50            C=$103          │
  │  │                     │             │
  │  ├──── o2c = +2.49% ───┤             │
  │  │                                   │
  │  Model sees open,                    │
  │  predicts direction                  │
  └──────────────────────────────────────┘

  If model predicted UP:  PnL = +2.49%
  If model predicted DN:  PnL = -2.49%
```

**Why o2c**: The model decides at bar open. It captures only the intra-bar
movement, which is what you'd realize by entering at open and exiting at close.

### Scenario C: Full Evaluation Matrix

For a model that predicts direction, evaluate both metrics:

```
                         Model Prediction
                     UP (long)     DN (short)
                 ┌──────────────┬──────────────┐
  Actual UP bar  │  c2c: +r     │  c2c: -r     │
  (c2c > 0)      │  o2c: +r'    │  o2c: -r'    │
                 ├──────────────┼──────────────┤
  Actual DN bar  │  c2c: -r     │  c2c: +r     │
  (c2c < 0)      │  o2c: -r'    │  o2c: +r'    │
                 └──────────────┴──────────────┘

  Cumulative PnL = Σ (predicted_direction × return)
```

---

## 8. The defer_open Effect on Labels

The Issue #46 fix changed how bar boundaries work:

### Before defer_open (Buggy)

```
Trade at price P breaches threshold:
  Bar N:   Close = P    (breaching trade closes bar N)
  Bar N+1: Open  = P    (SAME trade opens bar N+1)     ← BUG

  gap_return = 0  (always, because Open[N+1] = Close[N])
  c2c = o2c + 0   (close-to-close equals open-to-close)
```

### After defer_open (Correct)

```
Trade at price P breaches threshold:
  Bar N:   Close = P    (breaching trade closes bar N)
  Bar N+1: Open  = Q    (NEXT trade opens bar N+1)     ← CORRECT

  gap_return = (Q - P) / P  (nonzero gap between bars)
  c2c = gap + o2c           (close-to-close ≠ open-to-close)
```

**Impact on labels**: With defer_open, c2c and o2c are **genuinely different
metrics**. Before the fix, they were nearly identical (differing only by the
gap, which was always zero). This makes the 4-combination label scheme
meaningful — it was degenerate before the fix.

---

## 9. Connection to Triple Barrier (Future Enhancement)

The 4 directional labels above are **fixed-horizon labels** (h=1 bar). To
implement the full triple barrier method on range bars:

```
┌────────────────────────────────────────────────────────────┐
│              Triple Barrier on Range Bars                  │
│                                                            │
│  Entry: Close[i] (or Open[i+1])                            │
│                                                            │
│  Upper barrier: entry × (1 + pt × σ)                       │
│  Lower barrier: entry × (1 - sl × σ)                       │
│  Vertical:      i + h bars forward                         │
│                                                            │
│  σ = rolling volatility of c2c_return                      │
│      (exponentially-weighted, e.g., span=20 bars)          │
│                                                            │
│  For each bar j in [i+1, i+h]:                             │
│    if High[j] >= upper → label = +1, exit                  │
│    if Low[j]  <= lower → label = -1, exit                  │
│  If no barrier hit → label = sign(Close[i+h] - entry)      │
└────────────────────────────────────────────────────────────┘
```

This is a natural **Phase 2** enhancement. The 4 fixed-horizon labels are
Phase 1 — simple, correct, and immediately useful for baseline ML evaluation.

---

## 10. Summary: Label Taxonomy

```
┌──────────────────┬────────────┬──────────────────────────────────────┐
│ Method           │ Complexity │ What It Captures                     │
├──────────────────┼────────────┼──────────────────────────────────────┤
│ o2c_return       │ ★          │ Intra-bar return (bounded by thresh) │
│ c2c_return       │ ★          │ Inter-bar return (includes gap)      │
│ Directional PnL  │ ★★         │ Both returns × {up, down} = 4 cols   │
│ Fixed-horizon    │ ★★         │ h-bar forward return, thresholded    │
│ Triple barrier   │ ★★★        │ Volatility-scaled, path-dependent    │
│ Meta-labeling    │ ★★★★       │ Two-stage: direction + confidence    │
└──────────────────┴────────────┴──────────────────────────────────────┘

rangebar-py Phase 1 (this document):  ★–★★   (returns + directional PnL)
rangebar-py Phase 2 (future):         ★★★    (triple barrier)
rangebar-py Phase 3 (future):         ★★★★   (meta-labeling)
```

---

## References

- López de Prado, M. (2018). _Advances in Financial Machine Learning_. Wiley.
  Chapter 2 (Financial Data Structures), Chapter 3 (Labeling).
- López de Prado, M. (2017). "Meta-Labeling." SSRN working paper.
- Easley, D., López de Prado, M., O'Hara, M. (2012). "The Volume Clock."
  _Review of Financial Studies_.
- Kyle, A. (1985). "Continuous Auctions and Insider Trading." _Econometrica_.
- rangebar-py Issue #46: defer_open streaming/batch parity fix.
- rangebar-py Issue #25: Microstructure features (v7.0).
