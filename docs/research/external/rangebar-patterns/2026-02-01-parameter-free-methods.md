---
source_url: https://claude.ai/public/artifacts/15191277-a670-4ee9-87dc-2ec65116ecca
source_type: claude-artifact
scraped_at: 2026-02-01T23:07:51Z
purpose: Statistical rigor reference for exp078+ evaluation methodology - parameter-free alternatives to magic numbers
tags: [wfo, statistical-validation, e-values, adwin, tda, mintrl]
model_name: Claude
model_version: unknown (artifact)
tools: []
claude_code_uuid: 3d0512c5-fe15-4d74-86cc-b2e6ece321e9
claude_code_project_path: "~/.claude/projects/-Users-terryli-eon-alpha-forge-worktree-2025-12-24-ralph-ml-robustness-research/3d0512c5-fe15-4d74-86cc-b2e6ece321e9"
github_issue_url: TBD
---

# Parameter-Free Methods for Determining Minimum Reliable Evaluation Horizons in Algorithmic Trading

The "magic number" problem in quantitative finance—arbitrarily choosing **252-day rolling windows** or fixed lookback periods—can now be solved through six distinct mathematical frameworks that automatically determine minimum reliable evaluation horizons. These SOTA approaches eliminate hand-tuned parameters by using information-theoretic principles, topological invariants, Bayesian online learning, and anytime-valid sequential testing. For cryptocurrency markets with their rapid regime shifts and non-Gaussian returns, the combination of **ADWIN** for adaptive windowing, **e-values** for sequential strategy testing, and **persistence landscapes** from TDA provides a robust, principled alternative to fixed-window backtesting.

---

## E-values and anytime-valid inference solve the optional stopping problem

The most mathematically rigorous solution to determining "when do we have enough data" comes from **e-values** and **confidence sequences**—a framework that maintains statistical validity regardless of when you stop collecting data. Unlike p-values, which become invalid under optional stopping (the infamous p-hacking problem), e-values can be multiplied across observations while preserving Type I error control via **Ville's inequality**.

The `expectation` Python library ([https://github.com/jakorostami/expectation](https://github.com/jakorostami/expectation)) implements this framework for sequential strategy testing. An e-value exceeding **1/α** (e.g., 20 for α=0.05) provides valid evidence to reject the null hypothesis at any stopping time—no fixed sample size required. For Walk-Forward Optimization, this means each WFO window can terminate early when sufficient evidence accumulates that strategy edge is real (or absent), dramatically reducing evaluation time without statistical penalties.

The **Sequential Probability Ratio Test (SPRT)** offers a complementary approach with provably optimal expected sample size. The `sprt` package allows testing whether realized Sharpe exceeds a meaningful threshold (e.g., H₁: SR > 0.5 vs H₀: SR = 0), with decision boundaries derived from specified Type I/II error rates. Chess engine developers use this methodology extensively for strategy validation—the Stockfish testing framework demonstrates its robustness for detecting small but real performance edges.

For conformal prediction, **MAPIE** ([https://github.com/scikit-learn-contrib/MAPIE](https://github.com/scikit-learn-contrib/MAPIE), **6.5k+ GitHub stars**) provides distribution-free prediction intervals for time series via `MapieTimeSeriesRegressor`. The `partial_fit()` method enables online calibration as new data arrives, with coverage guarantees that hold regardless of the underlying return distribution—critical for fat-tailed crypto markets.

---

## ADWIN and Bayesian changepoint detection eliminate fixed rolling windows

**ADWIN (Adaptive Windowing)** from the River-ML library ([https://github.com/online-ml/river](https://github.com/online-ml/river)) automatically maintains a variable-length window where the data distribution has remained stationary. The algorithm uses **Hoeffding bounds** to statistically test whether two sub-windows have identical means; when they differ, the older data is discarded. The window size emerges from the data itself—no 252-day magic number required.

```python
from river import drift
adwin = drift.ADWIN(delta=0.002)  # delta controls false positive rate

for i, daily_return in enumerate(strategy_returns):
    adwin.update(daily_return)
    if adwin.drift_detected:
        print(f"Regime change at day {i}, new window size: {adwin.width}")
```

For offline historical analysis, the **ruptures** library ([https://github.com/deepcharles/ruptures](https://github.com/deepcharles/ruptures)) implements PELT with information-theoretic penalties that auto-determine breakpoint count. The **BIC penalty** (log(n) × p) and **CROPS method** (Changepoint for a Range of Penalties) eliminate manual penalty selection by exploring the full penalty landscape and identifying the "elbow" point. The `KernelCPD` algorithm with RBF kernel detects distributional changes beyond simple mean/variance shifts—capturing the complex regime transitions characteristic of crypto markets.

**Bayesian Online Changepoint Detection (BOCPD)** provides posterior probabilities over regime changes in real-time. The `bayesian-changepoint-detection` package ([https://github.com/hildensia/bayesian_changepoint_detection](https://github.com/hildensia/bayesian_changepoint_detection)) computes P(changepoint | data) at each timestep using conjugate priors. When this probability exceeds **0.8**, a regime change is signaled with quantified uncertainty. The Redpoll `changepoint` library (Rust backend, Python bindings) offers GPU-accelerated BOCPD for high-frequency applications.

---

## Topological data analysis detects regimes without specifying their number

Persistent homology from TDA offers a fundamentally different approach: instead of choosing correlation thresholds, **filtration scans all possible thresholds simultaneously**. Features that persist across multiple scales are meaningful; short-lived features are noise. The **L¹-norm of persistence landscapes** (Gidea & Katz, 2017) showed predictive power for market crashes—the norm exhibits significant growth **250 trading days before** major crashes like the 2008 Lehman collapse.

The **giotto-tda** library ([https://github.com/giotto-ai/giotto-tda](https://github.com/giotto-ai/giotto-tda)) provides scikit-learn-compatible TDA with automatic parameter selection for Takens embedding:

```python
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude

embedder = SingleTakensEmbedding(parameters_type="search")  # Auto-selects dim, delay
persistence = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
amplitude = Amplitude(metric="landscape", order=1)  # L1-norm

# Pipeline automatically determines embedding parameters via mutual information
```

The **Euler characteristic** (χ = β₀ - β₁ + β₂) provides a single topological invariant that distinguishes regimes: during crashes, **χ > 0** (fragmentation dominates), while normal markets show **χ < 0** (loop structures from diversification). GUDHI's financial time series notebook ([https://github.com/GUDHI/tda_financial_time_series_notebook](https://github.com/GUDHI/tda_financial_time_series_notebook)) implements this for correlation network analysis. For crypto regime detection, Gidea et al. (2020) demonstrated that k-means clustering on topological features correctly identified the January 2018 crash—with the number of regimes emerging from data topology rather than pre-specification.

---

## Information-theoretic metrics provide distribution-free alternatives to Sharpe

**Permutation Entropy** (PE) quantifies market predictability by analyzing ordinal patterns rather than absolute values. The metric is **scale-invariant** and requires no distributional assumptions—ideal for crypto's non-Gaussian returns. Using the `ordpy` or `antropy` packages:

- PE > 0.9: Market is efficient/random → avoid trend-following
- PE < 0.7: Detectable patterns → trading edge possible
- Rolling PE crossing 0.5 signals regime transitions

The **Time-Varying Hurst Exponent** determines whether a market is trending (H > 0.5) or mean-reverting (H < 0.5). The `nolds` library implements **Detrended Fluctuation Analysis (DFA)**, which is robust to polynomial non-stationarities that corrupt standard R/S estimation. For short samples (N > 100), the variance method in the `hurst-exponent` package provides reliable estimates with bootstrap confidence intervals.

**Transfer Entropy** from `pyinform` quantifies directional information flow between assets—detecting lead-lag relationships without assuming linearity (unlike Granger causality). The **net transfer entropy** between BTC and altcoins reveals which asset "leads" price discovery, enabling systematic exploitation of information propagation delays.

Sample size requirements vary by method: **Distribution Entropy** works with as few as **50 observations** (via `EntropyHub`), Permutation Entropy requires N >> d! (typically **1000+ for d=4**), and Transfer Entropy needs approximately **500+** observations depending on discretization. These thresholds replace arbitrary window lengths with information-theoretically justified minimums.

---

## Minimum Track Record Length provides frequentist convergence guarantees

Bailey & López de Prado's **MinTRL** formula explicitly calculates the minimum number of observations needed for a Sharpe ratio to be statistically significant:

**MinTRL = (1 - γ₃·SR + (γ₄-1)/4 · SR²) × (z₁₋α / (SR - SR\*))²**

where γ₃ = skewness, γ₄ = kurtosis, SR\* = benchmark Sharpe. The formula incorporates higher moments that Sharpe ignores—critical for crypto's fat tails. A strategy with SR = 1.0 and typical crypto kurtosis (~8) requires approximately **2.5× more observations** than normally-distributed returns to achieve the same confidence.

**QuantStats** ([https://github.com/ranaroussi/quantstats](https://github.com/ranaroussi/quantstats), **6.5k stars**) provides the most actively maintained implementation with rolling PSR support and an autocorrelation-adjusted "Smart Sharpe" that penalizes strategies exploiting serial correlation artifacts. The **pypbo** library ([https://github.com/esvhd/pypbo](https://github.com/esvhd/pypbo)) extends this with Probability of Backtest Overfitting (PBO) and stochastic dominance tests—directly addressing multiple testing concerns in WFO.

For streaming/online MinTRL, combine QuantStats' PSR with pandas rolling:

```python
def rolling_mintrl(returns, window=252, sr_benchmark=0.0, confidence=0.95):
    def calc_mintrl(r):
        sr = r.mean() / r.std() * np.sqrt(252)
        skew, kurt = r.skew(), r.kurtosis() + 3
        z = norm.ppf(confidence)
        if sr <= sr_benchmark: return np.inf
        return (1 - skew*sr + (kurt-1)/4*sr**2) * (z / (sr - sr_benchmark))**2
    return returns.rolling(window).apply(calc_mintrl)
```

---

## Integrated framework for crypto Walk-Forward Optimization

For BiLSTM with Hard Concrete gates on range bars, the following pipeline eliminates magic numbers across the entire WFO workflow:

| Problem                 | Magic-Number Solution  | Parameter-Free Alternative  | Library       |
| ----------------------- | ---------------------- | --------------------------- | ------------- |
| Window length           | Fixed 252 days         | ADWIN adaptive sizing       | `river`       |
| Number of regimes       | Pre-specified k        | TDA persistence clustering  | `giotto-tda`  |
| Significance threshold  | p < 0.05 fixed         | E-value > 20 anytime-valid  | `expectation` |
| Sharpe significance     | Arbitrary track record | MinTRL with higher moments  | `quantstats`  |
| Prediction uncertainty  | Point estimates        | Conformal intervals         | `MAPIE`       |
| Regime change detection | Rolling correlation    | BOCPD posterior probability | `changepoint` |
| Signal quality          | Rolling Sharpe         | Permutation Entropy         | `antropy`     |

The key insight is combining these methods: use **ADWIN** to dynamically size WFO windows, **BOCPD** to trigger re-optimization when regime probability exceeds 0.8, **e-values** to sequentially test strategy edge without fixed sample sizes, and **persistence landscapes** to validate that strategy performance is regime-independent. This creates a fully adaptive system where evaluation horizons emerge from the data rather than researcher intuition.

---

## Conclusion

The "time-to-convergence" problem in algorithmic trading has multiple mathematically principled solutions that eliminate arbitrary parameter choices. **E-values and confidence sequences** provide the strongest theoretical foundation for determining when enough data exists, maintaining validity under any stopping rule. **ADWIN and BOCPD** automatically determine window sizes through online change detection. **Topological data analysis** detects regimes without pre-specifying their number by leveraging scale-invariant persistence features. **Information-theoretic metrics** like permutation entropy provide distribution-free alternatives to Sharpe that work on short samples. The combination of `expectation` for sequential testing, `river` for adaptive windowing, `giotto-tda` for regime detection, and `MAPIE` for uncertainty quantification creates a complete framework for parameter-free strategy evaluation—particularly valuable for cryptocurrency markets where fixed historical windows fail to capture rapid regime dynamics.
