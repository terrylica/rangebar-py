---
source_url: https://claude.ai/public/artifacts/49769365-f2e6-4b24-a0b9-690147bf1e59
source_type: claude-artifact
scraped_at: 2026-02-02T08:00:27Z
purpose: SOTA methods for enriching range bars with tick-level microstructure features
tags:
  [
    intra-bar-features,
    microstructure,
    path-signatures,
    order-flow,
    hawkes-process,
  ]

# REQUIRED provenance
model_name: Claude Opus 4
model_version: claude-opus-4-5-20251101
tools: []

# REQUIRED for Claude Code backtracking + context
claude_code_uuid: 3d0512c5-fe15-4d74-86cc-b2e6ece321e9
claude_code_project_path: "~/.claude/projects/-Users-terryli-eon-alpha-forge-worktree-2025-12-24-ralph-ml-robustness-research/3d0512c5-fe15-4d74-86cc-b2e6ece321e9"

# REQUIRED backlink metadata (to be filled after issue creation)
github_issue_url: "https://github.com/terrylica/rangebar-py/issues/59"
---

# SOTA methods for enriching range bars with tick-level microstructure features

The most promising approaches for extracting intra-bar features from tick data to enrich cryptocurrency range bars combine **path signatures** for capturing price path dynamics, **set-based neural architectures** for handling variable tick counts, and **established microstructure metrics** (order flow imbalance, VPIN, Hawkes intensity) that directly leverage Binance's `is_buyer_maker` flag. Path signatures stand out as particularly powerful because they naturally handle variable-length sequences while producing fixed-dimensional features that capture the complete "shape" of how a bar formed. For practical implementation, the **tick** library for Hawkes processes, **signatory** for GPU-accelerated path signatures, and **polars** for high-performance aggregation form a production-ready stack that can process millions of bars efficiently.

The core challenge you face—100 dbps bars have signal but fail on costs, while 1000 dbps bars survive costs but may lack signal—has a direct solution in the microstructure literature: **extract the information content present at finer resolution and aggregate it into coarser-resolution features**. This is precisely what path signatures, volume profile statistics, and order flow features accomplish.

## Path signatures capture how price paths unfold within bars

**Path signatures** from rough path theory (Terry Lyons, Oxford) represent the mathematically optimal way to extract features from sequential data of variable length. The signature of a path X:[0,T]→ℝᵈ is an infinite series of iterated integrals that captures all relevant information about the path's shape, ordering, and dynamics. For practical use, **truncated signatures at depth 3-4** capture sufficient information while remaining computationally tractable.

The key insight for range bar enrichment is that path signatures are **reparametrization invariant**—they don't care about time, only about the geometric path traversed. This means two bars that both move 1% but do so differently (steady drift versus spike-and-reversal) will have distinct signatures. The **lead-lag transformation** converts a 1D price stream into a 2D path whose signed area encodes momentum and mean-reversion characteristics.

For implementation, **signatory** (PyTorch, GPU-accelerated) provides **132× speedup over CPU alternatives** and supports automatic differentiation for end-to-end training. The recommended configuration for tick-level crypto data: create a multivariate path with columns for (time_normalized, log_price, log_volume, order_imbalance), apply lead-lag transformation, compute log-signature at depth 3-4. This produces **39-120 fixed-dimensional features** regardless of whether the bar formed from 10 or 10,000 ticks.

python`# Signatory usage pattern
import signatory, torch
path = torch.tensor(augmented_ticks).unsqueeze(0).float()  # (1, n_ticks, d)
logsig = signatory.logsignature(path, depth=4)  # Fixed-size output`
The seminal reference is Chevyrev & Kormilitzin's "A Primer on the Signature Method in Machine Learning" (2016, arXiv:1603.03788). For financial applications specifically, see Lyons et al. (2014) "A feature set for streams and an application to high-frequency financial tick data" and Kalsi et al. (2020) "Optimal Execution with Rough Path Signatures" (SIAM Journal on Financial Mathematics).

## Order flow features directly exploit Binance's trade classification

Your aggTrade data includes `is_buyer_maker`, which indicates trade direction with **100% accuracy**—a major advantage over markets where Lee-Ready classification achieves only 75-85% accuracy. This enables precise computation of order flow metrics that predict short-term price movements.

**Order Flow Imbalance (OFI)** is the foundational metric: `OFI = (BuyVolume - SellVolume) / TotalVolume`. Compute this per bar and as rolling averages across bars. The academic foundation comes from Cont, Kukanov & Stoikov (2014) "The price impact of order book events." Implementation is trivial—a single pandas groupby operation—with **O(n)** complexity.

**VPIN (Volume-Synchronized PIN)** from Easley, López de Prado & O'Hara (2012) measures "flow toxicity" by tracking volume imbalances across fixed-volume buckets. With true trade classification from `is_buyer_maker`, your VPIN estimates will be substantially more accurate than implementations using bulk volume classification. However, note Andersen & Bondarenko's (2014) critique: VPIN correlates mechanically with volatility and volume, so **control for these when using as ML features**. The **flowrisk** library (`pip install flowrisk`) provides a production-ready recursive VPIN estimator.

**Kyle's lambda** measures price impact per unit of signed order flow: `Δp = λ × sign(v) × √|v|`. Estimate via regression within each bar or rolling window. The **frds** package (`pip install frds`) implements this directly: `frds.measures.kyle_lambda(returns, signed_dollar_volume)`. Higher lambda indicates lower liquidity and greater information asymmetry.

For microstructure noise estimation, **Two-Scale Realized Variance (TSRV)** from Zhang, Mykland & Aït-Sahalia (2005) separates true volatility from microstructure noise by comparing variance at different sampling frequencies. The noise component itself is an informative feature—high noise-to-signal ratios indicate market maker activity and potential mean-reversion.

MetricSeminal PaperImplementationComplexityOFICont et al. (2014)pandas nativeO(n), very lowVPINEasley et al. (2012)flowriskO(n), lowKyle's λKyle (1985)frdsO(n), lowTSRVZhang et al. (2005)custom ~30 linesO(n), mediumPINEasley et al. (1996)PINstimation (R)O(D×iter), high

## Volume profile and temporal features reveal bar formation dynamics

**Point of Control (POC)** is the price level with maximum volume within the bar—a concept from Market Profile (Steidlmayer, CBOT 1985). Compute by discretizing prices to tick size, summing volume per level, taking argmax. The POC's position relative to the bar's high/low reveals whether accumulation occurred at support or resistance. **Volume distribution entropy** measures whether volume concentrated at few prices (low entropy, potential breakout) or dispersed (high entropy, balanced trading).

**VWAP deviation from midpoint** captures directional pressure: if VWAP is above the bar's midpoint, buyers were more aggressive at higher prices. The full **volume-weighted distribution statistics** (mean, variance, skewness, kurtosis) provide a complete characterization of the intra-bar volume profile.

**Cumulative delta** (buy volume minus sell volume, running sum) tracks the net aggressor flow throughout bar formation. Its trajectory—monotonic, oscillating, or reversing—encodes distinct microstructure regimes.

python`# Volume profile computation
def volume_profile_features(trades):
total_vol = trades['quantity'].sum()
vwap = (trades['price'] \* trades['quantity']).sum() / total_vol
midpoint = (trades['price'].max() + trades['price'].min()) / 2

    # Volume-weighted moments
    var = ((trades['price'] - vwap)**2 * trades['quantity']).sum() / total_vol
    skew = ((trades['price'] - vwap)**3 * trades['quantity']).sum() / (total_vol * var**1.5)

    # POC
    vol_by_price = trades.groupby(trades['price'].round(2))['quantity'].sum()
    poc = vol_by_price.idxmax()
    entropy = -((vol_by_price/total_vol) * np.log(vol_by_price/total_vol + 1e-10)).sum()

    return {'vwap': vwap, 'vwap_deviation': (vwap - midpoint)/midpoint,
            'vol_std': np.sqrt(var), 'vol_skew': skew, 'poc': poc, 'vol_entropy': entropy}`

## Hawkes processes model trade arrival intensity and clustering

**Hawkes processes** are self-exciting point processes where each event increases the probability of subsequent events—exactly matching the empirical observation that trades cluster in bursts. The intensity function is `λ(t) = μ + Σ α×exp(-β×(t-tᵢ))`, where μ is baseline intensity, α is excitation amplitude, and β is decay rate.

For range bars, estimate Hawkes parameters per bar to capture **trade arrival dynamics**: high α/β ratio indicates strong clustering (potentially informed trading), while parameters close to a Poisson process suggest random flow. The **tick** library (`pip install tick`) from X-DataInitiative provides C++-accelerated estimation supporting exponential, sum-of-exponential, and power-law kernels. For bivariate Hawkes (separate buy/sell streams), the cross-excitation matrix reveals whether buys trigger more buys (momentum) or sells (mean-reversion).

**Burstiness parameter** (Goh & Barabási, 2008) provides a simpler alternative: `B = (σ_τ - μ_τ)/(σ_τ + μ_τ)` where τ is inter-trade time. B ranges from -1 (periodic) through 0 (Poisson) to +1 (maximally bursty). The **memory coefficient** measures autocorrelation of consecutive inter-trade times. Both are computationally trivial—just numpy operations on the diff of timestamps.

**Formation speed** (bar duration) is itself a powerful feature for variable-duration bars. Fast-forming bars indicate high activity and potentially stronger signals. Normalize other features by duration to create intensity metrics (volume/second, trades/second).

## Set functions handle variable tick counts elegantly

The fundamental challenge of variable tick counts per bar (10 versus 10,000 ticks forming one bar) has elegant solutions from the deep learning literature on **set functions**.

**DeepSets** (Zaheer et al., NeurIPS 2017) proves that any permutation-invariant function can be decomposed as `f(X) = ρ(Σ φ(xᵢ))`: encode each element independently, sum, then decode. For tick aggregation: each tick gets encoded by a shared MLP φ, embeddings are summed (or averaged), and ρ produces fixed-size bar features. Implementation is **5-10 lines of PyTorch** with complexity O(n) in tick count.

**Set Transformer** (Lee et al., ICML 2019) extends this with attention mechanisms to capture **pairwise interactions between ticks**. The Induced Set Attention Block (ISAB) uses **m learnable inducing points** to reduce complexity from O(n²) to O(mn), enabling scaling to thousands of ticks per bar. Pooling by Multihead Attention (PMA) provides learnable aggregation superior to simple sum/mean.

For your BiLSTM SelectiveNet architecture, a natural integration is: Set Transformer aggregates ticks → bar embedding, concatenate with OHLCV features, feed sequence of enriched bars to BiLSTM. The official implementation is at `github.com/juho-lee/set_transformer`.

**Temporal Fusion Transformer (TFT)** from Google (Lim et al., 2021) handles multiple input types and automatically learns which features matter via Variable Selection Networks. While more complex to implement, TFT provides interpretability (attention weights reveal which features and timesteps drive predictions). The **pytorch-forecasting** library (`pip install pytorch-forecasting`) provides a production-ready implementation.

## López de Prado's framework provides theoretical grounding for range bars

Marcos López de Prado's "Advances in Financial Machine Learning" (Wiley, 2018) Chapter 2 establishes the theoretical case against time bars and for **information-driven sampling**. The core argument: markets process information at irregular rates, so fixed-time sampling oversamples quiet periods and undersamples active ones, producing poor statistical properties (serial correlation, non-normality, heteroscedasticity).

**Range bars** (invented by Vicente Nicolellis, 1990s) sample based on price movement threshold—closely related to López de Prado's dollar/volume/tick bars that sample on market activity. Range bars exhibit **reduced serial correlation** and **clearer trend identification** compared to time bars. Your 1000 dbps bars (1% moves) are relatively large; the literature suggests 1/50 of daily range as a starting threshold, adjusted for your signal-to-cost tradeoff.

**Tick Imbalance Bars (TIB)** and **Dollar Imbalance Bars (DIB)** from Chapter 2 take this further: sample when the cumulative signed trade flow exceeds expected imbalance. This directly samples at moments of information arrival. Consider implementing as an alternative to fixed-range bars.

**Triple Barrier Labeling** (Chapter 3) provides superior training labels for ML: instead of predicting next-bar return, define profit-taking (upper barrier), stop-loss (lower barrier), and time expiration (vertical barrier). The label reflects which barrier is touched first—directly corresponding to realistic trade outcomes.

For implementation, **mlfinlab** from Hudson & Thames was the reference library but is now **commercial** (£100+/month via QuantConnect). The open-source **mlfinpy** (`pip install mlfinpy`) provides community reimplementations. For high-performance bar construction, the **polars-trading** plugin (`github.com/ngriffiths13/polars-trading`) implements tick/volume/dollar bars with polars' 10-100× speedup over pandas.

## Hierarchical architectures match the tick-to-bar-to-prediction structure

Your prediction problem has natural hierarchy: ticks aggregate to bars, bars form sequences for prediction. **Hierarchical Attention Networks (HAN)** explicitly model this structure with tick-level attention → bar embeddings → bar-level attention → prediction.

For irregular time series (variable bar durations), **Multi-Time Attention Network (mTAN)** from Shukla & Marlin (ICLR 2021) learns continuous-time embeddings rather than positional encodings, handling irregular sampling naturally. Implementation at `github.com/reml-lab/mTAN`.

**Neural Controlled Differential Equations (Neural CDEs)** from Kidger et al. (NeurIPS 2020) model continuous-time dynamics `dz/dt = g_θ(z) × dX/dt`, incorporating the incoming signal path. This provides principled handling of irregular observations but with higher computational cost. The **torchcde** library supports this.

**Wavelet decomposition** offers a simpler multi-scale approach: apply Discrete Wavelet Transform (PyWavelets library) to decompose price/volume into frequency bands, extract features from each scale. Studies show **15%+ RMSE improvement** over vanilla LSTM for financial prediction. The **pytorch-wavelets** library provides GPU-accelerated, differentiable wavelets for end-to-end training.

## Recommended feature engineering pipeline for production

Based on this research, the optimal pipeline for enriching 1000 dbps range bars with tick-level features:

**Tier 1 - Essential features (low complexity, high value):**

Order Flow Imbalance (OFI) using `is_buyer_maker`
Cumulative delta and its trajectory (slope, reversals)
VWAP, VWAP deviation from midpoint
Formation duration and volume/time intensity
Tick count per bar
Volume-weighted price moments (mean, std, skew)

**Tier 2 - Microstructure features (medium complexity):**

VPIN via flowrisk library, with volatility controls
Volume profile entropy
POC position relative to bar high/low
Burstiness parameter and memory coefficient of inter-trade times
Trade size distribution statistics

**Tier 3 - Advanced features (higher complexity):**

Path signatures (depth 3-4) via signatory, with lead-lag transformation
Hawkes process parameters (baseline intensity, excitation, decay) via tick library
Kyle's lambda estimated per rolling window
TSRV for noise-to-signal decomposition

**Aggregation architecture:**
For your BiLSTM SelectiveNet, prepend a **Set Transformer encoder** (16-32 inducing points) that converts variable-length tick sequences into fixed-size bar embeddings. Concatenate with pre-computed statistical features (Tier 1-2) before the LSTM layers. This hybrid approach captures both learned tick interactions and interpretable microstructure metrics.

**Computational budget** for 1 million bars with average 500 ticks each:

Tier 1 features: ~30 seconds (polars groupby)
Tier 2 features: ~2 minutes (vectorized numpy)
Path signatures: ~10 minutes (signatory GPU)
Hawkes estimation: ~30 minutes (tick library, parallelized)

## Conclusion

The signal present in 100 dbps bars can be partially recovered at 1000 dbps resolution through systematic extraction of **how each bar formed** rather than just its OHLCV summary. Path signatures provide the mathematically principled approach to this, while order flow metrics directly exploit your trade classification advantage. The combination of DeepSets/Set Transformer for tick aggregation with established microstructure features creates a feature space that preserves fine-grained information within a cost-feasible bar structure.

Key implementation priorities: start with Tier 1 features (immediate value, minimal effort), validate improvement on your BiLSTM before adding complexity. Path signatures via signatory should be the first advanced feature to test given their theoretical properties and computational efficiency. For production scale, polars for aggregation and numba for custom metrics will handle millions of bars efficiently.
