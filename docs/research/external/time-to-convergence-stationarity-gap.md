---
source_url: https://gemini.google.com/share/e78016a0eb8e
source_type: gemini-3-pro
scraped_at: 2026-02-01T00:54:05+00:00
purpose: Advanced quantitative finance methodology for regime-adaptive trading and OOD robustness validation
tags:
  [
    stationarity,
    regime-detection,
    adwin,
    probabilistic-sharpe-ratio,
    topological-data-analysis,
    fisher-information,
  ]
---

# Forensic Analysis: The "Time-to-Convergence" & Stationarity Gap

===============================================================

## The Epistemological Crisis of Stationarity in Quantitative Finance

The foundational edifice of modern quantitative finance rests precariously on a single, mathematically convenient, yet empirically fragile assumption: **Stationarity**. From the derivation of the Black-Scholes option pricing formula to the mean-variance optimization of Modern Portfolio Theory (MPT), the discipline assumes that the statistical properties of the underlying data generating process (DGP)—specifically the mean (μ), variance (σ2), and autocorrelation structure—remain constant over time, or at least evolve according to a stationary deterministic law. This assumption permits the invocation of the Law of Large Numbers (LLN) and the Central Limit Theorem (CLT), assuring the practitioner that with a sufficient sample size T, the sample estimator θ^ will converge to the true population parameter θ∗.

However, financial markets are not physical systems governed by immutable constants like the speed of light or the gravitational constant. They are complex, adaptive socio-technical systems characterized by feedback loops, structural breaks, and regime shifts. In this reality, stationarity is not a property of the data; it is a fleeting window of stability in a chaotic stream. This discrepancy creates a critical "blind spot" in risk management and signal processing: the **Stationarity Gap**.

The Stationarity Gap is defined as the divergence between the theoretical **Time-to-Convergence** (the sample size T required for an estimator to achieve statistical significance given the noise and non-normality of the data) and the **Regime Duration** (the actual time the market remains in a state where those parameters are valid). When the Time-to-Convergence exceeds the Regime Duration, the "Minimum Reliable Horizon" (MRH) is effectively infinite relative to the stability of the system. In such scenarios, standard quantitative metrics are not merely inaccurate; they are illusory. A Sharpe ratio calculated over a 3-year period for a strategy with a 5-year convergence horizon due to high kurtosis is statistically indistinguishable from a random walk, yet it is often treated as a valid performance metric.

This report serves as a forensic analysis of this gap. It rejects the use of arbitrary parameters—"magic numbers" like 20-day moving averages or fixed 0.05 p-value thresholds—which impose an artificial timescale on the market. Instead, it synthesizes a new framework for **Regime-Adaptive Algorithmic Trading** based on **Parameter-Free** metrics. We conduct an exhaustive survey of State-of-the-Art (SOTA) methodologies across four distinct mathematical domains:

1. **Adaptive Statistics:** specifically **Adaptive Windowing (ADWIN)**, which solves the "supply" side of the problem by dynamically determining the maximum length of history that is statistically consistent with the present.  

2. **Probabilistic Performance:** specifically the **Probabilistic Sharpe Ratio (PSR)**, **Minimum Track Record Length (MinTRL)**, and **Deflated Sharpe Ratio (DSR)**, which solve the "demand" side by quantifying exactly how much data is needed for significance given the higher moments (skewness, kurtosis) of the return distribution.  

3. **Topological Data Analysis (TDA):** utilizing **Persistent Homology** and **Persistence Landscapes** to detect geometric structural breaks that precede statistical moment shifts, offering an early warning system for regime collapse.  

4. **Information Geometry:** employing the **Fisher Information Metric (FIM)** and **Natural Gradient Descent (NGD)** to measure the "velocity" of distributional change and adapt learning rates to the curvature of the statistical manifold.  

By integrating these tools, we propose the **Minimum Reliable Horizon (MRH)** not merely as a concept, but as a calculable, dynamic metric that serves as the ultimate stress test for quantitative strategies in a non-stationary world.

---

## 1\. Adaptive Windowing (ADWIN): The Engine of Local Stationarity

### 1.1 The Failure of Fixed-Window Architectures

In standard practice, non-stationarity is handled heuristically via sliding windows (e.g., rolling 60-day volatility) or exponential decay (e.g., EWMA with λ\=0.94). These approaches introduce a fatal fragility: the **Memory Dilemma**.

- **Small Window / Fast Decay:** The estimator adapts quickly to new regimes but suffers from high variance (noise) in stable regimes.

- **Large Window / Slow Decay:** The estimator is robust (low variance) in stable regimes but suffers from high bias (lag) during transitions.

This trade-off is typically managed by optimizing the window size parameter (W) over historical data. However, this optimization assumes that the _frequency of regime shifts_ is itself stationary—that "60 days" is the eternal "natural frequency" of the market. Empirical reality contradicts this; markets exhibit multifractal behavior where stability exists on varying timescales.  

### 1.2 Theoretical Foundations of ADWIN

**Adaptive Windowing (ADWIN)**, developed by Bifet and Gavaldà, provides a rigorous, parameter-free solution to the Memory Dilemma. Unlike heuristics, ADWIN is grounded in learning theory bounds. It maintains a window W of variable size and automatically detects "concept drift" (distributional shift).  

The core mechanism is a hypothesis test performed on _every possible partition_ of the window. Let W be a sequence of real-valued observations x1​,x2​,…,xn​. ADWIN checks every split of W into two sub-windows W0​ (historical) and W1​ (recent) such that W\=W0​⋅W1​.

The algorithm compares the empirical means μ^​W0​​ and μ^​W1​​. The null hypothesis H0​ is that both sub-windows are drawn from the same distribution with mean μ. If the absolute difference ∣μ^​W0​​−μ^​W1​​∣ exceeds a theoretically derived threshold ϵcut​, the null hypothesis is rejected. The older data W0​ is deemed "stale" (belonging to a prior regime) and is discarded. The window shrinks to W1​, and the process repeats.

#### 1.2.1 The Hoeffding Bound Derivation

The "magic-number-free" nature of ADWIN stems from its definition of ϵcut​. It utilizes the **Hoeffding Bound**, a concentration inequality that provides statistical guarantees for the deviation of the sum of independent random variables from their expected value, assuming only that the variables are bounded (typically normalized to $$).

The threshold ϵcut​ is defined as:

ϵcut​\=2m1​⋅ln(δ4∣W∣​)![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="3.08em" viewBox="0 0 400000 3240" preserveAspectRatio="xMinYMin slice"><path d="M473,2793c339.3,-1799.3,509.3,-2700,510,-2702 l0 -0c3.3,-7.3,9.3,-11,18,-11 H400000v40H1017.7s-90.5,478,-276.2,1466c-185.7,988,-279.5,1483,-281.5,1485c-2,6,-10,9,-24,9c-8,0,-12,-0.7,-12,-2c0,-1.3,-5.3,-32,-16,-92c-50.7,-293.3,-119.7,-693.3,-207,-1200c0,-1.3,-5.3,8.7,-16,30c-10.7,21.3,-21.3,42.7,-32,64s-16,33,-16,33s-26,-26,-26,-26s76,-153,76,-153s77,-151,77,-151c0.7,0.7,35.7,202,105,604c67.3,400.7,102,602.7,104,606zM1001 80h400000v40H1017.7z"></path></svg>)​

Where:

- m is the harmonic mean of the lengths of the sub-windows: m\=1/∣W0​∣+1/∣W1​∣1​.

- ∣W∣ is the total length of the current window.

- δ is the confidence parameter (e.g., 10−3), representing the maximum probability of a False Positive (detecting drift when none exists).  

**Implications for Financial Time Series:**

1. **Dynamic Sensitivity:** The term ln(4∣W∣/δ) implies that as the window grows larger (accumulating more data in a stable regime), the threshold ϵcut​ increases slightly, but the denominator 2m (proportional to N) decreases the threshold significantly. This means ADWIN becomes _more sensitive_ to small drifts as it accumulates more history. It rigorously balances the confidence in the current mean against the evidence of a shift.  

2. **Rigorous Guarantees:** ADWIN guarantees that the mean of the window is statistically consistent with the current regime. If a regime shift of magnitude Δμ occurs, ADWIN will detect it and shrink the window within O(Δμ21​lnδ1​) steps.  

### 1.3 Algorithmic Architecture and "Bucket" Compression

A naive implementation of ADWIN would store all data points in W and recompute means for all splits, resulting in O(W2) or O(W) complexity per update, which is unacceptable for high-frequency trading (HFT). The SOTA implementation (found in libraries like `river` and `scikit-multiflow`) uses an **Exponential Histogram** data structure.  

The window is stored as a sequence of "buckets." Each bucket contains a summary of a specific number of original data points (2k).

- The system maintains a list of buckets of size 20 (1 point), 21 (2 points), 22 (4 points), etc.

- There is a maximum number of buckets (M) allowed per size level (typically M\=5).

- When a new data point arrives, it is added as a size 20 bucket. If the number of 20 buckets exceeds M, the two oldest are merged into a 21 bucket. This cascade can propagate up to larger sizes.

This structure reduces the memory and time complexity to O(logW). It allows ADWIN to maintain an effective window of millions of data points while only performing a few hundred operations per update. This is critical for forensic analysis of tick data, where "Time-to-Convergence" is measured in milliseconds.  

### 1.4 Forensic Application: The ADWIN-Vol and ADWIN-Corr

In our forensic framework, ADWIN is not just a preprocessing step; it is a primary signal generator.

#### 1.4.1 ADWIN-Vol (Adaptive Volatility)

Instead of feeding raw returns rt​ (which have a mean near zero), we feed **squared returns** rt2​ or **absolute returns** ∣rt​∣ into ADWIN. The algorithm maintains a window Wvol​ representing the period over which the variance is stationary.

- **The Estimator:** σ^t​\=∣Wvol​∣1​∑i∈Wvol​​ri2​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.88em" viewBox="0 0 400000 1944" preserveAspectRatio="xMinYMin slice"><path d="M983 90l0 -0c4,-6.7,10,-10,18,-10 H400000v40H1013.1s-83.4,268,-264.1,840c-180.7,572,-277,876.3,-289,913c-4.7,4.7,-12.7,7,-24,7s-12,0,-12,0c-1.3,-3.3,-3.7,-11.7,-7,-25c-35.3,-125.3,-106.7,-373.3,-214,-744c-10,12,-21,25,-33,39s-32,39,-32,39c-6,-5.3,-15,-14,-27,-26s25,-30,25,-30c26.7,-32.7,52,-63,76,-91s52,-60,52,-60s208,722,208,722c56,-175.3,126.3,-397.3,211,-666c84.7,-268.7,153.8,-488.2,207.5,-658.5c53.7,-170.3,84.5,-266.8,92.5,-289.5zM1001 80h400000v40h-400000z"></path></svg>)​.

- **The Meta-Signal:** The window length ∣Wvol​∣ itself.
  - If ∣Wvol​∣ is expanding, the volatility regime is stable. The "Time-to-Convergence" is being satisfied.

  - If ∣Wvol​∣ collapses (e.g., from 1000 days to 20 days), it indicates a **Volatility Shock**. The previous volatility estimate is no longer valid. The MRH has reset.

#### 1.4.2 ADWIN-Corr (Adaptive Correlation)

For pairs trading or portfolio optimization, stationarity of the correlation matrix is the "Achilles' heel." By feeding the product of standardized returns zX,t​⋅zY,t​ into ADWIN, we can detect **Correlation Breakdown**. Traditional rolling correlations often "ghost" (show high correlation long after the relationship has broken due to the window lag). ADWIN-Corr instantly discards the history once the covariance product distribution shifts, preventing the strategy from hedging with a broken instrument.  

### 1.5 The "Supply" of Stationarity

In the context of the MRH Framework, ADWIN defines the **Supply Side**.

Tavailable​(t)\=∣WADWIN​(t)∣

This is the maximum number of data points available at time t that can be essentially treated as Independent and Identically Distributed (IID) for the purpose of inference. If this supply is lower than the "demand" (MinTRL), the system is blind.

---

## 2\. Forensic Performance Measurement: The Probabilistic Paradigm

### 2.1 The Limits of the Standard Sharpe Ratio

While ADWIN measures the stability of the data, we must also measure the stability of the _strategy performance_. The Sharpe Ratio (SR), defined as σμ−rf​​, is the industry standard. However, the standard implementation of SR testing relies on a critical fallacy: the assumption of Normality.

Financial returns are characteristically leptokurtic (fat-tailed) and skewed. A strategy that sells out-of-the-money put options will exhibit a high Sharpe Ratio with small positive returns and rare, massive negative returns (negative skewness). Under the assumption of Normality, the standard error of the SR estimator is:

SE(SR^)≈T1+0.5SR2​![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="2.48em" viewBox="0 0 400000 2592" preserveAspectRatio="xMinYMin slice"><path d="M424,2478c-1.3,-0.7,-38.5,-172,-111.5,-514c-73,-342,-109.8,-513.3,-110.5,-514c0,-2,-10.7,14.3,-32,49c-4.7,7.3,-9.8,15.7,-15.5,25c-5.7,9.3,-9.8,16,-12.5,20s-5,7,-5,7c-4,-3.3,-8.3,-7.7,-13,-13s-13,-13,-13,-13s76,-122,76,-122s77,-121,77,-121s209,968,209,968c0,-2,84.7,-361.7,254,-1079c169.3,-717.3,254.7,-1077.7,256,-1081l0 -0c4,-6.7,10,-10,18,-10 H400000v40H1014.6s-87.3,378.7,-272.6,1166c-185.3,787.3,-279.3,1182.3,-282,1185c-2,6,-10,9,-24,9c-8,0,-12,-0.7,-12,-2z M1001 80h400000v40h-400000z"></path></svg>)​

This formula underestimates the error for negatively skewed strategies. It leads to **Sharpe Ratio Inflation**, where a strategy appears significant only because the "Time-to-Convergence" calculation (the denominator) assumed a Normal distribution that converges much faster than the actual fat-tailed distribution.  

### 2.2 The Probabilistic Sharpe Ratio (PSR)

Bailey and Lopez de Prado (2012) derived the correct standard error for the Sharpe Ratio under non-Normal conditions using a higher-order expansion. The variance of the Sharpe Ratio estimator is given by:

V\=T−11​(1−γ3​SR^+4γ4​−1​SR^2)

Where:

- γ3​ is the Skewness of returns.

- γ4​ is the Kurtosis of returns (Fisher's definition, Normal=3).

Notice the term −γ3​SR^. If a strategy has **negative skewness** (e.g., γ3​\=−2.5) and a positive Sharpe Ratio, this term becomes positive and _increases_ the variance. This quantifies the intuition that "slow up, fast down" strategies are statistically harder to validate.

The **Probabilistic Sharpe Ratio (PSR)** is the probability that the true Sharpe Ratio SR∗ exceeds a benchmark SRbm​ (e.g., 0):

\\widehat{PSR}(SR\_{bm}) = Z \\left

Where Z\[⋅\] is the Cumulative Distribution Function (CDF) of the Standard Normal distribution. The PSR serves as a **Resilient Metric**. A strategy with SR=2.0 but Skewness=-3.0 might have a lower PSR than a strategy with SR=1.5 and Skewness=0. The PSR correctly identifies that the former strategy has not yet converged to significance despite the higher nominal return.  

### 2.3 Minimum Track Record Length (MinTRL)

Inverting the PSR formula allows us to solve for T, the sample size. This yields the **Minimum Track Record Length (MinTRL)**, which represents the "Demand Side" of our forensic analysis:

MinTRL = 1 + \\left \\left( \\frac{Z\_\\alpha}{\\hat{SR} - SR\_{bm}} \\right)^2

This metric defines the **Minimum Reliable Horizon (MRH)** for the strategy. It answers the question: _"Given the non-normality of these returns, how many data points are required to reject the null hypothesis that the strategy is luck at the α confidence level?"_.  

**Table 1: MinTRL Sensitivity to Non-Normality (Target SR=1.0, Benchmark=0, Conf=95%)**

| Strategy Profile      | Skewness (γ3​) | Kurtosis (γ4​) | MinTRL (Years) | Interpretation                         |
| --------------------- | -------------- | -------------- | -------------- | -------------------------------------- |
| **Long-Short Equity** | 0.0            | 3.0 (Normal)   | **~3.2**       | Standard convergence.                  |
| **Trend Following**   | +0.5           | 4.0            | **~2.8**       | Positive skew accelerates validation.  |
| **Mean Reversion**    | \-1.5          | 6.0            | **~6.5**       | Negative skew doubles required time.   |
| **Short Volatility**  | \-4.0          | 15.0           | **~18.4**      | **Unvalidatable in standard regimes.** |

Export to Sheets

The "Short Volatility" example illustrates the **Stationarity Gap** perfectly. The strategy requires 18 years of stationary data to prove its skill. However, volatility regimes rarely last 18 years. Therefore, the strategy operates permanently in the blind spot; its performance is theoretically unknowable until it blows up.

### 2.4 The Deflated Sharpe Ratio (DSR) and Backtest Overfitting

The PSR/MinTRL framework addresses non-normality but ignores **Selection Bias**. Modern quantitative research involves running thousands of backtests (Hyperparameter Tuning). If a researcher tests 1,000 parameter combinations, the "best" result is expected to be high purely by chance. This is the problem of **Backtest Overfitting (PBO)**.  

The **Deflated Sharpe Ratio (DSR)** corrects for this by adjusting the benchmark SRbm​ using the expected maximum of N independent trials. Under Extreme Value Theory (EVT), the expected maximum of N independent standard normal variables is approximated by 2lnN![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429c69,-144,104.5,-217.7,106.5,-221l0 -0c5.3,-9.3,12,-14,20,-14H400000v40H845.2724s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47zM834 80h400000v40h-400000z"></path></svg>)​.

DSR≡PSR(SRbm​\=2lnN![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429c69,-144,104.5,-217.7,106.5,-221l0 -0c5.3,-9.3,12,-14,20,-14H400000v40H845.2724s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47zM834 80h400000v40h-400000z"></path></svg>)​⋅σSRtrials​​)

**Implementation Strategy:** To calculate DSR forensically, one must track the "Number of Trials" (N).

1. **Online Context:** In an automated system (e.g., using `quantstats` or `pypbo`), every strategy configuration tested counts as a trial.  

2. **Effective N:** Since trials are often correlated (e.g., Moving Average 50 vs Moving Average 51), simply counting backtests over-penalizes. The "Effective Number of Independent Trials" can be estimated by analyzing the correlation matrix of the backtest returns using clustering algorithms (e.g., K-Means or Varimax rotation).  

The DSR is the ultimate "Gatekeeper." It tells us: _"Even with this skewness and kurtosis, is the return high enough to overcome the probability that you simply data-mined this pattern from N attempts?"_

---

## 3\. The Geometry of Market Regimes: Topological Data Analysis (TDA)

While ADWIN and PSR focus on statistical moments, **Topological Data Analysis (TDA)** focuses on the **shape** of the data. Markets are dynamic systems that form geometric structures in phase space. Regime shifts often manifest as topological tears or the formation of "voids" (instability) before the mean or variance shifts significantly. TDA offers a SOTA, parameter-free method for early detection of these structural breaks.  

### 3.1 Persistent Homology: A Primer

The core tool of TDA is **Persistent Homology**. It analyzes the connectivity of a dataset at varying resolution scales (filtration).

1. **Point Cloud Embedding:** The time series rt​ is embedded into a d\-dimensional space using a sliding window (Takens' Embedding): vt​\=(rt​,rt−τ​,…,rt−(d−1)τ​). This converts the 1D series into a cloud of points in Rd.  

2. **Simplicial Complexes:** We place a ball of radius ϵ around each point. As ϵ increases, balls overlap, forming edges, triangles, and tetrahedrons (Simplicial Complexes, typically the Vietoris-Rips complex).

3. **Homological Features:** We track the appearance (birth) and disappearance (death) of topological features:
   - **H0​ (Components):** Connected clusters of points.

   - **H1​ (Loops):** Cycles or holes in the data. In financial data, loops represent recurrent, cyclic patterns (e.g., mean-reverting oscillations).

   - **H2​ (Voids):** Higher-dimensional hollows.

### 3.2 Persistence Landscapes and Crash Detection

The "persistence" of a feature is its lifespan: Death−Birth. Features with short persistence are noise; those with long persistence are structural signal. This distinction is **parameter-free**: we do not set a "noise threshold"; the topology sorts itself.

For machine learning integration, the collection of persistence intervals is transformed into a functional summary called a **Persistence Landscape** λ(k,t). The landscape is a sequence of continuous, piecewise-linear functions that encode the topological prominence of features.  

**Forensic Insight: The Lp\-Norm Precursor** Gidea and Katz (2018) demonstrated that the **Lp\-Norm** of the Persistence Landscape of H1​ features (loops) is a powerful predictor of crashes.

- **Mechanism:** Prior to a major crash (e.g., 2000 Dot-Com, 2008 Crisis), the market exhibits "Critical Slowing Down." Variance may remain low, but the _structure_ of returns becomes increasingly correlated and cyclic, forming persistent loops in the phase space.

- **The Signal:** The L1 or L2 norm of the landscape rises steadily.

- **Empirical Evidence:** In the 2008 crisis, the Lp\-norm of the persistence landscape showed a strong rising trend for **250 trading days** prior to the Lehman Brothers bankruptcy. This provided a "Time-to-Convergence" warning that statistical volatility models (like VIX) missed until the explosion occurred.  

### 3.3 Implementation and Software

SOTA implementation of TDA for financial time series utilizes libraries such as `GUDHI` (C++ with Python bindings) or `giotto-tda` (`gtda`).  

- **Pipeline:**
  1. `gtda.time_series.SlidingWindow`: Embeds the time series.

  2. `gtda.homology.VietorisRipsPersistence`: Computes persistence diagrams.

  3. `gtda.diagrams.PersistenceLandscape`: Converts diagrams to vector representations.

  4. Compute Norm: Calculate ∥λ∥p​ of the landscape.

- **Regime Detection:** A regime shift is flagged when the time-derivative of the Landscape Norm exceeds a threshold (which can be set adaptively via ADWIN). This combines the geometric sensitivity of TDA with the statistical rigor of ADWIN.

---

## 4\. Information Geometry and the Speed of Learning

The Stationarity Gap can be re-framed as a problem of **Learning Velocity**. If the market distribution changes faster than our algorithm can update its parameters, convergence is impossible. **Information Geometry** provides the tools to measure this velocity and optimize the learning path.

### 4.1 The Fisher Information Metric (FIM)

In standard Euclidean space, the distance between two parameter vectors θ and θ+Δθ is ∥Δθ∥2. However, in the space of probability distributions (Statistical Manifold), this metric is misleading. A change in the mean of a distribution with variance 0.1 is much more significant (distinguishable) than the same change in a distribution with variance 100.

The correct measure of distance on the statistical manifold is the **Fisher Information Metric**:

d2(θ,θ+dθ)\=i,j∑​gij​(θ)dθi​dθj​

Where gij​(θ) is the **Fisher Information Matrix (FIM)**:

gij​(θ)\=E\[(∂θi​∂​logp(x∣θ))(∂θj​∂​logp(x∣θ))\]

The FIM measures the "curvature" of the manifold. High Fisher Information implies high curvature, meaning the data imposes strong constraints on the parameters.

#### 4.1.1 Forensic Metric: Fisher Velocity

We can define the **Fisher Velocity** of the market regime as the rate of change of the distribution along the manifold:

vFisher​(t)\=ΔtdFisher​(pt​,pt−1​)​

A spike in Fisher Velocity indicates that the market is traversing the statistical manifold rapidly. This is a direct measure of **Non-Stationarity Intensity**. If vFisher​ is high, the "Time-to-Convergence" expands because the target (θ∗) is moving faster than the estimator (θ^) can converge.  

### 4.2 Natural Gradient Descent (NGD): Optimal Adaptation

In online learning (e.g., adaptive forecasting models), we update parameters iteratively: θt+1​\=θt​−η∇L. Standard Gradient Descent (SGD) uses the Euclidean gradient, which is sensitive to parameter scaling and inefficient in curved spaces.

**Natural Gradient Descent (NGD)** preconditions the update with the inverse of the FIM:

θt+1​\=θt​−ηF−1∇L(θt​)

**Why NGD solves "Time-to-Convergence":**

1. **Steepest Descent:** NGD moves in the direction of steepest descent _on the manifold_, not in parameter space. It represents the most efficient use of the information contained in the new data point.  

2. **Invariance:** NGD is invariant to parameterization. Whether we model volatility as σ or σ2 or logσ, NGD takes the same path.

3. **Adaptive Learning Rate:** The term F−1 acts as an automatic, optimal learning rate scheduler.
   - When uncertainty is high (low Fisher Info, flat curvature), F−1 is large → Large steps (Fast adaptation).

   - When uncertainty is low (high Fisher Info, sharp curvature), F−1 is small → Small steps (Precise convergence).

Recent research connects NGD to **Online Continual Learning (OCL)** and **Score-Driven Models (GAS)**. In financial time series, using NGD allows the model to "catch up" to regime shifts significantly faster than SGD. For high-dimensional models where F−1 is expensive to compute, approximations like **K-FAC** or block-diagonal approaches are used.  

### 4.3 Entropy and Transfer Entropy

Complementing the FIM are entropy-based metrics.

- **Shannon Entropy / Persistent Entropy:** Measures the "disorder" or flatness of the distribution. A sudden drop in entropy often signals a crystallization of the market into a deterministic trend or crash.  

- **Transfer Entropy (TE):** Measures the directional information flow between time series (e.g., VIX→SPY). Unlike correlation, TE captures non-linear causality. A rise in TE indicates **Systemic Coupling**, often a precursor to contagion events where stationarity assumptions break down across asset classes simultaneously.  

- **Caputo Fractional Fisher Information (CFFI):** A novel metric proposed to incorporate "long memory" (fractality) into the Fisher metric, bridging the gap between geometry and fractal analysis.  

---

## 5\. Fractal Dynamics and Memory Horizons

### 5.1 The Hurst Exponent (H) and Convergence Rates

The standard "Time-to-Convergence" derivation (σ/T![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429c69,-144,104.5,-217.7,106.5,-221l0 -0c5.3,-9.3,12,-14,20,-14H400000v40H845.2724s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47zM834 80h400000v40h-400000z"></path></svg>)​) assumes the IID condition, which implies a Hurst Exponent H\=0.5.

- **H\=0.5:** Random Walk (Brownian Motion).

- **0.5<H<1:** Persistence (Long Memory / Trend).

- **0<H<0.5:** Anti-Persistence (Mean Reversion).

If H\=0.5, the convergence rate changes. For a persistent process (H\>0.5), the variance of the mean estimator scales as T2H−2 rather than T−1.

SE∝T1−Hσ​

If H\=0.8, the standard error decays as T−0.2. This is _excruciatingly_ slow. A strategy might need **decades** of data to converge to the same confidence interval that an IID process reaches in one year.

### 5.2 Robust Estimation: R/S vs. DFA

Estimating H is notoriously difficult on short windows—exactly the context of the MRH stress test.

1. **Rescaled Range (R/S):** The classic Mandelbrot method. It is biased for short samples and sensitive to short-term autocorrelation.  

2. **Detrended Fluctuation Analysis (DFA):** The SOTA method for non-stationary data. It removes polynomial trends (DFA1​, DFA2​) before calculating scaling.

3. **The Short-Window Debate:** While DFA is superior asymptotically, recent studies utilizing the `nolds` and `hurst` Python packages suggest that for very short windows (N<50), a corrected R/S analysis may exhibit lower variance, albeit with bias.  

4. **Generalized Hurst Exponent (GHE):** Based on q\-order moments, GHE is often the most robust parameter-free estimator for financial multifractals.  

**Forensic Application:** We calculate the **Time-Varying Hurst Exponent**. If Ht​ significantly deviates from 0.5, we must adjust the MinTRL calculation. The "effective" sample size Teff​ is related to the actual sample size T by:

Teff​≈T2(1−H)

(Heuristic approximation for effective degrees of freedom). If H\=0.75, Teff​≈T![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429c69,-144,104.5,-217.7,106.5,-221l0 -0c5.3,-9.3,12,-14,20,-14H400000v40H845.2724s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47zM834 80h400000v40h-400000z"></path></svg>)​. We effectively have the square root of the data we thought we had.

---

## 6\. Synthesis: The Minimum Reliable Horizon (MRH) Framework

We now synthesize these disparate SOTA methodologies into a unified, executable framework for forensic analysis. The MRH is not a single number, but the outcome of an inequality between the **Supply of Stationarity** and the **Demand for Significance**.

### 6.1 The MRH Inequality

A quantitative signal is **Forensically Valid** at time t if and only if:

Supply![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="0.548em" viewBox="0 0 400000 548" preserveAspectRatio="xMinYMin slice"><path d="M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z"></path></svg>)![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="0.548em" viewBox="0 0 400000 548" preserveAspectRatio="xMidYMin slice"><path d="M199572 214c100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z"></path></svg>)![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="0.548em" viewBox="0 0 400000 548" preserveAspectRatio="xMaxYMin slice"><path d="M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z"></path></svg>)A(rt​,…,rt−W​)​​≥Demand![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="0.548em" viewBox="0 0 400000 548" preserveAspectRatio="xMinYMin slice"><path d="M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z"></path></svg>)![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="0.548em" viewBox="0 0 400000 548" preserveAspectRatio="xMidYMin slice"><path d="M199572 214c100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z"></path></svg>)![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="0.548em" viewBox="0 0 400000 548" preserveAspectRatio="xMaxYMin slice"><path d="M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z"></path></svg>)M(SR^,γ3​,γ4​,H)​​

Where:

- **A (Supply):** The **Available Stationary Window** determined by **ADWIN**.

  Tavail​\=∣WADWIN​(t)∣

  This measures how far back we can look before the regime changes (Drift Detection).

- **M (Demand):** The **Minimum Reliable Horizon** determined by **MinTRL** and **Fractality**.

  Treq​\=MinTRL(SRobs​,γ3​,γ4​,α)×ϕ(H)

  This measures how much data is required to distinguish the signal from noise, penalized by Skewness (γ3​), Kurtosis (γ4​), and slowed by Long Memory (ϕ(H)).

### 6.2 The Stationarity Gap Dashboard

The **Stationarity Gap** is defined as ΔGap​\=Treq​−Tavail​.

**Table 2: Forensic States of the Stationarity Gap**

| Forensic State        | ADWIN (Tavail​) | MinTRL (Treq​) | Gap Status       | Interpretation                                                           | Action          |
| --------------------- | --------------- | -------------- | ---------------- | ------------------------------------------------------------------------ | --------------- |
| **Converged (Ideal)** | High (500)      | Low (100)      | **Negative**     | Signal is statistically significant within the current regime.           | **Trade**       |
| **The Blind Spot**    | High (500)      | High (600)     | **Positive**     | Regime is stable, but strategy is too risky (fat tails) to validate yet. | **Reduce Size** |
| **Regime Reset**      | Low (20)        | Low (100)      | **Positive**     | Recent drift detected. History invalidated. Waiting for convergence.     | **Halt**        |
| **Geometric Crisis**  | _Any_           | _Any_          | **Forced Reset** | TDA Landscape Norm or Fisher Velocity spikes.                            | **Hard Stop**   |

Export to Sheets

### 6.3 Implementation Algorithm

1. **Data Ingestion:** Stream returns rt​.

2. **Regime Detection (Supply):**
   - Update **ADWIN** with rt2​ (volatility) and pairwise correlations.

   - Output Tavail​\=Wsize​.

   - _Override:_ If **TDA Lp\-Norm** or **Fisher Velocity** exceeds threshold (adaptive via secondary ADWIN), force Tavail​→0.

3. **Signal Analysis (Demand):**
   - Compute online moments: Mean, Var, Skew, Kurt within WADWIN​.

   - Compute online Hurst H (GHE).

   - Calculate DSR (Deflated Sharpe) to adjust benchmark for N trials.

   - Calculate Treq​\=MinTRL.

4. **Decision Logic:**
   - If Tavail​≥Treq​: **Valid Signal.**

   - If Tavail​<Treq​: **Invalid Signal (Stationarity Gap).**
     - The strategy is "guessing." It relies on a stationarity assumption that the data does not support.

### 6.4 Conclusion

The "Time-to-Convergence" problem is the central failure mode of algorithmic trading. By treating stationarity as a default assumption rather than a verifiable hypothesis, practitioners expose themselves to phantom alpha—performance that exists only in the "Stationarity Gap" and vanishes when the regime shifts.

This report establishes that "magic numbers" are unnecessary. The market _tells us_ its reliable horizon through its statistical moments (MinTRL), its drift (ADWIN), its geometry (Fisher Info), and its topology (Persistence Landscapes). By adopting this parameter-free, forensic framework, quantitative research moves from the alchemy of curve-fitting to the science of **Regime-Adaptive** signal processing. The Minimum Reliable Horizon is the only horizon that matters.

Learn more
