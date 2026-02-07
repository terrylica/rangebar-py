---
source_url: https://gemini.google.com/share/623be87160ee
source_type: gemini-3-pro
scraped_at: 2026-02-02T09:03:00-08:00
purpose: Research parameter-free feature selection methods for high-autocorrelation financial time series with OOD robustness
tags:
  [
    feature-selection,
    time-series,
    ood-robustness,
    causal-discovery,
    autocorrelation,
  ]

# REQUIRED provenance
model_name: Gemini 3 Pro Deep Research
model_version: "3.0"
tools: []

# REQUIRED for Claude Code backtracking + context
claude_code_uuid: 14ac6476-077b-4c82-a3fe-217dae94cff6
claude_code_project_path: "~/.claude/projects/-Users-terryli-eon-trading-fitness/14ac6476-077b-4c82-a3fe-217dae94cff6"

# REQUIRED backlink metadata (filled after ensuring issue exists)
github_issue_url: https://github.com/terrylica/cc-skills/issues/21
---

Title: ‎Gemini - direct access to Google AI

URL Source: <https://gemini.google.com/share/623be87160ee>

Markdown Content:
Exhaustive Research Report: Principled Feature Selection for High-Autocorrelation Financial Time Series with OOD Robustness

---

## Executive Summary

The quantitative analysis of financial time series presents a unique confluence of challenges that defy standard machine learning methodologies. The user's specific scenario—a dataset characterized by high dimensionality relative to effective information content ( compressing to effective dimensions), extreme temporal autocorrelation (ACF ), and the necessity for Out-of-Distribution (OOD) robustness—represents a textbook example of the "illusion of predictability" often encountered in financial engineering. In such environments, traditional feature selection techniques based on in-sample correlation or randomized cross-validation often yield spurious predictors that collapse during regime shifts (e.g., volatility spikes or liquidity crises).

This report conducts a rigorous, comprehensive examination of state-of-the-art, parameter-free (or minimal-hyperparameter) methodologies capable of addressing these specific pathologies. Moving beyond simple correlation filtering, we investigate advanced frameworks rooted in Causal Discovery, Information Theory, Invariant Risk Minimization, and Randomized Signal Processing.

Our analysis identifies four dominant methodological paradigms that satisfy the user's constraints for Python compatibility, scalability to , and handling of bounded $$ features:

1. **Time Series Knockoffs Inference (TSKI):** A statistically rigorous framework providing mathematically guaranteed False Discovery Rate (FDR) control under serial dependence. By utilizing block-sampling mechanisms to generate valid "shadow" variables, TSKI decouples predictive power from temporal autocorrelation, offering a solution to the "significance inflation" problem caused by high ACF.

2. **Causal Discovery via PCMCI+:** A two-phase algorithm (Condition Selection and Momentary Conditional Independence) implemented in the `tigramite` library. PCMCI explicitly models time lags and filters out spurious associations driven by autocorrelation, identifying true causal drivers rather than mere correlates.

3. **Nonlinear Redundancy Detection via HSIC Lasso:** Utilizing the Hilbert-Schmidt Independence Criterion, this method captures arbitrary nonlinear dependencies (essential for bounded indicators like RSI) without explicit density estimation. The block-based implementation in `pyHSICLasso` ensures scalability to large datasets.

4. **Implicit Scale Selection via Detach-ROCKET:** Addressing the arbitrary nature of fixed lookback windows, this approach leverages Random Convolutional Kernel Transforms (ROCKET) followed by Sequential Feature Detachment (SFD). It automatically selects the optimal receptive fields (temporal scales) from a pool of thousands, bypassing manual lag selection.

Furthermore, to address the critical requirement of **OOD Robustness**, we explore **Sparse Invariant Risk Minimization (SparseIRM)**. This technique integrates sparsity constraints into the IRM objective, forcing the model to select features that support an invariant predictor across disparate market regimes (environments), thereby ignoring features that only correlate spuriously within specific volatility clusters.

The following report details these methodologies, providing theoretical foundations, implementation guides, and comparative analyses to construct a robust feature selection pipeline.

---

1. The Pathology of Financial Feature Spaces

---

### 1.1 The Curse of Autocorrelation and Effective Sample Size

The user's dataset features a lag-1 autocorrelation function (ACF) of 0.94. This single statistic invalidates the vast majority of standard feature selection algorithms (like standard Lasso, Random Forest importance, or univariate tests), which assume Independent and Identically Distributed (i.i.d.) samples.

In a time series with high autocorrelation, the **Effective Sample Size ()** is drastically lower than the raw sample size . The relationship is approximated by:

For :

For a dataset of bars, the information content is equivalent to only independent observations. This massive reduction in degrees of freedom leads to **significance inflation**: spurious correlations that would vanish in i.i.d. data appear statistically significant simply because the series "trends" in the same direction for a duration. A feature selection algorithm that does not account for this (e.g., standard Pearson correlation filtering) will aggressively over-select features that happen to share a local trend with the target, leading to catastrophic failure OOD when the trend reverses.

### 1.2 Multicollinearity and the PCA Participation Ratio

The user notes that 160 features compress to effective dimensions. This suggests that 94% of the feature space is redundant. In financial technical analysis, this is common; an RSI(14), RSI(15), and Stochastic(14) are mathematically distinct but informationally identical.

Standard regularization (Lasso ) handles multicollinearity by arbitrarily selecting one feature and zeroing out the others. In finance, this arbitrariness is a source of instability. If Feature A (RSI-14) and Feature B (RSI-15) are nearly identical, selecting A in one training window and B in another (due to noise) makes the model uninterpretable and fragile. We require **Stability Selection** or **Group Selection** methods that treat these clusters coherently or select the representative based on _causal_ rather than _statistical_ dominance.

### 1.3 The OOD Generalization Gap: Covariate Shift

Financial markets are non-stationary systems characterized by distinct regimes (e.g., Mean Reversion vs. Momentum, Low Volatility vs. High Volatility). A feature set optimized for in-distribution (ID) performance often relies on correlations specific to the training regime.

- **In-Sample**: High volume correlates with price increases (Bull Market).

- **Out-of-Distribution (OOD)**: High volume correlates with price crashes (Panic Selling).

A robust feature selection process must identify features whose relationship is **invariant** across these shifts. This necessitates moving beyond Empirical Risk Minimization (ERM) toward **Invariant Risk Minimization (IRM)** and Causal Discovery.

---

1. Feature Selection for High-Autocorrelation Time Series

---

To address Research Question 1, we must employ methods that inherently respect or model serial dependence.

### 2.1 Time Series Knockoffs Inference (TSKI)

The **Knockoff Filter**, introduced by Barber and Candès, provides a rigorous statistical framework for variable selection that controls the False Discovery Rate (FDR) in finite samples. Unlike p-values, which test marginal independence, Knockoffs test conditional independence.

#### 2.1.1 Theoretical Mechanism

For each feature , we construct a "knockoff" variable . The vector of knockoffs is constructed to satisfy two properties:

1. **Exchangeability**: The joint distribution of is invariant to swapping and . This means mimics the correlation structure of perfectly.

2. **Null Property**: is conditionally independent of the response given .

Feature importance statistics are computed (e.g., the difference in Lasso coefficients between and ). If is a true signal, should be large and positive. If is noise, should be symmetrically distributed around zero.

#### 2.1.2 Adapting to Time Series (TSKI)

Standard knockoff generation assumes row-wise independence, which fails for financial time series (ACF=0.94). **Time Series Knockoffs Inference (TSKI)** adapts this by using **Block Sampling** or specialized generative models.

**Mechanism for TSKI:**

1. **Block Bootstrap/Sampling**: Instead of resampling individual rows, the algorithm samples contiguous blocks of data to preserve the short-term dependency structure (autocorrelation).

2. **Approximation**: TSKI relaxes the exact exchangeability requirement to "approximate exchangeability" suitable for time series, proving that FDR control is maintained asymptotically if the block size is sufficiently large to cover the mixing time of the process.

3. **Group Knockoffs**: Given the "extreme redundancy" (), TSKI can generate **Group Knockoffs**. Instead of testing if RSI-14 is significant vs. RSI-15, it tests if the _group_ of momentum indicators is significant. This increases power by reducing the degrees of freedom in the hypothesis space.

#### 2.1.3 Implementation Strategy

- **Library**: `knockpy` (Python).

- **Key Parameters**:
  - `fdr`: The target False Discovery Rate (e.g., `0.2`). This is the only critical hyperparameter and is statistically meaningful (not a "magic number").

  - `method`: Use `'maxent'` or `'gaussian'` for Model-X knockoff generation.

  - `groups`: Can be auto-detected via hierarchical clustering of the correlation matrix.

- **Relevance**: TSKI explicitly solves the "High Temporal Autocorrelation" problem by incorporating the dependency structure into the null hypothesis generation.

### 2.2 Stability Selection with Circular Block Bootstrap

**Stability Selection**, proposed by Meinshausen and Bühlmann, improves feature selection reliability by aggregating results across many subsamples.

#### 2.2.1 Time Series Adaptation

For time series, random subsampling destroys the lag structure. We must use **Circular Block Bootstrap**.

1. **Block Size Determination**: The block size should depend on the ACF decay. For ACF=0.94, a block size that covers the decay to negligible levels (e.g., ) is necessary.

2. **Procedure**:
   - Generate bootstrap samples using circular block sampling.

   - Run Lasso (or another selector) on each sample with randomized regularization penalties.

   - Calculate the **Selection Probability** for each feature .

   - Keep features where (e.g., 0.6).

This method filters out features that are only predictive during specific fleeting regimes (instability), satisfying the requirement for robustness.

---

1. Causal Discovery and Structure Learning

---

Research Question 5 asks for "Causal feature selection methods." In highly correlated financial data, purely statistical association is insufficient. We need to identify the _parents_ of the target in the causal graph.

### 3.1 PCMCI and PCMCI+ (Tigramite)

**PCMCI** is the current state-of-the-art for causal discovery in time series. It is specifically designed to handle **autocorrelation** and **high dimensionality**.

#### 3.1.1 The Algorithm

PCMCI decomposes the problem into two phases to ensure efficiency and robustness:

1. **Phase 1: Condition Selection (PC1)**
   - Uses a variant of the PC algorithm to identify a superset of parents for the target .

   - It iterates through lagged features and tests marginal and conditional independence.

   - **Goal**: Reduce the feature space from 160 to a manageable set of potential drivers, effectively acting as a coarse filter.

2. **Phase 2: Momentary Conditional Independence (MCI)**


    *   Performs the MCI test: .

    *   Crucially, this conditions on _both_ the parents of the target  AND the parents of the source . This double-conditioning removes the confounding effect of autocorrelation (serial dependence) that plagues Granger Causality.

#### 3.1.2 Implementation and Parameter-Free Nature

- **Library**: `tigramite` (Python).

- **Independence Tests**:
  - `ParCorr` (Partial Correlation): Fast, assumes linear dependencies. Suitable for the "10,000 to 1,000,000 samples" scale.

  - `CMIknn` (Conditional Mutual Information via k-NN): Handles nonlinearity but is computationally expensive ( or ). Given , `ParCorr` is preferred, perhaps after a nonlinear transformation of features (e.g., rank transformation).

- **Key Parameters**:
  - `tau_max`: The maximum time lag to consider.

  - **Auto-Selection**: The `pc_alpha` (significance threshold) can be optimized automatically using AIC/BIC within `tigramite`.`tau_max` should be set based on the ACF decay plot (e.g., where ACF drops below 0.05).

#### 3.1.3 Relevance to Financial OOD

Causal links are theoretically invariant. If causes , that mechanism should hold regardless of the market regime (volatility). PCMCI helps isolate these invariant links from the "spurious" correlations driven by shared trends (common causes).

### 3.2 Constraint-based Causal Discovery (FCI)

If unobserved confounders (e.g., global macro variables not in the dataset) are suspected, the **FCI (Fast Causal Inference)** algorithm is theoretically superior to PC. However, FCI is significantly slower and less stable on finite data. Given the user's high-dimensional setup (), PCMCI is the pragmatic "state-of-the-art" choice.

---

1. Nonlinear Redundancy Detection Beyond Correlation

---

Research Question 3 addresses the "extreme redundancy" where correlation () is insufficient because relationships in bounded $$ data (like oscillators) are often nonlinear (e.g., saturation effects).

### 4.1 HSIC Lasso (Hilbert-Schmidt Independence Criterion)

**HSIC** is a kernel-based measure of dependence that captures all orders of dependence (mean, variance, higher moments), unlike Pearson correlation which only captures linear covariance.

#### 4.1.1 The Optimization Problem

HSIC Lasso formulates feature selection as a convex optimization problem: $$ \min\_{\alpha} \frac{1}{2} |

| \bar{L} - \sum\_{k=1}^P \alpha_k \bar{K}^{(k)} ||\_F^2 + \lambda ||\alpha||\_1 $$ Where:

- is the centered kernel matrix of the target .

- is the centered kernel matrix of feature .

- is the vector of feature weights (sparse due to penalty).

- is the Frobenius norm.

This objective mathematically translates to: **Find a sparse set of features whose combined kernel matrix best approximates the target's kernel matrix.** It maximizes relevance (dependence with ) while automatically penalizing redundancy (dependence between selected 's).

#### 4.1.2 Scalability: Block HSIC Lasso

Standard HSIC requires computing kernel matrices, which is impossible for (requires terabytes of RAM). **Block HSIC Lasso** solves this by dividing data into blocks of size (e.g., ). It computes the HSIC statistics on blocks and averages them. This reduces memory complexity from to , making it linear in .

#### 4.1.3 Implementation Strategy

- **Library**: `pyHSICLasso` (Python).

- **Parameters**:
  - `B` (Block size): Data-driven based on RAM (e.g., 2048).

  - `kernel`: Gaussian kernel is standard for continuous $$ data.

- **Relevance**: It is "parameter-free" in the sense that the regularization path is solved automatically (dual augmented Lagrangian), giving a ranking of features without manual thresholding.

### 4.2 Partial Information Decomposition (PID)

While requested, PID (calculating Synergy/Redundancy/Uniqueness) scales as . For , it is computationally intractable. PID is best used _post-selection_, i.e., after TSKI or HSIC Lasso reduces the set to features, use PID to understand the interaction between the survivors.

---

1. Automatic Lookback and Scale Selection

---

Research Question 4 highlights the arbitrary nature of "lb20/lb100". The financial signal may exist at scale 13 or 89, not 20 or 100.

### 5.1 Detach-ROCKET: Random Convolutional Kernels

**ROCKET (Random Convolutional Kernel Transform)** represents a paradigm shift from hand-crafted features to randomized feature generation.

#### 5.1.1 Mechanism

Instead of selecting a specific lag, ROCKET generates a massive bank of **Random Convolutional Kernels** (e.g., 10,000 filters):

- **Random Length**: e.g.,

- **Random Dilation**: e.g., . This dilation effectively creates "lookbacks" that grow exponentially. A kernel of length 11 with dilation 32 covers a receptive field of time steps.

- **Random Weights**: Sampled from Normal distribution.

The time series is convolved with these kernels, and two features are extracted per kernel:

1. **Max Value**: Captures the strongest activation (e.g., "did this specific shape occur?").

2. **Proportion of Positive Values (PPV)**: Captures the prevalence of the pattern.

#### 5.1.2 Sequential Feature Detachment (SFD)

**Detach-ROCKET** adds a pruning layer to standard ROCKET.

1. Train a simple Ridge Classifier/Regressor on the 20,000 features.

2. **SFD Algorithm**: Iteratively removes features. It identifies features that, when removed, minimally impact the validation error (approximated via leave-one-out cross-validation concepts on the Ridge linear system).

3. **Result**: A small subset of kernels (e.g., 50 out of 10,000) that are highly predictive.

#### 5.1.3 Why this Solves "Arbitrary Lookback"

The method implicitly searches the space of all lookbacks (via random dilations). If the market signal is best captured by a 45-day pattern, a random kernel with receptive field will have high predictive power and will be retained by SFD. The user does not need to specify `lb45`.

- **Library**: `detach_rocket` (Python).

- **OOD Note**: Convolutional features are often more robust to shift than fixed lags because they detect _local shape_ rather than absolute position.

### 5.2 Wavelet Scattering Transform

Another "parameter-free" multi-scale approach is the **Wavelet Scattering Transform** (e.g., via `kymatio`). It produces features that are locally translation-invariant and stable to deformation.

- **Mechanism**: A cascade of wavelet transforms and modulus non-linearities.

- **Output**: Coefficients representing energy at scales .

- **Selection**: Apply TSKI or Lasso on the scattering coefficients to select the relevant scales.

---

1. OOD-Robust Feature Selection via Invariant Risk Minimization

---

Research Question 2 specifically targets **Out-of-Distribution (OOD)** validation.

### 6.1 The Failure of Standard ERM

Standard machine learning (Empirical Risk Minimization) minimizes average error on the training set.

In finance, the training set might be dominated by Low Volatility regimes. The model will learn spurious correlations valid only in Low Vol (e.g., "buy the dip"). When High Volatility hits, "buy the dip" becomes "catch the falling knife."

### 6.2 Sparse Invariant Risk Minimization (SparseIRM)

**Invariant Risk Minimization (IRM)** posits that while correlations change across environments, the causal mechanism remains invariant.

**SparseIRM** integrates feature selection directly into this framework. **Objective Function**:

Subject to being optimal for each environment individually.

#### 6.2.1 Workflow for Finance

1. **Environment Definition**: This is the key "parameter." For time series, use **Change Point Detection** (e.g., `ruptures` library ) on the volatility or correlation structure to segment history into regimes (Env A, Env B, Env C).

2. **Algorithm**:
   - Initialize a mask over the 160 features.

   - Optimize the mask to minimize loss across all environments simultaneously, penalizing the number of active features.

   - Features that require different weights in Env A vs Env B to minimize error are penalized (variance penalty). Features that work well with the _same_ weight in both are retained.

#### 6.2.2 Implementation

- **Library**: Research code `SparseIRM` or `ZIN` (Zhou et al.).

- **Parameter**: (regularization strength).

- **Advantages**: Explicitly optimizes for the user's "generalization to market regime changes" requirement.

---

1. Comparative Analysis and Integrated Methodology

---

The user's requirements create a tension:

- **PCMCI** is best for _causality_ but can be slow and sensitive to calibration.

- **TSKI** is best for _statistical rigor_ (FDR) but relies on Lasso (linear) statistics primarily.

- **Detach-ROCKET** is best for _predictive power_ and _scale selection_ but is less interpretable (kernels vs features).

- **HSIC Lasso** is best for _nonlinear redundancy_ but doesn't handle OOD explicitily.

### 7.1 The "Filter-Wrapper" Pipeline Recommendation

We propose a hierarchical pipeline to solve the "160 to 10" problem robustly.

#### Stage 1: Nonlinear Redundancy Reduction (N=160 N=50)

**Method**: **Block HSIC Lasso**.

- **Why**: Quickest way to remove the "extreme redundancy" ( to ) without assuming linearity.

- **Action**: Use `pyHSICLasso` with . Select top 50 features.

- **Addressing RQ3**: Handles nonlinear redundancy efficiently.

#### Stage 2: Causal & Robust Filtering (N=50 N=15)

**Method**: **PCMCI+ (Tigramite)**.

- **Why**: Explicitly handles autocorrelation (RQ1) and filters spurious links.

- **Action**: Run PCMCI on the surviving 50 features. Use `ParCorr` test for speed.

- **Auto-Tuning**: Use AIC to select `pc_alpha`. Set `tau_max` based on ACF decay.

- **Output**: The set of "Parent" features .

#### Stage 3: Invariant Validation (N=15 Final Set)

**Method**: **OOD Validation via Environment Splitting**.

- **Why**: Ensures OOD robustness (RQ2).

- **Action**:
  1. Use `ruptures` to segment data into 3 distinct regimes (e.g., Bull, Bear, Sideways).

  2. Train a simple Ridge model on the Stage 2 features on Regime 1 & 2.

  3. Test on Regime 3.

  4. Discard features with high coefficient variance across regimes.

### 7.2 Alternative "End-to-End" Path: Detach-ROCKET

If the user prefers a "black box" that just works:

- **Input**: Raw range bars (Open, High, Low, Close) + Volume.

- **Method**: **Detach-ROCKET**.

- **Why**: It internally handles scale selection (RQ4) and feature interaction. The SFD pruning acts as the feature selector.

- **Output**: A classifier/regressor using a minimal set of convolutional features.

- **Trade-off**: You lose the "160 named features" (RSI, MACD) in favor of "10 optimal kernels". Given the user has _already_ generated 160 features, they likely want to select _from_ them. Thus, the Pipeline in 7.1 is preferred.

---

1. Implementation Roadmap (Python)

---

### 8.1 Libraries and Dependencies

Requirement Library Source Key Capability
**TSKI / FDR Control**`knockpy`Group knockoffs, Block sampling.
**Causal Discovery**`tigramite`PCMCI+, `ParCorr`, `CMIknn`.
**Nonlinear Selection**`pyHSICLasso`()Block HSIC Lasso for Large N.
**Scale Selection**`detach_rocket`SFD Pruning, Random Kernels.
**Regime Segmentation**`ruptures`Change point detection (Pelt, BinSeg).

### 8.2 Parameter Selection Strategy (No Magic Numbers)

1. **Block Size ()**:
   - **Logic**: Must exceed the autocorrelation decay time.

   - **Data-Driven**: Plot ACF. Find lag where (95% CI). Set .

2. **Significance ()**:


    *   **Logic**: Controls False Positive Rate.

    *   **Data-Driven**: In TSKI, set FDR (q-value) = 0.2 (standard for exploratory selection). In PCMCI, use AIC/BIC for optimal .

3.  **Lookback (`tau_max`)**:


    *   **Logic**: Max plausible causal delay.

    *   **Data-Driven**: Use Detach-ROCKET to find max effective dilation, or set `tau_max` to the block size .

### 8.3 Limitations Assessment

- **Boundedness**: `pyHSICLasso` handles this naturally. `knockpy` (Gaussian knockoffs) might struggle with bounded edges; consider transforming features via `scipy.special.logit` to map before applying TSKI.

- **N=1,000,000**: `tigramite` with `ParCorr` scales linearly. `CMIknn` will be too slow; avoid nonlinear independence tests in PCMCI unless is subsampled. `pyHSICLasso` with blocking scales well.

- **Stationarity**: All methods assume _piecewise_ stationarity. If the market is continuously drifting (no stable regimes), SparseIRM and PCMCI will struggle to converge.

1. Conclusion

---

The user's current approach—intuition and correlation filtering—is statistically perilous given the ACF=0.94 and regime instability. By adopting **Time Series Knockoffs (TSKI)** for rigorous FDR control and **SparseIRM** for OOD robustness, the feature selection process moves from "art" to "science."

The most immediate, high-impact upgrade is to replace the arbitrary lookback selection with **Detach-ROCKET**, which provides an empirically validated, parameter-free method to discover the intrinsic time scales of the financial signal. For the specific 160-feature set, a pipeline of **HSIC Lasso (Redundancy)\*\***PCMCI (Causality)\*\* is the theoretical optimum for recovering the "10 effective dimensions" as true causal drivers.

---

## Detailed Analysis and Methodology

1. Introduction: The High-Dimensional Financial Time Series Problem

---

The analysis of financial time series data for trading strategy development is frequently obstructed by the "curse of dimensionality" and the "illusion of predictability" inherent in high-autocorrelation data. The user presents a scenario involving **160 bounded $$ features** derived from range bar data, characterized by:

1. **Extreme Redundancy**: An effective dimensionality of , implying a massive degree of multicollinearity.

2. **High Autocorrelation**: A lag-1 ACF of 0.94, indicating near-unit-root behavior where past values strongly predict future values, masking the true signal-to-noise ratio regarding the target variable (returns or direction).

3. **Regime Instability**: The critical need for Out-of-Distribution (OOD) robustness, as financial markets exhibit non-stationary behavior (Covariate Shift).

This report investigates methodologies that transcend standard i.i.d. assumptions. We focus on methods that are **parameter-free** (or data-driven), **open-source**, and capable of scaling to large datasets ().

---

1. Feature Selection for High-Autocorrelation Data (TSKI & Stability Selection)

---

### 2.1 The Failure of Marginal Testing

Standard feature selection (e.g., univariate correlation, T-tests) tests the marginal independence . In time series, two random walks and can exhibit high correlation (e.g., ) purely due to shared trends, even if no causal link exists (Spurious Regression). This leads to a massive False Discovery Rate (FDR).

### 2.2 Time Series Knockoffs Inference (TSKI)

**Method Name**: Time Series Knockoffs Inference (TSKI) / Model-X Knockoffs for Time Series. **Source**: "High-Dimensional Knockoffs Inference for Time Series Data" (2021).

**Why it's relevant**: TSKI is currently the only framework offering rigorous **FDR control** in finite samples for time series data. It replaces "magic number" thresholds (like correlation > 0.95) with a probabilistic guarantee (e.g., FDR ).

**Mechanism**: The core idea is to generate "knockoffs" —synthetic variables that mimic the correlation structure of the original variables but are conditionally independent of .

- **Block Sampling**: To handle the ACF=0.94, TSKI does not resample rows independently. It uses **Block Bootstrap** or **Block Sampling**. Data is divided into blocks of length . The blocks are shuffled or resampled to create the knockoff features. This preserves the internal temporal structure (the autocorrelation) within the blocks.

- **Feature Statistics**: The algorithm fits a model (e.g., Lasso) on the augmented set . It computes a statistic .
  - If is null, and are exchangeable, so is symmetric around 0.

  - If is predictive, is large positive.

- **Selection Threshold**: The threshold is computed adaptively: , where is the target FDR.

**Key Parameters (Data-Driven)**:

- `fdr` (q): Target false discovery rate (e.g., 0.2).

- `block_size`: Selected based on the spectral analysis or ACF decay of the features (e.g., where ACF drops < 0.1).

**Implementation**:

- **Library**: `knockpy` (Python).

- **GitHub**: `https://github.com/amspector100/knockpy`

- **Code Snippet Concept**:

**Limitations**:

- Generating valid knockoffs for non-Gaussian, bounded $$ data is challenging. The "Gaussian" sampler in `knockpy` might violate the bounded support.

- **Mitigation**: Transform features using `logit` () to map $$ to before applying TSKI, maximizing Gaussianity.

### 2.3 Stability Selection with Circular Block Bootstrap

**Method Name**: Stability Selection (Meinshausen & Bühlmann) adapted with Block Bootstrap. **Source**: "Stability of Feature Selection Algorithms".

**Why it's relevant**: Stability selection avoids the "instability" of Lasso where correlated features are selected arbitrarily. By bootstrapping, it computes the _probability_ of selection.

**Time Series Adaptation**: Standard bootstrap destroys autocorrelation. **Circular Block Bootstrap** samples blocks of length wrapping around the end of the series.

- **Algorithm**:
  1. Choose block size (e.g., 50 time steps).

  2. Resample blocks to form a new dataset .

  3. Run Lasso (L1) on . Record selected features.

  4. Repeat 100 times.

  5. Select features with frequency .

**Parameters**:

- `block_length`: Data-driven (ACF decay).

- `threshold`: Arbitrary (typically 0.6 - 0.9), but less sensitive than correlation thresholds.

---

1. Causal Discovery: Filtering Spurious Autocorrelation

---

Research Question 1 & 5 ask for causal understanding and handling temporal dependence.

### 3.1 PCMCI / PCMCI+

**Method Name**: PCMCI (Peter-Clark Momentary Conditional Independence). **Source**: "Detecting and quantifying causal associations in large nonlinear time series datasets" (Runge et al., 2019).

**Why it's relevant**: PCMCI is specifically designed to disentangle **autocorrelation** from **causal coupling**.

- In highly autocorrelated data (), correlates with even if the true cause is . Standard Lasso might pick due to higher correlation.

- PCMCI conditions on the _past of the driver_ and the _past of the target_, effectively isolating the new information flow.

**Mechanism**:

1. **PC Phase (Condition Selection)**: efficiently selects a set of parents using iterative independence tests. This handles the high dimensionality ().

2. **MCI Phase**: Computes the test statistic for conditioned on and .

**Implementation**:

- **Library**: `tigramite` (Python).

- **GitHub**: `https://github.com/jakobrunge/tigramite`

- **Independence Test**: Use `ParCorr` (Partial Correlation) for speed on . Use `CMIknn` (k-NN) for nonlinear relations (slower).

- **Parameters**:
  - `tau_max`: Max lag. Set based on domain knowledge or ACF.

  - `pc_alpha`: Significance level. `tigramite` supports **AIC-based optimization**, making it parameter-free.

**Limitations**:

- Assumes Causal Stationarity (relationships don't change over time). This conflicts with the OOD requirement unless combined with regime segmentation (see Section 6).

---

1. Nonlinear Redundancy Detection (Bounded Features)

---

Research Question 3 asks for redundancy detection beyond correlation.

### 4.1 HSIC Lasso (Hilbert-Schmidt Independence Criterion)

**Method Name**: HSIC Lasso / Block HSIC Lasso. **Source**: "High-Dimensional Nonlinear Feature Selection for Big Biological Data".

**Why it's relevant**:

- The user's features are bounded $$ (e.g., RSI). Relationships are often nonlinear (e.g., extreme values matter, middle values don't). Pearson correlation fails to capture this.

- HSIC measures dependence in RKHS (kernel space), capturing all moments.

- HSIC Lasso penalizes the HSIC between selected features (Redundancy) while rewarding HSIC with target (Relevance).

**Mechanism**:

- Solves $\min\_{\alpha} \frac{1}{2} |

| \bar{L} - \sum \alpha_k \bar{K}^{(k)} ||\_F^2 + \lambda ||\alpha||\_1$.

- **Block HSIC**: To handle , it computes the estimator on blocks of size (e.g., 2048) and averages, reducing memory complexity from quadratic to linear.

**Implementation**:

- **Library**: `pyHSICLasso` (Python).

- **GitHub**: `https://github.com/riken-aip/pyHSICLasso`

- **Parameters**:
  - `B` (Block Size): Set to 2048 or 4096 (hardware dependent).

  - `kernel`: `'Gaussian'` (standard for continuous).

- **Data-Driven**: The Lasso path () is solved automatically to return a ranked list of features.

**Limitations**:

- Does not explicitly model temporal order (treats blocks as i.i.d.). Ideally, input blocks should be contiguous time segments to respect local structure.

---

1. Automatic Lookback and Scale Selection

---

Research Question 4 asks to remove arbitrary lookbacks (`lb20`, `lb100`).

### 5.1 Detach-ROCKET

**Method Name**: Detach-ROCKET (Sequential Feature Detachment on ROCKET features). **Source**: "Detach-ROCKET: Sequential Feature Detachment for Time Series Classification".

**Why it's relevant**: This is a **constructive** approach to feature selection. Instead of selecting from pre-calculated lookbacks (SMA20, SMA100), ROCKET generates features with **random lookbacks** (via random kernel lengths and dilations).

- A kernel with length 9 and dilation 10 looks at a window of 90.

- A kernel with length 9 and dilation 1 looks at a window of 9.

- **SFD (Sequential Feature Detachment)** prunes the kernels that aren't predictive.

**Mechanism**:

1. **Generate**: 10,000 random convolutional features.

2. **Train**: A linear Ridge classifier/regressor.

3. **Prune**: Iteratively remove features based on the impact on the leave-one-out error metric.

4. **Result**: The optimal "lookbacks" are the receptive fields of the surviving kernels.

**Implementation**:

- **Library**: `detach_rocket` (Python).

- **GitHub**: `https://github.com/gon-uri/detach_rocket`

- **Parameters**:
  - `num_kernels`: 10,000 (default).

  - `model_type`: `'MiniRocket'` (fastest).

**Limitations**:

- It selects _kernels_, not the user's original named features. However, it can be applied to the multivariate time series of the 160 features to find the optimal temporal processing for them.

---

1. OOD-Robustness: Invariant Risk Minimization

---

Research Question 2 targets generalization to regime changes.

### 6.1 Sparse Invariant Risk Minimization (SparseIRM)

**Method Name**: SparseIRM. **Source**: "Sparse Invariant Risk Minimization" (Zhou et al., 2022).

**Why it's relevant**: Standard feature selection (Lasso, RFE) overfits to the "training regime" (e.g., Low Volatility). SparseIRM selects features that are **invariant** predictors across defined environments.

**Mechanism**:

- **Environments**: The user must define training environments. In finance, this is best done by segmenting the history into "Regimes" (e.g., Volatility Clusters) using a library like `ruptures`.

- **Objective**: Minimize error across all environments subject to the constraint that the optimal linear weights are _identical_ in all environments.

- **Sparsity**: Adds an or penalty to to force feature selection.

**Implementation**:

- **Code**: Research implementations (e.g., `ZIN` repo ).

- **GitHub**: `https://github.com/linyongver/ZIN_official`

- **Parameters**:
  - `penalty_weight`: Strength of invariance constraint.

  - `environments`: Derived from data (Change Point Detection).

### 6.2 Domain Adaptation via "Feature Dropping"

If full IRM is too complex to implement, a robust heuristic is **RandAugment** for time series: training on data where random subsets of features are masked (set to 0) forces the model to learn redundant, robust representations rather than relying on a single "fragile" feature.

---

1. Integrated "Filter-Wrapper" Pipeline

---

To answer the user's setup ("160 features -> 10 effective"), we propose a cascading pipeline.

### Step 1: Nonlinear Redundancy Filter (160 50)

- **Method**: **Block HSIC Lasso**.

- **Reason**: Efficiently removes the "Extreme Redundancy" (PCA ratio) without assuming linearity.

- **Library**: `pyHSICLasso`.

- **Parameter**: .

### Step 2: Causal/Statistically Significant Filter (50 20)

- **Method**: **TSKI (Knockoffs)** or **PCMCI**.

- **Reason**: Handles "High Autocorrelation". Removes features that are only statistically significant due to trend inflation.

- **Library**: `knockpy` (for FDR control) or `tigramite` (for causal rigor).

- **Parameter**: `fdr=0.1` (TSKI) or `pc_alpha` via AIC (PCMCI).

### Step 3: OOD Robustness Check (20 10)

- **Method**: **Environment Testing (IRM Principle)**.

- **Reason**: Ensures generalization.

- **Procedure**:
  1. Use `ruptures` to find 3 major regimes in history.

  2. Train Ridge Regressor on features from Step 2 on Regime A.

  3. Test on Regime B.

  4. Keep features with stable coefficients (low variance) and stable error contribution.

---

1. Comparison Table

---

1. Conclusion

---

The "magic numbers" in the current system should be replaced by:

1. **Data-Driven Blocks**: Use ACF decay to set block sizes for TSKI/HSIC.

2. **Probabilistic Thresholds**: Use FDR (0.1/0.2) in TSKI instead of correlation > 0.95.

3. **Automatic Scales**: Use Detach-ROCKET to find lookbacks instead of fixed `lb20`.

4. **Invariance Criteria**: Use stability across regimes (SparseIRM principle) instead of in-sample cross-validation.

This approach rigorously addresses the "illusion of predictability" by filtering out features that are redundant, spurious due to autocorrelation, or fragile to regime shifts.
