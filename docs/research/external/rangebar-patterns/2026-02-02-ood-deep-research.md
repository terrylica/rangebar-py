---
source_url: https://gemini.google.com/share/9bcd199946a0
source_type: gemini-3-pro
scraped_at: 2026-02-02T05:35:42Z
purpose: Deep research on parameter-free OOD robustness for financial ML with selective abstention
tags:
  [
    ood-robustness,
    selective-prediction,
    conformal-prediction,
    deep-gamblers,
    kelly-criterion,
  ]

# REQUIRED provenance
model_name: Gemini 3 Pro
model_version: Deep Research (2026-02)
tools: []

# REQUIRED for Claude Code backtracking + context
claude_code_uuid: 3d0512c5-fe15-4d74-86cc-b2e6ece321e9
claude_code_project_path: "~/.claude/projects/-Users-terryli-eon-alpha-forge-worktree-2025-12-24-ralph-ml-robustness-research/3d0512c5-fe15-4d74-86cc-b2e6ece321e9"

# REQUIRED backlink metadata (to be filled after issue creation)
github_issue_url: https://github.com/EonLabs-Spartan/alpha-forge/issues/131
---

# Parameter-Free OOD Robustness for Financial ML with Selective Abstention

## 1. Executive Summary

### 1.1 The Operational Context and Critical Pathologies

The deployment of machine learning models in financial markets is fundamentally a battle against non-stationarity. In the specific context of a **Walk-Forward Optimization (WFO)** framework utilizing **range bar** cryptocurrency data, this battle is complicated by the unique temporal warping properties of the data structure. While range bars stabilize the volatility dimension by mandating constant price magnitude per bar, they transfer market turbulence into the time domain—increasing the frequency of bars during high activity and sparsifying them during calm periods. The user's current architecture, a BiLSTM with a SelectiveNet-style gating mechanism, has hit a performance plateau characterized by several distinct pathologies: a reliance on brittle "magic numbers" (specifically a fixed validation window of 1200 bars), the failure of standard regime detection algorithms (PELT) due to the homogenized binary signal, and, most critically, a **negative Kelly edge**.

The negative Kelly edge indicates a profound failure of calibration: the model is structurally overconfident, assigning high probabilities to directional predictions that do not materialize with sufficient frequency to justify the implied position size. In a production trading environment, this leads to ruin, as the Kelly criterion penalizes estimation errors with geometric decay of capital. Furthermore, the high disagreement observed across random seeds for the same fold suggests that the loss landscape is flat or underspecified, indicating that the model is capturing noise rather than robust structural alpha.

### 1.2 The Research Mandate: Parameter-Free Robustness

The core research challenge is to transition from a fragile, hyperparameter-dependent system to a robust, **parameter-free** architecture. "Parameter-free" in this context is defined not as the absence of mathematical constants, but as the elimination of arbitrary tuning values (e.g., "gap_bars=50," "threshold=0.6") in favor of values derived from:

1. **Online Optimization:** Parameters learned dynamically via regret minimization.
2. **Statistical Bounds:** Parameters derived from martingale inequalities or conformal guarantees.
3. **Financial Constants:** Parameters anchored to real-world constraints like the risk-free rate or transaction costs.

### 1.3 Strategic Recommendations

Based on an exhaustive review of State-of-the-Art (SOTA) literature in selective classification, online learning, and conformal prediction, this report identifies three pillar technologies that directly address the user's constraints:

**1. Deep Gamblers with Portfolio Theory Loss (Replacing SelectiveNet)**
The SelectiveNet architecture requires a separate auxiliary head and a user-defined coverage target φ, which acts as a static hyperparameter unsuitable for shifting regimes. We recommend replacing this with the **Deep Gamblers** framework. By incorporating a "reservation" (abstention) class into the primary classification head and optimizing a loss function derived from the **Kelly Criterion**, the model learns to abstain based on the **opportunity cost of capital** (R). R is not a tuning parameter but a financial constant (risk-free rate + transaction friction). This explicitly solves the "negative Kelly edge" by penalizing overconfidence that would lead to capital destruction, aligning the training objective with the trading objective.

**2. Strongly Adaptive Online Conformal Prediction (SAOCP) (Replacing Static Gating)**
To handle run-time uncertainty without fixed thresholds, we recommend **SAOCP**. Unlike standard Conformal Prediction, which assumes exchangeability (violated in finance), SAOCP minimizes **strongly adaptive regret**, ensuring valid coverage on local time intervals. It employs a meta-algorithm to aggregate multiple learning rates, allowing it to adaptively tighten or loosen the abstention threshold in response to sudden volatility shocks without manual intervention. This provides a theoretical safety net that "breathes" with the market.

**3. OneNet for Adaptive Windowing (Replacing Fixed Lookback)**
The reliance on `val_bars=1200` is a fragility. **OneNet (Online Ensembling Network)** resolves this by maintaining multiple branches (e.g., Short-Term vs. Long-Term history) and combining their outputs using a reinforcement-learning-based weighting mechanism. This effectively allows the "window size" to be a fluid, dynamic parameter that shifts automatically based on which historical horizon is currently most predictive.

### 1.4 Implementation Trajectory

The report outlines a phased implementation roadmap:

1. **Phase I:** Pivot the loss function to Deep Gamblers to fix the calibration / Kelly edge issue.
2. **Phase II:** Implement Energy-Based OOD detection to identify "unknown unknowns" (regimes where the model is confused, not just uncertain).
3. **Phase III:** Wrap the inference in SAOCP to automate threshold management.
4. **Phase IV:** Deploy OneNet to dynamize the training window.

This architecture moves the system from a "Predict-then-Threshold" heuristic to a "Portfolio-Optimization-and-Adaptation" pipeline, rigorously grounded in statistical learning theory and financial mathematics.

---

## 2. The Pathology of Non-Stationarity in Range-Based Finance

To develop a robust solution, we must first rigorously define the problem space. The user's transition to range bars and the observation of a "negative Kelly edge" are not unrelated phenomena; they are intrinsic to how standard ML models misinterpret financial time (duration) and risk (probability).

### 2.1 The Geometry of Range Bar Distortion

Standard time series analysis usually operates on "Time Bars" (e.g., 1-hour candles). In this domain, volatility is observed as amplitude change (P_high - P_low). ML models learn to associate high variance features with specific market states.

In **Range Bars**, a new bar is formed only when the price traverses a cumulative distance threshold δ. This transformation has profound implications for machine learning:

1. **Amplitude Normalization:** Every bar has identical magnitude (approximately δ). The model cannot "see" volatility through price range features alone.
2. **Time Warping:** Volatility is encoded in the _frequency_ of bar formation. A high-volatility regime produces 100 bars/hour; a low-volatility regime produces 5 bars/hour.
3. **Information Density:** The user's finding that `val_bars=1200` is optimal suggests that the "memory" of the market (the period over which patterns repeat) is defined by _events_, not time. However, 1200 bars could represent 2 days (high volatility) or 2 months (low volatility).

**The OOD Trigger:** A model trained on a high-velocity regime (1200 bars = 2 days) learns short-term microstructure dependencies. If the market shifts to a low-velocity regime (1200 bars = 2 months), the same model attempts to apply microstructure logic to macro-structure trends. This is a classic **Covariate Shift** (P_train(X) ≠ P_test(X)), leading to the degradation observed.

### 2.2 The Negative Kelly Edge: A Crisis of Calibration

The Kelly Criterion defines the optimal fraction of capital f\* to wager on a repeated bet to maximize geometric growth:

```
f* = (bp - q) / b = (p(b+1) - 1) / b
```

Where:

- p is the probability of winning.
- q = 1 - p is the probability of losing.
- b is the gross odds received (in financial trading, the Reward/Risk ratio).

The user reports a **Negative Kelly Edge**. This means the model computes a predicted probability p̂ such that the derived f\* is positive, but the _realized_ outcome probability p_true is sufficiently lower than p̂ that the actual edge is negative.

Mathematically, this implies: E < 0

This is a failure of **Calibration**. Standard Cross-Entropy loss minimizes the divergence between the predicted distribution and the target, but it does not penalize _overconfidence_ specifically. A model can achieve low Cross-Entropy by being "mostly right" but "confident and wrong" on a few critical samples. In finance, those "confident and wrong" samples cause massive drawdowns via the Kelly mechanism.

The user's current SelectiveNet approach attempts to mitigate this by gating predictions based on a confidence threshold. However, SelectiveNet optimizes a surrogate objective (coverage vs. accuracy) that is decoupled from the financial utility of the bet. It does not "know" about the cost of capital or the asymmetry of geometric ruin. This necessitates a move to **Deep Gamblers**, which embeds the Kelly Criterion directly into the loss landscape.

### 2.3 The Failure of "Gap Detection" and "PELT"

The user noted that "ACF-based gap detection is worse than hardcoded." This highlights the failure of linear statistical methods on binary range bar data.

- **ACF (Auto-Correlation Function):** Measures linear dependence. Range bars often exhibit nonlinear, regime-switching dependence that ACF misses.
- **PELT (Pruned Exact Linear Time):** A changepoint detection algorithm that minimizes a cost function (usually mean/variance shift).
  - _Why it fails:_ The signal is binary direction {0,1} with p ≈ 0.5. The mean is constant (≈ 0.5). The variance is constant (≈ 0.25). PELT sees a stationary sequence of coin flips. The _structure_ of the dependence (e.g., HMM state transition probabilities) is changing, not the marginal moments.

This necessitates **OOD detection methods** that operate on the _learned representation space_ (Energy-based) or _prediction error stream_ (Martingales), rather than the raw signal.

---

## 3. Selective Prediction through Portfolio Theory: The Deep Gamblers Framework

The most direct solution to the "Negative Kelly Edge" and the need for parameter-free abstention is to restructure the learning objective itself. We recommend the **Deep Gamblers** method, also known as "Learning to Abstain with Portfolio Theory."

### 3.1 Theoretical Derivation

Consider a "horse race" (trading opportunity) with m mutually exclusive outcomes. In binary trading, m = 2 (Up, Down). The trader has wealth W_0.

- The model predicts a distribution p = (p_1, p_2).
- The trader can also reserve a portion of wealth in a safe asset (cash) with return O (reservation return).
- Let b*i be the portion of wealth bet on outcome i, and b*{m+1} be the portion reserved. Σb = 1.

The Deep Gamblers framework modifies the neural network to output m+1 values: p̂*1, p̂_2, ..., p̂*{m+1}. Here, p̂\_{m+1} represents the model's assigned probability that "none of the outcomes are safe enough to bet on," or effectively, the allocation to the safe asset.

The wealth after one step, given that the true outcome is y_k, is:

```
W_1 = W_0 · (p̂_k · Payoff + p̂_{m+1} · O)
```

The objective is to maximize the **Doubling Rate** (Geometric Growth):

```
Δ = E[log(W_1/W_0)]
```

This leads to the **Deep Gamblers Loss Function**:

```
L_DG(p̂, y) = -Σ_{k=1}^m y_k log(p̂_k + p̂_{m+1} · O)
```

For binary classification (y ∈ {0, 1}), where class 1 is the target:

```
L_DG = -log(p̂_target + p̂_abstain · O)
```

### 3.2 The Parameter O: Cost of Capital, Not Magic Number

The user requires "no magic numbers." The parameter O (often denoted R in finance) is the return on the reservation asset.

- **Financial Definition:** O is the opportunity cost of deploying capital.
- **Value:** O = 1.0 implies cash is stable.
- **Friction Adjustment:** In trading, doing nothing costs 0. Trading costs spread + fees. Therefore, the "reservation return" relative to a trade is effectively boosted by the avoided cost.

```
O ≈ 1.0 + Transaction Costs
```

Setting O > 1 forces the model to abstain unless the predicted probability p̂_target is significantly high.

**Mechanism:**

- If the model is unsure (p̂_up ≈ 0.5, p̂_down ≈ 0.5), maximizing log(0.5 + p_res · O) drives the network to shift probability mass to p_res.
- If the model is confident (p̂_up ≈ 0.9), the loss is minimized by shifting mass to p̂_up.

### 3.3 Comparison with SelectiveNet

| Feature              | SelectiveNet (Current)             | Deep Gamblers (Proposed)           |
| -------------------- | ---------------------------------- | ---------------------------------- |
| **Architecture**     | 3 Heads (Pred, Select, Aux)        | 1 Head (K+1 outputs)               |
| **Abstention Logic** | Gated Threshold g(x) > τ           | Argmax includes K+1                |
| **Optimization**     | Coverage vs. Accuracy (Lagrangian) | Portfolio Growth (Kelly)           |
| **Hyperparameters**  | Target Coverage φ, λ weight        | Reservation Return O               |
| **Calibration**      | No explicit calibration            | Inherently calibrated for growth   |
| **Edge Case**        | "Negative Kelly Edge" possible     | Penalizes negative edge explicitly |

### 3.4 Implementation Guidelines for Binary Time Series

1. **Network Modification:** Change the final layer of the BiLSTM to output **3 logits**: [z_up, z_down, z_reserve].
2. **Softmax:** Apply Softmax across all 3 logits.
3. **Inference:**
   - Get probabilities: [p_up, p_down, p_res].
   - **Decision:**
     - If p_res > max(p_up, p_down): **Abstain**.
     - Else: Predict argmax(p_up, p_down).

**GitHub Implementation:** <https://github.com/Z-T-WANG/NIPS2019DeepGamblers>

---

## 4. Key GitHub Repositories

| Method         | Repository                                               | Stars | Notes                   |
| -------------- | -------------------------------------------------------- | ----- | ----------------------- |
| Deep Gamblers  | <https://github.com/Z-T-WANG/NIPS2019DeepGamblers>       | -     | Official NeurIPS 2019   |
| SAOCP          | <https://github.com/salesforce/online_conformal>         | -     | Salesforce official     |
| Conformal PID  | <https://github.com/aangelopoulos/conformal-time-series> | -     | Tested on market data   |
| Energy OOD     | <https://github.com/wetliu/energy_ood>                   | -     | NeurIPS 2020            |
| pytorch-ood    | <https://github.com/kkirchheim/pytorch-ood>              | -     | Unified OOD library     |
| MAPIE          | <https://github.com/scikit-learn-contrib/MAPIE>          | -     | scikit-learn compatible |
| SelectiveNet   | <https://github.com/gatheluck/pytorch-SelectiveNet>      | -     | PyTorch implementation  |
| Learn to Defer | <https://github.com/clinicalml/learn-to-defer>           | -     | ICML 2020               |

---

## 5. Implementation Roadmap

### Phase I: Deep Gamblers Loss (exp080)

- Replace SelectiveNet 3-head with single head + abstention class
- Set O = 1.0 + transaction_costs (e.g., O = 1.001 for 10 bps)
- Compare Kelly edge before/after

### Phase II: Energy-Based OOD Detection (exp081)

- Add energy score computation from BiLSTM logits
- Correlate with fold performance degradation
- Threshold: abstain if E(x) > E_threshold (calibrate on validation)

### Phase III: SAOCP Wrapper (exp082)

- Wrap inference with online conformal prediction
- Use SF-OGD experts for strongly adaptive coverage
- Measure coverage stability across market regimes

### Phase IV: Adaptive Windowing via OneNet (exp083)

- Maintain multiple window branches (short/medium/long)
- RL-based weighting to select optimal lookback
- Replace fixed val_bars=1200
