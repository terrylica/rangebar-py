---
source_url: https://claude.ai/public/artifacts/27e12556-2f05-40da-a3fb-a092ab121084
source_type: claude-artifact
scraped_at: 2026-02-02T05:35:51Z
purpose: Synthesis of OOD robustness research for financial ML with selective abstention
tags:
  [
    ood-robustness,
    selective-prediction,
    conformal-prediction,
    ensemble-calibration,
    adaptive-windowing,
  ]

# REQUIRED provenance
model_name: Claude
model_version: Sonnet 4 (claude-sonnet-4-20250514)
tools: []

# REQUIRED for Claude Code backtracking + context
claude_code_uuid: 3d0512c5-fe15-4d74-86cc-b2e6ece321e9
claude_code_project_path: "~/.claude/projects/-Users-terryli-eon-alpha-forge-worktree-2025-12-24-ralph-ml-robustness-research/3d0512c5-fe15-4d74-86cc-b2e6ece321e9"

# REQUIRED backlink metadata (to be filled after issue creation)
github_issue_url: https://github.com/EonLabs-Spartan/alpha-forge/issues/131
---

# OOD-Robust Selective Prediction for Non-Stationary Financial Time Series

Your BiLSTM with SelectiveNet architecture operating on cryptocurrency range bars faces a fundamental challenge: maintaining reliable selective prediction under distribution shift without introducing magic numbers. After deep research across **35+ papers and methods**, three approaches emerge as most promising: **Conformal PID Control** for adaptive coverage guarantees explicitly tested on market data, **Deep Gamblers** for its natural trading interpretation and implementation simplicity, and **ensemble disagreement** leveraged through your existing multi-seed setup as a parameter-free uncertainty signal.

The critical finding across all research areas is that **truly parameter-free methods are rare**—most "adaptive" approaches hide hyperparameters. However, several methods achieve effective self-tuning through theoretical defaults or data-driven adaptation. Your experimental finding that val_bars=1200 (43% of train) works best suggests the system benefits from stability over reactivity, making gradual adaptation methods preferable to aggressive changepoint detection.

---

## Adaptive conformal prediction handles non-exchangeability through online threshold adjustment

**Conformal PID Control** (Angelopoulos, Candès, Tibshirani - NeurIPS 2023) stands out as the most directly applicable method, having been **explicitly tested on market returns prediction**. The approach reframes conformal prediction as a control theory problem using proportional-integral-derivative (PID) control, where the coverage rate is the process variable, the quantile threshold is the control variable, and an optional "scorecaster" provides anticipatory adjustment.

The update rule combines three components: the **P-term** reacts to current miscoverage, the **I-term** corrects for cumulative historical errors (providing long-run guarantees), and the **D-term** responds to the rate of change (enabling proactive adjustment before drift fully manifests). The authors provide sensible defaults that work across diverse time series without tuning:

**Implementation**: <https://github.com/aangelopoulos/conformal-time-series> (official)
**Parameter-free?**: Partial—uses theoretical defaults for K_I, saturation constant; the optional scorecaster requires domain knowledge but is not required
**Binary classification applicability**: Via score functions mapping probability outputs to conformal scores

**ACI (Adaptive Conformal Inference)** from Gibbs & Candès (NeurIPS 2021) provides the theoretical foundation. The key insight is modeling distribution shift as a single-parameter online learning problem: at each time t, form a prediction set using current threshold α*t, observe the outcome, compute error as `err_t = 1{y_t ∉ set} - α`, then update `α*{t+1} = α_t + γ·err_t`. This achieves **asymptotic 1-α coverage for ANY sequence**—including adversarial—with no assumptions on the data generating process. The main limitation is the learning rate γ, which later work (DtACI/FACI, AgACI) addresses through automatic tuning or expert aggregation.

**SAOCP (Strongly Adaptive OCP)** from Bhatnagar et al. (ICML 2023) goes further by minimizing strongly adaptive regret—the worst-case regret over ALL local time intervals of any fixed length. It maintains a library of SF-OGD experts with the same learning rate but different active intervals, continuously spawning new experts to react to sudden shifts. Catania & Ferraro (2024) validated SAOCP specifically for cryptocurrency VaR estimation, finding it excels at **extreme quantiles** relevant to tail risk management.

**GitHub**: <https://github.com/salesforce/online_conformal> (official Python)
**R package**: <https://github.com/herbps10/AdaptiveConformal>

**MAPIE library** (scikit-learn-contrib) provides the most accessible entry point with `MapieClassifier` supporting LAC, APS, and RAPS conformity scores for binary classification. However, its time series support via `MapieTimeSeriesRegressor` is regression-focused—adapting it for classification requires custom rolling calibration logic.

---

## OOD detection in feature space requires combining energy scores with Mahalanobis distance

For detecting when test distribution diverges from training, **energy-based OOD detection** (Liu et al., NeurIPS 2020) offers the simplest starting point. The energy score `E(x) = -T·log(Σ_i exp(f_i(x)/T))` computed from BiLSTM logits requires only a single forward pass with no retraining. Lower energy indicates in-distribution; higher energy signals OOD. The method is mostly parameter-free (temperature T=1 works well by default) and has official PyTorch implementation at <https://github.com/wetliu/energy_ood>.

However, energy scores don't detect shift in the **learned feature space**—they only use final logits. **Mahalanobis distance** (Lee et al., NeurIPS 2018) addresses this by fitting class-conditional Gaussians to intermediate features (e.g., BiLSTM final hidden states) and computing distance to the nearest class center: `M(x) = min_c [(f(x) - μ_c)ᵀ Σ⁻¹ (f(x) - μ_c)]`. The **Relative Mahalanobis Distance** variant (Ren et al., 2021) subtracts global background distance, improving near-OOD detection by up to **15% AUROC**.

For non-Gaussian financial features, **KNN-based OOD detection** (Sun et al., ICML 2022) offers a non-parametric alternative using k-th nearest neighbor distance in feature space without distributional assumptions. The **pytorch-ood library** (<https://github.com/kkirchheim/pytorch-ood>) implements Energy, Mahalanobis, KNN, MC Dropout, and more with consistent APIs.

**Uncertainty Drift Detection (UDD)** from Baier et al. (2021) specifically addresses label-free drift detection for financial applications. It combines MC Dropout uncertainty estimates with ADWIN-style windowing on uncertainty values rather than predictions—detecting drift from model uncertainty alone without requiring true labels. This is valuable for live trading where labels arrive with delay.

Recent work **TS-OOD** (arXiv:2502.15901, 2025) provides the first comprehensive benchmark finding that "OOD methods based on deep feature modeling may offer greater advantages for time-series OOD detection." Their evaluation on LSTM backbones suggests combining Energy (fast, no-retrain baseline) with Mahalanobis (feature-space shift) as a robust two-stage approach.

---

## Ensemble disagreement provides a parameter-free uncertainty signal for position sizing

Your high disagreement across random seeds is not a bug—it's a **feature** that provides free uncertainty quantification. The Deep Ensembles paper (Lakshminarayanan et al., NeurIPS 2017) established that ensemble members settle in **distant low-loss valleys**, making their disagreement a meaningful signal of epistemic uncertainty. Critically, MC Dropout explores within a single valley, producing redundant samples—**5 ensemble members significantly outperform 50 MC Dropout samples** for uncertainty quality.

For binary classification, the simplest disagreement metric is **standard deviation of probability predictions**: `σ = std([p_seed1, ..., p_seedM])`. This ranges from 0 (perfect agreement) to 0.5 (maximum disagreement at 50/50 split), is parameter-free, and directly relates to predictable variance. An alternative is **prediction entropy** on the averaged distribution.

Research on "Logit Disagreement" (arXiv:2502.15648, Feb 2025) shows measuring disagreement on **pre-softmax logits** rather than probabilities outperforms mutual information for OOD detection. The **Predictive Diversity Score** from "Scalable Ensemble Diversification" (arXiv:2409.16797) provides another formulation: `η_PDS = (1/C) Σ_c max_m p_m^c(x)`.

For **position sizing with uncertainty**, the research consensus points toward a hybrid approach:

```python
def calculate_position_size(mean_pred, disagreement, base_size=1.0):
    # Hard abstention threshold (step component)
    if disagreement > 0.35:  # ~70th percentile
        return 0.0
    # Proportional scaling within tradeable range
    agreement_factor = 1 - (disagreement / 0.35)
    confidence_factor = abs(mean_pred - 0.5) * 2
    position = base_size * agreement_factor * confidence_factor
    # Fractional Kelly cap (never exceed half-Kelly)
    return min(position, base_size * 0.5)
```

The step function prevents trading on high-uncertainty predictions while proportional scaling optimizes within the tradeable regime. **Uniform weighting** of ensemble members is recommended—optimized weights don't significantly outperform uniform for well-trained ensembles and add unnecessary complexity.

**Calibration under distribution shift** degrades for all methods, but ensembles degrade more gracefully than single models or MC Dropout. Temperature scaling effectiveness diminishes with shift severity. The practical recommendation: rely MORE on disagreement (inherently adaptive) and LESS on static calibration; use rolling isotonic regression if calibration is necessary.

---

## Truly parameter-free adaptive window selection remains elusive

Your finding that ACF-based gap detection performed **worse** than hardcoded gap_bars=50 and PELT had no effect aligns with the research: most adaptive methods hide hyperparameters, and highly homogeneous data (binary direction near i.i.d. at up-rate ≈ 0.496) provides insufficient signal for changepoint detection.

**EDDM (Early Drift Detection Method)** is the only truly parameter-free drift detector found. Unlike DDM's explicit warning/control thresholds or ADWIN's delta parameter, EDDM tracks distance between consecutive errors using hardcoded thresholds derived from PAC learning theory. Implementation is available in both River and scikit-multiflow libraries. Limitation: it requires error labels (predictions vs. actuals) and has higher detection delay than ADWIN—suitable for gradual rather than abrupt drift.

**MDL (Minimum Description Length)** provides the most theoretically principled parameter-free approach for window selection. The principle of minimizing `length(model) + length(data|model)` naturally penalizes overly complex models (long history) without hyperparameters. Davis & Yau (2013) proved MDL consistency for detecting breakpoints in piecewise stationary time series including ARMA and GARCH processes. The implementation requires defining a coding scheme but the selection criterion itself is parameter-free:

```python
for window_size in candidate_sizes:
    model_bits = compute_model_complexity(window_size)
    error_bits = compute_error_encoding_bits(window_size)
    mdl_score[window_size] = model_bits + error_bits
optimal_window = argmin(mdl_score)
```

**Bayesian Forgetting** via the Hierarchical Adaptive Forgetting Variational Filter (HAFVF) from PLOS Computational Biology (2019) offers a principled self-tuning approach. The two-level forgetting mechanism uses Bayesian inference: if new observations are unlikely given the posterior, the forgetting factor increases (reset toward prior); if consistent, it decreases (trust history more). This naturally handles regime detection without explicit changepoint methods.

Your empirical finding that **val_bars = 43% of train works best** may itself be a robust ratio. The research suggests using `val_bars = 0.4 × train_bars` as a stable rule rather than an absolute value, adapting proportionally as training windows change.

---

## Method-by-method applicability analysis

| Method             | Parameter-Free     | Binary OK | Handles Shift  | Open Source                                                | Best For                                 |
| ------------------ | ------------------ | --------- | -------------- | ---------------------------------------------------------- | ---------------------------------------- |
| **Conformal PID**  | Partial (defaults) | Yes       | Yes            | github.com/aangelopoulos/conformal-time-series             | Trending markets, coverage guarantees    |
| **SAOCP**          | No (D, g params)   | Yes       | Yes (strongly) | github.com/salesforce/online_conformal                     | Sudden regime changes, extreme quantiles |
| **ACI/FACI**       | No (γ, adaptive)   | Yes       | Yes            | github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries | General shift, theoretical guarantees    |
| **Energy OOD**     | Yes (T=1)          | Yes       | No             | github.com/wetliu/energy_ood                               | Fast baseline OOD detection              |
| **Mahalanobis**    | Yes                | Yes       | No             | github.com/kkirchheim/pytorch-ood                          | Feature-space shift detection            |
| **Deep Ensembles** | Yes (M=5)          | Yes       | Partial        | github.com/ENSTA-U2IS-AI/torch-uncertainty                 | Uncertainty quantification               |
| **EDDM**           | **Yes**            | Yes       | Detects        | River/scikit-multiflow                                     | Gradual drift detection                  |
| **MDL Window**     | **Yes**            | Yes       | N/A            | Custom                                                     | Principled window selection              |
| **SelectiveNet**   | No (c, λ, α)       | Yes       | No             | github.com/gatheluck/pytorch-SelectiveNet                  | End-to-end trainable abstention          |
| **Deep Gamblers**  | No (o param)       | Yes       | No             | github.com/Z-T-WANG/NIPS2019DeepGamblers                   | Trading interpretation, simplicity       |
| **Learn to Defer** | No (α)             | Yes       | No             | github.com/clinicalml/learn-to-defer                       | Expert-assisted decision-making          |
| **MAPIE**          | Varies             | Yes       | Via ACI        | github.com/scikit-learn-contrib/MAPIE                      | Easy integration, experimentation        |

---

## Implementation roadmap with ceteris paribus experiments

### Phase 1: Leverage existing infrastructure (no architecture changes)

**Experiment 1a**: Implement ensemble disagreement metric (σ of seed probabilities) as abstention signal. Compare to SelectiveNet gate output. Measure coverage-accuracy tradeoff.
**Experiment 1b**: Add energy-based OOD scoring to existing BiLSTM outputs. Correlate energy scores with prediction errors on held-out folds. Determine if energy predicts fold performance.
**Experiment 1c**: Apply Mahalanobis distance to BiLSTM final hidden states. Compare OOD detection performance to energy scores. Evaluate computational overhead for live trading.

### Phase 2: Add adaptive coverage (online threshold adjustment)

**Experiment 2a**: Implement ACI update rule on SelectiveNet gate threshold: `threshold_{t+1} = threshold_t + η·(error_t - α)`. Measure coverage stability across WFO folds compared to static threshold.
**Experiment 2b**: Implement Conformal PID control with scorecaster predicting future conformity scores. Compare coverage variance to ACI. The scorecaster can use a simple autoregressive model on recent scores.
**Experiment 2c**: Test SF-OGD from SAOCP for strongly adaptive coverage. Compare to Conformal PID on synthetic sudden-shift scenarios and across major market events in historical data.

### Phase 3: Replace SelectiveNet loss function (architecture change)

**Experiment 3a**: Convert to Deep Gamblers loss, adding (m+1)-th output neuron. Compare risk-coverage curve to SelectiveNet. Evaluate training stability.
**Experiment 3b**: Implement Learn to Defer loss with "expert" = no-trade outcome. Compare learned deferral policy to SelectiveNet gate.
**Experiment 3c**: Add asymmetric abstention thresholds (τ_long ≠ τ_short). Measure impact on risk-adjusted returns when long/short have different cost structures.

### Phase 4: Adaptive window selection (training pipeline change)

**Experiment 4a**: Replace hardcoded val_bars with ratio-based `val_bars = 0.4 × train_bars`. Validate across different train_bars values.
**Experiment 4b**: Implement EDDM drift detector feeding prediction errors. When drift detected, reduce effective train_bars by 50%. When stable, grow by 10%.
**Experiment 4c**: Implement MDL-based window selection. Compare to EDDM-triggered adaptation and fixed windows on out-of-sample performance.

---

## Open questions the literature doesn't address

**1. Binary classification near i.i.d. challenges all methods.** Your up-rate ≈ 0.496 means direction is nearly a coin flip, leaving little signal for drift detection or OOD methods. The literature assumes sufficient class structure for meaningful confidence scores. Research on selective prediction for **near-random streams** is essentially absent.

**2. Range bar non-uniform timing complicates all temporal methods.** Conformal PID's integral term, ACI's learning rate, and ADWIN's window sizing all implicitly assume uniform time steps. Range bars have variable duration, meaning "recent" samples may span vastly different calendar time. No method explicitly addresses non-uniform temporal spacing in sequential prediction.

**3. Negative Kelly despite positive profit factors indicates distribution mismatch.** This specific symptom—profitable average trades but negative growth rate—suggests the edge evaporates precisely when position sizes are Kelly-optimal. This may indicate time-varying edge that correlates with confidence, making static Kelly (or any fixed fraction) inappropriate. The literature on **adaptive Kelly with uncertainty-adjusted sizing** is thin.

**4. Pareto frontier epoch selection interacts with selective prediction.** Your WFO uses prior_epoch selected on (directional_accuracy, coverage) Pareto frontier. The interaction between epoch selection and abstention policy is unexplored—selecting epochs that optimize coverage may inadvertently select for overconfident gates.

**5. Transaction costs interact non-linearly with abstention.** Higher abstention → fewer trades → lower cumulative costs, but also fewer opportunities. The optimal abstention rate depends on transaction costs, but cost-sensitive selective prediction methods assume costs are per-prediction, not cumulative. The literature lacks models where abstention itself changes the cost structure.

**6. Ensemble seed disagreement vs. SelectiveNet gate redundancy.** Your architecture has both seed disagreement AND a trained selection head. These may capture overlapping uncertainty signals. Research on **combining learned rejection with ensemble uncertainty** for redundant signals is limited.

---

## Synthesis and final recommendations

For your specific cryptocurrency range bar binary direction prediction system with BiLSTM + SelectiveNet, the highest-impact interventions are:

**Immediate (no code changes needed)**: Use `σ = std([p_seed1, ..., p_seedM])` as primary abstention signal. This is fully parameter-free and already available from your multi-seed training. Abstain when σ > 0.25 (calibrate on validation).

**Short-term**: Implement ACI-style adaptive threshold on SelectiveNet gate: update threshold based on recent coverage. This adds robustness to distribution shift with minimal complexity (one learning rate η).

**Medium-term**: Replace SelectiveNet loss with Deep Gamblers—simpler, more stable training, natural trading interpretation. The single hyperparameter `o` is more interpretable than SelectiveNet's (c, λ, α).

**Longer-term**: Add Conformal PID with a simple AR scorecaster for anticipatory adjustment. This has the strongest empirical validation on market data and provides explicit coverage guarantees.

**Avoid**: Adding complexity through SAOCP (marginal gains over ACI for your likely smooth drift), MAML-based meta-learning (insufficient regime diversity to learn meaningful adaptation), or aggressive changepoint detection (your PELT results confirm the signal is too homogeneous).

The fundamental tension in your system is between **stability** (larger windows, gradual adaptation) and **reactivity** (smaller windows, aggressive drift detection). Your empirical evidence—val_bars=1200 best, PELT ineffective—strongly suggests favoring stability. The methods recommended above all bias toward gradual adaptation with strong theoretical guarantees over aggressive but unstable reactivity.
