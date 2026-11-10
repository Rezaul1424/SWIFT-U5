# SWIFT-U5: Final Phase 6 Results Freeze & Claims Policy
**Document ID:** `SWIFT-U5-PHASE6-RESULTS-FREEZE`  
**Date:** November 9, 2026  
**Status:** PERMANENT RESULTS FREEZE (Locked for Publication)  
**Primary Investigators / System:** SWIFT-U5 Core Research Team & Antigravity Agent  

---

## 1. Frozen Master Cohort & Accounting

The analytical cohort for the SWIFT-U5 study is frozen across all six Demographic and Health Surveys (BDHS 2004–2022) conducted in Bangladesh:
- **Eligible Cohort Definition:** All singleton and multiple live births occurring in the 0–59 months prior to maternal interview.
- **Target Outcome:** Recorded under-five mortality ($Y \in \{0, 1\}$), defined as child death before 60 months of age reported in maternal birth histories.
- **Total Harmonized Sample Size:** $N = \mathbf{51{,}489}$ children.
- **Total Recorded Mortality Events:** $\mathbf{2{,}338}$ deaths ($4.54\%$ unweighted prevalence; $4.12\%$ design-weighted prevalence).

### Frozen Wave-Level Accounting

| Survey Wave | Interview Years | Sample ($N$) | Deaths ($n$) | Unweighted Prevalence (%) | Survey-Weighted Mortality Rate (‰) | Partition Role in Primary Benchmark (Track A) |
|---|---|---|---|---|---|---|
| **BDHS 2004** | 2004 | 6,908 | 484 | 7.01% | 71.9 ‰ | Historical Development ($df\_train$) |
| **BDHS 2007** | 2007 | 6,150 | 361 | 5.87% | 55.9 ‰ | Historical Development ($df\_train$) |
| **BDHS 2011** | 2011 | 8,742 | 410 | 4.69% | 44.8 ‰ | Historical Development ($df\_train$) |
| **BDHS 2014** | 2014 | 7,886 | 319 | 4.05% | 41.0 ‰ | Historical Development ($df\_train$) |
| **BDHS 2017-18**| 2017–2018 | 8,684 | 356 | 4.10% | 40.3 ‰ | Forward Validation & Calibration Anchor ($df\_val$) |
| **BDHS 2022** | 2022 | 13,119 | 408 | 3.11% | 29.9 ‰ | Locked Prospective Out-of-Time Test ($df\_test$) |
| **Total Cohort**| 2004–2022 | **51,489** | **2,338** | **4.54%** | **41.2 ‰** | Mutually Exclusive & Completely Exhaustive |

---

## 2. Frozen Track A Prospective Evaluation Protocol

1. **Partition Definitions:**
   - **Development Partition ($df\_train$):** Waves 2004, 2007, 2011, 2014 ($N = 29,686$, deaths = $1,574$). Used for feature engineering, baseline training, and density-ratio base estimation.
   - **Validation Partition ($df\_val$):** Wave 2017-18 ($N = 8,684$, deaths = $356$). Strictly held out from development training. Used for Module C drift recalibration fitting, Module B shrinkage tuning, and Youden cutoff selection.
   - **Prospective Test Partition ($df\_test$):** Wave 2022 ($N = 13,119$, deaths = $408$). Locked prospective out-of-time evaluation cohort.
   - **Partition Exhaustiveness:** Development ($N=29,686$) + Validation ($N=8,684$) + Test ($N=13,119$) = **$51,489$** children; deaths: $1,574 + 356 + 408 = \mathbf{2,338}$ deaths.
2. **Strict 2022 Label Quarantine:**
   - **Zero** 2022 mortality outcome labels ($Y_{2022}$) were accessed during model training, feature selection, hyperparameter tuning, Module B shrinkage fitting, Module C drift recalibration, or threshold selection.
   - All classification cutoffs ($\tau_w = 0.04546$, $\tau_{uw} = 0.04758$) were locked on 2017-18 validation predictions prior to evaluating 2022 data.
3. **Transductive Covariate Adaptation Disclosure:**
   - Module A explicitly utilizes **unlabeled 2022 covariates** ($X_{2022}$) to estimate the target density ratio $w(x) = p_{2022}(x) / p_{\text{hist}}(x)$.
   - This reflects a realistic transductive public health deployment setting where target-population census features are accessible, but prospective mortality outcomes remain unknown. It is rigorously isolated from prohibited label leakage.

---

## 3. Frozen Primary Prospective Results (BDHS 2022 Test Cohort)

All prospective benchmark results on the locked 2022 test cohort ($N = 13,119$, $408$ deaths, weighted prevalence $2.99\%$) are frozen as follows:

| Metric Category | Performance Metric | Primary SWIFT-U5 (Full Framework) | Baseline Weighted XGBoost | Baseline Weighted Logistic | Baseline Weighted Random Forest | Uninformative Null Baseline |
|---|---|---|---|---|---|---|
| **Discrimination** | **Weighted AUROC** [95% CI] | **0.625** [0.587, 0.663] | 0.638 [0.603, 0.673] | 0.609 [0.571, 0.643] | 0.634 [0.599, 0.669] | 0.500 [Reference] |
| | **Weighted AUPRC** | **0.079** | 0.083 | 0.066 | 0.073 | 0.030 [Base Rate] |
| **Calibration** | **Weighted Brier Score** | **0.0287** | 0.1649 | 0.2091 | 0.1346 | 0.0290 [Null Brier] |
| | **Calibration Slope** ($\beta$) | **0.806** | 1.012 | 0.830 | 0.994 | N/A |
| | **Calibration Intercept** ($\alpha$)| **-0.905** | -3.135 | -3.385 | -2.919 | 0.000 |
| **Operational Triage**| **Screening Yield @ 10%** | **77.0 deaths / 1k** | 81.9 deaths / 1k | 84.7 deaths / 1k | 74.7 deaths / 1k | 29.9 deaths / 1k |
| | **Sensitivity @ 10%** | **24.60%** [19.8, 30.4] | 26.2% [21.1, 31.8] | 27.1% [22.0, 32.7] | 23.9% [19.0, 29.5] | 10.0% [Random] |
| | **NNS @ 10% Capacity** | **13.0** [10.4, 16.9] | 12.2 [10.1, 15.5] | 11.8 [9.8, 14.8] | 13.4 [10.9, 17.5] | 33.4 [Random] |
| | **Efficiency Ratio @ 10%** | **2.57×** over random | 2.73× over random | 2.83× over random | 2.50× over random | 1.00× [Reference] |
| | **NNS @ 5% Capacity** | **8.8** [6.9, 12.0] | 8.5 [6.7, 11.4] | 8.1 [6.4, 10.8] | 9.7 [7.5, 13.5] | 33.4 [Random] |

*Note on Baselines:* Baseline models (XGBoost, Logistic Regression, Random Forest) output uncalibrated discrimination scores distorted by class imbalance reweighting (`scale_pos_weight`), producing severe probability overconfidence (Brier scores $0.1346$--$0.2091$ vs. $0.0290$ for the null model, and intercepts $-2.92$ to $-3.38$).

---

## 4. Frozen Factorial Design & Methodological Interpretation

### 4.1 2×2 Factorial Architecture

The study preserves the 2×2 factorial design:
- **Factor 1:** Split Architecture (In-Distribution Random Split vs. Prospective Forward-Chained Temporal Split).
- **Factor 2:** Weighting Strategy (Unweighted vs. Survey / Module A Weighted).

### 4.2 Clarification of Experiment Configurations: Factorial Arm C vs. Track A SWIFT-U5

The repository contains two distinct 2022 temporal evaluation results that must be clearly distinguished:
1. **Factorial Benchmark (Arm C: Temporal / Unweighted, Table 2):** AUROC = **0.6568 (0.657)**.
2. **Primary Prospective Track A Framework (SWIFT-U5, Table 4):** AUROC = **0.6253 (0.625)**.

| Dimension | Factorial Benchmark (Arm C: Temporal / Unweighted) | Primary Prospective Benchmark (Track A: SWIFT-U5) |
|---|---|---|
| **Role in Study** | Retrospective methodological diagnostic (Table 2) | Primary prospective clinical/public health pipeline (Table 4) |
| **Model Configuration** | Raw XGBoost learner alone (No Modules A, B, or C) | Full 3-module framework (Module A + Module B + Module C) |
| **Training Partition** | **All 5 historical waves (2004–2018, $N=38,370$)** | **Development waves (2004–2014, $N=29,686$)** |
| **Validation Partition** | None (All historical data consumed in training) | **Held-out 2017-18 wave ($N=8,684$) for calibration/thresholds** |
| **Weighting Regime** | Completely unweighted ($\text{sample\_weight} = 1$) | Module A recency decay + transductive density-ratio shift weights |
| **Calibration Status** | **Uncalibrated raw scores** (Brier = $0.1763$, Cal Intercept $\approx -3.14$) | **Calibrated probabilities** (Brier = $0.0287$, Cal Intercept = $-0.905$) |
| **2022 Test AUROC** | **0.6568** [0.615, 0.682] | **0.6253** [0.587, 0.663] |

**Why the Results Differ:**
- **Training Set Composition:** Factorial Arm C trained directly on 2017-18 data ($N=8,684$, 356 deaths) alongside earlier waves ($N_{\text{train}} = 38,370$), providing recent pre-2022 feature associations. In contrast, Track A held out 2017-18 entirely from base training ($N_{\text{dev}} = 29,686$) to enable principled drift recalibration. Including 2017-18 in training accounts for $+0.019$ AUROC ($0.657$ vs. $0.638$ for unaugmented base model on 2004–2014).
- **Discrimination vs. Calibration Tradeoff:** Raw unweighted XGBoost in Arm C optimizes empirical binary cross-entropy on raw sample labels, maximizing sample pairwise concordance (AUROC $0.657$) but producing catastrophic probability miscalibration (Brier score $0.1763$). SWIFT-U5 optimizes population-calibrated risk via survey density-ratio weighting and recalibration, which substantially improves probability scaling (Brier score $0.0287$, beating the null) while shifting borderline scores to reflect true population risk, yielding an AUROC of $0.625$.

### 4.3 Frozen Factorial Effect Decomposition (XGBoost Benchmark)

- **Random Split Inflation ($\text{Arm A} - \text{Arm C}$):** $+0.0615$ AUROC on 2022 ($+0.063$ average across multi-round cuts). Standard random cross-validation yields optimistic out-of-sample estimates.
- **Survey Weighting Effect ($\text{Arm B} - \text{Arm A}$):** $-0.0123$ AUROC on 2022. Design-weighted loss optimizes population probability calibration at a slight cost to sample pairwise rank concordance.
- **Factorial Interaction ($(\text{Arm D} - \text{Arm C}) - (\text{Arm B} - \text{Arm A})$):** $+0.0034$ AUROC on 2022.
- **Provenance Track Distinction:** Track B (Random Split) contains 2022 survey records in the training fold by definition of random splitting. It is frozen exclusively as a retrospective methodological diagnostic and is strictly segregated from Track A prospective claims.

---

## 5. Frozen Calibration & Decision Curve Claims

1. **Temporal Cut Calibration Provenance:**
   - **Cut 1 (2007):** In-sample calibration anchor (2004). Frozen as **Exploratory In-Sample Calibration**.
   - **Cuts 2, 3, 4, 5:** Penultimate-round held-out calibration anchors (2007, 2011, 2014, 2017-18). Frozen as **Principled Out-of-Time Calibrations**.
2. **Decision Curve Analysis (DCA) Decision-Analytic Utility Bands:**
   - Evaluated on a pre-specified 0.5-percentage-point threshold grid ($p_t \in \{0.010, 0.015, \dots, 0.100\}$).
   - **Universal Intervention Zone ($p_t \le 0.020$):** Treat-All is preferred or clinically equivalent ($p_t < \bar{\pi}_w$). Model triage is not indicated.
   - **Defensible Positive Net Benefit Zone ($0.025 \le p_t \le 0.045$):** At evaluated grid points $p_t \in \{0.025, 0.030, 0.035, 0.040, 0.045\}$, SWIFT-U5 achieves positive decision-analytic net benefit over Treat-All (which becomes net-negative above prevalence) and Treat-None ($NB = 0$), with standardized net benefit between $2.1\%$ and $21.0\%$ ($NB = +0.00062$ to $+0.00628$).
   - **Clinical Parity Zone ($0.050 \le p_t \le 0.060$):** Decision-analytic net benefit is negligible ($+0.00005$ to $+0.00020$). Clinically indistinguishable from Treat-None.
   - **High-Specificity Triage Zone ($p_t \ge 0.065$):** Restricts triage to top $\le 10\%$ down to $2.7\%$ of infants.

---

## 6. Permitted and Prohibited Manuscript Claims

To uphold scientific integrity and prevent reviewer challenge, authors and automated agents must adhere to the following claims policy:

### Permitted Claims

| Claim Domain | Authoritative Permitted Formulation |
|---|---|
| **Screening Efficiency** | "At a 10% screening capacity nationwide, SWIFT-U5 captures 24.6% of national under-five deaths (NNS = 13.0, 77.0 deaths captured per 1,000 screened), achieving a 2.57-fold efficiency gain over random household triage." |
| **Probability Calibration** | "SWIFT-U5 substantially improves probability calibration, reducing the survey-weighted Brier score from 0.1649 (uncalibrated baseline XGBoost) to 0.0287, outperforming the uninformative null benchmark (0.0290) and achieving a calibration slope of 0.806." |
| **Geographic Disparity** | "Screening yield varied geographically, capturing 123.3 deaths per 1,000 screened in Rangpur (NNS = 8.1, 3.24-fold gain) versus 38.1 per 1,000 in Dhaka (NNS = 26.2)." |
| **Validation Leakage** | "Random cross-validation overestimates prospective out-of-time AUROC by +0.061 on the 2022 cohort (+0.063 multi-round mean), demonstrating the necessity of temporal forward-chaining for public health models." |
| **Decision-Analytic Net Benefit**| "Decision Curve Analysis on the evaluated 0.5-percentage-point grid establishes positive decision-analytic net benefit over Treat-All and Treat-None across risk thresholds from 2.5% to 4.5% ($p_t \in \{0.025, 0.030, 0.035, 0.040, 0.045\}$)." |

### Prohibited Claims

| Claim Domain | Strictly Prohibited Formulation | Exact Scientific Reason for Prohibition |
|---|---|---|
| **Causal Impact** | ❌ "Deploying SWIFT-U5 will save X children's lives." | Study is retrospective observational; ranking concordance does not establish causal public health intervention efficacy. |
| **Survival Completion** | ❌ "Living children under age five survived the full five-year follow-up." | Survey records age at interview; living children aged 0–59 months remain at risk of subsequent mortality. Outcome is recorded death by interview. |
| **Pure Causal Split Effect** | ❌ "Temporal splitting causes a 0.06 drop in intrinsic model discriminative ability." | Split strategy changes the test cohort distribution; difference reflects temporal covariate shift and secular trend, not purely split mechanics. |
| **Accuracy Framing** | ❌ "SWIFT-U5 achieves strong overall classification performance with 69.91% accuracy." | Under 33:1 imbalance (2.99% prevalence), predicting zero deaths achieves 97.01% accuracy. Standalone 69.91% accuracy is an operational recall tradeoff, not standalone excellence. |
| **Extended DCA Utility** | ❌ "SWIFT-U5 provides positive clinical utility up to a 6.0% risk threshold or continuously across all thresholds." | Net benefit at 5.0%–6.0% is +0.00005 to +0.00020, representing clinical equivalence with Treat-None. DCA was evaluated on a discrete 0.5% grid. |
| **Subgroup Fairness Proof** | ❌ "Algorithmic fairness is certified among uneducated mothers." | Subgroup contains only 28 death events; calibration slope (1.33, 95% CI: 0.81–1.85) has wide error bounds and is designated as Insufficient Evidence ($n < 50$). |
| **Zero Test Information** | ❌ "SWIFT-U5 operates with zero access to 2022 test survey data of any kind." | Module A explicitly uses unlabeled 2022 survey covariates for transductive density ratio estimation. Only 2022 mortality labels were quarantined. |
| **Significance Claims** | ❌ "SWIFT-U5 significantly improves calibration." | Replace inferential terms with descriptive statements (e.g. Brier score reduced from 0.1649 to 0.0287) unless a formal statistical hypothesis test is reported. |
