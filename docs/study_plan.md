# Study Plan Reference: SWIFT-U5

**Survey-Weighted and Temporally Validated Risk Modeling for Under-Five Mortality Using Bangladesh DHS Data**

---

## 1. Gap and Aim
Existing Bangladesh U5M models report inflated accuracy (random splits, SMOTE, no survey weights, leakage), or validate temporally without full survey weighting and with no division-level policy output.  
**Aim:** Measure how validation design and survey weighting change results, then deliver a division-level risk and screening scenario for policy.

---

## 2. Gaps in the Four Anchor Papers
| Paper | Main Gaps Identified |
| :--- | :--- |
| **Naznin 2025** *(U5M, PLOS One)* | Boruta run on full data before split, SMOTE with unclear test handling, random 70/30 split, no survey weights, no calibration, post-outcome predictors (breastfeeding, children ever born), 44% complete-case loss. |
| **Fahim 2026** *(U5M, arXiv)* | Single temporal cut, weights only as sensitivity check, mixed neonatal and later deaths, censoring, non-algorithmic feature selection, borderline division result (n=8), no CIs on NNS. |
| **Haque 2026** *(stunting)* | No survey weights, undersampling, feature set chosen on hold-out test data, weak calibration. |
| **Chandra 2026** *(hypertension)* | Weights only in one model, no formal feature selection, large train to test gap (overfitting). |

---

## 3. Research Questions
1. How much does temporal versus random validation change performance and screening workload?
2. How much do survey weights change prevalence, calibration, and division rankings?
3. Is SWIFT-U5 better calibrated and more stable across rounds than standard models?
4. Which divisions carry the highest risk and best screening yield, and which drivers are shifting?

---

## 4. Data
- **BDHS BR files:** 2004, 2007, 2011, 2014, 2017-18, 2022.
- **Births:** 5 years before each survey, exposure-complete only.
- **Design:** Weights (`v005`), PSU (`v021`), Strata (`v022`).
- **Division Harmonization:** 7 divisions for 2004 to 2014 trends (Mymensingh was part of Dhaka until 2015); 8 divisions for 2017-18 and 2022.

---

## 5. Predictors (Leakage-Controlled)
- **T0 Structural:** Maternal age at birth, maternal education, wealth quintile, residence (urban/rural), division, birth order, preceding birth interval, parity before.
- **T1 Antenatal:** ANC visits, tetanus toxoid (TT), iron supplementation.
- **T2 Birth:** Facility delivery, caesarean delivery, birth size.
- **Explicitly Excluded as Post-Outcome Leakage:** Breastfeeding, child age, children ever born, current vaccination.
- **Missing Data Handling:** Train-only imputation plus explicit missingness indicator flags.

---

## 6. Feature Selection (Training Data Only)
1. **Domain Filter:** Pre-outcome, comparable across rounds, <30% missing, drop one of any collinear pair ($r > 0.8$).
2. **Dual-Model Selection:** Weighted LASSO logistic regression + permutation importance on weighted XGBoost inside PSU-grouped cross-validation folds.
3. **Stability Selection:** Retain features chosen in $\ge 70\%$ of runs by both methods.
4. **Temporal Stability:** Repeat per rolling window to label features as *stable* vs. *drifting*.
5. **SHAP:** Used exclusively for post-hoc interpretation after models are fixed, never for feature selection.

---

## 7. The 2 × 2 Factorial Design
| Arm | Validation Scheme | Survey Weights |
| :---: | :--- | :---: |
| **Arm A** | Random Split | No |
| **Arm B** | Random Split | Yes |
| **Arm C** | Forward-chained temporal | No |
| **Arm D** | Forward-chained temporal | Yes |

- **Matched-n random arm** and random-split result scored on test round only to separate validation effect from test-set composition.
- **Identical features, models, and tuning** in every arm. No SMOTE. Class weights multiplied by survey weights, normalized within round.
- **Forward chaining sequence:** 2004 $\rightarrow$ 2007, 2004–07 $\rightarrow$ 2011, ..., up to 2022.
- **Mixed check:** Train unweighted & evaluate weighted, and vice-versa, to diagnose where weight damage arises.

---

## 8. SWIFT-U5 Framework
**S**urvey-**W**eighted, **I**mportance-corrected, **F**orward-validated **T**emporal risk model for U5M:
- **Module A (Recency & Shift Weighting):** Survey weight $\times$ time decay $\times$ clipped density ratio from a train vs. next-round domain classifier. Time decay rate tuned by forward chaining.
- **Module B (Division-Shrunk Head):** Division intercept and slope on the logit, shrunk toward national values by Empirical Bayes in proportion to observed death counts.
- **Module C (Drift-Aware Recalibration):** Calibration slope and intercept tracked per round and division, recalibrated on latest round, plus SHAP drift index.
- **Baselines:** Weighted logistic regression, XGBoost, Random Forest. Module ablation experiments.

---

## 9. Evaluation
- Metrics: Weighted and unweighted AUROC, AUPRC, Brier score, calibration slope and intercept, sensitivity, PPV, and NNS (Number Needed to Screen) at top 5%, 10%, and 20% screening capacity; Decision Curve Analysis (net benefit).
- Uncertainty: PSU-stratified cluster bootstrap CIs (1,000 replicates) and paired bootstrap tests for differences between arms.

---

## 10. Division Policy Layer
- Risk atlas and trend by division.
- Division NNS and capacity table with 95% CIs.
- Top drivers per division from SHAP.
- Equity audit across wealth quintiles and urban/rural residence.
- Priority table: Expected deaths captured per 1,000 births screened.

---

## 11. Results Structure
1. **Table 1:** Weighted sample, deaths, prevalence by round and division.
2. **Table 2:** 2 × 2 main result with CIs.
3. **Fig 1:** Inflation plot, random versus temporal, across six rounds.
4. **Table 3:** Effect decomposition: validation effect, weight effect, interaction.
5. **Fig 2:** Calibration curves, arms A to D.
6. **Fig 3:** Feature selection frequency, stable versus drifting.
7. **Table 4:** SWIFT-U5 versus baselines per test round.
8. **Table 5:** Ablation of modules.
9. **Fig 4:** Division risk atlas and trend.
10. **Table 6:** Division NNS and capacity.
11. **Fig 5:** SHAP drift by division.
12. **Table 7:** Equity audit.

