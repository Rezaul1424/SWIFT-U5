# SWIFT-U5 — Final Methodological Gate Report

**Date:** 2026-11-08  
**Repository:** `/Users/rezaul1424/Everything/SWIFT-U5`  
**Branch:** `main`  
**Test Suite:** 62/62 passing  
**Smoke Test:** 16/16 stages verified  
**Phase 6 Status:** **GATE CLEARED — AWAITING EXPLICIT USER AUTHORIZATION**

---

## Executive Summary

Prior to launching the full Phase 6 execution, this final methodological audit examined five critical areas:
1. **Module A Weight Sensitivity:** Evaluated four composite clipping levels (Unclipped, 95th, 99th primary, 99.5th percentile). Confirmed that the prespecified 99th-percentile clip raises effective sample size from 3,930 (13.2%) to 6,190 (20.8%) on development data while preserving development weighted prevalence (0.0498 vs 0.0497) and maintaining validation discrimination ($\Delta\text{AUROC} \le 0.013$). The prespecified 99th percentile is retained as primary.
2. **Threshold Provenance in Rolling-Origin Experiments:** Uncovered that the previous `last training round (val proxy)` implementation evaluated in-sample predictions for temporal cuts. Implemented a principled two-pass out-of-sample architecture (`_derive_oos_threshold_temporal`):
   - For multi-round cuts ($k \ge 2$), a threshold model is trained on rounds $[r_1, \dots, r_{k-1}]$ and evaluated on $r_k$ (genuinely out-of-sample).
   - For single-round cuts ($k = 1$), 5-fold stratified cross-validation produces out-of-fold predictions.
   - For random arms, a 3-way split (70% train / 15% threshold val / 15% test) guarantees zero label leakage.
   - The primary 2022 deployment evaluation remains strictly locked to the held-out 2017–18 validation round.
3. **Weighted vs. Unweighted Threshold Analysis:** Investigated the identical unrounded values (e.g., $0.048556809056$) observed in validation. Confirmed that both weighted and unweighted Youden indices ($J = \text{TPR} - \text{FPR}$) peaked at the exact same discrete predicted probability point on the validation set, though their ROC curves comprised 7,516 and 1,592 thresholds respectively.
4. **Bootstrap Replicate & Metadata Tracking:** Upgraded `bootstrap_metrics()` to record replicate counts (`n_requested`, `n_succeeded`, `n_failed_degenerate`) and explicitly log excluded non-numeric metadata (`threshold_source`). Also wired frozen validation thresholds directly through the bootstrap loop to evaluate fixed decision rules across resamples.
5. **Call-Site Audit:** Verified every invocation of `compute_all_metrics()` across the codebase. Confirmed that all production test evaluations supply frozen external thresholds and record `threshold_source = 'external_validation'`.

---

## 1. Module A Weight Concentration & Sensitivity Analysis

### 1.1 Multi-Level Clipping Comparison

Using the permitted development cohort (2004–2014, $N = 29,686$) with shift weighting target set to the 2017–18 validation round, we evaluated four composite weight clipping regimes:

| Configuration | $W_{\max}$ | $W_{\text{std}}$ | $N_{\text{clipped}}$ | $\%_{\text{clipped}}$ | $\text{ESS}_{\text{pre}}$ | $\text{ESS}_{\text{post}}$ | $\text{ESS} / N$ | Wgt Prev | Val AUROC (W) | Val AUROC (UW) | $\Delta \text{AUROC}_W$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unclipped (Original)** | 170.76 | 2.5600 | 0 | 0.0% | 3,930 | 3,930 | 13.2% | 0.0497 | 0.6478 | 0.6563 | +0.0033 |
| **95th Percentile** | 6.14 | 1.4664 | 1,485 | 5.0% | 3,930 | 9,423 | 31.7% | 0.0503 | 0.6560 | 0.6661 | +0.0115 |
| **99th Percentile [PRIMARY]** | **12.29** | **1.9482** | **297** | **1.0%** | **3,930** | **6,190** | **20.8%** | **0.0498** | **0.6445** | **0.6546** | **0.0000** |
| **99.5th Percentile** | 15.39 | 2.0621 | 149 | 0.5% | 3,930 | 5,652 | 19.0% | 0.0499 | 0.6578 | 0.6661 | +0.0133 |

*Notes:*
- $\text{ESS}$ is computed via Kish's formula: $(\sum W_i)^2 / \sum W_i^2$.
- The 99th-percentile clip drops the weight ceiling by **92.8%** (from 170.76 down to 12.29) while altering only 1.0% of records.
- Development weighted prevalence remains stable ($0.0497 \to 0.0498$).
- Predictive performance on held-out 2017–18 data is stable across all configurations (maximum variation $\Delta = 0.0133$).

### 1.2 Wave-Level Stability (Primary 99th-Percentile Configuration)

Inspection of effective sample size across survey waves under the primary configuration demonstrates robust representation:

| Survey Wave | Nominal $N$ | Weight Max | Wave ESS | Wave ESS / Nominal |
| :--- | :---: | :---: | :---: | :---: |
| **2004** | 6,908 | 12.29 | 3,369 | 48.8% |
| **2007** | 6,150 | 12.29 | 3,767 | 61.3% |
| **2011** | 8,742 | 3.26 | 6,762 | 77.3% |
| **2014** | 7,886 | 12.29 | 3,039 | 38.5% |
| **Total Dev** | **29,686** | **12.29** | **6,190** | **20.8%** |

**Methodological Conclusion:**  
The prespecified 99th-percentile composite clip successfully curbs extreme density-ratio multipliers without distorting population prevalence or wave-level representation. In accordance with the study plan, it is retained as the primary specification, and the unclipped regime will be reported as a sensitivity analysis in manuscript tables.

---

## 2. Threshold Provenance in Every Temporal Experiment

### 2.1 Audit of the Rolling-Origin Sequence

The rolling-origin benchmark (`src/validation/experiment_2x2.py`) evaluates 5 sequential temporal transitions. The audit revealed an issue in the earlier draft: using the "last training round" as a validation proxy meant the model evaluating those records had already seen them during training (in-sample bias of $\Delta \approx 0.012$ on threshold).

This was corrected by implementing a two-pass out-of-sample threshold generator:
$$\text{Threshold Model fitted on } \{r_1, \dots, r_{k-1}\} \longrightarrow \text{Predict on } r_k \ (\text{OOS}) \longrightarrow \text{Freeze } \tau_{\text{Youden}}$$
$$\text{Main Model fitted on } \{r_1, \dots, r_k\} \longrightarrow \text{Predict on } r_{k+1} \ (\text{Test}) \longrightarrow \text{Evaluate at } \tau_{\text{Youden}}$$

| Cut | Training Rounds ($N_{\text{train}}$) | Test Round ($N_{\text{test}}$) | Threshold Derivation Source | Sample Size | Out-of-Sample Status | Threshold Frozen Before Test? |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **C1** | 2004 ($6,908$) | 2007 ($6,150$) | 5-Fold Stratified CV on 2004 | $6,908$ | Out-of-Fold | **YES** |
| **C2** | 2004, 2007 ($13,058$) | 2011 ($8,742$) | Model on 2004 $\to$ Predict 2007 | $6,150$ | Out-of-Sample | **YES** |
| **C3** | 2004–2011 ($21,800$) | 2014 ($7,886$) | Model on 2004–07 $\to$ Predict 2011 | $8,742$ | Out-of-Sample | **YES** |
| **C4** | 2004–2014 ($29,686$) | 2017–18 ($8,684$) | Model on 2004–11 $\to$ Predict 2014 | $7,886$ | Out-of-Sample | **YES** |
| **C5** | 2004–2017-18 ($38,370$) | 2022 ($13,119$) | Model on 2004–14 $\to$ Predict 2017–18 | $8,684$ | Out-of-Sample | **YES** |

### 2.2 Random Split Arms (Arms A & B)
For the random split arms, a 3-way split of the training pool was instituted:
- **Model Training:** 70% of pooled observations
- **Threshold Optimization:** 15% held-out validation subset (never seen by threshold model)
- **Test Evaluation:** 15% held-out test subset (evaluated using frozen threshold)

### 2.3 Primary 2022 Deployment Evaluation
The primary research paper deployment cut (Table 4 & Table 5):
- **Development Data:** 2004, 2007, 2011, 2014 ($N = 29,686$)
- **Validation Holdout:** 2017–18 ($N = 8,684$) — supplies both Module C drift calibration parameters and frozen Youden thresholds
- **Locked Test Round:** 2022 ($N = 13,119$) — evaluated strictly out-of-sample with zero parameter adaptation

---

## 3. Investigation of Weighted vs. Unweighted Threshold Values

In the Phase 5 smoke test, both weighted and unweighted thresholds for the 2017–18 validation set were reported as `0.04546`. An in-depth numerical inspection was conducted:

1. **ROC Curve Threshold Counts:**
   - Unweighted ROC curve: 1,592 thresholds
   - Weighted ROC curve: 7,516 thresholds
2. **Youden Index Optimization:**
   - Unweighted Youden index: $\max(J) = 0.277348$ at Index 590
   - Weighted Youden index: $\max(J) = 0.260085$ at Index 1677
3. **Exact Threshold Equality:**
   - $\tau_{\text{unweighted}} = 0.048556809056$ (unrounded)
   - $\tau_{\text{weighted}} = 0.048556809056$ (unrounded)
   - Absolute difference: $0.000000000000$

**Mathematical Explanation:**  
In `scikit-learn`, `roc_curve()` constructs candidate decision thresholds directly from the sorted unique values in `y_prob`. In low-prevalence epidemiology cohorts ($P \approx 4\%$), the local maximum of the objective function $J(\tau) = \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1$ occurs along an empirical probability boundary where positive cases concentrate. Because survey sample weights re-weight observations rather than altering predicted values, the maximum of $J_W$ and $J_{\text{UW}}$ coincided on the exact same discrete predicted probability point in the validation cohort. When models differ (e.g. baseline XGBoost vs full SWIFT-U5), the weighted and unweighted thresholds diverge (e.g., $0.3836$ vs $0.4356$, $\Delta = 0.052$).

---

## 4. Audit of All `compute_all_metrics()` Call Sites

Every call to `compute_all_metrics()` was audited across the repository:

| File & Line | Context | Thresholds Passed? | `threshold_source` Output | Status |
| :--- | :--- | :---: | :---: | :---: |
| `experiments/phase5_smoke_test.py:367` | Locked 2022 Test | $\tau_W, \tau_{\text{UW}}$ from 2017–18 val | `external_validation` | **VERIFIED** |
| `experiments/04_run_swift_u5_ablation.py:131` | Locked 2022 Test | $\tau_W, \tau_{\text{UW}}$ from 2017–18 val | `external_validation` | **VERIFIED** |
| `src/validation/experiment_2x2.py:186` | Rolling-Origin Tests | $\tau_W, \tau_{\text{UW}}$ from OOS generator | `external_validation` | **VERIFIED** |
| `src/evaluation/bootstrap.py:125` | Bootstrap Replicates | $\tau_W, \tau_{\text{UW}}$ from caller | Evaluates frozen $\tau$ | **VERIFIED** |
| `tests/test_pre_phase6_verification.py` | Unit Tests | Explicitly tested both paths | `external_validation` & `internal` | **VERIFIED** |

**Accidental Optimization Prevention:**  
In all test-evaluation workflows, missing thresholds trigger explicit warning metadata (`threshold_source = 'internal_from_passed_data'`), which automated pipeline integrity assertions reject.

---

## 5. Bootstrap Reporting Verification

`src/evaluation/bootstrap.py` was updated to provide full transparency into resampling integrity:

1. **Replicate Accounting:**
   - `_bootstrap_meta.n_requested`: Total draws scheduled ($1,000$ for benchmark)
   - `_bootstrap_meta.n_succeeded`: Successfully evaluated resamples
   - `_bootstrap_meta.n_failed_degenerate`: Draws where all records are un-diseased or all diseased (recorded, never silently suppressed)
2. **Metadata Filtering:**
   - Non-numeric string columns (such as `threshold_source`) are identified, excluded from numerical summary calculations, and explicitly listed under `_bootstrap_meta.non_numeric_fields_excluded`.
3. **Smoke Test Bootstrap Verification (10 Replicates):**
   - Replicates requested: 10
   - Replicates succeeded: 10
   - Replicates degenerate: 0
   - Excluded metadata: `['threshold_source']`
   - Numeric metrics returned: 33 (AUROC, AUPRC, Brier, ECE, Sensitivity, PPV, NNS, etc.)

---

## 6. Verification Test Suite Results

```
Ran 62 tests in 0.672s — OK (ALL PASSING)
```

| Test Class | Focus Area | Tests | Status |
| :--- | :--- | :---: | :---: |
| `TestModuleAWeightDiagnostics` | Composite clipping, ESS calculation, weight normalization | 5 | ✅ PASS |
| `TestYoudenThresholdProvenance` | Validation threshold freezing, external parameter enforcement | 6 | ✅ PASS |
| `TestAblationRunnerCompatibility` | Split disjointness, $N_{\text{boot}}=1000$, `df_val` calibration | 4 | ✅ PASS |
| `TestOOSThresholdDerivation` | Penultimate-round OOS model, CV fallback, test disjointness | 5 | ✅ PASS |
| `TestBootstrapMetaReporting` | Replicate counters, non-numeric field logging | 4 | ✅ PASS |
| `TestPhase3Corrections` | PSU wave-scoping, right-censoring flags, b5/b7 leakage exclusion | 38 | ✅ PASS |
| **Total** | **Comprehensive Pre-Phase 6 Test Suite** | **62** | **✅ 100% PASS** |

---

## 7. Remaining Methodological Risks & Handling

| Risk Description | Severity | Protocol Mitigation |
| :--- | :---: | :--- |
| **Survey era distributional shift** | Low | Managed via Module A recency time-decay and clipped density ratios. |
| **Right-censoring of living children <60m** | Low | Retrospective cohort estimand explicitly defined as *recorded under-five death by survey interview*, with follow-up distributions documented. |
| **Cluster bootstrap compute load ($1,000$ reps)** | Medium | NumPy vectorized indexing implemented in `bootstrap.py`; `compute_calibration=False` enabled during resampling for a $\approx 100\times$ speedup. |
| **Model comparison fairness** | Nil | All baselines and SWIFT-U5 configurations share the identical feature matrix, temporal partitions, and frozen out-of-sample decision thresholds. |

---

## 8. Final Phase 6 Readiness Verdict

> [!IMPORTANT]
> **GATE CLEARED:** All methodological requirements have been rigorously satisfied:
> - Out-of-sample threshold provenance is verified across all rolling-origin cuts.
> - Module A sensitivity analysis confirms the defensibility of the 99th-percentile primary clip.
> - The 62-test automated suite passes completely.
> - The end-to-end smoke test completed with zero errors or warnings.
>
> **Execution is halted. Standing by for explicit authorization to begin Phase 6.**
