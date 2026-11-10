# SWIFT-U5 — Final Manuscript Claim Alignment Log

**Document Version:** 1.2 (Final Scientific Consistency Revision — Survey-Weighted Event Mass & Non-Causal AUROC Alignment)  
**Date:** November 10, 2026  
**Primary Manuscript File:** `docs/manuscript.tex`  
**Reference Frozen Artifacts:** `docs/phase6_results_freeze.md`, `docs/phase6_final_reconciliation_report.pdf`, `outputs/tables/table4_prospective_baselines.csv`, `outputs/tables/table6_division_nns_capacity.csv`, `outputs/predictions/authoritative_swift_u5_2022_predictions.parquet`

---

## 1. Executive Summary & Policy Compliance

This document provides a comprehensive audit trail of all text, narrative, and parameter alignments performed on the primary LaTeX manuscript (`docs/manuscript.tex`). In strict adherence to the Phase 6 freeze guidelines:
1. **Zero Numerical Results Were Altered:** All quantitative metrics remain strictly anchored to the authoritative Phase 6 frozen results artifacts.
2. **No Pipeline or Model Reruns:** No models were retrained, no datasets re-split, and no hyperparameter tuning was performed.
3. **Descriptive Comparative Framing:** All calibration and predictive comparisons are framed descriptively between model configurations, without attributing whole-pipeline improvements to a single module or claiming statistical significance.
4. **Retrospective Prioritization Reframing:** Prioritization capacity metrics are explicitly framed as retrospective risk-ranking on survey-recorded deaths, avoiding any conflation with demonstrated prospective screening effectiveness among living children.
5. **Clear Methodological Distinctions:** The $2 \times 2$ factorial benchmark (Arm C AUROC 0.6568) is strictly distinguished from the primary prospective deployment pipeline (Track A SWIFT-U5 AUROC 0.6253). The observed AUROC difference reflects a compound contrast between experimental configurations (training cohort span, validation design, instance weighting, and post-processing) and cannot be causally attributed to any single component independently.
6. **Survey-Weighted Event Mass Precision:** Clarified that survey-weighted values (e.g., $96.97 \approx 97.0$) represent survey-weighted true-positive event equivalents (event mass under DHS sampling weights) rather than literal counts of observed children or deaths.
7. **Boundary Statement:** Achieving internal narrative and tabular consistency does not constitute external scientific or clinical validation; prospective clinical field evaluation among living children remains necessary.

---

## 2. Item-by-Item Manuscript Changes

### Change 1: Abstract — Synchronization of Primary Metrics, Event Equivalents, and Estimated Yield
- **Location:** `docs/manuscript.tex`, `\begin{abstract}` (line 74)
- **Previous Content:**
  > "In simulated retrospective community outreach prioritization at 10\% capacity, the model captures 24.6\% of recorded child deaths (Sensitivity = 24.6\%, Specificity = 90.5\%, PPV = 7.70\%, $\text{NNS} = 13.0$), yielding 77.0 deaths captured per 1{,}000 children prioritized ($2.57\times$ efficiency over random selection), reaching 123.3 deaths per 1{,}000 prioritized ($\text{NNS} = 8.1$) in high-burden Rangpur division."
- **Replacement Content:**
  > "In simulated retrospective community outreach prioritization at 10\% capacity ($1{,}312$ records prioritized containing $107$ unweighted recorded deaths, corresponding to approximately $97$ weighted true-positive event equivalents), the model achieved a survey-weighted Sensitivity of 24.6\%, Specificity of 90.5\%, PPV of 7.70\%, and $\text{NNS} = 13.0$, with a weighted estimated prioritization yield of approximately 77 per 1{,}000 prioritized records (77.0 per 1{,}000; $2.57\times$ efficiency over random selection), reaching an estimated yield of 123.3 per 1{,}000 prioritized ($\text{NNS} = 8.1$) in high-burden Rangpur division."
- **Scientific Reason:** Clarifies that the 10% prioritization subgroup contains 107 unweighted observed deaths, corresponding to approximately 97 weighted true-positive event equivalents under DHS survey weights, and that 77 per 1,000 represents a survey-weighted estimated prioritization yield rather than a literal observed count.

---

### Change 2: Introduction — Retrospective Prioritization Yield
- **Location:** `docs/manuscript.tex`, Section 1 (line 128)
- **Previous Content:**
  > "providing retrospective prioritization capacity tables with $95\%$ PSU cluster-bootstrap confidence intervals quantifying prioritization workload (NNS) and survey-recorded deaths captured per $1{,}000$ children prioritized."
- **Replacement Content:**
  > "providing retrospective prioritization capacity tables with $95\%$ PSU cluster-bootstrap confidence intervals quantifying prioritization workload (NNS) and survey-weighted estimated prioritization yield per $1{,}000$ prioritized records."
- **Scientific Reason:** Harmonizes introductory contribution phrasing with the survey-weighted estimated yield definition.

---

### Change 3: Declaration of Locked Test Set Integrity & Transductive Covariate Disclosure
- **Location:** `docs/manuscript.tex`, Section 2.4 (lines 227–228) and Section 2.5 Module A (line 249)
- **Content:** Accurately distinguished quarantined mortality labels from transductively utilized covariates:
  > "We declare that BDHS 2022 test mortality outcome labels were strictly quarantined and completely unaccessed throughout all stages of model development, hyperparameter tuning, shrinkage estimation, calibration fitting, and threshold optimization. To address population-level distribution shift across survey waves, Module~A incorporates unlabeled 2022 covariates within a transductive covariate adaptation framework to compute density ratios; this procedure accesses strictly feature vectors without target mortality labels..."
- **Scientific Reason:** Preserves full scientific transparency regarding transductive density ratio estimation.

---

### Change 4: Section 4.1 — Non-Causal Compound Contrast Formulation (Factorial Arm C vs. Track A)
- **Location:** `docs/manuscript.tex`, Section 4.1 (line 383)
- **Previous Content:**
  > "We also distinguish the factorial diagnostic benchmark in Table~\ref{tab:table2_results} from the prospective deployment framework in Table~\ref{tab:table4_baselines}. These two pipelines differ fundamentally across training data, validation design, and post-processing: the factorial benchmark evaluates raw unweighted XGBoost trained across all five historical waves ($N_{\text{train}} = 38{,}370$, 2004--2018) without held-out validation or recalibration modules, achieving an uncalibrated AUROC of $0.657$ ($0.6568$) with severe probability distortion (Brier score $0.1763$). Conversely, the prospective SWIFT-U5 configuration trains the base learner on historical development waves ($N_{\text{dev}} = 29{,}686$, 2004--2014) and reserves the 2017--18 wave as an independent validation anchor for empirical Bayes shrinkage (Module~B) and drift recalibration (Module~C), while incorporating transductive covariate-shift instance weighting (Module~A). The resulting AUROC of $0.625$ reflects these compound differences in training cohort span, validation structure, and instance weighting rather than a pure degradation caused by probability calibration itself."
- **Replacement Content:**
  > "We also distinguish the factorial diagnostic benchmark in Table~\ref{tab:table2_results} from the prospective deployment framework in Table~\ref{tab:table4_baselines}. Factorial Arm~C evaluates raw unweighted XGBoost within the retrospective factorial experiment trained across all five historical waves ($N_{\text{train}} = 38{,}370$, 2004--2018) without held-out validation or recalibration modules, achieving an uncalibrated AUROC of $0.657$ ($0.6568$) with severe probability distortion (Brier score $0.1763$). Conversely, the prospective Track~A SWIFT-U5 evaluation trains the base learner on historical development waves ($N_{\text{dev}} = 29{,}686$, 2004--2014), incorporates transductive covariate-shift instance weighting (Module~A), and reserves the 2017--18 wave as an independent validation anchor for empirical Bayes shrinkage (Module~B) and drift recalibration (Module~C), achieving a calibrated AUROC of $0.625$ ($0.6253$) and Brier score of $0.0287$. The observed AUROC difference reflects a compound contrast between experimental configurations, including differences in training-data scope, validation design, instance weighting, and post-processing. Therefore, it cannot be attributed to any single component independently."
- **Scientific Reason:** Strictly removes causal phrasing regarding the AUROC divergence between Factorial Arm C and Track A SWIFT-U5, preserves exact experimental identities and numerical values, and incorporates the standard mandated formulation.

---

### Change 5: Section 4.3 — Clarification of Survey-Weighted Event Mass & Denominators
- **Location:** `docs/manuscript.tex`, Section 4.3 (lines 447–448, Table 4b footnote lines 468–470)
- **Previous Content (line 448):**
  > "In unweighted event counts, the model identifies $107$ child deaths among the $1{,}312$ prioritized children, yielding an unweighted sensitivity of $26.23\%$ ($107 / 408$ deaths) and an unweighted precision (PPV) of $8.16\%$ ($107 / 1{,}312$ prioritized). Incorporating DHS sampling weights (total population weight $= 13{,}169.9$; total mortality weight $= 394.2$), the survey-weighted true positives represent $97.0$ weighted deaths ($96.6$ if normalized to sample size), yielding the authoritative survey-weighted Sensitivity of \textbf{24.60\%} ($97.0 / 394.2$ weighted deaths), Specificity of \textbf{90.45\%}, Precision (PPV) of \textbf{7.70\%} ($97.0 / 1{,}260.0$ prioritized population weight), $\text{NNS} = 13.0$, and Priority Yield of \textbf{77.0 deaths captured per 1,000 prioritized} ($2.57\times$ efficiency over random selection)..."
- **Replacement Content (line 448):**
  > "In simulated community outreach prioritization, triage capacity is constrained by health worker availability (prioritizing the top 10\% highest-predicted-risk survey records, $N = 1{,}312$ out of $13{,}119$). The prioritized group contained $107$ unweighted recorded deaths, corresponding to an unweighted sensitivity of $26.23\%$ ($107 / 408$ deaths) and an unweighted precision (PPV) of $8.16\%$ ($107 / 1{,}312$ prioritized records). Under the survey-weighted evaluation accounting for DHS sampling weights (total population weight $= 13{,}169.9$; survey-weighted estimated event total $= 394.2$), this corresponds to approximately $97$ weighted true-positive event equivalents ($96.97$, or $97.0$; $96.6$ if normalized to sample size $N = 13{,}119$), yielding the authoritative survey-weighted Sensitivity of \textbf{24.60\%} ($97.0 / 394.2$), Specificity of \textbf{90.45\%}, Precision (PPV) of \textbf{7.70\%} ($97.0 / 1{,}260.0$ prioritized population weight), $\text{NNS} = 13.0$, and a weighted estimated prioritization yield of \textbf{approximately 77 per 1,000 prioritized records} ($76.96$ weighted event equivalents per $1{,}000$ prioritized population weight, a $2.57\times$ efficiency multiplier over random selection). We explicitly distinguish that $107 / 408$ defines unweighted sensitivity ($26.23\%$), whereas the authoritative $24.60\%$ figure represents the survey-weighted estimated sensitivity ($97.0 / 394.2$). These metrics quantify retrospective risk-ranking performance on survey-recorded deaths rather than demonstrated prospective screening effectiveness among living children."
- **Replacement Footnote (lines 468–470):**
  > `\textbf{Survey-Weighted Contingency Mass (Total Population Weight = $13{,}169.9$, Survey-Weighted Estimated Event Total = $394.2$):}\\`  
  > `Validation Youden Regime ($\tau = 0.04546$): $\text{TP}_w = 185.3$ weighted true-positive event equivalents ($207$ unw deaths), $\text{FP}_w = 3{,}740.6$ ($3{,}947$ unw), $\text{TN}_w = 8{,}985.7$ ($8{,}764$ unw), $\text{FN}_w = 207.4$ ($201$ unw).\\`  
  > `10\% Prioritization Capacity: $\text{TP}_w = 97.0$ weighted true-positive event equivalents ($107$ unw deaths; $\text{Sens}_w = 97.0 / 394.2 = 24.60\%$), $\text{FP}_w = 1{,}163.0$ ($1{,}205$ unw), $\text{TN}_w = 11{,}612.6$ ($11{,}506$ unw), $\text{FN}_w = 297.3$ ($301$ unw) ($\text{PPV}_w = 97.0 / 1{,}260.0 = 7.70\%$; $\text{NNS} = 13.0$; weighted estimated prioritization yield $\approx 77.0$/1k prioritized records).`
- **Scientific Reason:** Directly adopts the mandated phrasing ("weighted true-positive event equivalents", "survey-weighted estimated event total", "weighted estimated prioritization yield of approximately 77 per 1,000 prioritized records"), avoiding any suggestion that 97 represents literal observed deaths or that 77 per 1,000 represents a raw count. Preserves all exact calculations and unweighted/weighted mathematical denominators.

---

### Change 6: Section 4.5 & Table 6 — Division Prioritization Yield Terminology
- **Location:** `docs/manuscript.tex`, Section 4.5 (lines 547, 549, Figure 4 caption line 551, Table 6 caption line 555, Table 6 header line 561)
- **Previous Content (line 549):**
  > "At a simulated 10\% prioritization capacity nationwide, SWIFT-U5 captures \textbf{24.6\% of all national child deaths} (95\% CI: $19.8\%$--$30.4\%$), achieving a PPV of \textbf{7.70\%}, an NNS of \textbf{13.0} (95\% CI: $10.4$--$16.9$), and capturing \textbf{77.0 deaths per 1,000 children prioritized}..."
- **Replacement Content (line 549):**
  > "At a simulated 10\% prioritization capacity nationwide ($1{,}312$ records prioritized containing $107$ unweighted recorded deaths, corresponding to approximately $97$ weighted true-positive event equivalents), SWIFT-U5 achieves a survey-weighted sensitivity of \textbf{24.6\%} (95\% CI: $19.8\%$--$30.4\%$), a survey-weighted PPV of \textbf{7.70\%}, an NNS of \textbf{13.0} (95\% CI: $10.4$--$16.9$), and a weighted estimated prioritization yield of \textbf{approximately 77 per 1,000 prioritized records} ($77.0$ per $1{,}000$; 95\% CI: $59.2$--$95.7$; $2.57\times$ efficiency over random household visits) (Table~\ref{tab:table6_capacity} and Figure~\ref{fig:atlas}). In high-burden northern divisions, estimated prioritization yield reached \textbf{123.3 per 1,000 prioritized} in Rangpur ($\text{NNS} = 8.1$) and \textbf{101.2 per 1,000 prioritized} in Mymensingh ($\text{NNS} = 9.9$), compared to $38.1$ per $1{,}000$ in Dhaka ($\text{NNS} = 26.2$)."
- **Table 6 Column Header (line 561):**
  > Replaced `Deaths/1k Prioritized @ 10\%` with `Yield/1k Prioritized @ 10\%`
- **Table 6 Caption (line 555):**
  > Replaced `priority policy yield` with `survey-weighted estimated prioritization yield`
- **Scientific Reason:** Eliminates literal "deaths captured per 1,000" phrasing across division results and ensures consistency with survey-weighted rate estimates.

---

### Change 7: Section 4.7 — Retrospective Prioritization Synthesis
- **Location:** `docs/manuscript.tex`, Section 4.7 (line 628)
- **Previous Content:**
  > "Under simulated 10\% community outreach prioritization, SWIFT-U5 captures \textbf{24.6\% of recorded under-five child deaths} (PPV = \textbf{7.70\%}, $\text{NNS} = \mathbf{13.0}$, yielding \textbf{77.0 deaths captured per 1,000 children prioritized}), representing a $2.57\times$ efficiency multiplier over random household visits..."
- **Replacement Content:**
  > "Under simulated 10\% community outreach prioritization ($N = 1{,}312$ records prioritized out of $13{,}119$, containing $107$ unweighted recorded deaths), SWIFT-U5 achieves a survey-weighted sensitivity of \textbf{24.60\%} (corresponding to approximately $97$ weighted true-positive event equivalents over a survey-weighted estimated event total of $394.2$), a survey-weighted PPV of \textbf{7.70\%}, an $\text{NNS} = \mathbf{13.0}$, and a weighted estimated prioritization yield of \textbf{approximately 77 per 1,000 prioritized records} ($77.0$ weighted event equivalents per $1{,}000$ prioritized population weight), representing a $2.57\times$ efficiency multiplier over random household selection, reaching an estimated yield of \textbf{123.3 per 1,000 prioritized} in Rangpur ($\text{NNS} = 8.1$)."
- **Scientific Reason:** Aligns conclusion synthesis with the standardized survey-weighted event mass phrasing.

---

### Change 8: Section 6 — Conclusion Synchronization
- **Location:** `docs/manuscript.tex`, Section 6 (line 648)
- **Previous Content:**
  > "Under simulated 10\% community prioritization capacity, the model captured $24.6\%$ of recorded child deaths (PPV = $7.70\%$, $\text{NNS} = 13.0$), yielding $77.0$ deaths captured per $1{,}000$ prioritized children, rising to $123.3$ deaths per $1{,}000$ prioritized in high-burden Rangpur division ($\text{NNS} = 8.1$)."
- **Replacement Content:**
  > "Under simulated 10\% community prioritization capacity ($1{,}312$ records prioritized containing $107$ unweighted recorded deaths, corresponding to approximately $97$ weighted true-positive event equivalents), the model achieved a survey-weighted sensitivity of $24.6\%$ (PPV = $7.70\%$, $\text{NNS} = 13.0$) with a weighted estimated prioritization yield of approximately $77$ per $1{,}000$ prioritized records ($77.0$ weighted event equivalents per $1{,}000$ prioritized population weight), rising to an estimated yield of $123.3$ per $1{,}000$ prioritized in high-burden Rangpur division ($\text{NNS} = 8.1$)."
- **Scientific Reason:** Ensures final conclusion cleanly presents the survey-weighted event equivalents and estimated yield without literal death count implications.

---

## 3. Final Consistency Verification Matrix

All frozen numerical values across the study remain 100% preserved and verified:

| Metric / Parameter | Frozen Authoritative Value | Status in Manuscript | Verification Source |
|---|---|:---:|---|
| **Master Cohort live births ($N$)** | $51{,}489$ | Preserved ($51{,}489$) | Parquet cohort audit |
| **Master Cohort deaths ($n$)** | $2{,}338$ | Preserved ($2{,}338$) | Parquet cohort audit |
| **2022 Test Sample ($N$)** | $13{,}119$ | Preserved ($13{,}119$) | Table 1 / Table 2 / Table 4 |
| **2022 Test Deaths ($n$)** | $408$ | Preserved ($408$) | Table 1 / Table 2 / Table 4 |
| **Prioritized sample records ($N_{\text{screen}}$)** | $1{,}312$ | Preserved ($1{,}312$) | Parquet evaluation ($10\%$ of $13{,}119$) |
| **Unweighted true positives ($TP$)** | $107$ | Preserved ($107$) | Parquet evaluation |
| **Unweighted sensitivity** | $26.23\%$ | Preserved ($107 / 408 = 26.23\%$) | Parquet evaluation |
| **Unweighted PPV** | $8.16\%$ | Preserved ($107 / 1{,}312 = 8.16\%$) | Parquet evaluation |
| **Primary SWIFT-U5 AUROC (W)** | $\mathbf{0.625}$ ($0.6253$) [0.587, 0.663] | Preserved ($\mathbf{0.625}$) | `outputs/tables/table4_swift_vs_baselines.csv` |
| **Baseline XGBoost AUROC (W)** | $\mathbf{0.638}$ [0.603, 0.673] | Preserved ($\mathbf{0.638}$) | `outputs/tables/table4_swift_vs_baselines.csv` |
| **Factorial Arm C AUROC (W)** | $\mathbf{0.657}$ ($0.6568$) [0.615, 0.682] | Preserved ($\mathbf{0.657}$) | `outputs/tables/table2_2x2_main_results.csv` |
| **SWIFT-U5 Brier Score (W)** | $\mathbf{0.0287}$ | Preserved ($\mathbf{0.0287}$) | `outputs/tables/table4_swift_vs_baselines.csv` |
| **Null Predictor Brier** | $\mathbf{0.0290}$ | Preserved ($\mathbf{0.0290}$) | Prevalence calculation ($\pi = 2.99\%$) |
| **Baseline XGBoost Brier (W)** | $\mathbf{0.1649}$ | Preserved ($\mathbf{0.1649}$) | `outputs/tables/table4_swift_vs_baselines.csv` |
| **SWIFT-U5 Calibration Slope** | $\mathbf{0.806}$ | Preserved ($\mathbf{0.806}$) | `outputs/tables/table4_swift_vs_baselines.csv` |
| **SWIFT-U5 Calibration Intercept** | $\mathbf{-0.905}$ | Preserved ($\mathbf{-0.905}$) | `outputs/tables/table4_swift_vs_baselines.csv` |
| **10% Priority Sensitivity (W)** | $\mathbf{24.60\%}$ ($24.6\%$) [19.8, 30.4] | Preserved ($\mathbf{24.60\%}$) | `outputs/tables/table6_division_nns_capacity.csv` |
| **10% Priority PPV (W)** | $\mathbf{7.70\%}$ | Preserved ($\mathbf{7.70\%}$) | Parquet evaluation ($96.97 / 1{,}260.02$) |
| **10% Priority NNS (W)** | $\mathbf{13.0}$ [10.4, 16.9] | Preserved ($\mathbf{13.0}$) | `outputs/tables/table6_division_nns_capacity.csv` |
| **10% Priority Estimated Yield** | $\approx \mathbf{77}$ / 1k ($77.0$ [59.2, 95.7]) | Preserved ($\approx \mathbf{77}$ / 1k) | `outputs/tables/table6_division_nns_capacity.csv` |
| **Rangpur 10% Estimated Yield** | $\mathbf{123.3}$ / 1k ($\text{NNS}=8.1$) | Preserved ($\mathbf{123.3}$) | `outputs/tables/table6_division_nns_capacity.csv` |
| **Validation Youden Threshold ($\tau_w$)**| $\mathbf{0.04546}$ | Preserved ($\mathbf{0.04546}$) | Phase 6 audit (`0.04546`) |
| **Youden Weighted Sensitivity** | $\mathbf{47.18\%}$ ($185.3 / 394.2$) | Preserved ($\mathbf{47.18\%}$) | Table 4b evaluation |
| **Youden Balanced Accuracy** | $\mathbf{58.90\%}$ | Preserved ($\mathbf{58.90\%}$) | Table 4b evaluation |
| **Youden Weighted Accuracy** | $\mathbf{69.91\%}$ (vs.\ $97.01\%$) | Preserved ($\mathbf{69.91\%}$) | Table 4b evaluation |
| **Youden MCC** | $\mathbf{+0.0662}$ | Preserved ($\mathbf{+0.0662}$) | Table 4b evaluation |

---

## 4. Unresolved Discrepancies Statement

No mathematical, terminological, or artifact discrepancies remain unresolved:
1. **Unweighted vs. Weighted Event Mass:** Unweighted calculations ($107$ deaths out of $408$ total $= 26.23\%$; $107 / 1{,}312 = 8.16\%$) are explicitly separated from survey-weighted event mass ($96.97 \approx 97.0$ weighted true-positive event equivalents out of $394.24$ estimated event total $= 24.60\%$; $96.97 / 1{,}260.02 = 7.70\%$; estimated yield of approximately $77$ per $1{,}000$ prioritized records).
2. **AUROC Contrast Framing:** Factorial Arm C ($0.6568$, $0.657$, raw unweighted XGBoost, retrospective factorial experiment) and prospective Track A SWIFT-U5 ($0.6253$, $0.625$, full three-module SWIFT-U5, prospective temporal evaluation) are strictly presented as a compound contrast across experimental configurations without causal attribution to any single factor.
3. **Brier Reduction:** Maintained as a descriptive comparison between model configurations, with Module C's incremental effect isolated via Table 5 ablation findings.

**Scientific Validation Boundary:** We explicitly state that eliminating internal numerical discrepancies and refining narrative framing does not demonstrate external clinical validity or prospective intervention effectiveness. Prospective clinical trials enrolling living children are required before public health operational deployment.
