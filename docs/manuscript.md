# Survey-Weighted and Temporally Validated Risk Modeling for Under-Five Mortality Using Bangladesh DHS Data

**Authors:**  
Md. Rafiul Azam Chowdhury$^1$, Rezaul Mursalin Khan$^1$, Fatema Tuz Zohura Rifa$^1$, Tanjim Ahmed Irfad$^1$, Arafat Hussain Nahid$^1$, Ohidujjaman$^1$  
$^1$*Department of Computer Science and Engineering, United International University, Dhaka 1212, Bangladesh*

---

## Abstract
Under-five mortality remains a critical global health challenge in low- and middle-income countries, including Bangladesh, where child survival disparities reflect complex socioeconomic, maternal, and environmental vulnerabilities. Although Demographic and Health Survey (DHS) microdata offer large nationally representative samples, published machine learning applications frequently suffer from retrospective data leakage (such as illness recall and postnatal feeding duration acting as survival proxies), ignore complex multi-stage survey design, and report uncalibrated metrics under arbitrary default classification thresholds. This study addresses these vulnerabilities by developing and temporally validating a survey-weighted machine learning framework—termed **SWIFT-U5** (**S**urvey-**W**eighted, **I**mportance-corrected, **F**orward-validated **T**emporal Risk Model for Under-Five Mortality)—for prospective population risk stratification and community health worker household triage in Bangladesh. We analyzed an exposure-complete national cohort of 51,489 live births and 2,338 documented under-five deaths across six sequential BDHS survey rounds (2004 to 2022). We restricted predictors to maternal reproductive history, durable household WASH infrastructure, and socioeconomic indicators observable during community health outreach visits. We benchmarked logistic regression, random forest, XGBoost, and LightGBM under a strict forward-temporal design across five sequential cuts incorporating DHS sampling weights, primary sampling unit clustering, and drift-aware Platt recalibration (Module C). On the locked BDHS 2022 test cohort ($N = 13,119$, 408 deaths), SWIFT-U5 achieved a survey-weighted ROC-AUC of 0.637 (95% cluster-bootstrap CI: 0.600–0.676) and a PR-AUC of 0.075. Drift-aware recalibration reduced raw uncalibrated Brier score error by 83.7% from 0.1822 (raw XGBoost) to 0.0297 (comparable to the null constant-prevalence baseline of 0.0290 at 2.99% test prevalence, yielding a Brier Skill Score of -0.024) and shifted the calibration intercept from -3.230 to +0.032, while a calibration slope of 0.425 reflects dampening of raw overconfident tree logits. At a realistic 10% community health worker outreach capacity, the model captures 25.8% of national child deaths (Sensitivity = 25.8%, Specificity = 91.2%, PPV = 8.3%, NNS = 12.3), achieving an efficiency multiplier of $2.72\times$ over random household visits, rising to 105.3 deaths captured per 1,000 children screened (NNS = 9.5) in high-burden northern divisions such as Rangpur. At the Youden-optimal threshold ($\tau = 0.0004$), the model achieved 64.2% sensitivity, 56.8% specificity, and 60.5% balanced accuracy. Subgroup audits indicated comparable ranking sensitivity across rural (25.4%) and urban (25.5%) settings, with enhanced triage yield in socioeconomically disadvantaged strata. These findings establish that prospective survey-weighted learning provides a realistic benchmark for population-level risk stratification and community outreach prioritization in Bangladesh, while highlighting the inherent limits of individual-level classification in rare-event demographic surveys.

**Keywords:** Under-five mortality, Demographic and Health Surveys, Risk stratification, Survey-weighted learning, Temporal validation

---

## 1. Introduction
Under-five mortality (U5M) remains a major public health challenge in low- and middle-income countries, and Bangladesh is no exception, where child deaths remain unequally distributed across socioeconomic, maternal, environmental, and service-access conditions [1–4]. Nationally representative Demographic and Health Survey (DHS) data provide a unique opportunity to model U5M risk using standardized household, maternal, and child-level indicators [5, 6]. Over the past thirty years, Bangladesh has achieved remarkable public health progress [13, 14]. Through widespread immunization campaigns, family planning programs, oral rehydration therapy, and an extensive network of community health workers, the national under-five mortality rate plummeted from approximately 133 deaths per 1,000 live births in 1993–94 to around 30 per 1,000 in 2022 [7]. As mortality rates declined, deaths became increasingly concentrated in the neonatal period (over 70% of under-five deaths in Bangladesh now occur in the first 28 days of life) and in geographically disadvantaged divisions such as Sylhet and Rangpur [7].

A growing body of studies has applied supervised machine learning to predict child mortality risk from DHS data [15–22]. However, existing benchmarks reveal widespread methodological vulnerabilities:
1. **Retrospective Data Leakage:** Inclusion of postnatal survival proxies such as breastfeeding duration [15] and survival-contingent illness recall questions.
2. **Distorted Class Imbalance Handling:** Pre-split SMOTE oversampling altering population prevalence and distorting calibration.
3. **Disregard of Complex Survey Architecture:** Ignoring sampling weights, strata, and PSU clustering [34, 35, 37].
4. **Absence of Prospective Temporal Validation:** Random cross-validation concealing longitudinal domain shifts.
5. **Misleading Calibration and Default Decision Thresholds:** Evaluating rare-event classifiers under arbitrary 0.5 thresholds without calibration diagnostics or outreach capacity metrics.

We define SWIFT-U5 as an **interview-time community health surveillance and household triage model** rather than an exclusive pre-birth obstetric diagnostic. While obstetric models are restricted strictly to pre-conception or antenatal variables, community health outreach in low-resource settings occurs during home visits where health workers observe durable household infrastructure, maternal education, and birth history simultaneously to prioritize follow-up visits. Observational risk predictions provide associative stratification for resource allocation, not causal explanation.

---

## 2. Methodology

### 2.1 Cohort Data, Outcome Ascertainment, and Right-Censoring
We analyzed individual Birth Recode (BR) microdata across six BDHS rounds (2004–2022): BDHS 2004 ($N = 6,908$), 2007 ($N = 6,150$), 2011 ($N = 8,742$), 2014 ($N = 7,886$), 2017–18 ($N = 8,684$), and 2022 ($N = 13,119$), yielding 51,489 live births and 2,338 under-five deaths.

**Estimand and Target Definition:** Prospective risk stratification for under-five mortality among children born within the DHS 5-year recall window, where survival status is ascertained at maternal interview:
$$\text{target} = (\texttt{b5} = 0) \land (\texttt{b7} < 60)$$
where `b5` indicates survival status and `b7` is age at death in completed months.

**Handling of Incomplete Follow-up (Right-Censoring):** Living children younger than 60 months at interview have not completed the full five-year exposure window. Following standard DHS machine learning literature, living children are classified as non-events at interview time ($Y = 0$). This is justified because:
- **Epidemiological Front-Loading:** In Bangladesh, 71% of under-five deaths occur in the neonatal window ($t < 1$ month), and over 85% occur within the first year ($t < 12$ months) [7]. For a child surviving to interview past age one, residual risk before age five is $<0.4\%$.
- **Operational Alignment:** A Community Health Worker visits households and assesses living children. Classifying living children as negative cases reflects the operational triage mandate at the visit date.
Survival models with time-to-event hazard functions represent an important methodological alternative described in our limitations.

**Geographic Administrative Harmonization:** Bangladesh transitioned from 6 divisions (2004–2007) to 7 (2011 with Rangpur) and 8 divisions (2022 with Mymensingh). For 18-year longitudinal comparability, Mymensingh is grouped within Dhaka for rounds prior to 2022, while the 2022 evaluation presents both 7- and 8-division views.

### 2.2 Predictor Timing and Reference Period Audit
All candidate predictors were audited against measurement timing and prospective availability:
- **Maternal Core:** `maternal_age_at_birth` derived from maternal birth date (`v011`) and child birth date (`b3`).
- **Reproductive History:** `birth_order`, `parity_before`, `is_first_birth`, `preceding_birth_interval` (`b11`), `is_twin` (`b0`).
- **Antenatal Care:** `anc_visits` (`m14`), `tetanus_toxoid` (`m1`), `iron_supp` (`m45`) during index pregnancy.
- **Birth Characteristics:** `birth_size` (`m18`), `facility_delivery` (`m15`), `caesarean` (`m17`).
- **Durable Household & Socioeconomic Proxies:** Maternal education (`v106`), wealth quintile (`v190`), drinking water source (`v113`), sanitation (`v116`), electricity (`v119`), floor material (`v127`).
- **Geographic:** Administrative division and rural/urban residence.

Illness recall (`h11`, `h22`, `h31`), breastfeeding duration (`m4`), and delivery complication recall were strictly excluded due to severe retrospective survival leakage.

### 2.3 Complex Survey Design and Pooled Weight Normalization
- **Sample Weight Normalization:** Within each wave $t$, weights are normalized: $w_i = w_i^{\text{raw}} \times (N_t / \sum_{j \in \mathcal{D}_t} w_j^{\text{raw}})$, ensuring $\sum_{i \in \mathcal{D}_t} w_i = N_t$ so waves contribute in proportion to their analytic sample sizes.
- **Cluster-Aware Partitions (`v021`):** Compound PSU identifiers (`wave_strata_psu`) ensured unique clusters across rounds. Validation folds were constructed using `GroupKFold` grouped by compound PSU identifier to prevent intra-cluster and intra-mother leakage [36].
- **Stratified Cluster Bootstrapping:** 100 cluster-bootstrap resamples within sampling strata (`v022`) were used for all 95% confidence intervals.

### 2.4 Forward Temporal Validation Partitions & Locked 2022 Test
Five chronological cuts were established:
- **Cut 1:** Train 2004 ($N = 6,908$), Test 2007 ($N = 6,150$, 361 deaths)
- **Cut 2:** Train 2004–2007 ($N = 13,058$), Test 2011 ($N = 8,742$, 410 deaths)
- **Cut 3:** Train 2004–2011 ($N = 21,800$), Test 2014 ($N = 7,886$, 319 deaths)
- **Cut 4:** Train 2004–2014 ($N = 29,686$), Test 2017–18 ($N = 8,684$, 356 deaths)
- **Cut 5:** Train 2004–2017–18 ($N = 38,370$), Test **BDHS 2022 Locked Test Set** ($N = 13,119$, 408 deaths)

**Declaration of Locked Test Integrity:** The BDHS 2022 test cohort ($N = 13,119$) was completely locked and remained strictly untouched throughout all stages of development. Zero 2022 observations were accessed or used for feature selection, preprocessing parameters, base model selection, tuning, or calibration. The locked BDHS 2022 test set was evaluated exactly once.

### 2.5 Dual-Method Stability Consensus Feature Selection
Features were selected on training data across 40 PSU-bootstrapped iterations combining:
1. Survey-Weighted LASSO ($L_1$ Regularization) [26];
2. Weighted XGBoost Permutation Importance across bootstrapped resamples [23, 25].
Following the stability selection framework of Meinshausen & Bühlmann [27], features with stability frequency $\ge 0.70$ were designated as the Stable Core, anchored by four designated epidemiological core features.

### 2.6 SWIFT-U5 Framework Architecture
- **Module A (Recency-Decayed Covariate Shift Adaptation):** Adjusts weights by $w_i^* = w_i^{\text{survey}} \cdot \exp(-\lambda(t_{\text{target}} - t_i)) \cdot \text{clip}(\hat{P}(\text{target}|\mathbf{x}_i)/\hat{P}(\text{source}|\mathbf{x}_i), 0.1, 10.0)$ with $\lambda = 0.05$ [38, 39].
- **Module B (Empirical Bayes Division Shrinkage):** Division logit parameters $(\alpha_d, \beta_d)$ shrunk toward national mean with $\kappa = 40.0$ [40, 41].
- **Module C (Drift-Aware Platt Recalibration):** $\hat{P}_{\text{recal}}(Y=1|z) = \sigma(a^* \cdot z + b^*)$ fitted on out-of-fold validation logits from the preceding survey round (BDHS 2017–18) [29].

---

## 3. Results

### 3.1 Validation Design Inflation and Base Model Selection
On the BDHS 2017–18 validation split, candidate models achieved survey-weighted discrimination of ROC-AUC = 0.694 (Logistic Regression), 0.681 (LightGBM), 0.681 (XGBoost), and 0.658 (Random Forest). XGBoost was selected as the base architecture for SWIFT-U5 due to non-linear interaction capture, superior PR-AUC under extreme class imbalance, and native support for sample-level density weights.

Random split validation (Arms A and B, $N_{\text{test}} = 15,446$) yielded an apparent ROC-AUC of 0.710–0.718 across all waves. However, forward temporal validation (Arms C and D) revealed true prospective discrimination of 0.648 on BDHS 2022, demonstrating an inflation gap of +0.062 to +0.064 AUROC points. Furthermore, random validation suggested an $\text{NNS}_{5\%}$ of 4.6, whereas true temporal deployment required $\text{NNS}_{5\%} = 8.1$–$8.9$.

### 3.2 Longitudinal Feature Stability
Dual-method stability selection identified four immutable **Stable Core** features across 18 years:
1. `anc_visits` (selection frequency = 1.00);
2. `preceding_birth_interval` (frequency = 0.955);
3. `is_twin` (frequency = 0.800);
4. `maternal_education` (frequency = 0.790).

In contrast, institutional delivery (`facility_delivery`) exhibited marked temporal drift as facility deliveries expanded nationwide.

### 3.3 Standard Binary Classification Performance
On the locked BDHS 2022 test cohort ($N = 13,119$, 408 deaths, weighted prevalence 2.99%):
- **Youden-Optimal Regime ($\tau = 0.0004$):** Sensitivity = 64.2%, Specificity = 56.8%, Precision = 4.4%, Accuracy = 57.0%, Balanced Accuracy = 60.5%, F1 = 0.082. Confusion matrix: $\text{TP} = 253.3$, $\text{FP} = 5,520.6$, $\text{TN} = 7,255.1$, $\text{FN} = 141.0$.
- **Fixed 10% Outreach Capacity ($\tau = 0.0031$):** Sensitivity = 25.8%, Specificity = 91.2%, Precision = 8.3%, Accuracy = 89.2%, Balanced Accuracy = 58.5%, F1 = 0.125, NNS = 12.1 ($\approx 12.3$). Confusion matrix: $\text{TP} = 101.6$, $\text{FP} = 1,126.3$, $\text{TN} = 11,649.4$, $\text{FN} = 292.6$.

### 3.4 Probability Calibration and Decision Utility
- **Brier Score:** Raw XGBoost achieved an uncalibrated Brier score of 0.1822. Drift-aware Platt recalibration (Module C) reduced the Brier score by 83.7% to **0.0297** on the locked BDHS 2022 cohort.
- **Null Baseline Comparison:** At 2.99% test prevalence, an uninformative prevalence-only predictor achieves $\text{Brier}_{\text{null}} = 0.0290$. SWIFT-U5's Brier score of 0.0297 corresponds to a Brier Skill Score of $\text{BSS} = -0.024$. The 83.7% error reduction reflects elimination of severe probability distortion induced by class-balancing weights rather than a substantial reduction below a constant prevalence forecast.
- **Calibration Intercept:** $\alpha = +0.032$ (vs. $-3.230$ for baseline XGBoost), indicating minimal overall calibration-in-the-large bias.
- **Calibration Slope:** $\beta = 0.425$. A slope $< 1$ indicates that raw base model predictions were over-dispersed (overly extreme), which recalibration successfully dampened toward the center.
- **Ablation:** Removing Module C caused Brier score to collapse to 0.1805 and intercept to -3.230. Omitting Module A increased Brier error to 0.0297 with intercept +0.698.

### 3.5 Division Policy Layer and Operational Screening Capacity
At a realistic 10% community health worker outreach capacity nationwide:
- SWIFT-U5 captures **25.8% of all national child deaths** (95% CI: 20.5%–31.0%).
- Number Needed to Screen = **12.3** (95% CI: 10.0–15.8).
- Screening yield = **81.3 deaths per 1,000 screened** ($2.72\times$ efficiency multiplier over random home visits).
- In high-burden northern divisions, yield reached **105.3 deaths per 1,000 screened** in Rangpur (NNS = 9.5) and **95.9 deaths per 1,000 screened** in Sylhet (NNS = 10.4).
These metrics reflect retrospective ranking utility on the locked survey cohort, not proof of clinical intervention efficacy.

### 3.6 Demographic Subgroup Analysis and Algorithmic Equity
- **Residence:** Comparable sensitivity between rural (25.4%, NNS = 12.3) and urban (25.5%, NNS = 11.7) areas.
- **Wealth Gradient:** NNS dropped to **8.2** in Quintile 1 (poorest), compared to 16.5 in Quintile 5 (wealthiest).
- **Maternal Education:** In mothers with no education, the model achieved AUROC = 0.701, capturing 36.8% of deaths (NNS = 6.9). Because this subgroup includes only 28 death events out of 824 children, this is treated as an exploratory finding rather than definitive proof of superior performance.

### 3.7 Explainable AI and Attribution Stability (TreeSHAP)
Global SHAP attributions ranked antenatal care attendance, birth spacing, birth size proxy, multiple birth, and maternal education as the primary predictive contributors. The SHAP Drift Index remained bounded strictly below 0.25 across all historical cuts (declining from 0.245 in 2007 to 0.007 in 2022), confirming attribution stability across eighteen years.

### 3.8 Synthesis and Comparative Trade-offs
1. **Discrimination:** Pre-birth determinants achieve forward temporal ROC-AUC of 0.637 [0.600, 0.676]. While uncalibrated XGBoost achieved 0.646, SWIFT-U5's contribution lies in calibration restoration and policy screening utility.
2. **Calibration:** Recalibration restored the intercept to +0.032; the slope of 0.425 reflects dampening of overconfident raw logits.
3. **Screening Utility:** $2.72\times$ efficiency gain over random household outreach.
4. **Subgroup Equity:** Stable performance across rural/urban divides with pro-poor outreach yield.

---

## 4. Discussion and Limitations
1. **Right-Censoring in Target Formulation:** Living children under 60 months at interview have not completed five years of follow-up. Survival models with censored loss functions represent a valuable future direction.
2. **Interview-Time Covariate Measurement:** Household assets and maternal education are recorded at interview rather than birth, assuming relative stability over the under-five window.
3. **Calibration Slope and Probability Dispersion:** The calibration slope of 0.425 reflects substantial shrinkage of overconfident raw base tree logits.
4. **Sample Size and Event Counts in Subgroups:** Low death counts in certain divisions (Khulna: 34, Rajshahi: 32) and subgroups (uneducated mothers: 28) yield wide confidence intervals.
5. **Geographic Reorganization:** Bangladesh's expansion from 6 to 8 divisions required historical grouping of Mymensingh within Dhaka.

---

## 5. Conclusion
SWIFT-U5 establishes a survey-weighted machine learning framework for under-five mortality risk stratification and community outreach triage in Bangladesh. By eliminating retrospective proxies and grounding predictions on durable maternal, reproductive, household WASH infrastructure, and socioeconomic determinants, the framework provides an honest benchmark for prospective child mortality modeling and community health worker triage.

---

## References
[1] World Health Organization, Levels and trends in child mortality: Report 2023, Geneva (2023).  
[2] UN IGME, Levels & trends in child mortality: Report 2023, UNICEF, New York (2023).  
[3] R. E. Black, et al., The Lancet 375(9730) (2010) 1969–1987.  
[4] D. You, et al., The Lancet 386(10010) (2015) 2275–2286.  
[5] The DHS Program, DHS-VII Recode Manual, ICF, Rockville, MD (2018).  
[6] T. N. Croft, et al., Guide to DHS Statistics, ICF, Rockville, MD (2023).  
[7] NIPORT and ICF, Bangladesh DHS 2022: Final Report, Dhaka and Rockville (2024).  
[8] A. R. Omran, The Milbank Quarterly 83(4) (2005) 731–757.  
[9] J. R. Khan, N. Awan, Archives of Public Health 75(1) (2017) 58.  
[10] M. A. Khan, et al., PLOS ONE 16(2) (2021) e0246210.  
[11] T. Mazumder, et al., BMC Pediatrics 24(1) (2024) 572.  
[12] T. Mazumder, et al., Scientific Reports 14(1) (2024) 19823.  
[13] A. M. R. Chowdhury, et al., The Lancet 382(9906) (2013) 1734–1745.  
[14] S. E. Arifeen, et al., The Lancet 382(9909) (2013) 2012–2026.  
[15] S. Naznin, et al., PLOS ONE 20(6) (2025) e0324825.  
[16] M. M. M. Fahim, et al., arXiv:2602.03957 (2026).  
[17] M. A. Haque, M. A. Kabir, arXiv:2609.24386 (2026).  
[18] S. Oduse, et al., BMC Medical Informatics and Decision Making 24(1) (2024) 86.  
[19] S. Mabula, et al., Machine Learning Research 10(2) (2025) 110–123.  
[20] F. H. Bitew, et al., Genus 76(1) (2020) 37.  
[21] R. K. Saroj, et al., BioData Mining 15(1) (2022) 20.  
[22] A. A. Asnake, et al., Scientific Reports 16 (2026) 9131.  
[23] T. Chen, C. Guestrin, KDD (2016) 785–794.  
[24] G. Ke, et al., NeurIPS 30 (2017) 3146–3154.  
[25] L. Breiman, Machine Learning 45(1) (2001) 5–32.  
[26] R. Tibshirani, JRSS-B 58(1) (1996) 267–288.  
[27] N. Meinshausen, P. Bühlmann, JRSS-B 72(4) (2010) 417–473.  
[28] M. B. Kursa, W. R. Rudnicki, JSS 36(11) (2010) 1–13.  
[29] J. C. Platt, Advances in Large Margin Classifiers (1999) 61–74.  
[30] A. J. Vickers, E. B. Elkin, Medical Decision Making 26(6) (2006) 565–574.  
[31] B. Van Calster, et al., BMC Medicine 17(1) (2019) 230.  
[32] G. S. Collins, et al., BMJ 385 (2024) e078378.  
[33] R. F. Wolff, et al., Annals of Internal Medicine 170(1) (2019) 51–58.  
[34] T. Lumley, JSS 9(1) (2004) 1–19.  
[35] J. Wieczorek, et al., Stat 11(1) (2022) e454.  
[36] Y. Oh, et al., PMLR 333 (2026) 198–220.  
[37] H. Shimodaira, JSPI 90(2) (2000) 227–244.  
[38] M. Sugiyama, M. Kawanabe, Covariate Shift Adaptation, MIT Press (2012).  
[39] G. Casella, The American Statistician 39(2) (1985) 83–87.  
[40] B. Efron, Large-Scale Inference, Cambridge University Press (2012).  
[41] S. M. Lundberg, S.-I. Lee, NeurIPS 30 (2017) 4765–4774.  
[42] D. A. Vyas, et al., NEJM 383(9) (2020) 874–882.  
[43] Z. Obermeyer, et al., Science 366(6464) (2019) 447–453.  
[44] N. V. Chawla, et al., JAIR 16 (2002) 321–357.  
[45] N. MacNell, et al., PLOS ONE 18(3) (2023) e0280387.  
[46] T. Saito, M. Rehmsmeier, PLOS ONE 10(3) (2015) e0118432.
