# SWIFT-U5: Phase 6 Accounting & Provenance Correction Log
**Document ID:** `SWIFT-U5-PHASE6-ACCOUNTING-LOG`  
**Date:** November 9, 2026  
**Status:** COMPLETE & VERIFIED  
**Auditor / Agent:** SWIFT-U5 Verification Suite & Antigravity Agent  

---

## 1. Overview of Accounting Reconciliation

During final verification of `phase6_results_freeze.md` prior to results freeze, a clerical discrepancy was identified in the 2004 survey row:
- The previous draft table contained an outdated exploratory draft row: `2004: N=5,617, deaths=390`, which summed to $N=50,198$ and $2,244$ deaths, conflicting with the verified cohort total of $N=51,489$ and $2,338$ deaths.
- The authoritative harmonized cohort artifact (`data/processed/u5m_harmonized_cohort.parquet`) and the authoritative Table 1 artifact (`docs/tables/table1_cohort_summary.csv`) were audited directly.
- The true underlying data values are: **2004 $N = 6,908$ live births, $484$ deaths** (unweighted prevalence $7.01\%$, survey-weighted U5MR $71.9$ per 1,000).

---

## 2. Inspected Authoritative Source Artifacts

1. **`data/processed/u5m_harmonized_cohort.parquet` ($51,489$ records, $2,338$ recorded deaths):**
   - Verified wave-by-wave group counts of all rows, deaths (`target_u5m == 1`), sample weights (`sample_weight`), and resulting unweighted and design-weighted prevalence rates.
2. **`docs/tables/table1_cohort_summary.csv` & `docs/tables/table1_cohort_summary.tex`:**
   - Authoritative Table 1 baseline summary generated in Phase 5 / Phase 6. Line 2 confirms: `2004, National (Total), 6908, 484, 70.1, 7001.7, 503.6, 71.9`.
3. **`outputs/predictions/authoritative_swift_u5_2022_predictions.parquet` ($13,119$ test records, $408$ deaths):**
   - Verified exact test set size ($N=13,119$), raw deaths ($408$), and design-weighted deaths ($392.74$).
4. **`docs/manuscript.tex` (Line 316):**
   - Verified that the LaTeX manuscript already contained the authoritative 2004 national total: `\textbf{BDHS 2004} & National (Total) & 6{,}908 & 484 & 71.9 \\`.

---

## 3. Confirmed Authoritative Cohort Accounting Numbers

### Wave-Level Accounting (BDHS 2004–2022)

| Survey Wave | Harmonized Sample ($N$) | Recorded Deaths ($n$) | Unweighted Prevalence (%) | Weighted Sample ($N_w$) | Weighted Deaths ($n_w$) | Survey-Weighted U5MR (‰) | Partition Role in Primary Prospective Benchmark |
|---|---|---|---|---|---|---|---|
| **BDHS 2004** | 6,908 | 484 | 7.01% | 7,001.7 | 503.6 | 71.9 ‰ | Historical Development ($df\_train$) |
| **BDHS 2007** | 6,150 | 361 | 5.87% | 6,057.6 | 338.7 | 55.9 ‰ | Historical Development ($df\_train$) |
| **BDHS 2011** | 8,742 | 410 | 4.69% | 8,788.9 | 394.0 | 44.8 ‰ | Historical Development ($df\_train$) |
| **BDHS 2014** | 7,886 | 319 | 4.05% | 8,091.8 | 331.4 | 41.0 ‰ | Historical Development ($df\_train$) |
| **BDHS 2017-18**| 8,684 | 356 | 4.10% | 8,708.5 | 350.7 | 40.3 ‰ | Forward Validation & Calibration ($df\_val$) |
| **BDHS 2022** | 13,119 | 408 | 3.11% | 13,169.9 | 394.2 | 29.9 ‰ | Locked Prospective Out-of-Time Test ($df\_test$) |
| **Total Master Cohort** | **51,489** | **2,338** | **4.54%** | **51,818.4** | **2,312.6** | **41.2 ‰** | **Mutually Exclusive & Completely Exhaustive** |

### Benchmark Partition Totals (Track A Prospective Pipeline)

- **Development Partition ($df\_train$, 2004–2014):**
  $$N_{\text{dev}} = 6,908 + 6,150 + 8,742 + 7,886 = \mathbf{29{,}686}$$
  $$n_{\text{dev, deaths}} = 484 + 361 + 410 + 319 = \mathbf{1{,}574}$$
- **Validation Partition ($df\_val$, 2017-18):**
  $$N_{\text{val}} = \mathbf{8{,}684}, \quad n_{\text{val, deaths}} = \mathbf{356}$$
- **Prospective Test Partition ($df\_test$, 2022):**
  $$N_{\text{test}} = \mathbf{13{,}119}, \quad n_{\text{test, deaths}} = \mathbf{408}$$
- **Reconciliation Check:**
  $$N_{\text{total}} = 29,686 + 8,684 + 13,119 = \mathbf{51{,}489} \quad (\Delta = 0)$$
  $$n_{\text{deaths, total}} = 1,574 + 356 + 408 = \mathbf{2{,}338} \quad (\Delta = 0)$$

---

## 4. Log of Every Corrected Occurrence

| # | File Modified | Location | Prior Inconsistent Entry | Authoritative Corrected Entry | Reason for Correction |
|---|---|---|---|---|---|
| **1** | `docs/phase6_results_freeze.md` | Section 1 (Table row 1) | `BDHS 2004 \| 2004 \| 5,617 \| 390 \| 6.94% \| 65.2 ‰` | `BDHS 2004 \| 2004 \| 6,908 \| 484 \| 7.01% \| 71.9 ‰` | Corrected clerical typo carrying pre-cleaned exploratory numbers. |
| **2** | `docs/phase6_results_freeze.md` | Section 2 (Partition counts) | `Development Partition: N=29,686, deaths=1,480` | `Development Partition: N=29,686, deaths=1,574` | Updated death count to reflect true 2004 deaths ($484 + 361 + 410 + 319 = 1,574$). |
| **3** | `docs/phase6_results_freeze.md` | Section 4.2 | Missing comparative section | Added complete Section 4.2 detailing Factorial Arm C (AUROC 0.6568) vs. Track A SWIFT-U5 (AUROC 0.6253). | Disclosed underlying model configuration, training sets, and loss objectives. |
| **4** | `docs/phase6_results_freeze.md` | Section 6 (Claims policy) | Contained uncontextualized calibration and continuous DCA utility phrasing | Replaced "significantly improves calibration" with descriptive Brier reduction; framed DCA as decision-analytic net benefit on discrete 0.5% grid. | Enforced scientific claims integrity. |
| **5** | `docs/phase6_final_reconciliation_report.md` | Section 2.1 (Table Cut 1 & 2) | `2004 ($N=5,617$)` | `2004 ($N=6,908$, deaths=484)` | Reconciled training partition sizes across all cuts. |
| **6** | `docs/phase6_final_reconciliation_report.md` | Section 2.1 (Table Cut 5) | `2004-2014 ($N=29,686$, deaths=1,480)` | `2004-2014 ($N=29,686$, deaths=1,574)` | Reconciled development deaths. |
| **7** | `docs/phase6_final_reconciliation_report.md` | Section 1.5 | Missing comparative section | Added complete Section 1.5 analyzing AUROC 0.6568 vs. 0.6253. | Resolved reviewer ambiguity. |
| **8** | `docs/phase6_final_reconciliation_report.pdf` | Entire document | Generated from previous markdown | Recompiled via ReportLab with all corrections. Size: 114.0 KB. | Synchronized official PDF deliverable. |
| **9** | `docs/manuscript.tex` | Lines 443–474 | Stale pre-Phase 6 Youden threshold ($\tau=0.0004$) | Synchronized with authoritative validation-derived Youden threshold ($\tau=0.04546$, Sens $47.18\%$, Acc $69.91\%$, $\text{MCC} = +0.0662$) and 10% capacity ($\tau=0.05261$). | Harmonized LaTeX manuscript with authoritative Phase 6 audit. |
