# SWIFT-U5: Survey-Weighted and Temporally Validated Risk Modeling for Under-Five Mortality

Implementation of the complete study plan defined in `U5M_Study_Plan.pdf`: **"Survey-Weighted and Temporally Validated Risk Modeling for Under-Five Mortality Using Bangladesh DHS Data"**.

---

#### 📁 Repository Structure

```
SWIFT-U5/
├── U5M_Study_Plan.pdf          # Master study protocol
├── README.md                   # Project overview & roadmap
├── configs/
│   └── config.yaml             # Study configuration (rounds, predictors, design)
│
├── data/                       # Local-only; survey files and record-level cohorts are not tracked
│   ├── zips/                   # Place authorized Bangladesh DHS BR archives here
│   ├── raw/                    # Extracted Stata .dta files
│   ├── interim/                # Standardized & cleaned per-round datasets
│   └── processed/              # Harmonized cohorts & folds
│
├── src/
│   ├── data/                   # Data ingestion, harmonization, & cohort construction
│   │   ├── extract.py          # Extraction from ZIP/Stata formats
│   │   ├── harmonize.py        # Variable harmonization & division crosswalk (7 vs 8)
│   │   ├── cohort.py           # Exposure-complete filtering (<5 years) & leakage exclusion
│   │   └── dataset.py          # Master dataset generation pipeline
│   │
│   ├── features/               # Feature processing & selection
│   │   ├── build_features.py   # Train-only imputation & missingness indicators
│   │   └── select.py           # Weighted LASSO + XGBoost permutation + Stability selection
│   │
│   ├── models/                 # Predictive models & SWIFT-U5 components
│   │   ├── baselines.py        # Weighted Logistic, XGBoost, Random Forest
│   │   ├── module_a_weight.py  # Module A: Recency decay × density-ratio shift weighting
│   │   ├── module_b_shrink.py  # Module B: Division-shrunk EB logit head
│   │   ├── module_c_recalib.py # Module C: Drift-aware recalibration & SHAP drift index
│   │   └── swift_u5.py         # Full SWIFT-U5 integrated estimator & ablation wrapper
│   │
│   ├── validation/             # Validation splits & factorial design
│   │   ├── splits.py           # Random, Forward-chained temporal, & Matched-n splits
│   │   └── experiment_2x2.py   # Factorial design runner (Arms A, B, C, D + mixed checks)
│   │
│   ├── evaluation/             # Metrics, calibration, bootstrapping, & DCA
│   │   ├── metrics.py          # Weighted AUROC, AUPRC, Brier, ECE, NNS @ 5/10/20%
│   │   ├── bootstrap.py        # PSU-stratified cluster bootstrap & paired CIs
│   │   └── decision_curve.py   # Decision curve analysis (net benefit)
│   │
│   ├── policy/                 # Division policy layer & equity audit
│   │   ├── atlas.py            # Division risk atlas & temporal trends
│   │   ├── nns_capacity.py     # Division NNS & capacity tables
│   │   ├── shap_analysis.py    # Division-level SHAP drivers & temporal drift
│   │   └── equity_audit.py     # Equity audit across wealth quintile & residence
│   │
│   └── utils/
│       ├── config.py           # Python config loader
│       └── survey_weights.py   # Survey weights & cluster-aware utility functions
│
├── experiments/                # End-to-end execution scripts
│   ├── 01_run_harmonization.py     # Task 1: Build leakage-free dataset
│   ├── 02_run_2x2_benchmark.py     # Task 2: 2 x 2 Factorial design with baselines
│   ├── 03_run_feature_selection.py # Task 3: Feature selection & stability analysis
│   ├── 04_run_swift_u5_ablation.py # Task 4: SWIFT-U5 modules & ablation study
│   └── 05_run_policy_analysis.py   # Task 5: Division layer, policy tables, & equity audit
│
├── outputs/                    # Study deliverables (1:1 with Section 11 of Study Plan)
│   ├── tables/                 # Tables 1–7 (CSV and LaTeX format)
│   ├── figures/                # Figures 1–5 (high-resolution PNG / PDF)
│   └── reports/                # Model summaries, metrics logs, ablation outputs
│
├── notebooks/                  # Exploratory & inspection notebooks
└── tests/                      # Unit tests (leakage checks, survey weight tests)
```

## Data access

Raw Bangladesh DHS survey files and record-level derived cohorts are excluded from this repository. Obtain the survey files directly from the DHS Program under its access terms, then place the authorized archives in `data/zips/` before running the harmonization pipeline. The `data/` directory is ignored by Git; do not commit or redistribute survey microdata or record-level cohorts.

---

## 🎯 Deliverables Mapping (Section 11)

| Item              | Output File                                         | Description                                                         |
| :---------------- | :-------------------------------------------------- | :------------------------------------------------------------------ |
| **Table 1** | `outputs/tables/table1_cohort_summary.csv`        | Weighted sample, deaths, and prevalence by round & division         |
| **Table 2** | `outputs/tables/table2_2x2_main_results.csv`      | 2 × 2 main results (Arms A, B, C, D) with cluster bootstrap CIs    |
| **Fig 1**   | `outputs/figures/fig1_inflation_plot.png`         | Inflation plot: Random vs. Temporal validation across rounds        |
| **Table 3** | `outputs/tables/table3_effect_decomposition.csv`  | Effect decomposition: validation effect, weight effect, interaction |
| **Fig 2**   | `outputs/figures/fig2_calibration_curves.png`     | Calibration curves across Arms A to D                               |
| **Fig 3**   | `outputs/figures/fig3_feature_stability.png`      | Feature selection frequency (stable vs. drifting)                   |
| **Table 4** | `outputs/tables/table4_swift_vs_baselines.csv`    | SWIFT-U5 vs. Baselines per test round                               |
| **Table 5** | `outputs/tables/table5_ablation_modules.csv`      | Ablation of SWIFT-U5 modules (A, B, C)                              |
| **Fig 4**   | `outputs/figures/fig4_division_risk_atlas.png`    | Division risk atlas and temporal trends                             |
| **Table 6** | `outputs/tables/table6_division_nns_capacity.csv` | Division NNS and capacity table with CIs                            |
| **Fig 5**   | `outputs/figures/fig5_shap_drift_division.png`    | SHAP drift and top risk drivers by division                         |
| **Table 7** | `outputs/tables/table7_equity_audit.csv`          | Equity audit by wealth quintile and urban/rural residence           |

---

## 🔄 Order of Work (Section 14)

1. **Step 1: Harmonize variables & build leakage-free dataset** (`experiments/01_run_harmonization.py`)

   - Extract BR files from 6 rounds (2004, 2007, 2011, 2014, 2017-18, 2022).
   - Filter to births in 5 years prior to survey; ensure exposure completeness.
   - Extract design variables: weights (`v005`), PSU (`v021`), strata (`v022`).
   - Harmonize divisions (7 divisions 2004–2014, 8 divisions 2017–2022).
   - Select tiered predictors (T0 structural, T1 antenatal, T2 birth). Exclude post-outcome variables.
2. **Step 2: Run the 2 × 2 Factorial Design with baselines** (`experiments/02_run_2x2_benchmark.py`)

   - Arm A: Random split, unweighted.
   - Arm B: Random split, survey-weighted.
   - Arm C: Forward-chained temporal, unweighted.
   - Arm D: Forward-chained temporal, survey-weighted.
   - Matched-n random arm + mixed checks (train unweighted / test weighted & reverse).
3. **Step 3: Feature selection and stability** (`experiments/03_run_feature_selection.py`)

   - Pre-outcome filter (<30% missing, correlation < 0.8).
   - Weighted LASSO logistic + permutation importance on weighted XGBoost within PSU folds.
   - Stability selection (≥70% selection threshold).
   - Rolling window tracking for stable vs. drifting features.
4. **Step 4: SWIFT-U5 modules & ablation** (`experiments/04_run_swift_u5_ablation.py`)

   - Module A: Recency decay × density-ratio shift weighting.
   - Module B: Division-shrunk logit head (Empirical Bayes).
   - Module C: Drift-aware recalibration & SHAP drift tracking.
   - Evaluate baselines (Weighted Logistic, XGBoost, Random Forest) and module ablation.
5. **Step 5: Division policy layer & figures** (`experiments/05_run_policy_equity.py`)

   - Division risk atlas, trends, NNS tables at 5%, 10%, 20% capacity with 95% PSU cluster CIs.
   - Division-level SHAP drivers, temporal SHAP Drift Index (SDI), and equity audits across wealth and residence.
   - Generate all publication tables (1–7) and figures (1–5).

6. **Step 6: Academic Manuscript & Artifact Packaging** (`docs/manuscript.tex` and `docs/manuscript.md`)

   - Full academic paper prepared in Elsevier CAS single-column format (`cas-sc`).
   - Integrated BibTeX bibliography (`docs/cas-refs.bib`).
   - All 7 tables and 5 figures embedded with exact empirical results.
   - 21 integration unit tests passing across all pipeline modules.

---

## 🚀 Reproduction Instructions

To replicate the entire study from raw data to the final publication manuscript:

```bash
# 1. Environment Setup
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run Test Suite
python -m unittest discover -s tests

# 3. Execute Pipelines Sequentially
python experiments/01_run_harmonization.py
python experiments/02_run_2x2_benchmark.py
python experiments/03_run_feature_selection.py
python experiments/04_run_swift_u5_ablation.py
python experiments/05_run_policy_equity.py

# 4. View Compiled Manuscript
# LaTeX Source: docs/manuscript.tex
# BibTeX File:  docs/cas-refs.bib
# Markdown Doc: docs/manuscript.md
```
