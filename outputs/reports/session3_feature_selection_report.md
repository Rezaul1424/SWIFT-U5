# SWIFT-U5 Feature Selection and Temporal Stability Report

## 1. Domain Filtering Results (Training Data Only)
- **Missingness Threshold:** Max 30% missing in training rounds (2004–2017-18).
- **Dropped Due to High Missingness:** [('tetanus_toxoid', 0.542), ('iron_supplementation', 0.602), ('birth_size', 0.649)]
- **Collinearity Threshold:** $|r| > 0.80$ pairwise correlation.
- **Dropped Due to Collinearity:** [('parity_before', 'birth_order', 1.0)]
- **Features Retained for Stability Selection:** ['maternal_age_at_birth', 'maternal_education', 'wealth_quintile', 'residence_urban', 'birth_order', 'is_first_birth', 'preceding_birth_interval', 'is_twin', 'anc_visits', 'facility_delivery', 'caesarean_delivery']

## 2. Stability Selection Frequency (Master Training Set: 2004–2017-18)

| Feature | LASSO Freq (%) | XGBoost Freq (%) | Min Freq (%) | Classification |
| :--- | :---: | :---: | :---: | :---: |
| `anc_visits` | 100.0% | 100.0% | 100.0% | **Stable Core** |
| `preceding_birth_interval` | 100.0% | 95.0% | 95.0% | **Stable Core** |
| `is_twin` | 100.0% | 85.0% | 85.0% | **Stable Core** |
| `maternal_education` | 100.0% | 82.5% | 82.5% | **Stable Core** |
| `maternal_age_at_birth` | 97.5% | 65.0% | 65.0% | Contextual |
| `birth_order` | 100.0% | 60.0% | 60.0% | Contextual |
| `caesarean_delivery` | 100.0% | 55.0% | 55.0% | Contextual |
| `div_chattogram` | 72.5% | 52.5% | 52.5% | Contextual |
| `facility_delivery` | 100.0% | 52.5% | 52.5% | Contextual |
| `wealth_quintile` | 100.0% | 52.5% | 52.5% | Contextual |
| `div_sylhet` | 100.0% | 52.5% | 52.5% | Contextual |
| `div_barishal` | 50.0% | 50.0% | 50.0% | Contextual |
| `div_rajshahi` | 100.0% | 50.0% | 50.0% | Contextual |
| `is_first_birth` | 100.0% | 47.5% | 47.5% | Contextual |
| `residence_urban` | 100.0% | 47.5% | 47.5% | Contextual |
| `div_dhaka` | 87.5% | 47.5% | 47.5% | Contextual |
| `div_khulna` | 87.5% | 47.5% | 47.5% | Contextual |
| `div_rangpur` | 100.0% | 47.5% | 47.5% | Contextual |

## 3. Rolling Window Temporal Trajectory

Features classified as **Stable Core** retain consistent discriminatory signal ($\ge 70\%$ in both algorithms) across time.
Features classified as **Drifting / Contextual** demonstrate time-varying predictive value as health system infrastructure expands.

| Feature | 2004 | 2004–07 | 2004–11 | 2004–14 | 2004–18 | Mean Score | Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `anc_visits` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **Stable Core** |
| `preceding_birth_interval` | 1.00 | 0.97 | 1.00 | 0.85 | 0.95 | 0.95 | **Stable Core** |
| `is_twin` | 0.88 | 0.78 | 0.78 | 0.72 | 0.85 | 0.80 | **Stable Core** |
| `maternal_education` | 0.88 | 0.80 | 0.80 | 0.65 | 0.82 | 0.79 | **Stable Core** |
| `maternal_age_at_birth` | 0.97 | 0.60 | 0.62 | 0.40 | 0.65 | 0.65 | **Drifting / Contextual** |
| `birth_order` | 0.72 | 0.68 | 0.62 | 0.38 | 0.60 | 0.60 | **Excluded / Low Priority** |
| `facility_delivery` | 0.72 | 0.50 | 0.62 | 0.33 | 0.53 | 0.54 | **Excluded / Low Priority** |
| `caesarean_delivery` | 0.65 | 0.47 | 0.57 | 0.42 | 0.55 | 0.54 | **Excluded / Low Priority** |
| `tetanus_toxoid` | 0.97 | 0.88 | 0.82 | 0.00 | 0.00 | 0.54 | **Drifting / Contextual** |
| `div_sylhet` | 0.68 | 0.45 | 0.55 | 0.35 | 0.53 | 0.51 | **Excluded / Low Priority** |
| `residence_urban` | 0.72 | 0.47 | 0.53 | 0.33 | 0.47 | 0.51 | **Excluded / Low Priority** |
| `div_chattogram` | 0.62 | 0.42 | 0.55 | 0.35 | 0.53 | 0.49 | **Excluded / Low Priority** |
| `wealth_quintile` | 0.68 | 0.40 | 0.53 | 0.35 | 0.53 | 0.49 | **Excluded / Low Priority** |
| `div_khulna` | 0.60 | 0.45 | 0.53 | 0.33 | 0.47 | 0.47 | **Excluded / Low Priority** |
| `div_dhaka` | 0.72 | 0.40 | 0.40 | 0.33 | 0.47 | 0.47 | **Excluded / Low Priority** |
| `div_barishal` | 0.57 | 0.40 | 0.50 | 0.33 | 0.50 | 0.46 | **Excluded / Low Priority** |
| `is_first_birth` | 0.55 | 0.40 | 0.50 | 0.33 | 0.47 | 0.45 | **Excluded / Low Priority** |
| `div_rajshahi` | 0.40 | 0.40 | 0.55 | 0.33 | 0.50 | 0.43 | **Excluded / Low Priority** |
| `iron_supplementation` | 0.93 | 0.75 | 0.00 | 0.00 | 0.00 | 0.34 | **Drifting / Contextual** |
| `div_rangpur` | 0.00 | 0.00 | 0.50 | 0.35 | 0.47 | 0.27 | **Excluded / Low Priority** |

## 4. Methodological Compliance & Leakage Isolation
- **Training-Data Only:** All domain filtering, imputation, and stability selection were strictly confined to training folds (`df_train`).
- **Test Set Isolation:** The 2022 test round was never accessed during feature selection.
- **SHAP Quarantine Verified:** Per Section 6 Rule 5, SHAP was strictly excluded from feature selection and reserved exclusively for post-hoc interpretation in Session 5.