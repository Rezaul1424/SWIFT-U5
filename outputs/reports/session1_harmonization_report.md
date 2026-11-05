# SWIFT-U5 Harmonization & Data Integrity Report

- Total Births in Cohort (0-59 months pre-survey): 51,489
- Total Under-Five Deaths: 2,338 (45.4 per 1,000 unweighted)
- Total Weighted Deaths: 2,312.7

## Predictor Completeness Across Rounds (Percentage Available)

| Predictor | 2004 | 2007 | 2011 | 2014 | 2017-18 | 2022 | Total Missing (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `maternal_age_at_birth` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `maternal_education` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `wealth_quintile` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `residence_urban` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `division_name` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `birth_order` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `parity_before` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `is_first_birth` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `preceding_birth_interval` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `is_twin` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `anc_visits` | 77.6% | 80.0% | 83.6% | 56.9% | 57.7% | 38.6% | 37.6% |
| `tetanus_toxoid` | 77.5% | 80.1% | 83.5% | 0.0% | 0.0% | 38.5% | 56.0% |
| `iron_supplementation` | 77.6% | 80.0% | 0.0% | 0.0% | 57.6% | 38.6% | 60.5% |
| `facility_delivery` | 99.8% | 95.4% | 96.2% | 57.7% | 57.9% | 38.5% | 30.5% |
| `caesarean_delivery` | 99.9% | 100.0% | 99.8% | 57.0% | 61.0% | 40.6% | 28.3% |
| `birth_size` | 0.0% | 0.0% | 99.9% | 60.0% | 0.0% | 40.6% | 63.5% |

## Post-Outcome Leakage Check
- Breastfeeding (`m4`, `m5`, `v404`): **EXCLUDED** (0% leakage)
- Child age at survey / death (`b19`, `b4`): **EXCLUDED** (0% leakage)
- Children ever born at interview (`v201`): **EXCLUDED** (Replaced by strictly pre-outcome `parity_before = birth_order - 1`)
- Current child vaccination (`h*`): **EXCLUDED** (0% leakage)

## Exposure Completeness
- All observations have known and complete 5-year exposure tracking.
- Exposure months calculated explicitly: `exposure_months` = `b7` for deaths, `v008 - b3` for survivors.