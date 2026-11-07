"""
Cohort construction module for SWIFT-U5.

Filters raw BDHS Birth Recode (BR) datasets to the standard cohort of children
aged 0–59 completed months at the time of interview, extracts survey design
variables, and ensures zero post-outcome data leakage.

ESTIMAND NOTE
-------------
The outcome (target_u5m) is an *observed* under-five death recorded by the
interview date.  A living child aged < 60 months at interview has NOT completed
five years of potential follow-up — their eventual age-5 survival is unknown.
The flag ``completed_5yr_followup`` distinguishes children whose full five-year
window could have been observed (died before 60 months, OR alive and already
≥ 60 months — which cannot occur in this cohort by construction) from those
still within the follow-up window.  Sensitivity analyses should be conducted
on the sub-cohort of children with ``completed_5yr_followup == True``.

PSU IDENTITY NOTE
-----------------
DHS re-uses PSU identifier integers across survey waves.  PSU "361" in 2004 is
a different physical cluster from PSU "361" in 2022.  The columns ``psu_wave``
and ``strata_wave`` carry wave-prefixed composite identifiers that are unique
across the pooled multi-wave dataset and must be used for bootstrap resampling.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from src.utils.config import RAW_DATA_DIR, load_config

# Predictor columns to extract directly from BR files
COLUMNS_TO_LOAD = [
    'v008', 'b3', 'v011', 'v106', 'v133', 'v190', 'v025', 'v024',
    'bord', 'b0', 'b11',
    'm14', 'm1', 'm45', 'm15', 'm17', 'm18',
    'b5', 'b7',
    'v005', 'v021', 'v022'
]

# Explicit post-outcome leakage variables to drop/ensure absent
LEAKAGE_COLUMNS = [
    'b4',   # Child's age at death (only recorded if dead)
    'b19',  # Child's current age in months (only if alive)
    'm4',   # Duration of breastfeeding (stops if child dies)
    'm5',   # Months of breastfeeding
    'v404', # Currently breastfeeding
    'v201', # Total children ever born at interview (includes births after this child)
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10' # Vaccination indicators
]

def load_round_raw_br(round_name: str) -> pd.DataFrame:
    """Load raw Stata BR file for a given round, selecting only required pre-outcome columns."""
    round_dir = RAW_DATA_DIR / round_name
    files = list(round_dir.glob("*.dta")) + list(round_dir.glob("*.DTA"))
    if not files:
        raise FileNotFoundError(f"No Stata file found in {round_dir}")
    
    stata_file = files[0]
    
    # Inspect available columns (case-insensitive)
    reader = pd.read_stata(stata_file, iterator=True)
    available_cols = {col.lower(): col for col in reader.variable_labels().keys()}
    
    load_cols = []
    col_mapping = {}
    for col in COLUMNS_TO_LOAD:
        if col.lower() in available_cols:
            actual_col = available_cols[col.lower()]
            load_cols.append(actual_col)
            col_mapping[actual_col] = col.lower()
            
    df = pd.read_stata(stata_file, columns=load_cols)
    df = df.rename(columns=col_mapping)
    df['round'] = round_name
    return df

def filter_exposure_complete_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to births in the 5 years before interview (0-59 completed months)
    with complete exposure tracking (survival status and mortality timing known).
    """
    df = df.copy()
    
    # Age of child at survey in completed months
    df['age_at_survey_months'] = df['v008'] - df['b3']
    
    # Restrict to births occurring 0 to 59 months prior to interview
    cohort = df[(df['age_at_survey_months'] >= 0) & (df['age_at_survey_months'] < 60)].copy()
    
    # Encode under-five mortality outcome
    # b5: 1 / 'yes' = alive; 0 / 'no' = dead
    b5_str = cohort['b5'].astype(str).str.strip().str.lower()
    is_dead = b5_str.isin(['no', '0', 'dead', '0.0'])
    is_alive = b5_str.isin(['yes', '1', 'alive', '1.0'])
    
    # Exclude records with ambiguous survival status (exposure incomplete)
    valid_mask = is_dead | is_alive
    cohort = cohort[valid_mask].copy()
    
    # Under-five death event: died and age at death b7 < 60 months
    # (If child died, b7 is age at death in completed months)
    b7_numeric = pd.to_numeric(cohort['b7'], errors='coerce')
    cohort['target_u5m'] = (is_dead[valid_mask] & (b7_numeric < 60)).astype(int)
    cohort['age_at_death_months'] = np.where(cohort['target_u5m'] == 1, b7_numeric, np.nan)
    
    # Exposure in months: for dead children, b7; for alive children, age at survey
    cohort['exposure_months'] = np.where(
        cohort['target_u5m'] == 1,
        b7_numeric,
        cohort['age_at_survey_months']
    )
    # is_exposure_complete DEPRECATED — was incorrectly hardcoded to 1 for all
    # children regardless of age.  Retained as NaN for backward-compat.
    # Use completed_5yr_followup (computed below) instead.
    cohort['is_exposure_complete'] = np.nan

    # Normalize survey weights: DHS v005 has 6 implied decimal places
    cohort['sample_weight'] = pd.to_numeric(cohort['v005'], errors='coerce') / 1_000_000.0
    # Exclude records with zero or negative sample weights (DHS non-sample / pilot clusters)
    cohort = cohort[cohort['sample_weight'] > 0].copy()
    cohort['psu'] = cohort['v021'].astype(str)
    cohort['strata'] = cohort['v022'].astype(str)
    
    # ------------------------------------------------------------------ #
    # Follow-up completeness flag                                         #
    # ------------------------------------------------------------------ #
    # A child aged < 60 months at interview who is alive has NOT yet had  #
    # the opportunity to die before age 5.  Their survival to age 5 is   #
    # censored.  Only dead children (whose death age < 60 is recorded)   #
    # have a definitively completed 5-year follow-up window.             #
    # Note: by cohort construction age_at_survey_months is always < 60,  #
    # so "alive AND age >= 60" is impossible here.                        #
    cohort['completed_5yr_followup'] = (cohort['target_u5m'] == 1).astype(int)

    # Composite wave-scoped PSU / strata identifiers.                    #
    # DHS re-uses integer PSU IDs across survey waves; a raw ID of "361" #
    # in 2004 is a different physical cluster from "361" in 2022.        #
    # Always use psu_wave / strata_wave for bootstrap resampling.        #
    cohort['psu_wave']    = cohort['round'] + '_' + cohort['v021'].astype(str)
    cohort['strata_wave'] = cohort['round'] + '_' + cohort['v022'].astype(str)

    # Keep raw psu/strata for backward-compatibility with existing code,  #
    # but document that they must NOT be used for multi-wave bootstraps.  #
    cohort['psu']    = cohort['v021'].astype(str)
    cohort['strata'] = cohort['v022'].astype(str)

    # Strict verification: Ensure no leakage columns remain
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in cohort.columns, f"Data leakage detected: {leak_col} present in cohort!"

    # Ensure outcome source variables are not mistakenly used as features.
    # They remain in the frame for reference / audit but are marked here.
    # LeakageFreePreprocessor's CANDIDATE_FEATURES list must NOT include them.
    assert 'b5' in cohort.columns, "b5 expected for audit; drop explicitly in feature build step."
    assert 'b7' in cohort.columns, "b7 expected for audit; drop explicitly in feature build step."

    return cohort
