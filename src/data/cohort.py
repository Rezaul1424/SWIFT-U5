"""
Cohort construction module for SWIFT-U5.
Filters raw BDHS Birth Recode (BR) datasets to 5-year exposure-complete cohorts,
extracts survey design variables, and ensures zero post-outcome data leakage.
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
    cohort['is_exposure_complete'] = 1
    
    # Normalize survey weights: DHS v005 has 6 implied decimal places
    cohort['sample_weight'] = pd.to_numeric(cohort['v005'], errors='coerce') / 1_000_000.0
    # Exclude records with zero or negative sample weights (DHS non-sample / pilot clusters)
    cohort = cohort[cohort['sample_weight'] > 0].copy()
    cohort['psu'] = cohort['v021'].astype(str)
    cohort['strata'] = cohort['v022'].astype(str)
    
    # Strict verification: Ensure no leakage columns remain
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in cohort.columns, f"Data leakage detected: {leak_col} present in cohort!"
        
    return cohort
