"""
Harmonization module for SWIFT-U5.
Standardizes predictors across 6 BDHS survey rounds (2004–2022).
Implements tiered predictors: T0 Structural, T1 Antenatal, T2 Birth.
Harmonizes administrative divisions across historical boundary changes.
"""

from typing import Dict, List
import pandas as pd
import numpy as np

def harmonize_divisions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize division names and administrative boundaries:
    - Normalizes spellings (Barisal -> Barishal, Chittagong -> Chattogram)
    - Creates division_7 (Mymensingh merged into Dhaka for consistent 2004-2022 trends)
    - Creates division_8 (Available for 2017-18 and 2022)
    """
    df = df.copy()
    
    div_raw = df['v024'].astype(str).str.strip().str.lower()
    
    # Map division names to canonical spelling
    name_map = {
        'barisal': 'barishal',
        'barishal': 'barishal',
        '1': 'barishal',
        '1.0': 'barishal',
        
        'chittagong': 'chattogram',
        'chattogram': 'chattogram',
        '2': 'chattogram',
        '2.0': 'chattogram',
        
        'dhaka': 'dhaka',
        '3': 'dhaka',
        '3.0': 'dhaka',
        
        'khulna': 'khulna',
        '4': 'khulna',
        '4.0': 'khulna',
        
        'rajshahi': 'rajshahi',
        '5': 'rajshahi',
        '5.0': 'rajshahi',
        
        'rangpur': 'rangpur',
        '6': 'rangpur',
        '6.0': 'rangpur',
        
        'sylhet': 'sylhet',
        '7': 'sylhet',
        '7.0': 'sylhet',
        
        'mymensingh': 'mymensingh',
        '8': 'mymensingh',
        '8.0': 'mymensingh'
    }
    
    # In 2017-18 and 2022, division coding has 8 categories:
    # 1: Barishal, 2: Chattogram, 3: Dhaka, 4: Khulna, 5: Mymensingh, 6: Rajshahi, 7: Rangpur, 8: Sylhet
    # Let's handle numeric coding specific to 2017-2022 vs earlier rounds
    round_series = df['round'].astype(str)
    
    def resolve_division(row):
        val = str(row['v024']).strip().lower()
        rnd = str(row['round'])
        
        if rnd in ['2017-18', '2022']:
            if val in ['5', '5.0', 'mymensingh']:
                return 'mymensingh'
            elif val in ['6', '6.0', 'rajshahi']:
                return 'rajshahi'
            elif val in ['7', '7.0', 'rangpur']:
                return 'rangpur'
            elif val in ['8', '8.0', 'sylhet']:
                return 'sylhet'
            elif val in ['1', '1.0', 'barishal', 'barisal']:
                return 'barishal'
            elif val in ['2', '2.0', 'chattogram', 'chittagong']:
                return 'chattogram'
            elif val in ['3', '3.0', 'dhaka']:
                return 'dhaka'
            elif val in ['4', '4.0', 'khulna']:
                return 'khulna'
        elif rnd in ['2011', '2014']:
            if val in ['5', '5.0', 'rajshahi']:
                return 'rajshahi'
            elif val in ['6', '6.0', 'rangpur']:
                return 'rangpur'
            elif val in ['7', '7.0', 'sylhet']:
                return 'sylhet'
            elif val in ['1', '1.0', 'barishal', 'barisal']:
                return 'barishal'
            elif val in ['2', '2.0', 'chattogram', 'chittagong']:
                return 'chattogram'
            elif val in ['3', '3.0', 'dhaka']:
                return 'dhaka'
            elif val in ['4', '4.0', 'khulna']:
                return 'khulna'
        else: # 2004, 2007 (6 divisions: Rajshahi included Rangpur)
            if val in ['5', '5.0', 'rajshahi']:
                return 'rajshahi'
            elif val in ['6', '6.0', 'sylhet']:
                return 'sylhet'
            elif val in ['1', '1.0', 'barishal', 'barisal']:
                return 'barishal'
            elif val in ['2', '2.0', 'chattogram', 'chittagong']:
                return 'chattogram'
            elif val in ['3', '3.0', 'dhaka']:
                return 'dhaka'
            elif val in ['4', '4.0', 'khulna']:
                return 'khulna'
                
        return name_map.get(val, val)

    df['division_name'] = df.apply(resolve_division, axis=1)
    
    # 7-division harmonization: Mymensingh was historically part of Dhaka until late 2015
    df['division_7'] = df['division_name'].replace({'mymensingh': 'dhaka'})
    
    # 8-division harmonization: available for rounds where Mymensingh existed
    df['division_8'] = df['division_name']
    
    return df

def harmonize_t0_structural(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize T0 Structural Predictors:
    - maternal_age_at_birth: computed from (b3 - v011) // 12
    - maternal_education: 0: none, 1: primary, 2: secondary, 3: higher
    - maternal_education_years: single years of education from v133
    - wealth_quintile: 1: poorest to 5: richest from v190
    - residence_urban: 1: urban, 0: rural from v025
    - birth_order: integer birth order
    - parity_before: birth_order - 1 (leakage-free!)
    - is_first_birth: 1 if first birth else 0
    - birth_interval_months: preceding birth interval (with first-birth imputation & flag)
    - is_twin: 1 if multiple birth else 0
    """
    df = df.copy()
    
    # Maternal age at child's birth in completed years
    df['maternal_age_at_birth'] = ((df['b3'] - df['v011']) // 12).astype(float)
    
    # Maternal education level
    edu_str = df['v106'].astype(str).str.strip().str.lower()
    edu_map = {
        'no education': 0, 'none': 0, '0': 0, '0.0': 0,
        'primary': 1, '1': 1, '1.0': 1,
        'secondary': 2, '2': 2, '2.0': 2,
        'higher': 3, '3': 3, '3.0': 3
    }
    df['maternal_education'] = edu_str.map(edu_map)
    # Fallback to numeric if needed
    if df['maternal_education'].isna().any():
        df['maternal_education'] = df['maternal_education'].fillna(
            pd.to_numeric(df['v106'], errors='coerce')
        )
    
    # Maternal education in single years (if available)
    if 'v133' in df.columns:
        df['maternal_education_years'] = pd.to_numeric(df['v133'], errors='coerce')
        # Treat DHS special codes 98/99 as NaN
        df.loc[df['maternal_education_years'] > 30, 'maternal_education_years'] = np.nan
        
    # Wealth quintile (1: poorest, 2: poorer, 3: middle, 4: richer, 5: richest)
    wealth_str = df['v190'].astype(str).str.strip().str.lower()
    wealth_map = {
        'poorest': 1, '1': 1, '1.0': 1,
        'poorer': 2, '2': 2, '2.0': 2,
        'middle': 3, '3': 3, '3.0': 3,
        'richer': 4, '4': 4, '4.0': 4,
        'richest': 5, '5': 5, '5.0': 5
    }
    df['wealth_quintile'] = wealth_str.map(wealth_map)
    if df['wealth_quintile'].isna().any():
        df['wealth_quintile'] = df['wealth_quintile'].fillna(
            pd.to_numeric(df['v190'], errors='coerce')
        )
        
    # Residence (1: Urban, 0: Rural)
    res_str = df['v025'].astype(str).str.strip().str.lower()
    df['residence_urban'] = res_str.isin(['urban', '1', '1.0']).astype(int)
    
    # Birth order and preceding parity (leakage-free)
    df['birth_order'] = pd.to_numeric(df['bord'], errors='coerce').fillna(1).astype(int)
    df['parity_before'] = df['birth_order'] - 1
    df['is_first_birth'] = (df['birth_order'] == 1).astype(int)
    
    # Preceding birth interval in months
    b11_numeric = pd.to_numeric(df['b11'], errors='coerce')
    df['birth_interval_missing'] = b11_numeric.isna().astype(int)
    # For first births, preceding interval is not applicable -> fill with -1 or median indicator
    df['preceding_birth_interval'] = b11_numeric.fillna(-1.0)
    
    # Multiple birth (Twin / Triplet)
    b0_str = df['b0'].astype(str).str.strip().str.lower()
    df['is_twin'] = (~b0_str.isin(['single birth', '0', '0.0', 'single', 'nan'])).astype(int)
    
    return df

def harmonize_t1_antenatal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize T1 Antenatal Predictors:
    - anc_visits: Number of antenatal visits from m14 (0 for 'no visits')
    - tetanus_toxoid: Tetanus injections from m1
    - iron_supplementation: Iron tablets/syrup during pregnancy from m45
    """
    df = df.copy()
    
    # ANC Visits (m14)
    if 'm14' in df.columns:
        m14_str = df['m14'].astype(str).str.strip().str.lower()
        # 'no antenatal visits' -> 0
        df['anc_visits'] = np.where(
            m14_str.str.contains('no antenatal'),
            0.0,
            pd.to_numeric(df['m14'], errors='coerce')
        )
        # 98/99 are special DHS 'don't know' codes
        df.loc[df['anc_visits'] >= 90, 'anc_visits'] = np.nan
        df['anc_visits_missing'] = df['anc_visits'].isna().astype(int)
    else:
        df['anc_visits'] = np.nan
        df['anc_visits_missing'] = 1

    # Tetanus Toxoid (m1)
    if 'm1' in df.columns:
        m1_str = df['m1'].astype(str).str.strip().str.lower()
        df['tetanus_toxoid'] = np.where(
            m1_str.str.contains('received no'),
            0.0,
            pd.to_numeric(df['m1'], errors='coerce')
        )
        df.loc[df['tetanus_toxoid'] >= 90, 'tetanus_toxoid'] = np.nan
        df['tetanus_toxoid_missing'] = df['tetanus_toxoid'].isna().astype(int)
    else:
        df['tetanus_toxoid'] = np.nan
        df['tetanus_toxoid_missing'] = 1

    # Iron supplementation (m45)
    if 'm45' in df.columns:
        m45_str = df['m45'].astype(str).str.strip().str.lower()
        df['iron_supplementation'] = np.where(
            m45_str.isin(['yes', '1', '1.0']),
            1.0,
            np.where(m45_str.isin(['no', '0', '0.0']), 0.0, np.nan)
        )
        df['iron_supplementation_missing'] = df['iron_supplementation'].isna().astype(int)
    else:
        df['iron_supplementation'] = np.nan
        df['iron_supplementation_missing'] = 1

    return df

def harmonize_t2_birth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize T2 Birth Predictors:
    - facility_delivery: 1 if delivery in hospital/clinic/health center, 0 if home
    - caesarean_delivery: 1 if C-section, 0 if vaginal delivery
    - birth_size: 1: Very small to 5: Very large
    """
    df = df.copy()
    
    # Place of delivery (m15)
    if 'm15' in df.columns:
        m15_str = df['m15'].astype(str).str.strip().str.lower()
        is_home = m15_str.str.contains('home')
        is_facility = m15_str.str.contains(
            'hospital|clinic|center|centre|mcwc|govt|medical|hut|public|private|sector',
            regex=True
        ) & (~is_home)
        
        df['facility_delivery'] = np.where(
            is_facility, 1.0,
            np.where(is_home, 0.0, np.nan)
        )
        df['facility_delivery_missing'] = df['facility_delivery'].isna().astype(int)
    else:
        df['facility_delivery'] = np.nan
        df['facility_delivery_missing'] = 1

    # Caesarean delivery (m17)
    if 'm17' in df.columns:
        m17_str = df['m17'].astype(str).str.strip().str.lower()
        df['caesarean_delivery'] = np.where(
            m17_str.isin(['yes', '1', '1.0']),
            1.0,
            np.where(m17_str.isin(['no', '0', '0.0']), 0.0, np.nan)
        )
        df['caesarean_delivery_missing'] = df['caesarean_delivery'].isna().astype(int)
    else:
        df['caesarean_delivery'] = np.nan
        df['caesarean_delivery_missing'] = 1

    # Birth size (m18)
    if 'm18' in df.columns:
        m18_str = df['m18'].astype(str).str.strip().str.lower()
        size_map = {
            'very small': 1.0,
            'smaller than average': 2.0,
            'average': 3.0,
            'larger than average': 4.0,
            'very large': 5.0
        }
        df['birth_size'] = m18_str.map(size_map)
        df['birth_size_missing'] = df['birth_size'].isna().astype(int)
    else:
        df['birth_size'] = np.nan
        df['birth_size_missing'] = 1

    return df

def harmonize_round_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run full harmonization pipeline for a single round's cohort."""
    df = harmonize_divisions(df)
    df = harmonize_t0_structural(df)
    df = harmonize_t1_antenatal(df)
    df = harmonize_t2_birth(df)
    return df

