"""
Validation split generators for the 2x2 Factorial Design (SWIFT-U5).

Arms:
  A - Random 70/30 split, unweighted
  B - Random 70/30 split, survey-weighted
  C - Forward-chained temporal, unweighted
  D - Forward-chained temporal, survey-weighted

Also produces:
  - Matched-n random arm (test set matched to temporal test set size)
  - Mixed-weight diagnostic arms (train unweighted / eval weighted & vice versa)
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Fixed random state for reproducibility
RANDOM_SEED = 42

# Ordered forward-chaining sequence
ROUND_ORDER = ['2004', '2007', '2011', '2014', '2017-18', '2022']

# Predictors to include (the complete leakage-free feature set)
FEATURE_COLUMNS = [
    # T0 Structural
    'maternal_age_at_birth', 'maternal_education', 'maternal_education_years',
    'wealth_quintile', 'residence_urban', 'birth_order', 'parity_before',
    'is_first_birth', 'preceding_birth_interval', 'birth_interval_missing', 'is_twin',
    # Division (encoded separately in pipeline)
    # T1 Antenatal (with missingness indicators)
    'anc_visits', 'anc_visits_missing',
    'tetanus_toxoid', 'tetanus_toxoid_missing',
    'iron_supplementation', 'iron_supplementation_missing',
    # T2 Birth (with missingness indicators)
    'facility_delivery', 'facility_delivery_missing',
    'caesarean_delivery', 'caesarean_delivery_missing',
    'birth_size', 'birth_size_missing',
]

TARGET_COLUMN = 'target_u5m'
WEIGHT_COLUMN = 'sample_weight'
ROUND_COLUMN = 'round'
DIVISION_COLUMN = 'division_7'
# Wave-scoped composite identifiers: use these for bootstrap resampling.
# DHS re-uses raw PSU/strata IDs across waves; psu_wave = "<round>_<v021>"
# guarantees uniqueness across the pooled multi-wave dataset.
PSU_COLUMN = 'psu_wave'
STRATA_COLUMN = 'strata_wave'


def get_forward_chaining_cuts(df: pd.DataFrame) -> List[Dict]:
    """
    Generate all forward-chaining temporal splits.
    Each cut: train on rounds [0..i], test on round [i+1].
    Returns list of dicts with 'train_rounds', 'test_round', 'train_idx', 'test_idx'.
    """
    cuts = []
    available = [r for r in ROUND_ORDER if r in df[ROUND_COLUMN].unique()]
    for i in range(len(available) - 1):
        train_rounds = available[:i+1]
        test_round = available[i+1]
        train_idx = df[df[ROUND_COLUMN].isin(train_rounds)].index
        test_idx = df[df[ROUND_COLUMN] == test_round].index
        cuts.append({
            'train_rounds': train_rounds,
            'test_round': test_round,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        })
    return cuts


def get_random_split(df: pd.DataFrame, test_size: float = 0.30) -> Tuple[pd.Index, pd.Index]:
    """
    Stratified random 70/30 split on the full pooled dataset.
    Stratified by target to preserve prevalence.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    dead_idx = df[df[TARGET_COLUMN] == 1].index.tolist()
    alive_idx = df[df[TARGET_COLUMN] == 0].index.tolist()

    rng.shuffle(dead_idx)
    rng.shuffle(alive_idx)

    n_dead_test = int(len(dead_idx) * test_size)
    n_alive_test = int(len(alive_idx) * test_size)

    test_idx = pd.Index(dead_idx[:n_dead_test] + alive_idx[:n_alive_test])
    train_idx = pd.Index(dead_idx[n_dead_test:] + alive_idx[n_alive_test:])
    return train_idx, test_idx


def get_matched_n_random_split(df: pd.DataFrame, n_test_target: int,
                               random_state: Optional[int] = None) -> Tuple[pd.Index, pd.Index]:
    """
    Stratified random split where test set is exactly n_test_target observations.
    Used to match temporal test set size for fair comparison.
    """
    seed = random_state if random_state is not None else (RANDOM_SEED + 1)
    rng = np.random.default_rng(seed)
    dead_idx = df[df[TARGET_COLUMN] == 1].index.tolist()
    alive_idx = df[df[TARGET_COLUMN] == 0].index.tolist()

    prevalence = len(dead_idx) / len(df)
    n_dead_test = max(1, int(n_test_target * prevalence))
    n_alive_test = n_test_target - n_dead_test

    rng.shuffle(dead_idx)
    rng.shuffle(alive_idx)

    test_idx = pd.Index(dead_idx[:n_dead_test] + alive_idx[:n_alive_test])
    train_idx = df.index.difference(test_idx)
    return train_idx, test_idx


def encode_features(df: pd.DataFrame, train_idx: pd.Index,
                    imputer_medians: Optional[Dict] = None,
                    division_categories: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Prepare feature matrix with train-only imputation.

    Division_7 is one-hot encoded using **training categories only**.
    Any division value present in test but absent in training is mapped to
    all-zero dummy columns (unknown-category safe).

    Parameters
    ----------
    df : full DataFrame (all rows; slicing to train/test is done by callers)
    train_idx : Index of training rows; used to discover OHE categories and medians
    imputer_medians : pre-fitted median dict (pass on subsequent test-set calls)
    division_categories : pre-fitted list of division dummies (pass on test-set calls)

    Returns
    -------
    X : encoded float64 DataFrame aligned to all rows of df
    imputer_medians : fitted imputation dict
    """
    X = df[FEATURE_COLUMNS].copy()

    # ------------------------------------------------------------------ #
    # One-hot encode division_7 from TRAIN categories only                #
    # ------------------------------------------------------------------ #
    div_series = df[DIVISION_COLUMN].astype(str)

    if division_categories is None:
        # Discover categories from training data only
        train_div = div_series.loc[train_idx]
        division_categories = sorted(train_div.unique().tolist())

    # Encode the division column using discovered training categories
    for cat in division_categories:
        X[f'div_{cat}'] = (div_series == cat).astype(np.float64)
    # Unknown categories (present in test but not training) → all zeros → handled naturally

    # ------------------------------------------------------------------ #
    # Train-only imputation: fit medians on train, apply to all           #
    # ------------------------------------------------------------------ #
    if imputer_medians is None:
        imputer_medians = {}
        train_X = X.loc[train_idx]
        for col in X.columns:
            if X[col].isna().any():
                med = train_X[col].median()
                imputer_medians[col] = 0.0 if pd.isna(med) else float(med)

    for col, median_val in imputer_medians.items():
        if col in X.columns:
            X[col] = X[col].fillna(median_val)

    # Final guarantee: zero NaNs and numeric float64
    X = X.fillna(0.0).astype(np.float64)

    return X, imputer_medians


def compute_normalized_weights(df: pd.DataFrame, idx: pd.Index,
                                use_weights: bool = True) -> np.ndarray:
    """
    Compute normalized sample weights for a given index.
    When use_weights=False, returns uniform weights (equivalent to unweighted).
    """
    if not use_weights:
        return np.ones(len(idx))

    w = df.loc[idx, WEIGHT_COLUMN].values.astype(float)
    # Normalize to sum to n_obs (preserves scale-invariance in sklearn metrics)
    w = w / w.mean()
    return w

