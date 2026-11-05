"""
Feature engineering and train-only preprocessing pipeline for SWIFT-U5.
Applies train-fitted imputation, missingness indicator flags, and feature encoding
strictly preventing data leakage from test or future survey rounds.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

CANDIDATE_FEATURES = [
    # T0 Structural
    'maternal_age_at_birth',
    'maternal_education',
    'wealth_quintile',
    'residence_urban',
    'birth_order',
    'parity_before',
    'is_first_birth',
    'preceding_birth_interval',
    'is_twin',
    # T1 Antenatal
    'anc_visits',
    'tetanus_toxoid',
    'iron_supplementation',
    # T2 Birth
    'facility_delivery',
    'caesarean_delivery',
    'birth_size'
]

DIVISION_COLUMN = 'division_7'
TARGET_COLUMN = 'target_u5m'
WEIGHT_COLUMN = 'sample_weight'
PSU_COLUMN = 'psu'
STRATA_COLUMN = 'strata'
ROUND_COLUMN = 'round'


class LeakageFreePreprocessor:
    """
    Leakage-free preprocessor that learns statistics strictly on training data
    and applies them consistently across validation / future rounds.
    """
    def __init__(self, include_missing_flags: bool = True):
        self.include_missing_flags = include_missing_flags
        self.medians_: Dict[str, float] = {}
        self.division_categories_: List[str] = []
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame, feature_subset: Optional[List[str]] = None) -> 'LeakageFreePreprocessor':
        features = feature_subset if feature_subset is not None else CANDIDATE_FEATURES
        self.medians_ = {}

        for col in features:
            if col in df_train.columns:
                med = df_train[col].median()
                self.medians_[col] = 0.0 if pd.isna(med) else float(med)

        # Record unique division categories from train
        if DIVISION_COLUMN in df_train.columns:
            self.division_categories_ = sorted(df_train[DIVISION_COLUMN].unique().tolist())

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, feature_subset: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transform!")

        features = feature_subset if feature_subset is not None else [
            col for col in CANDIDATE_FEATURES if col in self.medians_
        ]

        transformed_cols = []
        feature_names = []

        # 1. Continuous / ordinal features + missingness flags
        for col in features:
            if col not in df.columns:
                col_data = np.full(len(df), self.medians_.get(col, 0.0))
                missing_flag = np.ones(len(df), dtype=float)
            else:
                series = pd.to_numeric(df[col], errors='coerce')
                missing_flag = series.isna().astype(float).values
                col_data = series.fillna(self.medians_.get(col, 0.0)).values

            transformed_cols.append(col_data)
            feature_names.append(col)

            if self.include_missing_flags and missing_flag.sum() > 0:
                flag_name = f"{col}_is_missing"
                transformed_cols.append(missing_flag)
                feature_names.append(flag_name)

        # 2. One-hot encoded division
        if self.division_categories_ and DIVISION_COLUMN in df.columns:
            div_series = df[DIVISION_COLUMN].astype(str)
            for div in self.division_categories_:
                div_flag = (div_series == div).astype(float).values
                transformed_cols.append(div_flag)
                feature_names.append(f"div_{div}")

        X = np.column_stack(transformed_cols).astype(np.float64)
        self.feature_names_ = feature_names
        return X, feature_names

    def fit_transform(self, df_train: pd.DataFrame, feature_subset: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        self.fit(df_train, feature_subset)
        return self.transform(df_train, feature_subset)

