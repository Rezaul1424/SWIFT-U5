"""
Baseline models for the 2x2 Factorial Benchmark (SWIFT-U5).

Models:
  - Weighted Logistic Regression (L2 penalized)
  - Weighted XGBoost
  - Weighted Random Forest
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_weighted_logistic(class_weight: str = 'balanced',
                            C: float = 0.1,
                            max_iter: int = 1000,
                            random_state: int = 42) -> LogisticRegression:
    """
    L2-penalized Logistic Regression using liblinear solver.
    liblinear is fast, robust, and stable on macOS ARM64 without numerical matmul warnings.
    """
    return LogisticRegression(
        penalty='l2',
        C=C,
        class_weight=class_weight,
        solver='liblinear',
        max_iter=max_iter,
        random_state=random_state
    )


def build_weighted_xgboost(scale_pos_weight: float = 1.0,
                           n_estimators: int = 150,
                           max_depth: int = 4,
                           learning_rate: float = 0.05,
                           subsample: float = 0.8,
                           colsample_bytree: float = 0.8,
                           random_state: int = 42) -> XGBClassifier:
    """
    XGBoost classifier with scale_pos_weight for imbalanced outcomes.
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=random_state,
        verbosity=0,
        n_jobs=-1
    )


def build_weighted_random_forest(n_estimators: int = 100,
                                 max_depth: Optional[int] = 10,
                                 class_weight: str = 'balanced',
                                 random_state: int = 42) -> RandomForestClassifier:
    """
    Random Forest with class balancing.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1
    )


def compute_scale_pos_weight(y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """
    Compute XGBoost scale_pos_weight as weighted ratio of negative to positive cases.
    For unweighted: n_negative / n_positive
    For weighted: sum_weight_negative / sum_weight_positive
    """
    if w is None:
        w = np.ones_like(y, dtype=float)
    pos_mask = y == 1
    neg_mask = y == 0
    w_pos = w[pos_mask].sum()
    w_neg = w[neg_mask].sum()
    if w_pos == 0:
        return 1.0
    return float(w_neg / w_pos)


BASELINE_CONFIGS = {
    'logistic': build_weighted_logistic,
    'xgboost': build_weighted_xgboost,
    'random_forest': build_weighted_random_forest,
}

