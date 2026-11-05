"""
Module C: Drift-Aware Recalibration for SWIFT-U5.

Tracks temporal calibration drift across sequential survey rounds,
applies Platt recalibration anchored on the latest pre-deployment round,
and computes the SHAP Drift Index (SDI) across survey intervals.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class ModuleCDriftRecalibrator:
    """
    Recalibrates predicted risk probabilities to counteract secular mortality decline
    (temporal drift) using the most recent available pre-deployment round.
    """
    def __init__(self, use_temporal_recalibration: bool = True):
        self.use_temporal_recalibration = use_temporal_recalibration
        self.recalib_slope_: float = 1.0
        self.recalib_intercept_: float = 0.0
        self.historical_calibrations_: Dict[str, Tuple[float, float]] = {}
        self.is_fitted: bool = False

    def fit_recalibration(self,
                          y_true_recent: np.ndarray,
                          y_prob_recent: np.ndarray,
                          w_recent: Optional[np.ndarray] = None,
                          round_label: str = "recent") -> 'ModuleCDriftRecalibrator':
        """
        Fit Platt recalibration on the most recent available survey round.
        """
        if not self.use_temporal_recalibration or len(y_true_recent) == 0:
            self.is_fitted = True
            return self

        eps = 1e-4
        y_prob_clipped = np.clip(y_prob_recent, eps, 1.0 - eps)
        z_recent = np.log(y_prob_clipped / (1.0 - y_prob_clipped)).reshape(-1, 1)

        if w_recent is None:
            w = np.ones(len(y_true_recent))
        else:
            w = w_recent / w_recent.mean()

        try:
            lr = LogisticRegression(penalty='l2', C=10.0, solver='liblinear', max_iter=200)
            lr.fit(z_recent, y_true_recent, sample_weight=w)
            self.recalib_slope_ = float(lr.coef_[0][0])
            self.recalib_intercept_ = float(lr.intercept_[0])
        except Exception:
            self.recalib_slope_ = 1.0
            self.recalib_intercept_ = 0.0

        self.historical_calibrations_[round_label] = (self.recalib_intercept_, self.recalib_slope_)
        self.is_fitted = True
        return self

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Recalibrate predicted probabilities using fitted drift parameters.
        """
        if not self.is_fitted or not self.use_temporal_recalibration:
            return y_prob

        eps = 1e-4
        y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
        z = np.log(y_prob_clipped / (1.0 - y_prob_clipped))

        z_recalib = self.recalib_intercept_ + self.recalib_slope_ * z
        recalib_prob = 1.0 / (1.0 + np.exp(-z_recalib))
        return recalib_prob

    @staticmethod
    def compute_shap_drift_index(shap_values_prev: np.ndarray,
                                shap_values_curr: np.ndarray) -> float:
        """
        Compute SHAP Drift Index (SDI) across sequential rounds:
          SDI = (1 / K) * sum_k |mean(|phi_k, t|) - mean(|phi_k, t-1|)|
        """
        mean_abs_prev = np.mean(np.abs(shap_values_prev), axis=0)
        mean_abs_curr = np.mean(np.abs(shap_values_curr), axis=0)
        sdi = float(np.mean(np.abs(mean_abs_curr - mean_abs_prev)))
        return sdi

