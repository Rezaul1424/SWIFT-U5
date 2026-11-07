"""
Module B: Division-Shrunk Head via Empirical Bayes for SWIFT-U5.

Fits a division-specific intercept and slope on the model logit scale,
shrunk toward the robust national values by Empirical Bayes in proportion
to observed division death volume (survey-weighted):

  D_d     = weighted death count in division d
  lambda_d = D_d / (D_d + kappa)
  alpha_d* = lambda_d * alpha_d + (1 - lambda_d) * alpha_0
  beta_d*  = lambda_d * beta_d  + (1 - lambda_d) * beta_0

Weighted death counts are used because the study estimand is a survey-weighted
population quantity; raw counts would give excessive shrinkage in divisions
with high sampling probabilities.

Failed division fits fall back to national parameters and are **explicitly
logged** in ``self.fallback_log_`` rather than failing silently.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class ModuleBDivisionShrunkHead:
    """
    Empirical Bayes shrinkage head for regional division calibration.
    Prevents overfitting in low-death divisions while allowing high-death
    divisions to adapt to local regional risk gradients.
    """
    def __init__(self, shrinkage_kappa: float = 40.0):
        self.shrinkage_kappa = shrinkage_kappa
        self.national_intercept_: float = 0.0
        self.national_slope_: float = 1.0
        self.shrunk_params_: Dict[str, Tuple[float, float]] = {}
        self.shrinkage_factors_: Dict[str, float] = {}
        self.division_deaths_: Dict[str, float] = {}   # weighted death counts
        self.fallback_log_: Dict[str, str] = {}        # records every fallback
        self.is_fitted: bool = False

    def fit(self,
            y_true: np.ndarray,
            y_prob_base: np.ndarray,
            division_series: pd.Series,
            sample_weight: Optional[np.ndarray] = None) -> 'ModuleBDivisionShrunkHead':
        """
        Fit national and division-specific heads on model logits, then apply EB shrinkage.
        """
        eps = 1e-4
        y_prob_clipped = np.clip(y_prob_base, eps, 1.0 - eps)
        z_base = np.log(y_prob_clipped / (1.0 - y_prob_clipped)).reshape(-1, 1)

        if sample_weight is None:
            sample_weight = np.ones(len(y_true))
        w = sample_weight / sample_weight.mean()

        # 1. Fit National calibration head
        lr_nat = LogisticRegression(penalty='l2', C=10.0, solver='liblinear', max_iter=200)
        lr_nat.fit(z_base, y_true, sample_weight=w)
        self.national_slope_ = float(lr_nat.coef_[0][0])
        self.national_intercept_ = float(lr_nat.intercept_[0])

        # 2. Fit and shrink local division heads
        unique_divs = division_series.unique().tolist()
        self.shrunk_params_ = {}
        self.shrinkage_factors_ = {}
        self.division_deaths_ = {}

        for div in unique_divs:
            div_mask = (division_series == div).values
            # Survey-weighted death count (more appropriate for a survey-weighted estimand
            # than unweighted raw counts, which overweight high-sampling-fraction divisions)
            div_deaths_w = float((w[div_mask] * y_true[div_mask]).sum())
            div_deaths_raw = int(y_true[div_mask].sum())
            self.division_deaths_[div] = div_deaths_w

            if div_deaths_raw < 5 or div_mask.sum() < 30:
                # Insufficient events for stable division fit — fall back to national
                reason = (f"div_deaths_raw={div_deaths_raw} < 5 "
                          f"or n_div={div_mask.sum()} < 30")
                self.shrunk_params_[div] = (self.national_intercept_, self.national_slope_)
                self.shrinkage_factors_[div] = 0.0
                self.fallback_log_[div] = f"FALLBACK_TO_NATIONAL: {reason}"
                continue

            z_div = z_base[div_mask]
            y_div = y_true[div_mask]
            w_div = w[div_mask]

            try:
                lr_div = LogisticRegression(penalty='l2', C=5.0, solver='liblinear', max_iter=200)
                lr_div.fit(z_div, y_div, sample_weight=w_div)
                div_slope = float(lr_div.coef_[0][0])
                div_intercept = float(lr_div.intercept_[0])
            except Exception as exc:
                # Fit failed — fall back to national and log the reason explicitly
                self.fallback_log_[div] = f"FALLBACK_TO_NATIONAL: fit failed — {exc}"
                div_slope = self.national_slope_
                div_intercept = self.national_intercept_

            # Empirical Bayes shrinkage factor (uses weighted death count)
            # lambda_d -> 1 when deaths are high, -> 0 when deaths are scarce
            lambda_d = float(div_deaths_w / (div_deaths_w + self.shrinkage_kappa))
            self.shrinkage_factors_[div] = lambda_d

            alpha_shrunk = lambda_d * div_intercept + (1.0 - lambda_d) * self.national_intercept_
            beta_shrunk = lambda_d * div_slope + (1.0 - lambda_d) * self.national_slope_

            self.shrunk_params_[div] = (alpha_shrunk, beta_shrunk)

        self.is_fitted = True
        return self

    def predict_proba(self,
                      y_prob_base: np.ndarray,
                      division_series: pd.Series) -> np.ndarray:
        """
        Apply division-shrunk logit transformation to base probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Module B must be fitted before predict_proba!")

        eps = 1e-4
        y_prob_clipped = np.clip(y_prob_base, eps, 1.0 - eps)
        z_base = np.log(y_prob_clipped / (1.0 - y_prob_clipped))

        shrunk_probs = np.zeros(len(y_prob_base), dtype=float)

        for div in division_series.unique():
            div_mask = (division_series == div).values
            if div in self.shrunk_params_:
                alpha, beta = self.shrunk_params_[div]
            else:
                alpha, beta = self.national_intercept_, self.national_slope_

            z_adj = alpha + beta * z_base[div_mask]
            shrunk_probs[div_mask] = 1.0 / (1.0 + np.exp(-z_adj))

        return shrunk_probs

