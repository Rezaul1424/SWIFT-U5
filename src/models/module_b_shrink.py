"""
Module B: Division-Shrunk Head via Empirical Bayes for SWIFT-U5.

Fits a division-specific intercept and slope on the model logit scale,
shrunk toward the robust national values by Empirical Bayes in proportion
to observed division death volume:
  lambda_d = D_d / (D_d + kappa)
  alpha_d* = lambda_d * alpha_d + (1 - lambda_d) * alpha_0
  beta_d*  = lambda_d * beta_d  + (1 - lambda_d) * beta_0
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
        self.division_deaths_: Dict[str, int] = {}
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
            div_deaths = int(y_true[div_mask].sum())
            self.division_deaths_[div] = div_deaths

            if div_deaths < 5 or div_mask.sum() < 30:
                # Fall back to national parameters if virtually no deaths
                self.shrunk_params_[div] = (self.national_intercept_, self.national_slope_)
                self.shrinkage_factors_[div] = 0.0
                continue

            z_div = z_base[div_mask]
            y_div = y_true[div_mask]
            w_div = w[div_mask]

            try:
                lr_div = LogisticRegression(penalty='l2', C=5.0, solver='liblinear', max_iter=200)
                lr_div.fit(z_div, y_div, sample_weight=w_div)
                div_slope = float(lr_div.coef_[0][0])
                div_intercept = float(lr_div.intercept_[0])
            except Exception:
                div_slope = self.national_slope_
                div_intercept = self.national_intercept_

            # Empirical Bayes shrinkage factor
            # lambda_d -> 1 when deaths are high, -> 0 when deaths are scarce
            lambda_d = float(div_deaths / (div_deaths + self.shrinkage_kappa))
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

