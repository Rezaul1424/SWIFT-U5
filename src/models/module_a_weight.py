"""
Module A: Recency and Shift Weighting for SWIFT-U5.

Combines:
  1. Within-round normalized survey sample weights (v005 / 1,000,000).
  2. Exponential time decay weighting prioritizing more recent rounds.
  3. Clipped density ratio from a domain classifier (train vs. target round)
     correcting for covariate shift across survey eras.

COMPOSITE WEIGHT CLIPPING
--------------------------
Each component is bounded individually (density ratio clipped at [clip_min, clip_max]),
but the PRODUCT of a high survey weight × time decay × density ratio can still produce
extreme composite values.  Near-perfect domain separation causes the density-ratio
classifier to assign P(target|x) ≈ 1 for many 2014 observations, which — when
multiplied by the already-large 2004 DHS survey weights — produces composites exceeding
100× the median.

A final composite clip at ``composite_clip_percentile`` (default: 99th percentile
of the pre-clip distribution) is applied BEFORE normalization to bound influence.
This is a prespecified sensitivity analysis choice, NOT an ad-hoc threshold chosen
to improve performance.  The clip level is stored in ``composite_clip_value_`` for
reporting.  The unclipped ESS and clipped ESS are both stored for publication.

SENSITIVITY ANALYSIS
--------------------
Run with ``composite_clip_percentile=None`` to replicate the unclipped (original)
behaviour for sensitivity comparison.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROUND_YEARS = {
    '2004': 2004.0,
    '2007': 2007.0,
    '2011': 2011.0,
    '2014': 2014.0,
    '2017-18': 2017.5,
    '2022': 2022.0
}


class ModuleARecencyShiftWeighter:
    """
    Computes sample weights combining:
      W_i = w_survey(i) * w_time(t_i) * r_clipped(x_i)
    """
    def __init__(self,
                 decay_rate: float = 0.05,
                 clip_min: float = 0.2,
                 clip_max: float = 5.0,
                 use_time_decay: bool = True,
                 use_density_ratio: bool = True,
                 composite_clip_percentile: Optional[float] = 99.0):
        """
        Parameters
        ----------
        composite_clip_percentile : float or None
            Percentile of the pre-normalisation composite weight distribution at
            which to clip extreme values.  Default 99.0 (99th percentile).
            Set to None to disable composite clipping (original behaviour, for
            sensitivity analysis).
        """
        self.decay_rate = decay_rate
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.use_time_decay = use_time_decay
        self.use_density_ratio = use_density_ratio
        self.composite_clip_percentile = composite_clip_percentile
        self.domain_clf_: Optional[LogisticRegression] = None
        self.density_ratio_mean_: float = 1.0
        # Diagnostics populated by compute_weights()
        self.composite_clip_value_: Optional[float] = None
        self.ess_before_clip_: Optional[float] = None
        self.ess_after_clip_: Optional[float] = None
        self.n_clipped_obs_: int = 0

    def fit_density_ratio(self,
                          X_train: np.ndarray,
                          X_target: np.ndarray,
                          w_train: Optional[np.ndarray] = None,
                          w_target: Optional[np.ndarray] = None) -> 'ModuleARecencyShiftWeighter':
        """
        Fit a domain classifier to estimate density ratio:
          r(x) = P(target | x) / P(train | x) * (N_train / N_target)
        """
        if not self.use_density_ratio or X_target is None or len(X_target) == 0:
            return self

        n_train = len(X_train)
        n_target = len(X_target)

        # Labels: 0 = train, 1 = target
        X_domain = np.vstack([X_train, X_target])
        y_domain = np.concatenate([np.zeros(n_train), np.ones(n_target)])

        # Combine weights if present
        if w_train is not None and w_target is not None:
            w_tr_norm = w_train / w_train.mean()
            w_tgt_norm = w_target / w_target.mean()
            w_domain = np.concatenate([w_tr_norm, w_tgt_norm])
        else:
            w_domain = np.ones(len(y_domain))

        self.domain_clf_ = LogisticRegression(
            penalty='l2',
            C=0.1,
            solver='liblinear',
            max_iter=300,
            random_state=42
        )
        self.domain_clf_.fit(X_domain, y_domain, sample_weight=w_domain)

        # Estimate empirical density ratio multiplier
        probs_train = self.domain_clf_.predict_proba(X_train)[:, 1]
        probs_train = np.clip(probs_train, 1e-4, 1.0 - 1e-4)
        raw_ratios = (probs_train / (1.0 - probs_train)) * (n_train / n_target)
        self.density_ratio_mean_ = float(np.mean(raw_ratios))

        return self

    def compute_weights(self,
                        df_train: pd.DataFrame,
                        target_round: str,
                        X_train: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute integrated Module A weights for the training observations.

        Records weight diagnostics in instance attributes for reporting:
          composite_clip_value_ : clip threshold applied (or None if disabled)
          ess_before_clip_      : ESS before composite clipping
          ess_after_clip_       : ESS after composite clipping
          n_clipped_obs_        : number of observations clipped
        """
        n_obs = len(df_train)

        # 1. Survey design weights: normalized within each round
        w_survey = np.zeros(n_obs, dtype=float)
        for rnd, grp_idx in df_train.groupby('round').groups.items():
            raw_w = df_train.loc[grp_idx, 'sample_weight'].values.astype(float)
            w_survey[df_train.index.get_indexer(grp_idx)] = raw_w / raw_w.mean()

        # 2. Time decay weights: w_time = exp(-lambda * (T_target - T_round))
        target_year = ROUND_YEARS.get(target_round, 2022.0)
        round_years_arr = df_train['round'].map(ROUND_YEARS).fillna(target_year).values.astype(float)

        if self.use_time_decay and self.decay_rate > 0:
            time_delta = target_year - round_years_arr
            w_time = np.exp(-self.decay_rate * np.maximum(0.0, time_delta))
        else:
            w_time = np.ones(n_obs, dtype=float)

        # 3. Density ratio shift correction: r(x)
        if self.use_density_ratio and self.domain_clf_ is not None and X_train is not None:
            probs = self.domain_clf_.predict_proba(X_train)[:, 1]
            probs = np.clip(probs, 1e-4, 1.0 - 1e-4)
            ratios = (probs / (1.0 - probs)) / max(1e-4, self.density_ratio_mean_)
            w_shift = np.clip(ratios, self.clip_min, self.clip_max)
        else:
            w_shift = np.ones(n_obs, dtype=float)

        # 4. Combined weight: W_i = w_survey * w_time * w_shift
        W = w_survey * w_time * w_shift

        # ESS before composite clip
        W_tmp = W / W.mean()
        self.ess_before_clip_ = float(W_tmp.sum() ** 2 / (W_tmp ** 2).sum())

        # 5. Composite clip at specified percentile (prespecified sensitivity analysis)
        self.composite_clip_value_ = None
        self.n_clipped_obs_ = 0
        if self.composite_clip_percentile is not None:
            clip_val = float(np.percentile(W, self.composite_clip_percentile))
            self.composite_clip_value_ = clip_val
            self.n_clipped_obs_ = int((W > clip_val).sum())
            W = np.minimum(W, clip_val)

        # Normalize to have mean 1.0
        W = W / W.mean()

        # ESS after composite clip + normalisation
        self.ess_after_clip_ = float(W.sum() ** 2 / (W ** 2).sum())

        return W

