"""
SWIFT-U5: Survey-Weighted, Importance-corrected, Forward-validated
Temporal Risk Model for Under-Five Mortality.

Unified framework combining:
  - Base Learner: Survey-aware calibrated gradient booster
  - Module A: Recency decay x Covariate shift density ratio weighting
  - Module B: Empirical Bayes Division-shrunk logit head
  - Module C: Drift-aware temporal recalibration
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.models.module_a_weight import ModuleARecencyShiftWeighter
from src.models.module_b_shrink import ModuleBDivisionShrunkHead
from src.models.module_c_recalib import ModuleCDriftRecalibrator
from src.features.build_features import LeakageFreePreprocessor, DIVISION_COLUMN, TARGET_COLUMN


class SWIFTU5Estimator:
    """
    SWIFT-U5 integrated risk prediction framework with full module ablation support.
    """
    def __init__(self,
                 use_module_a: bool = True,
                 use_module_b: bool = True,
                 use_module_c: bool = True,
                 decay_rate: float = 0.05,
                 shrinkage_kappa: float = 40.0,
                 n_estimators: int = 150,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 random_state: int = 42):
        self.use_module_a = use_module_a
        self.use_module_b = use_module_b
        self.use_module_c = use_module_c
        self.decay_rate = decay_rate
        self.shrinkage_kappa = shrinkage_kappa
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        # Component modules
        self.module_a = ModuleARecencyShiftWeighter(
            decay_rate=decay_rate,
            use_time_decay=use_module_a,
            use_density_ratio=use_module_a
        )
        self.module_b = ModuleBDivisionShrunkHead(shrinkage_kappa=shrinkage_kappa)
        self.module_c = ModuleCDriftRecalibrator(use_temporal_recalibration=use_module_c)

        self.preprocessor = LeakageFreePreprocessor(include_missing_flags=True)
        self.base_model_: Optional[XGBClassifier] = None
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def fit(self,
            df_train: pd.DataFrame,
            target_round: str,
            feature_subset: Optional[List[str]] = None,
            df_target_unlabeled: Optional[pd.DataFrame] = None) -> 'SWIFTU5Estimator':
        """
        Fit SWIFT-U5 on multi-round training cohort:
        1. Preprocess features using train-only statistics.
        2. Module A: Fit density ratio and compute composite sample weights.
        3. Fit base XGBoost learner with survey-shift weights.
        4. Module B: Fit Empirical Bayes division-shrunk logit head.
        5. Module C: Anchor temporal recalibration on the most recent training round.
        """
        y_train = df_train[TARGET_COLUMN].values.astype(int)
        div_train = df_train[DIVISION_COLUMN].astype(str)

        # 1. Feature Preprocessing
        X_train, self.feature_names_ = self.preprocessor.fit_transform(df_train, feature_subset)

        # 2. Module A: Recency + Shift Weighting
        if self.use_module_a:
            if df_target_unlabeled is not None and len(df_target_unlabeled) > 0:
                X_tgt, _ = self.preprocessor.transform(df_target_unlabeled, feature_subset)
                self.module_a.fit_density_ratio(X_train, X_tgt)
            w_train = self.module_a.compute_weights(df_train, target_round, X_train)
        else:
            # Baseline survey weight without time decay or density shift
            raw_w = df_train['sample_weight'].values.astype(float)
            w_train = raw_w / raw_w.mean()

        # 3. Base Model Fitting (XGBoost)
        spw = float((w_train[y_train == 0].sum()) / max(1.0, w_train[y_train == 1].sum()))
        self.base_model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric='logloss',
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1
        )
        self.base_model_.fit(X_train, y_train, sample_weight=w_train)

        # Base predicted probabilities on training data
        p_train_base = self.base_model_.predict_proba(X_train)[:, 1]

        # 4. Module B: Division-Shrunk Head
        if self.use_module_b:
            self.module_b.fit(y_train, p_train_base, div_train, sample_weight=w_train)

        # 5. Module C: Drift-Aware Recalibration (anchored on latest available train round)
        if self.use_module_c:
            train_rounds = sorted(df_train['round'].unique().tolist())
            latest_round = train_rounds[-1]
            latest_mask = (df_train['round'] == latest_round).values
            if latest_mask.sum() > 0:
                y_recent = y_train[latest_mask]
                p_recent = p_train_base[latest_mask]
                w_recent = w_train[latest_mask]
                self.module_c.fit_recalibration(y_recent, p_recent, w_recent, round_label=latest_round)

        self.is_fitted = True
        return self

    def predict_proba(self, df_test: pd.DataFrame, feature_subset: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate calibrated predictions for test observations through Modules A, B, C pipeline.
        """
        if not self.is_fitted:
            raise RuntimeError("SWIFT-U5 must be fitted before calling predict_proba!")

        X_test, _ = self.preprocessor.transform(df_test, feature_subset)
        div_test = df_test[DIVISION_COLUMN].astype(str)

        # Raw base learner predictions
        p = self.base_model_.predict_proba(X_test)[:, 1]

        # Apply Module B: Division-Shrunk Head
        if self.use_module_b:
            p = self.module_b.predict_proba(p, div_test)

        # Apply Module C: Drift-Aware Recalibration
        if self.use_module_c:
            p = self.module_c.predict_proba(p)

        return np.clip(p, 1e-4, 1.0 - 1e-4)

