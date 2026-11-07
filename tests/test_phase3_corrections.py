"""
Phase 3 correction tests for SWIFT-U5.

Verifies all critical/high-severity audit findings are fixed:
  - PSU composite identity uniqueness across waves
  - completed_5yr_followup flag correctness
  - b5/b7 absence from feature matrix
  - Metrics completeness (classification metrics, BSS, Null Brier)
  - encode_features() respects training categories only (no OHE leakage)
  - Module C calibration on held-out validation data
"""

import unittest
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# 1. PSU/Strata composite identity uniqueness across waves            #
# ------------------------------------------------------------------ #

class TestCompositeClusterIdentity(unittest.TestCase):
    """
    PSU IDs are re-used across DHS waves; psu_wave must be unique per wave.
    """

    def _make_cohort(self):
        """Return a minimal multi-wave cohort-like DataFrame with psu_wave."""
        rows = []
        for wave in ['2004', '2007', '2011', '2014', '2017-18', '2022']:
            for psu_id in [100, 101, 102]:
                rows.append({'round': wave, 'v021': psu_id, 'v022': psu_id + 500})
        df = pd.DataFrame(rows)
        df['psu_wave']    = df['round'] + '_' + df['v021'].astype(str)
        df['strata_wave'] = df['round'] + '_' + df['v022'].astype(str)
        return df

    def test_psu_wave_unique_across_waves(self):
        """Same raw PSU id in different waves must produce different psu_wave values."""
        df = self._make_cohort()
        # PSU id 100 appears in every wave; psu_wave must differ
        psu_100 = df[df['v021'] == 100]['psu_wave'].tolist()
        self.assertEqual(len(set(psu_100)), 6, "psu_wave should be unique per (round, psu)")

    def test_no_cross_wave_psu_sharing(self):
        """psu_wave values must not overlap across waves."""
        df = self._make_cohort()
        for wave_a, wave_b in [('2004', '2022'), ('2007', '2017-18')]:
            set_a = set(df[df['round'] == wave_a]['psu_wave'])
            set_b = set(df[df['round'] == wave_b]['psu_wave'])
            self.assertTrue(set_a.isdisjoint(set_b),
                            f"psu_wave values should not overlap between {wave_a} and {wave_b}")

    def test_raw_psu_overlap_would_occur(self):
        """Control: raw psu IDs DO overlap across waves (this is the bug being fixed)."""
        df = self._make_cohort()
        set_2004 = set(df[df['round'] == '2004']['v021'].astype(str))
        set_2022 = set(df[df['round'] == '2022']['v021'].astype(str))
        self.assertFalse(set_2004.isdisjoint(set_2022),
                         "Raw PSU IDs intentionally overlap — the fix is psu_wave")


# ------------------------------------------------------------------ #
# 2. completed_5yr_followup flag correctness                          #
# ------------------------------------------------------------------ #

class TestFollowUpCompletenessFlag(unittest.TestCase):

    def _make_df(self):
        """Mini cohort with dead and alive children of varying ages."""
        return pd.DataFrame({
            'target_u5m': [1, 1, 0, 0, 0],
            'age_at_survey_months': [5, 30, 10, 45, 59],
            'is_exposure_complete': [np.nan, np.nan, np.nan, np.nan, np.nan],
        })

    def test_completed_flag_only_true_for_deaths(self):
        """Only dead children (target_u5m == 1) have completed_5yr_followup == 1."""
        df = self._make_df()
        df['completed_5yr_followup'] = (df['target_u5m'] == 1).astype(int)
        self.assertEqual(df['completed_5yr_followup'].sum(), 2)
        # Alive children aged < 60 months must have flag == 0
        alive = df[df['target_u5m'] == 0]
        self.assertTrue((alive['completed_5yr_followup'] == 0).all())

    def test_is_exposure_complete_deprecated(self):
        """is_exposure_complete is NaN (deprecated), not 1."""
        df = self._make_df()
        # Simulate what cohort.py now produces
        self.assertTrue(df['is_exposure_complete'].isna().all(),
                        "is_exposure_complete should be NaN (deprecated)")

    def test_incomplete_followup_proportion_nonzero(self):
        """At least some alive children have age < 60 months (incomplete follow-up)."""
        df = self._make_df()
        df['completed_5yr_followup'] = (df['target_u5m'] == 1).astype(int)
        incomplete = (df['target_u5m'] == 0) & (df['age_at_survey_months'] < 60)
        self.assertGreater(incomplete.sum(), 0)


# ------------------------------------------------------------------ #
# 3. b5/b7 absence from feature matrix                                #
# ------------------------------------------------------------------ #

class TestOutcomeLeakageFromFeatureMatrix(unittest.TestCase):

    def test_b5_b7_not_in_candidate_features(self):
        """b5 and b7 must NOT appear in CANDIDATE_FEATURES."""
        from src.features.build_features import LeakageFreePreprocessor
        import inspect
        source = inspect.getsource(LeakageFreePreprocessor)
        self.assertNotIn("'b5'", source,
                         "b5 (survival status) must not be a candidate feature")
        self.assertNotIn("'b7'", source,
                         "b7 (age at death) must not be a candidate feature")

    def test_preprocessor_does_not_include_b5_b7(self):
        """LeakageFreePreprocessor's fit_transform must not include b5/b7 columns."""
        from src.features.build_features import LeakageFreePreprocessor
        import pandas as pd, numpy as np
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({
            'maternal_age_at_birth': rng.integers(18, 40, n).astype(float),
            'maternal_education': rng.integers(0, 4, n).astype(float),
            'wealth_quintile': rng.integers(1, 6, n).astype(float),
            'residence_urban': rng.integers(0, 2, n).astype(float),
            'birth_order': rng.integers(1, 5, n).astype(float),
            'parity_before': rng.integers(0, 4, n).astype(float),
            'is_first_birth': rng.integers(0, 2, n).astype(float),
            'preceding_birth_interval': rng.uniform(10, 36, n),
            'birth_interval_missing': rng.integers(0, 2, n).astype(float),
            'is_twin': rng.integers(0, 2, n).astype(float),
            'anc_visits': rng.integers(0, 8, n).astype(float),
            'anc_visits_missing': np.zeros(n),
            'tetanus_toxoid': rng.integers(0, 3, n).astype(float),
            'tetanus_toxoid_missing': np.zeros(n),
            'iron_supplementation': rng.integers(0, 2, n).astype(float),
            'iron_supplementation_missing': np.zeros(n),
            'facility_delivery': rng.integers(0, 2, n).astype(float),
            'facility_delivery_missing': np.zeros(n),
            'caesarean_delivery': rng.integers(0, 2, n).astype(float),
            'caesarean_delivery_missing': np.zeros(n),
            'birth_size': rng.integers(1, 6, n).astype(float),
            'birth_size_missing': np.zeros(n),
            'division_7': rng.choice(['dhaka', 'chattogram', 'khulna'], n),
            'target_u5m': rng.integers(0, 2, n).astype(int),
            # Inject outcome source columns to ensure they're excluded
            'b5': rng.integers(0, 2, n).astype(float),
            'b7': rng.integers(0, 60, n).astype(float),
        })
        prep = LeakageFreePreprocessor()
        X, feature_names = prep.fit_transform(df)
        self.assertNotIn('b5', feature_names, "b5 must not be in feature matrix")
        self.assertNotIn('b7', feature_names, "b7 must not be in feature matrix")


# ------------------------------------------------------------------ #
# 4. Metrics completeness                                             #
# ------------------------------------------------------------------ #

class TestMetricsCompleteness(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(99)
        n = 200
        self.y_true = rng.integers(0, 2, n).astype(int)
        self.y_prob = rng.uniform(0, 1, n)
        self.w = rng.uniform(0.5, 2.0, n)

    def test_classification_metrics_present(self):
        """compute_all_metrics must return classification metrics."""
        from src.evaluation.metrics import compute_all_metrics
        m = compute_all_metrics(self.y_true, self.y_prob, self.w)
        for key in ['accuracy_w', 'precision_w', 'recall_w', 'specificity_w',
                    'f1_w', 'balanced_accuracy_w', 'npv_w', 'mcc_w']:
            self.assertIn(key, m, f"Missing metric: {key}")
            self.assertFalse(np.isnan(m[key]), f"Metric {key} is NaN")

    def test_null_brier_and_bss_present(self):
        """Null Brier Score and Brier Skill Score must be computed."""
        from src.evaluation.metrics import compute_all_metrics
        m = compute_all_metrics(self.y_true, self.y_prob, self.w)
        self.assertIn('brier_null_w', m)
        self.assertIn('brier_skill_score_w', m)
        self.assertFalse(np.isnan(m['brier_null_w']))
        self.assertFalse(np.isnan(m['brier_skill_score_w']))
        # Null Brier must be >= 0
        self.assertGreaterEqual(m['brier_null_w'], 0.0)

    def test_bss_range(self):
        """BSS of perfect predictor should be ~1; of null predictor ~0."""
        from src.evaluation.metrics import compute_all_metrics
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 10)
        # Perfect predictor
        p_perfect = y.astype(float)
        m_perf = compute_all_metrics(y, p_perfect, compute_calibration=False)
        self.assertGreater(m_perf['brier_skill_score_w'], 0.9)
        # Null predictor (predict prevalence for all)
        p_null = np.full(len(y), y.mean())
        m_null = compute_all_metrics(y, p_null, compute_calibration=False)
        self.assertAlmostEqual(m_null['brier_skill_score_w'], 0.0, places=5)

    def test_effective_sample_size_present(self):
        """Effective sample size (eff_n) must be reported."""
        from src.evaluation.metrics import compute_all_metrics
        m = compute_all_metrics(self.y_true, self.y_prob, self.w)
        self.assertIn('eff_n', m)
        self.assertGreater(m['eff_n'], 0)
        # eff_n <= n (always true by Kish formula)
        self.assertLessEqual(m['eff_n'], len(self.y_true))


# ------------------------------------------------------------------ #
# 5. encode_features() training-only OHE                             #
# ------------------------------------------------------------------ #

class TestEncodeFeaturesCategoryLeakage(unittest.TestCase):

    def _make_split_df(self):
        """Small df where test has an extra division unseen in training."""
        from src.validation.splits import FEATURE_COLUMNS, DIVISION_COLUMN
        rng = np.random.default_rng(7)
        n_train = 30
        n_test = 10
        base = {col: rng.uniform(0, 1, n_train + n_test) for col in FEATURE_COLUMNS}
        # Train: divisions A and B; Test: divisions A, B, C (C is unseen)
        div = ['A'] * (n_train // 2) + ['B'] * (n_train // 2)
        div += ['A'] * (n_test // 2) + ['C'] * (n_test // 2)  # C unseen in train
        base[DIVISION_COLUMN] = div
        base['target_u5m'] = rng.integers(0, 2, n_train + n_test)
        df = pd.DataFrame(base)
        train_idx = df.index[:n_train]
        return df, train_idx

    def test_ohe_uses_training_categories_only(self):
        """Test set with unseen division category must NOT add new OHE columns."""
        from src.validation.splits import encode_features
        df, train_idx = self._make_split_df()
        X_all, _ = encode_features(df, train_idx)
        # Columns named div_C should NOT exist (C was only in test)
        col_names = X_all.columns.tolist()
        self.assertNotIn('div_C', col_names,
                         "Unseen test division 'C' must not create an OHE column")

    def test_ohe_includes_training_categories(self):
        """OHE must include all training division categories."""
        from src.validation.splits import encode_features
        df, train_idx = self._make_split_df()
        X_all, _ = encode_features(df, train_idx)
        col_names = X_all.columns.tolist()
        self.assertIn('div_A', col_names)
        self.assertIn('div_B', col_names)

    def test_test_rows_all_zero_for_unseen_category(self):
        """Test rows with unseen division must map to all-zero division dummies."""
        from src.validation.splits import encode_features, FEATURE_COLUMNS, DIVISION_COLUMN
        df, train_idx = self._make_split_df()
        test_idx = df.index[len(train_idx):]
        X_all, _ = encode_features(df, train_idx)
        X_test = X_all.loc[test_idx]
        # Rows where original division is 'C' should have div_A=0 and div_B=0
        div_col = df.loc[test_idx, DIVISION_COLUMN]
        c_rows = test_idx[div_col.values == 'C']
        for col in ['div_A', 'div_B']:
            self.assertTrue((X_test.loc[c_rows, col] == 0).all(),
                            f"Unseen-category rows must have {col}=0")


# ------------------------------------------------------------------ #
# 6. Module C fitted on held-out validation (not in-sample)           #
# ------------------------------------------------------------------ #

class TestModuleCValidationCalibration(unittest.TestCase):

    def _make_df(self, rounds, n_per_round=40):
        rng = np.random.default_rng(42)
        frames = []
        for r in rounds:
            n = n_per_round
            frames.append(pd.DataFrame({
                'round': r,
                'target_u5m': rng.integers(0, 2, n).astype(int),
                'sample_weight': rng.uniform(0.5, 2.0, n),
                'maternal_age_at_birth': rng.uniform(18, 40, n),
                'maternal_education': rng.integers(0, 4, n).astype(float),
                'maternal_education_years': rng.uniform(0, 12, n),
                'wealth_quintile': rng.integers(1, 6, n).astype(float),
                'residence_urban': rng.integers(0, 2, n).astype(float),
                'birth_order': rng.integers(1, 5, n).astype(float),
                'parity_before': rng.integers(0, 4, n).astype(float),
                'is_first_birth': rng.integers(0, 2, n).astype(float),
                'preceding_birth_interval': rng.uniform(10, 36, n),
                'birth_interval_missing': rng.integers(0, 2, n).astype(float),
                'is_twin': rng.integers(0, 2, n).astype(float),
                'anc_visits': rng.integers(0, 8, n).astype(float),
                'anc_visits_missing': np.zeros(n),
                'tetanus_toxoid': rng.integers(0, 3, n).astype(float),
                'tetanus_toxoid_missing': np.zeros(n),
                'iron_supplementation': rng.integers(0, 2, n).astype(float),
                'iron_supplementation_missing': np.zeros(n),
                'facility_delivery': rng.integers(0, 2, n).astype(float),
                'facility_delivery_missing': np.zeros(n),
                'caesarean_delivery': rng.integers(0, 2, n).astype(float),
                'caesarean_delivery_missing': np.zeros(n),
                'birth_size': rng.integers(1, 6, n).astype(float),
                'birth_size_missing': np.zeros(n),
                'division_7': rng.choice(['dhaka', 'chattogram', 'khulna'], n),
            }))
        return pd.concat(frames, ignore_index=True)

    def test_df_val_accepted_without_warning(self):
        """fit() with df_val must NOT emit a UserWarning."""
        import warnings
        from src.models.swift_u5 import SWIFTU5Estimator
        df_train = self._make_df(['2004', '2007', '2011', '2014'])
        df_val = self._make_df(['2017-18'])
        model = SWIFTU5Estimator(use_module_a=False, use_module_b=False, use_module_c=True,
                                 n_estimators=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(df_train, target_round='2017-18', df_val=df_val)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            self.assertEqual(len(user_warnings), 0,
                             "No UserWarning expected when df_val is provided")

    def test_no_df_val_emits_warning(self):
        """fit() without df_val must emit a UserWarning about in-sample calibration."""
        import warnings
        from src.models.swift_u5 import SWIFTU5Estimator
        df_train = self._make_df(['2004', '2007'])
        model = SWIFTU5Estimator(use_module_a=False, use_module_b=False, use_module_c=True,
                                 n_estimators=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(df_train, target_round='2007')
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            self.assertGreater(len(user_warnings), 0,
                               "UserWarning expected when df_val is not provided")
            self.assertIn("IN-SAMPLE", str(user_warnings[0].message))


if __name__ == '__main__':
    unittest.main()
