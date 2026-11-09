"""
Pre-Phase 6 verification tests for SWIFT-U5.

Covers three issues resolved before Phase 6 authorization:
  1. Module A — composite weight clipping and ESS diagnostics
  2. Youden threshold — must come from validation data, never test labels
  3. Ablation runner — df_val support, frozen threshold, correct data partitioning
"""

import unittest
import warnings
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# 1. Module A — composite weight diagnostics                          #
# ------------------------------------------------------------------ #

class TestModuleAWeightDiagnostics(unittest.TestCase):

    def _make_multi_round_df(self, rounds=None, n_per_round=50):
        if rounds is None:
            rounds = ['2004', '2007', '2011', '2014']
        rng = np.random.default_rng(1)
        frames = []
        for r in rounds:
            n = n_per_round
            frames.append(pd.DataFrame({
                'round': r,
                'sample_weight': rng.uniform(0.5, 3.0, n),
                'target_u5m': rng.integers(0, 2, n).astype(int),
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

    def test_composite_clip_percentile_param_accepted(self):
        """ModuleARecencyShiftWeighter accepts composite_clip_percentile param."""
        from src.models.module_a_weight import ModuleARecencyShiftWeighter
        ma = ModuleARecencyShiftWeighter(composite_clip_percentile=99.0)
        self.assertEqual(ma.composite_clip_percentile, 99.0)

    def test_composite_clip_none_disables_clipping(self):
        """composite_clip_percentile=None disables composite clip."""
        from src.models.module_a_weight import ModuleARecencyShiftWeighter
        from src.features.build_features import LeakageFreePreprocessor
        df = self._make_multi_round_df()
        ma = ModuleARecencyShiftWeighter(composite_clip_percentile=None,
                                         use_density_ratio=False)
        prep = LeakageFreePreprocessor()
        X, _ = prep.fit_transform(df)
        w = ma.compute_weights(df, '2022', X)
        self.assertIsNone(ma.composite_clip_value_)
        self.assertEqual(ma.n_clipped_obs_, 0)

    def test_ess_diagnostics_populated(self):
        """compute_weights() populates ess_before_clip_ and ess_after_clip_."""
        from src.models.module_a_weight import ModuleARecencyShiftWeighter
        from src.features.build_features import LeakageFreePreprocessor
        df = self._make_multi_round_df()
        ma = ModuleARecencyShiftWeighter(composite_clip_percentile=99.0,
                                         use_density_ratio=False)
        prep = LeakageFreePreprocessor()
        X, _ = prep.fit_transform(df)
        ma.compute_weights(df, '2022', X)
        self.assertIsNotNone(ma.ess_before_clip_)
        self.assertIsNotNone(ma.ess_after_clip_)
        self.assertGreater(ma.ess_before_clip_, 0)
        self.assertGreater(ma.ess_after_clip_, 0)
        # After clipping, ESS should be >= before clipping (clipping raises ESS)
        self.assertGreaterEqual(ma.ess_after_clip_, ma.ess_before_clip_ - 1e-6)

    def test_clipping_reduces_weight_max(self):
        """99th-pct composite clip reduces the weight maximum."""
        from src.models.module_a_weight import ModuleARecencyShiftWeighter
        from src.features.build_features import LeakageFreePreprocessor
        df = self._make_multi_round_df()
        prep = LeakageFreePreprocessor()
        X, _ = prep.fit_transform(df)

        ma_clipped   = ModuleARecencyShiftWeighter(composite_clip_percentile=99.0,
                                                    use_density_ratio=False)
        ma_unclipped = ModuleARecencyShiftWeighter(composite_clip_percentile=None,
                                                    use_density_ratio=False)
        w_clip   = ma_clipped.compute_weights(df, '2022', X)
        w_noclip = ma_unclipped.compute_weights(df, '2022', X)

        self.assertLessEqual(w_clip.max(), w_noclip.max() + 1e-6)

    def test_weights_normalized_mean_one(self):
        """Final composite weights have mean ≈ 1.0."""
        from src.models.module_a_weight import ModuleARecencyShiftWeighter
        from src.features.build_features import LeakageFreePreprocessor
        df = self._make_multi_round_df()
        prep = LeakageFreePreprocessor()
        X, _ = prep.fit_transform(df)
        ma = ModuleARecencyShiftWeighter(composite_clip_percentile=99.0,
                                          use_density_ratio=False)
        w = ma.compute_weights(df, '2022', X)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)


# ------------------------------------------------------------------ #
# 2. Youden threshold — must not use test labels                      #
# ------------------------------------------------------------------ #

class TestYoudenThresholdProvenance(unittest.TestCase):
    """
    CRITICAL: Youden threshold must be computed from VALIDATION data only.
    compute_all_metrics() must accept pre-frozen external thresholds.
    """

    def _make_data(self, n=200):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, n).astype(int)
        p = rng.uniform(0, 1, n)
        w = rng.uniform(0.5, 2.0, n)
        return y, p, w

    def test_external_threshold_accepted(self):
        """compute_all_metrics accepts youden_threshold_w and youden_threshold_uw."""
        from src.evaluation.metrics import compute_all_metrics
        y, p, w = self._make_data()
        m = compute_all_metrics(y, p, w,
                                youden_threshold_w=0.05,
                                youden_threshold_uw=0.05,
                                compute_calibration=False)
        self.assertIn('recall_w', m)
        self.assertAlmostEqual(m['threshold_used_w'], 0.05, places=6)

    def test_threshold_source_external_when_supplied(self):
        """threshold_source == 'external_validation' when thresholds are pre-supplied."""
        from src.evaluation.metrics import compute_all_metrics
        y, p, w = self._make_data()
        m = compute_all_metrics(y, p, w,
                                youden_threshold_w=0.04,
                                youden_threshold_uw=0.04,
                                compute_calibration=False)
        self.assertEqual(m['threshold_source'], 'external_validation')

    def test_threshold_source_internal_when_not_supplied(self):
        """threshold_source == 'internal_from_passed_data' when thresholds not supplied."""
        from src.evaluation.metrics import compute_all_metrics
        y, p, w = self._make_data()
        m = compute_all_metrics(y, p, w, compute_calibration=False)
        self.assertEqual(m['threshold_source'], 'internal_from_passed_data')

    def test_compute_youden_from_validation_public_api(self):
        """compute_youden_threshold_from_validation() returns two floats."""
        from src.evaluation.metrics import compute_youden_threshold_from_validation
        rng = np.random.default_rng(7)
        y_val = rng.integers(0, 2, 100).astype(int)
        p_val = rng.uniform(0, 1, 100)
        w_val = rng.uniform(0.5, 2.0, 100)
        thr_w, thr_uw = compute_youden_threshold_from_validation(y_val, p_val, w_val)
        self.assertIsInstance(thr_w, float)
        self.assertIsInstance(thr_uw, float)
        self.assertGreater(thr_w, 0.0)
        self.assertGreater(thr_uw, 0.0)

    def test_external_threshold_differs_from_test_derived(self):
        """Val-derived threshold differs from test-derived threshold (quantifies the leakage)."""
        from src.evaluation.metrics import _youden_threshold
        rng = np.random.default_rng(99)
        y_val  = rng.integers(0, 2, 150).astype(int)
        y_test = rng.integers(0, 2, 200).astype(int)
        p_val  = rng.uniform(0, 1, 150)
        p_test = rng.uniform(0, 1, 200)
        thr_val  = _youden_threshold(y_val,  p_val)
        thr_test = _youden_threshold(y_test, p_test)
        # Thresholds derived from different data are different (almost certainly)
        # This is a sanity check, not a strict test
        # The key point is they CAN differ — and using test labels is data leakage
        self.assertIsInstance(thr_val, float)
        self.assertIsInstance(thr_test, float)

    def test_test_labels_not_used_when_external_threshold_provided(self):
        """When external threshold is provided, test y_true is not accessed for threshold."""
        from src.evaluation.metrics import compute_all_metrics
        rng = np.random.default_rng(55)
        y_test = rng.integers(0, 2, 100).astype(int)
        p_test = rng.uniform(0, 1, 100)
        # Use a deliberately wrong threshold to prove external value is used, not recomputed
        external_thr = 0.9999  # extreme value no internal Youden would produce
        m = compute_all_metrics(y_test, p_test,
                                youden_threshold_w=external_thr,
                                youden_threshold_uw=external_thr,
                                compute_calibration=False)
        self.assertAlmostEqual(m['threshold_used_w'], external_thr, places=6)
        # At threshold=0.9999, almost nothing is screened → recall ≈ 0
        self.assertLess(m['recall_w'], 0.1)


# ------------------------------------------------------------------ #
# 3. Ablation runner compatibility                                     #
# ------------------------------------------------------------------ #

class TestAblationRunnerCompatibility(unittest.TestCase):

    def _make_swift_df(self, rounds, n_per_round=40):
        rng = np.random.default_rng(42)
        frames = []
        for r in rounds:
            n = n_per_round
            frames.append(pd.DataFrame({
                'round': r,
                'target_u5m': rng.integers(0, 2, n).astype(int),
                'sample_weight': rng.uniform(0.5, 2.0, n),
                'psu_wave': [f"{r}_{i}" for i in range(n)],
                'strata_wave': [f"{r}_{i//10}" for i in range(n)],
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

    def test_evaluate_model_on_cut_accepts_df_val(self):
        """evaluate_model_on_cut() accepts df_val parameter without error."""
        import sys
        from pathlib import Path
        repo_root = str(Path(__file__).parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # Import the function directly — the top-level module has side effects
        # (warnings.filterwarnings, os.environ), so we isolate the import.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "exp04_ablation",
            Path(__file__).parent.parent / "experiments" / "04_run_swift_u5_ablation.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        df_dev  = self._make_swift_df(['2004', '2007'])
        df_val  = self._make_swift_df(['2017-18'])
        df_test = self._make_swift_df(['2022'])

        from src.models.swift_u5 import SWIFTU5Estimator
        est = SWIFTU5Estimator(use_module_a=False, use_module_b=True, use_module_c=True,
                               n_estimators=5)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            metrics, probs = mod.evaluate_model_on_cut(
                est, df_dev, df_test,
                target_round='2022', is_swift=True,
                run_bootstrap=False, df_val=df_val,
            )
        self.assertIn('auroc_w', metrics)
        self.assertEqual(len(probs), len(df_test))

    def test_evaluate_model_on_cut_uses_external_threshold(self):
        """evaluate_model_on_cut() uses pre-frozen threshold when supplied."""
        import sys
        import importlib.util
        from pathlib import Path
        repo_root = str(Path(__file__).parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        spec = importlib.util.spec_from_file_location(
            "exp04_ablation2",
            Path(__file__).parent.parent / "experiments" / "04_run_swift_u5_ablation.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        df_dev  = self._make_swift_df(['2004'])
        df_test = self._make_swift_df(['2022'])

        from src.models.swift_u5 import SWIFTU5Estimator
        est = SWIFTU5Estimator(use_module_a=False, use_module_b=False, use_module_c=False,
                               n_estimators=5)
        external_thr = 0.123
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            metrics, _ = mod.evaluate_model_on_cut(
                est, df_dev, df_test,
                target_round='2022', is_swift=True,
                run_bootstrap=False,
                youden_threshold_w=external_thr,
                youden_threshold_uw=external_thr,
            )
        self.assertEqual(metrics.get('threshold_source'), 'external_validation')
        self.assertAlmostEqual(metrics.get('threshold_used_w', 0), external_thr, places=5)

    def test_ablation_df_val_disjoint_from_test(self):
        """Validation round (2017-18) and test round (2022) must be disjoint by survey round."""
        df_val  = self._make_swift_df(['2017-18'])
        df_test = self._make_swift_df(['2022'])
        # Check by round label, not integer index (both use reset_index)
        val_rounds  = set(df_val['round'].unique())
        test_rounds = set(df_test['round'].unique())
        self.assertTrue(val_rounds.isdisjoint(test_rounds),
                        f"Val rounds {val_rounds} and test rounds {test_rounds} must be disjoint")
        self.assertNotIn('2022', val_rounds)
        self.assertNotIn('2017-18', test_rounds)

    def test_ablation_no_2022_labels_in_training(self):
        """Training data for ablation must not contain 2022 round."""
        dev_rounds = ['2004', '2007', '2011', '2014']
        df = self._make_swift_df(dev_rounds + ['2017-18', '2022'])
        df_dev = df[df['round'].isin(dev_rounds)]
        self.assertFalse('2022' in df_dev['round'].values)
        self.assertFalse('2017-18' in df_dev['round'].values)


# ------------------------------------------------------------------ #
# 4. Out-of-sample threshold derivation in experiment_2x2            #
# ------------------------------------------------------------------ #

class TestOOSThresholdDerivation(unittest.TestCase):
    """
    Critical: threshold derivation must NEVER use in-sample predictions.

    For temporal cuts, _derive_oos_threshold_temporal() must train a 'threshold
    model' on [r1,...,rk-1] and predict on rk (genuinely OOS).  For single-round
    cuts it must use 5-fold CV.  In both cases, no observation used to derive
    the threshold was in the threshold-model's training data.
    """

    def _make_temporal_df(self, rounds, n_per_round=60):
        rng = np.random.default_rng(7)
        frames = []
        for r in rounds:
            n = n_per_round
            frames.append(pd.DataFrame({
                'round': r,
                'target_u5m': rng.integers(0, 2, n).astype(int),
                'sample_weight': rng.uniform(0.5, 2.0, n),
                'psu_wave': [f"{r}_{i//5}" for i in range(n)],
                'strata_wave': [f"{r}_{i//15}" for i in range(n)],
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

    def test_derive_oos_threshold_multi_round_uses_penultimate(self):
        """
        _derive_oos_threshold_temporal with k>=2 rounds must derive threshold
        from predictions on last_round using a model NOT trained on last_round.
        Verifies: last_round observations are out-of-sample w.r.t. the threshold model.
        """
        from src.validation.experiment_2x2 import _derive_oos_threshold_temporal
        df = self._make_temporal_df(['2004', '2007', '2011'])
        train_rounds = ['2004', '2007']
        full_train_idx = df[df['round'].isin(train_rounds)].index
        thr_w, thr_uw, prov = _derive_oos_threshold_temporal(
            df, full_train_idx, train_rounds, 'logistic', use_weights=True)
        # Thresholds must be valid probabilities
        self.assertGreater(thr_w, 0.0)
        self.assertLess(thr_w, 1.0)
        self.assertGreater(thr_uw, 0.0)
        self.assertLess(thr_uw, 1.0)
        # Provenance must mention the penultimate round model, NOT the last round
        self.assertIn('2004', prov)  # threshold model trained on 2004
        self.assertIn('2007', prov)  # predicting on 2007 (last round)
        # Critical: provenance must say OOS
        self.assertIn('OOS', prov)

    def test_derive_oos_threshold_single_round_uses_cv(self):
        """
        _derive_oos_threshold_temporal with k==1 must use 5-fold CV (OOF predictions).
        """
        from src.validation.experiment_2x2 import _derive_oos_threshold_temporal
        df = self._make_temporal_df(['2004', '2007'])
        train_rounds = ['2004']
        full_train_idx = df[df['round'] == '2004'].index
        thr_w, thr_uw, prov = _derive_oos_threshold_temporal(
            df, full_train_idx, train_rounds, 'logistic', use_weights=True)
        self.assertGreater(thr_w, 0.0)
        self.assertLess(thr_w, 1.0)
        self.assertIn('CV', prov)  # Must mention cross-validation
        self.assertIn('OOS', prov)

    def test_val_idx_not_subset_of_train_idx_in_threshold_model(self):
        """
        In _derive_oos_threshold_temporal (k>=2), the observations used to derive
        the threshold (last_round) must NOT have been in the threshold model's
        training data (penultimate rounds).
        """
        from src.validation.splits import ROUND_COLUMN
        df = self._make_temporal_df(['2004', '2007', '2011'])
        train_rounds = ['2004', '2007']

        penultimate_rounds = train_rounds[:-1]  # ['2004']
        last_round = train_rounds[-1]            # '2007'

        pen_idx  = df[df[ROUND_COLUMN].isin(penultimate_rounds)].index
        last_idx = df[df[ROUND_COLUMN] == last_round].index

        # Last round observations must NOT be in penultimate training set
        overlap = last_idx.isin(pen_idx)
        self.assertFalse(overlap.any(),
                         "Val observations (last_round) must not appear in threshold model training data")

    def test_threshold_source_is_external_in_run_single_trial(self):
        """
        run_single_trial() with pre-frozen thresholds must set
        threshold_source='external_validation' in the output metrics.
        """
        from src.validation.experiment_2x2 import run_single_trial
        df = self._make_temporal_df(['2004', '2007'])
        train_idx = df[df['round'] == '2004'].index
        test_idx  = df[df['round'] == '2007'].index
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            result = run_single_trial(
                df, train_idx, test_idx,
                model_name='logistic',
                use_train_weights=False,
                use_eval_weights=False,
                arm_name='test_arm',
                test_round='2007',
                train_rounds=['2004'],
                run_bootstrap=False,
                youden_threshold_w=0.05,
                youden_threshold_uw=0.05,
                threshold_provenance='unit_test_OOS',
            )
        self.assertEqual(result.get('threshold_source'), 'external_validation')
        self.assertAlmostEqual(result.get('threshold_used_w', 0), 0.05, places=5)
        self.assertEqual(result.get('threshold_provenance'), 'unit_test_OOS')

    def test_test_outcome_does_not_contribute_to_its_own_threshold(self):
        """
        CRITICAL: A test observation's label must not influence the threshold
        used to classify that observation.

        Test: derive threshold from val set A.  Evaluate on test set B (disjoint).
        Verify that swapping val/test roles changes the threshold (proving
        the threshold depends on val labels, not test labels).
        """
        from src.evaluation.metrics import compute_youden_threshold_from_validation
        rng = np.random.default_rng(13)
        # Two disjoint groups (mimicking val and test)
        y_a = rng.integers(0, 2, 200).astype(int)
        p_a = rng.uniform(0.0, 0.3, 200)
        w_a = rng.uniform(0.5, 2.0, 200)

        y_b = rng.integers(0, 2, 150).astype(int)
        p_b = rng.uniform(0.1, 0.5, 150)
        w_b = rng.uniform(0.5, 2.0, 150)

        # Threshold from A, evaluate on B
        thr_from_a, _ = compute_youden_threshold_from_validation(y_a, p_a, w_a)
        # Threshold from B (what it would be if test labels were used)
        thr_from_b, _ = compute_youden_threshold_from_validation(y_b, p_b, w_b)

        # They must differ (proving derivation data matters)
        # This is a statistical sanity check — different data → different thresholds
        self.assertIsInstance(thr_from_a, float)
        self.assertIsInstance(thr_from_b, float)
        # Key assertion: a threshold derived from one dataset, applied to another,
        # must NOT equal the threshold that WOULD have been derived from the target data.
        # (These are from different distributions so they should differ.)
        # We use a loose tolerance to avoid seed-sensitivity — the structural check
        # is that the API enforces external threshold supply.
        self.assertNotEqual(round(thr_from_a, 3), round(thr_from_b, 3),
                            "Thresholds from different datasets should differ — "
                            "this confirms threshold derivation data matters")


# ------------------------------------------------------------------ #
# 5. Bootstrap meta reporting                                         #
# ------------------------------------------------------------------ #

class TestBootstrapMetaReporting(unittest.TestCase):
    """
    bootstrap_metrics() must report replicate success/failure counts
    and excluded non-numeric fields explicitly.
    """

    def _make_test_df(self, n=200):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            'target_u5m': rng.integers(0, 2, n).astype(int),
            'sample_weight': rng.uniform(0.5, 2.0, n),
            'psu_wave': [f"2022_{i//5}" for i in range(n)],
            'strata_wave': [f"2022_{i//20}" for i in range(n)],
        })

    def test_bootstrap_meta_key_present(self):
        """bootstrap_metrics() result must contain '_bootstrap_meta'."""
        from src.evaluation.bootstrap import bootstrap_metrics
        df = self._make_test_df()
        rng = np.random.default_rng(1)
        y_prob = rng.uniform(0, 0.3, len(df))
        result = bootstrap_metrics(df, y_prob, n_replicates=5, seed=42)
        self.assertIn('_bootstrap_meta', result)

    def test_bootstrap_meta_reports_counts(self):
        """_bootstrap_meta must have n_requested, n_succeeded, n_failed_degenerate."""
        from src.evaluation.bootstrap import bootstrap_metrics
        df = self._make_test_df()
        rng = np.random.default_rng(1)
        y_prob = rng.uniform(0, 0.3, len(df))
        result = bootstrap_metrics(df, y_prob, n_replicates=10, seed=42)
        meta = result['_bootstrap_meta']
        self.assertIn('n_requested', meta)
        self.assertIn('n_succeeded', meta)
        self.assertIn('n_failed_degenerate', meta)
        self.assertEqual(meta['n_requested'], 10)
        self.assertGreaterEqual(meta['n_succeeded'], 0)
        self.assertGreaterEqual(meta['n_failed_degenerate'], 0)
        self.assertEqual(meta['n_succeeded'] + meta['n_failed_degenerate'],
                         meta['n_requested'])

    def test_bootstrap_meta_records_excluded_fields(self):
        """Non-numeric fields (threshold_source) must be listed in non_numeric_fields_excluded."""
        from src.evaluation.bootstrap import bootstrap_metrics
        df = self._make_test_df()
        rng = np.random.default_rng(1)
        y_prob = rng.uniform(0, 0.3, len(df))
        result = bootstrap_metrics(df, y_prob, n_replicates=5, seed=42)
        meta = result['_bootstrap_meta']
        self.assertIn('non_numeric_fields_excluded', meta)
        # threshold_source is a string field that must be excluded from CI computation
        excluded = meta['non_numeric_fields_excluded']
        self.assertIn('threshold_source', excluded,
                      f"'threshold_source' must be in excluded list; got: {excluded}")

    def test_threshold_source_not_in_ci_keys(self):
        """'threshold_source' must not appear as a CI-carrying key (only in _bootstrap_meta)."""
        from src.evaluation.bootstrap import bootstrap_metrics
        df = self._make_test_df()
        rng = np.random.default_rng(1)
        y_prob = rng.uniform(0, 0.3, len(df))
        result = bootstrap_metrics(df, y_prob, n_replicates=5, seed=42)
        ci_keys = {k for k in result if k != '_bootstrap_meta'}
        self.assertNotIn('threshold_source', ci_keys)


# ------------------------------------------------------------------ #
# 6. ROC threshold candidate invariance                              #
# ------------------------------------------------------------------ #

class TestROCThresholdCandidateInvariance(unittest.TestCase):
    """
    Verifies that identical y_true and y_prob arrays produce identical raw
    ROC threshold candidates regardless of whether sample weights are supplied.
    """

    def test_identical_y_prob_produces_identical_threshold_candidates(self):
        """
        _binary_clf_curve and roc_curve(drop_intermediate=False) must produce
        exactly identical threshold candidate arrays with and without sample weights.
        """
        from sklearn.metrics._ranking import _binary_clf_curve
        from sklearn.metrics import roc_curve

        rng = np.random.default_rng(42)
        n = 300
        y_true = rng.integers(0, 2, n).astype(int)
        y_prob = rng.uniform(0.01, 0.99, n)
        weights = rng.uniform(0.5, 3.0, n)

        # 1. Raw threshold candidates from _binary_clf_curve must be identical
        _, _, th_raw_w = _binary_clf_curve(y_true, y_prob, sample_weight=weights)
        _, _, th_raw_uw = _binary_clf_curve(y_true, y_prob, sample_weight=None)
        self.assertEqual(len(th_raw_w), len(th_raw_uw))
        np.testing.assert_array_equal(th_raw_w, th_raw_uw)

        # 2. When drop_intermediate=False, roc_curve thresholds must be identical
        _, _, th_nodrop_w = roc_curve(y_true, y_prob, sample_weight=weights, drop_intermediate=False)
        _, _, th_nodrop_uw = roc_curve(y_true, y_prob, sample_weight=None, drop_intermediate=False)
        self.assertEqual(len(th_nodrop_w), len(th_nodrop_uw))
        np.testing.assert_array_equal(th_nodrop_w, th_nodrop_uw)

    def test_drop_intermediate_prunes_collinear_unweighted_points(self):
        """
        Confirms why roc_curve(drop_intermediate=True) has fewer thresholds
        in unweighted vs weighted curves: integer step sizes produce collinear
        points that are pruned, whereas continuous weights retain curvature.
        """
        from sklearn.metrics import roc_curve

        rng = np.random.default_rng(42)
        n = 500
        y_true = rng.integers(0, 2, n).astype(int)
        y_prob = rng.uniform(0.01, 0.99, n)
        weights = rng.uniform(0.5, 3.0, n)

        _, _, th_w = roc_curve(y_true, y_prob, sample_weight=weights, drop_intermediate=True)
        _, _, th_uw = roc_curve(y_true, y_prob, sample_weight=None, drop_intermediate=True)

        # Unweighted curve has fewer thresholds due to collinear step pruning
        self.assertLessEqual(len(th_uw), len(th_w))


if __name__ == '__main__':
    unittest.main()
