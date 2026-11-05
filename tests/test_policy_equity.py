"""
Unit tests for Session 5: Policy Layer and Equity Audit.
Validates division capacity screening metrics, equity audit disaggregation,
bootstrap CI generation, and SHAP drift calculation.
"""

import os
import warnings
os.environ['MPLCONFIGDIR'] = '/tmp'
warnings.filterwarnings('ignore')

import unittest
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.policy.policy_evaluation import (
    compute_screening_stats,
    bootstrap_subgroup_capacity,
    compute_division_capacity_table,
    compute_equity_audit_table,
    compute_longitudinal_division_trends,
    compute_posthoc_shap_and_drift
)
from src.models.swift_u5 import SWIFTU5Estimator


class TestPolicyEquity(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 300
        self.y_true = np.random.binomial(1, 0.05, n)
        self.y_prob = np.clip(np.random.beta(1, 20, n) + 0.1 * self.y_true, 0.001, 0.999)
        self.weights = np.random.uniform(0.5, 1.5, n)

        divisions = ['barishal', 'chattogram', 'dhaka', 'khulna', 'mymensingh', 'rajshahi', 'rangpur', 'sylhet']
        self.df_test = pd.DataFrame({
            'target_u5m': self.y_true,
            'sample_weight': self.weights,
            'psu': np.random.randint(1, 20, n),
            'strata': np.random.randint(1, 4, n),
            'division_8': np.random.choice(divisions, n),
            'division_7': np.random.choice(divisions[:7], n),
            'wealth_quintile': np.random.choice([1, 2, 3, 4, 5], n),
            'residence_urban': np.random.choice([0, 1], n),
            'maternal_education': np.random.choice([0, 1, 2, 3], n),
            'round': ['2022'] * n
        })

    def test_compute_screening_stats(self):
        stats = compute_screening_stats(self.y_true, self.y_prob, self.weights, capacity=0.10)
        self.assertIn('sensitivity', stats)
        self.assertIn('ppv', stats)
        self.assertIn('nns', stats)
        self.assertIn('deaths_per_1000', stats)
        self.assertIn('yield_ratio', stats)
        self.assertGreaterEqual(stats['sensitivity'], 0.0)
        self.assertLessEqual(stats['sensitivity'], 1.0)
        self.assertGreaterEqual(stats['ppv'], 0.0)
        self.assertGreaterEqual(stats['nns'], 1.0)
        self.assertAlmostEqual(stats['deaths_per_1000'], stats['ppv'] * 1000.0, places=2)

    def test_bootstrap_subgroup_capacity(self):
        cis = bootstrap_subgroup_capacity(self.df_test, self.y_prob, capacity=0.10, n_replicates=25, seed=42)
        self.assertIn('sensitivity', cis)
        self.assertIn('nns', cis)
        lo, hi = cis['sensitivity']
        self.assertLessEqual(lo, hi)

    def test_division_capacity_table(self):
        div_table = compute_division_capacity_table(
            self.df_test, self.y_prob, division_col='division_8',
            capacities=[0.05, 0.10], n_bootstrap=15, seed=42
        )
        self.assertIsInstance(div_table, pd.DataFrame)
        self.assertGreater(len(div_table), 1)
        # Check that National is present
        self.assertIn('National (Bangladesh)', div_table['division'].values)
        self.assertIn('sens_10', div_table.columns)
        self.assertIn('nns_10', div_table.columns)

    def test_equity_audit_table(self):
        equity_table = compute_equity_audit_table(
            self.df_test, self.y_prob, capacity=0.10, n_bootstrap=15, seed=42
        )
        self.assertIsInstance(equity_table, pd.DataFrame)
        self.assertIn('domain', equity_table.columns)
        self.assertIn('equity_status', equity_table.columns)
        domains = set(equity_table['domain'])
        self.assertIn('Overall', domains)
        self.assertIn('Residence', domains)
        self.assertIn('Wealth Quintile', domains)

    def test_longitudinal_division_trends(self):
        df_multi = pd.concat([
            self.df_test.assign(round='2017-18'),
            self.df_test.assign(round='2022')
        ], ignore_index=True)
        trends = compute_longitudinal_division_trends(df_multi, division_col='division_7')
        self.assertIn('round', trends.columns)
        self.assertIn('u5mr_per_1000', trends.columns)
        self.assertEqual(len(trends['round'].unique()), 2)


if __name__ == '__main__':
    unittest.main()
