"""
Unit tests for Session 4: SWIFT-U5 Framework & Module Ablations.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import PROCESSED_DATA_DIR, TABLES_DIR, REPORTS_DIR
from src.models.module_a_weight import ModuleARecencyShiftWeighter
from src.models.module_b_shrink import ModuleBDivisionShrunkHead
from src.models.module_c_recalib import ModuleCDriftRecalibrator
from src.models.swift_u5 import SWIFTU5Estimator


class TestSWIFTU5Framework(unittest.TestCase):

    def setUp(self):
        parquet_path = PROCESSED_DATA_DIR / "u5m_harmonized_cohort.parquet"
        self.assertTrue(parquet_path.exists())
        self.df = pd.read_parquet(parquet_path)
        self.df_train = self.df[self.df['round'].isin(['2004', '2007', '2011', '2014', '2017-18'])].copy()
        self.df_test = self.df[self.df['round'] == '2022'].copy()

    def test_module_a_weighter(self):
        weighter = ModuleARecencyShiftWeighter(decay_rate=0.05, use_time_decay=True, use_density_ratio=False)
        w = weighter.compute_weights(self.df_train, target_round='2022')
        self.assertEqual(len(w), len(self.df_train))
        self.assertTrue((w > 0).all())
        self.assertAlmostEqual(w.mean(), 1.0, places=4)

    def test_module_b_eb_shrinkage(self):
        head = ModuleBDivisionShrunkHead(shrinkage_kappa=40.0)
        y = self.df_train['target_u5m'].values.astype(int)
        p = np.full(len(y), 0.04)
        div = self.df_train['division_7']
        head.fit(y, p, div)
        self.assertTrue(head.is_fitted)
        self.assertIn('dhaka', head.shrunk_params_)
        adj_p = head.predict_proba(p, div)
        self.assertEqual(len(adj_p), len(p))
        self.assertTrue((adj_p > 0).all() and (adj_p < 1).all())

    def test_module_c_recalibration(self):
        recalib = ModuleCDriftRecalibrator()
        y = (np.random.rand(1000) < 0.03).astype(int)
        p = np.full(1000, 0.06)  # Overpredicting model
        recalib.fit_recalibration(y, p)
        self.assertTrue(recalib.is_fitted)
        adj_p = recalib.predict_proba(p)
        self.assertLess(adj_p.mean(), p.mean())  # Should pull probability downward toward 0.03

    def test_swift_u5_estimator_pipeline(self):
        model = SWIFTU5Estimator(
            use_module_a=True, use_module_b=True, use_module_c=True,
            n_estimators=30
        )
        model.fit(self.df_train, target_round='2022')
        probs = model.predict_proba(self.df_test)
        self.assertEqual(len(probs), len(self.df_test))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_table4_and_table5_exist(self):
        t4_csv = TABLES_DIR / "table4_swift_vs_baselines.csv"
        t4_tex = TABLES_DIR / "table4_swift_vs_baselines.tex"
        t5_csv = TABLES_DIR / "table5_ablation_modules.csv"
        t5_tex = TABLES_DIR / "table5_ablation_modules.tex"
        rep = REPORTS_DIR / "session4_ablation_report.md"

        self.assertTrue(t4_csv.exists())
        self.assertTrue(t4_tex.exists())
        self.assertTrue(t5_csv.exists())
        self.assertTrue(t5_tex.exists())
        self.assertTrue(rep.exists())

        df_t4 = pd.read_csv(t4_csv)
        self.assertEqual(len(df_t4), 20)  # 4 models x 5 cuts
        df_t5 = pd.read_csv(t5_csv)
        self.assertEqual(len(df_t5), 5)   # 5 ablation configurations


if __name__ == '__main__':
    unittest.main()

