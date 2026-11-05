"""
Unit tests for Session 3: Feature Selection & Temporal Stability.
"""

import unittest
import pandas as pd
from pathlib import Path
from src.utils.config import PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR
from src.features.build_features import LeakageFreePreprocessor
from src.features.select import apply_domain_filter

class TestFeatureSelection(unittest.TestCase):

    def setUp(self):
        parquet_path = PROCESSED_DATA_DIR / "u5m_harmonized_cohort.parquet"
        self.assertTrue(parquet_path.exists())
        self.df = pd.read_parquet(parquet_path)
        self.df_train = self.df[self.df['round'].isin(['2004', '2007', '2011', '2014', '2017-18'])].copy()

    def test_domain_filter_collinearity_and_missingness(self):
        """Verify high-missingness (>30%) and collinear (|r|>0.8) features are dropped."""
        retained, log_info = apply_domain_filter(self.df_train, max_missing_rate=0.30, max_correlation=0.80)
        
        # parity_before must be dropped due to r=1.0 with birth_order
        self.assertNotIn('parity_before', retained)
        self.assertIn('birth_order', retained)

        # Variables with >30% missing in train pool must be dropped
        for dropped_var, _ in log_info['dropped_high_missing']:
            self.assertNotIn(dropped_var, retained)

    def test_leakage_free_preprocessor(self):
        """Verify preprocessor fits medians strictly on train and transforms without NaNs."""
        prep = LeakageFreePreprocessor(include_missing_flags=True)
        X_train, names_train = prep.fit_transform(self.df_train)
        self.assertEqual(len(X_train), len(self.df_train))
        self.assertFalse(pd.isna(X_train).any())

        # Transform on unseen 2022 test set using train-fitted statistics
        df_test = self.df[self.df['round'] == '2022'].copy()
        X_test, names_test = prep.transform(df_test)
        self.assertEqual(len(X_test), len(df_test))
        self.assertFalse(pd.isna(X_test).any())
        self.assertEqual(names_train, names_test)

    def test_fig3_and_table_exist(self):
        """Verify Fig 3 and stability table deliverables exist and are valid."""
        fig3 = FIGURES_DIR / "fig3_feature_stability.png"
        tab = TABLES_DIR / "table_feature_stability.csv"
        rep = REPORTS_DIR / "session3_feature_selection_report.md"

        self.assertTrue(fig3.exists())
        self.assertGreater(fig3.stat().st_size, 50_000)

        self.assertTrue(tab.exists())
        df_tab = pd.read_csv(tab)
        self.assertIn('classification', df_tab.columns)
        self.assertIn('Stable Core', df_tab['classification'].values)

        self.assertTrue(rep.exists())

    def test_zero_shap_leakage_in_selection(self):
        """Verify SHAP is not imported or used for feature selection (quarantine rule)."""
        select_file = Path("src/features/select.py")
        code = select_file.read_text(encoding="utf-8")
        self.assertNotIn("import shap", code)
        self.assertNotIn("TreeExplainer", code)

if __name__ == '__main__':
    unittest.main()

