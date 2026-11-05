"""
Unit tests for data integrity, exposure completeness, and leakage exclusion.
"""

import unittest
import pandas as pd
from pathlib import Path
from src.utils.config import PROCESSED_DATA_DIR, TABLES_DIR

class TestDataIntegrity(unittest.TestCase):

    def setUp(self):
        parquet_path = PROCESSED_DATA_DIR / "u5m_harmonized_cohort.parquet"
        self.assertTrue(parquet_path.exists(), "Master parquet file does not exist!")
        self.df = pd.read_parquet(parquet_path)

    def test_cohort_size_and_outcomes(self):
        """Verify total records, mortality cases, and exposure completeness."""
        self.assertEqual(len(self.df), 51489, f"Expected 51,489 records, got {len(self.df)}")
        self.assertEqual(self.df['target_u5m'].sum(), 2338, f"Expected 2,338 deaths, got {self.df['target_u5m'].sum()}")
        self.assertEqual(set(self.df['target_u5m'].unique()), {0, 1})
        self.assertGreaterEqual(self.df['exposure_months'].min(), 0)
        self.assertLessEqual(self.df['exposure_months'].max(), 60)

    def test_leakage_exclusion(self):
        """Ensure all post-outcome leakage variables are strictly absent."""
        forbidden_cols = ['b4', 'b19', 'm4', 'm5', 'v404', 'v201', 'h1', 'h2', 'h3', 'h4', 'h10']
        for col in forbidden_cols:
            self.assertNotIn(col, self.df.columns, f"Leakage detected: {col} present in dataset!")

    def test_survey_design_variables(self):
        """Verify survey weights, PSU, and strata exist and have no NaNs."""
        self.assertIn('sample_weight', self.df.columns)
        self.assertIn('psu', self.df.columns)
        self.assertIn('strata', self.df.columns)
        self.assertEqual(self.df['sample_weight'].isna().sum(), 0)
        self.assertTrue((self.df['sample_weight'] > 0).all())

    def test_table1_exists(self):
        """Verify Table 1 CSV and TeX are generated."""
        csv_file = TABLES_DIR / "table1_cohort_summary.csv"
        tex_file = TABLES_DIR / "table1_cohort_summary.tex"
        self.assertTrue(csv_file.exists())
        self.assertTrue(tex_file.exists())
        df_t1 = pd.read_csv(csv_file)
        self.assertGreater(len(df_t1), 0)
        self.assertIn("Weighted U5MR (per 1,000)", df_t1.columns)

if __name__ == '__main__':
    unittest.main()
