"""
Unit tests for Session 2: 2x2 Factorial Validation Benchmark.
"""

import unittest
import pandas as pd
from pathlib import Path
from src.utils.config import TABLES_DIR, FIGURES_DIR, REPORTS_DIR

class TestFactorialBenchmark(unittest.TestCase):

    def test_table2_structure(self):
        csv_file = TABLES_DIR / "table2_2x2_main_results.csv"
        self.assertTrue(csv_file.exists())
        df = pd.read_csv(csv_file)
        self.assertEqual(len(df), 20)  # 4 arms x 5 cuts
        required_cols = ['Arm', 'Test Round', 'AUROC (W)', 'AUROC CI', 'AUPRC (W)', 'Brier (W)', 'NNS @ 5% (W)']
        for col in required_cols:
            self.assertIn(col, df.columns)

    def test_table3_structure(self):
        csv_file = TABLES_DIR / "table3_effect_decomposition.csv"
        self.assertTrue(csv_file.exists())
        df = pd.read_csv(csv_file)
        self.assertEqual(len(df), 5)  # 5 cuts
        self.assertIn('Validation Effect AUROC', df.columns)
        self.assertIn('Weight Effect AUROC', df.columns)
        self.assertIn('Interaction AUROC', df.columns)

    def test_figures_exist_and_non_empty(self):
        fig1 = FIGURES_DIR / "fig1_inflation_plot.png"
        fig2 = FIGURES_DIR / "fig2_calibration_curves.png"
        self.assertTrue(fig1.exists())
        self.assertTrue(fig2.exists())
        self.assertGreater(fig1.stat().st_size, 50_000)
        self.assertGreater(fig2.stat().st_size, 50_000)

if __name__ == '__main__':
    unittest.main()

