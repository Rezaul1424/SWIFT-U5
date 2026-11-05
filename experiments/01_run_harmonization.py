"""
Session 1 Pipeline Runner: Data Harmonization and Leakage-Free Cohort Construction.
Executes BR extraction, exposure filtering, predictor harmonization,
compiles master datasets, and generates Table 1.
"""

import sys
from pathlib import Path

# Add project root to path
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(WORKSPACE_DIR))

from src.data.extract import extract_all_br_files
from src.data.dataset import build_master_dataset

def main():
    print("=" * 70)
    print("  SWIFT-U5: SESSION 1 EXECUTION PIPELINE")
    print("  Step 1: Data Harmonization & Leakage-Free Cohort Construction")
    print("=" * 70)
    
    # Step 1: Ensure all BR files are extracted
    extract_all_br_files()
    
    # Step 2: Build master dataset, apply exposure filter, harmonize, output Table 1
    master_df = build_master_dataset()
    
    print("\n" + "=" * 70)
    print("  SESSION 1 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  Total Harmonized Records: {len(master_df):,}")
    print(f"  Total Under-Five Deaths: {master_df['target_u5m'].sum():,}")
    print("=" * 70)

if __name__ == "__main__":
    main()

