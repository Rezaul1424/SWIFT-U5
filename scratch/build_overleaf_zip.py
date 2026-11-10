import zipfile
from pathlib import Path
import shutil

def create_overleaf_package():
    zip_path = Path("SWIFT_U5_Overleaf_Package.zip")
    
    # Also backup previous zip if needed, or overwrite directly
    readme_content = """SWIFT-U5 OVERLEAF SUBMISSION PACKAGE
=====================================
Title: SWIFT-U5: A Survey-Weighted and Temporally Calibrated Machine Learning Framework for Population-Level Child Mortality Risk Stratification in Bangladesh
Authors: Md. Rafiul Azam Chowdhury, Rezaul Mursalin Khan, Fatema Tuz Zohura Rifa, Tanjim Ahmed Irfad, Arafat Hussain Nahid, Ohidujjaman
Affiliation: Department of Computer Science and Engineering, United International University, Dhaka 1212, Bangladesh

PACKAGE CONTENTS:
1. main.tex          - Primary LaTeX document (configured for Overleaf auto-compilation; fully updated to Faculty Research Phenomenon guidelines and Phase 6 frozen results basis)
2. manuscript.tex    - Identical copy of main.tex
3. cas-refs.bib      - Comprehensive BibTeX database (54 entries)
4. figures/          - High-resolution (300 DPI) publication figures (Figures 1 to 7 plus diagnostic plots)
5. tables/           - Standalone LaTeX and CSV tables (Tables 1 through 7 and audit tables)
6. README.txt        - This guide

HOW TO COMPILE ON OVERLEAF:
1. Go to https://www.overleaf.com and log in to your account.
2. Click "New Project" -> "Upload Project".
3. Select this zip file: SWIFT_U5_Overleaf_Package.zip.
4. Overleaf will extract the project and automatically select "main.tex" as the root file.
5. In Overleaf Menu (top left):
   - Compiler: pdfLaTeX
   - TeX Live version: 2023 or 2024 (recommended)
   - Main document: main.tex (or manuscript.tex)
6. Click "Recompile". The PDF will compile cleanly with all 46 formatted references, 7 tables, and 7 high-resolution figures!

KEY SCIENTIFIC INTEGRITY & FROZEN RESULTS PRESERVED:
- Cohort: 6 BDHS waves (2004–2022), N = 51,489 records, 2,338 under-five deaths.
- Partitioning:
  * Development (2004–2014): N = 29,686, 1,574 deaths
  * Validation / Calibration (2017–18): N = 8,684, 356 deaths
  * Locked Prospective Test (2022): N = 13,119, 408 deaths (weighted prevalence 2.99%)
- Core Discrimination & Calibration Metrics:
  * Primary SWIFT-U5 Weighted AUROC: 0.625 [95% CI: 0.587–0.663]
  * Baseline Weighted XGBoost AUROC: 0.638 [95% CI: 0.603–0.673]
  * Unweighted SWIFT-U5 AUROC: 0.657 [95% CI: 0.628–0.686]
  * Primary SWIFT-U5 Weighted Brier Score: 0.0287
  * Uncalibrated Baseline XGBoost Brier Score: 0.1649 (class-rebalanced)
  * Constant-Prevalence Null Brier Score: 0.0290 (BSS = +0.0103)
  * Calibration Curve: Slope = 0.806, Intercept = -0.905
- Simulated 10% Prioritization Yield:
  * 1,312 records prioritized containing 107 unweighted deaths
  * Approximately 97 (96.97) weighted true-positive event equivalents under DHS sampling weights
  * Weighted Sensitivity: 24.60%, Specificity: 90.45%, PPV: 7.70%
  * NNS: 13.0 national (8.1 in Rangpur)
  * Estimated Yield: 77.0 per 1,000 prioritized national (123.3 per 1,000 in Rangpur)
"""

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        # 1. main.tex
        z.write('docs/manuscript.tex', arcname='main.tex')
        # 2. manuscript.tex
        z.write('docs/manuscript.tex', arcname='manuscript.tex')
        # 3. cas-refs.bib
        z.write('docs/cas-refs.bib', arcname='cas-refs.bib')
        # 4. README.txt
        z.writestr('README.txt', readme_content)
        
        # 5. Figures
        fig_dir = Path('docs/figures')
        for fig_file in sorted(fig_dir.glob('*.png')):
            z.write(fig_file, arcname=f'figures/{fig_file.name}')
            
        # 6. Tables
        tbl_dir = Path('docs/tables')
        for tbl_file in sorted(tbl_dir.glob('*.*')):
            if tbl_file.suffix in ('.tex', '.csv'):
                z.write(tbl_file, arcname=f'tables/{tbl_file.name}')

    print(f"Successfully generated {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")
    
    # Also copy to docs/ for convenience
    docs_zip = Path("docs/SWIFT_U5_Overleaf_Package.zip")
    shutil.copy2(zip_path, docs_zip)
    print(f"Copied package to {docs_zip}")

if __name__ == '__main__':
    create_overleaf_package()
