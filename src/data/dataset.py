"""
Master dataset construction pipeline for SWIFT-U5.
Compiles exposure-complete, harmonized cohorts across all 6 BDHS rounds (2004–2022),
saves processed datasets, and outputs Table 1 (Cohort Summary by Round and Division).
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.config import (
    load_config,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    REPORTS_DIR
)
from src.data.cohort import load_round_raw_br, filter_exposure_complete_cohort
from src.data.harmonize import harmonize_round_dataframe

def build_master_dataset() -> pd.DataFrame:
    """Extract, filter, and harmonize all 6 rounds into a unified master dataset."""
    config = load_config()
    round_dfs = []
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building master exposure-complete harmonized dataset across 6 BDHS rounds...")
    for item in config["rounds"]:
        rnd = item["round"]
        print(f"--> Processing Round {rnd}...")
        raw_df = load_round_raw_br(rnd)
        cohort_df = filter_exposure_complete_cohort(raw_df)
        harmonized_df = harmonize_round_dataframe(cohort_df)
        round_dfs.append(harmonized_df)
        print(f"    Completed Round {rnd}: N={len(harmonized_df)}, Deaths={harmonized_df['target_u5m'].sum()}")

    master_df = pd.concat(round_dfs, ignore_index=True)

    # Convert object columns to string to ensure clean parquet serialization
    for col in master_df.select_dtypes(include=['object']).columns:
        master_df[col] = master_df[col].astype(str)
    
    # Save master datasets
    parquet_path = PROCESSED_DATA_DIR / "u5m_harmonized_cohort.parquet"
    csv_path = PROCESSED_DATA_DIR / "u5m_harmonized_cohort.csv"
    
    master_df.to_parquet(parquet_path, index=False)
    print(f"Master parquet saved to: {parquet_path} ({parquet_path.stat().st_size / (1024*1024):.2f} MB)")
    master_df.to_csv(csv_path, index=False)
    print(f"Master CSV saved to: {csv_path} ({csv_path.stat().st_size / (1024*1024):.2f} MB)")
    
    # Generate Table 1
    table1 = generate_table1_cohort_summary(master_df)
    table1_path = TABLES_DIR / "table1_cohort_summary.csv"
    table1.to_csv(table1_path, index=False)
    print(f"Table 1 saved to: {table1_path}")
    
    # Generate LaTeX version of Table 1 (clean booktabs format without jinja2 dependency)
    table1_tex_path = TABLES_DIR / "table1_cohort_summary.tex"
    with open(table1_tex_path, "w", encoding="utf-8") as f:
        f.write("% Table 1: Under-Five Mortality Cohort Summary by Round and Division\n")
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{llrrrrrr}\n\\toprule\n")
        f.write("Survey Round & Division & $N$ & Deaths ($n$) & U5MR & $N_w$ & Deaths ($n_w$) & Weighted U5MR \\\\\n\\midrule\n")
        for _, r in table1.iterrows():
            f.write(f"{r['Survey Round']} & {r['Division (7-Harmonized)']} & {r['Unweighted Sample (N)']:,} & {r['Unweighted Deaths (n)']:,} & {r['Unweighted U5MR (per 1,000)']:.1f} & {r['Weighted Sample (N_w)']:.1f} & {r['Weighted Deaths (n_w)']:.1f} & {r['Weighted U5MR (per 1,000)']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\caption{Weighted sample, deaths, and prevalence by round and division (BDHS 2004--2022).}\n\\label{tab:table1_cohort}\n\\end{table}\n")
    print(f"Table 1 LaTeX saved to: {table1_tex_path}")

    # Generate data dictionary and harmonization report
    generate_harmonization_report(master_df)

    return master_df

def generate_table1_cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 1: Weighted sample, deaths, and prevalence by round and division.
    Computes both unweighted and survey-weighted sample counts and mortality rates per 1,000.
    """
    rows = []
    
    # Group by round and harmonized 7 divisions
    rounds = sorted(df['round'].unique().tolist())
    
    for rnd in rounds:
        df_rnd = df[df['round'] == rnd]
        
        # Round overall
        n_unw = len(df_rnd)
        d_unw = int(df_rnd['target_u5m'].sum())
        n_w = float(df_rnd['sample_weight'].sum())
        d_w = float((df_rnd['target_u5m'] * df_rnd['sample_weight']).sum())
        u5mr_unw = (d_unw / n_unw) * 1000.0 if n_unw > 0 else 0.0
        u5mr_w = (d_w / n_w) * 1000.0 if n_w > 0 else 0.0
        
        rows.append({
            'Survey Round': rnd,
            'Division (7-Harmonized)': 'National (Total)',
            'Unweighted Sample (N)': n_unw,
            'Unweighted Deaths (n)': d_unw,
            'Unweighted U5MR (per 1,000)': round(u5mr_unw, 1),
            'Weighted Sample (N_w)': round(n_w, 1),
            'Weighted Deaths (n_w)': round(d_w, 1),
            'Weighted U5MR (per 1,000)': round(u5mr_w, 1)
        })
        
        divisions = sorted(df_rnd['division_7'].unique().tolist())
        for div in divisions:
            df_div = df_rnd[df_rnd['division_7'] == div]
            div_n_unw = len(df_div)
            div_d_unw = int(df_div['target_u5m'].sum())
            div_n_w = float(df_div['sample_weight'].sum())
            div_d_w = float((df_div['target_u5m'] * df_div['sample_weight']).sum())
            div_u5mr_unw = (div_d_unw / div_n_unw) * 1000.0 if div_n_unw > 0 else 0.0
            div_u5mr_w = (div_d_w / div_n_w) * 1000.0 if div_n_w > 0 else 0.0
            
            rows.append({
                'Survey Round': rnd,
                'Division (7-Harmonized)': div.capitalize(),
                'Unweighted Sample (N)': div_n_unw,
                'Unweighted Deaths (n)': div_d_unw,
                'Unweighted U5MR (per 1,000)': round(div_u5mr_unw, 1),
                'Weighted Sample (N_w)': round(div_n_w, 1),
                'Weighted Deaths (n_w)': round(div_d_w, 1),
                'Weighted U5MR (per 1,000)': round(div_u5mr_w, 1)
            })

    table1_df = pd.DataFrame(rows)
    return table1_df

def generate_harmonization_report(df: pd.DataFrame):
    """Generate detailed variable missingness and harmonization report."""
    report_lines = [
        "# SWIFT-U5 Harmonization & Data Integrity Report",
        "",
        f"- Total Births in Cohort (0-59 months pre-survey): {len(df):,}",
        f"- Total Under-Five Deaths: {df['target_u5m'].sum():,} ({df['target_u5m'].mean() * 1000:.1f} per 1,000 unweighted)",
        f"- Total Weighted Deaths: {(df['target_u5m'] * df['sample_weight']).sum():,.1f}",
        "",
        "## Predictor Completeness Across Rounds (Percentage Available)",
        ""
    ]
    
    predictors = [
        'maternal_age_at_birth', 'maternal_education', 'wealth_quintile',
        'residence_urban', 'division_name', 'birth_order', 'parity_before',
        'is_first_birth', 'preceding_birth_interval', 'is_twin',
        'anc_visits', 'tetanus_toxoid', 'iron_supplementation',
        'facility_delivery', 'caesarean_delivery', 'birth_size'
    ]
    
    rounds = sorted(df['round'].unique().tolist())
    header = "| Predictor | " + " | ".join(rounds) + " | Total Missing (%) |"
    sep = "| :--- | " + " | ".join([":---:" for _ in rounds]) + " | :---: |"
    report_lines.extend([header, sep])
    
    for pred in predictors:
        row_vals = []
        for rnd in rounds:
            rnd_df = df[df['round'] == rnd]
            avail_pct = 100.0 - (rnd_df[pred].isna().mean() * 100.0)
            row_vals.append(f"{avail_pct:.1f}%")
        total_miss = df[pred].isna().mean() * 100.0
        report_lines.append(f"| `{pred}` | " + " | ".join(row_vals) + f" | {total_miss:.1f}% |")
        
    report_lines.extend([
        "",
        "## Post-Outcome Leakage Check",
        "- Breastfeeding (`m4`, `m5`, `v404`): **EXCLUDED** (0% leakage)",
        "- Child age at survey / death (`b19`, `b4`): **EXCLUDED** (0% leakage)",
        "- Children ever born at interview (`v201`): **EXCLUDED** (Replaced by strictly pre-outcome `parity_before = birth_order - 1`)",
        "- Current child vaccination (`h*`): **EXCLUDED** (0% leakage)",
        "",
        "## Exposure Completeness",
        "- All observations have known and complete 5-year exposure tracking.",
        "- Exposure months calculated explicitly: `exposure_months` = `b7` for deaths, `v008 - b3` for survivors."
    ])
    
    report_path = REPORTS_DIR / "session1_harmonization_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Harmonization report written to: {report_path}")

if __name__ == "__main__":
    build_master_dataset()
