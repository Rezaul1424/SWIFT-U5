"""
Session 2 Pipeline Runner: 2x2 Factorial Validation Benchmark & Baselines.
Generates Table 2 (main results), Fig 1 (inflation plot),
Table 3 (effect decomposition), and Fig 2 (calibration curves).
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['MPLCONFIGDIR'] = '/tmp'
from typing import Optional, Dict, List
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(WORKSPACE_DIR))

from src.validation.experiment_2x2 import run_2x2_factorial
from src.utils.config import PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR

# Screening capacities
SCREENING_CAPACITIES = [0.05, 0.10, 0.20]

ARM_COLORS = {
    'A_random_unweighted':     '#e74c3c',   # Red
    'B_random_weighted':       '#e67e22',   # Orange
    'C_temporal_unweighted':   '#2ecc71',   # Green
    'D_temporal_weighted':     '#2980b9',   # Blue
    'Mix1_train_uw_eval_w':    '#9b59b6',   # Purple (diagnostic)
    'Mix2_train_w_eval_uw':    '#1abc9c',   # Teal (diagnostic)
}

ARM_LABELS = {
    'A_random_unweighted':     'Arm A: Random / Unweighted',
    'B_random_weighted':       'Arm B: Random / Weighted',
    'C_temporal_unweighted':   'Arm C: Temporal / Unweighted',
    'D_temporal_weighted':     'Arm D: Temporal / Weighted',
    'Mix1_train_uw_eval_w':    'Mix1: Train UW / Eval W',
    'Mix2_train_w_eval_uw':    'Mix2: Train W / Eval UW',
}


def make_table2(results: list, model_filter: Optional[str] = 'xgboost') -> pd.DataFrame:
    """
    Table 2: 2x2 main results table.
    Filtered to selected model (default: XGBoost), or all models if model_filter is None.
    Arms A, B, C, D across all temporal cuts.
    """
    rows = []
    main_arms = ['A_random_unweighted', 'B_random_weighted',
                 'C_temporal_unweighted', 'D_temporal_weighted']

    for r in results:
        if model_filter is not None and r['model'] != model_filter:
            continue
        if r['arm'] not in main_arms:
            continue

        row = {
            'Model': r['model'].upper(),
            'Arm': ARM_LABELS.get(r['arm'], r['arm']),
            'Test Round': r['test_round'],
            'Train Rounds': r['train_rounds'],
            'N Test': r.get('n_test', ''),
            'Deaths (n)': r.get('n_deaths_test', ''),
            'AUROC (W)': f"{r.get('auroc_w', float('nan')):.3f}",
            'AUROC CI': f"[{r.get('auroc_w_ci_lo', float('nan')):.3f}, {r.get('auroc_w_ci_hi', float('nan')):.3f}]",
            'AUPRC (W)': f"{r.get('auprc_w', float('nan')):.3f}",
            'Brier (W)': f"{r.get('brier_w', float('nan')):.4f}",
            'Cal Slope (W)': f"{r.get('cal_slope_w', float('nan')):.3f}",
            'NNS @ 5% (W)': f"{r.get('nns_5pct_w', float('nan')):.1f}",
            'NNS @ 10% (W)': f"{r.get('nns_10pct_w', float('nan')):.1f}",
            'NNS @ 20% (W)': f"{r.get('nns_20pct_w', float('nan')):.1f}",
        }
        if model_filter is not None:
            row.pop('Model')
        rows.append(row)

    return pd.DataFrame(rows)


def make_table3(results: list, model_filter: Optional[str] = 'xgboost') -> pd.DataFrame:
    """
    Table 3: Effect decomposition.
    Measure validation effect (C vs A), weight effect (B vs A), interaction (D vs C).
    Calculated across temporal cuts and averaged.
    """
    rows = []
    main_arms = ['A_random_unweighted', 'B_random_weighted',
                 'C_temporal_unweighted', 'D_temporal_weighted']

    models_to_process = [model_filter] if model_filter is not None else ['xgboost', 'random_forest', 'logistic']

    for m in models_to_process:
        # Group by cut_label to compute diffs
        cut_data = {}
        for r in results:
            if r['model'] != m:
                continue
            if r['arm'] not in main_arms:
                continue
            key = (r.get('cut_label', ''), r['arm'])
            cut_data[key] = r

        cuts = sorted(set(k[0] for k in cut_data.keys()))
        for cut in cuts:
            a = cut_data.get((cut, 'A_random_unweighted'))
            b = cut_data.get((cut, 'B_random_weighted'))
            c = cut_data.get((cut, 'C_temporal_unweighted'))
            d = cut_data.get((cut, 'D_temporal_weighted'))

            if not all([a, b, c, d]):
                continue

            def val(d, k):
                return float(d.get(k, float('nan')))

            row = {
                'Model': m.upper(),
                'Temporal Cut': cut,
                'Test Round': c['test_round'],
                # Validation effect: Temporal (C) minus Random (A) -- negative indicates temporal deflation
                'Validation Effect AUROC': val(c, 'auroc_w') - val(a, 'auroc_w'),
                'Validation Effect NNS5': val(c, 'nns_5pct_w') - val(a, 'nns_5pct_w'),
                # Weight effect: Weighted (B) minus Unweighted (A) in Random split
                'Weight Effect AUROC': val(b, 'auroc_w') - val(a, 'auroc_w'),
                'Weight Effect NNS5': val(b, 'nns_5pct_w') - val(a, 'nns_5pct_w'),
                # Interaction: additional gain from combining temporal + weighting: (D - C) - (B - A)
                'Interaction AUROC': (val(d, 'auroc_w') - val(c, 'auroc_w')) - (val(b, 'auroc_w') - val(a, 'auroc_w')),
                # Inflation: Random (A) minus Temporal (C) -- positive indicates random split optimism
                'AUROC Inflation': val(a, 'auroc_w') - val(c, 'auroc_w'),
                'NNS5 Inflation': val(a, 'nns_5pct_w') - val(c, 'nns_5pct_w'),
            }
            if model_filter is not None:
                row.pop('Model')
            rows.append(row)

    return pd.DataFrame(rows)


def make_figure1(results: list, model_filter: str = 'xgboost',
                  save_path: Path = None) -> None:
    """Fig 1: AUROC inflation plot across temporal cuts — Arms A vs C."""
    main_arms = ['A_random_unweighted', 'C_temporal_unweighted',
                 'B_random_weighted', 'D_temporal_weighted']

    # Gather per-cut data
    cut_arm_data = {}
    for r in results:
        if r['model'] != model_filter or r['arm'] not in main_arms:
            continue
        key = (r.get('cut_label', r['test_round']), r['arm'])
        cut_arm_data[key] = r

    cuts = sorted(set(k[0] for k in cut_arm_data.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: AUROC by arm and cut ---
    ax = axes[0]
    for arm in main_arms:
        aurocs = []
        test_rounds = []
        for cut in cuts:
            entry = cut_arm_data.get((cut, arm))
            if entry:
                aurocs.append(entry.get('auroc_w', float('nan')))
                test_rounds.append(entry['test_round'])
        if aurocs:
            ax.plot(test_rounds, aurocs,
                    marker='o', linewidth=2.0, markersize=6,
                    color=ARM_COLORS[arm], label=ARM_LABELS[arm])

    ax.set_xlabel('Test Round', fontsize=12)
    ax.set_ylabel('Weighted AUROC', fontsize=12)
    ax.set_title('AUROC: Random vs. Temporal Validation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_ylim(0.55, 0.90)
    ax.grid(True, alpha=0.3)

    # --- Right: Inflation gap (A - C) per cut ---
    ax = axes[1]
    inflation_auroc = []
    inflation_nns = []
    test_rounds_r = []

    for cut in cuts:
        a = cut_arm_data.get((cut, 'A_random_unweighted'))
        c = cut_arm_data.get((cut, 'C_temporal_unweighted'))
        if a and c:
            inf_a = float(a.get('auroc_w', float('nan'))) - float(c.get('auroc_w', float('nan')))
            inflation_auroc.append(inf_a)
            test_rounds_r.append(a['test_round'])

    bars = ax.bar(test_rounds_r, inflation_auroc, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Test Round', fontsize=12)
    ax.set_ylabel('AUROC Inflation (Arm A − Arm C)', fontsize=12)
    ax.set_title('Inflation: Random vs. Temporal Validation', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Label bars
    for bar, val in zip(bars, inflation_auroc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'+{val:.3f}' if val >= 0 else f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fig 1 saved to: {save_path}")
    plt.close()


def make_figure2(results: list, model_filter: str = 'xgboost',
                  save_path: Path = None) -> None:
    """Fig 2: Calibration curves — Arms A, B, C, D on last temporal cut (test 2022)."""
    main_arms = ['A_random_unweighted', 'B_random_weighted',
                 'C_temporal_unweighted', 'D_temporal_weighted']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax_idx, arm in enumerate(main_arms):
        ax = axes[ax_idx]

        # Find the 2022 test round entry
        entry = None
        for r in results:
            if r['model'] == model_filter and r['arm'] == arm and r['test_round'] == '2022':
                entry = r
                break

        if entry is None:
            ax.text(0.5, 0.5, f"No data for {arm}", ha='center', va='center')
            continue

        y_test = entry['_y_test']
        y_prob = entry['_y_prob']
        w_test = entry['_w_test']

        # Calibration curve using quantile binning
        n_bins = 10
        order = np.argsort(y_prob)
        y_t_s = y_test[order]
        y_p_s = y_prob[order]
        w_s = w_test[order]

        bins = np.array_split(np.arange(len(y_t_s)), n_bins)
        mean_pred = [np.average(y_p_s[b], weights=w_s[b]) for b in bins if len(b) > 0]
        mean_obs = [np.average(y_t_s[b], weights=w_s[b]) for b in bins if len(b) > 0]

        ax.plot([0, 0.2], [0, 0.2], 'k--', linewidth=1.5, label='Perfect calibration')
        ax.scatter(mean_pred, mean_obs, color=ARM_COLORS[arm], s=60, zorder=5)
        ax.plot(mean_pred, mean_obs, color=ARM_COLORS[arm], linewidth=2.0,
                label=ARM_LABELS[arm])

        slope = entry.get('cal_slope_w', float('nan'))
        intercept = entry.get('cal_intercept_w', float('nan'))
        ax.set_title(f"{ARM_LABELS[arm]}\nSlope={slope:.2f}, Int={intercept:.3f}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Mean Predicted Probability', fontsize=10)
        ax.set_ylabel('Observed Event Fraction', fontsize=10)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Calibration Curves: Arms A–D (XGBoost, Test Round 2022)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fig 2 saved to: {save_path}")
    plt.close()


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("  SWIFT-U5: SESSION 2 — 2x2 FACTORIAL VALIDATION BENCHMARK")
    print("="*70)

    # Load master dataset
    df = pd.read_parquet(PROCESSED_DATA_DIR / "u5m_harmonized_cohort.parquet")
    print(f"Loaded master cohort: {len(df):,} records, {df['target_u5m'].sum():,} deaths")

    # Run full factorial experiment
    results = run_2x2_factorial(df, models=['logistic', 'xgboost', 'random_forest'],
                                 run_bootstrap=True)

    # --- Save raw results ---
    raw_results_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith('_')}
        for r in results
    ])
    raw_path = REPORTS_DIR / "session2_2x2_raw_results.csv"
    raw_results_df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved: {raw_path}")

    # --- Table 2: Main 2x2 Results (XGBoost) ---
    table2 = make_table2(results, model_filter='xgboost')
    table2_path = TABLES_DIR / "table2_2x2_main_results.csv"
    table2.to_csv(table2_path, index=False)
    (WORKSPACE_DIR / "docs" / "tables" / "table2_2x2_main_results.csv").write_text(table2.to_csv(index=False))
    print(f"Table 2 saved: {table2_path}")

    # --- Table 2: All Models (XGBoost, Random Forest, Logistic) ---
    table2_all = make_table2(results, model_filter=None)
    table2_all_path = TABLES_DIR / "table2_2x2_all_models.csv"
    table2_all.to_csv(table2_all_path, index=False)
    (WORKSPACE_DIR / "docs" / "tables" / "table2_2x2_all_models.csv").write_text(table2_all.to_csv(index=False))
    print(f"Table 2 (All Models) saved: {table2_all_path}")

    # --- Table 3: Effect Decomposition (XGBoost) ---
    table3 = make_table3(results, model_filter='xgboost')
    table3_path = TABLES_DIR / "table3_effect_decomposition.csv"
    table3.to_csv(table3_path, index=False)
    (WORKSPACE_DIR / "docs" / "tables" / "table3_effect_decomposition.csv").write_text(table3.to_csv(index=False))
    print(f"Table 3 saved: {table3_path}")

    # --- Table 3: Effect Decomposition (All Models) ---
    table3_all = make_table3(results, model_filter=None)
    table3_all_path = TABLES_DIR / "table3_effect_decomposition_all_models.csv"
    table3_all.to_csv(table3_all_path, index=False)
    (WORKSPACE_DIR / "docs" / "tables" / "table3_effect_decomposition_all_models.csv").write_text(table3_all.to_csv(index=False))
    print(f"Table 3 (All Models) saved: {table3_all_path}")

    # --- Fig 1: Inflation plot ---
    fig1_path = FIGURES_DIR / "fig1_inflation_plot.png"
    make_figure1(results, model_filter='xgboost', save_path=fig1_path)
    import shutil
    shutil.copyfile(fig1_path, WORKSPACE_DIR / "docs" / "figures" / "fig1_inflation_plot.png")

    # --- Fig 2: Calibration curves ---
    fig2_path = FIGURES_DIR / "fig2_calibration_curves.png"
    make_figure2(results, model_filter='xgboost', save_path=fig2_path)
    shutil.copyfile(fig2_path, WORKSPACE_DIR / "docs" / "figures" / "fig2_calibration_curves.png")

    print("\n" + "="*70)
    print("  SESSION 2 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  Table 2: {table2_path}")
    print(f"  Table 3: {table3_path}")
    print(f"  Fig 1:   {fig1_path}")
    print(f"  Fig 2:   {fig2_path}")
    print("="*70)


if __name__ == "__main__":
    main()

