#!/usr/bin/env python3
"""
experiments/generate_publication_figures.py

Generates and extracts high-resolution separated publication-ready figures
for the SWIFT-U5 manuscript. Each sub-panel from the multi-panel composite
figures is saved as an individual, standalone PNG with trimmed borders at 300 DPI.

Generated Individual Figures:
-----------------------------
Figure 1 (Evaluation Design & Inflation):
  - fig1a_auroc_comparison.png: AUROC across survey cuts (Random vs. Temporal Validation)
  - fig1b_inflation_gap.png: AUROC Inflation Gap (Arm A - Arm C) across survey rounds

Figure 2 (Probability Calibration):
  - fig2a_cal_random_unweighted.png: Arm A (Random Split, Unweighted XGBoost)
  - fig2b_cal_random_weighted.png: Arm B (Random Split, Weighted XGBoost)
  - fig2c_cal_temporal_unweighted.png: Arm C (Forward-Temporal Split, Unweighted XGBoost)
  - fig2d_cal_temporal_weighted.png: Arm D (Forward-Temporal Split, Weighted XGBoost)

Figure 3 (Feature Stability):
  - fig3a_stability_selection.png: Panel A - Dual-Method Selection Frequency (LASSO vs. XGBoost)
  - fig3b_rolling_window_stability.png: Panel B - Rolling Window Stability Heatmap

Figure 4 (Division Risk Atlas & Capacity):
  - fig4a_observed_vs_predicted_risk.png: Panel A - 2022 Observed Mortality vs. Predicted Risk
  - fig4b_screening_efficiency.png: Panel B - Screening / Prioritization Efficiency @ 10%
  - fig4c_longitudinal_trends.png: Panel C - Longitudinal Trends in U5MR (2004-2022)

Figure 5 (Explainability & Attribution Drift):
  - fig5a_shap_global_importance.png: Panel A - Global Top 12 Risk Drivers (|SHAP|)
  - fig5b_shap_regional_variation.png: Panel B - Regional Variation Heatmap of Top 8 Drivers
  - fig5c_shap_drift_index.png: Panel C - Temporal SHAP Drift Index (SDI) across cuts
"""

import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path

def trim_border(im, tol=250, pad=20):
    """Auto-trim white or transparent border padding around an image."""
    if im.mode == 'RGBA':
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        diff = np.abs(np.array(im).astype(float) - np.array(bg).astype(float))
    else:
        bg = Image.new('RGB', im.size, (255, 255, 255))
        diff = np.abs(np.array(im).astype(float) - np.array(bg).astype(float))
    mask = np.any(diff > (255 - tol), axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) and len(cols):
        r0 = max(0, rows[0] - pad)
        r1 = min(im.height, rows[-1] + pad + 1)
        c0 = max(0, cols[0] - pad)
        c1 = min(im.width, cols[-1] + pad + 1)
        return im.crop((c0, r0, c1, r1))
    return im

def extract_all_separated_figures():
    workspace = Path(__file__).resolve().parent.parent
    docs_fig = workspace / "docs" / "figures"
    out_fig = workspace / "outputs" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    separated = {}

    # --- Figure 1: Inflation Plot (1x2 layout) ---
    p1 = docs_fig / "fig1_inflation_plot.png"
    if p1.exists():
        im1 = Image.open(p1)
        split_x1 = 2043
        separated["fig1a_auroc_comparison.png"] = trim_border(im1.crop((0, 0, split_x1, im1.height)))
        separated["fig1b_inflation_gap.png"] = trim_border(im1.crop((split_x1, 0, im1.width, im1.height)))

    # --- Figure 2: Calibration Curves (2x2 layout) ---
    p2 = docs_fig / "fig2_calibration_curves.png"
    if p2.exists():
        im2 = Image.open(p2)
        split_x2 = 1792
        split_y2 = 1474
        separated["fig2a_cal_random_unweighted.png"] = trim_border(im2.crop((0, 80, split_x2, split_y2)))
        separated["fig2b_cal_random_weighted.png"] = trim_border(im2.crop((split_x2, 80, im2.width, split_y2)))
        separated["fig2c_cal_temporal_unweighted.png"] = trim_border(im2.crop((0, split_y2, split_x2, im2.height)))
        separated["fig2d_cal_temporal_weighted.png"] = trim_border(im2.crop((split_x2, split_y2, im2.width, im2.height)))

    # --- Figure 3: Feature Stability (1x2 layout) ---
    p3 = docs_fig / "fig3_feature_stability.png"
    if p3.exists():
        im3 = Image.open(p3)
        split_x3 = 1628
        separated["fig3a_stability_selection.png"] = trim_border(im3.crop((0, 0, split_x3, im3.height)))
        separated["fig3b_rolling_window_stability.png"] = trim_border(im3.crop((split_x3, 0, im3.width, im3.height)))

    # --- Figure 4: Division Risk Atlas (2x2 layout: 2 top, 1 bottom spanning) ---
    p4 = docs_fig / "fig4_division_risk_atlas.png"
    if p4.exists():
        im4 = Image.open(p4)
        split_x4 = 1919
        split_y4 = 1206
        separated["fig4a_observed_vs_predicted_risk.png"] = trim_border(im4.crop((0, 0, split_x4, split_y4)))
        separated["fig4b_screening_efficiency.png"] = trim_border(im4.crop((split_x4, 0, im4.width, split_y4)))
        separated["fig4c_longitudinal_trends.png"] = trim_border(im4.crop((0, split_y4, im4.width, im4.height)))

    # --- Figure 5: SHAP Drift & Attribution (Left column, 2 right rows) ---
    p5 = docs_fig / "fig5_shap_drift_division.png"
    if p5.exists():
        im5 = Image.open(p5)
        split_x5 = 2960
        split_y5 = 1231
        separated["fig5a_shap_global_importance.png"] = trim_border(im5.crop((0, 0, split_x5, im5.height)))
        separated["fig5b_shap_regional_variation.png"] = trim_border(im5.crop((split_x5, 0, im5.width, split_y5)))
        separated["fig5c_shap_drift_index.png"] = trim_border(im5.crop((split_x5, split_y5, im5.width, im5.height)))

    print(f"Generated {len(separated)} individual high-resolution figures:")
    for fname, img in separated.items():
        p_doc = docs_fig / fname
        p_out = out_fig / fname
        img.save(p_doc, format="PNG", dpi=(300, 300), optimize=True)
        img.save(p_out, format="PNG", dpi=(300, 300), optimize=True)
        print(f"  - {fname:<40} {img.width}x{img.height} px ({p_doc.stat().st_size/1024:.1f} KB)")

    return separated

if __name__ == '__main__':
    extract_all_separated_figures()
