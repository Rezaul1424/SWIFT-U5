"""
experiments/07_generate_external_validation_figures.py
=======================================================
Generates two publication-quality figures for the external multi-country
validation section (§5.5.5):

  Fig 6  –  AUC comparison panel (Nepal + Ethiopia):
            dot-and-CI plot across Track A, Track B, and Cross-Country transfer.

  Fig 7  –  Calibration curves for cross-country transfer:
            Nepal 2022 (zero-shot vs locally recalibrated) and
            Ethiopia 2024-25 (zero-shot vs locally recalibrated).

Outputs written to docs/figures/:
  fig6_external_auroc_comparison.png
  fig7_external_calibration.png

Data sources (already on disk, do NOT re-run validation pipeline):
  outputs/tables/table8_external_validation.csv
  outputs/predictions/authoritative_swift_u5_2022_predictions.parquet
                             (for colour palette consistency only)

Usage:
  python experiments/07_generate_external_validation_figures.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(ROOT, "outputs", "tables", "table8_external_validation.csv")
PRED_PATH = os.path.join(ROOT, "outputs", "predictions",
                         "authoritative_swift_u5_2022_predictions.parquet")
FIG_DIR   = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Colour / style palette (consistent with existing figs) ───────────────────
NEPAL_COLOR    = "#2166AC"   # blue
ETHIOPIA_COLOR = "#D6604D"   # orange-red
TRACK_A_COLOR  = "#4DAF4A"   # green
TRACK_B_COLOR  = "#FF7F00"   # orange
XFER_COLOR_ZS  = "#984EA3"   # purple
XFER_COLOR_LC  = "#1B9E77"   # teal

FONT_SIZE_TITLE = 11
FONT_SIZE_LABEL = 9
FONT_SIZE_TICK  = 8

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      FONT_SIZE_LABEL,
    "axes.titlesize": FONT_SIZE_TITLE,
    "axes.labelsize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
})

# ── Load results ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

def parse_ci(ci_str):
    """Parse '[lo, hi]' string into (lo, hi) floats."""
    if not isinstance(ci_str, str):
        return np.nan, np.nan
    ci_str = ci_str.strip("[]")
    parts = ci_str.split(",")
    if len(parts) != 2:
        return np.nan, np.nan
    try:
        return float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        return np.nan, np.nan

df["CI_lo"], df["CI_hi"] = zip(*df["AUROC_95CI"].map(parse_ci))


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 – External AUC comparison (dot-and-CI, two panels side by side)
# ══════════════════════════════════════════════════════════════════════════════

def _build_fig6_data(country):
    rows = df[df["Country"] == country].copy()

    # --- Track A: SWIFT-U5 Full and XGBoost baseline ---
    track_a_full = rows[
        (rows["Evaluation"].str.startswith("Track A")) &
        (rows["Model"] == "SWIFT-U5 (Full)")
    ].iloc[0]
    track_a_xgb = rows[
        (rows["Evaluation"].str.startswith("Track A")) &
        (rows["Model"] == "Weighted XGBoost")
    ].iloc[0]
    track_a_wof = rows[
        (rows["Evaluation"].str.startswith("Track A")) &
        (rows["Model"] == "SWIFT-U5 w/o Module A")
    ].iloc[0]

    # --- Track B: Random XGBoost ---
    track_b = rows[rows["Evaluation"].str.startswith("Track B")].iloc[0]

    # --- Cross-country ---
    xfer_zs = rows[
        (rows["Evaluation"].str.startswith("Cross-Country")) &
        (rows["Model"].str.contains("Zero-Shot"))
    ].iloc[0]
    xfer_lc = rows[
        (rows["Evaluation"].str.startswith("Cross-Country")) &
        (rows["Model"].str.contains("Local Calib"))
    ].iloc[0]

    return [
        # label,                   auc,                  lo,                   hi,                  color,         marker, zorder
        ("SWIFT-U5 (Full)\n[Track A]",    track_a_full["AUROC_W"], track_a_full["CI_lo"], track_a_full["CI_hi"], TRACK_A_COLOR,  "o", 4),
        ("w/o Module A\n[Track A]",       track_a_wof["AUROC_W"],  track_a_wof["CI_lo"],  track_a_wof["CI_hi"],  "#78C679",     "s", 4),
        ("XGBoost Baseline\n[Track A]",   track_a_xgb["AUROC_W"],  track_a_xgb["CI_lo"],  track_a_xgb["CI_hi"],  "#6BAED6",     "^", 4),
        ("XGBoost (Random)\n[Track B]",   track_b["AUROC_W"],       track_b["CI_lo"],       track_b["CI_hi"],       TRACK_B_COLOR, "D", 4),
        ("BDHS Transfer\n(Zero-Shot)",     xfer_zs["AUROC_W"],       xfer_zs["CI_lo"],       xfer_zs["CI_hi"],       XFER_COLOR_ZS, "P", 5),
        ("BDHS Transfer\n(Local Calib)",   xfer_lc["AUROC_W"],       xfer_lc["CI_lo"],       xfer_lc["CI_hi"],       XFER_COLOR_LC, "*", 5),
    ]


def make_fig6():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    for ax, country, color in zip(axes, ["Nepal", "Ethiopia"],
                                  [NEPAL_COLOR, ETHIOPIA_COLOR]):
        data = _build_fig6_data(country)
        y_pos = list(range(len(data)))[::-1]   # top-to-bottom

        for i, (label, auc, lo, hi, col, mkr, zo) in enumerate(data):
            yp = y_pos[i]
            err_lo = auc - lo if not np.isnan(lo) else 0
            err_hi = hi - auc if not np.isnan(hi) else 0
            ax.errorbar(auc, yp, xerr=[[err_lo], [err_hi]],
                        fmt=mkr, color=col, markersize=9,
                        ecolor=col, elinewidth=1.5, capsize=4,
                        zorder=zo, label=label)
            ax.text(auc, yp - 0.35, f"{auc:.3f}", ha="center",
                    va="top", fontsize=7, color=col)

        # ─ Reference lines ─
        ax.axvline(0.5, ls="--", lw=0.8, color="grey", alpha=0.6, label="Chance (0.5)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([d[0] for d in data], fontsize=8)
        ax.set_xlabel("Survey-Weighted ROC-AUC (95% CI)", fontsize=FONT_SIZE_LABEL)
        ax.set_xlim(0.45, 0.80)
        ax.set_title(f"({chr(96 + list(['Nepal','Ethiopia']).index(country) + 1)}) {country}", fontsize=FONT_SIZE_TITLE, fontweight="bold")
        ax.grid(axis="x", ls=":", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    # Shared legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TRACK_A_COLOR,
               markersize=9, label="SWIFT-U5 Full [Track A]"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#78C679",
               markersize=9, label="SWIFT-U5 w/o Module A [Track A]"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#6BAED6",
               markersize=9, label="XGBoost Baseline [Track A]"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=TRACK_B_COLOR,
               markersize=9, label="XGBoost – Random Split [Track B]"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor=XFER_COLOR_ZS,
               markersize=9, label="BDHS Transfer – Zero-Shot"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=XFER_COLOR_LC,
               markersize=11, label="BDHS Transfer – Local Recalibration"),
        Line2D([0], [0], ls="--", color="grey", label="Chance (0.5)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        "Fig 6. External Multi-Country Validation: ROC-AUC by Evaluation Design\n"
        "Nepal DHS 2022 (locked) and Ethiopia DHS 2024–25 (locked)",
        fontsize=FONT_SIZE_TITLE, y=1.01
    )
    out = os.path.join(FIG_DIR, "fig6_external_auroc_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 – Cross-country transfer calibration curves (2×2 panel)
# ══════════════════════════════════════════════════════════════════════════════

def make_fig7():
    """
    Draws 4-panel calibration figure using stored slope/intercept values to
    reconstruct logistic calibration curves (Platt model on logit scale).
    Shows the before/after recalibration contrast for each country.

    The calibration curve is modelled as:
        p_cal = sigmoid(slope * logit(p_raw) + intercept)
    where p_raw is the raw model score.  We show the population mean predicted
    probability as a single reference dot, the calibration line, and the
    perfect calibration diagonal.
    """
    # Cross-country rows
    xfer = df[df["Evaluation"].str.startswith("Cross-Country")].copy()

    configs = [
        ("Nepal",    "Zero-Shot",   "BDHS Transfer (Zero-Shot)",   "(a) Nepal 2022 — Zero-Shot"),
        ("Nepal",    "Local Calib", "BDHS Transfer + Local Calib", "(b) Nepal 2022 — Local Recalibration"),
        ("Ethiopia", "Zero-Shot",   "BDHS Transfer (Zero-Shot)",   "(c) Ethiopia 2024–25 — Zero-Shot"),
        ("Ethiopia", "Local Calib", "BDHS Transfer + Local Calib", "(d) Ethiopia 2024–25 — Local Recalibration"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.flatten()

    # Colour for country
    ctry_color = {"Nepal": NEPAL_COLOR, "Ethiopia": ETHIOPIA_COLOR}

    for ax, (country, variant, model_name, title) in zip(axes, configs):
        row = xfer[(xfer["Country"] == country) & (xfer["Model"] == model_name)]
        if row.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        row = row.iloc[0]

        slope     = row["Cal_Slope"]
        intercept = row["Cal_Intercept"]
        prev_w    = row["Prev_W_Pct"] / 100.0   # weighted prevalence
        auroc     = row["AUROC_W"]
        brier     = row["Brier_W"]

        col = ctry_color[country]

        # ─ Perfect calibration diagonal ─
        p_range = np.linspace(0.001, 0.25, 300)
        ax.plot(p_range, p_range, "--", color="gray", lw=1, label="Perfect calibration")

        # ─ Calibration line from slope/intercept (logit space) ─
        logit_p = np.log(p_range / (1 - p_range))
        logit_cal = slope * logit_p + intercept
        p_cal = 1 / (1 + np.exp(-logit_cal))
        ax.plot(p_range, p_cal, "-", color=col, lw=2,
                label=f"Calibration line\n(slope={slope:.3f}, int={intercept:.3f})")

        # ─ Mean predicted probability dot ─
        ax.scatter([prev_w], [prev_w], s=80, zorder=5, color=col,
                   marker="o", label=f"Weighted prevalence ({prev_w:.3f})")

        # ─ Null reference ─
        ax.axhline(prev_w, ls=":", lw=0.8, color="gray", alpha=0.5)
        ax.axvline(prev_w, ls=":", lw=0.8, color="gray", alpha=0.5)

        ax.set_xlim(0, 0.25)
        ax.set_ylim(0, 0.25)
        ax.set_xlabel("Mean Predicted Probability", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Observed Event Rate", fontsize=FONT_SIZE_LABEL)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(ls=":", alpha=0.4)

        # ─ Annotation ─
        textstr = f"AUC = {auroc:.3f}\nBrier = {brier:.4f}"
        ax.text(0.97, 0.05, textstr, transform=ax.transAxes,
                fontsize=7.5, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.suptitle(
        "Fig 7. Cross-Country Transfer Calibration: Bangladesh→Nepal and Bangladesh→Ethiopia\n"
        "Comparison of zero-shot and locally recalibrated BDHS 2004–2014 model",
        fontsize=FONT_SIZE_TITLE, y=1.01
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig7_external_calibration.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Fig 6: External AUC comparison …")
    make_fig6()
    print("Generating Fig 7: External calibration curves …")
    make_fig7()
    print("Done.")
