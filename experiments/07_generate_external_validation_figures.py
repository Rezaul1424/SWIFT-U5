"""
experiments/07_generate_external_validation_figures.py
=======================================================
Generates two publication-quality figures for the external multi-country
validation section (§5.5.5):

  fig6_external_auroc_comparison.png
      2-panel dot-and-CI AUROC comparison across Track A, Track B, and
      Cross-Country Transfer for Nepal DHS 2022 and Ethiopia DHS 2024-25.

  fig7_external_calibration.png
      2x2 empirical weighted decile calibration plot comparing Zero-Shot vs.
      Locally Recalibrated cross-country transfer (BDHS 2004-2014 -> Nepal
      2022 and Ethiopia 2024-25), matching the binning methodology of Fig 2.

Usage:
  python3 experiments/07_generate_external_validation_figures.py
"""

import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.utils.config import PROCESSED_DATA_DIR
RANDOM_STATE = 42
from src.features.build_features import (
    CANDIDATE_FEATURES, TARGET_COLUMN, WEIGHT_COLUMN, DIVISION_COLUMN
)
from src.models.baselines import compute_scale_pos_weight
from src.models.module_c_recalib import ModuleCDriftRecalibrator
from src.evaluation.metrics import compute_all_metrics

# Import exact country loader & preprocessor from 06_run_external_validation
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ext_val", ROOT / "experiments" / "06_run_external_validation.py"
)
ext_val = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext_val)
ConsistentPreprocessor = ext_val.ConsistentPreprocessor
load_country_cohort = ext_val.load_country_cohort

CSV_PATH = ROOT / "outputs" / "tables" / "table8_external_validation.csv"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour / typography palette ──────────────────────────────────────────────
NEPAL_COLOR    = "#2166AC"
ETHIOPIA_COLOR = "#D6604D"
TRACK_A_COLOR  = "#2E8B57"
TRACK_A_SUB1   = "#66C2A5"
TRACK_A_SUB2   = "#4393C3"
TRACK_B_COLOR  = "#E66101"
XFER_COLOR_ZS  = "#7B3294"
XFER_COLOR_LC  = "#008837"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       9,
    "axes.titlesize":  10.5,
    "axes.labelsize":  9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})


def parse_ci(ci_str):
    if not isinstance(ci_str, str):
        return np.nan, np.nan
    parts = ci_str.strip("[]").split(",")
    if len(parts) != 2:
        return np.nan, np.nan
    try:
        return float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        return np.nan, np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 1. AUROC Comparison Panel (fig6_external_auroc_comparison.png)
# ══════════════════════════════════════════════════════════════════════════════

def _build_fig6_data(df, country):
    rows = df[df["Country"] == country]

    def pick(ev_prefix, model_exact):
        sub = rows[rows["Evaluation"].str.startswith(ev_prefix) & (rows["Model"] == model_exact)]
        return sub.iloc[0]

    ta_full = pick("Track A", "SWIFT-U5 (Full)")
    ta_noa  = pick("Track A", "SWIFT-U5 w/o Module A")
    ta_xgb  = pick("Track A", "Weighted XGBoost")
    tb_xgb  = pick("Track B", "Weighted XGBoost (Random)")
    tr_zs   = pick("Cross-Country", "BDHS Transfer (Zero-Shot)")
    tr_lc   = pick("Cross-Country", "BDHS Transfer + Local Calib")

    return [
        ("SWIFT-U5 (Full)\n[Track A]",     ta_full["AUROC_W"], ta_full["CI_lo"], ta_full["CI_hi"], TRACK_A_COLOR, "o"),
        ("SWIFT-U5 w/o Mod. A\n[Track A]", ta_noa["AUROC_W"],  ta_noa["CI_lo"],  ta_noa["CI_hi"],  TRACK_A_SUB1,  "s"),
        ("XGBoost Baseline\n[Track A]",    ta_xgb["AUROC_W"],  ta_xgb["CI_lo"],  ta_xgb["CI_hi"],  TRACK_A_SUB2,  "^"),
        ("XGBoost (Random)\n[Track B]",    tb_xgb["AUROC_W"],  tb_xgb["CI_lo"],  tb_xgb["CI_hi"],  TRACK_B_COLOR, "D"),
        ("BDHS Transfer\n(Zero-Shot)",     tr_zs["AUROC_W"],   tr_zs["CI_lo"],   tr_zs["CI_hi"],   XFER_COLOR_ZS, "P"),
        ("BDHS Transfer\n(Local Calib)",   tr_lc["AUROC_W"],   tr_lc["CI_lo"],   tr_lc["CI_hi"],   XFER_COLOR_LC, "*"),
    ]


def make_fig6(df):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=False)
    panel_titles = [
        ("Nepal",    "(a) Nepal DHS 2022 (Locked)"),
        ("Ethiopia", "(b) Ethiopia DHS 2024–25 (Locked)"),
    ]

    for ax, (country, title) in zip(axes, panel_titles):
        data = _build_fig6_data(df, country)
        y_pos = list(range(len(data)))[::-1]

        # Subtle horizontal group separators (Track A | Track B | Cross-Country)
        ax.axhline(2.5, ls="-", lw=0.7, color="#CCCCCC", zorder=1)
        ax.axhline(1.5, ls="-", lw=0.7, color="#CCCCCC", zorder=1)

        for i, (label, auc, lo, hi, col, mkr) in enumerate(data):
            yp = y_pos[i]
            err_lo = auc - lo if not np.isnan(lo) else 0
            err_hi = hi - auc if not np.isnan(hi) else 0
            msize = 10.5 if mkr == "*" else 8
            ax.errorbar(
                auc, yp, xerr=[[err_lo], [err_hi]],
                fmt=mkr, color=col, markersize=msize,
                ecolor=col, elinewidth=1.6, capsize=4, zorder=4
            )
            # Place text cleanly ABOVE marker so bottom label never collides with x-axis
            ax.text(
                auc, yp + 0.18, f"{auc:.3f} [{lo:.3f}, {hi:.3f}]",
                ha="center", va="bottom", fontsize=7.2, color="#222222",
                fontweight="medium"
            )

        ax.axvline(0.5, ls="--", lw=0.9, color="grey", alpha=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([d[0] for d in data], fontsize=8.2)
        ax.set_ylim(-0.55, len(data) - 0.25)
        ax.set_xlim(0.47, 0.79)
        ax.set_xlabel("Survey-Weighted ROC-AUC (95% CI)")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.grid(axis="x", ls=":", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TRACK_A_COLOR, markersize=8.5, label="SWIFT-U5 Full [Track A]"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TRACK_A_SUB1,  markersize=8.5, label="SWIFT-U5 w/o Module A [Track A]"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=TRACK_A_SUB2,  markersize=8.5, label="XGBoost Baseline [Track A]"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=TRACK_B_COLOR, markersize=8.5, label="XGBoost – Random Split [Track B]"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor=XFER_COLOR_ZS, markersize=8.5, label="BDHS Transfer – Zero-Shot"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=XFER_COLOR_LC, markersize=10.5, label="BDHS Transfer – Local Recalibration"),
        Line2D([0], [0], ls="--", color="grey", label="Chance (0.50)"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4,
        fontsize=7.8, frameon=False, bbox_to_anchor=(0.5, -0.06)
    )
    fig.suptitle(
        "External Multi-Country Validation: Survey-Weighted ROC-AUC by Evaluation Design",
        fontsize=11.5, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out = FIG_DIR / "fig6_external_auroc_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Empirical Cross-Country Calibration Curves (fig7_external_calibration.png)
# ══════════════════════════════════════════════════════════════════════════════

def compute_cross_country_predictions(table8_df):
    """
    Reproduce deterministic cross-country transfer probability vectors
    (BDHS 2004-2014 -> Nepal 2022 & Ethiopia 2024-25) and assert exact agreement
    with locked table8_external_validation.csv metrics.
    """
    df_nepal = load_country_cohort("nepal", ["2006", "2011", "2016", "2022"])
    df_ethiopia = load_country_cohort("ethiopia", ["2005", "2011", "2016", "2019", "2024-25"])

    df_bd = pd.read_parquet(PROCESSED_DATA_DIR / "u5m_harmonized_cohort.parquet")
    df_bd_dev = df_bd[df_bd["round"].isin(["2004", "2007", "2011", "2014"])].copy().reset_index(drop=True)
    df_bd_val = df_bd[df_bd["round"] == "2017-18"].copy().reset_index(drop=True)

    df_bd_dev_nd = df_bd_dev.drop(columns=[DIVISION_COLUMN])
    df_bd_val_nd = df_bd_val.drop(columns=[DIVISION_COLUMN])

    prep = ConsistentPreprocessor(include_missing_flags=True)
    X_dev, _ = prep.fit_transform(df_bd_dev_nd, CANDIDATE_FEATURES)
    X_bd_val, _ = prep.transform(df_bd_val_nd, CANDIDATE_FEATURES)

    y_dev = df_bd_dev[TARGET_COLUMN].values.astype(int)
    w_dev = df_bd_dev[WEIGHT_COLUMN].values.astype(float)
    w_dev_norm = w_dev / w_dev.mean()
    spw = compute_scale_pos_weight(y_dev, w_dev_norm)

    clf = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
    )
    clf.fit(X_dev, y_dev, sample_weight=w_dev_norm)

    p_bd_val_raw = clf.predict_proba(X_bd_val)[:, 1]
    w_bd_val = df_bd_val[WEIGHT_COLUMN].values.astype(float)
    w_bd_val = w_bd_val / w_bd_val.mean()
    mod_c_bd = ModuleCDriftRecalibrator(use_temporal_recalibration=True)
    mod_c_bd.fit_recalibration(
        df_bd_val[TARGET_COLUMN].values.astype(int),
        p_bd_val_raw, w_bd_val, round_label="2017-18"
    )

    outputs = {}
    targets = [
        ("Nepal", df_nepal, "2016", "2022"),
        ("Ethiopia", df_ethiopia, "2019", "2024-25"),
    ]
    for cname, df_c, val_rnd, test_rnd in targets:
        df_c_val = df_c[df_c["round"] == val_rnd].drop(columns=[DIVISION_COLUMN]).reset_index(drop=True)
        df_c_test = df_c[df_c["round"] == test_rnd].copy().reset_index(drop=True)
        df_c_test_nd = df_c_test.drop(columns=[DIVISION_COLUMN])

        X_c_val, _ = prep.transform(df_c_val, CANDIDATE_FEATURES)
        X_c_test, _ = prep.transform(df_c_test_nd, CANDIDATE_FEATURES)

        p_raw_val = clf.predict_proba(X_c_val)[:, 1]
        p_raw_test = clf.predict_proba(X_c_test)[:, 1]

        p_zero = mod_c_bd.predict_proba(p_raw_test)

        mod_c_loc = ModuleCDriftRecalibrator(use_temporal_recalibration=True)
        w_c_val = df_c_val[WEIGHT_COLUMN].values.astype(float)
        w_c_val = w_c_val / w_c_val.mean()
        mod_c_loc.fit_recalibration(
            df_c_val[TARGET_COLUMN].values.astype(int),
            p_raw_val, w_c_val, round_label=val_rnd
        )
        p_loc = mod_c_loc.predict_proba(p_raw_test)

        y_test = df_c_test[TARGET_COLUMN].values.astype(int)
        w_test = df_c_test[WEIGHT_COLUMN].values.astype(float)

        for variant_key, model_label, p_vec in [
            ("Zero-Shot",   "BDHS Transfer (Zero-Shot)",   p_zero),
            ("Local Calib", "BDHS Transfer + Local Calib", p_loc),
        ]:
            m = compute_all_metrics(
                y_test, p_vec, sample_weight=w_test,
                compute_unweighted=False, compute_calibration=True
            )
            ref = table8_df[
                (table8_df["Country"] == cname) &
                (table8_df["Model"] == model_label)
            ].iloc[0]
            assert abs(round(m["auroc_w"], 3) - ref["AUROC_W"]) < 1e-6, f"AUROC mismatch {cname} {variant_key}"
            assert abs(round(m["cal_slope_w"], 3) - ref["Cal_Slope"]) < 1e-6, f"Slope mismatch {cname} {variant_key}"
            assert abs(round(m["brier_w"], 4) - ref["Brier_W"]) < 1e-6, f"Brier mismatch {cname} {variant_key}"
            outputs[(cname, variant_key)] = {
                "y_test": y_test,
                "w_test": w_test,
                "p_pred": p_vec,
                "metrics": m,
            }
            print(f"  Verified {cname} {variant_key}: AUC={m['auroc_w']:.3f}, Slope={m['cal_slope_w']:.3f}, Int={m['cal_intercept_w']:.3f}, Brier={m['brier_w']:.4f}")

    return outputs


def make_fig7(pred_dict):
    configs = [
        ("Nepal",    "Zero-Shot",   "(a) Nepal 2022 – Zero-Shot Transfer",          NEPAL_COLOR,    0.11),
        ("Nepal",    "Local Calib", "(b) Nepal 2022 – Local Recalibration (Mod. C)", NEPAL_COLOR,    0.11),
        ("Ethiopia", "Zero-Shot",   "(c) Ethiopia 2024–25 – Zero-Shot Transfer",     ETHIOPIA_COLOR, 0.14),
        ("Ethiopia", "Local Calib", "(d) Ethiopia 2024–25 – Local Recalibration (Mod. C)", ETHIOPIA_COLOR, 0.14),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.2))
    axes = axes.flatten()

    for ax, (country, variant, title, col, axis_max) in zip(axes, configs):
        d = pred_dict[(country, variant)]
        y_t = d["y_test"]
        w_t = d["w_test"]
        p_t = d["p_pred"]
        m = d["metrics"]

        # Empirical 10-quantile survey-weighted calibration curve (identical to Fig 2)
        n_bins = 10
        order = np.argsort(p_t)
        y_s, p_s, w_s = y_t[order], p_t[order], w_t[order]
        bins = np.array_split(np.arange(len(y_s)), n_bins)
        mean_pred = np.array([np.average(p_s[b], weights=w_s[b]) for b in bins if len(b) > 0])
        mean_obs  = np.array([np.average(y_s[b], weights=w_s[b]) for b in bins if len(b) > 0])

        hi_lim = max(axis_max, float(mean_pred.max()) * 1.12, float(mean_obs.max()) * 1.12)

        ax.plot([0, hi_lim], [0, hi_lim], "k--", lw=1.3, label="Perfect calibration")
        ax.scatter(mean_pred, mean_obs, color=col, s=55, zorder=5, edgecolor="white", linewidth=0.6)
        ax.plot(mean_pred, mean_obs, color=col, lw=2.0, label=f"{country} ({variant})")

        ax.set_xlim(0, hi_lim)
        ax.set_ylim(0, hi_lim)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Event Fraction")
        ax.set_title(
            f"{title}\nSlope={m['cal_slope_w']:.3f}, Int={m['cal_intercept_w']:.3f}, Brier={m['brier_w']:.4f}",
            fontsize=9.5, fontweight="bold", pad=6
        )
        ax.legend(fontsize=8, loc="upper left", frameon=True, framealpha=0.85)
        ax.grid(True, ls=":", alpha=0.45)
        ax.spines[["top", "right"]].set_visible(False)

        ax.text(
            0.96, 0.06,
            f"ROC-AUC = {m['auroc_w']:.3f}\nNull Brier = {m['brier_null_w']:.4f}",
            transform=ax.transAxes, fontsize=7.8, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#DDDDDD", alpha=0.9)
        )

    fig.suptitle(
        "Cross-Country Transfer Calibration Curves (BDHS 2004–2014 Model; 10 Weighted Quantiles)",
        fontsize=11.5, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out = FIG_DIR / "fig7_external_calibration.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    table8_df = pd.read_csv(CSV_PATH)
    table8_df["CI_lo"], table8_df["CI_hi"] = zip(*table8_df["AUROC_95CI"].map(parse_ci))

    print("Generating Fig 6: External AUROC comparison...")
    make_fig6(table8_df)

    print("Computing exact cross-country decile calibration data...")
    pred_dict = compute_cross_country_predictions(table8_df)

    print("Generating Fig 7: Empirical external calibration curves...")
    make_fig7(pred_dict)
    print("Done.")
