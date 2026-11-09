"""
Diagnostic script for Final Probability Calibration and Weight Dominance.
Computes all statistics requested in User Request:
1. Probability scale before/after Module C, quantiles, prevalence, Brier, calibration metrics.
2. Module A wave dominance table, component decomposition, and equal-wave-weight sensitivity analysis.
3. Rolling-origin threshold stability across all 5 cuts.
"""
import sys
sys.path.insert(0, '/Users/rezaul1424/Everything/SWIFT-U5')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.models.swift_u5 import SWIFTU5Estimator
from src.features.build_features import LeakageFreePreprocessor, TARGET_COLUMN, WEIGHT_COLUMN, DIVISION_COLUMN
from src.validation.splits import ROUND_COLUMN, get_forward_chaining_cuts, encode_features, compute_normalized_weights
from src.evaluation.metrics import (
    compute_all_metrics, compute_youden_threshold_from_validation,
    _calibration_slope_intercept, _expected_calibration_error,
    _weighted_brier_score, _null_brier_score
)
from src.models.module_a_weight import ModuleARecencyShiftWeighter, ROUND_YEARS
from src.models.baselines import build_weighted_xgboost, compute_scale_pos_weight
from src.validation.experiment_2x2 import _derive_oos_threshold_temporal

df = pd.read_parquet('data/processed/u5m_harmonized_cohort.parquet')

# ==============================================================================
# SECTION 1: Prediction Probability Scale Diagnostics
# ==============================================================================
print("=" * 80)
print("SECTION 1: PREDICTION PROBABILITY SCALE DIAGNOSTICS")
print("=" * 80)

dev_rounds = ['2004', '2007', '2011', '2014']
df_dev = df[df[ROUND_COLUMN].isin(dev_rounds)].copy()
df_val = df[df[ROUND_COLUMN] == '2017-18'].copy()
df_test = df[df[ROUND_COLUMN] == '2022'].copy()

y_test = df_test[TARGET_COLUMN].values.astype(int)
w_test = df_test[WEIGHT_COLUMN].values.astype(float)
obs_prev_uw = float(np.mean(y_test))
obs_prev_w = float(np.average(y_test, weights=w_test))
print(f"Test 2022 Observed Prevalence: Unweighted = {obs_prev_uw:.4f} ({obs_prev_uw*100:.2f}%), Weighted = {obs_prev_w:.4f} ({obs_prev_w*100:.2f}%)")

# Fit SWIFT-U5 Primary Model
swift_primary = SWIFTU5Estimator(
    use_module_a=True, use_module_b=True, use_module_c=True,
    n_estimators=100, random_state=42
)
swift_primary.fit(df_train=df_dev, target_round='2022', df_target_unlabeled=df_test, df_val=df_val)

# Extract stage predictions on 2022 test
X_test_trans, _ = swift_primary.preprocessor.transform(df_test)
div_test = df_test[DIVISION_COLUMN].astype(str)

p_stage1_base = swift_primary.base_model_.predict_proba(X_test_trans)[:, 1]
p_stage2_mod_b = swift_primary.module_b.predict_proba(p_stage1_base, div_test)
p_stage3_final = swift_primary.predict_proba(df_test) # has Module C

# Also extract validation predictions for threshold freezing
p_val_stage1_base = swift_primary.base_model_.predict_proba(swift_primary.preprocessor.transform(df_val)[0])[:, 1]
p_val_final = swift_primary.predict_proba(df_val)
y_val = df_val[TARGET_COLUMN].values.astype(int)
w_val = df_val[WEIGHT_COLUMN].values.astype(float)

thr_w_stage1, thr_uw_stage1 = compute_youden_threshold_from_validation(y_val, p_val_stage1_base, w_val)
thr_w_final, thr_uw_final = compute_youden_threshold_from_validation(y_val, p_val_final, w_val)

# Also fit Baseline XGBoost without SWIFT modules (standard baseline)
prep_base = LeakageFreePreprocessor()
X_dev_b, _ = prep_base.fit_transform(df_dev)
X_test_b, _ = prep_base.transform(df_test)
y_dev = df_dev[TARGET_COLUMN].values.astype(int)
w_dev = df_dev[WEIGHT_COLUMN].values.astype(float)
w_dev_n = w_dev / w_dev.mean()
spw_dev = compute_scale_pos_weight(y_dev, w_dev_n)
base_xgb = build_weighted_xgboost(scale_pos_weight=spw_dev)
base_xgb.fit(X_dev_b, y_dev, sample_weight=w_dev_n)
p_base_xgb_test = base_xgb.predict_proba(X_test_b)[:, 1]

# Also fit Refitted Baseline XGBoost (2004-2017-18)
df_full = df[df[ROUND_COLUMN].isin(['2004', '2007', '2011', '2014', '2017-18'])].copy()
prep_full = LeakageFreePreprocessor()
X_full, _ = prep_full.fit_transform(df_full)
X_test_full, _ = prep_full.transform(df_test)
y_full = df_full[TARGET_COLUMN].values.astype(int)
w_full = df_full[WEIGHT_COLUMN].values.astype(float)
w_full_n = w_full / w_full.mean()
spw_full = compute_scale_pos_weight(y_full, w_full_n)
refit_xgb = build_weighted_xgboost(scale_pos_weight=spw_full)
refit_xgb.fit(X_full, y_full, sample_weight=w_full_n)
p_refit_xgb_test = refit_xgb.predict_proba(X_test_full)[:, 1]

def summarize_probs(name, p, y_true, w, frozen_thr_w=None):
    quantiles = np.percentile(p, [1, 5, 25, 50, 75, 95, 99])
    mean_p = float(np.mean(p))
    median_p = float(np.median(p))
    min_p = float(np.min(p))
    max_p = float(np.max(p))
    
    brier = _weighted_brier_score(y_true, p, w)
    null_b = _null_brier_score(y_true, w)
    bss = 1.0 - brier / null_b if null_b > 0 else np.nan
    slope, intercept = _calibration_slope_intercept(y_true, p, w)
    ece = _expected_calibration_error(y_true, p, w)
    
    return {
        'name': name,
        'mean': mean_p,
        'median': median_p,
        'min': min_p,
        'max': max_p,
        'p1': quantiles[0],
        'p5': quantiles[1],
        'p25': quantiles[2],
        'p50': quantiles[3],
        'p75': quantiles[4],
        'p95': quantiles[5],
        'p99': quantiles[6],
        'brier_w': brier,
        'null_brier_w': null_b,
        'bss_w': bss,
        'cal_slope': slope,
        'cal_intercept': intercept,
        'ece_w': ece,
        'obs_prev_w': obs_prev_w,
        'ratio_mean_to_prev': mean_p / obs_prev_w,
        'frozen_thr_w': frozen_thr_w
    }

tracks = [
    ("SWIFT-U5 Base Stage (pre-Module B/C, w/ scale_pos_weight)", p_stage1_base, y_test, w_test, thr_w_stage1),
    ("SWIFT-U5 Post-Module B (Division EB shrunk, pre-Module C)", p_stage2_mod_b, y_test, w_test, None),
    ("SWIFT-U5 FINAL (Post-Module C Drift Platt Recalibrated)", p_stage3_final, y_test, w_test, thr_w_final),
    ("Baseline XGBoost 2004-14 (uncalibrated w/ scale_pos_weight)", p_base_xgb_test, y_test, w_test, None),
    ("Baseline XGBoost 2004-17-18 (refitted, uncalibrated)", p_refit_xgb_test, y_test, w_test, None),
]

summary_rows = [summarize_probs(name, p, y, w, thr) for name, p, y, w, thr in tracks]
sum_df = pd.DataFrame(summary_rows)

print("\n--- TABLE 1A: Probability Distribution & Scale ---")
cols_dist = ['name', 'mean', 'median', 'min', 'max', 'p5', 'p25', 'p50', 'p75', 'p95', 'ratio_mean_to_prev']
print(sum_df[cols_dist].to_string(index=False))

print("\n--- TABLE 1B: Calibration & Brier Diagnostics (Weighted) ---")
cols_cal = ['name', 'brier_w', 'null_brier_w', 'bss_w', 'cal_slope', 'cal_intercept', 'ece_w', 'frozen_thr_w']
print(sum_df[cols_cal].to_string(index=False))

# Check Module C recalibration parameters fitted on 2017-18
mc = swift_primary.module_c
print(f"\nModule C Fitted Recalibration Parameters on 2017-18 Val:")
print(f"  Recalibration Intercept (b*): {mc.recalib_intercept_:.4f}")
print(f"  Recalibration Slope     (a*): {mc.recalib_slope_:.4f}")
print(f"  Historical recalibrations dict: {mc.historical_calibrations_}")

# ==============================================================================
# SECTION 2: Module A Wave Dominance Diagnostic
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 2: MODULE A WAVE DOMINANCE DIAGNOSTIC")
print("=" * 80)

ma_primary = swift_primary.module_a

# Deconstruct component weights for df_dev
n_obs_dev = len(df_dev)

# 1. Survey weight (within-round normalized)
w_survey = np.zeros(n_obs_dev, dtype=float)
for rnd, grp_idx in df_dev.groupby('round').groups.items():
    raw_w = df_dev.loc[grp_idx, 'sample_weight'].values.astype(float)
    w_survey[df_dev.index.get_indexer(grp_idx)] = raw_w / raw_w.mean()

# 2. Time decay weight
target_year = ROUND_YEARS.get('2022', 2022.0)
round_years_arr = df_dev['round'].map(ROUND_YEARS).fillna(target_year).values.astype(float)
time_delta = target_year - round_years_arr
w_time = np.exp(-ma_primary.decay_rate * np.maximum(0.0, time_delta))

# 3. Density ratio
probs_dr = ma_primary.domain_clf_.predict_proba(X_dev_b)[:, 1]
probs_dr = np.clip(probs_dr, 1e-4, 1.0 - 1e-4)
ratios = (probs_dr / (1.0 - probs_dr)) / max(1e-4, ma_primary.density_ratio_mean_)
w_shift = np.clip(ratios, ma_primary.clip_min, ma_primary.clip_max)

# 4. Composite pre-clip and post-clip
W_raw = w_survey * w_time * w_shift
clip_val = ma_primary.composite_clip_value_
W_clipped = np.minimum(W_raw, clip_val)
W_final = W_clipped / W_clipped.mean()

df_dev_diag = df_dev[['round', 'sample_weight', TARGET_COLUMN]].copy()
df_dev_diag['w_survey'] = w_survey
df_dev_diag['w_time'] = w_time
df_dev_diag['w_shift'] = w_shift
df_dev_diag['W_final'] = W_final
df_dev_diag['W_final_sq'] = W_final ** 2

tot_W = W_final.sum()
tot_W_sq = (W_final ** 2).sum()
pooled_ess_prim = (tot_W ** 2) / tot_W_sq

wave_rows = []
for rnd in dev_rounds:
    sub = df_dev_diag[df_dev_diag['round'] == rnd]
    n_rec = len(sub)
    mean_wsurv = sub['w_survey'].mean()
    mean_wtime = sub['w_time'].mean()
    mean_wshift = sub['w_shift'].mean()
    mean_wcomp = sub['W_final'].mean()
    sum_wcomp = sub['W_final'].sum()
    sum_wcomp_sq = sub['W_final_sq'].sum()
    wave_ess = (sum_wcomp ** 2) / sum_wcomp_sq
    pct_sq_contrib = 100.0 * (sum_wcomp_sq / tot_W_sq)
    pct_tot_w_contrib = 100.0 * (sum_wcomp / tot_W)
    
    wave_rows.append({
        'wave': rnd,
        'records': n_rec,
        'mean_w_survey': mean_wsurv,
        'mean_w_time': mean_wtime,
        'mean_w_shift': mean_wshift,
        'mean_w_composite': mean_wcomp,
        'sum_w_composite': sum_wcomp,
        'sum_w_sq': sum_wcomp_sq,
        'wave_ess': wave_ess,
        'pct_sq_contrib': pct_sq_contrib,
        'pct_w_contrib': pct_tot_w_contrib,
    })

wave_df = pd.DataFrame(wave_rows)
print("\n--- TABLE 2A: Module A Decomposition Across Development Waves ---")
cols_w = ['wave', 'records', 'mean_w_survey', 'mean_w_time', 'mean_w_shift', 'mean_w_composite', 'pct_w_contrib', 'pct_sq_contrib', 'wave_ess']
print(wave_df[cols_w].to_string(index=False))

print(f"\nPooled Development Records: {n_obs_dev:,}")
print(f"Total Composite Weight Sum: {tot_W:,.1f}")
print(f"Total Composite Squared Weight Sum: {tot_W_sq:,.1f}")
print(f"Primary Pooled ESS: {pooled_ess_prim:.1f} ({pooled_ess_prim/n_obs_dev*100:.1f}% of nominal N)")
print(f"Sum of Wave ESS values: {wave_df['wave_ess'].sum():.1f}")

# Diagnostic Alternative: Equal Wave Weighting Sensitivity Analysis
# In this alternative, within-wave composite weights are normalized such that each of the 4 waves
# contributes exactly N_dev / 4 (25%) of total weight.
W_equal_wave = np.zeros(n_obs_dev, dtype=float)
target_per_wave = n_obs_dev / len(dev_rounds)
for rnd in dev_rounds:
    mask = (df_dev_diag['round'] == rnd).values
    w_wave = W_final[mask]
    W_equal_wave[mask] = w_wave * (target_per_wave / w_wave.sum())

tot_W_eq = W_equal_wave.sum()
tot_W_sq_eq = (W_equal_wave ** 2).sum()
pooled_ess_eq = (tot_W_eq ** 2) / tot_W_sq_eq

wgt_prev_prim = float(np.average(y_dev, weights=W_final))
wgt_prev_eq = float(np.average(y_dev, weights=W_equal_wave))

# Fit XGBoost with Equal Wave Weights on Dev, evaluate on 2017-18 Val
spw_eq = compute_scale_pos_weight(y_dev, W_equal_wave)
mdl_eq = build_weighted_xgboost(scale_pos_weight=spw_eq)
mdl_eq.fit(X_dev_b, y_dev, sample_weight=W_equal_wave)
X_val_b, _ = prep_base.transform(df_val)
p_val_eq = mdl_eq.predict_proba(X_val_b)[:, 1]

m_val_prim = compute_all_metrics(y_val, base_xgb.predict_proba(X_val_b)[:, 1], sample_weight=w_val, compute_calibration=False)
m_val_eq = compute_all_metrics(y_val, p_val_eq, sample_weight=w_val, compute_calibration=False)

print("\n--- TABLE 2B: Primary Weighting vs. Equal-Wave Diagnostic Alternative ---")
comp_rows = [
    {
        'Regime': 'Primary Module A (Recency + Shift)',
        'Pooled_ESS': pooled_ess_prim,
        'ESS_pct_N': pooled_ess_prim / n_obs_dev * 100,
        'W_max': W_final.max(),
        'W_std': W_final.std(),
        'Wgt_Dev_Prev': wgt_prev_prim,
        'Val_AUROC_w': m_val_prim['auroc_w'],
        'Val_AUPRC_w': m_val_prim['auprc_w'],
        '2014_pct_weight': wave_df.loc[wave_df['wave']=='2014', 'pct_w_contrib'].values[0],
        '2014_pct_sq_weight': wave_df.loc[wave_df['wave']=='2014', 'pct_sq_contrib'].values[0]
    },
    {
        'Regime': 'Diagnostic Alternative (Equal-Wave Composite)',
        'Pooled_ESS': pooled_ess_eq,
        'ESS_pct_N': pooled_ess_eq / n_obs_dev * 100,
        'W_max': W_equal_wave.max(),
        'W_std': W_equal_wave.std(),
        'Wgt_Dev_Prev': wgt_prev_eq,
        'Val_AUROC_w': m_val_eq['auroc_w'],
        'Val_AUPRC_w': m_val_eq['auprc_w'],
        '2014_pct_weight': 25.0,
        '2014_pct_sq_weight': 100.0 * ((W_equal_wave[df_dev_diag['round']=='2014']**2).sum() / tot_W_sq_eq)
    }
]
print(pd.DataFrame(comp_rows).to_string(index=False))

# ==============================================================================
# SECTION 3: Rolling-Origin Threshold Stability Across All 5 Transitions
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION 3: ROLLING-ORIGIN THRESHOLD STABILITY ACROSS ALL 5 TRANSITIONS")
print("=" * 80)

cuts = get_forward_chaining_cuts(df)
ro_rows = []

for idx, cut in enumerate(cuts):
    cut_id = f"C{idx+1}"
    tr_rounds = cut['train_rounds']
    tst_round = cut['test_round']
    train_idx = cut['train_idx']
    test_idx = cut['test_idx']
    
    # 1. Threshold derivation procedure
    thr_w, thr_uw, prov = _derive_oos_threshold_temporal(df, train_idx, tr_rounds, 'xgboost', use_weights=True)
    
    # 2. Main model fit on full training rounds
    df_cut_train = df.loc[train_idx].copy()
    df_cut_test = df.loc[test_idx].copy()
    
    X_cut_all, _ = encode_features(df, train_idx)
    X_cut_tr = X_cut_all.loc[train_idx].values
    X_cut_ts = X_cut_all.loc[test_idx].values
    y_cut_tr = df_cut_train[TARGET_COLUMN].values.astype(int)
    y_cut_ts = df_cut_test[TARGET_COLUMN].values.astype(int)
    w_cut_tr = compute_normalized_weights(df_cut_train, train_idx, use_weights=True)
    w_cut_ts = df_cut_test[WEIGHT_COLUMN].values.astype(float)
    
    spw_cut = compute_scale_pos_weight(y_cut_tr, w_cut_tr)
    main_cut_mdl = build_weighted_xgboost(scale_pos_weight=spw_cut)
    main_cut_mdl.fit(X_cut_tr, y_cut_tr, sample_weight=w_cut_tr)
    
    # Raw predictions on test
    p_test_raw = main_cut_mdl.predict_proba(X_cut_ts)[:, 1]
    
    # Calibration diagnostics of raw predictions on test
    slope_raw, int_raw = _calibration_slope_intercept(y_cut_ts, p_test_raw, w_cut_ts)
    brier_raw = _weighted_brier_score(y_cut_ts, p_test_raw, w_cut_ts)
    
    # Validation wave details
    last_tr = tr_rounds[-1]
    df_val_wave = df[df[ROUND_COLUMN] == last_tr]
    w_val_wave = df_val_wave[WEIGHT_COLUMN].values.astype(float)
    y_val_wave = df_val_wave[TARGET_COLUMN].values.astype(int)
    val_prev_w = float(np.average(y_val_wave, weights=w_val_wave))
    
    # Did threshold derivation involve refitting?
    involves_refit = (len(tr_rounds) >= 2)
    
    ro_rows.append({
        'Cut': cut_id,
        'Train_Waves': "+".join(tr_rounds),
        'Test_Wave': tst_round,
        'Threshold_Derivation': prov.split(':')[1].split('(')[0].strip() if ':' in prov else prov,
        'Frozen_Thr_W': thr_w,
        'Frozen_Thr_UW': thr_uw,
        'Val_Prev_W': val_prev_w,
        'Test_Mean_Prob': p_test_raw.mean(),
        'Test_Median_Prob': np.median(p_test_raw),
        'Test_Obs_Prev_W': np.average(y_cut_ts, weights=w_cut_ts),
        'Cal_Slope': slope_raw,
        'Cal_Int': int_raw,
        'Brier_W': brier_raw,
        'Threshold_Transferred_To_Refit': involves_refit
    })

ro_df = pd.DataFrame(ro_rows)
print(ro_df[['Cut', 'Train_Waves', 'Test_Wave', 'Frozen_Thr_W', 'Val_Prev_W', 'Test_Mean_Prob', 'Test_Obs_Prev_W', 'Cal_Slope', 'Cal_Int', 'Threshold_Transferred_To_Refit']].to_string(index=False))

print("\nDiagnostic complete.")
