"""
Session 6 External Multi-Country Validation Runner: Nepal DHS & Ethiopia DHS.

Tests SWIFT-U5 generalizability across two independent DHS settings without
modifying any frozen Bangladesh (BDHS) results:
  1. Nepal DHS (4 waves: 2006, 2011, 2016, 2022)
     - Track A Chronological: Dev 2006+2011 -> Val 2016 -> Locked Test 2022
     - Track B Random Diagnostic: 70/30 random split on pooled 2006-2022
  2. Ethiopia DHS (5 waves: 2005, 2011, 2016, 2019, 2024-25)
     - Track A Chronological: Dev 2005+2011+2016 -> Val 2019 -> Locked Test 2024-25
     - Track B Random Diagnostic: 70/30 random split on pooled 2005-2024-25
  3. Cross-Country Transfer (Bangladesh-trained SWIFT-U5 evaluated on
     locked Nepal 2022 & Ethiopia 2024-25 using shared country-invariant features,
     comparing zero-shot vs. locally recalibrated probabilities).

Saves summary tables to `outputs/tables/table8_external_validation.csv` (and `.tex`),
`outputs/tables/table8b_external_cohort_summary.csv`, and
`outputs/reports/session6_external_validation_report.md`.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')
os.environ['MPLCONFIGDIR'] = '/tmp'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(WORKSPACE_DIR))

from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIR, REPORTS_DIR
from src.data.cohort import COLUMNS_TO_LOAD, filter_exposure_complete_cohort
from src.data.harmonize import harmonize_t0_structural, harmonize_t1_antenatal
from src.features.build_features import (
    CANDIDATE_FEATURES, TARGET_COLUMN, WEIGHT_COLUMN, DIVISION_COLUMN,
    LeakageFreePreprocessor
)
from src.models.swift_u5 import SWIFTU5Estimator
from src.models.baselines import (
    build_weighted_xgboost, build_weighted_logistic, build_weighted_random_forest,
    compute_scale_pos_weight
)
from src.models.module_c_recalib import ModuleCDriftRecalibrator
from src.evaluation.metrics import (
    compute_all_metrics, compute_youden_threshold_from_validation, _null_brier_score
)
from src.evaluation.bootstrap import bootstrap_metrics

N_BOOTSTRAP = 1000
RANDOM_STATE = 42


class ConsistentPreprocessor(LeakageFreePreprocessor):
    """Subclass of LeakageFreePreprocessor ensuring train-locked missingness flags across waves/countries."""

    def fit(self, df_train: pd.DataFrame, feature_subset: List[str] = None) -> 'ConsistentPreprocessor':
        super().fit(df_train, feature_subset)
        features = feature_subset if feature_subset is not None else [
            c for c in CANDIDATE_FEATURES if c in self.medians_
        ]
        self.missing_flag_cols_ = []
        if self.include_missing_flags:
            for col in features:
                if col in df_train.columns:
                    s = pd.to_numeric(df_train[col], errors='coerce')
                    if s.isna().sum() > 0:
                        self.missing_flag_cols_.append(col)
        return self

    def transform(self, df: pd.DataFrame, feature_subset: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform!")
        features = feature_subset if feature_subset is not None else [
            col for col in CANDIDATE_FEATURES if col in self.medians_
        ]
        transformed_cols, feature_names = [], []
        for col in features:
            if col not in df.columns:
                col_data = np.full(len(df), self.medians_.get(col, 0.0))
                missing_flag = np.ones(len(df), dtype=float)
            else:
                series = pd.to_numeric(df[col], errors='coerce')
                missing_flag = series.isna().astype(float).values
                col_data = series.fillna(self.medians_.get(col, 0.0)).values
            transformed_cols.append(col_data)
            feature_names.append(col)
            if col in getattr(self, 'missing_flag_cols_', []):
                transformed_cols.append(missing_flag)
                feature_names.append(f"{col}_is_missing")

        if self.division_categories_ and DIVISION_COLUMN in df.columns:
            div_series = df[DIVISION_COLUMN].astype(str)
            for div in self.division_categories_:
                transformed_cols.append((div_series == div).astype(float).values)
                feature_names.append(f"div_{div}")

        X = np.column_stack(transformed_cols).astype(np.float64)
        self.feature_names_ = feature_names
        return X, feature_names


# ---------------------------------------------------------------------------
# Country-specific harmonization helpers
# ---------------------------------------------------------------------------

_NP_EASTERN_P2_DISTRICTS = {'saptari', 'siraha'}
_NP_WESTERN_P5_DISTRICTS = {
    'gulmi', 'palpa', 'nawalparasi', 'nawalparasi west',
    'rupandehi', 'kapilbastu', 'kapilvastu', 'arghakhanchi'
}


def harmonize_nepal_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize Nepal regions across 2006-2022 into 5 Development Regions."""
    df = df.copy()
    rnd = str(df['round'].iloc[0])

    if rnd == '2006':
        reg = df['v024'].astype(str).str.strip().str.lower()
    elif rnd in ('2011', '2016'):
        reg = df['sdevreg'].astype(str).str.strip().str.lower()
    elif rnd == '2022':
        prov = df['v024'].astype(str).str.strip().str.lower()
        dist = df['sdist'].astype(str).str.strip().str.lower()

        def map_2022(p, d):
            if 'koshi' in p or 'province 1' in p:
                return 'eastern'
            if 'madhesh' in p or 'province 2' in p:
                return 'eastern' if d in _NP_EASTERN_P2_DISTRICTS else 'central'
            if 'bagmati' in p or 'province 3' in p:
                return 'central'
            if 'gandaki' in p or 'province 4' in p:
                return 'western'
            if 'lumbini' in p or 'province 5' in p:
                return 'western' if d in _NP_WESTERN_P5_DISTRICTS else 'mid-western'
            if 'karnali' in p or 'province 6' in p:
                return 'mid-western'
            if 'sudurpashchim' in p or 'province 7' in p:
                return 'far-western'
            return p

        reg = [map_2022(p, d) for p, d in zip(prov, dist)]
    else:
        reg = df['v024'].astype(str).str.strip().str.lower()

    df['division_name'] = reg
    df['division_7'] = reg
    df['division_8'] = reg
    return df


_ET_REGION_MAP = {
    'addis abeba': 'addis_ababa',
    'addis ababa': 'addis_ababa',
    'addis adaba': 'addis_ababa',
    'afar': 'afar',
    'affar': 'afar',
    'amhara': 'amhara',
    'ben-gumz': 'benishangul_gumuz',
    'benishangul': 'benishangul_gumuz',
    'benishangul-gumuz': 'benishangul_gumuz',
    'dire dawa': 'dire_dawa',
    'gambela': 'gambela',
    'gambella': 'gambela',
    'harari': 'harari',
    'oromiya': 'oromia',
    'oromia': 'oromia',
    'snnp': 'snnpr',
    'snnpr': 'snnpr',
    # 2024-25 regions carved out of SNNPR merged back for 11-region longitudinal consistency
    'sidama': 'snnpr',
    'central ethiopia': 'snnpr',
    'south ethiopia': 'snnpr',
    'south west ethiopia': 'snnpr',
    'somali': 'somali',
    'tigray': 'tigray',
}


def harmonize_ethiopia_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize Ethiopia regions across 2005-2024-25 into 11 historical regions."""
    df = df.copy()
    raw = df['v024'].astype(str).str.strip().str.lower()
    mapped = raw.map(_ET_REGION_MAP).fillna(raw)
    df['division_name'] = mapped
    df['division_7'] = mapped
    df['division_8'] = mapped
    return df


def harmonize_t2_birth_external(df: pd.DataFrame) -> pd.DataFrame:
    """Extended facility-delivery regex covering Nepal health posts/NGOs and Ethiopia health posts."""
    df = df.copy()

    if 'm15' in df.columns:
        m15_str = df['m15'].astype(str).str.strip().str.lower()
        is_home = m15_str.str.contains('home', regex=False)
        is_facility = m15_str.str.contains(
            'hospital|clinic|center|centre|mcwc|govt|government|medical|medica|hut|'
            'public|private|sector|health post|sub-health post|community health|'
            'facility|facilities|ngo|fpan|umn',
            regex=True
        ) & (~is_home)
        df['facility_delivery'] = np.where(
            is_facility, 1.0,
            np.where(is_home, 0.0, np.nan)
        )
        df['facility_delivery_missing'] = df['facility_delivery'].isna().astype(int)
    else:
        df['facility_delivery'] = np.nan
        df['facility_delivery_missing'] = 1

    if 'm17' in df.columns:
        m17_str = df['m17'].astype(str).str.strip().str.lower()
        df['caesarean_delivery'] = np.where(
            m17_str.isin(['yes', '1', '1.0']),
            1.0,
            np.where(m17_str.isin(['no', '0', '0.0']), 0.0, np.nan)
        )
        df['caesarean_delivery_missing'] = df['caesarean_delivery'].isna().astype(int)
    else:
        df['caesarean_delivery'] = np.nan
        df['caesarean_delivery_missing'] = 1

    if 'm18' in df.columns:
        m18_str = df['m18'].astype(str).str.strip().str.lower()
        size_map = {
            'very small': 1.0,
            'smaller than average': 2.0,
            'average': 3.0,
            'larger than average': 4.0,
            'very large': 5.0
        }
        df['birth_size'] = m18_str.map(size_map)
        df['birth_size_missing'] = df['birth_size'].isna().astype(int)
    else:
        df['birth_size'] = np.nan
        df['birth_size_missing'] = 1

    return df


def load_country_cohort(country: str, rounds: List[str]) -> pd.DataFrame:
    """Load, filter (0-59 mo, leakage-free), and harmonize multi-wave cohort for Nepal or Ethiopia."""
    dfs = []
    extra_cols = ['sdevreg', 'sdist'] if country == 'nepal' else []

    for rnd in rounds:
        round_dir = RAW_DATA_DIR / country / rnd
        files = list(round_dir.glob('*.dta')) + list(round_dir.glob('*.DTA'))
        if not files:
            raise FileNotFoundError(f"No .dta file in {round_dir}")
        stata_file = files[0]

        reader = pd.read_stata(stata_file, iterator=True)
        available = {c.lower(): c for c in reader.variable_labels().keys()}

        req_cols = COLUMNS_TO_LOAD + [c for c in extra_cols if c in available]
        load_cols, rename_map = [], {}
        for c in req_cols:
            if c.lower() in available:
                orig = available[c.lower()]
                load_cols.append(orig)
                rename_map[orig] = c.lower()

        df_raw = pd.read_stata(stata_file, columns=load_cols).rename(columns=rename_map)
        df_raw['round'] = rnd

        cohort = filter_exposure_complete_cohort(df_raw)
        if country == 'nepal':
            cohort = harmonize_nepal_regions(cohort)
        else:
            cohort = harmonize_ethiopia_regions(cohort)
        cohort = harmonize_t0_structural(cohort)
        cohort = harmonize_t1_antenatal(cohort)
        cohort = harmonize_t2_birth_external(cohort)
        cohort['country'] = country
        dfs.append(cohort)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_predictions(df_test: pd.DataFrame,
                         y_prob: np.ndarray,
                         youden_thr_w: float,
                         run_bootstrap: bool = True) -> Tuple[Dict, Dict]:
    """Compute weighted metrics and 1,000-replicate PSU cluster-bootstrap CIs."""
    y_test = df_test[TARGET_COLUMN].values.astype(int)
    w_test = df_test[WEIGHT_COLUMN].values.astype(float)

    metrics = compute_all_metrics(
        y_test, y_prob,
        sample_weight=w_test,
        screening_capacities=[0.05, 0.10, 0.20],
        compute_unweighted=False,
        compute_calibration=True,
        youden_threshold_w=youden_thr_w,
    )
    boot = {}
    if run_bootstrap:
        boot = bootstrap_metrics(
            df_test, y_prob,
            n_replicates=N_BOOTSTRAP,
            psu_col='psu_wave',
            strata_col='strata_wave',
            screening_capacities=[0.05, 0.10, 0.20],
            seed=RANDOM_STATE,
            youden_threshold_w=youden_thr_w,
        )
    return metrics, boot


def run_within_country_suite(country: str,
                             df_all: pd.DataFrame,
                             dev_rounds: List[str],
                             val_round: str,
                             test_round: str) -> List[Dict]:
    """Run full Track A chronological benchmark + ablation + baselines + Track B random diagnostic."""
    df_dev = df_all[df_all['round'].isin(dev_rounds)].copy().reset_index(drop=True)
    df_val = df_all[df_all['round'] == val_round].copy().reset_index(drop=True)
    df_test = df_all[df_all['round'] == test_round].copy().reset_index(drop=True)

    y_val = df_val[TARGET_COLUMN].values.astype(int)
    w_val = df_val[WEIGHT_COLUMN].values.astype(float)
    y_test = df_test[TARGET_COLUMN].values.astype(int)
    w_test = df_test[WEIGHT_COLUMN].values.astype(float)

    print(f"\n--- {country.upper()} Track A ---")
    print(f"  Dev  ({','.join(dev_rounds)}): N={len(df_dev):,}, deaths={df_dev[TARGET_COLUMN].sum():,}")
    print(f"  Val  ({val_round}): N={len(df_val):,}, deaths={df_val[TARGET_COLUMN].sum():,}")
    print(f"  Test ({test_round}): N={len(df_test):,}, deaths={df_test[TARGET_COLUMN].sum():,}")

    rows = []

    def record(setting: str, model_label: str, sub_df: pd.DataFrame,
               y_p: np.ndarray, thr: float, run_boot: bool = True):
        m, b = evaluate_predictions(sub_df, y_p, thr, run_bootstrap=run_boot)
        auc_ci = (
            f"[{b['auroc_w']['ci_lower_95']:.3f}, {b['auroc_w']['ci_upper_95']:.3f}]"
            if 'auroc_w' in b else '--'
        )
        nns_ci = (
            f"[{b['nns_10pct_w']['ci_lower_95']:.1f}, {b['nns_10pct_w']['ci_upper_95']:.1f}]"
            if 'nns_10pct_w' in b else '--'
        )
        sens_ci = (
            f"[{b['sensitivity_10pct_w']['ci_lower_95']*100:.1f}, {b['sensitivity_10pct_w']['ci_upper_95']*100:.1f}]"
            if 'sensitivity_10pct_w' in b else '--'
        )
        w_norm = sub_df[WEIGHT_COLUMN].values / sub_df[WEIGHT_COLUMN].values.mean()
        prev_w = float(np.average(sub_df[TARGET_COLUMN].values, weights=w_norm))
        yield_1k = m['ppv_10pct_w'] * 1000.0
        mult = m['ppv_10pct_w'] / prev_w if prev_w > 0 else float('nan')

        row = {
            'Country': country.title(),
            'Evaluation': setting,
            'Model': model_label,
            'N_Test': len(sub_df),
            'Deaths_Test': int(sub_df[TARGET_COLUMN].sum()),
            'Prev_W_Pct': round(prev_w * 100, 2),
            'AUROC_W': round(m['auroc_w'], 3),
            'AUROC_95CI': auc_ci,
            'AUPRC_W': round(m['auprc_w'], 3),
            'Brier_W': round(m['brier_w'], 4),
            'Brier_Null_W': round(m['brier_null_w'], 4),
            'BSS_W': round(m['brier_skill_score_w'], 4),
            'Cal_Slope': round(m['cal_slope_w'], 3),
            'Cal_Intercept': round(m['cal_intercept_w'], 3),
            'Sens_10pct': round(m['sensitivity_10pct_w'] * 100, 1),
            'Sens_10pct_CI': sens_ci,
            'NNS_10pct': round(m['nns_10pct_w'], 1),
            'NNS_10pct_CI': nns_ci,
            'Yield_1k_10pct': round(yield_1k, 1),
            'Yield_Mult': round(mult, 2),
        }
        rows.append(row)
        print(f"  [{setting}] {model_label:28s}: AUC={row['AUROC_W']:.3f} {auc_ci} | "
              f"Brier={row['Brier_W']:.4f} (Null {row['Brier_Null_W']:.4f}) | "
              f"Slope={row['Cal_Slope']:.3f} Int={row['Cal_Intercept']:.3f} | "
              f"Sens@10%={row['Sens_10pct']:.1f}% NNS={row['NNS_10pct']:.1f} ({row['Yield_Mult']:.2f}x)")

    # 1. SWIFT-U5 (Full: Modules A + B + C)
    swift_full = SWIFTU5Estimator(
        use_module_a=True, use_module_b=True, use_module_c=True,
        n_estimators=150, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE
    )
    swift_full.preprocessor = ConsistentPreprocessor(include_missing_flags=True)
    swift_full.fit(df_dev, target_round=test_round, df_target_unlabeled=df_test, df_val=df_val)
    p_val_full = swift_full.predict_proba(df_val)
    thr_full, _ = compute_youden_threshold_from_validation(y_val, p_val_full, w_val)
    p_test_full = swift_full.predict_proba(df_test)
    record('Track A (Temporal)', 'SWIFT-U5 (Full)', df_test, p_test_full, thr_full)

    # 2. SWIFT-U5 w/o Module A (Ablation: B + C only)
    swift_no_a = SWIFTU5Estimator(
        use_module_a=False, use_module_b=True, use_module_c=True,
        n_estimators=150, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE
    )
    swift_no_a.preprocessor = ConsistentPreprocessor(include_missing_flags=True)
    swift_no_a.fit(df_dev, target_round=test_round, df_target_unlabeled=None, df_val=df_val)
    p_val_noa = swift_no_a.predict_proba(df_val)
    thr_noa, _ = compute_youden_threshold_from_validation(y_val, p_val_noa, w_val)
    p_test_noa = swift_no_a.predict_proba(df_test)
    record('Track A (Temporal)', 'SWIFT-U5 w/o Module A', df_test, p_test_noa, thr_noa)

    # 3. Baselines (Weighted XGBoost, Logistic Regression, Random Forest)
    prep = ConsistentPreprocessor(include_missing_flags=True)
    X_dev, _ = prep.fit_transform(df_dev)
    X_val, _ = prep.transform(df_val)
    X_test, _ = prep.transform(df_test)

    y_dev = df_dev[TARGET_COLUMN].values.astype(int)
    raw_w = df_dev[WEIGHT_COLUMN].values.astype(float)
    w_dev_norm = raw_w / raw_w.mean()
    spw = compute_scale_pos_weight(y_dev, w_dev_norm)

    baselines = [
        ('Weighted XGBoost', build_weighted_xgboost(scale_pos_weight=spw, n_estimators=100, random_state=RANDOM_STATE)),
        ('Weighted Logistic', build_weighted_logistic(random_state=RANDOM_STATE)),
        ('Weighted Random Forest', build_weighted_random_forest(n_estimators=100, random_state=RANDOM_STATE)),
    ]
    for name, clf in baselines:
        clf.fit(X_dev, y_dev, sample_weight=w_dev_norm)
        pv = clf.predict_proba(X_val)[:, 1]
        pt = clf.predict_proba(X_test)[:, 1]
        thr_b, _ = compute_youden_threshold_from_validation(y_val, pv, w_val)
        record('Track A (Temporal)', name, df_test, pt, thr_b)

    # 4. Track B Diagnostic: Pooled 70/30 Random Split (Weighted XGBoost)
    df_pool_tr, df_pool_te = train_test_split(
        df_all, test_size=0.30, stratify=df_all[TARGET_COLUMN], random_state=RANDOM_STATE
    )
    df_pool_tr = df_pool_tr.reset_index(drop=True)
    df_pool_te = df_pool_te.reset_index(drop=True)
    prep_r = ConsistentPreprocessor(include_missing_flags=True)
    Xr_tr, _ = prep_r.fit_transform(df_pool_tr)
    Xr_te, _ = prep_r.transform(df_pool_te)
    yr_tr = df_pool_tr[TARGET_COLUMN].values.astype(int)
    wr_tr = df_pool_tr[WEIGHT_COLUMN].values.astype(float)
    wr_tr = wr_tr / wr_tr.mean()
    spw_r = compute_scale_pos_weight(yr_tr, wr_tr)
    clf_r = build_weighted_xgboost(scale_pos_weight=spw_r, n_estimators=100, random_state=RANDOM_STATE)
    clf_r.fit(Xr_tr, yr_tr, sample_weight=wr_tr)
    pr_te = clf_r.predict_proba(Xr_te)[:, 1]
    record('Track B (Random 70/30)', 'Weighted XGBoost (Random)', df_pool_te, pr_te, 0.5)

    return rows


def run_cross_country_transfer(df_nepal: pd.DataFrame,
                               df_ethiopia: pd.DataFrame) -> List[Dict]:
    """
    Cross-country transportability test:
    Train SWIFT-U5 base learner on BDHS 2004-2014 using country-invariant
    biological/socioeconomic predictors (all CANDIDATE_FEATURES without country divisions),
    calibrate on BDHS 2017-18, and test on locked Nepal 2022 & Ethiopia 2024-25:
      (a) Zero-shot (BDHS-calibrated probabilities)
      (b) Locally recalibrated via Module C on target country's validation wave
    """
    bdhs_path = PROCESSED_DATA_DIR / 'u5m_harmonized_cohort.parquet'
    df_bd = pd.read_parquet(bdhs_path)
    df_bd_dev = df_bd[df_bd['round'].isin(['2004', '2007', '2011', '2014'])].copy().reset_index(drop=True)
    df_bd_val = df_bd[df_bd['round'] == '2017-18'].copy().reset_index(drop=True)

    # Fit preprocessor WITHOUT division one-hot (drop division_7 temporarily)
    df_bd_dev_nd = df_bd_dev.drop(columns=[DIVISION_COLUMN])
    df_bd_val_nd = df_bd_val.drop(columns=[DIVISION_COLUMN])

    prep = ConsistentPreprocessor(include_missing_flags=True)
    X_dev, fnames = prep.fit_transform(df_bd_dev_nd, CANDIDATE_FEATURES)
    X_bd_val, _ = prep.transform(df_bd_val_nd, CANDIDATE_FEATURES)

    y_dev = df_bd_dev[TARGET_COLUMN].values.astype(int)
    w_dev = df_bd_dev[WEIGHT_COLUMN].values.astype(float)
    w_dev_norm = w_dev / w_dev.mean()
    spw = compute_scale_pos_weight(y_dev, w_dev_norm)

    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
    )
    clf.fit(X_dev, y_dev, sample_weight=w_dev_norm)

    # Calibrate on BDHS 2017-18 (Module C)
    p_bd_val_raw = clf.predict_proba(X_bd_val)[:, 1]
    w_bd_val = df_bd_val[WEIGHT_COLUMN].values.astype(float)
    w_bd_val = w_bd_val / w_bd_val.mean()
    mod_c_bd = ModuleCDriftRecalibrator(use_temporal_recalibration=True)
    mod_c_bd.fit_recalibration(df_bd_val[TARGET_COLUMN].values.astype(int), p_bd_val_raw, w_bd_val, round_label='2017-18')

    rows = []
    targets = [
        ('Nepal', df_nepal, '2016', '2022'),
        ('Ethiopia', df_ethiopia, '2019', '2024-25'),
    ]

    print("\n--- Cross-Country Transportability (BDHS 2004-2014 -> External Test) ---")
    for cname, df_c, val_rnd, test_rnd in targets:
        df_c_val = df_c[df_c['round'] == val_rnd].drop(columns=[DIVISION_COLUMN]).reset_index(drop=True)
        df_c_test = df_c[df_c['round'] == test_rnd].copy().reset_index(drop=True)
        df_c_test_nd = df_c_test.drop(columns=[DIVISION_COLUMN])

        # Align feature transform cleanly with BDHS preprocessor
        X_c_val, _ = prep.transform(df_c_val, CANDIDATE_FEATURES)
        X_c_test, _ = prep.transform(df_c_test_nd, CANDIDATE_FEATURES)

        p_raw_val = clf.predict_proba(X_c_val)[:, 1]
        p_raw_test = clf.predict_proba(X_c_test)[:, 1]

        # 1. Zero-shot (BDHS Module C)
        p_zero = mod_c_bd.predict_proba(p_raw_test)
        m_z, b_z = evaluate_predictions(df_c_test, p_zero, 0.04546, run_bootstrap=True)
        auc_ci_z = f"[{b_z['auroc_w']['ci_lower_95']:.3f}, {b_z['auroc_w']['ci_upper_95']:.3f}]"
        nns_ci_z = f"[{b_z['nns_10pct_w']['ci_lower_95']:.1f}, {b_z['nns_10pct_w']['ci_upper_95']:.1f}]"

        # 2. Locally recalibrated Module C on target country's validation wave
        mod_c_loc = ModuleCDriftRecalibrator(use_temporal_recalibration=True)
        w_c_val = df_c_val[WEIGHT_COLUMN].values.astype(float)
        w_c_val = w_c_val / w_c_val.mean()
        mod_c_loc.fit_recalibration(df_c_val[TARGET_COLUMN].values.astype(int), p_raw_val, w_c_val, round_label=val_rnd)
        p_loc_val = mod_c_loc.predict_proba(p_raw_val)
        thr_loc, _ = compute_youden_threshold_from_validation(df_c_val[TARGET_COLUMN].values.astype(int), p_loc_val, w_c_val)
        p_loc = mod_c_loc.predict_proba(p_raw_test)
        m_l, b_l = evaluate_predictions(df_c_test, p_loc, thr_loc, run_bootstrap=True)
        auc_ci_l = f"[{b_l['auroc_w']['ci_lower_95']:.3f}, {b_l['auroc_w']['ci_upper_95']:.3f}]"
        nns_ci_l = f"[{b_l['nns_10pct_w']['ci_lower_95']:.1f}, {b_l['nns_10pct_w']['ci_upper_95']:.1f}]"

        for tag, m, auc_ci, nns_ci in [
            ('BDHS Transfer (Zero-Shot)', m_z, auc_ci_z, nns_ci_z),
            ('BDHS Transfer + Local Calib', m_l, auc_ci_l, nns_ci_l),
        ]:
            w_norm = df_c_test[WEIGHT_COLUMN].values / df_c_test[WEIGHT_COLUMN].values.mean()
            prev_w = float(np.average(df_c_test[TARGET_COLUMN].values, weights=w_norm))
            rows.append({
                'Country': cname,
                'Evaluation': 'Cross-Country Transfer',
                'Model': tag,
                'N_Test': len(df_c_test),
                'Deaths_Test': int(df_c_test[TARGET_COLUMN].sum()),
                'Prev_W_Pct': round(prev_w * 100, 2),
                'AUROC_W': round(m['auroc_w'], 3),
                'AUROC_95CI': auc_ci,
                'AUPRC_W': round(m['auprc_w'], 3),
                'Brier_W': round(m['brier_w'], 4),
                'Brier_Null_W': round(m['brier_null_w'], 4),
                'BSS_W': round(m['brier_skill_score_w'], 4),
                'Cal_Slope': round(m['cal_slope_w'], 3),
                'Cal_Intercept': round(m['cal_intercept_w'], 3),
                'Sens_10pct': round(m['sensitivity_10pct_w'] * 100, 1),
                'Sens_10pct_CI': '--',
                'NNS_10pct': round(m['nns_10pct_w'], 1),
                'NNS_10pct_CI': nns_ci,
                'Yield_1k_10pct': round(m['ppv_10pct_w'] * 1000.0, 1),
                'Yield_Mult': round(m['ppv_10pct_w'] / prev_w, 2),
            })
            print(f"  [{cname:8s}] {tag:28s}: AUC={m['auroc_w']:.3f} {auc_ci} | "
                  f"Brier={m['brier_w']:.4f} (Null {m['brier_null_w']:.4f}) | "
                  f"Slope={m['cal_slope_w']:.3f} Int={m['cal_intercept_w']:.3f} | "
                  f"NNS={m['nns_10pct_w']:.1f}")
    return rows


def build_cohort_summary(df_nepal: pd.DataFrame, df_ethiopia: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cname, df_c in [('Nepal', df_nepal), ('Ethiopia', df_ethiopia)]:
        for rnd, g in df_c.groupby('round', sort=False):
            w = g[WEIGHT_COLUMN].values / g[WEIGHT_COLUMN].values.mean()
            u5mr = float(np.average(g[TARGET_COLUMN].values, weights=w) * 1000.0)
            rows.append({
                'Country': cname,
                'Round': rnd,
                'Sample_N': len(g),
                'Deaths_n': int(g[TARGET_COLUMN].sum()),
                'Weighted_U5MR_per1000': round(u5mr, 1),
                'Num_Regions': g[DIVISION_COLUMN].nunique(),
                'Num_PSU': g['psu_wave'].nunique(),
            })
    return pd.DataFrame(rows)


def main():
    print("=================================================================")
    print("SWIFT-U5 External Multi-Country Validation (Nepal & Ethiopia DHS)")
    print("=================================================================")

    df_nepal = load_country_cohort('nepal', ['2006', '2011', '2016', '2022'])
    df_ethiopia = load_country_cohort('ethiopia', ['2005', '2011', '2016', '2019', '2024-25'])

    cohort_summary = build_cohort_summary(df_nepal, df_ethiopia)
    print("\n--- External Cohort Summary ---")
    print(cohort_summary.to_string(index=False))

    nepal_rows = run_within_country_suite('nepal', df_nepal, ['2006', '2011'], '2016', '2022')
    ethiopia_rows = run_within_country_suite('ethiopia', df_ethiopia, ['2005', '2011', '2016'], '2019', '2024-25')
    transfer_rows = run_cross_country_transfer(df_nepal, df_ethiopia)

    all_rows = pd.DataFrame(nepal_rows + ethiopia_rows + transfer_rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    docs_tables = WORKSPACE_DIR / 'docs' / 'tables'
    docs_tables.mkdir(parents=True, exist_ok=True)

    all_rows.to_csv(TABLES_DIR / 'table8_external_validation.csv', index=False)
    all_rows.to_csv(docs_tables / 'table8_external_validation.csv', index=False)
    cohort_summary.to_csv(TABLES_DIR / 'table8b_external_cohort_summary.csv', index=False)
    cohort_summary.to_csv(docs_tables / 'table8b_external_cohort_summary.csv', index=False)

    # Markdown report
    report_lines = [
        "# Session 6 External Multi-Country Validation Report",
        "",
        "## 1. Cohort Summary",
        "```",
        cohort_summary.to_string(index=False),
        "```",
        "",
        "## 2. External Validation Results",
        "```",
        all_rows.to_string(index=False),
        "```",
    ]
    with open(REPORTS_DIR / 'session6_external_validation_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print("\nSaved external validation outputs to outputs/tables/, docs/tables/, and outputs/reports/.")


if __name__ == '__main__':
    main()
