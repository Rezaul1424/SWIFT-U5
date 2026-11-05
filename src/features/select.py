"""
Feature selection module for SWIFT-U5.
Implements:
  1. Domain Filter: Pre-outcome, <30% missing, drop collinear pairs (r > 0.8).
  2. PSU-Grouped Stability Selection: Dual-method selection via Weighted LASSO
     Logistic Regression + Permutation Importance on Weighted XGBoost.
  3. Rolling Window Drift Tracking: Evaluates feature stability across sequential
     survey windows to classify features into Stable vs. Drifting.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.features.build_features import (
    CANDIDATE_FEATURES, DIVISION_COLUMN, TARGET_COLUMN,
    WEIGHT_COLUMN, PSU_COLUMN, STRATA_COLUMN, ROUND_COLUMN,
    LeakageFreePreprocessor
)


def apply_domain_filter(df_train: pd.DataFrame,
                        candidate_features: Optional[List[str]] = None,
                        max_missing_rate: float = 0.30,
                        max_correlation: float = 0.80) -> Tuple[List[str], Dict]:
    """
    Apply domain knowledge filter strictly on training data:
    1. Filter out features with > 30% missing values in df_train.
    2. Check pairwise correlation among numeric features; if |r| > 0.80,
       drop the feature with higher missingness (or lower variance).
    """
    if candidate_features is None:
        candidate_features = list(CANDIDATE_FEATURES)

    log_info = {
        'initial_candidates': len(candidate_features),
        'dropped_high_missing': [],
        'dropped_collinear': [],
        'retained_features': []
    }

    # Step 1: Missingness filter
    missing_filtered = []
    for col in candidate_features:
        if col in df_train.columns:
            miss_rate = df_train[col].isna().mean()
            if miss_rate <= max_missing_rate:
                missing_filtered.append(col)
            else:
                log_info['dropped_high_missing'].append((col, round(float(miss_rate), 3)))
        else:
            log_info['dropped_high_missing'].append((col, 1.0))

    # Step 2: Collinearity filter
    numeric_df = df_train[missing_filtered].copy()
    for col in missing_filtered:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce').fillna(
            numeric_df[col].median()
        )

    corr_matrix = numeric_df.corr().abs()
    to_drop = set()

    for i in range(len(missing_filtered)):
        col_i = missing_filtered[i]
        if col_i in to_drop:
            continue
        for j in range(i + 1, len(missing_filtered)):
            col_j = missing_filtered[j]
            if col_j in to_drop:
                continue
            r_val = corr_matrix.loc[col_i, col_j]
            if r_val > max_correlation:
                # Prefer birth_order over parity_before (exact duplicate)
                if col_i == 'birth_order' and col_j == 'parity_before':
                    to_drop.add(col_j)
                    log_info['dropped_collinear'].append((col_j, col_i, round(float(r_val), 3)))
                elif col_j == 'birth_order' and col_i == 'parity_before':
                    to_drop.add(col_i)
                    log_info['dropped_collinear'].append((col_i, col_j, round(float(r_val), 3)))
                else:
                    # Drop the one with higher missingness in df_train
                    miss_i = df_train[col_i].isna().mean()
                    miss_j = df_train[col_j].isna().mean()
                    drop_col = col_j if miss_j >= miss_i else col_i
                    to_drop.add(drop_col)
                    other_col = col_i if drop_col == col_j else col_j
                    log_info['dropped_collinear'].append((drop_col, other_col, round(float(r_val), 3)))

    retained = [col for col in missing_filtered if col not in to_drop]
    log_info['retained_features'] = retained
    return retained, log_info


def run_psu_stability_selection(df_train: pd.DataFrame,
                                feature_subset: List[str],
                                n_subsamples: int = 50,
                                stability_threshold: float = 0.70,
                                seed: int = 42) -> pd.DataFrame:
    """
    Run PSU-stratified stability selection inside training data:
    Dual methods:
      1. Weighted LASSO Logistic Regression (L1-penalized)
      2. Weighted XGBoost with Permutation Feature Importance
    Returns DataFrame with selection frequencies and stability indicator.
    """
    rng = np.random.default_rng(seed)

    # Preprocessor fitted on training set
    prep = LeakageFreePreprocessor(include_missing_flags=False)
    X_train_full, feat_names = prep.fit_transform(df_train, feature_subset)
    y_train_full = df_train[TARGET_COLUMN].values.astype(int)
    w_train_full = df_train[WEIGHT_COLUMN].values.astype(float)
    w_train_full = w_train_full / w_train_full.mean()

    # Precompute cluster index groups for fast resampling
    psu_series = df_train[PSU_COLUMN].values
    strata_series = df_train[STRATA_COLUMN].values
    unique_strata = np.unique(strata_series)

    strata_psu_map = {s: np.unique(psu_series[strata_series == s]) for s in unique_strata}
    psu_idx_map = {p: np.where(psu_series == p)[0] for p in np.unique(psu_series)}

    n_feats = len(feat_names)
    lasso_selections = np.zeros(n_feats, dtype=int)
    xgb_selections = np.zeros(n_feats, dtype=int)

    for b in range(n_subsamples):
        # 1. Cluster bootstrap resample
        selected_chunks = []
        for s in unique_strata:
            psus = strata_psu_map[s]
            sampled_psus = rng.choice(psus, size=len(psus), replace=True)
            for p in sampled_psus:
                selected_chunks.append(psu_idx_map[p])

        boot_idx = np.concatenate(selected_chunks)
        X_b = X_train_full[boot_idx]
        y_b = y_train_full[boot_idx]
        w_b = w_train_full[boot_idx]

        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue

        # 2. Method 1: Weighted LASSO Logistic Regression
        scaler = StandardScaler()
        X_b_scaled = scaler.fit_transform(X_b)

        lasso = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=0.1,
            class_weight='balanced',
            max_iter=300,
            random_state=seed + b
        )
        lasso.fit(X_b_scaled, y_b, sample_weight=w_b)
        lasso_active = (np.abs(lasso.coef_[0]) > 1e-4).astype(int)
        lasso_selections += lasso_active

        # 3. Method 2: Weighted XGBoost Permutation Importance
        spw = float((w_b[y_b == 0].sum()) / max(1.0, w_b[y_b == 1].sum()))
        xgb = XGBClassifier(
            n_estimators=40,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=spw,
            eval_metric='logloss',
            random_state=seed + b,
            verbosity=0,
            n_jobs=-1
        )
        xgb.fit(X_b, y_b, sample_weight=w_b)

        # Baseline AUROC on this sample
        base_probs = xgb.predict_proba(X_b)[:, 1]
        try:
            base_auc = roc_auc_score(y_b, base_probs, sample_weight=w_b)
        except Exception:
            base_auc = 0.5

        # Feature permutation check (sample 1000 rows for fast permutation)
        perm_sub_n = min(2000, len(X_b))
        sub_idx = rng.choice(len(X_b), size=perm_sub_n, replace=False)
        X_sub = X_b[sub_idx].copy()
        y_sub = y_b[sub_idx]
        w_sub = w_b[sub_idx]

        xgb_active = np.zeros(n_feats, dtype=int)
        for j in range(n_feats):
            X_perm = X_sub.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_probs = xgb.predict_proba(X_perm)[:, 1]
            try:
                p_auc = roc_auc_score(y_sub, perm_probs, sample_weight=w_sub)
                delta_auc = base_auc - p_auc
                if delta_auc > 0.001:  # Positive discriminatory contribution
                    xgb_active[j] = 1
            except Exception:
                pass

        xgb_selections += xgb_active

    freq_lasso = lasso_selections / n_subsamples
    freq_xgb = xgb_selections / n_subsamples
    both_selected = (freq_lasso >= stability_threshold) & (freq_xgb >= stability_threshold)

    results_df = pd.DataFrame({
        'feature': feat_names,
        'lasso_freq': freq_lasso,
        'xgb_freq': freq_xgb,
        'min_freq': np.minimum(freq_lasso, freq_xgb),
        'is_stable': both_selected
    }).sort_values(by='min_freq', ascending=False).reset_index(drop=True)

    return results_df


def track_rolling_feature_stability(df: pd.DataFrame,
                                     rolling_windows: Optional[List[Tuple[str, List[str]]]] = None,
                                     n_subsamples: int = 40,
                                     stability_threshold: float = 0.70) -> pd.DataFrame:
    """
    Repeat stability selection per rolling / sequential window to label features
    as stable or drifting across survey rounds.
    """
    if rolling_windows is None:
        rolling_windows = [
            ('Window 1 (2004)', ['2004']),
            ('Window 2 (2004–07)', ['2004', '2007']),
            ('Window 3 (2004–11)', ['2004', '2007', '2011']),
            ('Window 4 (2004–14)', ['2004', '2007', '2011', '2014']),
            ('Window 5 (2004–18)', ['2004', '2007', '2011', '2014', '2017-18'])
        ]

    window_records = {}

    for win_label, win_rounds in rolling_windows:
        print(f"--> Analyzing stability in {win_label} (Rounds: {win_rounds})...")
        win_df = df[df[ROUND_COLUMN].isin(win_rounds)].copy()

        # Domain filter on this window
        features_passed, _ = apply_domain_filter(win_df, max_missing_rate=0.35)
        # Stability selection on this window
        res = run_psu_stability_selection(
            win_df,
            features_passed,
            n_subsamples=n_subsamples,
            stability_threshold=stability_threshold
        )
        for _, row in res.iterrows():
            feat = row['feature']
            if feat not in window_records:
                window_records[feat] = {}
            window_records[feat][win_label] = float(row['min_freq'])

    # Build master tracking matrix
    all_feats = sorted(window_records.keys())
    matrix_rows = []
    window_cols = [w[0] for w in rolling_windows]

    for feat in all_feats:
        row_dict = {'feature': feat}
        scores = []
        for w_col in window_cols:
            score = window_records[feat].get(w_col, 0.0)
            row_dict[w_col] = round(score, 3)
            scores.append(score)

        mean_stability = float(np.mean(scores))
        std_stability = float(np.std(scores))
        row_dict['mean_stability'] = round(mean_stability, 3)
        row_dict['std_stability'] = round(std_stability, 3)

        # Classification criteria:
        # Stable: mean_stability >= 0.70 and std_stability <= 0.20
        # Drifting: selected significantly in some windows but decayed or surged
        if mean_stability >= stability_threshold and std_stability <= 0.20:
            classification = 'Stable Core'
        elif any(s >= stability_threshold for s in scores) and std_stability > 0.15:
            classification = 'Drifting / Contextual'
        else:
            classification = 'Excluded / Low Priority'

        row_dict['classification'] = classification
        matrix_rows.append(row_dict)

    tracking_df = pd.DataFrame(matrix_rows).sort_values(
        by='mean_stability', ascending=False
    ).reset_index(drop=True)

    return tracking_df

