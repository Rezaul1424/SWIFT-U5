"""
Policy Layer and Equity Audit Engine for SWIFT-U5.

Implements:
1. Division-level screening capacity and NNS estimation with PSU-stratified cluster CIs.
2. Priority metric: Expected deaths captured per 1,000 births screened.
3. Equity audit disaggregated by wealth quintile, residence, and education.
4. SHAP feature attribution and temporal SHAP Drift Index (SDI).
5. Longitudinal division mortality trajectories (2004-2022).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import shap

from src.evaluation.metrics import (
    _weighted_roc_auc, _weighted_brier_score, _calibration_slope_intercept
)


def compute_screening_stats(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           sample_weight: np.ndarray,
                           capacity: float) -> Dict[str, float]:
    """
    Compute screening yield metrics at a specified screening capacity fraction.
    """
    n = len(y_true)
    if n == 0 or y_true.sum() == 0:
        return {
            'sensitivity': float('nan'),
            'ppv': float('nan'),
            'nns': float('nan'),
            'deaths_per_1000': float('nan'),
            'yield_ratio': float('nan')
        }

    w = sample_weight / sample_weight.mean()
    n_screen = max(1, int(np.ceil(n * capacity)))
    top_indices = np.argsort(y_prob)[-n_screen:]
    screen_mask = np.zeros(n, dtype=bool)
    screen_mask[top_indices] = True

    total_deaths_w = float(np.sum(w * y_true))
    screened_w = float(np.sum(w * screen_mask))
    tp_w = float(np.sum(w * screen_mask * y_true))

    sens = float(tp_w / total_deaths_w) if total_deaths_w > 0 else float('nan')
    ppv = float(tp_w / screened_w) if screened_w > 0 else float('nan')
    nns = float(1.0 / ppv) if ppv > 0 else float('nan')
    deaths_per_1000 = float(1000.0 * ppv) if ppv > 0 else float('nan')

    base_rate = float(total_deaths_w / np.sum(w)) if np.sum(w) > 0 else float('nan')
    yield_ratio = float(ppv / base_rate) if base_rate > 0 else float('nan')

    return {
        'sensitivity': sens,
        'ppv': ppv,
        'nns': nns,
        'deaths_per_1000': deaths_per_1000,
        'yield_ratio': yield_ratio
    }


def bootstrap_subgroup_capacity(df_sub: pd.DataFrame,
                               y_prob_sub: np.ndarray,
                               capacity: float,
                               n_replicates: int = 100,
                               seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """
    PSU-stratified cluster bootstrap for screening metrics within a subgroup.
    Returns 95% CIs for sensitivity, ppv, nns, and deaths_per_1000.
    """
    rng = np.random.default_rng(seed)
    y_true_arr = df_sub['target_u5m'].values.astype(int)
    weight_arr = df_sub['sample_weight'].values.astype(float)
    y_prob_arr = np.asarray(y_prob_sub, dtype=float)

    psu_series = df_sub['psu'].values
    strata_series = df_sub['strata'].values if 'strata' in df_sub.columns else np.zeros(len(df_sub), dtype=int)

    unique_strata = np.unique(strata_series)
    strata_psu_map = {}
    for s in unique_strata:
        mask_s = strata_series == s
        strata_psu_map[s] = np.unique(psu_series[mask_s])

    psu_indices_map = {}
    for p in np.unique(psu_series):
        psu_indices_map[p] = np.where(psu_series == p)[0]

    boot_records = []
    for _ in range(n_replicates):
        selected_chunks = []
        for s in unique_strata:
            psus = strata_psu_map[s]
            sampled_psus = rng.choice(psus, size=len(psus), replace=True)
            for p in sampled_psus:
                selected_chunks.append(psu_indices_map[p])

        boot_idx = np.concatenate(selected_chunks)
        yt_b = y_true_arr[boot_idx]
        if yt_b.sum() == 0 or yt_b.sum() == len(yt_b):
            continue

        stats = compute_screening_stats(
            yt_b, y_prob_arr[boot_idx], weight_arr[boot_idx], capacity=capacity
        )
        boot_records.append(stats)

    cis = {}
    metric_keys = ['sensitivity', 'ppv', 'nns', 'deaths_per_1000', 'yield_ratio']
    if len(boot_records) >= 10:
        bdf = pd.DataFrame(boot_records)
        for k in metric_keys:
            vals = bdf[k].dropna()
            if len(vals) > 0:
                cis[k] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
            else:
                cis[k] = (float('nan'), float('nan'))
    else:
        for k in metric_keys:
            cis[k] = (float('nan'), float('nan'))

    return cis


def compute_division_capacity_table(df_test: pd.DataFrame,
                                   y_prob: np.ndarray,
                                   division_col: str = 'division_8',
                                   capacities: Optional[List[float]] = None,
                                   n_bootstrap: int = 100,
                                   seed: int = 42) -> pd.DataFrame:
    """
    Generate Table 6: Division NNS & Capacity Table.
    Evaluates observed mortality, predicted risk, and screening yield (Sens, NNS, Deaths/1k)
    across multiple screening capacities with 95% PSU-cluster bootstrap CIs.
    """
    if capacities is None:
        capacities = [0.05, 0.10, 0.20]

    divisions = sorted(df_test[division_col].unique().tolist())
    rows = []

    # Helper to process a cohort subset (either division or national)
    def _evaluate_subset(name: str, mask: np.ndarray) -> Dict:
        df_sub = df_test[mask].copy().reset_index(drop=True)
        yp_sub = y_prob[mask]
        n_unweighted = len(df_sub)
        w_sub = df_sub['sample_weight'].values.astype(float)
        y_sub = df_sub['target_u5m'].values.astype(int)

        w_norm = w_sub / w_sub.mean()
        w_deaths = float(np.sum(w_norm * y_sub))
        unw_deaths = int(np.sum(y_sub))
        obs_rate_per_1000 = float(w_deaths / len(df_sub) * 1000.0)
        mean_pred_risk_per_1000 = float(np.average(yp_sub, weights=w_norm) * 1000.0)

        row_dict = {
            'division': name.title(),
            'n_unweighted': n_unweighted,
            'deaths': unw_deaths,
            'obs_rate_per_1000': obs_rate_per_1000,
            'pred_risk_per_1000': mean_pred_risk_per_1000
        }

        # Compute metrics across capacities
        for cap in capacities:
            pct_label = int(cap * 100)
            stats = compute_screening_stats(y_sub, yp_sub, w_sub, capacity=cap)
            cis = bootstrap_subgroup_capacity(df_sub, yp_sub, capacity=cap, n_replicates=n_bootstrap, seed=seed)

            row_dict[f'sens_{pct_label}'] = stats['sensitivity']
            row_dict[f'sens_{pct_label}_ci_lo'] = cis['sensitivity'][0]
            row_dict[f'sens_{pct_label}_ci_hi'] = cis['sensitivity'][1]

            row_dict[f'nns_{pct_label}'] = stats['nns']
            row_dict[f'nns_{pct_label}_ci_lo'] = cis['nns'][0]
            row_dict[f'nns_{pct_label}_ci_hi'] = cis['nns'][1]

            row_dict[f'deaths_per_1000_cap{pct_label}'] = stats['deaths_per_1000']
            row_dict[f'deaths_per_1000_cap{pct_label}_ci_lo'] = cis['deaths_per_1000'][0]
            row_dict[f'deaths_per_1000_cap{pct_label}_ci_hi'] = cis['deaths_per_1000'][1]

            row_dict[f'yield_ratio_{pct_label}'] = stats['yield_ratio']

        return row_dict

    # Compute for each division
    for div in divisions:
        mask = (df_test[division_col] == div).values
        rows.append(_evaluate_subset(div, mask))

    # Compute National Total
    national_mask = np.ones(len(df_test), dtype=bool)
    rows.append(_evaluate_subset('National (Bangladesh)', national_mask))

    return pd.DataFrame(rows)


def compute_equity_audit_table(df_test: pd.DataFrame,
                              y_prob: np.ndarray,
                              capacity: float = 0.10,
                              n_bootstrap: int = 100,
                              seed: int = 42) -> pd.DataFrame:
    """
    Generate Table 7: Equity Audit by Wealth Quintile and Residence.
    Disaggregates SWIFT-U5 performance across vulnerable subpopulations,
    computes AUROC, Sensitivity@10%, PPV, NNS, calibration intercept,
    and applies clinical disparity flags.
    """
    subgroups = []

    # 1. Overall / National benchmark
    subgroups.append(('Overall', 'National', np.ones(len(df_test), dtype=bool)))

    # 2. Residence
    if 'residence_urban' in df_test.columns:
        subgroups.append(('Residence', 'Rural', (df_test['residence_urban'] == 0).values))
        subgroups.append(('Residence', 'Urban', (df_test['residence_urban'] == 1).values))

    # 3. Wealth Quintiles (1=Poorest to 5=Richest)
    wealth_labels = {
        1: 'Q1 (Poorest)',
        2: 'Q2 (Poorer)',
        3: 'Q3 (Middle)',
        4: 'Q4 (Richer)',
        5: 'Q5 (Richest)'
    }
    if 'wealth_quintile' in df_test.columns:
        for q in sorted(df_test['wealth_quintile'].unique()):
            label = wealth_labels.get(int(q), f'Q{int(q)}')
            subgroups.append(('Wealth Quintile', label, (df_test['wealth_quintile'] == q).values))

    # 4. Maternal Education (0=None, 1=Primary, 2=Secondary, 3=Higher)
    edu_labels = {
        0: 'No Education',
        1: 'Primary',
        2: 'Secondary',
        3: 'Higher'
    }
    if 'maternal_education' in df_test.columns:
        for e in sorted(df_test['maternal_education'].unique()):
            label = edu_labels.get(int(e), f'Edu_{int(e)}')
            subgroups.append(('Maternal Education', label, (df_test['maternal_education'] == e).values))

    rows = []
    national_sens_10 = None

    for domain, group_name, mask in subgroups:
        df_sub = df_test[mask].copy().reset_index(drop=True)
        yp_sub = y_prob[mask]
        n_unw = len(df_sub)
        w_sub = df_sub['sample_weight'].values.astype(float)
        y_sub = df_sub['target_u5m'].values.astype(int)
        deaths_cnt = int(np.sum(y_sub))

        w_norm = w_sub / w_sub.mean()
        prev_pct = float(np.sum(w_norm * y_sub) / n_unw * 100.0)
        mean_pred_pct = float(np.average(yp_sub, weights=w_norm) * 100.0)

        # Subgroup AUROC
        auroc = _weighted_roc_auc(y_sub, yp_sub, sample_weight=w_sub)

        # Screening yield at capacity (10%)
        stats_10 = compute_screening_stats(y_sub, yp_sub, w_sub, capacity=capacity)
        cis_10 = bootstrap_subgroup_capacity(df_sub, yp_sub, capacity=capacity, n_replicates=n_bootstrap, seed=seed)

        # Calibration intercept & slope
        slope, intercept = _calibration_slope_intercept(y_sub, yp_sub, sample_weight=w_sub)

        if domain == 'Overall':
            national_sens_10 = stats_10['sensitivity']

        # Equity disparity check:
        # 1. Sensitivity parity: Is sensitivity at least 75% of national sensitivity?
        # 2. Calibration parity: Is |calibration intercept| <= 0.60?
        is_sens_equitable = True
        if national_sens_10 is not None and not np.isnan(stats_10['sensitivity']):
            if stats_10['sensitivity'] < 0.75 * national_sens_10:
                is_sens_equitable = False

        is_cal_equitable = True
        if not np.isnan(intercept) and abs(intercept) > 0.60:
            is_cal_equitable = False

        if is_sens_equitable and is_cal_equitable:
            equity_status = "Satisfied"
        elif not is_sens_equitable and not is_cal_equitable:
            equity_status = "Flagged (Sens & Cal)"
        elif not is_sens_equitable:
            equity_status = "Flagged (Low Sens)"
        else:
            equity_status = "Flagged (Cal Drift)"

        rows.append({
            'domain': domain,
            'subgroup': group_name,
            'n_unweighted': n_unw,
            'deaths': deaths_cnt,
            'prevalence_pct': prev_pct,
            'mean_pred_pct': mean_pred_pct,
            'auroc': auroc,
            'sens_10': stats_10['sensitivity'],
            'sens_10_ci_lo': cis_10['sensitivity'][0],
            'sens_10_ci_hi': cis_10['sensitivity'][1],
            'ppv_10': stats_10['ppv'],
            'ppv_10_ci_lo': cis_10['ppv'][0],
            'ppv_10_ci_hi': cis_10['ppv'][1],
            'nns_10': stats_10['nns'],
            'nns_10_ci_lo': cis_10['nns'][0],
            'nns_10_ci_hi': cis_10['nns'][1],
            'cal_intercept': intercept,
            'cal_slope': slope,
            'equity_status': equity_status
        })

    return pd.DataFrame(rows)


def compute_longitudinal_division_trends(df: pd.DataFrame, division_col: str = 'division_7') -> pd.DataFrame:
    """
    Compute observed weighted under-5 mortality rates per 1,000 births
    across all survey rounds for each division.
    """
    records = []
    rounds = sorted(df['round'].unique().tolist())
    divisions = sorted(df[division_col].unique().tolist())

    for rnd in rounds:
        df_rnd = df[df['round'] == rnd]
        
        # National aggregate
        w_nat = df_rnd['sample_weight'].values.astype(float)
        y_nat = df_rnd['target_u5m'].values.astype(int)
        nat_rate = float(np.sum(w_nat * y_nat) / np.sum(w_nat) * 1000.0)
        records.append({
            'round': rnd,
            'division': 'National',
            'n': len(df_rnd),
            'deaths': int(y_nat.sum()),
            'u5mr_per_1000': nat_rate
        })

        for div in divisions:
            df_div = df_rnd[df_rnd[division_col] == div]
            if len(df_div) == 0:
                continue
            w_div = df_div['sample_weight'].values.astype(float)
            y_div = df_div['target_u5m'].values.astype(int)
            div_rate = float(np.sum(w_div * y_div) / np.sum(w_div) * 1000.0) if np.sum(w_div) > 0 else float('nan')
            records.append({
                'round': rnd,
                'division': div.title(),
                'n': len(df_div),
                'deaths': int(y_div.sum()),
                'u5mr_per_1000': div_rate
            })

    return pd.DataFrame(records)


def compute_posthoc_shap_and_drift(model,
                                  df_test: pd.DataFrame,
                                  feature_subset: Optional[List[str]] = None,
                                  cuts_data: Optional[List[Dict]] = None) -> Dict:
    """
    Perform post-hoc TreeSHAP interpretation on SWIFT-U5:
    1. Global SHAP values on 2022 test set.
    2. Per-division mean |SHAP| attribution matrix.
    3. SHAP Drift Index (SDI) across temporal cuts.
    """
    X_test, fnames = model.preprocessor.transform(df_test, feature_subset)
    explainer = shap.TreeExplainer(model.base_model_)
    shap_vals = explainer.shap_values(X_test)

    # Global importance
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    global_imp_df = pd.DataFrame({
        'feature': fnames,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    # Division-specific importance (8 divisions in 2022)
    div_col = 'division_8' if 'division_8' in df_test.columns else 'division_7'
    divisions = sorted(df_test[div_col].unique().tolist())
    div_shap_dict = {}

    for div in divisions:
        mask = (df_test[div_col] == div).values
        if mask.sum() > 0:
            div_mean_abs = np.mean(np.abs(shap_vals[mask]), axis=0)
            div_shap_dict[div.title()] = div_mean_abs

    div_shap_df = pd.DataFrame(div_shap_dict, index=fnames)

    # SHAP Drift Index across temporal cuts
    # Quantifies temporal shift in feature attribution relative to 2022 test model
    drift_records = []
    ref_series = pd.Series(mean_abs_shap, index=fnames)

    if cuts_data is not None:
        for cut in cuts_data:
            c_label = f"{cut.get('name', cut.get('test_round', ''))}"
            c_vals = cut.get('mean_abs_shap')
            c_fnames = cut.get('fnames', fnames)
            if c_vals is not None:
                c_series = pd.Series(c_vals, index=c_fnames)
                # Union of feature vocabularies
                all_vocab = sorted(list(set(ref_series.index).union(set(c_series.index))))
                p_c = c_series.reindex(all_vocab, fill_value=0.0).values
                p_ref = ref_series.reindex(all_vocab, fill_value=0.0).values

                if np.sum(p_c) > 0 and np.sum(p_ref) > 0:
                    p_c = p_c / np.sum(p_c)
                    p_ref = p_ref / np.sum(p_ref)
                    cos_sim = float(np.dot(p_c, p_ref) / (
                        np.linalg.norm(p_c) * np.linalg.norm(p_ref)
                    ))
                    sdi = max(0.0, 1.0 - cos_sim)
                else:
                    cos_sim = float('nan')
                    sdi = float('nan')

                drift_records.append({
                    'cut': c_label,
                    'test_round': cut.get('test_round', ''),
                    'cosine_similarity': cos_sim,
                    'shap_drift_index': sdi
                })

    drift_df = pd.DataFrame(drift_records)

    return {
        'shap_values': shap_vals,
        'feature_names': fnames,
        'X_test': X_test,
        'global_importance': global_imp_df,
        'division_importance': div_shap_df,
        'drift_df': drift_df
    }
