"""
PSU-stratified cluster bootstrap for SWIFT-U5.
Resamples PSUs within strata (DHS survey design-aware resampling)
to produce 95% CIs and paired bootstrap test statistics.
"""

from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd
from src.evaluation.metrics import compute_all_metrics


def _resample_cluster_indices(df: pd.DataFrame,
                               psu_col: str = 'psu',
                               strata_col: str = 'strata',
                               rng: Optional[np.random.Generator] = None) -> pd.Index:
    """
    Draw a bootstrap resample by resampling PSUs within each stratum (with replacement).
    Returns the row indices of the bootstrap sample.
    """
    if rng is None:
        rng = np.random.default_rng()

    boot_indices = []
    for stratum, s_df in df.groupby(strata_col):
        psus_in_stratum = s_df[psu_col].unique()
        # Resample PSUs with replacement (matching original number of PSUs)
        sampled_psus = rng.choice(psus_in_stratum, size=len(psus_in_stratum), replace=True)
        for psu in sampled_psus:
            boot_indices.extend(s_df[s_df[psu_col] == psu].index.tolist())

    return pd.Index(boot_indices)


def bootstrap_metrics(df_test: pd.DataFrame,
                      y_prob: np.ndarray,
                      n_replicates: int = 200,
                      psu_col: str = 'psu',
                      strata_col: str = 'strata',
                      target_col: str = 'target_u5m',
                      weight_col: str = 'sample_weight',
                      screening_capacities: Optional[List[float]] = None,
                      seed: int = 42) -> Dict:
    """
    Fast PSU-stratified cluster bootstrap to compute 95% CIs for metrics.
    Precomputes cluster-to-index mappings for near-instant execution.
    """
    if screening_capacities is None:
        screening_capacities = [0.05, 0.10, 0.20]

    rng = np.random.default_rng(seed)

    # Convert arrays to numpy for C-speed indexing
    y_true_arr = df_test[target_col].values.astype(int)
    weight_arr = df_test[weight_col].values.astype(float)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    # Precompute cluster index groups (avoids slow pandas groupby inside loop)
    psu_series = df_test[psu_col].values
    strata_series = df_test[strata_col].values
    
    unique_strata = np.unique(strata_series)
    strata_psu_map = {}
    for s in unique_strata:
        mask_s = strata_series == s
        strata_psu_map[s] = np.unique(psu_series[mask_s])

    psu_indices_map = {}
    for p in np.unique(psu_series):
        psu_indices_map[p] = np.where(psu_series == p)[0]

    boot_results = []
    for _ in range(n_replicates):
        selected_chunks = []
        for s in unique_strata:
            psus = strata_psu_map[s]
            sampled_psus = rng.choice(psus, size=len(psus), replace=True)
            for p in sampled_psus:
                selected_chunks.append(psu_indices_map[p])

        boot_idx = np.concatenate(selected_chunks)
        y_t = y_true_arr[boot_idx]
        y_p = y_prob_arr[boot_idx]
        y_w = weight_arr[boot_idx]

        # Skip degenerate replicates
        if y_t.sum() == 0 or y_t.sum() == len(y_t):
            continue

        m = compute_all_metrics(y_t, y_p, sample_weight=y_w,
                                screening_capacities=screening_capacities,
                                compute_unweighted=False,
                                compute_calibration=False)
        boot_results.append(m)

    if not boot_results:
        return {}

    boot_df = pd.DataFrame(boot_results)
    summary = {}
    for col in boot_df.columns:
        vals = boot_df[col].dropna().values
        if len(vals) == 0:
            continue
        summary[col] = {
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
            'ci_lower_95': float(np.percentile(vals, 2.5)),
            'ci_upper_95': float(np.percentile(vals, 97.5)),
            'std': float(np.std(vals)),
        }

    return summary


def paired_bootstrap_test(boot_a: List[float],
                           boot_b: List[float],
                           observed_diff: float) -> float:
    """
    Paired bootstrap hypothesis test: P(|B_a - B_b| >= |obs_diff|).
    Returns two-tailed p-value.
    """
    a = np.array(boot_a)
    b = np.array(boot_b)
    diffs = a - b
    centered_diffs = diffs - diffs.mean()
    p_val = float(np.mean(np.abs(centered_diffs) >= abs(observed_diff)))
    return p_val

