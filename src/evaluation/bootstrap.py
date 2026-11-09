"""
PSU-stratified cluster bootstrap for SWIFT-U5.

Resamples PSUs within strata (DHS survey design-aware resampling) to produce
95% CIs and paired bootstrap test statistics.

IMPORTANT — PSU IDENTITY
------------------------
DHS re-uses integer PSU identifiers across survey waves; a raw PSU ID of "361"
in 2004 is a *different* physical cluster from "361" in 2022.  When evaluating
on a pooled multi-wave test set, the ``psu_col`` and ``strata_col`` parameters
must therefore reference the **wave-scoped composite columns** ``psu_wave`` and
``strata_wave`` (created by ``cohort.filter_exposure_complete_cohort``), not the
raw ``psu`` / ``strata`` columns.  The defaults below enforce this.
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
                      n_replicates: int = 1000,
                      psu_col: str = 'psu_wave',
                      strata_col: str = 'strata_wave',
                      target_col: str = 'target_u5m',
                      weight_col: str = 'sample_weight',
                      screening_capacities: Optional[List[float]] = None,
                      seed: int = 42,
                      youden_threshold_w: Optional[float] = None,
                      youden_threshold_uw: Optional[float] = None) -> Dict:
    """
    PSU-stratified cluster bootstrap to compute 95% CIs for metrics.
    Default 1,000 replicates per approved Statistical Analysis Plan.
    Precomputes cluster-to-index mappings for efficient execution.

    Parameters
    ----------
    psu_col : str
        Column containing the PSU identifier.  For multi-wave test sets, pass
        ``'psu_wave'`` (the wave-prefixed composite, e.g. ``"2022_361"``), which
        guarantees uniqueness across waves.  Defaults to ``'psu_wave'``.
    strata_col : str
        Column containing the stratum identifier.  Use ``'strata_wave'`` for
        multi-wave test sets.  Defaults to ``'strata_wave'``.

    Returns
    -------
    Dict with keys:
      - One entry per numeric metric: ``{metric_name: {mean, median, ci_lower_95, ci_upper_95, std}}``
      - ``'_bootstrap_meta'``: dict recording ``n_requested``, ``n_succeeded``,
        ``n_failed_degenerate``, ``non_numeric_fields_excluded`` (list of field
        names skipped because they were non-numeric metadata).
    """
    if screening_capacities is None:
        screening_capacities = [0.05, 0.10, 0.20]

    rng = np.random.default_rng(seed)

    # Convert arrays to numpy for C-speed indexing
    y_true_arr = df_test[target_col].values.astype(int)
    weight_arr = df_test[weight_col].values.astype(float)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    # Precompute cluster index groups (avoids slow pandas groupby inside loop)
    psu_series    = df_test[psu_col].values
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
    n_failed_degenerate = 0

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

        # Skip degenerate replicates (all-positive or all-negative outcome)
        # and record them explicitly — do not silently suppress.
        if y_t.sum() == 0 or y_t.sum() == len(y_t):
            n_failed_degenerate += 1
            continue

        m = compute_all_metrics(y_t, y_p, sample_weight=y_w,
                                screening_capacities=screening_capacities,
                                compute_unweighted=False,
                                compute_calibration=False,
                                youden_threshold_w=youden_threshold_w,
                                youden_threshold_uw=youden_threshold_uw)
        boot_results.append(m)

    n_succeeded = len(boot_results)

    if not boot_results:
        return {
            '_bootstrap_meta': {
                'n_requested': n_replicates,
                'n_succeeded': 0,
                'n_failed_degenerate': n_failed_degenerate,
                'non_numeric_fields_excluded': [],
            }
        }

    boot_df = pd.DataFrame(boot_results)
    summary = {}
    non_numeric_excluded = []

    for col in boot_df.columns:
        vals = boot_df[col].dropna().values
        if len(vals) == 0:
            continue
        # Skip non-numeric columns (e.g., 'threshold_source' string metadata)
        # Record them explicitly rather than silently suppressing.
        try:
            vals = vals.astype(float)
        except (ValueError, TypeError):
            non_numeric_excluded.append(col)
            continue
        summary[col] = {
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
            'ci_lower_95': float(np.percentile(vals, 2.5)),
            'ci_upper_95': float(np.percentile(vals, 97.5)),
            'std': float(np.std(vals)),
        }

    # Attach bootstrap provenance metadata as a reserved key
    summary['_bootstrap_meta'] = {
        'n_requested': n_replicates,
        'n_succeeded': n_succeeded,
        'n_failed_degenerate': n_failed_degenerate,
        'non_numeric_fields_excluded': non_numeric_excluded,
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

