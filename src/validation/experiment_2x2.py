"""
2x2 Factorial Experiment Runner for SWIFT-U5.

Implements all experimental arms:
  Arm A: Random split, Unweighted
  Arm B: Random split, Survey-Weighted
  Arm C: Forward temporal, Unweighted
  Arm D: Forward temporal, Survey-Weighted

Plus diagnostic arms:
  Matched-N: Random split with test set matched to temporal test set size
  Mixed-W1: Train unweighted / Test weighted
  Mixed-W2: Train weighted / Test unweighted
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.validation.splits import (
    ROUND_ORDER, FEATURE_COLUMNS, TARGET_COLUMN, WEIGHT_COLUMN,
    ROUND_COLUMN, DIVISION_COLUMN, PSU_COLUMN, STRATA_COLUMN,
    get_forward_chaining_cuts, get_random_split, get_matched_n_random_split,
    encode_features, compute_normalized_weights
)
from src.models.baselines import (
    build_weighted_logistic, build_weighted_xgboost, build_weighted_random_forest,
    compute_scale_pos_weight
)
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.bootstrap import bootstrap_metrics


# Screening capacities for all reporting
SCREENING_CAPACITIES = [0.05, 0.10, 0.20]
N_BOOTSTRAP = 1000  # per approved Statistical Analysis Plan (was 100)


def run_single_trial(df: pd.DataFrame,
                      train_idx: pd.Index,
                      test_idx: pd.Index,
                      model_name: str,
                      use_train_weights: bool,
                      use_eval_weights: bool,
                      arm_name: str,
                      test_round: str,
                      train_rounds: List[str],
                      run_bootstrap: bool = True) -> Dict:
    """
    Run a single model training + evaluation trial for one arm/model combination.
    """
    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]

    y_train = df_train[TARGET_COLUMN].values
    y_test = df_test[TARGET_COLUMN].values
    w_test = df_test[WEIGHT_COLUMN].values if use_eval_weights else None

    # Encode features with train-only imputation
    X_all, imputer_medians = encode_features(df, train_idx)
    X_train = X_all.loc[train_idx].values
    X_test = X_all.loc[test_idx].values

    # Compute training weights
    w_train = compute_normalized_weights(df_train, train_idx, use_weights=use_train_weights)

    # Build and fit model
    if model_name == 'logistic':
        model = build_weighted_logistic()
        model.fit(X_train, y_train, sample_weight=w_train)
    elif model_name == 'xgboost':
        spw = compute_scale_pos_weight(y_train, w_train if use_train_weights else None)
        model = build_weighted_xgboost(scale_pos_weight=spw)
        model.fit(X_train, y_train, sample_weight=w_train)
    elif model_name == 'random_forest':
        model = build_weighted_random_forest()
        model.fit(X_train, y_train, sample_weight=w_train)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Predict probabilities on test set
    y_prob = model.predict_proba(X_test)[:, 1]

    # Point estimate metrics
    metrics = compute_all_metrics(
        y_test, y_prob,
        sample_weight=w_test,
        screening_capacities=SCREENING_CAPACITIES
    )

    # Bootstrap CIs
    if run_bootstrap:
        df_test_for_boot = df.loc[test_idx].copy()
        boot_summary = bootstrap_metrics(
            df_test_for_boot, y_prob,
            n_replicates=N_BOOTSTRAP,
            screening_capacities=SCREENING_CAPACITIES
        )
    else:
        boot_summary = {}

    result = {
        'arm': arm_name,
        'model': model_name,
        'test_round': test_round,
        'train_rounds': '+'.join(train_rounds),
        'use_train_weights': use_train_weights,
        'use_eval_weights': use_eval_weights,
        **metrics,
    }

    # Attach bootstrap CIs as flat columns
    for metric_name, ci_dict in boot_summary.items():
        result[f'{metric_name}_ci_lo'] = ci_dict.get('ci_lower_95', float('nan'))
        result[f'{metric_name}_ci_hi'] = ci_dict.get('ci_upper_95', float('nan'))

    # Store predictions for calibration / DCA plots
    result['_y_test'] = y_test
    result['_y_prob'] = y_prob
    result['_w_test'] = df.loc[test_idx, WEIGHT_COLUMN].values

    return result


def run_2x2_factorial(df: pd.DataFrame,
                       models: Optional[List[str]] = None,
                       run_bootstrap: bool = True) -> List[Dict]:
    """
    Run the full 2x2 factorial design experiment across all specified models.

    Returns list of result dicts (one per arm × model × temporal cut).
    """
    if models is None:
        models = ['logistic', 'xgboost', 'random_forest']

    results = []
    temporal_cuts = get_forward_chaining_cuts(df)

    print("\n" + "="*70)
    print("  SWIFT-U5: 2x2 FACTORIAL VALIDATION EXPERIMENT")
    print("="*70)

    for cut in temporal_cuts:
        tr = cut['train_rounds']
        test_rnd = cut['test_round']
        train_idx = cut['train_idx']
        test_idx = cut['test_idx']
        n_test = cut['n_test']

        print(f"\n--- Temporal Cut: Train={tr} | Test={test_rnd} | "
              f"N_train={cut['n_train']} | N_test={n_test} ---")

        # Matched-n random split (same test set size as temporal test)
        matched_train_idx, matched_test_idx = get_matched_n_random_split(df, n_test)
        # Random split across all pooled data
        rand_train_idx, rand_test_idx = get_random_split(df)

        for model_name in models:
            print(f"  Model: {model_name.upper()}")

            # --- Arm A: Random, Unweighted ---
            r = run_single_trial(df, rand_train_idx, rand_test_idx,
                                  model_name, False, False, 'A_random_unweighted',
                                  test_rnd, tr, run_bootstrap)
            r['cut_label'] = f"{'_'.join(tr)}->{test_rnd}"
            results.append(r)

            # --- Arm B: Random, Survey-Weighted ---
            r = run_single_trial(df, rand_train_idx, rand_test_idx,
                                  model_name, True, True, 'B_random_weighted',
                                  test_rnd, tr, run_bootstrap)
            r['cut_label'] = f"{'_'.join(tr)}->{test_rnd}"
            results.append(r)

            # --- Arm C: Temporal, Unweighted ---
            r = run_single_trial(df, train_idx, test_idx,
                                  model_name, False, False, 'C_temporal_unweighted',
                                  test_rnd, tr, run_bootstrap)
            r['cut_label'] = f"{'_'.join(tr)}->{test_rnd}"
            results.append(r)

            # --- Arm D: Temporal, Survey-Weighted ---
            r = run_single_trial(df, train_idx, test_idx,
                                  model_name, True, True, 'D_temporal_weighted',
                                  test_rnd, tr, run_bootstrap)
            r['cut_label'] = f"{'_'.join(tr)}->{test_rnd}"
            results.append(r)

            # --- Mixed-W1: Train unweighted, Eval weighted (temporal) ---
            r = run_single_trial(df, train_idx, test_idx,
                                  model_name, False, True, 'Mix1_train_uw_eval_w',
                                  test_rnd, tr, run_bootstrap=False)
            r['cut_label'] = f"{'_'.join(tr)}->{test_rnd}"
            results.append(r)

            # --- Mixed-W2: Train weighted, Eval unweighted (temporal) ---
            r = run_single_trial(df, train_idx, test_idx,
                                  model_name, True, False, 'Mix2_train_w_eval_uw',
                                  test_rnd, tr, run_bootstrap=False)
            r['cut_label'] = f"{'_'.join(tr)}->{test_rnd}"
            results.append(r)

    return results

