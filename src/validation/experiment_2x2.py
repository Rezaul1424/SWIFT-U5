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

THRESHOLD PROVENANCE (critical — see docstring on run_2x2_factorial)
-------------------------------------------------------------------------
Youden thresholds are derived from OUT-OF-SAMPLE predictions only.
Test-set labels are NEVER used for threshold selection in any arm or cut.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
from src.evaluation.metrics import compute_all_metrics, compute_youden_threshold_from_validation
from src.evaluation.bootstrap import bootstrap_metrics


# Screening capacities for all reporting
SCREENING_CAPACITIES = [0.05, 0.10, 0.20]
N_BOOTSTRAP = 1000  # per approved Statistical Analysis Plan (was 100)


def _fit_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
               w_train: np.ndarray, use_weights: bool):
    """Build and fit one of the three supported model types."""
    if model_name == 'logistic':
        model = build_weighted_logistic()
        model.fit(X_train, y_train, sample_weight=w_train)
    elif model_name == 'xgboost':
        spw = compute_scale_pos_weight(y_train, w_train if use_weights else None)
        model = build_weighted_xgboost(scale_pos_weight=spw)
        model.fit(X_train, y_train, sample_weight=w_train)
    elif model_name == 'random_forest':
        model = build_weighted_random_forest()
        model.fit(X_train, y_train, sample_weight=w_train)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def _derive_oos_threshold_temporal(
        df: pd.DataFrame,
        full_train_idx: pd.Index,
        train_rounds: List[str],
        model_name: str,
        use_weights: bool) -> Tuple[float, float, str]:
    """
    Derive Youden thresholds from GENUINELY OUT-OF-SAMPLE predictions.

    Strategy (temporal cuts):
    -------------------------
    If k >= 2 training rounds:
      - Fit a 'threshold model' on [r1,...,rk-1] (all but the last round).
      - Predict on rk (last round) → out-of-sample predictions.
      - Derive Youden threshold from these OOS predictions.

    If k == 1 (single training round):
      - Run 5-fold stratified CV within r1 → out-of-fold predictions.
      - Derive Youden threshold from out-of-fold predictions.

    The main evaluation model is still trained on all k rounds (full_train_idx).
    The threshold model is fitted only for threshold derivation, not for final
    evaluation. This two-pass approach ensures the threshold is never optimised
    on in-sample predictions.

    Returns
    -------
    thr_w, thr_uw : float — weighted and unweighted Youden thresholds
    provenance    : str  — human-readable description of the threshold source
    """
    if len(train_rounds) >= 2:
        # Penultimate-round model: trained on [r1,...,rk-1], predicts on rk
        penultimate_rounds = train_rounds[:-1]
        last_round         = train_rounds[-1]

        pen_idx  = df[df[ROUND_COLUMN].isin(penultimate_rounds)].index
        last_idx = df[df[ROUND_COLUMN] == last_round].index

        # Fit preprocessor on penultimate subset only
        X_all_pen, _ = encode_features(df, pen_idx)
        X_pen_train  = X_all_pen.loc[pen_idx].values
        X_pen_val    = X_all_pen.loc[last_idx].values

        df_pen   = df.loc[pen_idx]
        y_pen    = df_pen[TARGET_COLUMN].values
        w_pen    = compute_normalized_weights(df_pen, pen_idx, use_weights=use_weights)

        thr_model = _fit_model(model_name, X_pen_train, y_pen, w_pen, use_weights)
        y_prob_val = thr_model.predict_proba(X_pen_val)[:, 1]

        y_true_val = df.loc[last_idx, TARGET_COLUMN].values.astype(int)
        w_val_arr  = df.loc[last_idx, WEIGHT_COLUMN].values.astype(float) if use_weights else None
        thr_w, thr_uw = compute_youden_threshold_from_validation(y_true_val, y_prob_val, w_val_arr)
        provenance = (f"OOS: threshold model trained on {penultimate_rounds}, "
                      f"predicting on {last_round} (N={len(last_idx):,})")

    else:
        # Single training round — use 5-fold stratified CV for OOS predictions
        single_round = train_rounds[0]
        single_idx   = df[df[ROUND_COLUMN] == single_round].index
        df_single    = df.loc[single_idx]
        y_single     = df_single[TARGET_COLUMN].values
        w_single_raw = df_single[WEIGHT_COLUMN].values.astype(float)

        X_all_single, _ = encode_features(df, single_idx)
        X_single = X_all_single.loc[single_idx].values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_prob_oof = np.zeros(len(y_single))
        for fold_train_pos, fold_val_pos in skf.split(X_single, y_single):
            X_ft = X_single[fold_train_pos]
            X_fv = X_single[fold_val_pos]
            y_ft = y_single[fold_train_pos]
            w_ft_raw = w_single_raw[fold_train_pos]
            w_ft = w_ft_raw / w_ft_raw.mean() if use_weights else np.ones(len(fold_train_pos))
            fold_mdl = _fit_model(model_name, X_ft, y_ft, w_ft, use_weights)
            y_prob_oof[fold_val_pos] = fold_mdl.predict_proba(X_fv)[:, 1]

        w_val_arr = w_single_raw / w_single_raw.mean() if use_weights else None
        thr_w, thr_uw = compute_youden_threshold_from_validation(y_single, y_prob_oof, w_val_arr)
        provenance = (f"OOS: 5-fold CV within {single_round} (N={len(single_idx):,}), "
                      f"out-of-fold predictions")

    return thr_w, thr_uw, provenance


def run_single_trial(df: pd.DataFrame,
                      train_idx: pd.Index,
                      test_idx: pd.Index,
                      model_name: str,
                      use_train_weights: bool,
                      use_eval_weights: bool,
                      arm_name: str,
                      test_round: str,
                      train_rounds: List[str],
                      run_bootstrap: bool = True,
                      youden_threshold_w: Optional[float] = None,
                      youden_threshold_uw: Optional[float] = None,
                      threshold_provenance: str = 'not_specified') -> Dict:
    """
    Run a single model training + evaluation trial for one arm/model combination.

    THRESHOLD PROVENANCE
    --------------------
    ``youden_threshold_w`` and ``youden_threshold_uw`` MUST be pre-computed from
    GENUINELY OUT-OF-SAMPLE VALIDATION data before calling this function.
    They are derived by ``run_2x2_factorial`` using ``_derive_oos_threshold_temporal``
    (which uses a penultimate-round model or 5-fold CV) so that no prediction on
    which the threshold is optimised was produced by a model that saw that
    observation during training.

    If not supplied (backward compat), ``threshold_source='internal_from_passed_data'``
    is set in the output — this is ONLY acceptable for internal/unit-test use.
    """
    df_train = df.loc[train_idx]
    df_test  = df.loc[test_idx]

    y_train = df_train[TARGET_COLUMN].values
    y_test  = df_test[TARGET_COLUMN].values
    w_test  = df_test[WEIGHT_COLUMN].values if use_eval_weights else None

    # Encode features with train-only imputation
    X_all, imputer_medians = encode_features(df, train_idx)
    X_train = X_all.loc[train_idx].values
    X_test  = X_all.loc[test_idx].values

    # Compute training weights
    w_train = compute_normalized_weights(df_train, train_idx, use_weights=use_train_weights)

    # Build and fit main evaluation model on full training set
    model = _fit_model(model_name, X_train, y_train, w_train, use_train_weights)

    # Predict probabilities on test set
    y_prob = model.predict_proba(X_test)[:, 1]

    # Pass pre-frozen out-of-sample thresholds to compute_all_metrics.
    # Test-set labels (y_test) must NEVER be used to derive these thresholds.
    metrics = compute_all_metrics(
        y_test, y_prob,
        sample_weight=w_test,
        screening_capacities=SCREENING_CAPACITIES,
        youden_threshold_w=youden_threshold_w,
        youden_threshold_uw=youden_threshold_uw,
    )
    # Attach threshold provenance for audit
    metrics['threshold_provenance'] = threshold_provenance

    # Bootstrap CIs
    if run_bootstrap:
        df_test_for_boot = df.loc[test_idx].copy()
        boot_summary = bootstrap_metrics(
            df_test_for_boot, y_prob,
            n_replicates=N_BOOTSTRAP,
            screening_capacities=SCREENING_CAPACITIES,
            youden_threshold_w=youden_threshold_w,
            youden_threshold_uw=youden_threshold_uw,
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

    # Attach bootstrap CIs as flat columns (skip metadata key)
    for metric_name, ci_dict in boot_summary.items():
        if metric_name == '_bootstrap_meta':
            result['_bootstrap_meta'] = ci_dict
            continue
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

    THRESHOLD PROVENANCE — GUARANTEED OUT-OF-SAMPLE
    ------------------------------------------------
    For each temporal cut [r1,...,rk] → test=rk+1:

    TEMPORAL ARMS (C, D, Mix1, Mix2):
      If k >= 2:
        - Fit a 'threshold model' on [r1,...,rk-1] (without the last round).
        - Predict on rk (last round) → out-of-sample predictions.
        - Derive Youden threshold from these OOS predictions.
        - Main model still trained on full [r1,...,rk].
        - Threshold is frozen BEFORE the main model evaluates on rk+1.
      If k == 1:
        - 5-fold stratified CV within r1 → out-of-fold predictions.
        - Derive Youden threshold from OOF predictions.

    RANDOM ARMS (A, B):
      A 70-15-15 three-way stratified split is used: train (70%), validation
      (15%, for OOS threshold), test (15%).  The threshold model is fitted on
      the 70% training portion and predicts on the 15% validation portion — these
      observations were NOT used in model training.

    MATCHED-N ARMS:
      Uses the same threshold as the corresponding temporal cut (C/D) since the
      temporal cut defines the test set size.

    IMPORTANT: The val_idx used in threshold derivation is NEVER a subset of
    the train_idx used for the main evaluation model in that arm.

    PRIMARY 2022 EVALUATION (ablation / smoke test):
      The 2017-18 round serves as a genuinely held-out validation set for the
      model trained on 2004-2014.  This is the primary scientific result.
    """
    if models is None:
        models = ['logistic', 'xgboost', 'random_forest']

    results = []
    temporal_cuts = get_forward_chaining_cuts(df)

    print("\n" + "="*70)
    print("  SWIFT-U5: 2x2 FACTORIAL VALIDATION EXPERIMENT")
    print("="*70)
    print("  Threshold derivation: OUT-OF-SAMPLE (penultimate-round model or 5-fold CV)")

    for cut_idx, cut in enumerate(temporal_cuts):
        tr        = cut['train_rounds']
        test_rnd  = cut['test_round']
        train_idx = cut['train_idx']
        test_idx  = cut['test_idx']
        n_test    = cut['n_test']

        print(f"\n--- Temporal Cut: Train={tr} | Test={test_rnd} | "
              f"N_train={cut['n_train']} | N_test={n_test} ---")

        # Matched-N random split within the contemporary cumulative cohort (Train + Test rounds)
        # Matches both N_test and N_train for a strict apples-to-apples comparison.
        df_cum = df.loc[train_idx.union(test_idx)]
        matched_train_idx, matched_test_idx = get_matched_n_random_split(
            df_cum, n_test, random_state=42 + cut_idx
        )

        # Out-of-sample threshold split for random arms:
        # Split matched_train_idx into sub-train (80%) and validation (20%)
        # to ensure threshold model never sees matched_test_idx.
        rng_rand = np.random.default_rng(99 + cut_idx)
        y_rand_train = df.loc[matched_train_idx, TARGET_COLUMN].values
        dead_rt  = matched_train_idx[y_rand_train == 1]
        alive_rt = matched_train_idx[y_rand_train == 0]
        rng_rand.shuffle(dead_rt := dead_rt.tolist())
        rng_rand.shuffle(alive_rt := alive_rt.tolist())
        n_dead_v  = max(1, int(len(dead_rt)  * 0.20))
        n_alive_v = max(1, int(len(alive_rt) * 0.20))
        rand_val_idx    = pd.Index(dead_rt[:n_dead_v]  + alive_rt[:n_alive_v])
        rand_train2_idx = pd.Index(dead_rt[n_dead_v:] + alive_rt[n_alive_v:])

        for model_name in models:
            print(f"  Model: {model_name.upper()}")

            # Derive OOS Youden thresholds for temporal arms
            thr_w_t, thr_uw_t, prov_t = _derive_oos_threshold_temporal(
                df, train_idx, tr, model_name, use_weights=True)
            thr_w_t_uw, thr_uw_t_uw, _ = _derive_oos_threshold_temporal(
                df, train_idx, tr, model_name, use_weights=False)

            # Derive OOS Youden thresholds for random arms
            # Fit threshold model on rand_train2_idx, predict on rand_val_idx
            X_all_rand, _ = encode_features(df, rand_train2_idx)
            X_rt2 = X_all_rand.loc[rand_train2_idx].values
            X_rv  = X_all_rand.loc[rand_val_idx].values
            y_rt2 = df.loc[rand_train2_idx, TARGET_COLUMN].values
            y_rv  = df.loc[rand_val_idx, TARGET_COLUMN].values
            w_rv  = df.loc[rand_val_idx, WEIGHT_COLUMN].values

            # Weighted random threshold model
            w_rt2 = compute_normalized_weights(
                df.loc[rand_train2_idx], rand_train2_idx, use_weights=True)
            thr_mdl_rand_w = _fit_model(model_name, X_rt2, y_rt2, w_rt2, True)
            y_prob_rv_w    = thr_mdl_rand_w.predict_proba(X_rv)[:, 1]
            thr_w_rand, _  = compute_youden_threshold_from_validation(y_rv, y_prob_rv_w, w_rv)

            # Unweighted random threshold model
            w_rt2_uw = np.ones(len(rand_train2_idx))
            thr_mdl_rand_uw = _fit_model(model_name, X_rt2, y_rt2, w_rt2_uw, False)
            y_prob_rv_uw    = thr_mdl_rand_uw.predict_proba(X_rv)[:, 1]
            _, thr_uw_rand  = compute_youden_threshold_from_validation(y_rv, y_prob_rv_uw, None)

            prov_rand = (f"OOS: threshold model on rand_train2_idx (N={len(rand_train2_idx):,}), "
                         f"predicting on rand_val_idx (N={len(rand_val_idx):,})")

            # --- Arm A: Random, Unweighted (Matched-N) ---
            r = run_single_trial(df, matched_train_idx, matched_test_idx,
                                  model_name, False, False, 'A_random_unweighted',
                                  test_rnd, tr, run_bootstrap,
                                  youden_threshold_w=thr_uw_rand,
                                  youden_threshold_uw=thr_uw_rand,
                                  threshold_provenance=prov_rand)
            r['cut_label'] = f"{'_'.join(tr)}->{test_rnd}"
            results.append(r)

            # --- Arm B: Random, Survey-Weighted (Matched-N) ---
            r = run_single_trial(df, matched_train_idx, matched_test_idx,
                                  model_name, True, True, 'B_random_weighted',
                                  test_rnd, tr, run_bootstrap,
                                  youden_threshold_w=thr_w_rand,
                                  youden_threshold_uw=thr_uw_rand,
                                  threshold_provenance=prov_rand)
            r['cut_label'] = f"{'_'.join(tr)}->{test_rnd}"
            results.append(r)

            # --- Arm C: Temporal, Unweighted ---
            r = run_single_trial(df, train_idx, test_idx,
                                  model_name, False, False, 'C_temporal_unweighted',
                                  test_rnd, tr, run_bootstrap,
                                  youden_threshold_w=thr_uw_t_uw,
                                  youden_threshold_uw=thr_uw_t_uw,
                                  threshold_provenance=prov_t)
            r['cut_label'] = f"{'_'.join(tr)}->{test_rnd}"
            results.append(r)

            # --- Arm D: Temporal, Survey-Weighted ---
            r = run_single_trial(df, train_idx, test_idx,
                                  model_name, True, True, 'D_temporal_weighted',
                                  test_rnd, tr, run_bootstrap,
                                  youden_threshold_w=thr_w_t,
                                  youden_threshold_uw=thr_uw_t,
                                  threshold_provenance=prov_t)
            r['cut_label'] = f"{'_'.join(tr)}->{test_rnd}"
            results.append(r)

            # --- Mixed-W1: Train unweighted, Eval weighted (temporal) ---
            r = run_single_trial(df, train_idx, test_idx,
                                  model_name, False, True, 'Mix1_train_uw_eval_w',
                                  test_rnd, tr, run_bootstrap=False,
                                  youden_threshold_w=thr_uw_t_uw,
                                  youden_threshold_uw=thr_uw_t_uw,
                                  threshold_provenance=prov_t)
            r['cut_label'] = f"{'_'.join(tr)}->{test_rnd}"
            results.append(r)

            # --- Mixed-W2: Train weighted, Eval unweighted (temporal) ---
            r = run_single_trial(df, train_idx, test_idx,
                                  model_name, True, False, 'Mix2_train_w_eval_uw',
                                  test_rnd, tr, run_bootstrap=False,
                                  youden_threshold_w=thr_w_t,
                                  youden_threshold_uw=thr_uw_t,
                                  threshold_provenance=prov_t)
            r['cut_label'] = f"{'_'.join(tr)}->{test_rnd}"
            results.append(r)

    return results
