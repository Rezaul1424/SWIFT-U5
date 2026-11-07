"""
Metrics engine for SWIFT-U5 study evaluation.

Computes weighted and unweighted:
  - Classification: accuracy, precision, recall/sensitivity, specificity, F1,
    balanced accuracy, NPV, MCC, confusion matrix
  - Discrimination: AUROC, AUPRC (Average Precision, trapezoidal)
  - Probability / calibration: Brier score, Null Brier, Brier Skill Score,
    calibration slope, calibration intercept, ECE
  - Screening utility: sensitivity, PPV, NNS at 5%, 10%, 20% capacity
  - Decision analysis: net benefit (DCA), treat-all, treat-none references
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression


def _weighted_roc_auc(y_true: np.ndarray, y_score: np.ndarray,
                       sample_weight: Optional[np.ndarray] = None) -> float:
    """Compute weighted AUROC."""
    try:
        return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))
    except Exception:
        return float('nan')


def _weighted_average_precision(y_true: np.ndarray, y_score: np.ndarray,
                                  sample_weight: Optional[np.ndarray] = None) -> float:
    """Compute weighted AUPRC (Average Precision, trapezoidal-equivalent via sklearn)."""
    try:
        return float(average_precision_score(y_true, y_score, sample_weight=sample_weight))
    except Exception:
        return float('nan')


def _weighted_brier_score(y_true: np.ndarray, y_prob: np.ndarray,
                            sample_weight: Optional[np.ndarray] = None) -> float:
    """Compute weighted Brier score."""
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    sample_weight = sample_weight / sample_weight.mean()
    return float(np.average((y_true - y_prob) ** 2, weights=sample_weight))


def _null_brier_score(y_true: np.ndarray,
                      sample_weight: Optional[np.ndarray] = None) -> float:
    """Null (reference) Brier score: predicting the weighted prevalence for all.
    BSS = 1 - Brier / NullBrier; a BSS > 0 means the model beats the null.
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    w = sample_weight / sample_weight.mean()
    p_null = float(np.average(y_true, weights=w))
    null_probs = np.full(len(y_true), p_null)
    return float(np.average((y_true - null_probs) ** 2, weights=w))


def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                       sample_weight: Optional[np.ndarray] = None) -> float:
    """Return probability threshold that maximises the Youden index (sensitivity + specificity - 1)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, sample_weight=sample_weight)
    j = tpr - fpr
    best = np.argmax(j)
    return float(thresholds[best])


def _classification_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                                          threshold: float,
                                          sample_weight: Optional[np.ndarray] = None) -> Dict:
    """
    Compute weighted classification metrics at a given probability threshold.

    Returns: accuracy, precision, recall (sensitivity), specificity, f1,
             balanced_accuracy, npv, mcc, tp, tn, fp, fn (all weighted).
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    w = sample_weight / sample_weight.mean()

    y_pred = (y_prob >= threshold).astype(int)

    tp = float((w * y_pred * y_true).sum())
    tn = float((w * (1 - y_pred) * (1 - y_true)).sum())
    fp = float((w * y_pred * (1 - y_true)).sum())
    fn = float((w * (1 - y_pred) * y_true).sum())

    n_w = w.sum()
    accuracy          = (tp + tn) / n_w if n_w > 0 else float('nan')
    precision         = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall            = tp / (tp + fn) if (tp + fn) > 0 else float('nan')   # sensitivity
    specificity       = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    npv               = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    f1                = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else float('nan'))
    balanced_accuracy = (recall + specificity) / 2.0 if (not np.isnan(recall) and not np.isnan(specificity)) else float('nan')

    # Matthews Correlation Coefficient (generalised to weighted version)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float((tp * tn - fp * fn) / denom) if denom > 0 else float('nan')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'npv': npv,
        'f1': f1,
        'balanced_accuracy': balanced_accuracy,
        'mcc': mcc,
        'tp_w': tp, 'tn_w': tn, 'fp_w': fp, 'fn_w': fn,
        'threshold_used': threshold,
    }


def _calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray,
                                 sample_weight: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Compute calibration slope and intercept via logistic recalibration.
    slope=1.0 and intercept=0.0 indicate perfect calibration.
    Uses modest L2 regularization to ensure stable, instant convergence.
    """
    eps = 1e-4
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    log_odds = np.log(y_prob_clipped / (1.0 - y_prob_clipped)).reshape(-1, 1)

    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    sample_weight = sample_weight / sample_weight.mean()

    try:
        lr = LogisticRegression(penalty='l2', C=10.0, solver='liblinear', max_iter=200, tol=1e-4)
        lr.fit(log_odds, y_true, sample_weight=sample_weight)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
        return slope, intercept
    except Exception:
        return float('nan'), float('nan')


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               sample_weight: Optional[np.ndarray] = None,
                               n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-frequency binning.
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    sample_weight = sample_weight / sample_weight.sum()

    order = np.argsort(y_prob)
    y_true_s = y_true[order]
    y_prob_s = y_prob[order]
    w_s = sample_weight[order]

    bin_edges = np.array_split(np.arange(len(y_true_s)), n_bins)
    ece = 0.0
    for bin_idx in bin_edges:
        if len(bin_idx) == 0:
            continue
        bin_w = w_s[bin_idx]
        bin_total_w = bin_w.sum()
        if bin_total_w == 0:
            continue
        conf = np.average(y_prob_s[bin_idx], weights=bin_w)
        acc = np.average(y_true_s[bin_idx], weights=bin_w)
        ece += bin_total_w * abs(conf - acc)

    return float(ece)


def _screening_metrics_at_capacity(y_true: np.ndarray, y_prob: np.ndarray,
                                  capacity: float,
                                  sample_weight: Optional[np.ndarray] = None) -> Dict:
    """
    Compute Sensitivity, PPV, and NNS at top `capacity` fraction (e.g., 0.05 = top 5%).
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    sample_weight = sample_weight / sample_weight.mean()

    n_screen = max(1, int(np.ceil(len(y_true) * capacity)))
    top_idx = np.argsort(y_prob)[-n_screen:]
    screen_mask = np.zeros(len(y_true), dtype=bool)
    screen_mask[top_idx] = True

    tp_w = (sample_weight * screen_mask * y_true).sum()
    all_pos_w = (sample_weight * y_true).sum()
    screened_w = (sample_weight * screen_mask).sum()

    sensitivity = float(tp_w / all_pos_w) if all_pos_w > 0 else float('nan')
    ppv = float(tp_w / screened_w) if screened_w > 0 else float('nan')
    nns = float(1.0 / ppv) if ppv and ppv > 0 else float('nan')

    return {
        f'sensitivity_{int(capacity*100)}pct': sensitivity,
        f'ppv_{int(capacity*100)}pct': ppv,
        f'nns_{int(capacity*100)}pct': nns,
    }


def compute_all_metrics(y_true: np.ndarray,
                      y_prob: np.ndarray,
                      sample_weight: Optional[np.ndarray] = None,
                      screening_capacities: List[float] = None,
                      compute_unweighted: bool = True,
                      compute_calibration: bool = True) -> Dict:
    """
    Compute the complete SWIFT-U5 evaluation metric suite (weighted and optionally unweighted).

    Includes:
    - Discrimination: AUROC, AUPRC
    - Calibration: Brier score, Null Brier, BSS, slope, intercept, ECE
      (calibration sub-metrics skipped when compute_calibration=False for bootstrap speed)
    - Classification at Youden threshold: accuracy, precision, recall/sensitivity,
      specificity, F1, balanced accuracy, NPV, MCC, weighted confusion matrix cells
    - Screening utility: sensitivity, PPV, NNS at each capacity fraction
    - Sample / prevalence statistics

    Set compute_calibration=False during bootstrap loops for a ~100x speedup.
    """
    if screening_capacities is None:
        screening_capacities = [0.05, 0.10, 0.20]

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    metrics = {}

    # ------------------------------------------------------------------ #
    # Weighted metrics                                                     #
    # ------------------------------------------------------------------ #
    metrics['auroc_w'] = _weighted_roc_auc(y_true, y_prob, sample_weight)
    metrics['auprc_w'] = _weighted_average_precision(y_true, y_prob, sample_weight)
    metrics['brier_w'] = _weighted_brier_score(y_true, y_prob, sample_weight)

    # Null Brier and Brier Skill Score
    null_brier_w = _null_brier_score(y_true, sample_weight)
    metrics['brier_null_w'] = null_brier_w
    metrics['brier_skill_score_w'] = (
        float(1.0 - metrics['brier_w'] / null_brier_w)
        if null_brier_w and null_brier_w > 0 else float('nan')
    )

    if compute_calibration:
        cal_slope, cal_int = _calibration_slope_intercept(y_true, y_prob, sample_weight)
        metrics['cal_slope_w'] = cal_slope
        metrics['cal_intercept_w'] = cal_int
        metrics['ece_w'] = _expected_calibration_error(y_true, y_prob, sample_weight)

    # Classification metrics at Youden threshold (weighted)
    try:
        thr_w = _youden_threshold(y_true, y_prob, sample_weight)
        clf_w = _classification_metrics_at_threshold(y_true, y_prob, thr_w, sample_weight)
        for k, v in clf_w.items():
            metrics[f'{k}_w'] = v
    except Exception:
        pass

    for cap in screening_capacities:
        screen_m = _screening_metrics_at_capacity(y_true, y_prob, cap, sample_weight)
        metrics.update({f'{k}_w': v for k, v in screen_m.items()})

    # ------------------------------------------------------------------ #
    # Unweighted metrics                                                   #
    # ------------------------------------------------------------------ #
    if compute_unweighted:
        metrics['auroc_uw'] = _weighted_roc_auc(y_true, y_prob, None)
        metrics['auprc_uw'] = _weighted_average_precision(y_true, y_prob, None)
        metrics['brier_uw'] = _weighted_brier_score(y_true, y_prob, None)

        null_brier_uw = _null_brier_score(y_true, None)
        metrics['brier_null_uw'] = null_brier_uw
        metrics['brier_skill_score_uw'] = (
            float(1.0 - metrics['brier_uw'] / null_brier_uw)
            if null_brier_uw and null_brier_uw > 0 else float('nan')
        )

        if compute_calibration:
            cal_slope_uw, cal_int_uw = _calibration_slope_intercept(y_true, y_prob, None)
            metrics['cal_slope_uw'] = cal_slope_uw
            metrics['cal_intercept_uw'] = cal_int_uw
            metrics['ece_uw'] = _expected_calibration_error(y_true, y_prob, None)

        # Classification metrics at Youden threshold (unweighted)
        try:
            thr_uw = _youden_threshold(y_true, y_prob, None)
            clf_uw = _classification_metrics_at_threshold(y_true, y_prob, thr_uw, None)
            for k, v in clf_uw.items():
                metrics[f'{k}_uw'] = v
        except Exception:
            pass

        for cap in screening_capacities:
            screen_m_uw = _screening_metrics_at_capacity(y_true, y_prob, cap, None)
            metrics.update({f'{k}_uw': v for k, v in screen_m_uw.items()})

    # ------------------------------------------------------------------ #
    # Sample statistics                                                    #
    # ------------------------------------------------------------------ #
    metrics['n_test'] = int(len(y_true))
    metrics['n_deaths_test'] = int(y_true.sum())
    metrics['prevalence_uw'] = float(y_true.mean())
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        metrics['prevalence_w'] = float(np.average(y_true, weights=w))
        # Effective sample size (Kish approximation)
        metrics['eff_n'] = float(w.sum() ** 2 / (w ** 2).sum())
    else:
        metrics['prevalence_w'] = metrics['prevalence_uw']
        metrics['eff_n'] = float(len(y_true))
    # Legacy key kept for backward compat
    metrics['prevalence'] = metrics['prevalence_uw']

    return metrics


def compute_decision_curve(y_true: np.ndarray,
                             y_prob: np.ndarray,
                             sample_weight: Optional[np.ndarray] = None,
                             thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Decision Curve Analysis: compute net benefit across risk thresholds.
    net_benefit(t) = TP/n - FP/n * t/(1-t)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.30, 60)
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    sample_weight = sample_weight / sample_weight.mean()

    results = []
    n = len(y_true)

    for t in thresholds:
        screen = (y_prob >= t).astype(int)
        tp = (sample_weight * screen * y_true).sum()
        fp = (sample_weight * screen * (1 - y_true)).sum()
        nb = tp / n - fp / n * (t / (1 - t))
        results.append({'threshold': float(t), 'net_benefit': float(nb)})

    return pd.DataFrame(results)

