"""
Metrics engine for SWIFT-U5 study evaluation.

Computes weighted and unweighted:
  - AUROC, AUPRC (Average Precision)
  - Brier Score
  - Calibration slope and intercept (logistic recalibration)
  - Expected Calibration Error (ECE)
  - Screening metrics at top 5%, 10%, 20% capacity: Sensitivity, PPV, NNS
  - Decision Curve Analysis net benefit
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
    """Compute weighted AUPRC (Average Precision)."""
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
    Compute evaluation metrics (weighted and optionally unweighted).
    Set compute_calibration=False during bootstrap loops for 100x speedup.
    """
    if screening_capacities is None:
        screening_capacities = [0.05, 0.10, 0.20]

    metrics = {}

    # --- Weighted metrics ---
    metrics['auroc_w'] = _weighted_roc_auc(y_true, y_prob, sample_weight)
    metrics['auprc_w'] = _weighted_average_precision(y_true, y_prob, sample_weight)
    metrics['brier_w'] = _weighted_brier_score(y_true, y_prob, sample_weight)

    if compute_calibration:
        cal_slope, cal_int = _calibration_slope_intercept(y_true, y_prob, sample_weight)
        metrics['cal_slope_w'] = cal_slope
        metrics['cal_intercept_w'] = cal_int
        metrics['ece_w'] = _expected_calibration_error(y_true, y_prob, sample_weight)

    for cap in screening_capacities:
        screen_m = _screening_metrics_at_capacity(y_true, y_prob, cap, sample_weight)
        metrics.update({f'{k}_w': v for k, v in screen_m.items()})

    # --- Unweighted metrics ---
    if compute_unweighted:
        metrics['auroc_uw'] = _weighted_roc_auc(y_true, y_prob, None)
        metrics['auprc_uw'] = _weighted_average_precision(y_true, y_prob, None)
        metrics['brier_uw'] = _weighted_brier_score(y_true, y_prob, None)
        if compute_calibration:
            cal_slope_uw, cal_int_uw = _calibration_slope_intercept(y_true, y_prob, None)
            metrics['cal_slope_uw'] = cal_slope_uw
            metrics['cal_intercept_uw'] = cal_int_uw
            metrics['ece_uw'] = _expected_calibration_error(y_true, y_prob, None)

        for cap in screening_capacities:
            screen_m_uw = _screening_metrics_at_capacity(y_true, y_prob, cap, None)
            metrics.update({f'{k}_uw': v for k, v in screen_m_uw.items()})

    metrics['n_test'] = int(len(y_true))
    metrics['n_deaths_test'] = int(y_true.sum())
    metrics['prevalence'] = float(y_true.mean())

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

