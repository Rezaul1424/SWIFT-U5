"""
Decision Curve Analysis (DCA) module for SWIFT-U5.
Computes net benefit across risk thresholds for model comparison.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_decision_curve(y_true: np.ndarray,
                            y_prob: np.ndarray,
                            sample_weight: Optional[np.ndarray] = None,
                            thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Decision Curve Analysis: compute net benefit across risk thresholds.
    net_benefit(t) = TP/n - FP/n * t/(1-t)

    Also computes 'treat all' and 'treat none' reference curves.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.30, 60)
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    sample_weight = sample_weight / sample_weight.mean()

    n = len(y_true)
    prevalence = np.average(y_true, weights=sample_weight)
    results = []

    for t in thresholds:
        screen = (y_prob >= t).astype(float)
        tp = float((sample_weight * screen * y_true).sum())
        fp = float((sample_weight * screen * (1 - y_true)).sum())
        nb_model = tp / n - fp / n * (t / (1 - t)) if t < 1.0 else float('nan')

        # 'Treat all' net benefit
        nb_treat_all = prevalence - (1 - prevalence) * (t / (1 - t)) if t < 1.0 else float('nan')

        results.append({
            'threshold': float(t),
            'net_benefit_model': float(nb_model),
            'net_benefit_treat_all': float(nb_treat_all),
            'net_benefit_treat_none': 0.0,
        })

    return pd.DataFrame(results)

