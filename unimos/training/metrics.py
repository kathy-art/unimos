"""
metrics.py — UniMoS evaluation metrics (SPEC-07).

Two public functions:

  find_best_threshold(prob, y_class) -> float
      Searches [0.01, 0.99] for the threshold maximising F1 on a given
      dataset (call on val set, pass result to test compute_metrics).

  compute_metrics(prob_synergy, y_class, ..., threshold=0.5) -> dict
      Returns the report metrics (keys in REPORT_KEYS):
      auroc, auprc, accuracy, balanced_accuracy, f1, precision, recall,
      specificity, mcc, and confusion_matrix (nested {tp, fp, tn, fn}).
      Balanced accuracy + MCC are robust on imbalanced data where raw
      accuracy is dominated by the majority (negative) class.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPORT_KEYS = (
    "auroc", "auprc", "accuracy", "balanced_accuracy",
    "f1", "precision", "recall", "specificity", "mcc",
    "confusion_matrix",
)
_EXPECTED_KEYS = frozenset(REPORT_KEYS)


def find_best_threshold(
    prob_synergy: np.ndarray,
    y_class: np.ndarray,
    n_thresholds: int = 200,
) -> float:
    """
    Return the threshold in [0.01, 0.99] that maximises F1 on the given split.

    Call on the validation set; pass the result as ``threshold`` when calling
    ``compute_metrics`` on the test set.  NaN rows in y_class are excluded.
    Returns 0.5 when no positive labels are present.
    """
    prob = np.asarray(prob_synergy, dtype=np.float64)
    y    = np.asarray(y_class,      dtype=np.float64)

    valid = np.isfinite(y)
    prob, y = prob[valid], y[valid]

    if len(y) == 0 or (y == 1).sum() == 0:
        return 0.5

    candidates = np.linspace(0.01, 0.99, n_thresholds)
    best_thr, best_f1 = 0.5, -1.0
    y_int = y.astype(int)
    for thr in candidates:
        pred = (prob >= thr).astype(int)
        score = float(f1_score(y_int, pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr


def compute_metrics(
    prob_synergy: np.ndarray,   # (N,) sigmoid(logit_class)
    y_class: np.ndarray,        # (N,) 0/1, NaN rows skipped
    y_ri_pred_A: np.ndarray,    # (N,) kept for API compatibility; unused
    y_ri_true_A: np.ndarray,    # (N,) kept for API compatibility; unused
    y_ri_pred_B: np.ndarray,    # (N,) kept for API compatibility; unused
    y_ri_true_B: np.ndarray,    # (N,) kept for API compatibility; unused
    ri_sum_prob: np.ndarray,    # (N,) kept for API compatibility; unused
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute the 6 report metrics.

    Parameters
    ----------
    prob_synergy : predicted synergy probability (after sigmoid)
    y_class      : ground-truth binary labels (0/1); NaN rows are excluded
    y_ri_pred_*  : unused; retained for backward-compatible call signatures
    y_ri_true_*  : unused; retained for backward-compatible call signatures
    ri_sum_prob  : unused; retained for backward-compatible call signatures
    threshold    : decision threshold for accuracy / f1 / precision / recall

    Returns
    -------
    dict with keys in REPORT_KEYS; all values are Python float
    (or float('nan') for undefined metrics).
    """
    prob = np.asarray(prob_synergy, dtype=np.float64)
    y    = np.asarray(y_class,      dtype=np.float64)

    valid = np.isfinite(y)
    yv = y[valid]
    pv = prob[valid]

    n_total = len(yv)
    n_pos   = int((yv == 1).sum())
    n_neg   = int((yv == 0).sum())
    has_both = (n_pos > 0) and (n_neg > 0)

    if has_both and n_total > 0:
        try:
            auroc = float(roc_auc_score(yv, pv))
            auprc = float(average_precision_score(yv, pv))
        except Exception:
            auroc = auprc = float("nan")
    else:
        auroc = auprc = float("nan")

    if n_total > 0:
        pred = (pv >= threshold).astype(int)
        yv_int = yv.astype(int)
        f1     = float(f1_score(yv_int, pred, zero_division=0))
        acc    = float(accuracy_score(yv_int, pred))
        recall = float(recall_score(yv_int, pred, zero_division=0))
        prec   = float(precision_score(yv_int, pred, zero_division=0))
        # confusion_matrix with labels=[0,1] → (tn, fp, fn, tp) even if a class is absent
        tn, fp, fn, tp = (int(x) for x in
                          confusion_matrix(yv_int, pred, labels=[0, 1]).ravel())
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        if n_pos > 0 and n_neg > 0:
            mcc     = float(matthews_corrcoef(yv_int, pred))
            bal_acc = float(balanced_accuracy_score(yv_int, pred))
        else:
            mcc = bal_acc = float("nan")
    else:
        f1 = acc = recall = prec = spec = mcc = bal_acc = float("nan")
        tp = fp = tn = fn = 0

    return {
        "auroc":             auroc,
        "auprc":             auprc,
        "accuracy":          acc,
        "balanced_accuracy": bal_acc,
        "f1":                f1,
        "precision":         prec,
        "recall":            recall,
        "specificity":       spec,
        "mcc":               mcc,
        "confusion_matrix":  {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
