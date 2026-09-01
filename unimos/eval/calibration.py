"""
calibration.py — Phase-0 metrics not covered by unimos/training/metrics.py
(design doc §5.2: "报 AUPRC + top-k precision + 校准/排序指标").

  precision_at_k(prob, y, k)        — fixed-k precision (e.g. k=100)
  precision_at_k_pct(prob, y, pct)  — percentage-k precision (e.g. top 1%)
  brier_score(prob, y)              — mean squared error of predicted
                                       probability vs. 0/1 label; a proper
                                       calibration score that needs no binning
                                       (unlike ECE), which matters at 6.6%
                                       positive rate where bins run thin.
  spearman_ranking(prob, continuous)— rank correlation between predicted
                                       probability and a continuous synergy
                                       score (Loewe), when available, as a
                                       ranking-quality check independent of
                                       the binary threshold.

All functions drop NaN rows (in y or continuous) the same way
unimos.training.metrics.compute_metrics does.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _valid(y: np.ndarray) -> np.ndarray:
    return np.isfinite(y)


def precision_at_k(prob: np.ndarray, y: np.ndarray, k: int) -> float:
    """Precision among the top-k highest-scored rows (fixed k).

    Returns NaN if fewer than k valid rows exist (k is not meaningful on a
    test set smaller than k; report percentage-k instead in that case).
    """
    m = _valid(y)
    p, yv = prob[m], y[m]
    if len(yv) < k or k <= 0:
        return float("nan")
    order = np.argsort(-p)[:k]
    return float(yv[order].sum() / k)


def precision_at_k_pct(prob: np.ndarray, y: np.ndarray, pct: float) -> float:
    """Precision among the top `pct` fraction of rows (0 < pct <= 1)."""
    m = _valid(y)
    p, yv = prob[m], y[m]
    k = max(1, int(round(len(yv) * pct)))
    order = np.argsort(-p)[:k]
    return float(yv[order].sum() / k)


def brier_score(prob: np.ndarray, y: np.ndarray) -> float:
    m = _valid(y)
    p, yv = prob[m], y[m]
    if len(yv) == 0:
        return float("nan")
    return float(np.mean((p - yv) ** 2))


def spearman_ranking(prob: np.ndarray, continuous: np.ndarray) -> float:
    """Spearman rho between predicted probability and a continuous synergy
    score (e.g. raw Loewe value). NaN if fewer than 2 valid pairs or the
    continuous column is unavailable / constant.
    """
    m = np.isfinite(prob) & np.isfinite(continuous)
    if m.sum() < 2:
        return float("nan")
    p, c = prob[m], continuous[m]
    if np.all(p == p[0]) or np.all(c == c[0]):
        return float("nan")
    rho, _ = spearmanr(p, c)
    return float(rho)


def phase0_metrics(
    prob: np.ndarray,
    y: np.ndarray,
    loewe_continuous: "np.ndarray | None" = None,
    fixed_k: int = 100,
    pct_ks: tuple[float, ...] = (0.01, 0.05),
) -> dict[str, float]:
    """Bundle of §5.2 metrics beyond AUROC/AUPRC (which compute_metrics already
    covers). k values report NaN gracefully when the split is smaller than k.
    """
    out = {
        f"precision_at_{fixed_k}": precision_at_k(prob, y, fixed_k),
        "brier": brier_score(prob, y),
    }
    for pct in pct_ks:
        out[f"precision_at_top{int(pct * 100)}pct"] = precision_at_k_pct(prob, y, pct)
    if loewe_continuous is not None:
        out["spearman_vs_loewe"] = spearman_ranking(prob, loewe_continuous)
    return out
