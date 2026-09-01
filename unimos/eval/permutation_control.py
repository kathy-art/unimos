"""
permutation_control.py — Phase-0 implementation of the §5.3 hard gate:
"打乱对照是硬性关卡:任何'提升'都要能扛住'把身份/配对随机打乱后提升应消失'
这一击,才能排除'参数变多'的平凡效应。"

First axis implemented (per user decision): drug-pair <-> cell PAIRING shuffle.
The row's label y and (ik_A, ik_B, pA, pB) stay put; the cell-side fields
(cell_id, c_fn, c_mut_fn) are permuted across rows with a fixed seed, breaking
the true (drug-pair, cell) correspondence while preserving each side's
marginal distribution. If a baseline's real-vs-shuffled metric gap does not
shrink towards ~0, its apparent skill does not come from actually modelling
which cell the pair was tested in — it's a red flag for that baseline (or,
in later phases, for a cell-conditioned component under test).

This module only shuffles and re-scores with an *already-fit* baseline
(fit on the untouched, correctly-paired training data) — the permutation
happens only at evaluation time on val/test, per the CLAUDE.md gate.
"""

from __future__ import annotations

import copy

import numpy as np


def shuffle_pairing(feat: dict, seed: int = 0) -> dict:
    """Return a copy of `feat` with cell-side fields permuted relative to the
    drug-pair-side fields. y (label) travels with the drug-pair side, i.e.
    unchanged — only the cell context attached to it changes.
    """
    rng = np.random.default_rng(seed)
    n = len(feat["y"])
    perm = rng.permutation(n)

    out = copy.copy(feat)  # shallow copy; overwrite cell-side keys below
    for key in ("cell_id", "c_fn", "c_mut_fn"):
        if key in feat:
            out[key] = feat[key][perm]
    # pair_key/ik_A/ik_B/pA/pB/y/loewe stay in original row order (unshuffled)
    return out


def run_control(
    baseline,
    real_feat: dict,
    metric_fn,
    seed: int = 0,
) -> dict:
    """
    Score a fitted `baseline` on `real_feat` and on a pairing-shuffled copy,
    both through the same `metric_fn(prob, y) -> dict[str, float]`.

    Returns {"real": {...}, "shuffled": {...}, "delta": {...}} where delta =
    real - shuffled for every metric key present in both (positive delta =
    real pairing helps, as expected for a genuinely cell-conditioned signal).
    """
    shuffled_feat = shuffle_pairing(real_feat, seed=seed)

    prob_real = baseline.predict_proba(real_feat)
    prob_shuf = baseline.predict_proba(shuffled_feat)

    m_real = metric_fn(prob_real, real_feat["y"])
    m_shuf = metric_fn(prob_shuf, shuffled_feat["y"])

    delta = {}
    for k, v in m_real.items():
        if k not in m_shuf or not isinstance(v, (int, float)):
            continue
        v2 = m_shuf[k]
        if isinstance(v2, (int, float)) and np.isfinite(v) and np.isfinite(v2):
            delta[k] = v - v2
    return {"real": m_real, "shuffled": m_shuf, "delta": delta}
