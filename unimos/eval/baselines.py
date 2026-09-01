"""
baselines.py — Phase-0 sanity baselines (design doc §5.1 / CLAUDE.md GATE
discipline: "每个组件必须打赢基线才接入").

Four baseline families, all with a common fit(train_feat) / predict_proba(test_feat)
interface, where *_feat is the dict returned by unimos.eval.features.extract_bulk:

  MeanBaseline        — pair_mean -> drug_mean -> cell_mean -> global_mean
                         fallback chain (Virtual Cell Challenge's "hard to beat
                         the mean" baseline family; drug_mean/cell_mean/global_mean
                         are reported separately too, see `variant=`).
  OneHotBaseline      — logistic regression on one-hot(drug1) + one-hot(drug2)
                         + one-hot(cell)  (SynVerse's "killer baseline": pure
                         entity-identity memorisation, no structure/features at all).
  RFBaseline          — RandomForestClassifier on cat(pA, pB, c_fn, c_mut_fn).
  XGBBaseline         — XGBoost on the same 268-dim input.

All fit calls use train-fold-only statistics (no leakage): one-hot vocabularies,
per-pair/per-drug/per-cell means, and RF/XGB parameters are all derived from
the training split only.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

MeanVariant = Literal["pair_drug_cell_global", "drug_mean", "cell_mean", "global_mean"]


def _finite_mask(feat: dict) -> np.ndarray:
    return np.isfinite(feat["y"])


class MeanBaseline:
    """
    variant="pair_drug_cell_global": fallback chain
        pair_mean(pair_key) if seen in train
        else drug_mean(avg over both drugs' individual train positive rates)
        else cell_mean(cell_id) if seen in train
        else global_mean
    variant="drug_mean" / "cell_mean" / "global_mean": single-level means only,
        falling back to global_mean when the entity is unseen in train.
    """

    def __init__(self, variant: MeanVariant = "pair_drug_cell_global"):
        self.variant = variant

    def fit(self, feat: dict) -> "MeanBaseline":
        m = _finite_mask(feat)
        y = feat["y"][m]
        self.global_mean_ = float(y.mean()) if len(y) else 0.5

        pair_key, ik_A, ik_B, cell_id = (
            feat["pair_key"][m], feat["ik_A"][m], feat["ik_B"][m], feat["cell_id"][m]
        )

        self.pair_mean_ = _group_mean(pair_key, y)
        drug_id = np.concatenate([ik_A, ik_B])
        drug_y = np.concatenate([y, y])
        self.drug_mean_ = _group_mean(drug_id, drug_y)
        self.cell_mean_ = _group_mean(cell_id, y)
        return self

    def predict_proba(self, feat: dict) -> np.ndarray:
        n = len(feat["y"])
        out = np.full(n, self.global_mean_, dtype=np.float64)

        if self.variant == "global_mean":
            return out

        if self.variant == "cell_mean":
            for i, c in enumerate(feat["cell_id"]):
                out[i] = self.cell_mean_.get(c, self.global_mean_)
            return out

        if self.variant == "drug_mean":
            for i, (a, b) in enumerate(zip(feat["ik_A"], feat["ik_B"])):
                da = self.drug_mean_.get(a)
                db = self.drug_mean_.get(b)
                vals = [v for v in (da, db) if v is not None]
                out[i] = float(np.mean(vals)) if vals else self.global_mean_
            return out

        # pair_drug_cell_global fallback chain
        for i, (pk, a, b, c) in enumerate(
            zip(feat["pair_key"], feat["ik_A"], feat["ik_B"], feat["cell_id"])
        ):
            if pk in self.pair_mean_:
                out[i] = self.pair_mean_[pk]
                continue
            da, db = self.drug_mean_.get(a), self.drug_mean_.get(b)
            vals = [v for v in (da, db) if v is not None]
            if vals:
                out[i] = float(np.mean(vals))
                continue
            if c in self.cell_mean_:
                out[i] = self.cell_mean_[c]
                continue
            out[i] = self.global_mean_
        return out


def _group_mean(keys: np.ndarray, y: np.ndarray) -> dict:
    d: dict = {}
    for k in np.unique(keys):
        d[k] = float(y[keys == k].mean())
    return d


class OneHotBaseline:
    """Logistic regression on one-hot(drug1) + one-hot(drug2) + one-hot(cell).

    Pure entity-identity memorisation — no pA/pB/c structure at all. Unseen
    entities at test time map to the all-zero row for that block (handled via
    OneHotEncoder(handle_unknown="ignore")).
    """

    def __init__(self, C: float = 1.0, max_iter: int = 200):
        self.C = C
        self.max_iter = max_iter

    def fit(self, feat: dict) -> "OneHotBaseline":
        m = _finite_mask(feat)
        self.enc_ = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
        X = self._design(feat, m, fit=True)
        y = feat["y"][m]
        self.clf_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter, class_weight="balanced"
        )
        self.clf_.fit(X, y)
        return self

    def _design(self, feat: dict, mask: np.ndarray, fit: bool) -> csr_matrix:
        cols = np.stack(
            [feat["ik_A"][mask], feat["ik_B"][mask], feat["cell_id"][mask]], axis=1
        )
        if fit:
            return self.enc_.fit_transform(cols)
        return self.enc_.transform(cols)

    def predict_proba(self, feat: dict) -> np.ndarray:
        mask = np.ones(len(feat["y"]), dtype=bool)
        X = self._design(feat, mask, fit=False)
        return self.clf_.predict_proba(X)[:, 1]


class RFBaseline:
    def __init__(self, n_estimators: int = 300, max_depth: int | None = None,
                 n_jobs: int = -1, random_state: int = 0):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.n_jobs, self.random_state = n_jobs, random_state

    def fit(self, feat: dict) -> "RFBaseline":
        from unimos.eval.features import kernel_equivalent_matrix

        m = _finite_mask(feat)
        X = kernel_equivalent_matrix(feat)[m]
        y = feat["y"][m]
        self.clf_ = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            n_jobs=self.n_jobs, random_state=self.random_state,
            class_weight="balanced",
        )
        self.clf_.fit(X, y)
        return self

    def predict_proba(self, feat: dict) -> np.ndarray:
        from unimos.eval.features import kernel_equivalent_matrix

        X = kernel_equivalent_matrix(feat)
        return self.clf_.predict_proba(X)[:, 1]


class XGBBaseline:
    def __init__(self, n_estimators: int = 300, max_depth: int = 6,
                 learning_rate: float = 0.1, n_jobs: int = -1, random_state: int = 0):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.learning_rate = learning_rate
        self.n_jobs, self.random_state = n_jobs, random_state

    def fit(self, feat: dict) -> "XGBBaseline":
        from xgboost import XGBClassifier
        from unimos.eval.features import kernel_equivalent_matrix

        m = _finite_mask(feat)
        X = kernel_equivalent_matrix(feat)[m]
        y = feat["y"][m]
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        spw = (n_neg / n_pos) if n_pos > 0 else 1.0
        self.clf_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, n_jobs=self.n_jobs,
            random_state=self.random_state, scale_pos_weight=spw,
            eval_metric="logloss", tree_method="hist",
        )
        self.clf_.fit(X, y)
        return self

    def predict_proba(self, feat: dict) -> np.ndarray:
        from unimos.eval.features import kernel_equivalent_matrix

        X = kernel_equivalent_matrix(feat)
        return self.clf_.predict_proba(X)[:, 1]


BASELINE_REGISTRY = {
    "global_mean": lambda: MeanBaseline(variant="global_mean"),
    "drug_mean": lambda: MeanBaseline(variant="drug_mean"),
    "cell_mean": lambda: MeanBaseline(variant="cell_mean"),
    "pair_drug_cell_global_mean": lambda: MeanBaseline(variant="pair_drug_cell_global"),
    "one_hot": OneHotBaseline,
    "random_forest": RFBaseline,
    "xgboost": XGBBaseline,
}
