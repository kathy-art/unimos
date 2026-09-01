"""
dataset.py — UniMoS PyTorch Dataset.

Loads all pre-computed features (cell ssGSEA, mutation burden, HVG expression,
drug function vectors, Morgan FPs) and pairs them with binary synergy label and
RI labels from the main pair CSV.

HVG selection is fold-aware: if `hvg_train_expr` is provided (an expression
DataFrame restricted to training cells), top-150 HVGs are selected from that
to avoid leakage into LCO test cells.

Row filtering: only rows where `cell_feature_id` is present in the processed
cell index are retained.  This drops rows where cell_feature_mask == 0 or
the cell has no processed omics features, preventing zero-vector pollution.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_bio_embeddings(emb_path: str, index_path: str) -> tuple[np.ndarray, dict]:
    """L2-normalised (Signaturizer B1) embeddings + inchikey->row index.
    Pass the returned tuple as ``bio_emb`` to UniMoSDataset."""
    emb = np.load(emb_path)
    idx: dict[str, int] = json.load(open(index_path))
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return norm.astype(np.float32), idx

# ── Default file locations ────────────────────────────────────────────────────

_REPO = Path(__file__).resolve().parent.parent.parent
_DATA = _REPO / "data"
_PROC = _DATA / "processed"
_FUNC = _DATA / "Drugcombv15" / "function_nodes"

# Continuous DrugComb synergy scores used for optional multi-task regression.
SYNERGY_METRIC_COLS: tuple[str, ...] = ("loewe", "zip", "hsa", "bliss")


class UniMoSDataset(Dataset):
    """
    Dataset for one split (train, val, or test).

    Each item is a dict of tensors matching UniMoS.REQUIRED_KEYS:
      pA, pB          : (67,)   drug function vectors
      fp_A, fp_B      : (2048,) Morgan fingerprints (float32 0/1)
      c_fn            : (67,)   cell ssGSEA-derived function-node activity  [z-scored]
      c_mut_fn        : (67,)   cell mutation burden projected to fn space  [z-scored]
      c_raw           : (150,)  cell top-HVG raw log1p expression
      cell_row_idx    : ()      long, row index into cell_feature_index.json order —
                                 used by VirtualCellModule to look up cell_fm_emb.npy
                                 (unimos/data/precompute_fm_emb.py); same row order as
                                 c_fn/c_mut_fn/c_raw
      y_class         : ()      Loewe > 10 binary label (float32 0/1, may be NaN)
      y_ri_A, y_ri_B  : ()      single-drug RI (float32, may be NaN)
      y_syn_metrics   : (K,)    continuous synergy scores (loewe/zip/hsa/bliss);
                                 always present when columns exist, else NaNs
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split_indices: np.ndarray,
        *,
        proc_dir: Path = _PROC,
        func_dir: Path = _FUNC,
        hvg_gene_ids: Optional[list[str]] = None,
        cell_norm_stats: Optional[dict] = None,
        use_pair_prior: bool = False,
        pair_prior_stats: Optional[tuple] = None,
        use_bio_prior: bool = False,
        bio_emb: Optional[tuple] = None,
        bio_prior_stats: Optional[tuple] = None,
        bio_knn_k: int = 5,
        synergy_metric_cols: tuple[str, ...] = SYNERGY_METRIC_COLS,
    ) -> None:
        """
        Parameters
        ----------
        df              : full pair DataFrame (all rows, all splits)
        split_indices   : integer indices into df for this split
        proc_dir        : directory containing processed .npy / .json files
        func_dir        : directory containing drug_function_vectors.npy
        hvg_gene_ids    : if provided, use these gene symbols as the HVG subset
                          (pass training-derived HVGs to avoid leakage in val/test)
        cell_norm_stats : if provided, use these z-score stats for c_fn / c_mut_fn
                          instead of computing from this split's cells.
                          Pass ``train_ds.norm_stats()`` to val/test constructors.
                          Keys: "c_fn_mean", "c_fn_std", "c_mut_mean", "c_mut_std".
        use_pair_prior  : if True, __getitem__ returns a "pair_prior" scalar —
                          the (drug1,drug2)-pair's historical positive rate,
                          unordered (frozenset key), matching the
                          pair_drug_cell_global_mean baseline's pair_mean_
                          (see unimos/eval/baselines.py). Exploits the fact
                          that drug-pair synergy tendency is largely
                          cell-independent, which is why that baseline beats
                          the model on some LCO seeds — this exposes the same
                          signal to the model itself.
        pair_prior_stats: (sums, counts, global_mean) computed from TRAIN rows
                          — pass ``train_ds.pair_prior_stats()`` to val/test
                          constructors. If None (the train dataset itself),
                          stats are computed from this dataset's own rows and
                          each row's __getitem__ leaves its own label out of
                          its pair's aggregate (leave-one-out) to avoid
                          trivially leaking the label through the prior.
        use_bio_prior   : if True, __getitem__ also returns "bio_prior" — a
                          per-drug historical positive-rate prior, generalised
                          via k-NN in Signaturizer B1 (target/MoA) embedding
                          space. Unlike pair_prior, this works even for drugs
                          with ZERO train-set history (e.g. every LDO test
                          drug): a drug seen in train uses its own (LOO)
                          per-drug rate; an unseen drug falls back to the
                          similarity-weighted average rate of its k nearest
                          train-drug neighbours in B1 space. Row value =
                          mean(prior(ik_A), prior(ik_B)).
        bio_emb         : (norm_embeddings, ik->row_index) from
                          load_bio_embeddings(). Required if use_bio_prior.
        bio_prior_stats : (drug_sum, drug_count, global_mean, knn_cache) —
                          pass ``train_ds.bio_prior_stats()`` to val/test
                          constructors. If None (the train dataset itself),
                          computed from this dataset's own rows + bio_emb.
        bio_knn_k       : number of nearest train-drug neighbours to average
                          over for out-of-train drugs.
        synergy_metric_cols : continuous DrugComb score columns to expose as
                          y_syn_metrics (default: loewe/zip/hsa/bliss).
        """
        self.synergy_metric_cols = tuple(synergy_metric_cols)
        self._syn_clip_lo: list[float] | None = None
        self._syn_clip_hi: list[float] | None = None

        # ── Load cell feature index first so we can filter rows ───────────────
        cell_idx: dict[str, int] = json.load(open(proc_dir / "cell_feature_index.json"))
        self.cell_to_row = cell_idx

        # ── Filter split to rows with resolvable cell features ────────────────
        raw_rows = df.iloc[split_indices].reset_index(drop=True)
        self.rows = raw_rows[
            raw_rows["cell_feature_id"].isin(cell_idx)
        ].reset_index(drop=True)

        # ── Optional pair-history prior (leakage-safe, see docstring) ─────────
        self.use_pair_prior = use_pair_prior
        if use_pair_prior:
            if pair_prior_stats is not None:
                self._pp_sum, self._pp_count, self._pp_global = pair_prior_stats
                self._pp_loo = False
            else:
                self._pp_sum, self._pp_count, self._pp_global = self._compute_pair_prior_stats()
                self._pp_loo = True
        else:
            self._pp_sum, self._pp_count, self._pp_global, self._pp_loo = {}, {}, 0.5, False

        # ── Optional bio (Signaturizer B1) k-NN prior ──────────────────────────
        self.use_bio_prior = use_bio_prior
        self._bio_norm, self._bio_idx = bio_emb if bio_emb is not None else (None, None)
        if use_bio_prior:
            if bio_prior_stats is not None:
                self._bp_sum, self._bp_count, self._bp_global, self._bp_knn = bio_prior_stats
                self._bp_loo = False
            else:
                self._bp_sum, self._bp_count, self._bp_global, self._bp_knn = \
                    self._compute_bio_prior_stats(bio_knn_k)
                self._bp_loo = True
        else:
            self._bp_sum, self._bp_count, self._bp_global, self._bp_knn, self._bp_loo = {}, {}, 0.5, {}, False

        # ── Cell features ─────────────────────────────────────────────────────
        self.c_fn     = np.load(proc_dir / "cell_fn_vectors.npy")          # (168, 67)
        self.c_mut_fn = np.load(proc_dir / "cell_mut_fn_vectors.npy")      # (168, 67)
        c_raw_full    = np.load(proc_dir / "cell_raw_expr.npy")             # (168, 150)

        stored_hvg: list[str] = json.load(open(proc_dir / "cell_hvg_gene_ids.json"))

        # Fold-aware HVG selection: restrict to training-derived genes
        if hvg_gene_ids is not None and hvg_gene_ids != stored_hvg:
            hvg_positions = [stored_hvg.index(g) for g in hvg_gene_ids if g in stored_hvg]
            c_raw_full = c_raw_full[:, hvg_positions]

        self.c_raw = c_raw_full                                             # (168, n_hvg)
        self.n_hvg = self.c_raw.shape[1]

        # ── Drug function vectors ──────────────────────────────────────────────
        drug_fn = np.load(func_dir / "drug_function_vectors.npy")           # (3814, 67)
        drug_fn_idx_df = pd.read_parquet(func_dir / "drug_function_vector_index.parquet")
        self.drug_fn = drug_fn
        self.drug_fn_idx: dict[str, int] = dict(zip(
            drug_fn_idx_df["inchikey"], range(len(drug_fn_idx_df))
        ))

        # ── Morgan fingerprints ────────────────────────────────────────────────
        self.morgan_fps = np.load(proc_dir / "drug_morgan_fps.npy")         # (3814, 2048)
        self.morgan_idx: dict[str, int] = json.load(
            open(proc_dir / "drug_morgan_index.json")
        )

        # ── c_fn / c_mut_fn z-score stats (on this split's cells) ─────────────
        # After filtering, all cell_feature_id values are in cell_idx.
        split_cell_rows = sorted({cell_idx[cid] for cid in self.rows["cell_feature_id"]})

        if cell_norm_stats is not None:
            self._c_fn_mean  = cell_norm_stats["c_fn_mean"]
            self._c_fn_std   = cell_norm_stats["c_fn_std"]
            self._c_mut_mean = cell_norm_stats["c_mut_mean"]
            self._c_mut_std  = cell_norm_stats["c_mut_std"]
        else:
            c_fn_sub  = self.c_fn[split_cell_rows]  if split_cell_rows else self.c_fn
            c_mut_sub = self.c_mut_fn[split_cell_rows] if split_cell_rows else self.c_mut_fn
            self._c_fn_mean  = c_fn_sub.mean(axis=0)                        # (67,)
            self._c_fn_std   = c_fn_sub.std(axis=0) + 1e-6                  # (67,)
            self._c_mut_mean = c_mut_sub.mean(axis=0)                       # (67,)
            self._c_mut_std  = c_mut_sub.std(axis=0) + 1e-6                 # (67,)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def norm_stats(self) -> dict:
        """
        Return c_fn / c_mut_fn normalization stats.

        Pass the result to the val/test UniMoSDataset constructors as
        ``cell_norm_stats`` so they use training-cell statistics.
        """
        return {
            "c_fn_mean":  self._c_fn_mean,
            "c_fn_std":   self._c_fn_std,
            "c_mut_mean": self._c_mut_mean,
            "c_mut_std":  self._c_mut_std,
        }

    def _compute_pair_prior_stats(self) -> tuple[dict, dict, float]:
        """Per-unordered-pair (sum, count) of label_loewe_gt_10 over this
        dataset's own rows, plus the global mean (NaN labels excluded)."""
        sums: dict = {}
        counts: dict = {}
        labels = self.rows["label_loewe_gt_10"]
        for a, b, y in zip(self.rows["drug1_inchikey"], self.rows["drug2_inchikey"], labels):
            if pd.isna(y):
                continue
            key = frozenset((a, b))
            sums[key] = sums.get(key, 0.0) + float(y)
            counts[key] = counts.get(key, 0) + 1
        valid = labels.dropna()
        global_mean = float(valid.mean()) if len(valid) else 0.5
        return sums, counts, global_mean

    def pair_prior_stats(self) -> tuple[dict, dict, float]:
        """(sums, counts, global_mean) — pass to val/test UniMoSDataset(pair_prior_stats=...)."""
        return self._pp_sum, self._pp_count, self._pp_global

    def _compute_bio_prior_stats(self, k: int) -> tuple[dict, dict, float, dict]:
        """Per-drug (sum, count) of label_loewe_gt_10 over this dataset's own
        rows (both drug1 and drug2 roles), global mean, and a k-NN fallback
        cache (in B1 space) for every bio-embedded drug NOT covered by these
        train rows — so out-of-train (LDO test) drugs still get a value."""
        sums: dict = {}
        counts: dict = {}
        labels = self.rows["label_loewe_gt_10"]
        for a, b, y in zip(self.rows["drug1_inchikey"], self.rows["drug2_inchikey"], labels):
            if pd.isna(y):
                continue
            for d in (a, b):
                sums[d] = sums.get(d, 0.0) + float(y)
                counts[d] = counts.get(d, 0) + 1
        valid = labels.dropna()
        global_mean = float(valid.mean()) if len(valid) else 0.5

        knn_cache: dict = {}
        if self._bio_norm is not None:
            covered = [d for d in counts if d in self._bio_idx]
            if covered:
                cov_rows = np.array([self._bio_idx[d] for d in covered])
                cov_emb = self._bio_norm[cov_rows]                       # (M, 128)
                cov_vals = np.array([sums[d] / counts[d] for d in covered])
                kk = min(k, len(covered))
                for ik, row in self._bio_idx.items():
                    if ik in counts:
                        continue  # has its own history — handled at read time
                    sims = cov_emb @ self._bio_norm[row]
                    top = np.argpartition(-sims, kk - 1)[:kk]
                    w = np.clip(sims[top], 0, None) + 1e-6
                    knn_cache[ik] = float(np.average(cov_vals[top], weights=w))
        return sums, counts, global_mean, knn_cache

    def bio_prior_stats(self) -> tuple[dict, dict, float, dict]:
        """(sums, counts, global_mean, knn_cache) — pass to val/test UniMoSDataset(bio_prior_stats=...)."""
        return self._bp_sum, self._bp_count, self._bp_global, self._bp_knn

    def _bio_prior_for(self, ik: str, own_label) -> float:
        s = self._bp_sum.get(ik)
        n = self._bp_count.get(ik, 0)
        if s is not None and n > 0:
            if self._bp_loo and pd.notna(own_label):
                s, n = s - float(own_label), n - 1
            if n > 0:
                return s / n
        return self._bp_knn.get(ik, self._bp_global)

    def compute_pos_weight(self) -> float:
        """
        Return n_neg / n_pos from this split's y_class labels.

        Call on the training dataset before constructing UniMoSLoss so that
        pos_weight reflects the actual class balance in the training set.
        NaN labels are excluded from the calculation.
        Returns 1.0 if no positive labels are present.
        """
        col = self.rows["label_loewe_gt_10"].dropna()
        n_pos = int((col == 1).sum())
        n_neg = int((col == 0).sum())
        return float(n_neg / n_pos) if n_pos > 0 else 1.0

    def compute_syn_metric_stats(
        self,
        clip_pct: tuple[float, float] = (1.0, 99.0),
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """Train-split mean/std + winsorize bounds for synergy metrics.

        Returns ``(means, stds, clip_lo, clip_hi)``.  Stats are computed on
        values clipped to ``clip_pct`` percentiles so ZIP/Bliss outliers do
        not inflate σ.  Callers should also apply the same bounds via
        ``set_syn_clip_bounds`` so train/val/test targets are winsorized.
        """
        means: list[float] = []
        stds: list[float] = []
        clip_lo: list[float] = []
        clip_hi: list[float] = []
        for col in self.synergy_metric_cols:
            if col not in self.rows.columns:
                means.append(0.0)
                stds.append(1.0)
                clip_lo.append(float("-inf"))
                clip_hi.append(float("inf"))
                continue
            v = pd.to_numeric(self.rows[col], errors="coerce").to_numpy(dtype=np.float64)
            finite = v[np.isfinite(v)]
            if finite.size == 0:
                means.append(0.0)
                stds.append(1.0)
                clip_lo.append(float("-inf"))
                clip_hi.append(float("inf"))
                continue
            lo, hi = np.percentile(finite, list(clip_pct))
            clipped = np.clip(finite, lo, hi)
            means.append(float(clipped.mean()))
            stds.append(float(max(clipped.std(), 1e-3)))
            clip_lo.append(float(lo))
            clip_hi.append(float(hi))
        return means, stds, clip_lo, clip_hi

    def set_syn_clip_bounds(
        self,
        clip_lo: list[float],
        clip_hi: list[float],
    ) -> None:
        """Winsorize y_syn_metrics in __getitem__ to train-derived bounds."""
        if len(clip_lo) != len(self.synergy_metric_cols) or len(clip_hi) != len(self.synergy_metric_cols):
            raise ValueError("clip_lo/hi length must match synergy_metric_cols")
        self._syn_clip_lo = [float(x) for x in clip_lo]
        self._syn_clip_hi = [float(x) for x in clip_hi]

    def _drug_fn_vec(self, inchikey: str) -> np.ndarray:
        idx = self.drug_fn_idx.get(inchikey)
        if idx is None:
            return np.zeros(self.drug_fn.shape[1], dtype=np.float32)
        return self.drug_fn[idx].astype(np.float32)

    def _morgan_fp(self, inchikey: str) -> np.ndarray:
        idx = self.morgan_idx.get(inchikey)
        if idx is None:
            return np.zeros(self.morgan_fps.shape[1], dtype=np.float32)
        return self.morgan_fps[idx].astype(np.float32)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        row = self.rows.iloc[i]

        ik_A    = row["drug1_inchikey"]
        ik_B    = row["drug2_inchikey"]
        cell_id = row["cell_feature_id"]

        pA   = self._drug_fn_vec(ik_A)
        pB   = self._drug_fn_vec(ik_B)
        fp_A = self._morgan_fp(ik_A)
        fp_B = self._morgan_fp(ik_B)

        cell_row = self.cell_to_row.get(cell_id)
        if cell_row is not None:
            c_fn     = (self.c_fn[cell_row]     - self._c_fn_mean)  / self._c_fn_std
            c_mut_fn = (self.c_mut_fn[cell_row] - self._c_mut_mean) / self._c_mut_std
            c_raw    = self.c_raw[cell_row].copy()
            cell_row_idx = cell_row
        else:
            # Dead branch after __init__ filtering — safety guard only
            c_fn     = np.zeros(self.c_fn.shape[1],     dtype=np.float32)
            c_mut_fn = np.zeros(self.c_mut_fn.shape[1], dtype=np.float32)
            c_raw    = np.zeros(self.n_hvg,              dtype=np.float32)
            cell_row_idx = 0

        loewe_gt10 = row.get("label_loewe_gt_10")
        ri_a       = row.get("ri_drug1")
        ri_b       = row.get("ri_drug2")

        syn_vals = []
        for i, col in enumerate(self.synergy_metric_cols):
            if col in self.rows.columns:
                v = row.get(col)
                val = float(v) if pd.notna(v) else float("nan")
            else:
                val = float("nan")
            if (
                math.isfinite(val)
                and self._syn_clip_lo is not None
                and self._syn_clip_hi is not None
            ):
                val = float(min(max(val, self._syn_clip_lo[i]), self._syn_clip_hi[i]))
            syn_vals.append(val)

        pair_prior = self._pp_global
        if self.use_pair_prior:
            key = frozenset((ik_A, ik_B))
            s = self._pp_sum.get(key)
            n = self._pp_count.get(key, 0)
            if s is not None and n > 0:
                if self._pp_loo and pd.notna(loewe_gt10):
                    s, n = s - float(loewe_gt10), n - 1
                pair_prior = (s / n) if n > 0 else self._pp_global

        bio_prior = 0.5
        if self.use_bio_prior:
            bio_prior = 0.5 * (
                self._bio_prior_for(ik_A, loewe_gt10) + self._bio_prior_for(ik_B, loewe_gt10)
            )

        return {
            "pA":       torch.from_numpy(pA),
            "pB":       torch.from_numpy(pB),
            "fp_A":     torch.from_numpy(fp_A),
            "fp_B":     torch.from_numpy(fp_B),
            "c_fn":     torch.from_numpy(c_fn.astype(np.float32)),
            "c_mut_fn": torch.from_numpy(c_mut_fn.astype(np.float32)),
            "c_raw":    torch.from_numpy(c_raw.astype(np.float32)),
            "cell_row_idx": torch.tensor(cell_row_idx, dtype=torch.long),
            "y_class":  torch.tensor(
                float(loewe_gt10) if pd.notna(loewe_gt10) else float("nan"),
                dtype=torch.float32,
            ),
            "y_ri_A":   torch.tensor(
                float(ri_a) if pd.notna(ri_a) else float("nan"),
                dtype=torch.float32,
            ),
            "y_ri_B":   torch.tensor(
                float(ri_b) if pd.notna(ri_b) else float("nan"),
                dtype=torch.float32,
            ),
            "y_syn_metrics": torch.tensor(syn_vals, dtype=torch.float32),
            "ik_A":     ik_A,
            "ik_B":     ik_B,
            "cell_id":  cell_id,
            "pair_prior": torch.tensor(float(pair_prior), dtype=torch.float32),
            "bio_prior": torch.tensor(float(bio_prior), dtype=torch.float32),
        }


def select_hvg(
    expr_df: pd.DataFrame,
    train_cell_ids: list[str],
    n_top: int = 150,
) -> list[str]:
    """
    Select top-n_top high-variance genes using only training cells.
    expr_df must be (genes × cells) with gene symbols as index.

    Returns list of gene symbol strings.
    """
    train_cols = [c for c in train_cell_ids if c in expr_df.columns]
    var = expr_df[train_cols].var(axis=1)
    return var.nlargest(n_top).index.tolist()
