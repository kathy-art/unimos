"""
train.py — UniMoS single-run training entry (SPEC-09).

Usage
-----
python train.py --split ldo --seed 0 [--config configs/ldo.yaml]
                [--output-dir checkpoints/ldo/seed_0] [--max-epochs 2] [--fast-dev-run]

Output
------
checkpoints/{split}/seed_{seed}/
    best.ckpt          — Lightning checkpoint at best val_auroc
    metrics.json       — val + test metrics (SPEC-07 keys) + best_threshold
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import EarlyStopping as LightningEarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


class _SmoothedMonitor(Callback):
    """Publish a trailing-K-epoch mean of `src` under the name `dst`.

    This runs on ``on_train_epoch_end``, where Lightning calls non-monitoring
    callbacks before ModelCheckpoint and EarlyStopping (see
    ``_FitLoop.on_advance_end``).  By that point the epoch's validation metrics
    are in ``callback_metrics``, and the value written here is still there when
    the two monitoring callbacks read it.  Neither ``on_validation_epoch_end``
    nor ``pl_module.log`` works for this: the raw metric is not published that
    early, and the dict is rebuilt afterwards.  The raw per-epoch history is
    kept for the run record.
    """

    def __init__(self, src: str, dst: str, k: int) -> None:
        self.src, self.dst, self.k = src, dst, k
        self.raw: list[float] = []
        self.smooth: list[float] = []

    @property
    def current(self) -> float | None:
        """Latest value of whichever series is being monitored."""
        series = self.smooth if self.k > 1 else self.raw
        return series[-1] if series else None

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get(self.src)
        if v is None:
            return
        self.raw.append(float(v))
        window = self.raw[-self.k:]
        mean = sum(window) / len(window)
        self.smooth.append(mean)
        trainer.callback_metrics[self.dst] = torch.tensor(mean)


class _TopKWeightPool(Callback):
    """Keep CPU copies of the `n` epochs scoring highest on `monitor`.

    Selection uses only the validation monitor, so averaging these weights is
    a val-side decision like any other early-stopping rule.
    """

    def __init__(self, source: "_SmoothedMonitor", n: int) -> None:
        self.source, self.n = source, n
        self.entries: list[tuple[float, int, dict]] = []

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        # Read the score off the monitor callback, which sits earlier in the
        # callback list and so has already recorded this epoch's value.
        v = self.source.current
        if v is None:
            return
        state = {k: t.detach().cpu().clone() for k, t in pl_module.state_dict().items()}
        self.entries.append((float(v), int(trainer.current_epoch), state))
        self.entries.sort(key=lambda e: e[0], reverse=True)
        del self.entries[self.n:]

    def averaged_state(self) -> tuple[dict, list[int], list[float]]:
        """Parameter-wise mean of the pooled weights (float tensors only)."""
        states = [e[2] for e in self.entries]
        epochs = [e[1] for e in self.entries]
        scores = [e[0] for e in self.entries]
        out = {}
        for key in states[0]:
            ref = states[0][key]
            if ref.is_floating_point():
                acc = torch.zeros_like(ref, dtype=torch.float64)
                for s in states:
                    acc += s[key].to(torch.float64)
                out[key] = (acc / len(states)).to(ref.dtype)
            else:
                # Integer buffers (e.g. BatchNorm num_batches_tracked) are not
                # meaningfully averageable; they get reset by the BN pass below.
                out[key] = ref.clone()
        return out, epochs, scores


def _reestimate_bn(model, loader, device, max_batches: int) -> int:
    """Recompute BatchNorm running statistics for an averaged weight vector.

    Averaged parameters do not match the running statistics inherited from any
    single epoch, so they are re-accumulated over training batches.  Only the
    BatchNorm layers are put in train mode; dropout stays off so the statistics
    reflect inference-time activations.
    """
    bns = [m for m in model.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    model.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average over the pass
        m.train()
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= max_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            model(batch)
            seen += 1
    model.eval()
    return seen


class _EpochPrinter(Callback):
    """Prints one plain line per epoch so progress is visible even when the
    terminal doesn't support tqdm's carriage-return progress bar."""

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        auroc = trainer.callback_metrics.get("val_auroc")
        auroc_str = f"{float(auroc):.4f}" if auroc is not None else "n/a"
        # Prefer the configured early-stop monitor (e.g. val_zip_pearson).
        monitor = getattr(pl_module, "early_stop_metric", None) or "val_auroc"
        mon = trainer.callback_metrics.get(monitor)
        syn = trainer.callback_metrics.get("val_syn_pearson")
        zip_r = trainer.callback_metrics.get("val_zip_pearson")
        parts = [f"[epoch {trainer.current_epoch:>3}] val_auroc={auroc_str}"]
        if zip_r is not None:
            parts.append(f"val_zip_pearson={float(zip_r):.4f}")
        if syn is not None and monitor != "val_syn_pearson":
            parts.append(f"val_syn_pearson={float(syn):.4f}")
        elif syn is not None and zip_r is None:
            parts.append(f"val_syn_pearson={float(syn):.4f}")
        if mon is not None and monitor not in ("val_auroc", "val_zip_pearson", "val_syn_pearson"):
            parts.append(f"{monitor}={float(mon):.4f}")
        print(" ".join(parts), flush=True)

from unimos.data.dataset import UniMoSDataset, select_hvg, load_bio_embeddings
from unimos.data.splits import build_splits, build_cv_splits
from unimos.model.unimos import UniMoS
from unimos.training.metrics import compute_metrics, find_best_threshold

# Continuous DrugComb synergy scores for optional multi-task regression.
_SYN_METRIC_COLS = ("loewe", "zip", "hsa", "bliss")

# ── Default paths ─────────────────────────────────────────────────────────────

_REPO       = Path(__file__).resolve().parent
# train_data/ is the single, self-contained source for everything training reads:
# features, function-node vectors, the pair table, and the materialised splits.
# Nothing here reaches outside the repository.
_TRAIN_DATA = _REPO / "train_data"
_PROC     = _TRAIN_DATA / "processed"
_FUNC     = _TRAIN_DATA / "function_nodes"
# The pair table that the splits were built from (251,541 rows). Using any other
# table with an index-based split silently selects the wrong rows -- see
# scripts/materialize_splits.py.
_PAIR_CSV = _TRAIN_DATA / "pairs" / "unimos_stratified_dataset.parquet"

# Config path fields that may still carry legacy data/ or data_vc_frozen/ prefixes.
_PATH_FIELDS = (
    "fm_emb_path", "fm_kernel_emb_path", "graph_cache", "w_prior_path",
    "bio_emb_path_prior", "bio_index_path_prior", "desc_cache",
    "sens_drug_cache", "sens_cell_cache", "w_prior_percell_path",
    "gi_prior_path", "film_rescue_emb_path",
)
_TRAIN_DATA_SUBDIRS = ("processed", "processed/signaturizer", "function_nodes", "pairs")


def _redirect_to_train_data(value: str) -> str:
    """Resolve a config path against train_data/ by file name.

    Archived configs under checkpoints_*/ still say `data/processed/...`, and
    rewriting those files would destroy the provenance of past runs. Instead the
    same file is looked up inside train_data/; if it is there, that copy wins.
    Paths whose file name is absent from train_data/ are left untouched.
    """
    if not value:
        return value
    name = Path(value).name
    for sub in _TRAIN_DATA_SUBDIRS:
        candidate = _TRAIN_DATA / sub / name
        if candidate.is_file():
            return str(candidate)
    return value


# ── Hyperparameter container ──────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Optimiser
    lr: float = 5e-4
    # Model
    dropout: float = 0.2
    rank_r: int = 16
    struct_hidden: int = 256
    cell_hidden: int = 128
    # Loss weights
    lambda_ri: float = 0.5
    lambda_res: float = 0.01
    lambda_W: float = 1e-3
    lambda_gamma: float = 1e-3
    lambda_gate: float = 1e-3
    # Training loop
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    # Model selection (see docs/MODEL_SELECTION.md).  Both default to the
    # historical behaviour: no smoothing, single best checkpoint.
    # es_smooth K > 1 monitors a trailing K-epoch mean instead of the raw
    # per-epoch value, so one lucky epoch can neither be selected nor end the
    # run.  ckpt_avg N > 0 replaces the single best checkpoint with the
    # parameter-wise mean of the N epochs scoring highest on that monitor.
    es_smooth: int = 1
    ckpt_avg: int = 0
    ckpt_avg_bn_batches: int = 100
    # Optional biological vertical-synergy prior on the pathway kernel W
    w_prior_path: str = ""
    w_prior_scale: float = 0.0
    # Objective function (SPEC-06 ext): bce | focal, + class-imbalance knobs
    loss_type: str = "bce"
    focal_gamma: float = 2.0
    pos_weight_scale: float = 1.0

    struct_encoder: str = "fp"
    graph_cache: str = ""
    gnn_hidden: int = 256
    gnn_layers: int = 3
    use_descriptors: bool = False
    desc_cache: str = ""
    desc_dim: int = 217
    use_sensitivity_profile: bool = False
    sens_drug_cache: str = ""
    sens_cell_cache: str = ""
    sens_dim: int = 6

    # VirtualCellModule (design doc §2.1/§4, entrance one; Phase 1)
    use_virtual_cell: bool = False
    fm_emb_path: str = ""
    fm_out_dim: int = 128
    z_cell_dim: "int | None" = None  # None -> defaults to cell_hidden (see UniMoS)

    # Gate-2 z_fm structural rescue (PHASE1_GATE.md §9): FiLM(z_fm) on pA/pB
    use_film_rescue: bool = False
    film_rescue_emb_path: str = ""
    film_hidden: int = 64

    # Phase 2 (design doc §2.2): per-cell FiLM-modulated W_prior(cell)
    cell_specific_w_prior: bool = False
    w_prior_percell_path: str = ""
    tau_prior: "float | None" = None

    # Phase 3 (design doc §2.3): GI prior additive init for W_base
    gi_prior_path: str = ""
    gi_prior_init_scale: float = 0.0

    # NBE selling-point ablations (train-time; see configs/ablation_nbe_ldo_*.yaml)
    # none|full|core_only|resid_only|no_deltaw|zero_p|
    # core_only_no_deltaw|resid_only_no_deltaw (track ablation vs VC+ΔW− full)
    ablation_mode: str = "none"

    # PathwayAggregationKernel cell conditioning: hypernet | fm_kernel_interp
    cell_cond_mode: str = "hypernet"
    fm_kernel_emb_path: str = ""   # empty → reuse fm_emb_path
    fm_kernel_tau: float = 0.1
    fm_kernel_self_mask: bool = True

    # Drug-pair historical positive-rate prior (leakage-safe; train-only,
    # leave-one-out for train rows). See UniMoSDataset use_pair_prior.
    use_pair_prior: bool = False

    # Per-drug k-NN prior in Signaturizer B1 (target/MoA) embedding space —
    # generalises to drugs with zero train history. See UniMoSDataset
    # use_bio_prior / load_bio_embeddings.
    use_bio_prior: bool = False
    bio_emb_path_prior: str = "data_vc_frozen/processed/signaturizer/b1_embeddings.npy"
    bio_index_path_prior: str = "data_vc_frozen/processed/signaturizer/b1_inchikey_index.json"
    bio_knn_k: int = 5

    # LR schedule: "cosine" (default, T_max=max_epochs) or "plateau"
    # (ReduceLROnPlateau on val_auroc — adapts per-run instead of assuming a
    # fixed max_epochs convergence horizon). See UniMoS.configure_optimizers.
    lr_scheduler_type: str = "cosine"
    lr_patience: int = 5
    lr_factor: float = 0.5

    # Multi-task regression on continuous DrugComb synergy scores.
    # Off by default.  Columns are configurable — e.g. drop loewe when the
    # binary head already uses label_loewe_gt_10.
    use_synergy_regression: bool = False
    lambda_syn_reg: float = 0.05
    # Classification loss weight.  Use <1 (e.g. 0.1) for ZIP-primary runs.
    lambda_cls: float = 1.0
    syn_metric_hidden: int = 128
    syn_metric_cols: tuple = ("loewe", "zip", "hsa", "bliss")
    # Winsorize continuous synergy targets to these train-split percentiles
    # before standardisation / loss / metrics.  (1, 99) matches the v2 recipe.
    syn_winsorize_pct: tuple = (1.0, 99.0)
    # If True: concentrate syn_reg loss on syn_primary_metrics (others get
    # syn_aux_weight).  Typical: primary=["zip"], aux=0.0, lambda_cls=0.1.
    syn_reg_primary: bool = False
    syn_primary_metrics: tuple = ("zip",)
    syn_aux_weight: float = 0.0
    # Explicit per-metric Huber weights aligned with syn_metric_cols.
    # When set, overrides syn_reg_primary.  None → equal (or primary scheme).
    syn_metric_weight_values: tuple | None = None
    # Lightning early-stop / checkpoint monitor.  Use "val_syn_pearson" for
    # equal-weight multi-metric, "val_zip_pearson" for ZIP-primary.
    early_stop_metric: str = "val_auroc"

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        # Allow YAML lists for tuple fields
        for key in (
            "syn_winsorize_pct",
            "syn_primary_metrics",
            "syn_metric_cols",
            "syn_metric_weight_values",
        ):
            if key in d and isinstance(d[key], list):
                d[key] = tuple(d[key])
        for key in _PATH_FIELDS:
            if isinstance(d.get(key), str):
                resolved = _redirect_to_train_data(d[key])
                if resolved != d[key]:
                    print(f"[train_data] {key}: {d[key]} -> {resolved}", flush=True)
                    d[key] = resolved
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def active_syn_cols(self) -> tuple[str, ...]:
        """Columns fed to the synergy regression head."""
        if not self.use_synergy_regression:
            return ()
        cols = tuple(str(c).lower() for c in self.syn_metric_cols)
        if not cols:
            raise ValueError("syn_metric_cols must be non-empty when use_synergy_regression")
        return cols

    def syn_metric_weights(self, cols: tuple[str, ...] | None = None) -> list[float] | None:
        """Per-metric loss weights for UniMoSLoss (None → equal weights)."""
        if not self.use_synergy_regression:
            return None
        use_cols = cols if cols is not None else self.active_syn_cols()
        if self.syn_metric_weight_values is not None:
            w = [float(x) for x in self.syn_metric_weight_values]
            if len(w) != len(use_cols):
                raise ValueError(
                    f"syn_metric_weight_values length {len(w)} != cols {len(use_cols)}"
                )
            if sum(w) <= 0:
                raise ValueError("syn_metric_weight_values sum to 0")
            return w
        if not self.syn_reg_primary:
            return None
        primary = {str(x).lower() for x in self.syn_primary_metrics}
        w = []
        for c in use_cols:
            w.append(1.0 if c.lower() in primary else float(self.syn_aux_weight))
        if sum(w) <= 0:
            raise ValueError(
                f"syn_reg_primary weights sum to 0; "
                f"primary={self.syn_primary_metrics} cols={use_cols}"
            )
        return w


# ── EarlyStopper (kept for API compatibility and unit tests) ──────────────────

class EarlyStopper:
    """Stop when monitored metric doesn't improve for `patience` checks."""

    def __init__(self, patience: int, mode: str = "max"):
        self.patience   = patience
        self.mode       = mode
        self._best      = -math.inf if mode == "max" else math.inf
        self._wait      = 0
        self.best_epoch = 0

    def improved(self, value: float) -> bool:
        if not math.isfinite(value):
            return False
        if (self.mode == "max" and value > self._best) or \
           (self.mode == "min" and value < self._best):
            self._best = value
            self._wait = 0
            return True
        self._wait += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self._wait >= self.patience

    @property
    def best_value(self) -> float:
        return self._best


# ── Data setup ────────────────────────────────────────────────────────────────

def build_loaders(
    df: pd.DataFrame,
    split_type: str,
    seed: int,
    cfg: TrainConfig,
    proc_dir: Path = _PROC,
    func_dir: Path = _FUNC,
    precomputed_splits: Optional[dict] = None,
) -> tuple[DataLoader, DataLoader, DataLoader, UniMoSDataset]:
    """
    Returns (train_loader, val_loader, test_loader, train_ds).
    train_ds is exposed so callers can read norm_stats / compute_pos_weight.
    precomputed_splits: if provided, skips build_splits() (used for CV folds).
    """
    splits = precomputed_splits if precomputed_splits is not None \
             else build_splits(df, split_type, seed=seed)

    bio_emb = (load_bio_embeddings(cfg.bio_emb_path_prior, cfg.bio_index_path_prior)
               if cfg.use_bio_prior else None)
    syn_cols = cfg.active_syn_cols() if cfg.use_synergy_regression else _SYN_METRIC_COLS
    train_ds = UniMoSDataset(df, splits["train"], proc_dir=proc_dir, func_dir=func_dir,
                              use_pair_prior=cfg.use_pair_prior,
                              use_bio_prior=cfg.use_bio_prior, bio_emb=bio_emb,
                              bio_knn_k=cfg.bio_knn_k,
                              synergy_metric_cols=syn_cols)
    norm_stats = train_ds.norm_stats()
    pair_prior_stats = train_ds.pair_prior_stats() if cfg.use_pair_prior else None
    bio_prior_stats = train_ds.bio_prior_stats() if cfg.use_bio_prior else None

    # LCO: compute fold-aware HVG from training cells only
    if split_type == "lco":
        c_raw_all    = np.load(proc_dir / "cell_raw_expr.npy")
        hvg_ids      = json.load(open(proc_dir / "cell_hvg_gene_ids.json"))
        cell_idx     = json.load(open(proc_dir / "cell_feature_index.json"))
        train_cells  = list({
            r for r in df.iloc[splits["train"]]["cell_feature_id"]
            if pd.notna(r)
        })
        known_cells  = [c for c in train_cells if c in cell_idx]
        rows         = [cell_idx[c] for c in known_cells]
        expr_df      = pd.DataFrame(
            c_raw_all[rows, :].T, index=hvg_ids, columns=known_cells
        )
        hvg_gene_ids = select_hvg(expr_df, known_cells, n_top=150)
    else:
        hvg_gene_ids = None

    val_ds  = UniMoSDataset(df, splits["val"],  proc_dir=proc_dir, func_dir=func_dir,
                             cell_norm_stats=norm_stats, hvg_gene_ids=hvg_gene_ids,
                             use_pair_prior=cfg.use_pair_prior, pair_prior_stats=pair_prior_stats,
                             use_bio_prior=cfg.use_bio_prior, bio_emb=bio_emb,
                             bio_prior_stats=bio_prior_stats, bio_knn_k=cfg.bio_knn_k,
                             synergy_metric_cols=syn_cols)
    test_ds = UniMoSDataset(df, splits["test"], proc_dir=proc_dir, func_dir=func_dir,
                             cell_norm_stats=norm_stats, hvg_gene_ids=hvg_gene_ids,
                             use_pair_prior=cfg.use_pair_prior, pair_prior_stats=pair_prior_stats,
                             use_bio_prior=cfg.use_bio_prior, bio_emb=bio_emb,
                             bio_prior_stats=bio_prior_stats, bio_knn_k=cfg.bio_knn_k,
                             synergy_metric_cols=syn_cols)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size * 2,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size * 2,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds


# ── Inference helper ──────────────────────────────────────────────────────────

def _collect_predictions(
    model: UniMoS,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """
    Run inference over `loader`.  Returns numpy arrays keyed by:
        prob, y_class, yhat_ri_A, y_ri_A, yhat_ri_B, y_ri_B
        (+ yhat_syn, y_syn when use_synergy_regression)
    """
    model.eval()
    buckets: dict[str, list] = {
        "prob": [], "y_class": [],
        "yhat_ri_A": [], "y_ri_A": [],
        "yhat_ri_B": [], "y_ri_B": [],
    }
    want_syn = bool(getattr(model, "use_synergy_regression", False))
    if want_syn:
        buckets["yhat_syn"] = []
        buckets["y_syn"] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out   = model(batch)
            buckets["prob"].append(torch.sigmoid(out["logit_class"]).cpu())
            buckets["y_class"].append(batch["y_class"].cpu())
            buckets["yhat_ri_A"].append(out["yhat_ri_A"].cpu())
            buckets["y_ri_A"].append(batch["y_ri_A"].cpu())
            buckets["yhat_ri_B"].append(out["yhat_ri_B"].cpu())
            buckets["y_ri_B"].append(batch["y_ri_B"].cpu())
            if want_syn:
                buckets["yhat_syn"].append(out["yhat_syn"].cpu())
                buckets["y_syn"].append(batch["y_syn_metrics"].cpu())

    return {k: torch.cat(v).numpy() for k, v in buckets.items()}


def _syn_reg_metrics(
    yhat_syn: np.ndarray,
    y_syn: np.ndarray,
    mean: list[float],
    std: list[float],
    cols: tuple[str, ...] = _SYN_METRIC_COLS,
) -> dict[str, float]:
    """Pearson / RMSE of de-standardised predictions vs raw synergy scores."""
    from scipy.stats import pearsonr

    out: dict[str, float] = {}
    mean_a = np.asarray(mean, dtype=np.float64)
    std_a = np.asarray(std, dtype=np.float64)
    pred = yhat_syn.astype(np.float64) * std_a + mean_a
    for i, col in enumerate(cols):
        yt = y_syn[:, i].astype(np.float64)
        yp = pred[:, i]
        m = np.isfinite(yt) & np.isfinite(yp)
        if m.sum() < 3:
            out[f"{col}_pearson"] = float("nan")
            out[f"{col}_rmse"] = float("nan")
            continue
        r, _ = pearsonr(yt[m], yp[m])
        out[f"{col}_pearson"] = float(r)
        out[f"{col}_rmse"] = float(np.sqrt(np.mean((yt[m] - yp[m]) ** 2)))
    return out


# ── Training entry ────────────────────────────────────────────────────────────

def train_run(
    split_type: str,
    seed: int,
    cfg: TrainConfig,
    output_dir: Path,
    fold: Optional[int] = None,
    n_folds: int = 5,
    fast_dev_run: bool = False,
    pair_csv: Path = _PAIR_CSV,
    proc_dir: Path = _PROC,
    func_dir: Path = _FUNC,
    split_npz: Optional[Path] = None,
    split_parquet: Optional[Path] = None,
) -> dict:
    """
    Full training run using Lightning Trainer.
    Returns the metrics dict (also written to metrics.json).

    fold     : CV fold index (0-based). None = single 80/10/10 split.
    n_folds  : total number of CV folds (used only when fold is not None).
    fast_dev_run=True caps training at 2 epochs (for CI / smoke tests).
    """
    L.seed_everything(seed, workers=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ─────────────────────────────────────────────────────────────────
    if str(pair_csv).endswith('.parquet'):
        df = pd.read_parquet(pair_csv)
    else:
        df = pd.read_csv(pair_csv, low_memory=False)

    precomputed = None
    if split_parquet is not None:
        # Materialised split: the rows and their train/val/test assignment live in
        # one file, so the assignment cannot be paired with the wrong table the way
        # a bare index file can (see scripts/materialize_splits.py). Overrides
        # whatever --pair-csv pointed at.
        df = pd.read_parquet(split_parquet)
        if "split" not in df.columns:
            raise ValueError(f"{split_parquet} has no 'split' column")
        assign = df["split"].to_numpy()
        precomputed = {k: np.flatnonzero(assign == k) for k in ("train", "val", "test")}
        empty = [k for k, v in precomputed.items() if v.size == 0]
        if empty:
            raise ValueError(f"{split_parquet}: empty split(s) {empty}")
        df = df.drop(columns=["split"])
        print(f"[split] materialised {split_parquet.name}: "
              + " ".join(f"{k}={v.size:,}" for k, v in precomputed.items()))
    elif split_npz is not None:
        split_file = np.load(split_npz, allow_pickle=True)
        precomputed = {
            "train": split_file["train_indices"].astype(int),
            "val": split_file["val_indices"].astype(int),
            "test": split_file["test_indices"].astype(int),
        }
        # Index files are only meaningful against the table they were built from.
        n_idx = sum(v.size for v in precomputed.values())
        if n_idx != len(df):
            raise ValueError(
                f"split index/table mismatch: {split_npz.name} covers {n_idx:,} rows "
                f"but {pair_csv} has {len(df):,}. These indices were built for a "
                f"different table; using them here would silently select the wrong "
                f"rows and destroy the held-out semantics. Use the matching table, "
                f"or a materialised split via --split-parquet."
            )
    elif fold is not None:
        cv_splits  = build_cv_splits(df, split_type, n_folds=n_folds, seed=seed)
        precomputed = cv_splits[fold]

    train_loader, val_loader, test_loader, train_ds = build_loaders(
        df, split_type, seed, cfg,
        proc_dir=proc_dir, func_dir=func_dir,
        precomputed_splits=precomputed,
    )
    n_hvg      = train_ds[0]["c_raw"].shape[0]
    pos_weight = train_ds.compute_pos_weight() * cfg.pos_weight_scale

    syn_mean, syn_std = (None, None)
    syn_clip_lo, syn_clip_hi = (None, None)
    syn_cols = cfg.active_syn_cols() if cfg.use_synergy_regression else ()
    if cfg.use_synergy_regression:
        syn_mean, syn_std, syn_clip_lo, syn_clip_hi = train_ds.compute_syn_metric_stats(
            clip_pct=tuple(cfg.syn_winsorize_pct),
        )
        train_ds.set_syn_clip_bounds(syn_clip_lo, syn_clip_hi)
        # Val/test loaders share the same Dataset objects already built — apply
        # train-derived winsorize bounds to every split's dataset.
        for loader in (train_loader, val_loader, test_loader):
            loader.dataset.set_syn_clip_bounds(syn_clip_lo, syn_clip_hi)
        print(
            f"[syn-reg] metrics={list(syn_cols)} "
            f"winsorize={cfg.syn_winsorize_pct} "
            f"clip_lo={syn_clip_lo} clip_hi={syn_clip_hi} "
            f"mean={syn_mean} std={syn_std} λ_syn={cfg.lambda_syn_reg} "
            f"λ_cls={cfg.lambda_cls} primary={cfg.syn_reg_primary} "
            f"primary_metrics={list(cfg.syn_primary_metrics) if cfg.syn_reg_primary else 'all'} "
            f"weights={cfg.syn_metric_weights(syn_cols)} monitor={cfg.early_stop_metric}",
            flush=True,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UniMoS(
        dropout     = cfg.dropout,
        rank_r      = cfg.rank_r,
        struct_hidden = cfg.struct_hidden,
        cell_hidden = cfg.cell_hidden,
        n_hvg       = n_hvg,
        lr          = cfg.lr,
        max_epochs  = cfg.max_epochs,
        pos_weight  = pos_weight,
        lambda_ri   = cfg.lambda_ri,
        lambda_res  = cfg.lambda_res,
        lambda_gate = cfg.lambda_gate,
        lambda_W    = cfg.lambda_W,
        lambda_gamma= cfg.lambda_gamma,
        w_prior_path = cfg.w_prior_path,
        w_prior_scale= cfg.w_prior_scale,
        loss_type    = cfg.loss_type,
        focal_gamma  = cfg.focal_gamma,
        struct_encoder = cfg.struct_encoder,
        graph_cache  = cfg.graph_cache,
        gnn_hidden   = cfg.gnn_hidden,
        gnn_layers   = cfg.gnn_layers,
        use_descriptors = cfg.use_descriptors,
        desc_cache   = cfg.desc_cache,
        desc_dim     = cfg.desc_dim,
        use_sensitivity_profile = cfg.use_sensitivity_profile,
        sens_drug_cache = cfg.sens_drug_cache,
        sens_cell_cache = cfg.sens_cell_cache,
        sens_dim     = cfg.sens_dim,
        use_virtual_cell = cfg.use_virtual_cell,
        fm_emb_path  = cfg.fm_emb_path,
        fm_out_dim   = cfg.fm_out_dim,
        z_cell_dim   = cfg.z_cell_dim,
        use_film_rescue = cfg.use_film_rescue,
        film_rescue_emb_path = cfg.film_rescue_emb_path,
        film_hidden  = cfg.film_hidden,
        cell_specific_w_prior = cfg.cell_specific_w_prior,
        w_prior_percell_path = cfg.w_prior_percell_path,
        tau_prior    = cfg.tau_prior,
        gi_prior_path = cfg.gi_prior_path,
        gi_prior_init_scale = cfg.gi_prior_init_scale,
        ablation_mode = cfg.ablation_mode,
        cell_cond_mode = cfg.cell_cond_mode,
        fm_kernel_emb_path = cfg.fm_kernel_emb_path,
        fm_kernel_tau = cfg.fm_kernel_tau,
        fm_kernel_self_mask = cfg.fm_kernel_self_mask,
        use_pair_prior = cfg.use_pair_prior,
        use_bio_prior = cfg.use_bio_prior,
        lr_scheduler_type = cfg.lr_scheduler_type,
        lr_patience = cfg.lr_patience,
        lr_factor = cfg.lr_factor,
        use_synergy_regression = cfg.use_synergy_regression,
        lambda_syn_reg = cfg.lambda_syn_reg,
        lambda_cls = cfg.lambda_cls,
        n_syn_metrics = len(syn_cols) if syn_cols else len(_SYN_METRIC_COLS),
        syn_metric_hidden = cfg.syn_metric_hidden,
        syn_metric_mean = syn_mean,
        syn_metric_std = syn_std,
        syn_metric_weights = cfg.syn_metric_weights(syn_cols) if syn_cols else None,
        early_stop_metric = cfg.early_stop_metric,
        syn_metric_names = syn_cols if syn_cols else _SYN_METRIC_COLS,
    )

    # FM-kernel interp: dictionary = train-fold cells only (LCO-safe)
    if cfg.cell_cond_mode == "fm_kernel_interp":
        cell_idx = json.load(open(proc_dir / "cell_feature_index.json"))
        train_rows = sorted({
            int(cell_idx[c])
            for c in train_ds.rows["cell_feature_id"].unique()
            if c in cell_idx
        })
        model.set_train_cell_mask(train_rows)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    raw_monitor = cfg.early_stop_metric if cfg.use_synergy_regression else "val_auroc"
    extra_cbs: list[Callback] = []

    # Optional trailing-mean monitor: checkpointing and early stopping key off
    # the smoothed series, while the raw one is still recorded per epoch.
    smooth_cb = _SmoothedMonitor(raw_monitor, f"{raw_monitor}_smooth{cfg.es_smooth}",
                                 cfg.es_smooth)
    extra_cbs.append(smooth_cb)
    monitor = smooth_cb.dst if cfg.es_smooth > 1 else raw_monitor

    ckpt_callback = ModelCheckpoint(
        dirpath   = str(output_dir),
        filename  = "best",
        monitor   = monitor,
        mode      = "max",
        save_top_k= 1,
    )
    early_stop_cb = LightningEarlyStopping(
        monitor  = monitor,
        mode     = "max",
        patience = cfg.patience,
    )

    pool_cb = _TopKWeightPool(smooth_cb, cfg.ckpt_avg) if cfg.ckpt_avg > 0 else None
    if pool_cb is not None:
        extra_cbs.append(pool_cb)

    max_epochs = 2 if fast_dev_run else cfg.max_epochs

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs          = max_epochs,
        callbacks           = [*extra_cbs, ckpt_callback, early_stop_cb, _EpochPrinter()],
        enable_progress_bar = True,
        enable_model_summary= False,
        logger              = False,
        accelerator         = "auto",
        devices             = 1,
    )
    trainer.fit(
        model,
        train_dataloaders = train_loader,
        val_dataloaders   = val_loader,
    )

    # ── Load best checkpoint ──────────────────────────────────────────────────
    # Use Lightning's resolved path. If output_dir already contains best.ckpt,
    # ModelCheckpoint may version the new file as best-v1.ckpt.
    best_ckpt = Path(ckpt_callback.best_model_path) if ckpt_callback.best_model_path else output_dir / "best.ckpt"
    best_model = UniMoS.load_from_checkpoint(str(best_ckpt), map_location=device)
    best_model.eval()

    # ── Optional checkpoint averaging ─────────────────────────────────────────
    avg_info: dict | None = None
    if pool_cb is not None and pool_cb.entries:
        avg_state, avg_epochs, avg_scores = pool_cb.averaged_state()
        best_model.load_state_dict(avg_state)
        best_model.to(device)
        bn_batches = _reestimate_bn(best_model, train_loader, device,
                                    cfg.ckpt_avg_bn_batches)
        best_model.eval()
        avg_info = {
            "n_averaged": len(avg_epochs),
            "epochs": avg_epochs,
            "monitor_values": avg_scores,
            "bn_batches": bn_batches,
        }
        torch.save({"state_dict": avg_state, "avg_info": avg_info},
                   output_dir / "avg.ckpt")
        print(f"[ckpt-avg] averaged {len(avg_epochs)} epochs {sorted(avg_epochs)} "
              f"(BN re-estimated over {bn_batches} batches)")

    # ── Val predictions (used for threshold selection) ────────────────────────
    pred_v = _collect_predictions(best_model, val_loader, device)
    prob_v, y_v = pred_v["prob"], pred_v["y_class"]
    yhat_A_v, yri_A_v = pred_v["yhat_ri_A"], pred_v["y_ri_A"]
    yhat_B_v, yri_B_v = pred_v["yhat_ri_B"], pred_v["y_ri_B"]
    best_threshold = find_best_threshold(prob_v, y_v)

    # ── Test predictions ──────────────────────────────────────────────────────
    pred_t = _collect_predictions(best_model, test_loader, device)
    prob_t, y_t = pred_t["prob"], pred_t["y_class"]
    yhat_A_t, yri_A_t = pred_t["yhat_ri_A"], pred_t["y_ri_A"]
    yhat_B_t, yri_B_t = pred_t["yhat_ri_B"], pred_t["y_ri_B"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    _REPORT_KEYS = (
        "auroc", "auprc", "accuracy", "f1", "precision", "recall",
        "balanced_accuracy", "mcc", "specificity",
    )

    ri_sum_v = _sigmoid(yhat_A_v + yhat_B_v)
    val_full = compute_metrics(prob_v, y_v, yhat_A_v, yri_A_v, yhat_B_v, yri_B_v, ri_sum_v,
                               threshold=best_threshold)
    val_out  = {k: val_full[k] for k in _REPORT_KEYS}

    ri_sum_t  = _sigmoid(yhat_A_t + yhat_B_t)
    test_full = compute_metrics(prob_t, y_t, yhat_A_t, yri_A_t, yhat_B_t, yri_B_t, ri_sum_t,
                                threshold=best_threshold)
    test_out  = {k: test_full[k] for k in _REPORT_KEYS}
    test_out["best_threshold"] = best_threshold

    if cfg.use_synergy_regression and syn_mean is not None and syn_std is not None:
        val_out.update(_syn_reg_metrics(
            pred_v["yhat_syn"], pred_v["y_syn"], syn_mean, syn_std, cols=syn_cols,
        ))
        test_out.update(_syn_reg_metrics(
            pred_t["yhat_syn"], pred_t["y_syn"], syn_mean, syn_std, cols=syn_cols,
        ))

    # Best epoch comes from the saved Lightning checkpoint payload
    _ckpt_raw  = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
    best_epoch = int(_ckpt_raw.get("epoch", -1))
    best_monitor = (
        float(ckpt_callback.best_model_score)
        if ckpt_callback.best_model_score is not None
        else float(val_out.get("auroc", 0.0))
    )
    # Keep best_val_auroc key for downstream parsers; also store the actual monitor.
    best_auroc = float(val_out["auroc"])

    metrics = {
        "split":          split_type,
        "seed":           seed,
        "fold":           fold,
        "best_epoch":     best_epoch,
        "best_checkpoint": str(best_ckpt),
        "early_stop_metric": monitor,
        "best_monitor_value": best_monitor,
        "best_val_auroc": best_auroc,
        # Model-selection settings and the raw per-epoch monitor series, so a
        # run's selection behaviour can be audited without re-training.
        "selection": {
            "raw_monitor": raw_monitor,
            "es_smooth": cfg.es_smooth,
            "patience": cfg.patience,
            "ckpt_avg": cfg.ckpt_avg,
            "ckpt_avg_info": avg_info,
        },
        "val_curve_raw": smooth_cb.raw,
        "val_curve_smooth": smooth_cb.smooth if cfg.es_smooth > 1 else None,
        "use_synergy_regression": cfg.use_synergy_regression,
        "lambda_syn_reg": cfg.lambda_syn_reg if cfg.use_synergy_regression else 0.0,
        "lambda_cls": cfg.lambda_cls,
        "syn_reg_primary": cfg.syn_reg_primary,
        "syn_primary_metrics": list(cfg.syn_primary_metrics) if cfg.syn_reg_primary else list(syn_cols),
        "syn_aux_weight": cfg.syn_aux_weight if cfg.syn_reg_primary else None,
        "syn_metric_weights": cfg.syn_metric_weights(syn_cols) if cfg.use_synergy_regression else None,
        "syn_metric_cols": list(syn_cols) if cfg.use_synergy_regression else [],
        "syn_winsorize_pct": list(cfg.syn_winsorize_pct) if cfg.use_synergy_regression else None,
        "syn_clip_lo": syn_clip_lo,
        "syn_clip_hi": syn_clip_hi,
        "syn_metric_mean": syn_mean,
        "syn_metric_std": syn_std,
        "val":            val_out,
        "test":           test_out,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Save test predictions (schema matches baselines: binary_label/pred_prob/pred)
    preds_df = pd.DataFrame({
        "binary_label": y_t.astype(int),
        "pred_prob":    prob_t,
        "pred":         (prob_t >= best_threshold).astype(int),
    })
    if cfg.use_synergy_regression and syn_mean is not None and syn_std is not None:
        mean_a = np.asarray(syn_mean, dtype=np.float32)
        std_a = np.asarray(syn_std, dtype=np.float32)
        yhat_raw = pred_t["yhat_syn"].astype(np.float32) * std_a + mean_a
        for i, col in enumerate(syn_cols):
            preds_df[f"y_{col}"] = pred_t["y_syn"][:, i]
            preds_df[f"yhat_{col}"] = yhat_raw[:, i]
    preds_df.to_parquet(output_dir / "test_predictions.parquet", index=False)

    print(f"\nSaved checkpoint → {best_ckpt}")
    print(f"Saved metrics    → {output_dir / 'metrics.json'}")
    print(f"Saved predictions → {output_dir / 'test_predictions.parquet'}")
    return metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_checkpoint(ckpt_path: Path, device: Optional[torch.device] = None) -> UniMoS:
    """Load a UniMoS model from a Lightning checkpoint saved by train_run()."""
    if device is None:
        device = torch.device("cpu")
    model = UniMoS.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval()
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UniMoS single training run")
    p.add_argument("--split",      default="ldo",
                   choices=["ldo", "lco", "lpo", "random"])
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--fold",       type=int, default=None,
                   help="CV fold index (0-based). Omit for single 80/10/10 split.")
    p.add_argument("--n-folds",    type=int, default=5,
                   help="Total CV folds (default: 5). Used only with --fold.")
    p.add_argument("--config",     type=Path, default=None,
                   help="Hyperparameter YAML. Default: configs/{split}.yaml")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-epochs", type=int,  default=None)
    p.add_argument("--patience",   type=int,  default=None,
                   help="Override the config's early-stopping patience.")
    p.add_argument("--es-smooth",  type=int,  default=None,
                   help="Select and early-stop on a trailing K-epoch mean of the val "
                        "monitor instead of its raw per-epoch value (1 = off).")
    p.add_argument("--ckpt-avg",   type=int,  default=None,
                   help="Final model = parameter-wise mean of the N epochs scoring "
                        "highest on the val monitor, with BatchNorm re-estimated "
                        "(0 = off, use the single best checkpoint).")
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--pair-csv",    type=Path, default=None,
                   help="Drug-pair dataset (CSV or parquet). Defaults to the built-in DrugCombv15 CSV.")
    p.add_argument("--split-npz",   type=Path, default=None,
                   help="Optional precomputed split .npz with train_indices/val_indices/test_indices. "
                        "Only valid against the table it was built from; a row-count mismatch is a "
                        "hard error.")
    p.add_argument("--split-parquet", type=Path, default=None,
                   help="Materialised split: one parquet holding the rows plus a 'split' column "
                        "(train/val/test). Self-contained, so it cannot be paired with the wrong "
                        "table. Overrides --pair-csv and --split-npz. "
                        "See scripts/materialize_splits.py and splits_materialized/.")
    p.add_argument("--proc-dir",    type=Path, default=None,
                   help="Override for the processed-features directory (default: data/processed). "
                        "Point at a frozen snapshot (e.g. data_vc_frozen/processed) to decouple a run "
                        "from the shared, mutable data/ symlink.")
    p.add_argument("--func-dir",    type=Path, default=None,
                   help="Override for the function-node-vectors directory "
                        "(default: data/Drugcombv15/function_nodes). See --proc-dir.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = args.config if args.config is not None else _REPO / "configs" / f"{args.split}.yaml"
    cfg  = TrainConfig.from_yaml(cfg_path)
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.patience is not None:
        cfg.patience = args.patience
    if args.es_smooth is not None:
        cfg.es_smooth = args.es_smooth
    if args.ckpt_avg is not None:
        cfg.ckpt_avg = args.ckpt_avg

    if args.output_dir is not None:
        out_dir = args.output_dir
    elif args.fold is not None:
        out_dir = _REPO / "checkpoints" / args.split / f"seed_{args.seed}" / f"fold_{args.fold}"
    else:
        out_dir = _REPO / "checkpoints" / args.split / f"seed_{args.seed}"

    train_run(
        split_type   = args.split,
        seed         = args.seed,
        cfg          = cfg,
        output_dir   = out_dir,
        fold         = args.fold,
        n_folds      = args.n_folds,
        fast_dev_run = args.fast_dev_run,
        pair_csv     = args.pair_csv if args.pair_csv is not None else _PAIR_CSV,
        split_npz    = args.split_npz,
        split_parquet= args.split_parquet,
        proc_dir     = args.proc_dir if args.proc_dir is not None else _PROC,
        func_dir     = args.func_dir if args.func_dir is not None else _FUNC,
    )


if __name__ == "__main__":
    main()
