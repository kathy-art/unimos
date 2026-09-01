"""
tune.py — UniMoS HPO with Optuna (SPEC-10).

Usage
-----
python tune.py \
  --n-trials 50 \
  --split ldo --seed 0 \
  --output configs/best_hparams.yaml \
  [--storage sqlite:///opt/optuna_study.db]

Each trial runs a full Lightning training run (with early-stopping + pruning)
and returns best val_auroc.  MedianPruner cancels under-performing trials after
n_startup_trials=5.

Output
------
configs/best_hparams.yaml    — best trial hparams (loadable by TrainConfig.from_yaml)
[optuna_study.db]            — SQLite study (only if --storage is given)
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, replace as _replace_cfg
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import EarlyStopping as LightningEarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner

from train import TrainConfig, build_loaders, _collect_predictions, _sigmoid
from unimos.model.unimos import UniMoS
from unimos.training.metrics import compute_metrics

# ── Default paths ─────────────────────────────────────────────────────────────

_REPO       = Path(__file__).resolve().parent
# Same self-contained source as train.py; nothing reaches outside the repository.
_TRAIN_DATA = _REPO / "train_data"
_PROC     = _TRAIN_DATA / "processed"
_FUNC     = _TRAIN_DATA / "function_nodes"
_PAIR_CSV = _TRAIN_DATA / "pairs" / "unimos_stratified_dataset.parquet"


def _read_pair_table(pair_csv: Path) -> pd.DataFrame:
    if str(pair_csv).endswith(".parquet"):
        return pd.read_parquet(pair_csv)
    return pd.read_csv(pair_csv, low_memory=False)


def _load_split(
    pair_csv: Path,
    split_npz: Optional[Path],
    split_parquet: Optional[Path],
    seed: int,
):
    """Return (df, precomputed_splits).

    A materialised split (--split-parquet) carries its own rows, so the table
    comes out of the same file as the assignment and the two cannot be mismatched.
    A bare index file (--split-npz) is only valid against the table it was built
    from; the row count is checked to make that failure loud instead of silent.
    """
    if split_parquet is not None:
        path = Path(str(split_parquet).format(seed=seed))
        df = pd.read_parquet(path)
        assign = df["split"].to_numpy()
        pre = {k: np.flatnonzero(assign == k) for k in ("train", "val", "test")}
        return df.drop(columns=["split"]), pre

    df = _read_pair_table(pair_csv)
    if split_npz is None:
        return df, None

    path = Path(str(split_npz).format(seed=seed))
    z = np.load(path, allow_pickle=True)
    pre = {
        "train": z["train_indices"].astype(int),
        "val": z["val_indices"].astype(int),
        "test": z["test_indices"].astype(int),
    }
    n_idx = sum(v.size for v in pre.values())
    if n_idx != len(df):
        raise ValueError(
            f"split index/table mismatch: {path.name} covers {n_idx:,} rows but "
            f"{pair_csv} has {len(df):,}. Those indices were built for a different "
            f"table; applying them here would select the wrong rows."
        )
    return df, pre


# ── Optuna → Lightning pruning bridge ────────────────────────────────────────

class _OptunaPruningCallback(Callback):
    """
    Reports val_auroc to Optuna every `report_interval` epochs.
    Raises TrialPruned (propagates through trainer.fit) when Optuna decides
    the trial should be stopped early.
    """

    def __init__(self, trial: optuna.Trial, report_interval: int = 5):
        super().__init__()
        self.trial           = trial
        self.report_interval = report_interval

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        if (epoch + 1) % self.report_interval != 0:
            return
        val_auroc = float(trainer.callback_metrics.get("val_auroc", 0.0))
        self.trial.report(val_auroc, epoch)
        if self.trial.should_prune():
            raise TrialPruned(f"Trial pruned at epoch {epoch}")


# ── Search space ──────────────────────────────────────────────────────────────

def _suggest_config(
    trial: optuna.Trial,
    base_cfg: Optional[TrainConfig] = None,
    refine: bool = False,
    refine_v2: bool = False,
    broad_v4: bool = False,
    workflow: bool = False,
    tune_syn_weights: bool = False,
    ldo_wide: bool = False,
    split_type: str = "ldo",
) -> TrainConfig:
    """
    Sample a TrainConfig from the search space.
    Fixed fields (batch_size, max_epochs, patience) are taken from `base_cfg`.

    refine_v2=True  — tight trust-region around best_hparams.yaml (random-split optimum).
      Architecture fixed: rank_r=16, struct_hidden=512, cell_hidden=256.
      Lambda ranges ≈ ±3× the default values (log-scale); lr=[4e-5, 4e-4].
      For LCO/LDO, lambda_ri is searched in a smaller range because the RI
      regression auxiliary loss did not improve held-out generalization when
      tuned at high weights.
      Use --seeds 0 1 to average val_auroc across two seeds for a stable signal.

    refine=True  — original narrow mode (kept for backward compat):
      ldo: lr=[8e-5,3e-4], rank_r=16 fixed, struct/cell searched, lambdas wide.
      lco: lr=[1.1e-4,2.0e-4], rank_r=24 fixed, struct/cell fixed, lambdas wide.
    """
    if base_cfg is None:
        base_cfg = TrainConfig()

    # Objective-function knobs (only searched in workflow mode; else defaults).
    loss_type        = "bce"
    focal_gamma      = 2.0
    pos_weight_scale = 1.0

    # Multi-task loss balance. Left at the base config unless --tune-syn-weights
    # is passed: the rebuild's first HPO round fixed lambda_syn_reg=1.0 /
    # lambda_cls=0.5, and on LCO/LDO that config trails the baselines on AUPRC
    # by ~5x the AUROC gap while leading on F1/MCC — i.e. the regression head
    # is pulling the positive-class ranking around. Searching the balance is
    # opt-in so existing studies keep their parameter space.
    lambda_syn_reg = base_cfg.lambda_syn_reg
    lambda_cls     = base_cfg.lambda_cls
    wide_ldo = bool(ldo_wide and split_type == "ldo")
    if tune_syn_weights:
        # ldo-wide: round-2 best sat on lambda_cls=0.256 (floor 0.25).
        syn_lo, syn_hi = (0.02, 3.0) if wide_ldo else (0.05, 2.0)
        cls_lo, cls_hi = (0.08, 4.0) if wide_ldo else (0.25, 4.0)
        lambda_syn_reg = trial.suggest_float("lambda_syn_reg", syn_lo, syn_hi, log=True)
        lambda_cls     = trial.suggest_float("lambda_cls",     cls_lo, cls_hi, log=True)

    if workflow and wide_ldo:
        # Modest expansion of the LDO workflow box. Round-2 optima sat on the
        # lower edge of lr / pos_weight_scale / lambda_cls; all top-8 trials
        # used cell_hidden=256. New study required (Optuna cannot widen an
        # existing distribution).
        lr           = trial.suggest_float("lr",           2e-5,  6e-4, log=True)
        dropout      = trial.suggest_float("dropout",      0.05,  0.50)
        lambda_ri    = trial.suggest_float("lambda_ri",    0.02,  1.5,  log=True)
        lambda_res   = trial.suggest_float("lambda_res",   5e-4,  0.15, log=True)
        lambda_W     = trial.suggest_float("lambda_W",     5e-6,  2e-3, log=True)
        lambda_gamma = trial.suggest_float("lambda_gamma", 5e-6,  8e-3, log=True)
        lambda_gate  = trial.suggest_float("lambda_gate",  5e-6,  8e-3, log=True)
        rank_r       = trial.suggest_categorical("rank_r",        [16, 24, 32, 48])
        struct_hidden= trial.suggest_categorical("struct_hidden", [256, 512])
        cell_hidden  = trial.suggest_categorical("cell_hidden",   [128, 256])
        loss_type    = trial.suggest_categorical("loss_type", ["bce", "focal"])
        if loss_type == "focal":
            focal_gamma = trial.suggest_float("focal_gamma", 0.5, 3.0)
        pos_weight_scale = trial.suggest_float("pos_weight_scale", 0.25, 2.5)
    elif workflow:
        # Full-budget per-split search to push accuracy toward the targets.
        # in-distribution splits (random/lpo) want LOW lambda_ri; cold-start
        # (lco/ldo) benefit from a stronger RI regulariser.
        lr           = trial.suggest_float("lr",           5e-5,  5e-4, log=True)
        dropout      = trial.suggest_float("dropout",      0.10,  0.45)
        if split_type in {"lco", "ldo"}:
            lambda_ri = trial.suggest_float("lambda_ri", 0.05, 1.0, log=True)
        else:  # random, lpo
            lambda_ri = trial.suggest_float("lambda_ri", 1e-3, 0.3, log=True)
        lambda_res   = trial.suggest_float("lambda_res",   1e-3,  0.1,  log=True)
        lambda_W     = trial.suggest_float("lambda_W",     1e-5,  1e-3, log=True)
        lambda_gamma = trial.suggest_float("lambda_gamma", 1e-5,  5e-3, log=True)
        lambda_gate  = trial.suggest_float("lambda_gate",  1e-5,  5e-3, log=True)
        rank_r       = trial.suggest_categorical("rank_r",        [16, 24, 32])
        struct_hidden= trial.suggest_categorical("struct_hidden", [256, 512])
        cell_hidden  = trial.suggest_categorical("cell_hidden",   [128, 256])
        loss_type    = trial.suggest_categorical("loss_type", ["bce", "focal"])
        if loss_type == "focal":
            focal_gamma = trial.suggest_float("focal_gamma", 0.5, 3.0)
        pos_weight_scale = trial.suggest_float("pos_weight_scale", 0.5, 2.0)
    elif broad_v4:
        # AUROC-only heldout search.  This deliberately extends beyond the v3
        # trust region because several v3 optima landed close to a boundary.
        lr           = trial.suggest_float("lr",           1e-5,  1e-3, log=True)
        dropout      = trial.suggest_float("dropout",      0.05,  0.55)
        lambda_ri    = trial.suggest_float("lambda_ri",    1e-6,  0.5,  log=True)
        lambda_res   = trial.suggest_float("lambda_res",   1e-4,  0.3,  log=True)
        lambda_W     = trial.suggest_float("lambda_W",     1e-6,  2e-3, log=True)
        lambda_gamma = trial.suggest_float("lambda_gamma", 1e-5,  1e-2, log=True)
        lambda_gate  = trial.suggest_float("lambda_gate",  1e-5,  1e-2, log=True)
        rank_r       = trial.suggest_categorical("rank_r",        [8, 16, 24, 32])
        struct_hidden= trial.suggest_categorical("struct_hidden", [128, 256, 512])
        cell_hidden  = trial.suggest_categorical("cell_hidden",   [64, 128, 256])
    elif refine_v2:
        # Centered on best_hparams.yaml defaults:
        #   lr=1.24e-4, dropout=0.293, lambda_ri=0.475, lambda_res=0.0151,
        #   lambda_W=1.1e-4, lambda_gamma=5.2e-4, lambda_gate=1.67e-4
        lr           = trial.suggest_float("lr",           4e-5,  4e-4,  log=True)
        dropout      = trial.suggest_float("dropout",      0.15,  0.45)
        if split_type in {"lco", "ldo"}:
            # Trust region centered on best_hparams lambda_ri=0.475 (full-budget
            # optimum is ~0.5-0.8; the old 1e-4..0.3 cap biased it down and caused
            # the v4 regression — see project_lambda_ri_hpo).
            lambda_ri = trial.suggest_float("lambda_ri", 0.15, 0.9, log=True)
        else:
            lambda_ri = trial.suggest_float("lambda_ri", 0.01, 1.5, log=True)
        lambda_res   = trial.suggest_float("lambda_res",   3e-3,  0.08,  log=True)
        lambda_W     = trial.suggest_float("lambda_W",     3e-5,  4e-4,  log=True)
        lambda_gamma = trial.suggest_float("lambda_gamma", 1e-4,  2e-3,  log=True)
        lambda_gate  = trial.suggest_float("lambda_gate",  5e-5,  6e-4,  log=True)
        rank_r       = 16
        struct_hidden= 512
        cell_hidden  = 256
    elif refine and split_type == "lco":
        lr           = trial.suggest_float("lr",      1.1e-4, 2.0e-4, log=True)
        dropout      = trial.suggest_float("dropout", 0.15,   0.35)
        lambda_ri    = trial.suggest_float("lambda_ri",    1e-4,  0.3,   log=True)
        lambda_res   = trial.suggest_float("lambda_res",   1e-3,  0.1,   log=True)
        lambda_W     = trial.suggest_float("lambda_W",     1e-4,  1e-2,  log=True)
        lambda_gamma = trial.suggest_float("lambda_gamma", 1e-4,  1e-2,  log=True)
        lambda_gate  = trial.suggest_float("lambda_gate",  1e-4,  1e-2,  log=True)
        rank_r       = 24
        struct_hidden= 512
        cell_hidden  = 256
    elif refine:  # ldo
        lr           = trial.suggest_float("lr",           8e-5,  3e-4,  log=True)
        dropout      = trial.suggest_float("dropout",      0.1,   0.5)
        lambda_ri    = trial.suggest_float("lambda_ri",    1e-4,  0.3,   log=True)
        lambda_res   = trial.suggest_float("lambda_res",   1e-3,  0.1,   log=True)
        lambda_W     = trial.suggest_float("lambda_W",     1e-4,  1e-2,  log=True)
        lambda_gamma = trial.suggest_float("lambda_gamma", 1e-4,  1e-2,  log=True)
        lambda_gate  = trial.suggest_float("lambda_gate",  1e-4,  1e-2,  log=True)
        rank_r       = 16
        struct_hidden= trial.suggest_categorical("struct_hidden", [256, 512])
        cell_hidden  = trial.suggest_categorical("cell_hidden",   [128, 256])
    else:
        lr           = trial.suggest_float("lr",           1e-4,  5e-3,  log=True)
        dropout      = trial.suggest_float("dropout",      0.1,   0.5)
        if split_type in {"lco", "ldo"}:
            lambda_ri = trial.suggest_float("lambda_ri", 1e-4, 0.3, log=True)
        else:
            lambda_ri = trial.suggest_float("lambda_ri", 0.1, 2.0)
        lambda_res   = trial.suggest_float("lambda_res",   1e-3,  0.1,   log=True)
        lambda_W     = trial.suggest_float("lambda_W",     1e-4,  1e-2,  log=True)
        lambda_gamma = trial.suggest_float("lambda_gamma", 1e-4,  1e-2,  log=True)
        lambda_gate  = trial.suggest_float("lambda_gate",  1e-4,  1e-2,  log=True)
        rank_r       = trial.suggest_categorical("rank_r",        [8, 16, 24, 32])
        struct_hidden= trial.suggest_categorical("struct_hidden",  [128, 256, 512])
        cell_hidden  = trial.suggest_categorical("cell_hidden",    [64, 128, 256])

    # dataclasses.replace() keeps every base_cfg field not listed below as-is —
    # in particular use_virtual_cell/cell_cond_mode/fm_kernel_* (VC architecture
    # knobs) and batch_size/max_epochs/patience, none of which are searched here.
    return _replace_cfg(
        base_cfg,
        lr           = lr,
        dropout      = dropout,
        lambda_ri    = lambda_ri,
        lambda_res   = lambda_res,
        lambda_W     = lambda_W,
        lambda_gamma = lambda_gamma,
        lambda_gate  = lambda_gate,
        rank_r       = rank_r,
        struct_hidden= struct_hidden,
        cell_hidden  = cell_hidden,
        # Objective function (workflow mode; defaults otherwise)
        loss_type       = loss_type,
        focal_gamma     = focal_gamma,
        pos_weight_scale= pos_weight_scale,
        # Multi-task balance (searched only with --tune-syn-weights)
        lambda_syn_reg  = lambda_syn_reg,
        lambda_cls      = lambda_cls,
    )


# ── Single-trial training ─────────────────────────────────────────────────────

def _train_trial(
    trial: optuna.Trial,
    cfg: TrainConfig,
    split_type: str,
    seed: int,
    tmp_dir: Path,
    pair_csv: Path,
    proc_dir: Path,
    func_dir: Path,
    split_npz: Optional[Path] = None,
    split_parquet: Optional[Path] = None,
    enable_pruning: bool = True,
) -> dict[str, float]:
    """
    Run one HPO trial.  Returns validation metrics from the best val_auroc checkpoint.
    Raises TrialPruned if Optuna decides to stop early.
    """
    L.seed_everything(seed, workers=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, precomputed = _load_split(pair_csv, split_npz, split_parquet, seed)
    train_loader, val_loader, _, train_ds = build_loaders(
        df, split_type, seed, cfg,
        proc_dir=proc_dir,
        func_dir=func_dir,
        precomputed_splits=precomputed,
    )
    n_hvg      = train_ds[0]["c_raw"].shape[0]
    pos_weight = train_ds.compute_pos_weight() * cfg.pos_weight_scale

    model = UniMoS(
        dropout      = cfg.dropout,
        rank_r       = cfg.rank_r,
        struct_hidden= cfg.struct_hidden,
        cell_hidden  = cfg.cell_hidden,
        n_hvg        = n_hvg,
        lr           = cfg.lr,
        max_epochs   = cfg.max_epochs,
        pos_weight   = pos_weight,
        lambda_ri    = cfg.lambda_ri,
        lambda_res   = cfg.lambda_res,
        lambda_gate  = cfg.lambda_gate,
        lambda_W     = cfg.lambda_W,
        lambda_gamma = cfg.lambda_gamma,
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
    )

    # FM-kernel interp: dictionary = train-fold cells only (LCO-safe) — mirrors
    # train.py's train_run(); without this the kernel would leak val/test cells.
    if cfg.cell_cond_mode == "fm_kernel_interp":
        cell_idx = json.load(open(proc_dir / "cell_feature_index.json"))
        train_rows = sorted({
            int(cell_idx[c])
            for c in train_ds.rows["cell_feature_id"].unique()
            if c in cell_idx
        })
        model.set_train_cell_mask(train_rows)

    ckpt_cb    = ModelCheckpoint(
        dirpath=str(tmp_dir), filename="best",
        monitor="val_auroc", mode="max", save_top_k=1,
    )
    early_cb   = LightningEarlyStopping(
        monitor="val_auroc", mode="max", patience=cfg.patience
    )
    callbacks = [ckpt_cb, early_cb]
    if enable_pruning:
        callbacks.append(_OptunaPruningCallback(trial, report_interval=5))

    trainer = L.Trainer(
        max_epochs          = cfg.max_epochs,
        callbacks           = callbacks,
        enable_progress_bar = False,
        enable_model_summary= False,
        logger              = False,
        accelerator         = "auto",
        devices             = 1,
    )
    # TrialPruned raised inside pruning_cb propagates through trainer.fit().
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Compute validation metrics from checkpoint inference.  The checkpoint is
    # still selected by val_auroc, but the HPO objective can combine metrics.
    best_ckpt = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else tmp_dir / "best.ckpt"
    if best_ckpt.exists():
        best_model = UniMoS.load_from_checkpoint(str(best_ckpt), map_location=device)
        best_model.eval()
        # _collect_predictions returns a dict (it gained extra keys when the
        # synergy-regression head was added); unpacking it as a 6-tuple silently
        # yielded the key strings instead of the arrays.
        pred_v = _collect_predictions(best_model, val_loader, device)
        prob_v, y_v = pred_v["prob"], pred_v["y_class"]
        yhat_A_v, yri_A_v = pred_v["yhat_ri_A"], pred_v["y_ri_A"]
        yhat_B_v, yri_B_v = pred_v["yhat_ri_B"], pred_v["y_ri_B"]
        ri_sum = _sigmoid(yhat_A_v + yhat_B_v)
        m = compute_metrics(prob_v, y_v, yhat_A_v, yri_A_v, yhat_B_v, yri_B_v, ri_sum)
        return {
            "auroc": float(m.get("auroc", 0.0)),
            "auprc": float(m.get("auprc", 0.0)),
        }

    return {
        "auroc": float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else 0.0,
        "auprc": 0.0,
    }


# ── Objective ─────────────────────────────────────────────────────────────────

def objective(
    trial: optuna.Trial,
    split_type: str = "ldo",
    seeds: list = None,
    base_cfg: Optional[TrainConfig] = None,
    refine: bool = False,
    refine_v2: bool = False,
    broad_v4: bool = False,
    workflow: bool = False,
    tune_syn_weights: bool = False,
    ldo_wide: bool = False,
    auprc_weight: float = 0.0,
    std_penalty: float = 0.0,
    pair_csv: Path = _PAIR_CSV,
    proc_dir: Path = _PROC,
    func_dir: Path = _FUNC,
    split_npz: Optional[Path] = None,
    split_parquet: Optional[Path] = None,
) -> float:
    """
    Optuna objective function.  Returns the configured validation score across seeds.

    The per-seed score is (1 - auprc_weight) * AUROC + auprc_weight * AUPRC.
    For multi-seed objectives, pruning is disabled so a trial is not discarded
    based only on the first seed.
    """
    if seeds is None:
        seeds = [0]
    cfg = _suggest_config(
        trial,
        base_cfg,
        refine=refine,
        refine_v2=refine_v2,
        broad_v4=broad_v4,
        workflow=workflow,
        tune_syn_weights=tune_syn_weights,
        ldo_wide=ldo_wide,
        split_type=split_type,
    )
    scores = []
    aurocs = []
    auprcs = []
    enable_pruning = len(seeds) == 1
    for seed in seeds:
        with tempfile.TemporaryDirectory() as tmp:
            val_metrics = _train_trial(
                trial      = trial,
                cfg        = cfg,
                split_type = split_type,
                seed       = seed,
                tmp_dir    = Path(tmp) / f"trial_seed{seed}",
                pair_csv   = pair_csv,
                proc_dir   = proc_dir,
                func_dir   = func_dir,
                split_npz  = split_npz,
                split_parquet = split_parquet,
                enable_pruning = enable_pruning,
            )
        auroc = val_metrics["auroc"]
        auprc = val_metrics["auprc"]
        aurocs.append(auroc)
        auprcs.append(auprc)
        scores.append((1.0 - auprc_weight) * auroc + auprc_weight * auprc)

    mean_score = float(sum(scores) / len(scores))
    if len(scores) > 1 and std_penalty > 0:
        mean = mean_score
        variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
        mean_score -= std_penalty * float(variance ** 0.5)

    trial.set_user_attr("mean_val_auroc", float(sum(aurocs) / len(aurocs)))
    trial.set_user_attr("mean_val_auprc", float(sum(auprcs) / len(auprcs)))
    trial.set_user_attr("objective_score", mean_score)
    return mean_score


# ── HPO entry point ───────────────────────────────────────────────────────────

def _create_study_with_retry(
    study_name: str,
    storage: Optional[str],
    max_retries: int = 10,
) -> optuna.Study:
    """
    Create or load an Optuna study, retrying on SQLite init race conditions.

    When multiple processes start simultaneously with a shared SQLite storage,
    they can all try to CREATE TABLE at the same moment and fail with
    'table studies already exists'.  Retrying with backoff resolves this.
    """
    for attempt in range(max_retries):
        try:
            return optuna.create_study(
                study_name     = study_name,
                direction      = "maximize",
                pruner         = MedianPruner(n_startup_trials=5, n_warmup_steps=10),
                storage        = storage,
                load_if_exists = True,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.3 * (attempt + 1))
            else:
                raise


def run_hpo(
    n_trials: int,
    split_type: str = "ldo",
    seeds: list = None,
    max_epochs: int = 200,
    patience: int = 20,
    auprc_weight: float = 0.0,
    std_penalty: float = 0.0,
    study_name: str = "unimos_hpo",
    output: Path = _REPO / "configs" / "best_hparams.yaml",
    storage: Optional[str] = None,
    refine: bool = False,
    refine_v2: bool = False,
    broad_v4: bool = False,
    workflow: bool = False,
    tune_syn_weights: bool = False,
    ldo_wide: bool = False,
    pair_csv: Path = _PAIR_CSV,
    proc_dir: Path = _PROC,
    func_dir: Path = _FUNC,
    split_npz: Optional[Path] = None,
    split_parquet: Optional[Path] = None,
    base_config: Optional[Path] = None,
) -> dict:
    """
    Run Optuna HPO.  Returns best hparams dict (also written to `output`).

    Parameters
    ----------
    n_trials   : number of Optuna trials (per worker when running in parallel)
    split_type : dataset split for objective evaluation
    seeds      : list of random seeds; objective is averaged across seeds
    max_epochs : fixed training epochs per trial
    patience   : EarlyStopping patience per trial
    auprc_weight: mix AUPRC into the objective score; 0.0 = AUROC-only
    std_penalty : subtract this coefficient times the across-seed score std
    study_name : Optuna study name — must be the same across all parallel workers
    output     : path to write best_hparams.yaml
    storage    : Optuna storage URL (None = in-memory; use sqlite:/// for parallel)
    refine_v2  : narrow trust-region search around best_hparams.yaml defaults
    base_config: optional yaml (e.g. configs/lco_fm_kernel_interp_vc.yaml) providing
                 fixed, non-searched fields — in particular VC architecture knobs
                 (use_virtual_cell, cell_cond_mode, fm_kernel_*). Without this,
                 base_cfg defaults to the non-VC architecture (use_virtual_cell=False).
    """
    if seeds is None:
        seeds = [0]
    base_cfg = TrainConfig.from_yaml(base_config) if base_config is not None else TrainConfig()
    base_cfg = _replace_cfg(base_cfg, max_epochs=max_epochs, patience=patience)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _create_study_with_retry(study_name, storage)

    study.optimize(
        lambda t: objective(
            t,
            split_type = split_type,
            seeds      = seeds,
            base_cfg   = base_cfg,
            refine     = refine,
            refine_v2  = refine_v2,
            broad_v4   = broad_v4,
            workflow   = workflow,
            tune_syn_weights = tune_syn_weights,
            ldo_wide   = ldo_wide,
            auprc_weight = auprc_weight,
            std_penalty  = std_penalty,
            pair_csv   = pair_csv,
            proc_dir   = proc_dir,
            func_dir   = func_dir,
            split_npz  = split_npz,
            split_parquet = split_parquet,
        ),
        n_trials = n_trials,
        catch    = (Exception,),
    )

    best    = study.best_trial
    # Start from every base_cfg field (carries VC/w_prior/etc. fixed settings from
    # --base-config) so the output yaml is directly loadable by train.py, then
    # overlay the searched params on top.
    hparams = {k: v for k, v in asdict(base_cfg).items()}
    hparams.update(best.params)
    # refine_v2 fixes architecture — add them explicitly so YAML is complete
    if refine_v2:
        hparams.setdefault("rank_r",        16)
        hparams.setdefault("struct_hidden", 512)
        hparams.setdefault("cell_hidden",   256)
    hparams.update({
        "batch_size": base_cfg.batch_size,
        "max_epochs": max_epochs,
        "patience":   patience,
    })

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.dump(hparams, f, default_flow_style=False, sort_keys=True)

    print(f"\nBest trial #{best.number}: val_auroc={best.value:.4f}")
    print(f"Saved best hparams → {output}")
    return hparams


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UniMoS HPO with Optuna")
    p.add_argument("--n-trials",   type=int,  default=50)
    p.add_argument("--split",      default="ldo",
                   choices=["ldo", "lco", "lpo", "random"])
    p.add_argument("--seeds",      type=int,  nargs="+", default=[0],
                   help="Seeds for objective evaluation; objective score is averaged. "
                        "E.g. --seeds 0 1 averages two seeds per trial.")
    p.add_argument("--max-epochs", type=int,  default=200)
    p.add_argument("--patience",   type=int,  default=20)
    p.add_argument("--auprc-weight", type=float, default=0.0,
                   help="Objective mix: score=(1-w)*val_auroc + w*val_auprc.")
    p.add_argument("--std-penalty", type=float, default=0.0,
                   help="For multi-seed HPO, subtract this times score std.")
    p.add_argument("--output",     type=Path,
                   default=_REPO / "configs" / "best_hparams.yaml")
    p.add_argument("--storage",    type=str,  default=None,
                   help="Optuna storage, e.g. sqlite:///opt/optuna_study.db")
    p.add_argument("--study-name", type=str,  default="unimos_hpo",
                   help="Optuna study name (must match across all parallel workers).")
    p.add_argument("--refine",     action="store_true", default=False,
                   help="Original narrow search (kept for backward compat).")
    p.add_argument("--refine-v2",  action="store_true", default=False,
                   help="Trust-region search centered on best_hparams.yaml: "
                        "architecture fixed (rank_r=16, struct=512, cell=256), "
                        "lambdas ≈ ±3× defaults, lr=[4e-5,4e-4]. "
                        "Recommended with --seeds 0 1 for stable signal.")
    p.add_argument("--broad-v4", action="store_true", default=False,
                   help="Wide heldout search over optimization, auxiliary-loss, "
                        "and architecture parameters. Intended for a new study.")
    p.add_argument("--workflow", action="store_true", default=False,
                   help="Full-budget per-split accuracy-push search: split-specific "
                        "lambda_ri range + objective function (bce/focal), focal_gamma, "
                        "pos_weight_scale, capacity. Pair with --auprc-weight as needed.")
    p.add_argument("--tune-syn-weights", action="store_true", default=False,
                   help="Also search the multi-task loss balance lambda_syn_reg "
                        "(0.05-2.0) and lambda_cls (0.25-4.0). Off by default so "
                        "existing studies keep their parameter space; use a fresh "
                        "study name when enabling it.")
    p.add_argument("--ldo-wide", action="store_true", default=False,
                   help="LDO-only modest expansion of --workflow/--tune-syn-weights "
                        "bounds (lr, pos_weight, lambda_cls, rank_r=48). Requires a "
                        "new study name; no-op for other splits.")
    p.add_argument("--pair-csv", type=Path, default=_PAIR_CSV,
                   help="Drug-pair dataset (CSV or parquet).")
    p.add_argument("--split-parquet", type=Path, default=None,
                   help="Materialised split (rows + 'split' column). May contain "
                        "{seed} to substitute the seed. Overrides --pair-csv/--split-npz.")
    p.add_argument("--split-npz", type=Path, default=None,
                   help="Optional precomputed split .npz. Use '{seed}' in the path for seed-specific files.")
    p.add_argument("--proc-dir",  type=Path, default=None,
                   help="Override for the processed-features directory (default: data/processed).")
    p.add_argument("--func-dir",  type=Path, default=None,
                   help="Override for the function-node-vectors directory "
                        "(default: data/Drugcombv15/function_nodes).")
    p.add_argument("--base-config", type=Path, default=None,
                   help="Yaml providing fixed (non-searched) base fields, in particular "
                        "VC architecture knobs (use_virtual_cell, cell_cond_mode, "
                        "fm_kernel_*) — e.g. configs/lco_fm_kernel_interp_vc.yaml. "
                        "Without this, HPO tunes the non-VC architecture.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_hpo(
        n_trials   = args.n_trials,
        split_type = args.split,
        seeds      = args.seeds,
        max_epochs = args.max_epochs,
        patience   = args.patience,
        auprc_weight = args.auprc_weight,
        std_penalty  = args.std_penalty,
        study_name = args.study_name,
        output     = args.output,
        storage    = args.storage,
        refine     = args.refine,
        refine_v2  = args.refine_v2,
        broad_v4   = args.broad_v4,
        workflow   = args.workflow,
        tune_syn_weights = args.tune_syn_weights,
        ldo_wide   = args.ldo_wide,
        pair_csv   = args.pair_csv,
        split_npz  = args.split_npz,
        split_parquet = args.split_parquet,
        proc_dir   = args.proc_dir if args.proc_dir is not None else _PROC,
        func_dir   = args.func_dir if args.func_dir is not None else _FUNC,
        base_config = args.base_config,
    )


if __name__ == "__main__":
    main()
