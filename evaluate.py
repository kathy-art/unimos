"""
evaluate.py — UniMoS Evaluation + Interpretability Export (SPEC-11).

Usage
-----
# Single-run evaluation
python evaluate.py run \
  --checkpoint checkpoints/ldo/seed_0/best.ckpt \
  --split ldo --seed 0

# Cross-seed kernel stability (after ≥2 seeds evaluated)
python evaluate.py stability \
  checkpoints/ldo/seed_0/kernel_weights.npz \
  checkpoints/ldo/seed_1/kernel_weights.npz \
  checkpoints/ldo/seed_2/kernel_weights.npz \
  --output results/kernel_stability.json

# Phase 4 interpretability hardening (G-INTERP) — prefer the dedicated driver:
#   python scripts/run_phase4_interp.py --ckpt-dir ... --split lco

Output files (per run)
----------------------
{output_dir}/
    test_predictions.csv    — per-row predictions + labels
    kernel_weights.npz      — W_sym (67,67) + gamma (67,)

results/
    kernel_stability.json   — pairwise W_sym cosine similarities across seeds
    phase4_interp/          — stability + necessity + reconnection (Phase 4)
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from unimos.data.dataset import UniMoSDataset, select_hvg
from unimos.data.splits import build_splits
from unimos.model.unimos import UniMoS
from unimos.training.metrics import compute_metrics

# ── Default paths ─────────────────────────────────────────────────────────────

_REPO     = Path(__file__).resolve().parent
_PROC     = _REPO / "data" / "processed"
_FUNC     = _REPO / "data" / "Drugcombv15" / "function_nodes"
_PAIR_CSV = _REPO / "data" / "Drugcombv15" / "drugcombv15_unimos_pair_study_both_pathway.csv"

# CSV columns written by run_evaluate (order matches SPEC-11)
PRED_CSV_COLS = [
    "drug1_inchikey", "drug2_inchikey", "cell_feature_id",
    "y_loewe", "y_class", "prob_synergy",
    "y_ri_A", "yhat_ri_A", "y_ri_B", "yhat_ri_B",
    "split_type", "seed",
]


# ── Pure-computation helpers ──────────────────────────────────────────────────

def extract_kernel_weights(model: UniMoS) -> dict[str, np.ndarray]:
    """
    Extract static, interpretable kernel weights from a UniMoS model.

    Returns
    -------
    W_sym : (67, 67) — symmetric part of W_base: (W_base + W_base^T) / 2.
            Represents the baseline pathway co-activation structure learned
            across all cells (no cell-conditioned ΔW).
    gamma : (67,) — signed per-node gating coefficients.
    """
    with torch.no_grad():
        W_base = model.kernel.W_base.detach().cpu()
        W_sym  = ((W_base + W_base.t()) / 2).numpy()
        gamma  = model.kernel.gamma.detach().cpu().numpy()
    return {"W_sym": W_sym, "gamma": gamma}


def compute_kernel_stability(npz_paths: list[Path]) -> dict[str, float]:
    """
    Compute pairwise cosine similarity of W_sym vectors across seeds.

    Parameters
    ----------
    npz_paths : paths to kernel_weights.npz files, one per seed.

    Returns
    -------
    Dict with keys "W_sym_cosine_{i}{j}" for each pair and
    "W_sym_mean_cosine" (mean over all pairs).
    """
    w_syms = [np.load(p)["W_sym"].ravel().astype(np.float64) for p in npz_paths]
    result: dict[str, float] = {}
    cos_vals: list[float] = []

    for i, j in itertools.combinations(range(len(w_syms)), 2):
        a, b   = w_syms[i], w_syms[j]
        norms  = np.linalg.norm(a) * np.linalg.norm(b)
        cos    = float(np.dot(a, b) / norms) if norms > 0 else 0.0
        result[f"W_sym_cosine_{i}{j}"] = cos
        cos_vals.append(cos)

    result["W_sym_mean_cosine"] = float(np.mean(cos_vals)) if cos_vals else float("nan")
    return result


# ── Data loading helper ───────────────────────────────────────────────────────

def _make_test_dataset_and_loader(
    df: pd.DataFrame,
    split_type: str,
    seed: int,
    proc_dir: Path,
    func_dir: Path,
    batch_size: int = 512,
) -> tuple[UniMoSDataset, DataLoader]:
    """
    Build test dataset with the same norm_stats and HVG selection as training.
    Mirrors build_loaders() in train.py for the test split.
    """
    splits   = build_splits(df, split_type, seed=seed)
    train_ds = UniMoSDataset(df, splits["train"], proc_dir=proc_dir, func_dir=func_dir)
    norm_stats = train_ds.norm_stats()

    # LCO: fold-aware HVG (must match training — unique cells, not per-row)
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

    test_ds = UniMoSDataset(
        df, splits["test"],
        proc_dir        = proc_dir,
        func_dir        = func_dir,
        cell_norm_stats = norm_stats,
        hvg_gene_ids    = hvg_gene_ids,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_ds, test_loader


# ── Main evaluation entry point ───────────────────────────────────────────────

def run_evaluate(
    ckpt_path: Path,
    split_type: str,
    seed: int,
    output_dir: Optional[Path] = None,
    pair_csv: Path = _PAIR_CSV,
    proc_dir: Path = _PROC,
    func_dir: Path = _FUNC,
) -> dict:
    """
    Evaluate a checkpoint on the test split.

    Writes
    ------
    {output_dir}/test_predictions.csv
    {output_dir}/kernel_weights.npz

    Returns
    -------
    Summary dict with keys: split, seed, n_test, metrics, pred_path, npz_path.
    """
    if output_dir is None:
        output_dir = ckpt_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    model = UniMoS.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval()

    # ── Build test data ───────────────────────────────────────────────────────
    df      = pd.read_csv(pair_csv, low_memory=False)
    test_ds, test_loader = _make_test_dataset_and_loader(
        df, split_type, seed, proc_dir, func_dir
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    probs, yhat_A, yhat_B = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(batch)
            probs.append(torch.sigmoid(out["logit_class"]).cpu().numpy())
            yhat_A.append(out["yhat_ri_A"].cpu().numpy())
            yhat_B.append(out["yhat_ri_B"].cpu().numpy())

    prob_arr  = np.concatenate(probs)
    yhat_A_arr= np.concatenate(yhat_A)
    yhat_B_arr= np.concatenate(yhat_B)

    # ── Assemble predictions CSV ──────────────────────────────────────────────
    rows = test_ds.rows
    df_out = pd.DataFrame({
        "drug1_inchikey": rows["drug1_inchikey"].values,
        "drug2_inchikey": rows["drug2_inchikey"].values,
        "cell_feature_id": rows["cell_feature_id"].values,
        "y_loewe":    rows["loewe"].values if "loewe" in rows.columns else np.nan,
        "y_class":    rows["label_loewe_gt_10"].values,
        "prob_synergy": prob_arr,
        "y_ri_A":     rows["ri_drug1"].values if "ri_drug1" in rows.columns else np.nan,
        "yhat_ri_A":  yhat_A_arr,
        "y_ri_B":     rows["ri_drug2"].values if "ri_drug2" in rows.columns else np.nan,
        "yhat_ri_B":  yhat_B_arr,
        "split_type": split_type,
        "seed":       seed,
    })

    pred_path = output_dir / "test_predictions.csv"
    df_out.to_csv(pred_path, index=False)

    # ── Kernel weights ────────────────────────────────────────────────────────
    kw       = extract_kernel_weights(model)
    npz_path = output_dir / "kernel_weights.npz"
    np.savez(str(npz_path), **kw)

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_class = rows["label_loewe_gt_10"].values.astype(float)
    y_ri_A  = rows["ri_drug1"].values.astype(float) if "ri_drug1" in rows.columns \
              else np.full_like(yhat_A_arr, np.nan)
    y_ri_B  = rows["ri_drug2"].values.astype(float) if "ri_drug2" in rows.columns \
              else np.full_like(yhat_B_arr, np.nan)
    ri_sum  = 1.0 / (1.0 + np.exp(-(yhat_A_arr + yhat_B_arr)))

    metrics = compute_metrics(
        prob_arr, y_class, yhat_A_arr, y_ri_A, yhat_B_arr, y_ri_B, ri_sum
    )

    summary = {
        "split":     split_type,
        "seed":      seed,
        "n_test":    len(df_out),
        "metrics":   metrics,
        "pred_path": str(pred_path),
        "npz_path":  str(npz_path),
    }
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cmd_run(args: argparse.Namespace) -> None:
    ckpt_path  = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    result     = run_evaluate(
        ckpt_path  = ckpt_path,
        split_type = args.split,
        seed       = args.seed,
        output_dir = output_dir,
    )
    print(f"n_test={result['n_test']}  "
          f"auroc={result['metrics'].get('auroc', float('nan')):.4f}")
    print(f"Predictions → {result['pred_path']}")
    print(f"Kernel      → {result['npz_path']}")


def _cmd_stability(args: argparse.Namespace) -> None:
    npz_paths  = [Path(p) for p in args.npz_files]
    result     = compute_kernel_stability(npz_paths)
    output     = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"W_sym mean cosine: {result.get('W_sym_mean_cosine', float('nan')):.4f}")
    print(f"Stability → {output}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UniMoS Evaluation + Interpretability")
    sub = p.add_subparsers(dest="command", required=True)

    # run subcommand
    run_p = sub.add_parser("run", help="Evaluate a checkpoint on the test split")
    run_p.add_argument("--checkpoint", required=True, type=str)
    run_p.add_argument("--split", required=True, choices=["ldo","lco","lpo","random"])
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--output-dir", type=str, default=None)

    # stability subcommand
    stab_p = sub.add_parser("stability", help="Compute kernel stability across seeds")
    stab_p.add_argument("npz_files", nargs="+", help="kernel_weights.npz paths")
    stab_p.add_argument("--output", default="results/kernel_stability.json")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        _cmd_run(args)
    elif args.command == "stability":
        _cmd_stability(args)


if __name__ == "__main__":
    main()
