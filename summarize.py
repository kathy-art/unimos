"""
summarize.py — UniMoS results aggregation (SPEC-12).

Two modes:

Single-split mode (default):
  Reads checkpoints/{split}/seed_{seed}/metrics.json
  Aggregates mean±std across seeds.

  python summarize.py --outputs-dir checkpoints/ --results-dir results/

CV mode (--cv):
  Reads checkpoints/{split}/seed_{seed}/fold_{fold}/metrics.json.
  For each (split, seed): averages across folds first.
  Then aggregates mean±std across seeds.

  python summarize.py --cv --folds 0 1 2 3 4 \
    --outputs-dir checkpoints/ --results-dir results/

Output (single-split):
  results/full_comparison.csv, results/{split}_summary.json

Output (CV):
  results/cv_full_comparison.csv, results/cv_{split}_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

_REPO    = Path(__file__).resolve().parent
_SPLITS  = ["ldo", "lco", "lpo", "random"]
_SEEDS   = [0, 1, 2]
_FOLDS   = [0, 1, 2, 3, 4]

# Keys never included as performance metrics
_SKIP_KEYS: frozenset[str] = frozenset({"best_threshold"})

# Preferred column order for the CSV (remaining keys appended alphabetically)
_PREFERRED_METRIC_ORDER = [
    "auroc", "auprc", "accuracy", "f1", "precision", "recall",
]


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _aggregate(values: list[float]) -> dict[str, float]:
    """
    Compute mean and std from a list of floats.  NaN values are excluded.

    Returns {"mean": float, "std": float}.
    std uses ddof=1 (unbiased); returns NaN when fewer than 2 finite values.
    """
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return {"mean": float("nan"), "std": float("nan")}
    mean = float(np.mean(clean))
    std  = float(np.std(clean, ddof=1)) if len(clean) > 1 else float("nan")
    return {"mean": mean, "std": std}


def compute_summary(
    metrics_by_split: dict[str, list[dict]],
    splits: list[str] = _SPLITS,
) -> tuple[pd.DataFrame, dict]:
    """
    Aggregate per-seed metric dicts into mean±std per split.

    Parameters
    ----------
    metrics_by_split : {"ldo": [test_dict_seed0, ...], "lco": [...], ...}
        Each inner dict maps metric_name → float.  Missing keys treated as NaN.
    splits : ordered list of splits to include (one row each in the CSV).

    Returns
    -------
    df        : pd.DataFrame, one row per split.
                Columns: split, n_seeds, {metric}_mean, {metric}_std, ...
    summaries : dict[str, dict]  — per-split summary with nested {"mean", "std"}.
    """
    # Collect all metric keys present in any seed, excluding skip set
    all_keys: set[str] = set()
    for ms in metrics_by_split.values():
        for d in ms:
            all_keys.update(
                k for k, v in d.items()
                if k not in _SKIP_KEYS and isinstance(v, (int, float))
            )

    # Order: preferred first, then remaining keys alphabetically
    ordered_keys = [k for k in _PREFERRED_METRIC_ORDER if k in all_keys]
    ordered_keys += sorted(all_keys - set(ordered_keys))

    rows: list[dict] = []
    summaries: dict[str, dict] = {}

    for split in splits:
        ms   = metrics_by_split.get(split, [])
        n    = len(ms)
        row  = {"split": split, "n_seeds": n}
        summary: dict = {"split": split, "n_seeds": n}

        for key in ordered_keys:
            vals = [d[key] for d in ms if key in d and isinstance(d[key], (int, float))]
            agg  = _aggregate(vals)
            row[f"{key}_mean"] = agg["mean"]
            row[f"{key}_std"]  = agg["std"]
            summary[key]       = agg

        rows.append(row)
        summaries[split] = summary

    df = pd.DataFrame(rows)
    return df, summaries


# ── CV helper ─────────────────────────────────────────────────────────────────

def _average_dicts(dicts: list[dict]) -> dict:
    """
    NaN-safe per-key mean across a list of metric dicts.

    Skips _SKIP_KEYS and non-numeric values.
    Returns an empty dict if dicts is empty.
    """
    if not dicts:
        return {}
    all_keys = {
        k for d in dicts for k, v in d.items()
        if k not in _SKIP_KEYS and isinstance(v, (int, float))
    }
    result: dict[str, float] = {}
    for key in all_keys:
        vals  = [d[key] for d in dicts if key in d and isinstance(d[key], (int, float))]
        clean = [v for v in vals if math.isfinite(v)]
        result[key] = float(np.mean(clean)) if clean else float("nan")
    return result


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_run_metrics(
    outputs_dir: Path,
    splits: list[str] = _SPLITS,
    seeds: list[int]  = _SEEDS,
    source: str       = "test",
) -> dict[str, list[dict]]:
    """
    Scan outputs_dir/{split}/seed_{seed}/metrics.json and load the `source` key.

    Missing files are silently skipped (contributes to a lower n_seeds count).
    Returns {"ldo": [test_dict_seed0, test_dict_seed1, ...], ...}.
    """
    result: dict[str, list[dict]] = {s: [] for s in splits}

    for split in splits:
        for seed in seeds:
            path = outputs_dir / split / f"seed_{seed}" / "metrics.json"
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
                metrics = data.get(source)
                if isinstance(metrics, dict):
                    result[split].append(metrics)
            except Exception:
                pass  # corrupt file → skip

    return result


def load_cv_metrics(
    outputs_dir: Path,
    splits: list[str] = _SPLITS,
    seeds: list[int]  = _SEEDS,
    folds: list[int]  = _FOLDS,
    source: str       = "test",
) -> dict[str, list[dict]]:
    """
    Scan outputs_dir/{split}/seed_{seed}/fold_{fold}/metrics.json.

    For each (split, seed), averages metrics across available folds → one dict.
    Seeds with zero valid folds are skipped (not counted in n_seeds).
    Returns {"ldo": [seed0_avg_dict, seed1_avg_dict, ...], ...}.
    """
    result: dict[str, list[dict]] = {s: [] for s in splits}

    for split in splits:
        for seed in seeds:
            fold_dicts: list[dict] = []
            for fold in folds:
                path = outputs_dir / split / f"seed_{seed}" / f"fold_{fold}" / "metrics.json"
                if not path.exists():
                    continue
                try:
                    data = json.loads(path.read_text())
                    metrics = data.get(source)
                    if isinstance(metrics, dict):
                        fold_dicts.append(metrics)
                except Exception:
                    pass
            if fold_dicts:
                result[split].append(_average_dicts(fold_dicts))

    return result


# ── Main entry points ─────────────────────────────────────────────────────────

def run_summarize(
    outputs_dir: Path,
    results_dir: Path,
    splits: list[str] = _SPLITS,
    seeds: list[int]  = _SEEDS,
    source: str       = "test",
) -> tuple[pd.DataFrame, dict]:
    """
    Full summarize pipeline.  Returns (df, summaries) and writes output files.

    Parameters
    ----------
    outputs_dir : root directory containing {split}/seed_{seed}/metrics.json
    results_dir : where to write full_comparison.csv and {split}_summary.json
    splits      : splits to include (default: all 4)
    seeds       : seed indices to scan (default: 0, 1, 2)
    source      : "test" or "val" — which metrics dict to aggregate
    """
    metrics_by_split = load_run_metrics(outputs_dir, splits, seeds, source)
    df, summaries    = compute_summary(metrics_by_split, splits)

    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "full_comparison.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")

    for split, summary in summaries.items():
        json_path = results_dir / f"{split}_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

    _print_summary(csv_path, summaries, splits)
    return df, summaries


def run_summarize_cv(
    outputs_dir: Path,
    results_dir: Path,
    splits: list[str] = _SPLITS,
    seeds: list[int]  = _SEEDS,
    folds: list[int]  = _FOLDS,
    source: str       = "test",
) -> tuple[pd.DataFrame, dict]:
    """
    CV summarize pipeline: folds → seed average → mean±std across seeds.

    For each (split, seed): averages across available folds first.
    Then aggregates mean±std across seeds — same output shape as run_summarize.

    Writes cv_full_comparison.csv and cv_{split}_summary.json.
    """
    metrics_by_split = load_cv_metrics(outputs_dir, splits, seeds, folds, source)
    df, summaries    = compute_summary(metrics_by_split, splits)

    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "cv_full_comparison.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")

    for split, summary in summaries.items():
        json_path = results_dir / f"cv_{split}_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

    _print_summary(csv_path, summaries, splits)
    return df, summaries


def _print_summary(csv_path: Path, summaries: dict, splits: list[str]) -> None:
    print(f"Saved {csv_path}")
    for split in splits:
        print(f"  {split}: n_seeds={summaries[split]['n_seeds']}", end="")
        auroc = summaries[split].get("auroc", {}).get("mean", float("nan"))
        if math.isfinite(auroc):
            print(f"  auroc={auroc:.4f}", end="")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize UniMoS training runs")
    p.add_argument("--outputs-dir", type=Path,
                   default=_REPO / "checkpoints")
    p.add_argument("--results-dir", type=Path,
                   default=_REPO / "results")
    p.add_argument("--seeds",  nargs="+", type=int, default=_SEEDS)
    p.add_argument("--splits", nargs="+", default=_SPLITS,
                   choices=_SPLITS + ["all"],
                   help="Splits to include (default: all 4)")
    p.add_argument("--source", default="test", choices=["test", "val"])
    p.add_argument("--cv",    action="store_true",
                   help="CV mode: reads fold_{k} subdirectories and averages across folds first.")
    p.add_argument("--folds", nargs="+", type=int, default=_FOLDS,
                   help="Fold indices to include in CV mode (default: 0 1 2 3 4).")
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    splits = _SPLITS if "all" in args.splits else args.splits
    if args.cv:
        run_summarize_cv(
            outputs_dir = args.outputs_dir,
            results_dir = args.results_dir,
            splits      = splits,
            seeds       = args.seeds,
            folds       = args.folds,
            source      = args.source,
        )
    else:
        run_summarize(
            outputs_dir = args.outputs_dir,
            results_dir = args.results_dir,
            splits      = splits,
            seeds       = args.seeds,
            source      = args.source,
        )


if __name__ == "__main__":
    main()
