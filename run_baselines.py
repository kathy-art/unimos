"""
run_baselines.py — Phase-0 evaluation harness entry point.

Builds the same (split_type, seed) train/val/test partitions that
`train.py` uses (via `unimos.data.splits.build_splits`, identical defaults:
val_frac=test_frac=0.10, ldo_strict=False), fits each §5.1 sanity baseline
on train, and reports test-set metrics (AUROC/AUPRC from
unimos.training.metrics + top-k/Brier/Spearman from unimos.eval.calibration)
alongside a pairing-permutation control (unimos.eval.permutation_control).

Usage
-----
python run_baselines.py --splits random ldo lco lpo \
    --baselines global_mean drug_mean cell_mean pair_drug_cell_global_mean one_hot random_forest xgboost \
    --output-dir checkpoints_baselines

Notes
-----
* Uses the current main pool as-is (drugcombv15_unimos_pair_study_both_pathway.parquet,
  O'Neil rows included) — matches the seeds/splits of the existing
  checkpoints/{split}/seed_*/metrics.json UniMoS runs for a like-for-like
  comparison. O'Neil-pool exclusion and a leave-study-out split type were
  both explicitly deferred to a later phase (see PHASE0_GATE.md).
* Only seeds that already have a UniMoS reference run under checkpoints/
  are used, so every baseline row has a matching "v2 现状模型" row to
  compare against in the summary table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from unimos.data.splits import build_splits
from unimos.data.dataset import UniMoSDataset
from unimos.eval.baselines import BASELINE_REGISTRY
from unimos.eval.calibration import phase0_metrics
from unimos.eval.features import extract_bulk
from unimos.eval.permutation_control import run_control
from unimos.training.metrics import compute_metrics, find_best_threshold

_REPO = Path(__file__).resolve().parent
_PAIR = _REPO / "data" / "Drugcombv15" / "drugcombv15_unimos_pair_study_both_pathway.parquet"
_PROC = _REPO / "data" / "processed"
_FUNC = _REPO / "data" / "Drugcombv15" / "function_nodes"
_CKPT = _REPO / "checkpoints"


def existing_seeds(split_type: str) -> list[int]:
    d = _CKPT / split_type
    if not d.is_dir():
        return []
    return sorted(int(p.name.split("_")[1]) for p in d.glob("seed_*") if (p / "metrics.json").exists())


def load_v2_metrics(split_type: str, seed: int) -> "dict | None":
    p = _CKPT / split_type / f"seed_{seed}" / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def full_metrics(prob: np.ndarray, y: np.ndarray, loewe: "np.ndarray | None" = None,
                  threshold: float = 0.5) -> dict:
    base = compute_metrics(prob, y, prob, y, prob, y, prob, threshold=threshold)
    base.update(phase0_metrics(prob, y, loewe_continuous=loewe))
    return base


def _metric_fn_factory(loewe: np.ndarray):
    def _fn(prob, y):
        return full_metrics(prob, y, loewe=loewe)
    return _fn


def run_one(split_type: str, seed: int, baseline_name: str, df: pd.DataFrame) -> dict:
    splits = build_splits(df, split_type, seed=seed)
    train_ds = UniMoSDataset(df, splits["train"], proc_dir=_PROC, func_dir=_FUNC)
    norm_stats = train_ds.norm_stats()
    val_ds = UniMoSDataset(df, splits["val"], proc_dir=_PROC, func_dir=_FUNC, cell_norm_stats=norm_stats)
    test_ds = UniMoSDataset(df, splits["test"], proc_dir=_PROC, func_dir=_FUNC, cell_norm_stats=norm_stats)

    train_feat = extract_bulk(train_ds)
    val_feat = extract_bulk(val_ds)
    test_feat = extract_bulk(test_ds)

    baseline = BASELINE_REGISTRY[baseline_name]()
    baseline.fit(train_feat)

    val_prob = baseline.predict_proba(val_feat)
    thr = find_best_threshold(val_prob, val_feat["y"])

    test_prob = baseline.predict_proba(test_feat)
    test_metrics = full_metrics(test_prob, test_feat["y"], loewe=test_feat.get("loewe"), threshold=thr)

    control = run_control(baseline, test_feat, _metric_fn_factory(test_feat.get("loewe")), seed=0)

    return {
        "split": split_type,
        "seed": seed,
        "baseline": baseline_name,
        "n_train": len(train_feat["y"]),
        "n_val": len(val_feat["y"]),
        "n_test": len(test_feat["y"]),
        "best_threshold": thr,
        "test": test_metrics,
        "permutation_control": control,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["random", "ldo", "lco", "lpo"])
    ap.add_argument("--baselines", nargs="+", default=list(BASELINE_REGISTRY))
    ap.add_argument("--output-dir", type=Path, default=_REPO / "checkpoints_baselines")
    ap.add_argument("--pair-parquet", type=Path, default=_PAIR)
    ap.add_argument("--max-seeds-per-split", type=int, default=None,
                     help="cap seeds per split (for a fast smoke run)")
    ap.add_argument("--force", action="store_true",
                     help="recompute even if a result JSON already exists")
    ap.add_argument("--restrict", nargs="*", default=[],
                     help="restrict seeds for a split, e.g. --restrict ldo:1,10,15")
    args = ap.parse_args()

    df = pd.read_parquet(args.pair_parquet)

    restrict = {}
    for r in args.restrict:
        split_name, seed_list = r.split(":")
        restrict[split_name] = [int(s) for s in seed_list.split(",")]

    rows = []
    for split_type in args.splits:
        seeds = existing_seeds(split_type)
        if split_type in restrict:
            requested = restrict[split_type]
            missing = [s for s in requested if s not in seeds]
            if missing:
                raise ValueError(
                    f"--restrict {split_type}: seeds {missing} have no existing "
                    f"checkpoints/{split_type}/seed_*/metrics.json (available: {seeds})"
                )
            seeds = requested
        if args.max_seeds_per_split:
            seeds = seeds[: args.max_seeds_per_split]
        if not seeds:
            print(f"[skip] no existing checkpoints/{split_type}/seed_* found — "
                  f"nothing to compare against, skipping split")
            continue
        for seed in seeds:
            for baseline_name in args.baselines:
                out_dir = args.output_dir / split_type / f"seed_{seed}"
                out_path = out_dir / f"{baseline_name}.json"
                if out_path.exists() and not args.force:
                    print(f"[skip] split={split_type} seed={seed} baseline={baseline_name} (already done)")
                    result = json.loads(out_path.read_text())
                else:
                    print(f"[run] split={split_type} seed={seed} baseline={baseline_name}")
                    result = run_one(split_type, seed, baseline_name, df)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(result, indent=2, default=float))

                v2 = load_v2_metrics(split_type, seed)
                rows.append({
                    "split": split_type, "seed": seed, "model": baseline_name,
                    "auroc": result["test"]["auroc"], "auprc": result["test"]["auprc"],
                    "brier": result["test"]["brier"],
                    "precision_at_100": result["test"].get("precision_at_100"),
                    "precision_at_top1pct": result["test"].get("precision_at_top1pct"),
                    "spearman_vs_loewe": result["test"].get("spearman_vs_loewe"),
                    "pairing_shuffle_auroc_delta": result["permutation_control"]["delta"].get("auroc"),
                })
                if v2 is not None:
                    rows.append({
                        "split": split_type, "seed": seed, "model": "unimos_v2",
                        "auroc": v2["test"]["auroc"], "auprc": v2["test"]["auprc"],
                        "brier": None, "precision_at_100": None,
                        "precision_at_top1pct": None, "spearman_vs_loewe": None,
                        "pairing_shuffle_auroc_delta": None,
                    })

    summary = pd.DataFrame(rows).drop_duplicates(subset=["split", "seed", "model"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "phase0_summary.csv", index=False)
    print(f"\nWrote {args.output_dir / 'phase0_summary.csv'}")
    print(summary.groupby(["split", "model"])[["auroc", "auprc"]].mean().round(4))


if __name__ == "__main__":
    main()
