"""
splits.py — Train/val/test split builders for UniMoS.

Single-split strategies (build_splits):
  LDO (Leave-Drug-Out):  test drugs never seen in train
  LCO (Leave-Cell-Out):  test cells never seen in train
  LPO (Leave-Pair-Out):  test drug-pairs never seen in train (single drugs may appear)
  Random:                row-level shuffle
  Split sizes: 80 / 10 / 10 (train / val / test).

Cross-validation (build_cv_splits):
  Same four strategies, partitioned into n_folds (default 5) non-overlapping folds.
  For fold k: test=fold_k, val=fold_(k+1)%n_folds, train=remaining folds.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
import pandas as pd

SplitType = Literal["ldo", "lco", "lpo", "random"]


def _greedy_ldo_drug_partition(
    df: pd.DataFrame,
    drug1_col: str,
    drug2_col: str,
    seed: int,
    val_frac: float,
    test_frac: float,
) -> tuple[set, set]:
    """Greedily assign drugs to test/val so the resulting ROW-level split
    actually lands near (val_frac, test_frac) — plain per-drug sampling
    (whether by permutation or hash bucket) systematically overshoots test/val
    row share under the OR condition (a row is held out if EITHER drug is
    held out, so a drug pair has two independent chances to be excluded from
    train; e.g. sampling 10% of drugs by count yields ~19% of rows).

    Processes drugs in a fully seeded-random order and greedily assigns each
    to test, then val, then train — whichever bucket still has budget (with a
    10% overshoot tolerance) for the NEW rows this drug would add (rows not
    already covered by a previously-assigned drug in that bucket). Hub drugs
    naturally end up in train: adding one anywhere else would blow past the
    tolerance in a single step, so they only fail both checks and fall
    through. An earlier version processed drugs lowest-degree-first instead of
    fully at random, on the theory that this made quota control easier — it
    does, but at a steep cost: with ~1887 drugs where 722 have degree 1
    (appear in exactly one row) and the row quota requires climbing well past
    that, the low-degree tiers are consumed almost to exhaustion regardless of
    tie-break order, so the resulting test-drug set was ~99.9% IDENTICAL
    across different seeds (measured directly) — seeds stopped being
    meaningfully different splits. Fully-random order with a tolerance check
    fixes this (~1% overlap between seeds' test-drug sets, measured) while
    still landing within ~3 percentage points of (val_frac, test_frac) at the
    row level, because hub drugs are excluded by the tolerance check
    regardless of where they land in the random order.

    Trades away the hash-bucket's growth-invariance: assignment depends on the
    full drug-degree distribution, so appending new drug-pair rows can shift
    existing drugs across buckets. Use ldo_balance_mode="hash" instead if
    stability under dataset growth matters more than hitting 80/10/10 tightly.
    """
    rng = np.random.default_rng(seed)
    d1 = df[drug1_col].to_numpy()
    d2 = df[drug2_col].to_numpy()
    drug_rows: dict = {}
    for i in range(len(df)):
        a, b = d1[i], d2[i]
        if isinstance(a, str):
            drug_rows.setdefault(a, set()).add(i)
        if isinstance(b, str):
            drug_rows.setdefault(b, set()).add(i)

    drugs = list(drug_rows.keys())
    order = rng.permutation(np.array(drugs, dtype=object))

    n = len(df)
    target_test = test_frac * n
    target_val = val_frac * n
    overshoot_tol = 1.1
    test_covered: set = set()
    val_covered: set = set()
    test_drugs: set = set()
    val_drugs: set = set()

    for d in order:
        rows = drug_rows[d]
        new_test = rows - test_covered
        if len(test_covered) < target_test and len(test_covered) + len(new_test) <= target_test * overshoot_tol:
            test_drugs.add(d)
            test_covered |= new_test
            continue
        new_val = rows - test_covered - val_covered
        if len(val_covered) < target_val and len(val_covered) + len(new_val) <= target_val * overshoot_tol:
            val_drugs.add(d)
            val_covered |= new_val
        # else: leave for train (implicit — not added to either set)

    return test_drugs, val_drugs


def _stable_unit_interval(key: str) -> float:
    """Deterministic float in [0, 1) from a string key.

    Stable across Python versions/processes/machines — unlike the builtin
    hash(), which is salted per-process unless PYTHONHASHSEED is fixed.
    Used so a drug's LDO bucket depends only on (seed, drug_id), never on
    which other drugs happen to be present in the current dataframe.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _train_val_test_split(
    indices: np.ndarray,
    rng: np.random.Generator,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an index array into train / val / test with the given fractions."""
    idx = rng.permutation(indices)
    n = len(idx)
    n_test = int(np.ceil(n * test_frac))
    n_val = int(np.ceil(n * val_frac))
    return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]


def build_splits(
    df: pd.DataFrame,
    split_type: SplitType,
    seed: int = 0,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    drug1_col: str = "drug1_inchikey",
    drug2_col: str = "drug2_inchikey",
    cell_col: str = "cell_feature_id",
    ldo_strict: bool = False,
    ldo_balance_mode: Literal["greedy", "hash", "hash_prune", "legacy_permutation"] = "greedy",
) -> dict[str, np.ndarray]:
    """
    Return {"train": idx, "val": idx, "test": idx} — integer positions into df.

    Parameters
    ----------
    df : full training DataFrame
    split_type : one of ldo / lco / lpo / random
    seed : random seed (0 / 1 / 2)
    val_frac, test_frac : proportions for val / test sets (row count)
    ldo_strict : only used when split_type == "ldo". Default (False) reproduces the
        existing reference LDO split: a row is held out if EITHER drug is a test
        drug (OR condition), i.e. "at least one drug unseen". Set True for the
        stricter "both drugs unseen" (AND condition) variant referenced in
        ABLATION_PLAN.md §2.1 item 5 / L-5. This is strictly additive — default
        behaviour and all existing checkpoints/splits are unaffected.
    ldo_balance_mode : only used when split_type == "ldo". How drugs are assigned
        to train/val/test:
          "greedy" (default) — per-drug greedy assignment that actually lands
              near (val_frac, test_frac) at the ROW level (see
              _greedy_ldo_drug_partition); fixes the long-standing LDO
              imbalance (plain per-drug sampling overshoots test/val row share
              under the OR condition — two chances per pair to be excluded).
              Depends on the full drug-degree distribution, so is NOT stable
              under dataset growth: appending rows can reshuffle existing
              drugs across buckets.
          "hash" — per-drug stable hash bucket (see _stable_unit_interval): a
              drug's assignment depends only on (seed, drug_id), so appending
              new drug-pair rows never reshuffles already-assigned drugs under
              the same seed. Does NOT fix the row-level imbalance.
          "hash_prune" — same per-drug hash assignment as "hash" (no greedy
              per-drug quota tuning at all), then if the resulting val/test
              row counts overshoot (val_frac, test_frac) — which they will,
              structurally, under the OR condition — randomly DROP the excess
              rows from val/test until each hits its target count. Dropped
              rows are excluded entirely (not moved to train, since their
              drugs are still meant to be held out); only val/test shrink,
              train is untouched. Simpler than "greedy" (no per-drug quota
              bookkeeping) at the cost of discarding some drug-pairs outright.
          "legacy_permutation" — original behaviour: permutation over
              range(len(unique_drugs)). Same seed gives a DIFFERENT split if
              the drug universe size changes. This is what main's cached
              splits/*.npz files were built with; regenerating them requires
              this mode, or (preferably) loading the .npz directly via
              --split-npz.
    """
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(df))

    if split_type == "random":
        train, val, test = _train_val_test_split(all_idx, rng, val_frac, test_frac)

    elif split_type == "ldo":
        # Hold out drugs (test entity = union of drug1 and drug2 InChIKeys)
        # Sort before applying the seeded permutation.  Iteration order of a
        # Python set depends on hash randomisation, so an unsorted set makes a
        # nominally seeded split change across processes.
        if ldo_balance_mode == "greedy":
            test_drugs, val_drugs = _greedy_ldo_drug_partition(
                df, drug1_col, drug2_col, seed, val_frac, test_frac
            )
        else:
            unique_drugs = np.array(
                sorted(set(df[drug1_col].dropna()) | set(df[drug2_col].dropna()))
            )
            if ldo_balance_mode == "legacy_permutation":
                n_test = max(1, int(np.ceil(len(unique_drugs) * test_frac)))
                n_val = max(1, int(np.ceil(len(unique_drugs) * val_frac)))
                perm = rng.permutation(len(unique_drugs))
                test_drugs = set(unique_drugs[perm[:n_test]])
                val_drugs = set(unique_drugs[perm[n_test:n_test + n_val]])
            elif ldo_balance_mode in ("hash", "hash_prune"):
                buckets = {d: _stable_unit_interval(f"{seed}:{d}") for d in unique_drugs}
                test_drugs = {d for d, b in buckets.items() if b < test_frac}
                val_drugs = {
                    d for d, b in buckets.items() if test_frac <= b < test_frac + val_frac
                }
            else:
                raise ValueError(f"unknown ldo_balance_mode: {ldo_balance_mode!r}")

        if ldo_strict:
            is_test = df[drug1_col].isin(test_drugs) & df[drug2_col].isin(test_drugs)
            is_val = (~is_test) & (df[drug1_col].isin(val_drugs) & df[drug2_col].isin(val_drugs))
        else:
            is_test = df[drug1_col].isin(test_drugs) | df[drug2_col].isin(test_drugs)
            is_val = (~is_test) & (df[drug1_col].isin(val_drugs) | df[drug2_col].isin(val_drugs))
        is_train = ~is_test & ~is_val

        train = all_idx[is_train.values]
        val = all_idx[is_val.values]
        test = all_idx[is_test.values]

        if ldo_balance_mode == "hash_prune":
            target_val = int(round(val_frac * len(df)))
            target_test = int(round(test_frac * len(df)))
            if len(val) > target_val:
                keep = rng.choice(len(val), size=target_val, replace=False)
                val = np.sort(val[keep])
            if len(test) > target_test:
                keep = rng.choice(len(test), size=target_test, replace=False)
                test = np.sort(test[keep])
            # Dropped rows are excluded outright — not returned to train, since
            # their drugs are still meant to be held out of train.
            return {"train": train, "val": val, "test": test}

    elif split_type == "lco":
        unique_cells = np.array(df[cell_col].unique())
        n_test = max(1, int(np.ceil(len(unique_cells) * test_frac)))
        n_val = max(1, int(np.ceil(len(unique_cells) * val_frac)))
        perm = rng.permutation(len(unique_cells))
        test_cells = set(unique_cells[perm[:n_test]])
        val_cells = set(unique_cells[perm[n_test:n_test + n_val]])

        is_test = df[cell_col].isin(test_cells)
        is_val = (~is_test) & df[cell_col].isin(val_cells)
        is_train = ~is_test & ~is_val

        train = all_idx[is_train.values]
        val = all_idx[is_val.values]
        test = all_idx[is_test.values]

    elif split_type == "lpo":
        # Entity = frozenset({drug1, drug2}) — unordered pair
        pairs = np.array([
            frozenset({d1, d2})
            for d1, d2 in zip(df[drug1_col], df[drug2_col])
        ])
        unique_pairs = np.array(list(set(pairs)))
        n_test = max(1, int(np.ceil(len(unique_pairs) * test_frac)))
        n_val = max(1, int(np.ceil(len(unique_pairs) * val_frac)))
        perm = rng.permutation(len(unique_pairs))
        test_pairs = set(unique_pairs[perm[:n_test]])
        val_pairs = set(unique_pairs[perm[n_test:n_test + n_val]])

        is_test = np.array([p in test_pairs for p in pairs])
        is_val = np.array([(not t) and (p in val_pairs) for p, t in zip(pairs, is_test)])
        is_train = ~is_test & ~is_val

        train = all_idx[is_train]
        val = all_idx[is_val]
        test = all_idx[is_test]

    else:
        raise ValueError(f"Unknown split_type: {split_type!r}")

    assert len(train) + len(val) + len(test) == len(df), (
        f"Split size mismatch: {len(train)}+{len(val)}+{len(test)} != {len(df)}"
    )
    return {"train": train, "val": val, "test": test}


# ── Cross-validation ──────────────────────────────────────────────────────────

def _assign_folds(
    entities: np.ndarray,
    n_folds: int,
    rng: np.random.Generator,
) -> list[set]:
    """Randomly partition entities into n_folds balanced, non-overlapping groups."""
    perm = rng.permutation(len(entities))
    return [set(entities[perm[i::n_folds]]) for i in range(n_folds)]


def build_cv_splits(
    df: pd.DataFrame,
    split_type: SplitType,
    n_folds: int = 5,
    seed: int = 0,
    drug1_col: str = "drug1_inchikey",
    drug2_col: str = "drug2_inchikey",
    cell_col: str = "cell_feature_id",
) -> list[dict[str, np.ndarray]]:
    """
    Build n_folds cross-validation splits.

    Returns a list of n_folds dicts, each with {"train", "val", "test"}.
    For fold k:
      test  = entities in fold k
      val   = entities in fold (k+1) % n_folds   (never overlaps test)
      train = entities in all remaining folds

    Uses the same entity-level strategy as build_splits:
      random → row-level folds
      ldo    → drug-level folds (unique drug identity)
      lco    → cell-level folds
      lpo    → drug-pair-level folds (unordered pairs)
    """
    rng     = np.random.default_rng(seed)
    all_idx = np.arange(len(df))
    result: list[dict[str, np.ndarray]] = []

    if split_type == "random":
        row_folds = _assign_folds(all_idx, n_folds, rng)
        for k in range(n_folds):
            test_set  = row_folds[k]
            val_set   = row_folds[(k + 1) % n_folds]
            train_mask = np.ones(len(df), dtype=bool)
            for idx in test_set:
                train_mask[idx] = False
            for idx in val_set:
                train_mask[idx] = False
            result.append({
                "train": all_idx[train_mask],
                "val":   np.fromiter(sorted(val_set),  dtype=int),
                "test":  np.fromiter(sorted(test_set), dtype=int),
            })

    elif split_type == "ldo":
        unique_drugs = np.array(
            sorted(set(df[drug1_col].dropna()) | set(df[drug2_col].dropna()))
        )
        drug_folds = _assign_folds(unique_drugs, n_folds, rng)
        for k in range(n_folds):
            test_drugs = drug_folds[k]
            val_drugs  = drug_folds[(k + 1) % n_folds]
            is_test  = (df[drug1_col].isin(test_drugs) | df[drug2_col].isin(test_drugs)).values
            is_val   = (~is_test) & (df[drug1_col].isin(val_drugs) | df[drug2_col].isin(val_drugs)).values
            is_train = ~is_test & ~is_val
            result.append({
                "train": all_idx[is_train],
                "val":   all_idx[is_val],
                "test":  all_idx[is_test],
            })

    elif split_type == "lco":
        unique_cells = np.array(sorted(df[cell_col].dropna().unique()))
        cell_folds = _assign_folds(unique_cells, n_folds, rng)
        for k in range(n_folds):
            test_cells = cell_folds[k]
            val_cells  = cell_folds[(k + 1) % n_folds]
            is_test  = df[cell_col].isin(test_cells).values
            is_val   = (~is_test) & df[cell_col].isin(val_cells).values
            is_train = ~is_test & ~is_val
            result.append({
                "train": all_idx[is_train],
                "val":   all_idx[is_val],
                "test":  all_idx[is_test],
            })

    elif split_type == "lpo":
        pairs = np.array([
            frozenset({d1, d2})
            for d1, d2 in zip(df[drug1_col], df[drug2_col])
        ])
        unique_pairs = np.array(sorted(set(pairs), key=str))
        pair_folds = _assign_folds(unique_pairs, n_folds, rng)
        for k in range(n_folds):
            test_pairs = pair_folds[k]
            val_pairs  = pair_folds[(k + 1) % n_folds]
            is_test  = np.array([p in test_pairs for p in pairs])
            is_val   = np.array([(not t) and (p in val_pairs)
                                 for p, t in zip(pairs, is_test)])
            is_train = ~is_test & ~is_val
            result.append({
                "train": all_idx[is_train],
                "val":   all_idx[is_val],
                "test":  all_idx[is_test],
            })

    else:
        raise ValueError(f"Unknown split_type: {split_type!r}")

    for k, s in enumerate(result):
        total = len(s["train"]) + len(s["val"]) + len(s["test"])
        assert total == len(df), f"CV fold {k} size mismatch: {total} != {len(df)}"

    return result
