"""
features.py — Vectorised feature extraction for Phase-0 sanity baselines.

Reuses the exact same UniMoSDataset (same fold-aware HVG selection, same
cell z-score norm stats fit on train only, same row filtering) that the real
UniMoS model trains on, so the sanity baselines see an identical row set /
identical potential-leakage surface. Extraction is vectorised (bulk numpy
indexing on the dataset's internal arrays) instead of looping __getitem__,
since RF/XGBoost need bulk (N, D) matrices anyway.

Feature block used by the RF / XGBoost baselines: cat(pA, pB, c_fn, c_mut_fn)
— the same 4*67=268-dim input the UniMoS core (interpretable) track consumes.
This keeps the "must beat" comparison apples-to-apples: if RF/XGBoost on the
*same inputs* matches or beats UniMoS, the architecture is not earning its
complexity from those inputs.
"""

from __future__ import annotations

import numpy as np

from unimos.data.dataset import UniMoSDataset


def extract_bulk(ds: UniMoSDataset) -> dict[str, np.ndarray]:
    """
    Vectorised extraction of drug/cell identity + kernel-equivalent features
    from a UniMoSDataset split.

    Returns
    -------
    dict with:
      ik_A, ik_B   : (N,) object arrays of InChIKeys
      cell_id      : (N,) object array
      pair_key     : (N,) object array, order-independent "ikA|ikB" pair key
      pA, pB       : (N, 67) drug function vectors
      c_fn, c_mut_fn : (N, 67) cell features (already z-scored per ds's norm_stats)
      y            : (N,) float32, Loewe>10 label (NaN rows already dropped)
    """
    rows = ds.rows
    ik_A = rows["drug1_inchikey"].to_numpy()
    ik_B = rows["drug2_inchikey"].to_numpy()
    cell_id = rows["cell_feature_id"].to_numpy()

    n = len(rows)
    pA = np.zeros((n, ds.drug_fn.shape[1]), dtype=np.float32)
    pB = np.zeros((n, ds.drug_fn.shape[1]), dtype=np.float32)
    for i, (a, b) in enumerate(zip(ik_A, ik_B)):
        ia = ds.drug_fn_idx.get(a)
        ib = ds.drug_fn_idx.get(b)
        if ia is not None:
            pA[i] = ds.drug_fn[ia]
        if ib is not None:
            pB[i] = ds.drug_fn[ib]

    cell_rows = np.array([ds.cell_to_row[c] for c in cell_id], dtype=np.int64)
    c_fn = (ds.c_fn[cell_rows] - ds._c_fn_mean) / ds._c_fn_std
    c_mut_fn = (ds.c_mut_fn[cell_rows] - ds._c_mut_mean) / ds._c_mut_std

    y = rows["label_loewe_gt_10"].to_numpy(dtype=np.float32)
    loewe = rows["loewe"].to_numpy(dtype=np.float32) if "loewe" in rows.columns else np.full(n, np.nan, dtype=np.float32)

    pair_key = np.array(
        ["|".join(sorted((a, b))) for a, b in zip(ik_A, ik_B)], dtype=object
    )

    return {
        "ik_A": ik_A,
        "ik_B": ik_B,
        "cell_id": cell_id,
        "pair_key": pair_key,
        "pA": pA.astype(np.float32),
        "pB": pB.astype(np.float32),
        "c_fn": c_fn.astype(np.float32),
        "c_mut_fn": c_mut_fn.astype(np.float32),
        "y": y,
        "loewe": loewe,
    }


def kernel_equivalent_matrix(feat: dict[str, np.ndarray]) -> np.ndarray:
    """cat(pA, pB, c_fn, c_mut_fn) -> (N, 268) for RF / XGBoost."""
    return np.concatenate(
        [feat["pA"], feat["pB"], feat["c_fn"], feat["c_mut_fn"]], axis=1
    )
