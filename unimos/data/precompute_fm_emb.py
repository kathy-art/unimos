"""
precompute_fm_emb.py — Phase 1 / entrance-one offline derivation (design doc
§2.1 / §9.1): DepMap bulk expression -> scFoundation cell embedding, looked
up by cell_feature_id at train time (frozen; see §2.1 "落地建议").

*** This is a GPU job. Per CLAUDE.md discipline it is written here but must
be *run* by the user on their own GPU allocation, not executed automatically. ***

Pipeline (mirrors the official model/get_embedding.py DeepCDR bulk-cell-line
recipe exactly, chosen because it's the closest official precedent to our
use case — bulk expression -> cell embedding -> drug-response-style task):

    --input_type bulk --output_type cell --pool_type max
    --tgthighres f1 --pre_normalized T --version ce

`pre_normalized=T` is correct here (not the default F) because our DepMap
source file is already log1p(TPM) — exactly the "already normalized" bulk
input the official recipe's T branch expects (same as DeepCDR's own
`normalized19264.npy`), so we skip the internal normalize_total+log1p step.

Output: cell_fm_emb.npy, shape (168, 4*hidden_dim) — pool_type=max here uses
the model's 4-way concat pooling (last two summary tokens + max-pool +
mean-pool over the rest, see get_embedding.py `output_type=='cell'` branch);
row order matches data/processed/cell_feature_index.json exactly (same
convention as cell_fn_vectors.npy / cell_mut_fn_vectors.npy), so it can be
indexed with the same `cell_to_row` dict UniMoSDataset already uses — no new
index file needed.

Usage
-----
python -m unimos.data.precompute_fm_emb \
    --depmap-csv data/Depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv \
    --scfoundation-dir data/external/scfoundation/official \
    --ckpt data/external/scfoundation/hf_mirror/models.ckpt \
    --cell-feature-index data/processed/cell_feature_index.json \
    --output data/processed/cell_fm_emb.npy
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _load_scfoundation(scfoundation_dir: Path, ckpt_path: Path):
    """Import the official scFoundation inference code (vendored, read-only,
    see data/external/scfoundation/official/) and load the 'cell' sub-model.
    """
    sys.path.insert(0, str(scfoundation_dir))
    from load import load_model_frommmf, gatherData  # noqa: E402  (official code, unmodified)

    model, config = load_model_frommmf(str(ckpt_path), key="cell")
    model.eval()
    return model, config, gatherData


def _gene_list(scfoundation_dir: Path) -> list[str]:
    df = pd.read_csv(scfoundation_dir / "OS_scRNA_gene_index.19264.tsv", sep="\t")
    return list(df["gene_name"])


def _load_depmap_bulk(depmap_csv: Path, cell_ids: list[str]) -> pd.DataFrame:
    """Load DepMap bulk expression for the given ModelIDs (= cell_feature_id),
    keeping only the default profile per model, gene columns stripped of the
    trailing " (entrez)" suffix down to the bare gene symbol.
    """
    df = pd.read_csv(depmap_csv)
    df = df[df["IsDefaultEntryForModel"] == "Yes"]
    df = df[df["ModelID"].isin(cell_ids)].set_index("ModelID")
    df = df.reindex(cell_ids)  # preserve cell_feature_index.json row order; NaN if missing
    missing = df.index[df.isna().all(axis=1)].tolist()
    if missing:
        raise ValueError(
            f"{len(missing)} cell_feature_id(s) have no DepMap bulk expression row: {missing}"
        )
    meta_cols = [c for c in df.columns if not c.endswith(")")]
    gene_cols = [c for c in df.columns if c.endswith(")")]
    df = df[gene_cols]
    df.columns = [c.split(" (")[0] for c in gene_cols]
    return df


def _main_gene_selection(X_df: pd.DataFrame, gene_list: list[str]) -> pd.DataFrame:
    """Official main_gene_selection (load.py) — pad missing genes with 0,
    reorder to the model's 19264-gene vocabulary."""
    to_fill = list(set(gene_list) - set(X_df.columns))
    if to_fill:
        pad = pd.DataFrame(0.0, index=X_df.index, columns=to_fill)
        X_df = pd.concat([X_df, pad], axis=1)
    return X_df[gene_list]


def compute_embeddings(
    depmap_csv: Path,
    scfoundation_dir: Path,
    ckpt_path: Path,
    cell_feature_index: Path,
    tgthighres: str = "f1",
    device: str = "cuda",
    use_amp: bool = False,
) -> tuple[np.ndarray, list[str]]:
    cell_idx = json.load(open(cell_feature_index))
    cell_ids = [c for c, _ in sorted(cell_idx.items(), key=lambda kv: kv[1])]

    gene_list = _gene_list(scfoundation_dir)
    expr = _load_depmap_bulk(depmap_csv, cell_ids)
    expr = _main_gene_selection(expr, gene_list)
    assert expr.shape[1] == len(gene_list) == 19264

    model, config, gatherData = _load_scfoundation(scfoundation_dir, ckpt_path)
    model = model.to(device)

    # Bulk expression is far denser than the sparse single-cell data this
    # recipe is usually run on (~85% nonzero genes here vs. typically far
    # fewer in scRNA-seq), so the effective token sequence length into the
    # encoder is ~16k+ — one layer's quadratic attention matrix alone is
    # ~12GB in fp32 (confirmed: OOM size matches 16264^2 * 12 heads * 4B),
    # too big for a 24GB GPU regardless of which GPU. bfloat16 (not fp16):
    # same exponent range as fp32, so it doesn't have fp16's overflow/NaN
    # failure mode on this input's unnormalized activation magnitudes,
    # while still halving activation memory.
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp and device.startswith("cuda")
        else contextlib.nullcontext()
    )

    embs = []
    with torch.no_grad(), autocast_ctx:
        for i in range(expr.shape[0]):
            # pre_normalized == "T": bulk row is already log1p(TPM) — use as-is,
            # totalcount = sum of the (already-normalized) values, no extra log10.
            tmpdata = expr.iloc[i, :].tolist()
            totalcount = expr.iloc[i, :].sum()
            assert tgthighres[0] == "f", "only fold-change resolution supported here"
            pretrain_gene_x = torch.tensor(
                tmpdata + [float(totalcount) * float(tgthighres[1:]), float(totalcount)]
            ).unsqueeze(0).to(device)
            data_gene_ids = torch.arange(
                19266, device=pretrain_gene_x.device
            ).repeat(pretrain_gene_x.shape[0], 1)

            value_labels = pretrain_gene_x > 0
            x, x_padding = gatherData(pretrain_gene_x, value_labels, config["pad_token_id"])
            position_gene_ids, _ = gatherData(data_gene_ids, value_labels, config["pad_token_id"])

            x = model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
            x = x + model.pos_emb(position_gene_ids)
            geneemb = model.encoder(x, x_padding)

            # pool_type=max (DeepCDR precedent): single max-pool over all tokens
            geneembmerge, _ = torch.max(geneemb, dim=1)
            embs.append(geneembmerge.float().detach().cpu().numpy())

            del x, geneemb, geneembmerge, pretrain_gene_x, position_gene_ids, data_gene_ids
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    return np.concatenate(embs, axis=0), cell_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depmap-csv", type=Path, required=True)
    ap.add_argument("--scfoundation-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cell-feature-index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tgthighres", type=str, default="f1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true",
                     help="enable bfloat16 autocast to roughly halve activation "
                          "memory — needed here: one encoder layer's attention "
                          "matrix at this input's ~16k effective sequence length "
                          "is ~12GB in fp32, too big for a single 24GB GPU "
                          "regardless of which GPU (confirmed: same OOM size on "
                          "an otherwise-idle GPU). bf16, not fp16 — fp16 was "
                          "observed to produce all-NaN output on this dense, "
                          "unnormalized bulk-expression input; bf16 shares fp32's "
                          "exponent range and doesn't have that overflow mode.")
    args = ap.parse_args()

    emb, cell_ids = compute_embeddings(
        args.depmap_csv, args.scfoundation_dir, args.ckpt, args.cell_feature_index,
        tgthighres=args.tgthighres, device=args.device, use_amp=args.amp,
    )
    print(f"cell_fm_emb shape: {emb.shape} for {len(cell_ids)} cells (row order == cell_feature_index.json)")
    n_nan_rows = int(np.isnan(emb).any(axis=1).sum())
    if n_nan_rows:
        raise RuntimeError(
            f"{n_nan_rows}/{emb.shape[0]} rows are NaN — not saving. This was "
            f"observed with --amp (fp16 overflow on this dense bulk-expression "
            f"input); retry without --amp, or on a GPU with more free VRAM."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, emb.astype(np.float32))
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
