# Data layout

Training reads from `train_data/` by default. Override with `--proc-dir` and `--func-dir` if needed.

This repository does not ship the files below.

## Expected files

| Path | Contents |
| --- | --- |
| `train_data/pairs/unimos_stratified_dataset.parquet` | Drug-pair table used to build the splits |
| `train_data/processed/cell_fn_vectors.npy` | Cell function-node activity |
| `train_data/processed/cell_mut_fn_vectors.npy` | Cell mutation burden in function-node space |
| `train_data/processed/cell_raw_expr.npy` | Cell expression matrix |
| `train_data/processed/drug_morgan_fps.npy` | Morgan fingerprints |
| `train_data/processed/cell_feature_index.json` | Cell name → row |
| `train_data/processed/drug_morgan_index.json` | InChIKey → row |
| `train_data/processed/cell_fm_emb.npy` | scFoundation cell embeddings (virtual-cell track) |
| `train_data/function_nodes/` | One function-node `.npy` per drug |

Official configs that set `struct_encoder: gnn` also need `data/processed/drug_graphs.pt` (or the path in `graph_cache`).

## Virtual-cell embeddings

```bash
python -m unimos.data.precompute_fm_emb \
    --depmap-csv data/Depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv \
    --scfoundation-dir data/external/scfoundation/official \
    --ckpt data/external/scfoundation/hf_mirror/models.ckpt \
    --cell-feature-index data/processed/cell_feature_index.json \
    --output data/processed/cell_fm_emb.npy
```

Copy or symlink the output into `train_data/processed/` before training with `use_virtual_cell: true`.
