#!/usr/bin/env python3
"""
predict_novel_pair.py — score a (drug A, drug B, cell) triple with UniMoS,
including novel drugs supplied by SMILES, ensembled across split-type models,
with mechanism-oriented interpretability readouts.

Pipeline per drug:
  * known drug  -> resolve by name/inchikey, look up Morgan FP + function vector
  * novel drug  -> --smiles-a "<SMILES>" [--nodes-a F005,F000]  ; FP computed via
                   RDKit Morgan(radius=2, fpSize=2048); function vector = the given
                   function-node bits (default: zeros).

Ensemble: for each requested checkpoint set, reconstruct the exact training
dataset (norm stats + fold-aware HVG) so cell features match how the model was
trained, then forward and average synergy probability.

Readouts: prob (+per-model), decision vs mean best_threshold, per-drug predicted
RI, alpha gate, core/resid logits, same_fn (diagonal same-community overlap) and
cross (off-diagonal cross-community / vertical-synergy interaction).

Example
-------
  python scripts/predict_novel_pair.py \
      --smiles-a "C(=O)[C@H](...)O" --name-a "Fucostatin I" --nodes-a F000,F005,F006 \
      --drug-b defactinib --cells AGS,SNU-16,"KATO III" \
      --ensembles ldo_heldout,lco_heldout,random
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from unimos.data.dataset import UniMoSDataset, select_hvg  # noqa: E402
from unimos.model.unimos import UniMoS  # noqa: E402

DATA = REPO / "train_data/pairs/unimos_stratified_dataset.parquet"
PROC = REPO / "train_data/processed"
FUNC = REPO / "train_data/function_nodes"
SPLITS = REPO / "splits"

# name -> (checkpoint dir, split_type, seeds, provenance dict)
# npz_tmpl: Path template for a frozen split .npz, or None to call build_splits().
ENSEMBLES = {
    "random": ("checkpoints/random/seed_{s}", "random", [0, 1, 2], dict(
        data=DATA, proc=PROC, func=FUNC, npz_tmpl=None,
    )),
    "lco_heldout": ("checkpoints/lco/seed_{s}", "lco", [34, 453, 876], dict(
        data=DATA, proc=PROC, func=FUNC,
        npz_tmpl=SPLITS / "leave_cell_out_seed{s}.npz",
    )),
    "ldo_heldout": ("checkpoints/ldo/seed_{s}", "ldo", [1, 10, 15], dict(
        data=DATA, proc=PROC, func=FUNC,
        npz_tmpl=SPLITS / "leave_drug_out_seed{s}.npz",
    )),
}


def morgan_fp(smiles: str) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return gen.GetFingerprintAsNumPy(mol).astype(np.float32)


def resolve_known(df: pd.DataFrame, token: str):
    """Return (inchikey, targets_uniprot) for a known drug given name or inchikey."""
    if token.endswith("-N") and len(token) >= 27:  # looks like an InChIKey
        m1 = df["drug1_inchikey"] == token
        if m1.any():
            return token, df[m1].iloc[0]["targets_drug1_uniprot"]
        m2 = df["drug2_inchikey"] == token
        return token, (df[m2].iloc[0]["targets_drug2_uniprot"] if m2.any() else [])
    t = token.lower()
    rl = df["drug_row_names_raw"].astype(str).str.lower()
    cl = df["drug_col_names_raw"].astype(str).str.lower()
    if (rl == t).any():
        r = df[rl == t].iloc[0]
        return r["drug1_inchikey"], r["targets_drug1_uniprot"]
    if (cl == t).any():
        r = df[cl == t].iloc[0]
        return r["drug2_inchikey"], r["targets_drug2_uniprot"]
    raise ValueError(f"drug '{token}' not found in dataset (supply --smiles for novel drugs)")


def resolve_cell(df: pd.DataFrame, token: str) -> str:
    if token in set(df["cell_feature_id"]):
        return token
    m = df["cell_name_canonical"].astype(str).str.upper() == token.upper()
    if m.any():
        return df[m].iloc[0]["cell_feature_id"]
    raise ValueError(f"cell '{token}' not found")


def function_vector_from_nodes(nodes: list[str], F: int = 67) -> np.ndarray:
    v = np.zeros(F, dtype=np.float32)
    for n in nodes:
        idx = int(n[1:]) if n.upper().startswith("F") else int(n)
        v[idx] = 1.0
    return v


def build_train_ds(df, split_type, seed, npz_tmpl, proc_dir=PROC, func_dir=FUNC):
    """npz_tmpl: Path template (str(path).format(s=seed)) to a split .npz, or
    None to reconstruct via build_splits() — only equivalent to the original
    training split for "random"; LCO/LDO MUST use their pinned npz_tmpl (see
    ENSEMBLES provenance notes) since build_splits() does not reproduce them."""
    if npz_tmpl is not None:
        npz = np.load(str(npz_tmpl).format(s=seed), allow_pickle=True)
        train_idx = npz["train_indices"].astype(int)
    else:
        from unimos.data.splits import build_splits
        train_idx = build_splits(df, split_type, seed=seed)["train"]
    ds = UniMoSDataset(df, train_idx, proc_dir=proc_dir, func_dir=func_dir)
    # fold-aware HVG only for LCO (mirrors train.build_loaders)
    hvg_ids = None
    if split_type == "lco":
        c_raw_all = np.load(proc_dir / "cell_raw_expr.npy")
        stored = json.load(open(proc_dir / "cell_hvg_gene_ids.json"))
        cell_idx = json.load(open(proc_dir / "cell_feature_index.json"))
        train_cells = [c for c in set(df.iloc[train_idx]["cell_feature_id"]) if c in cell_idx]
        rows = [cell_idx[c] for c in train_cells]
        expr = pd.DataFrame(c_raw_all[rows, :].T, index=stored, columns=train_cells)
        hvg_ids = select_hvg(expr, train_cells, n_top=150)
    return ds, hvg_ids


def cell_tensors(ds, hvg_ids, cell_id, proc_dir=PROC):
    stored = json.load(open(proc_dir / "cell_hvg_gene_ids.json"))
    cr = ds.cell_to_row[cell_id]
    c_fn = (ds.c_fn[cr] - ds._c_fn_mean) / ds._c_fn_std
    c_mut = (ds.c_mut_fn[cr] - ds._c_mut_mean) / ds._c_mut_std
    c_raw_full = np.load(proc_dir / "cell_raw_expr.npy")[cr]
    if hvg_ids is not None and hvg_ids != stored:
        pos = [stored.index(g) for g in hvg_ids if g in stored]
        c_raw = c_raw_full[pos]
    else:
        c_raw = ds.c_raw[cr]
    return c_fn, c_mut, c_raw.copy()


def drug_feats(ds, inchikey, fp_override, fnvec_override):
    pv = fnvec_override if fnvec_override is not None else ds._drug_fn_vec(inchikey)
    fp = fp_override if fp_override is not None else ds._morgan_fp(inchikey)
    return pv.astype(np.float32), fp.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug-a"); ap.add_argument("--smiles-a"); ap.add_argument("--name-a", default="Drug A")
    ap.add_argument("--nodes-a", default="")
    ap.add_argument("--drug-b"); ap.add_argument("--smiles-b"); ap.add_argument("--name-b", default="Drug B")
    ap.add_argument("--nodes-b", default="")
    ap.add_argument("--cells", required=True, help="comma-separated cell names or ACH ids")
    ap.add_argument("--ensembles", default="ldo_heldout,lco_heldout,random")
    args = ap.parse_args()

    df = pd.read_parquet(DATA)

    def prep(drug, smiles, nodes):
        if smiles:
            ik = None
            try:
                from rdkit import Chem
                ik = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
            except Exception:
                pass
            fp = morgan_fp(smiles)
            fnv = function_vector_from_nodes([x for x in nodes.split(",") if x]) if nodes else np.zeros(67, np.float32)
            tgt = "(novel; targets user-supplied)"
            return ik, fp, fnv, tgt
        ik, tgt = resolve_known(df, drug)
        return ik, None, None, tgt

    ikA, fpA, fnA, tgtA = prep(args.drug_a, args.smiles_a, args.nodes_a)
    ikB, fpB, fnB, tgtB = prep(args.drug_b, args.smiles_b, args.nodes_b)
    cells = [resolve_cell(df, c.strip()) for c in args.cells.split(",")]
    cell_names = [df[df["cell_feature_id"] == c].iloc[0]["cell_name_canonical"] for c in cells]

    print(f"Drug A: {args.name_a}  ik={ikA}  targets={tgtA}")
    print(f"Drug B: {args.name_b}  ik={ikB}  targets={tgtB}")
    print(f"Cells : {list(zip(cell_names, cells))}\n")

    df_cache = {}
    for cell_id, cname in zip(cells, cell_names):
        print(f"================ CELL {cname} ({cell_id}) ================")
        all_probs, all_thr = [], []
        for ens in args.ensembles.split(","):
            ens = ens.strip()
            cdir, stype, seeds, prov = ENSEMBLES[ens]
            if ens not in df_cache:
                df_cache[ens] = pd.read_parquet(prov["data"])
            df_ens = df_cache[ens]
            probs = []
            agg = {"riA": [], "riB": [], "alpha": [], "core": [], "resid": [], "same": [], "cross": []}
            for s in seeds:
                ckpt = REPO / cdir.format(s=s) / "best.ckpt"
                if not ckpt.exists():
                    continue
                ds, hvg = build_train_ds(df_ens, stype, s, prov["npz_tmpl"],
                                          proc_dir=prov["proc"], func_dir=prov["func"])
                if cell_id not in ds.cell_to_row:
                    continue
                pA, faA = drug_feats(ds, ikA, fpA, fnA)
                pB, faB = drug_feats(ds, ikB, fpB, fnB)
                c_fn, c_mut, c_raw = cell_tensors(ds, hvg, cell_id, proc_dir=prov["proc"])
                cell_row = ds.cell_to_row[cell_id]
                t = lambda x: torch.from_numpy(np.asarray(x, np.float32)).unsqueeze(0)
                batch = dict(pA=t(pA), pB=t(pB), fp_A=t(faA), fp_B=t(faB),
                             c_fn=t(c_fn), c_mut_fn=t(c_mut), c_raw=t(c_raw),
                             cell_row_idx=torch.tensor([cell_row], dtype=torch.long))
                m = UniMoS.load_from_checkpoint(str(ckpt), map_location="cpu"); m.eval()
                with torch.no_grad():
                    out = m(batch)
                    z_cell = m.virtual_cell(batch["cell_row_idx"], batch["c_fn"] + m.alpha_mut * batch["c_mut_fn"],
                                             m.cell_enc(batch["c_raw"])) if m.use_virtual_cell else m.cell_enc(batch["c_raw"])
                    isc = m.kernel.interaction_scores(batch["pA"], batch["pB"], z_cell, batch["cell_row_idx"])
                p = torch.sigmoid(out["logit_class"]).item()
                probs.append(p)
                agg["riA"].append(out["yhat_ri_A"].item()); agg["riB"].append(out["yhat_ri_B"].item())
                agg["alpha"].append(out["alpha"].mean().item())
                agg["core"].append(out["core_logit"].item()); agg["resid"].append(out["resid_logit"].item())
                agg["same"].append(isc["diag"].item()); agg["cross"].append(isc["cross"].item())
                mj = REPO / cdir.format(s=s) / "metrics.json"
                if mj.exists():
                    all_thr.append(json.load(open(mj))["test"].get("best_threshold", 0.5))
            if not probs:
                print(f"  [{ens}] no usable checkpoints for this cell"); continue
            all_probs += probs
            print(f"  [{ens:12s}] prob={np.mean(probs):.3f} {['%.3f'%x for x in probs]} "
                  f"RI_A={np.mean(agg['riA']):.2f} RI_B={np.mean(agg['riB']):.2f} "
                  f"same_fn={np.mean(agg['same']):+.3f} cross={np.mean(agg['cross']):+.3f} "
                  f"alpha={np.mean(agg['alpha']):.3f}")
        if all_probs:
            mp = float(np.mean(all_probs)); thr = float(np.mean(all_thr)) if all_thr else 0.5
            print(f"  >>> OVERALL prob={mp:.3f}  thr~{thr:.3f}  -> {'SYNERGY' if mp >= thr else 'NOT synergy'}\n")


if __name__ == "__main__":
    main()
