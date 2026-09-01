"""
interpretability.py — Phase 4 interpretability hardening (design doc §3 / G-INTERP).

Three deliverables:
  1. Multi-seed stability selection on W_sym / γ (§3.1)
  2. Necessity + sufficiency ablation of highlighted edges (§3.2)
  3. Per-cell reconnection matrices W(cell) − W_base (+ optional W_prior) (§3.2/表)

Pre-registered G-INTERP thresholds are passed in by the driver (see
scripts/run_phase4_interp.py / PHASE4_INTERP_GATE.md) — this module only
computes the quantities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unimos.model.unimos import UniMoS


# ── Kernel extraction ─────────────────────────────────────────────────────────

def extract_static_kernel(model: UniMoS) -> dict[str, np.ndarray]:
    """W_base-symmetrised + gamma (cell-independent baseline readout)."""
    with torch.no_grad():
        Wb = model.kernel.W_base.detach().cpu()
        W_sym = ((Wb + Wb.t()) / 2).numpy().astype(np.float64)
        gamma = model.kernel.gamma.detach().cpu().numpy().astype(np.float64)
    return {"W_sym": W_sym, "gamma": gamma}


def _build_W_sym(
    kernel: nn.Module,
    z_cell: torch.Tensor,
    cell_row_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Replicate PathwayAggregationKernel's W_sym path without dropout.

    Dispatches through kernel._delta_w so this works for both
    cell_cond_mode="hypernet" (ΔW = f(z_cell)) and "fm_kernel_interp"
    (ΔW = FM-neighbor-interpolated per-cell factors, a function of
    cell_row_idx only — the production champion checkpoints use this mode
    and have no hyper_U/hyper_V attributes at all).
    """
    B = z_cell.shape[0]
    dW = kernel._delta_w(z_cell, cell_row_idx, apply_dropout=False)
    W = kernel.W_base.unsqueeze(0) + dW
    if kernel.w_prior is not None:
        W = W + kernel.w_prior_scale * kernel.w_prior
    if kernel.cell_specific_w_prior:
        assert cell_row_idx is not None
        wp_cell = kernel.w_prior_percell[cell_row_idx]
        gamma = kernel.film_gamma(z_cell)
        beta = kernel.film_beta(z_cell)
        gate = (1 + gamma).unsqueeze(2) * (1 + gamma).unsqueeze(1)
        shift = beta.unsqueeze(2) + beta.unsqueeze(1)
        prior_raw = gate * wp_cell + shift
        prior_norm = prior_raw.norm(dim=(1, 2), keepdim=True)
        prior_term = prior_raw * (
            kernel.tau_prior * torch.tanh(prior_norm / kernel.tau_prior)
            / (prior_norm + 1e-8)
        )
        W = W + prior_term
    return (W + W.transpose(1, 2)) / 2


def _core_logit_from_parts(
    model: UniMoS,
    W_sym: torch.Tensor,
    pA: torch.Tensor,
    pB: torch.Tensor,
    gamma: torch.Tensor,
    h_struct: torch.Tensor,
    z_cell: torch.Tensor,
    ri_repr_A: torch.Tensor,
    ri_repr_B: torch.Tensor,
    use_resid: bool = True,
) -> dict[str, torch.Tensor]:
    """core_logit (+ optional full logit) given an explicit W_sym."""
    Wpb = torch.bmm(W_sym, pB.unsqueeze(2)).squeeze(2)
    Wpa = torch.bmm(W_sym, pA.unsqueeze(2)).squeeze(2)
    contrib_A = pA * Wpb
    contrib_B = pB * Wpa
    kern_repr = torch.cat([contrib_A, contrib_B], dim=-1)
    same_fn = (pA * pB * gamma).sum(dim=-1, keepdim=True)
    head = model.synergy_head
    F = head.kern_dim // 2
    C_A, C_B = kern_repr[:, :F], kern_repr[:, F:]
    core_input = torch.cat([C_A + C_B, (C_A - C_B).abs(), same_fn], dim=-1)
    core_logit = head.core_head(core_input).squeeze(-1)
    out = {
        "core_logit": core_logit,
        "contrib_A": contrib_A,
        "contrib_B": contrib_B,
        "same_fn": same_fn,
        "kern_repr": kern_repr,
    }
    if use_resid:
        ri_sum = ri_repr_A + ri_repr_B
        ri_prod = ri_repr_A * ri_repr_B
        resid_input = torch.cat([h_struct, z_cell, ri_sum, ri_prod], dim=-1)
        resid_logit = head.resid_head(resid_input).squeeze(-1)
        alpha = torch.sigmoid(head.gate(z_cell)).squeeze(-1)
        out["resid_logit"] = resid_logit
        out["alpha"] = alpha
        out["logit_class"] = core_logit + alpha * resid_logit
    return out


@torch.no_grad()
def _encode_batch(model: UniMoS, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Mirror UniMoS.forward up to (and including) RI / struct / z_cell."""
    fp_A = batch["fp_A"]
    fp_B = batch["fp_B"]
    pA = batch["pA"]
    pB = batch["pB"]
    c_fn = batch["c_fn"]
    c_mut_fn = batch["c_mut_fn"]
    c_raw = batch["c_raw"]

    if model.use_film_rescue:
        pA, pB = model.film_rescue(pA, pB, batch["cell_row_idx"])

    c = c_fn + model.alpha_mut * c_mut_fn
    if model.struct_encoder == "gnn":
        gA = model._graphs_for(batch["ik_A"], fp_A.device)
        gB = model._graphs_for(batch["ik_B"], fp_A.device)
        h_struct = model.gin(gA) + model.gin(gB)
    else:
        h_struct = model.struct_enc(fp_A) + model.struct_enc(fp_B)

    if model.use_descriptors:
        dev = fp_A.device
        last = model._desc_mat.shape[0] - 1
        iA = torch.tensor([model._desc_ik.get(k, last) for k in batch["ik_A"]], device=dev)
        iB = torch.tensor([model._desc_ik.get(k, last) for k in batch["ik_B"]], device=dev)
        h_struct = h_struct + model.desc_enc(model._desc_mat[iA]) + model.desc_enc(model._desc_mat[iB])

    if model.use_sensitivity_profile:
        dev = fp_A.device
        last_d = model._sens_drug_mat.shape[0] - 1
        iA = torch.tensor([model._sens_drug_ik.get(k, last_d) for k in batch["ik_A"]], device=dev)
        iB = torch.tensor([model._sens_drug_ik.get(k, last_d) for k in batch["ik_B"]], device=dev)
        h_struct = (
            h_struct
            + model.sens_drug_enc(model._sens_drug_mat[iA])
            + model.sens_drug_enc(model._sens_drug_mat[iB])
        )

    h_cell = model.cell_enc(c_raw)
    if model.use_sensitivity_profile:
        last_c = model._sens_cell_mat.shape[0] - 1
        iC = torch.tensor(
            [model._sens_cell_ik.get(k, last_c) for k in batch["cell_id"]], device=fp_A.device
        )
        h_cell = h_cell + model.sens_cell_enc(model._sens_cell_mat[iC])

    if model.use_virtual_cell:
        z_cell = model.virtual_cell(batch["cell_row_idx"], c, h_cell)
    else:
        z_cell = h_cell

    ri_repr_A = model.ri_enc(pA, z_cell)
    ri_repr_B = model.ri_enc(pB, z_cell)
    return {
        "pA": pA,
        "pB": pB,
        "z_cell": z_cell,
        "h_struct": h_struct,
        "ri_repr_A": ri_repr_A,
        "ri_repr_B": ri_repr_B,
        "cell_row_idx": batch["cell_row_idx"],
    }


# ── §3.1 Stability selection ──────────────────────────────────────────────────

def stability_selection(
    kernels: list[dict[str, np.ndarray]],
    top_k: int = 50,
    min_sign_agree: float = 0.7,
    node_labels: Optional[list[str]] = None,
) -> dict:
    """
    Cross-seed stability on static W_sym / gamma.

    An off-diagonal edge (i,j), i<j, is *stable* iff among the top_k edges by
    |mean W_sym| it also has sign-agreement ≥ min_sign_agree across seeds.
    Same rule for gamma entries (top_k by |mean|).
    """
    n = len(kernels)
    assert n >= 2
    W_stack = np.stack([k["W_sym"] for k in kernels], axis=0)  # (S, F, F)
    g_stack = np.stack([k["gamma"] for k in kernels], axis=0)  # (S, F)
    F = W_stack.shape[1]
    if node_labels is None:
        node_labels = [f"F{i:03d}" for i in range(F)]

    mean_W = W_stack.mean(axis=0)
    std_W = W_stack.std(axis=0, ddof=1)
    # Rank by mean(|W_s|) — NOT |mean W| — so magnitude ranking is independent
    # of sign agreement (selecting by |mean| would circularly favour same-sign edges).
    mean_abs_W = np.abs(W_stack).mean(axis=0)
    tri = np.triu_indices(F, k=1)
    mag = mean_abs_W[tri]
    order = np.argsort(-mag)
    top_idx = order[: min(top_k, len(order))]

    edge_rows = []
    n_stable = 0
    for t in top_idx:
        i, j = int(tri[0][t]), int(tri[1][t])
        vals = W_stack[:, i, j]
        signs = np.sign(vals)
        if (signs != 0).sum() == 0:
            agree = 0.0
        else:
            agree = float(max((signs > 0).sum(), (signs < 0).sum()) / n)
        # magnitude consistency: IQR / median(|v|) small → consistent scale
        abs_v = np.abs(vals)
        med = float(np.median(abs_v))
        iqr = float(np.percentile(abs_v, 75) - np.percentile(abs_v, 25))
        mag_cv = iqr / max(med, 1e-8)
        # "量级一致": relative IQR ≤ 1.5 (pre-registered with the driver)
        mag_ok = mag_cv <= 1.5
        stable = (agree >= min_sign_agree) and mag_ok
        if stable:
            n_stable += 1
        lo, hi = np.percentile(vals, [2.5, 97.5])
        edge_rows.append({
            "i": i, "j": j,
            "label_i": node_labels[i], "label_j": node_labels[j],
            "mean": float(mean_W[i, j]),
            "mean_abs": float(mean_abs_W[i, j]),
            "std": float(std_W[i, j]),
            "ci95_lo": float(lo), "ci95_hi": float(hi),
            "sign_agree": agree,
            "mag_cv": mag_cv,
            "mag_ok": bool(mag_ok),
            "stable": bool(stable),
            "rank_by_mean_abs": int(np.where(order == t)[0][0] + 1),
        })

    mean_g = g_stack.mean(axis=0)
    std_g = g_stack.std(axis=0, ddof=1)
    mean_abs_g = np.abs(g_stack).mean(axis=0)
    g_order = np.argsort(-mean_abs_g)
    g_top = g_order[: min(top_k, F)]
    gamma_rows = []
    n_g_stable = 0
    for rank, k in enumerate(g_top, start=1):
        vals = g_stack[:, k]
        signs = np.sign(vals)
        if (signs != 0).sum() == 0:
            agree = 0.0
        else:
            agree = float(max((signs > 0).sum(), (signs < 0).sum()) / n)
        abs_v = np.abs(vals)
        med = float(np.median(abs_v))
        iqr = float(np.percentile(abs_v, 75) - np.percentile(abs_v, 25))
        mag_cv = iqr / max(med, 1e-8)
        mag_ok = mag_cv <= 1.5
        stable = (agree >= min_sign_agree) and mag_ok
        if stable:
            n_g_stable += 1
        lo, hi = np.percentile(vals, [2.5, 97.5])
        gamma_rows.append({
            "k": int(k), "label": node_labels[k],
            "mean": float(mean_g[k]), "mean_abs": float(mean_abs_g[k]),
            "std": float(std_g[k]),
            "ci95_lo": float(lo), "ci95_hi": float(hi),
            "sign_agree": agree, "mag_cv": mag_cv, "mag_ok": bool(mag_ok),
            "stable": bool(stable),
            "rank_by_mean_abs": rank,
        })

    # pairwise cosine of flattened W_sym (legacy metric from evaluate.py)
    flat = W_stack.reshape(n, -1)
    cos_vals = []
    for a in range(n):
        for b in range(a + 1, n):
            na = np.linalg.norm(flat[a]) * np.linalg.norm(flat[b])
            cos_vals.append(float(np.dot(flat[a], flat[b]) / na) if na > 0 else 0.0)

    n_top = len(edge_rows)
    n_g_top = len(gamma_rows)
    return {
        "n_seeds": n,
        "top_k": top_k,
        "min_sign_agree": min_sign_agree,
        "edge_stable_rate": float(n_stable / n_top) if n_top else float("nan"),
        "n_stable_edges": n_stable,
        "n_top_edges": n_top,
        "gamma_stable_rate": float(n_g_stable / n_g_top) if n_g_top else float("nan"),
        "n_stable_gamma": n_g_stable,
        "n_top_gamma": n_g_top,
        "W_sym_mean_cosine": float(np.mean(cos_vals)) if cos_vals else float("nan"),
        "edges": edge_rows,
        "gamma": gamma_rows,
        "mean_W_sym": mean_W,
        "mean_gamma": mean_g,
    }


# ── §3.2 Necessity / sufficiency ──────────────────────────────────────────────

@torch.no_grad()
def necessity_ablation(
    model: UniMoS,
    loader: DataLoader,
    edges: list[dict],
    device: torch.device,
    max_batches: Optional[int] = None,
    rel_threshold: float = 0.05,
) -> dict:
    """
    For each highlighted edge (i,j): zero W_sym[:,i,j] and W_sym[:,j,i],
    measure mean |Δ core_logit| / mean |core_logit| over the loader.

    Edge *passes necessity* if relative Δ ≥ rel_threshold.

    Implementation note: ablate via a single batched mask over edges rather than
    cloning W per edge (40 edges × full test would otherwise be prohibitive).
    """
    model.eval()
    model.to(device)
    stable_edges = [e for e in edges if e.get("stable", True)]
    if not stable_edges:
        return {
            "n_edges": 0, "necessity_pass_rate": float("nan"),
            "rel_threshold": rel_threshold, "edges": [],
        }

    E = len(stable_edges)
    ij = [(e["i"], e["j"]) for e in stable_edges]
    sum_abs_delta = np.zeros(E, dtype=np.float64)
    sum_abs_core = 0.0
    n_samples = 0

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        enc = _encode_batch(model, batch)
        W_sym = _build_W_sym(model.kernel, enc["z_cell"], enc["cell_row_idx"])
        gamma = model.kernel.gamma
        B, F = enc["pA"].shape

        # Expand batch × edges: (B*E, F, F) with one edge zeroed per slice.
        # For large E this is memory-heavy; chunk edges.
        chunk = 8
        core0 = None
        for start in range(0, E, chunk):
            sl = ij[start:start + chunk]
            e_chunk = len(sl)
            W_rep = W_sym.unsqueeze(0).expand(e_chunk, -1, -1, -1).clone()  # (E',B,F,F)
            for local, (i, j) in enumerate(sl):
                W_rep[local, :, i, j] = 0.0
                W_rep[local, :, j, i] = 0.0
            W_flat = W_rep.reshape(e_chunk * B, F, F)
            pA = enc["pA"].unsqueeze(0).expand(e_chunk, -1, -1).reshape(e_chunk * B, F)
            pB = enc["pB"].unsqueeze(0).expand(e_chunk, -1, -1).reshape(e_chunk * B, F)
            zc = enc["z_cell"].unsqueeze(0).expand(e_chunk, -1, -1).reshape(
                e_chunk * B, enc["z_cell"].shape[-1]
            )
            hs = enc["h_struct"].unsqueeze(0).expand(e_chunk, -1, -1).reshape(
                e_chunk * B, enc["h_struct"].shape[-1]
            )
            riA = enc["ri_repr_A"].unsqueeze(0).expand(e_chunk, -1, -1).reshape(
                e_chunk * B, enc["ri_repr_A"].shape[-1]
            )
            riB = enc["ri_repr_B"].unsqueeze(0).expand(e_chunk, -1, -1).reshape(
                e_chunk * B, enc["ri_repr_B"].shape[-1]
            )
            ab = _core_logit_from_parts(
                model, W_flat, pA, pB, gamma, hs, zc, riA, riB, use_resid=False,
            )["core_logit"].view(e_chunk, B)
            if core0 is None:
                # compute once
                core0 = _core_logit_from_parts(
                    model, W_sym, enc["pA"], enc["pB"], gamma,
                    enc["h_struct"], enc["z_cell"], enc["ri_repr_A"], enc["ri_repr_B"],
                    use_resid=False,
                )["core_logit"]
                sum_abs_core += float(core0.abs().sum().item())
                n_samples += int(core0.numel())
            delta = (core0.unsqueeze(0) - ab).abs()  # (E', B)
            sum_abs_delta[start:start + e_chunk] += delta.sum(dim=1).cpu().numpy()

    mean_abs_core = sum_abs_core / max(n_samples, 1)
    rows = []
    n_pass = 0
    for ei, e in enumerate(stable_edges):
        mean_abs_delta = float(sum_abs_delta[ei] / max(n_samples, 1))
        rel = mean_abs_delta / max(mean_abs_core, 1e-8)
        passes = rel >= rel_threshold
        if passes:
            n_pass += 1
        rows.append({
            **{k: e[k] for k in (
                "i", "j", "label_i", "label_j", "mean", "mean_abs",
                "sign_agree", "mag_cv", "stable",
            ) if k in e},
            "mean_abs_delta_core": mean_abs_delta,
            "rel_delta": float(rel),
            "necessity_pass": bool(passes),
        })

    return {
        "n_edges": len(stable_edges),
        "n_samples": n_samples,
        "mean_abs_core_logit": mean_abs_core,
        "rel_threshold": rel_threshold,
        "necessity_pass_rate": float(n_pass / len(stable_edges)) if stable_edges else float("nan"),
        "n_pass": n_pass,
        "edges": rows,
    }


@torch.no_grad()
def sufficiency_ablation(
    model: UniMoS,
    loader: DataLoader,
    device: torch.device,
    keep_ks: tuple[int, ...] = (10, 25, 50, 100),
    max_batches: Optional[int] = None,
) -> dict:
    """
    Keep only the top-k |W_base_sym| edges (global, static), zero the rest of
    W_sym (including ΔW entries outside those positions), measure Spearman of
    ablated core_logit vs full core_logit.
    """
    from scipy.stats import spearmanr

    model.eval()
    model.to(device)
    with torch.no_grad():
        Wb = model.kernel.W_base.detach()
        W0 = ((Wb + Wb.t()) / 2).cpu().numpy()
    F = W0.shape[0]
    tri = np.triu_indices(F, k=1)
    mag = np.abs(W0[tri])
    order = np.argsort(-mag)

    cores_full: list[np.ndarray] = []
    cores_by_k: dict[int, list[np.ndarray]] = {k: [] for k in keep_ks}

    masks = {}
    for k in keep_ks:
        m = np.zeros((F, F), dtype=np.float32)
        for t in order[:k]:
            i, j = int(tri[0][t]), int(tri[1][t])
            m[i, j] = m[j, i] = 1.0
        # always keep diagonal (same-community self terms)
        np.fill_diagonal(m, 1.0)
        masks[k] = torch.from_numpy(m).to(device)

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        enc = _encode_batch(model, batch)
        W_sym = _build_W_sym(model.kernel, enc["z_cell"], enc["cell_row_idx"])
        gamma = model.kernel.gamma
        full = _core_logit_from_parts(
            model, W_sym, enc["pA"], enc["pB"], gamma,
            enc["h_struct"], enc["z_cell"], enc["ri_repr_A"], enc["ri_repr_B"],
            use_resid=False,
        )["core_logit"].cpu().numpy()
        cores_full.append(full)
        for k, mask in masks.items():
            W_k = W_sym * mask.unsqueeze(0)
            ab = _core_logit_from_parts(
                model, W_k, enc["pA"], enc["pB"], gamma,
                enc["h_struct"], enc["z_cell"], enc["ri_repr_A"], enc["ri_repr_B"],
                use_resid=False,
            )["core_logit"].cpu().numpy()
            cores_by_k[k].append(ab)

    y = np.concatenate(cores_full)
    out = {"n_samples": int(y.size), "spearman": {}}
    for k, parts in cores_by_k.items():
        yk = np.concatenate(parts)
        rho, _ = spearmanr(y, yk)
        out["spearman"][str(k)] = float(rho) if np.isfinite(rho) else float("nan")
    return out


# ── Per-cell reconnection ─────────────────────────────────────────────────────

@torch.no_grad()
def per_cell_reconnection(
    model: UniMoS,
    loader: DataLoader,
    device: torch.device,
    cell_ids: list[str],
    cell_feature_index: dict[str, int],
    w_prior_percell: Optional[np.ndarray] = None,
    max_batches: Optional[int] = None,
) -> dict[str, dict]:
    """
    For selected cells: mean W_sym(cell) over test samples of that cell,
    Δ = W(cell) − W_base_sym, and optional offline W_prior(cell).
    """
    model.eval()
    model.to(device)
    wanted = {cid: cell_feature_index[cid] for cid in cell_ids if cid in cell_feature_index}
    if not wanted:
        return {}

    Wb = model.kernel.W_base.detach().cpu()
    W_base_sym = ((Wb + Wb.t()) / 2).numpy().astype(np.float64)

    # row_idx -> accumulators
    inv = {row: cid for cid, row in wanted.items()}
    sum_W = {row: np.zeros_like(W_base_sym) for row in inv}
    counts = {row: 0 for row in inv}

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        enc = _encode_batch(model, batch)
        W_sym = _build_W_sym(model.kernel, enc["z_cell"], enc["cell_row_idx"]).cpu().numpy()
        rows = batch["cell_row_idx"].cpu().numpy()
        for t, row in enumerate(rows):
            row = int(row)
            if row in sum_W:
                sum_W[row] += W_sym[t]
                counts[row] += 1

    out: dict[str, dict] = {}
    for row, cid in inv.items():
        n = counts[row]
        if n == 0:
            continue
        W_cell = sum_W[row] / n
        entry = {
            "cell_id": cid,
            "n_samples": n,
            "W_cell": W_cell,
            "W_base": W_base_sym,
            "delta": W_cell - W_base_sym,
            "delta_fro": float(np.linalg.norm(W_cell - W_base_sym)),
        }
        if w_prior_percell is not None and row < w_prior_percell.shape[0]:
            wp = w_prior_percell[row].astype(np.float64)
            entry["W_prior"] = wp
            entry["prior_vs_base_fro"] = float(np.linalg.norm(wp - W_base_sym))
            entry["prior_vs_cell_fro"] = float(np.linalg.norm(wp - W_cell))
            entry["prior_cell_corr"] = float(
                np.corrcoef(wp.ravel(), W_cell.ravel())[0, 1]
            )
        out[cid] = entry
    return out


def load_node_labels(func_nodes_json: Path) -> list[str]:
    raw = json.loads(Path(func_nodes_json).read_text())
    labels = []
    for i, item in enumerate(raw):
        lab = item.get("manual_label") or item.get("auto_label") or item.get("function_id") or f"F{i:03d}"
        # shorten for tables
        labels.append(str(lab)[:48])
    return labels


def plot_reconnection_panel(
    cell_payload: dict,
    node_labels: list[str],
    out_path: Path,
    top_n: int = 20,
) -> None:
    """Save a 1×2 (or 1×3) heatmap panel for one cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta = cell_payload["delta"]
    F = delta.shape[0]
    # show top_n × top_n sub-block by row |Δ| energy
    energy = np.abs(delta).sum(axis=1)
    keep = np.argsort(-energy)[:top_n]
    labs = [node_labels[i][:18] for i in keep]

    panels = [("ΔW = W(cell)−W_base", delta[np.ix_(keep, keep)])]
    if "W_prior" in cell_payload:
        panels.append(("W_prior(cell)", cell_payload["W_prior"][np.ix_(keep, keep)]))
        panels.append(("W(cell)", cell_payload["W_cell"][np.ix_(keep, keep)]))

    n_p = len(panels)
    fig, axes = plt.subplots(1, n_p, figsize=(4.2 * n_p, 4.0))
    if n_p == 1:
        axes = [axes]
    for ax, (title, mat) in zip(axes, panels):
        vmax = np.percentile(np.abs(mat), 99) + 1e-8
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(top_n))
        ax.set_yticks(range(top_n))
        ax.set_xticklabels(labs, rotation=90, fontsize=6)
        ax.set_yticklabels(labs, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cid = cell_payload["cell_id"]
    fig.suptitle(
        f"{cid}  n={cell_payload['n_samples']}  ‖Δ‖_F={cell_payload['delta_fro']:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
