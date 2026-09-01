"""
pathway_kernel.py — PathwayAggregationKernel for UniMoS (SPEC-03).

Computation:
  U(z_cell), V(z_cell) = hyper_U/V(z_cell) reshaped to (B, F, rank_r)
  ΔW_raw      = U @ V^T                           (B, F, F)
  ΔW          = ΔW_raw * τ·tanh(‖ΔW_raw‖_F/τ) / (‖ΔW_raw‖_F + ε)
                → ‖ΔW‖_F = τ·tanh(‖ΔW_raw‖_F/τ) ≤ τ  (guaranteed)
  W           = W_base + ΔW [+ w_prior term, see below]  (B, F, F)
  W_sym       = (W + W^T) / 2                      per-batch
  contrib_A   = pA ⊙ (W_sym @ pB)                 (B, F)
  contrib_B   = pB ⊙ (W_sym @ pA)                 (B, F)
  kern_repr   = cat([contrib_A, contrib_B], dim=1) (B, 2F = 134)
  same_fn     = -(pA ⊙ pB ⊙ γ).sum(-1, keepdim=True)  (B, 1)

Design notes:
  * kern_repr is NOT symmetric under pA↔pB: swapping drugs swaps the two
    67-dim halves.  End-to-end logit_class A/B invariance is enforced in
    SynergyHead (SPEC-04) which treats the two halves symmetrically.
  * τ is a fixed architectural hyperparameter (default 1.0), not HPO.
  * same_fn is returned separately; SynergyHead cats it to kern_repr to form
    the 135-dim input of core_head.
  * UniMoS-VC (design doc §4): the cell-conditioning input to hyper_U/V was
    the 67-dim functional-activity vector `c`; it is now the fused `z_cell`
    (VirtualCellModule output, default 128-dim) — see `cell_ctx_dim`. This is
    a breaking change to hyper_U/V's input shape versus pre-VC checkpoints.

  * cell_cond_mode="fm_kernel_interp" (DESIGN_FM_KERNEL_INTERP.md): replace
    HyperNet(z_cell)→(U,V) with softmax interpolation of per-train-cell
    low-rank factors (U_j, V_j) in frozen FM embedding space:
        w_j = softmax_j( <z_cell, z_j> / τ_fm )   j ∈ train cells
        U   = Σ_j w_j U_j ,  V = Σ_j w_j V_j
    FM supplies a cell-similarity prior, not a perturbation prediction.
    Requires cell_row_idx + set_train_cell_mask(...) before training.

Phase 2 (design doc §2.2, entrance two): W_prior can be either
  (a) a single global (F,F) buffer added as `w_prior_scale * W_prior`
      (original, unchanged — default when `cell_specific_w_prior=False`), or
  (b) a per-cell (n_cells, F, F) frozen lookup table (`build_cell_prior.py`),
      FiLM-modulated by z_cell per the design doc's
      `W(cell) = W_base + FiLM_γ(z_cell) ⊙ W_prior(cell) + FiLM_β(z_cell) + ΔW_lowrank(z_cell)`:
        γ, β = Linear(z_cell) -> (B, F) each
        gate[i,j]  = (1+γ[i])·(1+γ[j])          (per-node, symmetric outer product)
        shift[i,j] = β[i] + β[j]                 (per-node, symmetric additive)
        prior_term = gate ⊙ W_prior(cell) + shift, then Frobenius-gated to
        ‖prior_term‖_F ≤ tau_prior (same ceiling mechanism as ΔW, separate τ)
      FiLM Linear layers are zero-initialised so at init gate=all-ones,
      shift=0 — prior_term reduces to the raw W_prior(cell) unmodified,
      matching this project's "start at an interpretable, unmodulated
      baseline" convention (FoundationCellEncoder, FilmRescueModule).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PathwayAggregationKernel(nn.Module):
    """
    Cell-conditioned bilinear synergy kernel in function-node space.

    Parameters
    ----------
    F_nodes     : number of function nodes (default 67)
    rank_r      : low-rank dimension for hyper-network (default 16)
    tau         : Frobenius norm ceiling for ΔW; fixed, not HPO (default 1.0)
    dropout     : applied to U and V before outer product (default 0.1)
    cell_ctx_dim: dimensionality of the cell-conditioning input to hyper_U/V.
                  Pre-VC default was F_nodes (fed the raw 67-dim `c`); UniMoS-VC
                  passes z_cell instead — set to VirtualCellModule's z_cell_dim
                  (default 128 here to match its default).
    cell_cond_mode : "hypernet" (default) | "fm_kernel_interp"
    fm_emb_path    : precomputed cell FM embeddings (.npy, n_cells × d); required
                     for fm_kernel_interp
    fm_kernel_tau  : temperature for FM cosine softmax (default 0.1)
    fm_kernel_self_mask : leave-one-out mask of query cell during training
    """

    def __init__(
        self,
        F_nodes: int = 67,
        rank_r: int = 16,
        tau: float = 1.0,
        dropout: float = 0.1,
        w_prior: "torch.Tensor | None" = None,
        w_prior_scale: float = 0.0,
        cell_ctx_dim: int = 128,
        cell_specific_w_prior: bool = False,
        w_prior_percell_path: str = "",
        tau_prior: "float | None" = None,
        gi_prior: "torch.Tensor | None" = None,
        gi_prior_init_scale: float = 0.0,
        use_delta_w: bool = True,
        cell_cond_mode: str = "hypernet",
        fm_emb_path: str = "",
        fm_kernel_tau: float = 0.1,
        fm_kernel_self_mask: bool = True,
    ) -> None:
        super().__init__()
        self.F = F_nodes
        self.r = rank_r
        self.tau = tau
        self.tau_prior = tau if tau_prior is None else tau_prior
        # NBE ablation: False → W = W_base[+prior] only (no cell-conditioned ΔW)
        self.use_delta_w = use_delta_w

        mode = (cell_cond_mode or "hypernet").lower()
        if mode not in {"hypernet", "fm_kernel_interp"}:
            raise ValueError(
                f"cell_cond_mode must be 'hypernet' or 'fm_kernel_interp', got {cell_cond_mode!r}"
            )
        self.cell_cond_mode = mode
        self.fm_kernel_tau = float(fm_kernel_tau)
        self.fm_kernel_self_mask = bool(fm_kernel_self_mask)

        # Fixed biological vertical-synergy prior added to the effective W:
        #   W = W_base + ΔW + w_prior_scale · W_prior
        # W_prior[i,j] (i≠j) = cross-community pathway-graph adjacency (data-derived),
        # giving biologically-linked-but-distinct communities (e.g. RTK↔PI3K) a
        # positive interaction floor.  Non-trainable buffer; scale 0 = disabled.
        # Mutually exclusive with cell_specific_w_prior (Phase 2 below).
        self.w_prior_scale = float(w_prior_scale)
        if w_prior is not None and self.w_prior_scale != 0.0:
            wp = torch.as_tensor(w_prior, dtype=torch.float32)
            wp = (wp + wp.t()) / 2  # ensure symmetric
            self.register_buffer("w_prior", wp)
        else:
            self.w_prior = None

        # Phase 2 (design doc §2.2): per-cell FiLM-modulated W_prior(cell).
        self.cell_specific_w_prior = cell_specific_w_prior
        if cell_specific_w_prior:
            wp_percell = np.load(w_prior_percell_path).astype(np.float32)  # (n_cells, F, F)
            self.register_buffer("w_prior_percell", torch.from_numpy(wp_percell))
            self.film_gamma = nn.Linear(cell_ctx_dim, F_nodes)
            self.film_beta = nn.Linear(cell_ctx_dim, F_nodes)
            nn.init.zeros_(self.film_gamma.weight); nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight); nn.init.zeros_(self.film_beta.bias)

        # Base interaction matrix — init via temp tensor to avoid in-place on Parameter view
        base_init = nn.init.xavier_uniform_(torch.empty(1, F_nodes, F_nodes)).squeeze(0)
        # Phase 3 (design doc §2.3, entrance three): GI prior additive init.
        # base_init = Xavier(seed-controlled) + scale * W_base_gi_prior — keeps the
        # same random-init hygiene (no dead/all-zero regions) everywhere, only nudges
        # the ~29/2211 node pairs with real Norman-et-al. epistasis evidence. scale=0
        # (default) reproduces plain Xavier init exactly, so this is a strict
        # superset/no-op unless explicitly enabled — the GATE-G-GI ablation is:
        # same seed, same Xavier draw, only this additive term differs.
        if gi_prior is not None and gi_prior_init_scale != 0.0:
            gp = torch.as_tensor(gi_prior, dtype=torch.float32)
            base_init = base_init + gi_prior_init_scale * gp
        self.W_base = nn.Parameter(base_init)

        if mode == "hypernet":
            # Hyper-network: z_cell → U, V ∈ R^{F × r}
            self.hyper_U = nn.Linear(cell_ctx_dim, F_nodes * rank_r)
            self.hyper_V = nn.Linear(cell_ctx_dim, F_nodes * rank_r)
            nn.init.normal_(self.hyper_U.weight, std=0.01)
            nn.init.normal_(self.hyper_V.weight, std=0.01)
            nn.init.zeros_(self.hyper_U.bias)
            nn.init.zeros_(self.hyper_V.bias)
        else:
            if not fm_emb_path:
                raise ValueError("cell_cond_mode='fm_kernel_interp' requires fm_emb_path")
            emb = np.load(fm_emb_path).astype(np.float32)  # (n_cells, d_fm)
            # Same per-dim z-score as FoundationCellEncoder, then L2 → cosine via matmul
            mean = emb.mean(axis=0, keepdims=True)
            std = emb.std(axis=0, keepdims=True) + 1e-6
            emb = (emb - mean) / std
            emb_t = torch.from_numpy(emb)
            emb_t = F.normalize(emb_t, p=2, dim=1)
            self.register_buffer("fm_z", emb_t)  # (n_cells, d_fm), frozen
            n_cells = emb_t.shape[0]
            self.beta_U = nn.Parameter(torch.empty(n_cells, F_nodes, rank_r))
            self.beta_V = nn.Parameter(torch.empty(n_cells, F_nodes, rank_r))
            nn.init.normal_(self.beta_U, std=0.01)
            nn.init.normal_(self.beta_V, std=0.01)
            # Overwritten by set_train_cell_mask before fit; default all-True is safe
            # for smoke tests that skip the setter (IID / random split).
            self.register_buffer("train_cell_mask", torch.ones(n_cells, dtype=torch.bool))

        # Per-node signed overlap coefficient (γ > 0 = redundancy penalty)
        self.gamma = nn.Parameter(torch.zeros(F_nodes))

        self.dropout = nn.Dropout(p=dropout)

    def set_train_cell_mask(self, train_cell_rows: "list[int] | torch.Tensor") -> None:
        """Restrict FM-kernel dictionary to train-fold cells (LCO-safe)."""
        if self.cell_cond_mode != "fm_kernel_interp":
            return
        n = int(self.fm_z.shape[0])
        mask = torch.zeros(n, dtype=torch.bool, device=self.fm_z.device)
        if isinstance(train_cell_rows, torch.Tensor):
            rows = train_cell_rows.detach().cpu().long().tolist()
        else:
            rows = [int(r) for r in train_cell_rows]
        for r in rows:
            if 0 <= r < n:
                mask[r] = True
        if not bool(mask.any()):
            raise ValueError("set_train_cell_mask: empty train cell set")
        self.train_cell_mask = mask

    def _fm_interp_UV(self, cell_row_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Softmax-interpolate per-cell (U_j, V_j) in FM embedding space."""
        assert cell_row_idx is not None, "fm_kernel_interp requires cell_row_idx"
        B = cell_row_idx.shape[0]
        z_q = self.fm_z[cell_row_idx]                          # (B, d)
        # logits[b, j] = <z_q[b], z_j> / τ
        logits = (z_q @ self.fm_z.T) / max(self.fm_kernel_tau, 1e-8)  # (B, n_cells)
        logits = logits.masked_fill(~self.train_cell_mask.unsqueeze(0), float("-inf"))
        if self.fm_kernel_self_mask:
            # Leave-one-out, train AND eval: a cell's β row is only ever trained
            # to serve as a neighbor for OTHER cells, so it must never be allowed
            # to copy itself at inference either — train/eval must use the same
            # objective, or val/test degrade as the row specializes for neighbors
            # instead of self (confirmed empirically on random/ldo/lpo: val_auroc
            # peaks at epoch 0 and monotonically drops once self-mask is train-only).
            logits.scatter_(1, cell_row_idx.view(-1, 1), float("-inf"))
        # If a row is all -inf (degenerate), fall back to uniform over train cells.
        bad = ~torch.isfinite(logits).any(dim=1)
        if bool(bad.any()):
            fill = torch.zeros_like(logits)
            fill = fill.masked_fill(~self.train_cell_mask.unsqueeze(0), float("-inf"))
            logits = torch.where(bad.unsqueeze(1), fill, logits)
        w = torch.softmax(logits, dim=1)                       # (B, n_cells)
        U = torch.einsum("bn,nfr->bfr", w, self.beta_U)        # (B, F, r)
        V = torch.einsum("bn,nfr->bfr", w, self.beta_V)
        return U, V

    def _lowrank_UV(
        self,
        z_cell: torch.Tensor,
        cell_row_idx: "torch.Tensor | None",
        apply_dropout: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = z_cell.shape[0]
        if self.cell_cond_mode == "fm_kernel_interp":
            U, V = self._fm_interp_UV(cell_row_idx)
        else:
            U = self.hyper_U(z_cell).view(B, self.F, self.r)
            V = self.hyper_V(z_cell).view(B, self.F, self.r)
        if apply_dropout:
            U = self.dropout(U)
            V = self.dropout(V)
        return U, V

    def _delta_w(
        self,
        z_cell: torch.Tensor,
        cell_row_idx: "torch.Tensor | None",
        apply_dropout: bool = True,
    ) -> torch.Tensor:
        B = z_cell.shape[0]
        if not self.use_delta_w:
            return torch.zeros(B, self.F, self.F, device=z_cell.device, dtype=z_cell.dtype)
        U, V = self._lowrank_UV(z_cell, cell_row_idx, apply_dropout=apply_dropout)
        dW_raw = torch.bmm(U, V.transpose(1, 2))
        norm_raw = dW_raw.norm(dim=(1, 2), keepdim=True)
        return dW_raw * (self.tau * torch.tanh(norm_raw / self.tau) / (norm_raw + 1e-8))

    def forward(
        self,
        pA: torch.Tensor,      # (B, F)
        pB: torch.Tensor,      # (B, F)
        z_cell: torch.Tensor,  # (B, cell_ctx_dim)  fused cell context (VirtualCellModule)
        cell_row_idx: "torch.Tensor | None" = None,  # (B,) long; required iff cell_specific_w_prior
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        kern_repr : (B, 2F)  cat([contrib_A, contrib_B])
        same_fn   : (B, 1)   node-overlap signal
        """
        B, F = pA.shape

        # ── Low-rank cell-conditioned ΔW ──────────────────────────────────
        dW = self._delta_w(z_cell, cell_row_idx)

        # ── W(z_cell) and per-batch symmetrization ────────────────────────
        W = self.W_base.unsqueeze(0) + dW                      # (B, F, F)
        if self.w_prior is not None:
            W = W + self.w_prior_scale * self.w_prior          # broadcast (F,F)
        if self.cell_specific_w_prior:
            assert cell_row_idx is not None, "cell_specific_w_prior=True requires cell_row_idx"
            wp_cell = self.w_prior_percell[cell_row_idx]        # (B, F, F)
            gamma = self.film_gamma(z_cell)                     # (B, F)
            beta = self.film_beta(z_cell)                       # (B, F)
            gate = (1 + gamma).unsqueeze(2) * (1 + gamma).unsqueeze(1)   # (B, F, F)
            shift = beta.unsqueeze(2) + beta.unsqueeze(1)                # (B, F, F)
            prior_raw = gate * wp_cell + shift                  # (B, F, F)
            prior_norm = prior_raw.norm(dim=(1, 2), keepdim=True)
            prior_term = prior_raw * (self.tau_prior * torch.tanh(prior_norm / self.tau_prior)
                                       / (prior_norm + 1e-8))
            W = W + prior_term
        W_sym = (W + W.transpose(1, 2)) / 2                    # (B, F, F)

        # ── Per-node contribution vectors ──────────────────────────────────
        Wpb = torch.bmm(W_sym, pB.unsqueeze(2)).squeeze(2)     # (B, F)
        Wpa = torch.bmm(W_sym, pA.unsqueeze(2)).squeeze(2)     # (B, F)
        contrib_A = pA * Wpb                                    # (B, F)
        contrib_B = pB * Wpa                                    # (B, F)
        kern_repr = torch.cat([contrib_A, contrib_B], dim=-1)  # (B, 2F)

        # ── Same-function-node overlap ─────────────────────────────────────
        # same_fn > 0 (γ>0): redundancy — SynergyHead's learned negative weight penalises
        # same_fn < 0 (γ<0): vertical synergy bonus
        same_fn = (pA * pB * self.gamma).sum(dim=-1, keepdim=True)   # (B, 1)

        return kern_repr, same_fn

    def interaction_scores(
        self,
        pA: torch.Tensor,      # (B, F)
        pB: torch.Tensor,      # (B, F)
        z_cell: torch.Tensor,  # (B, cell_ctx_dim)
        cell_row_idx: "torch.Tensor | None" = None,
    ) -> dict[str, torch.Tensor]:
        """Interpretability readout: decompose the bilinear pathway interaction
        pAᵀ W_sym pB into diagonal (same-community) and off-diagonal
        (cross-community / vertical) parts.  No effect on training.

        Returns dict with (B,1) tensors: total, diag, cross.
        cross > 0 → the model assigns positive cross-pathway interaction between
        the two drugs' *different* communities (vertical-synergy signal).
        """
        B, F = pA.shape
        with torch.no_grad():
            dW = self._delta_w(z_cell, cell_row_idx, apply_dropout=False)
            W = self.W_base.unsqueeze(0) + dW
            if self.w_prior is not None:
                W = W + self.w_prior_scale * self.w_prior
            if self.cell_specific_w_prior:
                assert cell_row_idx is not None, "cell_specific_w_prior=True requires cell_row_idx"
                wp_cell = self.w_prior_percell[cell_row_idx]
                gamma = self.film_gamma(z_cell)
                beta = self.film_beta(z_cell)
                gate = (1 + gamma).unsqueeze(2) * (1 + gamma).unsqueeze(1)
                shift = beta.unsqueeze(2) + beta.unsqueeze(1)
                prior_raw = gate * wp_cell + shift
                prior_norm = prior_raw.norm(dim=(1, 2), keepdim=True)
                prior_term = prior_raw * (self.tau_prior * torch.tanh(prior_norm / self.tau_prior)
                                           / (prior_norm + 1e-8))
                W = W + prior_term
            W_sym = (W + W.transpose(1, 2)) / 2                 # (B, F, F)
            Wpb = torch.bmm(W_sym, pB.unsqueeze(2)).squeeze(2)  # (B, F)
            total = (pA * Wpb).sum(dim=-1, keepdim=True)        # pAᵀ W pB
            diag_w = torch.diagonal(W_sym, dim1=1, dim2=2)      # (B, F)
            diag = (pA * diag_w * pB).sum(dim=-1, keepdim=True)
            cross = total - diag
        return {"total": total, "diag": diag, "cross": cross}

    def nuclear_norm_loss(self) -> torch.Tensor:
        """Nuclear norm of W_base for sparse-interaction regularisation."""
        return torch.linalg.matrix_norm(self.W_base, ord="nuc")
