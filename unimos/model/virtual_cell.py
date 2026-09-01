"""
virtual_cell.py — VirtualCellModule for UniMoS-VC (design doc §2.1, §4).

Phase 1 (entrance one only): FoundationCellEncoder is a frozen *lookup* into
precomputed scFoundation cell embeddings (unimos/data/precompute_fm_emb.py),
not a live forward pass through the transformer — see design doc §2.1
"落地建议". FuseMLP merges it with the existing functional-activity vector
`c` and the existing raw-expression residual context `h_cell` into a single
`z_cell` that now conditions both the pathway kernel and the RI/gate tracks
(design doc §4: RIEncoder / SynergyHead gate / PathwayAggregationKernel all
take z_cell instead of the old 67-dim `c`).

s_struct (§2.2, per-cell structural summary of W_prior(cell)) does not exist
yet — Phase 2. FuseMLP's input list is additive; Phase 2 adds a third input
without changing FoundationCellEncoder or this module's existing behaviour.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class FoundationCellEncoder(nn.Module):
    """
    Frozen lookup of precomputed scFoundation cell embeddings + a trainable
    projection MLP. Row order of the precomputed .npy must match
    data/processed/cell_feature_index.json (same convention as
    cell_fn_vectors.npy) — the caller passes the same integer row index
    UniMoSDataset already resolves via cell_to_row.

    Per-dimension z-scored at load time (mean/std computed once over the
    fixed 168-cell table, not fold-aware — this is a marginal statistic of
    the embedding table itself, not a label-conditional aggregate, so it
    carries the same low leakage risk as e.g. the Morgan-fingerprint lookup,
    not the train-fold-only HVG/w_prior fitting CLAUDE.md's data discipline
    is about). Motivation: raw scFoundation output has large per-dimension
    means (up to ~4.3) against small per-dimension std (0.03-0.22) — most of
    each dimension's magnitude is a constant offset shared across all 168
    cells, with the actual cross-cell signal riding on a small residual.
    Feeding that raw scale into a freshly-initialised Linear leaves the
    informative residual variance swamped by the constant component,
    observed to correlate with unstable/inconsistent downstream training
    (see PHASE1_GATE.md).

    Input  : cell_row_idx (B,) long
    Output : z_fm (B, out_dim)
    """

    def __init__(self, emb_path: str, out_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        emb = np.load(emb_path).astype(np.float32)
        mean = emb.mean(axis=0, keepdims=True)
        std = emb.std(axis=0, keepdims=True) + 1e-6
        emb = (emb - mean) / std
        self.register_buffer("emb_table", torch.from_numpy(emb))
        self.proj = _mlp(emb.shape[1], out_dim, out_dim, dropout)

    def forward(self, cell_row_idx: torch.Tensor) -> torch.Tensor:
        emb = self.emb_table[cell_row_idx]  # frozen; no grad needed on the table itself
        return self.proj(emb)


class FuseMLP(nn.Module):
    """cat(z_fm, c, h_cell) -> z_cell (design doc §2.1: "z_cell = FuseMLP([z_fm ; c ; s_struct])",
    s_struct not yet available — h_cell (existing raw-expression residual context)
    fills its slot until Phase 2.
    """

    def __init__(
        self,
        z_fm_dim: int,
        c_dim: int,
        h_cell_dim: int,
        out_dim: int = 128,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = _mlp(z_fm_dim + c_dim + h_cell_dim, hidden, out_dim, dropout)

    def forward(self, z_fm: torch.Tensor, c: torch.Tensor, h_cell: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_fm, c, h_cell], dim=-1))


class FilmRescueModule(nn.Module):
    """
    Gate-2 structural rescue for z_fm (2026-07-22, PHASE1_GATE.md §9): a
    literal SynCell-style FiLM modulation of the drug function vectors
    pA/pB, driven ONLY by z_fm — architecturally distinct from
    VirtualCellModule/PathwayAggregationKernel's existing hyper_U/V
    low-rank-ΔW-on-the-kernel mechanism (already end-to-end tested and
    failed G-VC, see PHASE1_GATE.md §1-3, §5b). This module leaves that
    mechanism and the rest of the v2 core untouched (meant to be used with
    use_virtual_cell=False, c_fn still drives the kernel as before) so any
    effect measured here is attributable to this one new mechanism only.

        [γ_A, β_A, γ_B, β_B] = MLP(z_fm)     (SynCell: MLP(concat[h_dA,h_dB,h_c]))
        pA' = pA ⊙ (1 + γ_A) + β_A
        pB' = pB ⊙ (1 + γ_B) + β_B

    Last layer is zero-initialised so the module is an identity map
    (pA'=pA, pB'=pB) at init, matching the Frobenius-gated ΔW's "start
    near the unconditioned baseline" discipline in PathwayAggregationKernel.

    `emb_path` may point to the real cell_fm_emb.npy (rescue) or a
    once-permuted copy of it (parameter-matched control, §9.2) — same
    architecture and parameter count either way, only the frozen table's
    row↔cell correspondence differs.
    """

    def __init__(self, emb_path: str, fn_dim: int = 67, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        emb = np.load(emb_path).astype(np.float32)
        mean = emb.mean(axis=0, keepdims=True)
        std = emb.std(axis=0, keepdims=True) + 1e-6
        emb = (emb - mean) / std
        self.register_buffer("emb_table", torch.from_numpy(emb))
        self.fn_dim = fn_dim
        self.net = nn.Sequential(
            nn.Linear(emb.shape[1], hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4 * fn_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pA: torch.Tensor, pB: torch.Tensor, cell_row_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_fm = self.emb_table[cell_row_idx]
        gA, bA, gB, bB = self.net(z_fm).split(self.fn_dim, dim=-1)
        pA2 = pA * (1 + gA) + bA
        pB2 = pB * (1 + gB) + bB
        return pA2, pB2


class VirtualCellModule(nn.Module):
    """
    z_cell = FuseMLP([FoundationCellEncoder(cell_row_idx) ; c ; h_cell])

    Parameters
    ----------
    fm_emb_path : path to precomputed cell_fm_emb.npy (unimos/data/precompute_fm_emb.py)
    fn_dim      : c's dimensionality (67, functional activity)
    cell_hidden : h_cell's dimensionality (CellEncoder output, default 128)
    fm_out_dim  : FoundationCellEncoder projection output dim (default 128)
    z_cell_dim  : final fused cell-context dim (default 128 — matches the old
                  h_cell dim so downstream modules' input size is unchanged
                  whether or not the virtual-cell branch is enabled)
    """

    def __init__(
        self,
        fm_emb_path: str,
        fn_dim: int = 67,
        cell_hidden: int = 128,
        fm_out_dim: int = 128,
        z_cell_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fm_encoder = FoundationCellEncoder(fm_emb_path, out_dim=fm_out_dim, dropout=dropout)
        self.fuse = FuseMLP(
            fm_out_dim, fn_dim, cell_hidden, out_dim=z_cell_dim, dropout=dropout
        )

    def forward(
        self, cell_row_idx: torch.Tensor, c: torch.Tensor, h_cell: torch.Tensor
    ) -> torch.Tensor:
        z_fm = self.fm_encoder(cell_row_idx)
        return self.fuse(z_fm, c, h_cell)
