"""
encoders.py — Feature encoders for UniMoS.

SPEC-01  MLP Base Encoders
  _mlp             : reusable two-layer MLP factory (Linear → LayerNorm → GELU → Dropout → Linear)
  StructureEncoder : Morgan FP (fp_dim=2048) → structural embedding (struct_hidden=256)
  CellEncoder      : Raw HVG expression (n_hvg=150) → cell residual context (cell_hidden=128)

SPEC-02  RIEncoder, RIHead
  RIEncoder : cat(p:67, z_cell:cell_ctx_dim=128) → ri_repr (B, ri_dim=64)
  RIHead    : ri_repr → ŷ_ri (B,)  auxiliary RI regression

SPEC-04  SynergyHead
  SynergyHead : (kern_repr, same_fn, h_struct, z_cell, ri_repr_A, ri_repr_B)
                → {logit_class, core_logit, resid_logit, alpha}

UniMoS-VC (design doc §4): RIEncoder and SynergyHead's cell-context input is
now z_cell (VirtualCellModule's fused output), replacing the old separate
c(67-dim functional activity) + h_cell(128-dim raw-expression context) pair.
CellEncoder is unchanged in isolation — it still produces h_cell from
c_raw — but h_cell is now only consumed as one of z_cell's fusion inputs
(unimos/model/virtual_cell.py::FuseMLP), not passed standalone downstream.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Shared building block ─────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    """Two-layer MLP: Linear → LayerNorm → GELU → Dropout → Linear."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


# ── SPEC-01: MLP Base Encoders ────────────────────────────────────────────────

class StructureEncoder(nn.Module):
    """
    Morgan fingerprint → structural embedding.

    Input  : fp  (B, fp_dim)   float32 ∈ {0, 1}
    Output : h   (B, out_dim)
    """

    def __init__(
        self,
        fp_dim: int = 2048,
        hidden: int = 256,
        out_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = _mlp(fp_dim, hidden, out_dim, dropout)

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return self.net(fp)


class CellEncoder(nn.Module):
    """
    Raw HVG log1p expression → cell residual context vector.

    Input  : c_raw  (B, n_hvg)   float32, fold-internal HVG
    Output : h_cell (B, out_dim)
    """

    def __init__(
        self,
        n_hvg: int = 150,
        hidden: int = 128,
        out_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = _mlp(n_hvg, hidden, out_dim, dropout)

    def forward(self, c_raw: torch.Tensor) -> torch.Tensor:
        return self.net(c_raw)


# ── SPEC-02: RIEncoder + RIHead ───────────────────────────────────────────────

class RIEncoder(nn.Module):
    """
    Single-drug response encoder.

    Input  : cat(p:fn_dim, z_cell:cell_ctx_dim)  →  ri_repr (B, ri_dim)
    p      : drug functional-node perturbation vector
    z_cell : fused cell context (VirtualCellModule output)
    """

    def __init__(
        self,
        fn_dim: int = 67,
        cell_ctx_dim: int = 128,
        ri_dim: int = 64,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fn_dim = fn_dim
        self.cell_ctx_dim = cell_ctx_dim
        self.net = _mlp(fn_dim + cell_ctx_dim, hidden, ri_dim, dropout)

    def forward(self, p: torch.Tensor, z_cell: torch.Tensor) -> torch.Tensor:
        if p.shape[-1] != self.fn_dim:
            raise ValueError(
                f"RIEncoder: p expected last dim {self.fn_dim}, got {p.shape[-1]}"
            )
        if z_cell.shape[-1] != self.cell_ctx_dim:
            raise ValueError(
                f"RIEncoder: z_cell expected last dim {self.cell_ctx_dim}, got {z_cell.shape[-1]}"
            )
        return self.net(torch.cat([p, z_cell], dim=-1))


class RIHead(nn.Module):
    """
    Auxiliary RI regression head.

    Input  : ri_repr (B, ri_dim)
    Output : ŷ_ri   (B,)
    """

    def __init__(self, ri_dim: int = 64) -> None:
        super().__init__()
        self.fc = nn.Linear(ri_dim, 1)

    def forward(self, ri_repr: torch.Tensor) -> torch.Tensor:
        return self.fc(ri_repr).squeeze(-1)


# ── SPEC-04: SynergyHead ──────────────────────────────────────────────────────

class SynergyHead(nn.Module):
    """
    Final classification head combining core-kernel and residual tracks.

    logit_class = core_logit + α · resid_logit

    Core track  — interpretable, directly supervised:
      core_input  = cat(kern_repr:kern_dim, same_fn:1)    (B, kern_dim+1 = 135)
      core_logit  = Linear(135 → 1).squeeze()             (B,)

    Residual track — structural/cell context, α-gated:
      ri_sum       = ri_repr_A + ri_repr_B                 (B, ri_dim)   symmetric
      ri_prod      = ri_repr_A * ri_repr_B                 (B, ri_dim)   symmetric
      resid_input  = cat(h_struct, z_cell, ri_sum, ri_prod) (B, 512)
      resid_logit  = MLP(512 → resid_hidden → 1).squeeze() (B,)

    Gate (UniMoS-VC §4: scalar → z_cell-conditioned):
      gate_logit   = Linear(cell_ctx_dim → 1)(z_cell)  →  α = sigmoid(gate_logit)  (B,)
      logit_class  = core_logit + α · resid_logit                                  (B,)
      Pre-VC this was a single learned scalar (init sigmoid(-3)≈0.047, same
      residual-dormant-at-init behaviour reproduced here via the Linear's bias init).

    A/B symmetry (full-model invariance under pA↔pB):
      Core track  — kern_repr halves swap when pA↔pB, so we decompose:
                    C_sum = C_A + C_B  (symmetric)
                    C_diff = |C_A - C_B|  (symmetric)
                    core_input = cat(C_sum, C_diff, same_fn)   dim = 67+67+1 = 135
      Residual track — ri_sum and ri_prod are symmetric under ri_repr_A↔ri_repr_B.
    """

    def __init__(
        self,
        kern_dim: int = 134,
        h_struct_dim: int = 256,
        cell_ctx_dim: int = 128,
        ri_dim: int = 64,
        resid_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kern_dim = kern_dim
        self.ri_dim = ri_dim

        # Core track: single linear layer (interpretable)
        self.core_head = nn.Linear(kern_dim + 1, 1)   # input dim = 135

        # Residual track: MLP over symmetric RI + structural + cell context
        # cat(h_struct:256, z_cell:128, ri_sum:64, ri_prod:64) = 512
        resid_in = h_struct_dim + cell_ctx_dim + ri_dim * 2
        self.resid_head = _mlp(resid_in, resid_hidden, 1, dropout)

        # α gate — z_cell-conditioned (UniMoS-VC §4). Bias init -3.0 reproduces
        # the pre-VC sigmoid(-3)≈0.047 residual-dormant-at-init behaviour;
        # weight zero-init means α starts purely at that constant regardless
        # of z_cell, then differentiates per-sample as training progresses.
        self.gate = nn.Linear(cell_ctx_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(
        self,
        kern_repr: torch.Tensor,    # (B, kern_dim=134)
        same_fn: torch.Tensor,      # (B, 1)
        h_struct: torch.Tensor,     # (B, 256)
        z_cell: torch.Tensor,       # (B, cell_ctx_dim=128)
        ri_repr_A: torch.Tensor,    # (B, ri_dim=64)
        ri_repr_B: torch.Tensor,    # (B, ri_dim=64)
        ablation_mode: str = "none",
    ) -> dict[str, torch.Tensor]:
        # Core track — symmetric decomposition of kern_repr halves
        F = self.kern_dim // 2                                       # 67
        C_A, C_B = kern_repr[:, :F], kern_repr[:, F:]
        C_sum  = C_A + C_B                                          # (B, 67) symmetric
        C_diff = (C_A - C_B).abs()                                  # (B, 67) symmetric
        core_input = torch.cat([C_sum, C_diff, same_fn], dim=-1)    # (B, 135)
        core_logit = self.core_head(core_input).squeeze(-1)         # (B,)

        # Residual track — symmetric RI combination
        ri_sum  = ri_repr_A + ri_repr_B                             # (B, 64)
        ri_prod = ri_repr_A * ri_repr_B                             # (B, 64)
        resid_input = torch.cat(
            [h_struct, z_cell, ri_sum, ri_prod], dim=-1
        )                                                           # (B, 512)
        resid_logit = self.resid_head(resid_input).squeeze(-1)      # (B,)

        # Gated combination — per-sample α(z_cell)
        alpha = torch.sigmoid(self.gate(z_cell)).squeeze(-1)         # (B,)
        logit_class = core_logit + alpha * resid_logit              # (B,)

        # NBE selling-point training ablations (applied after both tracks run so
        # graphs stay comparable; gradients only flow through the kept track).
        mode = (ablation_mode or "none").lower()
        if mode.endswith("_no_deltaw") and mode != "no_deltaw":
            mode = mode[: -len("_no_deltaw")]
        if mode in ("none", "full", "no_deltaw", "zero_p"):
            pass
        elif mode == "core_only":
            alpha = torch.zeros_like(alpha)
            logit_class = core_logit
        elif mode == "resid_only":
            core_logit = torch.zeros_like(core_logit)
            logit_class = resid_logit
        else:
            raise ValueError(f"unknown ablation_mode={ablation_mode!r}")

        return {
            "logit_class": logit_class,
            "core_logit":  core_logit,
            "resid_logit": resid_logit,
            "alpha":       alpha,
            # Shared features for optional multi-metric synergy regression
            # (loewe / zip / hsa / bliss). cat(core_input, resid_input).
            "syn_feat":    torch.cat([core_input, resid_input], dim=-1),
        }


class SynergyMetricHead(nn.Module):
    """
    Auxiliary multi-output regression head for continuous synergy scores
    (loewe / zip / hsa / bliss).  Operates on SynergyHead.syn_feat and
    predicts *standardised* targets (train-set mean/std applied in the loss).
    """

    def __init__(
        self,
        in_dim: int,
        n_targets: int = 4,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, syn_feat: torch.Tensor) -> torch.Tensor:
        """Return (B, n_targets) standardised score predictions."""
        return self.net(syn_feat)
