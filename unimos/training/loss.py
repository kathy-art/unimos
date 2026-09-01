"""
loss.py — UniMoS multi-task loss (SPEC-06).

Formula
-------
L = λ_cls · BCE_pw(logit_class, y_class)
  + λ_ri   · [Huber(ŷ_ri_A, y_ri_A, δ=5) + Huber(ŷ_ri_B, y_ri_B, δ=5)]
  + λ_syn  · Σ_k w_k · Huber(ŷ_syn_k, (y_syn_k - μ_k)/σ_k, δ=1)   # optional
  + λ_res  · mean(h_struct²)
  + λ_gate · mean(alpha²)   # alpha is per-sample (B,) since UniMoS-VC's z_cell-conditioned gate
  + λ_W    · nuclear_norm(W_base)
  + λ_γ    · mean(gamma²)

NaN rules
---------
- y_class NaN  → row excluded from BCE
- y_ri_A/B NaN → row excluded from that Huber term
- y_syn NaN per-metric → that cell excluded from syn_reg Huber
- all-NaN batch for one label → that term returns 0 (regularisers unaffected)

pos_weight
----------
Pass (n_neg / n_pos) from the training split at construction time.
Not part of HPO — computed once before training starts.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniMoSLoss(nn.Module):
    """
    Parameters
    ----------
    pos_weight : float
        Class-imbalance weight = n_neg / n_pos from the training split.
    lambda_cls  : weight for classification BCE/focal (default 1.0; <1 for
        regression-primary runs)
    lambda_ri   : weight for RI Huber terms (default 0.1)
    lambda_syn_reg : weight for continuous synergy-metric regression (default 0)
    lambda_res  : weight for h_struct L2 regularisation (default 1e-4)
    lambda_gate : weight for alpha² regularisation (default 1e-3)
    lambda_W    : weight for W_base nuclear-norm regularisation (default 1e-4)
    lambda_gamma: weight for gamma L2 regularisation (default 1e-4)
    huber_delta : δ for Huber loss on RI regression (default 5.0)
    syn_huber_delta : δ for standardised synergy-metric Huber (default 1.0)
    syn_metric_weights : per-metric weights (loewe/zip/hsa/bliss).  Renormalised
        to sum to 1 over metrics with finite predictions; None → equal weights.
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_ri: float = 0.1,
        lambda_syn_reg: float = 0.0,
        lambda_res: float = 1e-4,
        lambda_gate: float = 1e-3,
        lambda_W: float = 1e-4,
        lambda_gamma: float = 1e-4,
        huber_delta: float = 5.0,
        syn_huber_delta: float = 1.0,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        syn_metric_mean: Optional[list[float]] = None,
        syn_metric_std: Optional[list[float]] = None,
        syn_metric_weights: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        if pos_weight < 0:
            raise ValueError(f"pos_weight must be non-negative, got {pos_weight}")
        if lambda_cls < 0:
            raise ValueError(f"lambda_cls must be non-negative, got {lambda_cls}")
        if loss_type not in ("bce", "focal"):
            raise ValueError(f"loss_type must be 'bce' or 'focal', got {loss_type}")
        self.register_buffer("_pos_weight", torch.tensor(pos_weight))
        self.lambda_cls   = lambda_cls
        self.lambda_ri    = lambda_ri
        self.lambda_syn_reg = lambda_syn_reg
        self.lambda_res   = lambda_res
        self.lambda_gate  = lambda_gate
        self.lambda_W     = lambda_W
        self.lambda_gamma = lambda_gamma
        self.huber_delta  = huber_delta
        self.syn_huber_delta = syn_huber_delta
        self.loss_type    = loss_type
        self.focal_gamma  = focal_gamma
        if syn_metric_mean is not None and syn_metric_std is not None:
            self.register_buffer(
                "_syn_mean", torch.tensor(syn_metric_mean, dtype=torch.float32)
            )
            self.register_buffer(
                "_syn_std", torch.tensor(syn_metric_std, dtype=torch.float32).clamp_min(1e-3)
            )
        else:
            self._syn_mean = None
            self._syn_std = None
        if syn_metric_weights is not None:
            w = torch.tensor(syn_metric_weights, dtype=torch.float32)
            if (w < 0).any():
                raise ValueError(f"syn_metric_weights must be >= 0, got {syn_metric_weights}")
            self.register_buffer("_syn_w", w)
        else:
            self._syn_w = None

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _nan_mask(t: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(t)

    def _bce(
        self,
        logit: torch.Tensor,   # (B,)
        y: torch.Tensor,       # (B,)
    ) -> torch.Tensor:
        mask = self._nan_mask(y)
        if not mask.any():
            return logit.sum() * 0.0
        l, t = logit[mask], y[mask]
        if self.loss_type == "focal":
            # Focal loss: pos_weight acts as the class-balancing alpha,
            # (1 - p_t)^gamma down-weights easy examples.
            ce = F.binary_cross_entropy_with_logits(
                l, t, pos_weight=self._pos_weight, reduction="none",
            )
            p   = torch.sigmoid(l)
            p_t = p * t + (1.0 - p) * (1.0 - t)
            return ((1.0 - p_t).clamp_min(1e-6) ** self.focal_gamma * ce).mean()
        return F.binary_cross_entropy_with_logits(
            l, t, pos_weight=self._pos_weight,
        )

    def _huber(
        self,
        pred: torch.Tensor,    # (B,)
        target: torch.Tensor,  # (B,)
        delta: "float | None" = None,
    ) -> torch.Tensor:
        mask = self._nan_mask(target)
        if not mask.any():
            return pred.sum() * 0.0
        d = self.huber_delta if delta is None else delta
        return F.huber_loss(pred[mask], target[mask], delta=d)

    def _syn_reg(
        self,
        yhat_syn: "torch.Tensor | None",  # (B, K) standardised preds
        y_syn: "torch.Tensor | None",     # (B, K) raw continuous scores
    ) -> torch.Tensor:
        """Weighted mean of per-metric Huber on standardised targets (NaN-safe)."""
        if (
            yhat_syn is None
            or y_syn is None
            or self.lambda_syn_reg == 0.0
            or self._syn_mean is None
            or self._syn_std is None
        ):
            # Keep graph connected when the head exists but loss is off.
            if yhat_syn is not None:
                return yhat_syn.sum() * 0.0
            return torch.tensor(0.0)

        mean = self._syn_mean.to(dtype=y_syn.dtype, device=y_syn.device)
        std = self._syn_std.to(dtype=y_syn.dtype, device=y_syn.device)
        y_std = (y_syn - mean) / std
        k = yhat_syn.shape[-1]
        if self._syn_w is None:
            weights = torch.ones(k, dtype=y_syn.dtype, device=y_syn.device)
        else:
            weights = self._syn_w.to(dtype=y_syn.dtype, device=y_syn.device)
            if weights.numel() != k:
                raise ValueError(
                    f"syn_metric_weights length {weights.numel()} != n_metrics {k}"
                )

        losses = []
        used_w = []
        for i in range(k):
            w_i = weights[i]
            if float(w_i) <= 0.0:
                continue
            mask = torch.isfinite(y_std[:, i]) & torch.isfinite(yhat_syn[:, i])
            if not mask.any():
                continue
            losses.append(
                F.huber_loss(
                    yhat_syn[mask, i], y_std[mask, i], delta=self.syn_huber_delta,
                )
            )
            used_w.append(w_i)

        if not losses:
            return yhat_syn.sum() * 0.0
        w_stack = torch.stack(used_w)
        w_stack = w_stack / w_stack.sum().clamp_min(1e-8)
        return sum(w * L for w, L in zip(w_stack, losses))

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        *,
        logit_class: torch.Tensor,   # (B,)
        y_class: torch.Tensor,       # (B,)
        yhat_ri_A: torch.Tensor,     # (B,)
        yhat_ri_B: torch.Tensor,     # (B,)
        y_ri_A: torch.Tensor,        # (B,)
        y_ri_B: torch.Tensor,        # (B,)
        h_struct: torch.Tensor,      # (B, 256)
        alpha: torch.Tensor,         # (B,) — UniMoS-VC: per-sample, was scalar pre-VC
        W_base: torch.Tensor,        # (67, 67)
        gamma: torch.Tensor,         # (67,)
        yhat_syn: "torch.Tensor | None" = None,  # (B, K)
        y_syn: "torch.Tensor | None" = None,     # (B, K)
    ) -> dict[str, torch.Tensor]:
        bce_loss = self._bce(logit_class, y_class)
        ri_loss  = self._huber(yhat_ri_A, y_ri_A) + self._huber(yhat_ri_B, y_ri_B)
        syn_loss = self._syn_reg(yhat_syn, y_syn)

        reg_res   = h_struct.pow(2).mean()
        reg_gate  = alpha.pow(2).mean()
        reg_W     = torch.linalg.matrix_norm(W_base, ord="nuc")
        reg_gamma = gamma.pow(2).mean()

        loss = (
            self.lambda_cls   * bce_loss
            + self.lambda_ri      * ri_loss
            + self.lambda_syn_reg * syn_loss
            + self.lambda_res     * reg_res
            + self.lambda_gate    * reg_gate
            + self.lambda_W       * reg_W
            + self.lambda_gamma   * reg_gamma
        )

        return {
            "loss":     loss,
            "bce_loss": bce_loss,
            "ri_loss":  ri_loss,
            "syn_loss": syn_loss,
            "reg_res":  reg_res,
            "reg_gate": reg_gate,
            "reg_W":    reg_W,
            "reg_gamma":reg_gamma,
        }
