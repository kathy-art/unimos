"""
unimos.py — UniMoS-VC full model (SPEC-05 + design doc §4 VC refactor).

Assembles SPEC-01~04 sub-modules (+ VirtualCellModule) into a LightningModule
that owns the multi-task loss (SPEC-06) and logs val_auroc for Lightning
callbacks.

Forward (strict order):
  1.  c          = c_fn + alpha_mut * c_mut_fn          (B, 67)
  2.  h_struct   = struct_enc(fp_A) + struct_enc(fp_B)  (B, 256)
  3.  h_cell     = cell_enc(c_raw)                       (B, 128)
  3b. z_cell     = virtual_cell(cell_row_idx, c, h_cell) (B, z_cell_dim=128)
                   if use_virtual_cell else h_cell (fallback; same dim,
                   reproduces pre-VC behaviour when the branch is disabled)
  4.  ri_repr_A  = ri_enc(pA, z_cell)                   (B, 64)
      yhat_ri_A  = ri_head(ri_repr_A)                   (B,)
      ri_repr_B  = ri_enc(pB, z_cell)                   (B, 64)
      yhat_ri_B  = ri_head(ri_repr_B)                   (B,)
  5.  kern_repr, same_fn = kernel(pA, pB, z_cell)        (B,134), (B,1)
  6.  out        = synergy_head(kern_repr, same_fn,
                                h_struct, z_cell,
                                ri_repr_A, ri_repr_B)

Output dict keys: logit_class, core_logit, resid_logit, alpha,
                  yhat_ri_A, yhat_ri_B, h_struct, W_base, gamma

UniMoS-VC (design doc §4): c and h_cell no longer reach RIEncoder/kernel/
SynergyHead directly — they are folded into z_cell via VirtualCellModule
(unimos/model/virtual_cell.py). z_cell_dim defaults to cell_hidden (128) so
disabling use_virtual_cell (default) reproduces the exact pre-VC dataflow,
just with h_cell standing in for z_cell everywhere.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import expit as _expit
import torch
import torch.nn as nn
from lightning import LightningModule

from unimos.model.encoders import (
    CellEncoder,
    RIEncoder,
    RIHead,
    StructureEncoder,
    SynergyHead,
    SynergyMetricHead,
)
from unimos.model.pathway_kernel import PathwayAggregationKernel
from unimos.model.virtual_cell import FilmRescueModule, VirtualCellModule
from unimos.training.loss import UniMoSLoss
from unimos.training.metrics import compute_metrics


class UniMoS(LightningModule):
    """
    UniMoS: interpretable drug-synergy prediction model.

    Parameters
    ----------
    fn_dim        : function-node count (default 67)
    fp_dim        : Morgan fingerprint length (default 2048)
    n_hvg         : HVG expression features per fold (default 150)
    struct_hidden : StructureEncoder hidden / output dim (default 256)
    cell_hidden   : CellEncoder hidden / output dim (default 128)
    ri_dim        : RIEncoder output dim (default 64)
    rank_r        : PathwayAggregationKernel low-rank dim (default 16)
    resid_hidden  : SynergyHead residual MLP hidden dim (default 256)
    dropout       : shared dropout rate (default 0.1)
    lr            : base learning rate for AdamW (default 1e-3)
    max_epochs    : T_max for CosineAnnealingLR (default 200)
    pos_weight    : BCE pos_weight for class imbalance (default 1.0)
    lambda_ri     : weight for RI Huber loss (default 0.1)
    lambda_res    : weight for h_struct L2 regularisation (default 1e-4)
    lambda_gate   : weight for alpha² regularisation (default 1e-3)
    lambda_W      : weight for W_base nuclear-norm regularisation (default 1e-4)
    lambda_gamma  : weight for gamma L2 regularisation (default 1e-4)
    use_virtual_cell : enable VirtualCellModule (design doc §4, entrance one); default
                       False reproduces pre-VC dataflow exactly (h_cell stands in for z_cell)
    fm_emb_path      : path to precomputed cell_fm_emb.npy (required if use_virtual_cell)
    fm_out_dim       : FoundationCellEncoder projection output dim (default 128)
    z_cell_dim       : fused cell-context dim; must equal cell_hidden when
                       use_virtual_cell=False (default 128)
    """

    # ── Canonical batch key set (SPEC-05 Table) ───────────────────────────────
    # cell_row_idx: integer row index into cell_feature_index.json order, used
    # to look up the precomputed scFoundation embedding (VirtualCellModule);
    # always present in the batch, only consumed when use_virtual_cell=True.
    REQUIRED_KEYS = frozenset({
        "pA", "pB", "fp_A", "fp_B",
        "c_fn", "c_mut_fn", "c_raw", "cell_row_idx",
        "y_class", "y_ri_A", "y_ri_B",
    })

    def __init__(
        self,
        fn_dim: int = 67,
        fp_dim: int = 2048,
        n_hvg: int = 150,
        struct_hidden: int = 256,
        cell_hidden: int = 128,
        ri_dim: int = 64,
        rank_r: int = 16,
        resid_hidden: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_epochs: int = 200,
        pos_weight: float = 1.0,
        lambda_ri: float = 0.1,
        lambda_res: float = 1e-4,
        lambda_gate: float = 1e-3,
        lambda_W: float = 1e-4,
        lambda_gamma: float = 1e-4,
        w_prior_path: str = "",
        w_prior_scale: float = 0.0,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        struct_encoder: str = "fp",
        graph_cache: str = "",
        gnn_hidden: int = 256,
        gnn_layers: int = 3,
        use_descriptors: bool = False,
        desc_cache: str = "",
        desc_dim: int = 217,
        use_sensitivity_profile: bool = False,
        sens_drug_cache: str = "",
        sens_cell_cache: str = "",
        sens_dim: int = 6,
        use_virtual_cell: bool = False,
        fm_emb_path: str = "",
        fm_out_dim: int = 128,
        z_cell_dim: "int | None" = None,
        use_film_rescue: bool = False,
        film_rescue_emb_path: str = "",
        film_hidden: int = 64,
        cell_specific_w_prior: bool = False,
        w_prior_percell_path: str = "",
        tau_prior: "float | None" = None,
        gi_prior_path: str = "",
        gi_prior_init_scale: float = 0.0,
        ablation_mode: str = "none",
        cell_cond_mode: str = "hypernet",
        # FM path for fm_kernel_interp ΔW; falls back to fm_emb_path when empty
        fm_kernel_emb_path: str = "",
        fm_kernel_tau: float = 0.1,
        fm_kernel_self_mask: bool = True,
        # Drug-pair historical positive-rate prior (leakage-safe; see
        # UniMoSDataset use_pair_prior docstring). Additive in logit space via
        # a learnable scalar gate, zero-initialised so behaviour at init is
        # identical to use_pair_prior=False.
        use_pair_prior: bool = False,
        # k-NN (Signaturizer B1 space) generalisation of the pair prior —
        # works for drugs with zero train history (e.g. every LDO test
        # drug), unlike use_pair_prior. Same additive-logit-gate design.
        use_bio_prior: bool = False,
        # LR schedule: "cosine" (CosineAnnealingLR, T_max=max_epochs — the
        # long-standing default) or "plateau" (ReduceLROnPlateau on
        # val_auroc). cosine assumes training runs ~max_epochs; when a run
        # actually peaks/overfits far earlier (seed-dependent — observed
        # anywhere from ~epoch 8 to 20+), LR barely decays over that window
        # (cosine is flat near t=0), so the post-peak val_auroc noise is
        # partly just large-LR drift, not only overfitting. plateau adapts
        # per-run instead of assuming a fixed horizon.
        lr_scheduler_type: str = "cosine",
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        # Optional multi-task regression on continuous DrugComb synergy scores
        # (loewe / zip / hsa / bliss). Off by default — identical to production
        # checkpoints_vc when use_synergy_regression=False.
        use_synergy_regression: bool = False,
        lambda_syn_reg: float = 0.05,
        lambda_cls: float = 1.0,
        n_syn_metrics: int = 4,
        syn_metric_hidden: int = 128,
        syn_metric_mean: "list[float] | None" = None,
        syn_metric_std: "list[float] | None" = None,
        syn_metric_weights: "list[float] | None" = None,
        # Lightning monitor for early-stop / ModelCheckpoint when multi-task
        # regression is on.  Default stays val_auroc for backward compat.
        # Use val_zip_pearson for ZIP-primary runs.
        early_stop_metric: str = "val_auroc",
        syn_metric_names: "tuple[str, ...] | list[str]" = ("loewe", "zip", "hsa", "bliss"),
    ) -> None:
        super().__init__()
        # z_cell_dim=None -> default to cell_hidden, so every existing config
        # (many use cell_hidden=256, not the model default 128) keeps working
        # unchanged when use_virtual_cell=False, with no need to also set
        # z_cell_dim explicitly. Resolved before save_hyperparameters() so the
        # concrete value (not None) is what gets checkpointed.
        if z_cell_dim is None:
            z_cell_dim = cell_hidden
        self.save_hyperparameters()

        mode = (ablation_mode or "none").lower()
        # Composite *_no_deltaw modes keep track ablations while matching
        # production VC+ΔW− (checkpoints_vc): ΔW off.
        _valid = {
            "none", "full", "core_only", "resid_only", "no_deltaw", "zero_p",
            "core_only_no_deltaw", "resid_only_no_deltaw",
        }
        if mode not in _valid:
            raise ValueError(f"ablation_mode must be one of {_valid}, got {ablation_mode!r}")
        self.ablation_mode = mode
        self.use_delta_w = not mode.endswith("no_deltaw")

        # Optional biological vertical-synergy prior for the pathway kernel
        w_prior = None
        if w_prior_path and w_prior_scale != 0.0:
            import numpy as _np
            w_prior = torch.from_numpy(_np.load(w_prior_path)).float()

        self.lr         = lr
        self.max_epochs = max_epochs
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.early_stop_metric = early_stop_metric
        self.syn_metric_names = tuple(syn_metric_names)

        # ── Sub-modules ───────────────────────────────────────────────────────
        self.struct_encoder = struct_encoder
        if struct_encoder == "gnn":
            # GIN drug-structure encoder over molecular graphs (LDO generalisation).
            from unimos.model.gin_encoder import GINDrugEncoder
            cache = torch.load(graph_cache, weights_only=False)
            self._graph_bank = cache["graphs"]      # inchikey -> PyG Data (CPU, static)
            self._atom_dim = cache["dims"]["atom_dim"]
            self._bond_dim = cache["dims"]["bond_dim"]
            self.gin = GINDrugEncoder(
                atom_dim=self._atom_dim, bond_dim=self._bond_dim,
                hidden=gnn_hidden, out_dim=struct_hidden,
                n_layers=gnn_layers, dropout=dropout,
            )
            self._zero_graph = None
        else:
            self.struct_enc = StructureEncoder(
                fp_dim=fp_dim, hidden=struct_hidden, out_dim=struct_hidden,
                dropout=dropout,
            )
        self.cell_enc = CellEncoder(
            n_hvg=n_hvg, hidden=cell_hidden, out_dim=cell_hidden,
            dropout=dropout,
        )

        # VirtualCellModule (design doc §2.1/§4, entrance one). Disabled by
        # default: h_cell (dim cell_hidden) stands in for z_cell (dim
        # z_cell_dim) unchanged, so cell_hidden and z_cell_dim must match in
        # that case — enforced below rather than silently reshaping.
        self.use_virtual_cell = use_virtual_cell
        self.z_cell_dim = z_cell_dim
        if use_virtual_cell:
            self.virtual_cell = VirtualCellModule(
                fm_emb_path=fm_emb_path, fn_dim=fn_dim, cell_hidden=cell_hidden,
                fm_out_dim=fm_out_dim, z_cell_dim=z_cell_dim, dropout=dropout,
            )
        elif z_cell_dim != cell_hidden:
            raise ValueError(
                f"use_virtual_cell=False falls back to h_cell as z_cell; "
                f"z_cell_dim ({z_cell_dim}) must equal cell_hidden ({cell_hidden})"
            )

        # Gate-2 z_fm structural rescue (PHASE1_GATE.md §9): FiLM(z_fm) on
        # pA/pB, independent of use_virtual_cell — meant to be combined with
        # use_virtual_cell=False so the existing c_fn-driven kernel/z_cell
        # path is untouched and this is the only new mechanism under test.
        self.use_film_rescue = use_film_rescue
        if use_film_rescue:
            self.film_rescue = FilmRescueModule(
                emb_path=film_rescue_emb_path, fn_dim=fn_dim, hidden=film_hidden, dropout=dropout,
            )

        self.ri_enc = RIEncoder(
            fn_dim=fn_dim, cell_ctx_dim=z_cell_dim, ri_dim=ri_dim,
            dropout=dropout,
        )
        self.ri_head = RIHead(ri_dim=ri_dim)
        gi_prior = None
        if gi_prior_path and gi_prior_init_scale != 0.0:
            import numpy as _np
            gi_prior = torch.from_numpy(_np.load(gi_prior_path).astype(_np.float32))

        self.kernel = PathwayAggregationKernel(
            F_nodes=fn_dim, rank_r=rank_r, dropout=dropout,
            w_prior=w_prior, w_prior_scale=w_prior_scale,
            cell_ctx_dim=z_cell_dim,
            cell_specific_w_prior=cell_specific_w_prior,
            w_prior_percell_path=w_prior_percell_path,
            tau_prior=tau_prior,
            gi_prior=gi_prior,
            gi_prior_init_scale=gi_prior_init_scale,
            use_delta_w=self.use_delta_w,
            cell_cond_mode=cell_cond_mode,
            fm_emb_path=(fm_kernel_emb_path or fm_emb_path),
            fm_kernel_tau=fm_kernel_tau,
            fm_kernel_self_mask=fm_kernel_self_mask,
        )
        self.synergy_head = SynergyHead(
            kern_dim=fn_dim * 2,
            h_struct_dim=struct_hidden,
            cell_ctx_dim=z_cell_dim,
            ri_dim=ri_dim,
            resid_hidden=resid_hidden,
            dropout=dropout,
        )

        # Optional continuous synergy-metric regression (loewe/zip/hsa/bliss).
        # Input dim = core_input (135) + resid_input (h_struct+z_cell+2*ri_dim).
        self.use_synergy_regression = use_synergy_regression
        self.n_syn_metrics = n_syn_metrics
        if use_synergy_regression:
            syn_in = (fn_dim * 2 + 1) + (struct_hidden + z_cell_dim + ri_dim * 2)
            self.syn_metric_head = SynergyMetricHead(
                in_dim=syn_in,
                n_targets=n_syn_metrics,
                hidden=syn_metric_hidden,
                dropout=dropout,
            )

        # Optional RDKit physicochemical descriptor branch (added to h_struct)
        self.use_descriptors = use_descriptors
        if use_descriptors:
            import numpy as _np
            dz = _np.load(desc_cache, allow_pickle=True)
            self._desc_ik = {ik: i for i, ik in enumerate(dz["inchikeys"])}
            mat = torch.from_numpy(dz["desc"].astype("float32"))
            mat = torch.cat([mat, torch.zeros(1, mat.shape[1])], 0)  # last row = missing
            self.register_buffer("_desc_mat", mat)
            self.desc_enc = nn.Sequential(
                nn.Linear(desc_dim, struct_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(struct_hidden, struct_hidden),
            )

        # Optional "baseline sensitivity profile" branch — per-drug and
        # per-cell aggregate single-agent RI statistics (train-split-only,
        # NN-imputed for cold-start entities; see
        # opt/cell_sensitivity_profile/build_profiles.py). Drug profiles are
        # added into h_struct (per drug, like the descriptor branch); the
        # cell profile is added into h_cell.
        self.use_sensitivity_profile = use_sensitivity_profile
        if use_sensitivity_profile:
            import numpy as _np
            dz = _np.load(sens_drug_cache, allow_pickle=True)
            self._sens_drug_ik = {ik: i for i, ik in enumerate(dz["inchikeys"])}
            dmat = torch.from_numpy(dz["profile"].astype("float32"))
            dmat = torch.cat([dmat, torch.zeros(1, dmat.shape[1])], 0)  # last row = missing
            self.register_buffer("_sens_drug_mat", dmat)
            self.sens_drug_enc = nn.Sequential(
                nn.Linear(sens_dim, struct_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(struct_hidden, struct_hidden),
            )

            cz = _np.load(sens_cell_cache, allow_pickle=True)
            self._sens_cell_ik = {cid: i for i, cid in enumerate(cz["cell_ids"])}
            cmat = torch.from_numpy(cz["profile"].astype("float32"))
            cmat = torch.cat([cmat, torch.zeros(1, cmat.shape[1])], 0)  # last row = missing
            self.register_buffer("_sens_cell_mat", cmat)
            self.sens_cell_enc = nn.Sequential(
                nn.Linear(sens_dim, cell_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(cell_hidden, cell_hidden),
            )

        # Learnable mutation-activity mixing scalar (SPEC-05)
        self.alpha_mut = nn.Parameter(torch.ones(1))

        # Drug-pair historical prior gate (zero-init -> no-op until learned)
        self.use_pair_prior = use_pair_prior
        if use_pair_prior:
            self.pair_prior_gamma = nn.Parameter(torch.zeros(1))

        self.use_bio_prior = use_bio_prior
        if use_bio_prior:
            self.bio_prior_gamma = nn.Parameter(torch.zeros(1))

        # Multi-task loss — owns pos_weight buffer, no trainable params
        self.loss_fn = UniMoSLoss(
            pos_weight  = pos_weight,
            lambda_cls  = lambda_cls,
            lambda_ri   = lambda_ri,
            lambda_syn_reg = lambda_syn_reg if use_synergy_regression else 0.0,
            lambda_res  = lambda_res,
            lambda_gate = lambda_gate,
            lambda_W    = lambda_W,
            lambda_gamma= lambda_gamma,
            loss_type   = loss_type,
            focal_gamma = focal_gamma,
            syn_metric_mean = syn_metric_mean if use_synergy_regression else None,
            syn_metric_std  = syn_metric_std if use_synergy_regression else None,
            syn_metric_weights = syn_metric_weights if use_synergy_regression else None,
        )

        # Validation output accumulation (single-GPU)
        self._val_outputs: list[dict] = []

    # ── GNN graph batching helper ─────────────────────────────────────────────

    def _graphs_for(self, inchikeys, device):
        from torch_geometric.data import Batch
        if self._zero_graph is None:
            from torch_geometric.data import Data
            self._zero_graph = Data(
                x=torch.zeros(1, self._atom_dim), edge_index=torch.zeros(2, 1, dtype=torch.long),
                edge_attr=torch.zeros(1, self._bond_dim))
        datas = [self._graph_bank.get(ik, self._zero_graph) for ik in inchikeys]
        return Batch.from_data_list(datas).to(device)

    def set_train_cell_mask(self, train_cell_rows: "list[int] | torch.Tensor") -> None:
        """LCO-safe dictionary for fm_kernel_interp (no-op under hypernet)."""
        self.kernel.set_train_cell_mask(train_cell_rows)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pA, pB     = batch["pA"], batch["pB"]
        fp_A, fp_B = batch["fp_A"], batch["fp_B"]
        c_fn       = batch["c_fn"]
        c_mut_fn   = batch["c_mut_fn"]
        c_raw      = batch["c_raw"]

        # 0. Gate-2 z_fm structural rescue: FiLM(z_fm) modulates pA/pB before
        # anything downstream (kernel, RIEncoder) consumes them. h_struct
        # uses fp_A/fp_B (unaffected). Identity map at init (zero-init last
        # layer in FilmRescueModule).
        if self.use_film_rescue:
            pA, pB = self.film_rescue(pA, pB, batch["cell_row_idx"])

        # NBE ablations on function-node vectors:
        #   zero_p     — no p anywhere (kernel + RI), structure/cell remain
        #   resid_only — structure(+cell) residual; RI must not see p either,
        #                otherwise "−core" still leaks function-node info
        #   resid_only_no_deltaw — same track ablation with ΔW off
        _track = self.ablation_mode
        if _track.endswith("_no_deltaw") and _track != "no_deltaw":
            _track = _track[: -len("_no_deltaw")]
        if _track in ("zero_p", "resid_only"):
            pA = torch.zeros_like(pA)
            pB = torch.zeros_like(pB)

        # 1. Cell joint functional activity
        c = c_fn + self.alpha_mut * c_mut_fn                      # (B, 67)

        # 2. Structural embeddings (symmetric sum)
        if self.struct_encoder == "gnn":
            gA = self._graphs_for(batch["ik_A"], fp_A.device)
            gB = self._graphs_for(batch["ik_B"], fp_A.device)
            h_struct = self.gin(gA) + self.gin(gB)               # (B, 256)
        else:
            h_struct = self.struct_enc(fp_A) + self.struct_enc(fp_B)  # (B, 256)

        if self.use_descriptors:
            dev = fp_A.device
            last = self._desc_mat.shape[0] - 1
            iA = torch.tensor([self._desc_ik.get(k, last) for k in batch["ik_A"]], device=dev)
            iB = torch.tensor([self._desc_ik.get(k, last) for k in batch["ik_B"]], device=dev)
            h_struct = h_struct + self.desc_enc(self._desc_mat[iA]) + self.desc_enc(self._desc_mat[iB])

        if self.use_sensitivity_profile:
            dev = fp_A.device
            last_d = self._sens_drug_mat.shape[0] - 1
            iA = torch.tensor([self._sens_drug_ik.get(k, last_d) for k in batch["ik_A"]], device=dev)
            iB = torch.tensor([self._sens_drug_ik.get(k, last_d) for k in batch["ik_B"]], device=dev)
            h_struct = h_struct + self.sens_drug_enc(self._sens_drug_mat[iA]) + self.sens_drug_enc(self._sens_drug_mat[iB])

        # 3. Cell residual context
        h_cell = self.cell_enc(c_raw)                             # (B, 128)

        if self.use_sensitivity_profile:
            dev = fp_A.device
            last_c = self._sens_cell_mat.shape[0] - 1
            iC = torch.tensor([self._sens_cell_ik.get(k, last_c) for k in batch["cell_id"]], device=dev)
            h_cell = h_cell + self.sens_cell_enc(self._sens_cell_mat[iC])

        # 3b. Fused cell context (VirtualCellModule, design doc §4)
        if self.use_virtual_cell:
            z_cell = self.virtual_cell(batch["cell_row_idx"], c, h_cell)  # (B, z_cell_dim)
        else:
            z_cell = h_cell                                       # fallback: reproduces pre-VC dataflow

        # 4. Per-drug RI representations
        ri_repr_A = self.ri_enc(pA, z_cell)                      # (B, 64)
        y_ri_A    = self.ri_head(ri_repr_A)                      # (B,)
        ri_repr_B = self.ri_enc(pB, z_cell)                      # (B, 64)
        y_ri_B    = self.ri_head(ri_repr_B)                      # (B,)

        # 5. Pathway aggregation kernel
        kern_repr, same_fn = self.kernel(pA, pB, z_cell, batch["cell_row_idx"])  # (B,134), (B,1)

        # 6. Synergy head (+ track ablations)
        out = self.synergy_head(
            kern_repr, same_fn, h_struct, z_cell, ri_repr_A, ri_repr_B,
            ablation_mode=self.ablation_mode,
        )

        if self.use_pair_prior:
            prior = batch["pair_prior"].clamp(1e-4, 1 - 1e-4)
            prior_logit = torch.logit(prior)
            out["logit_class"] = out["logit_class"] + self.pair_prior_gamma * prior_logit

        if self.use_bio_prior:
            bprior = batch["bio_prior"].clamp(1e-4, 1 - 1e-4)
            bprior_logit = torch.logit(bprior)
            out["logit_class"] = out["logit_class"] + self.bio_prior_gamma * bprior_logit

        out["yhat_ri_A"] = y_ri_A
        out["yhat_ri_B"] = y_ri_B
        out["h_struct"]  = h_struct
        out["W_base"]    = self.kernel.W_base
        out["gamma"]     = self.kernel.gamma
        if self.use_synergy_regression:
            out["yhat_syn"] = self.syn_metric_head(out["syn_feat"])  # (B, K)
        return out

    # ── Shared forward + loss ─────────────────────────────────────────────────

    def _shared_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict, dict]:
        out      = self(batch)
        loss_kwargs = dict(
            logit_class = out["logit_class"],
            y_class     = batch["y_class"],
            yhat_ri_A   = out["yhat_ri_A"],
            yhat_ri_B   = out["yhat_ri_B"],
            y_ri_A      = batch["y_ri_A"],
            y_ri_B      = batch["y_ri_B"],
            h_struct    = out["h_struct"],
            alpha       = out["alpha"],
            W_base      = out["W_base"],
            gamma       = out["gamma"],
        )
        if self.use_synergy_regression:
            loss_kwargs["yhat_syn"] = out["yhat_syn"]
            loss_kwargs["y_syn"] = batch["y_syn_metrics"]
        loss_out = self.loss_fn(**loss_kwargs)
        return out, loss_out

    # ── Lightning hooks ───────────────────────────────────────────────────────

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int = 0
    ) -> torch.Tensor:
        _, loss_out = self._shared_step(batch)
        self._safe_log("train_loss", loss_out["loss"],
                       on_step=False, on_epoch=True, prog_bar=True)
        if self.use_synergy_regression:
            self._safe_log("train_syn_loss", loss_out["syn_loss"],
                           on_step=False, on_epoch=True, prog_bar=False)
        return loss_out["loss"]

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int = 0
    ) -> None:
        out, loss_out = self._shared_step(batch)
        self._safe_log("val_loss", loss_out["loss"],
                       on_step=False, on_epoch=True, prog_bar=False)
        entry = {
            "prob":      torch.sigmoid(out["logit_class"]).detach().cpu(),
            "y_class":   batch["y_class"].detach().cpu(),
            "yhat_ri_A": out["yhat_ri_A"].detach().cpu(),
            "y_ri_A":    batch["y_ri_A"].detach().cpu(),
            "yhat_ri_B": out["yhat_ri_B"].detach().cpu(),
            "y_ri_B":    batch["y_ri_B"].detach().cpu(),
        }
        if self.use_synergy_regression:
            entry["yhat_syn"] = out["yhat_syn"].detach().cpu()
            entry["y_syn"] = batch["y_syn_metrics"].detach().cpu()
            self._safe_log("val_syn_loss", loss_out["syn_loss"],
                           on_step=False, on_epoch=True, prog_bar=False)
        self._val_outputs.append(entry)

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            self._safe_log("val_auroc", 0.0, prog_bar=True)
            if self.use_synergy_regression:
                self._safe_log("val_syn_pearson", 0.0, prog_bar=True)
                self._safe_log("val_zip_pearson", 0.0, prog_bar=True)
            return

        prob    = torch.cat([o["prob"]      for o in self._val_outputs]).numpy()
        y_class = torch.cat([o["y_class"]   for o in self._val_outputs]).numpy()
        yhat_A  = torch.cat([o["yhat_ri_A"] for o in self._val_outputs]).numpy()
        y_A     = torch.cat([o["y_ri_A"]    for o in self._val_outputs]).numpy()
        yhat_B  = torch.cat([o["yhat_ri_B"] for o in self._val_outputs]).numpy()
        y_B     = torch.cat([o["y_ri_B"]    for o in self._val_outputs]).numpy()
        yhat_syn = None
        y_syn = None
        if self.use_synergy_regression and "yhat_syn" in self._val_outputs[0]:
            yhat_syn = torch.cat([o["yhat_syn"] for o in self._val_outputs]).numpy()
            y_syn = torch.cat([o["y_syn"] for o in self._val_outputs]).numpy()
        self._val_outputs.clear()

        ri_sum_prob = _expit(yhat_A + yhat_B)
        try:
            metrics   = compute_metrics(prob, y_class, yhat_A, y_A, yhat_B, y_B, ri_sum_prob)
            val_auroc = float(metrics["auroc"]) if math.isfinite(metrics["auroc"]) else 0.0
        except Exception:
            val_auroc = 0.0

        self._safe_log("val_auroc", val_auroc, prog_bar=True)

        if self.use_synergy_regression and yhat_syn is not None and y_syn is not None:
            per = self._per_syn_pearson(yhat_syn, y_syn)
            val_syn_pearson = float(np.mean(list(per.values()))) if per else 0.0
            self._safe_log("val_syn_pearson", val_syn_pearson, prog_bar=True)
            # Always expose ZIP for primary-regression early-stop.
            self._safe_log("val_zip_pearson", float(per.get("zip", 0.0)), prog_bar=True)
            for name, r in per.items():
                self._safe_log(f"val_{name}_pearson", float(r), prog_bar=False)

    def _per_syn_pearson(self, yhat_syn: np.ndarray, y_syn: np.ndarray) -> dict[str, float]:
        """Per-metric Pearson r (de-standardised preds vs raw targets)."""
        from scipy.stats import pearsonr

        mean = getattr(self.loss_fn, "_syn_mean", None)
        std = getattr(self.loss_fn, "_syn_std", None)
        if mean is None or std is None:
            return {}
        mean_a = mean.detach().cpu().numpy().astype(np.float64)
        std_a = std.detach().cpu().numpy().astype(np.float64)
        pred = yhat_syn.astype(np.float64) * std_a + mean_a
        yt = y_syn.astype(np.float64)
        names = list(self.syn_metric_names)
        if len(names) < yt.shape[1]:
            names = names + [f"m{i}" for i in range(len(names), yt.shape[1])]
        out: dict[str, float] = {}
        for i in range(yt.shape[1]):
            m = np.isfinite(yt[:, i]) & np.isfinite(pred[:, i])
            if int(m.sum()) < 3:
                continue
            r, _ = pearsonr(yt[m, i], pred[m, i])
            if math.isfinite(float(r)):
                out[names[i]] = float(r)
        return out

    def _mean_syn_pearson(self, yhat_syn: np.ndarray, y_syn: np.ndarray) -> float:
        """Mean Pearson r over synergy metrics (de-standardised preds vs targets)."""
        per = self._per_syn_pearson(yhat_syn, y_syn)
        return float(np.mean(list(per.values()))) if per else 0.0

    # ── Optimiser + scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=1e-2
        )
        monitor = self.early_stop_metric if self.use_synergy_regression else "val_auroc"
        if self.lr_scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=self.lr_factor,
                patience=self.lr_patience, min_lr=self.lr * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler, "interval": "epoch", "monitor": monitor,
                },
            }
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr * 0.01
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _safe_log(self, name: str, value, **kwargs) -> None:
        """Log metric; silently ignore if called outside a Trainer context."""
        try:
            self.log(name, value, **kwargs)
        except Exception:
            pass
