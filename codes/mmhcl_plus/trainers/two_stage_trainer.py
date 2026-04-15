"""
Two-Stage Contrastive Learning Trainer --- Revision 5.1

Algorithm 1 decomposed into three phases per mini-batch:

  Stage 1 --- Intra-branch CL (only after warmup):
    u2u branch: VICReg on ExpandedProjector outputs (D=1024).
    i2i branch: Chunked InfoNCE with FAISS hard negatives + temperature tau(t).

  Stage 2 --- Cross-branch Alignment + Ego-Final Anchoring:
    Soft BYOL between final hypergraph embedding and EMA teacher target.
    Ego-Final Anchor: VICReg between Layer 0 and Layer L embeddings.

  Aggregation:
    All 6 losses combined by Homoscedastic Uncertainty Balancer (single backward).

Changes from Rev44:
  - Barlow Twins -> VICReg (u2u + ego-final)
  - GradNorm (5 tasks) -> UncertaintyLossBalancer (6 tasks)
  - Added ego-final anchor loss (L_ego_final)
  - Added hard_negatives support in i2i branch
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.amp import GradScaler, autocast
import torch.nn as nn

from ..contrast.neighbor_pairs import build_neighbor_layer_pairs
from ..topology.dynamic_ema_weights import WEMAManager, update_ema_teacher


class TwoStageTrainer:
    """
    Manages one training step of MMHCL+ (Rev5.1).

    Args:
        model:         MMHCLPlus instance.
        optimizer:     Optimiser covering model + projector + balancer params.
        balancer:      UncertaintyLossBalancer (6 tasks).
        cfg:           MMHCLPlusConfig dataclass.
        projector:     ExpandedProjector for VICReg (d -> D=1024).
        w_ema_u:       Optional WEMAManager for user-side soft weights.
        w_ema_i:       Optional WEMAManager for item-side soft weights.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        balancer: nn.Module,
        cfg,
        projector: nn.Module,
        w_ema_u: WEMAManager | None = None,
        w_ema_i: WEMAManager | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.balancer = balancer
        self.cfg = cfg
        self.projector = projector
        self.w_ema_u = w_ema_u
        self.w_ema_i = w_ema_i
        self._epoch: int = 0

        sys_cfg = getattr(cfg, "system", None)
        amp = getattr(sys_cfg, "amp_dtype", "off") if sys_cfg is not None else "off"
        self._amp_enabled: bool = amp in ("bf16", "fp16") and torch.cuda.is_available()
        self._amp_dtype: torch.dtype = (
            torch.bfloat16 if amp == "bf16" else torch.float16
        )
        self._use_grad_scaler: bool = amp == "fp16" and torch.cuda.is_available()
        self._scaler = GradScaler("cuda", enabled=self._use_grad_scaler)
        self._grad_clip: float = float(
            getattr(sys_cfg, "grad_clip_max_norm", 1.0) if sys_cfg is not None else 1.0
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Called by the outer epoch loop to update the epoch counter."""
        self._epoch = epoch

    def current_tau(self) -> float:
        """Exponential temperature annealing: tau(t) = max(tau_min, tau_max * gamma^t)"""
        lc = self.cfg.loss
        tau_max = getattr(lc, "tau_max", 0.5)
        tau_min = getattr(lc, "tau_min", 0.05)
        tau_gamma = getattr(lc, "tau_gamma", 0.99)
        return max(tau_min, tau_max * (tau_gamma**self._epoch))

    def _in_warmup(self) -> bool:
        warmup = getattr(self.cfg.loss, "warmup_epochs", 0)
        return self._epoch < warmup

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch: dict[str, object],
        checkpoint_fn: Callable | None,
        loss_fns: dict[str, Callable],
    ) -> dict[str, float]:
        """
        Execute one mini-batch step (forward -> losses -> backward -> update).

        Returns:
            Dictionary of scalar loss values (detached, CPU float).
        """
        self.model.train()
        self.projector.train()

        device = next(self.model.parameters()).device
        amp_dev = "cuda" if device.type == "cuda" else "cpu"

        with autocast(
            device_type=amp_dev,
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            out = self.model(batch["x"], checkpoint_fn)

            loss_u = out["hyper_final"].new_tensor(0.0)
            loss_i = out["bip_final"].new_tensor(0.0)
            loss_ego = out["hyper_final"].new_tensor(0.0)

            if not self._in_warmup():
                tau = self.current_tau()
                lc = self.cfg.loss

                u_pairs = build_neighbor_layer_pairs(
                    out["hyper_layers"], max_hops=self.cfg.model.max_g_layers
                )
                i_pairs = build_neighbor_layer_pairs(
                    out["bip_layers"], max_hops=self.cfg.model.max_g_layers
                )

                # -- u2u branch: VICReg + Expanded Projector + ASW --
                u_terms: list[torch.Tensor] = []
                user_idx = batch.get("user_idx")
                n_u = user_idx.numel() if user_idx is not None else None
                for a, b in u_pairs:
                    if n_u is not None and n_u < a.size(0):
                        a_u, b_u = a[:n_u], b[:n_u]
                    else:
                        a_u, b_u = a, b

                    soft_w = None
                    if self.w_ema_u is not None and user_idx is not None:
                        w_rows = self.w_ema_u.get_batch_weights(user_idx, device=device)
                        soft_w = w_rows.mean(dim=-1)

                    u_terms.append(
                        loss_fns["vicreg"](
                            self.projector(a_u),
                            self.projector(b_u),
                            sim_weight=getattr(lc, "vicreg_sim_weight", 25.0),
                            var_weight=getattr(lc, "vicreg_var_weight", 25.0),
                            cov_weight=getattr(lc, "vicreg_cov_weight", 1.0),
                            soft_weights=soft_w,
                        )
                    )
                if u_terms:
                    loss_u = torch.stack(u_terms).mean()

                # -- i2i branch: Chunked InfoNCE + dynamic weights + hard negatives --
                i_terms: list[torch.Tensor] = []
                item_idx = batch.get("item_idx")
                n_i = item_idx.numel() if item_idx is not None else None
                n_u_off = n_u if n_u is not None else 0
                for a, b in i_pairs:
                    if n_i is not None and (n_u_off + n_i) <= a.size(0):
                        a_i = a[n_u_off : n_u_off + n_i]
                        b_i = b[n_u_off : n_u_off + n_i]
                    else:
                        a_i, b_i = a, b

                    dyn_w = None
                    if self.w_ema_i is not None and item_idx is not None:
                        w_full = self.w_ema_i.get_batch_weights(item_idx, device=device)
                        col_idx = item_idx.cpu().long()
                        if col_idx.max() < w_full.size(1):
                            dyn_w = w_full[:, col_idx].to(device)
                        if dyn_w is not None and dyn_w.size(0) != a_i.size(0):
                            dyn_w = None

                    # Hard negatives from batch (if provided)
                    hard_neg = batch.get("hard_negatives")

                    i_terms.append(
                        loss_fns["infonce"](
                            a_i,
                            b_i,
                            tau=tau,
                            dynamic_weights=dyn_w,
                            hard_negatives=hard_neg,
                        )
                    )
                if i_terms:
                    loss_i = torch.stack(i_terms).mean()

                # -- Ego-Final Anchor: VICReg between Layer 0 and Layer L --
                hyper_layers = out["hyper_layers"]
                if len(hyper_layers) >= 2:
                    ego_emb = hyper_layers[0]   # Layer 0
                    final_emb = hyper_layers[-1]  # Layer L
                    if n_u is not None and n_u < ego_emb.size(0):
                        ego_emb = ego_emb[:n_u]
                        final_emb = final_emb[:n_u]
                    loss_ego = loss_fns["vicreg"](
                        self.projector(ego_emb),
                        self.projector(final_emb.detach()),
                        sim_weight=getattr(lc, "vicreg_sim_weight", 25.0),
                        var_weight=getattr(lc, "vicreg_var_weight", 25.0),
                        cov_weight=getattr(lc, "vicreg_cov_weight", 1.0),
                    )

            align_weights = batch.get("align_weights")
            loss_align = loss_fns["soft_byol"](
                out["hyper_final"], out["bip_final"], align_weights
            )

            loss_bpr = loss_fns["bpr"](batch["pos_scores"], batch["neg_scores"])

        # Dirichlet energy regularisation (anti-over-smoothing)
        loss_dir = loss_fns["dirichlet"](
            out["hyper_final"], batch["node_idx"], batch["lap_getter"]
        )

        # ----------------------------------------------------------------
        # Aggregation via Homoscedastic Uncertainty Balancer (6 tasks)
        # ----------------------------------------------------------------
        raw_losses = [loss_bpr, loss_u, loss_i, loss_align, loss_dir, loss_ego]
        total = self.balancer(raw_losses)

        # ----------------------------------------------------------------
        # Backward + grad clip + parameter update
        # ----------------------------------------------------------------
        self.optimizer.zero_grad(set_to_none=True)
        opt_params = [p for g in self.optimizer.param_groups for p in g["params"]]

        if self._use_grad_scaler:
            self._scaler.scale(total).backward()
            if self._grad_clip > 0:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(opt_params, self._grad_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            total.backward()
            if self._grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(opt_params, self._grad_clip)
            self.optimizer.step()

        # EMA teacher refresh
        update_ema_teacher(
            self.model.bip_encoder,
            self.model.ema_teacher,
            momentum=self.cfg.topology.ema_momentum,
        )

        # Lazy W_ema update
        u_idx = batch.get("user_idx")
        if self.w_ema_u is not None and u_idx is not None:
            n_u = u_idx.numel()
            h_full = out["hyper_final"]
            h_u = h_full[:n_u] if n_u <= h_full.size(0) else h_full
            if h_u.size(0) == n_u:
                self.w_ema_u.step_update(h_u.detach(), u_idx, self._epoch)

        i_idx = batch.get("item_idx")
        if self.w_ema_i is not None and i_idx is not None:
            n_i = i_idx.numel()
            h_full = out["bip_final"]
            n_u_offset = u_idx.numel() if u_idx is not None else 0
            h_i = (
                h_full[n_u_offset : n_u_offset + n_i]
                if (n_u_offset + n_i) <= h_full.size(0)
                else h_full
            )
            if h_i.size(0) == n_i:
                self.w_ema_i.step_update(h_i.detach(), i_idx, self._epoch)

        # Get uncertainty weights for logging
        unc_weights = self.balancer.get_weights().cpu().tolist() if hasattr(self.balancer, "get_weights") else []

        return {
            "loss": float(total.detach().cpu()),
            "bpr": float(loss_bpr.detach().cpu()),
            "u2u": float(loss_u.detach().cpu()),
            "i2i": float(loss_i.detach().cpu()),
            "align": float(loss_align.detach().cpu()),
            "dir": float(loss_dir.detach().cpu()),
            "ego_final": float(loss_ego.detach().cpu()),
            "tau": self.current_tau(),
            "warmup": self._in_warmup(),
            "uncertainty_weights": unc_weights,
        }
