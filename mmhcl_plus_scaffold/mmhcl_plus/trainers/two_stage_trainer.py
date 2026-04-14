"""
Two-Stage Contrastive Learning Trainer — TEX Section 4, Step 3.

Algorithm 1 (TEX §4.4) decomposed into three phases per mini-batch:

  Stage 1 — Intra-branch CL (only after warmup):
    For each cached layer pair (l, l+1) with l ≤ g:
      u2u branch: Barlow Twins on ExpandedProjector outputs.
                  Soft weights W_ema[batch_users] are fetched lazily from
                  WEMAManager and applied as per-sample importance (ASW).
      i2i branch: Chunked InfoNCE with temperature τ(t).
                  Dynamic weights W_ema[batch_items] are passed as log-prior
                  corrections to the logit matrix (Eq. 12–13 in NLGCL+).

  Stage 2 — Cross-branch Alignment:
    Soft BYOL between the final hypergraph embedding and the EMA teacher's
    bipartite embedding (stop-gradient on teacher).

  Aggregation:
    All five losses are combined by the GradNorm balancer for dynamic gradient
    normalisation (Chen et al., ICML 2018).

Temperature annealing schedule (TEX §4.3):
    τ(t) = max(τ_min,  τ_max · γ^t)
    Starts high (broad contrastive signal) and cools as representations sharpen.

Warmup:
    For the first `warmup_epochs` epochs the CL losses are skipped entirely so
    the BPR loss can establish a reasonable embedding space before contrastive
    objectives are applied.  This prevents representation collapse early in training.

EMA teacher refresh (TEX §4.2):
    After every gradient step: ξ ← β·ξ + (1-β)·θ  (BYOL momentum update).
    W_ema is also offered a chance to update its batch rows via WEMAManager.step_update.
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
    Manages one training step of MMHCL+.

    Args:
        model:         MMHCLPlus instance (hyper_encoder, bip_encoder, fusion,
                       projector_u2u, ema_teacher).
        optimizer:     Optimiser covering model + projector + balancer params.
        balancer:      GradNormLossBalancer (5 tasks).
        cfg:           MMHCLPlusConfig dataclass.
        projector:     ExpandedProjector for the u2u branch (d → D=8192).
        w_ema_u:       Optional WEMAManager for user-side soft weights.
                       If None, no ASW is applied to the u2u branch.
        w_ema_i:       Optional WEMAManager for item-side soft weights.
                       If None, no ASW is applied to the i2i branch.
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
        """
        Exponential temperature annealing schedule (TEX §4.3):
            τ(t) = max(τ_min,  τ_max · γ^t)
        """
        lc = self.cfg.loss
        tau_max = getattr(lc, "tau_max", lc.tau)
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
        Execute one mini-batch step (forward → losses → backward → update).

        Expected batch keys (see README_MMHCL_PLUS.md):
            x            : [B, d] input embeddings / features.
            pos_scores   : [B] positive pair scores (for BPR).
            neg_scores   : [B] negative pair scores (for BPR).
            node_idx     : [B] integer node indices (for Dirichlet).
            lap_getter   : callable(idx) → sparse Laplacian block.
            user_idx     : optional [B_u] user indices for W_ema_u lookup.
            item_idx     : optional [B_i] item indices for W_ema_i lookup.
            align_weights: optional [B] soft BYOL alignment weights.

        Returns:
            Dictionary of scalar loss values (detached, CPU float).
        """
        self.model.train()
        self.projector.train()

        device = next(self.model.parameters()).device
        amp_dev = "cuda" if device.type == "cuda" else "cpu"

        # ----------------------------------------------------------------
        # Forward + CL + BPR + alignment under autocast (report Opt. 2).
        # Dirichlet trace is computed *outside* autocast for numerical stability.
        # ----------------------------------------------------------------
        with autocast(
            device_type=amp_dev,
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            out = self.model(batch["x"], checkpoint_fn)

            loss_u = out["hyper_final"].new_tensor(0.0)
            loss_i = out["bip_final"].new_tensor(0.0)

            if not self._in_warmup():
                tau = self.current_tau()

                u_pairs = build_neighbor_layer_pairs(
                    out["hyper_layers"], max_hops=self.cfg.model.max_g_layers
                )
                i_pairs = build_neighbor_layer_pairs(
                    out["bip_layers"], max_hops=self.cfg.model.max_g_layers
                )

                # -- u2u branch: Barlow Twins + Expanded Projector + ASW --
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
                        loss_fns["barlow"](
                            self.projector(a_u),
                            self.projector(b_u),
                            lambd=self.cfg.loss.barlow_lambda,
                            soft_weights=soft_w,
                        )
                    )
                if u_terms:
                    loss_u = torch.stack(u_terms).mean()

                # -- i2i branch: Chunked InfoNCE + dynamic weights + temperature --
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

                    i_terms.append(
                        loss_fns["infonce"](
                            a_i,
                            b_i,
                            tau=tau,
                            dynamic_weights=dyn_w,
                        )
                    )
                if i_terms:
                    loss_i = torch.stack(i_terms).mean()

            align_weights = batch.get("align_weights")
            loss_align = loss_fns["soft_byol"](
                out["hyper_final"], out["bip_final"], align_weights
            )

            loss_bpr = loss_fns["bpr"](batch["pos_scores"], batch["neg_scores"])

        # ----------------------------------------------------------------
        # Dirichlet energy regularisation (anti-over-smoothing)
        # ----------------------------------------------------------------
        loss_dir = loss_fns["dirichlet"](
            out["hyper_final"], batch["node_idx"], batch["lap_getter"]
        )

        # ----------------------------------------------------------------
        # Aggregation via GradNorm
        # ----------------------------------------------------------------
        weighted_losses = [
            self.cfg.loss.bpr_weight * loss_bpr,
            self.cfg.loss.u2u_weight * loss_u,
            self.cfg.loss.i2i_weight * loss_i,
            self.cfg.loss.align_weight * loss_align,
            self.cfg.loss.dirichlet_weight * loss_dir,
        ]
        total, grad_weights = self.balancer.combine(weighted_losses)

        # ----------------------------------------------------------------
        # Backward + grad clip (report Opt. 6) + parameter update
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

        # ----------------------------------------------------------------
        # EMA teacher refresh (BYOL-style momentum update)
        # ----------------------------------------------------------------
        update_ema_teacher(
            self.model.bip_encoder,
            self.model.ema_teacher,
            momentum=self.cfg.topology.ema_momentum,
        )

        # ----------------------------------------------------------------
        # Lazy W_ema update for batch nodes (only every update_interval epochs)
        # ----------------------------------------------------------------
        # W_ema lazy update — embeddings passed to step_update MUST have the same
        # number of rows as idx.  The caller organises the batch as [users | items],
        # so we take the first n_u rows for users and the rest for items.
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
            # Items occupy the tail of the batch after users
            n_u_offset = u_idx.numel() if u_idx is not None else 0
            h_i = (
                h_full[n_u_offset : n_u_offset + n_i]
                if (n_u_offset + n_i) <= h_full.size(0)
                else h_full
            )
            if h_i.size(0) == n_i:
                self.w_ema_i.step_update(h_i.detach(), i_idx, self._epoch)

        return {
            "loss": float(total.detach().cpu()),
            "bpr": float(loss_bpr.detach().cpu()),
            "u2u": float(loss_u.detach().cpu()),
            "i2i": float(loss_i.detach().cpu()),
            "align": float(loss_align.detach().cpu()),
            "dir": float(loss_dir.detach().cpu()),
            "tau": self.current_tau(),
            "warmup": self._in_warmup(),
            "grad_weights": grad_weights.cpu().tolist(),
        }
