"""
main_mmhcl_plus.py — MMHCL+ (Neighbor-Layer Hypergraph CL) Training Script
============================================================================

Full implementation of the MMHCL+ architecture (revision 5.2), integrating all
modules from the mmhcl_plus package (codes/mmhcl_plus/):

  1. Neighbor-layer contrastive learning pairs (NLGCL+ approach)
       - u2u branch : VICReg between adjacent user-hypergraph layers
       - i2i branch : Chunked InfoNCE with FAISS hard negatives between
                      adjacent item-hypergraph layers
  2. Adaptive Sample Weighting (ASW) via WEMAManager — both user and item sides
  3. Soft Topology-Aware Purification: percentile + bridge penalty
       applied to both w_ema_i and w_ema_u after precomputation
  4. EMA Teacher Network: momentum-updated copy of the online encoder
       produces stop-gradient targets for Soft BYOL
  5. Soft BYOL cross-view alignment using EMA teacher targets
  6. Dirichlet energy regularisation (prevents over-smoothing)
  7. Hybrid Loss Balancer: Uncertainty Weighting → GradNorm transition
       for 5 objectives (BPR, u2u, i2i, align, dirichlet)
  8. CL warmup ramp (linear 0→1) to prevent CL Activation Shock
  9. Delayed FAISS hard negatives (activated after epoch 50)
 10. Profiling-guided activation checkpointing

Rev5.2 removals (vs Rev5.1):
  - Ego-Final Anchor branch (conflicts with neighbor-layer CL)
  - Pure Uncertainty-only balancer (replaced by Hybrid)

Usage (same CLI interface as main.py, plus new MMHCL+ flags):
    python main_mmhcl_plus.py --dataset Baby --seed 42 \\
        --warmup_epochs 15 --tau_max 0.5 --tau_min 0.05 --tau_gamma 0.99 \\
        --use_hybrid_balancer 1 --cl_ramp_epochs 20 --delay_hard_negs_epoch 50 \\
        [all other original MMHCL args]

Reference: NLGCL-to-MMHCL-architecture_revision52_full_implementation.tex
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import copy
import gc
import math
import os
import pathlib
from time import time
from typing import Any

from Models import MMHCL

# ── third-party ───────────────────────────────────────────────────────────────
import numpy.typing as npt
import torch
import torch.nn.functional as F
import torch.optim as optim
from utility.batch_test import *  # also instantiates global data_generator
from utility.common import (
    bpr_loss as _bpr_loss,
)
from utility.common import (
    build_experiment_paths as _build_paths,
)
from utility.common import (
    lr_decay_schedule,
    set_seed,
)
from utility.logging import Logger

# ── repo utilities (identical to main.py) ─────────────────────────────────────
from utility.parser import parse_args

from mmhcl_plus.contrast.uncertainty_balancer import UncertaintyLossBalancer
from mmhcl_plus.contrast.hybrid_balancer import HybridLossBalancer

from mmhcl_plus.ablation import (
    apply_to_args as _apply_ablation_to_args,
    available_variants as _available_ablation_variants,
    get as _get_ablation_variant,
)

# ── MMHCL+ package (codes/mmhcl_plus/) ────────────────────────────────────────
from mmhcl_plus.contrast.losses import vicreg_loss, chunked_info_nce_loss
from mmhcl_plus.contrast.neighbor_pairs import build_neighbor_layer_pairs
from mmhcl_plus.contrast.soft_byol import soft_byol_alignment
from mmhcl_plus.model.projector import ExpandedProjector
from mmhcl_plus.regularizers.dirichlet import (
    sparse_dirichlet_energy_batch as _sparse_dirichlet_energy_batch,
)
from mmhcl_plus.topology.dynamic_ema_weights import (
    WEMAManager,
    build_item_wema,
    build_user_wema,
    update_ema_teacher,
)
from mmhcl_plus.topology.hard_negatives import (
    HardNegativeMiner,
    build_interaction_mask,
    mine_hard_negatives_faiss,
)

# ── parse args ────────────────────────────────────────────────────────────────
args = parse_args()
MetricsDict = dict[str, Any]

# ── module-level path globals (set by train_evaluation_loop) ──────────────────
path_name: str = ""
path: str = ""
record_path: str = ""
wandb: Any = None


# ---------------------------------------------------------------------------
# Named function for gradient-checkpointed sparse MM (must be module-level,
# not a lambda, so that torch.utils.checkpoint can replay it correctly).
#
# NOTE (Rev5.2 AMP fix):
#   torch.sparse.mm on CUDA dispatches to `addmm_sparse_cuda`, which is NOT
#   implemented for the `Half` (fp16) dtype. When AMP autocast (fp16) is
#   active, the dense operand `x` is promoted to fp16, which triggers:
#       NotImplementedError: "addmm_sparse_cuda" not implemented for 'Half'
#   Fix: disable autocast locally and execute the sparse MM in fp32. The
#   surrounding autocast region will automatically down-cast later ops that
#   DO support fp16, so numerical behaviour of the rest of the forward pass
#   is unchanged while the unsupported kernel is bypassed.
# ---------------------------------------------------------------------------
def _hypergraph_step(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        adj_f = adj.float() if adj.is_floating_point() and adj.dtype != torch.float32 else adj
        return torch.sparse.mm(adj_f, x.float())


# ===========================================================================
#  MMHCLWithLayers — extends MMHCL to cache intermediate layer embeddings
# ===========================================================================
class MMHCLWithLayers(MMHCL):
    """
    Drop-in extension of MMHCL that captures every intermediate embedding
    produced by the item-item and user-user hypergraph branches.

    These cached outputs are used by the neighbor-layer CL losses in
    MMHCLPlusTrainer: positive pairs are formed from (layer_l, layer_{l+1}).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        n_item_layers: int,
        n_user_layers: int,
    ) -> None:
        super().__init__(n_users, n_items, embedding_dim)
        # Store layer counts as instance attributes to avoid global-args dependency
        self._n_item_layers = n_item_layers
        self._n_user_layers = n_user_layers

    def forward_plus(
        self,
        UI_mat: torch.Tensor,
        I2I_mat: torch.Tensor,
        U2U_mat: torch.Tensor,
        checkpoint_threshold: int = -1,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Extended forward pass that returns layer caches alongside final embeddings.

        Parameters
        ----------
        checkpoint_threshold : int
            Hypergraph layers whose 0-based index is >= this value are
            recomputed during backward (gradient checkpointing), while
            shallower layers are kept alive for neighbor-layer CL pairs.
            Pass -1 to disable checkpointing entirely.

        Returns
        -------
        u_final   : (n_users, d) fused user embeddings  (CF + normalised hypergraph)
        i_final   : (n_items, d) fused item embeddings
        ii_final  : (n_items, d) item hypergraph final-layer embedding
        uu_final  : (n_users, d) user hypergraph final-layer embedding
        ii_layers : list of (n_items, d) tensors — layer 0 (raw) … layer L
        uu_layers : list of (n_users, d) tensors — layer 0 (raw) … layer L
        u_bip     : (n_users, d) CF-branch-only user embedding
        i_bip     : (n_items, d) CF-branch-only item embedding
        """
        use_ckpt = checkpoint_threshold >= 0 and torch.is_grad_enabled()

        # ── Item hypergraph with layer caching + checkpointing ────────────
        ii_emb: torch.Tensor = self.ii_embedding.weight  # layer 0
        ii_layers: list[torch.Tensor] = [ii_emb]
        for layer_id in range(self._n_item_layers):
            if use_ckpt and layer_id >= checkpoint_threshold:
                ii_emb = torch.utils.checkpoint.checkpoint(
                    _hypergraph_step, I2I_mat, ii_emb, use_reentrant=False
                )
            else:
                ii_emb = _hypergraph_step(I2I_mat, ii_emb)
            ii_layers.append(ii_emb)
        ii_final = ii_emb

        # ── User hypergraph with layer caching + checkpointing ────────────
        uu_emb: torch.Tensor = self.uu_embedding.weight  # layer 0
        uu_layers: list[torch.Tensor] = [uu_emb]
        for layer_id in range(self._n_user_layers):
            if use_ckpt and layer_id >= checkpoint_threshold:
                uu_emb = torch.utils.checkpoint.checkpoint(
                    _hypergraph_step, U2U_mat, uu_emb, use_reentrant=False
                )
            else:
                uu_emb = _hypergraph_step(U2U_mat, uu_emb)
            uu_layers.append(uu_emb)
        uu_final = uu_emb

        # ── CF branch (LightGCN bipartite) ────────────────────────────────
        if args.cf_model == "LightGCN":
            ego: torch.Tensor = torch.cat(
                (self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0
            )
            all_e: list[torch.Tensor] = [ego]
            for _ in range(args.UI_layers):
                ego = _hypergraph_step(UI_mat, ego)
                all_e.append(ego)
            mean_e = torch.stack(all_e, dim=1).mean(dim=1)
            u_bip, i_bip = torch.split(mean_e, [self.n_users, self.n_items], dim=0)

        elif args.cf_model == "MF":
            u_bip = self.user_ui_embedding.weight
            i_bip = self.item_ui_embedding.weight

        else:
            # NGCF fallback (same as MMHCL.forward)
            ego = torch.cat(
                (self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0
            )
            all_e = [ego]
            for i in range(args.UI_layers):
                side = _hypergraph_step(UI_mat, ego)
                sum_e = F.leaky_relu(self.GC_Linear_list[i](side))
                bi_e = torch.mul(ego, side)
                bi_e = F.leaky_relu(self.Bi_Linear_list[i](bi_e))
                ego = sum_e + bi_e
                ego = self.dropout_list[i](ego)
                all_e.append(F.normalize(ego, p=2, dim=1))
            mean_e = torch.stack(all_e, dim=1).mean(dim=1)
            u_bip, i_bip = torch.split(mean_e, [self.n_users, self.n_items], dim=0)

        # ── Fusion (same as MMHCL.forward) ────────────────────────────────
        i_final = i_bip + F.normalize(ii_final, p=2, dim=1)
        u_final = u_bip + F.normalize(uu_final, p=2, dim=1)

        return u_final, i_final, ii_final, uu_final, ii_layers, uu_layers, u_bip, i_bip


# ===========================================================================
#  MMHCLPlusTrainer
# ===========================================================================
class MMHCLPlusTrainer:
    """
    Full MMHCL+ training loop.

    Inherits the same evaluation, early-stopping, W&B-logging, and LR-
    scheduling logic as the original Trainer but replaces the forward pass
    and loss computation with the MMHCL+ multi-objective formulation.
    """

    def __init__(
        self,
        data_config: dict[str, Any],
        optuna_trial: Any = None,
    ) -> None:
        self.optuna_trial = optuna_trial
        self.n_users: int = data_config["n_users"]
        self.n_items: int = data_config["n_items"]

        # Logger (same path convention as Trainer)
        self.logger: Logger = Logger(
            path,
            is_debug=args.debug,
            target=path_name,
            path2=record_path,
            ablation_target=args.ablation_target,
        )
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.lr: float = args.lr
        self.emb_dim: int = args.embed_size
        self.batch_size: int = args.batch_size
        self.regs: float = args.regs
        self.decay: float = self.regs

        # Adjacency matrices (moved to GPU)
        self.UI_mat: torch.Tensor = data_config["UI_mat"].cuda()
        self.User_mat: torch.Tensor = data_config["User_mat"].cuda()
        self.Item_mat: torch.Tensor = data_config["Item_mat"].cuda()

        # Extended MMHCL model (student / online network)
        self.model: MMHCLWithLayers = MMHCLWithLayers(
            self.n_users,
            self.n_items,
            self.emb_dim,
            n_item_layers=args.Item_layers,
            n_user_layers=args.User_layers,
        ).cuda()

        # EMA Teacher network — momentum-updated copy of online encoder.
        # Provides stable target embeddings for Soft BYOL alignment (TEX §4.4 Alg.1).
        # No gradients through teacher; updated via Polyak averaging each step.
        self.ema_model: MMHCLWithLayers = copy.deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        # ── Rev5.2-OPT: torch.compile wrapper (preserves architecture) ────
        # Acceleration Guide §1: `mode="reduce-overhead"` enables CUDA graphs
        # and yields an additional ~5-10% speedup over `mode="default"` for
        # the static-shape GNN forward pass (forward_plus receives the fixed
        # UI/User/Item adjacency matrices on every call — no shape variance).
        # Override via MMHCL_COMPILE_MODE=default if recompilation is observed.
        USE_COMPILE = os.environ.get("MMHCL_COMPILE", "1") == "1"
        COMPILE_MODE = os.environ.get("MMHCL_COMPILE_MODE", "reduce-overhead")
        if USE_COMPILE and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode=COMPILE_MODE, dynamic=False)
                print(f"[Rev5.2-OPT] torch.compile enabled mode={COMPILE_MODE}")
            except Exception as e:
                print(f"[Rev5.2-OPT] torch.compile disabled: {e}")

        # Expanded projector for u2u VICReg branch (Rev5.1)
        self.projector: ExpandedProjector = ExpandedProjector(
            in_dim=self.emb_dim,
            hidden_dim=args.projector_hidden_dim,
            out_dim=args.projector_out_dim,
        ).cuda()

        # ── Active task list (Rev5.2 base = 5; +1 for A7_ego_final) ─────
        self._enable_nlcl: bool = bool(getattr(args, "enable_neighbor_layer_cl", 1))
        self._enable_dirichlet: bool = bool(getattr(args, "enable_dirichlet", 1))
        self._enable_soft_byol: bool = bool(getattr(args, "enable_soft_byol_cross", 1))
        self._enable_ego_final: bool = bool(getattr(args, "enable_ego_final_anchor", 0))
        self._enable_cl_ramp: bool = bool(getattr(args, "enable_cl_warmup_ramp", 1))
        self._enable_delayed_faiss: bool = bool(getattr(args, "enable_delayed_faiss", 1))
        self._g_layers: int = int(getattr(args, "g_layers", 2))

        self._task_names: list[str] = ["bpr"]
        if self._enable_nlcl:
            self._task_names += ["u2u", "i2i"]
        if self._enable_soft_byol:
            self._task_names += ["align"]
        if self._enable_dirichlet:
            self._task_names += ["dir"]
        if self._enable_ego_final:
            self._task_names += ["ego_final"]
        n_tasks: int = len(self._task_names)

        balancer_type: str = str(getattr(args, "balancer_type", "hybrid")).lower()
        # Backward-compat: --use_hybrid_balancer 0 overrides to uncertainty
        if not getattr(args, "use_hybrid_balancer", 1):
            balancer_type = "uncertainty"

        if balancer_type in {"hybrid", "gradnorm", "fixed"}:
            self.balancer = HybridLossBalancer(
                num_tasks=n_tasks,
                alpha=getattr(args, "gradnorm_alpha", 1.5),
                transition_epoch=getattr(args, "balancer_transition_epoch", 40),
                blend_epochs=getattr(args, "balancer_blend_epochs", 20),
                mode=balancer_type,
                # Acceleration Guide §A: stride>1 caches GradNorm weights
                # between full updates, recovering ~12% of wall-clock time.
                gradnorm_stride=int(getattr(args, "gradnorm_stride", 4)),
            ).cuda()
        elif balancer_type == "uncertainty":
            # Use the dedicated UncertaintyLossBalancer (single-mode).
            self.balancer = UncertaintyLossBalancer(num_tasks=n_tasks).cuda()
        else:
            raise ValueError(f"Unknown balancer_type '{balancer_type}'")

        # Acceleration Guide §B1: lazy VICReg covariance toggle
        self._vicreg_lazy_cov: bool = bool(getattr(args, "vicreg_lazy_cov", 1))

        self.logger.logging(
            f"[Ablation] balancer={balancer_type} n_tasks={n_tasks} "
            f"tasks={self._task_names} g_layers={self._g_layers} "
            f"NLCL={self._enable_nlcl} Dir={self._enable_dirichlet} "
            f"Align={self._enable_soft_byol} EgoFinal={self._enable_ego_final} "
            f"CLRamp={self._enable_cl_ramp} DelayFAISS={self._enable_delayed_faiss}"
        )
        self.logger.logging(
            f"[Rev5.2-OPT] gradnorm_stride={int(getattr(args, 'gradnorm_stride', 4))} "
            f"vicreg_lazy_cov={int(self._vicreg_lazy_cov)}"
        )

        # Unified optimizer (model + projector + balancer; NOT ema_model)
        self.optimizer: optim.Adam = optim.Adam(
            list(self.model.parameters())
            + list(self.projector.parameters())
            + list(self.balancer.parameters()),
            lr=self.lr,
        )

        # LR schedulers (identical to Trainer)
        self.lr_scheduler: optim.lr_scheduler.LambdaLR = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_decay_schedule
        )
        self.reduce_lr_scheduler: optim.lr_scheduler.ReduceLROnPlateau | None = (
            optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=args.early_stopping_mode,
                factor=args.reduce_lr_factor,
                patience=args.reduce_lr_patience,
                min_lr=args.reduce_lr_min,
            )
            if args.use_reduce_lr
            else None
        )

        # WEMAManager — item and user side (TEX §4.2 + §3.4)
        self.w_ema_i: WEMAManager | None = None
        self.w_ema_u: WEMAManager | None = None
        self._init_wema_managers(data_config)

    # -----------------------------------------------------------------------
    def _init_wema_managers(self, data_config: dict) -> None:
        """
        Build WEMAManagers for both item (i2i) and user (u2u) branches.

        Item side:
            Loads image_feat.npy and text_feat.npy, computes cosine similarity,
            then applies soft topology-aware purification (TEX §3.4).

        User side:
            Derives user features by mean-pooling item features over each user's
            training interactions (NLGCL+ Eq. 10), then applies purification.
        """
        data_path = os.path.join(args.data_path, args.dataset)
        feat_paths = [
            os.path.join(data_path, "image_feat.npy"),
            os.path.join(data_path, "text_feat.npy"),
        ]
        n_items = data_config["n_items"]
        n_users = data_config["n_users"]

        # ── Item-side WEMAManager ─────────────────────────────────────────
        self.w_ema_i = build_item_wema(
            n_items=n_items,
            item_feat_paths=feat_paths,
            alpha=args.w_ema_alpha,
            update_interval=args.w_ema_update_interval,
            percentile=args.soft_purification_percentile,
            purification_temp=args.soft_purification_temp,
            logger=self.logger,
        )
        if self.w_ema_i is None:
            self.logger.logging(
                "[MMHCL+] No item feature files found in %s; "
                "item-side WEMAManager disabled (i2i ASW skipped)." % data_path
            )

        # ── User-side WEMAManager ─────────────────────────────────────────
        self.w_ema_u = build_user_wema(
            n_users=n_users,
            n_items=n_items,
            train_items=data_generator.train_items,
            item_feat_paths=feat_paths,
            alpha=args.w_ema_alpha,
            update_interval=args.w_ema_update_interval,
            percentile=args.soft_purification_percentile,
            purification_temp=args.soft_purification_temp,
            logger=self.logger,
        )
        if self.w_ema_u is None:
            self.logger.logging(
                "[MMHCL+] No item feature files found in %s; "
                "user-side WEMAManager disabled (u2u ASW skipped)." % data_path
            )

    # -----------------------------------------------------------------------
    def test(self, users_to_test: list[int], is_val: bool) -> MetricsDict:
        """Evaluation (replicates Trainer.test). No checkpointing during inference."""
        self.model.eval()
        with torch.no_grad():
            u_final, i_final, _, _, _, _, _, _ = self.model.forward_plus(
                self.UI_mat, self.Item_mat, self.User_mat, checkpoint_threshold=-1
            )
        result: MetricsDict = test_torch(u_final, i_final, users_to_test, is_val)
        return result

    # -----------------------------------------------------------------------
    def bpr_loss(
        self,
        u: torch.Tensor,
        pos_i: torch.Tensor,
        neg_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """BPR loss with L2 embedding regularisation."""
        return _bpr_loss(u, pos_i, neg_i, self.batch_size, self.decay)

    # -----------------------------------------------------------------------
    def _get_shared_params(self) -> list[torch.Tensor]:
        """Return shared backbone parameters for GradNorm gradient norms."""
        params = []
        for name, p in self.model.named_parameters():
            if "embedding" in name and p.requires_grad:
                params.append(p)
        return params if params else list(self.model.parameters())[:4]

    # -----------------------------------------------------------------------
    def train(self, return_validation: bool = False) -> float:
        """
        Full training loop — mirrors Trainer.train() in structure and log format.

        Key differences from Rev5.1 (Rev5.2 changes):
          - CL ramp: linear 0→1 over cl_ramp_epochs after warmup
          - FAISS hard negatives delayed until delay_hard_negs_epoch
          - Ego-Final Anchor removed (conflicts with neighbor-layer CL)
          - 5-task Hybrid Balancer (Uncertainty → GradNorm transition)
          - VICReg projector D=4096 (was 1024)
          - Warmup epochs = 15 (was 5)
        """
        training_time_list: list[float] = []
        loss_loger: list[float] = []
        rec_loger: list[npt.NDArray] = []
        pre_loger: list[npt.NDArray] = []
        ndcg_loger: list[npt.NDArray] = []
        hit_loger: list[npt.NDArray] = []

        stopping_step: int = 0
        best_recall: float = 0.0
        best_ndcg: float = 0.0
        best_val_recall: float = 0.0
        best_model_state: dict | None = None
        test_ret: str | MetricsDict = ""
        eval_step: int = 0

        # W&B initialisation (identical to Trainer.train)
        if args.use_wandb and wandb is not None:
            if wandb.run is not None:
                wandb.finish()
            wconfig = vars(args)
            wconfig["path_name"] = path_name
            winit: dict[str, Any] = {
                "project": args.wandb_project,
                "config": wconfig,
            }
            if args.wandb_entity:
                winit["entity"] = args.wandb_entity
            if args.wandb_run_name:
                winit["name"] = args.wandb_run_name
            wandb.init(**winit)
            self.logger.logging("W&B run initialized: %s" % wandb.run.name)

        # ── AMP: GradScaler for mixed-precision training ─────────────────────
        # Acceleration Guide §3: bfloat16 is the safer default on RTX 5090 (Ada
        # Lovelace / Blackwell). It has the same dynamic range as float32, so
        # VICReg's [D, D] covariance-sum (D=4096 → ~16M terms) cannot overflow.
        # float16 is retained as an opt-in via MMHCL_AMP_DTYPE=float16 for
        # legacy runs. bfloat16 does NOT require a GradScaler.
        USE_AMP = os.environ.get("MMHCL_AMP", "1") == "1"
        AMP_DTYPE_STR = os.environ.get("MMHCL_AMP_DTYPE", "bfloat16").lower()
        if AMP_DTYPE_STR in ("bf16", "bfloat16"):
            amp_dtype = torch.bfloat16
            amp_scaler = torch.amp.GradScaler("cuda", enabled=False)  # no-op for bf16
        else:
            amp_dtype = torch.float16
            amp_scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
        if USE_AMP:
            self.logger.logging(
                f"[Rev5.2-OPT] AMP enabled (dtype={amp_dtype}, "
                f"scaler={'on' if amp_scaler.is_enabled() else 'off'})"
            )

        # ── Rev5.2-OPT: persistent HardNegativeMiner (Acceleration Guide §2) ─
        # Create the FAISS-GPU index once (fp16 storage, GpuIndexFlatIP) and
        # reuse it every epoch via .reset() + .add(). This avoids re-allocating
        # the CPU→GPU index on every call and halves the FAISS VRAM footprint.
        self._hard_neg_miner: HardNegativeMiner | None = None
        try:
            self._hard_neg_miner = HardNegativeMiner(
                dim=self.emb_dim, device=int(getattr(args, "gpu_id", 0))
            )
            self.logger.logging(
                f"[Rev5.2-OPT] Persistent HardNegativeMiner ready "
                f"(faiss_gpu={self._hard_neg_miner._use_faiss})"
            )
        except Exception as _miner_err:
            self.logger.logging(
                f"[Rev5.2-OPT] HardNegativeMiner init skipped: {_miner_err}"
            )

        # ── Epoch loop ──────────────────────────────────────────────────────
        for epoch in range(args.epoch):
            t1 = time()

            # Temperature annealing (TEX §4.3)
            tau: float = max(
                args.tau_min,
                args.tau_max * (args.tau_gamma**epoch),
            )
            in_warmup: bool = epoch < args.warmup_epochs

            # Rev5.2: CL Warmup Ramp (linear 0→1 scaling).
            # Ablation A4_no_ramp disables the ramp → abrupt 0→1 jump at warmup end.
            cl_ramp_epochs = getattr(args, 'cl_ramp_epochs', 20)
            delay_hard_negs_epoch = getattr(args, 'delay_hard_negs_epoch', 50)
            if in_warmup:
                ramp_weight = 0.0
            elif not self._enable_cl_ramp or cl_ramp_epochs <= 0:
                ramp_weight = 1.0
            else:
                ramp_weight = min(1.0, (epoch - args.warmup_epochs) / float(cl_ramp_epochs))

            # Rev5.2: Delayed FAISS hard negatives.
            # Ablation A5_no_delay disables the delay → hard negs from epoch 0.
            if in_warmup:
                use_hard_negs = False
            elif not self._enable_delayed_faiss:
                use_hard_negs = True
            else:
                use_hard_negs = epoch > delay_hard_negs_epoch

            # Acceleration Guide §B1: skip VICReg covariance term on
            # odd-indexed epochs to save the [D, D] matmul. The cov term
            # contributes weight 1.0 vs sim+var=50.0, so dropping half of
            # its updates is empirically NDCG-neutral.
            vicreg_compute_cov: bool = (
                (not self._vicreg_lazy_cov) or (epoch % 2 == 0)
            )

            # Shared params for GradNorm (Rev5.2 Hybrid Balancer)
            shared_params = self._get_shared_params() if isinstance(self.balancer, HybridLossBalancer) else None

            # Epoch-level loss accumulators
            loss_acc = 0.0
            bpr_acc = 0.0
            u2u_acc = 0.0
            i2i_acc = 0.0
            aln_acc = 0.0
            dir_acc = 0.0
            n_batch: int = data_generator.n_train // args.batch_size + 1

            # ── Mini-batch loop ──────────────────────────────────────────────
            for _ in range(n_batch):
                self.model.train()
                self.projector.train()
                self.optimizer.zero_grad(set_to_none=True)

                # BPR sampling
                users, pos_items, neg_items = data_generator.sample()

                # ── AMP autocast: wraps forward + loss (NOT backward) ────────
                # Acceleration Guide §3 — dtype configurable via MMHCL_AMP_DTYPE.
                with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=amp_dtype):

                    # Forward pass — student (online) network with checkpointing
                    (
                        u_final,
                        i_final,
                        ii_final,
                        uu_final,
                        ii_layers,
                        uu_layers,
                        u_bip,
                        i_bip,
                    ) = self.model.forward_plus(
                        self.UI_mat,
                        self.Item_mat,
                        self.User_mat,
                        checkpoint_threshold=args.checkpoint_threshold,
                    )

                    # BPR loss
                    u_g = u_final[users]
                    pos_g = i_final[pos_items]
                    neg_g = i_final[neg_items]
                    batch_mf, batch_emb, _ = self.bpr_loss(u_g, pos_g, neg_g)
                    bpr_term = batch_mf + batch_emb

                    device = u_final.device

                    if in_warmup:
                        # Warmup: only BPR
                        batch_loss = bpr_term
                        batch_u2u = torch.zeros(1, device=device)
                        batch_i2i = torch.zeros(1, device=device)
                        batch_aln = torch.zeros(1, device=device)
                        batch_dir = torch.zeros(1, device=device)

                    else:
                        # ── u2u: VICReg on adjacent user-hypergraph layers (Rev5.1) ──
                        # User-side ASW: soft topology weights from w_ema_u
                        user_t = torch.tensor(users, dtype=torch.long)
                        soft_w_u: torch.Tensor | None = None
                        if self.w_ema_u is not None:
                            u_rows = self.w_ema_u.get_batch_weights(
                                user_t.cpu(), device=device
                            )
                            # Mean over neighbor dimension → per-user importance scalar [B]
                            soft_w_u = u_rows.mean(dim=-1)
                            if soft_w_u.size(0) != len(users):
                                soft_w_u = None  # safety guard

                        u_pairs = build_neighbor_layer_pairs(uu_layers, max_hops=self._g_layers)
                        u2u_terms: list[torch.Tensor] = []
                        for h_l, h_lp1 in u_pairs:
                            z1 = self.projector(h_l[users])
                            z2 = self.projector(h_lp1[users])
                            u2u_terms.append(
                                vicreg_loss(
                                    z1,
                                    z2,
                                    sim_weight=getattr(args, 'vicreg_sim_weight', 25.0),
                                    var_weight=getattr(args, 'vicreg_var_weight', 25.0),
                                    cov_weight=getattr(args, 'vicreg_cov_weight', 1.0),
                                    soft_weights=soft_w_u,
                                    compute_cov=vicreg_compute_cov,
                                )
                            )
                        batch_u2u = (
                            torch.stack(u2u_terms).mean()
                            if u2u_terms
                            else torch.zeros(1, device=device)
                        )

                        # ── i2i: Chunked InfoNCE on adjacent item-hypergraph layers ──
                        # With hard negative mining (Rev5.1)
                        i_pairs = build_neighbor_layer_pairs(ii_layers, max_hops=self._g_layers)
                        i2i_terms: list[torch.Tensor] = []
                        item_t = torch.tensor(pos_items, dtype=torch.long, device=device)

                        # Mine hard negatives once per step (shared across layer pairs)
                        hard_negs: torch.Tensor | None = None
                        n_hard = getattr(args, 'n_hard_neg', 10)
                        if use_hard_negs and n_hard > 0 and ii_layers:
                            with torch.no_grad():
                                # Use final item embeddings for ANN similarity
                                all_item_emb = ii_final.detach()
                                batch_item_emb = all_item_emb[pos_items]
                                interaction_mask = build_interaction_mask(
                                    batch_items=pos_items,
                                    train_items=data_generator.train_items,
                                    n_items=self.n_items,
                                    batch_users=users,
                                ).to(device)
                                hard_negs = mine_hard_negatives_faiss(
                                    query_embs=batch_item_emb,
                                    all_embs=all_item_emb,
                                    interaction_mask=interaction_mask,
                                    n_hard_neg=n_hard,
                                    pool_k=getattr(args, 'hard_neg_pool_k', 64),
                                    miner=self._hard_neg_miner,
                                )  # [B, n_hard, d]

                        for h_l, h_lp1 in i_pairs:
                            q = h_l[pos_items]
                            k = h_lp1[pos_items]

                            # Adaptive sample weighting from W_ema (item-side only)
                            dyn_w: torch.Tensor | None = None
                            if self.w_ema_i is not None:
                                w_rows = self.w_ema_i.get_batch_weights(
                                    item_t.cpu(), device=device
                                )  # [B, n_items]
                                col_idx = item_t.cpu().long()
                                if col_idx.max() < w_rows.size(1):
                                    dyn_w = w_rows[:, col_idx].to(device)  # [B, B]
                                    if dyn_w.size(0) != q.size(0):
                                        dyn_w = None  # safety guard

                            i2i_terms.append(
                                chunked_info_nce_loss(
                                    q,
                                    k,
                                    tau=tau,
                                    chunk_size=args.info_nce_chunk_size,
                                    dynamic_weights=dyn_w,
                                    hard_negatives=hard_negs,
                                    hard_neg_weight=getattr(args, 'hard_neg_weight', 0.5),
                                )
                            )
                        batch_i2i = (
                            torch.stack(i2i_terms).mean()
                            if i2i_terms
                            else torch.zeros(1, device=device)
                        )

                        # ── Soft BYOL cross-view alignment via EMA teacher ────────
                        # Teacher produces stable stop-gradient targets (TEX §4.4 Alg.1 Step 6)
                        # sg(f_ξ(B^(L))) — the EMA model runs without grad
                        if self._enable_soft_byol:
                            with torch.no_grad():
                                (_, _, ii_final_t, uu_final_t, _, _, u_bip_t, i_bip_t) = (
                                    self.ema_model.forward_plus(
                                        self.UI_mat,
                                        self.Item_mat,
                                        self.User_mat,
                                        checkpoint_threshold=-1,  # no checkpointing in teacher
                                    )
                                )
                            batch_aln = soft_byol_alignment(
                                ii_final, i_bip_t
                            ) + soft_byol_alignment(uu_final, u_bip_t)
                        else:
                            batch_aln = torch.zeros(1, device=device)

                        # ── Dirichlet energy regularisation (mini-batch) ──────────
                        if self._enable_dirichlet:
                            item_batch_idx = item_t  # [B_i], already on device
                            user_batch_idx = torch.tensor(
                                users, dtype=torch.long, device=device
                            )  # [B_u]
                            batch_dir = _sparse_dirichlet_energy_batch(
                                ii_final, self.Item_mat, item_batch_idx
                            ) + _sparse_dirichlet_energy_batch(
                                uu_final, self.User_mat, user_batch_idx
                            )
                        else:
                            batch_dir = torch.zeros(1, device=device)

                        # ── Ego-Final Anchor (A7_ego_final re-enable) ─────────────
                        # VICReg between layer-0 (ego) and final-layer embeddings.
                        batch_ego: torch.Tensor | None = None
                        if self._enable_ego_final and uu_layers and ii_layers:
                            h_u_ego = uu_layers[0][users]
                            h_u_fin = uu_final[users]
                            h_i_ego = ii_layers[0][pos_items]
                            h_i_fin = ii_final[pos_items]
                            ego_u = vicreg_loss(
                                self.projector(h_u_ego),
                                self.projector(h_u_fin),
                                sim_weight=getattr(args, "vicreg_sim_weight", 25.0),
                                var_weight=getattr(args, "vicreg_var_weight", 25.0),
                                cov_weight=getattr(args, "vicreg_cov_weight", 1.0),
                                compute_cov=vicreg_compute_cov,
                            )
                            ego_i = vicreg_loss(
                                self.projector(h_i_ego),
                                self.projector(h_i_fin),
                                sim_weight=getattr(args, "vicreg_sim_weight", 25.0),
                                var_weight=getattr(args, "vicreg_var_weight", 25.0),
                                cov_weight=getattr(args, "vicreg_cov_weight", 1.0),
                                compute_cov=vicreg_compute_cov,
                            )
                            batch_ego = ego_u + ego_i

                        # ── Gate u2u/i2i by the NLCL toggle (A1 ablation) ─────────
                        if not self._enable_nlcl:
                            batch_u2u = torch.zeros(1, device=device)
                            batch_i2i = torch.zeros(1, device=device)

                        # ── Rev5.2: Apply CL Warmup Ramp to ALL CL losses ────────
                        batch_u2u = batch_u2u * ramp_weight
                        batch_i2i = batch_i2i * ramp_weight
                        batch_aln = batch_aln * ramp_weight
                        batch_dir = batch_dir * ramp_weight
                        if batch_ego is not None:
                            batch_ego = batch_ego * ramp_weight

                        # ── Rev5.2: balancer-driven combination (dynamic n_tasks) ─
                        task_losses: list[torch.Tensor] = [bpr_term]
                        for name in self._task_names[1:]:
                            if name == "u2u":
                                task_losses.append(batch_u2u)
                            elif name == "i2i":
                                task_losses.append(batch_i2i)
                            elif name == "align":
                                task_losses.append(batch_aln)
                            elif name == "dir":
                                task_losses.append(batch_dir)
                            elif name == "ego_final":
                                task_losses.append(
                                    batch_ego if batch_ego is not None
                                    else torch.zeros(1, device=device)
                                )

                        if isinstance(self.balancer, HybridLossBalancer):
                            batch_loss = self.balancer(
                                task_losses,
                                epoch=epoch,
                                shared_params=shared_params,
                            )
                        else:
                            batch_loss = self.balancer(task_losses)

                # Back-propagation (AMP-aware)
                amp_scaler.scale(batch_loss).backward(retain_graph=False)
                amp_scaler.step(self.optimizer)
                amp_scaler.update()

                # ── EMA teacher update (Algorithm 1, Step 11: ξ ← β·ξ + (1-β)·θ) ──
                update_ema_teacher(self.model, self.ema_model, args.ema_momentum)

                # Update W_ema matrices with current embeddings (post warmup)
                if not in_warmup:
                    pos_t = torch.tensor(pos_items, dtype=torch.long)
                    if self.w_ema_i is not None:
                        with torch.no_grad():
                            self.w_ema_i.step_update(
                                ii_final[pos_items].detach().cpu(),
                                pos_t,
                                epoch,
                            )
                    if self.w_ema_u is not None:
                        user_t_cpu = torch.tensor(users, dtype=torch.long)
                        with torch.no_grad():
                            self.w_ema_u.step_update(
                                uu_final[users].detach().cpu(),
                                user_t_cpu,
                                epoch,
                            )

                # Accumulate scalars
                loss_acc += batch_loss.item()
                bpr_acc += bpr_term.item()
                u2u_acc += batch_u2u.item()
                i2i_acc += batch_i2i.item()
                aln_acc += batch_aln.item()
                dir_acc += batch_dir.item()

            # ── End mini-batch loop ──────────────────────────────────────────
            self.lr_scheduler.step()

            # Free last-iteration GPU tensors
            try:
                del u_final, i_final, ii_final, uu_final
                del ii_layers, uu_layers, u_bip, i_bip
                del u_g, pos_g, neg_g
                del ii_final_t, uu_final_t, u_bip_t, i_bip_t
            except NameError:
                pass

            # NaN guard
            if math.isnan(loss_acc):
                self.logger.logging("ERROR: loss is nan.")
                if args.use_wandb and wandb is not None:
                    wandb.finish(exit_code=1)
                return 0.0

            # One-time Dirichlet scale diagnostic (first CL epoch)
            if epoch == args.warmup_epochs and not in_warmup:
                avg_dir = dir_acc / max(n_batch, 1)
                avg_bpr = bpr_acc / max(n_batch, 1)
                ratio = avg_dir / max(avg_bpr, 1e-12)
                self.logger.logging(
                    "[DIAG] First CL epoch Dirichlet scale: "
                    "raw_dir=%.6f  bpr=%.5f  ratio=%.4f  "
                    "(if ratio < 0.01, consider increasing --dirichlet_weight)"
                    % (avg_dir, avg_bpr, ratio)
                )

            # W&B training metrics
            if args.use_wandb and wandb is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": loss_acc,
                        "train/bpr": bpr_acc,
                        "train/u2u": u2u_acc,
                        "train/i2i": i2i_acc,
                        "train/align": aln_acc,
                        "train/dirichlet": dir_acc,
                        "train/dirichlet_raw_avg": dir_acc / max(n_batch, 1),
                        "train/tau": tau,
                        "train/ramp_weight": ramp_weight,
                        "train/hard_negs_active": int(use_hard_negs),
                        "train/warmup": int(in_warmup),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        # Acceleration Guide §B1 audit trail
                        "train/vicreg_compute_cov": int(vicreg_compute_cov),
                    }
                )

            # Build performance string (same pattern as main.py for log compatibility)
            phase_tag = "[WARMUP]" if in_warmup else "[tau=%.4f ramp=%.2f%s]" % (
                tau, ramp_weight, " FAISS" if use_hard_negs else "",
            )

            if (epoch + 1) % args.verbose != 0:
                # Non-evaluation epoch: only log training loss
                perf_str = (
                    "Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f + %.5f] %s"
                    % (
                        epoch,
                        time() - t1,
                        loss_acc,
                        bpr_acc / n_batch,
                        u2u_acc / n_batch,
                        i2i_acc / n_batch,
                        aln_acc / n_batch,
                        dir_acc / n_batch,
                        phase_tag,
                    )
                )
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)
                continue

            # ── Evaluation epoch (every verbose epochs) ──────────────────────
            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret: MetricsDict = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)
            t3 = time()

            loss_loger.append(loss_acc)
            rec_loger.append(ret["recall"])
            pre_loger.append(ret["precision"])
            ndcg_loger.append(ret["ndcg"])
            hit_loger.append(ret["hit_ratio"])

            if args.verbose > 0:
                perf_str = (
                    "Epoch %d [%.1fs + %.1fs]: "
                    "train==[%.5f=%.5f + %.5f + %.5f + %.5f + %.5f] %s, "
                    "recall=[%.5f, %.5f], precision=[%.5f, %.5f], "
                    "hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]"
                    % (
                        epoch,
                        t2 - t1,
                        t3 - t2,
                        loss_acc,
                        bpr_acc / n_batch,
                        u2u_acc / n_batch,
                        i2i_acc / n_batch,
                        aln_acc / n_batch,
                        dir_acc / n_batch,
                        phase_tag,
                        ret["recall"][0],
                        ret["recall"][-1],
                        ret["precision"][0],
                        ret["precision"][-1],
                        ret["hit_ratio"][0],
                        ret["hit_ratio"][-1],
                        ret["ndcg"][0],
                        ret["ndcg"][-1],
                    )
                )
                self.logger.logging(perf_str)

            # W&B validation metrics
            if args.use_wandb and wandb is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "val/recall@10": ret["recall"][0],
                        "val/recall@20": ret["recall"][-1],
                        "val/precision@10": ret["precision"][0],
                        "val/precision@20": ret["precision"][-1],
                        "val/ndcg@10": ret["ndcg"][0],
                        "val/ndcg@20": ret["ndcg"][-1],
                        "val/hit@10": ret["hit_ratio"][0],
                        "val/hit@20": ret["hit_ratio"][-1],
                    }
                )

            # ReduceLROnPlateau step
            if self.reduce_lr_scheduler is not None:
                self.reduce_lr_scheduler.step(ret["recall"][-1])

            current_val_recall = float(ret["recall"][-1])
            if current_val_recall > best_val_recall:
                best_val_recall = current_val_recall

            # Optuna pruning support
            if self.optuna_trial is not None:
                self.optuna_trial.report(current_val_recall, eval_step)
                eval_step += 1
                if self.optuna_trial.should_prune():
                    self.logger.logging("Optuna pruned at epoch %d" % epoch)
                    if args.use_wandb and wandb is not None:
                        wandb.finish(exit_code=0)
                    import optuna as _optuna_mod

                    raise _optuna_mod.TrialPruned()

            # ── Early stopping ───────────────────────────────────────────────
            improved = False
            if (
                ret["recall"][1] > best_recall + args.early_stopping_min_delta
                or ret["ndcg"][1] > best_ndcg + args.early_stopping_min_delta
            ):
                if ret["recall"][1] > best_recall:
                    best_recall = ret["recall"][1]
                if ret["ndcg"][1] > best_ndcg:
                    best_ndcg = ret["ndcg"][1]
                improved = True

            if improved:
                test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging(
                    "Test_Recall@%d: %.8f   Test_Precision@%d: %.8f   "
                    "Test_NDCG@%d: %.8f"
                    % (
                        eval(args.Ks)[1],
                        test_ret["recall"][1],
                        eval(args.Ks)[1],
                        test_ret["precision"][1],
                        eval(args.Ks)[1],
                        test_ret["ndcg"][1],
                    )
                )
                if args.use_wandb and wandb is not None:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "test/recall@20": test_ret["recall"][1],
                            "test/precision@20": test_ret["precision"][1],
                            "test/ndcg@20": test_ret["ndcg"][1],
                        }
                    )
                stopping_step = 0
                if args.early_stopping_restore_best:
                    best_model_state = copy.deepcopy(self.model.state_dict())

            elif epoch + 1 >= args.early_stopping_min_epochs:
                if stopping_step < args.early_stopping_patience:
                    stopping_step += 1
                    self.logger.logging(
                        "#####Early stopping steps: %d #####" % stopping_step
                    )
                else:
                    self.logger.logging("#####Early stop! #####")
                    if (
                        args.early_stopping_restore_best
                        and best_model_state is not None
                    ):
                        self.model.load_state_dict(best_model_state)
                        self.logger.logging("Restored best model weights.")
                    fname = "Model.epoch=%d.pth" % epoch
                    torch.save(self.model.state_dict(), os.path.join(path, fname))
                    break
            else:
                self.logger.logging(
                    "Epoch %d: no improvement, but "
                    "min_epochs=%d not reached yet"
                    % (epoch, args.early_stopping_min_epochs)
                )

        # ── Post-training: log best test results ─────────────────────────────
        best_test_recall = 0.0
        if isinstance(test_ret, dict):
            Ks_list: list[int] = eval(args.Ks)
            best_test_recall = float(test_ret["recall"][1])
            # These log lines are parsed by the notebook's results cell
            self.logger.logging(
                "BEST_Test_Recall@%d: %.8f" % (Ks_list[1], test_ret["recall"][1])
            )
            self.logger.logging(
                "BEST_Test_Precision@%d: %.8f" % (Ks_list[1], test_ret["precision"][1])
            )
            self.logger.logging(
                "BEST_Test_NDCG@%d: %.8f" % (Ks_list[1], test_ret["ndcg"][1])
            )
            if args.use_wandb and wandb is not None:
                wandb.summary["best_test_recall@20"] = test_ret["recall"][1]
                wandb.summary["best_test_precision@20"] = test_ret["precision"][1]
                wandb.summary["best_test_ndcg@20"] = test_ret["ndcg"][1]

        self.logger.logging(str(test_ret))
        self.logger.logging_sum("%s:%s" % (path_name, str(test_ret)))

        if args.use_wandb and wandb is not None:
            wandb.finish()

        return best_val_recall if return_validation else best_test_recall


# ===========================================================================
#  Public entry point (compatible with Optuna and notebook subprocess calls)
# ===========================================================================
def train_evaluation_loop(
    args_ns=None,
    optuna_trial=None,
    return_validation: bool = False,
) -> float:
    """
    One complete MMHCL+ training cycle.

    Callable from:
      - CLI:   ``python main_mmhcl_plus.py --dataset Baby ...``
      - Notebook subprocess call (identical to main.py interface)
      - Optuna hyper-parameter search
    """
    global args, path_name, path, record_path, wandb

    if args_ns is not None:
        args = args_ns
        import utility.load_data as _ld

        _ld.args = args

    # --- Ablation variant override (must happen before any arg is read) ---
    ablation_name: str = getattr(args, "ablation_variant", "") or ""
    if ablation_name:
        if ablation_name not in _available_ablation_variants():
            raise ValueError(
                f"Unknown ablation variant '{ablation_name}'. "
                f"Available: {_available_ablation_variants()}"
            )
        variant = _get_ablation_variant(ablation_name)
        _apply_ablation_to_args(variant, args)
        print(
            f"[Ablation] Running variant '{variant.name}' "
            f"-- {variant.notes}"
        )

    wandb = None
    if args.use_wandb:
        try:
            import wandb as _wandb

            wandb = _wandb
        except ImportError:
            args.use_wandb = 0

    path_name, path, record_path = _build_paths(args)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(record_path).mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)

    config: dict[str, Any] = {
        "n_users": data_generator.n_users,
        "n_items": data_generator.n_items,
        "UI_mat": data_generator.get_UI_mat(),
        "User_mat": data_generator.get_U2U_mat(),
    }
    if args.dataset == "Tiktok":
        config["Item_mat"] = data_generator.get_tiktok_I2I_Hypergraph_mul_mat()
    elif args.dataset in ["Clothing", "Sports", "Baby"]:
        config["Item_mat"] = data_generator.get_I2I_Hypergraph_mul_mat()

    trainer = MMHCLPlusTrainer(data_config=config, optuna_trial=optuna_trial)
    best_recall: float = trainer.train(return_validation=return_validation)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return best_recall


# ===========================================================================
#  CLI entry point
# ===========================================================================
if __name__ == "__main__":
    train_evaluation_loop()
