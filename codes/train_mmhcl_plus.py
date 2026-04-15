"""
MMHCL+ demo training script — wires the full two-stage pipeline end-to-end.

This script uses synthetic DummyDataset / DummyBranchLayer objects so it can run
without the actual MMHCL dataloader.  See README_MMHCL_PLUS.md for integration
points with the real MMHCL repository.

Run:
    python train_mmhcl_plus.py --config configs/mmhcl_plus.yaml

Key design choices implemented (TEX §4.2–4.4):
  1. W_ema is pre-computed from synthetic raw features before training starts.
  2. Per-epoch call to trainer.set_epoch(epoch) drives temperature annealing
     and warmup gating.
  3. batch['user_idx'] / batch['item_idx'] are passed so the trainer can fetch
     soft weights via WEMAManager.get_batch_weights (lazy row fetch).
  4. profiling_guided_checkpoint is injected as checkpoint_fn to honour the
     checkpoint_layers config.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mmhcl_plus.config import load_config
from mmhcl_plus.contrast import (
    UncertaintyLossBalancer,
    bpr_loss,
    chunked_info_nce_loss,
    soft_byol_alignment,
    temperature_free_info_nce_loss,
    vicreg_loss,
)
from mmhcl_plus.model import ExpandedProjector, FusionMLP, LayerwiseEncoder, MMHCLPlus
from mmhcl_plus.regularizers import dirichlet_energy_minibatch
from mmhcl_plus.systems import profiling_guided_checkpoint, to_device_async
from mmhcl_plus.topology.dynamic_ema_weights import WEMAManager
from mmhcl_plus.trainers import TwoStageTrainer
from mmhcl_plus.utils import count_parameters, set_seed

# ---------------------------------------------------------------------------
# Dummy components (replace with real MMHCL layers in production)
# ---------------------------------------------------------------------------


class DummyBranchLayer(nn.Module):
    """Residual-style dummy conv layer (stands in for a real hypergraph conv)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DummyDataset(Dataset):
    """
    Synthetic dataset that mimics the batch dictionary expected by TwoStageTrainer.

    Each 'item' in the dataset is a single mini-batch dict (batch_size=128 nodes).
    The dataset returns 16 such batches per epoch (simulating ~2 000 users).
    """

    def __init__(self, n_nodes: int = 512, dim: int = 64) -> None:
        self.n_nodes = n_nodes
        self.dim = dim
        self.x_all = torch.randn(n_nodes, dim)
        # Synthetic Laplacian (random doubly-stochastic matrix → I - Θ)
        theta = torch.softmax(torch.randn(n_nodes, n_nodes), dim=-1)
        self.theta = theta

    def __len__(self) -> int:
        return 16

    def __getitem__(self, _idx: int) -> dict:
        node_idx = torch.randperm(self.n_nodes)[:128]
        x = self.x_all[node_idx]
        pos_scores = torch.randn(128)
        neg_scores = torch.randn(128)

        # Synthetic user / item split: first 64 nodes = users, last 64 = items
        user_idx = node_idx[:64]
        item_idx = node_idx[64:]

        theta = self.theta  # capture for closure

        def lap_getter(bidx: torch.Tensor) -> torch.Tensor:
            block = theta.index_select(0, bidx.cpu()).index_select(1, bidx.cpu())
            eye = torch.eye(block.size(0), dtype=block.dtype)
            return (eye - block).to_sparse()

        return {
            "x": x,
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "node_idx": torch.arange(x.size(0)),
            "lap_getter": lap_getter,
            "user_idx": user_idx,
            "item_idx": item_idx,
        }


def collate_single(batch: list) -> dict:
    """Identity collate: the dataset already returns a single-batch dict."""
    return batch[0]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_demo_model(cfg) -> tuple[nn.Module, nn.Module]:
    """Build MMHCLPlus with DummyBranchLayer encoders."""
    hyper_layers = nn.ModuleList(
        [DummyBranchLayer(cfg.model.in_dim) for _ in range(cfg.model.n_layers)]
    )
    bip_layers = nn.ModuleList(
        [DummyBranchLayer(cfg.model.in_dim) for _ in range(cfg.model.n_layers)]
    )
    hyper_encoder = LayerwiseEncoder(
        hyper_layers,
        max_g_layers=cfg.model.max_g_layers,
        use_checkpoint=cfg.model.use_checkpoint,
    )
    bip_encoder = LayerwiseEncoder(
        bip_layers,
        max_g_layers=cfg.model.max_g_layers,
        use_checkpoint=cfg.model.use_checkpoint,
    )
    fusion = FusionMLP(cfg.model.in_dim, cfg.model.fusion_hidden_dim)
    projector = ExpandedProjector(
        cfg.model.in_dim,
        cfg.model.projector_hidden_dim,
        cfg.model.projector_out_dim,
    )
    model = MMHCLPlus(hyper_encoder, bip_encoder, fusion, projector)
    return model, projector


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="MMHCL+ demo trainer")
    parser.add_argument("--config", type=str, default="configs/mmhcl_plus.yaml")
    parser.add_argument(
        "--temperature_free",
        action="store_true",
        help="Use temperature-free InfoNCE (τ=1) for the i2i branch",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.system.seed)
    device = torch.device(cfg.system.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model, projector = build_demo_model(cfg)
    model = model.to(device)
    projector = projector.to(device)

    if getattr(cfg.system, "compile_projector", False) and hasattr(torch, "compile"):
        try:
            projector = torch.compile(projector, mode="reduce-overhead")
            print("[train_mmhcl_plus] torch.compile(ExpandedProjector) enabled.")
        except Exception as exc:
            print(f"[train_mmhcl_plus] torch.compile(projector) skipped: {exc}")

    balancer = UncertaintyLossBalancer(num_tasks=getattr(cfg.loss, 'num_tasks', 6)).to(device)

    # projector_u2u is already registered inside MMHCLPlus, so model.parameters()
    # already covers it.  Adding projector.parameters() again would trigger a
    # "duplicate parameters" warning in PyTorch >= 2.0.
    all_params = list(model.parameters()) + list(balancer.parameters())
    optimizer = torch.optim.Adam(all_params, lr=cfg.system.lr)

    # ------------------------------------------------------------------
    # Pre-compute W_ema from synthetic raw features (TEX §4.2, Step 1)
    # In production: pass actual visual / textual raw feature matrices here.
    # ------------------------------------------------------------------
    n_nodes = 512
    raw_visual = torch.randn(n_nodes, 64)
    raw_textual = torch.randn(n_nodes, 64)

    # User-side W: first 256 nodes are "users"
    w_ema_u = WEMAManager(
        n_nodes=n_nodes,
        alpha=cfg.topology.w_ema_alpha,
        update_interval=cfg.topology.update_interval,
    )
    w_ema_u.precompute_from_raw([raw_visual[:n_nodes], raw_textual[:n_nodes]])

    # Item-side W: same shape for the demo (in practice item count may differ)
    w_ema_i = WEMAManager(
        n_nodes=n_nodes,
        alpha=cfg.topology.w_ema_alpha,
        update_interval=cfg.topology.update_interval,
    )
    w_ema_i.precompute_from_raw([raw_visual[:n_nodes], raw_textual[:n_nodes]])

    # ------------------------------------------------------------------
    # Loss function registry
    # ------------------------------------------------------------------
    if args.temperature_free:

        def infonce_fn(a, b, tau=None, dynamic_weights=None, hard_negatives=None, hard_neg_weight=0.5):
            return temperature_free_info_nce_loss(a, b)
    else:

        def infonce_fn(a, b, tau=None, dynamic_weights=None, hard_negatives=None, hard_neg_weight=0.5):
            t = tau if tau is not None else cfg.loss.tau_max
            return chunked_info_nce_loss(
                a,
                b,
                tau=t,
                chunk_size=cfg.loss.info_nce_chunk_size,
                dynamic_weights=dynamic_weights,
                hard_negatives=hard_negatives,
                hard_neg_weight=hard_neg_weight,
            )

    loss_fns = {
        "vicreg": vicreg_loss,
        "infonce": infonce_fn,
        "soft_byol": soft_byol_alignment,
        "bpr": bpr_loss,
        "dirichlet": dirichlet_energy_minibatch,
    }

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = TwoStageTrainer(
        model=model,
        optimizer=optimizer,
        balancer=balancer,
        cfg=cfg,
        projector=projector,
        w_ema_u=w_ema_u,
        w_ema_i=w_ema_i,
    )

    # Profiling-guided checkpoint function (injected so encoder stays policy-agnostic)
    def checkpoint_fn(module: nn.Module, x: torch.Tensor, lid: int) -> torch.Tensor:
        return profiling_guided_checkpoint(
            module, x, lid, checkpoint_layers=cfg.system.checkpoint_layers
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = DummyDataset(n_nodes=n_nodes, dim=cfg.model.in_dim)
    nw = int(getattr(cfg.system, "dataloader_num_workers", 0))
    dl_common = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_single,
    )
    if nw > 0:
        dataloader = DataLoader(
            **dl_common,
            num_workers=nw,
            pin_memory=(device.type == "cuda"),
            persistent_workers=True,
            prefetch_factor=int(getattr(cfg.system, "dataloader_prefetch_factor", 2)),
        )
    else:
        dataloader = DataLoader(**dl_common)

    print(
        f"Model params: {count_parameters(model) + count_parameters(projector) + count_parameters(balancer):,}"
    )
    print(
        f"Device: {device}  |  Warmup epochs: {cfg.loss.warmup_epochs}"
        f"  |  Total epochs: {cfg.system.epochs}"
    )
    print(
        f"Temperature schedule: tau_max={cfg.loss.tau_max}, "
        f"tau_min={cfg.loss.tau_min}, gamma={cfg.loss.tau_gamma}"
    )
    print()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(cfg.system.epochs):
        trainer.set_epoch(epoch)  # drives warmup gate + temperature annealing

        epoch_metrics: dict[str, float] = {}

        for step, batch in enumerate(dataloader, start=1):
            gpu_batch = to_device_async(batch, device)

            metrics = trainer.train_step(
                batch=gpu_batch,
                checkpoint_fn=checkpoint_fn,
                loss_fns=loss_fns,
            )

            for k, val in metrics.items():
                if isinstance(val, float):
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + val

            if step % cfg.system.log_every == 0:
                mode = "WARMUP" if metrics["warmup"] else f"tau={metrics['tau']:.4f}"
                ego_str = f"  ego={metrics.get('ego_final', 0.0):.4f}"
                print(
                    f"epoch={epoch + 1:3d}/{cfg.system.epochs} "
                    f"step={step:2d} "
                    f"[{mode}] "
                    f"loss={metrics['loss']:.4f}  "
                    f"bpr={metrics['bpr']:.4f}  "
                    f"u2u={metrics['u2u']:.4f}  "
                    f"i2i={metrics['i2i']:.4f}  "
                    f"align={metrics['align']:.4f}  "
                    f"dir={metrics['dir']:.6f}"
                    + ego_str
                )

        n_steps = len(dataloader)
        print(
            f"  >> epoch {epoch + 1} avg: "
            + "  ".join(
                f"{k}={v / n_steps:.4f}"
                for k, v in epoch_metrics.items()
                if isinstance(v, float)
            )
        )
        print()


if __name__ == "__main__":
    main()
