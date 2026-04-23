"""Quick smoke test — verifies all new modules import and execute correctly."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

# ── 1. Config ────────────────────────────────────────────────────────────────
from mmhcl_plus.config import load_config

cfg_path = os.path.join(os.path.dirname(__file__), "configs", "mmhcl_plus.yaml")
cfg = load_config(cfg_path)
assert cfg.loss.warmup_epochs == 5
assert cfg.loss.tau_max == 0.5
assert cfg.loss.tau_min == 0.05
assert cfg.loss.tau_gamma == 0.99
print(
    f"[1] Config OK  warmup={cfg.loss.warmup_epochs}  tau_max={cfg.loss.tau_max}  gamma={cfg.loss.tau_gamma}"
)

# ── 2. Checkpointing (use_reentrant=False) ───────────────────────────────────
from mmhcl_plus.systems.checkpointing import profiling_guided_checkpoint

layer = nn.Linear(8, 8)
x = torch.randn(4, 8, requires_grad=True)
out = profiling_guided_checkpoint(layer, x, layer_id=1, checkpoint_layers=[1, 2])
assert out.shape == (4, 8)
out.sum().backward()
print(f"[2] Checkpointing OK  shape={tuple(out.shape)}")

# ── 3. WEMAManager ────────────────────────────────────────────────────────────
from mmhcl_plus.topology.dynamic_ema_weights import WEMAManager

mgr = WEMAManager(n_nodes=64, alpha=0.9, update_interval=5)
mgr.precompute_from_raw([torch.randn(64, 32), torch.randn(64, 16)])
assert mgr.W.shape == (64, 64)
assert mgr.W.dtype == torch.float16
rows = mgr.get_batch_weights(torch.arange(10))
assert rows.shape == (10, 64)
# Lazy EMA update
embs = torch.randn(10, 8)
mgr.step_update(embs, torch.arange(10), epoch=5)
print(f"[3] WEMAManager OK  W={tuple(mgr.W.shape)}  rows={tuple(rows.shape)}")

# ── 4. Losses ─────────────────────────────────────────────────────────────────
from mmhcl_plus.contrast.losses import (
    barlow_twins_loss,
    bpr_loss,
    chunked_info_nce_loss,
    temperature_free_info_nce_loss,
)

z = torch.randn(16, 128)
bt = barlow_twins_loss(z, z + 0.01, lambd=5e-3, soft_weights=torch.rand(16))
ci = chunked_info_nce_loss(
    z[:, :64], z[:, :64], tau=0.2, chunk_size=8, dynamic_weights=torch.rand(16, 16)
)
tf = temperature_free_info_nce_loss(z[:, :32], z[:, :32])
bp = bpr_loss(torch.randn(16), torch.randn(16))
assert bt.ndim == 0 and ci.ndim == 0 and tf.ndim == 0 and bp.ndim == 0
print(
    f"[4] Losses OK  BT={bt.item():.3f}  CI={ci.item():.3f}  TF={tf.item():.3f}  BPR={bp.item():.3f}"
)

# ── 5. LayerwiseEncoder ───────────────────────────────────────────────────────
from mmhcl_plus.model.hypergraph_encoder import LayerwiseEncoder

layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
enc = LayerwiseEncoder(layers, max_g_layers=2, use_checkpoint=True)
h0 = torch.randn(10, 8)


def ckpt_fn(m, x, lid):
    return profiling_guided_checkpoint(m, x, lid, checkpoint_layers=[1, 2])


final, cached = enc(h0, checkpoint_fn=ckpt_fn)
# max_g_layers=2: cache h0 + layers 0 and 1 → 3 entries (indices 0,1 < 2)
assert final.shape == (10, 8)
assert len(cached) == 3, f"Expected 3 cached, got {len(cached)}"
print(
    f"[5] LayerwiseEncoder OK  final={tuple(final.shape)}  cached={len(cached)} layers"
)

# ── 6. Full TwoStageTrainer step ─────────────────────────────────────────────
from mmhcl_plus.contrast import (
    GradNormLossBalancer,
    barlow_twins_loss,
    bpr_loss,
    chunked_info_nce_loss,
    soft_byol_alignment,
)
from mmhcl_plus.model import ExpandedProjector, FusionMLP, MMHCLPlus
from mmhcl_plus.regularizers import dirichlet_energy_minibatch
from mmhcl_plus.trainers import TwoStageTrainer

dim, L, g = 8, 3, 2


def make_enc():
    return LayerwiseEncoder(
        nn.ModuleList([nn.Linear(dim, dim) for _ in range(L)]), max_g_layers=g
    )


model = MMHCLPlus(
    make_enc(), make_enc(), FusionMLP(dim, 16), ExpandedProjector(dim, 64, 128)
)
proj = model.projector_u2u
bal = GradNormLossBalancer(alpha=1.5)
opt = torch.optim.Adam(list(model.parameters()) + list(bal.parameters()), lr=1e-3)

w_u = WEMAManager(n_nodes=16)
w_u.precompute_from_raw([torch.randn(16, 8)])
w_i = WEMAManager(n_nodes=16)
w_i.precompute_from_raw([torch.randn(16, 8)])

trainer = TwoStageTrainer(model, opt, bal, cfg, proj, w_ema_u=w_u, w_ema_i=w_i)

# -- Warmup epoch: CL losses must be zero
trainer.set_epoch(0)
assert trainer._in_warmup(), "Epoch 0 should be in warmup"

# -- Post-warmup epoch
trainer.set_epoch(10)
assert not trainer._in_warmup(), "Epoch 10 should be past warmup"
tau = trainer.current_tau()
expected = max(0.05, 0.5 * (0.99**10))
assert abs(tau - expected) < 1e-6, f"tau mismatch: {tau} vs {expected}"

# Build a tiny batch
n = 16
theta = torch.softmax(torch.randn(n, n), dim=-1)


def lap(idx):
    b = theta[idx][:, idx]
    e = torch.eye(b.shape[0])
    return (e - b).to_sparse()


batch = {
    "x": torch.randn(n, dim),
    "pos_scores": torch.randn(n),
    "neg_scores": torch.randn(n),
    "node_idx": torch.arange(n),
    "lap_getter": lap,
    "user_idx": torch.arange(8),
    "item_idx": torch.arange(8, 16),
}


def infonce_fn(a, b, tau=0.2, dynamic_weights=None):
    return chunked_info_nce_loss(a, b, tau=tau, dynamic_weights=dynamic_weights)


loss_fns = {
    "barlow": barlow_twins_loss,
    "infonce": infonce_fn,
    "soft_byol": soft_byol_alignment,
    "bpr": bpr_loss,
    "dirichlet": dirichlet_energy_minibatch,
}

metrics = trainer.train_step(batch, checkpoint_fn=None, loss_fns=loss_fns)
assert isinstance(metrics["loss"], float)
assert metrics["u2u"] > 0.0, "u2u loss must be non-zero after warmup"
assert metrics["i2i"] > 0.0, "i2i loss must be non-zero after warmup"
print(
    f"[6] TwoStageTrainer OK  loss={metrics['loss']:.4f}  "
    f"bpr={metrics['bpr']:.4f}  u2u={metrics['u2u']:.4f}  "
    f"i2i={metrics['i2i']:.4f}  tau={metrics['tau']:.4f}"
)

print()
print("=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
