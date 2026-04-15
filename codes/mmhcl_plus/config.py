"""
MMHCL+ Configuration --- Revision 5.1

Changes from Rev44:
  - BARLOW_PROJ_DIM=8192 -> VICREG_PROJ_DIM=1024
  - projector_hidden_dim: 2048 -> 512
  - Removed: barlow_lambda, gradnorm_alpha
  - Added: vicreg_{sim,var,cov}_weight, num_tasks=6, ego_final_weight
  - Added: TopologyConfig.use_svd_filtering, svd_top_k
  - warmup_epochs: 10 -> 5 (Rev5.1 spec)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import yaml

# -- Global architectural constant (TEX Rev5.1 Section 2.5) --------------
# VICReg projection dimension D=1024 (was BARLOW_PROJ_DIM=8192 in Rev44).
# Eliminates O(B*D^2) cross-correlation bottleneck -> ~80% VRAM saving.
VICREG_PROJ_DIM: int = 1024

# Backward compatibility alias (deprecated --- will be removed in v6)
BARLOW_PROJ_DIM: int = VICREG_PROJ_DIM


@dataclass
class ModelConfig:
    in_dim: int = 64
    hidden_dim: int = 64
    projector_hidden_dim: int = 512       # was 2048 for Barlow's D=8192
    projector_out_dim: int = VICREG_PROJ_DIM  # 1024
    n_layers: int = 3
    max_g_layers: int = 2
    fusion_hidden_dim: int = 128
    use_checkpoint: bool = True
    checkpoint_threshold: int | None = None  # None = disabled for 24GB GPU


@dataclass
class LossConfig:
    # Temperature annealing schedule (TEX Rev5.1 Section 5.2 code snippet):
    #   tau(t) = max(tau_min, tau_max * tau_gamma^t)
    tau_max: float = 0.5
    tau_min: float = 0.05
    tau_gamma: float = 0.99

    # Warmup: skip CL losses for first N epochs
    warmup_epochs: int = 5

    # VICReg coefficients (Bardes et al., ICLR 2022)
    vicreg_sim_weight: float = 25.0   # invariance (MSE)
    vicreg_var_weight: float = 25.0   # variance (std >= 1)
    vicreg_cov_weight: float = 1.0    # covariance (decorrelation)

    # Chunked InfoNCE
    info_nce_chunk_size: int = 512

    # Homoscedastic Uncertainty Balancing --- 6 tasks
    num_tasks: int = 6

    # Per-task initial weights (informational; uncertainty balancer learns sigma_i)
    bpr_weight: float = 1.0
    u2u_weight: float = 1.0
    i2i_weight: float = 1.0
    align_weight: float = 1.0
    dirichlet_weight: float = 1.0
    ego_final_weight: float = 1.0

    # Hard negative weighting in InfoNCE denominator (Rev5.1 canonical: 0.5)
    # Conservative default prevents hard negatives dominating before warm-up completes.
    hard_neg_weight: float = 0.5


@dataclass
class TopologyConfig:
    ann_backend: str = "faiss"
    ann_k: int = 32
    ann_metric: str = "ip"
    purification_tau: float = 0.2
    ema_momentum: float = 0.99
    w_ema_alpha: float = 0.9
    update_interval: int = 5

    # SVD Spectral Augmentation (TEX Rev5.1 Section 2.2, Corollary 2.1)
    use_svd_filtering: bool = True
    svd_top_k: int = 10

    # FAISS Hard Negative Mining (TEX Rev5.1 Section 2.4)
    n_hard_neg: int = 10
    hard_neg_pool_k: int = 64


@dataclass
class SystemConfig:
    device: str = "cuda"
    seed: int = 2026
    checkpoint_layers: list[int] = field(default_factory=lambda: [1, 2])
    log_every: int = 10
    epochs: int = 200
    lr: float = 1e-3


@dataclass
class MMHCLPlusConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> MMHCLPlusConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    defaults = {
        "model": ModelConfig().__dict__.copy(),
        "loss": LossConfig().__dict__.copy(),
        "topology": TopologyConfig().__dict__.copy(),
        "system": SystemConfig().__dict__.copy(),
    }
    merged = _deep_update(defaults, raw)
    return MMHCLPlusConfig(
        model=ModelConfig(**merged["model"]),
        loss=LossConfig(**merged["loss"]),
        topology=TopologyConfig(**merged["topology"]),
        system=SystemConfig(**merged["system"]),
    )
