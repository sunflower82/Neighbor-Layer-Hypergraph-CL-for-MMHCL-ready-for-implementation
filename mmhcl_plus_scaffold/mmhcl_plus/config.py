from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ModelConfig:
    in_dim: int = 64
    hidden_dim: int = 64
    projector_hidden_dim: int = 2048
    projector_out_dim: int = 8192
    n_layers: int = 3
    max_g_layers: int = 2
    fusion_hidden_dim: int = 128
    use_checkpoint: bool = True


@dataclass
class LossConfig:
    # Temperature — used as τ_max; anneals toward τ_min (TEX §4.3)
    tau: float = 0.2
    tau_max: float = 0.5  # initial (warm) temperature
    tau_min: float = 0.05  # floor temperature after annealing
    tau_gamma: float = 0.99  # exponential decay factor per epoch
    # Warmup: skip CL losses for the first N epochs (TEX §4.4 code snippet)
    warmup_epochs: int = 5
    # Barlow Twins off-diagonal penalty coefficient
    barlow_lambda: float = 5e-3
    # Chunked InfoNCE chunk size
    info_nce_chunk_size: int = 1024
    # GradNorm α — higher values enforce stricter balance
    gradnorm_alpha: float = 1.5
    # Per-task loss weights (overridden by GradNorm at runtime)
    dirichlet_weight: float = 0.1
    bpr_weight: float = 1.0
    u2u_weight: float = 1.0
    i2i_weight: float = 1.0
    align_weight: float = 1.0


@dataclass
class TopologyConfig:
    ann_backend: str = (
        "auto"  # auto | torch | faiss (YAML / demos; main_mmhcl_plus uses CLI)
    )
    ann_k: int = 32
    ann_metric: str = "ip"
    purification_tau: float = 0.2
    ema_momentum: float = 0.99
    w_ema_alpha: float = 0.9
    update_interval: int = 5


@dataclass
class SystemConfig:
    device: str = "cuda"
    seed: int = 2026
    checkpoint_layers: list[int] = field(default_factory=lambda: [1, 2])
    log_every: int = 10
    epochs: int = 3
    lr: float = 1e-3
    # MMHCL+ Optimization Report: autocast (bf16 preferred) + optional GradScaler (fp16)
    amp_dtype: str = "off"  # off | bf16 | fp16
    grad_clip_max_norm: float = 1.0  # 0.0 disables clipping
    dataloader_num_workers: int = 2
    dataloader_prefetch_factor: int = 2
    compile_projector: bool = False  # torch.compile(projector, mode="reduce-overhead")


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
