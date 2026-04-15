"""
parser.py — Command-Line Argument Definitions
===============================================

Centralises every hyperparameter and configuration flag used by MMHCL.
All other modules import `parse_args()` to access these values.

Usage examples:
    # Train on Clothing with default settings
    python main.py --dataset Clothing

    # Train with a specific seed and W&B logging
    python main.py --dataset Clothing --seed 42 --use_wandb 1

    # Override key hyperparameters
    python main.py --dataset Sports --batch_size 512 --epoch 500 --lr 0.001
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse and return all MMHCL command-line arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="MMHCL: Multi-Modal Hypergraph Contrastive Learning "
        "for Recommendation"
    )

    # =====================================================================
    #  General / Data
    # =====================================================================
    parser.add_argument(
        "--data_path",
        nargs="?",
        default="../data/",
        help="Root path to all dataset folders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for reproducibility (Python, NumPy, PyTorch).",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="Tiktok",
        help="Dataset name: {Tiktok, Sports, Clothing, Baby}. "
        "Must match a folder under data_path.",
    )

    # =====================================================================
    #  Training Schedule
    # =====================================================================
    parser.add_argument(
        "--verbose",
        type=int,
        default=5,
        help="Run validation every N epochs (e.g. 5 → evaluate at epoch 4, 9, 14, …).",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=250,
        help="Maximum number of training epochs (250 fits with original paper).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Number of BPR triplets per mini-batch (1024 fits with original paper).",
    )
    parser.add_argument(
        "--regs",
        type=float,
        default=1e-3,
        help="L2 regularisation coefficient for BPR embedding loss.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Initial learning rate for Adam optimiser.",
    )
    parser.add_argument("--train_dir", default="train")

    # =====================================================================
    #  Model Architecture
    # =====================================================================
    parser.add_argument(
        "--embed_size",
        type=int,
        default=64,
        help="Dimensionality of user/item embeddings.",
    )
    parser.add_argument(
        "--weight_size",
        nargs="?",
        default="[64,64,64]",
        help="Output sizes of each GNN layer (as a Python list string). "
        "Only used by NGCF backbone.",
    )
    parser.add_argument(
        "--core",
        type=int,
        default=5,
        help="K-core filtering threshold. "
        "5 = warm-start (remove users/items with < 5 interactions); "
        "0 = cold-start (keep everything).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of nearest neighbours to keep in the k-NN "
        "sparsification of the modality similarity graphs.",
    )
    parser.add_argument(
        "--cf_model",
        nargs="?",
        default="LightGCN",
        help="Collaborative filtering backbone: "
        "{MF, NGCF, LightGCN}. LightGCN is recommended.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=20,
        help="Number of consecutive non-improving evaluation "
        "cycles before stopping training. "
        "Analysis shows peak around epoch 120-180 with "
        "plateau oscillating within 1.3%%; patience=20 "
        "is sufficient to capture the peak.",
    )

    parser.add_argument(
        "--sparse",
        type=int,
        default=0,
        help="1 = use sparse adjacency matrices for the UI graph; "
        "0 = use dense (default).",
    )
    parser.add_argument(
        "--debug",
        default="True",
        help='If "True", enable file logging in addition to console output.',
    )

    parser.add_argument(
        "--norm_type",
        nargs="?",
        default="sym",
        help="Adjacency matrix normalisation type: "
        "{sym, rw, origin}. "
        "sym = symmetric (D^-½ A D^-½); "
        "rw  = random-walk (D^-1 A).",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="Index of the GPU to use (for multi-GPU machines).",
    )

    # =====================================================================
    #  Evaluation
    # =====================================================================
    parser.add_argument(
        "--Ks",
        nargs="?",
        default="[10,20]",
        help="Values of K for Recall@K, Precision@K, NDCG@K, Hit@K "
        '(as a Python list string, e.g. "[10,20]").',
    )
    parser.add_argument(
        "--test_flag",
        nargs="?",
        default="part",
        help="{part, full}. "
        "part = mini-batch evaluation (faster, uses heapq for top-K); "
        "full = full-sort evaluation (slower, also computes AUC).",
    )

    # =====================================================================
    #  MMHCL-specific: Hypergraph & Contrastive Learning
    # =====================================================================
    parser.add_argument(
        "--UI_layers",
        type=int,
        default=2,
        help="Number of GNN layers on the user-item bipartite graph "
        "(LightGCN / NGCF message-passing depth).",
    )
    parser.add_argument(
        "--User_layers",
        type=int,
        default=3,
        help="Number of GNN layers on the user-user co-interaction hypergraph.",
    )
    parser.add_argument(
        "--Item_layers",
        type=int,
        default=2,
        help="Number of GNN layers on the item-item multi-modal hypergraph.",
    )
    parser.add_argument(
        "--user_loss_ratio",
        type=float,
        default=0.03,
        help="Weight (λ_user) for the user-side contrastive loss. "
        "Set to 0 to disable user hypergraph branch entirely.",
    )
    parser.add_argument(
        "--item_loss_ratio",
        type=float,
        default=0.07,
        help="Weight (λ_item) for the item-side contrastive loss. "
        "Set to 0 to disable item hypergraph branch entirely.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature (τ) for InfoNCE contrastive loss. "
        "Lower values produce sharper similarity distributions.",
    )

    parser.add_argument(
        "--ablation_target",
        type=str,
        default="",
        help="Tag for ablation experiments (appears in log folder name). "
        "Leave empty for standard training.",
    )

    # =====================================================================
    #  Enhanced Early Stopping
    # =====================================================================
    parser.add_argument(
        "--early_stopping_min_epochs",
        type=int,
        default=0,
        help="Minimum epochs before early stopping can trigger. "
        "During warm-up the model may not improve every cycle.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0001,
        help="Minimum metric improvement to count as progress. "
        "Prevents premature stopping on tiny fluctuations.",
    )
    parser.add_argument(
        "--early_stopping_monitor",
        type=str,
        default="val_recall@20",
        help="Metric name to monitor (informational; the actual "
        "logic in main.py checks recall@20 and ndcg@20).",
    )
    parser.add_argument(
        "--early_stopping_mode",
        type=str,
        default="max",
        help='{max, min}. "max" means higher is better (for recall/ndcg).',
    )
    parser.add_argument(
        "--early_stopping_restore_best",
        type=int,
        default=1,
        help="1 = restore the best model weights when early stopping "
        "triggers; 0 = keep the last model weights.",
    )
    parser.add_argument(
        "--adaptive_patience",
        type=int,
        default=0,
        help="1 = enable adaptive patience (scales patience by dataset "
        "size); 0 = disabled.",
    )

    # =====================================================================
    #  ReduceLROnPlateau Scheduler
    # =====================================================================
    parser.add_argument(
        "--use_reduce_lr",
        type=int,
        default=0,
        help="1 = enable ReduceLROnPlateau (in addition to the "
        "exponential decay scheduler); 0 = disabled.",
    )
    parser.add_argument(
        "--reduce_lr_factor",
        type=float,
        default=0.5,
        help="Factor by which LR is multiplied when plateau is detected.",
    )
    parser.add_argument(
        "--reduce_lr_patience",
        type=int,
        default=3,
        help="Number of non-improving evaluation cycles before reducing LR.",
    )
    parser.add_argument(
        "--reduce_lr_min",
        type=float,
        default=1e-6,
        help="Minimum learning rate (floor for the scheduler).",
    )

    # =====================================================================
    #  Weights & Biases (W&B) Experiment Tracking
    # =====================================================================
    parser.add_argument(
        "--use_wandb",
        type=int,
        default=0,
        help="1 = enable W&B logging; 0 = disabled. "
        "Requires `pip install wandb` and `wandb login`.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mmhcl",
        help="W&B project name (creates one if it does not exist).",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="W&B entity (team or username). Leave empty to use your default entity.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="Custom name for this W&B run. Leave empty for an auto-generated name.",
    )

    # =====================================================================
    #  MMHCL+ — Neighbor-Layer Hypergraph Contrastive Learning
    #  (all args below are ignored by the original main.py;
    #   they are only consumed by main_mmhcl_plus.py)
    # =====================================================================

    # Warmup & temperature schedule (TEX §4.3–4.4)
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Epochs to run BPR-only warmup before enabling CL losses. "
        "Rev5.1 reduces to 5 warmup epochs (down from 10 in Rev44).",
    )
    parser.add_argument(
        "--tau_max",
        type=float,
        default=0.5,
        help="Initial (maximum) contrastive temperature for i2i InfoNCE. "
        "Anneals toward tau_min over training.",
    )
    parser.add_argument(
        "--tau_min",
        type=float,
        default=0.05,
        help="Floor temperature for the exponential annealing schedule.",
    )
    parser.add_argument(
        "--tau_gamma",
        type=float,
        default=0.99,
        help="Per-epoch decay factor: tau(t) = max(tau_min, tau_max * gamma^t).",
    )

    # Loss function configuration (Rev5.1: VICReg replaces Barlow Twins)
    parser.add_argument(
        "--vicreg_sim_weight",
        type=float,
        default=25.0,
        help="VICReg invariance (MSE) coefficient (u2u & ego-final branches).",
    )
    parser.add_argument(
        "--vicreg_var_weight",
        type=float,
        default=25.0,
        help="VICReg variance (hinge) coefficient.",
    )
    parser.add_argument(
        "--vicreg_cov_weight",
        type=float,
        default=1.0,
        help="VICReg covariance (off-diagonal) coefficient.",
    )
    parser.add_argument(
        "--info_nce_chunk_size",
        type=int,
        default=512,
        help="Chunk size for memory-safe chunked InfoNCE (i2i branch). "
        "Rev5.1 default: 512 (down from 1024).",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=6,
        help="Number of loss tasks for UncertaintyLossBalancer. "
        "Rev5.1: 6 (BPR, u2u, i2i, align, dir, ego_final).",
    )
    # Backward compatibility: kept but ignored by Rev5.1 code paths
    parser.add_argument(
        "--barlow_lambda",
        type=float,
        default=5e-3,
        help="[DEPRECATED] Barlow Twins lambda. Ignored in Rev5.1 (VICReg).",
    )
    parser.add_argument(
        "--gradnorm_alpha",
        type=float,
        default=1.5,
        help="[DEPRECATED] GradNorm alpha. Ignored in Rev5.1 (UncertaintyBalancer).",
    )

    # Per-task initial loss weights (GradNorm adjusts these dynamically at runtime)
    parser.add_argument(
        "--bpr_weight",
        type=float,
        default=1.0,
        help="Initial weight for the BPR loss term.",
    )
    parser.add_argument(
        "--u2u_weight",
        type=float,
        default=1.0,
        help="Initial weight for the u2u VICReg loss term.",
    )
    parser.add_argument(
        "--i2i_weight",
        type=float,
        default=1.0,
        help="Initial weight for the i2i Chunked InfoNCE loss term.",
    )
    parser.add_argument(
        "--align_weight",
        type=float,
        default=1.0,
        help="Initial weight for the soft BYOL cross-view alignment term.",
    )
    # Audit finding: with d=64 and shallow propagation (L=2–3), the raw
    # mini-batch Dirichlet energy is only ~0.0003–0.00045, which is ~400×
    # smaller than BPR (~0.13).  Raising to 1.0 gives the balancer a
    # meaningful signal to work with.
    parser.add_argument(
        "--dirichlet_weight",
        type=float,
        default=1.0,
        help="Initial weight for the Dirichlet energy regularisation term. "
        "Set higher (1.0–10.0) when the raw energy is very small "
        "(< 0.001) so the uncertainty balancer can balance it.",
    )
    parser.add_argument(
        "--ego_final_weight",
        type=float,
        default=1.0,
        help="Initial weight for the ego-final anchor VICReg loss (Rev5.1).",
    )

    # Expanded projector for u2u branch
    # Rev5.1: VICReg needs only 64→512→1024 (vs Barlow's 64→2048→8192)
    parser.add_argument(
        "--projector_hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension of the expanded projector MLP (u2u branch). "
        "Rev5.1: 512 (VICReg). Rev44: 2048 (Barlow Twins).",
    )
    parser.add_argument(
        "--projector_out_dim",
        type=int,
        default=1024,
        help="Output dimension of the expanded projector MLP (u2u branch). "
        "Rev5.1: 1024 (VICReg). Rev44: 8192 (Barlow Twins).",
    )

    # WEMAManager — dynamic EMA similarity weights (TEX §4.2)
    parser.add_argument(
        "--w_ema_alpha",
        type=float,
        default=0.9,
        help="EMA momentum for updating the item similarity matrix W_ema.",
    )
    parser.add_argument(
        "--w_ema_update_interval",
        type=int,
        default=5,
        help="Update W_ema every N training steps (0 = never update).",
    )

    # Soft topology-aware purification (TEX §3.4, §4.2)
    parser.add_argument(
        "--soft_purification_percentile",
        type=float,
        default=0.8,
        help="Percentile threshold for soft topology weight computation. "
        "Similarities below this percentile receive near-zero weight.",
    )
    parser.add_argument(
        "--soft_purification_temp",
        type=float,
        default=0.2,
        help="Softmax temperature applied when normalising soft topology weights.",
    )

    # EMA teacher network for Soft BYOL alignment (TEX §4.4, Algorithm 1 Step 11)
    parser.add_argument(
        "--ema_momentum",
        type=float,
        default=0.99,
        help="Momentum beta for the EMA teacher network: "
        "xi <- beta*xi + (1-beta)*theta. Higher = slower teacher update.",
    )

    # Profiling-guided activation checkpointing (TEX §4.3, Table 2)
    parser.add_argument(
        "--checkpoint_threshold",
        type=int,
        default=-1,
        help="Checkpoint hypergraph layers from this index onward "
        "(0-indexed). -1 disables checkpointing entirely.",
    )

    # =====================================================================
    #  MMHCL+ Rev5.1 — SVD Spectral Augmentation & Hard Negatives
    # =====================================================================
    parser.add_argument(
        "--use_svd_filtering",
        type=int,
        default=1,
        help="1 = enable SVD spectral augmentation (filter top-K singular "
        "values from incidence matrix); 0 = disabled.",
    )
    parser.add_argument(
        "--svd_top_k",
        type=int,
        default=10,
        help="Number of top singular values to zero out in SVD filtering. "
        "These correspond to popularity-dominated directions.",
    )
    parser.add_argument(
        "--n_hard_neg",
        type=int,
        default=10,
        help="Number of hard negatives per query in FAISS-based mining.",
    )
    parser.add_argument(
        "--hard_neg_pool_k",
        type=int,
        default=64,
        help="Pool size K for ANN search when mining hard negatives.",
    )
    parser.add_argument(
        "--hard_neg_weight",
        type=float,
        default=0.5,
        help="Relative weight of hard negatives in the InfoNCE denominator.",
    )
    parser.add_argument(
        "--temperature_free",
        type=int,
        default=0,
        help="1 = use temperature-free InfoNCE; 0 = standard InfoNCE.",
    )

    return parser.parse_args()
