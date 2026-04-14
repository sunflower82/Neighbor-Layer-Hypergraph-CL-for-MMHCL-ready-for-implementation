"""
main.py — Training Orchestrator for MMHCL
==========================================

This is the **entry point** of the MMHCL (Multi-Modal Hypergraph Contrastive
Learning) recommendation system.  It wires together every other module:

    parser.py  ──>  args          (hyperparameters)
    load_data.py ──>  Data        (user-item interactions + multi-modal graphs)
    Models.py  ──>  MMHCL         (the neural network)
    batch_test.py ──>  test_torch (evaluation on val / test splits)
    logging.py ──>  Logger        (file + console logging)

Overall training pipeline (per epoch):
    1.  Sample (user, positive_item, negative_item) triplets  — BPR paradigm
    2.  Forward pass through MMHCL to get embeddings
    3.  Compute total loss  =  BPR loss  +  embedding regularisation
                              +  item contrastive loss  +  user contrastive loss
    4.  Back-propagate and update parameters
    5.  Every ``verbose`` epochs, evaluate on the validation set
    6.  If validation metrics improve, evaluate on the test set and save the best
        model state; otherwise increment the early-stopping counter

Reference:
    "MMHCL: Multi-Modal Hypergraph Contrastive Learning for Recommendation"
    (ACM, 2024)
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import os
import pathlib
from time import time
from typing import Any

from Models import MMHCL
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.optim as optim
from utility.batch_test import *  # also creates `data_generator` (global)
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
from utility.common import (
    sparse_mx_to_torch_sparse_tensor as _sparse_mx_to_torch,
)
from utility.logging import Logger
from utility.parser import parse_args

# Type alias for metric result dictionaries
MetricsDict = dict[str, Any]

# ---------------------------------------------------------------------------
# 1.  Parse command-line arguments (see utility/parser.py for all defaults)
# ---------------------------------------------------------------------------
args: argparse.Namespace = parse_args()

# ---------------------------------------------------------------------------
# 2.  Optional Weights & Biases (W&B) integration for experiment tracking
#     Set --use_wandb 1 to enable; requires ``pip install wandb``.
# ---------------------------------------------------------------------------
wandb: Any = None
if args.use_wandb:
    try:
        import wandb as _wandb

        wandb = _wandb
    except ImportError:
        print("[WARNING] wandb not installed. Disabling W&B logging.")
        args.use_wandb = 0


# ---------------------------------------------------------------------------
# 3.  Experiment directory paths (computed per-run by train_evaluation_loop)
# ---------------------------------------------------------------------------
path_name: str = ""
path: str = ""
record_path: str = ""


# ===========================================================================
#  Trainer — the main class that owns the model, optimizer, and training loop
# ===========================================================================
class Trainer:
    """
    Encapsulates:
        - Model initialisation (MMHCL on GPU)
        - Adam optimiser + LambdaLR exponential-decay scheduler
        - Optional ReduceLROnPlateau scheduler
        - BPR (Bayesian Personalised Ranking) loss computation
        - The full train-evaluate loop with early stopping
        - Optional Optuna trial integration for intermediate reporting & pruning
    """

    def __init__(self, data_config: dict[str, Any], optuna_trial: Any = None) -> None:
        """
        Args:
            data_config: must contain keys
                'n_users', 'n_items',
                'UI_mat'   — normalised user-item bipartite adjacency (sparse, GPU),
                'User_mat' — user-user co-interaction hypergraph      (sparse, GPU),
                'Item_mat' — item-item multi-modal hypergraph          (sparse, GPU).
            optuna_trial: An optuna.trial.Trial object (or None). When provided,
                          the trainer reports intermediate validation metrics for
                          pruning support.
        """
        # --- Optuna trial handle (None when not tuning) ---
        self.optuna_trial: Any = optuna_trial

        # --- dataset dimensions ---
        self.n_users: int = data_config["n_users"]
        self.n_items: int = data_config["n_items"]

        # --- logger (writes to both per-run and aggregated directories) ---
        self.logger: Logger = Logger(
            path,
            is_debug=args.debug,
            target=path_name,
            path2=record_path,
            ablation_target=args.ablation_target,
        )
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        # --- store frequently-used hyperparameters ---
        self.lr: float = args.lr  # initial learning rate
        self.emb_dim: int = args.embed_size  # embedding dimensionality
        self.batch_size: int = args.batch_size  # BPR training batch size
        self.weight_size: list[int] = eval(args.weight_size)  # per-layer sizes
        self.n_layers: int = len(self.weight_size)  # number of GNN layers
        self.regs: float = args.regs  # L2 regularisation coefficient
        self.decay: float = self.regs  # alias used in bpr_loss()

        # --- move the three adjacency matrices to GPU ---
        self.UI_mat: torch.Tensor = data_config["UI_mat"].cuda()
        self.User_mat: torch.Tensor = data_config["User_mat"].cuda()
        self.Item_mat: torch.Tensor = data_config["Item_mat"].cuda()

        # --- instantiate the MMHCL model and move to GPU ---
        self.model: MMHCL = MMHCL(self.n_users, self.n_items, self.emb_dim)
        self.model = self.model.cuda()

        # --- optimizer and learning-rate schedulers ---
        self.optimizer: optim.Adam = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler: optim.lr_scheduler.LambdaLR = self.set_lr_scheduler()
        self.reduce_lr_scheduler: optim.lr_scheduler.ReduceLROnPlateau | None = (
            self.set_reduce_lr_scheduler() if args.use_reduce_lr else None
        )

    # -----------------------------------------------------------------------
    #  Learning-rate schedulers
    # -----------------------------------------------------------------------
    def set_lr_scheduler(self) -> optim.lr_scheduler.LambdaLR:
        """
        Exponential decay scheduler:  lr(epoch) = lr_0 * 0.96^(epoch / 50)

        This provides a smooth, gradual decay — after 50 epochs the LR is
        multiplied by 0.96, after 100 epochs by 0.96^2 ≈ 0.922, etc.
        """
        scheduler: optim.lr_scheduler.LambdaLR = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_decay_schedule
        )
        return scheduler

    def set_reduce_lr_scheduler(self) -> optim.lr_scheduler.ReduceLROnPlateau:
        """
        ReduceLROnPlateau: automatically halves the LR when the monitored
        validation metric stops improving for ``reduce_lr_patience`` evaluation
        cycles.  This is an *additional* scheduler layered on top of the
        exponential decay.
        """
        scheduler: optim.lr_scheduler.ReduceLROnPlateau = (
            optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=args.early_stopping_mode,
                factor=args.reduce_lr_factor,
                patience=args.reduce_lr_patience,
                min_lr=args.reduce_lr_min,
            )
        )
        return scheduler

    # -----------------------------------------------------------------------
    #  Evaluation helper
    # -----------------------------------------------------------------------
    def test(self, users_to_test: list[int], is_val: bool) -> MetricsDict:
        """
        Run a full evaluation pass (no gradient computation).

        Args:
            users_to_test: list of user IDs to evaluate.
            is_val: True → use validation set; False → use test set.

        Returns:
            dict with keys 'recall', 'precision', 'ndcg', 'hit_ratio', 'auc'.
        """
        self.model.eval()
        with torch.no_grad():
            ua_embeddings: torch.Tensor
            ia_embeddings: torch.Tensor
            ii: torch.Tensor
            uu: torch.Tensor
            ua_embeddings, ia_embeddings, ii, uu = self.model(
                self.UI_mat, self.Item_mat, self.User_mat
            )
        result: MetricsDict = test_torch(
            ua_embeddings, ia_embeddings, users_to_test, is_val
        )
        return result

    # -----------------------------------------------------------------------
    #  Main training loop
    # -----------------------------------------------------------------------
    def train(self, return_validation: bool = False) -> float:
        """
        Full training procedure with early stopping and optional W&B logging.

        Args:
            return_validation: If True, return best *validation* Recall@20
                               instead of best test Recall@20.  Used by Optuna
                               to avoid data leakage into the test set.

        Returns:
            Best validation (or test) Recall@20 at the early-stopping checkpoint.
            Returns 0.0 if training fails (e.g. NaN loss).
        """
        training_time_list: list[float] = []
        loss_loger: list[float] = []
        pre_loger: list[npt.NDArray[np.floating]] = []
        rec_loger: list[npt.NDArray[np.floating]] = []
        ndcg_loger: list[npt.NDArray[np.floating]] = []
        hit_loger: list[npt.NDArray[np.floating]] = []
        stopping_step: int = 0

        # n_batch = how many mini-batches per epoch
        n_batch: int = data_generator.n_train // args.batch_size + 1
        best_recall: float = 0.0  # best validation recall@20 seen so far
        best_ndcg: float = 0.0  # best validation ndcg@20 seen so far
        best_val_recall: float = 0.0  # tracked for Optuna return value
        best_model_state: dict[str, Any] | None = None
        test_ret: str | MetricsDict = ""
        eval_step: int = 0  # counter for Optuna intermediate reports

        # ===== Initialise W&B run (if enabled) =====
        if args.use_wandb and wandb is not None:
            wandb_config: dict[str, Any] = vars(args)
            wandb_config["path_name"] = path_name
            wandb_init_kwargs: dict[str, Any] = {
                "project": args.wandb_project,
                "config": wandb_config,
                "reinit": True,
            }
            if args.wandb_entity:
                wandb_init_kwargs["entity"] = args.wandb_entity
            if args.wandb_run_name:
                wandb_init_kwargs["name"] = args.wandb_run_name
            wandb.init(**wandb_init_kwargs)
            self.logger.logging(f"W&B run initialized: {wandb.run.name}")

        # ===== Epoch loop =====
        for epoch in range(args.epoch):
            t1: float = time()
            # Accumulators for epoch-level loss components
            loss: float = 0.0
            mf_loss: float = 0.0
            emb_loss: float = 0.0
            reg_loss: float = 0.0
            contrastive_loss: float = 0.0
            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time: float = 0.0

            # ----- Mini-batch loop within one epoch -----
            for idx in range(n_batch):
                self.model.train()
                self.optimizer.zero_grad()

                # (1) Sample a BPR triplet batch
                sample_t1: float = time()
                users: list[int]
                pos_items: list[int]
                neg_items: list[int]
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1

                # (2) Forward pass — get all user/item embeddings + hypergraph views
                ua_embeddings: torch.Tensor
                ia_embeddings: torch.Tensor
                ii: torch.Tensor
                uu: torch.Tensor
                ua_embeddings, ia_embeddings, ii, uu = self.model(
                    self.UI_mat, self.Item_mat, self.User_mat
                )

                # Look up embeddings for this batch's users and items
                u_g_embeddings: torch.Tensor = ua_embeddings[users]
                pos_i_g_embeddings: torch.Tensor = ia_embeddings[pos_items]
                neg_i_g_embeddings: torch.Tensor = ia_embeddings[neg_items]

                # (3) BPR loss
                batch_mf_loss: torch.Tensor
                batch_emb_loss: torch.Tensor
                batch_reg_loss: float
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
                    u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
                )

                # (4) Contrastive losses (InfoNCE)
                batch_contrastive_loss1: torch.Tensor = (
                    self.model.batched_contrastive_loss(ia_embeddings, ii)
                )
                batch_contrastive_loss1 *= args.item_loss_ratio

                batch_contrastive_loss2: torch.Tensor = (
                    self.model.batched_contrastive_loss(ua_embeddings, uu)
                )
                batch_contrastive_loss2 *= args.user_loss_ratio

                batch_contrastive_loss: torch.Tensor = (
                    batch_contrastive_loss1 + batch_contrastive_loss2
                )

                # (5) Total loss = BPR + regularisation + contrastive
                batch_loss: torch.Tensor = (
                    batch_mf_loss
                    + batch_emb_loss
                    + batch_reg_loss
                    + batch_contrastive_loss
                )

                # (6) Back-propagation and parameter update
                batch_loss.backward(retain_graph=False)
                self.optimizer.step()

                # Accumulate for epoch-level logging (.item() avoids UserWarning
                # about converting a tensor that requires grad to a Python float)
                loss += batch_loss.item()
                mf_loss += batch_mf_loss.item()
                emb_loss += batch_emb_loss.item()
                reg_loss += float(batch_reg_loss)
                contrastive_loss += batch_contrastive_loss.item()

            # Step the exponential LR decay scheduler (once per epoch)
            self.lr_scheduler.step()

            # Free GPU memory for embeddings computed during training
            del ua_embeddings, ia_embeddings, u_g_embeddings
            del neg_i_g_embeddings, pos_i_g_embeddings, ii, uu

            # ----- NaN guard -----
            if math.isnan(loss):
                self.logger.logging("ERROR: loss is nan.")
                if args.use_wandb and wandb is not None:
                    wandb.finish(exit_code=1)
                return 0.0

            # ----- W&B: log training loss every epoch -----
            if args.use_wandb and wandb is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": loss,
                        "train/mf_loss": mf_loss,
                        "train/emb_loss": emb_loss,
                        "train/contrastive_loss": contrastive_loss,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # ----- Non-evaluation epoch: just log training loss and move on -----
            if (epoch + 1) % args.verbose != 0:
                perf_str: str = "Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]" % (
                    epoch,
                    time() - t1,
                    loss,
                    mf_loss,
                    emb_loss,
                    contrastive_loss,
                )
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)
                continue

            # ===== Evaluation epoch (every ``verbose`` epochs) =====
            t2: float = time()
            users_to_test: list[int] = list(data_generator.test_set.keys())
            users_to_val: list[int] = list(data_generator.val_set.keys())
            ret: MetricsDict = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)

            t3: float = time()

            # Store validation metrics for later analysis
            loss_loger.append(loss)
            rec_loger.append(ret["recall"])
            pre_loger.append(ret["precision"])
            ndcg_loger.append(ret["ndcg"])
            hit_loger.append(ret["hit_ratio"])

            # Pretty-print epoch summary
            if args.verbose > 0:
                perf_str = (
                    "Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], "
                    "recall=[%.5f, %.5f], precision=[%.5f, %.5f], "
                    "hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]"
                    % (
                        epoch,
                        t2 - t1,
                        t3 - t2,
                        loss,
                        mf_loss,
                        emb_loss,
                        contrastive_loss,
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

            # ----- W&B: log validation metrics -----
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

            # ----- ReduceLROnPlateau step (monitors val recall@20) -----
            if self.reduce_lr_scheduler is not None:
                self.reduce_lr_scheduler.step(ret["recall"][-1])

            # ----- Optuna: report intermediate value & check pruning -----
            current_val_recall: float = float(ret["recall"][-1])
            if current_val_recall > best_val_recall:
                best_val_recall = current_val_recall

            if self.optuna_trial is not None:
                self.optuna_trial.report(current_val_recall, eval_step)
                eval_step += 1
                if self.optuna_trial.should_prune():
                    self.logger.logging(
                        f"Optuna pruned at epoch {epoch} "
                        f"(val recall@20 = {current_val_recall:.6f})"
                    )
                    if args.use_wandb and wandb is not None:
                        wandb.finish(exit_code=0)
                    import optuna as _optuna_mod

                    raise _optuna_mod.TrialPruned(
                        f"Pruned at epoch {epoch}, val recall@20={current_val_recall:.6f}"
                    )

            # ===== Early-stopping logic =====
            improved: bool = False
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
                # ---- Improvement found → evaluate on the TEST set ----
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

                # W&B: log test metrics at this checkpoint
                if args.use_wandb and wandb is not None:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "test/recall@20": test_ret["recall"][1],
                            "test/precision@20": test_ret["precision"][1],
                            "test/ndcg@20": test_ret["ndcg"][1],
                            "best_recall": best_recall,
                            "best_ndcg": best_ndcg,
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
                    fname: str = f"Model.epoch={epoch}.pth"
                    torch.save(self.model.state_dict(), os.path.join(path, fname))
                    break
            else:
                self.logger.logging(
                    f"Epoch {epoch}: no improvement, but "
                    f"min_epochs={args.early_stopping_min_epochs} not reached yet"
                )

        # ===== Post-training: log and return the BEST test results =====
        best_test_recall: float = 0.0
        if isinstance(test_ret, dict):
            Ks_list: list[int] = eval(args.Ks)
            best_test_recall = float(test_ret["recall"][1])
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
        self.logger.logging_sum(f"{path_name}:{str(test_ret)}")

        if args.use_wandb and wandb is not None:
            wandb.finish()

        if return_validation:
            return best_val_recall
        return best_test_recall

    # -----------------------------------------------------------------------
    #  BPR (Bayesian Personalised Ranking) Loss
    # -----------------------------------------------------------------------
    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """BPR pairwise ranking loss with L2 embedding regularisation."""
        return _bpr_loss(users, pos_items, neg_items, self.batch_size, self.decay)

    # -----------------------------------------------------------------------
    #  Utility: scipy sparse → torch sparse
    # -----------------------------------------------------------------------
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx: sp.spmatrix) -> torch.Tensor:
        """Convert a scipy sparse matrix to a torch sparse COO tensor."""
        return _sparse_mx_to_torch(sparse_mx)


# ===========================================================================
#  Public API: train_evaluation_loop — callable by Optuna or the notebook
# ===========================================================================
def train_evaluation_loop(
    args_ns: argparse.Namespace | None = None,
    optuna_trial: Any = None,
    return_validation: bool = False,
) -> float:
    """
    Run one complete MMHCL training cycle and return the best Recall@20.

    This function is the single entry point for both standalone training
    (``python main.py``) and Optuna hyper-parameter search.

    Args:
        args_ns: A complete argparse.Namespace with all hyperparameters.
                 If *None*, the module-level ``args`` (from ``parse_args()``)
                 is used unchanged — this is the normal CLI path.
        optuna_trial: An ``optuna.trial.Trial`` object (or *None*).  Passed
                      through to ``Trainer`` so it can report intermediate
                      validation metrics and honour pruning requests.
        return_validation: If *True*, return the best **validation** Recall@20
                           instead of test Recall@20.  This prevents data
                           leakage when Optuna optimises the objective.

    Returns:
        Best validation (or test) Recall@20 at the early-stopping checkpoint.
        Returns 0.0 if training fails (e.g. NaN loss).
    """
    global args, path_name, path, record_path, wandb

    if args_ns is not None:
        args = args_ns
        import utility.load_data as _ld

        _ld.args = args

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

    config: dict[str, Any] = dict()
    config["n_users"] = data_generator.n_users
    config["n_items"] = data_generator.n_items
    config["UI_mat"] = data_generator.get_UI_mat()
    config["User_mat"] = data_generator.get_U2U_mat()

    if args.dataset == "Tiktok":
        config["Item_mat"] = data_generator.get_tiktok_I2I_Hypergraph_mul_mat()
    elif args.dataset in ["Clothing", "Sports", "Baby"]:
        config["Item_mat"] = data_generator.get_I2I_Hypergraph_mul_mat()

    trainer: Trainer = Trainer(data_config=config, optuna_trial=optuna_trial)
    best_recall: float = trainer.train(return_validation=return_validation)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return best_recall


# ===========================================================================
#  Entry point (CLI)
# ===========================================================================
if __name__ == "__main__":
    train_evaluation_loop()
