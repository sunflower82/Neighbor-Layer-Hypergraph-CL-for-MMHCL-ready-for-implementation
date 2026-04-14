"""
tune_optuna.py — Optuna Hyperparameter Search for MMHCL on Amazon Baby
=======================================================================

Uses Tree-structured Parzen Estimator (TPE) with **MedianPruner** to
efficiently search a search space *tailored to the Amazon Baby dataset*.

Amazon Baby characteristics:
  - Extremely high sparsity (~99.96% empty cells)
  - Severe long-tail item distribution
  - Only ~7k items / ~19k users after 5-core filtering

Design rationale for the search space:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  K (topk / k-NN neighbours)                                        │
  │    Range: 2–15  (default paper: 5).  The data is very sparse;      │
  │    large K captures noise rather than signal on Baby.               │
  ├─────────────────────────────────────────────────────────────────────┤
  │  tau (temperature)                                                  │
  │    Range: 0.1–0.8, step 0.05.  Long-tail data benefits from a      │
  │    flexible temperature so that niche items are not pushed too far  │
  │    away in the contrastive loss.                                    │
  ├─────────────────────────────────────────────────────────────────────┤
  │  L2 regularisation (regs)                                           │
  │    Range: 1e-5 – 1e-1 (log-uniform).  Critical for preventing      │
  │    overfitting when users have very few interactions.               │
  ├─────────────────────────────────────────────────────────────────────┤
  │  alpha / beta (user/item contrastive loss weights)                  │
  │    Range: 0.01–0.15, step 0.01.  Finer resolution than the paper's │
  │    coarse {0.01, 0.03, 0.05, 0.07, 0.09} grid.                    │
  ├─────────────────────────────────────────────────────────────────────┤
  │  GNN depths (User_layers, Item_layers)                              │
  │    Range: 1–3 (as in the original paper).                           │
  └─────────────────────────────────────────────────────────────────────┘

Pruning:
  MedianPruner (n_startup_trials=5, n_warmup_steps=20) immediately kills
  trials whose intermediate validation Recall@20 falls below the median
  of prior trials, freeing GPU time for more promising configurations.

Data-leakage prevention:
  The objective function returns the best **validation** Recall@20 (not
  test).  The test set is only touched during the final multi-seed run
  after the best configuration is determined.

Output:
  - ``optuna_results_amazon_baby.csv``  — full trial dataframe
  - ``best_mmhcl_baby_config.json``     — best hyperparameters (JSON)

Usage (run from project root):
    python tune_optuna.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

# =====================================================================
#  Tuning configuration
# =====================================================================
N_TRIALS: int = 50
OPTUNA_SEED: int = 42
STUDY_NAME: str = "MMHCL_Baby_Tuning_v2"
TRAINING_SEED: int = 2024

# MedianPruner settings
PRUNER_N_STARTUP_TRIALS: int = 5  # run at least 5 trials before pruning
PRUNER_N_WARMUP_STEPS: int = 20  # don't prune before eval-step 20

# =====================================================================
#  Path & sys.argv setup — MUST happen before importing MMHCL modules
# =====================================================================
PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
CODES_DIR: str = os.path.join(PROJECT_ROOT, "codes")
os.chdir(CODES_DIR)
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)

# Minimal CLI args — Optuna overrides the rest via `args_ns`
sys.argv = [
    "main.py",
    "--dataset",
    "Baby",
    "--batch_size",
    "1024",
    "--epoch",
    "250",
    "--verbose",
    "5",
    "--embed_size",
    "64",
    "--core",
    "5",
    "--early_stopping_patience",
    "20",
    "--early_stopping_min_epochs",
    "50",
]

# =====================================================================
#  Now safe to import MMHCL modules (data_generator loads once here)
# =====================================================================
from main import train_evaluation_loop  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402
from utility.parser import parse_args  # noqa: E402

_default_args = parse_args()


# =====================================================================
#  Objective — one Optuna trial
# =====================================================================
def objective(trial: optuna.Trial) -> float:
    """
    Single Optuna trial: suggest hyperparameters tailored to Amazon Baby,
    train the model, and return the best **validation** Recall@20.

    The trial object is forwarded to main.py so that ``Trainer.train()``
    can call ``trial.report()`` / ``trial.should_prune()`` at every
    evaluation epoch.
    """
    args = copy.deepcopy(_default_args)

    # ----- Search space targeting Baby's sparsity & long-tail -----
    args.topk = trial.suggest_int("topk", 2, 15)
    args.temperature = trial.suggest_float("temperature", 0.1, 0.8, step=0.05)
    args.regs = trial.suggest_float("regs", 1e-5, 1e-1, log=True)
    args.user_loss_ratio = trial.suggest_float("user_loss_ratio", 0.01, 0.15, step=0.01)
    args.item_loss_ratio = trial.suggest_float("item_loss_ratio", 0.01, 0.15, step=0.01)
    args.User_layers = trial.suggest_int("User_layers", 1, 3)
    args.Item_layers = trial.suggest_int("Item_layers", 1, 3)

    # Fixed across trials
    args.seed = TRAINING_SEED
    args.use_wandb = 0

    print(f"\n{'=' * 60}")
    print(f"[Trial {trial.number + 1}/{N_TRIALS}] Hyperparameters:")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")
    print(f"{'=' * 60}")

    try:
        # return_validation=True → Optuna optimises val recall, NOT test
        best_val_recall: float = train_evaluation_loop(
            args_ns=args,
            optuna_trial=trial,
            return_validation=True,
        )
    except optuna.TrialPruned:
        # Re-raise so Optuna records: status=PRUNED
        print(f"[Trial {trial.number + 1}] Pruned early.")
        torch.cuda.empty_cache()
        raise
    except Exception as e:
        print(f"[Trial {trial.number + 1}] Failed: {e}")
        torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()

    print(
        f"[Trial {trial.number + 1}] Best Validation Recall@20 = {best_val_recall:.6f}"
    )
    return best_val_recall


# =====================================================================
#  Entry point
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MMHCL Optuna Hyperparameter Search — Amazon Baby (v2)")
    print(f"  Trials: {N_TRIALS}  |  Sampler: TPE (seed={OPTUNA_SEED})")
    print(
        f"  Pruner: MedianPruner "
        f"(startup={PRUNER_N_STARTUP_TRIALS}, warmup={PRUNER_N_WARMUP_STEPS})"
    )
    print(f"  Training seed: {TRAINING_SEED} (fixed for fair comparison)")
    gpu_name: str = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    print(f"  GPU: {gpu_name}")
    print("=" * 70)

    study: optuna.Study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=PRUNER_N_STARTUP_TRIALS,
            n_warmup_steps=PRUNER_N_WARMUP_STEPS,
        ),
    )

    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

    # ---- Print results ----
    print("\n" + "=" * 70)
    print("OPTIMISATION COMPLETE")
    print("=" * 70)
    print(f"  Best Validation Recall@20: {study.best_value:.6f}")
    print(f"  Best Trial: #{study.best_trial.number + 1}")
    print("\n  Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # ---- Save full trial dataframe ----
    results_path: str = os.path.join(PROJECT_ROOT, "optuna_results_amazon_baby.csv")
    study.trials_dataframe().to_csv(results_path, index=False)
    print(f"\n  Trial results saved to: {results_path}")

    # ---- Auto-export best config for final multi-seed run ----
    best_config_path: str = os.path.join(PROJECT_ROOT, "best_mmhcl_baby_config.json")
    best_config: dict = {
        "study_name": STUDY_NAME,
        "best_validation_recall@20": study.best_value,
        "best_trial_number": study.best_trial.number + 1,
        "n_trials_completed": len(study.trials),
        "best_params": study.best_params,
    }
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=4)

    print(f"  Best config saved to: {best_config_path}")
    print(
        "\n  Next step: load best_mmhcl_baby_config.json in the notebook "
        "and run the final multi-seed training with these optimised "
        "hyperparameters."
    )
    print("=" * 70)
