"""Standalone CLI runner for the MMHCL+ Q1-style ablation study (Rev 5.2).

This mirrors the logic of the notebook's "Section 7" ablation driver, but is
invokable from a plain terminal so long-running sweeps do not require a live
Jupyter kernel. It launches `main_mmhcl_plus.py` once per (variant, seed)
combination, parses the per-run log, and writes three CSVs + one XLSX with
raw results, seed-averaged summary, and paired-test statistics.

Usage examples
--------------

    # Full 15 x 3 sweep (Baby, 250 epochs per run, ~hours per run)
    python run_ablation.py

    # Shorter smoke sweep on two variants, 1 seed, 20 epochs
    python run_ablation.py --variants A0_full,A7_ego_final --seeds 1 --epochs 20

    # Custom dataset / GPU
    python run_ablation.py --dataset Baby --gpu 0 --seeds 3 --epochs 250

The script is resilient to individual-run failures: it records the return
code and continues with the next (variant, seed). All CSV/XLSX outputs are
written to --out-dir (default: ablation_outputs/).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE: Path = Path(__file__).resolve().parent
REPO: Path = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mmhcl_plus.ablation import REGISTRY, available_variants, get as get_variant  # noqa: E402


DEFAULT_VARIANTS: list[str] = [
    "A0_full",
    "A1_no_nlcl",
    "A2_no_svd",
    "A3_small_proj",
    "A4_no_ramp",
    "A5_no_delay",
    "A6_no_dirichlet",
    "A7_ego_final",
    "A8_no_cross",
    "B1_g1",
    "B2_g2",
    "B3_g3",
    "C1_uncertainty",
    "C2_gradnorm",
    "C3_fixed",
]


_BEST_PATTERNS: dict[str, re.Pattern[str]] = {
    "recall@20": re.compile(r"BEST_Test_Recall@20:\s+([\d.]+)"),
    "precision@20": re.compile(r"BEST_Test_Precision@20:\s+([\d.]+)"),
    "ndcg@20": re.compile(r"BEST_Test_NDCG@20:\s+([\d.]+)"),
}
_FALLBACK_PATTERNS: dict[str, re.Pattern[str]] = {
    "recall@20": re.compile(r"Test_Recall@20:\s+([\d.]+)"),
    "precision@20": re.compile(r"Test_Precision@20:\s+([\d.]+)"),
    "ndcg@20": re.compile(r"Test_NDCG@20:\s+([\d.]+)"),
}


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="Clothing")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=250,
                   help="Max epochs per run (early stopping may cut it shorter).")
    p.add_argument("--seeds", type=int, default=3,
                   help="Number of random seeds per variant.")
    p.add_argument("--seed-list", default="",
                   help="Optional explicit comma-separated seed list (overrides --seeds).")
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                   help="Comma-separated variant names from the ablation registry.")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--regs", type=float, default=1e-3)
    p.add_argument("--embed-size", type=int, default=64)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--core", type=int, default=5)
    p.add_argument("--user-layers", type=int, default=3)
    p.add_argument("--item-layers", type=int, default=2)
    p.add_argument("--user-loss-ratio", type=float, default=0.03)
    p.add_argument("--item-loss-ratio", type=float, default=0.07)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--out-dir", default=str(REPO / "ablation_outputs"))
    p.add_argument("--python", default=sys.executable,
                   help="Interpreter to launch subprocesses with "
                        "(default: same Python as this runner).")
    p.add_argument("--use-wandb", type=int, default=0)
    p.add_argument("--wandb-project", default="snips-local-mmhcl-plus-baby-ablation")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--early-stopping-patience", type=int, default=30)
    p.add_argument("--verbose", type=int, default=5)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands but do not execute them.")
    return p.parse_args()


def resolve_variants(args: argparse.Namespace) -> list[str]:
    requested: list[str] = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown: list[str] = [v for v in requested if v not in REGISTRY]
    if unknown:
        print(f"[fatal] unknown variants (not in registry): {unknown}")
        print(f"         available: {available_variants()}")
        sys.exit(2)
    return requested


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seed_list.strip():
        return [int(s) for s in args.seed_list.split(",") if s.strip()]
    random.seed(int(time.time_ns() % (2**31)))
    return [random.randint(1, 2**31 - 1) for _ in range(args.seeds)]


def build_cmd(args: argparse.Namespace, variant: str, seed: int) -> list[str]:
    return [
        args.python,
        str(HERE / "main_mmhcl_plus.py"),
        "--dataset", args.dataset,
        "--gpu_id", str(args.gpu),
        "--seed", str(seed),
        "--epoch", str(args.epochs),
        "--verbose", str(args.verbose),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--regs", str(args.regs),
        "--embed_size", str(args.embed_size),
        "--topk", str(args.topk),
        "--core", str(args.core),
        "--User_layers", str(args.user_layers),
        "--Item_layers", str(args.item_layers),
        "--user_loss_ratio", str(args.user_loss_ratio),
        "--item_loss_ratio", str(args.item_loss_ratio),
        "--temperature", str(args.temperature),
        "--early_stopping_patience", str(args.early_stopping_patience),
        "--early_stopping_min_epochs", str(min(75, max(args.epochs // 2, 1))),
        "--early_stopping_min_delta", "0.0001",
        "--early_stopping_monitor", "val_recall@20",
        "--early_stopping_mode", "max",
        "--early_stopping_restore_best", "1",
        "--use_reduce_lr", "1",
        "--reduce_lr_factor", "0.5",
        "--reduce_lr_patience", "3",
        "--reduce_lr_min", "1e-6",
        "--ablation_variant", variant,
        "--use_wandb", str(args.use_wandb),
        "--wandb_project", args.wandb_project,
        "--wandb_entity", args.wandb_entity,
        "--wandb_run_name", f"abl_{variant}_seed_{seed}",
    ]


def log_path_for(args: argparse.Namespace, variant: str, seed: int) -> Path:
    """Mirror build_experiment_paths() naming convention."""
    name: str = (
        f"uu_ii={args.user_layers}_{args.item_layers}"
        f"_{args.user_loss_ratio}_{args.item_loss_ratio}"
        f"_topk={args.topk}_t={args.temperature}"
        f"_regs={args.regs}_dim={args.embed_size}"
        f"_seed={seed}_{variant}"
    )
    return REPO / args.dataset / name / f"{name}.txt"


def parse_log(log_file: Path) -> dict[str, float] | None:
    if not log_file.exists():
        print(f"  [warn] log file missing: {log_file}")
        return None
    text: str = log_file.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float] = {}
    for metric, pat in _BEST_PATTERNS.items():
        m = pat.search(text) or _FALLBACK_PATTERNS[metric].search(text)
        if m:
            out[metric] = float(m.group(1))
    if len(out) != len(_BEST_PATTERNS):
        print(f"  [warn] partial metrics parsed from {log_file}: {out}")
        return None
    out["log_file"] = str(log_file)  # type: ignore[assignment]
    return out


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    variants: list[str] = resolve_variants(args)
    seeds: list[int] = resolve_seeds(args)

    print("=" * 80)
    print(f"Ablation sweep: {len(variants)} variants x {len(seeds)} seeds "
          f"(epochs<=" f"{args.epochs} per run)")
    print(f"Variants: {variants}")
    print(f"Seeds:    {seeds}")
    print("=" * 80)

    os.chdir(HERE)  # main_mmhcl_plus.py expects cwd == codes/

    env: dict[str, str] = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    results: list[dict[str, Any]] = []
    total: int = len(variants) * len(seeds)
    idx: int = 0

    for variant in variants:
        v = get_variant(variant)
        for seed in seeds:
            idx += 1
            run_id: str = f"{variant}/seed={seed}"
            print(f"\n{'#' * 80}")
            print(f"# [{idx}/{total}]  variant={variant}  seed={seed}")
            print(f"# notes: {v.notes}")
            print(f"{'#' * 80}", flush=True)

            cmd: list[str] = build_cmd(args, variant, seed)
            if args.dry_run:
                print("  DRY-RUN cmd:", " ".join(cmd))
                continue

            t0: float = time.time()
            proc = subprocess.run(cmd, env=env, text=True,
                                  encoding="utf-8", errors="replace")
            dt: float = time.time() - t0

            if proc.returncode != 0:
                print(f"[WARNING] {run_id} exited with code {proc.returncode} "
                      f"after {dt / 60:.1f} min")

            metrics = parse_log(log_path_for(args, variant, seed))
            row: dict[str, Any] = {
                "variant": variant,
                "seed": seed,
                "notes": v.notes,
                "duration_sec": dt,
                "return_code": proc.returncode,
                "recall@20": float("nan"),
                "precision@20": float("nan"),
                "ndcg@20": float("nan"),
            }
            if metrics is not None:
                row.update(metrics)
                print(f"  metrics: recall@20={row['recall@20']:.6f} "
                      f"precision@20={row['precision@20']:.6f} "
                      f"ndcg@20={row['ndcg@20']:.6f} "
                      f"(elapsed {dt / 60:.1f} min)")
            else:
                print(f"  metrics: UNAVAILABLE  (elapsed {dt / 60:.1f} min)")
            results.append(row)
    return results


def write_outputs(results: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv: Path = out_dir / "ablation_raw.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["variant", "seed", "notes", "duration_sec",
                        "return_code", "recall@20", "precision@20", "ndcg@20",
                        "log_file"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\n[write] raw -> {raw_csv}")

    # Summary per variant
    from collections import defaultdict
    bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        bucket[row["variant"]].append(row)

    summary_csv: Path = out_dir / "ablation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["variant", "n_seeds",
                         "recall@20_mean", "recall@20_std",
                         "precision@20_mean", "precision@20_std",
                         "ndcg@20_mean", "ndcg@20_std"])
        import statistics
        for variant, rows in bucket.items():
            def _vals(k: str) -> list[float]:
                return [r[k] for r in rows if not (r[k] != r[k])]  # drop NaN
            row_out: list[Any] = [variant, len(rows)]
            for metric in ("recall@20", "precision@20", "ndcg@20"):
                vals = _vals(metric)
                if len(vals) >= 1:
                    row_out.append(f"{sum(vals) / len(vals):.6f}")
                    row_out.append(f"{(statistics.stdev(vals) if len(vals) > 1 else 0.0):.6f}")
                else:
                    row_out.extend(["", ""])
            writer.writerow(row_out)
    print(f"[write] summary -> {summary_csv}")

    # Raw json for downstream analysis
    json_path: Path = out_dir / "ablation_raw.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"[write] json -> {json_path}")


def main() -> None:
    args = parse_cli()
    out_dir: Path = Path(args.out_dir).resolve()
    results: list[dict[str, Any]] = run_sweep(args)
    if not args.dry_run:
        write_outputs(results, out_dir)
        print("\n" + "=" * 80)
        print(f"Completed {len(results)} runs. Outputs in {out_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()
